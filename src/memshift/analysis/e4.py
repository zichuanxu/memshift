"""E4 analysis (plan §6 E4, §2.5 H_S): ContractValid by (mode, lag, B, system), the lag-window
split, state divergence, and the sync-semantics x B interaction.

Cells        (mode, lag, system); B is 0 / 1 for the HoldMem cells and None for the zero system.
             The ideal stream is recorded with lag 0.
Window       a probe is inside the lag window when its delay after the triggering true event is
             shorter than L (never, for ideal). `unsynced` / `synced` splits by whether the
             triggering object was identical in the delivered and the true log at as_of, which
             also counts the poll interval.
Drop         CV(official, L) - CV(ideal) for one system, paired by probe, clustered by world.
Interaction  per L: Drop(B=1) - Drop(B=0), paired by (world, probe). H_S predicts a drop that does
             not depend on B and sits inside the lag window; an interval that excludes 0 is the
             plan's trigger for promoting sync semantics to a main axis.
All intervals are world-level bootstraps (scoring.stats).
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memshift.experiments.common import read_jsonl
from memshift.holdmem.system import CellConfig
from memshift.scoring.stats import Estimate, paired_diff, world_bootstrap

Row = dict[str, Any]
Cond = Callable[[Row], bool]
Vec = dict[str, dict[str, float]]

HM = {b: CellConfig(1, b, "source").name for b in (0, 1)}


def _lag(r: Row) -> float:
    return float(r["tags"]["lag"])


def in_window(r: Row) -> bool:
    return float(r["tags"]["delta"]) < _lag(r)


SPLITS: dict[str, Cond] = {
    "all": lambda r: True,
    "in_window": in_window,
    "outside_window": lambda r: not in_window(r),
    "unsynced": lambda r: not r["tags"]["synced"],
    "synced": lambda r: bool(r["tags"]["synced"]),
    "gold_changed": lambda r: bool(r["tags"]["gold_changed"]),
    "gold_unchanged": lambda r: not r["tags"]["gold_changed"],
}


def _by_world(rows: list[Row], field: str, cond: Cond) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if cond(r):
            out[r["world_id"]].append(float(r[field]))
    return out


def _vec(rows: list[Row], mode: str, lag: float, system: str, cond: Cond = SPLITS["all"]) -> Vec:
    out: Vec = defaultdict(dict)
    for r in rows:
        t = r["tags"]
        if t["mode"] == mode and _lag(r) == lag and r["system"] == system and cond(r):
            out[r["world_id"]][r["probe_id"]] = float(r["contract_valid"])
    return out


def _minus(a: Vec, b: Vec) -> Vec:
    return {w: {k: a[w][k] - b[w][k] for k in a[w].keys() & b[w].keys()} for w in a if w in b}


def _json(e: Estimate) -> dict[str, Any]:
    return {**e.to_json(), "excludes_zero": bool(e.lo > 0 or e.hi < 0)}


def cells(rows: list[Row]) -> dict[str, Any]:
    groups: dict[tuple[str, float, str], list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r["tags"]["mode"], _lag(r), r["system"])].append(r)
    out = {}
    for (mode, lag, system), rs in sorted(groups.items()):
        out[f"{mode}_L{lag:g}:{system}"] = {
            "mode": mode,
            "lag": lag,
            "system": system,
            "B": rs[0]["tags"]["B"],
            **{
                name: world_bootstrap(_by_world(rs, "contract_valid", cond)).to_json()
                for name, cond in SPLITS.items()
            },
        }
    return out


def divergence_table(div: list[Row]) -> dict[str, Any]:
    groups: dict[tuple[str, float], list[Row]] = defaultdict(list)
    for r in div:
        groups[(r["mode"], float(r["lag"]))].append(r)
    out = {}
    for (mode, lag), rs in sorted(groups.items()):
        parts = {
            "all": rs,
            "in_window": [r for r in rs if float(r["delta"]) < lag],
            "outside_window": [r for r in rs if float(r["delta"]) >= lag],
        }
        out[f"{mode}_L{lag:g}"] = {
            "mode": mode,
            "lag": lag,
            **{
                name: world_bootstrap(_by_world(part, "divergence", SPLITS["all"])).to_json()
                for name, part in parts.items()
            },
        }
    return out


def contrasts(rows: list[Row]) -> dict[str, Any]:
    """Per official lag: drop per system, and the B interaction, overall and in the window."""
    lags = sorted({_lag(r) for r in rows if r["tags"]["mode"] == "official"})
    systems = sorted({r["system"] for r in rows})
    ideal = {s: _vec(rows, "ideal", 0.0, s) for s in systems}
    out: dict[str, Any] = {}
    for lag in lags:
        res: dict[str, Any] = {}
        for split in ("all", "in_window", "outside_window", "unsynced"):
            off = {s: _vec(rows, "official", lag, s, SPLITS[split]) for s in systems}
            for s in systems:
                res[f"drop_{s}_{split}"] = _json(paired_diff(off[s], ideal[s]))
            if all(HM[b] in off for b in (0, 1)):
                d1, d0 = (_minus(off[HM[b]], ideal[HM[b]]) for b in (1, 0))
                res[f"interaction_B_{split}"] = _json(paired_diff(d1, d0))
        out[f"L{lag:g}"] = res
    return out


def analyse(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "answers.jsonl")
    div = read_jsonl(run_dir / "divergence.jsonl")
    result = {
        "cells": cells(rows),
        "divergence": divergence_table(div),
        "contrasts": contrasts(rows),
        "n_worlds": len({r["world_id"] for r in rows}),
        "n_probes": len({r["probe_id"] for r in rows}),
    }
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    (out / "e4_metrics.json").write_text(json.dumps(result, indent=1))
    return result

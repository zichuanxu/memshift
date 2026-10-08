"""E4 extension analysis (prereg v1 §4 "E4 扩展"): ContractValid by (stream, system) per source
and pooled, the official - ideal drop per system, and the sync-semantics x B interaction, over all
probes and over touched probes.

Source       the source of the true event that triggered a probe (tag `source`); "pooled" is every
             probe.
Cells        ContractValid per (stream, system), per source and pooled, all and touched probes.
Drop         CV(official, L) - CV(ideal) for one system, paired by probe, clustered by world.
Interaction  Drop(B=1) - Drop(B=0), paired by (world, probe). The calendar value over all probes is
             the pre-registered primary result (replication of the stage-1 E4 interaction); the
             other sources and the pooled value are descriptive.
Divergence   state divergence per (stream, source) over the probe times, and pooled over sources.
All intervals are world-level bootstraps (scoring.stats: 2,000 draws, seed 996).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from memshift.analysis.e4 import HM, Cond, Row, _by_world, _json, _lag, _minus, _vec
from memshift.calsync.sim import ALL_SOURCES
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import paired_diff, world_bootstrap

SCOPES = ("pooled", *ALL_SOURCES)
SUBSETS: dict[str, Cond] = {"all": lambda r: True, "touched": lambda r: bool(r["tags"]["touched"])}
PRIMARY = ("all", "calendar", "interaction_B")


def _cond(subset: str, scope: str) -> Cond:
    keep = SUBSETS[subset]
    return lambda r: keep(r) and scope in ("pooled", r["tags"]["source"])


def _row_in(scope: str) -> Cond:
    return lambda r: scope in ("pooled", r["source"])


def cells(rows: list[Row]) -> dict[str, Any]:
    groups: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r["tags"]["stream"], r["system"])].append(r)
    out = {}
    for (stream, system), rs in sorted(groups.items()):
        out[f"{stream}:{system}"] = {
            "stream": stream,
            "mode": rs[0]["tags"]["mode"],
            "lag": _lag(rs[0]),
            "system": system,
            "B": rs[0]["tags"]["B"],
            **{
                subset: {
                    scope: world_bootstrap(
                        _by_world(rs, "contract_valid", _cond(subset, scope))
                    ).to_json()
                    for scope in SCOPES
                }
                for subset in SUBSETS
            },
        }
    return out


def contrasts(rows: list[Row]) -> dict[str, Any]:
    """Per official lag, subset and source: the drop per system and the B interaction."""
    lags = sorted({_lag(r) for r in rows if r["tags"]["mode"] == "official"})
    systems = sorted({r["system"] for r in rows})
    out: dict[str, Any] = {}
    for lag in lags:
        res: dict[str, Any] = {}
        for subset in SUBSETS:
            res[subset] = {}
            for scope in SCOPES:
                cond = _cond(subset, scope)
                off = {s: _vec(rows, "official", lag, s, cond) for s in systems}
                ideal = {s: _vec(rows, "ideal", 0.0, s, cond) for s in systems}
                c = {f"drop_{s}": _json(paired_diff(off[s], ideal[s])) for s in systems}
                if all(HM[b] in systems for b in (0, 1)):
                    d1, d0 = (_minus(off[HM[b]], ideal[HM[b]]) for b in (1, 0))
                    c["interaction_B"] = _json(paired_diff(d1, d0))
                res[subset][scope] = c
        out[f"L{lag:g}"] = res
    return out


def divergence_table(div: list[Row]) -> dict[str, Any]:
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in div:
        groups[r["stream"]].append(r)
    out = {}
    for stream, rs in sorted(groups.items()):
        out[stream] = {
            "mode": rs[0]["mode"],
            "lag": float(rs[0]["lag"]),
            **{
                scope: world_bootstrap(_by_world(rs, "divergence", _row_in(scope))).to_json()
                for scope in SCOPES
            },
        }
    return out


def analyse(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "answers.jsonl")
    div = read_jsonl(run_dir / "divergence.jsonl")
    con = contrasts(rows)
    probes = {r["probe_id"]: r["tags"] for r in rows}
    subset, scope, name = PRIMARY
    result = {
        "primary": {
            "contrast": f"{name} ({scope}, {subset} probes): [official - ideal]_B=1 - "
            "[official - ideal]_B=0, ContractValid",
            **{lag: c[subset][scope][name] for lag, c in con.items() if name in c[subset][scope]},
        },
        "cells": cells(rows),
        "contrasts": con,
        "divergence": divergence_table(div),
        "n_worlds": len({r["world_id"] for r in rows}),
        "n_probes": len(probes),
        "n_probes_by_source": {
            src: {
                "all": sum(t["source"] == src for t in probes.values()),
                "touched": sum(t["source"] == src and bool(t["touched"]) for t in probes.values()),
            }
            for src in ALL_SOURCES
        },
    }
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    (out / "e4x_metrics.json").write_text(json.dumps(result, indent=1))
    return result

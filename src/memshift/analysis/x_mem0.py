"""Block X of prereg/v4_round3.md: per-item diagnosis of Mem0 and the two interventions.

Reads the runs of `experiments.x_mem0` (probes, answers, `fork_deps.jsonl`, and per store
`ledger.jsonl` and `events.jsonl`). Definitions, as registered before any run:

detection   an item-time (world, branch, item, checkpoint) is detected when at least one UPDATE or
            DELETE event lies after the fork and at or before the checkpoint, on a memory linked
            to a record in the item's gold dependency set at the checkpoint or at the fork. A
            memory is linked to a record (a source object) by the ledger (the unit key
            `object_id#revision` -> the memory ids its `add` returned) and by the event log (the
            unit whose ADD or UPDATE event names the memory). Event times are the simulated times
            of the records being ingested; only the served (extracted) scope counts. Events of
            every origin count: Mem0's own and, in `mem0_det`, the adapter's deletions.
            `mem0_rep` is served from the `mem0` store, so it has that store's events.
m, r, a     Eq. (1) over the changed item-times with an answer (gold value or status differs from
            the branch's gold at t0): m the share not detected, r the ContractValid rate of the
            detected ones, a that of the others; Update = (1 - m) r + m a. Per class (definition
            items `k.`; other: every other item) and branch, world bootstrap (2,000 draws, seed
            996, `ex_e1_classes.world_estimate`).
dominant    of a (class, branch) cell, from the `mem0` arm's point estimates: detection if
            m(1 - a) > (1 - m)(1 - r), else repair (a term with weight 0 is 0, so r undefined at
            m = 1 adds nothing); judged only where the larger of the two losses is >= 5 pp.
effects     each arm's Update minus `mem0`'s, item-paired on the changed item-times both answered,
            world-clustered bootstrap (`stats.paired_diff`, 2,000 draws, seed 996).
X1          in at least 4 of the judged cells the arm that targets the dominant failure
            (`mem0_det` for detection, `mem0_rep` for repair) raises Update more than the other
            arm (point estimates); not judged if fewer than 3 cells are judged.

`support` states what the logs can carry: when the native arm logged no UPDATE or DELETE after
the fork (mem0ai 2.2.1 adds only, see `adapters.native_mem0x`), m = 1 in every `mem0` cell by
construction and r is undefined, so the dominant failure is detection wherever the loss is judged.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from memshift.adapters.native import LEDGER_FILE, Ledger, parse_key
from memshift.adapters.native_mem0x import EVENTS_FILE, MUTATIONS, read_events
from memshift.analysis import e1
from memshift.analysis.ex_e1_classes import CLASS_SETS, item_class, world_estimate
from memshift.experiments.common import read_jsonl
from memshift.gen.model import parse_ts
from memshift.scoring.stats import paired_diff

Row = dict[str, Any]
ARMS = ("mem0", "mem0_det", "mem0_rep")
INTERVENTIONS = ("mem0_det", "mem0_rep")
TARGET = {"detection": "mem0_det", "repair": "mem0_rep"}
BRANCHES = ("src", "def", "orphan")
CLASSES = ("definition", "other")
STATS = ("n", "cv", "det", "cvdet")
MIN_LOSS = 0.05
MIN_JUDGED = 3
MIN_FAVOUR = 4
DRAWS = 2000
SEED = 996


def store_of(arm: str, branch: str) -> str:
    """The store directory an arm is served from (`mem0_rep` shares the `mem0` store)."""
    return f"{branch}_det" if arm == "mem0_det" else branch


class StoreLog:
    """Links (memory id -> source objects) and mutations (time, memory id, event, origin) of one
    store, from its ledger and its event log."""

    def __init__(self, ledger: Mapping[str, Mapping[str, Any]], events: list[Row]):
        self.links: dict[str, set[str]] = defaultdict(set)
        self.mutations: list[tuple[datetime, str, str, str]] = []
        for key, rec in ledger.items():
            p = parse_key(key)
            for mid in rec.get("memory_ids") or []:
                if p is not None and mid:
                    self.links[str(mid)].add(p[0])
        for line in events:
            ts = parse_ts(str(line["ts"]))
            for ev in line.get("events") or []:
                if ev.get("scope") != "extracted":
                    continue
                mid = str(ev.get("memory_id"))
                if ev.get("event") in ("ADD", "UPDATE"):
                    self.links[mid].add(str(line["object_id"]))
                if ev.get("event") in MUTATIONS:
                    self.mutations.append((ts, mid, str(ev["event"]), str(ev.get("origin"))))
        self.mutations.sort()

    @classmethod
    def read(cls, store: Path) -> StoreLog:
        return cls(Ledger(store / LEDGER_FILE).entries, read_events(store / EVENTS_FILE))

    def detected(self, deps: set[str], fork: datetime, as_of: datetime) -> bool:
        """At least one UPDATE / DELETE in (fork, as_of] on a memory linked to one of `deps`."""
        return any(
            fork < ts <= as_of and self.links.get(mid, set()) & deps
            for ts, mid, _, _ in self.mutations
        )

    def counts(self, fork: datetime) -> dict[str, int]:
        """Mutations after the fork by event and origin."""
        out: dict[str, int] = defaultdict(int)
        for ts, _, ev, origin in self.mutations:
            if ts > fork:
                out[f"{origin}/{ev}"] += 1
        return dict(sorted(out.items()))


def load(run_dirs: list[Path]) -> dict[str, Any]:
    data: dict[str, Any] = {"probes": [], "answers": [], "fork_deps": [], "results": []}
    stores: dict[tuple[str, str], StoreLog] = {}
    for d in run_dirs:
        data["probes"] += read_jsonl(d / "probes.jsonl")
        data["answers"] += read_jsonl(d / "answers.jsonl")
        data["fork_deps"] += read_jsonl(d / "fork_deps.jsonl")
        data["results"] += read_jsonl(d / "world_results.jsonl")
        for world_dir in sorted((d / "stores").glob("*_mem0")):
            world = world_dir.name.removesuffix("_mem0")
            for store in sorted(p for p in world_dir.iterdir() if p.is_dir()):
                if store.name in ("prefix", "snapshot"):
                    continue
                stores[(world, store.name)] = StoreLog.read(store)
    data["stores"] = stores
    return data


def item_times(data: Mapping[str, Any]) -> list[Row]:
    """Changed, answered item-times of every arm with their detection flag."""
    probes = {e1._key(p): p for p in data["probes"]}
    fork = {(r["world_id"], r["branch"], r["fact_id"]): r for r in data["fork_deps"]}
    rows = e1.item_table({"probes": data["probes"], "answers": data["answers"], "items": []})
    out = []
    for r in rows:
        if r["system"] not in ARMS or not r["changed"] or r["cv"] is None:
            continue
        w, b, f, k = r["world_id"], r["branch"], r["fact_id"], r["k"]
        p, fd = probes[(w, b, f, k)], fork[(w, b, f)]
        deps = {str(o) for o, _ in p["deps"]} | {str(o) for o, _ in fd["deps"]}
        log = data["stores"].get((w, store_of(r["system"], b)))
        det = bool(log and log.detected(deps, parse_ts(fd["fork"]), parse_ts(p["as_of"])))
        out.append({**r, "class": _cls(f), "detected": det})
    return out


def _cls(fact_id: str) -> str:
    return next(c for c in CLASSES if item_class(fact_id) in CLASS_SETS[c])


def per_world(rows: list[Row], arm: str, branch: str, cls: str) -> np.ndarray:
    """(worlds, STATS) sums over the changed item-times of one cell."""
    acc: dict[str, list[float]] = defaultdict(lambda: [0.0] * len(STATS))
    for r in rows:
        if r["system"] == arm and r["branch"] == branch and r["class"] == cls:
            cv, det = float(r["cv"]), float(r["detected"])
            s = acc[r["world_id"]]
            s[0], s[1], s[2], s[3] = s[0] + 1, s[1] + cv, s[2] + det, s[3] + cv * det
    return np.array([acc[w] for w in sorted(acc)], dtype=float).reshape(-1, len(STATS))


TERMS = {
    "Update": lambda t: t[..., 1] / t[..., 0],
    "m": lambda t: 1 - t[..., 2] / t[..., 0],
    "r": lambda t: t[..., 3] / t[..., 2],
    "a": lambda t: (t[..., 1] - t[..., 3]) / (t[..., 0] - t[..., 2]),
}


def losses(m: float, r: float, a: float) -> dict[str, float]:
    """The two loss terms of 1 - Update = m(1 - a) + (1 - m)(1 - r); a term of weight 0 is 0."""
    det = m * (1 - a) if m > 0 else 0.0
    rep = (1 - m) * (1 - r) if m < 1 else 0.0
    return {"detection": det, "repair": rep}


def dominant(m: float, r: float, a: float) -> dict[str, Any]:
    loss = losses(m, r, a)
    larger = max(loss.values())
    return {
        "loss": loss,
        "dominant": "detection" if loss["detection"] > loss["repair"] else "repair",
        "judged": bool(larger >= MIN_LOSS),
    }


def cv_by_item(rows: list[Row], arm: str, branch: str, cls: str) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["system"] == arm and r["branch"] == branch and r["class"] == cls:
            out[r["world_id"]][f"{r['fact_id']}|{r['k']}"] = float(r["cv"])
    return dict(out)


def x1(cells: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The X1 verdict from the cells' diagnosis and effects (point estimates)."""
    judged = {}
    for name, c in cells.items():
        diag, eff = c.get("diagnosis"), c.get("effects") or {}
        if not diag or not diag["judged"] or any(eff.get(a) is None for a in INTERVENTIONS):
            continue
        target = TARGET[diag["dominant"]]
        other = next(a for a in INTERVENTIONS if a != target)
        judged[name] = {
            "dominant": diag["dominant"],
            "target": target,
            "target_effect": eff[target]["est"],
            "other_effect": eff[other]["est"],
            "favours_target": bool(eff[target]["est"] > eff[other]["est"]),
        }
    n_fav = sum(j["favours_target"] for j in judged.values())
    ok = len(judged) >= MIN_JUDGED
    return {
        "rule": f"judged if >= {MIN_JUDGED} cells are judged; holds if the targeting arm raises "
        f"Update more than the other arm in >= {MIN_FAVOUR} judged cells",
        "cells": judged,
        "n_judged": len(judged),
        "n_favour_target": n_fav,
        "judged": ok,
        "holds": (n_fav >= MIN_FAVOUR) if ok else None,
    }


def analyse(run_dirs: list[Path], out: Path | None = None, draws: int = DRAWS) -> dict[str, Any]:
    data = load(run_dirs)
    rows = item_times(data)
    arms = [a for a in ARMS if any(r["system"] == a for r in rows)]
    cells: dict[str, dict[str, Any]] = {}
    for b in BRANCHES:
        for cls in CLASSES:
            cell: dict[str, Any] = {"branch": b, "class": cls, "arms": {}}
            for arm in arms:
                pw = per_world(rows, arm, b, cls)
                if not len(pw):
                    continue
                n = int(pw[:, 0].sum())
                terms = {k: world_estimate(pw, f, n, draws, SEED) for k, f in TERMS.items()}
                cell["arms"][arm] = {"n": n, "detected": int(pw[:, 2].sum()), **terms}
            if not cell["arms"]:
                continue
            base = cell["arms"].get("mem0")
            if base:
                cell["diagnosis"] = dominant(base["m"]["est"], base["r"]["est"], base["a"]["est"])
                ref = cv_by_item(rows, "mem0", b, cls)
                cell["effects"] = {
                    a: (
                        paired_diff(cv_by_item(rows, a, b, cls), ref, draws, SEED).to_json()
                        if a in cell["arms"]
                        else None
                    )
                    for a in INTERVENTIONS
                }
            cells[f"{b}/{cls}"] = cell
    forks = {(r["world_id"], r["branch"]): parse_ts(r["fork"]) for r in data["fork_deps"]}
    mutations: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for (w, store), log in data["stores"].items():
        branch = store.removesuffix("_det")
        arm = "mem0_det" if store.endswith("_det") else "mem0"
        if (w, branch) in forks:
            for k, v in log.counts(forks[(w, branch)]).items():
                mutations[arm][k] += v
    native_mut = sum(mutations.get("mem0", {}).values())
    res: dict[str, Any] = {
        "status": "registered (prereg/v4_round3.md, block X)",
        "runs": [d.name for d in run_dirs],
        "worlds": sorted({r["world_id"] for r in rows}),
        "arms": arms,
        "prefix_sources": {
            r["world"]["world_id"]: r.get("prefix", {}).get("source") for r in data["results"]
        },
        "anchor": {r["world"]["world_id"]: r.get("anchor") for r in data["results"]},
        "units_without_events": {
            f"{r['world']['world_id']}/{b}/{arm}": len(v.get("units_without_events") or [])
            for r in data["results"]
            for b, br in (r.get("branches") or {}).items()
            for arm, v in br.items()
            if isinstance(v, dict) and "units_without_events" in v
        },
        "mutations_after_fork": {a: dict(v) for a, v in mutations.items()},
        "cells": cells,
        "X1": x1(cells),
        "support": (
            "the mem0 arm logged no UPDATE or DELETE after the fork: m = 1 and r is undefined in "
            "every mem0 cell by construction (an ADD-only pipeline), so the diagnosis can only "
            "name detection"
            if "mem0" in arms and native_mut == 0
            else "the mem0 arm logged UPDATE or DELETE events after the fork"
        ),
    }
    adir = (out or run_dirs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "x_mem0.json").write_text(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    r = analyse([Path(p) for p in a.runs], Path(a.out) if a.out else None)
    print(json.dumps({"X1": r["X1"], "support": r["support"]}, indent=1))

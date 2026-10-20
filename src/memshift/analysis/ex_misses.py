"""Why the cited-cluster key misses changes. Every key starts with `explore_`.

The cited-cluster key moves when an event touches a cluster that holds one of the records the writer
cited. A model's citations cannot include a record that enters an item's scope after the derivation,
however good they are (the phantom problem of predicate locks), so a miss may be the writer citing
too little or the key having nothing to cite. This analysis classifies every changed item-time the
cited key missed, from the world alone (no model) and the citations the writer stored at t0:

  a  a changed record was in the item's gold dependency set at derivation time (`gold_at(t0).deps`)
     and was not among the writer's citations; "changed" is a revision or a deletion of that record
     between t0 and the checkpoint
  b  otherwise, a record that was not in the gold dependency set at derivation time is in it at the
     checkpoint: `b_existing` for a record that already existed at t0 and entered the scope (the
     window or the definition moved), `b_new` for a record created after t0 (an insertion)
  c  otherwise (no record of the dependency set changed and none entered): `c_definition` when
     the definition version differs between t0 and the checkpoint, `c_left_scope` when records
     of the set left it unchanged (a time window or a definition narrowed the scope: no record
     changed at all), `c_other` for the rest

The classes are exclusive (a first, then b, then c); `a_and_b` counts the item-times of class a
that also have an entering record, `version_changed` the item-times of any class whose definition
version differs between t0 and the checkpoint, and `cited_touched` the item-times where a cited
record itself changed although the key did not move (none is expected: a revision moves the
cluster epoch).
Shares are per branch and item class (`ex_e1_classes.CLASS_SETS`), with world-clustered bootstrap
intervals (2,000 draws, seed 996). The worlds are regenerated from their seeds and checked against
the `events_sha256` of the run manifests; the writer's citations are those of the t0 item of the
D=0, B=0 cell (every cell derives its t0 items from the same writer output).

Usage: python -m memshift.analysis.ex_misses OUT --runs RUNS_DIR main:STAGE [rep:STAGE ...]
Writes OUT/analysis/ex_misses.json and OUT/manifest.json (the stage's own runs are found through
its manifest; RUNS_DIR holds the raw runs with items.jsonl).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import read_jsonl
from memshift.gen.gold import gold_at
from memshift.gen.model import Slot, World
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.runinfo import world_record

DRAWS = ex.DRAWS
CITED = "holdmem_D1B1_cited"  # the cited-cluster key at D=1 (D=0 misses the same item-times)
CITED_D0 = "holdmem_D0B1_cited"
WRITER_CELL = "holdmem_D0B0"  # its t0 items carry the writer's citations
COUNTS = (
    "missed",
    "a",
    "b",
    "b_existing",
    "b_new",
    "c",
    "c_definition",
    "c_left_scope",
    "c_other",
    "a_and_b",
    "version_changed",
    "cited_touched",
)
SHARES = COUNTS[1:]


def classify(world: World, log: SourceLog, slot: Slot, k: int, cited: set[str]) -> dict[str, bool]:
    """The class of one missed changed item-time (world, slot, checkpoint k weeks after the fork)
    given the ids of the records the writer cited at t0."""
    t0 = world.fork - timedelta(minutes=1)
    tk = world.fork + timedelta(weeks=k)
    deps0 = dict(gold_at(world, log, slot, t0).deps)
    depsk = dict(gold_at(world, log, slot, tk).deps)
    snap0, snapk = log.snapshot(t0), log.snapshot(tk)
    touched = [
        oid
        for oid, rev in deps0.items()
        if (o := snapk.get(oid)) is None or o.revision != rev  # revised or deleted
    ]
    entered = [oid for oid in depsk if oid not in deps0]
    a = any(oid not in cited for oid in touched)
    b = not a and bool(entered)
    b_new = b and any(snap0.get(oid) is None for oid in entered)
    c = not a and not b
    left = any(oid not in depsk for oid in deps0)
    kid = slot.kappa_id
    changed_version = kid is not None and world.kappa_version(kid, t0) != world.kappa_version(
        kid, tk
    )
    return {
        "a": a,
        "b": b,
        "b_existing": b and not b_new,
        "b_new": b_new,
        "c": c,
        "c_definition": c and changed_version,
        "c_left_scope": c and not changed_version and left,
        "c_other": c and not changed_version and not left,
        "a_and_b": a and bool(entered),
        "version_changed": changed_version,
        "cited_touched": any(oid in cited for oid in touched),
    }


def citations(run_dir: Path) -> dict[tuple[str, str, str], set[str]]:
    """(world, branch, slot) -> the ids of the records the writer cited at t0."""
    out: dict[tuple[str, str, str], set[str]] = {}
    with (run_dir / "items.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if WRITER_CELL not in line or "#0" not in line:
                continue
            r = json.loads(line)
            if r["system"] == WRITER_CELL and r["item_id"].endswith("#0"):
                out[r["world_id"], r["branch"], r["slot_id"]] = {oid for oid, _ in r["refs"]}
    return out


def missed_item_times(tab: ex.Table, system: str) -> dict[tuple[str, str], list[tuple[str, int]]]:
    """(world, branch) -> the sorted (slot, k) of the changed item-times `system` did not
    invalidate."""
    missed: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
    for b in tab.branches:
        for r in tab.rows(system, b):
            if r["changed"] and r.get("cv") is not None and not r.get("detected"):
                missed[r["world_id"], b].add((r["fact_id"], r["k"]))
    return {key: sorted(v) for key, v in missed.items()}


def classify_stage(tab: ex.Table, run_dirs: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    """Every missed changed item-time of one E1 stage (the item table and the runs it merges)."""
    cited: dict[tuple[str, str, str], set[str]] = {}
    hashes: dict[tuple[str, str], str] = {}
    for d in run_dirs:
        cited |= citations(d)
        man = json.loads((d / "manifest.json").read_text())
        hashes |= {(w["world_id"], w["branch"]): w["events_sha256"] for w in man["worlds"]}
    missed = missed_item_times(tab, CITED)
    missed_d0 = missed_item_times(tab, CITED_D0)
    worlds = sorted({w for w, _ in missed})
    pos = {w: i for i, w in enumerate(worlds)}
    arr = np.zeros((len(worlds), len(tab.branches), len(ex.BASE_CLASSES), len(COUNTS)))
    checks = {"world_branches": 0, "events_hash_matches": 0, "missed": 0, "missed_same_at_D0": 0}
    for (w, b), keys in sorted(missed.items()):
        world = generate_branch(int(w[1:]), b, None)
        checks["world_branches"] += 1
        checks["events_hash_matches"] += world_record(world)["events_sha256"] == hashes[w, b]
        checks["missed"] += len(keys)
        checks["missed_same_at_D0"] += len(keys) * (missed_d0.get((w, b)) == keys)
        log = SourceLog(world.events)
        slots = {s.slot_id: s for s in world.slots}
        for slot_id, k in keys:
            flags = {
                "missed": True,
                **classify(world, log, slots[slot_id], k, cited[w, b, slot_id]),
            }
            base = ex.BASE_CLASSES.index(ex.item_class(slot_id))
            for i, name in enumerate(COUNTS):
                arr[pos[w], tab.branches.index(b), base, i] += flags[name]
    out: dict[str, Any] = {"checks": checks, "worlds": len(worlds)}
    for bi, b in enumerate(tab.branches):
        for cls, members in ex.CLASS_SETS.items():
            idx = [ex.BASE_CLASSES.index(c) for c in members]
            pw = arr[:, bi, idx, :].sum(axis=1)
            n = int(pw[:, 0].sum())
            node: dict[str, Any] = {"missed_item_times": n}
            for name in SHARES:
                j = COUNTS.index(name)
                node[name] = ex.world_estimate(pw, _share(j), n, draws)
                node[f"{name}_count"] = int(pw[:, j].sum())
            out.setdefault(b, {})[cls] = node
    return out


def _share(j: int) -> Callable[[np.ndarray], Any]:
    """Count j over the missed item-times, from per-world sums."""
    return lambda t: t[..., j] / t[..., 0]


def stage_runs(stage: Path, runs: Path) -> list[Path]:
    man = json.loads((stage / "manifest.json").read_text())
    return [runs / s["run_id"] for s in man["sources"]]


def analyse(
    stages: Mapping[str, Path], runs: Path, out: Path, draws: int = DRAWS
) -> dict[str, Any]:
    res: dict[str, Any] = {
        "explore_status": ex.STATUS,
        "explore_draws": draws,
        "explore_seed": ex.SEED,
    }
    res["explore_miss_classes"] = {
        role: classify_stage(
            ex.Table(read_jsonl(stage / "analysis" / "e1_item_table.jsonl")),
            stage_runs(stage, runs),
            draws,
        )
        for role, stage in stages.items()
    }
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "ex_misses.json").write_text(json.dumps(res, indent=1))
    write_stage_manifest(out, "ex_misses", list(stages.values()), {})
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--runs", type=Path, required=True, help="directory of the raw runs")
    ap.add_argument("stages", nargs="+", help="role:STAGE_DIR")
    a = ap.parse_args()
    stages = {s.split(":", 1)[0]: Path(s.split(":", 1)[1]) for s in a.stages}
    res = analyse(stages, a.runs, a.out)
    for role, r in res["explore_miss_classes"].items():
        print(role, r["checks"])


if __name__ == "__main__":
    main()

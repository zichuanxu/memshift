"""Offline audit of the E1 binding keys: what each key detects, from the worlds alone (no model).

The source-epoch key, the oracle key (`gold`) and the scope key on definition items are functions of
the world, so their detection is known before any model runs: for every (world, branch, slot,
checkpoint) the audit replays the key with `HoldMem.key_at`, re-deriving (resetting the key) at
each change, and compares detection with the change of the gold at t0, as `analysis.e1.item_table`
does for a run. Reported per branch, key and item class (`analysis.e1.ITEM_CLASSES`):

  changed / unchanged   item-checkpoints whose gold differs from / equals the gold at t0
  missed                share of changed item-checkpoints the key had not invalidated by then
  false_invalidation    share of unchanged item-checkpoints it had invalidated by then
  rederivations         invalidations over the four checkpoints (each one writer request)

The cited key (and the non-definition items of the scope key) depend on what the writer cites, so
only a run measures them. These rates are facts about the keys; a run's rates for the same keys
must equal them (a check of the implementation, not a result).

    python -m memshift.analysis.e1_keys --seeds 1000-1019 --branches src,def,orphan
    python -m memshift.analysis.e1_keys --seeds 1000-1019 --branches src,def,orphan \\
        --out runs/stage5_key_audit

`--out` writes the audit as a stage directory (analysis/e1_keys.json and manifest.json with the
command and seeds; the paper's `KeyAudit` macros and table read it, `analysis.key_audit`) instead
of printing it.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from memshift.analysis.e1 import ITEM_CLASSES
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import world_params
from memshift.experiments.e1 import BRANCHES, OFFSETS_WEEKS, parse_seeds
from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem

KEYS = {
    "source": CellConfig(0, 1, "source"),
    "gold": CellConfig(0, 1, "gold"),
    "scope": CellConfig(0, 1, "scope"),  # definition items only
}


def item_class(slot_id: str) -> str:
    return next(c for c, prefixes in ITEM_CLASSES.items() if slot_id.startswith(prefixes))


def _rates(counts: dict[str, int]) -> dict[str, Any]:
    ch, un = counts["changed"], counts["unchanged"]
    return {
        **counts,
        "missed": 1 - counts["changed_detected"] / ch if ch else None,
        "false_invalidation": counts["unchanged_detected"] / un if un else None,
    }


def audit(seeds: list[int], branches: tuple[str, ...], months: int = 12) -> dict[str, Any]:
    no_writer: Any = None  # keys are functions of the world; nothing is written
    totals: dict[tuple[str, str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for seed in seeds:
        for branch in branches:
            world = generate_branch(seed, branch, world_params(months))
            log = SourceLog(world.events)
            t0 = world.fork - timedelta(minutes=1)
            cells = {
                name: HoldMem(world, log, cfg, no_writer, "audit") for name, cfg in KEYS.items()
            }
            for slot in world.slots:
                cls = item_class(slot.slot_id)
                g0 = gold_at(world, log, slot, t0)
                bound = {name: hm.key_at(slot, [], t0) for name, hm in cells.items()}
                detected = dict.fromkeys(cells, False)
                for k in OFFSETS_WEEKS:
                    t = world.fork + timedelta(weeks=k)
                    g = gold_at(world, log, slot, t)
                    changed = (g.value, g.status) != (g0.value, g0.status)
                    for name, hm in cells.items():
                        key = hm.key_at(slot, [], t)
                        if key != bound[name]:
                            detected[name], bound[name] = True, key
                            if name != "scope" or slot.kappa_id:
                                totals[branch, name, cls]["rederivations"] += 1
                        if name == "scope" and not slot.kappa_id:
                            continue
                        c = totals[branch, name, cls]
                        c["changed" if changed else "unchanged"] += 1
                        c["changed_detected"] += changed and detected[name]
                        c["unchanged_detected"] += (not changed) and detected[name]
    out: dict[str, Any] = {"seeds": seeds, "branches": list(branches), "months": months}
    for (branch, name, cls), counts in sorted(totals.items()):
        for field in ("changed", "unchanged", "changed_detected", "unchanged_detected"):
            counts.setdefault(field, 0)
        counts.setdefault("rederivations", 0)
        out.setdefault(branch, {}).setdefault(name, {})[cls] = _rates(dict(counts))
    return out


def write_stage(out: Path, result: dict[str, Any], command: str, seeds: str) -> None:
    """The audit as a stage directory: no run is behind it, so the manifest lists no sources."""
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "e1_keys.json").write_text(json.dumps(result, indent=1))
    extra = {"command": command, "seeds": seeds, "branches": result["branches"]}
    write_stage_manifest(out, "key_audit", [], extra)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--branches", default=",".join((*BRANCHES, "orphan")))
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--out", type=Path, help="write a stage directory instead of printing")
    a = ap.parse_args()
    res = audit(parse_seeds(a.seeds), tuple(a.branches.split(",")), a.months)
    if a.out is None:
        print(json.dumps(res, indent=1))
    else:
        cmd = f"python -m memshift.analysis.e1_keys --seeds {a.seeds} --branches {a.branches}"
        write_stage(a.out, res, f"{cmd} --months {a.months}", a.seeds)

"""E6 analysis (plan §3.2 T2, prereg v1 §5): the leave test through Portable Memory (.mem).

Each imported store is paired with the native HoldMem (1,1) source-epoch store per probe and
clustered by world. T2 holds for an import when all three differences have their 90% world
bootstrap interval inside +/- T2_MARGIN (TOST, alpha = 0.05):
  ContractValid(import) - ContractValid(native) at t0 (all probes);
  the same difference after S_src and S_def (checkpoints pooled);
  MissedInvalidation(import) - MissedInvalidation(native) (changed items; stores with item
  histories only).
ExportFidelity = ContractValid(import) / ContractValid(native), at t0 and after the shift, is
reported as a descriptive ratio. The import without the derivation extension is predicted to
fail (its detection degrades to B=0); Mem0 arms are reported when present.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.scoring.stats import paired_diff, tost

NATIVE = "native_holdmem_D1B1_source"
T2_MARGIN = 0.03  # prereg v1 §5: epsilon of plan §4.6
Vec = dict[str, dict[str, float]]


def _mean(v: Vec) -> float:
    xs = [x for d in v.values() for x in d.values()]
    return sum(xs) / len(xs) if xs else float("nan")


def vectors(data: dict[str, list[e1.Row]]) -> dict[str, dict[str, Vec]]:
    """system -> measure ('pre', 'post', 'missed') -> world -> item key -> value."""
    out: dict[str, dict[str, Vec]] = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for a in data["answers"]:
        if int(a["tags"].get("k", 0)) == 0:
            key = f"{a['branch']}:{a['fact_id']}"
            out[a["system"]]["pre"][a["world_id"]][key] = float(a["contract_valid"])
    for r in e1.item_table(data):
        key = f"{r['branch']}:{r['fact_id']}@{r['k']}"
        if r["cv"] is not None:
            out[r["system"]]["post"][r["world_id"]][key] = float(r["cv"])
        if r["changed"] and "detected" in r:
            out[r["system"]]["missed"][r["world_id"]][key] = float(not r["detected"])
    return out


def analyse(*run_dirs: Path, out: Path | None = None) -> dict[str, Any]:
    data = e1.load(*run_dirs)
    vec = vectors(data)
    native = vec[NATIVE]
    result: dict[str, Any] = {"native": NATIVE, "margin": T2_MARGIN, "imports": {}}
    for system in sorted(s for s in vec if s != NATIVE):
        imp = vec[system]
        diffs = {
            m: paired_diff(imp[m], native[m], alpha=0.10)
            for m in ("pre", "post", "missed")
            if imp.get(m) and native.get(m)
        }
        result["imports"][system] = {
            "diff_90": {m: d.to_json() for m, d in diffs.items()},
            "measured": sorted(diffs),
            "t2_equivalent": len(diffs) == 3 and all(tost(d, T2_MARGIN) for d in diffs.values()),
            "export_fidelity": {
                m: _mean(imp[m]) / _mean(native[m]) for m in ("pre", "post") if imp.get(m)
            },
        }
    adir = (out or run_dirs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "e6_t2.json").write_text(json.dumps(result, indent=1))
    return result


if __name__ == "__main__":
    import sys

    # python -m memshift.analysis.e6 OUT_DIR RUN_DIR [RUN_DIR ...]
    res = analyse(*(Path(p) for p in sys.argv[2:]), out=Path(sys.argv[1]))
    print(json.dumps(res, indent=1)[:4000])

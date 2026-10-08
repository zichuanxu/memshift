"""E3 analysis (plan §3.2 T3/T4, §6 E3; prereg v0 §6).

Per horizon and system: static ContractValid (world bootstrap), the five cost parts, serving
cost per query relative to one full re-read, and the membership tests
  T3  ContractValid(system) >= ContractValid(oracle) - epsilon (3 pp)
  T4  serving tokens per query <= c (0.5) x full re-read tokens per query.
Exploratory: `explore_no_definition` per system repeats ContractValid and T3 with the definition
items (class `definition` of `e1.ITEM_CLASSES`) left out of both the system and the oracle.
Exploratory (decisions.md 2026-09-30; `ex_tests`): `explore_t3_by_class` gives the T3 margin per
item class (and the oracle's own ContractValid per class), `explore_t4_slack` the factor by which a
system's serving cost is below the T4 ceiling.
The zero-system decision (H_U) needs Update after S_src / S_def from E1; `decide_hu` combines both.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from memshift.analysis import ex_tests
from memshift.analysis.e1 import ITEM_CLASSES
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import paired_diff, world_bootstrap

EPSILON = 0.03
COST_CEILING = 0.5
HM = "holdmem_D1B1_source"
DEFINITION_PREFIXES = ITEM_CLASSES["definition"]


def _costs(results: list[dict[str, Any]], h: int, system: str) -> dict[str, Any] | None:
    rows = [r["horizons"][str(h)] for r in results if str(h) in r["horizons"]]
    costs = [r["costs"][system] for r in rows if system in r["costs"]]
    if not costs:
        return None
    mean = {k: sum(c[k] for c in costs) / len(costs) for k in costs[0]}
    queries = sum(r["queries"] for r in rows) / len(rows)
    reread = sum(r["full_reread_tokens_per_query"] for r in rows) / len(rows)
    per_query = mean["serve_tokens"] / queries
    return {
        "cost_mean": mean,
        "serve_tokens_per_query": per_query,
        "full_reread_tokens_per_query": reread,
        "serve_ratio_to_full_reread": per_query / reread,
        "T4_pass": per_query / reread <= COST_CEILING,
    }


def analyse(run_dir: Path) -> dict[str, Any]:
    answers = read_jsonl(run_dir / "answers.jsonl")
    results = read_jsonl(run_dir / "world_results.jsonl")
    cv: dict[tuple[int, str], dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for a in answers:
        h = int(a["tags"]["horizon_months"])
        cv[(h, a["system"])][a["world_id"]][a["fact_id"]] = float(a["contract_valid"])
    out: dict[str, Any] = {"horizons": {}}
    for h in sorted({h for h, _ in cv}):
        systems = sorted(s for hh, s in cv if hh == h)
        per: dict[str, Any] = {}
        for s in systems:
            est = world_bootstrap({w: list(v.values()) for w, v in cv[(h, s)].items()})
            per[s] = {"contract_valid": est.to_json(), **(_costs(results, h, s) or {})}
        if "oracle" in per:
            for s in systems:
                if s != "oracle":
                    d = paired_diff(cv[(h, s)], cv[(h, "oracle")]).to_json()
                    per[s]["minus_oracle"] = d
                    per[s]["T3_pass"] = d["est"] >= -EPSILON
        if "zero" in per and HM in per:
            per["zero_minus_holdmem"] = paired_diff(cv[(h, "zero")], cv[(h, HM)]).to_json()
        # Exploratory (decisions.md 2026-09-28): ContractValid by item class; D=1's programmatic
        # extraction coincides with the gold computation on the rule and definition classes.
        for s in systems:
            per[s]["explore_by_class"] = {
                cls: world_bootstrap(
                    {
                        w: [x for f, x in v.items() if f.startswith(prefixes)]
                        for w, v in cv[(h, s)].items()
                    }
                ).to_json()
                for cls, prefixes in ITEM_CLASSES.items()
            }
        # Exploratory (not pre-registered): ContractValid and T3 without the definition items, on
        # which the oracle context is weak (the reader must evaluate a definition over every record
        # it ranges over) while D=1 computes them exactly; the same items in system and oracle.
        kept = {
            s: {
                w: {f: x for f, x in v.items() if not f.startswith(DEFINITION_PREFIXES)}
                for w, v in cv[(h, s)].items()
            }
            for s in systems
        }
        for s in systems:
            entry: dict[str, Any] = {
                "contract_valid": world_bootstrap(
                    {w: list(v.values()) for w, v in kept[s].items()}
                ).to_json()
            }
            if s != "oracle" and "oracle" in kept:
                d = paired_diff(kept[s], kept["oracle"]).to_json()
                entry.update({"minus_oracle": d, "T3_pass": d["est"] >= -EPSILON})
            per[s]["explore_no_definition"] = entry
        # Exploratory (decisions.md 2026-09-30): the T3 margin per item class, and how far each
        # system's serving cost is from the T4 ceiling
        for s, by_class in ex_tests.t3_by_class({s: cv[(h, s)] for s in systems}, EPSILON).items():
            per[s]["explore_t3_by_class"] = by_class
            per[s]["explore_t4_slack"] = ex_tests.t4_slack(
                per[s].get("serve_ratio_to_full_reread"), COST_CEILING
            )
        out["horizons"][h] = per
    adir = run_dir / "analysis"
    adir.mkdir(exist_ok=True)
    (adir / "e3_metrics.json").write_text(json.dumps(out, indent=1, default=str))
    return out


def decide_hu(e3: dict[str, Any], e1_update: dict[str, float]) -> dict[str, Any]:
    """Zero system "meets the bar" (prereg v0 §6): (a) static CV at 12 months within 3 pp of
    HoldMem(1,1); (b) Update after S_src and S_def within 3 pp (from E1); (c) serving cost per
    query <= 0.5 x full re-read. `e1_update` maps branch -> Update(zero) - Update(HoldMem)."""
    h12 = e3["horizons"].get(12) or e3["horizons"].get("12") or {}
    a = h12.get("zero_minus_holdmem", {}).get("est", float("-inf")) >= -EPSILON
    b = bool(e1_update) and all(v >= -EPSILON for v in e1_update.values())
    c = bool(h12.get("zero", {}).get("T4_pass", False))
    return {"a_static": a, "b_update": b, "c_cost": c, "H_U_holds": a and b and c}


if __name__ == "__main__":
    import sys

    print(json.dumps(analyse(Path(sys.argv[1])), indent=1, default=str)[:6000])

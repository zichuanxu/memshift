"""Block M2 (prereg/v2.md §2): is the two-factor pattern a property of the layer or of one model
family? For an E1 stage run with another family as writer and reader, the pre-registered checks:

(i)   path 2: the D effect on missed invalidation is not triggered at either granularity or branch;
(ii)  B effect on Update (source epoch, D pooled) on S_src > +2 pp, 95% interval excluding 0;
(iii) D effect on repair correctness (source epoch) > 0 on S_src and on S_def (interval > 0);
(iv)  B x D interaction on Update (source epoch) on S_def > 0 (interval > 0);
(v)   after S_def, the zero system's ContractValid is below HoldMem(1,1)'s: item-paired difference
      zero - HoldMem(1,1) over the asked post-shift probes, world bootstrap, interval below 0.
The family verdict ("two-factor pattern holds") is (ii)-(iv) all true.

Usage: python -m memshift.analysis.families OUT LABEL=E1_STAGE [LABEL=E1_STAGE ...]
(each stage directory holds analysis/e1_metrics.json and analysis/e1_item_table.jsonl)
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from memshift.analysis.gate import PATH1_MIN, path2
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import paired_diff

HM11 = "holdmem_D1B1_source"


def _above(est: dict[str, Any], floor: float = 0.0) -> bool:
    return bool(est["est"] > floor and est["lo"] > 0)


def zero_hm_vectors(
    items: list[dict[str, Any]], branch: str = "def"
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
    """Post-shift ContractValid of the zero system and of HoldMem(1,1): world -> item -> value."""
    vec: dict[str, dict[str, dict[str, float]]] = {
        "zero": defaultdict(dict),
        HM11: defaultdict(dict),
    }
    for r in items:
        if r["branch"] != branch or int(r["k"]) == 0 or r.get("cv") is None:
            continue
        if r["system"] in vec:
            vec[r["system"]][r["world_id"]][f"{r['fact_id']}@{r['k']}"] = float(r["cv"])
    return vec["zero"], vec[HM11]


def zero_minus_hm(items: list[dict[str, Any]], branch: str = "def") -> dict[str, Any]:
    zero, hm = zero_hm_vectors(items, branch)
    return paired_diff(zero, hm).to_json()


def family_checks(e1: dict[str, Any], items: list[dict[str, Any]]) -> dict[str, Any]:
    c = e1["contrasts"]
    checks = {
        "i_path2_not_triggered": not path2(e1)["triggered"],
        "ii_B_update_src": _above(c["src"]["B_effect_update_pooled_source"], PATH1_MIN),
        "iii_D_repair_src": _above(c["src"]["D_effect_repair_source"]),
        "iii_D_repair_def": _above(c["def"]["D_effect_repair_source"]),
        "iv_BxD_def": _above(c["def"]["BxD_interaction_update_source"]),
    }
    zmh = zero_minus_hm(items)
    checks["v_zero_below_hm11_def"] = bool(zmh["hi"] < 0)
    core = ("ii_B_update_src", "iii_D_repair_src", "iii_D_repair_def", "iv_BxD_def")
    return {
        "checks": checks,
        "pattern_holds": all(checks[k] for k in core),
        "estimates": {
            "B_update_src": c["src"]["B_effect_update_pooled_source"],
            "D_repair_src": c["src"]["D_effect_repair_source"],
            "D_repair_def": c["def"]["D_effect_repair_source"],
            "BxD_def": c["def"]["BxD_interaction_update_source"],
            "BxD_src": c["src"]["BxD_interaction_update_source"],
            "zero_minus_hm11_def": zmh,
        },
        "n_worlds": e1.get("n_worlds"),
    }


def analyse(out: Path, stages: dict[str, Path]) -> dict[str, Any]:
    """`family_checks` of every labelled E1 stage directory; writes OUT/analysis/families.json."""
    res: dict[str, Any] = {}
    for label, stage in stages.items():
        d = stage / "analysis"
        res[label] = family_checks(
            json.loads((d / "e1_metrics.json").read_text()), read_jsonl(d / "e1_item_table.jsonl")
        )
    res["families_holding"] = sum(v["pattern_holds"] for v in res.values() if isinstance(v, dict))
    Path(out, "analysis").mkdir(parents=True, exist_ok=True)
    Path(out, "analysis", "families.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    res = analyse(
        Path(sys.argv[1]),
        {spec.split("=", 1)[0]: Path(spec.split("=", 1)[1]) for spec in sys.argv[2:]},
    )
    print(json.dumps({k: v["pattern_holds"] for k, v in res.items() if isinstance(v, dict)}))

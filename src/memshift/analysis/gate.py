"""Gate A decision (plan §7, prereg v1 §3, §5, §6), computed mechanically from committed analyses.

Path 1  B effect on Update (source epoch, D pooled) on S_src: triggered when the point estimate
        is <= +0.02.
Path 2  D effect on MissedInvalidation at either granularity on S_src, S_def or orphan:
        triggered when |estimate| >= 0.02 and the 95% interval excludes 0.
Path 3  pooled calibration over the writer tiers: triggered when the slope's 90% lower bound is
        <= 0 or AUROC < 0.60 (from `analysis.e2`).
Claim level from the table of prereg v1 §6. Membership tests T1 (per tier, from E2), T3 and T4
(per system and horizon, from E3) are reported beside the claim level, not folded into it.

Usage: python -m memshift.analysis.gate E1_STAGE E2_STAGE E3_STAGE OUT_DIR
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PATH1_MIN = 0.02
PATH2_MIN = 0.02


def path1(e1: dict[str, Any]) -> dict[str, Any]:
    est = e1["contrasts"]["src"]["B_effect_update_pooled_source"]
    return {"estimate": est, "triggered": bool(est["est"] <= PATH1_MIN)}


def path2(e1: dict[str, Any]) -> dict[str, Any]:
    hits: dict[str, Any] = {}
    for branch, cons in e1["contrasts"].items():
        for key, est in cons.items():
            if not key.startswith("D_effect_missed_"):
                continue
            excl = est["lo"] > 0 or est["hi"] < 0
            hits[f"{branch}:{key}"] = {
                "estimate": est,
                "triggered": bool(abs(est["est"]) >= PATH2_MIN and excl),
            }
    return {"checks": hits, "triggered": any(h["triggered"] for h in hits.values())}


def path3(e2: dict[str, Any]) -> dict[str, Any]:
    p = e2["path3"]
    return {
        "pooled_slope": p["pooled_slope"],
        "pooled_auroc": p["pooled_auroc"],
        "tiers": p["tiers"],
        "tier_caveats": p["tier_caveats"],
        "triggered": bool(p["triggered"]),
    }


def claim_level(p1: bool, p2: bool, p3: bool) -> str:
    """prereg v1 §6 (path 2 is taken as unrepaired here; a repair re-runs E1 first)."""
    if p1 and p2 and p3:
        return "measurement methodology + zero-system conclusion"
    if p2:
        return "measurement methodology"
    if p1:
        return "repair single factor (D)"
    if p3:
        return "two-factor theory, qualitative predictions"
    return "two-factor theory + quantitative prediction"


def membership(e2: dict[str, Any], e3: dict[str, Any]) -> dict[str, Any]:
    t3t4: dict[str, Any] = {}
    for h, per in e3["horizons"].items():
        for s, v in per.items():
            if isinstance(v, dict) and "contract_valid" in v:
                t3t4[f"{h}m:{s}"] = {"T3_pass": v.get("T3_pass"), "T4_pass": v.get("T4_pass")}
    return {
        "T1": {
            t: {"equivalent": v["equivalent"], "delta_90": v["delta_90"]}
            for t, v in e2["t1"].items()
        },
        "T3_T4": t3t4,
    }


def gate_a(e1: dict[str, Any], e2: dict[str, Any], e3: dict[str, Any]) -> dict[str, Any]:
    p1, p2, p3 = path1(e1), path2(e1), path3(e2)
    return {
        "path1": p1,
        "path2": p2,
        "path3": p3,
        "claim_level": claim_level(p1["triggered"], p2["triggered"], p3["triggered"]),
        "membership": membership(e2, e3),
        "n_worlds_e1": e1.get("n_worlds"),
    }


if __name__ == "__main__":
    import sys

    s1, s2, s3, out = (Path(p) for p in sys.argv[1:5])
    res = gate_a(
        json.loads((s1 / "analysis" / "e1_metrics.json").read_text()),
        json.loads((s2 / "analysis" / "e2_metrics.json").read_text()),
        json.loads((s3 / "analysis" / "e3_metrics.json").read_text()),
    )
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "gate_a.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("claim_level", "n_worlds_e1")}, indent=1))
    print(json.dumps({k: res[k]["triggered"] for k in ("path1", "path2", "path3")}, indent=1))

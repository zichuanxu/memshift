"""Block R: the unchanged stage-2 analyses re-estimated on 20 fresh test worlds.

Each E1 contrast of `e1.contrasts` is estimated on the replication worlds alone and compared with
the stage-2 estimate:
- effects predicted non-zero (B on detection and on Update, D on repair, the B x D interaction),
  `replication`: "replicated" = same sign and the 95% interval excludes 0; "same direction" = same
  sign, interval includes 0; "not replicated" = opposite sign;
- the D effect on missed invalidation (predicted zero): `within_threshold` of path 2;
- T1 (per tier) and T3 (per system and horizon): replicated when the outcome is the same;
- the three paths of `analysis.gate` on the replication worlds (`within_threshold`);
- every other contrast (false invalidations, granularity, JSON arm) is reported side by side.

Usage: python -m memshift.analysis.replication ORIG_E1 ORIG_E2 ORIG_E3 REP_E1 REP_E2 REP_E3 OUT
(stage directories, each with analysis/e{1,2,3}_metrics.json)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from memshift.analysis.gate import PATH2_MIN, gate_a

PREDICTED_NONZERO = (
    "B_effect_detected_",
    "B_effect_update_D1_",
    "B_effect_update_pooled_",
    "D_effect_repair_source",
    "D_effect_repair_cited",
    "D_effect_repair_mech_",
    "BxD_interaction_update_",
)


def replication_class(orig: dict[str, Any], rep: dict[str, Any]) -> str:
    """The replicate estimate of an effect predicted non-zero, in the direction of the stage-2
    estimate: "replicated", "same direction" or "not replicated"."""
    sign = 1 if orig["est"] >= 0 else -1
    if sign * rep["est"] <= 0:
        return "not replicated"
    excludes = rep["lo"] > 0 if sign > 0 else rep["hi"] < 0
    return "replicated" if excludes else "same direction"


def in_predicted_direction(kind: str, est: dict[str, Any] | None) -> bool | None:
    """Is one stage's estimate in the predicted direction? Every effect predicted non-zero is
    predicted positive, so the point estimate must be > 0 (this is not `replication_class`, which
    compares R with the sign of the stage-2 estimate); the D effect on missed invalidation is
    predicted zero: within the threshold of path 2. None for the descriptive contrasts, which
    carry no prediction, and for a missing estimate."""
    if not est or est["est"] != est["est"]:  # missing or NaN
        return None
    if kind == "effect":
        return bool(est["est"] > 0)
    if kind == "path2":
        return path2_within_threshold(est)
    return None


def path2_within_threshold(rep: dict[str, Any]) -> bool:
    """Path 2 of `analysis.gate` on one estimate: not (|est| >= 2 pp, interval excluding 0)."""
    excl = rep["lo"] > 0 or rep["hi"] < 0
    return not (abs(rep["est"]) >= PATH2_MIN and excl)


def contrast_rows(orig_e1: dict[str, Any], rep_e1: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for branch, cons in orig_e1["contrasts"].items():
        for key, o in cons.items():
            r = rep_e1["contrasts"].get(branch, {}).get(key)
            if r is None or key.startswith("explore_"):
                continue
            kind = (
                "path2"
                if key.startswith("D_effect_missed_")
                else "effect"
                if key.startswith(PREDICTED_NONZERO)
                else "descriptive"
            )
            rows.append(
                {
                    "branch": branch,
                    "contrast": key,
                    "kind": kind,
                    "orig": o,
                    "rep": r,
                    "replication": replication_class(o, r) if kind == "effect" else None,
                    "within_threshold": path2_within_threshold(r) if kind == "path2" else None,
                }
            )
    return rows


def membership_rows(
    orig_e2: dict[str, Any], rep_e2: dict[str, Any], orig_e3: dict[str, Any], rep_e3: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for tier, o in orig_e2["t1"].items():
        r = rep_e2["t1"].get(tier)
        if r is not None:
            rows.append(
                {
                    "test": "T1",
                    "unit": tier,
                    "orig": o["equivalent"],
                    "rep": r["equivalent"],
                    "rep_delta_90": r["delta_90"],
                    "replication": (
                        "replicated" if o["equivalent"] == r["equivalent"] else "differs"
                    ),
                }
            )
    for h, per in orig_e3["horizons"].items():
        for system, o in per.items():
            r = rep_e3["horizons"].get(h, {}).get(system)
            if not isinstance(o, dict) or not isinstance(r, dict) or "T3_pass" not in o:
                continue
            rows.append(
                {
                    "test": "T3",
                    "unit": f"{h}m:{system}",
                    "orig": o["T3_pass"],
                    "rep": r.get("T3_pass"),
                    "rep_minus_oracle": r.get("minus_oracle"),
                    "replication": "replicated" if o["T3_pass"] == r.get("T3_pass") else "differs",
                }
            )
    return rows


def replicate(
    orig: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    rep: tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
) -> dict[str, Any]:
    contrasts = contrast_rows(orig[0], rep[0])
    membership = membership_rows(orig[1], rep[1], orig[2], rep[2])
    effects = [c for c in contrasts if c["kind"] == "effect"]
    rep_gate = gate_a(*rep)
    return {
        "contrasts": contrasts,
        "membership": membership,
        "paths_within_threshold": {
            "path1": rep_gate["path1"]["within_threshold"],
            "path2": rep_gate["path2"]["within_threshold"],
            "path3": rep_gate["path3"]["within_threshold"],
        },
        "summary": {
            "effects": len(effects),
            "replicated": sum(c["replication"] == "replicated" for c in effects),
            "same_direction": sum(c["replication"] == "same direction" for c in effects),
            "not_replicated": sum(c["replication"] == "not replicated" for c in effects),
            "path2_outside_threshold": sum(c["within_threshold"] is False for c in contrasts),
            "membership_same": sum(m["replication"] == "replicated" for m in membership),
            "membership": len(membership),
            "n_worlds_rep": rep[0].get("n_worlds"),
        },
    }


def _load(stage: Path, n: int) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((stage / "analysis" / f"e{n}_metrics.json").read_text())
    return data


if __name__ == "__main__":
    *dirs, out = (Path(p) for p in sys.argv[1:8])
    o = (_load(dirs[0], 1), _load(dirs[1], 2), _load(dirs[2], 3))
    r = (_load(dirs[3], 1), _load(dirs[4], 2), _load(dirs[5], 3))
    res = replicate(o, r)
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "replication.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({**res["summary"], **res["paths_within_threshold"]}, indent=1))

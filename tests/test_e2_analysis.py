import json
import math
import random

from memshift.analysis.e2 import (
    T1_SYSTEM,
    _logit,
    analyse,
    calibration_curve,
    logistic_fit,
    logistic_slope,
    slope_on_logit,
)


def _rows(tier: str, lost_every: int) -> list[dict]:
    """10 worlds x 20 items; D=1 items lose one item in `lost_every`; D=0 survival tracks p."""
    rows = []
    for w in range(10):
        for i in range(20):
            lost = lost_every and i % lost_every == 0
            rows.append(
                {
                    "tier": tier,
                    "system": T1_SYSTEM,
                    "world_id": f"w{w}",
                    "slot_id": f"s{i}",
                    "D": 1,
                    "B": 1,
                    "valid_before": True,
                    "valid_after": not lost,
                    "predicted": None,
                    "type": "t",
                }
            )
            p = 0.9 if i % 2 else 0.3
            rows.append(
                {
                    "tier": tier,
                    "system": "holdmem_D0B1_source",
                    "world_id": f"w{w}",
                    "slot_id": f"s{i}",
                    "D": 0,
                    "B": 1,
                    "valid_before": True,
                    "valid_after": i % 2 == 1 or i % 4 == 0,
                    "predicted": p,
                    "type": f"t{i % 2}",
                }
            )
    return rows


def test_t1_tost_and_pooled_path3(tmp_path):
    rows = _rows("same", 0) + _rows("upgrade", 5)
    (tmp_path / "survival.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = analyse(tmp_path)
    # no loss: delta is exactly 0, inside +/- 3 pp; one in five lost: -20 pp, outside
    assert res["t1"]["same"]["equivalent"]
    assert not res["t1"]["upgrade"]["equivalent"]
    assert abs(res["t1"]["upgrade"]["delta_90"]["est"] + 0.2) < 1e-9
    # higher agreement -> higher survival in both tiers: slope > 0, path 3 within its threshold
    assert res["path3"]["tiers"] == ["same", "upgrade"]
    assert res["path3"]["pooled_slope"]["est"] > 0
    assert res["path3"]["within_threshold"]
    assert res["path3"]["tier_caveats"] == []
    # M1-b: D=1 survival minus D=0 survival, paired by item (D=0 keeps 15 of 20 items)
    m1b = res["m1b_d_protects_switch"]
    assert abs(m1b["same"]["d1_minus_d0_survival"]["est"] - 0.25) < 1e-9
    assert abs(m1b["upgrade"]["d1_minus_d0_survival"]["est"] - 0.05) < 1e-9
    assert m1b["same"]["est_at_least_zero"] and m1b["upgrade"]["est_at_least_zero"]
    # T1 for every cell: the primary T1 cell is the same computation, and
    # HoldMem(0,1) (D=0 keeps 15 of 20 items) is 25 pp below retention 1, outside the margin
    cells = res["explore_t1_cells"]
    assert cells["same"][T1_SYSTEM] == res["t1"]["same"]
    assert cells["upgrade"][T1_SYSTEM] == res["t1"]["upgrade"]
    d0b1 = cells["same"]["holdmem_D0B1_source"]
    assert abs(d0b1["delta_90"]["est"] + 0.25) < 1e-9 and not d0b1["equivalent"]
    assert d0b1["margin"] == res["t1"]["same"]["margin"]


def test_gate_paths_compare_each_estimate_with_its_threshold():
    from memshift.analysis import gate

    zero = {"est": 0.0, "lo": 0.0, "hi": 0.0}
    moved = {"est": 0.03, "lo": 0.01, "hi": 0.05}
    e1 = {
        "contrasts": {
            "src": {"B_effect_update_pooled_source": dict(moved), "D_effect_missed_source": zero},
            "def": {"D_effect_missed_cited": moved},
        }
    }
    assert gate.path1(e1)["within_threshold"]  # +3 pp is above +2 pp
    p2 = gate.path2(e1)
    assert p2["checks"]["src:D_effect_missed_source"]["within_threshold"]
    assert not p2["checks"]["def:D_effect_missed_cited"]["within_threshold"]
    assert not p2["within_threshold"]
    e1["contrasts"]["src"]["B_effect_update_pooled_source"]["est"] = 0.02
    assert not gate.path1(e1)["within_threshold"]


def test_cross_family_tier_tokens():
    from memshift.experiments.e2 import MAIN_MODEL, tier_models, tier_name

    tok = "cross:meta-llama/Llama-3.1-8B-Instruct"
    assert tier_models(tok) == (MAIN_MODEL, "meta-llama/Llama-3.1-8B-Instruct")
    assert tier_name(tok) == "cross_meta-llama_Llama-3.1-8B-Instruct"
    assert tier_name("same") == "same" and tier_models("same")[0] == MAIN_MODEL


def _worlds() -> dict[str, list[tuple[float, float]]]:
    """Twelve worlds of items whose survival rises with the agreement rate p, with noise."""
    rng = random.Random(7)
    out = {}
    for w in range(12):
        pts = []
        for _ in range(40):
            p = rng.choice([0.3, 0.6, 0.83, 0.97, 1.0])
            pts.append((p, float(rng.random() < 0.15 + 0.85 * p)))
        out[f"w{w}"] = pts
    return out


def test_calibration_curve_is_the_fit_behind_the_reported_slope():
    by_world = _worlds()
    curve = calibration_curve(by_world, draws=300)
    pts = [p for w in sorted(by_world) for p in by_world[w]]
    a, b = logistic_fit([_logit(p) for p, _ in pts], [y for _, y in pts])
    assert curve["slope"] == b == slope_on_logit([p for p, _ in pts], [y for _, y in pts])
    assert curve["slope"] == logistic_slope([_logit(p) for p, _ in pts], [y for _, y in pts])
    assert curve["intercept"] == a and curve["n_items"] == 480 and curve["n_worlds"] == 12
    grid = curve["grid"]
    assert len(grid) == len(curve["est"]) == len(curve["lo90"]) == len(curve["hi90"])
    for g, est, lo, hi in zip(grid, curve["est"], curve["lo90"], curve["hi90"], strict=True):
        assert est == 1 / (1 + math.exp(-(a + b * _logit(g))))
        assert 0 < lo <= hi < 1  # a proper band, not a point
    assert curve["est"] == sorted(curve["est"])  # survival rises with agreement (b > 0)
    assert any(hi - lo > 0.01 for lo, hi in zip(curve["lo90"], curve["hi90"], strict=True))
    assert calibration_curve(by_world, draws=300) == curve  # deterministic


def test_calibration_curve_without_points_is_missing_not_zero():
    curve = calibration_curve({}, draws=10)
    assert math.isnan(curve["slope"]) and curve["n_items"] == 0
    assert all(math.isnan(v) for v in curve["est"] + curve["lo90"] + curve["hi90"])


def test_analyse_stores_the_pooled_curve_beside_path_3(tmp_path):
    rows = _rows("same", 0) + _rows("upgrade", 5)
    (tmp_path / "survival.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = analyse(tmp_path)
    curve = res["explore_pooled_curve"]
    assert curve["slope"] == res["path3"]["pooled_slope"]["est"]
    assert curve["n_items"] == res["path3"]["pooled_slope"]["n_items"]

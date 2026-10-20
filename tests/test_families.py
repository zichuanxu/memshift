import json
from pathlib import Path

from memshift.analysis.families import HM11, family_checks

E1 = Path(__file__).resolve().parents[1] / "runs" / "stage2_e1" / "analysis" / "e1_metrics.json"


def _items(zero_cv: int, hm_cv: int) -> list[dict]:
    rows = []
    for w in ("w1", "w2", "w3"):
        for f in ("a", "b", "c"):
            for system, cv in (("zero", zero_cv), (HM11, hm_cv)):
                rows.append(
                    {
                        "world_id": w,
                        "branch": "def",
                        "system": system,
                        "fact_id": f,
                        "k": 1,
                        "cv": cv,
                    }
                )
    return rows


def test_stage2_main_family_passes_all_checks():
    res = family_checks(json.loads(E1.read_text()), _items(0, 1))
    assert res["ii_to_iv_true"]
    assert all(res["checks"].values())
    assert res["estimates"]["zero_minus_hm11_def"]["est"] == -1.0


def test_zero_not_below_fails_check_v_only():
    res = family_checks(json.loads(E1.read_text()), _items(1, 1))
    assert res["ii_to_iv_true"] and not res["checks"]["v_zero_below_hm11_def"]

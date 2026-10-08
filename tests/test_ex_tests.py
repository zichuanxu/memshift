"""Exploratory T1 and T3 sensitivity (`analysis.ex_tests`): survival, recovery and the net change,
the eps grid, T1 and T3 by item class, wired into `e2.analyse` and `e3.analyse`."""

import json
from pathlib import Path

import pytest

from memshift.analysis import e2, e3, ex_tests
from memshift.scoring.stats import paired_diff, tost, world_bootstrap
from tests.test_e3_analysis import _analyse as analyse_e3
from tests.test_e3_analysis import _answers as answers_e3

WORLDS = [f"w{i}" for i in range(10)]
# slot id, valid before, valid after: a class each for rule, free text and definition items
ITEMS = [(f"c.x{i}", True, True) for i in range(6)] + [
    ("u.employer", True, False),  # lost
    ("u.manager", True, True),
    ("k.active_project", False, True),  # gained
    ("s.series", False, False),
]


def _rows(tier: str = "same", system: str = "holdmem_D1B1_source", frozen: bool = False):
    return [
        {
            "tier": tier,
            "system": system,
            "world_id": w,
            "slot_id": slot,
            "valid_before": before,
            "valid_after": before if frozen else after,
        }
        for w in WORLDS
        for slot, before, after in ITEMS
    ]


def test_survival_recovery_and_net_change_of_one_cell():
    sr = ex_tests.survival_recovery(_rows())
    assert sr["n_items"] == 100 and sr["n_valid_before"] == 80
    assert sr["gains"] == 10 and sr["losses"] == 10 and sr["stay_valid"] == 70
    assert sr["stay_invalid"] == 10 and not sr["trivial"]
    assert sr["valid_before"]["est"] == pytest.approx(0.8)
    assert sr["valid_after"]["est"] == pytest.approx(0.8)
    assert sr["survival"]["est"] == pytest.approx(70 / 80)  # the registered retention
    assert sr["recovery"]["est"] == pytest.approx(0.5)  # one of the two items wrong before
    assert sr["net"]["est"] == pytest.approx(0.0)  # a gain for every loss
    frozen = ex_tests.survival_recovery(_rows(frozen=True))
    assert frozen["trivial"] and frozen["survival"]["est"] == 1.0
    assert frozen["net"]["est"] == 0.0 and frozen["net"]["lo"] == frozen["net"]["hi"] == 0.0


def test_the_registered_retention_is_the_same_computation():
    rows = _rows() + _rows(tier="upgrade", system="holdmem_D0B0", frozen=True)
    out = ex_tests.t1_sensitivity(rows, 0.03)
    sr = out["explore_survival_recovery"]["same:holdmem_D1B1_source"]
    by_w = {w: [1.0] * 7 + [0.0] for w in WORLDS}  # 7 of the 8 valid-before items stay valid
    want = world_bootstrap({w: [v - 1 for v in xs] for w, xs in by_w.items()}, alpha=0.10)
    got = sr["retention_delta"]
    assert (got["est"], got["lo"], got["hi"]) == pytest.approx((want.est, want.lo, want.hi))
    # every (tier, system) present is covered, the trivial cell included
    assert set(out["explore_survival_recovery"]) == {
        "same:holdmem_D1B1_source",
        "upgrade:holdmem_D0B0",
    }
    assert out["explore_survival_recovery"]["upgrade:holdmem_D0B0"]["trivial"]


def test_eps_grid_gives_non_inferiority_and_equivalence_for_retention_and_net_change():
    rows = _rows()  # retention -12.5 pp, net change 0
    grid = ex_tests.t1_sensitivity(rows, 0.03)["explore_t1_eps"]["same:holdmem_D1B1_source"]
    assert grid["margin"] == 0.03 and set(grid["by_eps_pp"]) == {"1", "2", "3", "5"}
    for v in grid["by_eps_pp"].values():
        assert not v["retention_noninferior"] and not v["retention_equivalent"]
        assert v["net_noninferior"] and v["net_equivalent"]  # net 0 [0, 0] inside every margin
    # a cell that only gains: equivalent retention, net change above every margin
    gain = [
        {
            "tier": "t",
            "system": "s",
            "world_id": w,
            "slot_id": f"c.x{i}",
            "valid_before": i < 6,
            "valid_after": True,
        }
        for w in WORLDS
        for i in range(10)
    ]
    g = ex_tests.t1_sensitivity(gain, 0.03)["explore_t1_eps"]["t:s"]
    for v in g["by_eps_pp"].values():
        assert v["retention_noninferior"] and v["retention_equivalent"]
        assert v["net_noninferior"] and not v["net_equivalent"]  # +40 pp is not "no change"
    assert g["net_90"]["est"] == pytest.approx(0.4)
    # the verdicts follow the world-bootstrap interval and the same strict inequalities as TOST
    d = world_bootstrap({w: [0.0] * 5 + [-1.0] for w in WORLDS}, alpha=0.10)
    for e in (1, 2, 3, 5):
        assert tost(d, e / 100) == (-e / 100 < d.lo and d.hi < e / 100)


def test_t1_by_item_class_shows_where_a_margin_hides_a_loss():
    cls = ex_tests.t1_sensitivity(_rows(), 0.03)["explore_t1_by_class"]["same:holdmem_D1B1_source"]
    assert set(cls) == {"rule", "definition", "norule"}
    free, rule, definition = cls["norule"], cls["rule"], cls["definition"]
    assert free["n_valid_before"] == 20 and free["share_of_valid_before"] == pytest.approx(0.25)
    assert free["retention_delta_90"]["est"] == pytest.approx(-0.5)
    assert not free["equivalent"] and not free["noninferior"]
    # a loss of a fraction l inside the class moves the pooled delta by -share * l: 0.03 / 0.25
    assert free["tolerated_loss_at_margin"] == pytest.approx(0.12)
    assert rule["retention_delta_90"]["est"] == 0.0 and rule["equivalent"]
    assert rule["share_of_valid_before"] == pytest.approx(0.75)
    assert rule["tolerated_loss_at_margin"] == pytest.approx(0.04)
    # the definition items were all wrong before: no valid-before item, nothing to retain
    assert definition["n_valid_before"] == 0 and definition["tolerated_loss_at_margin"] is None
    assert definition["net_90"]["est"] == pytest.approx(1.0) and not definition["equivalent"]
    # the class shares add up to the valid-before items
    assert sum(c["n_valid_before"] for c in cls.values()) == 80


def test_e2_analyse_stores_the_t1_fields_and_keeps_the_registered_t1(tmp_path: Path):
    from tests.test_e2_analysis import _rows as e2_rows

    rows = e2_rows("same", 0) + e2_rows("upgrade", 5)
    (tmp_path / "survival.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    res = e2.analyse(tmp_path)
    eps = res["explore_t1_eps"]["upgrade:" + e2.T1_SYSTEM]
    assert eps["retention_delta_90"]["est"] == pytest.approx(
        res["t1"]["upgrade"]["delta_90"]["est"]
    )
    assert eps["by_eps_pp"]["3"]["retention_equivalent"] == res["t1"]["upgrade"]["equivalent"]
    assert res["explore_survival_recovery"]["same:" + e2.T1_SYSTEM]["trivial"]  # nothing lost
    assert res["explore_t1_by_class"]["same:" + e2.T1_SYSTEM]["rule"]["n_valid_before"] == 0


# ------------------------------------------------------------------------------------------- T3
def _cv(rates: dict[str, tuple[float, float, float]]) -> dict:
    """system -> world -> fact -> ContractValid; ten worlds, 4 rule, 2 definition, 2 free-text
    items, the rate of each class given as (rule, definition, free text)."""
    facts = {
        "rule": ["c.a", "c.b", "c.c", "c.d"],
        "definition": ["k.a", "k.b"],
        "norule": ["u.employer", "u.manager"],
    }
    out: dict = {}
    for system, (r, d, f) in rates.items():
        out[system] = {}
        for w in range(10):
            per = {}
            for cls, rate in (("rule", r), ("definition", d), ("norule", f)):
                for i, fact in enumerate(facts[cls]):
                    per[fact] = float((w + i) % 10 < round(rate * 10))
            out[system][f"w{w}"] = per
    return out


def test_t3_margin_per_class_contributions_and_oracle_ceiling():
    cv = _cv({"oracle": (0.9, 0.3, 1.0), "hold": (0.8, 1.0, 0.6), "zero": (0.9, 0.3, 0.8)})
    by = ex_tests.t3_by_class(cv, 0.03)
    assert set(by) == {"oracle", "hold", "zero"}
    hold = by["hold"]
    assert hold["rule"]["minus_oracle"]["est"] == pytest.approx(-0.1)
    assert hold["definition"]["minus_oracle"]["est"] == pytest.approx(0.7)
    assert hold["norule"]["minus_oracle"]["est"] == pytest.approx(-0.4)
    assert [hold[c]["T3_pass"] for c in ("rule", "definition", "norule")] == [False, True, False]
    assert hold["rule"]["T3_pass_at_eps_pp"] == {"1": False, "2": False, "3": False, "5": False}
    assert hold["definition"]["T3_pass_at_eps_pp"]["1"]
    # the oracle's own accuracy per class
    assert by["oracle"]["definition"]["contract_valid"]["est"] == pytest.approx(0.3)
    assert "minus_oracle" not in by["oracle"]["rule"]
    # the contributions add up to the pooled margin of the registered T3
    per_world_items = 8
    pooled = paired_diff(
        {w: cv["hold"][w] for w in cv["hold"]}, {w: cv["oracle"][w] for w in cv["oracle"]}
    )
    assert sum(hold[c]["contribution"] for c in hold) == pytest.approx(pooled.est)
    assert hold["rule"]["share_of_items"] == pytest.approx(4 / per_world_items)
    # pooled, the definition class carries a pass that the other classes do not support
    assert pooled.est >= -0.03 and not hold["rule"]["T3_pass"] and not hold["norule"]["T3_pass"]
    # the zero system keeps pace with the oracle on rule and definition items
    assert by["zero"]["rule"]["minus_oracle"]["est"] == pytest.approx(0.0)
    assert by["zero"]["rule"]["T3_pass"] and by["zero"]["norule"]["minus_oracle"]["est"] < 0


def test_t3_by_class_without_an_oracle_has_no_margin():
    by = ex_tests.t3_by_class(_cv({"zero": (0.9, 0.3, 0.8)}), 0.03)
    assert by["zero"]["rule"]["contract_valid"]["est"] == pytest.approx(0.9)
    assert "minus_oracle" not in by["zero"]["rule"]


def test_e3_analyse_adds_the_class_margins_and_keeps_the_registered_t3(tmp_path):
    per = analyse_e3(
        tmp_path, answers_e3({"oracle": (0.3, 0.9), "holdmem_D1B1_source": (1.0, 0.8)})
    )
    hold = per["holdmem_D1B1_source"]
    assert hold["minus_oracle"]["est"] == pytest.approx(0.3) and hold["T3_pass"]
    assert hold["explore_t3_by_class"]["rule"]["minus_oracle"]["est"] == pytest.approx(-0.1)
    assert hold["explore_t3_by_class"]["definition"]["minus_oracle"]["est"] == pytest.approx(0.7)
    assert set(hold["explore_t3_by_class"]) == {"rule", "definition", "norule"}
    assert hold["explore_t4_slack"] is None  # no cost rows in the fixture


def test_t4_slack_is_the_factor_below_the_ceiling():
    assert ex_tests.t4_slack(0.01, 0.5) == pytest.approx(50.0)
    assert ex_tests.t4_slack(None, 0.5) is None and ex_tests.t4_slack(0.0, 0.5) is None
    assert e3.COST_CEILING == 0.5

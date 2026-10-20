"""Further E1 analyses (`analysis.ex_e1_more`): the out-of-sample baselines agree with the
earlier test where they overlap, the pooled cell mean and the missed-item and zero-system contrasts
are checked by hand on tables small enough to add up, and the committed stage analysis holds the
numbers recomputed by hand."""

import json
from pathlib import Path

import numpy as np
import pytest

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_e1_cross as cross
from memshift.analysis import ex_e1_more as em
from tests.test_ex_e1_classes import CV, DET, FACTS, _rows

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "runs/stage5_ex_more/analysis/ex_e1_more.json"
FOUR = ("holdmem_D0B0", "holdmem_D1B0", "holdmem_D0B1_source", "holdmem_D1B1_source")


def _table3(worlds: int, branches=("src", "def", "orphan"), cv=CV, det=DET) -> ex.Table:
    return ex.Table(r for b in branches for w in range(worlds) for r in _rows(f"w{w}", cv, det, b))


def _same(a: float, b: float) -> bool:
    return a == b or (np.isnan(a) and np.isnan(b))


# ------------------------------------------------------------------------------ baselines
def test_pooled_cell_mean_by_hand():
    tab = _table3(3)
    w, arr, names = ex.cell_arrays(tab, "src", "D1", em.cross.GRANS)
    pred = em._predict(arr.sum(axis=0), em.CELLS6, tuple(em.PREDICTOR_LABELS))
    # D1B0: cv [1, 1, 1, 0] over the four facts: pooled Update 3/4, whatever the class
    assert np.allclose(pred["pooled_cell_mean"]["D1B0"], 0.75)
    assert np.allclose(pred["pooled_cell_mean"]["D0B0"], 0.25)
    assert np.allclose(pred["pooled_cell_mean"]["D1B1_cited"], 0.5)
    # the class terms differ by class: definition 1, rule 1, free text 0 at D1B0
    assert list(pred["cell"]["D1B0"]) == [1, 1, 0]
    assert set(pred) == set(em.PREDICTOR_LABELS)


def test_baselines_agree_with_the_earlier_out_of_sample_test():
    fit, target = _table3(3), _table3(4)
    mine = em.transfer(fit, target, em.CELLS6, tuple(em.PREDICTOR_LABELS), draws=40)
    theirs = cross.oos(fit, target, draws=40)
    for b in ("src", "def", "orphan"):
        for cls in em.CLASSES:
            for p in ("cell", "structural", "literal"):
                a, t = mine["branches"][b][cls][p], theirs["branches"][b][cls][p]
                assert a["mean_abs_error"] == t["mean_abs_error"]
                assert a["cells_within_5pp"] == t["cells_within_5pp"]
                for cell in em.CELLS6:
                    assert a["cells"][cell]["error"] == t["cells"][cell]["error"]
    for cls in em.CLASSES:
        for p in ("cell", "structural", "literal"):
            a = mine["pooled_over_branches"][cls][p]
            t = theirs["pooled_over_branches"][cls][p]
            assert a["mean_abs_error"] == t["mean_abs_error"]
            assert a["cells_within_5pp"] == t["cells_within_5pp"]
    # the pooled cell mean is the earlier test's "cell" predictor at the fit worlds' own mix
    for b in ("src", "def", "orphan"):
        a = mine["branches"][b]["all"]["pooled_cell_mean"]
        t = theirs["branches"][b]["all_fit_mix"]["cell"]
        assert a["mean_abs_error"]["est"] == pytest.approx(t["mean_abs_error"]["est"])
        for cell in em.CELLS6:
            assert a["cells"][cell]["predicted"] == pytest.approx(t["cells"][cell]["predicted"])


def test_class_blind_baseline_fails_where_the_class_terms_do_not():
    worse = {**CV, "holdmem_D1B0": [1, 1, 1, 0]}
    fit, target = _table3(3), _table3(3, cv=worse)
    res = em.transfer(fit, target, em.CELLS6, tuple(em.PREDICTOR_LABELS), draws=20)["branches"]
    d = res["src"]["definition"]
    # definition items at D0B0 are 0 in the fit, and the pooled mean over all classes is 1/4
    assert d["pooled_cell_mean"]["cells"]["D0B0"]["predicted"] == pytest.approx(0.25)
    assert d["cell"]["cells"]["D0B0"]["predicted"] == 0
    assert d["pooled_cell_mean"]["mean_abs_error"]["est"] > d["cell"]["mean_abs_error"]["est"]
    assert (
        res["src"]["all"]["pooled_cell_mean"]["mean_abs_error"]["est"] == 0
    )  # same mix, same cells


def test_horizon_transfer_uses_the_four_source_cells():
    cv4 = {k: v for k, v in CV.items() if k in FOUR}
    det4 = {k: v for k, v in DET.items() if k in FOUR}
    tables = {"main": _table3(3), "short": _table3(2, cv=cv4, det=det4)}
    res = em.horizon_transfer(tables, draws=20)
    assert res["cells"] == list(em.CELLS4) and set(res["predictors"]) == {
        "cell",
        "pooled_cell_mean",
    }
    node = res["main_to_short"]["branches"]["src"]["definition"]["cell"]
    assert set(node["cells"]) == set(em.CELLS4) and node["n_cells"] == 4
    assert node["mean_abs_error"]["est"] == 0 and node["cells_within_5pp"] == 4
    assert em.horizon_transfer({"main": tables["main"]}) == {}
    # a short world where D=0 re-derives right: the error is seen in the D0B1 cell
    better = {**cv4, "holdmem_D0B1_source": [1, 1, 1, 1]}
    short = _table3(2, cv=better, det=det4)
    err = em.horizon_transfer({"main": tables["main"], "short": short}, draws=20)
    cell = err["main_to_short"]["branches"]["src"]["definition"]["cell"]["cells"]["D0B1_source"]
    assert cell["predicted"] == 0 and cell["observed"] == 1 and cell["error"]["est"] == 1


# ------------------------------------------------------------------------------ missed items
def test_missed_items_by_hand():
    res = em.missed_items(_table3(3, branches=("src",)), draws=20)["src"]
    d1 = res["D1"]
    # cited key misses k.b (definition) and u.employer (free text); facts: k.a k.b c.s1 u.employer
    assert (
        d1["definition"]["changed_item_times"] == 6 and d1["definition"]["missed_item_times"] == 3
    )
    assert d1["definition"]["missed_share"]["est"] == 0.5
    assert d1["definition"]["a_missed_B1_cited"]["est"] == 0  # k.b is wrong in the cited cell
    assert d1["definition"]["a_missed_B0"]["est"] == 1  # the B=0 cell re-derives k.b right
    assert d1["definition"]["a_missed_B1_minus_B0"]["est"] == -1
    assert (
        d1["definition"]["update_B1_cited"]["est"] == 0.5
        and d1["definition"]["update_B0"]["est"] == 1
    )
    assert d1["definition"]["B_effect_cited"]["est"] == -0.5
    assert d1["definition"]["r_invalidated_B1_cited"]["est"] == 1
    assert d1["other"]["missed_item_times"] == 3 and d1["other"]["a_missed_B1_minus_B0"]["est"] == 0
    assert d1["all"]["changed_item_times"] == 12 and d1["all"]["missed_item_times"] == 6
    d0 = res["D0"]["definition"]
    assert d0["update_B1_cited"]["est"] == 0 and d0["update_B0"]["est"] == 0
    # an undefined accuracy (no missed item) is NaN, and it stays so in the bootstrap
    full = {**DET, "holdmem_D1B1_cited": [True] * 4, "holdmem_D0B1_cited": [True] * 4}
    none = em.missed_items(_table3(3, branches=("src",), det=full), draws=20)["src"]["D1"]
    assert np.isnan(none["definition"]["a_missed_B1_cited"]["est"])
    assert none["definition"]["missed_share"]["est"] == 0


def test_missed_items_need_both_cells():
    tab = ex.Table(r for r in _table3(2, branches=("src",)).rows("holdmem_D1B1_cited", "src"))
    assert em.missed_items(tab, draws=10) == {}


# ------------------------------------------------------------------------------- zero system
def _with_zero(cv_zero: list[int], worlds: int = 3) -> ex.Table:
    rows = [r for w in range(worlds) for r in _rows(f"w{w}", CV, DET, "src")]
    zero = {"zero": cv_zero}
    det_zero = {"zero": [False] * 4}
    for w in range(worlds):
        for r in _rows(f"w{w}", zero, det_zero, "src"):
            r.pop("detected"), r.pop("repaired_now")
            rows.append(r)
            if r["fact_id"] == "u.employer":  # an unchanged, asked item of the free-text class
                rows.append({**r, "fact_id": "u.manager", "changed": False, "cv": True})
    return ex.Table(rows)


def test_zero_system_update_stability_and_paired_differences_by_hand():
    tab = _with_zero([0, 1, 1, 1])
    assert "u.manager" not in FACTS  # the unchanged item added above is outside the four facts
    res = em.zero_system(tab, draws=20)["src"]
    a = res["all"]
    # zero: changed cv [0,1,1,1] -> 3/4; HoldMem(1,1) all 1; HoldMem(0,0) [0,0,1,0] -> 1/4
    assert a["update"]["zero"]["est"] == 0.75 and a["update"]["holdmem_D1B1_source"]["est"] == 1
    assert a["update"]["holdmem_D0B0"]["est"] == 0.25
    assert a["zero_minus"]["holdmem_D1B1"]["update"]["est"] == pytest.approx(-0.25)
    assert a["zero_minus"]["holdmem_D0B0"]["update"]["est"] == pytest.approx(0.5)
    d = res["definition"]["update"]
    assert d["zero"]["est"] == 0.5 and d["holdmem_D1B1_source"]["est"] == 1
    assert res["definition"]["zero_minus"]["holdmem_D1B1"]["update"]["est"] == pytest.approx(-0.5)
    # stability: only the free-text class has an unchanged item here; no cell answered it
    assert res["norule"]["stability"]["zero"]["est"] == 1
    assert np.isnan(res["definition"]["stability"]["zero"]["est"])
    assert set(res["all"]["update"]) == {
        "zero",
        "holdmem_D0B0",
        "holdmem_D1B0",
        "holdmem_D0B1_source",
        "holdmem_D1B1_source",
    }


def test_zero_system_without_a_zero_system_has_no_differences():
    res = em.zero_system(_table3(2, branches=("src",)), draws=10)["src"]["all"]
    assert res["zero_minus"] == {} and "zero" not in res["update"]


# ------------------------------------------------------------------------------- end to end
def test_analyse_tables_and_reproducibility():
    tables = {"dev": _table3(3), "main": _table3(3), "rep": _table3(2), "short": _table3(2)}
    res = em.analyse_tables(tables, draws=20)
    assert set(res["explore_oos_baselines"]) >= {"dev_to_main", "dev_to_rep", "main_to_rep"}
    assert "explore_horizon_transfer" in res
    assert set(res["explore_missed_items"]) == {"main", "rep"}
    assert set(res["explore_zero_system"]) == {"main", "rep"}
    assert json.dumps(res) == json.dumps(em.analyse_tables(tables, draws=20))  # NaN-safe
    assert "explore_horizon_transfer" not in em.analyse_tables({"dev": tables["dev"]}, draws=10)


def test_load_tables_reads_the_item_tables_of_the_stages(tmp_path):
    for name in ("d1", "d2", "m"):
        (tmp_path / name / "analysis").mkdir(parents=True)
        rows = [r for w in range(2) for r in _rows(f"{name}{w}", CV, DET, "src")]
        (tmp_path / name / "analysis" / "e1_item_table.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n"
        )
    tables = em.load_tables(
        {"dev": [tmp_path / "d1"], "devorphan": [tmp_path / "d2"], "main": [tmp_path / "m"]}
    )
    assert set(tables) == {"dev", "main"}
    assert len(tables["dev"].worlds) == 4 and len(tables["main"].worlds) == 2


# ------------------------------------------------------------------- the committed analysis
committed = pytest.mark.skipif(not STAGE.exists(), reason="stage not staged")


@committed
def test_committed_baselines_are_the_numbers_a_reviewer_recomputed():
    res = json.loads(STAGE.read_text())["explore_oos_baselines"]["dev_to_main"][
        "pooled_over_branches"
    ]
    pooled = res["all"]["pooled_cell_mean"]
    assert (
        round(100 * pooled["mean_abs_error"]["est"], 1) == 2.1 and pooled["cells_within_5pp"] == 17
    )
    assert (
        res["all"]["cell"]["cells_within_5pp"] == 16
        and res["all"]["structural"]["cells_within_5pp"] == 15
    )
    assert res["all"]["literal"]["cells_within_5pp"] == 10
    # class-blind, the baseline cannot follow the classes
    assert res["definition"]["pooled_cell_mean"]["mean_abs_error"]["est"] > 0.15


@committed
def test_committed_baselines_reproduce_the_staged_out_of_sample_test():
    mine = json.loads(STAGE.read_text())["explore_oos_baselines"]
    theirs = json.loads((ROOT / "runs/stage5_ex_e1/analysis/ex_e1.json").read_text())["explore_oos"]
    for pair in ("dev_to_main", "dev_to_rep", "main_to_rep"):
        for cls in em.CLASSES:
            for p in ("cell", "structural", "literal"):
                a = mine[pair]["pooled_over_branches"][cls][p]
                t = theirs[pair]["pooled_over_branches"][cls][p]
                assert a["mean_abs_error"]["est"] == pytest.approx(t["mean_abs_error"]["est"])
                assert a["cells_within_5pp"] == t["cells_within_5pp"]


@committed
def test_committed_horizon_transfer_fails_by_pp_beyond_five():
    res = json.loads(STAGE.read_text())["explore_horizon_transfer"]["main_to_short"]
    assert res["branches"]["src"]["all"]["cell"]["cells_within_5pp"] == 4
    assert res["branches"]["def"]["all"]["cell"]["cells_within_5pp"] <= 1
    assert res["pooled_over_branches"]["all"]["cell"]["mean_abs_error"]["est"] > 0.05


@committed
def test_committed_missed_items_and_zero_system_are_the_reviewers_numbers():
    res = json.loads(STAGE.read_text())

    def pct(e):
        return round(100 * e["est"], 1)

    miss = res["explore_missed_items"]["main"]
    # the changed items a cited key did not invalidate, B=1 against B=0 (D=1)
    assert pct(miss["src"]["D1"]["definition"]["a_missed_B1_cited"]) == 3.8
    assert pct(miss["src"]["D1"]["other"]["a_missed_B1_cited"]) == 1.0
    assert pct(miss["def"]["D1"]["other"]["a_missed_B1_cited"]) == 0.0
    assert pct(miss["src"]["D1"]["definition"]["update_B0"]) == 65.8
    assert pct(miss["src"]["D1"]["other"]["update_B0"]) == 85.9
    assert pct(miss["def"]["D1"]["other"]["update_B0"]) == 71.3
    assert [pct(miss[b]["D1"]["other"]["B_effect_cited"]) for b in ("src", "def")] == [-3.4, -14.0]
    assert [
        pct(miss[b]["D1"]["definition"]["B_effect_cited"]) for b in ("src", "def", "orphan")
    ] == [
        6.6,
        9.8,
        15.9,
    ]
    zero = res["explore_zero_system"]["main"]
    assert pct(zero["orphan"]["all"]["update"]["zero"]) == 71.3
    assert pct(zero["orphan"]["all"]["update"]["holdmem_D1B1_source"]) == 73.1
    assert pct(zero["orphan"]["other"]["update"]["zero"]) == 85.9
    assert pct(zero["orphan"]["other"]["update"]["holdmem_D1B1_source"]) == 67.5
    assert pct(zero["def"]["other"]["update"]["zero"]) == 70.2
    assert pct(zero["orphan"]["all"]["stability"]["zero"]) == 64.8
    assert pct(zero["orphan"]["all"]["stability"]["holdmem_D1B1_source"]) == 88.4

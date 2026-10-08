"""Exploratory test of the account across writers and item classes (`analysis.ex_account`): the cell
terms are the staged Eq. (1) decomposition, the identity holds, the fit statistics are checked by
hand on cells that add up, and the committed stage analysis is the one the paper reads."""

import json
import math
import random
from pathlib import Path

import numpy as np
import pytest

from memshift.analysis import ex_account as ea
from memshift.analysis import ex_e1_classes as ex

ROOT = Path(__file__).resolve().parents[1]
FACTS = ("k.a", "k.b", "k.c", "c.s1", "c.s2", "c.s3", "u.employer", "u.manager")
CELLS = ("holdmem_D0B0", "holdmem_D1B0", "holdmem_D0B1_source", "holdmem_D1B1_source")


def _table(seed: int, worlds: int = 4, branches=("src", "def", "orphan"), detect: bool = True):
    """Random ContractValid in the four source-epoch cells; with `detect` the key invalidates
    every changed item (m = 0), as the source-epoch key does."""
    rng = random.Random(seed)
    rows = []
    for w in range(worlds):
        for b in branches:
            for cell in CELLS:
                for fact in FACTS:
                    definition = fact.startswith("k.")
                    p = 0.9 if cell.endswith("D1B1_source") else 0.15 if definition else 0.6
                    b1 = "B1" in cell
                    rows.append(
                        {
                            "world_id": f"w{w}",
                            "branch": b,
                            "system": cell,
                            "fact_id": fact,
                            "vtype": "city",
                            "k": 1,
                            "changed": True,
                            "cv": rng.random() < p,
                            "cv_pre": True,
                            "detected": b1 and (detect or rng.random() < 0.8),
                            "repaired_now": b1,
                        }
                    )
    return ex.Table(rows)


def _same(a, b):
    return a == b or (isinstance(a, float) and math.isnan(a) and math.isnan(b))


# -------------------------------------------------------------------------------- cell terms
def test_cell_terms_are_the_staged_decomposition():
    tab = _table(1)
    cells = ea.cell_nodes(tab, draws=40)
    dec = ex.decomposition(tab, draws=40)["D1"]
    for b in ea.BRANCHES:
        for cls in ea.CELL_CLASSES:
            mine, theirs = cells[b][cls], dec[b][cls]
            src = theirs["cells"]["source"]
            it = theirs["interaction"]["source"]
            pairs = {
                "U_D0B0": src["D0B0"]["U"],
                "U_D1B1": src["D1B1"]["U"],
                "r0": src["D0B1"]["r"],
                "r1": src["D1B1"]["r"],
                "m": src["D1B1"]["m"],
                "a0": src["D0B0"]["U"],
                "a1": src["D1B0"]["U"],
                "BxD": it["observed"],
                "B_D0": it["B_effect_D0"],
                "B_D1": it["B_effect_D1"],
                "D_B0": it["D_effect_B0"],
                "D_B1": it["D_effect_B1"],
                "corrected": it["corrected_formula"],
            }
            for k, want in pairs.items():
                for f in ("est", "lo", "hi", "n_worlds"):
                    assert _same(mine[k][f], want[f]), (b, cls, k, f)
            assert mine["changed_item_times"] == src["D1B1"]["n"]


def test_repair_gap_is_r1_minus_r0_and_the_identity_holds_without_misses():
    cells = ea.cell_nodes(_table(2), draws=40)
    for b in ea.BRANCHES:
        for cls in ea.CELL_CLASSES:
            n = cells[b][cls]
            assert n["m"]["est"] == 0 and n["m_D0"]["est"] == 0  # the source key detects all
            assert n["gap"]["est"] == pytest.approx(n["r1"]["est"] - n["r0"]["est"])
            # with m = 0: B x D = (r1 - a1) - (r0 - a0) = gap - D|B0
            assert n["BxD"]["est"] == pytest.approx(n["corrected"]["est"])
            assert n["BxD"]["est"] == pytest.approx(n["gap"]["est"] - n["D_B0"]["est"])
            assert n["a0"]["est"] == n["U_D0B0"]["est"] and n["a1"]["est"] == n["U_D1B0"]["est"]


def test_other_is_field_plus_free_text():
    cells = ea.cell_nodes(_table(3), draws=20)["src"]
    n_other = cells["other"]["changed_item_times"]
    assert n_other == cells["rule"]["changed_item_times"] + cells["norule"]["changed_item_times"]
    assert cells["definition"]["changed_item_times"] == 4 * 3  # worlds x definition facts


def test_misses_enter_the_terms_as_they_should():
    cells = ea.cell_nodes(_table(4, detect=False), draws=20)
    assert any(cells[b]["other"]["m"]["est"] > 0 for b in ea.BRANCHES)
    n = cells["src"]["other"]
    # with misses the identity needs the missed terms: the corrected formula no longer equals B x D
    assert abs(n["BxD"]["est"] - n["corrected"]["est"]) > 1e-9


# ----------------------------------------------------------------------------------- the fit
def _cells(x, y, cls=None, wr=None, br=None):
    n = len(x)
    return {
        "x": np.asarray(x, float),
        "y": np.asarray(y, float),
        "corr": np.asarray(y, float),
        "cls": np.asarray(cls if cls is not None else [0] * n),
        "br": np.asarray(br if br is not None else [0] * n),
        "wr": np.asarray(wr if wr is not None else [0] * n),
    }


EMPTY_HELD = _cells([], [])


def test_ols_and_predictors_by_hand():
    assert ea._ols(np.array([0.0, 1.0, 2.0]), np.array([1.0, 3.0, 5.0])) == pytest.approx(
        (2.0, 1.0)
    )
    assert all(math.isnan(v) for v in ea._ols(np.array([1.0, 1.0, 1.0]), np.array([1.0, 2.0, 3.0])))
    tr = _cells([0, 1, 2, 3], [0, 1, 4, 5], cls=[0, 0, 1, 1])
    te = _cells([10], [0], cls=[1], br=[2])
    assert ea._predict("grand_mean", tr, te)[0] == pytest.approx(2.5)
    assert ea._predict("class_mean", tr, te)[0] == pytest.approx(4.5)
    assert ea._predict("class_branch_mean", tr, te)[0] == pytest.approx(4.5)  # falls back to class
    assert ea._predict("gap_identity", tr, te)[0] == 10
    s, i = ea._ols(tr["x"], tr["y"])
    assert ea._predict("gap_linear", tr, te)[0] == pytest.approx(i + 10 * s)
    # a class the training cells lack has no class mean
    assert math.isnan(ea._predict("class_mean", tr, _cells([1], [0], cls=[2]))[0])


def test_fit_statistics_on_an_exact_line_and_on_a_class_effect():
    x = np.array([0.1, 0.4, 0.2, 0.9, 0.3, 0.7, 0.5, 0.8])
    wr = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    fit = ea.fit_statistics(_cells(x, 0.5 * x + 0.1, wr=wr), EMPTY_HELD)
    assert fit["spearman"] == pytest.approx(1.0) and fit["slope"] == pytest.approx(0.5)
    assert fit["intercept"] == pytest.approx(0.1)
    assert fit["lowo_mae"]["gap_linear"] == pytest.approx(0, abs=1e-12)
    assert fit["lowo_mae"]["class_mean"] > 0.01  # one class: no better than the mean
    # y set by the class alone, x unrelated: the class label predicts, the gap does not
    cls = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y = np.where(cls == 0, 0.6, 0.0)
    fit2 = ea.fit_statistics(_cells(x, y, cls=cls, wr=wr), EMPTY_HELD)
    assert fit2["lowo_mae"]["class_mean"] == pytest.approx(0, abs=1e-12)
    assert fit2["lowo_mae"]["gap_linear"] > 0.05
    assert fit2["lowo_mae_by_class"][0]["class_mean"] == pytest.approx(0, abs=1e-12)
    assert set(fit2["lowo_mae_by_writer"]) == {0, 1, 2, 3}
    # cells with an undefined value are dropped
    bad = _cells(
        np.append(x, np.nan), np.append(y, 0.3), cls=np.append(cls, 0), wr=np.append(wr, 3)
    )
    assert ea.fit_statistics(bad, EMPTY_HELD)["n_cells"] == 8


def test_held_out_cells_are_predicted_from_the_fit_not_in_it():
    x = np.linspace(0, 1, 8)
    wr = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    held = _cells([0.25, 0.75], [0.35, 0.85], wr=[-1, -1])
    fit = ea.fit_statistics(_cells(x, x + 0.1, wr=wr), held)
    # y = x + 0.1 is the line: held cells (x, y) = (.25, .35) and (.75, .85) are on it
    assert fit["heldout_mae_all_fitted"]["gap_linear"] == pytest.approx(0.0, abs=1e-12)
    # training without writer 0 (the main study) still recovers the same line
    assert fit["heldout_mae_other_lineages"]["gap_linear"] == pytest.approx(0.0, abs=1e-12)
    assert fit["heldout_mae_all_fitted"]["gap_identity"] == pytest.approx(0.1)


# ---------------------------------------------------------------------------- end to end
def _tables(draws_seed: int = 10):
    return {
        w: _table(draws_seed + i, worlds=3 + (w in ("main", "rep")))
        for i, (w, _) in enumerate(ea.WRITERS)
    }


def test_analyse_tables_end_to_end_and_reproducible():
    res = ea.analyse_tables(_tables(), draws=30)
    assert set(res["explore_account_writers"]) == {w for w, _ in ea.WRITERS}
    assert (
        res["explore_account_writers"]["rep"]["fitted"] is False
        and res["explore_account_writers"]["gemma"]["fitted"] is True
    )
    assert (
        res["explore_account_writers"]["main"]["worlds"] == 4
        and res["explore_account_writers"]["llama"]["worlds"] == 3
    )
    fit = res["explore_account_fit"]
    assert fit["two_class"]["n_cells"] == 4 * 2 * 3 and fit["three_class"]["n_cells"] == 4 * 3 * 3
    for scheme in fit.values():
        assert scheme["spearman"]["lo"] <= scheme["spearman"]["hi"]
        assert -1 <= scheme["spearman"]["est"] <= 1
        assert set(scheme["lowo"]["mae"]) == set(ea.PREDICTORS)
        assert scheme["lowo"]["mae"]["corrected"]["est"] == pytest.approx(0, abs=1e-9)
        lowo = scheme["lowo"]
        assert lowo["mae_gap_minus_class"]["est"] == pytest.approx(
            lowo["mae"]["gap_linear"]["est"] - lowo["mae"]["class_mean"]["est"]
        )
        assert set(lowo["by_class_mae_gap_minus_class"]) == set(lowo["by_class"])
        assert "heldout_all_fitted_mae_gap_minus_class" in scheme
        assert len(scheme["heldout_cells"]) == scheme["n_cells"] // 4
        assert len(scheme["fit_line"]["grid"]) == len(scheme["fit_line"]["lo"]) == 25
    assert all(c["writer"] != "rep" for c in fit["two_class"]["cells"])
    assert json.dumps(res) == json.dumps(ea.analyse_tables(_tables(), draws=30))  # NaN-safe


def test_the_fit_needs_all_fitted_writers_and_r():
    tables = _tables()
    del tables["llama"]
    res = ea.analyse_tables(tables, draws=10)
    assert "explore_account_fit" not in res and "llama" not in res["explore_account_cells"]


def test_writers_are_resampled_independently():
    arrays = {w: ea.writer_arrays(t, 10, ea.SEED + i) for i, (w, t) in enumerate(_tables().items())}
    same_n = [arrays["gemma"], arrays["llama"], arrays["gptoss"]]  # 3 worlds each, distinct seeds
    a, b = (w["src"]["definition"][1]["BxD"] for w in same_n[:2])
    assert not np.array_equal(a, b)


# ------------------------------------------------------------------- the committed analysis
STAGE = ROOT / "runs/stage5_ex_account/analysis/ex_account.json"
STAGES = {
    "main": "runs/stage2_e1",
    "rep": "runs/stage4_rep_e1",
    "gemma": "runs/stage4_m2_google_gemma-4-12b-it",
    "llama": "runs/stage4_m2_meta-llama_llama-3_1-8b-instruct",
    "gptoss": "runs/stage4_m2_openai_gpt-oss-20b",
}


@pytest.mark.skipif(not STAGE.exists(), reason="stage not staged")
def test_staged_cells_are_the_staged_decompositions_of_every_writer():
    acc = json.loads(STAGE.read_text())["explore_account_cells"]
    for w, stage in STAGES.items():
        metrics = json.loads((ROOT / stage / "analysis/e1_metrics.json").read_text())
        for b in ea.BRANCHES:
            for cls in ("definition", "other"):
                dec = metrics["explore_decomposition"]["D1"][b][cls]
                src, it = dec["cells"]["source"], dec["interaction"]["source"]
                mine = acc[w][b][cls]
                for k, want in (
                    ("r0", src["D0B1"]["r"]),
                    ("r1", src["D1B1"]["r"]),
                    ("a0", src["D0B0"]["U"]),
                    ("a1", src["D1B0"]["U"]),
                    ("BxD", it["observed"]),
                    ("B_D0", it["B_effect_D0"]),
                    ("D_B0", it["D_effect_B0"]),
                ):
                    for f in ("est", "lo", "hi"):
                        assert _same(mine[k][f], want[f]), (w, b, cls, k, f)


@pytest.mark.skipif(not STAGE.exists(), reason="stage not staged")
def test_the_numbers_a_reviewer_recomputed_are_the_staged_ones():
    acc = json.loads(STAGE.read_text())["explore_account_cells"]

    def bxd(w, b):
        return round(100 * acc[w][b]["other"]["BxD"]["est"], 1)

    # the 42 other items: Qwen3.5-9B adds up, the other writers do not after a deletion
    assert [bxd("main", b) for b in ("src", "def", "orphan")] == [-0.6, -0.6, -1.4]
    assert bxd("gemma", "orphan") == 6.0 and bxd("gptoss", "orphan") == 4.5
    assert bxd("llama", "orphan") == 27.6 and bxd("llama", "def") == 19.3
    # Llama's single-field items are not re-derived well after S_src (D=0, D=1)
    llama = acc["llama"]["src"]["other"]
    assert round(100 * llama["r0"]["est"], 1) == 75.7 and round(100 * llama["r1"]["est"], 1) == 86.0


@pytest.mark.skipif(not STAGE.exists(), reason="stage not staged")
def test_staged_fit_has_the_registered_shape():
    res = json.loads(STAGE.read_text())
    assert res["explore_draws"] == 2000 and res["explore_seed"] == 996
    two = res["explore_account_fit"]["two_class"]
    assert two["n_cells"] == 24 and len(two["cells"]) == 24 and len(two["heldout_cells"]) == 6
    assert set(res["explore_account_headline"]) <= set(two["lowo"]["mae"])
    assert two["lowo"]["mae"]["corrected"]["est"] < 0.01  # the identity, to a fraction of a pp

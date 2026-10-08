"""Exploratory multi-stage analyses of the E1 family (`analysis.ex_e1_cross`): the out-of-sample
predictors on tables small enough to add up, the replication accounting on the committed stage-2
contrasts, the short-world prediction, the cost model against the committed E3 analysis, and the
stage driver end to end on synthetic stages."""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_e1_cross as cross
from memshift.analysis.replication import PREDICTED_NONZERO
from tests.test_ex_e1_classes import CV, DET, _rows, _table

ROOT = Path(__file__).resolve().parents[1]


def _arrays(worlds: int = 3):
    tab = _table(worlds)
    w, a, names = ex.cell_arrays(tab, "src", "D1", cross.GRANS)
    assert tuple(names) == cross.FIT_CELLS
    return w, a


# ------------------------------------------------------------------------ out-of-sample terms
def test_predictors_by_hand():
    _, arr = _arrays()
    pred = cross.predict(arr.sum(axis=0))
    d = ex.BASE_CLASSES.index("definition")
    # definition items: source-epoch r is 0 at D=0 and 1 at D=1, the key never misses them;
    # the cited key misses k.b (half of the definition items) and k.b is wrong (a = 0)
    assert pred["structural"]["D0B1_source"][d] == 0 and pred["structural"]["D1B1_source"][d] == 1
    assert pred["structural"]["D1B1_cited"][d] == pytest.approx(0.5 * 1 + 0.5 * 0)
    assert pred["structural"]["D0B0"][d] == 0 and pred["structural"]["D1B0"][d] == 1
    # the cell predictor is each cell's own Update: here that is also the structural one
    for cell in cross.FIT_CELLS:
        assert pred["cell"][cell][d] == pytest.approx(pred["structural"][cell][d])
    # the paper's Eq. (1) has one stale-value accuracy: 0, so it predicts 0 at B=0 for both D
    assert pred["literal"]["D1B0"][d] == 0 and pred["literal"]["D0B0"][d] == 0
    r = ex.BASE_CLASSES.index("rule")
    assert pred["cell"]["D1B0"][r] == 1 and pred["literal"]["D1B0"][r] == 0


def test_combine_weights_the_base_classes_by_the_target_mix():
    pred = np.array([0.2, 0.6, 1.0])  # definition, rule, norule
    w = np.array([1.0, 1.0, 2.0])
    assert cross._combine(pred, w, "definition") == pytest.approx(0.2)
    assert cross._combine(pred, w, "other") == pytest.approx((0.6 + 2.0) / 3)
    assert cross._combine(pred, w, "all") == pytest.approx((0.2 + 0.6 + 2.0) / 4)
    # a class the target does not have weighs nothing, even where its prediction is undefined
    assert cross._combine(np.array([0.2, np.nan, 1.0]), np.array([1.0, 0.0, 1.0]), "all") == (
        pytest.approx(0.6)
    )


def test_oos_zero_error_for_the_cell_predictor_on_identical_worlds():
    res = cross.oos(_table(3), _table(4), draws=40)
    b = res["branches"]["src"]
    for cls in cross.OOS_CLASSES:
        cell = b[cls]["cell"]
        assert cell["mean_abs_error"]["est"] == 0 and cell["cells_within_5pp"] == 6
        assert all(c["error"]["est"] == 0 for c in cell["cells"].values())
    # the literal Eq. (1) is wrong by the whole Update at B=0, for D=1 definition items by 1
    lit = b["definition"]["literal"]["cells"]
    assert lit["D1B0"]["error"]["est"] == 1 and lit["D0B0"]["error"]["est"] == 0
    assert lit["D1B0"]["predicted"] == 0 and lit["D1B0"]["observed"] == 1
    assert b["definition"]["literal"]["mean_abs_error"]["est"] == pytest.approx(1 / 6)
    assert b["all"]["cell"]["cells"]["D0B0"]["error"]["n_worlds"] == 4  # target worlds
    mix = res["class_mix"]["src"]
    assert mix["fit"]["definition"]["share"] == mix["target"]["definition"]["share"] == 0.5
    assert res["pooled_over_branches"]["all"]["cell"]["n_cells"] == 6


def test_oos_uses_only_the_branches_both_tables_have():
    both = _table(3)
    other = ex.Table(r for w in range(3) for r in _rows(f"w{w}", CV, DET, branch="def"))
    res = cross.oos(both, other, draws=20)
    assert res["branches"] == {} and "pooled_over_branches" not in res


def test_oos_error_responds_to_a_shifted_target():
    worse = {**CV, "holdmem_D1B1_source": [0, 0, 1, 1]}  # the target's D=1 source cell fails
    tgt = ex.Table(r for w in range(3) for r in _rows(f"w{w}", worse, DET))
    res = cross.oos(_table(3), tgt, draws=40)["branches"]["src"]
    cell = res["definition"]["structural"]["cells"]["D1B1_source"]
    assert cell["predicted"] == 1 and cell["observed"] == 0 and cell["error"]["est"] == -1
    assert cell["abs_error"]["est"] == 1


# ------------------------------------------------------------------------------ short worlds
def test_short_world_prediction_from_the_main_terms():
    res = cross.short_prediction(_table(3), _table(2), draws=40)["src"]
    # B at D=0 on definition items: r_0 - a_0 = 0 - 0 predicted, and observed
    d0 = res["definition"]["D0"]
    assert d0["predicted"]["est"] == 0 and d0["observed"]["est"] == 0
    assert d0["observed_minus_predicted"]["est"] == 0
    o1 = res["other"]["D1"]  # other items: r_1 = 1, a_1 = 0.5
    assert o1["predicted"]["est"] == pytest.approx(0.5)
    assert o1["observed_minus_predicted"]["est"] == pytest.approx(0)
    t = res["terms"]["definition"]
    assert t["r_D1"]["short_minus_main"]["est"] == 0 and t["a_B0_D1"]["main"]["est"] == 1


def test_short_world_regime_change_shows_in_the_prediction_error():
    better = {**CV, "holdmem_D0B1_source": [1, 1, 1, 1]}  # D=0 re-derives definitions right
    short = ex.Table(r for w in range(2) for r in _rows(f"s{w}", better, DET))
    res = cross.short_prediction(_table(3), short, draws=40)["src"]["definition"]["D0"]
    assert res["predicted"]["est"] == 0 and res["observed"]["est"] == 1
    assert res["observed_minus_predicted"]["est"] == 1
    assert res["observed_minus_predicted"]["n_worlds"] == 2


# --------------------------------------------------------------------- replication accounting
STAGE2 = json.loads((ROOT / "runs/stage2_e1/analysis/e1_metrics.json").read_text())


def test_source_epoch_accounting_counts_identities_and_the_rest():
    acc = cross.source_epoch_accounting(STAGE2, STAGE2)  # a stage against itself replicates
    assert acc["n_contrasts"] == 21 and acc["n_identities"] == 6 and acc["n_non_identity"] == 15
    detection = [c for c in acc["contrasts"] if c["identity_by_construction"]]
    assert len(detection) == 6 and all(
        c["contrast"].startswith("B_effect_detected_") for c in detection
    )
    assert acc["n_d_effect_on_repair"] == 6 and acc["n_update_contrasts"] == 9
    assert acc["independent_update_estimates"] == 6
    assert acc["verdicts_non_identity"]["replicated"] == 15
    flipped = json.loads(json.dumps(STAGE2))
    est = flipped["contrasts"]["src"]["B_effect_update_D1_source"]
    flipped["contrasts"]["src"]["B_effect_update_D1_source"] = {**est, "est": -est["est"]}
    again = cross.source_epoch_accounting(STAGE2, flipped)
    assert again["verdicts_non_identity"]["not replicated"] == 1


def _split(est_def, est_other):
    def e(x):
        return {"est": x, "lo": x - 1, "hi": x + 1, "n_worlds": 20, "n_items": 10}

    return {
        "explore_class_split": {
            "D1": {
                "src": {
                    "source": {
                        "definition": {k: e(est_def) for k in cross.CONTRAST_ORDER},
                        "other": {k: e(est_other) for k in cross.CONTRAST_ORDER},
                    }
                }
            }
        }
    }


def test_class_replication_reads_sign_and_interval_on_r():
    orig, rep = _split(30, 0.5), _split(28, -0.4)
    res = cross.class_replication(orig, rep)
    assert len(res["rows"]) == 16  # 8 contrasts x 2 classes
    d = res["summary"]["definition"]
    assert d["rep_same_sign"] == 8 and d["rep_excludes_zero"] == 8 and d["replicated"] == 8
    o = res["summary"]["other"]
    assert o["rep_same_sign"] == 0 and o["rep_excludes_zero"] == 0 and o["replicated"] == 0
    row = next(r for r in res["rows"] if r["class"] == "other")
    assert row["verdict"] == "not replicated" and not row["same_sign"]
    assert cross.class_replication({}, {})["rows"] == []


def test_holm_family_leaves_out_detection_and_uses_exact_p():
    rows = [r for w in range(6) for r in _rows(f"w{w}", CV, DET)]
    metrics = {"contrasts": {"src": e1.contrasts(rows, "src")}}
    res = cross.holm_family(rows, metrics, draws=200)
    names = [c["contrast"] for c in res["contrasts"]]
    assert len(names) == res["family_size"] == 10 and not any("detected" in n for n in names)
    assert all(n.startswith(PREDICTED_NONZERO) for n in names)
    assert res["smallest_possible_p"] == pytest.approx(1 / 201)
    assert res["smallest_possible_holm_p"] == pytest.approx(10 / 201)
    assert all(c["p"] >= 1 / 201 for c in res["contrasts"])
    assert res["n_survive_holm"] == sum(c["survives_holm"] for c in res["contrasts"])
    # the pooled B effect is +0.25 in every world: no draw at or below zero, so p is the floor
    sure = next(c for c in res["contrasts"] if c["contrast"] == "B_effect_update_pooled_source")
    assert sure["est"] == 0.25 and sure["draws_at_or_below_zero"] == 0
    assert sure["p"] == pytest.approx(1 / 201)


# ---------------------------------------------------------------------------- cost and noise
def test_cost_model_matches_the_committed_e3_analysis():
    e3 = json.loads((ROOT / "runs/stage2_e3/analysis/e3_metrics.json").read_text())
    res = cross.cost_model(e3)
    assert "not an execution" in res["label"]
    hm = res["horizons"]["12"]["holdmem_D1B1_source"]
    zero = res["horizons"]["12"]["zero"]
    assert zero["total_over_zero"] == 1 and zero["maintain_tokens"] == 0
    assert hm["total_tokens"] == pytest.approx(
        hm["ingest_tokens"] + hm["maintain_tokens"] + hm["serve_tokens"]
    )
    assert hm["total_over_zero"] == pytest.approx(hm["total_tokens"] / zero["total_tokens"])
    assert hm["maintain_over_serve"] > 1 and hm["tokens_per_counted_write"] > 1000
    assert hm["queries"] == pytest.approx(zero["queries"])
    assert cross.cost_model({}) == {"label": cross.COST_LABEL, "horizons": {}}


def test_pooled_noise_floor_pools_only_the_named_stages():
    def m(v, a, worlds=20):
        return {
            "explore_noise_floor": {
                "total": {
                    "world_branches": 60,
                    "items": 2880,
                    "values_differ": v,
                    "answers_differ": a,
                },
                "world_level": {
                    "worlds": worlds,
                    "items": worlds * 48,
                    "values_differ": v // 3,
                    "answers_differ": a // 3,
                },
            }
        }

    res = cross.pooled_noise_floor(
        {"main": m(39, 3), "rep": m(27, 15), "c": m(39, 3), "none": {}}, pooled=["main", "rep"]
    )
    p = res["pooled"]
    assert p["items"] == 5760 and p["values_differ"] == 66 and p["answers_differ"] == 18
    assert p["values_differ_share"] == pytest.approx(66 / 5760) and set(res["stages"]) == {
        "main",
        "rep",
        "c",
    }
    w = p["world_level"]  # every world counted once: 40 worlds of 48 items
    assert (w["worlds"], w["items"], w["values_differ"], w["answers_differ"]) == (40, 1920, 22, 6)
    assert w["values_differ_share"] == pytest.approx(22 / 1920)
    assert "pooled" not in cross.pooled_noise_floor({"c": m(1, 1)})


# --------------------------------------------------------------------------------- the driver
def _stage(tmp_path: Path, name: str, tab_rows: list[dict], with_metrics: bool = True) -> Path:
    d = tmp_path / name
    (d / "analysis").mkdir(parents=True)
    (d / "analysis" / "e1_item_table.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in tab_rows)
    )
    metrics = {
        "contrasts": {"src": e1.contrasts(tab_rows, "src")},
        **ex.analyse_rows(tab_rows, (), draws=30),
    }
    (d / "analysis" / "e1_metrics.json").write_text(json.dumps(metrics))
    return d


def test_driver_writes_the_analysis_and_a_manifest(tmp_path):
    rows = lambda prefix, n: [  # noqa: E731
        r for w in range(n) for r in _rows(f"{prefix}{w}", CV, DET)
    ]
    dev = _stage(tmp_path, "stage1_e1", rows("d", 3))
    main = _stage(tmp_path, "stage2_e1", rows("m", 4))
    rep = _stage(tmp_path, "stage4_rep_e1", rows("r", 4))
    short = _stage(tmp_path, "stage5_short_e1", rows("s", 3))
    e3 = ROOT / "runs/stage2_e3"
    out = tmp_path / "stage5_ex_e1"
    res = cross.analyse(
        {
            "dev": [dev],
            "main": [main],
            "rep": [rep],
            "short": [short],
            "e3": [e3],
            "noisepool": [main, rep],
        },
        out,
        draws=30,
    )
    assert set(res) >= {
        "explore_oos",
        "explore_replication",
        "explore_short",
        "explore_cost",
        "explore_noise_floor",
        "status",
    }
    assert set(res["explore_oos"]) == {"dev_to_main", "dev_to_rep", "main_to_rep"}
    written = json.loads((out / "analysis" / "ex_e1.json").read_text())
    assert written["explore_short"]["src"]["all"]["D0"]["predicted"]["n_worlds"] == 3
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["experiment"] == "merged-ex_e1"
    assert {s["run_id"] for s in manifest["sources"]} >= {"stage1_e1", "stage2_e3"}
    assert "analysis/ex_e1.json" in manifest["files_sha256"]
    assert math.isfinite(written["explore_cost"]["main"]["horizons"]["12"]["zero"]["queries"])

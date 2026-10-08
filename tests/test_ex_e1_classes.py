"""Exploratory class-stratified E1 analyses (`analysis.ex_e1_classes`): the bootstrap replays the
paper's estimator exactly, the classes are the ones of `e1.ITEM_CLASSES`, and every quantity of the
Eq. (1) decomposition is checked by hand on a table small enough to add up."""

import json
import math
import random

import numpy as np
import pytest

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.experiments.e1 import run_world
from memshift.scoring.stats import paired_diff, world_bootstrap
from tests.fakes import FakeLLM


# ------------------------------------------------------------------------------------- bootstrap
def _random_groups(seed: int, n_worlds: int = 7) -> dict[str, list[float]]:
    rng = random.Random(seed)
    return {
        f"w{i}": [float(rng.randint(-1, 1)) for _ in range(rng.randint(0, 9))]
        for i in range(n_worlds)
    }


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_mean_estimate_replays_world_bootstrap_exactly(seed):
    groups = _random_groups(seed)  # some worlds are empty: they are dropped, as in the paper
    want = world_bootstrap(groups).to_json()
    got = ex.mean_estimate(groups)
    assert got == want


def test_paired_estimate_is_paired_diff():
    rng = random.Random(4)
    a = {f"w{i}": {f"f{j}@1": float(rng.random() < 0.6) for j in range(8)} for i in range(6)}
    b = {f"w{i}": {f"f{j}@1": float(rng.random() < 0.4) for j in range(3, 11)} for i in range(6)}
    assert ex.paired_estimate(a, b) == paired_diff(a, b).to_json()


def test_boot_index_is_the_bootstrap_resampling():
    idx = ex.boot_index(5, draws=3, seed=996)
    rng = random.Random(996)
    assert idx.tolist() == [[rng.randrange(5) for _ in range(5)] for _ in range(3)]
    assert ex.boot_index(5, draws=3, seed=996) is idx  # cached


def test_world_estimate_with_no_world_is_nan():
    e = ex.mean_estimate({})
    assert math.isnan(e["est"]) and e["n_worlds"] == 0


# --------------------------------------------------------------------------------------- classes
def test_item_class_follows_item_classes_and_rejects_unknown_ids():
    assert ex.item_class("k.important_contact") == "definition"
    assert ex.item_class("c.p05.city") == "rule"
    assert ex.item_class("u.timezone") == "rule"
    assert ex.item_class("u.employer") == "norule"
    assert ex.item_class("t.t001") == "rule"
    with pytest.raises(ValueError):
        ex.item_class("z.unknown")
    assert set(ex.CLASS_SETS["other"]) == {"rule", "norule"}
    assert set(ex.CLASS_SETS["all"]) == set(e1.ITEM_CLASSES)


# ------------------------------------------------------------------------ a table that adds up
FACTS = ("k.a", "k.b", "c.s1", "u.employer")  # two definitions, one structured, one free-text


def _rows(world: str, cv: dict[str, list[int]], detected: dict[str, list[bool]], branch="src"):
    """cv[cell][i] and detected[cell][i] for the four FACTS at k=1, all items changed."""
    out = []
    for cell, vals in cv.items():
        for i, fact in enumerate(FACTS):
            out.append(
                {
                    "world_id": world,
                    "branch": branch,
                    "system": cell,
                    "fact_id": fact,
                    "vtype": "city",
                    "k": 1,
                    "changed": True,
                    "cv": bool(vals[i]),
                    "cv_pre": True,
                    "detected": detected[cell][i],
                    "repaired_now": detected[cell][i],
                }
            )
    return out


CV = {
    "holdmem_D0B0": [0, 0, 1, 0],
    "holdmem_D1B0": [1, 1, 1, 0],
    "holdmem_D0B1_source": [0, 0, 1, 1],
    "holdmem_D1B1_source": [1, 1, 1, 1],
    # cited: the key misses the second definition and the free-text item
    "holdmem_D0B1_cited": [0, 0, 1, 0],
    "holdmem_D1B1_cited": [1, 0, 1, 0],
}
DET = {
    "holdmem_D0B0": [False] * 4,
    "holdmem_D1B0": [False] * 4,
    "holdmem_D0B1_source": [True] * 4,
    "holdmem_D1B1_source": [True] * 4,
    "holdmem_D0B1_cited": [True, False, True, False],
    "holdmem_D1B1_cited": [True, False, True, False],
}


def _table(worlds: int = 3) -> ex.Table:
    rows = []
    for i in range(worlds):
        rows += _rows(f"w{i}", CV, DET)
    return ex.Table(rows)


def test_table_arms_and_granularities():
    tab = _table()
    assert tab.arms() == ["D1"]
    assert tab.granularities("D1") == ["source", "cited"]
    assert ex.cell_system("D1s", 1, 1, "cited") == "holdmem_D1sB1_cited"
    assert ex.cell_system("D1", 0, 0, "source") == "holdmem_D0B0"
    assert ex.Table([]).arms() == []


def test_class_mix_counts_item_times_and_world_items():
    mix = ex.class_mix(_table())["src"]
    assert mix["changed_item_times"] == 12
    c = mix["classes"]
    assert c["definition"]["changed_item_times"] == 6 and c["definition"]["share"] == 0.5
    assert c["other"]["changed_item_times"] == 6 and c["other"]["world_items"] == 6
    assert c["rule"]["changed_item_times"] == 3 and c["norule"]["changed_item_times"] == 3
    assert c["all"]["world_items"] == 12 and c["all"]["worlds"] == 3


def test_class_contrasts_by_hand():
    tab = _table()
    d = ex.class_contrasts(tab, "src", "D1", "source", "definition")
    # definition items: D0B0 0, D1B0 1, D0B1 0, D1B1 1 -> B|D0 0, B|D1 0, D|B0 1, D|B1 1, BxD 0
    assert d["B_effect_update_D0"]["est"] == 0 and d["B_effect_update_D1"]["est"] == 0
    assert d["D_effect_update_B0"]["est"] == 1 and d["D_effect_update_B1"]["est"] == 1
    assert d["BxD_interaction_update"]["est"] == 0
    assert d["D_effect_repair_changed"]["est"] == 1 and d["D_effect_repair"]["n_items"] == 6
    o = ex.class_contrasts(tab, "src", "D1", "source", "other")
    # other items: D0B0 (1+0)/2, D1B0 1/2, D0B1 1, D1B1 1: B|D0 +.5, B|D1 +.5, D|B0 0, D|B1 0
    assert o["B_effect_update_D0"]["est"] == 0.5 and o["B_effect_update_pooled"]["est"] == 0.5
    assert o["D_effect_update_B1"]["est"] == 0 and o["BxD_interaction_update"]["est"] == 0
    c = ex.class_contrasts(tab, "src", "D1", "cited", "all")
    assert c["B_effect_update_D1"]["est"] == pytest.approx(
        0.5 - 0.75
    )  # cited: [1,0,1,0] vs [1,1,1,0]


def test_decomposition_terms_identity_and_formulas_by_hand():
    dec = ex.decomposition(_table(), draws=40)["D1"]["src"]
    # definition items, source epoch: nothing missed, r is the Update of the B=1 cells
    src = dec["definition"]
    cells = src["cells"]["source"]
    assert cells["D0B0"]["n"] == 6 and cells["D0B0"]["miss"] == 6
    assert cells["D0B0"]["m"]["est"] == 1 and cells["D0B0"]["a"]["est"] == 0
    assert cells["D1B1"]["m"]["est"] == 0 and cells["D1B1"]["r"]["est"] == 1
    assert math.isnan(cells["D1B1"]["a"]["est"])  # no missed item to be accurate about
    it = src["interaction"]["source"]
    assert it["observed"]["est"] == 0
    assert it["paper_formula"]["est"] == 1  # (m0 - m1)(r1 - r0) = 1 x (1 - 0)
    assert it["corrected_formula"]["est"] == 0  # (r1 - a1) - (r0 - a0) = (1 - 1) - (0 - 0)
    assert it["D_effect_B0"]["est"] == 1 and it["gap_paper"]["est"] == -1
    # the cited key misses the second definition (missed 3 of 6, accuracy 0 on those)
    cited = src["cells"]["cited"]
    assert cited["D1B1"]["m"]["est"] == 0.5 and cited["D1B1"]["a"]["est"] == 0
    assert cited["D1B1"]["r"]["est"] == 1 and cited["D1B1"]["U"]["est"] == 0.5
    i2 = src["interaction"]["cited"]
    # observed [(0.5 - 1) - (0 - 0)] = -0.5; identity residual is zero
    assert i2["observed"]["est"] == pytest.approx(-0.5)
    assert i2["exact_from_terms"]["est"] == pytest.approx(i2["observed"]["est"])
    for cls in ex.CLASS_SETS:
        assert dec[cls]["identity_max_abs_residual"] < 1e-12
    # the free-text class: the B=1 cells repair it correctly, B=0 leaves the old value
    free = dec["norule"]["cells"]["source"]
    assert free["D0B0"]["U"]["est"] == 0 and free["D0B1"]["U"]["est"] == 1


def test_decomposition_invariance_compares_granularities():
    inv = ex.decomposition(_table(), draws=20)["D1"]["src"]["definition"]["invariance"]["cited"]
    assert inv["m_source_minus_cited_D1"]["est"] == -0.5  # source misses none, cited misses half
    assert inv["r_source_minus_cited_D1"]["est"] == 0  # both are right on what they re-derive


def test_invalidation_precision_and_false_invalidation():
    inv = ex.invalidation(_table(), draws=20)["D1"]["src"]
    src = inv["source"]["all"]["D1"]
    assert src["false_invalidation"]["n_items"] == 12
    # every item changes here: false invalidation is undefined, precision and recall are 1
    assert math.isnan(src["false_invalidation"]["est"]) and src["precision"]["est"] == 1
    assert src["missed_invalidation"]["est"] == 0 and src["base_rate_of_change"]["est"] == 1
    cited = inv["cited"]["definition"]["D1"]
    assert cited["missed_invalidation"]["est"] == 0.5 and cited["invalidated_share"]["est"] == 0.5
    assert (
        inv["cited"]["all"]["D0"]["precision"]["est"]
        == inv["cited"]["all"]["D1"]["precision"]["est"]
    )


def test_unchanged_rows_count_for_false_invalidation_not_for_update():
    rows = []
    for w in ("w0", "w1"):
        rows += _rows(w, CV, DET)
        # one unchanged item per cell: the source-epoch key invalidates it, the cited key does not
        for cell in CV:
            rows.append(
                {
                    "world_id": w,
                    "branch": "src",
                    "system": cell,
                    "fact_id": "c.s2",
                    "vtype": "city",
                    "k": 1,
                    "changed": False,
                    "cv": True,
                    "cv_pre": True,
                    "detected": "source" in cell,
                    "repaired_now": "source" in cell,
                }
            )
    tab = ex.Table(rows)
    inv = ex.invalidation(tab, draws=20)["D1"]["src"]
    assert inv["source"]["all"]["D1"]["false_invalidation"]["est"] == 1.0
    assert inv["cited"]["all"]["D1"]["false_invalidation"]["est"] == 0.0
    assert inv["source"]["all"]["D1"]["precision"]["est"] == pytest.approx(4 / 5)
    # Update is over the changed items only; Stability over the unchanged ones
    assert ex.class_contrasts(tab, "src", "D1", "source", "all")["D_effect_update_B1"]["est"] == 0.5
    st = ex.stability(tab, draws=20)["src"]["all"]
    assert st["stability"]["holdmem_D0B1_source"]["est"] == 1.0
    assert st["D1"]["source"]["B_effect_stability_D1"]["est"] == 0.0
    # all asked probes: 4 changed + 1 unchanged per world; the net B effect is their pooled mean
    assert st["D1"]["source"]["net_B_effect_all_probes_D1"]["est"] == pytest.approx(
        (1 - 0.75) * 4 / 5  # changed: D1B1 1.0 vs D1B0 0.75, unchanged: 0
    )


def test_json_arm_against_the_sk_arm():
    cv = {**CV, "holdmem_json_arm": [0, 1, 1, 1]}
    det = {**DET, "holdmem_json_arm": [True] * 4}
    tab = ex.Table([r for w in ("w0", "w1") for r in _rows(w, cv, det)])
    res = ex.json_arm(tab, draws=20)["src"]
    assert res["all"]["update_json_minus_sk"]["est"] == pytest.approx(-0.25)
    assert res["definition"]["update_json_minus_sk"]["est"] == pytest.approx(-0.5)
    assert res["other"]["update_json_minus_sk"]["est"] == 0
    assert ex.json_arm(_table(), draws=20) == {}  # no JSON arm in the run


def test_per_definition_types_and_leave_one_out():
    res = ex.per_definition(_table(), draws=20)["D1"]["src"]
    assert set(res) == {"k.a", "k.b"}
    # k.a: D1B1 and D1B0 both 1, D0B1 and D0B0 both 0 -> D|B1 = 1, B|D1 = 0
    a = res["k.a"]["type"]["source"]
    assert a["D_effect_update_B1"]["est"] == 1 and a["B_effect_update_D1"]["est"] == 0
    assert res["k.a"]["update"]["D1B1"]["est"] == 1 and res["k.a"]["without_type"]["source"]
    assert res["k.b"]["type"]["cited"]["B_effect_update_D1"]["est"] == -1  # cited misses it


def test_noise_floor_and_cost_counts():
    def world(a, b):
        return {
            "seed": 1,
            "branches": {
                "src": {
                    "noise_floor": {
                        "items": 48,
                        "values_differ": a,
                        "cv_first": 0.8,
                        "cv_second": 0.9,
                        "answers_differ": b,
                    },
                    "costs": {"holdmem_D0B0": {"ingest_tokens": 10, "maintain_tokens": 20}},
                },
                "def": {"noise_floor": {}, "costs": {}},  # a non-LLM writer: nothing to count
            },
        }

    res = ex.noise_floor([world(3, 1), world(0, 0)])
    t = res["total"]
    # one world reports two branches with the same second ingestion: it counts once at world level
    twice = {
        **world(3, 1),
        "branches": {**world(3, 1)["branches"], "def": world(3, 1)["branches"]["src"]},
    }
    both = ex.noise_floor([twice, {**world(0, 0), "seed": 2}])
    assert both["total"]["world_branches"] == 3 and both["world_level"]["worlds"] == 2
    assert both["world_level"]["values_differ"] == 3 and both["world_level"]["items"] == 96
    assert both["world_level"]["answers_differ_share"] == pytest.approx(1 / 96)
    assert t["items"] == 96 and t["values_differ"] == 3 and t["answers_differ"] == 1
    assert t["values_differ_share"] == pytest.approx(3 / 96) and t["world_branches"] == 2
    assert t["cv_first_mean"] == pytest.approx(0.8) and "def" not in res["by_branch"]
    assert ex.noise_floor([{"branches": {}}]) == {}
    cost = ex.cost_e1([world(0, 0), world(1, 1)])["src"]["holdmem_D0B0"]
    assert cost["maintain_tokens_per_world_branch"] == 20 and cost["world_branches"] == 2


def test_summaries_are_read_from_run_directories(tmp_path):
    (tmp_path / "world_summaries.jsonl").write_text(
        json.dumps({"seed": 1, "branches": {"src": {"noise_floor": {"items": 4, "values_differ": 1,
                    "cv_first": 1.0, "cv_second": 1.0, "answers_differ": 0}}}}) + "\n"
    )  # fmt: skip
    (tmp_path / "other").mkdir()
    out = ex.analyse_rows(_table().cells[("holdmem_D0B0", "src")], [tmp_path, tmp_path / "other"])
    assert out["explore_noise_floor"]["total"]["values_differ"] == 1
    assert "explore_status" in out and all(k.startswith("explore_") for k in out)


def test_class_split_of_a_run_without_binding_cells_is_empty():
    tab = ex.Table([r for r in _rows("w0", CV, DET) if r["system"] == "holdmem_D0B0"])
    assert ex.class_split(tab) == {} and ex.decomposition(tab) == {}


# ------------------------------------------------------------------------- the real pipeline
@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    d = tmp_path_factory.mktemp("e1run")
    run_world(5, d, branches=("src", "orphan"), baselines=True, writer=FakeLLM())
    return e1.analyse(d), d


def test_pipeline_all_class_reproduces_the_registered_contrasts(pipeline):
    res, _ = pipeline
    assert all(k in res for k in ("explore_class_split", "explore_decomposition"))
    for b in ("src", "orphan"):
        reg = res["contrasts"][b]
        allc = res["explore_class_split"]["D1"][b]["source"]["all"]
        for new, old in (
            ("B_effect_update_D0", "B_effect_update_D0_source"),
            ("B_effect_update_D1", "B_effect_update_D1_source"),
            ("B_effect_update_pooled", "B_effect_update_pooled_source"),
            ("BxD_interaction_update", "BxD_interaction_update_source"),
            ("D_effect_repair", "D_effect_repair_source"),
            ("D_effect_repair_changed", "D_effect_repair_changed_source"),
        ):
            for f in ("est", "lo", "hi", "n_items", "n_worlds"):
                a, o = allc[new][f], reg[old][f]
                assert a == o or (math.isnan(a) and math.isnan(o)), (b, new, f)


def test_pipeline_identity_and_detection_by_construction(pipeline):
    res, _ = pipeline
    for b in ("src", "orphan"):
        for cls, node in res["explore_decomposition"]["D1"][b].items():
            cells = node["cells"]["source"]
            if cells["D0B0"]["n"] == 0:  # the one fake world has no changed item of this class
                assert math.isnan(node["identity_max_abs_residual"])
                continue
            assert node["identity_max_abs_residual"] < 1e-12, (b, cls)
            assert cells["D0B0"]["m"]["est"] in (1.0, None) or math.isnan(cells["D0B0"]["m"]["est"])
            assert cells["D1B1"]["m"]["est"] == 0.0 or cells["D1B1"]["n"] == 0
    # the exact recomposition equals the observed interaction
    it = res["explore_decomposition"]["D1"]["src"]["all"]["interaction"]["source"]
    assert it["exact_from_terms"]["est"] == pytest.approx(it["observed"]["est"], abs=1e-12)
    assert it["observed"]["est"] == pytest.approx(
        res["contrasts"]["src"]["BxD_interaction_update_source"]["est"], abs=1e-12
    )


def test_pipeline_json_arm_and_no_run_summaries(pipeline):
    res, _ = pipeline
    assert "src" in res["explore_json_arm"]
    # the run summaries are written by the driver, not by `run_world`: nothing to count here
    assert "explore_noise_floor" not in res and "explore_cost_e1" not in res
    assert np.isfinite(res["explore_class_mix"]["src"]["classes"]["all"]["share"])

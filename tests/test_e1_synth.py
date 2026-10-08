"""Block C analysis (prereg v2 §4): C1-C4 and program statistics, on synthetic rows and one
offline end-to-end run."""

import json
import math

import pytest

from memshift.analysis import e1, e1_synth
from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_cell, run_world
from memshift.experiments.e2 import item_type as e2_item_type
from memshift.gen.shifts import generate_branch
from tests.fakes import FakeLLM

WORLDS = ("w1", "w2", "w3")
N = 10
CLASS_KEYS = {
    f"explore_C{n}_repair_{cells}_source_{cls}"
    for n, cells in ((1, "D1s_minus_D0"), (2, "D1_minus_D1s"))
    for cls in e1.ITEM_CLASSES
}


def rows_for(
    system, cv, branch="def", repaired=True, detected=True, worlds=WORLDS, prefix="k.item"
):
    """One row per (world, item) at k = 1; `cv(world_index, item_index)` gives ContractValid."""
    return [
        {
            "world_id": w,
            "branch": branch,
            "system": system,
            "fact_id": f"{prefix}{i}",
            "vtype": "set_person",
            "k": 1,
            "changed": True,
            "cv": int(cv(wi, i)),
            "cv_pre": 1,
            "detected": detected,
            "repaired_now": repaired,
            "repair_mech": bool(cv(wi, i)),
        }
        for wi, w in enumerate(worlds)
        for i in range(N)
    ]


def d1s_fraction(wi, i):  # the share of items the writer's own program gets right, per world
    return i < (6, 4, 8)[wi]


def table():
    return (
        rows_for("holdmem_D0B0", lambda wi, i: 0, repaired=False, detected=False)
        + rows_for("holdmem_D1sB0", lambda wi, i: 0, repaired=False, detected=False)
        + rows_for("holdmem_D0B1_source", lambda wi, i: 0)
        + rows_for("holdmem_D1B1_source", lambda wi, i: 1)
        + rows_for("holdmem_D1sB1_source", d1s_fraction)
    )


def test_block_c_contrasts_on_synthetic_rows():
    c = e1_synth.contrasts(table(), "def")
    mean_fraction = (6 + 4 + 8) / (3 * N)
    assert c["C1_repair_D1s_minus_D0_source"].est == pytest.approx(mean_fraction)
    assert c["C2_repair_D1_minus_D1s_source"].est == pytest.approx(1 - mean_fraction)
    # the original D effect is 1, so the share that depends on authored rules is 1 - fraction
    assert c["C2_share_of_D_effect_source"].est == pytest.approx(1 - mean_fraction)
    assert c["C3_BxD_interaction_update_D1s_source"].est == pytest.approx(mean_fraction)
    assert c["C4_D1s_effect_missed_source"].est == 0.0
    assert set(c) == {
        "C1_repair_D1s_minus_D0_source",
        "C2_repair_D1_minus_D1s_source",
        "C2_share_of_D_effect_source",
        "C3_BxD_interaction_update_D1s_source",
        "C4_D1s_effect_missed_source",
        *CLASS_KEYS,
    }
    for key in ("C1_repair_D1s_minus_D0_source", "C3_BxD_interaction_update_D1s_source"):
        assert c[key].n_worlds == 3 and c[key].n_items == 3 * N
        assert 0 < c[key].lo <= c[key].est <= c[key].hi < 1  # worlds differ: a real interval


def classed_table():
    """Definition (k.), structured-field (c.) and free-text (u.employer) items; per class the cells
    repair 0 (D0), a class-specific share (D1s: definition 0.6, rule 0.9, free text 0.2) and all
    (D1) of the items, so that C1 and C2 differ by class."""
    d1s_share = {"k.item": 0.6, "c.item": 0.9, "u.employer": 0.2}
    rows = []
    for prefix, share in d1s_share.items():
        kw = {"prefix": prefix}
        rows += rows_for("holdmem_D0B0", lambda wi, i: 0, repaired=False, detected=False, **kw)
        rows += rows_for("holdmem_D1sB0", lambda wi, i: 0, repaired=False, detected=False, **kw)
        rows += rows_for("holdmem_D0B1_source", lambda wi, i: 0, **kw)
        rows += rows_for("holdmem_D1B1_source", lambda wi, i: 1, **kw)
        rows += rows_for("holdmem_D1sB1_source", lambda wi, i, s=share: i < s * N, **kw)
    return rows


def test_c1_and_c2_are_split_by_item_class_with_the_e1_class_definition():
    c = e1_synth.contrasts(classed_table(), "def")
    for cls, share in (("definition", 0.6), ("rule", 0.9), ("norule", 0.2)):
        c1 = c[f"explore_C1_repair_D1s_minus_D0_source_{cls}"]
        c2 = c[f"explore_C2_repair_D1_minus_D1s_source_{cls}"]
        assert c1.est == pytest.approx(share) and c2.est == pytest.approx(1 - share), cls
        assert c1.n_worlds == 3 and c1.n_items == 3 * N and c2.n_items == 3 * N, cls
    # the pooled contrasts weigh the three classes equally here (10 items each)
    assert c["C1_repair_D1s_minus_D0_source"].est == pytest.approx((0.6 + 0.9 + 0.2) / 3)
    # the same class assignment as the D-effect split of the E1 contrasts
    e = e1.contrasts(classed_table(), "def")
    for cls in e1.ITEM_CLASSES:
        assert e[f"explore_D_effect_repair_source_{cls}"]["est"] == pytest.approx(1.0)
        d = e[f"explore_C1_repair_D1s_minus_D0_source_{cls}"]["est"]
        d2 = e[f"explore_C2_repair_D1_minus_D1s_source_{cls}"]["est"]
        assert d + d2 == pytest.approx(1.0), cls  # C1 + C2 = D effect, item by item


def test_class_split_reports_an_empty_class_as_missing_not_as_zero():
    rows = [r for r in classed_table() if not r["fact_id"].startswith("u.employer")]
    c = e1_synth.contrasts(rows, "def")
    empty = c["explore_C1_repair_D1s_minus_D0_source_norule"]
    assert empty.n_items == 0 and math.isnan(empty.est)
    assert c["explore_C1_repair_D1s_minus_D0_source_rule"].n_items == 3 * N


def test_share_is_a_bootstrapped_ratio_of_pooled_means():
    rows = table()
    share = e1_synth.contrasts(rows, "def")["C2_share_of_D_effect_source"]
    assert share.n_worlds == 3 and share.n_items == 3 * N
    assert share.lo < share.est < share.hi
    # a D effect of zero has no share
    flat = [r for r in rows if r["system"] != "holdmem_D1B1_source"] + rows_for(
        "holdmem_D1B1_source", lambda wi, i: 0
    )
    assert math.isnan(e1_synth.contrasts(flat, "def")["C2_share_of_D_effect_source"].est)


def test_c4_sees_a_d_effect_on_detection_if_there_is_one():
    rows = [r for r in table() if r["system"] != "holdmem_D1sB1_source"]
    rows += rows_for("holdmem_D1sB1_source", d1s_fraction, detected=False)
    c = e1_synth.contrasts(rows, "def")
    assert c["C4_D1s_effect_missed_source"].est == 1.0  # D1s missed everything, D0 nothing


def test_contrasts_need_their_cells_and_label_the_cited_ones_exploratory():
    assert e1_synth.contrasts(rows_for("holdmem_D0B0", lambda wi, i: 1), "def") == {}
    only_d1s = [
        r for r in table() if r["system"] in ("holdmem_D1sB1_source", "holdmem_D0B1_source")
    ]
    c = e1_synth.contrasts(only_d1s, "def")
    assert set(c) == {
        "C1_repair_D1s_minus_D0_source",
        "C4_D1s_effect_missed_source",
        *(k for k in CLASS_KEYS if k.startswith("explore_C1")),  # C2 needs the D1 cell
    }
    cited = [
        {**r, "system": r["system"].replace("_source", "_cited")}
        for r in table()
        if "B1" in r["system"]
    ]
    c = e1_synth.contrasts(table() + cited, "def")
    assert "C1_repair_D1s_minus_D0_source" in c and "explore_C1_repair_D1s_minus_D0_cited" in c
    assert "C1_repair_D1s_minus_D0_cited" not in c  # only the source-epoch cells are on the list
    assert "explore_C3_BxD_interaction_update_D1s_cited" in c  # B=0 cells are shared
    assert "explore_C2_share_of_D_effect_cited" in c and "explore_C4_D1s_effect_missed_cited" in c
    assert e1_synth.synth_granularities(table() + cited) == ["source", "cited"]
    assert e1_synth.synth_granularities(rows_for("holdmem_D1B1_source", lambda wi, i: 1)) == []


def test_contrasts_are_merged_into_the_e1_contrasts_only_when_d1s_is_present():
    plain = [r for r in table() if not e1_synth.is_synth(r["system"])]
    assert not any(k.startswith(("C1", "C2", "C3", "C4")) for k in e1.contrasts(plain, "def"))
    merged = e1.contrasts(table(), "def")
    assert merged["C1_repair_D1s_minus_D0_source"]["est"] == pytest.approx(0.6)
    assert "D_effect_repair_source" in merged  # the existing contrasts are still there


def test_item_type_matches_the_experiment_taxonomy():
    w = generate_branch(1000, "natural", world_params(3))
    types = {e1_synth.item_type(s.slot_id): e2_item_type(s) for s in w.slots}
    assert all(k == v for k, v in types.items())
    assert len(types) > 15  # user attributes, contact fields, kinds, definitions


# ---------------------------------------------------------------------------- program statistics
def gold(w, fact_id, value, status="answer", vtype="set_person"):
    return {
        "world_id": w,
        "branch": "def",
        "fact_id": fact_id,
        "vtype": vtype,
        "tags": {"k": 0},
        "gold_status": status,
        "gold_value": value,
    }


def item(w, system, slot, *notes):
    history = []
    for event, ok, value in notes:
        history.append({"event": event})
        if ok is not None:
            history[-1]["program"] = {"ok": ok, "value": value}
    return {"world_id": w, "branch": "def", "system": system, "slot_id": slot, "history": history}


def test_program_statistics_per_item_type():
    s, other = "holdmem_D1sB1_source", "holdmem_D1B1_source"
    probes = []
    items = []
    for w in ("w1", "w2"):
        probes += [
            gold(w, "k.close_friend", ["Ann", "Bo"]),
            gold(w, "u.timezone", "Europe/Paris", vtype="tz"),
            gold(w, "c.p03.city", "Rome", vtype="city"),
            gold(w, "c.p04.city", None, status="abstain", vtype="city"),
        ]
        items += [
            # runs and agrees at t0 (set comparison ignores order); its repair also runs
            item(
                w,
                s,
                "k.close_friend",
                ("derive", True, ["Bo", "Ann"]),
                ("invalidate", None, None),
                ("repair", True, ["Ann"]),
            ),  # fmt: skip
            # does not run at t0, runs after the shift
            item(w, s, "u.timezone", ("derive", False, None), ("repair", True, "Europe/Paris")),
            # runs but disagrees at t0 (w1); w2 agrees
            item(w, s, "c.p03.city", ("derive", True, "Rome" if w == "w2" else "Oslo")),
            # gold is abstain: a program that returns None agrees
            item(w, s, "c.p04.city", ("derive", True, None), ("repair", False, None)),
            item(w, other, "k.close_friend", ("derive", None, None)),  # not a D1s cell
        ]
    stats = e1_synth.program_stats({"probes": probes, "items": items})
    assert list(stats) == ["def"] and list(stats["def"]) == [s]  # only the D1s system
    by_type = stats["def"][s]
    assert set(by_type) == {"all", "kappa.close_friend", "u.timezone", "contact.city"}
    ck = by_type["kappa.close_friend"]
    assert ck["n"] == 2
    assert ck["exec_t0"]["est"] == 1.0 and ck["agree_t0"]["est"] == 1.0
    assert ck["agree_t0_given_exec"]["est"] == 1.0 and ck["exec_post"]["est"] == 1.0
    tz = by_type["u.timezone"]
    assert tz["exec_t0"]["est"] == 0.0 and tz["agree_t0"]["est"] == 0.0
    assert tz["agree_t0_given_exec"] is None  # nothing ran
    assert tz["exec_post"]["est"] == 1.0
    city = by_type["contact.city"]
    assert city["n"] == 4 and city["exec_t0"]["est"] == 1.0
    assert city["agree_t0"]["est"] == pytest.approx(0.75)  # Oslo is wrong once, None is right
    assert city["exec_post"]["est"] == 0.0
    total = by_type["all"]
    assert total["n"] == 8 and total["exec_t0"]["est"] == pytest.approx(0.75)  # 2 of 8 do not run
    assert total["agree_t0"]["est"] == pytest.approx(5 / 8)  # 5 of 8 items agree
    assert total["agree_t0_given_exec"]["est"] == pytest.approx(5 / 6)
    assert total["exec_post"]["est"] == pytest.approx(4 / 6)  # 4 of the 6 later programs ran
    assert total["exec_post"]["n_worlds"] == 2


def test_program_statistics_are_empty_without_d1s_items():
    assert e1_synth.program_stats({"probes": [], "items": []}) == {}


# ---------------------------------------------------------------------------- end to end
def test_e1_run_with_d1s_cells_and_analysis(tmp_path):
    cells = [parse_cell(t) for t in ("D0B0", "D0B1_source", "D1B1_source", "D1sB0", "D1sB1_source")]
    summary = run_world(
        5,
        tmp_path,
        branches=("def",),
        baselines=False,
        writer=FakeLLM(),
        cells_cfg=cells,
        world_params=world_params(3),
    )
    assert summary["branches"]["def"]["hash_ok"]  # the hash assertion covers the D1s cells
    res = e1.analyse(tmp_path)
    assert res["pre_intervention"]["cells_identical_answers"]
    c = res["contrasts"]["def"]
    for key in (
        "C1_repair_D1s_minus_D0_source",
        "C2_repair_D1_minus_D1s_source",
        "C2_share_of_D_effect_source",
        "C3_BxD_interaction_update_D1s_source",
        "C4_D1s_effect_missed_source",
    ):
        assert key in c and c[key]["n_worlds"] == 1
    assert c["C4_D1s_effect_missed_source"]["est"] == 0.0  # D is not in the key
    assert "holdmem_D1sB1_source" in res["cells"]["def"]
    stats = res["synth_programs"]["def"]
    assert set(stats) == {"holdmem_D1sB0", "holdmem_D1sB1_source"}
    for by_type in stats.values():
        assert by_type["all"]["exec_t0"]["est"] == 1.0  # the fake writer's programs all run
        assert "kappa.close_friend" in by_type and "u.timezone" in by_type
    saved = json.loads((tmp_path / "analysis" / "e1_metrics.json").read_text())
    assert saved["synth_programs"]["def"]["holdmem_D1sB0"]["all"]["n"] > 0

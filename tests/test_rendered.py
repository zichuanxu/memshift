import json
from pathlib import Path

import pytest

from memshift.analysis import e1, families, rendered
from memshift.experiments import e1 as driver
from memshift.experiments.common import world_params
from memshift.gen.llm_render import RenderReport, SourceCounts
from memshift.holdmem.system import CellConfig
from memshift.scoring.stats import item_diffs
from tests.fakes import FakeLLM

HM11 = families.HM11
ITEMS = [f"c.p{i:02d}.city" for i in range(4)]
# system -> (ContractValid after the shift, detected, repaired now); zero has no item history
PATTERN = {
    "holdmem_D0B0": (0, False, False),
    "holdmem_D1B0": (0, False, False),
    "holdmem_D0B1_source": (0, True, True),
    "holdmem_D1B1_source": (1, True, True),
    "zero": (0, None, None),
}


def _table(wrong: dict[str, set[int]]) -> list[dict]:
    """E1 item table of one condition: every item changed at k = 1 on the src and def branches.
    `wrong`: world -> indices of items whose HoldMem(1,1) answer is invalid after the shift."""
    rows = []
    for world, bad in wrong.items():
        for branch in ("src", "def"):
            for i, fact in enumerate(ITEMS):
                for system, (cv, detected, repaired) in PATTERN.items():
                    row: dict = {
                        "world_id": world,
                        "branch": branch,
                        "system": system,
                        "fact_id": fact,
                        "vtype": "org",
                        "k": 1,
                        "changed": True,
                        "cv": 0 if system == HM11 and i in bad else cv,
                        "cv_pre": 1,
                    }
                    if detected is not None:
                        row |= {"detected": detected, "repaired_now": repaired}
                    rows.append(row)
    return rows


def _stage(path: Path, rows: list[dict]) -> Path:
    (path / "analysis").mkdir(parents=True)
    with (path / "analysis" / "e1_item_table.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    return path


RENDERED = {"w1000": {0, 1}, "w1001": set(), "w1002": {0, 1, 2, 3}}
TEMPLATE: dict[str, set[int]] = {"w1000": set(), "w1001": set(), "w1002": set()}


def _by_name(res: dict) -> dict[str, dict]:
    return {p["name"]: p for p in res["paired"]}


def test_known_paired_differences(tmp_path):
    # w1003 exists only in the template data and is wildly different: it must not matter
    template = _table({**TEMPLATE, "w1003": {0, 1, 2, 3}})
    res = rendered.analyse(
        _stage(tmp_path / "r", _table(RENDERED)), _stage(tmp_path / "t", template), tmp_path / "o"
    )
    assert res["worlds"] == ["w1000", "w1001", "w1002"]
    assert res["worlds_missing_in_template"] == []
    by = _by_name(res)
    # HoldMem(1,1) repairs 6 of 12 items in the rendered worlds and all 12 in the template worlds
    d = by["D_repair_src"]
    assert d["template"]["est"] == 1.0 and d["rendered"]["est"] == 0.5
    assert d["diff"]["est"] == pytest.approx(-0.5)
    assert d["diff"]["n_worlds"] == 3 and d["diff"]["n_items"] == 12
    assert d["per_world"]["w1000"]["diff"] == pytest.approx(-0.5)
    assert d["per_world"]["w1001"]["diff"] == 0.0
    assert d["per_world"]["w1002"]["diff"] == pytest.approx(-1.0)
    assert d["diff_world_mean"]["est"] == pytest.approx(-0.5)  # (-0.5 + 0 - 1) / 3 worlds
    lo, est, hi = (d["diff"][k] for k in ("lo", "est", "hi"))
    assert -1.0 <= lo < est < hi <= 0.0  # three worlds with different differences: a real interval
    # the other contrasts behind (ii)-(v), by hand: B effect on Update pools D=0 (0) and D=1
    # (6 of 12 items right) -> 0.25 vs 0.5; the interaction is the D=1 effect; zero - HoldMem
    assert by["B_update_src"]["diff"]["est"] == pytest.approx(0.25 - 0.5)
    assert by["D_repair_def"]["diff"]["est"] == pytest.approx(-0.5)
    assert by["BxD_def"]["diff"]["est"] == pytest.approx(0.5 - 1.0)
    assert by["zero_minus_hm11_def"]["diff"]["est"] == pytest.approx(-0.5 - -1.0)
    assert by["zero_minus_hm11_def"]["check"] == "v"
    # path 2: nothing changes detection
    assert by["src:D_effect_missed_source"]["check"] == "i"
    assert by["src:D_effect_missed_source"]["diff"]["est"] == 0.0
    assert by["def:D_effect_missed_source"]["diff"]["est"] == 0.0


def test_estimates_equal_the_e1_contrasts_and_the_difference_is_their_difference():
    rows_r, rows_t = _table(RENDERED), _table(TEMPLATE)
    sum_r, sum_t = rendered.summarise(rows_r), rendered.summarise(rows_t)
    for row in rendered.contrast_rows(rows_r, rows_t, sum_r, sum_t):
        assert row["items"]["rendered"] == row["items"]["template"] == row["items"]["common"]
        if row["name"] == "zero_minus_hm11_def":
            continue
        # the rendered and template estimates are e1.contrasts' own
        assert row["rendered"] == e1.contrasts(rows_r, row["branch"])[row["contrast"]]
        assert row["template"] == e1.contrasts(rows_t, row["branch"])[row["contrast"]]
        # same items in both: paired difference of item contrasts == difference of the estimates
        assert row["diff"]["est"] == pytest.approx(row["rendered"]["est"] - row["template"]["est"])
        assert row["mean_rendered"] == pytest.approx(row["rendered"]["est"])


def test_contrast_vectors_match_paired_diff_of_every_contrast():
    rows = _table(RENDERED)
    vecs = rendered.contrast_vectors(rows)
    for branch in ("src", "def"):
        for key, est in e1.contrasts(rows, branch).items():
            if (branch, key) in vecs and est["n_items"]:
                items = [v for d in vecs[branch, key].values() for v in d.values()]
                assert sum(items) / len(items) == pytest.approx(est["est"])


def test_checks_on_rendered_and_template_are_family_checks(tmp_path):
    rows_r = _table({w: {0, 1, 2, 3} for w in RENDERED})  # HoldMem(1,1) wrong on every item
    rows_t = _table(TEMPLATE)
    res = rendered.analyse(
        _stage(tmp_path / "r", rows_r), _stage(tmp_path / "t", rows_t), tmp_path / "o"
    )
    assert res["template_same_seeds"] == families.family_checks(rendered.summarise(rows_t), rows_t)
    assert res["rendered"] == families.family_checks(rendered.summarise(rows_r), rows_r)
    cmp = res["check_comparison"]
    assert cmp["ii_to_iv_true"] == {"rendered": False, "template": True}
    assert all(v["template"] for v in cmp["checks"].values())  # true by construction
    # no repair, no Update benefit and no gap to the zero system in the rendered worlds
    for check in ("ii_B_update_src", "iii_D_repair_src", "iv_BxD_def", "v_zero_below_hm11_def"):
        assert cmp["checks"][check] == {"rendered": False, "template": True, "agree": False}
    assert cmp["checks"]["i_D_missed_within_threshold"]["agree"]  # detection does not move


def test_identical_conditions_give_zero_differences_and_full_agreement(tmp_path):
    same = _table(TEMPLATE)
    res = rendered.analyse(
        _stage(tmp_path / "r", same), _stage(tmp_path / "t", same), tmp_path / "o"
    )
    assert all(
        p["diff"]["est"] == 0.0 and p["diff"]["lo"] == p["diff"]["hi"] == 0.0 for p in res["paired"]
    )
    assert all(c["diff"]["est"] == 0.0 for c in res["cells"])
    assert all(v["agree"] for v in res["check_comparison"]["checks"].values())
    assert res["rendered"]["ii_to_iv_true"] and res["probe_identity"]["identical"]


def test_restriction_to_the_rendered_seeds_and_determinism(tmp_path):
    rstage = _stage(tmp_path / "r", _table(RENDERED))
    full = _table({**TEMPLATE, "w1003": {0, 1, 2, 3}, "w1004": set()})
    a = rendered.analyse(rstage, _stage(tmp_path / "t1", full), tmp_path / "o1")
    b = rendered.analyse(rstage, _stage(tmp_path / "t2", _table(TEMPLATE)), tmp_path / "o2")
    assert a == b  # the template's extra worlds change nothing, and the bootstrap is deterministic
    assert a["worlds"] == ["w1000", "w1001", "w1002"]
    assert a["template_same_seeds"]["n_worlds"] == 3
    written = json.loads((tmp_path / "o1" / "analysis" / "rendered.json").read_text())
    assert written["paired"] == a["paired"]
    # a rendered world without a template counterpart is reported and left out of the pairing
    one = _stage(tmp_path / "t3", _table({"w1000": set()}))
    partial = rendered.analyse(rstage, one, tmp_path / "o3")
    assert partial["worlds_missing_in_template"] == ["w1001", "w1002"]
    assert partial["paired"][0]["diff"]["n_worlds"] == 1


def test_pooled_and_equal_weight_world_differences_by_hand():
    # world a: two items, +1 each; world b: one item, -1. Pooled over the three items: 1/3; the
    # per-world differences are +1 and -1: equal-weight mean 0. The item present in one condition
    # only (b's "z") is not paired.
    vr = {"a": {"x": 1.0, "y": 1.0}, "b": {"x": 0.0, "z": 1.0}}
    vt = {"a": {"x": 0.0, "y": 0.0}, "b": {"x": 1.0}}
    row = rendered.paired_row(vr, vt)
    assert row["diff"]["est"] == pytest.approx(1 / 3)
    assert row["diff_world_mean"]["est"] == 0.0
    assert row["items"] == {"rendered": 4, "template": 3, "common": 3}
    assert row["per_world"]["a"] == {"rendered": 1.0, "template": 0.0, "diff": 1.0, "n_items": 2}
    assert row["per_world"]["b"] == {"rendered": 0.0, "template": 1.0, "diff": -1.0, "n_items": 1}
    assert (row["mean_rendered"], row["mean_template"]) == (2 / 3, 1 / 3)


def test_probe_level_cell_table():
    rows_r, rows_t = _table(RENDERED), _table(TEMPLATE)
    cells = {(c["branch"], c["system"], c["period"]): c for c in rendered.cell_rows(rows_r, rows_t)}
    hm = cells["src", HM11, "post"]
    assert (hm["mean_rendered"], hm["mean_template"]) == (0.5, 1.0)
    assert hm["diff"]["est"] == pytest.approx(-0.5) and hm["items"]["common"] == 12
    assert cells["src", HM11, "k1"]["diff"]["est"] == pytest.approx(-0.5)
    assert cells["src", HM11, "pre"]["diff"]["est"] == 0.0  # the shift cannot move t0 answers
    for system in ("holdmem_D0B0", "holdmem_D0B1_source", "zero"):
        assert cells["def", system, "post"]["diff"]["est"] == 0.0
    # paired at the probe level: an item wrong only in the rendered world is a -1 for that probe
    vr, vt = rendered._cv_vectors(rows_r), rendered._cv_vectors(rows_t)
    d = item_diffs(vr["src", HM11, "post"], vt["src", HM11, "post"])
    assert d["w1000"] == {
        f"{ITEMS[0]}@1": -1.0,
        f"{ITEMS[1]}@1": -1.0,
        f"{ITEMS[2]}@1": 0.0,
        f"{ITEMS[3]}@1": 0.0,
    }


def test_probe_identity_flags_a_changed_gold():
    rows = _table(TEMPLATE)
    other = [{**r} for r in rows]
    for r in other:  # one probe, which every system answers: w1000, src, first item, k = 1
        if (r["world_id"], r["branch"], r["fact_id"]) == ("w1000", "src", ITEMS[0]):
            r["changed"] = False
    assert rendered.probe_identity(rows, rows)["identical"]
    diff = rendered.probe_identity(rows, other)
    assert not diff["identical"] and diff["changed_or_vtype_mismatch"] == 1


# ---------------------------------------------------------------------------- (c) descriptives
def _report(rate: float = 1.0) -> dict:
    counts = {
        "email": SourceCounts(
            events=10, skipped_background=4, rendered_first=4, rendered_retry=1, kept_template=1,
            anchored_events=3, anchors=5, outside_length_band=1, first_failures={"anchor": 2},
        ),
        "messages": SourceCounts(events=5, rendered_first=5),
        "notes": SourceCounts(
            events=2, rendered_first=1, kept_template=1, first_failures={"anchor": 1, "number": 1}
        ),
    }  # fmt: skip
    return RenderReport(
        "fake/renderer", counts, 9, {"calls": 9, "cached": 2}, 100, rate, 1.0
    ).to_json()


def _summaries(run: Path, rate: float = 1.0) -> None:
    run.mkdir(parents=True)
    rows = [
        {
            "seed": seed,
            "branches": {
                b: {"render": _report(rate if (seed, b) == (1001, "def") else 1.0)}
                for b in ("src", "def")
            },
            "worlds": [{"world_id": f"w{seed}", "branch": b} for b in ("src", "def")],
        }
        for seed in (1000, 1001, 1002)
    ]
    (run / "world_summaries.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_render_descriptives_per_source_kind(tmp_path):
    _summaries(tmp_path / "run")
    res = rendered.render_descriptives([tmp_path / "run"], {"w1000", "w1001"})  # w1002 left out
    assert res["source"] == "world_summaries.jsonl" and res["models"] == ["fake/renderer"]
    assert res["world_branches"] == 4
    email = res["per_source"]["email"]
    assert (email["events"], email["skipped_background"], email["attempted"]) == (40, 16, 24)
    assert (email["rendered"], email["retried"], email["kept_template"]) == (20, 8, 4)
    assert email["kept_template_share"] == pytest.approx(4 / 24)
    assert email["retried_share"] == pytest.approx(8 / 24)
    assert email["first_failures"] == {"anchor": 8}
    assert res["per_source"]["messages"]["kept_template_share"] == 0.0
    notes = res["per_source"]["notes"]
    assert notes["kept_template_share"] == 0.5 and notes["first_failures"] == {
        "anchor": 4,
        "number": 4,
    }
    total = res["total"]
    assert total["attempted"] == 24 + 20 + 8 and total["kept_template"] == 4 + 0 + 4
    assert total["kept_template_share"] == pytest.approx(8 / 52)
    assert res["fidelity"] == {
        "checked": 400, "min_rate": 1.0, "min_rate_template": 1.0, "all_one": True, "below_one": []
    }  # fmt: skip
    assert res["requests"] == 36 and res["usage"] == {"cached": 8, "calls": 36}
    assert {(r["world_id"], r["branch"]) for r in res["by_world_branch"]} == {
        (w, b) for w in ("w1000", "w1001") for b in ("src", "def")
    }


def test_render_descriptives_flag_a_fidelity_below_one(tmp_path):
    _summaries(tmp_path / "run", rate=0.98)
    res = rendered.render_descriptives([tmp_path / "run"])
    assert not res["fidelity"]["all_one"] and res["fidelity"]["min_rate"] == 0.98
    assert res["fidelity"]["below_one"] == ["w1001/def"]


def test_render_descriptives_fall_back_to_the_manifest(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    render = {"model": "m", "sources": ["email"], "kept_template": 2}
    worlds = [
        {"world_id": w, "branch": b, "render": {**render, "skipped_background": 3}}
        for w in ("w1000", "w1001")
        for b in ("src", "def")
    ]
    (run / "manifest.json").write_text(json.dumps({"worlds": worlds}))
    res = rendered.render_descriptives([run], {"w1000"})
    assert (
        res["source"] == "manifest.json" and res["per_source"] is None and res["fidelity"] is None
    )
    assert res["total"] == {"skipped_background": 6, "kept_template": 4}
    assert rendered.render_descriptives([tmp_path / "nowhere"])["source"] is None


# --------------------------------------------------------------------------- end to end (raw runs)
def test_end_to_end_on_raw_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(driver, "client", lambda model, run_dir=None, **kw: FakeLLM(model))
    params = {**(world_params(3) or {}), "target_tokens": 30_000}
    cells = [CellConfig(d, 1) for d in (0, 1)]  # two cells keep the driver runs short
    runs = {}
    for name, render in (("template", None), ("rendered", "fake/renderer")):
        out = tmp_path / name
        out.mkdir()
        runs[name] = driver.run(
            [5], out=out, cells_cfg=cells, baselines=False, branches=("src", "def"),
            world_params=params, render_model=render, parallel_worlds=1,
        )  # fmt: skip
    res = rendered.analyse(runs["rendered"], runs["template"], tmp_path / "out", [runs["rendered"]])
    assert res["worlds"] == ["w0005"] and res["worlds_missing_in_template"] == []
    assert res["probe_identity"]["identical"]  # rendering leaves gold and the probes alone
    assert res["rendering"]["source"] == "world_summaries.jsonl"
    assert res["rendering"]["fidelity"]["all_one"] and res["rendering"]["world_branches"] == 2
    assert set(res["rendering"]["per_source"]) == {"email", "messages", "notes"}
    assert {"D_repair_src", "D_repair_def", "src:D_effect_missed_source"} <= set(_by_name(res))
    # the fake reader abstains everywhere, so rendering cannot move any cell
    assert all(c["diff"]["est"] == 0.0 for c in res["cells"] if c["diff"]["n_items"])
    assert (tmp_path / "out" / "analysis" / "rendered.json").exists()

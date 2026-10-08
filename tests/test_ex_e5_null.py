"""Exploratory native null baselines (`analysis.ex_e5_null`, stored by `e5.analyse` as
`explore_native_null`) on the synthetic multi-system fixture of `tests/ex_e5_fixture.py`."""

import json
import math
from typing import Any

import pytest

from memshift.analysis import e1, e5, e5_infer, ex_e5_null
from memshift.scoring.stats import paired_diff, world_bootstrap
from tests.ex_e5_fixture import BASELINES, NATIVE, RATES, WORLDS, build_runs


@pytest.fixture
def runs(tmp_path, monkeypatch):
    return build_runs(tmp_path, monkeypatch)


def _analyse(runs, tmp_path, noise=True, baselines=True) -> dict[str, Any]:
    native, white, noise_dir, base = runs
    return e5.analyse(
        [native],
        [white],
        tmp_path / "out",
        [noise_dir] if noise else None,
        [base] if baselines else None,
    )


# ------------------------------------------------------------------------------------ helpers
def test_b_identification_is_unidentified_when_the_interval_straddles_the_threshold():
    straddle = {f"w{i}": [1.0, 0.0, 1.0, 0.0] for i in range(5)}  # P = 0.5 exactly
    got = ex_e5_null.b_identification(straddle, label=1)
    assert got["rate"] == 0.5 and got["straddles"] and not got["identified"]
    assert got["threshold"] == e5_infer.B_THRESHOLD and got["n"] == 20
    low = {f"w{i}": [0.0, 0.0, 0.0, 1.0] if i % 2 else [0.0] * 4 for i in range(6)}
    got = ex_e5_null.b_identification(low, label=1)
    assert got["hi"] < 0.5 and not got["straddles"] and got["identified"]
    # a label that could not be inferred (too few probes) is unidentified whatever the interval
    assert not ex_e5_null.b_identification(low, label=None)["identified"]
    none = ex_e5_null.b_identification({}, label=None)
    assert none["n"] == 0 and not none["identified"] and not none["straddles"]


def test_ingestion_variance_is_half_the_mean_square_of_the_per_world_differences():
    first = {"a": {"x": 1.0, "y": 1.0}, "b": {"x": 0.0, "y": 0.0}, "c": {"x": 1.0}}
    second = {"a": {"x": 0.0, "y": 1.0}, "b": {"x": 0.0, "y": 0.0}, "d": {"x": 1.0}}
    v = ex_e5_null.ingestion_variance(first, second)
    # worlds a and b are shared: d_a = -0.5, d_b = 0; variance = (0.25 + 0) / 2 / 2
    assert v is not None and v["worlds"] == 2
    assert v["variance"] == pytest.approx(0.0625) and v["sd"] == pytest.approx(0.25)
    assert ex_e5_null.ingestion_variance({"a": {"x": 1.0}}, {"b": {"x": 1.0}}) is None


def test_inflation_widens_the_interval_by_the_ingestion_variance_and_is_zero_without_it():
    native = {f"w{i}": {f"f{j}": float(j < 3 + i % 3) for j in range(6)} for i in range(6)}
    ref = {f"w{i}": {f"f{j}": float(j < 2) for j in range(6)} for i in range(6)}
    plain = paired_diff(native, ref)
    p0 = ex_e5_null.paired(native, ref, {"variance": 0.0, "sd": 0.0, "worlds": 2})
    assert p0 is not None and p0["lo_inflated"] == pytest.approx(plain.lo)
    assert p0["hi_inflated"] == pytest.approx(plain.hi)
    noise = {"variance": 0.01, "sd": 0.1, "worlds": 3}
    p = ex_e5_null.paired(native, ref, noise)
    assert p is not None and p["est"] == pytest.approx(plain.est)
    se = math.sqrt(0.01 * 6 * (1 / 6) ** 2)  # six worlds of equal size
    assert p["ingestion_se"] == pytest.approx(se)
    assert p["lo_inflated"] == pytest.approx(
        plain.est - math.hypot(plain.est - plain.lo, 1.959964 * se)
    )
    assert p["lo_inflated"] < plain.lo and p["hi_inflated"] > plain.hi
    no_noise = ex_e5_null.paired(native, ref, None)
    assert no_noise is not None and "lo_inflated" not in no_noise
    assert ex_e5_null.paired({}, ref, None) is None


# ---------------------------------------------------------------------------- the full analysis
def _interval(rows: list[dict], system: str, branch: str, metric: str) -> dict[str, float]:
    by = e5.carry_samples(rows, system, branch, set(WORLDS))[metric]
    return world_bootstrap(by).to_json()


def test_every_system_is_analysed_and_existing_fields_do_not_change(runs, tmp_path, monkeypatch):
    res = _analyse(runs, tmp_path)
    ex = res["explore_native_null"]
    assert set(ex["systems"]) == set(NATIVE) and ex["total"]["systems"] == 3
    monkeypatch.setattr(ex_e5_null, "analyse", lambda *a, **k: {})
    plain = e5.analyse([runs[0]], [runs[1]], tmp_path / "out2", [runs[2]], [runs[3]])

    def without(r: dict[str, Any]) -> str:  # through JSON: NaNs compare equal as text
        return json.dumps(
            {k: v for k, v in r.items() if k != "explore_native_null"}, sort_keys=True
        )

    assert without(res) == without(plain)
    # the document-level system is placed against the cited-cluster cells
    light = ex["systems"]["lightrag"]
    assert light["predicted_by"] == "holdmem_D0B1_cited"
    assert light["other_label_cell"] == "holdmem_D0B0"
    assert set(light["chance_coverage"]["label_cells"]) == {
        "holdmem_D0B0",
        "holdmem_D1B0",
        "holdmem_D0B1_cited",
        "holdmem_D1B1_cited",
    }
    mem = ex["systems"]["mem0"]
    assert set(mem["chance_coverage"]["label_cells"]) == {
        "holdmem_D0B0",
        "holdmem_D1B0",
        "holdmem_D0B1_source",
        "holdmem_D1B1_source",
    }
    assert mem["baseline_references"] == ["zero", "rag", "oracle"]
    assert set(mem["white_box_references"]) == set(RATES) & {
        c for c in RATES if c.startswith("holdmem_")
    }


def test_contained_by_lists_the_reference_intervals_that_hold_the_native_value(runs, tmp_path):
    res = _analyse(runs, tmp_path)
    native, white, _, base = runs
    data = e1.load(white)
    data["answers"] += [
        json.loads(line)
        for d in (native, base)
        for line in (d / "answers.jsonl").read_text().splitlines()
    ]
    rows = e1.item_table(data)
    for system in NATIVE:
        entry = res["explore_native_null"]["systems"][system]
        assert len(entry["comparisons"]) == 6
        for key, c in entry["comparisons"].items():
            branch, metric = key.split(":")
            nat = _interval(rows, system, branch, metric)
            expect = [
                ref
                for ref in (*entry["white_box_references"], *entry["baseline_references"])
                if (r := _interval(rows, ref, branch, metric))["lo"] <= nat["est"] <= r["hi"]
            ]
            assert c["contained_by"] == expect and c["n_contained"] == len(expect)
            assert c["n_references"] == len(c["references"]) == 9
            assert c["native"]["est"] == pytest.approx(nat["est"])
            assert c["label_circular"] == (branch == "orphan")
    # mem0 behaves like (0,0) and like the zero system: both hold its value, (0,1) does not
    src = res["explore_native_null"]["systems"]["mem0"]["comparisons"]["src:Update"]
    assert {"holdmem_D0B0", "zero"} <= set(src["contained_by"])
    assert "holdmem_D0B1_source" not in src["contained_by"]


def test_counts_split_label_circular_from_out_of_sample(runs, tmp_path):
    res = _analyse(runs, tmp_path)
    ex = res["explore_native_null"]
    for system, entry in ex["systems"].items():
        c = entry["counts"]
        for k in ("comparisons", "inside_predicted", "inside_other", "discriminating"):
            assert c["all"][k] == c["label_circular"][k] + c["out_of_sample"][k]
        assert c["label_circular"]["comparisons"] == 2 and c["out_of_sample"]["comparisons"] == 4
        # the counts are the ones `stage4_summary` already reports
        s4 = res["stage4"]["discriminability"]["per_system"][system]
        assert c["all"]["comparisons"] == s4["comparisons"]
        assert c["all"]["inside_predicted"] == s4["inside_predicted"]
        assert c["all"]["discriminating"] == s4["discriminating"]
    total = ex["total"]["counts"]["all"]
    assert total["comparisons"] == 18
    assert total["discriminating"] == res["stage4"]["discriminability"]["total"]["discriminating"]


def test_label_free_baselines_are_scored_under_the_same_rule(runs, tmp_path):
    res = _analyse(runs, tmp_path)
    mem = res["explore_native_null"]["systems"]["mem0"]
    for b in BASELINES:
        counts = mem["baseline_counts"][b]
        assert counts["baseline"]["all"]["comparisons"] == 6  # run on all three branches
        assert counts["baseline"]["label_circular"]["comparisons"] == 2
    # the zero system has mem0's rates, so it is inside wherever mem0 is, and the oracle is not
    zero = mem["baseline_counts"]["zero"]
    assert zero["baseline"]["all"] == zero["native_same_comparisons"]["all"] == mem["counts"]["all"]
    oracle = mem["baseline_counts"]["oracle"]["baseline"]["all"]
    assert oracle["inside_predicted"] == 0 and oracle["discriminating"] == 0
    # by hand: each baseline estimate against mem0's predicted and other intervals
    for b in BASELINES:
        inside = inside_other = 0
        for key, c in mem["comparisons"].items():
            br = res["systems"]["mem0"]["branches"][key.split(":")[0]]
            m = key.split(":")[1]
            est = c["references"][b]["est"]
            inside += br["white_box"][m]["lo"] <= est <= br["white_box"][m]["hi"]
            inside_other += br["white_box_other"][m]["lo"] <= est <= br["white_box_other"][m]["hi"]
        got = mem["baseline_counts"][b]["baseline"]["all"]
        assert (got["inside_predicted"], got["inside_other"]) == (inside, inside_other)
    total = res["explore_native_null"]["total"]["baseline_counts"]["zero"]["baseline"]["all"]
    assert total["comparisons"] == 18


def test_chance_coverage_is_the_share_of_reference_intervals_holding_the_native_value(
    runs, tmp_path
):
    res = _analyse(runs, tmp_path)
    for system, entry in res["explore_native_null"]["systems"].items():
        cov = entry["chance_coverage"]
        cells = list(cov["label_cells"])
        hits = {c: 0 for c in cells}
        pairs = 0
        for c in entry["comparisons"].values():
            for cell in cells:
                hits[cell] += c["references"][cell]["contains"]
                pairs += 1
        for cell in cells:
            assert cov["label_cells"][cell] == pytest.approx(hits[cell] / 6)
        assert cov["labels"] == pytest.approx(sum(hits.values()) / pairs)
        assert cov["labels_pooled"] == pytest.approx(cov["labels"])
        with_null = [
            sum(c["references"][n]["contains"] for n in (*cells, "zero", "rag")) / (len(cells) + 2)
            for c in entry["comparisons"].values()
        ]
        assert cov["references"] == pytest.approx(sum(with_null) / 6)
        flags = [
            ok
            for br in res["systems"][system]["branches"].values()
            for ok in br["consistent"].values()
        ]
        assert cov["observed"] == pytest.approx(sum(flags) / len(flags))
    # mem0 lies in (0,0) and nowhere near (0,1): coverage of a random label is below the observed
    mem = res["explore_native_null"]["systems"]["mem0"]["chance_coverage"]
    assert mem["observed"] == 1.0 and mem["labels"] < 0.5
    assert res["explore_native_null"]["total"]["mean_labels_coverage"] == pytest.approx(
        sum(e["chance_coverage"]["labels"] for e in res["explore_native_null"]["systems"].values())
        / 3
    )


def test_b_is_marked_unidentified_where_the_persistence_interval_straddles_one_half(runs, tmp_path):
    res = _analyse(runs, tmp_path)
    ident = {s: e["b_identified"] for s, e in res["explore_native_null"]["systems"].items()}
    assert ident["mem0"]["rate"] == pytest.approx(0.5) and ident["mem0"]["straddles"]
    assert not ident["mem0"]["identified"]
    assert ident["lightrag"]["rate"] == pytest.approx(4 / 36) and ident["lightrag"]["identified"]
    assert ident["amem"]["rate"] == pytest.approx(32 / 36) and ident["amem"]["identified"]
    assert ident["amem"]["label"] == 0 and ident["lightrag"]["label"] == 1
    assert res["explore_native_null"]["total"]["b_identified"] == 2


def test_paired_difference_to_the_predicted_cell_with_the_ingestion_noise(runs, tmp_path):
    res = _analyse(runs, tmp_path)
    native, white, noise_dir, base = runs
    data = e1.load(white)
    data["answers"] += [
        json.loads(line)
        for d in (native, base)
        for line in (d / "answers.jsonl").read_text().splitlines()
    ]
    rows = e1.item_table(data)
    mem = res["explore_native_null"]["systems"]["mem0"]
    p = mem["paired_difference"]["src:Update"]["holdmem_D0B0"]
    nat = ex_e5_null.items(ex_e5_null._index(rows), "mem0", "src", WORLDS, "Update")
    ref = ex_e5_null.items(ex_e5_null._index(rows), "holdmem_D0B0", "src", WORLDS, "Update")
    want = paired_diff(nat, ref)
    assert (p["est"], p["lo"], p["hi"], p["n_items"]) == pytest.approx(
        (want.est, want.lo, want.hi, want.n_items)
    )
    assert p["n_items"] == 36  # six changed items in each of six worlds
    # the noise: the second ingestion of w0 and w1 answers the first two changed items wrong
    assert p["ingestion_worlds"] == 2 and p["ingestion_sd"] > 0
    assert p["lo_inflated"] < p["lo"] and p["hi_inflated"] > p["hi"]
    # native - zero is also paired (the zero system was run on the same items); the oracle is not
    assert "zero" in mem["paired_difference"]["src:Update"]
    assert "oracle" not in mem["paired_difference"]["src:Update"]
    assert mem["paired_difference"]["src:Update"]["zero"]["est"] == pytest.approx(0.0)


def test_analysis_without_baselines_or_noise_runs_keeps_the_white_box_references(runs, tmp_path):
    res = _analyse(runs, tmp_path, noise=False, baselines=False)
    mem = res["explore_native_null"]["systems"]["mem0"]
    assert mem["baseline_references"] == [] and mem["baseline_counts"] == {}
    assert mem["counts"]["all"]["comparisons"] == 6
    p = mem["paired_difference"]["src:Update"]["holdmem_D0B0"]
    assert "lo_inflated" not in p and "zero" not in mem["paired_difference"]["src:Update"]
    assert all(c["n_references"] == 6 for c in mem["comparisons"].values())
    assert res["explore_native_null"]["total"]["baseline_counts"] == {}


def test_a_system_without_an_inferred_label_has_no_predicted_cell(runs, tmp_path, monkeypatch):
    lab = {
        s: {"B": {"label": None, "rate": 0.5, "n": 3}, "D": {"label": None, "rate": 1.0, "n": 0}}
        for s in NATIVE
    }
    monkeypatch.setattr(e5_infer, "infer", lambda *a, **k: lab)
    res = _analyse(runs, tmp_path)
    mem = res["explore_native_null"]["systems"]["mem0"]
    assert mem["predicted_by"] is None and mem["counts"]["all"]["comparisons"] == 0
    assert not mem["b_identified"]["identified"]
    # the references that contain the native value are still listed
    assert all(c["inside_predicted"] is None for c in mem["comparisons"].values())
    assert any(c["contained_by"] for c in mem["comparisons"].values())

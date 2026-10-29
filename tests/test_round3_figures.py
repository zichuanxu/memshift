"""The data functions and figures of `analysis.round3_figures` (round-3 figures and the
diagnostic-tests table): hand-built inputs for the logic, the staged analyses for what the paper
shows, and every figure drawn with full, partial and no inputs."""

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memshift.analysis import round3_figures as rf
from memshift.analysis.paper_data import stage_json
from tests.memtab_check import check_memtab

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = rf.load(ROOT, SOURCES)
EMPTY = rf.load(ROOT, {})


def est(x: float, lo: float | None = None, hi: float | None = None, n: int = 20) -> dict:
    return {"est": x, "lo": x if lo is None else lo, "hi": x if hi is None else hi, "n_worlds": n}


def need(*keys: str) -> None:
    missing = [k for k in keys if INP.get(k) is None]
    if missing:
        pytest.skip(f"not staged: {', '.join(missing)}")


# ------------------------------------------------------------------------------- writers
OPUS, LLAMA = "claude-opus-5-5", "meta-llama/Llama-3.1-8B-Instruct"


def hand_writers() -> dict:
    sets = {f"{OPUS}#t0#s{i}": OPUS for i in range(3)} | {
        f"{LLAMA}#t0.7#s{i}": LLAMA for i in (0, 1)
    }
    per_set = {s: est(0.39, 0.32, 0.45) for s in sets if s.startswith(OPUS)}
    per_set[f"{LLAMA}#t0.7#s0"] = est(0.07, 0.02, 0.12)
    per_set[f"{LLAMA}#t0.7#s1"] = est(0.02, -0.03, 0.08)
    agree: dict[str, dict[str, Any]] = {
        s: {"agreement_t0": {"def": est(1.0)}, "writer": w} for s, w in sets.items()
    }
    agree[f"{LLAMA}#t0.7#s0"]["agreement_t0"]["def"] = est(0.125)
    del agree[f"{LLAMA}#t0.7#s1"]  # its x is not staged: kept in `programs`, not drawn
    branches = {
        b: {
            "r0": {"writer": est(0.45 + i), "qwen_same_worlds": est(0.12 + i)},
            "bxd_D1": {"writer": est(0.16 + i), "qwen_same_item_times": est(0.30 + i)},
        }
        for i, b in enumerate(("src", "def", "orphan"))
    }
    return {
        "g_b": {
            "quantities": {
                "programs": {
                    "branches": {"def": {"definition": {"per_set": per_set}}},
                    "sets": {s: {"writer": w} for s, w in sets.items()},
                }
            }
        },
        "writers": {
            "program_sets": {"sets": agree},
            "definition_writers": {
                "writers": {
                    "claude-sonnet-5-5": {"branches": branches},
                    OPUS: {"branches": branches},
                }
            },
        },
        "e1": {
            "explore_class_split": {
                "D1": {
                    "def": {
                        "source": {"definition": {"BxD_interaction_update": est(0.39, 0.32, 0.45)}}
                    }
                }
            }
        },
    }


def test_writer_points_join_the_sets_x_and_y():
    d = rf.writer_points(hand_writers())
    assert [p["set"] for p in d["programs"]] == sorted(p["set"] for p in d["programs"])
    assert len(d["programs"]) == 5
    nox = next(p for p in d["programs"] if p["set"] == f"{LLAMA}#t0.7#s1")
    assert nox["x"] is None and nox["y"]["est"] == 0.02
    opus = [p for p in d["programs"] if p["writer"] == OPUS]  # identical sets, each kept
    assert len(opus) == 3 and all(p["x"]["est"] == 1.0 and p["y"]["est"] == 0.39 for p in opus)
    llama = next(p for p in d["programs"] if p["set"] == f"{LLAMA}#t0.7#s0")
    assert llama["x"]["est"] == 0.125
    assert d["reference"] == {"est": 0.39, "lo": 0.32, "hi": 0.45}


def test_definition_writers_put_qwen_once_per_shift_on_the_same_worlds():
    d = rf.writer_points(hand_writers())["defwriters"]
    assert [(p["writer"], p["shift"]) for p in d] == [
        (rf.QWEN, "src"),
        ("claude-sonnet-5-5", "src"),
        (OPUS, "src"),
        (rf.QWEN, "def"),
        ("claude-sonnet-5-5", "def"),
        (OPUS, "def"),
    ]  # S_del is not drawn; the Claude writers in the order of `H_MODELS`
    q = d[0]
    assert q["x"]["est"] == pytest.approx(0.12) and q["y"]["est"] == pytest.approx(0.30)
    assert d[4]["x"]["est"] == pytest.approx(1.45) and d[4]["y"]["est"] == pytest.approx(1.16)


def test_sets_on_one_spot_share_one_square_at_their_own_values():
    pts = [p for p in rf.writer_points(hand_writers())["programs"] if p["x"]]
    spots = rf._spots(pts)
    opus = [(x, y, mem) for x, y, mem in spots if pts[mem[0]]["writer"] == OPUS]
    assert [(x, y, len(mem)) for x, y, mem in opus] == [(100.0, 39.0, 3)]  # one square, 3 sets
    llama = next((x, y) for x, y, mem in spots if pts[mem[0]]["writer"] == LLAMA)
    assert llama[0] == pytest.approx(12.5)  # every spot at its own share
    assert sum(len(mem) for *_, mem in spots) == len(pts)


def test_nudges_part_the_marks_that_collide_and_leave_the_rest():
    def mark(w: str, sh: str, x: float, y: float) -> dict:
        h = rf.WRITERS_SQ_B / 2 + 0.1 if sh == "def" else rf.WRITERS_CIRC_B / 2 + 0.1
        return {"key": (w, sh), "w": w, "cx": x, "cy": y, "h": h, "round": sh == "src",
                "xlo": x - 3.0, "xhi": x + 3.1, "ylo": y - 2.9, "yhi": y + 3.2}  # fmt: skip

    marks = [mark("a", "def", 10.0, 10.0), mark("b", "def", 10.4, 10.3), mark("c", "src", 40, 30)]
    nud = rf._nudges(marks)
    assert set(nud) == {("a", "def"), ("b", "def")}  # the far mark stays
    assert all(max(abs(dx), abs(dy)) <= rf.WRITERS_NUDGE_MAX for dx, dy in nud.values())
    (ax, ay), (bx, by) = nud["a", "def"], nud["b", "def"]
    assert not rf._clash(marks[0], np.array([[ax, ay]]), marks[1], np.array([[bx, by]]))[0, 0]
    assert rf._nudges(marks[2:]) == {}


def test_writer_points_without_inputs_are_empty():
    d = rf.writer_points(EMPTY)
    assert d == {"programs": [], "reference": None, "defwriters": []}


def test_staged_writer_points_are_the_staged_values():
    need("g_b", "e1")
    d = rf.writer_points(INP)
    per_set = INP["g_b"]["quantities"]["programs"]["branches"]["def"]["definition"]["per_set"]
    assert {p["set"] for p in d["programs"]} == set(per_set)
    for p in d["programs"]:
        assert p["y"]["est"] == per_set[p["set"]]["est"]
    ref = INP["e1"]["explore_class_split"]["D1"]["def"]["source"]["definition"]
    assert d["reference"]["est"] == ref["BxD_interaction_update"]["est"]
    assert all(p["x"] is not None for p in d["programs"])  # every staged set has its share


def test_staged_definition_writers_agree_with_block_h():
    """The figure's axes reproduce block H's quantities where both report them."""
    h_res = {k: stage_json(ROOT, SOURCES, "round3", f"{k}.json") for k in ("h_sonnet", "h_opus")}
    need("writers")
    if None in h_res.values():
        pytest.skip("not staged: block H")
    dw = INP["writers"]["definition_writers"]
    assert dw["pairing_ok"] is True
    for key, model in (("h_sonnet", "claude-sonnet-5-5"), ("h_opus", "claude-opus-5-5")):
        h = h_res[key]["quantities"]["writers"][model]
        w = dw["writers"][model]
        assert w["worlds"] == h["worlds"]
        for k in ("writer", "qwen_same_worlds", "paired_difference"):
            assert w["branches"]["src"]["r0"][k]["est"] == pytest.approx(h["r0"][k]["est"])
        for b in ("src", "def", "orphan"):
            hb = h["bxd"][b]["definition"]["D1"]
            assert w["branches"][b]["bxd_D1"]["writer"]["est"] == pytest.approx(hb["est"])
            assert w["branches"][b]["bxd_D1"]["writer"]["lo"] == pytest.approx(hb["lo"])


# -------------------------------------------------------------------------------- reader
def test_reader_rows_take_the_paired_contrasts_and_the_store_part():
    need("y", "sr")
    rows = rf.reader_rows(INP)
    assert [(r["cls"], r["branch"]) for r in rows] == [
        (c, b) for c in ("definition", "other") for b in ("src", "def", "orphan")
    ]
    con = INP["y"]["quantities"]["contrasts"]
    ysr = INP["y"]["quantities"]["store_reader"]["results"]["D1"]
    sr = INP["sr"]["results"]["D1"]
    for r in rows:
        node = con[r["branch"]][r["cls"]]["D1"]
        assert r["qwen"]["est"] == node["committed_reader"]["est"]
        assert r["sonnet"]["est"] == node["reader"]["est"]
        store = ysr[r["branch"]][r["cls"]]["interaction"]["store"]
        assert r["store"]["est"] == store["est"]
        # the store part is the same with either reader: the committed split agrees
        committed = sr[r["branch"]][r["cls"]]["interaction"]["store"]
        assert r["store"]["est"] == pytest.approx(committed["est"])


def test_reader_bars_are_the_store_part_and_both_readers_in_their_model_colours():
    from memshift.analysis import figstyle as fs

    bars = {key: fill for key, _, fill in rf.READER_BARS}
    assert list(bars) == ["store", "qwen", "sonnet"]  # back to front within a shift
    r, g, b = (int(bars["store"][i : i + 2], 16) for i in (1, 3, 5))
    assert min(r, g, b) > 0xDD and max(r, g, b) - min(r, g, b) < 0x10  # a pale grey
    r, g, b = (int(bars["qwen"][i : i + 2], 16) for i in (1, 3, 5))
    assert max(r, g, b) < 0x70  # Qwen3.5-9B's near black, as a dark slate
    assert bars["sonnet"] == fs.MODELS["claude-sonnet-5-5"][1]
    assert [k for k, *_ in rf.READERS] == ["qwen", "sonnet"]
    assert [m for _, m, *_ in rf.READERS] == [rf.QWEN, "claude-sonnet-5-5"]


def test_reader_rows_fall_back_to_the_committed_split_and_to_nothing():
    rows = rf.reader_rows({"sr": INP["sr"], "y": None}) if INP.get("sr") else []
    for r in rows:
        assert r["sonnet"] is None and r["store"] is not None and r["qwen"] is not None
    assert all(r["store"] is r["qwen"] is r["sonnet"] is None for r in rf.reader_rows(EMPTY))


# -------------------------------------------------------------------------------- misses
def hand_misses() -> dict:
    node = {
        "missed_item_times": 10,
        "a_count": 1,
        "b_existing_count": 2,
        "b_new_count": 4,
        "c_count": 3,
        "c_left_scope_count": 2,
    }
    return {"miss": {"explore_miss_classes": {"main": {"src": {"definition": node}}}}}


def test_miss_rows_are_exclusive_shares_of_the_missed_item_times():
    (r,) = rf.miss_rows(hand_misses())
    assert (r["stage"], r["branch"], r["cls"], r["missed"]) == ("main", "src", "definition", 10)
    assert r["shares"] == {"a": 0.1, "b_old": 0.2, "b_new": 0.4, "c_left": 0.2, "c_rest": 0.1}
    assert r["uncoverable"] == pytest.approx(0.6)
    assert rf.miss_rows(EMPTY) == []


def test_miss_causes_are_grey_when_coverable_and_warm_when_not():
    from memshift.analysis import figstyle as fs

    fills = {key: fill for key, _, fill, _ in rf.MISS_CAUSES}
    covered = [covered for *_, covered in rf.MISS_CAUSES]
    assert covered == sorted(covered, reverse=True)  # the coverable causes first
    for key, _, fill, covered in rf.MISS_CAUSES:
        r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
        if covered:  # a write-time set could have contained the record: a cool grey
            assert r <= g <= b and b - r < 0x20, key
        else:  # it could not: warm, red above green above blue
            assert r > g > b and r - b > 0x60, key
    assert fills["b_new"] == fs.VERMILION  # the strong colour on the dominant cause
    assert fills["c_left"] != fs.VERMILION


def test_staged_miss_rows_add_up_and_show_both_studies():
    need("miss")
    rows = rf.miss_rows(INP)
    assert {r["stage"] for r in rows} == {"main", "rep"}
    assert len(rows) == 12  # two studies, three shifts, two classes
    for r in rows:
        assert sum(r["shares"].values()) == pytest.approx(1.0)
        assert r["uncoverable"] > 0.5  # most misses: records no write-time set can contain


# ---------------------------------------------------------------------------- diagnostic tests
def test_diagnostic_tests_list_the_four_tests_with_their_outcomes():
    assert rf.diagnostic_tests(EMPTY) == []
    need("e2", "e3", "m1", "nat", "e6", "rep_e2", "rep_e3")
    tests = rf.diagnostic_tests(INP)
    labels = [t.label for t in tests]
    for prefix in ("Retention test", "Leaving test", "Utility floor", "Cost ceiling"):
        assert any(lab.startswith(prefix) for lab in labels), prefix
    # every row is an outcome of the systems: how many units pass, per system where several
    for t in tests:
        assert re.fullmatch(r"\d+/\d+(\\,\$\\cdot\$\\,\d+/\d+)*", t.outcome), t.label
    row = {t.label: t for t in tests if not t.fresh}  # the main study's rows
    assert row[r"Retention test, \holdmem{}(1,1), three tiers"].outcome == "3/3"
    five = row[r"Retention test, \holdmem{}(1,1), five other writers"]
    assert five.outcome == "2/5" and len(five.passing) == 2  # the passing writers, named
    assert row[r"Leaving test, \holdmem{}(1,1) with its keys"].outcome == "1/1"


def test_the_fresh_world_rows_give_the_estimates_and_passes_of_the_replication():
    need("rep_e2", "rep_e3")
    fresh = {t.label.split(", ")[0]: t for t in rf.diagnostic_tests(INP) if t.fresh}
    assert set(fresh) == {"Retention test", "Utility floor", "Cost ceiling"}
    t1 = [INP["rep_e2"]["t1"][k] for k in rf.TIER_KEYS]
    lo = min(t["delta_90"]["est"] for t in t1)
    hi = max(t["delta_90"]["est"] for t in t1)
    assert fresh["Retention test"].estimate == rf._pp_range([lo, hi])
    passes = sum(bool(t["equivalent"]) for t in t1)
    assert fresh["Retention test"].outcome == f"{passes}/{len(t1)}"
    per = rf._per_horizon(INP["rep_e3"])
    cost = [x for s, _ in rf.HORIZON_SYSTEMS for x in per[s]]
    ratio = max(x["serve_ratio_to_full_reread"] for x in cost)
    assert fresh["Cost ceiling"].estimate == f"at most {100 * ratio:.1f}\\%"
    assert fresh["Cost ceiling"].outcome == f"{sum(bool(x['T4_pass']) for x in cost)}/{len(cost)}"
    assert fresh["Utility floor"].outcome == rf._floor_counts(per)
    # the fresh rows repeat the main study's tests on the fresh worlds, with their thresholds
    main = {t.label: t for t in rf.diagnostic_tests(INP) if not t.fresh}
    for t in fresh.values():
        assert t.label in main and t.threshold == main[t.label].threshold
    assert not [t for t in rf.diagnostic_tests({**INP, "rep_e3": None}) if t.fresh and
                not t.label.startswith("Retention")]  # fmt: skip


@pytest.mark.paper
def test_the_committed_diagnostic_tests_table_is_the_staged_one():
    need("e2", "e3", "m1", "nat", "e6", "rep_e2", "rep_e3")
    tex = rf.tab_diagnostic_tests(rf.diagnostic_tests(INP))
    committed = (ROOT / "paper/generated/tab_diagnostic_tests.tex").read_text()
    assert committed == tex, "regenerate: figures --only tab_diagnostic_tests"
    assert r"\label{tab:diagnostic-tests}" in tex and tex.count(r"\begin{table*}") == 1
    assert "Test & Systems or units & Estimate & Passes & Estimate & Passes" in tex
    assert r"\multicolumn{2}{c}{Main study} & \multicolumn{2}{c@{}}{Replication}" in tex
    for word in ("regist", "predict", "verdict", r"\bholds", "exploratory", "same in", "fresh"):
        assert not re.search(word, tex), word
    assert "\nRetention test & \\holdmem{}(1,1), three tiers & " in tex
    # the replication beside the main study, a dash where it did not run the test
    leaving = next(ln for ln in tex.splitlines() if ln.startswith("Leaving test"))
    assert leaving.endswith(r"& \mtna & \mtna \\")
    notes = tex.split(r"\mtnotes{", 1)[1]
    for term in ("Thresholds: retention test 90", "point estimates", "Passing: ",
                 "---: not run in the replication"):  # fmt: skip
        assert term in notes, term
    caption = tex.split(r"\label{tab:diagnostic-tests}")[0]
    assert "Estimate" not in caption and "Threshold" not in caption


def test_the_outcome_counts_follow_the_systems_named():
    """A count per system, in the order the units name them; each name in one box."""
    per = {
        "holdmem_D1B1_source": [{"T3_pass": True}] * 3,
        "zero": [{"T3_pass": False}] * 3,
        "rag": [{"T3_pass": True}, {"T3_pass": False}],
    }
    assert rf._floor_counts(per) == r"3/3\,$\cdot$\,0/3\,$\cdot$\,1/2"
    assert rf._floor_units(per) == (
        r"\mbox{\holdmem{}(1,1)}~$\cdot$ \mbox{no-store BM25}~$\cdot$ \mbox{plain RAG}, "
        "three horizons"
    )
    assert rf._floor_counts({**per, "zero": []}) == r"3/3\,$\cdot$\,1/2"
    assert rf._passes([True, False, True]) == "2/3"


def test_the_diagnostic_tests_table_puts_the_replication_beside_the_main_study():
    """One row per test and units, the replication's estimate and passes beside the main
    study's (a dash where it did not run); a gap between tests, none between the rows of one
    test, and a cost ceiling stays with the utility floor before it."""
    t = rf.DiagnosticTest
    tests = [
        t("Retention test, a", "x", "1", "1/1"),
        t("Retention test, b", "x", "2", "0/1", passing=("W",)),
        t("Leaving test, c", "y", "3", "1/1"),
        t("Utility floor, d", "z", "4", "1/1"),
        t("Cost ceiling, the same systems", "w", "5", "1/1"),
        t("Retention test, a", "x", "6", "1/1", fresh=True),
        t("Utility floor, d", "z", "7", "0/1", fresh=True),
        t("Cost ceiling, the same systems", "w", "8", "1/1", fresh=True),
    ]
    tex = rf.tab_diagnostic_tests(tests)
    check_memtab(tex)
    body = tex.split(r"\midrule", 1)[1].split(r"\bottomrule")[0].strip().splitlines()
    gap = r"\addlinespace[2pt]"
    assert [ln.split(" & ")[0] if ln != gap else "gap" for ln in body] == [
        "Retention test", "", "gap", "Leaving test", "gap", "Utility floor", "Cost ceiling",
    ]  # fmt: skip
    assert body[0] == r"Retention test & a & 1 & 1/1 & 6 & 1/1 \\"
    assert body[1] == r" & b & 2 & 0/1 & \mtna & \mtna \\"
    assert body[5] == r"Utility floor & d & 4 & 1/1 & 7 & 0/1 \\"
    notes = tex.split(r"\mtnotes{", 1)[1]
    assert "Thresholds: retention test x, leaving test y, utility floor z, cost ceiling w." in notes
    assert "Passing: W of the b." in notes
    assert "mtgroup" not in tex


def test_the_passing_parts_are_kept_apart_by_semicolons():
    """A part may itself contain "and" (its units "the native systems and the two baselines"),
    so a comma between parts would read as one more passing unit."""
    t = rf.DiagnosticTest
    tex = rf.tab_diagnostic_tests([
        t("Retention test, five other writers", "x", "1", "2/5", passing=("A", "B")),
        t("Utility floor, native systems and the two baselines", "z", "2", "1/8", passing=("C",)),
    ])  # fmt: skip
    assert (
        "Passing: A and B of the five other writers; C of the native systems and the two "
        "baselines." in tex
    )


# --------------------------------------------------------------------------------- drawing
@pytest.mark.parametrize("name", sorted(rf.FIGURES))
def test_figures_draw_with_full_partial_and_no_inputs(name, tmp_path):
    fig = rf.FIGURES[name]
    partial = dict(INP)
    for key in ("g_b", "y", "rep_e2"):
        partial[key] = None  # a block not staged yet: the figure draws the rest
    outs = []
    for tag, inp in (("full", INP), ("partial", partial), ("empty", EMPTY)):
        d = tmp_path / tag
        d.mkdir()
        path = fig(inp, d)
        assert path.name == f"{name}.pdf" and path.read_bytes().startswith(b"%PDF")
        outs.append(path.read_bytes())
    assert outs[0] != outs[2]


@pytest.mark.paper
def test_the_figure_hook_draws_the_three_figures():
    figures = (ROOT / "src/memshift/analysis/figures.py").read_text()
    assert "round3_figures.FIGURES" in figures
    assert set(rf.FIGURES) == {"fig_writers", "fig_reader", "fig_misses"}
    for name in rf.FIGURES:
        assert (ROOT / "paper/figures" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    assert '"tab_diagnostic_tests" in a.only' in figures  # the diagnostic tests table, too

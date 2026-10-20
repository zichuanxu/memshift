"""The data functions and figures of `analysis.round3_figures` (round-3 figures and the
diagnostic-tests table): hand-built inputs for the logic, the staged analyses for what the paper
shows, and every figure drawn with full, partial and no inputs."""

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import round3_figures as rf
from memshift.analysis.paper_data import stage_json

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


def test_writer_points_join_the_sets_x_and_y_and_merge_identical_sets():
    d = rf.writer_points(hand_writers())
    assert [p["set"] for p in d["programs"]] == sorted(p["set"] for p in d["programs"])
    assert len(d["programs"]) == 5
    nox = next(p for p in d["programs"] if p["set"] == f"{LLAMA}#t0.7#s1")
    assert nox["x"] is None and nox["y"]["est"] == 0.02
    opus = [g for g in d["groups"] if g["writer"] == OPUS]
    assert len(opus) == 1 and opus[0]["n"] == 3 and len(opus[0]["sets"]) == 3
    llama = [g for g in d["groups"] if g["writer"] == LLAMA]
    assert len(llama) == 1 and llama[0]["n"] == 1 and llama[0]["x"]["est"] == 0.125
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


def test_coincide_merges_one_writers_sets_only():
    a = {"set": "a", "writer": "w", "x": est(0.5), "y": est(0.2)}
    b = {"set": "b", "writer": "w", "x": est(0.5002), "y": est(0.2)}
    c = {"set": "c", "writer": "v", "x": est(0.5), "y": est(0.2)}
    d = {"set": "d", "writer": "w", "x": est(0.6), "y": est(0.2)}
    groups = rf.coincide([a, b, c, d, {"set": "e", "writer": "w", "x": None, "y": est(0.1)}])
    assert [(g["writer"], g["n"]) for g in groups] == [("w", 2), ("v", 1), ("w", 1)]


def test_neighbouring_writers_take_opposite_sides_for_their_labels():
    def p(w: str, x: float, y: float) -> dict:
        return {"writer": w, "x": est(x), "y": est(y)}

    pts = [p("a", 0.43, 0.24), p("a", 0.53, 0.16), p("b", 0.45, 0.16), p("b", 0.54, 0.16),
           p("q", 0.05, 0.39)]  # fmt: skip
    assert rf._side("a", pts) == "above" and rf._side("b", pts) == "below"
    assert rf._side("q", pts) is None  # no other writer within 5 pp of its x range


def test_writer_points_without_inputs_are_empty():
    d = rf.writer_points(EMPTY)
    assert d == {"programs": [], "groups": [], "reference": None, "defwriters": []}


def test_staged_writer_points_are_the_staged_values():
    need("g_b", "e1")
    d = rf.writer_points(INP)
    per_set = INP["g_b"]["quantities"]["programs"]["branches"]["def"]["definition"]["per_set"]
    assert {p["set"] for p in d["programs"]} == set(per_set)
    for p in d["programs"]:
        assert p["y"]["est"] == per_set[p["set"]]["est"]
    ref = INP["e1"]["explore_class_split"]["D1"]["def"]["source"]["definition"]
    assert d["reference"]["est"] == ref["BxD_interaction_update"]["est"]
    assert sum(g["n"] for g in d["groups"]) == sum(p["x"] is not None for p in d["programs"])


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
    assert list(bars) == ["store", "qwen", "sonnet"]  # left to right within a shift
    r, g, b = (int(bars["store"][i : i + 2], 16) for i in (1, 3, 5))
    assert r == g == b > 0x99  # the store part: light grey, the same for both readers
    assert bars["qwen"] == fs.MODELS[rf.QWEN][1]
    assert bars["sonnet"] == fs.MODELS["claude-sonnet-5-5"][1]


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
    for key, _, fill, covered in rf.MISS_CAUSES:
        r, g, b = (int(fill[i : i + 2], 16) for i in (1, 3, 5))
        if covered:  # a write-time set could have contained the record: grey
            assert r == g == b, key
        else:  # it could not: warm, red above green above blue
            assert r > g > b, key
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
    # every row is an outcome of the systems: how many units pass
    for t in tests:
        assert t.outcome.startswith(("passes", "does not pass")), t.label
    row = {t.label: t for t in tests if not t.fresh}  # the main study's rows
    assert row[r"Retention test, \holdmem{}(1,1), three tiers (change)"].outcome == "passes 3 of 3"
    five = row[r"Retention test, \holdmem{}(1,1), five other writers (change)"]
    assert five.outcome.startswith("passes 2 of 5")


def test_the_fresh_world_rows_give_the_estimates_and_passes_of_the_replication():
    need("rep_e2", "rep_e3")
    fresh = {t.label.split(", ")[0]: t for t in rf.diagnostic_tests(INP) if t.fresh}
    assert set(fresh) == {"Retention test", "Utility floor", "Cost ceiling"}
    t1 = [INP["rep_e2"]["t1"][k] for k in rf.TIER_KEYS]
    lo = min(t["delta_90"]["est"] for t in t1)
    hi = max(t["delta_90"]["est"] for t in t1)
    assert fresh["Retention test"].estimate == rf._pp_range([lo, hi])
    passes = sum(bool(t["equivalent"]) for t in t1)
    assert fresh["Retention test"].outcome == f"passes {passes} of {len(t1)}"
    per = rf._per_horizon(INP["rep_e3"])
    cost = [x for s, _ in rf.HORIZON_SYSTEMS for x in per[s]]
    ratio = max(x["serve_ratio_to_full_reread"] for x in cost)
    assert fresh["Cost ceiling"].estimate == f"at most {100 * ratio:.1f}\\%"
    assert fresh["Cost ceiling"].outcome == (
        f"passes {sum(bool(x['T4_pass']) for x in cost)} of {len(cost)}"
    )
    assert fresh["Utility floor"].outcome == "passes: " + rf._floor_counts(per)
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
    assert r"\label{tab:diagnostic-tests}" in tex and tex.count(r"\begin{table}") == 1
    assert "test & systems or units & threshold & estimate & outcome" in tex
    for word in ("regist", "predict", "verdict", "holds", "exploratory", "same in"):
        assert word not in tex, word
    assert "passes 3 of 3" in tex and r"\emph{fresh worlds of the replication}" in tex
    assert "Retention test & \\holdmem{}(1,1), three tiers (change)" in tex


# --------------------------------------------------------------------------------- drawing
@pytest.mark.skipif(shutil.which("latex") is None, reason="no LaTeX for the figure text")
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


# ------------------------------------------------------------------------------ the mark
@pytest.mark.skipif(shutil.which("latex") is None, reason="no LaTeX for the figure text")
def test_interval_bar_has_exact_ends_and_an_outlined_variant():
    from memshift.analysis import figstyle as fs

    plt = fs.plt()
    fig, ax = plt.subplots()
    fs.interval_bar(ax, 1.0, 5.0, 2.0, 8.0, fs.BLUE)
    bar, stroke = ax.lines
    assert list(bar.get_xdata()) == [2.0, 8.0] and list(bar.get_ydata()) == [1.0, 1.0]
    assert bar.get_solid_capstyle() == "butt" and bar.get_linewidth() == fs.BAR_PT
    assert 0.3 <= bar.get_alpha() <= 0.4
    assert list(stroke.get_xdata()) == [5.0] and stroke.get_marker() == "|"
    assert stroke.get_markersize() > fs.BAR_PT  # a little longer than the bar is thick
    fig2, ax2 = plt.subplots()
    fs.interval_bar(ax2, 0.0, None, -1.0, 3.0, fs.VERMILION, filled=False, vertical=True)
    assert len(ax2.lines) == 4  # two long edges and two ends, no fill, no stroke
    assert all(line.get_alpha() is None for line in ax2.lines)
    edges = [line for line in ax2.lines if len(line.get_ydata()) == 2]
    assert all(list(line.get_ydata()) == [-1.0, 3.0] for line in edges)
    handle = fs.interval_handle(fs.BLUE, False, "fails")
    leg = fig.legend(handles=[handle], handler_map=fs.INTERVAL_HANDLERS)
    assert [t.get_text() for t in leg.get_texts()] == ["fails"]
    plt.close(fig)
    plt.close(fig2)


@pytest.mark.skipif(shutil.which("latex") is None, reason="no LaTeX for the figure text")
def test_ci_bar_draws_the_interval_white_only_where_it_crosses_a_dark_bar():
    from matplotlib.colors import to_hex

    from memshift.analysis import figstyle as fs

    plt = fs.plt()
    fig, ax = plt.subplots()
    fs.ci_bar(ax, 0.0, 10.0, 6.0, 14.0, fs.BLACK, 0.3)  # a dark bar from 0 to 10
    (bar,) = ax.patches
    assert bar.get_height() == 10.0 and bar.get_width() == pytest.approx(0.3)
    segs = {tuple(line.get_ydata()): to_hex(line.get_color()) for line in ax.lines
            if line.get_marker() in ("None", None, "")}  # fmt: skip
    assert segs == {(6.0, 10.0): "#ffffff", (10.0, 14.0): to_hex(fs.BLACK)}
    caps = {line.get_ydata()[0]: to_hex(line.get_color()) for line in ax.lines
            if line.get_marker() == "_"}  # fmt: skip
    assert caps == {6.0: "#ffffff", 14.0: to_hex(fs.BLACK)}
    fig2, ax2 = plt.subplots()
    fs.ci_bar(ax2, 0.0, -2.0, -5.0, 3.0, fs.SKY, 0.3, filled=False)  # outlined, below zero
    (bar,) = ax2.patches
    assert to_hex(bar.get_facecolor()) == "#ffffff"
    assert to_hex(bar.get_edgecolor()) == fs.SKY.lower()
    assert all(to_hex(line.get_color()) == to_hex(fs.BLACK) for line in ax2.lines)
    assert fs.ink(fs.SKY) == fs.ink(fs.PURPLE) == fs.ink(fs.VERMILION) == fs.BLACK
    plt.close(fig)
    plt.close(fig2)

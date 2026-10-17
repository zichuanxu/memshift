"""The data functions and figures of `analysis.round3_figures` (round-3 figures and the forest
plot of the registered predictions): hand-built inputs for the logic, the staged analyses for what
the paper shows, and every figure drawn with full, partial and no inputs."""

import json
import math
import re
import shutil
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import round3_figures as rf

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
    ]  # S_del is not drawn; the Claude writers in the registered order
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


def test_staged_writer_points_are_the_registered_values():
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
    """The exploratory axes reproduce block H's registered quantities where both report them."""
    need("writers", "h_sonnet", "h_opus")
    dw = INP["writers"]["definition_writers"]
    assert dw["pairing_ok"] is True
    for key, model in (("h_sonnet", "claude-sonnet-5-5"), ("h_opus", "claude-opus-5-5")):
        h = INP[key]["quantities"]["writers"][model]
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


def test_staged_miss_rows_add_up_and_show_both_studies():
    need("miss")
    rows = rf.miss_rows(INP)
    assert {r["stage"] for r in rows} == {"main", "rep"}
    assert len(rows) == 12  # two studies, three shifts, two classes
    for r in rows:
        assert sum(r["shares"].values()) == pytest.approx(1.0)
        assert r["uncoverable"] > 0.5  # most misses: records no write-time set can contain


# ---------------------------------------------------------------------------- registered
PANEL_KEYS = {key for key, *_ in rf.PANELS}


def test_registered_rows_without_inputs_are_empty():
    rows = rf.registered_rows(EMPTY)
    assert set(rows) == PANEL_KEYS
    assert all(v == [] for v in rows.values())


def test_a_prediction_that_is_not_judged_has_no_row():
    inp = {
        "q": {
            "predictions": {"Q1": {"verdict": "not_judged"}, "Q2": {"verdict": "holds"}},
            "quantities": {"Q1": {"difference": est(0.0)}, "Q2": {"D1": {"unnamed": est(0.4)}}},
        }
    }
    rows = rf.registered_rows(inp)
    assert [r.label for k in rows for r in rows[k]] == ["Interaction, unnamed probes"]
    (q2,) = rows["generality"]
    assert q2.points == [
        {"est": 0.4, "lo": 0.4, "hi": 0.4, "shift": "def", "sub": rf.SHIFT_TEXT["def"],
         "verdict": rf.HOLDS}
    ]  # fmt: skip
    assert q2.bar == ("gt", (0.0,)) and q2.panel == "generality" and q2.scale == "pp"


def test_subrows_follow_the_points_and_merge_identical_marks():
    p = {"est": 0.0, "lo": 0.0, "hi": 0.0, "verdict": rf.HOLDS}
    row = rf.Row("x", "missed", ("abs_lt", (2.0,)), [
        {**p, "shift": "src", "sub": "a"}, {**p, "shift": "src", "sub": "a"},
        {**p, "shift": "def", "sub": "b"},
        {**p, "est": 0.01, "hi": 0.02, "shift": "def", "sub": "b"},
    ])  # fmt: skip
    subs = row.subrows()
    assert [s for s, _ in subs] == ["a", "b"]
    assert [len(pts) for _, pts in subs] == [1, 2]  # two keys with equal values are one mark


def _counted_row(label: str, *verdicts: str) -> rf.Row:
    p = {"est": 0.1, "lo": 0.0, "hi": 0.2, "shift": None}
    points = [{**p, "sub": f"s{i}", "verdict": v} for i, v in enumerate(verdicts)]
    return rf.Row(label, "binding", ("gt", (0.0,)), points)


def test_registered_counts_classify_each_row_by_its_pass_and_fail_points():
    rows = {
        "binding": [
            _counted_row("all hold", rf.HOLDS, rf.HOLDS),
            _counted_row("all fail", rf.FAILS),
            _counted_row("mixed", rf.HOLDS, rf.FAILS, rf.HOLDS),
            _counted_row("reading only", rf.READING),  # no pass or fail: not counted
            _counted_row("hold and reading", rf.HOLDS, rf.READING),  # the reading is left out
        ],
        "ratio": [_counted_row("fail and reading", rf.READING, rf.FAILS)],
    }
    counts = rf.registered_counts(rows)
    assert counts == {"RegJudged": 5, "RegHeld": 2, "RegFailed": 2, "RegMixed": 1}
    assert counts["RegJudged"] == counts["RegHeld"] + counts["RegFailed"] + counts["RegMixed"]
    assert rf.registered_counts(rf.registered_rows(EMPTY)) == dict.fromkeys(rf.COUNT_MACROS, 0)
    tex = rf.counts_tex(counts)
    for name, value in counts.items():
        assert f"\\newcommand{{\\{name}}}{{{value}}}" in tex
    assert all(line.startswith(("%", "\\newcommand")) for line in tex.splitlines())


@pytest.mark.paper
def test_the_committed_counts_are_those_of_the_staged_rows():
    need("gate", "e1", "ctl", "u", "q", "y", "h_sonnet", "n")
    rows = rf.registered_rows(INP)
    counts = rf.registered_counts(rows)
    judged = [r for k in rows for r in rows[k] if {p["verdict"] for p in r.points} - {rf.READING}]
    assert counts["RegJudged"] == len(judged) > 0
    assert counts["RegJudged"] == counts["RegHeld"] + counts["RegFailed"] + counts["RegMixed"]
    every = rf.paper_counts(INP)
    assert {k: every[k] for k in rf.COUNT_MACROS} == counts  # the figure's rows, as before
    for fig, tab, both in zip(rf.COUNT_MACROS, rf.TABLE_COUNT_MACROS, rf.ALL_COUNT_MACROS,
                              strict=True):  # fmt: skip
        assert every[both] == every[fig] + every[tab]
    committed = (ROOT / "paper/generated/registered_counts.tex").read_text()
    assert committed == rf.counts_tex(every), "regenerate: figures --only fig_registered"


# ------------------------------------------------- registered verdicts outside the figure
def test_a_registered_row_holds_fails_or_is_mixed_by_its_units_or_its_joint_rule():
    def reg(units: list[bool], **kw: Any) -> rf.Registered:
        return rf.Registered("g", "x", "c", "---", units, **kw)

    assert reg([True, True]).verdict() == rf.HOLDS
    assert reg([False]).verdict() == rf.FAILS
    assert reg([True, False, True]).verdict() == rf.MIXED
    assert reg([]).verdict() is None
    # a rule over its units (a count to reach) decides by the staged verdict, not by the units
    assert reg([True, False], joint=False).verdict() == rf.FAILS
    assert reg([True, False], joint=True).verdict() == rf.HOLDS
    assert reg([True, False, True]).verdict_text() == "mixed (2 of 3)"
    assert reg([True]).verdict_text() == "holds"
    assert reg([True], test=True, note="passes 1 of 1").verdict_text() == "passes 1 of 1"
    rows = [reg([True]), reg([False]), reg([True, False]), reg([False], test=True)]
    assert rf.table_counts(rows) == dict(zip(rf.TABLE_COUNT_MACROS, (3, 1, 1, 1), strict=True))
    assert rf.table_counts([]) == dict.fromkeys(rf.TABLE_COUNT_MACROS, 0)


def test_registered_tests_without_inputs_are_empty():
    assert rf.registered_tests(EMPTY) == []
    assert rf.paper_counts(EMPTY) == dict.fromkeys(
        (*rf.COUNT_MACROS, *rf.TABLE_COUNT_MACROS, *rf.ALL_COUNT_MACROS), 0
    )


def test_staged_registered_tests_carry_the_registered_verdicts():
    need("e1", "e2", "e3", "m1", "ctl", "nat", "e6", "e4x", "rendered")
    tests = rf.registered_tests(INP)

    def row(prefix: str) -> rf.Registered:
        found: rf.Registered = next(t for t in tests if t.label.startswith(prefix))
        return found

    # D=1 survival of a switch far above the checker pass rate (G.1: falsified)
    surv = row("$D{=}1$ survival minus checker")
    assert surv.verdict() == rf.FAILS and len(surv.units) == 3
    dc = INP["e2"]["d1_vs_checker"]["same"]
    assert dc["d1_b1_survival"] - dc["checker_pass_rate_calibration"] > rf.EPS
    # D survival per switch: judged per switch (decisions.md 2026-09-29), 6 of 8
    mb = row("Survival of $D{=}1$ minus $D{=}0$")
    assert len(mb.units) == 8 and mb.verdict() == rf.MIXED
    assert sum(mb.units) == sum(v["holds"] for v in INP["m1"]["m1b_d_protects_switch"].values())
    ctl = INP["ctl"]["verdicts"]
    assert row("Query-time view minus zero system, other").verdict() == rf.FAILS
    assert ctl["Z2"]["status"] == "fails"
    assert {t.label.split(" ")[0] for t in tests if t.group == "Controls"} >= {"Oracle", "Thinking"}
    assert row("Calendar interaction").verdict() == rf.FAILS  # did not reproduce
    assert row("Import without the keys").verdict() == rf.HOLDS
    s4 = INP["nat"]["stage4"]
    assert row("Comparisons inside, all").verdict() == (rf.HOLDS if s4["n3_holds"] else rf.FAILS)
    assert row("Comparisons inside the inferred").verdict() == rf.MIXED
    assert row("Inferred label").verdict() == rf.HOLDS
    # the figure's rows are not repeated: no check (ii) or (iv) of the families, no gain check
    labels = [t.label for t in tests]
    assert not any("Gain check" in lab or "$B{\\times}D$" in lab for lab in labels)
    # every prediction has a verdict; the diagnostic tests are listed and not counted
    for t in tests:
        assert t.units and (t.test or t.verdict() in (rf.HOLDS, rf.FAILS, rf.MIXED))
    diag = [t for t in tests if t.test]
    assert {t.group for t in diag} == {rf.TEST_PART} and len(diag) >= 7
    assert row("Retention test, three").verdict() == rf.HOLDS  # the three registered tiers
    assert row("Retention test, five").verdict() == rf.MIXED
    counts = rf.table_counts(tests)
    assert counts["RegTabJudged"] == len(tests) - len(diag) > 0


@pytest.mark.paper
def test_the_committed_table_is_the_staged_one_and_is_set_in_g1():
    need("e1", "e2", "e3", "m1", "ctl", "nat")
    tex = rf.tab_registered_tests(rf.registered_tests(INP))
    committed = (ROOT / "paper/generated/tab_registered_tests.tex").read_text()
    assert committed == tex, "regenerate: figures --only fig_registered"
    assert r"\label{tab:registered-tests}" in tex and tex.count(r"\begin{table}") == 1
    caption = tex.split(r"\label")[0]
    counts = rf.table_counts(rf.registered_tests(INP))
    assert f"Of the {counts['RegTabJudged']} predictions" in caption
    app = (ROOT / "paper/sections/app_results.tex").read_text()
    g1 = app.split(r"\subsection{Where the Interaction Lies}")[0]
    assert r"\input{generated/tab_registered_tests}" in g1
    main = (ROOT / "paper/sections/results.tex").read_text()
    for name in rf.ALL_COUNT_MACROS:  # the main text quotes the totals
        assert f"\\{name}" in main and f"\\{name}" in g1


def test_u5_and_g3_rows_appear_once_staged():
    fam = {"pooled_definition_D1_Dm_src_def": {"upsert": {"R": est(0.02, 0.0, 0.05)}}}
    u = {
        "predictions": {"U5": {"verdict": "holds"}, "U3": {"verdict": "judged"}},
        "quantities": {"families": {"main": fam, "google/gemma-4-12B-it": fam, "x|y|3": fam}},
    }
    g = {
        "predictions": {"G3": {"verdict": "fails"}},
        "quantities": {
            "programs": {"branches": {"def": {"definition": {"two_way": est(0.1, -0.02, 0.2)}}}}
        },
    }
    rows = rf.registered_rows({"u": u, "g_b": g})
    u5 = next(r for r in rows["ratio"] if r.label.startswith("Upsert, other families"))
    assert len(u5.points) == 1 and u5.points[0]["verdict"] == rf.HOLDS  # the M2 family only
    assert u5.points[0]["sub"] == "gemma-4-12B"  # the label of the other family rows
    u3 = next(r for r in rows["ratio"] if r.label.startswith("Upsert, share"))
    assert u3.points[0]["verdict"] == rf.READING and u3.bar == ("reading", (0.5, 0.8))
    assert u3.points[0]["sub"] == "pooled"
    g3 = next(r for r in rows["generality"] if r.label.startswith("Program sets"))
    assert g3.points[0]["verdict"] == rf.FAILS and g3.points[0]["lo"] == -0.02
    assert not any(r.label.startswith("Upsert, other") for r in rf.registered_rows({"u": u | {
        "quantities": {"families": {"main": fam}}}})["ratio"])  # fmt: skip


def test_staged_registered_rows_carry_the_verdicts_of_the_analyses():
    need("gate", "e1", "ctl", "u", "q", "y", "h_sonnet", "n")
    rows = rf.registered_rows(INP)
    every: list[rf.Row] = [r for k in rows for r in rows[k]]
    for k, rs in rows.items():
        for r in rs:
            assert r.points and r.panel == k and r.scale == ("ratio" if k == "ratio" else "pp")
            for p in r.points:
                assert p["verdict"] in {rf.HOLDS, rf.FAILS, rf.READING} and p["sub"]
                assert p["lo"] <= p["est"] + 1e-12 and p["est"] <= p["hi"] + 1e-12
                assert all(math.isfinite(p[k]) for k in ("est", "lo", "hi"))
            labels = [s for s, _ in r.subrows()]
            assert len(labels) == len(set(labels))  # one sub-row per shift or arm

    def row(prefix: str) -> rf.Row:
        return next(r for r in every if r.label.startswith(prefix))

    assert row("Gain check:").points[0]["verdict"] == rf.HOLDS
    assert {p["verdict"] for p in row("Unnamed minus named").points} == {rf.FAILS}
    assert {p["verdict"] for p in row("Sonnet minus Qwen3.5-9B").points} == {rf.FAILS}
    oracle_stab = row("Oracle minus source-epoch key, Stability")
    assert {p["verdict"] for p in oracle_stab.points} == {rf.FAILS}
    assert len(oracle_stab.subrows()) == 6  # D=0 and D=1 after each shift
    cited = row("Cited-cluster key: binding's")
    assert [p["verdict"] for p in cited.points][0] == rf.FAILS  # S_src
    assert row("Claude Sonnet 5.5 writes").points[0]["verdict"] == rf.READING
    assert row("Claude Sonnet 5.5 writes").points[0]["est"] == pytest.approx(
        INP["h_sonnet"]["predictions"]["H1"]["tests"][0]["value"]
    )
    assert row("Upsert minus append-only").points[0]["est"] == pytest.approx(
        INP["u"]["quantities"]["families"]["main"]["U4"]["difference"]["est"]
    )
    ctl = INP["ctl"]["verdicts"]  # a control's points carry its prediction's word
    assert {p["verdict"] for p in row("The same, all items").points} == {ctl["T3"]["status"]}
    assert {p["verdict"] for p in row("Repair, $D{=}1$ minus a").points} == {ctl["F1"]["status"]}
    assert len(row("Leak check:").subrows()) == 3  # both keys are exactly 0 after each shift
    pooled = row("Cited-cluster key: binding's effect, pooled")  # Section 4.3: both failed
    assert [p["verdict"] for p in pooled.points] == [rf.FAILS, rf.FAILS, rf.HOLDS]
    assert row("Other families: binding's effect").bar == ("gt", (2.0,))  # check (ii)
    # the printed labels carry no registration IDs, block letters or path numbers
    for r in every:
        assert not re.match(r"(P-|[A-Z][0-9]|[A-Z]:|Path)", r.label), r.label


def test_every_panel_axis_holds_its_rows_and_cut_points():
    need("gate", "e1", "ctl", "u")
    for key, rows in rf.registered_rows(INP).items():
        if not rows:
            continue
        k = 1.0 if key == "ratio" else 100.0
        lo, hi = rf._span(rows, k)
        for r in rows:
            for p in r.points:
                assert lo < k * p["lo"] and k * p["hi"] < hi
            for c in r.bar[1]:
                assert lo < c < hi


# --------------------------------------------------------------------------------- drawing
@pytest.mark.skipif(shutil.which("latex") is None, reason="no LaTeX for the figure text")
@pytest.mark.parametrize("name", sorted(rf.FIGURES))
def test_figures_draw_with_full_partial_and_no_inputs(name, tmp_path):
    fig = rf.FIGURES[name]
    partial = dict(INP)
    for key in ("g_b", "y", "rep_e1", "ctl", "h_opus"):
        partial[key] = None  # a block not staged yet: the figure draws the rest
    outs = []
    for tag, inp in (("full", INP), ("partial", partial), ("empty", EMPTY)):
        d = tmp_path / tag
        d.mkdir()
        path = fig(inp, d)
        assert path.name == f"{name}.pdf" and path.read_bytes().startswith(b"%PDF")
        outs.append(path.read_bytes())
        if name == "fig_registered":  # two pages
            assert (d / "fig_registered_2.pdf").read_bytes().startswith(b"%PDF")
    assert outs[0] != outs[2]


@pytest.mark.paper
def test_the_figure_hook_draws_the_four_figures():
    figures = (ROOT / "src/memshift/analysis/figures.py").read_text()
    assert "round3_figures.FIGURES" in figures
    assert set(rf.FIGURES) == {"fig_writers", "fig_reader", "fig_misses", "fig_registered"}
    for name in (*rf.FIGURES, "fig_registered_2"):
        assert (ROOT / "paper/figures" / f"{name}.pdf").read_bytes().startswith(b"%PDF")
    app = (ROOT / "paper/sections/app_results.tex").read_text()
    assert "figures/fig_registered_2.pdf" in app and "\\ContinuedFloat" in app
    for name in rf.COUNT_MACROS:  # the counts of its rows are generated and quoted by macro
        assert f"\\{name}" in app


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

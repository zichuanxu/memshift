"""Paper numbers and tables of the control cells (`analysis.controls_paper`): a stage that is not
staged renders only MISSING markers, a staged one carries the estimates, verdicts and counts the
analysis produced, the tables are well-formed and compile, and the committed generated files are
current."""

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from memshift.analysis import controls as ctl
from memshift.analysis import controls_paper as cp
from memshift.analysis import paper_tables
from memshift.analysis.paper_data import MISSING, camel
from memshift.analysis.paper_numbers import _macro, with_estimates
from tests.test_controls import Builder, ev
from tests.test_paper_stage4 import check_tex

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "paper/results_sources.json").read_text())


@pytest.fixture(scope="module")
def staged() -> dict:
    return ev(Builder().table())


@pytest.fixture(scope="module")
def partial() -> dict:
    b = Builder()
    b.drop |= {"zeroDm", "D1B0"}
    return ev(b.table())


# ---------------------------------------------------------------------------------- macros
def test_unstaged_control_cells_render_only_missing_markers():
    macros = cp.numbers(None)
    assert len(macros) > 450
    for name, value in macros.items():
        _macro(name, value)  # letters only
        assert name.startswith("Ctl") and value == MISSING, (name, value)


def test_every_contrast_prediction_and_check_has_a_macro_name():
    names = set(cp.numbers(None))
    for key in ctl.contrast_keys():
        assert f"Ctl{camel(key)}" in names, key
    for pred in ctl.PREDICTIONS:
        for suffix in ("Verdict", "Hits", "Total", "Need", "Count"):
            assert f"Ctl{camel(pred)}{suffix}" in names
    for tag in ("Hash", "Probes", "Anchor", "Audit", "ScopeCited"):
        assert f"CtlCheck{tag}Pass" in names
    for extra in ("CtlGoneStrong", "CtlTonePresentation", "CtlChecksPassed", "CtlSummaryHolds"):
        assert extra in names


def test_the_full_access_control_has_its_macros_for_src_and_def_only():
    names = set(cp.numbers(None))
    for name in (
        "CtlFoneVerdict",
        "CtlFoneReading",
        "CtlFRepSrcDefinition",
        "CtlFRepDefDefinition",
        "CtlFRepmechSrcDefinition",
        "CtlFRepSrcUntruncated",
        "CtlFRepmechDefUntruncated",
        "CtlFGainRepSrcDefinition",
        "CtlFGainUpdDefDefinition",
        "CtlFGainRepSrcUntruncated",
        "CtlFullLimit",
        "CtlFullTruncatedAll",
        "CtlFullTruncatedShareSrc",
        "CtlFullWritesDef",
        "CtlFullCloseFriendTruncatedSrc",
        "CtlFullCloseFriendMessagesTruncatedSrc",
        "CtlFullVipSenderEmailDroppedDef",
        "CtlFullVipSenderContactsTruncatedAll",
        "CtlFullWorkingHoursNotesTruncatedSrc",
        "CtlFullAsRegistered",
        "CtlFullWhichTruncated",
    ):
        assert name in names, name
    # the orphan branch was not run: no contrast and no note macro for it
    assert not [n for n in names if n.startswith(("CtlFRep", "CtlFGain")) and "Orphan" in n]
    assert not [n for n in names if n.startswith("CtlFull") and n.endswith("Orphan")]
    assert len([n for n in names if n.startswith("CtlFull")]) > 100


def test_staged_macros_carry_estimates_intervals_verdicts_and_counts(staged):
    m = with_estimates(cp.numbers(staged))
    # an estimate with its interval, and the estimate alone
    assert m["CtlZUpdSrcDefinition"] == "$+$80.0\\,pp [$+$80.0, $+$80.0]"
    assert m["CtlZUpdSrcDefinitionEst"] == "$+$80.0\\,pp"
    assert m["CtlGUpdDzeroSrcAll"].startswith("0.0\\,pp")
    assert m["CtlRRatioSrcDefinition"].startswith("0.65")  # (0.80 - 0.50) / (0.96 - 0.50)
    # verdicts, counts and flags
    assert m["CtlGoneVerdict"] == "holds" and m["CtlGoneCount"] == "6 of 6 contrasts"
    assert (m["CtlGthreeHits"], m["CtlGthreeNeed"], m["CtlGthreeTotal"]) == ("6", "5", "6")
    assert m["CtlGoneStrong"] == "yes" and m["CtlTonePresentation"] == "no"
    assert m["CtlZtwoCount"] == "8 of 8 contrasts" and m["CtlRoneCount"] == "3 of 3 contrasts"
    assert (
        m["CtlSummaryHolds"] == "20"
        and m["CtlSummaryFails"] == "0"
        and m["CtlSummaryTotal"] == "20"
    )
    # the D0full control: verdict, reading, the gap with its interval, the sensitivity analysis
    assert m["CtlFoneVerdict"] == "holds" and m["CtlFoneCount"] == "4 of 4 contrasts"
    assert m["CtlFoneReading"] == "execution"
    assert m["CtlFRepSrcDefinitionEst"] == "$+$66.0\\,pp"
    assert m["CtlFRepmechDefDefinitionEst"] == "$+$66.0\\,pp"
    assert m["CtlFGainRepSrcDefinitionEst"] == "$+$16.0\\,pp"
    assert m["CtlFRepSrcUntruncatedEst"].startswith("$+$6")  # 94 - 28 over 98 items
    assert "CtlFRepOrphanDefinition" not in m
    assert m["CtlWorlds"] == "3"
    # the reported quantities
    assert m["CtlProgOkRaAll"].startswith("100.0\\%") and m["CtlProgOkRatioAll"].startswith("1.00")
    assert m["CtlThinkWritesAll"] == "90" and m["CtlThinkLimit"] == "8,192"
    assert m["CtlThinkUnreadableAll"].startswith("0.0\\%")
    assert m["CtlRaLineageFalseOrphanAll"].startswith("10.0\\%")
    assert m["CtlOverActiveProjectRepairedRaOrphan"] == "6"  # worlds x checkpoints
    assert m["CtlOverActiveProjectWithdrawnRaOrphan"] == "0"
    # the truncation counts of the D0full writer, pooled, per definition and per source
    assert m["CtlFullLimit"] == "25,000" and m["CtlFullWritesAll"] == "72"
    assert (m["CtlFullWritesSrc"], m["CtlFullTruncatedAll"], m["CtlFullBySourceAll"]) == (
        "36", "24", "24",
    )  # fmt: skip
    assert m["CtlFullNewestAll"] == "0" and m["CtlFullDroppedSrc"] == "312"
    assert m["CtlFullTruncatedShareAll"].startswith("33.3\\%")
    assert (m["CtlFullTokensMedianAll"], m["CtlFullTokensMaxAll"]) == ("9,000", "24,900")
    assert m["CtlFullCloseFriendTruncatedSrc"] == "6" and m["CtlFullCloseFriendDroppedSrc"] == "72"
    assert m["CtlFullCloseFriendMessagesTruncatedAll"] == "12"
    assert m["CtlFullVipSenderEmailDroppedDef"] == "240"
    assert m["CtlFullVipSenderContactsDroppedDef"] == "0"  # the small source is given whole
    assert m["CtlFullActiveProjectTasksTruncatedAll"] == "0"
    assert m["CtlFullAsRegistered"] == "yes"
    assert m["CtlFullWhichTruncated"] == "close\\_friend, vip\\_sender"


NOTHING_TO_SHOW = re.compile(
    r"CtlChecks?\w*"  # the checks of a run: the synthetic table has none
    r"|CtlRa\w*Free"  # the synthetic items have no free-text class
    r"|CtlRaFinalUndeclaredOfFailed\w+"  # and no failed item
    r"|Ctl[GZ]Stab\w*Definition"  # and no definition whose gold does not change
)


def test_every_estimate_and_verdict_macro_has_a_number_when_everything_ran(staged):
    macros = cp.numbers(staged)
    missing = sorted(k for k, v in macros.items() if v == MISSING)
    assert all(NOTHING_TO_SHOW.fullmatch(k) for k in missing), missing[:10]
    for key in ctl.contrast_keys():
        if not NOTHING_TO_SHOW.fullmatch(f"Ctl{camel(key)}"):
            assert macros[f"Ctl{camel(key)}"] != MISSING, key
    for pred in ctl.PREDICTIONS:
        assert macros[f"Ctl{camel(pred)}Verdict"] in ("holds", "fails", "not judged")


def test_a_prediction_that_was_not_judged_says_so_and_a_missing_cell_leaves_missing_contrasts(
    partial,
):
    m = cp.numbers(partial)
    assert m["CtlZoneVerdict"] == "not judged" and m["CtlZoneCount"] == "not judged"
    assert m["CtlZoneHits"] == MISSING and m["CtlZUpdSrcDefinition"] == MISSING
    assert m["CtlRtwoVerdict"] == "not judged" and m["CtlGoneVerdict"] == "holds"
    assert m["CtlSummaryNotJudged"] == "6" and m["CtlSummaryHolds"] == "14"
    assert m["CtlFoneVerdict"] == "holds"  # D1 and D0full were run in `partial`
    assert "CtlZUpdSrcDefinitionEst" not in with_estimates(m)  # `with_estimates` invents none


def test_macro_values_compile_as_text(staged, tmp_path):
    if shutil.which("pdflatex") is None:
        pytest.skip("no LaTeX")
    macros = cp.numbers(staged)
    (tmp_path / "numbers.tex").write_text(
        "\n".join(_macro(k, v) for k, v in sorted(macros.items()))
    )
    lines = "".join(f"\\noindent\\texttt{{{k}}} {{\\{k}}}\\par\n" for k in sorted(macros))
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,newtxtext,newtxmath}",
                r"\input{numbers}\begin{document}",
                lines,
                r"\end{document}",
            ]
        )
    )
    done = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-1500:]


# ---------------------------------------------------------------------------------- tables
def test_tables_are_wellformed_staged_partly_staged_and_unstaged(staged, partial):
    for view in (staged, partial, None):
        for name, tex in cp.tables(view).items():
            check_tex(tex)
            assert tex.count("\\begin{table}") == (1 if name == "tab_controls.tex" else 8)
    assert MISSING in cp.tab_controls(None) and MISSING in cp.tab_controls_full(None)
    # a table cell that cannot be computed is MISSING, the rest stay
    assert cp.tab_controls(partial).count(MISSING) > 10
    assert cp.tab_controls(partial).count(MISSING) < cp.tab_controls(None).count(MISSING)


def test_compact_table_rows_are_the_registered_predictions_with_their_verdicts(staged):
    tex = cp.tab_controls(staged)
    for pred in (
        "P-G1, P-G2",
        "P-G3",
        "P-R1",
        "P-R2",
        "P-T1, P-T3",
        "P-T2",
        "P-M1",
        "P-M2",
        "P-M3",
        "P-K1",
        "P-K2",
        "P-Z1",
        "P-Z2",
        "P-Z3",
        "P-D1",
        "P-D2",
    ):
        assert f"\n{pred} &" in tex, pred
    assert "registered before running" in tex  # the caption says it
    assert tex.count("\\pass\\,6/6") >= 2  # G1 and G2 carry a verdict each, T1 and T3 as well
    rows = [r for r in cp.ROWS if r.verdict]
    assert {p for r in rows for p in r.verdict} == set(
        ctl.PREDICTIONS
    )  # every prediction has a row
    # the cells a prediction judges are in bold, the others are not
    g = next(line for line in tex.splitlines() if line.startswith("P-G1, P-G2"))
    assert g.count("\\textbf{") == 6  # definition and all items, three branches each
    z1 = next(line for line in tex.splitlines() if line.startswith("P-Z1 &"))
    assert z1.count("\\textbf{") == 2  # the two branches P-Z1 judges
    assert "$+$80.0" in z1


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.removesuffix(r"\\").split(" & ")]


def cells_of(tex: str, start: str) -> list[str]:
    """The cells of the table row that begins with `start`."""
    return split_row(next(line for line in tex.splitlines() if line.startswith(start)))


def table_of(tex: str, label: str) -> str:
    """The table with `\\label{tab:<label>}` (its caption, the label and its rows)."""
    return tex.split("\\begin{table}")[
        [f"\\label{{tab:{label}}}" in t for t in tex.split("\\begin{table}")].index(True)
    ]


def contrast_rows(tex: str, label: str) -> list[list[str]]:
    """The rows of the full table (contrast, items, S_src, S_def, orphan) of one section."""
    return [
        split_row(line)
        for line in table_of(tex, f"controls_full_{label}").splitlines()
        if line.count(" & ") == 4 and not line.startswith("contrast")
    ]


def test_compact_table_has_the_full_access_rows_for_src_and_def_only(staged):
    cells = cells_of(cp.tab_controls(staged), "P-F1 &")
    # predicted, contrast, definition (3 branches), non-definition (3), all items (3), verdict
    assert len(cells) == 12 and cells[0] == "P-F1" and "full-access" in cells[1]
    assert cells[2:5] == [r"\textbf{$+$66.0}", r"\textbf{$+$66.0}", cp.UNDEFINED]
    assert cells[5:8] == [cp.UNDEFINED] * 3  # the prediction is about the definition items
    assert cells[8:11] == ["$+$32.2", "$+$32.2", cp.UNDEFINED]  # not judged, so not in bold
    assert cells[11] == r"\pass\,4/4"
    lines = cp.tab_controls(staged).splitlines()
    mech = split_row(lines[lines.index(next(x for x in lines if x.startswith("P-F1 &"))) + 1])
    assert mech[:2] == ["", r"\quad mechanistic repair"]  # the row after P-F1's
    assert mech[2:5] == [r"\textbf{$+$66.0}", r"\textbf{$+$66.0}", cp.UNDEFINED]
    assert "P-F1" in cp.tab_controls(staged) and "$\\Ssrc$ and $\\Sdef$ only" in cp.CAPTION


def test_the_full_access_rows_are_missing_when_unstaged_and_not_judged_without_the_cell():
    cells = cells_of(cp.tab_controls(None), "P-F1 &")
    assert cells[2:4] == [MISSING, MISSING] and cells[4] == cp.UNDEFINED and cells[11] == MISSING
    b = Builder()
    b.drop |= {"D0fullB1"}
    cells = cells_of(cp.tab_controls(ev(b.table())), "P-F1 &")
    assert cells[2:4] == [MISSING, MISSING] and cells[4] == cp.UNDEFINED  # `---`: no orphan run
    assert cells[11] == cp.NOT_JUDGED
    m = cp.numbers(ev(b.table()))
    assert m["CtlFoneVerdict"] == "not judged" and m["CtlFoneReading"] == MISSING
    assert m["CtlFRepSrcDefinition"] == MISSING and m["CtlFullTruncatedAll"] == MISSING
    assert m["CtlFullCloseFriendMessagesTruncatedSrc"] == MISSING
    assert m["CtlFullAsRegistered"] == "no"  # the flag is a statement about the notes: none given
    # the full table: MISSING where nothing is staged, `---` where the contrast cannot be computed
    # (as every contrast of a cell that was not run) and for the branch the control was not run on
    rows = contrast_rows(cp.tab_controls_full(None), "e")
    assert len(rows) == 12 and all(r[2:] == [MISSING, MISSING, cp.UNDEFINED] for r in rows)
    rows = contrast_rows(cp.tab_controls_full(ev(b.table())), "e")
    assert len(rows) == 12 and all(r[2:] == [cp.UNDEFINED] * 3 for r in rows)
    trunc = table_of(cp.tab_controls_full(ev(b.table())), "controls_truncation")
    assert cells_of(trunc, "close\\_friend &")[2:] == [MISSING] * 6


def test_the_full_table_carries_the_full_access_contrasts_the_sensitivity_and_the_truncations(
    staged,
):
    tex = cp.tab_controls_full(staged)
    f = table_of(tex, "controls_full_e")
    rows = contrast_rows(tex, "e")  # four families, three classes each
    assert len(rows) == 12 and all(r[4] == cp.UNDEFINED and "scriptsize" in r[2] for r in rows)
    assert [r[1] for r in rows[:3]] == ["all items", "definition", "definition, truncated left out"]
    assert "Full-access writer (P-F1" in f and "close\\_friend and vip\\_sender" in f
    for label in (
        "Repair, $D{=}1$ $-$ full-access $D{=}0$",
        "Mechanistic repair, $D{=}1$ $-$ full-access $D{=}0$",
        "Repair, full-access $-$ $D{=}0$ (access gain)",
        "Update, full-access $-$ $D{=}0$ (access gain)",
        "definition, truncated left out",
    ):
        assert label in f, label
    # the reported table and the truncation table carry the notes' counts
    assert "full-access writes (definition items re-derived after the shift) & 72 \\\\" in tex
    assert "of which truncated (records of a declared source left out) & 24 \\\\" in tex
    assert "approximate tokens given, median and maximum (cap 25,000) & 9,000, 24,900" in tex
    assert "the two registered & yes (close\\_friend, vip\\_sender)" in tex
    trunc = table_of(tex, "controls_truncation")
    assert cells_of(trunc, "close\\_friend &") == [
        "close\\_friend", "messages", "6", "6", "72", "6", "6", "72",
    ]  # fmt: skip
    assert cells_of(trunc, "vip\\_sender &") == [
        "vip\\_sender", "email", "6", "6", "240", "6", "6", "240",
    ]  # fmt: skip
    assert cells_of(trunc, " & contacts &")[1:] == ["contacts", "6", "0", "0", "6", "0", "0"]
    assert cells_of(trunc, "active\\_project &")[2:] == ["6", "0", "0", "6", "0", "0"]
    assert "Counts over the re-derivations" in trunc and "estimates in pp" not in trunc.lower()


def test_compact_table_shows_a_failed_and_a_not_judged_verdict():
    b = Builder({("D1B1gold", "definition", "orphan"): 0.50})
    b.drop |= {"zeroDm"}
    res = ev(b.table())
    tex = cp.tab_controls(res)
    g = next(line for line in tex.splitlines() if line.startswith("P-G1, P-G2"))
    assert g.endswith("\\fail\\,5/6 \\\\") or "\\fail" in g
    z = next(line for line in tex.splitlines() if line.startswith("P-Z1 &"))
    assert z.rstrip(" \\").endswith("n.j.")


def test_full_table_lists_every_contrast_with_its_interval(staged):
    tex = cp.tab_controls_full(staged)
    assert tex.count("\\label{tab:controls_full_") == 5
    assert "tab:controls_reported" in tex and "tab:controls_checks" in tex
    assert "tab:controls_truncation" in tex
    assert "\\tiny" not in tex and "world-clustered bootstrap (2,000 draws, seed 996)" in tex
    assert "{\\scriptsize [" in tex and ",\\," in tex  # the one interval format of the tables
    for label in (
        "Update, oracle $-$ source, $D{=}1$",
        "Mechanistic repair, thinking $-$ $D{=}0$",
        "task definitions",
    ):
        assert label in tex
    assert "(add-only $-$ $B{=}0$)/($B{=}1$ $-$ $B{=}0$)" in tex
    # every contrast family of the analysis has rows in the full table
    for tag in ctl.families():
        assert cp.TAG_LABELS[tag] in tex, tag


def test_checks_table_marks_each_check():
    anchors = {"r": {"holdmem_D0B1_source": {"share": 0.05, "answers": 100}}}
    audit_cell = {"pass": True, "branches": {"src": {"definition": {"pass": True}}}}
    view = {
        "n_worlds": 3,
        "checks": {
            "hash": {"pass": True},
            "probe_sets": {"pass": True},
            "anchors": {"pass": False, "runs": anchors},
            "audit": {"pass": True, "cells": {"holdmem_D0B1_gold": audit_cell}},
            "scope_cited": {"pass": None, "reason": "missing"},
            "passed": 3,
            "failed": 1,
            "not_evaluated": 1,
        },
    }
    tex = cp.tab_controls_full(view)
    assert "worst cell 5.0\\% of answers differ, fail" in tex
    assert "D0B1\\_gold equal the audit & 1/1 groups, pass" in tex
    assert "not evaluated" in tex
    assert "\\fail\\,3/5" in cp.tab_controls(view)
    m = cp.numbers(view)
    assert m["CtlCheckAnchorPass"] == "fail" and m["CtlCheckScopeCitedPass"] == "not evaluated"
    assert m["CtlCheckAnchorWorst"].startswith("5.0\\%") and m["CtlChecksPassed"] == "3 of 5"


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile_and_fit_the_page(staged, tmp_path):
    for name, tex in cp.tables(staged).items():
        (tmp_path / name).write_text(tex)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage[letterpaper,margin=1in]{geometry}",
                r"\usepackage{amsmath,newtxtext,newtxmath,xcolor,booktabs}",
                r"\usepackage[font=small,labelfont=bf,skip=5pt]{caption}",
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}",
                r"\newcommand{\pass}{\checkmark}\newcommand{\fail}{$\times$}",
                r"\begin{document}\input{tab_controls.tex}\clearpage\input{tab_controls_full.tex}",
                r"\end{document}",
            ]
        )
    )
    done = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-1500:]
    assert not re.search(r"Overfull \\hbox \(([5-9]|\d{2,})\.", done.stdout)  # nothing sticks out


def test_compact_caption_defines_every_arm_and_term_and_rows_count_each_verdict(staged):
    tex = cp.tab_controls(staged)
    caption = tex.split("\\label{tab:controls}")[0]
    assert r"\texttt{prereg/v3\_controls.md}" in caption  # a path, not hyphenated text
    for term in (
        r"\emph{oracle}",
        r"\emph{scope}",
        r"\emph{add-only}",
        r"\emph{template}",
        r"\emph{deletion-aware}",
        r"\emph{view}",
        r"\emph{thinking}",
        r"\emph{full-access}",
        "Repair: share of the re-derived items the reader answers correctly",
        "mechanistic repair: share whose re-derived value equals gold",
        "$t_0$, $k{=}0$: before the shift",
        r"($\Sdel$)",
    ):
        assert term in caption, term
    assert "retraction" not in tex and "gold $-$" not in tex and "refresh $-$" not in tex
    # a row that carries two predictions gives the count of each, in their order
    t = cells_of(tex, "P-T1, P-T3 &")
    v = staged["verdicts"]
    assert t[-1] == ", ".join(cp._mark(v[p]) for p in ("T1", "T3"))
    assert t[-1].count("/") == 2


def test_a_negative_ratio_has_a_typeset_minus():
    ratio = next(r for r in cp.ROWS if r.unit == "ratio")
    assert cp._compact_cell(-0.01, ratio, bold=False) == "$-$0.01"
    assert cp._cell({"est": -0.01, "lo": -0.02, "hi": 0.03}, "ratio") == (
        r"$-$0.01\,{\scriptsize [$-$0.02,\,0.03]}"
    )


# --------------------------------------------------------------------- sources and generated files
def test_results_sources_name_the_control_stage_and_it_is_not_staged_yet():
    assert SOURCES["ctl"] == "runs/stage5_controls"
    if not (ROOT / SOURCES["ctl"] / "analysis" / "ctl_metrics.json").exists():
        assert cp.load(ROOT, SOURCES) is None


def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in cp.tables(cp.load(ROOT, SOURCES)).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name


def test_paper_tables_generator_writes_the_control_tables(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["paper_tables", "--root", str(ROOT), "--out", str(tmp_path)])
    paper_tables.main()
    capsys.readouterr()
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in cp.tables(cp.load(ROOT, SOURCES)).items():
        assert (tmp_path / name).read_text() == head + tex


def test_the_number_generator_hook_adds_the_control_macros():
    src = (ROOT / "src/memshift/analysis/paper_numbers.py").read_text()
    assert "controls_paper.numbers(controls_paper.load(root, sources))" in src

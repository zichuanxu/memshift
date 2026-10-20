"""Paper numbers and the table of the control cells (`analysis.controls_paper`): a stage that is
not staged renders only MISSING markers, a staged one carries the estimates and counts the
analysis produced, the table is well-formed and compiles, and the committed generated files are
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
SOURCES = json.loads((ROOT / "results/sources.json").read_text())


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


def test_every_contrast_and_check_has_a_macro_name_and_no_threshold_outcome_has_one():
    names = set(cp.numbers(None))
    for key in ctl.contrast_keys():
        assert f"Ctl{camel(key)}" in names, key
    for tag in ("Hash", "Probes", "Anchor", "Audit", "ScopeCited"):
        assert f"CtlCheck{tag}Pass" in names
    assert "CtlChecksPassed" in names and "CtlWorlds" in names
    for pred in ctl.CONTROLS:
        for suffix in ("Verdict", "Hits", "Total", "Need", "Count"):
            assert f"Ctl{camel(pred)}{suffix}" not in names
    assert not [n for n in names if re.search(r"Summary|Strong|Presentation|Reading|AsReg", n)]


def test_the_full_access_control_has_its_macros_for_src_and_def_only():
    names = set(cp.numbers(None))
    for name in (
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
        "CtlFullWhichTruncated",
    ):
        assert name in names, name
    # the orphan branch was not run: no contrast and no note macro for it
    assert not [n for n in names if n.startswith(("CtlFRep", "CtlFGain")) and "Orphan" in n]
    assert not [n for n in names if n.startswith("CtlFull") and n.endswith("Orphan")]
    assert len([n for n in names if n.startswith("CtlFull")]) > 100


def test_staged_macros_carry_estimates_intervals_and_counts(staged):
    m = with_estimates(cp.numbers(staged))
    # an estimate with its interval, and the estimate alone
    assert m["CtlZUpdSrcDefinition"] == "$+$80.0\\,pp [$+$80.0, $+$80.0]"
    assert m["CtlZUpdSrcDefinitionEst"] == "$+$80.0\\,pp"
    assert m["CtlGUpdDzeroSrcAll"].startswith("0.0\\,pp")
    assert m["CtlRRatioSrcDefinition"].startswith("0.65")  # (0.80 - 0.50) / (0.96 - 0.50)
    # the D0full control: the gap with its interval, the sensitivity analysis
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
    # per definition and declared source after S_src and S_def: writes, writes cut, records left out
    for b in ("Src", "Def"):
        assert (
            m[f"CtlFullCloseFriendWrites{b}"],
            m[f"CtlFullCloseFriendMessagesTruncated{b}"],
            m[f"CtlFullCloseFriendMessagesDropped{b}"],
        ) == ("6", "6", "72")
        assert (
            m[f"CtlFullVipSenderWrites{b}"],
            m[f"CtlFullVipSenderEmailTruncated{b}"],
            m[f"CtlFullVipSenderEmailDropped{b}"],
        ) == ("6", "6", "240")
        assert (
            m[f"CtlFullVipSenderContactsTruncated{b}"],
            m[f"CtlFullVipSenderContactsDropped{b}"],
        ) == (
            "0",
            "0",
        )
        assert (
            m[f"CtlFullActiveProjectWrites{b}"],
            m[f"CtlFullActiveProjectTasksTruncated{b}"],
            m[f"CtlFullActiveProjectTasksDropped{b}"],
        ) == ("6", "0", "0")
    assert m["CtlFullWhichTruncated"] == "close\\_friend, vip\\_sender"


NOTHING_TO_SHOW = re.compile(
    r"CtlChecks?\w*"  # the checks of a run: the synthetic table has none
    r"|CtlRa\w*Free"  # the synthetic items have no free-text class
    r"|CtlRaFinalUndeclaredOfFailed\w+"  # and no failed item
    r"|Ctl[GZ]Stab\w*Definition"  # and no definition whose gold does not change
)


def test_every_estimate_macro_has_a_number_when_everything_ran(staged):
    macros = cp.numbers(staged)
    missing = sorted(k for k, v in macros.items() if v == MISSING)
    assert all(NOTHING_TO_SHOW.fullmatch(k) for k in missing), missing[:10]
    for key in ctl.contrast_keys():
        if not NOTHING_TO_SHOW.fullmatch(f"Ctl{camel(key)}"):
            assert macros[f"Ctl{camel(key)}"] != MISSING, key


def test_a_missing_cell_leaves_its_contrasts_missing(partial):
    m = cp.numbers(partial)
    assert m["CtlZUpdSrcDefinition"] == MISSING  # the zero system's D_m cell was dropped
    assert m["CtlGUpdDzeroSrcAll"] != MISSING
    assert m["CtlFRepSrcDefinition"] != MISSING  # D1 and D0full were run in `partial`
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
        assert set(cp.tables(view)) == {"tab_controls.tex"}
        for tex in cp.tables(view).values():
            check_tex(tex)
            assert tex.count("\\begin{table}") == 1
    assert MISSING in cp.tab_controls(None)
    # a table cell that cannot be computed is MISSING, the rest stay
    assert cp.tab_controls(partial).count(MISSING) > 10
    assert cp.tab_controls(partial).count(MISSING) < cp.tab_controls(None).count(MISSING)


def test_compact_table_rows_are_the_control_arms_with_their_contrasts(staged):
    tex = cp.tab_controls(staged)
    for arm, contrast in (  # the arm's name on the first row of its block, once
        ("oracle", "Update, oracle"),
        ("", "Stability, oracle"),
        ("add-only", "Update, add-only"),
        ("template", "Update, template"),
        ("scope", "Update, scope $-$ source"),
        ("", "Update, scope $-$ cited"),
        ("view", "Update, view $-$ no-store BM25"),
        ("", "\\quad single-field items"),
        ("", "Stability, view $-$ no-store BM25"),
        ("", "Update, view $-$ $D_"),
        ("", "Accuracy at $t_0$"),
        ("full access", "Acc.\\ on re-derived, $D{=}1$ $-$ full-access"),
        ("add-only", "(add-only"),  # the two ratios in a block of their own, last
        ("template", "(template"),
    ):
        assert f"\n{arm} & {contrast}" in tex, (arm, contrast)
    ratios = tex.split("\\midrule")[-1]
    assert ratios.count("ratio") == 0 and ratios.count(" \\\\") == 2
    assert tex.index("\nfull access &") < tex.index("\nadd-only & (add-only")
    # the two arms that could not be tested as designed are not in the table
    assert "deletion-aware" not in tex and "thinking" not in tex.lower()
    assert "P-" not in tex and "prereg" not in tex and "regist" not in tex
    assert "verdict" not in tex and "\\pass" not in tex and "\\fail" not in tex
    assert "\\textbf{" not in tex.replace(MISSING, "") and "n.j." not in tex
    assert "Implementation checks" not in tex
    z1 = next(
        line for line in tex.splitlines() if line.startswith("view & Update, view $-$ no-store")
    )
    assert "$+$80.0" in z1


def split_row(line: str) -> list[str]:
    return [c.strip() for c in line.removesuffix(r"\\").split(" & ")]


def cells_of(tex: str, start: str) -> list[str]:
    """The cells of the table row that begins with `start`."""
    return split_row(next(line for line in tex.splitlines() if line.startswith(start)))


def test_compact_table_has_the_full_access_rows_for_src_and_def_only(staged):
    cells = cells_of(cp.tab_controls(staged), "full access &")
    # arm, contrast, definition (3 branches), other items (3), all items (3)
    assert len(cells) == 11 and cells[0] == "full access" and "full-access" in cells[1]
    assert cells[2:5] == ["$+$66.0", "$+$66.0", cp.UNDEFINED]
    assert cells[5:8] == [cp.NOT_TAKEN] * 3  # empty: the contrast is about definition items
    assert cells[8:11] == ["$+$32.2", "$+$32.2", cp.UNDEFINED]
    lines = cp.tab_controls(staged).splitlines()
    mech = split_row(
        lines[lines.index(next(x for x in lines if x.startswith("full access &"))) + 1]
    )
    assert mech[:2] == ["", r"\quad value equals gold"]  # the row after the full-access row
    assert mech[2:5] == ["$+$66.0", "$+$66.0", cp.UNDEFINED]
    assert "full access &" in cp.tab_controls(staged) and "$\\Ssrc$ and $\\Sdef$ only" in cp.CAPTION


def test_the_full_access_rows_are_missing_when_unstaged_or_without_the_cell():
    cells = cells_of(cp.tab_controls(None), "full access &")
    assert cells[2:4] == [MISSING, MISSING] and cells[4] == cp.UNDEFINED
    b = Builder()
    b.drop |= {"D0fullB1"}
    cells = cells_of(cp.tab_controls(ev(b.table())), "full access &")
    assert cells[2:4] == [MISSING, MISSING] and cells[4] == cp.UNDEFINED  # `---`: no orphan run
    m = cp.numbers(ev(b.table()))
    assert m["CtlFRepSrcDefinition"] == MISSING and m["CtlFullTruncatedAll"] == MISSING
    assert m["CtlFullCloseFriendMessagesTruncatedSrc"] == MISSING


def test_compact_table_shows_the_estimates_of_every_arm():
    b = Builder({("D1B1gold", "definition", "orphan"): 0.50})
    b.drop |= {"zeroDm"}
    res = ev(b.table())
    tex = cp.tab_controls(res)
    g = cells_of(tex, "oracle & Update, oracle")
    assert len(g) == 11 and "\\fail" not in tex and "n.j." not in tex
    z = cells_of(tex, "view & Update, view $-$ no-store")
    assert len(z) == 11


def test_check_macros_mark_each_check():
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
                r"\newcommand{\pass}{\checkmark}\newcommand{\fail}{$\times$}"
                r"\newcommand{\cref}[1]{Appendix~A}",
                r"\begin{document}\input{tab_controls.tex}",
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


def test_compact_caption_defines_every_arm_and_term(staged):
    tex = cp.tab_controls(staged)
    caption = tex.split("\\label{tab:controls}")[0]
    assert "prereg" not in caption and "P-" not in caption and "regist" not in caption
    for term in (
        "The arms are those of \\cref{app:controls}",  # defined once, in the appendix
        "Source, cited: the source key and the citation key",
        "Full access: the full-access writer",
        "Acc.\\ on re-derived: accuracy on re-derived items",
        "Value equals gold: the share of re-derived values that equal gold",
        "The last two rows are ratios",
        "$t_0$: before the shift",
        r"$\Ssrc$, $\Sdef$ and $\Sdel$",
    ):
        assert term in caption, term
    assert "Deletion-aware" not in caption and "Thinking" not in caption
    assert "retraction" not in tex and "gold $-$" not in tex and "refresh $-$" not in tex
    t = cells_of(tex, "template & Update, template")
    assert len(t) == 11  # arm, contrast and the nine cells


def test_a_negative_ratio_has_a_typeset_minus():
    ratio = next(r for r in cp.ROWS if r.unit == "ratio")
    assert cp._compact_cell(-0.01, ratio) == "$-$0.01"


# --------------------------------------------------------------------- sources and generated files
def test_results_sources_name_the_control_stage_and_it_is_not_staged_yet():
    assert SOURCES["ctl"] == "runs/stage5_controls"
    if not (ROOT / SOURCES["ctl"] / "analysis" / "ctl_metrics.json").exists():
        assert cp.load(ROOT, SOURCES) is None


@pytest.mark.paper
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

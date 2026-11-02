"""Paper numbers and tables of `analysis.ex_round2`: computed from committed run outputs and the
seeded generator, reproducing the staged values they build on, and the committed generated files."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import ex_round2 as r2
from memshift.analysis import paper_numbers as pn
from memshift.analysis.ex_e1_paper import UNDEFINED
from memshift.analysis.paper_data import MISSING
from tests.memtab_check import MEMTAB_PREAMBLE, check_memtab

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = r2.load(ROOT, SOURCES)
EMPTY = r2.load(ROOT, {})
NUMBERS = ROOT / "paper/generated/numbers.tex"  # read by the tests marked `paper` only
# the attribution and the thinking writer read raw outputs (item tables, probes and items of the
# source runs), which the public release does not ship
needs_raw = pytest.mark.skipif(
    INP["attr"] is None or INP["think"] is None,
    reason="raw run outputs (item tables, probes, items) are not in this checkout",
)


def committed(name: str) -> str:
    """The value of a macro in the committed numbers.tex."""
    m = re.search(rf"\\newcommand{{\\{name}}}{{(.*)}}\n", NUMBERS.read_text())
    assert m, name
    return m.group(1)


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = r2.numbers(EMPTY)
    assert len(macros) > 250
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only
        assert name.startswith(("Ex", "ETwoExWriter")), name
        assert value == MISSING, (name, value)


@needs_raw
def test_every_macro_has_a_number_when_everything_is_staged():
    macros = r2.numbers(INP)
    assert not [k for k, v in macros.items() if v == MISSING]
    # a redefinition changes no item outside the definition class: a contrast over no item
    assert macros["ExAttrDefShiftOtherItems"] == "0"
    assert macros["ExAttrDefShiftOtherBxD"] == UNDEFINED


@pytest.mark.paper
def test_attribution_reproduces_the_published_counts_and_splits_them():
    m = r2.numbers(INP)
    # the changed item-times of the class mix, by branch
    for b in ("Src", "Def", "Orphan"):
        assert int(m[f"ExAttr{b}Changed"].replace(",", "")) == int(
            committed(f"ExMixMain{b}AllItems")
        )
        assert m[f"ExAttr{b}ItemTimes"] == "3,840"
        parts = sum(int(m[f"ExAttr{b}{c}AllItems"].replace(",", "")) for c in ("Shift", "Drift"))
        assert parts == int(m[f"ExAttr{b}Changed"].replace(",", ""))
    # S_src changes 30.4% of the item-times; the revision itself 20.5%; it targets 10 of 48 items
    assert m["ExAttrSrcChangedShareOfAll"] == "30.4\\%"
    assert m["ExAttrSrcShiftShareOfAll"] == "20.5\\%"
    assert m["ExAttrSrcTargetSlots"] == "10" and m["ExAttrWorldSlots"] == "48"
    # after S_def: by definition version 235 redefined and 152 kept item-times (as the review),
    # by value 222 changed by the redefinition and 165 by drift only
    assert m["ExAttrDefKappaRedefinedItems"] == "235" and m["ExAttrDefKappaKeptItems"] == "152"
    assert m["ExAttrDefKappaRedefinedBxD"].startswith("$+$38.3\\,pp [")
    assert m["ExAttrDefKappaKeptBxD"].startswith("$+$39.5\\,pp [")
    assert m["ExAttrDefShiftDefinitionItems"] == "222"
    assert m["ExAttrDefDriftDefinitionItems"] == "165"
    assert m["ExAttrDefDriftOtherItems"] == committed("ExMixMainDefOtherItems")


@pytest.mark.paper
def test_attribution_on_drift_of_other_items_is_the_published_class_contrast():
    # after S_def every other item changed by drift: the part is the whole class
    m = r2.numbers(INP)
    assert m["ExAttrDefDriftOtherBxD"] == committed("ExClsMainDefOtherBxD")
    assert m["ExAttrDefDriftOtherBDzero"] == committed("ExClsMainDefOtherBDzero")


@pytest.mark.paper
def test_thinking_writer_reproduces_the_control_stage_shares():
    m = r2.numbers(INP)
    assert m["ExThinkAllAtLimit"] == committed("CtlThinkAtLimitAll")
    assert m["ExThinkAllUnreadable"] == committed("CtlThinkUnreadableAll")
    assert m["ExThinkAllWrites"] == "708"
    assert m["ExThinkImportantContactAtLimit"].startswith("98.3\\%")
    assert m["ExThinkCloseFriendAtLimit"].startswith("60.8\\%")
    assert m["ExThinkOverdueTaskAtLimit"].startswith("0.0\\%")
    # every asked re-derivation of important contacts after S_src returned no value
    assert m["ExThinkImportantContactSrcNoValue"] == m["ExThinkImportantContactSrcRederived"]
    assert m["ExThinkImportantContactSrcCorrect"] == "0"
    # the thinking gain on repair is the control stage's D_gain_rep (think minus plain, same items)
    assert m["ExThinkAllSrcRederived"] == m["ExThinkAllSrcPlainRederived"]
    gain = int(m["ExThinkAllSrcCorrect"]) - int(m["ExThinkAllSrcPlainCorrect"])
    n = int(m["ExThinkAllSrcRederived"])
    assert committed("CtlDGainRepSrcDefinitionEst") == f"$+${100 * gain / n:.1f}\\,pp"


@pytest.mark.paper
def test_writer_predictor_intervals_under_type_resampling():
    m = r2.numbers(INP)
    assert m["ETwoExWriterAgreePooledAurocTypeNinety"] == "0.832 [0.389, 0.934]"
    assert m["ETwoExWriterNoDefAgreePooledAurocNinety"] == "0.275 [0.175, 0.347]"
    assert m["ETwoExWriterNonDefPooledAurocNinety"].startswith("0.869 [")
    assert not any("Verdict" in k for k in m)
    assert committed("ETwoExWriterAgreePooledSlopeType") == "1.01 [$-$0.53, 1.79]"


@pytest.mark.paper
def test_leave_one_definition_out_range():
    m = r2.numbers(INP)
    assert (m["ExPerDefLooMainSrcBxDMin"], m["ExPerDefLooMainSrcBxDMax"]) == (
        "$+$16.5\\,pp",
        "$+$33.8\\,pp",
    )
    assert m["ExPerDefLooMainDefBxDMin"] == "$+$34.6\\,pp"
    assert committed("ExPerDefWithoutMainSrcImportantContactBxDEst") == "$+$16.5\\,pp"


# ------------------------------------------------------------------------------------ tables
# panel (a) of the per-definition table, whose float, caption and labels are in the appendix
# source (the class split with D_m is part of ex_e1_paper.tab_classsplit)
PANELS = ("tab_perdef.tex",)


def test_tables_are_wellformed_with_data_and_without():
    for inp in (INP, EMPTY):
        assert set(r2.tables(inp)) == set(PANELS)
        for name, tex in r2.tables(inp).items():
            check_memtab(tex, panel=True)
            assert tex.startswith("\\begin{mttabular}") and "\\label" not in tex, name
            assert "\\caption" not in tex and "(a) Definition" in tex, name
            # the paper's table style: intervals as muted secondary text, notes under the panel
            assert "\\scriptsize" not in tex and "\\mtnotes{" in tex, name
    for name, tex in r2.tables(EMPTY).items():
        assert MISSING in tex, name


@needs_raw
def test_tables_have_no_missing_marker_when_everything_is_staged():
    for name, tex in r2.tables(INP).items():
        assert MISSING not in tex, name


@pytest.mark.paper
def test_the_levels_of_the_decomposition_are_the_account_macros():
    levels = {  # r_0, r_1, a_0, a_1 of the source-epoch key in the main study
        ("Src", "Definition"): ("16.2", "99.1", "10.2", "65.8"),
        ("Src", "Other"): ("89.1", "90.7", "83.7", "85.9"),
        ("Def", "Definition"): ("6.7", "99.7", "7.5", "61.8"),
        ("Orphan", "Other"): ("68.1", "67.5", "10.0", "10.9"),
    }
    for (b, c), want in levels.items():
        got = [committed(f"ExAcctMain{b}{c}{t}Est") for t in ("RZero", "ROne", "AZero", "AOne")]
        assert [g.removesuffix("\\%") for g in got] == list(want), (b, c)
        assert committed(f"ExAcctMain{b}{c}BxD") == committed(f"ExClsMain{b}{c}BxD")
    assert committed("ExAcctMainSrcDefinitionBxD") == "$+$27.3\\,pp [$+$21.6, $+$32.6]"


@pytest.mark.paper
def test_perdef_table_matches_the_per_definition_macros():
    tex = r2.tab_perdef(INP)
    for d in ("ImportantContact", "CloseFriend", "WorkingHours", "ActiveProject", "VipSender"):
        for b in ("Src", "Def", "Orphan"):
            est = committed(f"ExPerDefMain{b}{d}BxDEst").replace("\\,pp", "")
            assert f"{est} & " in tex, (d, b)  # its interval in the next cell, if it has width
    assert "$+$16.5 to $+$33.8" in tex
    assert "0.0 & \\mtint{[0.0,\\,0.0]}" not in tex  # every world the same: no interval
    # the class value of all six definitions is the class split's, not repeated here
    assert "all six" not in tex and "\\cref{tab:classsplit}" not in tex


@pytest.mark.paper  # the table style, paper/memtab-style.tex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile_at_the_paper_width(tmp_path):
    names = []
    for name, tex in r2.tables(INP).items():
        (tmp_path / name).write_text(tex)
        names.append(name)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[10pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage[textwidth=6.75in,textheight=9in]{geometry}",
                r"\makeatletter\renewcommand{\footnotesize}"  # 9 pt, as preprint.sty
                r"{\@setfontsize\footnotesize\@ixpt\@xpt}\makeatother",
                r"\usepackage{amsmath,newtxtext,newtxmath,xcolor,booktabs}",
                MEMTAB_PREAMBLE,
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}\newcommand{\cref}[1]{Table~1}",
                r"\begin{document}"
                + "".join(
                    r"\begin{table*}[t]\centering\mtsetup\setlength{\tabcolsep}{4pt}"
                    rf"\input{{{n}}}\end{{table*}}"
                    for n in names
                )
                + r"\end{document}",
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
    assert "Overfull \\hbox" not in done.stdout  # every table fits the 6.75in text block


# ------------------------------------------------------------------------- generated files
@pytest.mark.paper
def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in r2.tables(INP).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name
    numbers = NUMBERS.read_text()
    for name, value in r2.numbers(INP).items():
        assert pn._macro(name, value) in numbers, name

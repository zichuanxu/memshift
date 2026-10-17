"""Paper numbers, tables and figure of the second-round exploratory analyses
(`analysis.ex_review_paper`), the leftover-wording fixes of the generators, and the committed
generated files."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import ex_review_paper as rp
from memshift.analysis import paper_data as pd
from memshift.analysis import paper_numbers as pn
from memshift.analysis import paper_tables as pt
from memshift.analysis.ex_e1_paper import UNDEFINED
from memshift.analysis.paper_data import MISSING
from tests.test_paper_stage4 import check_tex

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = rp.load(ROOT, SOURCES)
EMPTY = rp.load(ROOT, {})


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = rp.numbers(EMPTY)
    assert len(macros) > 2000
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only
        assert name.startswith("Ex"), name
        assert value == MISSING, (name, value)


def test_every_macro_has_a_number_when_everything_is_staged():
    macros = rp.numbers(INP)
    assert not [k for k, v in macros.items() if v == MISSING][:10]
    assert all(v for v in macros.values())
    # a quantity over no item is a dash, not a missing number
    assert macros["ExMissClassRepDefFreeACount"] == "0"
    assert macros["ExMissClassRepDefFreeA"] == UNDEFINED


def test_macros_carry_the_staged_account_test():
    m = rp.numbers(INP)
    assert m["ExAcctFitTwoCells"] == "24" and m["ExAcctFitThreeCells"] == "36"
    assert m["ExAcctFitTwoSpearman"].startswith("0.75 [")
    assert m["ExAcctFitTwoSlope"].startswith("0.30 [")
    assert m["ExAcctLowoTwoGapLinearMae"].startswith("10.5\\,pp [")
    assert m["ExAcctLowoTwoClassMeanMae"].startswith("11.5\\,pp [")
    assert m["ExAcctLowoTwoGapLinearHits"] == "7 of 24"
    # within the definition cells the repair gap carries no rank information
    assert m["ExAcctFitTwoDefinitionSpearman"].startswith("$-$0.21")
    # the account terms by writer, class and branch (a reviewer's numbers)
    assert m["ExAcctLlamaOrphanOtherBxD"].startswith("$+$27.6")
    assert m["ExAcctMainSrcDefinitionROne"].startswith("99.1")
    assert m["ExAcctMainSrcDefinitionRZero"].startswith("16.2")
    assert m["ExAcctLlamaSrcOtherRZero"].startswith("75.7")
    assert m["ExAcctMainSrcOtherItems"] == "835"
    # R is shown but is not a fitted writer
    assert "ExAcctLowoTwoRepGapLinearMae" not in m and "ExAcctRepSrcOtherBxD" in m


def test_macros_carry_the_staged_baselines_and_horizon_transfer():
    m = rp.numbers(INP)
    assert m["ExBaseDevMainPooledPooledAllHits"] == "17 of 18"
    assert m["ExBaseDevMainPooledPooledAllMae"].startswith("2.1\\,pp [")
    assert m["ExBaseDevMainCellPooledAllHits"] == "16 of 18"
    assert m["ExBaseDevMainLiteralPooledAllHits"] == "10 of 18"
    assert m["ExBaseDevMainPooledPooledDefinitionMae"].startswith("23.")  # class-blind
    assert m["ExHorizonCellPooledAllHits"].endswith("of 12")
    assert m["ExHorizonPooledPooledAllHits"].endswith("of 12")
    assert m["ExHorizonCellDefAllDzeroBoneSourceErr"].startswith("$+$14.1")


def test_macros_carry_the_staged_missed_items_zero_system_and_miss_classes():
    m = rp.numbers(INP)
    assert m["ExMissedMainSrcDoneDefinitionACited"].startswith("3.8\\%")
    assert m["ExMissedMainSrcDoneOtherUpdateBzero"].startswith("85.9\\%")
    assert m["ExMissedMainSrcDoneOtherBEffect"].startswith("$-$3.4")
    assert m["ExMissedMainOrphanDoneDefinitionBEffect"].startswith("$+$15.9")
    assert m["ExZeroMainOrphanAllUpdateZero"].startswith("71.3")
    assert m["ExZeroMainOrphanAllUpdateHoldOneOne"].startswith("73.1")
    assert m["ExZeroMainOrphanOtherUpdateZero"].startswith("85.9")
    assert m["ExZeroMainOrphanAllStabZero"].startswith("64.8")
    assert m["ExZeroMainOrphanAllUpdateDiffHoldOneOne"].startswith("$-$1.8")
    assert m["ExMissClassMainSrcAllItems"] == "174"
    assert m["ExMissClassMainVerified"] == "60 of 60" and m["ExMissClassRepVerified"] == "59 of 59"
    assert (
        m["ExMissClassMainSrcAllACount"] == "8"
        and m["ExMissClassMainSrcAllCitedTouchedCount"] == "0"
    )


def test_formatters():
    assert rp.snum(-0.214) == "$-$0.21" and rp.snum(0.3) == "0.30" and rp.snum(None) == MISSING
    assert rp.snum(float("nan")) == MISSING
    assert rp.mae_frac(0.0212) == "2.1\\,pp" and rp.mae_pp(10.66) == "10.7\\,pp"
    assert (
        rp.dpp(-1.04) == "$-$1.0\\,pp"
        and rp.dpp(0.0) == "0.0\\,pp"
        and rp.dpp(2.0) == "$+$2.0\\,pp"
    )
    assert rp.of(7, 24) == "7 of 24" and rp.of(None, 24) == MISSING and rp.count(None) == MISSING


# -------------------------------------------------------------------------- tables, figure
LABELS = {
    "tab_writers.tex": "tab:writers",
    "tab_baselines.tex": "tab:baselines",
    "tab_misses.tex": "tab:misses",
    "tab_zero_after.tex": "tab:zero-after",
    "tab_json_arm.tex": "tab:json-arm",
}


def test_tables_are_wellformed_with_data_and_without():
    for inp in (INP, EMPTY):
        assert set(rp.tables(inp)) == set(LABELS)
        for name, tex in rp.tables(inp).items():
            check_tex(tex)
            assert tex.startswith("\\begin{table}") and f"\\label{{{LABELS[name]}}}" in tex
    for name, tex in rp.tables(EMPTY).items():
        assert MISSING in tex, name
    for name, tex in rp.tables(INP).items():
        assert MISSING not in tex, name


def test_writers_table_has_every_writer_and_class_and_the_test():
    tex = rp.tab_writers(INP)
    for label in ("Qwen3.5-9B", "Qwen3.5-9B, repl.", "gemma-4-12B", "Llama-3.1-8B", "gpt-oss-20b"):
        assert label in tex
    for cls in ("definition", "field", "free text", "other"):
        assert len(re.findall(rf"& {re.escape(cls)} & \d", tex)) == 5
    assert "leave-one-writer-out error" in tex and "class label" in tex
    assert "rule" not in re.sub(r"\\(top|mid|bottom|cmid)rule", "", tex)


def test_misses_and_zero_tables_show_both_studies():
    tex = rp.tab_misses(INP)
    assert tex.count("\\quad\\textcolor{gray}{repl.}") == 9  # one replication line per cell
    assert "{R}" not in tex and "R's" not in rp.tab_writers(INP)  # no block letter
    zero = rp.tab_zero_after(INP)
    assert "\\emph{main study}} \\\\" in zero and "\\emph{replication}} \\\\" in zero


def test_misses_zero_and_json_tables_are_one_grid_each():
    # one tabular per table, so the blocks share their column widths (main study and R; (a), (b))
    for tex in (rp.tab_misses(INP), rp.tab_zero_after(INP), rp.tab_json_arm(INP)):
        assert tex.count("\\begin{tabular}") == 1
    misses = rp.tab_misses(INP)
    assert "\n(b) & & missed &" in misses and "\\midrule[\\heavyrulewidth]" in misses


def test_misses_caption_defines_its_columns_and_says_where_r_is():
    caption = rp.tab_misses(INP).split("\\label{tab:misses}")[0]
    for term in (
        "Missed: the number of missed item-times",  # a count, the rest are shares
        "shares of them in percent",
        r"$a{\wedge}b$: item-times of class $a$ into whose dependency set a record also entered",
        "(a) Why, main study and, in grey, the replication",
        "main study. Missed: the number of missed item-times at $D{=}1$",  # (b): no R
    ):
        assert term in caption, term


def test_interval_format_is_signed_scriptsize_with_a_thin_space():
    zero = rp.tab_zero_after(INP)
    assert "\\tiny" not in zero and "{\\scriptsize [" in zero and ",\\," in zero
    assert rp._mae_cell({"est": 10.66, "lo": 9.4, "hi": 12.7}) == (
        r"10.7\,{\scriptsize [9.4,\,12.7]}"
    )
    assert rp._snum_cell({"est": -0.214, "lo": -0.54, "hi": 0.3}) == (
        r"$-$0.21\,{\scriptsize [$-$0.54,\,0.30]}"
    )


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path):
    names = []
    for name, tex in rp.tables(INP).items():
        (tmp_path / name).write_text(tex)
        names.append(name)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage[letterpaper,margin=1in]{geometry}",
                r"\usepackage{amsmath,newtxtext,newtxmath,xcolor,booktabs}",
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}",
                r"\begin{document}" + "".join(rf"\input{{{n}}}" for n in names) + r"\end{document}",
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


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_figure_draws_with_and_without_data(tmp_path):
    full = rp.fig_account(INP, tmp_path)
    assert full.read_bytes().startswith(b"%PDF")
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare = rp.fig_account(EMPTY, bare_dir)
    assert bare.read_bytes().startswith(b"%PDF") and bare.read_bytes() != full.read_bytes()


# -------------------------------------------------------------- leftover wording of the generators
def test_claim_level_is_a_registered_claim_not_a_theory():
    assert pd.claim_label("two-factor theory + quantitative prediction") == (
        "registered claim: two-factor account with quantitative prediction"
    )
    assert pd.claim_label("two-factor theory, qualitative predictions") == (
        "registered claim: two-factor account, qualitative predictions"
    )
    assert pd.claim_label("measurement methodology") == "registered claim: measurement methodology"
    assert pd.claim_label(None) == MISSING and pd.claim_label("") == MISSING
    gate = json.loads((ROOT / SOURCES["gate"] / "analysis/gate_a.json").read_text())
    tex = pt.tab_gate(gate)
    assert "registered claim: two-factor account with quantitative prediction" in tex
    assert "theory" not in tex and "what the result supports" not in tex
    assert pn.gate_numbers(gate)["GateClaimLevel"].startswith("registered claim: ")


GENERATED = sorted((ROOT / "paper/generated").glob("*.tex"))
LEFTOVERS = re.compile(r"stage.?[24]|theory|\bD1s\b|certif", re.IGNORECASE)
PLAIN_RULE = re.compile(r"(?<![A-Za-z\\])rule\b")  # the word, not \toprule or \midrule


@pytest.mark.parametrize("path", GENERATED, ids=lambda p: p.name)
@pytest.mark.paper
def test_generated_files_carry_no_leftover_wording(path):
    text = path.read_text()
    assert not LEFTOVERS.search(text), (path.name, LEFTOVERS.search(text))
    if path.name.startswith("tab_"):  # the structured-field class is "field" in every table
        assert not PLAIN_RULE.search(text), (path.name, PLAIN_RULE.search(text))


# ------------------------------------------------------------------------- generated files
@pytest.mark.paper
def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in rp.tables(INP).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name
    assert (ROOT / "paper/figures/fig_account.pdf").read_bytes().startswith(b"%PDF")
    numbers = (ROOT / "paper/generated/numbers.tex").read_text()
    for name, value in rp.numbers(INP).items():
        assert pn._macro(name, value) in numbers, name


def test_results_sources_name_the_second_round_stages():
    assert SOURCES["ex_account"] == "runs/stage5_ex_account"
    assert SOURCES["ex_more"] == "runs/stage5_ex_more"
    assert SOURCES["ex_misses"] == "runs/stage5_ex_misses"

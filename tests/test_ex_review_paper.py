"""Paper numbers, tables and figure of the account across writers and the further analyses of
the main-study outputs (`analysis.ex_review_paper`), the wording of the generated files, and the
committed generated files."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import ex_review_paper as rp
from memshift.analysis import paper_numbers as pn
from memshift.analysis.ex_e1_paper import UNDEFINED, mt_ci_cells
from memshift.analysis.paper_data import MISSING
from tests.test_ex_e1_paper import check_mt_tex, memtab_preamble

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
    "tab_misses.tex": "tab:misses",
    "tab_zero_after.tex": "tab:zero-after",
    "tab_json_arm.tex": "tab:json-arm",
}


def test_tables_are_wellformed_with_data_and_without():
    for inp in (INP, EMPTY):
        assert set(rp.tables(inp)) == set(LABELS)
        for name, tex in rp.tables(inp).items():
            check_mt_tex(tex)
            assert tex.startswith("\\begin{table}") and f"\\label{{{LABELS[name]}}}" in tex
    for name, tex in rp.tables(EMPTY).items():
        assert MISSING in tex, name
    for name, tex in rp.tables(INP).items():
        assert MISSING not in tex, name


def test_writers_table_has_every_writer_and_class_and_the_fit():
    tex = rp.tab_writers(INP)
    for label in ("Qwen3.5-9B", "Qwen3.5-9B, repl.", "gemma-4-12B", "Llama-3.1-8B", "gpt-oss-20b"):
        assert label in tex
    for cls in ("definition", "other", "\\mtin field", "\\mtin free text"):  # other, its kinds
        assert len(re.findall(rf"& {re.escape(cls)} & \d", tex)) == 5
    assert "(b) Across the cells & $\\rho$, all & $\\rho$, definition" in tex
    assert "Spearman's $\\rho$" in tex and "least-squares line" in tex
    # the leave-one-writer-out error comparison is not part of the table
    assert "leave-one-writer-out" not in tex and "class label" not in tex
    caption = tex.split(r"\label{tab:writers}")[0]
    assert r"(a) repair correctness $r_D$ at $D$ (\%)" in caption
    assert "(writer, class, shift) cells" in caption and "branch" not in caption
    notes = tex.split("\\mtnotes{", 1)[1]
    assert "Field: single-field items. Free text: free-text items. Other: both." in notes
    # two panels of one width: (b) is set at the width of (a)
    assert tex.count("\\begin{mttabular}{") == 1 and "\\begin{mttabular*}{\\mtwidth}" in tex
    assert "exploratory" not in tex
    # a statistic and its interval in one cell, the boxes as wide as the widest of the panel
    assert "\\mtpair{$-$0.21}{[$-$0.53,\\,0.31]}" in tex
    assert "\\mtpairwidths{$-$0.21}{[$-$0.53,\\,0.31]}" in tex
    assert "rule" not in re.sub(r"\\(top|mid|bottom|cmid)rule", "", tex)


def test_misses_and_zero_tables_show_their_rows():
    tex = rp.tab_misses(INP)
    # every shift and class, each at both D, and n once per class
    assert len(re.findall(r"\n(?:\$\\S\w+\$)? & (?:definition|other|all) & \d+ & 0 & ", tex)) == 9
    assert len(re.findall(r"\n &  &  & 1 & ", tex)) == 9
    assert "{R}" not in tex and "R's" not in rp.tab_writers(INP)  # no block letter
    zero = rp.tab_zero_after(INP)
    assert "\\mtgroup{10}{Main study}\n" in zero and "\\mtgroup{10}{Replication}\n" in zero
    # the main study's totals are in the main text's overview table, the replication's here
    main, rep = zero.split("\\mtgroup{10}{Replication}")
    assert "& all &" not in main and rep.count("& all &") == 3
    assert zero.count("\\mtin$\\S") == 6  # each shift's rows under its banner
    assert "Table~\\ref{tab:overview}" in zero


def test_misses_table_holds_the_accuracies_of_the_missed_items():
    """The cited and B=0 columns of `tab_misses` (forced.json) are the accuracies of the missed
    items of `ex_e1_more` at the printed precision, so the appendix prints them once."""
    more = INP["more"]["explore_missed_items"]["main"]
    forced = INP["forced"]["results"]
    for b in ("src", "def", "orphan"):
        for d in ("D0", "D1"):
            for cls in ("definition", "other", "all"):
                a, f = more[b][d][cls], forced[d][b][cls]
                assert f["n_items"] == a["missed_item_times"]
                for mine, theirs in (("cited", "a_missed_B1_cited"), ("B0", "a_missed_B0")):
                    assert rp._num(f["update"][mine]["est"]) == rp._num(a[theirs]["est"])


def test_misses_zero_and_json_tables_are_one_grid_each():
    # one tabular per table, so the blocks share their column widths (main study and R)
    for tex in (rp.tab_misses(INP), rp.tab_zero_after(INP), rp.tab_json_arm(INP)):
        assert tex.count("\\begin{mttabular}") == 1
    assert "\\label{tab:forced}\\label{tab:misses}" in rp.tab_misses(INP)


def test_misses_notes_define_its_columns():
    tex = rp.tab_misses(INP)
    caption = tex.split("\\label{tab:forced}")[0]
    assert "that the citation key did not invalidate" in caption
    notes = tex.split("\\mtnotes{", 1)[1]
    for term in (
        "Cited: the citation-key cell, which served the stale item",
        "Forced: the source-key cell, which re-derived the same item-times",
        "$B{=}0$: the append-only cell",
        "Gain: forced minus cited",
        "$n$: missed item-times",
    ):
        assert term in notes and term not in caption, term


def test_interval_format_is_signed_small_and_muted_with_a_thin_space():
    zero = rp.tab_zero_after(INP)
    assert "\\scriptsize" not in zero and "\\mtint{[" in zero and ",\\," in zero
    # a statistic and its interval as two aligned cells, the minus typeset
    assert mt_ci_cells({"est": -0.214, "lo": -0.54, "hi": 0.3}, f=rp.snum) == (
        r"$-$0.21 & \mtint{[$-$0.54,\,0.30]}"
    )


def test_percentage_columns_are_as_wide_as_a_whole_share():
    """The whole percentages sit in boxes as wide as 100 and the counts n in a box as wide as the
    largest count, so every value fits its box."""
    misses = rp.tab_misses(INP)
    n = [int(m) for m in re.findall(r"\n(?:\$\\S\w+\$)? & \w+ & (\d+) & 0 & ", misses)]
    assert f"N{{{'0' * len(str(max(n)))}}} c" in misses
    for tex in (rp.tab_writers(INP), rp.tab_zero_after(INP), rp.tab_json_arm(INP)):
        assert "N{100}" in tex and not re.search(r"N\{(?!100\})", tex)


@pytest.mark.paper  # the table style, paper/memtab-style.tex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path):
    names = []
    for name, tex in rp.tables(INP).items():
        (tmp_path / name).write_text(tex)
        names.append(name)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[10pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage[textwidth=6.75in,textheight=9in]{geometry}",  # the appendix block
                r"\makeatletter\renewcommand{\footnotesize}"  # 9 pt, as preprint.sty
                r"{\@setfontsize\footnotesize\@ixpt\@xpt}\makeatother",
                r"\usepackage{amsmath,newtxtext,newtxmath,xcolor,booktabs}",
                memtab_preamble(tmp_path),
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


def test_figure_draws_with_and_without_data(tmp_path):
    full = rp.fig_account(INP, tmp_path)
    assert full.read_bytes().startswith(b"%PDF")
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare = rp.fig_account(EMPTY, bare_dir)
    assert bare.read_bytes().startswith(b"%PDF") and bare.read_bytes() != full.read_bytes()


def test_figure_colours_each_writer_by_its_model_and_marks_the_class_by_shape():
    from memshift.analysis import figstyle as fs

    for w, _, _ in rp.WRITER_TAGS:  # every writer of the account has its model colour
        assert rp.WRITER_MODELS[w] in fs.MODELS, w
    assert rp.WRITER_MODELS["main"] == rp.WRITER_MODELS["rep"]  # R: hollow, not another hue
    assert set(rp.CLASS_MARKERS) == {"definition", "other"}
    assert len(set(rp.CLASS_MARKERS.values())) == 2
    fit = (INP.get("acct") or {}).get("explore_account_fit", {}).get("two_class", {})
    for c, _ in rp._fit_cells(fit):
        assert c["writer"] in rp.WRITER_MODELS and c["class"] in rp.CLASS_MARKERS


def test_spread_sets_apart_only_the_markers_that_cover_one_another():
    from memshift.analysis import figstyle as fs

    with fs.printed() as plt:
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        dx = rp._spread(ax, [(50, 50), (50.1, 50), (10, 10)], 5.0)
    assert dx[2] == 0.0  # alone
    assert sorted(dx[:2]) == [-2.5, 2.5]  # side by side around their centre, in the order of x
    assert dx[0] < dx[1]
    plt.close(fig)


# -------------------------------------------------------------- leftover wording of the generators
def test_the_design_quantities_are_estimates_only():
    gate = json.loads((ROOT / SOURCES["gate"] / "analysis/gate_a.json").read_text())
    out = pn.gate_numbers(gate)
    assert set(out) == {"GatePathOneEstimate", "GatePathTwoMaxAbs", "GateWorlds"}
    assert out["GatePathOneEstimate"] == pn.ci(gate["path1"]["estimate"], pn.pp)
    assert out["GatePathTwoMaxAbs"] == pn.pp(0.0) and out["GateWorlds"] == "20"


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


def _inputs() -> set[str]:
    """The generated files the paper inputs."""
    sources = [ROOT / "paper/main.tex", *sorted((ROOT / "paper/sections").glob("*.tex"))]
    return {
        m
        for p in sources
        if p.exists()
        for m in re.findall(r"\\input\{generated/([\w-]+)\}", p.read_text())
    }


def _shown() -> list[Path]:
    """The generated tables the paper inputs, except the main file of macros."""
    return [ROOT / "paper/generated" / f"{n}.tex" for n in sorted(_inputs() - {"numbers"})]


@pytest.mark.paper
def test_every_generated_file_is_one_the_paper_inputs():
    assert {p.stem for p in GENERATED} == _inputs()


VERDICT_WORDING = re.compile(
    r"regist|exploratory|post hoc|as pred\.|n\.j\.|not judged|triggered|falsif|verdict"
    r"|gain check|leak check|agreement check|behavioural|mechanistic",
    re.IGNORECASE,
)


@pytest.mark.parametrize("path", _shown(), ids=lambda p: p.name)
@pytest.mark.paper
def test_tables_the_paper_shows_state_estimates_not_verdicts(path):
    text = "\n".join(ln for ln in path.read_text().splitlines() if not ln.startswith("%"))
    assert not VERDICT_WORDING.search(text), (path.name, VERDICT_WORDING.search(text))


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

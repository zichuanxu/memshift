"""Paper numbers, tables and figure of the further E1-family analyses (`analysis.ex_e1_paper`),
and the committed stage analyses they read: the class `all` of the split reproduces the main
contrasts to the last digit, the Eq. (1) identity holds in every cell, the numbers a reviewer
recomputed from the item tables are the ones staged, and the committed generated files are
current."""

import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import ex_e1_paper as xp
from memshift.analysis import ex_review_paper as rp
from memshift.analysis import ex_round2 as r2
from memshift.analysis import ex_round3 as r3
from memshift.analysis import ex_round3_review as rv
from memshift.analysis import ex_round4_detection as det
from memshift.analysis import paper_numbers as pn
from memshift.analysis.paper_data import MISSING
from tests.test_paper_stage4 import check_tex

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = xp.load(ROOT, SOURCES)
EMPTY = xp.load(ROOT, {})
MAIN_CONTRASTS = {  # class-split key -> the e1.contrasts key (source-epoch cells)
    "B_effect_update_D0": "B_effect_update_D0_source",
    "B_effect_update_D1": "B_effect_update_D1_source",
    "B_effect_update_pooled": "B_effect_update_pooled_source",
    "BxD_interaction_update": "BxD_interaction_update_source",
    "D_effect_repair": "D_effect_repair_source",
    "D_effect_repair_changed": "D_effect_repair_changed_source",
}


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


def check_mt_tex(tex: str) -> None:
    """`check_tex` for a table in the shared table style (paper/memtab-style.tex): an mttabular
    is a tabular and an N{template} column one column."""
    plain = re.sub(r"\\(begin|end)\{mttabular(\*?)\}", r"\\\1{tabular\2}", tex)
    check_tex(re.sub(r"N\{[^{}]*\}", "c", plain))


def memtab_preamble(tmp_path: Path) -> str:
    """Preamble lines that load the shared table style, copied next to the test document, after
    the packages and the colour it builds on."""
    shutil.copy(ROOT / "paper/memtab-style.tex", tmp_path)
    return r"\usepackage{array}\definecolor{cgrey}{HTML}{6E6E6E}\input{memtab-style}"


# ---------------------------------------------------------------- the committed stage analyses
@pytest.mark.parametrize("key", ["main", "rep", "gem", "llama", "gpt", "dev"])
def test_class_all_reproduces_the_main_contrasts_to_the_last_digit(key):
    m = INP[key]
    for b, contrasts in m["contrasts"].items():
        allc = m["explore_class_split"]["D1"][b]["source"]["all"]
        for new, old in MAIN_CONTRASTS.items():
            for f in ("est", "lo", "hi", "n_worlds", "n_items"):
                assert _same(allc[new][f], contrasts[old][f]), (key, b, new, f)


def test_block_c_split_uses_the_model_written_arm():
    c = INP["c"]
    assert set(c["explore_class_split"]) == {"D1", "D1s"}
    allc = c["explore_class_split"]["D1s"]["src"]["source"]["all"]
    for new, old in (
        ("BxD_interaction_update", "C3_BxD_interaction_update_D1s_source"),
        ("D_effect_repair", "C1_repair_D1s_minus_D0_source"),
    ):
        for f in ("est", "lo", "hi"):
            assert _same(allc[new][f], c["contrasts"]["src"][old][f]), (new, f)


@pytest.mark.parametrize("key", ["main", "rep", "c", "gem", "llama", "gpt", "dev", "short"])
def test_eq1_identity_holds_in_every_cell_and_class(key):
    dec = INP[key]["explore_decomposition"]
    for arm in dec.values():
        for b in arm.values():
            for cls, node in b.items():
                assert node["identity_max_abs_residual"] < 1e-12, (key, cls)
                for g in node["interaction"].values():
                    assert abs(g["exact_from_terms"]["est"] - g["observed"]["est"]) < 1e-12


def test_staged_numbers_are_the_ones_a_reviewer_recomputed_from_the_item_tables():
    main, rep = INP["main"], INP["rep"]

    def est(m, b, cls, key):
        e = m["explore_class_split"]["D1"][b]["source"][cls][key]
        return tuple(round(100 * e[k], 1) for k in ("est", "lo", "hi"))

    # B x D on Update outside the definition items (a reviewer's recomputation)
    assert est(main, "src", "other", "BxD_interaction_update") == (-0.6, -3.8, 2.5)
    assert est(main, "def", "other", "BxD_interaction_update") == (-0.6, -4.0, 2.7)
    assert est(main, "orphan", "other", "BxD_interaction_update") == (-1.4, -5.4, 2.3)
    assert est(rep, "src", "other", "BxD_interaction_update") == (-1.9, -3.9, 0.2)
    assert est(main, "src", "definition", "BxD_interaction_update") == (27.3, 21.6, 32.6)
    # the D effect at B = 0 that Eq. (1) leaves out: +17.4, +37.8, +9.8 pp
    for b, want in (("src", 17.4), ("def", 37.8), ("orphan", 9.8)):
        e = main["explore_decomposition"]["D1"][b]["all"]["interaction"]["source"]["D_effect_B0"]
        assert round(100 * e["est"], 1) == want
    # definition items are 29%, 69% and 17% of the changed item-times
    shares = [
        main["explore_class_mix"][b]["classes"]["definition"]["share"]
        for b in main["explore_class_mix"]
    ]
    assert sorted(round(100 * s) for s in shares) == [17, 29, 69]
    # the source-epoch key invalidates 99.7 / 99.3 / 100 percent of the unchanged items
    fi = {
        b: main["explore_invalidation"]["D1"][b]["source"]["all"]["D1"]["false_invalidation"]["est"]
        for b in ("src", "def", "orphan")
    }
    assert [round(100 * fi[b], 1) for b in ("src", "def", "orphan")] == [99.7, 99.3, 100.0]
    # the JSON arm after the shift: Update 83.7 against 93.1 (S_src)
    j = main["explore_json_arm"]["src"]["all"]
    assert (
        round(100 * j["update_json"]["est"], 1) == 83.7
        and round(100 * j["update_sk"]["est"], 1) == 93.1
    )


def test_noise_floor_pools_the_test_and_replication_worlds():
    p = INP["cross"]["explore_noise_floor"]["pooled"]
    assert (p["items"], p["values_differ"], p["answers_differ"]) == (5760, 66, 18)
    assert p["stages"] == ["stage2_e1", "stage4_rep_e1"]
    # the three branches of a world repeat one second ingestion: 40 worlds, 1,920 items
    w = p["world_level"]
    assert (w["worlds"], w["items"], w["values_differ"], w["answers_differ"]) == (40, 1920, 22, 6)


def test_replication_accounting_and_holm_on_the_committed_stages():
    rep = INP["cross"]["explore_replication"]
    se = rep["source_epoch"]
    assert (se["n_contrasts"], se["n_identities"], se["n_non_identity"]) == (21, 6, 15)
    assert se["replication_non_identity"]["replicated"] == 15
    holm = rep["holm_non_detection"]
    for stage in ("main", "rep"):
        h = holm[stage]
        assert h["family_size"] == 30 and h["draws"] == 10_000
        assert h["smallest_possible_holm_p"] == pytest.approx(30 / 10_001)
        assert not any("detected" in c["contrast"] for c in h["contrasts"])
    cs = rep["class_split"]["summary"]
    assert cs["definition"]["rep_same_sign"] == cs["definition"]["n_contrasts"]


def test_oos_uses_development_worlds_that_match_the_test_design():
    oos = INP["cross"]["explore_oos"]["dev_to_main"]
    assert set(oos["branches"]) == {"src", "def", "orphan"}
    pooled = oos["pooled_over_branches"]["all"]
    # terms carried from the ten development worlds predict the test cells within a few pp; the
    # paper's Eq. (1) with one stale-value accuracy does not
    assert pooled["cell"]["mean_abs_error"]["est"] < 0.05
    assert pooled["structural"]["mean_abs_error"]["est"] < 0.05
    assert pooled["literal"]["mean_abs_error"]["est"] > 0.10


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = xp.numbers(EMPTY)
    assert len(macros) > 4000
    for name, value in macros.items():
        pn._macro(name, value)
        assert name.startswith("Ex"), name
        assert value == MISSING, (name, value)


def test_every_macro_has_a_number_when_everything_is_staged():
    macros = xp.numbers(INP)
    missing = sorted(k for k, v in macros.items() if v == MISSING)
    assert not missing, missing[:10]
    assert all(v for v in macros.values())


def test_macro_names_do_not_collide_with_the_earlier_ones():
    macros = pn.build_macros(ROOT)
    ex = xp.numbers(INP) | rp.numbers(rp.load(ROOT, SOURCES))  # the second-round block, too
    round2 = r2.numbers(r2.load(ROOT, SOURCES))  # and the review-round-2 block
    assert round2.keys() <= macros.keys()
    ex |= {k: v for k, v in round2.items() if k.startswith("Ex")}  # not its ETwoExWriter ones
    ex |= r3.numbers(r3.load(ROOT, SOURCES))  # and the block of `ex_round3` (ExRThree)
    ex |= rv.numbers(rv.load(ROOT, SOURCES))  # and the block of `ex_round3_review` (ExRev)
    ex |= det.numbers(det.load(ROOT, SOURCES))  # and the block of `ex_round4_detection` (ExDet)
    assert ex.keys() <= macros.keys() and all(k.startswith("Ex") for k in ex)
    assert not any(k.startswith("Ex") for k in macros.keys() - ex.keys() if not k.endswith("Est"))


def test_macros_carry_the_staged_estimates():
    m = xp.numbers(INP)
    assert m["ExClsMainSrcOtherBxD"] == "$-$0.6\\,pp [$-$3.8, $+$2.5]"
    assert m["ExClsMainSrcDefinitionBxD"].startswith("$+$27.3")
    # the other families: the same class split, from the M2 stages over all ten worlds each
    assert m["ExClsLlamaOrphanOtherBxD"].startswith("$+$27.6")
    assert m["ExClsGemOrphanOtherBxD"].startswith("$+$6.0")
    assert m["ExClsGptOrphanOtherBxD"].startswith("$+$4.5")
    assert m["ExMixLlamaSrcDefinitionItems"] == m["ExMixGemSrcDefinitionItems"] == "165"
    assert m["ExDecLlamaSrcOtherDzeroBoneRepair"].startswith("75.7")
    assert m["ExRepIdentities"] == "6" and m["ExRepNonIdentity"] == "15"
    assert m["ExRepClassDefinitionN"] == m["ExRepClassOtherN"] == "24"
    assert m["ExRepClassDefinitionExcludesZero"] == "23" and m["ExRepClassOtherSameSign"] == "17"
    assert not any("Replicated" in k or k.startswith("ExRepNonIdentity") and k != "ExRepNonIdentity"
                   for k in m)  # fmt: skip
    assert m["ExNoiseHoldPooledValues"] == "66" and m["ExNoiseHoldPooledAnswers"] == "18"
    assert m["ExNoiseHoldPooledItems"] == "5,760" and m["ExNoiseHoldPooledWorlds"] == "40"
    assert m["ExNoiseHoldPooledWorldValues"] == "22" and m["ExNoiseHoldPooledWorldAnswers"] == "6"
    assert m["ExHolmMainFamily"] == "30" and m["ExHolmMainDraws"] == "10,000"
    assert (
        m["ExCostModelMainTwelveHoldTotal"].endswith("M")
        and m["ExCostModelMainTwelveZeroRatioZero"] == "1.00"
    )
    assert m["ExDecResidualMax"].startswith("$") and "10^{" in m["ExDecResidualMax"]
    # a quantity that is undefined by construction is a dash, not a missing number
    assert m["ExDecMainSrcDefinitionDoneBoneStale"] == xp.UNDEFINED


def test_formatters():
    assert xp.tokens(78_632_437) == "78.6M" and xp.tokens(10_771.6) == "10.8k"
    assert xp.tokens(None) == MISSING and xp.sci(float("nan")) == MISSING
    assert xp.sci(1.1e-16) == r"$1.1\times10^{-16}$"
    assert xp.fmt(None) == MISSING and xp.fmt({"est": float("nan")}) == xp.UNDEFINED
    assert xp.point(None) == MISSING and xp.point(float("nan")) == xp.UNDEFINED


# ------------------------------------------------------------------------------- tables, figure
def test_tables_are_wellformed_with_data_and_without():
    for inp in (INP, EMPTY):
        for name, tex in xp.tables(inp).items():
            check_mt_tex(tex)
            assert tex.startswith("\\begin{table}") and f"\\label{{tab:{name[4:-4]}}}" in tex
    assert set(xp.tables(INP)) == {"tab_classsplit.tex"}
    assert MISSING in xp.tab_classsplit(EMPTY) and MISSING not in xp.tab_classsplit(INP)


def test_class_table_shows_every_arm_for_every_shift_and_class():
    tex = xp.tab_classsplit(INP)
    for label in ("$B$ effect", "$D$ effect", "$B{\\times}D$", "Arm"):
        assert label in tex
    assert "{Share}" not in tex  # the class mix is in the main table of results
    # a block per study, the arms in the Arm column: per shift and class (definition, other,
    # all) exact recomputation and the writer's programs in the main study, and exact
    # recomputation in the replication
    body = tex.split("\\mtgroup{13}{Main study: 20 test worlds}\n")[1].split("\\bottomrule")[0]
    main, rep = body.split("\\mtgroup{13}{Replication: 20 fresh worlds}\n")
    assert main.count(" & exact & ") == 9 and main.count(" & programs & ") == 9
    assert rep.count(" & exact & ") == 9 and "programs" not in rep
    assert "replication &" not in tex  # a study, not an arm
    assert "\\color{mtmuted}" not in tex  # the studies are told apart by their banners
    assert tex.count(" \\\\\n") == 2 + 27  # two header rows, then the rows of each study
    # estimate and interval as two aligned cells: signed, the interval small and muted, a thin
    # space after the comma
    assert "$-$0.6 & \\mtint{[$-$3.8,\\,$+$2.5]}" in tex and "\\scriptsize" not in tex
    assert tex.count("r@{\\,}l") == 5  # one column pair per contrast
    assert "\\%" not in tex.split("\\midrule", 1)[1]  # units in the header only


def test_class_table_sets_exact_recomputation_and_the_programs_in_one_column():
    """The D effect at B=1 of exact recomputation and of the writer's programs on definition
    items after a source revision share a column, so the two can be read off one table."""
    tex = xp.tab_classsplit(INP)
    lines = tex.splitlines()
    first = next(i for i, line in enumerate(lines) if line.startswith("$\\Ssrc$ & definition"))
    rows = {line.split(" & ")[2]: line.split(" & ") for line in lines[first : first + 2]}
    main = INP["main"]["explore_class_split"]["D1"]["src"]["source"]["definition"]
    dm = INP["c"]["explore_class_split"]["D1s"]["src"]["source"]["definition"]
    key = "D_effect_update_B1"
    exact = xp._num(main[key]["est"], sign=True)
    programs = xp._num(dm[key]["est"], sign=True)
    assert rows["exact"][9] == exact and rows["programs"][9] == programs  # column B=1 of D


@pytest.mark.paper
def test_class_table_holds_the_c3_values():
    """The programs' rows over all items are block C's B x D (macros SynthCThree...)."""
    tex = xp.tab_classsplit(INP)
    numbers = (ROOT / "paper/generated/numbers.tex").read_text()
    for name in ("SynthCThreeSrc", "SynthCThreeDef", "SynthExCThreeOrphan"):
        m = re.search(rf"\\newcommand{{\\{name}Est}}{{(.*)}}\n", numbers)
        assert m, name
        est = m.group(1).replace("\\,pp", "")
        programs = [line for line in tex.splitlines() if " & programs & " in line]
        assert any(f"{est} & \\mtint{{[" in line for line in programs), name


def test_class_table_caption_and_notes_name_the_arms():
    tex = xp.tab_classsplit(INP)
    caption = tex.split("\\label{tab:classsplit}")[0]
    assert "with exact recomputation and with the writer's programs" in caption
    notes = tex.split("\\mtnotes{", 1)[1]
    for term in ("Exact: exact recomputation as the $D{=}1$ arm", "Programs: the writer's"):
        assert term in notes, term
    assert "regist" not in tex and "exploratory" not in tex
    # the contrasts are defined once, in the opening of the appendix of results
    assert "minus" not in caption + notes and "95\\%" not in caption + notes


def test_cells_of_the_shared_table_style():
    e = {"est": -0.214, "lo": -0.54, "hi": 0.3}
    assert xp.mt_ci_cells(e, f=rp.snum) == r"$-$0.21 & \mtint{[$-$0.54,\,0.30]}"
    assert xp.mt_ci_cells({"est": 0.0, "lo": 0.0, "hi": 0.0}) == "0.0 &"  # every world the same
    assert xp.mt_ci_pair_cells({"est": 0.0, "lo": 0.0, "hi": 0.0}) == "0.0 & "
    assert xp.mt_ci_cells(None, span="c@{}") == rf"\multicolumn{{2}}{{c@{{}}}}{{{MISSING}}}"
    assert xp.mt_ci_cells({"est": float("nan")}).endswith(f"{{{xp.UNDEFINED}}}")
    assert xp.mt_ci_cells(e, sign=True, style=xp.MUTED).startswith(r"\color{mtmuted}$-$21.4 &")
    assert xp.mt_pair(e, f=rp.snum) == r"\mtpair{$-$0.21}{[$-$0.54,\,0.30]}"
    assert xp.mt_pair(None) == MISSING and xp.mt_pair({"est": float("nan")}) == xp.UNDEFINED
    # an N column is as wide as its widest value; a printed 100 stays, blanks and dashes don't count
    assert xp.n_template(["333", "1,459", "", MISSING]) == "0,000"
    assert xp.n_template(["99.7", "100.0", xp.UNDEFINED]) == "100.0"
    assert xp.n_template(["15.0", "0.4"]) == "00.0" and xp.n_template([]) == "0"
    # the boxes of a column of pairs: its widest estimate and interval, as printed or as zeros
    pairs = [("0.75", "[0.68,\\,0.84]"), ("$-$0.21", "[$-$0.53,\\,0.31]"), None]
    pairs.append(("$+$3.2", "[$+$0.6,\\,$+$5.0]"))  # longer to type, narrower in print
    assert xp.pair_widths(pairs) == r"\mtpairwidths{$-$0.21}{[$-$0.53,\,0.31]}"
    signed = [("$-$22.2", "[$+$28.3,\\,$+$16.0]"), ("$+$60.0", "[0.0,\\,$+$100.0]")]
    assert xp.pair_widths(signed, template=True) == r"\mtpairwidths{$+$00.0}{[$-$00.0,\,$-$00.0]}"
    full = [("$-$20.0", "[$-$100.0,\\,$+$100.0]")]
    assert xp.pair_widths(full, template=True) == r"\mtpairwidths{$+$00.0}{[$-$100.0,\,$+$100.0]}"
    assert xp.pair_widths([None]) == r"\mtpairwidths{}{}"


@pytest.mark.paper  # the table style, paper/memtab-style.tex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path):
    for name, tex in xp.tables(INP).items():
        (tmp_path / name).write_text(tex)
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
                r"\begin{document}\input{tab_classsplit.tex}\end{document}",
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
    full = xp.fig_classsplit(INP, tmp_path)
    assert full.read_bytes().startswith(b"%PDF")
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare = xp.fig_classsplit(EMPTY, bare_dir)
    assert bare.read_bytes().startswith(b"%PDF") and bare.read_bytes() != full.read_bytes()


# ------------------------------------------------------------------------- generated files
@pytest.mark.paper
def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in xp.tables(INP).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name
    assert (ROOT / "paper/figures/fig_classsplit.pdf").read_bytes().startswith(b"%PDF")
    numbers = (ROOT / "paper/generated/numbers.tex").read_text()
    for name, value in xp.numbers(INP).items():
        assert pn._macro(name, value) in numbers, name


def test_results_sources_name_the_further_stages():
    assert SOURCES["dev_e1"] == "runs/stage1_e1"
    assert SOURCES["dev_orphan_e1"] == "runs/stage1_orphan"
    assert SOURCES["short_e1"] == "runs/stage5_short_e1"
    assert SOURCES["ex"] == "runs/stage5_ex_e1"

"""Paper numbers, tables and figure of the exploratory E1-family analyses (`analysis.ex_e1_paper`),
and the committed stage analyses they read: the class `all` of the split reproduces the registered
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
from memshift.analysis import paper_numbers as pn
from memshift.analysis.paper_data import MISSING
from tests.test_paper_stage4 import check_tex

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "paper/results_sources.json").read_text())
INP = xp.load(ROOT, SOURCES)
EMPTY = xp.load(ROOT, {})
REGISTERED = {  # class-split key -> the registered e1.contrasts key (source-epoch cells)
    "B_effect_update_D0": "B_effect_update_D0_source",
    "B_effect_update_D1": "B_effect_update_D1_source",
    "B_effect_update_pooled": "B_effect_update_pooled_source",
    "BxD_interaction_update": "BxD_interaction_update_source",
    "D_effect_repair": "D_effect_repair_source",
    "D_effect_repair_changed": "D_effect_repair_changed_source",
}


def _same(a: float, b: float) -> bool:
    return a == b or (math.isnan(a) and math.isnan(b))


# ---------------------------------------------------------------- the committed stage analyses
@pytest.mark.parametrize("key", ["main", "rep", "gem", "llama", "gpt", "dev"])
def test_class_all_reproduces_the_registered_contrasts_to_the_last_digit(key):
    m = INP[key]
    for b, contrasts in m["contrasts"].items():
        allc = m["explore_class_split"]["D1"][b]["source"]["all"]
        for new, old in REGISTERED.items():
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
    assert se["verdicts_non_identity"]["replicated"] == 15
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
            check_tex(tex)
            assert tex.startswith("\\begin{table}") and f"\\label{{tab:{name[4:-4]}}}" in tex
    assert MISSING in xp.tab_classsplit(EMPTY) and MISSING in xp.tab_decomp(EMPTY)
    assert MISSING not in xp.tab_classsplit(INP) and MISSING not in xp.tab_decomp(INP)


def test_class_table_shows_both_studies_for_every_contrast():
    tex = xp.tab_classsplit(INP)
    for label in ("$B$ effect, $D{=}0$", "$D$ effect, $B{=}1$", "$B{\\times}D$ interaction"):
        assert label in tex
    assert tex.count("\\quad\\textcolor{gray}{R}") == 5  # one replication line per contrast
    # the one interval format: estimate and bounds signed, scriptsize, a thin space after the comma
    assert "$-$0.6\\,{\\scriptsize [$-$3.8,\\,$+$2.5]}" in tex and "\\tiny" not in tex


def test_class_table_caption_defines_the_contrasts_and_the_status_of_each_study():
    caption = xp.tab_classsplit(INP).split("\\label{tab:classsplit}")[0]
    for term in (
        "$B$ effect: $B{=}1$ minus $B{=}0$ at the given $D$",
        "$D$ effect: $D{=}1$ minus $D{=}0$ at the given $B$",
        "$B{\\times}D$: the $B$ effect at $D{=}1$ minus that at $D{=}0$",
        "share: the class's share of the changed item-times in the main study",
        "whose class split is exploratory",
        "whose registered class rows replicate it",
    ):
        assert term in caption, term
    assert "pre-registered" not in caption and "\nshare & " in xp.tab_classsplit(INP)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path):
    for name, tex in xp.tables(INP).items():
        (tmp_path / name).write_text(tex)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage[letterpaper,margin=1in]{geometry}",
                r"\usepackage{amsmath,newtxtext,newtxmath,xcolor,booktabs}",
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}",
                r"\begin{document}\input{tab_classsplit.tex}\input{tab_decomp.tex}\end{document}",
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
    full = xp.fig_classsplit(INP, tmp_path)
    assert full.read_bytes().startswith(b"%PDF")
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    bare = xp.fig_classsplit(EMPTY, bare_dir)
    assert bare.read_bytes().startswith(b"%PDF") and bare.read_bytes() != full.read_bytes()


# ------------------------------------------------------------------------- generated files
def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in xp.tables(INP).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name
    assert (ROOT / "paper/figures/fig_classsplit.pdf").read_bytes().startswith(b"%PDF")
    numbers = (ROOT / "paper/generated/numbers.tex").read_text()
    for name, value in xp.numbers(INP).items():
        assert pn._macro(name, value) in numbers, name


def test_results_sources_name_the_exploratory_stages():
    assert SOURCES["dev_e1"] == "runs/stage1_e1"
    assert SOURCES["dev_orphan_e1"] == "runs/stage1_orphan"
    assert SOURCES["short_e1"] == "runs/stage5_short_e1"
    assert SOURCES["ex"] == "runs/stage5_ex_e1"

"""Macros and appendix tables of the exploratory analyses (`analysis.ex_numbers`,
`analysis.ex_tables`): they read the keys the analyses write, render MISSING for a stage that is
not there, and do not collide with the existing macros."""

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import e2, e3, e5, ex_e5_null, ex_numbers, ex_tables
from memshift.analysis import paper_data as pd
from memshift.analysis import paper_numbers as pn
from memshift.analysis.paper_data import MISSING, NATIVES
from memshift.experiments.e2 import tier_name
from tests.ex_e5_fixture import build_runs
from tests.test_ex_e2_predictors import _synthetic
from tests.test_ex_tests import _cv
from tests.test_paper_stage4 import check_tex

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())


def _roundtrip(x: Any) -> dict[str, Any]:
    return dict(json.loads(json.dumps(x, default=str)))


CROSS = tier_name("cross:meta-llama/Llama-3.1-8B-Instruct")


def _e2_analysis(
    tmp_path: Path, tiers: tuple[str, ...] = ("same", "upgrade", "downgrade")
) -> dict[str, Any]:
    """E2 on the given tiers and the four cells: the calibration rows of the predictors tests, and
    the other cells as copies whose items gain and lose in a fixed pattern."""
    rows = _synthetic(tmp_path, tiers=tiers)
    extra = []
    for n, r in enumerate(rows):
        for system, d, b in (
            ("holdmem_D1B1_source", 1, 1),
            ("holdmem_D0B0", 0, 0),
            ("holdmem_D1B0", 1, 0),
        ):
            before = n % 7 != 0
            after = before if b == 0 else (n % 11 != 0 or not before)
            extra.append(
                {
                    **r,
                    "system": system,
                    "D": d,
                    "B": b,
                    "valid_before": before,
                    "valid_after": after,
                }
            )
    (tmp_path / "survival.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + extra))
    return _roundtrip(e2.analyse(tmp_path))


def _e3_analysis(tmp_path: Path) -> dict[str, Any]:
    systems = {
        "oracle": (0.9, 0.3, 1.0),
        "holdmem_D1B1_source": (0.8, 1.0, 0.6),
        "rag": (0.7, 0.2, 0.5),
        "zero": (0.9, 0.3, 0.8),
    }
    rows = []
    for h in (1, 3, 12):
        for system, per in _cv(systems).items():
            for w, facts in per.items():
                rows += [
                    {
                        "system": system,
                        "world_id": w,
                        "fact_id": f,
                        "contract_valid": x,
                        "tags": {"horizon_months": h},
                    }
                    for f, x in facts.items()
                ]
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "world_results.jsonl").write_text("")
    return _roundtrip(e3.analyse(tmp_path))


def _e5_analysis(tmp_path: Path, monkeypatch: Any) -> dict[str, Any]:
    native, white, noise, base = build_runs(tmp_path, monkeypatch)
    return _roundtrip(e5.analyse([native], [white], tmp_path / "out", [noise], [base]))


def _six(res: dict[str, Any]) -> dict[str, Any]:
    """The analysis with every native system of block N: the first system's entry cloned."""
    res = copy.deepcopy(res)
    first = res["explore_native_null"]["systems"]["mem0"]
    lab = res["labels"]["mem0"]
    entries = {n.key: copy.deepcopy(first) for n in NATIVES}
    res["explore_native_null"] = {
        "systems": entries,
        "total": _roundtrip(ex_e5_null._total(entries)),
    }
    res["labels"] = {n.key: copy.deepcopy(lab) for n in NATIVES}
    return res


@pytest.fixture
def e2_res(tmp_path):
    return _e2_analysis(tmp_path)


@pytest.fixture
def e3_res(tmp_path_factory):
    return _e3_analysis(tmp_path_factory.mktemp("e3"))


@pytest.fixture
def e5_res(tmp_path, monkeypatch):
    return _e5_analysis(tmp_path, monkeypatch)


# ------------------------------------------------------------------------------------ macros
def _letters(macros: dict[str, str]) -> None:
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only names
        assert "Ex" in name, name


def test_a_stage_that_is_not_staged_renders_only_missing_markers():
    empty = {
        **ex_numbers.e2_numbers({}),
        **ex_numbers.m1_numbers({}),
        **ex_numbers.t3_numbers(None, "EThree"),
        **ex_numbers.t3_numbers(None, "Rep"),
        **ex_numbers.mem0_null_numbers(None),
        **ex_numbers.native_null_numbers(None),
    }
    assert len(empty) > 1500
    assert all(v == MISSING for v in empty.values()), [k for k, v in empty.items() if v != MISSING][
        :5
    ]
    _letters(empty)


def test_slopes_print_as_bounds_once_the_fit_separates():
    assert ex_numbers.sn(1.014) == "1.01" and ex_numbers.sn(-0.534) == "$-$0.53"
    assert ex_numbers.sn(25.0) == "$>20$" and ex_numbers.sn(-101.4) == "$<-20$"
    assert ex_numbers.sn(0.055, 3, signed=True) == "$+$0.055"
    assert ex_numbers.sn(float("nan")) == MISSING and ex_numbers.sn(None) == MISSING


def test_e2_macros_follow_the_analysis(e2_res):
    out = ex_numbers.e2_numbers(e2_res)
    _letters(out)
    pooled = e2_res["explore_writer_predictors"]["sets"]["all"]["pooled"]
    assert out["ETwoExWriterItemsPooled"] == f"{pooled['n_items']:,}" == "900"
    assert out["ETwoExWriterFailuresPooled"] == str(pooled["n_failures"])
    agree = pooled["predictors"]["agree"]
    # the same estimate the registered macro prints, with both interval types beside it
    slope = out["ETwoExWriterAgreePooledSlope"].replace("$-$", "-")
    assert slope.startswith(f"{agree['slope']['est']:.2f} [")
    w, t = agree["slope"]["world"], agree["slope"]["type"]
    assert f"[{w['lo90']:.2f}, {w['hi90']:.2f}]" in slope
    assert t["lo90"] != w["lo90"]
    assert out["ETwoExWriterAgreePooledSlopeType"] != out["ETwoExWriterAgreePooledSlope"]
    assert "ETwoExWriterAgreePooledAurocDiff" not in out  # agreement is the reference
    assert out["ETwoExWriterNewAccPooledAurocDiff"].startswith(("$+$", "$-$"))
    assert "ETwoExWriterNoDefNonDefPooledSlope" not in out  # constant on non-definition items
    assert out["ETwoExWriterNoDefItemsPooled"] == "540"
    # leave-one-type-out and failure counts
    loto = e2_res["explore_writer_loto"]["pooled"]["range"]["agree"]
    assert out["ETwoExLotoSlopeMin"] == ex_numbers.sn(loto["slope_min"])
    assert out["ETwoExLotoSlopeMinType"] == loto["slope_min_type"].replace("_", r"\_")
    assert out["ETwoExFailTotal"] == str(
        e2_res["explore_failures_by_type"]["summary"]["n_failures"]
    )
    assert out["ETwoExFailKappaOverdueTaskPooled"].endswith(" of 180")
    # survival, recovery, net change, the eps grid and T1 by item class, per cell and tier
    assert out["ETwoExGainsDzeroBzeroSame"] == out["ETwoExLossesDzeroBzeroSame"] == "0"
    assert out["ETwoExTrivialDzeroBzeroSame"] == "yes" and out["ETwoExTrivialDoneBoneSame"] == "no"
    net = e2_res["explore_survival_recovery"]["same:holdmem_D1B1_source"]["net"]
    assert out["ETwoExNetDoneBoneSame"].startswith(pn.pp(net["est"]))
    assert out["ETwoExTOneRetEpsThreeDoneBoneSame"] in ("within", "outside")
    assert out["ETwoExTOneNetEqEpsOneDzeroBzeroSame"] == "within"  # no item changed
    assert out["ETwoExTOneRetEpsFiveWithinDoneBone"].endswith(" of 3")
    assert out["ETwoExTOneClassFreetextDoneBoneSameShare"].endswith(r"\%")
    # M1's switches: the stage-2 tiers are there, a switch that is not staged is missing
    m1 = ex_numbers.m1_numbers(e2_res)
    _letters(m1)
    assert m1["MOneExWriterAgreePooledSlope"] == out["ETwoExWriterAgreePooledSlope"]
    assert m1["MOneExWriterAgreeSameSlope"] == out["ETwoExWriterAgreeSameSlope"]
    assert m1["MOneExWriterAgreeLlamaSlope"] == MISSING


def test_a_cross_family_switch_is_covered_without_a_change(tmp_path):
    """Block M1's cross-family tiers (`cross_<slug>`): the predictors, T1 fields and macros of the
    stage-2 tiers apply to them as soon as they are in the merged E2 analysis."""
    res = _e2_analysis(tmp_path, ("same", CROSS))
    w = res["explore_writer_predictors"]
    assert w["tiers"] == sorted(["same", CROSS]) and CROSS in res["explore_failures_by_type"]
    assert f"{CROSS}:holdmem_D1B1_source" in res["explore_survival_recovery"]
    out = ex_numbers.m1_numbers(res)
    _letters(out)
    assert (
        out["MOneExWriterAgreeLlamaSlope"] != MISSING and out["MOneExNetDoneBoneLlama"] != MISSING
    )
    assert out["MOneExWriterAgreeGemmaSlope"] == MISSING  # that switch is not staged
    assert out["MOneExWriterAgreePooledSlope"] != MISSING
    # the pooled scope covers both switches
    assert w["sets"]["all"]["pooled"]["n_items"] == 2 * 300
    tex = ex_tables.tab_writer_predictors(None, res)
    assert "block M1, pooled over its switches; 600 items" in tex


def test_t3_macros_by_class_horizon_and_system(e3_res):
    out = ex_numbers.t3_numbers(e3_res, "EThree")
    _letters(out)
    assert out["EThreeExTThreeHoldRuleOne"].startswith("$-$10.0")  # 0.8 against 0.9
    assert out["EThreeExTThreeHoldDefinitionOneResult"] == "pass"
    assert out["EThreeExTThreeHoldRuleOneResult"] == "fail"
    assert out["EThreeExTThreePassHoldDefinition"] == "3 of 3"
    assert out["EThreeExTThreePassHoldFreetext"] == "0 of 3"
    assert out["EThreeExTThreePassZeroRule"] == "3 of 3"
    assert out["EThreeExTThreeOracleCvDefinitionOne"].startswith("30.0")
    assert "EThreeExTFourSlackOracleOne" not in out  # the oracle context has no serving cost
    assert out["EThreeExTFourSlackHoldOne"] == MISSING  # no cost rows in the fixture
    rep = ex_numbers.t3_numbers(e3_res, "Rep")
    assert rep["RepExTThreeHoldRuleOne"] == out["EThreeExTThreeHoldRuleOne"]


def test_null_macros_for_one_system_and_the_totals_need_all_six(e5_res):
    mem = ex_numbers.null_numbers(e5_res["explore_native_null"]["systems"]["mem0"], "EFiveExMem")
    _letters(mem)
    assert (
        mem["EFiveExMemNullInsideOf"] == "6 of 6"
        and mem["EFiveExMemNullInsideCircularOf"] == "2 of 2"
    )
    assert mem["EFiveExMemNullZeroInsideOf"] == "6 of 6"
    assert mem["EFiveExMemNullBIdentified"] == "no" and mem["EFiveExMemNullBStraddles"] == "yes"
    assert mem["EFiveExMemNullContainedUpdateSrc"].startswith("(0,0)")
    assert mem["EFiveExMemNullContainedCountUpdateSrc"].endswith(" of 9")
    assert mem["EFiveExMemNullChanceLabels"] == pn.pct(
        e5_res["explore_native_null"]["systems"]["mem0"]["chance_coverage"]["labels"]
    )
    assert mem["EFiveExMemNullPairedInflatedUpdateSrc"] != mem["EFiveExMemNullPairedUpdateSrc"]
    assert mem["EFiveExMemNullPairedZeroUpdateSrc"].startswith(r"0.0\,pp")
    three = ex_numbers.native_null_numbers(e5_res)  # mem0, amem and lightrag: not all six
    assert three["NatMemExNullInsideOf"] == "6 of 6" and three["NatMemosExNullInsideOf"] == MISSING
    assert three["NatExNullInsideOf"] == MISSING and three["NatExNullBIdentified"] == MISSING
    six = ex_numbers.native_null_numbers(_six(e5_res))
    _letters(six)
    assert six["NatExNullInsideOf"] == "36 of 36" and six["NatExNullInsideOosOf"] == "24 of 24"
    assert six["NatExNullZeroInsideOf"] == "36 of 36" and six["NatExNullBIdentified"] == "0 of 6"
    assert six["NatExNullChanceLabels"].endswith(r"\%")
    assert six["NatCogneeExNullInsideOf"] == "6 of 6"


def test_all_exploratory_names_are_new_and_letters_only():
    inp = pd.load_stage4(ROOT, SOURCES)
    macros = pn.build_macros(ROOT)  # raises when an exploratory name collides with another macro
    assert not [k for k in macros if not re.fullmatch(r"[A-Za-z]+", k)]
    ex = ex_numbers.build(ROOT, SOURCES, inp)
    assert len(ex) > 2500 and set(ex) <= set(macros)
    assert all(macros[k] == v for k, v in ex.items())


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_the_exploratory_macros_compile_as_text(tmp_path):
    macros = pn.with_estimates(ex_numbers.build(ROOT, SOURCES, pd.load_stage4(ROOT, SOURCES)))
    (tmp_path / "numbers.tex").write_text(
        "\n".join(pn._macro(k, v) for k, v in sorted(macros.items())) + "\n"
    )
    lines = "".join(f"\\noindent\\texttt{{{k}}} {{\\{k}}}\\par\n" for k in sorted(macros))
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,newtxtext,newtxmath}",
                r"\newcommand{\holdmem}{\textsc{HoldMem}}",
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
        errors="replace",  # TeX's console output can carry 8-bit glyphs from box reports
    )
    assert done.returncode == 0, done.stdout[-1500:]


def test_the_committed_analyses_answer_the_review_questions():
    """What the staged data say (pinned: a restaged analysis that changes these is a finding)."""
    macros = pn.build_macros(ROOT)
    assert macros["ETwoExFailTotal"] == "41" and macros["ETwoExFailItems"] == "2,446"
    assert macros["ETwoExWriterAgreePooledSlopeEst"] == "1.01"
    assert macros["ETwoExWriterAgreePooledAurocEst"] == "0.832"
    assert macros["ETwoExWriterNewAccPooledAurocEst"] == "0.887"  # the baseline ranks better
    assert macros["ETwoExNetDoneBoneSame"].startswith(r"$+$8.3")
    assert macros["EFiveExMemNullBIdentified"] == "no"
    assert macros["EFiveExMemNullZeroInsideOf"] == macros["EFiveExMemNullZeroNativeInsideOf"]


# ------------------------------------------------------------------------------------- tables
def _tables(e2_res, e3_res, e5_res) -> dict[str, str]:
    return {
        "writer": ex_tables.tab_writer_predictors(e2_res, e2_res),
        "tests": ex_tables.tab_tests_sensitivity(e2_res, e3_res, e3_res),
        "native": ex_tables.tab_native_null(_six(e5_res)),
    }


def test_tables_are_wellformed_with_data_and_without(e2_res, e3_res, e5_res):
    for tex in _tables(e2_res, e3_res, e5_res).values():
        check_tex(tex)
    for tex in (
        ex_tables.tab_writer_predictors(None, None),
        ex_tables.tab_tests_sensitivity(None, None, None),
        ex_tables.tab_native_null(None),
    ):
        check_tex(tex)
        assert MISSING in tex
    # the block that is not staged (M1, replication, the five other native systems) is missing
    tests = ex_tables.tab_tests_sensitivity(e2_res, e3_res, None)
    assert "(d) replication" in tests and MISSING in tests
    assert "\\label{tab:writer-predictors}" in _tables(e2_res, e3_res, e5_res)["writer"]


def test_writer_table_carries_the_baselines_and_both_interval_types(e2_res):
    tex = ex_tables.tab_writer_predictors(e2_res, None)
    for label in ("agreement (registered)", "new-writer accuracy", "old-writer accuracy"):
        assert label in tex
    pooled = e2_res["explore_writer_predictors"]["sets"]["all"]["pooled"]["predictors"]["agree"]
    t = pooled["slope"]["type"]
    w = pooled["slope"]["world"]
    assert f"[{ex_numbers.sn(w['lo90'])},\\,{ex_numbers.sn(w['hi90'])}]" in tex
    assert f"[{ex_numbers.sn(t['lo90'])},\\,{ex_numbers.sn(t['hi90'])}]" in tex
    assert "block M1" in tex and MISSING in tex  # M1 is not staged
    assert tex.count(r"\midrule") == 6


def test_tests_table_orders_the_cells_and_states_what_holds_by_construction(e2_res, e3_res):
    tex = ex_tables.tab_tests_sensitivity(e2_res, e3_res, None)
    heads = [r for r in ex_tables._t1_cell_rows(e2_res) if r.startswith(r"\multicolumn")]
    assert "(1,1)" in heads[0] and "(0,1)" in heads[1]  # the T1 cell first
    assert "(1,0)" in heads[2] and "(0,0)" in heads[3] and "no item changes" in heads[3]
    assert "no item changes in any tier" in tex  # the B = 0 cells
    by = ex_tables._by_construction(e2_res, None)
    assert [r[0] for r in by] == ["T1", "T1", "T2", "T3", "T4"]
    assert by[0][2].startswith("6 of 6 B=0 cell-tiers")
    assert r"\label{tab:tests-sensitivity}" in tex and r"\label{tab:tests-sensitivity-t3}" in tex


def test_native_table_lists_every_system_and_the_references_that_hold_the_value(e5_res):
    tex = ex_tables.tab_native_null(e5_res)
    body = tex.split(r"\toprule", 1)[1]
    for nat in NATIVES:
        assert nat.label in body
    assert "Mem0, predicted by (0,0)" in tex.replace(r"\emph{Mem0}", "Mem0")
    assert "zero, RAG" in tex or "(0,0)" in tex
    assert r"$^\circ$" in tex  # the label-circular comparisons are marked
    check_tex(tex)


def test_native_table_defines_its_counts_and_gives_the_out_of_sample_baselines(e5_res):
    tex = ex_tables.tab_native_null(e5_res)
    caption = tex.split(r"\label{tab:native-null}")[0]
    for term in (
        r"Inside: inside the 95\% interval of the cell with the inferred label",
        r"Discriminating: inside, and outside that of the cell with the other $B$",
        "the share of deleted items still answered",
        "gold-dep.: gold-dependency context",
        "Last row: sums",
        r"$^\circ$: label-circular comparison",
    ):
        assert term in caption, term
    head = next(ln for ln in tex.splitlines() if ln.startswith("system & label"))
    assert head.count("out-of-sample") == 3  # the native estimate, the zero system, plain RAG
    assert "oracle" not in tex.split(r"\label{tab:native-null}")[1]  # named as in the paper
    # with all six systems: per system and in the last row, the baselines out of sample
    six = _six(e5_res)
    tex = ex_tables.tab_native_null(six)
    ex = six["explore_native_null"]
    mem = next(ln for ln in tex.splitlines() if ln.startswith("Mem0 &")).split(" & ")
    total = next(ln for ln in tex.splitlines() if ln.startswith("total &")).split(" & ")
    assert len(mem) == len(total) == 13
    for k, name in ((7, "zero"), (9, "rag")):
        bc = ex["systems"]["mem0"]["baseline_counts"].get(name)
        tbc = ex["total"]["baseline_counts"].get(name)
        for j, part in ((0, "all"), (1, "out_of_sample")):
            want = ex_tables._pair(bc["baseline"][part]) if bc else MISSING
            assert mem[k + j] == want, (name, part)
            if tbc:
                assert total[k + j] == ex_tables._pair(tbc["baseline"][part]), (name, part)
    assert total[4] == ex_tables._pair(ex["total"]["counts"]["all"])
    assert total[3] == f"{ex['total']['b_identified']}/6"


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path, e2_res, e3_res, e5_res):
    tables = _tables(e2_res, e3_res, e5_res)
    tables |= {
        "writer_missing": ex_tables.tab_writer_predictors(None, None),
        "tests_missing": ex_tables.tab_tests_sensitivity(None, None, None),
        "native_missing": ex_tables.tab_native_null(None),
    }
    for name, tex in tables.items():
        (tmp_path / f"{name}.tex").write_text(tex.replace("\\label{tab:", f"\\label{{tab:{name}"))
    body = "".join(f"\\input{{{name}.tex}}\n" for name in tables)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[letterpaper,margin=1in]{geometry}\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,newtxtext,newtxmath,booktabs,array,xcolor,caption}",
                r"\newcommand{\holdmem}{\textsc{HoldMem}}\newcommand{\pass}{\checkmark}",
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}",
                r"\newcommand{\fail}{$\times$}\newcommand{\cref}[1]{#1}",
                r"\begin{document}",
                body,
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


@pytest.mark.paper
def test_committed_exploratory_tables_are_current():
    """paper/generated/tab_{writer_predictors,tests_sensitivity,native_null}.tex are what the
    generators print for the staged data: regenerate them with `python -m
    memshift.analysis.paper_tables` whenever a stage or a generator changes."""
    inp = pd.load_stage4(ROOT, SOURCES)
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in ex_tables.build(ROOT, SOURCES, inp).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name

"""Macros and the appendix table of the further analyses (`analysis.ex_numbers`,
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
from tests.test_paper_stage4 import PAPER_PAGE, check_tex, table_style

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
    # the same estimate the main macro prints, with both interval types beside it
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


def test_all_ex_names_are_new_and_letters_only():
    inp = pd.load_stage4(ROOT, SOURCES)
    macros = pn.build_macros(ROOT)  # raises when an `Ex` name collides with another macro
    assert not [k for k in macros if not re.fullmatch(r"[A-Za-z]+", k)]
    ex = ex_numbers.build(ROOT, SOURCES, inp)
    assert len(ex) > 2500 and set(ex) <= set(macros)
    assert all(macros[k] == v for k, v in ex.items())


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_the_ex_macros_compile_as_text(tmp_path):
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


def test_the_committed_analyses_give_the_pinned_values():
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
def test_table_is_wellformed_with_data_and_without(e5_res):
    tex = ex_tables.tab_native_null(_six(e5_res))
    check_tex(tex)
    assert MISSING not in tex.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]
    empty = ex_tables.tab_native_null(None)
    check_tex(empty)
    assert MISSING in empty
    # the five other native systems are not staged: missing
    assert MISSING in ex_tables.tab_native_null(e5_res)
    assert set(ex_tables.build(ROOT, SOURCES, pd.load_stage4(ROOT, SOURCES))) == {
        "tab_native_null.tex"
    }


def test_native_table_lists_every_system_and_the_references_that_hold_the_value(e5_res):
    tex = ex_tables.tab_native_null(e5_res)
    body = tex.split(r"\toprule", 1)[1]
    for nat in NATIVES:
        assert nat.label in body
    assert "\nMem0 & " in tex and "\ncell (0,0) & " in tex  # the system above its cell
    assert "zero, RAG" in tex or "(0,0)" in tex
    assert r"$^\circ$" in tex  # the comparisons after S_del, from which B is inferred
    assert "predicted" not in tex and "exploratory" not in tex and r"\pass" not in tex
    check_tex(tex)


def test_native_table_gives_each_systems_label_persistence_and_count(e5_res):
    """One table for the native systems (it carries both labels): per system and comparison the
    references that contain the native value and the paired differences, and in the system's
    first column its inferred label, its deletion persistence P with the 95% interval and its
    comparisons inside the interval of the label cell (k/n, or k, j/n where some inside are not
    outside the cell with the other B)."""
    tex = ex_tables.tab_native_null(e5_res)
    assert tex.count(r"\begin{table*}") == 1
    assert r"\label{tab:native-diff}\label{tab:native-null}" in tex
    notes = " ".join(re.findall(r"\\mtnotes\{(.*)\}\n", tex))
    for term in (
        r"Inside $k$/$n$: $k$ comparisons inside the 95\% interval of the cell",
        r"$(D,B)$: \holdmem{} cell, c: citation key",
        "the share of deleted items still answered, with its 95\\% interval",
        "no-store: no-store BM25, RAG: plain RAG, oracle: oracle context",
        r"$^\circ$: after $\Sdel$, from which $B$ is inferred",
        "Difference: native minus the",
    ):
        assert term in notes, term
    for word in ("circular", "out-of-sample", "discriminating", "inflated", "exploratory"):
        assert word not in tex, word
    assert "white-box" not in tex and "(d,b)" not in tex and "Chance" not in tex
    assert "zero" not in tex.split(r"\label{tab:native-null}")[1]  # named as in the paper
    six = _six(e5_res)
    ex = six["explore_native_null"]
    lines = ex_tables.tab_native_null(six).splitlines()
    first = next(i for i, ln in enumerate(lines) if ln.startswith("Mem0 & "))
    stub = [ln.split(" & ")[0] for ln in lines[first : first + 6]]
    mem = ex["systems"]["mem0"]
    b, k = mem["b_identified"], mem["counts"]["all"]
    assert stub[:2] == ["Mem0", "cell (0,0)"]
    assert stub[2] == rf"$P$ {ex_tables._num(b['rate'])}\%"
    assert stub[3] == rf"\mtint{{[{ex_tables._num(b['lo'])},\,{ex_tables._num(b['hi'])}]}}"
    assert stub[4] == f"inside {k['inside_predicted']}/{k['comparisons']}" and stub[5] == ""
    # k, j/n where one comparison inside is not outside the other cell
    some = next(
        c["counts"]["all"] for c in ex["systems"].values() if c["counts"]["all"]["inside_predicted"]
    )
    some["discriminating"] = some["inside_predicted"] - 1
    tex = ex_tables.tab_native_null(six)
    assert r"$j$ of them also outside that of the cell with the other $B$" in tex
    assert f"inside {some['inside_predicted']}, {some['discriminating']}/" in tex


@pytest.mark.paper  # the table style, paper/memtab-style.tex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path, e5_res):
    tables = {
        "native": ex_tables.tab_native_null(_six(e5_res)),
        "native_missing": ex_tables.tab_native_null(None),
    }
    for name, tex in tables.items():
        (tmp_path / f"{name}.tex").write_text(tex.replace("\\label{tab:", f"\\label{{tab:{name}"))
    body = "".join(f"\\input{{{name}.tex}}\n" for name in tables)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                PAPER_PAGE + r"\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,newtxtext,newtxmath,booktabs,array,xcolor,caption}",
                table_style(tmp_path),
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
def test_committed_table_is_current():
    """paper/generated/tab_native_null.tex is what the generator prints for the staged data:
    regenerate it with `python -m memshift.analysis.paper_tables` whenever a stage or a generator
    changes."""
    inp = pd.load_stage4(ROOT, SOURCES)
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in ex_tables.build(ROOT, SOURCES, inp).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name

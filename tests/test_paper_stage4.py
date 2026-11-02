"""Stage-4 paper generators: every number, table and figure come from staged analyses; a stage
that has not run renders as the MISSING marker, and nothing is invented for it. Real analyses
(committed stages, and the analysis functions run on small synthetic data) supply the input
structures, so the readers follow the writers' actual keys."""

import copy
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import e2, e5, e7, families, paper_data, rank_corr, rendered, replication
from memshift.analysis import figures as figs
from memshift.analysis import paper_numbers as pn
from memshift.analysis import paper_tables as pt
from memshift.analysis.paper_data import FAMILIES, MISSING
from memshift.experiments.e2 import tier_name
from tests.test_e2_analysis import _rows as e2_rows
from tests.test_families import _items as family_items
from tests.test_rank_corr import LEVELS
from tests.test_rank_corr import WORLDS as RANK_WORLDS
from tests.test_rank_corr import _table as rank_table
from tests.test_rendered import RENDERED, TEMPLATE
from tests.test_rendered import _stage as rendered_stage
from tests.test_rendered import _table as rendered_table

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
BLOCK_PREFIXES = ("Rep", "Synth", "MOne", "MTwo", "MThree", "Nat", "Rend", "Rank", "Mult")
VERDICT_WORDS = {
    "holds",
    "fails",
    "does not hold",
    "not judged",
    "judged",
    "triggered",
    "not triggered",
    "replicated",
    "not replicated",
    "same direction",
}


def _load(stage: str, name: str) -> Any:
    return json.loads((RUNS / stage / "analysis" / name).read_text())


def _roundtrip(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


ORIG = {f"e{n}": _load(f"stage2_e{n}", f"e{n}_metrics.json") for n in (1, 2, 3)}
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
MULT_RESULT = _load("stage4_multiplicity", "multiplicity.json")
REP_PARTIAL = {  # what is staged today: E1 and E3 on the replication worlds, E2 is not
    "e1": _load("stage4_rep_e1", "e1_metrics.json"),
    "e2": None,
    "e3": _load("stage4_rep_e3", "e3_metrics.json"),
}
NONE3 = {"e1": None, "e2": None, "e3": None}
EMPTY = {
    "orig": NONE3,
    "rep": NONE3,
    "pool": NONE3,
    "c": None,
    "m1": None,
    "m2": {f.key: None for f in FAMILIES},
    "m3": {},
    "n": None,
    "n3": None,
    "ncost": {},
    "rendered": None,
    "rank": None,
    "mult": {},
}


def _est(x, lo=None, hi=None):
    return {
        "est": x,
        "lo": x - 0.02 if lo is None else lo,
        "hi": x + 0.02 if hi is None else hi,
        "n_worlds": 5,
        "n_items": 50,
    }


def no_verdicts(out: dict[str, str]) -> None:
    """No macro states a verdict: no such name, and no value that is a verdict word."""
    for name, value in out.items():
        assert not re.search(r"Verdict|Holds|Triggered|Replicated|Claim|Pattern|Check[IV]", name)
        assert value not in VERDICT_WORDS, (name, value)


def _first_body(tex: str) -> str:
    """The rows of a table's first tabular, without the caption (which explains the marks)."""
    return tex.split(r"\toprule", 1)[1].split(r"\bottomrule", 1)[0]


# --------------------------------------------------------------------------- names and MISSING
def test_camel_spells_digits_and_drops_separators():
    assert paper_data.camel("B_effect_update_D1_cited") == "BEffectUpdateDoneCited"
    assert paper_data.camel("granularity_false_inval_D0_cited") == "GranularityFalseInvalDzeroCited"
    assert paper_data.camel("BxD_interaction_update_source") == "BxDInteractionUpdateSource"
    assert pn.type_tag("u.home_city") == "UserHomeCity"
    assert pn.type_tag("kappa.vip_sender") == "KappaVipSender"
    assert pn.type_tag("task_due") == "TaskDue"


def test_model_slugs_match_the_stage_directories_and_tier_names():
    assert paper_data.model_slug("meta-llama/Llama-3.1-8B-Instruct") == (
        "meta-llama_llama-3_1-8b-instruct"
    )
    assert paper_data.model_slug("google/gemma-4-12B-it") == "google_gemma-4-12b-it"
    assert paper_data.family_of("openai/gpt-oss-20b").key == "gptoss"
    assert paper_data.family_of("mistralai/Ministral-3-14B-Instruct-2512-BF16") is None
    tier = tier_name("cross:meta-llama/Llama-3.1-8B-Instruct")
    llama = next(t for t in paper_data.TIERS if t.tag == "Llama")
    assert paper_data.tier_key({"calibration": {tier: {}, "same": {}}}, llama) == tier
    assert paper_data.tier_key({"calibration": {"same": {}}}, llama) is None


def test_no_stage_is_staged_renders_only_missing_markers():
    macros = pn.stage4_numbers(EMPTY)
    assert len(macros) > 800
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only names
        assert name.startswith(BLOCK_PREFIXES), name
        # the documented labels come from the system audit, not from a stage
        assert value == MISSING or name.endswith("Documented"), (name, value)
    assert macros["NatLightRagDocumented"] == "(0,1)" and macros["NatMemDocumented"] == "(0,0)"


def test_an_estimate_whose_printed_interval_has_zero_width_is_printed_alone():
    assert pn._macro("A", r"0.0\,pp [0.0, 0.0]") == r"\newcommand{\A}{0.0\,pp}"
    assert pn._macro("A", r"$+$100.0\,pp [$+$100.0, $+$100.0]") == r"\newcommand{\A}{$+$100.0\,pp}"
    assert pn._macro("A", r"0.0\% [0.0, 0.0]") == r"\newcommand{\A}{0.0\%}"
    assert pn._macro("A", "1.00 [1.00, 1.00]") == r"\newcommand{\A}{1.00}"
    kept = r"0.0\,pp [$-$0.9, $+$0.9]"
    assert pn._macro("A", kept) == rf"\newcommand{{\A}}{{{kept}}}"
    assert pn._macro("A", "12 of 20") == r"\newcommand{\A}{12 of 20}"


def test_stage4_names_do_not_collide_with_the_earlier_macros():
    legacy: dict[str, str] = {}
    for fn in (
        pn.e1_numbers,
        pn.nofb_numbers,
        pn.e2_numbers,
        pn.e3_numbers,
        pn.e4_numbers,
        pn.e4x_numbers,
        pn.e5_numbers,
        pn.e5cost_numbers,
        pn.e5wb_numbers("Dev"),
        pn.e6_numbers,
        pn.e7_numbers,
        pn.gate_numbers,
    ):
        legacy |= fn({})
    assert legacy and not any(k.startswith(BLOCK_PREFIXES) for k in legacy)


def test_a_name_defined_twice_in_a_block_is_an_error():
    macros = pn._Macros()
    macros["RepA"] = "1"
    macros.update({"RepB": "2"})
    with pytest.raises(ValueError, match="RepA"):
        macros["RepA"] = "3"
    with pytest.raises(ValueError, match="RepB"):
        macros.update(RepB="4")


def test_a_name_produced_by_two_blocks_is_an_error(monkeypatch):
    monkeypatch.setattr(pn, "rank_numbers", lambda m: {"RepWorlds": "1"})
    with pytest.raises(ValueError, match="RepWorlds"):
        pn.stage4_numbers(EMPTY)


# ---------------------------------------------------------------- stage 2 and stage 3 Ex macros
def test_t1_macros_for_the_cells_other_than_the_primary_one():
    e2m = ORIG["e2"]
    out = pn.e2_numbers(e2m)
    cells = e2m["explore_t1_cells"]
    for tier, tag in (("same", "Same"), ("upgrade", "Up"), ("downgrade", "Down")):
        # the primary cell's T1 is the same computation as `t1`
        assert cells[tier]["holdmem_D1B1_source"] == e2m["t1"][tier]
        c = cells[tier]["holdmem_D0B1_source"]
        assert out[f"ETwoExTOneDzeroBone{tag}"] == pn.ci(c["delta_90"], pn.pp)
        assert out[f"ETwoExTOnePassDzeroBone{tag}"] == ("within" if c["equivalent"] else "outside")
        d = c["delta_90"]
        assert out[f"ETwoExTOneRetentionDzeroBone{tag}"] == pn.ci(
            {k: 1 + d[k] for k in ("est", "lo", "hi")}
        )
        assert f"ETwoExTOneDoneBone{tag}" not in out  # the primary cell keeps ETwoTOne
        assert out[f"ETwoExTOnePassDzeroBzero{tag}"] == "within"  # retention 1 by construction
    # HoldMem(0,1) passes T1 for the same-size and the upgrade writer, not for the downgrade
    assert [out[f"ETwoExTOnePassDzeroBone{t}"] for t in ("Same", "Up", "Down")] == [
        "within",
        "within",
        "outside",
    ]
    assert out["ETwoExTOneWithinDzeroBone"] == "2 of 3"
    assert out["ETwoExTOneWithinDoneBzero"] == out["ETwoExTOneWithinDzeroBzero"] == "3 of 3"
    empty = pn.e2_numbers({})
    assert empty["ETwoExTOneDzeroBoneSame"] == MISSING
    assert empty["ETwoExTOnePassDzeroBoneDown"] == MISSING
    assert empty["ETwoExTOneWithinDzeroBone"] == MISSING
    assert empty["ETwoExTOneRetentionDzeroBoneUp"] == MISSING


def test_e3_macros_without_definition_items_and_the_t3_counts():
    m = ORIG["e3"]
    out = pn.e3_numbers(m)
    hold12 = m["horizons"]["12"]["holdmem_D1B1_source"]["explore_no_definition"]
    assert out["EThreeExNoDefMinusOracleHoldTwelve"] == pn.ci(hold12["minus_oracle"], pn.pp)
    oracle12 = m["horizons"]["12"]["oracle"]["explore_no_definition"]["contract_valid"]
    assert out["EThreeExNoDefCvOracleTwelve"] == pn.ci(oracle12)
    # HoldMem(1,1) passes T3 at every horizon with all items, and only at twelve months without
    # the definition items, on which the oracle context is strong at the short horizons
    assert [out[f"EThreeTThreeResultHold{h}"] for h in ("One", "Three", "Twelve")] == ["pass"] * 3
    assert [out[f"EThreeExNoDefTThreeResultHold{h}"] for h in ("One", "Three", "Twelve")] == [
        "fail",
        "fail",
        "pass",
    ]
    assert out["EThreeTThreePassHold"] == "3 of 3"
    assert out["EThreeExNoDefTThreePassHold"] == "1 of 3"
    assert out["EThreeTThreePassZero"] == out["EThreeExNoDefTThreePassRag"] == "0 of 3"
    assert out["EThreeTThreePassAll"] == "3 of 9" and out["EThreeExNoDefTThreePassAll"] == "1 of 9"
    empty = pn.e3_numbers({})
    assert empty["EThreeExNoDefTThreePassAll"] == empty["EThreeTThreePassHold"] == MISSING
    assert empty["EThreeExNoDefMinusOracleZeroOne"] == MISSING
    partial = copy.deepcopy(m)
    del partial["horizons"]["3"]["zero"]["explore_no_definition"]["T3_pass"]
    assert pn.e3_numbers(partial)["EThreeExNoDefTThreePassZero"] == MISSING  # incomplete: no count


def test_replication_e3_macros_give_both_values_on_the_replication_worlds():
    rep = REP_PARTIAL["e3"]
    out = pn.rep_e3_numbers(ORIG["e3"], rep)
    per = rep["horizons"]["12"]
    assert out["RepZeroMinusHoldTwelve"] == pn.ci(per["zero_minus_holdmem"], pn.pp)
    # the paired difference is the difference of the two system means the RepCv macros print
    zero, hold = per["zero"]["contract_valid"], per["holdmem_D1B1_source"]["contract_valid"]
    assert per["zero_minus_holdmem"]["est"] == pytest.approx(zero["est"] - hold["est"])
    full = pn.with_estimates(pn.rep_numbers(paper_data.replication_view(ORIG, REP_PARTIAL)))
    assert full["RepCvZeroTwelveEst"] == pn.pct(zero["est"])
    assert full["RepCvHoldTwelveEst"] == pn.pct(hold["est"])
    assert full["RepCvOracleTwelveEst"] == pn.pct(per["oracle"]["contract_valid"]["est"])
    # without definition items: the same T3 outcome as stage 2 in every unit
    nd = per["holdmem_D1B1_source"]["explore_no_definition"]
    assert out["RepExNoDefTThreeHoldTwelve"] == pn.ci(nd["minus_oracle"], pn.pp)
    assert out["RepExNoDefTThreeResultHoldOne"] == "fail"
    for h, htag in paper_data.HORIZON_TAGS:
        for system, stag in paper_data.SYSTEM_TAGS:
            orig_nd = ORIG["e3"]["horizons"][h][system]["explore_no_definition"]
            assert out[f"RepExNoDefTThreeResult{stag}{htag}"] == pn._pass(orig_nd["T3_pass"])
    assert out["RepTThreePassAll"] == "3 of 9" and out["RepExNoDefTThreePassAll"] == "1 of 9"
    oracle_nd = per["oracle"]["explore_no_definition"]["contract_valid"]
    assert out["RepExNoDefCvOracleTwelve"] == pn.ci(oracle_nd)
    assert set(pn.rep_e3_numbers(None, None).values()) == {MISSING}
    only_rep = pn.rep_e3_numbers(None, rep)
    assert only_rep["RepZeroMinusHoldOne"] != MISSING
    no_verdicts(out)


def test_mem0_discriminability_macros_from_stage_3():
    m = _load("stage3_e5", "e5_natives.json")
    out = pn.e5_numbers(m)
    branches = m["systems"]["mem0"]["branches"]
    flags = [d for br in branches.values() for d in br["discrimination"].values()]
    assert len(flags) == 6
    assert out["EFiveExMemComparisons"] == "6"
    assert out["EFiveExMemInsidePredicted"] == str(sum(d["inside_predicted"] for d in flags)) == "5"
    assert out["EFiveExMemInsideOther"] == str(sum(d["inside_other"] for d in flags)) == "0"
    assert out["EFiveExMemDiscriminating"] == str(sum(d["discriminates"] for d in flags)) == "5"
    assert out["EFiveExMemDiscriminatingOf"] == "5 of 6"
    assert out["EFiveExMemOtherReference"] == r"\holdmem{}(0,1), source epoch"
    # the one comparison that does not discriminate is the one outside the predicted interval
    assert out["EFiveExMemUpdateDiscriminatesSrc"] == "no"
    assert out["EFiveMemUpdateConsistentSrc"] == "outside"
    assert out["EFiveExMemStabilityDiscriminatesSrc"] == "yes"
    assert out["EFiveExMemUpdateInsideOtherSrc"] == "outside"
    assert out["EFiveExMemUpdateOtherSrc"] == pn.ci(branches["src"]["white_box_other"]["Update"])
    assert out["EFiveExMemStabilityOtherOrphan"] == pn.ci(
        branches["orphan"]["white_box_other"]["Stability"]
    )
    missing = pn.e5_numbers({})
    assert missing["EFiveExMemDiscriminating"] == MISSING
    assert missing["EFiveExMemUpdateDiscriminatesSrc"] == MISSING
    assert missing["EFiveExMemOtherReference"] == MISSING


# ------------------------------------------------------------------------------------- block R
def test_replication_view_partial_stage_keeps_what_needs_only_e1_and_e3():
    view = paper_data.replication_view(ORIG, REP_PARTIAL)
    assert set(view["paths"]) == {"path1", "path2"} and "claim_level" not in view
    assert view["t1"] == {} and len(view["t3"]) == 9 and len(view["t4"]) == 9
    assert view["n_worlds"] == {"orig": 20, "rep": 20}
    out = pn.rep_numbers(view)
    p1 = view["paths"]["path1"]["estimate"]
    assert out["RepPathOneEstimate"] == pn.ci(p1, pn.pp) and out["RepPathTwoMaxAbs"] == pn.pp(0.0)
    assert out["RepPathThreeSlope"] == MISSING and out["RepTOneSame"] == MISSING
    # T3 and T4 have the main study's outcome in every unit
    for unit in ("t3", "t4"):
        assert all(v["rep"] == v["orig"] for v in view[unit].values()), unit
    assert out["RepTThreeResultHoldOne"] == "pass" and out["RepTThreeResultRagOne"] == "fail"
    no_verdicts(out)


def test_replication_macros_give_both_estimates_of_every_contrast():
    view = paper_data.replication_view(ORIG, REP_PARTIAL)
    out = pn.rep_numbers(view)
    rows = replication.contrast_rows(ORIG["e1"], REP_PARTIAL["e1"])
    effects = [r for r in rows if r["kind"] == "effect"]
    assert out["RepEffects"] == str(len(effects)) and out["RepEffectsSource"] == "21"
    for r in rows:
        name = f"Rep{paper_data.camel(r['contrast'])}{dict(paper_data.BRANCH_TAGS)[r['branch']]}"
        assert out[name] == pn.ci(r["rep"], pn.pp) and out[f"{name}Orig"] == pn.ci(r["orig"], pn.pp)
    assert "RepDEffectFalseInvalSourceSrc" in out and "RepDEffectMissedSourceDefOrig" in out
    assert out["RepWorlds"] == out["RepWorldsOrig"] == "20"
    no_verdicts(out)


def test_self_replication_with_all_three_stages_gives_every_quantity():
    view = paper_data.replication_view(ORIG, ORIG)
    out = pn.rep_numbers(view)
    assert out["RepTOneSame"] == pn.ci(ORIG["e2"]["t1"]["same"]["delta_90"], pn.pp)
    assert out["RepTOnePassSame"] == "within"
    assert out["RepPathThreeSlope"].startswith("1.01 (90\\% lower bound 0.90)")
    assert MISSING not in [v for k, v in out.items() if k.startswith(("RepPath", "RepTOne"))]
    no_verdicts(out)


def test_a_flipped_effect_shows_in_the_fresh_estimate():
    rep = copy.deepcopy(ORIG)
    flipped = rep["e1"]["contrasts"]["src"]["B_effect_update_pooled_source"]
    flipped.update(est=-0.01, lo=-0.03, hi=0.01)
    view = paper_data.replication_view(ORIG, rep)
    out = pn.rep_numbers(view)
    assert out["RepBEffectUpdatePooledSourceSrc"] == pn.ci(flipped, pn.pp)
    assert out["RepPathOneEstimate"] == pn.ci(flipped, pn.pp)  # binding's effect pooled over D
    tab = pt.tab_replication(view)
    assert pt.mt_ci_cells(flipped, sign=True) in tab and r"\fail" not in tab


def test_pooled_forty_worlds_macros_and_partial_stage():
    pooled = paper_data.pooled_view(ORIG)  # any stage set with the three analyses will do
    out = pn.rep_pool_numbers(pooled)
    est = ORIG["e1"]["contrasts"]["src"]["B_effect_update_pooled_source"]
    assert out["RepPoolBEffectUpdatePooledSourceSrc"] == pn.ci(est, pn.pp)
    assert out["RepPoolPathOneEstimate"] == pn.ci(est, pn.pp)
    assert out["RepPoolTOneSame"] == pn.ci(ORIG["e2"]["t1"]["same"]["delta_90"], pn.pp)
    assert out["RepPoolTThreeResultHoldTwelve"] == "pass"
    only_e1 = pn.rep_pool_numbers(
        paper_data.pooled_view({"e1": ORIG["e1"], "e2": None, "e3": None})
    )
    assert only_e1["RepPoolPathOneEstimate"] == pn.ci(est, pn.pp)
    assert only_e1["RepPoolTOneSame"] == MISSING and only_e1["RepPoolTThreeHoldOne"] == MISSING
    no_verdicts(out)


def test_replication_direction_counts_come_from_the_estimates():
    out = pn.rep_numbers(paper_data.replication_view(ORIG, REP_PARTIAL))
    # 42 contrasts expected to be non-zero; 37 intervals exclude 0 in either stage, 36 of them in
    # the expected (positive) direction and one, the orphan cited B x D, the other way
    assert out["RepDirectional"] == out["RepOrigDirectional"] == "42"
    assert (out["RepOrigExcludesZero"], out["RepExcludesZero"]) == ("37", "37")
    assert (out["RepOrigExcludesPredicted"], out["RepExcludesPredicted"]) == ("36", "36")
    assert out["RepOrigExcludesOpposite"] == out["RepExcludesOpposite"] == "1"
    assert (out["RepOrigEstimatePredicted"], out["RepEstimatePredicted"]) == ("37", "38")
    assert out["RepDirectionalSource"] == out["RepDirectionalCited"] == "21"
    assert out["RepOrigExcludesPredictedSource"] == "21"  # every source-epoch contrast
    view = paper_data.multiplicity_view(MULT_RESULT, SOURCES)
    assert out["RepOrigExcludesPredicted"] == pn.mult_numbers(view)["MultExOrigInterval"]
    assert out["RepExcludesPredicted"] == pn.mult_numbers(view)["MultExRepInterval"]
    assert int(out["RepExcludesPredicted"]) + int(out["RepExcludesOpposite"]) == int(
        out["RepExcludesZero"]
    )
    # counts partition the family, and every one is missing without the contrasts
    empty = pn.rep_numbers(paper_data.replication_view(NONE3, NONE3))
    assert {v for k, v in empty.items() if k.startswith(("RepDirectional", "RepExcludes"))} == {
        MISSING
    }


def test_replication_table_gives_both_estimates_with_their_intervals():
    view = paper_data.replication_view(ORIG, REP_PARTIAL)
    tab = pt.tab_replication(view)
    assert "agrees" not in tab and "as pred." not in tab and "repl." not in tab
    gap = r"@{\hspace{14pt}}"  # between the shifts' column pairs
    assert rf"\begin{{mttabular}}{{@{{}}ll r@{{\,}}l {gap} r@{{\,}}l {gap} r@{{\,}}l@{{}}}}" in tab
    assert r"\mtgroup{8}{Source key}" in tab and r"\mtgroup{8}{Citation key}" in tab
    lines = [ln.removesuffix(r" \\").split(" & ") for ln in _first_body(tab).splitlines()]
    rows = [c for c in lines if len(c) == 8 and c[1] in ("main study", "replication")]
    # the detection effects at D=0 and D=1 print alike (one row), and D's effect on missed
    # invalidation prints 0.0 in every cell (a line of the notes): both keys, both studies
    shown = [c[0] for c in rows if c[0]]
    assert r"\mtin $B$ effect, detection (either $D$)" in shown
    assert not any("missed" in lab or "{=}0$)" in lab for lab in shown)
    # both studies per contrast; the source key leaves two contrasts to the class split
    shown_rows = 2 * (len(pt.REP_ROWS) - 2)
    assert len(rows) == 2 * shown_rows - 2 * len(pt.IN_CLASSSPLIT)
    assert "$D$'s effect on missed invalidation was 0.0 with both keys" in tab
    assert r"\cref{tab:classsplit}" not in tab
    source = _first_body(tab).split(r"\mtgroup{8}{Citation key}")[0]
    assert r"$B{\times}D$ interaction" not in source and "Update ($D{=}1$)" not in source
    label = r"\mtin $B{\times}D$ interaction, Update"
    for g, k in zip(("cited",), [i for i, c in enumerate(rows) if c[0] == label],
                    strict=True):  # fmt: skip
        main, fresh = rows[k], rows[k + 1]
        assert (main[1], fresh[0], fresh[1]) == ("main study", "", "replication")
        for j, b in zip((2, 4, 6), ("src", "def", "orphan"), strict=True):
            row = view["rows"][b, f"BxD_interaction_update_{g}"]
            # every estimate with its interval, in two cells
            assert " & ".join(main[j : j + 2]) == pt.mt_ci_cells(row["orig"], sign=True)
            assert " & ".join(fresh[j : j + 2]) == pt.mt_ci_cells(row["rep"], sign=True)


def test_replication_table_without_marks_and_partial_stage():
    full = pt.tab_replication(paper_data.replication_view(ORIG, REP_PARTIAL))
    assert r"\pass" not in full and r"$\sim$" not in full and r"\fail" not in full
    assert "slope of writer agreement" not in full  # the design quantities are in the text
    assert "same outcome" not in full and "diagnostic tests" not in full
    assert "regist" not in full and "triggered" not in full and "prediction" not in full
    assert "path~" not in full and "block R" not in full and "App.~" not in full
    assert MISSING not in _first_body(full)  # the contrast rows need only E1
    empty = pt.tab_replication({})
    rows = 2 * len(pt.REP_ROWS) - len(pt.IN_CLASSSPLIT)  # both keys; the source key's in Table 20
    assert _first_body(empty).count(MISSING) == 2 * rows * 3


# ------------------------------------------------------------------------------------- block C
C_E1 = _load("stage4_c_e1", "e1_metrics.json")


def test_construct_macros():
    out = pn.synth_numbers(C_E1)
    con = C_E1["contrasts"]
    assert out["SynthCOneSrc"] == pn.ci(con["src"]["C1_repair_D1s_minus_D0_source"], pn.pp)
    assert out["SynthCTwoShareDef"] == pn.ci(con["def"]["C2_share_of_D_effect_source"], pn.pct)
    for name, key in (
        ("COne", "C1_repair_D1s_minus_D0_source"),
        ("CTwo", "C2_repair_D1_minus_D1s_source"),
        ("CThree", "C3_BxD_interaction_update_D1s_source"),
        ("CFour", "C4_D1s_effect_missed_source"),
    ):
        for b, btag in paper_data.BRANCH_TAGS[:2]:
            assert out[f"Synth{name}{btag}"] == pn.ci(con[b][key], pn.pp)
    assert "SynthCOneOrphan" not in out and "SynthExCOneOrphan" in out  # the orphan branch: Ex
    no_verdicts(out)
    assert out["SynthExCOneCitedSrc"] == pn.ci(
        con["src"]["explore_C1_repair_D1s_minus_D0_cited"], pn.pp
    )
    assert out["SynthWorlds"] == "20"
    for cls, cname in pn.CLASS_NAMES:  # C1 and C2 by item class
        for b, btag in paper_data.BRANCH_TAGS:
            c = con[b]
            assert out[f"SynthExCOne{cname}{btag}"] == pn.ci(
                c[f"explore_C1_repair_D1s_minus_D0_source_{cls}"], pn.pp
            )
            assert out[f"SynthExCTwo{cname}{btag}"] == pn.ci(
                c[f"explore_C2_repair_D1_minus_D1s_source_{cls}"], pn.pp
            )
    assert out["SynthExCItemsDefinitionSrc"] == "387" and out["SynthExCItemsRuleOrphan"] == "2,662"
    sp = C_E1["synth_programs"]["src"]["holdmem_D1sB1_source"]
    assert out["SynthExecAll"] == pn.ci(sp["all"]["exec_t0"])
    assert out["SynthAgreeKappaVipSender"] == pn.ci(sp["kappa.vip_sender"]["agree_t0"])
    assert out["SynthAgreeExecKappaWorkingHours"] == "n/a"  # no executable program
    assert out["SynthItemsAll"] == "960"
    assert out["SynthItemsGroupDefinition"] == "120"
    group = [sp[t] for t in sp if t.startswith("kappa.")]
    want = sum(e["exec_t0"]["est"] * e["n"] for e in group) / sum(e["n"] for e in group)
    assert out["SynthExecGroupDefinition"] == pn.pct(want)
    assert set(paper_data.ITEM_TYPES) == set(sp) - {"all"}  # the static list is the data's


# ------------------------------------------------------------------------------------ block M1
CROSS = (
    "meta-llama/Llama-3.1-8B-Instruct",
    "google/gemma-4-12B-it",
    "openai/gpt-oss-20b",
    "mistralai/Ministral-3-14B-Instruct-2512-BF16",
    "Qwen/Qwen3-8B",
)


def _m1(tmp_path: Path, tiers: dict[str, int]) -> Any:
    """The E2 analysis of synthetic survival rows: tier -> one D=1 item lost in `lost_every`."""
    rows = [r for tier, lost in tiers.items() for r in e2_rows(tier, lost)]
    (tmp_path / "survival.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return _roundtrip(e2.analyse(tmp_path))


def _eight(tmp_path: Path) -> Any:
    tiers = {"same": 0, "upgrade": 5, "downgrade": 2}
    tiers |= {tier_name(f"cross:{m}"): 0 for m in CROSS}
    return _m1(tmp_path, tiers)


def test_m1_macros_per_switch_and_pooled(tmp_path):
    m = _eight(tmp_path)
    out = pn.m1_numbers(m)
    assert out["MOneStaged"] == out["MOneSwitches"] == "8"
    assert out["MOnePathThreeN"] == f"{8 * 10 * 20:,}"  # eight tiers, ten worlds, twenty items
    # D=1 loses nothing in six tiers (survival 1 vs 0.75), one item in five in `upgrade` (0.8),
    # one in two in `downgrade` (0.5 < 0.75: D does not protect this switch)
    mb = m["m1b_d_protects_switch"]
    for tier in paper_data.TIERS:
        key = paper_data.tier_key(m, tier)
        assert out[f"MOneMB{tier.tag}"] == pn.ci(mb[key]["d1_minus_d0_survival"], pn.pp)
    assert mb["downgrade"]["d1_minus_d0_survival"]["est"] < 0
    assert out["MOneMBHolding"] == "7 of 8"
    assert out["MOneTOnePassSame"] == "within" and out["MOneTOnePassUp"] == "outside"
    assert out["MOneTOneHolding"] == "6 of 8"
    assert out["MOneTOneQwenPrev"] == pn.ci(
        m["t1"][tier_name("cross:Qwen/Qwen3-8B")]["delta_90"], pn.pp
    )
    assert out["MOneCaveats"] == "none"
    no_verdicts(out)
    caveat = tier_name("cross:meta-llama/Llama-3.1-8B-Instruct")  # its underscores are escaped
    m["path3"]["tier_caveats"] = ["downgrade", caveat, "cross_unknown_model"]
    assert pn.m1_numbers(m)["MOneCaveats"] == (
        r"the switch down from Qwen3.8-27B, the switch to Llama-3.1-8B, and cross\_unknown\_model"
    )
    assert out["MOneSlopeGemma"] != MISSING and out["MOneAurocGemma"] != MISSING
    tab = pt.tab_families(m, {})
    assert "Ministral-3-14B" in tab and "pooled, 8 switches" in tab
    assert r"\pass" not in tab and r"\fail" not in tab and "triggered" not in tab
    down = next(ln for ln in tab.splitlines() if ln.startswith("downgrade"))
    surv = m["m1b_d_protects_switch"]["downgrade"]["d1_minus_d0_survival"]
    assert pt.mt_ci_cells(surv, sign=True) in down  # the survival difference, interval


def test_m1_partial_stage_counts_only_when_all_eight_are_staged(tmp_path):
    m = _m1(tmp_path, {"same": 0, "upgrade": 5, "downgrade": 0})
    out = pn.m1_numbers(m)
    assert out["MOneStaged"] == "3" and out["MOneSwitches"] == "3"
    assert out["MOneSlopeLlama"] == MISSING and out["MOneMBGemma"] == MISSING
    assert out["MOneMBHolding"] == out["MOneTOneHolding"] == MISSING
    assert out["MOneMBSame"] != MISSING  # the staged switches still render
    tab = pt.tab_families(m, {})
    assert "same tier" in tab and MISSING in tab


# ------------------------------------------------------------------------------------ block M2
STAGE2_E1 = ORIG["e1"]


def _family(zero_cv: int = 0, break_pattern: bool = False) -> Any:
    e1 = copy.deepcopy(STAGE2_E1)
    if break_pattern:
        e1["contrasts"]["def"]["BxD_interaction_update_source"].update(est=0.01, lo=-0.02)
    return _roundtrip(families.family_checks(e1, family_items(zero_cv, 1)))


def test_m2_macros_estimates_and_family_count():
    fams = {"llama": _family(), "gemma": _family(zero_cv=1), "gptoss": _family(break_pattern=True)}
    out = pn.m2_numbers(fams)
    for fam in FAMILIES:
        for key, name in paper_data.FAMILY_ESTIMATES:
            want = pn.ci(fams[fam.key]["estimates"][key], pn.pp)
            assert out[f"MTwo{fam.tag}{name}"] == want, (fam.key, key)
    assert out["MTwoGptOssBxDDef"] != out["MTwoLlamaBxDDef"]  # the broken pattern
    assert out["MTwoStaged"] == "3"
    no_verdicts(out)
    assert out["MTwoLlamaWorlds"] == "20"


def test_m2_count_needs_all_three_families():
    out = pn.m2_numbers({"llama": _family(), "gemma": None, "gptoss": None})
    assert out["MTwoStaged"] == "1"
    assert out["MTwoLlamaBxDDef"] != MISSING and out["MTwoGemmaBxDDef"] == MISSING


def test_load_families_reads_the_saved_json_then_the_item_table(tmp_path):
    stage = tmp_path / "stage4_m2_x"
    (stage / "analysis").mkdir(parents=True)
    assert paper_data.load_families(tmp_path, "stage4_m2_x") is None  # nothing staged
    (stage / "analysis" / "e1_metrics.json").write_text(json.dumps(STAGE2_E1))
    assert paper_data.load_families(tmp_path, "stage4_m2_x") is None  # no item table either
    items = family_items(0, 1)
    (stage / "analysis" / "e1_item_table.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in items)
    )
    computed = paper_data.load_families(tmp_path, "stage4_m2_x")
    assert computed and computed["ii_to_iv_true"] and all(computed["checks"].values())
    res = families.analyse(stage, {stage.name: stage})  # what `analysis.stage m2` writes
    saved = json.loads((stage / "analysis" / "families.json").read_text())
    assert (
        saved["families_ii_to_iv_true"] == 1 and saved[stage.name]["checks"] == computed["checks"]
    )
    assert res["families_ii_to_iv_true"] == 1
    assert paper_data.load_families(tmp_path, "stage4_m2_x") == computed


def test_stage_m2_writes_e1_and_families_json(tmp_path, monkeypatch):
    from memshift.analysis import stage
    from memshift.experiments.e1 import run_world
    from memshift.holdmem.system import CellConfig
    from tests.fakes import FakeLLM

    cells = [CellConfig(d, b) for d in (0, 1) for b in (0, 1)]
    run = tmp_path / "run"
    run.mkdir()
    run_world(5, run, branches=("src", "def"), baselines=True, writer=FakeLLM(), cells_cfg=cells)
    out = tmp_path / "stage4_m2_x"
    monkeypatch.setattr("sys.argv", ["stage", "m2", str(out), str(run)])
    stage.main()
    assert (out / "analysis" / "e1_metrics.json").exists()
    saved = json.loads((out / "analysis" / "families.json").read_text())
    assert saved[out.name]["n_worlds"] == 1 and "families_ii_to_iv_true" in saved
    manifest = json.loads((out / "manifest.json").read_text())
    assert "analysis/families.json" in manifest["files_sha256"]
    assert paper_data.load_families(tmp_path, out.name) == saved[out.name]


# ------------------------------------------------------------------------------------ block M3
THETA_R = _load("stage3_e7", "e7_metrics.json")["theta_r"]  # a real swap (5 worlds)


def test_m3_macros_from_a_real_reader_swap():
    out = pn.m3_numbers({"llama": THETA_R})
    assert out["MThreeLlamaWorlds"] == "5"
    for b, btag in paper_data.BRANCH_TAGS[:2]:
        for key, name in pn.M3_CONTRASTS:
            c = THETA_R[b]["contrasts"][key]
            assert out[f"MThreeLlama{name}Main{btag}"] == pn.ci(c["main"], pn.pp)
            assert out[f"MThreeLlama{name}Swap{btag}"] == pn.ci(c["swap"], pn.pp)
    assert out["MThreeLlamaCvHoldSwapDef"] == pn.ci(
        THETA_R["def"]["systems"]["holdmem_D1B1_source"]["swap"]["ContractValid_post"]
    )
    assert out["MThreeGemmaBxDMainSrc"] == MISSING
    assert out["MThreeStaged"] == "1"
    no_verdicts(out)


def test_m3_flipped_sign_and_the_reader_table():
    flipped = copy.deepcopy(THETA_R)
    flipped["def"]["contrasts"]["BxD_interaction_update_source"]["swap"]["est"] = -0.05
    out = pn.m3_numbers({"llama": THETA_R, "gemma": flipped, "gptoss": THETA_R})
    swap = flipped["def"]["contrasts"]["BxD_interaction_update_source"]["swap"]
    assert out["MThreeGemmaBxDSwapDef"] == pn.ci(swap, pn.pp)
    assert out["MThreeStaged"] == "3"
    tab = pt.tab_families(None, {"llama": THETA_R})
    missed = THETA_R["src"]["contrasts"]["D_effect_missed_source"]["swap"]
    assert "6/6" not in tab and "triggered" not in tab
    assert pt.mt_ci_cells(missed, sign=True).split(" & ")[0] in tab  # 0.0: no interval
    # one row per quantity: the swapped reader of each family, then the main reader
    readers = tab.split("As reader only", 1)[1].splitlines()
    assert readers[0].split(" & ")[1:] == [
        *(rf"\multicolumn{{2}}{{c}}{{{f.label}}}" for f in FAMILIES),
        r"\multicolumn{2}{c@{}}{Qwen3.5-9B} \\",
    ]
    missing = rf"\multicolumn{{2}}{{c}}{{{MISSING}}}"
    for label, key, b in pt.READER_ROWS:
        c = THETA_R[b]["contrasts"][key]
        line = next(ln for ln in readers if ln.startswith(label + " &")).removesuffix(r" \\")
        swap, rest = line.split(" & ", 1)[1].split(f" & {missing} & {missing} & ")
        assert swap == pt.mt_ci_cells(c["swap"], sign=True), label  # Llama, then not staged
        assert rest == pt.mt_ci_cells(c["main"], sign=True), label  # the main reader last
    labels = [lab for lab, _, _ in pt.READER_ROWS]
    for quantity in ("$B$ effect on Update", "$D$ effect on accuracy on re-derived items"):
        assert {f"{quantity}, $\\Ssrc$", f"{quantity}, $\\Sdef$"} <= set(labels), quantity
    cv = next(ln for ln in readers if ln.startswith(r"\holdmem{}(1,1) accuracy"))
    hold = THETA_R["def"]["systems"]["holdmem_D1B1_source"]
    cells = cv.removesuffix(r" \\").split(" & ")
    assert [cells[1], cells[-2]] == [
        pt._est(hold[arm]["ContractValid_post"]) for arm in ("swap", "main")
    ]


def test_reader_groups_from_run_manifests_and_load_readers(tmp_path: Path):
    def run(name: str, model: str | None) -> Path:
        d = tmp_path / name
        d.mkdir()
        if model:
            (d / "manifest.json").write_text(json.dumps({"models": {"reader": {"model": model}}}))
        return d

    llama, gemma = "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-4-12B-it"
    runs = [run("a", llama), run("b", gemma), run("c", llama)]
    groups = e7.reader_groups(runs)
    assert list(groups) == sorted([llama, gemma]) and groups[llama] == [runs[0], runs[2]]
    assert e7.reader_groups([*runs, run("d", None)]) == {}  # a run without a record: no grouping
    swaps = {llama: THETA_R, gemma: {"worlds": ["w1"]}, "some/other-model": THETA_R}
    readers = paper_data.load_readers({"theta_r_by_reader": swaps})
    assert set(readers) == {"llama", "gemma"} and readers["llama"] is THETA_R
    assert paper_data.load_readers(None) == {}


def test_e7_stage_without_e7_runs_keeps_the_readers_apart(tmp_path: Path):
    """`analysis.stage e7 runs/stage4_m3 e1:RUN theta_r:RUN ...` with runs of two readers."""
    from memshift.experiments.e1 import run_world
    from memshift.holdmem.system import CellConfig
    from tests.fakes import FakeLLM

    cells = [CellConfig(d, b) for d in (0, 1) for b in (0, 1)]
    llama, gemma = "meta-llama/Llama-3.1-8B-Instruct", "google/gemma-4-12B-it"
    runs = {}
    for name, seed, model in (("main", 5, None), ("llama", 5, llama), ("gemma", 5, gemma)):
        runs[name] = tmp_path / name
        runs[name].mkdir()
        run_world(seed, runs[name], branches=("src",), baselines=False, writer=FakeLLM(),
                  cells_cfg=cells)  # fmt: skip
        if model:
            manifest = {"models": {"reader": {"model": model}}}
            (runs[name] / "manifest.json").write_text(json.dumps(manifest))
    out = tmp_path / "stage4_m3"
    res = e7.analyse([], [runs["main"]], out, [runs["llama"], runs["gemma"]])
    assert list(res["theta_r_by_reader"]) == sorted([llama, gemma])
    saved = json.loads((out / "analysis" / "e7_metrics.json").read_text())
    readers = paper_data.load_readers(saved)
    assert set(readers) == {"llama", "gemma"} and readers["llama"]["worlds"] == ["w0005"]
    assert pn.m3_numbers(readers)["MThreeStaged"] == "2"


# ------------------------------------------------------------------------------------- block N
ORDER = [(b, m) for b in ("src", "def", "orphan") for m in ("Update", "Stability")]
DOCUMENTED = e5.DOCUMENTED_LABELS
SYSTEMS = ("mem0", "amem", "memos", "lightrag", "hipporag", "cognee")
COST = {
    s: {
        "t_i": {
            "llm_calls": 1.0 + i,
            "prompt_tokens": 10000.0 / (i + 1),
            "completion_tokens": 200.0,
        },
        "attempted": 600 + i,
        "wall_s_per_unit": 30.0 + i,
    }
    for i, s in enumerate(SYSTEMS)
}


def _natives(inferred: dict[str, tuple[int, int]], inside: dict[str, list[bool]]) -> Any:
    """An E5 analysis of the systems in `inferred`: the committed Mem0 entry cloned per system
    with chosen labels and consistency flags, and the real stage-4 summary computed over it."""
    base = _load("stage3_e5", "e5_natives.json")
    res = {"labels": {}, "systems": {}, "failures": {}, "noise_floor": base["noise_floor"]}
    res["membership"] = {}
    for s, (d, b) in inferred.items():
        per = copy.deepcopy(base["systems"]["mem0"])
        per["predicted_by"] = e5.predicted_cell(s, d, b)
        per["other_label_cell"] = e5.other_cell(s, d, b)
        for (br, m), ok in zip(ORDER, inside[s], strict=True):
            per["branches"][br]["consistent"][m] = ok
            d = per["branches"][br]["discrimination"][m]  # cloned from Mem0: keep it coherent
            d["inside_predicted"], d["discriminates"] = ok, ok and not d["inside_other"]
        res["systems"][s] = per
        res["labels"][s] = {
            "B": {"label": b, "rate": 0.3 + 0.1 * b, "n": 90},
            "D": {"label": None, "rate": 1.0, "n": 12},
        }
        res["failures"][s] = {"ingested": 990, "failed": 10, "rate": 0.01}
        res["membership"][s] = base["membership"]["mem0"]
    res["stage4"] = e5.stage4_summary(res)
    return _roundtrip(res)


ALL_SIX = _natives(
    {s: DOCUMENTED[s] for s in SYSTEMS},  # every inferred label as documented
    {s: [True] * 6 for s in SYSTEMS} | {"lightrag": [True, False, True, True, False, True]},
)


def test_native_macros_for_all_six_systems():
    out = pn.nat_numbers(ALL_SIX, COST)
    assert out["NatLightRagLabel"] == "(0,1)" and out["NatLightRagLabelAgrees"] == "yes"
    assert out["NatLightRagReference"] == r"\holdmem{}(0,1), cited cluster"
    assert out["NatMemReference"] == r"\holdmem{}(0,0)"
    assert out["NatLightRagInsideOf"] == "4 of 6"
    assert out["NatAmemInsideOf"] == "6 of 6"
    assert out["NatLightRagUpdateConsistentDef"] == "inside"
    assert out["NatLightRagStabilityConsistentSrc"] == "outside"
    assert out["NatStaged"] == "6"
    assert out["NatNOneAgree"] == "5 of 5"
    assert out["NatNThreeInside"] == "34 of 36"
    assert out["NatNThreeShare"] == pn.pct(34 / 36)
    assert out["NatNFourStabilityBOne"] != MISSING  # B=0 and B=1 systems both present
    assert out["NatNFourStabilityBZero"] != MISSING
    no_verdicts(out)
    assert out["NatMemFailRate"] == "1.00\\%" and out["NatMemUnits"] == "1,000"
    assert out["NatMemTThreeResult"] == "fail" and out["NatMemTFourResult"] == "pass"
    # cost per event, and its token ratio to Mem0 (10,200 tokens per event)
    assert out["NatMemCostCalls"] == "1.00" and out["NatMemCostRatio"] == "1.00"
    assert out["NatAmemCostPrompt"] == "5.0k" and out["NatAmemCostTokens"] == "5.2k"
    assert out["NatAmemCostRatio"] == pn.num(5200 / 10200, 2)


def test_native_counts_need_every_system_and_disagreeing_labels_count():
    inside = {s: [True] * 6 for s in SYSTEMS}
    three = {s: DOCUMENTED[s] for s in ("mem0", "amem", "lightrag")}
    partial = _natives(three, inside)
    out = pn.nat_numbers(partial, COST)
    assert out["NatStaged"] == "3" and out["NatAmemInsideOf"] == "6 of 6"
    for k in ("NatNOneAgree", "NatNThreeInside", "NatNThreeShare", "NatNFourStabilityBOne"):
        assert out[k] == MISSING
    assert out["NatMemosLabel"] == MISSING and out["NatMemosDocumented"] == "(0,0)"
    # two systems whose inferred label differs from the documented one: 3 of 5 agree
    wrong = {s: DOCUMENTED[s] for s in SYSTEMS} | {"amem": (0, 1), "memos": (0, 1)}
    res = _natives(wrong, {s: [True] * 6 for s in SYSTEMS} | {"amem": [False] * 6})
    out = pn.nat_numbers(res, COST)
    assert out["NatAmemLabelAgrees"] == "no" and out["NatNOneAgree"] == "3 of 5"
    assert out["NatAmemInsideOf"] == "0 of 6" and out["NatNThreeInside"] == "30 of 36"
    assert out["NatAmemReference"] == r"\holdmem{}(0,1), source epoch"


def test_native_discriminability_macros_and_totals_need_every_system():
    out = pn.nat_numbers(ALL_SIX, COST)
    disc = ALL_SIX["stage4"]["discriminability"]
    for nat in paper_data.NATIVES:
        d = disc["per_system"][nat.key]
        assert out[f"Nat{nat.tag}ExDiscriminating"] == str(d["discriminating"])
        assert out[f"Nat{nat.tag}ExInsideOther"] == str(d["inside_other"])
        assert out[f"Nat{nat.tag}ExDiscriminatingOf"] == f"{d['discriminating']} of 6"
    # a comparison discriminates only if it is inside the predicted interval: the two comparisons
    # outside it (LightRAG's) are not counted, and none lies inside the other label's interval
    assert out["NatExComparisons"] == "36" and out["NatExInsidePredicted"] == "34"
    assert out["NatExDiscriminating"] == "34" and out["NatExDiscriminatingOf"] == "34 of 36"
    assert out["NatExInsideOther"] == "0"
    assert out["NatLightRagExDiscriminating"] == "4"
    assert out["NatLightRagExOtherReference"] == r"\holdmem{}(0,0)"  # the other B label's cell
    assert out["NatAmemExOtherReference"] == r"\holdmem{}(0,1), source epoch"
    assert out["NatLightRagExStabilityDiscriminatesSrc"] == "no"  # outside the predicted interval
    # totals only once all six systems are staged
    inside = {s: [True] * 6 for s in SYSTEMS}
    two = {s: DOCUMENTED[s] for s in ("mem0", "amem")}
    partial = pn.nat_numbers(_natives(two, inside), COST)
    assert partial["NatExDiscriminating"] == partial["NatExDiscriminatingOf"] == MISSING
    assert partial["NatAmemExDiscriminating"] == "6"
    assert partial["NatMemosExDiscriminating"] == MISSING
    assert pn.nat_numbers(None, {})["NatMemExDiscriminating"] == MISSING


def test_native_results_falls_back_to_stage3_for_mem0_only():
    s3 = {"systems": {"mem0": {}}}
    assert paper_data.native_results({"n": None, "n3": s3}) is s3
    assert paper_data.native_results({"n": ALL_SIX, "n3": s3}) is ALL_SIX
    assert paper_data.native_results({"n": None, "n3": None}) is None


def test_merge_costs_prefers_the_later_analysis():
    a = {"by_system": {"mem0": {"attempted": 1}, "graphiti": {"attempted": 2}}}
    b = {"by_system": {"amem": {"attempted": 3}, "mem0": {"attempted": 4}}}
    merged = paper_data.merge_costs(a, None, b)
    assert {k: v["attempted"] for k, v in merged.items()} == {"mem0": 4, "graphiti": 2, "amem": 3}


# ------------------------------------------------------------------------------------- block L
RENDERING = {
    "source": "world_summaries.jsonl",
    "models": ["google/gemma-4-12B-it"],
    "requests": 1200,
    "total": {
        "attempted": 1000,
        "retried": 50,
        "retried_share": 0.05,
        "kept_template": 20,
        "kept_template_share": 0.02,
        "skipped_background": 300,
    },
    "fidelity": {"checked": 5000, "min_rate": 1.0, "all_one": True},
    "per_source": {
        "email": {"attempted": 400, "retried_share": 0.04, "kept_template_share": 0.01},
        "messages": {"attempted": 400, "retried_share": 0.06, "kept_template_share": 0.03},
        "notes": {"attempted": 200, "retried_share": 0.05, "kept_template_share": 0.02},
    },
}


def _rendered(tmp_path: Path) -> Any:
    res = rendered.analyse(
        rendered_stage(tmp_path / "r", rendered_table(RENDERED)),
        rendered_stage(tmp_path / "t", rendered_table(TEMPLATE)),
        tmp_path / "o",
    )
    return _roundtrip(res) | {"rendering": RENDERING}


def test_rendered_macros_checks_differences_and_descriptives(tmp_path):
    m = _rendered(tmp_path)
    out = pn.rend_numbers(m)
    assert out["RendWorlds"] == "3" and out["RendProbesIdentical"] == "yes"
    # the fixture: HoldMem(1,1) repairs half of the items in the rendered worlds, all in template;
    # the D effect on repair is 0.5 - 0 in the rendered worlds, with an interval over three worlds
    # that reaches 0, and 1 in the template worlds
    assert out["RendDiffDRepairSrc"] == pn.ci(m["paired"][1]["diff"], pn.pp)
    assert out["RendDRepairSrc"] == pn.ci(m["rendered"]["estimates"]["D_repair_src"], pn.pp)
    assert m["rendered"]["estimates"]["D_repair_src"]["lo"] <= 0
    assert out["RendTemplateDRepairSrc"].startswith("$+$100.0")
    assert out["RendDMissedSourceSrc"].startswith("0.0")
    no_verdicts(out)
    assert out["RendCvDiffHoldDef"] != MISSING
    assert out["RendKeptTemplate"] == "2.0\\%" and out["RendRetriedEmail"] == "4.0\\%"
    assert out["RendFidelityMin"] == "100.0\\%" and out["RendFidelityAllOne"] == "yes"
    assert out["RendFidelityChecked"] == "5,000" and out["RendModel"] == "google/gemma-4-12B-it"


def test_rendered_without_render_reports_leaves_the_descriptives_missing(tmp_path):
    m = _rendered(tmp_path) | {"rendering": {"source": None, "note": "no reports"}}
    out = pn.rend_numbers(m)
    assert out["RendKeptTemplate"] == MISSING and out["RendFidelityMin"] == MISSING
    assert out["RendKeptEmail"] == MISSING and out["RendModel"] == MISSING
    assert out["RendDRepairSrc"] != MISSING  # the estimates do not depend on the reports


# --------------------------------------------------------------------------------- section 6
def test_rank_macros_from_the_real_correlations():
    rows = rank_table({w: LEVELS for w in sorted(RANK_WORLDS)})
    samples = rank_corr.measure(rows, RANK_WORLDS)
    m = _roundtrip(
        {
            "systems": rank_corr.system_rows(samples, {"mem0"}),
            "correlations": rank_corr.correlations(samples),
        }
    )
    out = pn.rank_numbers(m)
    assert out["RankExRhoSrcDef"] == "0.00 [0.00, 0.00]"  # identical worlds: a point interval
    assert out["RankExRhoSrcOrphan"] == "$-$1.00 [$-$1.00, $-$1.00]"  # a typeset minus
    assert out["RankExSrcDefSystems"] == "5" and out["RankExSrcDefWorlds"] == "3"
    assert out["RankExSystems"] == "5"
    # the subsets: not in this result yet, so MISSING
    assert out["RankExRhoSrcDefHoldMem"] == MISSING == out["RankExSrcDefNativeSystems"]
    assert pn.rank_numbers({}) == dict.fromkeys(out, MISSING)
    among = rank_corr.subsets(list(samples), {"mem0"})
    m["subsets"] = _roundtrip({k: rank_corr.correlations(samples, s) for k, s in among.items()})
    out = pn.rank_numbers(m)
    # HoldMem cells of LEVELS: S_src ranks (D 1, C 2, B 3, A 4), S_def (D 1, A 2, B 3, C 4)
    assert out["RankExRhoSrcDefHoldMem"] == "0.20 [0.20, 0.20]"
    assert out["RankExSrcDefHoldMemSystems"] == "4" and out["RankExSrcDefHoldMemWorlds"] == "3"
    assert out["RankExRhoSrcDefNative"] == MISSING  # one native system: no correlation
    assert out["RankExSrcDefNativeSystems"] == "1" == out["RankExSrcDefNativeBaseSystems"]


# ------------------------------------------------------------------------------- multiplicity
MULT_STAGE_DIRS = {"orig": "stage2_e1", "rep": "stage4_rep_e1"}


def test_multiplicity_macros_are_the_analysis_counts():
    view = paper_data.multiplicity_view(MULT_RESULT, SOURCES)
    assert set(view) == {"orig", "rep"}
    out = pn.mult_numbers(view)
    assert sorted(out) == sorted(
        f"MultEx{stage}{field}"
        for stage in ("Orig", "Rep")
        for field in ("Contrasts", "Interval", "Holm", "Bh")
    )
    for tag, stage in (("orig", "Orig"), ("rep", "Rep")):
        s = MULT_RESULT["stages"][MULT_STAGE_DIRS[tag]]["summary"]
        assert out[f"MultEx{stage}Contrasts"] == str(s["n_contrasts"])
        assert out[f"MultEx{stage}Interval"] == str(s["n_interval_excludes_zero"])
        assert out[f"MultEx{stage}Holm"] == str(s["n_survive_holm"])
        assert out[f"MultEx{stage}Bh"] == str(s["n_survive_bh"])
        assert int(out[f"MultEx{stage}Holm"]) <= int(out[f"MultEx{stage}Bh"])
        assert int(out[f"MultEx{stage}Bh"]) <= int(out[f"MultEx{stage}Interval"])
        assert int(out[f"MultEx{stage}Interval"]) <= int(out[f"MultEx{stage}Contrasts"])
    for name, value in out.items():
        pn._macro(name, value)  # letters-only names, plain digits
        assert value.isdigit(), (name, value)


def test_multiplicity_stage_that_is_not_staged_renders_missing(tmp_path):
    out = pn.mult_numbers(paper_data.multiplicity_view(MULT_RESULT, SOURCES))
    assert pn.mult_numbers({}) == dict.fromkeys(out, MISSING)
    assert paper_data.multiplicity_view(None, SOURCES) == {}
    assert paper_data.multiplicity_view(MULT_RESULT, {}) == {}
    # stage 2 is in the result, but R's source group names a stage that the result does not hold
    partial = paper_data.multiplicity_view(MULT_RESULT, {**SOURCES, "rep_e1": "runs/other_stage"})
    assert set(partial) == {"orig"}
    got = pn.mult_numbers(partial)
    assert all(v == MISSING for k, v in got.items() if k.startswith("MultExRep"))
    assert all(v != MISSING for k, v in got.items() if k.startswith("MultExOrig"))
    # nothing staged: the loader gives no view, and every macro is the marker
    assert paper_data.load_stage4(tmp_path, SOURCES)["mult"] == {}
    assert {k: pn.stage4_numbers(EMPTY)[k] for k in out} == dict.fromkeys(out, MISSING)


def test_multiplicity_view_is_loaded_from_the_committed_stage():
    view = paper_data.load_stage4(ROOT, SOURCES)["mult"]
    assert set(view) == {"orig", "rep"} and view["orig"]["n_contrasts"] == 42
    assert view == paper_data.multiplicity_view(MULT_RESULT, SOURCES)


# ------------------------------------------------------------------------------ full inputs
def _rank_result() -> Any:
    # three native systems and two no-store baselines, so that every subset has a correlation
    spec = {**LEVELS, "amem": (5, 7, 1), "lightrag": (3, 3, 5), "zero": (9, 9, 9), "rag": (1, 1, 1)}
    native = {"mem0", "amem", "lightrag"}
    samples = rank_corr.measure(rank_table({w: spec for w in sorted(RANK_WORLDS)}), RANK_WORLDS)
    among = rank_corr.subsets(list(samples), native)
    return _roundtrip(
        {
            "systems": rank_corr.system_rows(samples, native),
            "correlations": rank_corr.correlations(samples),
            "subsets": {k: rank_corr.correlations(samples, s) for k, s in among.items()},
        }
    )


def _full_input(tmp_path: Path) -> dict[str, Any]:
    """Every stage staged (synthetic where the real analysis is not there yet)."""
    (tmp_path / "m1").mkdir()
    return {
        "orig": ORIG,
        "rep": ORIG,
        "pool": ORIG,
        "c": C_E1,
        "m1": _eight(tmp_path / "m1"),
        "m2": {"llama": _family(), "gemma": _family(zero_cv=1), "gptoss": _family(True)},
        "m3": {f.key: THETA_R for f in FAMILIES},
        "n": ALL_SIX,
        "n3": None,
        "ncost": COST,
        "rendered": _rendered(tmp_path),
        "rank": _rank_result(),
        "mult": paper_data.multiplicity_view(MULT_RESULT, SOURCES),
    }


def test_all_stages_staged_leaves_only_the_fixtures_gaps_missing(tmp_path):
    """A macro that reads a key the analysis does not write would stay MISSING for good: with every
    stage staged, only what the synthetic inputs cannot hold (noise floors of the five new systems,
    the rendered fixture's other systems, branches and granularity) may be missing."""
    out = pn.stage4_numbers(_full_input(tmp_path))
    missing = sorted(k for k, v in out.items() if v == MISSING)
    assert all(
        re.fullmatch(r"Nat(Amem|Memos|LightRag|HippoRag|Cognee)Noise\w+", k)
        or re.fullmatch(
            r"Rend(CvDiff(Oracle|Rag)(Src|Def)|(Diff)?DMissed(Cited\w+|SourceOrphan))", k
        )
        for k in missing
    ), missing
    assert len(out) > 1200 and len(missing) < 60


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_all_stages_staged_macros_compile_as_text(tmp_path):
    macros = pn.with_estimates(pn.stage4_numbers(_full_input(tmp_path)))
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
    )
    assert done.returncode == 0, done.stdout[-1500:]


# ------------------------------------------------------------------------------ table structure
def _group(s: str, i: int) -> tuple[str, int]:
    """The brace group that opens at s[i] (nested groups included) and the index after it."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1 : j], j + 1
    raise ValueError(s[i : i + 40])


# For the tables' compile tests: the paper's text width (paper/preprint.sty: a 6.75 in text block
# on US letter) and its table style (paper/memtab-style.tex), which needs booktabs, array and
# xcolor and the grey of the figures
PAPER_PAGE = r"\usepackage[letterpaper,hmargin=0.875in,vmargin=1in]{geometry}"


def table_style(tmp_path: Path) -> str:
    """The preamble line that loads the table style in a compile test run in `tmp_path` (after
    booktabs, array and xcolor); the style file is copied there."""
    shutil.copy(ROOT / "paper" / "memtab-style.tex", tmp_path)
    return r"\definecolor{cgrey}{HTML}{6E6E6E}\input{memtab-style}"


# column types with arguments: p{width}; the table style's (paper/memtab-style.tex) N{template}
# and H{width}{indent}; P{width} and C{width} as a table defines them with \newcolumntype
COLUMN_ARGS = {"p": 1, "N": 1, "P": 1, "C": 1, "H": 2}
ENVIRONMENTS = ("tabular", "tabular*", "mttabular", "mttabular*")
FLOATS = ("table", "table*")  # a table in one column or across both


def _columns(spec: str) -> int:
    """The columns of a tabular spec: l, c, r and the types of `COLUMN_ARGS`; @{...}, >{...} and
    <{...} are not columns."""
    count, i = 0, 0
    while i < len(spec):
        c = spec[i]
        if spec[i + 1 : i + 2] == "{" and (c in "@><" or c in COLUMN_ARGS):
            i += 1
            for _ in range(COLUMN_ARGS.get(c, 1)):
                _, i = _group(spec, i)
            count += c in COLUMN_ARGS
            continue
        count += c in "lcr"
        i += 1
    return count


def _tabulars(tex: str) -> list[tuple[int, list[str]]]:
    """(column count, body lines) of every tabular, tabular* and table-style mttabular(*) in
    `tex`."""
    out = []
    for m in re.finditer(r"\\begin\{((?:mt)?tabular\*?)\}(\[[a-z]\])?", tex):
        i = m.end()
        if m.group(1).endswith("*"):
            _, i = _group(tex, i)  # the width of a tabular*
        spec, i = _group(tex, i)
        end = tex.index(f"\\end{{{m.group(1)}}}", i)
        out.append((_columns(spec), tex[i:end].strip("\n").splitlines()))
    return out


def check_tex(tex: str, panel: bool = False) -> None:
    """Environments balance, and every table row has as many cells as its tabular has columns.
    `panel`: a panel of a table whose float is in the paper's source, so it has none of its own."""
    floats = sum(tex.count(f"\\begin{{{env}}}") for env in FLOATS)
    assert floats == sum(tex.count(f"\\end{{{env}}}") for env in FLOATS) >= (0 if panel else 1)
    assert not (panel and floats), "a panel has no float of its own"
    begins = sum(tex.count(f"\\begin{{{env}}}") for env in ENVIRONMENTS)
    assert begins == sum(tex.count(f"\\end{{{env}}}") for env in ENVIRONMENTS) >= 1
    assert tex.count("{") == tex.count("}")
    tabulars = _tabulars(tex)
    assert tabulars
    for columns, lines in tabulars:
        for line in lines:
            if not line.endswith(r"\\") or line.startswith(("\\toprule", "\\midrule")):
                continue
            cells = re.split(r"\s*&\s*", line[:-2])
            span = sum(
                int(m.group(1)) if (m := re.match(r"\s*\\multicolumn\{(\d+)\}", c)) else 1
                for c in cells
            )
            assert span == columns, line


def _all_tables() -> dict[str, str]:
    return {
        "replication": pt.tab_replication(paper_data.replication_view(ORIG, REP_PARTIAL)),
        "families": pt.tab_families(None, {"llama": THETA_R}),
    }


def test_tables_are_wellformed_with_data_and_without():
    for name, tex in _all_tables().items():
        check_tex(tex)
        assert tex.startswith("\\begin{table*}") and f"\\label{{tab:{name}}}" in tex
    for tex in (
        pt.tab_replication({}),
        pt.tab_families(None, {}),
    ):
        check_tex(tex)
        assert MISSING in tex


def test_tables_and_macros_are_deterministic(tmp_path):
    assert _all_tables() == _all_tables()
    assert pn.stage4_numbers(EMPTY) == pn.stage4_numbers(EMPTY)
    inp = paper_data.load_stage4(ROOT, json.loads((ROOT / "results/sources.json").read_text()))
    assert pn.stage4_numbers(inp) == pn.stage4_numbers(inp)
    assert pn.stage4_numbers(inp) == pn.stage4_numbers(
        paper_data.load_stage4(ROOT, json.loads((ROOT / "results/sources.json").read_text()))
    )


@pytest.mark.paper  # the table style, paper/memtab-style.tex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path):
    tables = _all_tables()
    tables |= {f"{k}_missing": v for k, v in {
        "replication": pt.tab_replication({}),
        "families": pt.tab_families(None, {}),
    }.items()}  # fmt: skip
    body = "".join(f"\\input{{{name}.tex}}\n" for name in tables)
    for name, tex in tables.items():
        (tmp_path / f"{name}.tex").write_text(tex.replace("\\label{tab:", f"\\label{{tab:{name}"))
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
                r"\newcommand{\fail}{$\times$}\newcommand{\Cref}[1]{Table~1}"
                r"\newcommand{\cref}[1]{Table~1}",
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


# --------------------------------------------------------------------------------------- figure
def test_native_panels_for_all_systems_and_for_missing_ones():
    panels = figs.native_panels(ALL_SIX)
    assert [p["system"] for p in panels] == list(SYSTEMS) and all(p["staged"] for p in panels)
    light = next(p for p in panels if p["system"] == "lightrag")
    assert light["inferred"] == "(0,1)" and light["label"] == "LightRAG"
    cell = light["values"]["src", "Update"]
    src = ALL_SIX["systems"]["lightrag"]["branches"]["src"]
    assert cell["native"] == src["native"]["Update"] and cell["pred"] == src["white_box"]["Update"]
    assert cell["inside"] is True and light["values"]["src", "Stability"]["inside"] is False
    only_mem = figs.native_panels(
        paper_data.native_results({"n": None, "n3": _load("stage3_e5", "e5_natives.json")})
    )
    assert [p["staged"] for p in only_mem] == [True] + [False] * 5
    assert (
        only_mem[1]["inferred"] is None and only_mem[1]["values"]["src", "Update"]["native"] is None
    )
    assert not any(p["staged"] for p in figs.native_panels(None))
    assert [p["system"] for p in figs.native_panels({"systems": {"letta": {}}})][-1] == "letta"


def test_native_systems_figure_is_a_vector_pdf_with_and_without_systems(tmp_path):
    for name, res in (("all", ALL_SIX), ("none", None)):
        out = tmp_path / name
        out.mkdir()
        path = figs.fig_native_systems(res, out)
        data = path.read_bytes()
        assert data.startswith(b"%PDF") and b"/Subtype /Image" not in data
        assert figs.fig_native_systems(res, out).read_bytes()[:5] == b"%PDF-"


def test_calibration_figure_draws_with_and_without_the_curve(tmp_path):
    with_curve = figs.fig_calibration(ORIG["e2"], tmp_path).read_bytes()
    assert with_curve.startswith(b"%PDF") and b"/Subtype /Image" not in with_curve
    box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", with_curve)
    assert box and float(box.group(1)) == pytest.approx(72 * figs.fs.TEXT)  # full width
    assert float(box.group(2)) == pytest.approx(figs.CALIBRATION_PT[1])
    bare = {k: v for k, v in ORIG["e2"].items() if k != "explore_pooled_curve"}
    (tmp_path / "bare").mkdir()
    without = figs.fig_calibration(bare, tmp_path / "bare").read_bytes()
    assert without.startswith(b"%PDF") and with_curve != without
    (tmp_path / "empty").mkdir()
    assert figs.fig_calibration({}, tmp_path / "empty").read_bytes().startswith(b"%PDF")


def test_direct_labels_keep_their_order_and_spacing():
    assert figs._spread([0.9, 0.5, 0.1], 0.05) == [0.9, 0.5, 0.1]  # far apart: left alone
    ys = figs._spread([0.66, 0.62, 0.92], 0.05)  # the zero system and plain RAG at 12 months
    assert ys[2] == 0.92 and ys[0] > ys[1] and ys[0] - ys[1] == pytest.approx(0.05)
    assert (ys[0] + ys[1]) / 2 == pytest.approx(0.64)  # the pair stays centred on its targets
    assert figs._spread([0.5, 0.5, 0.5], 0.1) == pytest.approx([0.4, 0.5, 0.6])


def test_calibration_counts_reproduce_the_pooled_auroc():
    e2 = ORIG["e2"]
    counts = figs.calibration_counts(e2)
    alive, dead = sum(s for _, s, _ in counts), sum(f for *_, f in counts)
    assert alive + dead == e2["path3"]["pooled_auroc"]["n_items"] == 2446 and dead == 41
    assert [a for a, *_ in counts] == sorted({a for a, *_ in counts})  # one row per agreement
    # the ROC of agreement as a predictor: flag the items at or below a cut, share of surviving
    # (x) and of failed (y) items flagged, straight lines between the cuts (ties count one half).
    # Its area is the AUROC of the analysis, so the counts are the items the analysis scored
    xs, ys = [0.0], [0.0]
    for _, s, f in counts:
        xs.append(xs[-1] + s / alive)
        ys.append(ys[-1] + f / dead)
    steps = zip(xs, xs[1:], ys, ys[1:], strict=False)
    area = sum((x1 - x0) * (y0 + y1) / 2 for x0, x1, y0, y1 in steps)
    assert area == pytest.approx(e2["path3"]["pooled_auroc"]["est"], abs=1e-12)
    assert round(area, 4) == 0.8319


def test_calibration_bins_hold_every_item_and_none_is_empty():
    e2 = ORIG["e2"]
    bins = figs.calibration_bins(e2)
    edges = figs.CALIBRATION_BINS
    pairs = list(zip(edges, edges[1:], strict=False))
    assert [(b["lo"], b["hi"]) for b in bins] == pairs  # no bin is empty
    assert sum(b["n"] for b in bins) == 2446
    for b in bins:
        assert b["lo"] <= b["agreement"] <= b["hi"] and 0 <= b["survival"] <= 1
    counts = figs.calibration_counts(e2)
    assert sum(b["survival"] * b["n"] for b in bins) == pytest.approx(sum(s for _, s, _ in counts))
    assert figs.calibration_bins({}) == []
    bad = {"calibration": {"same": {"per_type": {"t": {"predicted": 1, "observed": 0.5, "n": 3}}}}}
    with pytest.raises(ValueError):
        figs.calibration_counts(bad)  # 1.5 survivals: not whole items


def test_native_cells_take_the_colour_of_their_binding_arm():
    fs = figs.fs
    none = figs.native_cell_style("holdmem_D0B0")
    cited = figs.native_cell_style("holdmem_D0B1_cited")
    source = figs.native_cell_style("holdmem_D0B1_source")
    assert none and cited and source and figs.native_cell_style(None) is None
    assert none["rail"] == fs.ORANGE and none["band"] == fs.tint(fs.ORANGE, 0.8)  # B=0 amber
    assert cited["rail"] == fs.SKY and source["rail"] == fs.BLUE  # B=1: the key's blues
    assert "".join(s for s, _ in cited["segments"]).endswith("(0,1), citation key")
    assert "".join(s for s, _ in none["segments"]) == "HOLDMEM(0,0)"
    panels = figs.native_panels(ALL_SIX)
    assert [p["cell"] for p in panels[:2]] == ["holdmem_D0B0", "holdmem_D0B0"]
    assert [(c, len(ks)) for c, ks in figs.native_groups(panels)][0] == ("holdmem_D0B0", 3)


def test_zero_cost_figure_is_a_full_width_vector_pdf(tmp_path):
    path = figs.fig_zero_cost(ORIG["e3"], tmp_path)
    data = path.read_bytes()
    assert data.startswith(b"%PDF") and b"/Subtype /Image" not in data
    box = re.search(rb"/MediaBox \[\s*0 0 ([\d.]+) ([\d.]+)\s*\]", data)
    assert box and float(box[1]) == pytest.approx(72 * figs.fs.TEXT)
    assert float(box[2]) == pytest.approx(figs.ZERO_PT[1])
    (tmp_path / "empty").mkdir()  # without the stage: the two panels' titles only
    assert figs.fig_zero_cost({}, tmp_path / "empty").read_bytes().startswith(b"%PDF")


# ------------------------------------------------------------------------- sources and the paper
def test_results_sources_name_every_stage4_group():
    sources = json.loads((ROOT / "results/sources.json").read_text())
    want = {"rep_e1", "rep_e2", "rep_e3", "pool_e1", "pool_e2", "pool_e3", "c", "m1", "m3", "n"}
    want |= {"ncost", "l", "rank", "mult"} | {f"m2_{f.key}" for f in FAMILIES}
    assert want <= set(sources)
    assert sources["mult"] == "runs/stage4_multiplicity"
    assert sources["m2_llama"] == "runs/stage4_m2_meta-llama_llama-3_1-8b-instruct"
    assert sources["m2_gemma"] == "runs/stage4_m2_google_gemma-4-12b-it"
    assert sources["m2_gptoss"] == "runs/stage4_m2_openai_gpt-oss-20b"
    assert all(str(v).startswith("runs/") for v in sources.values())


@pytest.mark.paper
def test_committed_generated_files_are_current():
    """paper/generated/* is what the generators print for the staged data: regenerate it (see
    paper/main.tex and the module docstrings) whenever a stage or a generator changes."""
    macros = pn.build_macros(ROOT)
    head = "% generated by memshift.analysis.paper_numbers; do not edit"
    text = "\n".join([head, *(pn._macro(k, macros[k]) for k in sorted(macros))]) + "\n"
    assert (ROOT / "paper/generated/numbers.tex").read_text() == text
    sources = json.loads((ROOT / "results/sources.json").read_text())
    inp = paper_data.load_stage4(ROOT, sources)
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    want = {
        "tab_replication.tex": pt.tab_replication(
            paper_data.replication_view(inp["orig"], inp["rep"])
        ),
    }
    want["tab_switches.tex"], want["tab_families.tex"] = pt.split_families(
        pt.tab_families(inp["m1"], inp["m3"])
    )
    for name, tex in want.items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name


# ------------------------------------------------------------ captions and the interval format
def test_interval_format_is_signed_scriptsize_with_a_thin_space():
    assert pt._ci({"est": 0.092, "lo": 0.067, "hi": 0.119}, sign=True) == (
        r"$+$9.2\,{\scriptsize [$+$6.7,\,$+$11.9]}"
    )
    assert pt._ci({"est": -0.005, "lo": -0.02, "hi": 0.0}, sign=True) == (
        r"$-$0.5\,{\scriptsize [$-$2.0,\,0.0]}"
    )
    # an interval whose ends equal the estimate as printed: the estimate alone
    assert pt.ci_pair_cells({"est": 0.0, "lo": 0.0, "hi": 0.0}, sign=True) == "0.0 & "
    assert pt.ci_pair_cells({"est": 1.0, "lo": 0.9996, "hi": 1.0}) == "100.0 & "
    assert pt.est_ci_cells({"est": 1.0, "lo": 0.9996, "hi": 1.0}) == "100.0 &"
    # both helpers print one cell text, apart from the space after an estimate alone
    e = {"est": 0.092, "lo": 0.067, "hi": 0.119}
    assert (
        pt.ci_pair_cells(e, sign=True)
        == pt.est_ci_cells(e, sign=True)
        == (r"$+$9.2 & {\scriptsize [$+$6.7,\,$+$11.9]}")
    )


def test_table_style_interval_formats():
    r"""The tables in the paper's table style set an interval as secondary text (`\mtint`),
    beside its estimate in a column pair or with it in one cell (`\mtpair`), and the estimate
    alone where the printed ends equal it, as the formats above do."""
    e = {"est": 0.092, "lo": 0.067, "hi": 0.119}
    assert pt.mt_ci_cells(e, sign=True) == r"$+$9.2 & \mtint{[$+$6.7,\,$+$11.9]}"
    assert pt.mt_pair(e, sign=True) == r"\mtpair{$+$9.2}{[$+$6.7,\,$+$11.9]}"
    alone = {"est": 0.0, "lo": 0.0, "hi": 0.0}
    assert pt.mt_ci_cells(alone, sign=True) == "0.0 & " and pt.mt_pair(alone) == r"\mtpair{0.0}{}"
    assert pt.mt_ci_cells(None) == rf"\multicolumn{{2}}{{c}}{{{MISSING}}}"
    assert pt.mt_pair({"est": float("nan"), "lo": 0.0, "hi": 0.0}) == pt.UNDEFINED
    assert pt._slope_cells({"est": -3.52, "lo90": -99.29, "hi90": -0.96}, unusable=True) == (
        r"\mtmuted{$-$3.52} & \mtint{[$-$99.29,\,$-$0.96]}"
    )
    assert pt.centred(pt.UNDEFINED, last=True, cols=2) == r"\multicolumn{2}{c@{}}{---}"


def _notes(tex: str) -> str:
    """The notes under the tables of `tex`."""
    return " ".join(re.findall(r"\\mtnotes\{(.*)\}\n", tex))


def test_stage4_captions_and_notes_define_their_rows_marks_and_cells():
    inp = paper_data.load_stage4(ROOT, SOURCES)
    rep = pt.tab_replication(paper_data.replication_view(inp["orig"], inp["rep"]))
    for term in (
        "Detection: share",
        "Value equals gold: share of the re-derived item-times whose value equals gold",
    ):
        assert term in _notes(rep), term
    assert "in the main study and in the replication" in rep.split(r"\label{tab:replication}")[0]
    assert "$D$'s effect on missed invalidation was 0.0" in _notes(rep)
    assert "behavioural" not in rep and "mechanistic" not in rep
    assert "the retention test in" not in rep and "Membership" not in rep
    fam = pt.tab_families(inp["m1"], inp["m3"])
    switches, families_ = fam.split("\n\n\\begin{table*}")  # two tables
    assert r"\label{tab:switches}" in switches and r"\label{tab:families}" in families_
    for term in ("slope (gray) is not usable", "(90\\% interval)", "AUROC: point estimates"):
        assert term in _notes(switches), term
    for term in (
        "$D$'s effect on missed invalidation was 0.0 in every column",
        "Qwen3.5-9B: the main reader, on the same worlds",
        r"\holdmem{}(1,1) accuracy: point estimates",
    ):
        assert term in _notes(families_), term
    # the families as writer and reader are in Table 4(b) of the main text, not here
    caption = families_.split(r"\label{tab:families}")[0]
    assert r"\Cref{tab:main}b gives them as writer and reader" in caption
    for label in ("\n$B$ effect on Update, $\\Ssrc$, pooled over $D$ & ", "\nNo-store BM25 $-$"):
        assert label not in families_, label
    assert "(i)" not in families_ and "(ii)" not in families_
    # Qwen3.5-9B: the main reader of the same stores, in the last column pair
    bxd = next(
        ln for ln in families_.splitlines() if ln.startswith("$B{\\times}D$ on Update, $\\Sdef$")
    )
    main = next(r for f in FAMILIES if (r := inp["m3"].get(f.key)))
    main_bxd = main["def"]["contrasts"]["BxD_interaction_update_source"]["main"]
    assert bxd.removesuffix(r" \\").endswith(pt.mt_ci_cells(main_bxd, sign=True))
    text = _notes(fam) + fam.split(r"\label{tab:families}")[0]
    assert "same sign" not in fam and "keep the main reader's sign" not in text
    assert r"$\pm3$" not in text and "regist" not in text and "check" not in text
    # every family's E1 analysis is staged, and D's effect on missed invalidation is 0.0 in all
    assert not any(ln.startswith("$D$ effect on missed") for ln in fam.splitlines())
    upgrade = next(ln for ln in fam.splitlines() if ln.startswith("upgrade (27B)$^\\ast$"))
    assert r"\mtmuted{" in upgrade  # the unusable slope is grey
    gemma = next(ln for ln in fam.splitlines() if ln.startswith("gemma-4-12B$^\\ast$"))
    assert r" & \mtna & " in gemma  # no item failed: no AUROC
    # the main study's three tiers, then the five other writers
    assert switches.count(r"\mtgap") == 1
    assert switches.index("downgrade") < switches.index(r"\mtgap") < switches.index("Llama")

"""Stage-4 paper generators (prereg v2 §8): every number, table and the predicted-vs-observed figure
come from staged analyses; a stage that has not run renders as the MISSING marker, and nothing is
invented for it. Real analyses (committed stages, and the analysis functions run on small
synthetic data) supply the input structures, so the readers follow the writers' actual keys."""

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


def _load(stage: str, name: str) -> Any:
    return json.loads((RUNS / stage / "analysis" / name).read_text())


def _roundtrip(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))


ORIG = {f"e{n}": _load(f"stage2_e{n}", f"e{n}_metrics.json") for n in (1, 2, 3)}
SOURCES = json.loads((ROOT / "paper/results_sources.json").read_text())
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
        # the documented labels are pre-registered predictions, not results
        assert value == MISSING or name.endswith("Documented"), (name, value)
    assert macros["NatLightRagDocumented"] == "(0,1)" and macros["NatMemDocumented"] == "(0,0)"


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


# --------------------------------------------------------- stage 2 and stage 3 exploratory macros
def test_t1_macros_for_the_cells_other_than_the_primary_one():
    e2m = ORIG["e2"]
    out = pn.e2_numbers(e2m)
    cells = e2m["explore_t1_cells"]
    for tier, tag in (("same", "Same"), ("upgrade", "Up"), ("downgrade", "Down")):
        # the pre-registered cell is the same computation as the primary T1
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
    # without definition items, and against the stage-2 verdicts
    nd = per["holdmem_D1B1_source"]["explore_no_definition"]
    assert out["RepExNoDefTThreeHoldTwelve"] == pn.ci(nd["minus_oracle"], pn.pp)
    assert out["RepExNoDefTThreeResultHoldOne"] == "fail"
    assert out["RepExNoDefTThreeReplicated"] == "9 of 9"
    assert out["RepTThreePassAll"] == "3 of 9" and out["RepExNoDefTThreePassAll"] == "1 of 9"
    oracle_nd = per["oracle"]["explore_no_definition"]["contract_valid"]
    assert out["RepExNoDefCvOracleTwelve"] == pn.ci(oracle_nd)
    assert set(pn.rep_e3_numbers(None, None).values()) == {MISSING}
    only_rep = pn.rep_e3_numbers(None, rep)
    assert only_rep["RepZeroMinusHoldOne"] != MISSING
    assert only_rep["RepExNoDefTThreeReplicated"] == MISSING


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
    assert set(view["paths"]) == {"path1", "path2"} and view["claim_level"] is None
    assert view["t1"] == {} and len(view["t3"]) == 9 and len(view["t4"]) == 9
    assert view["n_worlds"] == {"orig": 20, "rep": 20}
    out = pn.rep_numbers(view)
    assert out["RepPathOne"] == "not triggered" and out["RepPathTwo"] == "not triggered"
    assert out["RepPathOneOrig"] == "not triggered"
    assert out["RepPathThree"] == MISSING and out["RepClaimLevel"] == MISSING
    assert out["RepTOneReplicated"] == MISSING and out["RepTOneVerdictSame"] == MISSING
    assert out["RepTThreeReplicated"] == "9 of 9" and out["RepTFourReplicated"] == "9 of 9"
    assert out["RepTThreeResultHoldOne"] == "pass" and out["RepTThreeResultRagOne"] == "fail"


def test_replication_macros_follow_the_analysis_verdicts():
    view = paper_data.replication_view(ORIG, REP_PARTIAL)
    out = pn.rep_numbers(view)
    rows = replication.contrast_rows(ORIG["e1"], REP_PARTIAL["e1"])
    effects = [r for r in rows if r["kind"] == "effect"]
    assert out["RepEffects"] == str(len(effects))
    assert int(out["RepReplicated"]) + int(out["RepSameDirection"]) + int(
        out["RepNotReplicated"]
    ) == len(effects)
    assert out["RepEffectsSource"] == out["RepReplicatedSource"] == "21"  # all replicate
    row = next(
        r for r in rows if (r["branch"], r["contrast"]) == ("src", "B_effect_update_pooled_source")
    )
    assert out["RepBEffectUpdatePooledSourceSrc"] == pn.ci(row["rep"], pn.pp)
    assert out["RepBEffectUpdatePooledSourceSrcOrig"] == pn.ci(row["orig"], pn.pp)
    assert out["RepBEffectUpdatePooledSourceSrcVerdict"] == row["verdict"] == "replicated"
    # descriptive contrasts have an estimate but no verdict macro
    assert "RepDEffectFalseInvalSourceSrc" in out
    assert "RepDEffectFalseInvalSourceSrcVerdict" not in out
    assert out["RepDEffectMissedSourceDefVerdict"] == "not triggered"
    assert out["RepWorlds"] == out["RepWorldsOrig"] == "20"


def test_self_replication_with_all_three_stages_gives_the_claim_level():
    view = paper_data.replication_view(ORIG, ORIG)
    out = pn.rep_numbers(view)
    assert (
        out["RepClaimLevel"] == "registered claim: two-factor account with quantitative prediction"
    )
    assert out["RepPathThree"] == "not triggered"
    assert out["RepTOneReplicated"] == "3 of 3" and out["RepTThreeReplicated"] == "9 of 9"
    # the estimate for the tier called "same" and the count of replicated tiers are two macros
    assert out["RepTOneSame"] == pn.ci(ORIG["e2"]["t1"]["same"]["delta_90"], pn.pp)
    assert out["RepTOnePassSame"] == "within" and out["RepTOneVerdictUp"] == "replicated"
    assert out["RepNotReplicated"] == "0"
    assert out["RepPathThreeSlope"].startswith("1.01 (90\\% lower bound 0.90)")
    assert MISSING not in [
        v for k, v in out.items() if k.startswith(("RepPath", "RepTOne", "RepClaim"))
    ]


def test_a_flipped_effect_is_not_replicated_and_marked_so():
    rep = copy.deepcopy(ORIG)
    rep["e1"]["contrasts"]["src"]["B_effect_update_pooled_source"].update(
        est=-0.01, lo=-0.03, hi=0.01
    )
    view = paper_data.replication_view(ORIG, rep)
    out = pn.rep_numbers(view)
    assert out["RepBEffectUpdatePooledSourceSrcVerdict"] == "not replicated"
    assert out["RepPathOne"] == "triggered"  # the path-1 threshold is +2 pp
    assert out["RepClaimLevel"] == "registered claim: repair single factor (D)"
    assert int(out["RepNotReplicated"]) >= 1
    tab = pt.tab_replication(view)
    assert r"\fail" in tab and "$-$1.0" in tab


def test_pooled_forty_worlds_macros_and_partial_stage():
    pooled = paper_data.pooled_view(ORIG)  # any stage set with the three analyses will do
    out = pn.rep_pool_numbers(pooled)
    est = ORIG["e1"]["contrasts"]["src"]["B_effect_update_pooled_source"]
    assert out["RepPoolBEffectUpdatePooledSourceSrc"] == pn.ci(est, pn.pp)
    assert out["RepPoolPathThree"] == "not triggered"
    assert out["RepPoolTOneSame"] == pn.ci(ORIG["e2"]["t1"]["same"]["delta_90"], pn.pp)
    assert out["RepPoolTThreeResultHoldTwelve"] == "pass"
    only_e1 = pn.rep_pool_numbers(
        paper_data.pooled_view({"e1": ORIG["e1"], "e2": None, "e3": None})
    )
    assert only_e1["RepPoolPathOne"] == "not triggered"
    assert only_e1["RepPoolPathThree"] == MISSING and only_e1["RepPoolTThreeHoldOne"] == MISSING


def test_replication_direction_counts_come_from_the_estimates():
    out = pn.rep_numbers(paper_data.replication_view(ORIG, REP_PARTIAL))
    # 42 contrasts predicted non-zero; 37 intervals exclude 0 in either stage, 36 of them in the
    # predicted (positive) direction and one, the orphan cited B x D, the other way
    assert out["RepDirectional"] == out["RepOrigDirectional"] == "42"
    assert (out["RepOrigExcludesZero"], out["RepExcludesZero"]) == ("37", "37")
    assert (out["RepOrigExcludesPredicted"], out["RepExcludesPredicted"]) == ("36", "36")
    assert out["RepOrigExcludesOpposite"] == out["RepExcludesOpposite"] == "1"
    assert (out["RepOrigEstimatePredicted"], out["RepEstimatePredicted"]) == ("37", "38")
    # 37 replicate, 36 as predicted: the 37th replicates a negative effect
    assert out["RepReplicated"] == "37"
    assert out["RepReplicatedAsPredicted"] == "36"
    assert out["RepReplicatedAgainstPrediction"] == "1"
    assert out["RepReplicatedAgainstPredictionSource"] == "0"
    assert out["RepReplicatedAgainstPredictionCited"] == "1"
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


def test_replication_table_has_a_separate_as_predicted_column():
    tab = pt.tab_replication(paper_data.replication_view(ORIG, REP_PARTIAL))
    assert "as pred." in tab and "repl." in tab
    assert r"\begin{tabular}{@{}l rrcc rrcc rrcc@{}}" in tab
    label = r"$B{\times}D$ interaction, Update"
    rows = [line.split(" & ") for line in tab.splitlines() if line.startswith(rf"\quad {label}")]
    assert len(rows) == 2  # source-epoch and cited-cluster binding
    cited = rows[1]
    assert len(cited) == 13
    # orphan B x D at cited-cluster binding: negative in both stages and replicated (same sign,
    # interval excludes 0), so replicated and against the prediction
    assert cited[11] == r"\pass" and cited[12].removesuffix(r" \\") == r"\fail\,\fail"
    # every source-epoch row (the D effect on missed invalidation, judged by path 2, included) is
    # as predicted in both stages, on all three branches
    source_rows = [
        line.split(" & ")
        for line in tab.split("cited-cluster binding")[0].splitlines()
        if line.startswith(r"\quad ")
    ]
    assert len(source_rows) == 8 and any("missed invalidation" in c[0] for c in source_rows)
    for cells in source_rows:
        for k in (4, 8, 12):
            assert cells[k].removesuffix(r" \\") == r"\pass\,\pass", cells[0]


def test_replication_table_marks_and_partial_stage():
    full = pt.tab_replication(paper_data.replication_view(ORIG, REP_PARTIAL))
    assert r"\pass" in full and r"$\sim$" in full and r"\fail" in full  # cited-cluster rows
    assert "path~3 " + MISSING in full  # E2 not staged: path 3 and T1 are missing
    assert "T3 9 of 9" in full
    assert MISSING not in _first_body(full)  # the contrast rows need only E1
    empty = pt.tab_replication({})
    assert empty.count(MISSING) > 100 and r"\pass" not in _first_body(empty)


# ------------------------------------------------------------------------------------- block C
C_E1 = _load("stage4_c_e1", "e1_metrics.json")


def test_construct_macros_and_verdict_rules():
    out = pn.synth_numbers(C_E1)
    con = C_E1["contrasts"]
    assert out["SynthCOneSrc"] == pn.ci(con["src"]["C1_repair_D1s_minus_D0_source"], pn.pp)
    assert out["SynthCTwoShareDef"] == pn.ci(con["def"]["C2_share_of_D_effect_source"], pn.pct)
    for name in ("COne", "CTwo", "CThree", "CFour"):
        assert out[f"Synth{name}VerdictSrc"] == out[f"Synth{name}VerdictDef"] == "holds"
    assert "SynthCOneVerdictOrphan" not in out and "SynthExCOneOrphan" in out  # exploratory
    assert out["SynthExCOneCitedSrc"] == pn.ci(
        con["src"]["explore_C1_repair_D1s_minus_D0_cited"], pn.pp
    )
    assert out["SynthWorlds"] == "20"
    for cls, cname in pn.CLASS_NAMES:  # exploratory: C1 and C2 by item class
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


def test_synthetic_verdict_rules():
    holds = paper_data.synth_holds
    assert holds("above", _est(0.05, 0.02, 0.08)) is True
    assert holds("above", _est(0.05, -0.01, 0.08)) is False  # interval reaches 0
    assert holds("above", None) is None
    assert holds("nonneg", _est(0.0, -0.02, 0.02)) is True  # the point estimate decides
    assert holds("nonneg", _est(-0.001, -0.02, 0.02)) is False
    assert holds("path2", _est(0.0, 0.0, 0.0)) is True
    assert holds("path2", _est(0.05, 0.03, 0.07)) is False


def test_construct_table_and_missing():
    tab = pt.tab_construct(C_E1)
    # C1, C2, C4 judged on S_src and S_def, C3 on S_def only (prereg v2 section 4)
    assert _first_body(tab).count(r"\pass") == 7 and r"\fail" not in _first_body(tab)
    c3 = next(line for line in tab.splitlines() if line.startswith("C3 "))
    assert c3.split(" & ")[3] == "" and c3.split(" & ")[5] == r"\pass"  # S_src: no mark
    assert r"C3 is registered for $\Sdef$ only" in tab.split(r"\label{tab:construct}")[0]
    assert "$+$7.3" in tab and "46.7" in tab and "960" in tab
    # the share is an em dash where C1 is not positive (orphan: D_m falls below D=0)
    share = next(line for line in tab.splitlines() if line.startswith("C2 share"))
    assert share.count("---") == 1 and share.split(" & ")[-1].startswith("---")
    assert "182.9" not in tab and "35.7" in share
    assert C_E1["contrasts"]["orphan"]["C1_repair_D1s_minus_D0_source"]["est"] < 0
    # the D1s cell is displayed as D_m; no display string spells it D1s
    assert r"$D_{\mathrm{m}}-D_0$" in tab and r"$D_1-D_{\mathrm{m}}$" in tab
    assert "D1s" not in tab
    flipped = copy.deepcopy(C_E1)
    flipped["contrasts"]["src"]["C1_repair_D1s_minus_D0_source"]["est"] = -0.01
    assert (
        next(
            line for line in pt.tab_construct(flipped).splitlines() if line.startswith("C2 share")
        ).count("---")
        == 2
    )
    empty = pt.tab_construct(None)
    assert MISSING in empty and r"\pass" not in _first_body(empty)
    bad = copy.deepcopy(C_E1)
    bad["contrasts"]["def"]["C1_repair_D1s_minus_D0_source"].update(est=0.01, lo=-0.01)
    assert r"\fail" in pt.tab_construct(bad)
    assert pn.synth_numbers(bad)["SynthCOneVerdictDef"] == "does not hold"


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
    assert out["MOnePathThreeTriggered"] == "not triggered" and out["MOnePathThreeHolds"] == "holds"
    assert out["MOnePathThreeN"] == f"{8 * 10 * 20:,}"  # eight tiers, ten worlds, twenty items
    # D=1 loses nothing in six tiers (survival 1 vs 0.75), one item in five in `upgrade` (0.8),
    # one in two in `downgrade` (0.5 < 0.75: D does not protect this switch)
    assert out["MOneMBHoldsSame"] == out["MOneMBHoldsUp"] == out["MOneMBHoldsLlama"] == "holds"
    assert out["MOneMBHoldsDown"] == "does not hold"
    assert out["MOneMBHolding"] == "7 of 8"
    assert out["MOneTOnePassSame"] == "within" and out["MOneTOnePassUp"] == "outside"
    assert out["MOneTOneHolding"] == "6 of 8"
    assert out["MOneTOneQwenPrev"] == pn.ci(
        m["t1"][tier_name("cross:Qwen/Qwen3-8B")]["delta_90"], pn.pp
    )
    assert out["MOneQuantHolding"] == "8 of 8"
    assert out["MOneCaveats"] == "none"
    caveat = tier_name("cross:meta-llama/Llama-3.1-8B-Instruct")  # its underscores are escaped
    m["path3"]["tier_caveats"] = ["downgrade", caveat, "cross_unknown_model"]
    assert pn.m1_numbers(m)["MOneCaveats"] == r"downgrade, Llama-3.1-8B, cross\_unknown\_model"
    assert out["MOneSlopeGemma"] != MISSING and out["MOneAurocGemma"] != MISSING
    tab = pt.tab_families(m, {f.key: None for f in FAMILIES}, {})
    assert "Ministral-3-14B" in tab and "pooled, 8 switches" in tab
    assert r"\pass" in tab and r"\fail" in tab


def test_m1_partial_stage_counts_only_when_all_eight_are_staged(tmp_path):
    m = _m1(tmp_path, {"same": 0, "upgrade": 5, "downgrade": 0})
    out = pn.m1_numbers(m)
    assert out["MOneStaged"] == "3" and out["MOneSwitches"] == "3"
    assert out["MOneSlopeLlama"] == MISSING and out["MOneMBHoldsGemma"] == MISSING
    assert out["MOneMBHolding"] == out["MOneTOneHolding"] == out["MOneQuantHolding"] == MISSING
    assert out["MOneMBHoldsSame"] == "holds"  # the staged switches still render
    tab = pt.tab_families(m, {f.key: None for f in FAMILIES}, {})
    assert "same tier" in tab and MISSING in tab


# ------------------------------------------------------------------------------------ block M2
STAGE2_E1 = ORIG["e1"]


def _family(zero_cv: int = 0, break_pattern: bool = False) -> Any:
    e1 = copy.deepcopy(STAGE2_E1)
    if break_pattern:
        e1["contrasts"]["def"]["BxD_interaction_update_source"].update(est=0.01, lo=-0.02)
    return _roundtrip(families.family_checks(e1, family_items(zero_cv, 1)))


def test_m2_macros_checks_estimates_and_family_count():
    fams = {"llama": _family(), "gemma": _family(zero_cv=1), "gptoss": _family(break_pattern=True)}
    out = pn.m2_numbers(fams)
    assert out["MTwoLlamaPattern"] == "holds" and out["MTwoLlamaCheckV"] == "holds"
    assert out["MTwoGemmaPattern"] == "holds" and out["MTwoGemmaCheckV"] == "fails"
    assert out["MTwoGptOssPattern"] == "does not hold" and out["MTwoGptOssCheckIV"] == "fails"
    assert out["MTwoGptOssCheckIIISrc"] == "holds"
    assert out["MTwoLlamaBxDDef"] == pn.ci(fams["llama"]["estimates"]["BxD_def"], pn.pp)
    assert out["MTwoHolding"] == "2 of 3" and out["MTwoStaged"] == "3"
    assert out["MTwoLlamaWorlds"] == "20"
    tab = pt.tab_families(None, fams, {})
    assert "families where (ii)--(iv) all hold: 2 of 3" in tab


def test_m2_count_needs_all_three_families():
    out = pn.m2_numbers({"llama": _family(), "gemma": None, "gptoss": None})
    assert out["MTwoStaged"] == "1" and out["MTwoHolding"] == MISSING
    assert out["MTwoLlamaPattern"] == "holds" and out["MTwoGemmaPattern"] == MISSING


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
    assert computed and computed["pattern_holds"] and all(computed["checks"].values())
    res = families.analyse(stage, {stage.name: stage})  # what `analysis.stage m2` writes
    saved = json.loads((stage / "analysis" / "families.json").read_text())
    assert saved["families_holding"] == 1 and saved[stage.name]["checks"] == computed["checks"]
    assert res["families_holding"] == 1
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
    assert saved[out.name]["n_worlds"] == 1 and "families_holding" in saved
    manifest = json.loads((out / "manifest.json").read_text())
    assert "analysis/families.json" in manifest["files_sha256"]
    assert paper_data.load_families(tmp_path, out.name) == saved[out.name]


# ------------------------------------------------------------------------------------ block M3
THETA_R = _load("stage3_e7", "e7_metrics.json")["theta_r"]  # a real swap (5 worlds)


def test_m3_macros_from_a_real_reader_swap():
    out = pn.m3_numbers({"llama": THETA_R})
    assert out["MThreeLlamaWorlds"] == "5"
    assert out["MThreeLlamaSameSign"] == "6 of 6" and out["MThreeLlamaHolds"] == "holds"
    assert out["MThreeLlamaBUpdateSignSrc"] == "same"
    assert out["MThreeLlamaBUpdateMainSrc"] == pn.ci(
        THETA_R["src"]["contrasts"]["B_effect_update_pooled_source"]["main"], pn.pp
    )
    assert out["MThreeLlamaDMissedPathDef"] == "not triggered"
    assert out["MThreeLlamaCvHoldSwapDef"] == pn.ci(
        THETA_R["def"]["systems"]["holdmem_D1B1_source"]["swap"]["ContractValid_post"]
    )
    assert out["MThreeGemmaSameSign"] == MISSING and out["MThreeGemmaBxDMainSrc"] == MISSING
    assert out["MThreeStaged"] == "1" and out["MThreeHolding"] == MISSING


def test_m3_flipped_sign_and_full_count():
    flipped = copy.deepcopy(THETA_R)
    flipped["def"]["contrasts"]["BxD_interaction_update_source"]["swap"]["est"] = -0.05
    out = pn.m3_numbers({"llama": THETA_R, "gemma": flipped, "gptoss": THETA_R})
    assert out["MThreeGemmaBxDSignDef"] == "flipped" and out["MThreeGemmaSameSign"] == "5 of 6"
    assert out["MThreeGemmaHolds"] == "does not hold"
    assert out["MThreeStaged"] == "3" and out["MThreeHolding"] == "2 of 3"
    tab = pt.tab_families(None, {f.key: None for f in FAMILIES}, {"llama": THETA_R})
    assert "6/6" in tab and "not triggered" in tab


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
    assert out["NatLightRagInsideOf"] == "4 of 6" and out["NatLightRagNTwo"] == "holds"
    assert out["NatAmemInsideOf"] == "6 of 6"
    assert out["NatLightRagUpdateConsistentDef"] == "inside"
    assert out["NatLightRagStabilityConsistentSrc"] == "outside"
    assert out["NatStaged"] == "6"
    assert out["NatNOneAgree"] == "5 of 5" and out["NatNOneHolds"] == "holds"
    assert out["NatNTwoHolding"] == "6 of 6"
    assert out["NatNThreeInside"] == "34 of 36" and out["NatNThreeHolds"] == "holds"
    assert out["NatNThreeShare"] == pn.pct(34 / 36)
    assert out["NatNFourHolds"] in ("holds", "does not hold")  # B=0 and B=1 systems both present
    assert out["NatNFourStabilityBOne"] != MISSING
    assert out["NatMemFailRate"] == "1.00\\%" and out["NatMemUnits"] == "1,000"
    assert out["NatMemTThreeResult"] == "fail" and out["NatMemTFourResult"] == "pass"
    # cost per event, and its token ratio to Mem0 (10,200 tokens per event)
    assert out["NatMemCostCalls"] == "1.00" and out["NatMemCostRatio"] == "1.00"
    assert out["NatAmemCostPrompt"] == "5.0k" and out["NatAmemCostTokens"] == "5.2k"
    assert out["NatAmemCostRatio"] == pn.num(5200 / 10200, 2)


def test_native_verdicts_need_every_system_and_disagreeing_labels_count():
    inside = {s: [True] * 6 for s in SYSTEMS}
    three = {s: DOCUMENTED[s] for s in ("mem0", "amem", "lightrag")}
    partial = _natives(three, inside)
    out = pn.nat_numbers(partial, COST)
    assert out["NatStaged"] == "3" and out["NatAmemInsideOf"] == "6 of 6"
    for k in ("NatNOneAgree", "NatNOneHolds", "NatNTwoHolding", "NatNThreeShare", "NatNFourHolds"):
        assert out[k] == MISSING
    assert out["NatMemosLabel"] == MISSING and out["NatMemosDocumented"] == "(0,0)"
    # two systems whose inferred label differs from the documented one: N-1 fails (3 of 5)
    wrong = {s: DOCUMENTED[s] for s in SYSTEMS} | {"amem": (0, 1), "memos": (0, 1)}
    res = _natives(wrong, {s: [True] * 6 for s in SYSTEMS} | {"amem": [False] * 6})
    out = pn.nat_numbers(res, COST)
    assert out["NatAmemLabelAgrees"] == "no" and out["NatNOneAgree"] == "3 of 5"
    assert out["NatNOneHolds"] == "does not hold" and out["NatNTwoHolding"] == "5 of 6"
    assert out["NatAmemNTwo"] == "does not hold"
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


def test_native_table_and_marks():
    tab = pt.tab_native(ALL_SIX, COST)
    assert MISSING not in tab
    assert r"$\bullet\circ\bullet\bullet\circ\bullet$" in tab and "4/6" in tab
    assert tab.count("6/6") == 5 and "LightRAG & (0,1) & (0,1)" in tab
    only_mem = pt.tab_native(
        paper_data.native_results({"n": None, "n3": _load("stage3_e5", "e5_natives.json")}), {}
    )
    mem_row = next(line for line in only_mem.splitlines() if line.startswith("Mem0"))
    assert r"$\circ\bullet\bullet\bullet\bullet\bullet$" in mem_row and "5/6" in mem_row
    assert "70.6" in mem_row and MISSING in mem_row  # no cost measured: missing
    amem = next(line for line in only_mem.splitlines() if line.startswith("A-Mem"))
    assert amem.count(MISSING) == 8 and "(0,0)" in amem
    none = pt.tab_native(None, {})
    assert none.count(MISSING) == 6 * 8


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
    # the fixture: HoldMem(1,1) repairs half of the items in the rendered worlds, all in template
    # the fixture's D effect on repair is 0.5 - 0 in the rendered worlds but its interval over
    # three worlds reaches 0 (check fails); in the template worlds the effect is 1 (holds)
    assert m["rendered"]["checks"]["iii_D_repair_src"] is False
    assert out["RendCheckIIISrc"] == "fails" and out["RendTemplateCheckIIISrc"] == "holds"
    assert out["RendAgreeCheckIIISrc"] == "differ"
    assert out["RendDiffDRepairSrc"] == pn.ci(m["paired"][1]["diff"], pn.pp)
    assert out["RendDRepairSrc"] == pn.ci(m["rendered"]["estimates"]["D_repair_src"], pn.pp)
    assert out["RendTemplateDRepairSrc"].startswith("$+$100.0")
    assert out["RendPattern"] in ("holds", "does not hold") and out["RendAgreeCheckI"] == "agree"
    assert out["RendDMissedSourceSrc"].startswith("0.0")
    assert out["RendCvDiffHoldDef"] != MISSING
    assert out["RendKeptTemplate"] == "2.0\\%" and out["RendRetriedEmail"] == "4.0\\%"
    assert out["RendFidelityMin"] == "100.0\\%" and out["RendFidelityAllOne"] == "yes"
    assert out["RendFidelityChecked"] == "5,000" and out["RendModel"] == "google/gemma-4-12B-it"
    tab = pt.tab_rendered(m)
    assert "chat messages & 400 & 6.0 & 3.0" in tab and r"\pass" in tab


def test_rendered_without_render_reports_leaves_the_descriptives_missing(tmp_path):
    m = _rendered(tmp_path) | {"rendering": {"source": None, "note": "no reports"}}
    out = pn.rend_numbers(m)
    assert out["RendKeptTemplate"] == MISSING and out["RendFidelityMin"] == MISSING
    assert out["RendKeptEmail"] == MISSING and out["RendModel"] == MISSING
    assert out["RendCheckIIISrc"] != MISSING  # the checks do not depend on the reports


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
    assert out["RankExRhoSrcOrphan"] == "-1.00 [-1.00, -1.00]"
    assert out["RankExSrcDefSystems"] == "5" and out["RankExSrcDefWorlds"] == "3"
    assert out["RankExSystems"] == "5"
    assert pn.rank_numbers({}) == dict.fromkeys(out, MISSING)


# ------------------------------------------------------------------ multiplicity (exploratory)
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
    samples = rank_corr.measure(rank_table({w: LEVELS for w in sorted(RANK_WORLDS)}), RANK_WORLDS)
    return _roundtrip(
        {
            "systems": rank_corr.system_rows(samples, {"mem0"}),
            "correlations": rank_corr.correlations(samples),
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
def _tabulars(tex: str) -> list[tuple[int, list[str]]]:
    """(column count, body lines) of every tabular in `tex`."""
    out = []
    for m in re.finditer(r"\\begin\{tabular\}\{([^\n]*)\}\n(.*?)\\end\{tabular\}", tex, re.S):
        spec = re.sub(r"@\{[^}]*\}", "", m.group(1))
        out.append((len(re.findall(r"[lcrp]", spec)), m.group(2).splitlines()))
    return out


def check_tex(tex: str) -> None:
    """Environments balance, and every table row has as many cells as its tabular has columns."""
    for env in ("table", "tabular"):
        assert tex.count(f"\\begin{{{env}}}") == tex.count(f"\\end{{{env}}}") >= 1
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
    fams = {"llama": _family(), "gemma": _family(zero_cv=1), "gptoss": _family(True)}
    return {
        "replication": pt.tab_replication(paper_data.replication_view(ORIG, REP_PARTIAL)),
        "construct": pt.tab_construct(C_E1),
        "native": pt.tab_native(ALL_SIX, COST),
        "families": pt.tab_families(None, fams, {"llama": THETA_R}),
    }


def test_tables_are_wellformed_with_data_and_without():
    for name, tex in _all_tables().items():
        check_tex(tex)
        assert tex.startswith("\\begin{table}") and f"\\label{{tab:{name}}}" in tex
    for tex in (
        pt.tab_replication({}),
        pt.tab_construct(None),
        pt.tab_native(None, {}),
        pt.tab_families(None, {f.key: None for f in FAMILIES}, {}),
        pt.tab_rendered(None),
    ):
        check_tex(tex)
        assert MISSING in tex


def test_rendered_table_wellformed(tmp_path):
    tex = pt.tab_rendered(_rendered(tmp_path))
    check_tex(tex)
    assert "(iii) $D$ effect on repair, $\\Sdef$" in tex and "\\label{tab:rendered}" in tex


def test_tables_and_macros_are_deterministic(tmp_path):
    assert _all_tables() == _all_tables()
    assert pn.stage4_numbers(EMPTY) == pn.stage4_numbers(EMPTY)
    inp = paper_data.load_stage4(
        ROOT, json.loads((ROOT / "paper/results_sources.json").read_text())
    )
    assert pn.stage4_numbers(inp) == pn.stage4_numbers(inp)
    assert pn.stage4_numbers(inp) == pn.stage4_numbers(
        paper_data.load_stage4(ROOT, json.loads((ROOT / "paper/results_sources.json").read_text()))
    )


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile(tmp_path):
    tables = _all_tables() | {"rendered": pt.tab_rendered(_rendered(tmp_path))}
    tables |= {f"{k}_missing": v for k, v in {
        "replication": pt.tab_replication({}),
        "native": pt.tab_native(None, {}),
        "families": pt.tab_families(None, {f.key: None for f in FAMILIES}, {}),
    }.items()}  # fmt: skip
    body = "".join(f"\\input{{{name}.tex}}\n" for name in tables)
    for name, tex in tables.items():
        (tmp_path / f"{name}.tex").write_text(tex.replace("\\label{tab:", f"\\label{{tab:{name}"))
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}",
                r"\usepackage[letterpaper,margin=1in]{geometry}\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,newtxtext,newtxmath,booktabs,array,xcolor,caption}",
                r"\newcommand{\holdmem}{\textsc{HoldMem}}\newcommand{\pass}{\checkmark}",
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}",
                r"\newcommand{\fail}{$\times$}",
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


@pytest.mark.skipif(shutil.which("latex") is None, reason="no LaTeX for the figure text")
def test_native_systems_figure_is_a_vector_pdf_with_and_without_systems(tmp_path):
    for name, res in (("all", ALL_SIX), ("none", None)):
        out = tmp_path / name
        out.mkdir()
        path = figs.fig_native_systems(res, out)
        data = path.read_bytes()
        assert data.startswith(b"%PDF") and b"/Subtype /Image" not in data
        assert figs.fig_native_systems(res, out).read_bytes()[:5] == b"%PDF-"


def test_by_design_detection_lines_are_exactly_zero_or_one_in_the_data():
    """fig_dissociation marks the B=0 and source-epoch detection lines as 0 or 100 by design;
    that is only honest if the data have exactly those values on every branch, and the cited
    arm (not by design) must not."""
    for branch in ("src", "def"):
        cells = ORIG["e1"]["cells"][branch]
        for arm, want in (("none", 1.0), ("source", 0.0)):
            for cell in figs.ARM_CELLS[arm]:
                e = cells[cell]["MissedInvalidation"]
                assert e["est"] == want and e["lo"] == want and e["hi"] == want, (branch, cell)
        assert set(figs.BY_DESIGN) == {"none", "source"}
        for cell in figs.ARM_CELLS["cited"]:
            assert 0 < cells[cell]["MissedInvalidation"]["est"] < 1


@pytest.mark.skipif(shutil.which("latex") is None, reason="no LaTeX for the figure text")
def test_calibration_and_dissociation_figures_draw_with_and_without_the_curve(tmp_path):
    with_curve = figs.fig_calibration(ORIG["e2"], tmp_path).read_bytes()
    assert with_curve.startswith(b"%PDF") and b"/Subtype /Image" not in with_curve
    bare = {k: v for k, v in ORIG["e2"].items() if k != "explore_pooled_curve"}
    (tmp_path / "bare").mkdir()
    without = figs.fig_calibration(bare, tmp_path / "bare").read_bytes()
    assert without.startswith(b"%PDF") and with_curve != without
    assert figs.fig_dissociation(ORIG["e1"], tmp_path).read_bytes().startswith(b"%PDF")


# ------------------------------------------------------------------------- sources and the paper
def test_results_sources_name_every_stage4_group():
    sources = json.loads((ROOT / "paper/results_sources.json").read_text())
    want = {"rep_e1", "rep_e2", "rep_e3", "pool_e1", "pool_e2", "pool_e3", "c", "m1", "m3", "n"}
    want |= {"ncost", "l", "rank", "mult"} | {f"m2_{f.key}" for f in FAMILIES}
    assert want <= set(sources)
    assert sources["mult"] == "runs/stage4_multiplicity"
    assert sources["m2_llama"] == "runs/stage4_m2_meta-llama_llama-3_1-8b-instruct"
    assert sources["m2_gemma"] == "runs/stage4_m2_google_gemma-4-12b-it"
    assert sources["m2_gptoss"] == "runs/stage4_m2_openai_gpt-oss-20b"
    assert all(str(v).startswith("runs/") for v in sources.values())


def test_committed_generated_files_are_current():
    """paper/generated/* is what the generators print for the staged data: regenerate it (see
    paper/main.tex and the module docstrings) whenever a stage or a generator changes."""
    macros = pn.build_macros(ROOT)
    head = "% generated by memshift.analysis.paper_numbers; do not edit"
    text = "\n".join([head, *(pn._macro(k, macros[k]) for k in sorted(macros))]) + "\n"
    assert (ROOT / "paper/generated/numbers.tex").read_text() == text
    sources = json.loads((ROOT / "paper/results_sources.json").read_text())
    inp = paper_data.load_stage4(ROOT, sources)
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    want = {
        "tab_replication.tex": pt.tab_replication(
            paper_data.replication_view(inp["orig"], inp["rep"])
        ),
        "tab_construct.tex": pt.tab_construct(inp["c"]),
        "tab_native.tex": pt.tab_native(paper_data.native_results(inp), inp["ncost"]),
        "tab_families.tex": pt.tab_families(inp["m1"], inp["m2"], inp["m3"]),
        "tab_rendered.tex": pt.tab_rendered(inp["rendered"]),
    }
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
    assert pt._slope({"est": -3.52, "lo90": -99.29, "hi90": -0.96}, unusable=True) == (
        r"\textcolor{gray}{$-$3.52\,{\scriptsize [$-$99.29,\,$-$0.96]}}"
    )


def test_gate_table_reads_not_triggered_and_defines_its_paths():
    gate = json.loads((ROOT / SOURCES["gate"] / "analysis/gate_a.json").read_text())
    tex = pt.tab_gate(gate)
    assert "(held)" not in tex and tex.count("(not triggered)") == 3
    assert "path & triggered if & observed (status)" in tex
    caption = tex.split(r"\label{tab:gate}")[0]
    for term in (
        "A path is triggered when its condition holds",
        r"95\% world-bootstrap interval",
        "for both keys (source epoch, cited cluster)",
        r"with 90\% bounds",
    ):
        assert term in caption, term


def test_stage4_captions_define_their_rows_marks_and_cells():
    inp = paper_data.load_stage4(ROOT, SOURCES)
    rep = pt.tab_replication(paper_data.replication_view(inp["orig"], inp["rep"]))
    caption = rep.split(r"\label{tab:replication}")[0]
    for term in (
        "Detection: share",
        "the reader then answers correctly (behavioural)",
        "whose re-derived value equals gold (mechanistic)",
    ):
        assert term in caption, term
    assert "retention test T1" in rep and "Membership" not in rep
    con = pt.tab_construct(inp["c"]).split(r"\label{tab:construct}")[0]
    for term in (
        "$D_0$: the $D{=}0$ cell",
        "authored derivation",
        "$t_0$, before any shift",
        r"C2 share: $(D_1-D_{\mathrm{m}})/(D_1-D_0)$",
    ):
        assert term in con, term
    assert "benchmark-authored" not in con
    fam = pt.tab_families(inp["m1"], inp["m2"], inp["m3"])
    caption = fam.split(r"\label{tab:families}")[0]
    for term in (
        "(i) path~2 not triggered",
        "(ii) $B$ effect on Update",
        "(v) zero system below",
        r"registered 97\% bar",
        "slope (grey) is not usable",
    ):
        assert term in caption, term
    assert r"$\pm3$" not in caption
    upgrade = next(ln for ln in fam.splitlines() if ln.startswith("upgrade (27B)$^\\ast$"))
    assert r"\textcolor{gray}{" in upgrade  # the unusable slope is grey
    ren = pt.tab_rendered(inp["rendered"])
    first = next(ln for ln in ren.splitlines() if ln.startswith("(i) "))
    assert first.removesuffix(r" \\").endswith(pt.UNDEFINED)
    assert "has no paired difference" in ren.split(r"\label{tab:rendered}")[0]

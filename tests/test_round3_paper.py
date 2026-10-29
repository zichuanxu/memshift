"""Paper numbers and tables of the round-3 blocks (`analysis.round3_paper`: blocks U, N, H, Q and
G, the Claude blocks Y and V, the upsert split of `analysis.ex_round3_upsert`, and block X): the
stage is complete and traceable, the values the paper states are pinned, the staged results agree
with the published analyses they pair with, and the committed generated files are current."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import paper_numbers as pn
from memshift.analysis import round3_paper as r3p
from memshift.analysis.ex_e1_paper import UNDEFINED
from memshift.analysis.paper_data import FAMILIES, MISSING
from memshift.gen import kappa
from memshift.runinfo import file_sha256
from tests.memtab_check import MEMTAB_PREAMBLE, check_memtab
from tests.paper_sources import paper_text

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = r3p.load(ROOT, SOURCES)
EMPTY = r3p.load(ROOT, {})
NUMBERS = ROOT / "paper/generated/numbers.tex"  # read by the tests marked `paper` only
BASE = r3p.numbers(INP)
MACROS = pn.with_estimates(BASE)  # as paper_numbers emits them, with the `Est` forms
BRANCHES = ("src", "def", "orphan")


def committed(name: str) -> str:
    """The value of a macro in the committed numbers.tex."""
    m = re.search(rf"\\newcommand{{\\{name}}}{{(.*)}}\n", NUMBERS.read_text())
    assert m, name
    return m.group(1)


# ------------------------------------------------------------------------------------ stage
def test_results_sources_name_the_stage_and_it_is_staged():
    assert SOURCES["round3"] == r3p.STAGE_DIR == "runs/stage7_round3"
    for key in r3p.FILES:
        assert INP[key] is not None, key


def test_stage_manifest_lists_its_files_commands_and_sources():
    stage = ROOT / r3p.STAGE_DIR
    man = json.loads((stage / "manifest.json").read_text())
    assert man["run_id"] == "stage7_round3" and man["experiment"] == "merged-round3"
    assert man["analysis_code"]["dirty"] is False
    files = {f"analysis/{name}" for name in r3p.FILES.values()}
    assert set(man["files_sha256"]) == files == set(man["producers"])
    for rel, digest in man["files_sha256"].items():
        assert file_sha256(stage / rel) == digest, rel
    assert man["producers"] == r3p.PRODUCERS and man["note"] == r3p.NOTE
    want = [r.removeprefix("runs/") for r in r3p.source_runs()]
    # every run once: the upsert runs feed two files, and so do the G-b sets of V's three writers
    assert len(want) == len(set(want)) == 226
    assert [s["run_id"] for s in man["sources"]] == want
    for s in man["sources"]:
        assert file_sha256(ROOT / "runs" / s["run_id"] / "manifest.json") == s["manifest_sha256"]
    assert set(man["response_caches"]) == set(r3p.CACHES.values())
    for key, name in r3p.FILES.items():  # each result names its inputs relative to the root
        res = json.loads((stage / "analysis" / name).read_text())
        assert res["inputs"]["runs"] == list(r3p.RUNS[key])
        assert res["inputs"]["compare"] == list(r3p._compare(key))
        assert res["inputs"].get("response_cache") == r3p.CACHES.get(key)
        assert res["code"] == man["analysis_code"]
        assert "/Users/" not in (stage / "analysis" / name).read_text()
        assert f"--out {r3p.STAGE_DIR}/analysis/{name}" in man["producers"][f"analysis/{name}"]


def test_response_cache_snapshots_are_the_ones_the_manifest_hashes():
    man = json.loads((ROOT / r3p.STAGE_DIR / "manifest.json").read_text())
    for path, digest in man["response_caches"].items():
        if not (ROOT / path).exists():
            pytest.skip(f"{path} is not here (raw runs not linked)")
        assert file_sha256(ROOT / path) == digest, path


def test_the_generator_hooks_call_this_block():
    numbers = (ROOT / "src/memshift/analysis/paper_numbers.py").read_text()
    tables = (ROOT / "src/memshift/analysis/paper_tables.py").read_text()
    assert "round3_paper.numbers(round3_paper.load(root, sources))" in numbers
    assert "round3_paper.tables(round3_paper.load(a.root, sources))" in tables


# ----------------------------------------------------------------------------------- pairing
def test_u_n_y_h_q_g_and_v_pass_every_pairing_check():
    for key in ("u", "n", "y", "h_sonnet", "h_opus", "h_b", "q", "g_a", "g_b", "v"):
        pair = INP[key]["pairing"]
        assert pair["ok"] and not any(pair["counts"].values()), key
    assert INP["u_delta"]["pairing_ok"] is True
    # the refresh runs' 9,782 anchor answers, as many again from the upsert runs, and 2,060 from
    # the U5 runs of each M2 family (paired with its M2 stage)
    assert MACROS["RThreeUAnchorCompared"] == "25,744"
    per_run = INP["u"]["pairing"]["detail"]["runs"]
    for r in r3p.U5_RUNS:
        assert per_run[r.removeprefix("runs/")]["anchor_answers_compared"] > 0, r
    assert MACROS["RThreeNAnchorCompared"] == "3,840"  # 4 cells x 20 worlds x 48 items at t0
    # H-a and H-c anchor at k = 0 only: 6 cells and the JSON arm x 10 worlds x 3 branches x 48 items
    assert MACROS["RThreeHSonnetAnchorCompared"] == MACROS["RThreeHOpusAnchorCompared"] == "10,080"
    # Q's runs share every cell with the committed S_def branch; their anchor answers are those on
    # the probes whose text is unchanged (all but the definition items)
    assert MACROS["RThreeQAnchorCompared"] == "10,892"
    # a held-out-definition run has no committed counterpart, so it has no anchor
    assert INP["g_a"]["pairing"]["anchor_answers_compared"] == 0
    # the V and G-b runs anchor on every cell they share with the committed runs (D1s at k = 0)
    assert MACROS["RThreeVAnchorCompared"] == MACROS["RThreeGBAnchorCompared"] == "103,680"
    # the thinking writer's runs anchor on D0B1_source at every checkpoint
    assert MACROS["RThreeHThinkAnchorCompared"] == "4,898"


def test_y_contexts_equal_the_committed_ones():
    """With the restaged block C stage every answer of the Claude reader has the committed
    reader's context, so no answer is left out and no comparison carries a pairing note."""
    y = INP["y"]
    assert not any(y["pairing"]["counts"].values())
    assert y["pairing"]["detail"].get("context_differs") is None
    assert y["quantities"]["answers_left_out_context_differs"] == 0
    for name in ("Y1", "Y2", "Y3"):
        assert "pairing_note" not in y["threshold_tests"][name], name
    assert MACROS["RThreeYContextDiffers"] == MACROS["RThreeYStoreRequestsSent"] == "0"
    assert MACROS["RThreeYContextWorldBranches"] == "0"


# ---------------------------------------------------------------------- the staged results
def test_the_staged_block_results_and_no_outcome_macro():
    assert not [k for k in MACROS if re.search(r"Verdict|Reading|Registered|UOneTrue|Same$", k)]
    fams = INP["u"]["threshold_tests"]["U5"]["families"]
    assert all(f["tests"][0]["value"] <= 0.5 for f in fams.values())  # 3 of 3 below 0.5
    assert INP["u"]["threshold_tests"]["U3"]["tests"][0]["value"] <= 0.5
    tests = INP["h_sonnet"]["threshold_tests"]
    assert tests["H3"] == {"tests": []}  # H-b is computed in h_b.json
    assert set(tests) == {"H1", "H2", "H3"}  # no other writer's comparisons
    # Sonnet's r_0 after S_src was below 70% (the Opus runs follow it)
    r0 = INP["h_sonnet"]["reported"]["sonnet_r0_below_70pct"]
    assert r0["below_70pct"] is True and r0["on_all_worlds"] is True
    tests = INP["h_opus"]["threshold_tests"]  # another writer's comparisons carry its model name
    assert set(tests) == {"H1", "H2", "H3", "H1_claude-opus-5-5", "H2_claude-opus-5-5"}
    assert all(tests[k] == {"tests": []} for k in ("H1", "H2", "H3"))
    assert INP["h_opus"]["reported"]["sonnet_r0_below_70pct"] is None
    tests = INP["g_a"]["threshold_tests"]  # G3 is computed on the program-set runs (g_b.json)
    assert set(tests) == {"G1", "G2", "G3"}
    assert tests["G3"] == {"tests": []}
    tests = INP["g_b"]["threshold_tests"]  # G1 and G2 are computed on the held-out runs
    assert set(tests) == {"G1", "G2", "G3"}
    assert tests["G1"] == tests["G2"] == {"tests": []}
    assert tests["G3"]["sets"] == tests["G3"]["sets_total"] == 18
    tests = INP["h_b"]["threshold_tests"]
    assert set(tests) == {"H1", "H2", "H3"}
    assert tests["H3"] == {"tests": []}  # 12.6% of the thinking writes cut off, above 10%
    assert tests["H1"] == tests["H2"] == {"tests": []}  # no Claude writer
    tests = INP["v"]["threshold_tests"]  # over all nine program sets
    assert set(tests) == {"V1", "V2"}
    assert tests["V1"]["sets"] == tests["V2"]["sets"] == 9


# ---------------------------------------------------------------------------------- block X
def test_x_is_void_and_carries_no_x1_outcome():
    """Block X's anchor check failed: every world's prefix was restored from E5, yet 3 of the 480
    answers at the fork differ from E5's committed ones. The staged result computes no comparison
    for X1: no tests and no count of cells that favour an arm."""
    x = INP["x"]
    pair = x["pairing"]
    assert pair["ok"] is False and pair["counts"] == {"anchor_answers_differ": 3}
    assert pair["anchor_answers_compared"] == 480
    assert pair["detail"]["anchor_contexts_differ"] == 50
    assert set(pair["detail"]["prefix_sources"].values()) == {"e5"}
    per_world = pair["detail"]["anchor_per_world"]
    assert sorted(per_world) == [f"w{s}" for s in range(1000, 1010)]
    assert {w for w, a in per_world.items() if a["answer_identical"] < a["compared"]} == {
        "w1000",
        "w1001",
        "w1008",
    }
    assert sum(a["compared"] - a["context_identical"] for a in per_world.values()) == 50
    assert x["threshold_tests"] == {"X1": {"tests": []}}
    assert "X1_before_amendment" not in x["reported"]
    text = (ROOT / r3p.STAGE_DIR / "analysis" / "x.json").read_text()
    assert "favour" not in text and '"holds"' not in text
    # the paper reports the cells, not the anchor counts: no macro or table states them
    assert MACROS["RThreeXWorlds"] == "10"
    assert not [k for k in MACROS if k.startswith(("RThreeXAnchor", "RThreeXPrefix"))]
    tables = r3p.tables(INP)
    assert "anchor" not in "".join(tables.values())
    assert "void" not in "".join(tables.values())


def test_x_values_the_paper_states():
    """Mem0 only adds, so it invalidates nothing in any class and shift (m = 1, r
    undefined, a = Update); the query-time view acts on definition items only; invalidation is
    the only arm that deletes."""
    cells = INP["x"]["quantities"]["cells"]
    assert set(cells) == {f"{b}/{c}" for b in BRANCHES for c in ("definition", "other")}
    for name, cell in cells.items():
        native = cell["arms"]["mem0"]
        assert native["detected"] == 0 and native["m"]["est"] == 1.0, name
        assert native["r"]["est"] != native["r"]["est"], name  # undefined (NaN)
        assert native["a"]["est"] == native["Update"]["est"], name
        assert cell["arms"]["mem0_rep"]["n"] == native["n"], name
        if name.endswith("/other"):  # the view leaves the other items' answers as they were
            rep = cell["effects"]["mem0_rep"]
            assert (rep["est"], rep["lo"], rep["hi"]) == (0.0, 0.0, 0.0), name
    assert INP["x"]["reported"]["mutations_after_fork"] == {"mem0_det": {"adapter/DELETE": 1312}}
    assert MACROS["RThreeXNativeMutations"] == "0" and MACROS["RThreeXDetMutations"] == "1,312"
    assert MACROS["RThreeXSrcDefinitionAEst"] == "12.2\\%"
    assert MACROS["RThreeXDefDefinitionAEst"] == "8.9\\%"
    assert MACROS["RThreeXOrphanDefinitionAEst"] == "10.7\\%"
    assert MACROS["RThreeXSrcDefinitionM"] == "100.0\\% [100.0, 100.0]"
    assert MACROS["RThreeXSrcDefinitionR"] == UNDEFINED
    assert MACROS["RThreeXSrcDefinitionRepEffect"] == "$+$72.6\\,pp [$+$63.1, $+$81.2]"
    assert MACROS["RThreeXDefDefinitionRepEffect"] == "$+$64.4\\,pp [$+$57.4, $+$71.0]"
    assert MACROS["RThreeXOrphanDefinitionRepEffect"] == "$+$71.4\\,pp [$+$52.0, $+$88.6]"
    assert MACROS["RThreeXSrcDefinitionDetEffect"] == "$+$0.6\\,pp [$-$3.5, $+$4.2]"
    assert MACROS["RThreeXOrphanDefinitionDetEffect"] == "$+$3.6\\,pp [0.0, $+$10.7]"
    assert MACROS["RThreeXOrphanOtherDetEffect"] == "$+$35.9\\,pp [$+$26.1, $+$44.8]"
    assert MACROS["RThreeXDefOtherDetEffect"] == "$-$8.5\\,pp [$-$15.1, $-$1.1]"
    # the text gives the invalidation arm's effect and r on definition items as min to max
    det = [cells[f"{b}/definition"] for b in BRANCHES]
    effects = [c["effects"]["mem0_det"]["est"] for c in det]
    assert min(effects) == effects[0] and max(effects) == effects[2]
    rs = [c["arms"]["mem0_det"]["r"]["est"] for c in det]
    assert min(rs) == rs[1] and max(rs) == rs[2]
    assert MACROS["RThreeXDefDefinitionDetREst"] == "25.0\\%"
    assert MACROS["RThreeXOrphanDefinitionDetREst"] == "30.8\\%"


def test_upsert_values_the_paper_states():
    """U3 pools the four combinations; the upsert recovers almost none of binding's effect on
    definition items, and after S_del it is below the append-only cell (U4)."""
    assert MACROS["RThreeUThreeR"] == "0.01 [$-$0.02, 0.04]"
    assert MACROS["RThreeUThreeItems"] == "1,440"
    assert MACROS["RThreeUFourDiff"] == "$-$2.8\\,pp [$-$4.3, $-$1.4]"
    assert MACROS["RThreeUFourItems"] == "1,767"
    assert MACROS["RThreeUDoneDefDefinitionUpdateUpsertEst"] == "61.8\\%"
    assert MACROS["RThreeUDoneDefDefinitionUpdateBzeroEst"] == "61.8\\%"
    assert MACROS["RThreeUDoneDefDefinitionUpdateBoneEst"] == "99.7\\%"
    assert MACROS["RThreeUUpsertWorlds"] == MACROS["RThreeUWorlds"] == "20"
    pooled = INP["u"]["quantities"]["families"]["main"]["pooled_definition_D1_Dm_src_def"]
    assert pooled["upsert"]["combinations"] == ["D1/src", "D1/def", "Dm/src", "Dm/def"]
    # the upsert cells pair with the same committed cells as the refresh cells
    fam = INP["u"]["quantities"]["families"]["main"]["R"]
    for b in BRANCHES:
        for arm in ("D0", "D1", "Dm"):
            ref, ups = fam["refresh"][b][arm]["definition"], fam["upsert"][b][arm]["definition"]
            assert ref["update"]["B0"] == ups["update"]["B0"], (b, arm)
            assert ref["update"]["B1"] == ups["update"]["B1"], (b, arm)


def test_the_upsert_acts_only_where_the_delta_yields_an_item():
    """Split by the delta: binding pays on definition items where the delta yielded no new item,
    where both B=0 memories were almost never right; where it yielded one, the append-only cell
    already answered from it and the upsert moved almost no answer. Its few moves are on the
    other items."""
    for b, share, effect in (("Src", "27\\%", "79\\%"), ("Def", "36\\%", "93\\%")):
        head = f"RThreeUDeltaDone{b}DefinitionNone"
        assert MACROS[f"{head}Share"] == share
        assert MACROS[f"{head}EffectShare"] == effect
        for cell in ("Bzero", "Upsert"):
            assert float(MACROS[f"{head}Update{cell}Est"].removesuffix("\\%")) <= 2.5
        assert float(MACROS[f"{head}UpdateBoneEst"].removesuffix("\\%")) >= 97.5
        assert float(MACROS[f"RThreeUDeltaDone{b}DefinitionYieldedUpdateBzeroEst"][:4]) >= 90
    # the appendix bounds both B=0 cells by the largest of the four levels, S_src's upsert
    levels = {
        (b, cell): float(MACROS[f"RThreeUDeltaDone{b}DefinitionNoneUpdate{cell}Est"][:-2])
        for b in ("Src", "Def")
        for cell in ("Bzero", "Upsert")
    }
    assert max(levels, key=levels.__getitem__) == ("Src", "Upsert")
    assert MACROS["RThreeUDeltaDoneOrphanOtherNoneShare"] == "84\\%"
    assert MACROS["RThreeUDeltaDoneShiftsDefinitionMoved"] == "4"
    assert MACROS["RThreeUDeltaDoneShiftsOtherMoved"] == "120"
    arms = INP["u_delta"]["families"]["main"]["arms"]
    for arm in ("D1", "Dm"):
        for b in BRANCHES:  # the parts add up to the binding effect of the block's quantity
            q = arms[arm][b]["definition"]
            ref = INP["u"]["quantities"]["families"]["main"]["R"]["upsert"][b][arm]["definition"]
            assert q["n_items"] == ref["n_items"]
            assert abs(q["B_effect"] - ref["denominator"]["est"]) < 1e-9, (arm, b)


@pytest.mark.paper
def test_definition_writer_values_the_paper_states():
    assert MACROS["RThreeHSonnetWrites"] == "944"
    assert MACROS["RThreeHSonnetAtLimit"] == MACROS["RThreeHSonnetUnreadable"] == "0"
    assert MACROS["RThreeHSonnetWorlds"] == "10"
    assert MACROS["RThreeHSonnetDefDefinitionDoneBxD"] == "$+$16.4\\,pp [$+$9.7, $+$22.4]"
    assert MACROS["RThreeHSonnetRZero"] == "45.5\\% [36.0, 53.1]"
    assert MACROS["RThreeHSonnetRZeroQwenEst"] == "12.7\\%"
    assert MACROS["RThreeHSonnetRZeroDiff"] == "$+$32.7\\,pp [$+$26.6, $+$38.5]"
    v = INP["h_sonnet"]["quantities"]["writers"]["claude-sonnet-5-5"]["validity"]
    assert v["within_limits"] is True and v["unreadable_basis"].startswith("distinct requests")
    assert v["cache"]["missing_in_cache"] == 0
    # Qwen3.5-9B's r_0 in the main study, the reference of the Claude writers
    assert committed("ExAcctMainSrcDefinitionRZeroEst") == "16.2\\%"


def test_opus_writer_values_the_paper_states():
    """H-c (Claude Opus 5.5) on the same worlds, cells and Qwen3.5-9B comparison as H-a. Its
    interaction after S_def has the same point estimate as Sonnet's, which the text states once
    for both writers; the item-level values behind it differ."""
    assert MACROS["RThreeHOpusWrites"] == "944"
    assert MACROS["RThreeHOpusAtLimit"] == MACROS["RThreeHOpusUnreadable"] == "0"
    assert MACROS["RThreeHOpusTokensMax"] == "2,118"
    assert MACROS["RThreeHOpusWorlds"] == "10"
    assert MACROS["RThreeHOpusDefDefinitionDoneBxD"] == "$+$16.4\\,pp [$+$9.8, $+$23.0]"
    assert MACROS["RThreeHOpusSrcDefinitionDoneBxDEst"] == "$+$23.6\\,pp"
    assert MACROS["RThreeHOpusOrphanDefinitionDoneBxDEst"] == "$+$30.5\\,pp"
    assert MACROS["RThreeHOpusRZero"] == "43.0\\% [32.9, 51.1]"
    est = "RThreeH{}DefDefinitionDoneBxDEst"
    assert MACROS[est.format("Opus")] == MACROS[est.format("Sonnet")]
    for name in ("RZeroQwen", "Worlds", "Writes", "AnchorCompared"):  # same Qwen side and design
        assert MACROS[f"RThreeHOpus{name}"] == MACROS[f"RThreeHSonnet{name}"], name
    v = INP["h_opus"]["quantities"]["writers"]["claude-opus-5-5"]["validity"]
    assert v["within_limits"] is True and v["unreadable_basis"].startswith("distinct requests")
    assert v["cache"]["missing_in_cache"] == 0
    son = INP["h_sonnet"]["quantities"]["writers"]["claude-sonnet-5-5"]["bxd"]
    opus = INP["h_opus"]["quantities"]["writers"]["claude-opus-5-5"]["bxd"]
    for b in BRANCHES:  # each writer's interaction is over the same item-times
        for cls in ("definition", "other", "all"):
            assert son[b][cls]["D1"]["n_items"] == opus[b][cls]["D1"]["n_items"], (b, cls)


def test_refresh_ratios_the_paper_states():
    assert MACROS["RThreeUDoneSrcDefinitionREst"] == "1.00"
    assert MACROS["RThreeUDoneDefDefinitionREst"] == "0.99"
    assert MACROS["RThreeUDmSrcDefinitionREst"] == "0.94"
    assert MACROS["RThreeUDmDefDefinitionREst"] == "0.92"
    assert MACROS["RThreeUTwoSrc"] == "1.15 [0.86, 1.88]"
    assert MACROS["RThreeUTwoDef"] == UNDEFINED  # binding's effect at D=0 after S_def < 5 pp
    assert MACROS["RThreeUDzeroDefDefinitionBEffect"] == "$-$0.8\\,pp [$-$5.7, $+$4.2]"
    assert MACROS["RThreeUWorlds"] == "20"
    # the committed cells are those of the main study and block C (B=1 levels as published)
    assert MACROS["RThreeUDoneSrcDefinitionUpdateBone"].startswith("99.1\\% [")


@pytest.mark.paper
def test_natural_branch_values_and_their_published_counterparts():
    assert MACROS["RThreeNSrcDefinitionDiff"] == "$-$4.5\\,pp [$-$17.5, $+$9.6]"
    assert MACROS["RThreeNDefDefinitionDiff"] == "$+$13.5\\,pp [$+$1.8, $+$25.3]"
    assert MACROS["RThreeNSrcDefinitionNaturalMovedEst"] == "60.4\\%"
    # the shifted-branch side is the published attribution of round 2, paired on every item-time
    for b in ("Src", "Def", "Orphan"):
        assert MACROS[f"RThreeN{b}DefinitionShifted"] == committed(f"ExAttr{b}ShiftDefinitionBxD")
        assert MACROS[f"RThreeN{b}DefinitionItems"] == committed(f"ExAttr{b}ShiftDefinitionItems")
    cf = INP["n"]["quantities"]["counterfactual"]
    for b in BRANCHES:
        for cls in ("definition", "other"):
            node = cf[b]["shift"][cls]
            assert node["item_times_with_natural"] == node["item_times_in_branch"], (b, cls)


@pytest.mark.paper
def test_claude_reader_values_and_validity():
    assert MACROS["RThreeYAnswers"] == MACROS["RThreeYCalls"] == "68,474"
    assert MACROS["RThreeYUnreadable"] == MACROS["RThreeYCutOff"] == "0"
    assert MACROS["RThreeYDefOtherDoneBxD"] == "$+$9.9\\,pp [$+$3.4, $+$17.3]"
    assert MACROS["RThreeYOrphanOtherDoneBxD"] == "$+$5.3\\,pp [$+$2.3, $+$8.4]"
    assert MACROS["RThreeYSrcDefinitionDoneDiffEst"] == "$-$8.7\\,pp"
    assert MACROS["RThreeYDefDefinitionDoneDiffEst"] == "$-$15.2\\,pp"
    assert MACROS["RThreeYSRMainShiftsOtherLost"] == "5"
    assert MACROS["RThreeYSRMainShiftsOtherLostAbstain"] == "1"
    # the committed reader's B x D is the published class split, cell for cell
    for b in ("Src", "Def", "Orphan"):
        for c in ("Definition", "Other", "All"):
            assert MACROS[f"RThreeY{b}{c}DoneCommitted"] == committed(f"ExClsMain{b}{c}BxD"), (b, c)


def test_same_stores_same_store_part():
    """The Claude reader read the committed stores: on the authored D=1 arm its store part of
    B x D is the committed analysis's to the last digit, so the readers differ only in reading."""
    mine = INP["y"]["quantities"]["store_reader"]["results"]["D1"]
    theirs = json.loads((ROOT / "runs/stage6_ex_round3/analysis/store_reader.json").read_text())
    theirs = theirs["results"]["D1"]
    for b in BRANCHES:
        for cls in ("definition", "other", "all"):
            a, c = mine[b][cls]["interaction"]["store"], theirs[b][cls]["interaction"]["store"]
            assert (a["est"], a["lo"], a["hi"]) == (c["est"], c["lo"], c["hi"]), (b, cls)
            for cell in ("D0B0", "D0B1", "D1B0", "D1B1"):
                assert mine[b][cls]["counts"][cell]["ds"] == theirs[b][cls]["counts"][cell]["ds"]


@pytest.mark.paper
def test_unnamed_probe_values_the_paper_states():
    """Q: dropping the version from the definition-item probes leaves binding's effect at D=1
    after S_def where it was. The named side is the published class split, item for item."""
    assert MACROS["RThreeQOneDiff"] == "0.0\\,pp [$-$1.3, $+$1.2]"
    assert MACROS["RThreeQBEffectUnnamedEst"] == MACROS["RThreeQBEffectNamedEst"] == "$+$38.0\\,pp"
    assert MACROS["RThreeQDoneBxDUnnamed"] == "$+$39.5\\,pp [$+$33.3, $+$45.2]"
    assert MACROS["RThreeQItems"] == "387" and MACROS["RThreeQWorlds"] == "20"
    assert MACROS["RThreeQBEffectNamed"] == committed("ExClsMainDefDefinitionBDone")
    assert MACROS["RThreeQDoneBxDNamed"] == committed("ExClsMainDefDefinitionBxD")
    assert MACROS["RThreeQDmBxDNamed"] == committed("ExClsDmDefDefinitionBxD")
    assert INP["q"]["quantities"]["worlds"] == [f"w{s}" for s in range(1000, 1020)]


def test_heldout_definition_values_the_paper_states():
    """G-a: three draws of six held-out definitions on worlds 1000-1009. Every definition has
    changed item-times after every shift, and resampling definitions as well as worlds widens
    each interval, which stays above 0."""
    q = INP["g_a"]["quantities"]
    assert q["heldout_draws"] == [f"heldout:{s}" for s in r3p.G_A_SEEDS]
    assert q["program_sets"] == [] and q["ignored_runs"] == []
    assert MACROS["RThreeGADraws"] == "3" and MACROS["RThreeGAWorlds"] == "10"
    assert MACROS["RThreeGADefDoneBxD"] == "$+$33.0\\,pp [$+$22.1, $+$44.9]"
    assert MACROS["RThreeGASrcDoneBxD"] == "$+$26.7\\,pp [$+$16.6, $+$38.3]"
    assert MACROS["RThreeGADefDmBxD"] == "$+$27.4\\,pp [$+$15.2, $+$40.7]"
    assert MACROS["RThreeGASrcDmBxD"] == "$+$23.3\\,pp [$+$12.9, $+$36.4]"
    assert MACROS["RThreeGADefDoneBxDOneWay"] == "$+$33.0\\,pp [$+$29.2, $+$36.2]"
    assert MACROS["RThreeGADefDonePositive"] == "17"
    ids = {f"heldout:{s}|k.{d.kappa_id}" for s in r3p.G_A_SEEDS for d in kappa.heldout_draw(s).defs}
    for b in BRANCHES:
        for arm in ("D1", "Dm"):
            node = q["heldout"][b][arm]
            two, one = node["two_way"], node["one_way_worlds"]
            assert two["est"] == one["est"] and two["lo"] > 0, (b, arm)
            assert two["hi"] - two["lo"] > one["hi"] - one["lo"], (b, arm)
            assert two["n_clusters"] == len(node["definitions_with_items"]) == 18, (b, arm)
            assert set(node["per_definition"]) == ids, (b, arm)


def test_user_words_values_the_paper_states():
    """V: programs from the definition in the user's words keep the interaction at D_m on
    definition items over the nine sets (resampling sets widens each interval). Against the same
    writer's programs from the codified statement (G-b, same samples), Qwen3.5-9B's Update is
    unchanged and the Claude writers' is lower, as is agreement with gold before the shift."""
    q = INP["v"]["quantities"]
    assert q["V_sets"] == q["G_b_sets"] and len(q["V_sets"]) == 9 and q["ignored_runs"] == []
    assert {s.split("#")[0] for s in q["V_sets"]} == {m for m, _ in r3p.V_WRITERS}
    assert MACROS["RThreeVDefBxD"] == "$+$32.8\\,pp [$+$24.2, $+$41.3]"
    assert MACROS["RThreeVSrcBxD"] == "$+$22.4\\,pp [$+$14.7, $+$29.6]"
    assert MACROS["RThreeVOrphanBxD"] == "$+$29.0\\,pp [$+$19.8, $+$37.0]"
    for b in BRANCHES:
        node = q["programs"]["branches"][b]["definition"]
        two, one = node["two_way"], node["one_way_worlds"]
        assert two["n_clusters"] == len(node["sets_with_items"]) == 9, b
        assert two["est"] == one["est"] and two["lo"] > 0, b
        assert two["hi"] - two["lo"] > one["hi"] - one["lo"], b
    for tag, update, agree in (
        ("Qwen", "$+$1.4\\,pp [$-$0.8, $+$3.7]", "$-$4.4\\,pp [$-$5.3, $-$3.3]"),
        ("Sonnet", "$-$7.3\\,pp [$-$8.3, $-$6.2]", "$-$5.4\\,pp [$-$5.6, $-$5.1]"),
        ("Opus", "$-$18.1\\,pp [$-$20.5, $-$15.7]", "$-$16.0\\,pp [$-$16.4, $-$15.6]"),
    ):
        assert MACROS[f"RThreeV{tag}UpdatePooledDiff"] == update, tag
        assert MACROS[f"RThreeV{tag}AgreeDiff"] == agree, tag
    # G3 is computed in g_b.json, on the 18 program sets of six writers, not in v.json
    assert "G3" not in INP["v"]["threshold_tests"]
    assert {s.split("#")[0] for s in q["G_b_sets"]} <= {m for m, _ in r3p.G_B_WRITERS}


def _pp(e: dict) -> str:
    """A staged estimate as the paper prints it (pp with its interval)."""
    return pn.ci(e, pn.pp)


def test_program_set_values_the_paper_states():
    """G-b (G3): 18 program sets of six writers keep the interaction at D_m on definition items
    after every shift with intervals over worlds and program sets; per writer it ranges from
    near zero (Llama, gpt-oss) to the authored derivation's level (Sonnet, Opus), and the writer
    holds almost all of the variance between sets after S_def."""
    q = INP["g_b"]["quantities"]
    assert q["heldout"] == {} and q["ignored_runs"] == [] and len(q["program_sets"]) == 18
    writers = {m for m, _ in r3p.G_B_WRITERS}
    assert {s["writer"] for s in q["programs"]["sets"].values()} == writers
    assert MACROS["RThreeGBDefBxD"] == "$+$23.1\\,pp [$+$14.6, $+$31.2]"
    assert MACROS["RThreeGBSrcBxD"] == "$+$15.1\\,pp [$+$8.0, $+$22.4]"
    assert MACROS["RThreeGBOrphanBxD"] == "$+$19.4\\,pp [$+$10.9, $+$27.6]"
    assert MACROS["RThreeGBQwenDefBxD"] == "$+$20.4\\,pp [$+$15.6, $+$24.9]"
    assert MACROS["RThreeGBSonnetDefBxDEst"] == "$+$38.6\\,pp"
    assert MACROS["RThreeGBOpusDefBxDEst"] == "$+$38.8\\,pp"
    assert MACROS["RThreeGBGemmaDefBxDEst"] == "$+$30.7\\,pp"
    assert MACROS["RThreeGBLlamaDefBxD"] == "$+$5.4\\,pp [$+$0.2, $+$10.7]"
    assert MACROS["RThreeGBGptOssDefBxD"] == "$+$4.7\\,pp [$-$0.4, $+$9.8]"
    assert MACROS["RThreeGBDefWriterEtaSq"] == "0.99" and MACROS["RThreeGBWriters"] == "6"
    assert MACROS["RThreeGBDefSets"] == "18" and MACROS["RThreeGBWorlds"] == "20"
    for b, bt in (("src", "Src"), ("def", "Def"), ("orphan", "Orphan")):
        node = q["programs"]["branches"][b]["definition"]
        two, one = node["two_way"], node["one_way_worlds"]
        assert MACROS[f"RThreeGB{bt}BxD"] == _pp(two), b
        assert two["n_clusters"] == len(node["sets_with_items"]) == 18, b
        assert two["est"] == one["est"] and two["lo"] > 0, b
        assert two["hi"] - two["lo"] > one["hi"] - one["lo"], b
        for model, tag in r3p.G_B_WRITERS:
            assert MACROS[f"RThreeGB{tag}{bt}BxD"] == _pp(node["per_writer"][model]), (b, tag)
    # eta squared is over the 18 per-set values after S_def, the G3 contrast
    share = q["programs"]["writer_variance_share"]
    assert share["sets"] == 18 and share["writers"] == 6
    per_set = q["programs"]["branches"]["def"]["definition"]["per_set"]
    vals = {s: e["est"] for s, e in per_set.items()}
    grand = sum(vals.values()) / len(vals)
    by: dict[str, list[float]] = {}
    for s, x in vals.items():
        by.setdefault(q["programs"]["sets"][s]["writer"], []).append(x)
    ss_b = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in by.values())
    ss_t = sum((x - grand) ** 2 for x in vals.values())
    assert abs(ss_b / ss_t - share["eta_squared"]) < 1e-12
    assert MACROS["RThreeGBDefWriterEtaSq"] == r3p.ratio(share["eta_squared"])
    # the S_del per-writer values the block reports are those of the deletion branch
    rep_del = INP["g_b"]["reported"]["S_del_interaction_per_writer"]["definition"]
    assert rep_del == q["programs"]["branches"]["orphan"]["definition"]["per_writer"]


@pytest.mark.paper
def test_thinking_writer_validity_the_paper_states():
    """H-b (H3): 89 of the 708 thinking writes were cut off at the 32,768-token limit, above the
    10% validity limit; the same writes are the unreadable ones."""
    v = INP["h_b"]["quantities"]["think"]["validity"]
    assert v["limits"] == [32768] and v["within_limits"] is False
    assert MACROS["RThreeHThinkLimit"] == "32,768"
    assert MACROS["RThreeHThinkWrites"] == f"{v['writes']:,}" == "708"
    assert MACROS["RThreeHThinkCutOff"] == f"{v['cut_off']:,}" == "89"
    assert MACROS["RThreeHThinkCutOffShare"] == pn.pct(v["cut_off_share"]) == "12.6\\%"
    assert v["cut_off_share"] == v["cut_off"] / v["writes"] > 0.10
    assert MACROS["RThreeHThinkUnreadable"] == MACROS["RThreeHThinkCutOff"]
    assert MACROS["RThreeHThinkWorlds"] == "10"
    # at the controls' 8,192-token limit a larger share was cut off (an existing macro)
    assert committed("CtlThinkLimit") == "8,192"
    assert committed("CtlThinkAtLimitAllEst") == "32.1\\%"


def test_upsert_in_other_families_values_the_paper_states():
    """U5: each M2 family's upsert recovers almost none of binding's effect on definition items
    (D=1, S_src and S_def pooled), below the 0.5 bound as for the main writer, in 3 of 3."""
    fam = INP["u"]["quantities"]["families"]
    u5 = INP["u"]["threshold_tests"]["U5"]
    for f in FAMILIES:
        node = fam[f.model]["pooled_definition_D1_Dm_src_def"]["upsert"]
        assert node["combinations"] == ["D1/src", "D1/def"], f.key
        assert MACROS[f"RThreeUFive{f.tag}R"] == pn.ci(node["R"], r3p.ratio), f.key
        assert f"RThreeUFive{f.tag}Reading" not in MACROS, f.key
        assert u5["families"][f.model]["tests"][0]["value"] == node["R"]["est"], f.key
        assert fam[f.model]["worlds"]["upsert_D1"] == [f"w{s}" for s in range(1000, 1010)]
    assert MACROS["RThreeUFiveGemmaR"] == "0.07 [$-$0.04, 0.18]"
    assert MACROS["RThreeUFiveLlamaR"] == "$-$0.16 [$-$0.44, 0.01]"
    assert MACROS["RThreeUFiveGptOssR"] == "$-$0.03 [$-$0.08, 0.02]"
    assert MACROS["RThreeUFiveFamilies"] == "3" and "RThreeUFiveSame" not in MACROS


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = r3p.numbers(EMPTY)
    assert set(macros) == set(BASE) and len(macros) > 500
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only
        assert name.startswith("RThree"), name
        assert value == MISSING, (name, value)
    assert not [k for k, v in BASE.items() if v == MISSING]


# ------------------------------------------------------------------------------------ tables
LABELS = {
    "tab_reader.tex": "tab:reader",
    "tab_mem0x.tex": "tab:mem0x",
}
# panels of a table whose float, caption and labels are in the appendix source: the held-out
# definitions (b) of the per-definition table, the user's words (c) of the programs table
PANELS = ("tab_heldout.tex", "tab_userwords.tex")


def test_tables_are_wellformed_with_data_and_without():
    for inp in (INP, EMPTY):
        assert set(r3p.tables(inp)) == set(LABELS) | set(PANELS)
        for name, tex in r3p.tables(inp).items():
            check_memtab(tex, panel=name in PANELS)
            if name in PANELS:
                assert tex.startswith("\\begin{mttabular}") and "\\label" not in tex, name
                assert "\\caption" not in tex and "table*" not in tex, name
            else:
                assert tex.startswith("\\begin{table*}") and f"\\label{{{LABELS[name]}}}" in tex
                assert "\\centering\\mtsetup" in tex, name
            # the paper's table style: its type and rules, intervals as muted secondary text
            assert "\\begin{mttabular}" in tex, name
            assert "\\scriptsize" not in tex and "\\addlinespace" not in tex, name
    for tex in r3p.tables(EMPTY).values():
        assert MISSING in tex
    for tex in r3p.tables(INP).values():
        assert MISSING not in tex


def test_reader_table_sets_both_readers_of_the_same_stores_in_one_block():
    """One table for both readers of the same stores (it carries the label of either table), one
    block of shift x class x arm: the store part, the committed reader's reader part, and the
    Claude reader's reader part, B x D and difference from the committed reader, with no empty
    cell. The committed reader's B x D is the class split's (store part plus reader part), so the
    table does not repeat it."""
    tex = r3p.tables(INP)["tab_reader.tex"]
    assert "\\label{tab:store-reader-bxd}\\label{tab:reader}" in tex
    body = tex.split("\\mtnotes{")[0]
    assert "(a)" not in body and "(b)" not in body and body.count("Reader part") == 2
    head = body.split("\\toprule", 1)[1].split("\\midrule", 1)[0]  # the header rows
    assert "Store part" in head and "Sonnet $-$ Qwen" in head and head.count("$B{\\times}D$") == 1
    rows = [
        re.split(r"\s*&\s*", line.removesuffix(r" \\"))
        for line in body.splitlines()
        if re.search(r"& \$D(\{=\}1|_\{\\mathrm\{m\}\})\$ &", line)
    ]
    assert len(rows) == 3 * 3 * 2  # shift x class x arm
    # no empty cell: an estimate in each column pair (an interval of zero width is not printed)
    assert all(len(cells) == 13 and all(cells[i] for i in (3, 5, 7, 9, 11)) for cells in rows)
    assert "store part plus its reader part (\\cref{tab:classsplit})" in tex
    # the Sonnet x D_m values the text cites are the Claude reader's B x D
    for name in ("RThreeYDefOtherDmBxD", "RThreeYOrphanOtherDmBxD"):
        est = MACROS[name].split("\\,pp")[0]
        assert any(c[2] == "$D_{\\mathrm{m}}$" and c[9] == est for c in rows), name


def test_tables_name_the_experiments_without_codes_or_record_files():
    for tex in r3p.tables(INP).values():
        caption = tex.split("\\label{", 1)[0]
        assert "prereg" not in tex and "block" not in caption and "round" not in caption
        assert not re.search(r"\\b(worlds|seeds) \\d{3,4}--", tex)  # no seed ranges
        assert not re.search(r"\\b([UYHGQVNX][1-5]|P-[A-Z])\\b", tex)  # no prediction IDs
    draws = r3p.tables(INP)["tab_heldout.tex"]
    assert all(f"\n{k} & " in draws for k in ("1", "2", "3"))  # draws by number, not seed


@pytest.mark.paper  # the table style, paper/memtab-style.tex
@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile_at_the_paper_width(tmp_path):
    names = []
    for name, tex in r3p.tables(INP).items():
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
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}\newcommand{\pass}{\checkmark}"
                r"\newcommand{\fail}{$\times$}\newcommand{\cref}[1]{Table~1}",
                r"\begin{document}" + "".join(map(_set, names)) + r"\end{document}",
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


def _set(name: str) -> str:
    """The input of a generated table in a test document; a panel inside a float set as the
    appendix source sets it."""
    if name not in PANELS:
        return rf"\input{{{name}}}"
    return (
        r"\begin{table*}[t]\centering\mtsetup\setlength{\tabcolsep}{4pt}"
        rf"\input{{{name}}}\end{{table*}}"
    )


# ------------------------------------------------------------------------- generated files
@pytest.mark.paper
def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in r3p.tables(INP).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name
    numbers = NUMBERS.read_text()
    for name, value in MACROS.items():
        assert pn._macro(name, value) in numbers, name


@pytest.mark.paper
def test_the_text_reports_the_blocks_with_defined_macros():
    text = paper_text()
    used = set(re.findall(r"\\(RThree[A-Za-z]+)", text))
    assert used, "the text cites no round-3 macro"
    defined = set(MACROS) | {f"{k}Est" for k, v in MACROS.items() if " [" in v}
    assert used <= defined, sorted(used - defined)[:5]
    main = (ROOT / "paper/main.tex").read_text()
    assert "NatPending" not in text + main  # the placeholder is gone
    for name in ("tab_reader", "tab_heldout", "tab_userwords", "tab_mem0x"):
        assert f"\\input{{generated/{name}}}" in text, name
    # U3, U4, H-a and H-c are reported
    for name in ("RThreeUThreeR", "RThreeUFourDiff", "RThreeHSonnetRZero"):
        assert name in used, name
    for name in ("RThreeHSonnetDefDefinitionDoneBxDEst", "RThreeHOpusRZero"):
        assert name in used, name
    # G3, H3 and U5 are reported
    for name in ("RThreeGBDefBxD", "RThreeGBDefWriterEtaSq", "RThreeHSonnetWrites"):
        assert name in used, name
    assert {f"RThreeUFive{f.tag}R" for f in FAMILIES} <= used
    # Q, G-a and V are reported
    for name in ("RThreeQOneDiff", "RThreeGADefDoneBxD", "RThreeGASrcDoneBxD", "RThreeVDefBxD"):
        assert name in used, name
    # X: the effects are reported (its anchor counts are not)
    assert "RThreeXSrcDefinitionRepEffect" in used

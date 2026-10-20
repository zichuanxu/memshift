"""The analyses of `analysis.ex_round3_review`: read from the staged analyses, they reproduce the
published values they build on (tab_redef_only, the class split, tab_refresh_upsert, the Eq. (1)
terms, block H's r_0, the full-access control), the pinned anchor values hold, and the committed
numbers.tex carries every macro of the block."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import ex_round3_review as rv
from memshift.analysis import paper_numbers as pn
from memshift.analysis.ex_e1_paper import UNDEFINED, fmt
from memshift.analysis.paper_data import MISSING
from memshift.runinfo import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = rv.load(ROOT, SOURCES)
EMPTY = rv.load(ROOT, {})
NUMBERS = ROOT / "paper/generated/numbers.tex"  # read by the tests marked `paper` only
RAW = rv.numbers(INP)
MACROS = pn.with_estimates(RAW)  # with the `...Est` macros paper_numbers adds
BRANCHES = (("src", "Src"), ("def", "Def"), ("orphan", "Orphan"))
CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))


def committed(name: str) -> str:
    """The value of a macro in the committed numbers.tex."""
    m = re.search(rf"\\newcommand{{\\{name}}}{{(.*)}}\n", NUMBERS.read_text())
    assert m, name
    return m.group(1)


def node(part: str, *keys: str) -> Any:
    x: Any = INP[part]
    for k in keys:
        x = x[k]
    return x


# ------------------------------------------------------------------------------------ stage
def test_results_sources_name_the_stage_and_it_is_staged():
    assert SOURCES["ex_round3_review"] == rv.STAGE_DIR == "runs/stage8_ex_round3_review"
    assert rv.STAGE_KEY == "ex_round3_review"
    for key in rv.FILES:
        assert INP[key] is not None, key
        assert INP[key]["status"] == rv.STATUS, key


def test_stage_manifest_lists_its_files_producers_and_sources():
    stage = ROOT / rv.STAGE_DIR
    man = json.loads((stage / "manifest.json").read_text())
    assert man["run_id"] == "stage8_ex_round3_review"
    assert man["experiment"] == "merged-ex_round3_review"
    files = {f"analysis/{name}" for name in rv.FILES.values()}
    assert set(man["files_sha256"]) == files == set(man["producers"])
    for rel, digest in man["files_sha256"].items():
        assert file_sha256(stage / rel) == digest, rel
    assert man["producers"] == rv.PRODUCERS and man["note"] == rv.NOTE
    want = [s.removeprefix("runs/") for s in rv.source_runs(SOURCES)]
    assert want[: len(rv.SOURCE_KEYS)] == [SOURCES[k].removeprefix("runs/") for k in rv.SOURCE_KEYS]
    assert len(want) == len(set(want)) == len(rv.SOURCE_KEYS) + 5 + 10  # refresh, then H runs
    assert [s["run_id"] for s in man["sources"]] == want
    for s in man["sources"]:
        if "manifest_sha256" in s:
            got = file_sha256(ROOT / "runs" / s["run_id"] / "manifest.json")
            assert got == s["manifest_sha256"], s["run_id"]
    # no local path in the staged files: the tokenizer is named by its SHA256
    for name in rv.FILES.values():
        text = (stage / "analysis" / name).read_text()
        assert "/Users/" not in text and "/home/" not in text, name
    assert INP["sizes"]["tokenizer"]["file"] == "tokenizer.json"
    assert re.fullmatch(r"[0-9a-f]{64}", INP["sizes"]["tokenizer"]["sha256"])


def test_the_generator_hook_calls_this_block():
    numbers = (ROOT / "src/memshift/analysis/paper_numbers.py").read_text()
    assert "ex_round3_review.numbers(ex_round3_review.load(root, sources))" in numbers


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = rv.numbers(EMPTY)
    assert len(macros) > 1000
    assert set(macros) == set(RAW)
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only
        assert name.startswith("ExRev"), name
        assert value == MISSING, (name, value)


def test_every_macro_has_a_number_when_staged():
    assert not [k for k, v in MACROS.items() if v == MISSING]
    # no other item changes after a redefinition; no definition is half complete in scope
    assert MACROS["ExRevCauseMainDefShiftOtherItems"] == "0"
    assert MACROS["ExRevCauseMainDefShiftOtherBxD"] == UNDEFINED
    assert MACROS["ExRevCovCloseFriendCompleteShare"] == "0.0\\% [0.0, 0.0]"


# ------------------------------------------------------------- refresh against binding
@pytest.mark.paper
def test_refresh_reproduces_block_u_and_the_class_split():
    for arm, at in (("D0", "Dzero"), ("D1", "Done"), ("Dm", "Dm")):
        for b, bt in BRANCHES:
            for cls, ct in CLASSES:
                q = node("refresh", "results", b, cls, arm)
                name = f"RThreeU{at}{bt}{ct}"
                assert f"{q['n_items']:,}" == committed(f"{name}Items"), name
                for level, tag in (("B0", "Bzero"), ("refresh", "Refresh"), ("B1", "Bone")):
                    assert fmt(q["update"][level]) == committed(f"{name}Update{tag}"), name
                assert fmt(q["B1_minus_B0"], pn.pp) == committed(f"{name}BEffect"), name
                assert fmt(q["refresh_minus_B0"], pn.pp) == committed(f"{name}Gain"), name
                if arm != "D0":
                    split = "ExClsMain" if arm == "D1" else "ExClsDm"
                    assert fmt(q["BxD_B1"], pn.pp) == committed(f"{split}{bt}{ct}BxD"), name


def test_binding_is_the_refresh_on_definition_items_and_removes_stale_items():
    want = {  # B=1 minus the refresh on changed definition item-times, pp (the verification)
        "Dzero": ("$-$0.9", "0.0", "0.0"),
        "Done": ("0.0", "$+$0.3", "0.0"),
        "Dm": ("$+$2.1", "$+$2.8", "$+$3.6"),
    }
    for at, values in want.items():
        for (_, bt), v in zip(BRANCHES, values, strict=True):
            got = MACROS[f"ExRevRefresh{at}{bt}DefinitionBoneMinusRefreshEst"]
            assert got == f"{v}\\,pp", (at, bt)
    other = MACROS["ExRevRefreshDoneOrphanOtherBoneMinusRefresh"]
    assert other == "$+$9.9\\,pp [$+$4.9, $+$14.6]"


# ---------------------------------------------------------------------- shift and drift
@pytest.mark.paper
def test_cause_reproduces_tab_redef_only_and_the_class_split():
    for _, bt in BRANCHES:
        for pt in ("Shift", "Drift"):
            for _, ct in CLASSES:
                name = f"{bt}{pt}{ct}"
                assert MACROS[f"ExRevCauseMain{name}Items"] == committed(f"ExAttr{name}Items")
                assert MACROS[f"ExRevCauseMain{name}BxD"] == committed(f"ExAttr{name}BxD")
                assert MACROS[f"ExRevCauseDm{name}Items"] == committed(f"ExAttr{name}Items")
        for _, ct in CLASSES:
            for arm, split in (("Main", "ExClsMain"), ("Dm", "ExClsDm")):
                got = MACROS[f"ExRevCause{arm}{bt}Pooled{ct}BxD"]
                assert got == committed(f"{split}{bt}{ct}BxD"), (arm, bt, ct)
            # the cited-cluster key's misses, pooled, are the Eq. (1) m of D1B1_cited
            missed = committed(f"ExDecMain{bt}{ct}DoneBoneMissedCited")
            assert MACROS[f"ExRevCited{bt}Pooled{ct}MissShare"] == missed, (bt, ct)


def test_shift_caused_contrasts_and_cited_misses():
    assert MACROS["ExRevCauseMainSrcPooledAllBxD"] == "$+$7.4\\,pp [$+$5.0, $+$9.8]"
    assert MACROS["ExRevCauseMainSrcShiftAllBxD"] == "$+$2.5\\,pp [$+$0.5, $+$4.4]"
    assert MACROS["ExRevCauseMainOrphanShiftAllBxDEst"] == "$-$0.1\\,pp"
    assert MACROS["ExRevCauseDmDefShiftAllBxD"] == "$+$32.4\\,pp [$+$23.4, $+$41.5]"
    # the cited-cluster key misses no change a redefinition caused; its misses are drift
    assert MACROS["ExRevCitedDefShiftAllMissed"] == "0"
    assert MACROS["ExRevCitedDefShiftAllItems"] == "222"
    assert MACROS["ExRevCitedDefDriftAllMissShareEst"] == "34.5\\%"
    assert MACROS["ExRevCitedSrcShiftAllMissShareEst"] == "7.5\\%"
    assert MACROS["ExRevCitedSrcDriftAllMissShareEst"] == "30.2\\%"
    assert MACROS["ExRevCitedOrphanShiftAllMissShareEst"] == "0.4\\%"
    assert MACROS["ExRevCitedOrphanDriftAllMissShareEst"] == "33.0\\%"
    for b in node("cause", "cited").values():  # the key's detection does not depend on D
        assert all(c["same_at_D0"] for part in b.values() for c in part.values())


# ------------------------------------------------------------------------------- clock
@pytest.mark.paper
def test_clock_reproduces_the_main_terms():
    for study, st in (("main", "Main"), ("rep", "Rep")):
        for b, bt in BRANCHES:
            for cls, ct in CLASSES:
                for cell, cellt in (("D0B1", "DzeroBone"), ("D1B1", "DoneBone")):
                    t = node("clock", study, "terms", b, cls, cell, "since_t0")
                    name = f"ExDec{st}{bt}{ct}{cellt}"
                    assert fmt(t["m"]) == committed(f"{name}Missed"), name
                    assert fmt(t["r"]) == committed(f"{name}Repair"), name


def test_clock_only_changes_and_the_staleness_errors():
    for study, st, wrong, clock in (("main", "Main", "6", "5"), ("rep", "Rep", "7", "7")):
        assert MACROS[f"ExRevClock{st}ShiftsErrorsOneOne"] == wrong
        assert MACROS[f"ExRevClock{st}ShiftsErrorsOneOneClock"] == clock
        cases = node("clock", study, "errors_one_one", "shifts", "cases")
        assert sum(c["clock_staleness"] for c in cases) == int(clock)
        assert all(not c["detected_after_last_change"] for c in cases if c["clock_staleness"])
    assert MACROS["ExRevClockMainShiftsDefinitionChanged"] == "1,028"
    assert MACROS["ExRevClockMainShiftsDefinitionClockOnly"] == "99"
    assert MACROS["ExRevClockMainShiftsDefinitionClockCaught"] == "91"
    assert MACROS["ExRevClockMainShiftsDefinitionClockSilent"] == "8"
    # per interval, the verification's audit: 842 changes of definition items, 86 clock-only
    assert MACROS["ExRevClockMainShiftsDefinitionChanges"] == "842"
    assert MACROS["ExRevClockMainShiftsDefinitionChangesClockOnly"] == "86"
    assert MACROS["ExRevClockMainShiftsDefinitionChangesClockSilent"] == "5"
    assert MACROS["ExRevClockRepShiftsDefinitionChangesClockSilent"] == "7"
    # detection relative to the last change: the stale items move from r to m
    assert MACROS["ExRevClockMainSrcDefinitionDoneBoneMissedLastEst"] == "0.6\\%"
    assert MACROS["ExRevClockMainSrcDefinitionDoneBoneRepairLastEst"] == "99.7\\%"
    assert MACROS["ExRevClockRepDefDefinitionDoneBoneRepairLastEst"] == "100.0\\%"
    for study in ("main", "rep"):
        assert node("clock", study, "world_branches_hashed") == 60


# ------------------------------------------------------------------------------ forced
def test_forced_rederivation_on_the_cited_keys_misses():
    assert MACROS["ExRevForcedDzeroSrcDefinitionItems"] == "78"
    assert MACROS["ExRevForcedDzeroSrcDefinitionGain"] == "$+$2.6\\,pp [0.0, $+$6.2]"
    assert MACROS["ExRevForcedDoneSrcDefinitionGain"] == "$+$94.9\\,pp [$+$87.3, $+$100.0]"
    assert MACROS["ExRevForcedDoneDefDefinitionForcedEst"] == "100.0\\%"
    assert MACROS["ExRevForcedDzeroDefOtherCitedEst"] == "0.0\\%"


# ---------------------------------------------------------------------------- coverage
@pytest.mark.paper
def test_coverage_reproduces_r0_and_the_interaction():
    # all item-times together: the published r_0 and B x D of each writer
    for name, src in (
        ("qwen", "ExDecMainSrcDefinitionDzeroBoneRepair"),
        ("qwen10", "RThreeHOpusRZeroQwen"),
        ("sonnet", "RThreeHSonnetRZero"),
        ("opus", "RThreeHOpusRZero"),
    ):
        assert fmt(node("coverage", "r", name, "src", "all", "r0")) == committed(src), name
    for b, bt in BRANCHES:
        for arm, at, split in (("D1", "Done", "ExClsMain"), ("Dm", "Dm", "ExClsDm")):
            e = node("coverage", "bxd", "qwen", b, arm, "all")
            assert fmt(e, pn.pp) == committed(f"{split}{bt}DefinitionBxD"), (b, arm)
            for w, wt in (("sonnet", "Sonnet"), ("opus", "Opus")):
                e = node("coverage", "bxd", w, b, arm, "all")
                assert fmt(e, pn.pp) == committed(f"RThreeH{wt}{bt}Definition{at}BxD"), (w, b)
    assert MACROS["ExRevFullAllTruncatedShare"] == committed("CtlFullTruncatedShareAll")


def test_evidence_coverage_of_the_d0_writer():
    for d in ("CloseFriend", "VipSender"):
        assert MACROS[f"ExRevCov{d}CompleteShareEst"] == "0.0\\%"
    for d in ("WorkingHours", "ActiveProject", "OverdueTask"):
        assert MACROS[f"ExRevCov{d}CompleteShareEst"] == "100.0\\%"
        assert MACROS[f"ExRevCov{d}ScopeShareEst"] == "100.0\\%"
    assert MACROS["ExRevCovCloseFriendScopeShareEst"] == "19.8\\%"
    assert MACROS["ExRevCovVipSenderScopeShareEst"] == "23.4\\%"
    # with the whole scope, the Claude writers re-derive about as well as the authored rule
    for w, values in (("Opus", ("91.7", "97.1", "95.2")), ("Sonnet", ("94.4", "96.2", "93.5"))):
        for (_, bt), v in zip(BRANCHES, values, strict=True):
            assert MACROS[f"ExRevCov{w}{bt}CompleteRZeroEst"] == f"{v}\\%", (w, bt)
    assert MACROS["ExRevCovOpusSrcCompleteDoneBxDEst"] == "$-$23.6\\,pp"
    assert MACROS["ExRevCovOpusDefIncompleteDoneBxDEst"] == "$+$55.7\\,pp"
    assert MACROS["ExRevCovQwenSrcCompleteRZeroEst"] == "36.8\\%"
    # the full-access control: truncated on close friends and VIP senders, the D=0 request on
    # the task definitions
    assert MACROS["ExRevFullAllRederived"] == "472"
    for d, trunc, same in (
        ("CloseFriend", "100.0", "0.0"),
        ("VipSender", "100.0", "0.0"),
        ("ActiveProject", "0.0", "100.0"),
        ("OverdueTask", "0.0", "100.0"),
    ):
        assert MACROS[f"ExRevFull{d}TruncatedShareEst"] == f"{trunc}\\%", d
        assert MACROS[f"ExRevFull{d}SameRequestShareEst"] == f"{same}\\%", d


# ------------------------------------------------------------------------------- sizes
def test_typed_input_sizes():
    assert INP["sizes"]["fit_tokens"] == 31_000
    main = node("sizes", "main", "definitions")
    assert main["all"]["item_times"] == 20 * 3 * 4 * 6
    assert node("sizes", "heldout", "definitions", "all", "item_times") == 10 * 3 * 4 * 18
    assert MACROS["ExRevSizeMainCloseFriendDictsMax"] == "63,375"
    assert MACROS["ExRevSizeMainVipSenderColumnsNoBodyMax"] == "47,326"
    for d in ("CloseFriend", "VipSender"):
        for v in ("Dicts", "DictsNoBody", "ColumnsNoBody"):
            assert MACROS[f"ExRevSizeMain{d}{v}Fit"] == "0.0\\%", (d, v)
    for d in ("ImportantContact", "WorkingHours", "ActiveProject", "OverdueTask"):
        assert MACROS[f"ExRevSizeMain{d}DictsFit"] == "100.0\\%", d
    assert len([k for k in MACROS if k.startswith("ExRevSizeHeldout")]) == 19 * 10


def test_typed_sizes_counts_records_as_the_verification_did():
    recs = [
        {"source": "email", "kind": "", "object_id": "e1", "body": "x y", "text": "t"},
        {"source": "notes", "kind": "", "object_id": "n1", "body": "z", "text": "u"},
    ]
    sizes = rv.typed_sizes(recs, lambda s, v: len(s))  # one "token" per character
    b0, b1 = (json.dumps({k: v for k, v in r.items() if k != "text"}, separators=(",", ":"))
              for r in recs)  # fmt: skip
    assert sizes["dicts"] == 2 + len(b0) + 1 + len(b1) + 1
    c0 = json.dumps({"source": "email", "kind": "", "object_id": "e1"}, separators=(",", ":"))
    assert sizes["dicts_no_body"] == 2 + len(c0) + 1 + len(b1) + 1


# ------------------------------------------------------------------------------- fixed
def test_the_estimates_that_the_design_fixes():
    assert not [k for k in MACROS if "Fixed" in k]  # no macro counts these rows
    rows = node("fixed", "rows")
    assert len(rows) == len(rv.FIXED) == 6
    assert rv.fixed(ROOT, SOURCES)["rows"] == rows  # recomputed from the staged estimates
    assert all(r["fixed"] and r["row_found"] for r in rows)
    assert sum(r["estimates"] for r in rows) == 6 + 2 + 18 + 6 + 6 + 1


# ------------------------------------------------------------------------- generated file
@pytest.mark.paper
def test_committed_numbers_carry_every_macro_of_the_block():
    numbers = NUMBERS.read_text()
    for name, value in MACROS.items():
        assert pn._macro(name, value) in numbers, name

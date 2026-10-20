"""The analyses of `analysis.ex_round4_detection`: read from the staged analyses, detection since
the fork reproduces the published Eq. (1) terms and the citation key's misses by cause, detection
from the latest change reproduces the clock analysis for the source key, the B x D of block H and
the denominators hold, the pinned values hold, and the committed numbers.tex carries every macro of
the block."""

import json
import re
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import ex_round4_detection as det
from memshift.analysis import paper_numbers as pn
from memshift.analysis.ex_e1_paper import fmt
from memshift.analysis.paper_data import MISSING
from memshift.runinfo import file_sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = det.load(ROOT, SOURCES)
EMPTY = det.load(ROOT, {})
NUMBERS = ROOT / "paper/generated/numbers.tex"  # read by the tests marked `paper` only
RAW = det.numbers(INP)
MACROS = pn.with_estimates(RAW)  # with the `...Est` macros paper_numbers adds
BRANCHES = (("src", "Src"), ("def", "Def"), ("orphan", "Orphan"))
CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))
CELLS = (("D0B1", "DzeroBone"), ("D1B1", "DoneBone"))


def committed(name: str) -> str:
    """The value of a macro in the committed numbers.tex."""
    m = re.search(rf"\\newcommand{{\\{name}}}{{(.*)}}\n", NUMBERS.read_text())
    assert m, name
    return m.group(1)


def shown(value: str) -> str:
    """A macro value as numbers.tex prints it (an interval of zero width is left out)."""
    m = re.fullmatch(r"\\newcommand\{\\X\}\{(.*)\}", pn._macro("X", value))
    assert m, value
    return m.group(1)


def node(part: str, *keys: str) -> Any:
    x: Any = INP[part]
    for k in keys:
        x = x[k]
    return x


# ------------------------------------------------------------------------------------ stage
def test_results_sources_name_the_stage_and_it_is_staged():
    assert SOURCES["ex_round4_detection"] == det.STAGE_DIR == "runs/stage10_ex_round4_detection"
    assert det.STAGE_KEY == "ex_round4_detection"
    for key in det.FILES:
        assert INP[key] is not None, key
        assert INP[key]["status"] == det.STATUS, key


def test_stage_manifest_lists_its_files_producers_and_sources():
    stage = ROOT / det.STAGE_DIR
    man = json.loads((stage / "manifest.json").read_text())
    assert man["run_id"] == "stage10_ex_round4_detection"
    assert man["experiment"] == "merged-ex_round4_detection"
    files = {f"analysis/{name}" for name in det.FILES.values()}
    assert set(man["files_sha256"]) == files == set(man["producers"])
    for rel, digest in man["files_sha256"].items():
        assert file_sha256(stage / rel) == digest, rel
    assert man["producers"] == det.PRODUCERS and man["note"] == det.NOTE
    want = [s.removeprefix("runs/") for s in det.source_runs(SOURCES)]
    assert len(want) == len(set(want)) == len(det.SOURCE_KEYS) + 10  # the stages, then H runs
    assert [s["run_id"] for s in man["sources"]] == want
    for s in man["sources"]:
        if "manifest_sha256" in s:
            got = file_sha256(ROOT / "runs" / s["run_id"] / "manifest.json")
            assert got == s["manifest_sha256"], s["run_id"]
    for name in det.FILES.values():
        text = (stage / "analysis" / name).read_text()
        assert "/Users/" not in text and "/home/" not in text, name


def test_the_generator_hook_calls_this_block():
    numbers = (ROOT / "src/memshift/analysis/paper_numbers.py").read_text()
    assert "ex_round4_detection.numbers(ex_round4_detection.load(root, sources))" in numbers


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = det.numbers(EMPTY)
    assert len(macros) > 500
    assert set(macros) == set(RAW)
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only
        assert name.startswith("ExDet"), name
        assert value == MISSING, (name, value)


def test_every_macro_has_a_number_when_staged():
    assert not [k for k, v in MACROS.items() if v == MISSING]


# ------------------------------------------------------------------- since the fork, as before
@pytest.mark.paper
def test_since_fork_reproduces_the_published_terms():
    for study, st in (("main", "Main"), ("rep", "Rep")):
        for b, bt in BRANCHES:
            for cls, ct in CLASSES:
                for key, kt in det.KEYS:
                    for cell, cellt in CELLS:
                        t = node("event", study, "terms", b, cls, key, "cells", cell, "since_fork")
                        name = f"ExDec{st}{bt}{ct}{cellt}"
                        assert shown(fmt(t["m"])) == committed(f"{name}Missed{kt}"), (name, kt)
                        assert shown(fmt(t["r"])) == committed(f"{name}Repair{kt}"), (name, kt)
                        assert shown(fmt(t["U"])) == committed(f"{name}Update{kt}"), (name, kt)


@pytest.mark.paper
def test_latest_change_reproduces_the_clock_analysis_for_the_source_key():
    for st in ("Main", "Rep"):
        for _, bt in BRANCHES:
            for _, ct in CLASSES:
                for _, cellt in CELLS:
                    name = f"{st}{bt}{ct}{cellt}"
                    m, r = MACROS[f"ExDet{name}Missed"], MACROS[f"ExDet{name}Repair"]
                    assert shown(m) == committed(f"ExRevClock{name}MissedLast"), name
                    assert shown(r) == committed(f"ExRevClock{name}RepairLast"), name


@pytest.mark.paper
def test_cited_misses_since_the_fork_reproduce_the_published_split():
    for _, bt in BRANCHES:
        for part in ("Pooled", "Shift", "Drift"):
            for _, ct in CLASSES:
                name = f"{bt}{part}{ct}"
                assert MACROS[f"ExDetCited{name}Items"] == committed(f"ExRevCited{name}Items")
                assert MACROS[f"ExDetCited{name}MissedFork"] == committed(f"ExRevCited{name}Missed")
                share = committed(f"ExRevCited{name}MissShare")
                assert shown(MACROS[f"ExDetCited{name}MissShareFork"]) == share, name


def test_update_and_b0_cells_do_not_depend_on_the_count():
    for study in ("main", "rep"):
        for b, _ in BRANCHES:
            for cls, _ in CLASSES:
                for key, _ in det.KEYS:
                    for cell, _ in CELLS:
                        c = node("event", study, "terms", b, cls, key, "cells", cell)
                        assert c["since_fork"]["U"] == c["latest_change"]["U"]
                        assert c["since_fork"]["n"] == c["latest_change"]["n"]
                        # counted from the latest change, a key misses at least as much
                        assert c["latest_change"]["miss"] >= c["since_fork"]["miss"]


# --------------------------------------------------------------------------- pinned values
def test_the_citation_key_from_the_latest_change():
    # it registers every redefinition but misses a later move of 54 of the 222 changes it caused
    assert MACROS["ExDetCitedDefShiftAllItems"] == "222"
    assert MACROS["ExDetCitedDefShiftAllMissedFork"] == "0"
    assert MACROS["ExDetCitedDefShiftAllMissedLast"] == "54"
    assert MACROS["ExDetCitedDefDriftAllMissedLast"] == "127"
    assert MACROS["ExDetMainSrcAllDoneBoneMissedCitedEst"] == "16.9\\%"
    assert MACROS["ExDetMainDefAllDoneBoneMissedCitedEst"] == "32.4\\%"
    # after a redefinition every wrong D=1 answer on a changed definition item is a missed detection
    assert MACROS["ExDetMainDefDefinitionDoneBoneRepairCitedEst"] == "100.0\\%"
    assert MACROS["ExDetRepDefDefinitionDoneBoneRepairCitedEst"] == "100.0\\%"
    # the key's detection does not depend on D
    for _, bt in BRANCHES:
        for _, ct in CLASSES:
            for kt in ("", "Cited"):
                assert (
                    MACROS[f"ExDetMain{bt}{ct}DzeroBoneMissed{kt}"]
                    == MACROS[f"ExDetMain{bt}{ct}DoneBoneMissed{kt}"]
                )


def test_the_source_key_from_the_latest_change():
    assert MACROS["ExDetMainSourceMissedMax"] == "0.6\\%"
    assert MACROS["ExDetMainDefDefinitionDoneBoneMissedEst"] == "0.3\\%"
    assert MACROS["ExDetMainSrcDefinitionDzeroBoneRepairEst"] == "16.3\\%"
    assert MACROS["ExDetMainSrcDefinitionDoneBoneRepairEst"] == "99.7\\%"
    # Update does not depend on the count: Figure 1's bars
    assert MACROS["ExDetMainDefDefinitionDoneBoneUpdate"] == "99.7\\% [99.2, 100.0]"
    assert MACROS["ExDetMainDefDefinitionDzeroBzeroUpdate"] == "7.5\\% [4.2, 10.8]"


def test_the_denominators_of_update_and_stability():
    want = {"Src": ("1,168", "1,168"), "Def": ("558", "558"), "Orphan": ("1,767", "1,683")}
    for bt, (upd, stab) in want.items():
        assert MACROS[f"ExDetMain{bt}UpdateItems"] == upd
        assert MACROS[f"ExDetMain{bt}StabilityItems"] == stab
    # the reader was asked exactly the seeded sample, for every HoldMem cell and D_m
    for study in ("main", "rep"):
        for cells in node("event", study, "item_times").values():
            for c in cells.values():
                assert c["unchanged_asked"] == c["unchanged_sample"] <= c["unchanged_all"]
    assert set(node("event", "main", "item_times_dm", "src")) == set(det.DM_SYSTEMS)


def test_the_oracle_key_keeps_the_pattern():
    assert MACROS["ExDetKeyWorlds"] == "10"
    assert MACROS["ExDetKeyOracleSrcDefinitionBxDEst"] == "$+$23.6\\,pp"
    assert MACROS["ExDetKeyOracleDefDefinitionBxDEst"] == "$+$38.8\\,pp"
    assert MACROS["ExDetKeyOracleOrphanDefinitionBxDEst"] == "$+$41.6\\,pp"
    assert MACROS["ExDetKeySourceSrcDefinitionBxDEst"] == "$+$25.5\\,pp"
    assert MACROS["ExDetKeySourceOrphanDefinitionBxDEst"] == "$+$40.9\\,pp"
    assert MACROS["ExDetKeyOracleSrcDefinitionBDzeroEst"] == "$+$6.7\\,pp"
    # the range of its effect at D=0 over the three shifts
    assert MACROS["ExDetKeyOracleDefinitionBDzeroMin"] == "$-$1.5\\,pp"
    assert MACROS["ExDetKeyOracleDefinitionBDzeroMax"] == "$+$6.7\\,pp"


@pytest.mark.paper
def test_the_claude_writers_key_effect():
    assert MACROS["ExDetWriterWorlds"] == committed("RThreeHSonnetWorlds")
    for _, wt in det.WRITER_TAGS:
        for _, bt in BRANCHES:
            got = MACROS[f"ExDetWriter{wt}{bt}DefinitionBxD"]
            assert got == committed(f"RThreeH{wt}{bt}DefinitionDoneBxD"), (wt, bt)
    want = {"Sonnet": ("32.7", "38.8", "32.5"), "Opus": ("30.3", "38.8", "32.5")}
    for wt, values in want.items():
        for (_, bt), v in zip(BRANCHES, values, strict=True):
            assert MACROS[f"ExDetWriter{wt}{bt}DefinitionBDzeroEst"] == f"$+${v}\\,pp"
    assert MACROS["ExDetWriterDefinitionBDzeroMin"] == "$+$30.3\\,pp"
    assert MACROS["ExDetWriterDefinitionBDzeroMax"] == "$+$38.8\\,pp"


# ------------------------------------------------------------------------- generated file
@pytest.mark.paper
def test_committed_numbers_carry_every_macro_of_the_block():
    numbers = NUMBERS.read_text()
    for name, value in MACROS.items():
        assert pn._macro(name, value) in numbers, name

"""Paper numbers of the control cells (`analysis.controls_paper`): a stage that is not staged
renders only MISSING markers, and a staged one carries the estimates and counts the analysis
produced."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import controls as ctl
from memshift.analysis import controls_paper as cp
from memshift.analysis.paper_data import MISSING, camel
from memshift.analysis.paper_numbers import _macro, with_estimates
from tests.test_controls import Builder, ev

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())


@pytest.fixture(scope="module")
def staged() -> dict:
    return ev(Builder().table())


@pytest.fixture(scope="module")
def partial() -> dict:
    b = Builder()
    b.drop |= {"zeroDm", "D1B0"}
    return ev(b.table())


# ---------------------------------------------------------------------------------- macros
def test_unstaged_control_cells_render_only_missing_markers():
    macros = cp.numbers(None)
    assert len(macros) > 450
    for name, value in macros.items():
        _macro(name, value)  # letters only
        assert name.startswith("Ctl") and value == MISSING, (name, value)


def test_every_contrast_and_check_has_a_macro_name_and_no_threshold_outcome_has_one():
    names = set(cp.numbers(None))
    for key in ctl.contrast_keys():
        assert f"Ctl{camel(key)}" in names, key
    for tag in ("Hash", "Probes", "Anchor", "Audit", "ScopeCited"):
        assert f"CtlCheck{tag}Pass" in names
    assert "CtlChecksPassed" in names and "CtlWorlds" in names
    for pred in ctl.CONTROLS:
        for suffix in ("Verdict", "Hits", "Total", "Need", "Count"):
            assert f"Ctl{camel(pred)}{suffix}" not in names
    assert not [n for n in names if re.search(r"Summary|Strong|Presentation|Reading|AsReg", n)]


def test_the_full_access_control_has_its_macros_for_src_and_def_only():
    names = set(cp.numbers(None))
    for name in (
        "CtlFRepSrcDefinition",
        "CtlFRepDefDefinition",
        "CtlFRepmechSrcDefinition",
        "CtlFRepSrcUntruncated",
        "CtlFRepmechDefUntruncated",
        "CtlFGainRepSrcDefinition",
        "CtlFGainUpdDefDefinition",
        "CtlFGainRepSrcUntruncated",
        "CtlFullLimit",
        "CtlFullTruncatedAll",
        "CtlFullTruncatedShareSrc",
        "CtlFullWritesDef",
        "CtlFullCloseFriendTruncatedSrc",
        "CtlFullCloseFriendMessagesTruncatedSrc",
        "CtlFullVipSenderEmailDroppedDef",
        "CtlFullVipSenderContactsTruncatedAll",
        "CtlFullWorkingHoursNotesTruncatedSrc",
        "CtlFullWhichTruncated",
    ):
        assert name in names, name
    # the orphan branch was not run: no contrast and no note macro for it
    assert not [n for n in names if n.startswith(("CtlFRep", "CtlFGain")) and "Orphan" in n]
    assert not [n for n in names if n.startswith("CtlFull") and n.endswith("Orphan")]
    assert len([n for n in names if n.startswith("CtlFull")]) > 100


def test_staged_macros_carry_estimates_intervals_and_counts(staged):
    m = with_estimates(cp.numbers(staged))
    # an estimate with its interval, and the estimate alone
    assert m["CtlZUpdSrcDefinition"] == "$+$80.0\\,pp [$+$80.0, $+$80.0]"
    assert m["CtlZUpdSrcDefinitionEst"] == "$+$80.0\\,pp"
    assert m["CtlGUpdDzeroSrcAll"].startswith("0.0\\,pp")
    assert m["CtlRRatioSrcDefinition"].startswith("0.65")  # (0.80 - 0.50) / (0.96 - 0.50)
    # the D0full control: the gap with its interval, the sensitivity analysis
    assert m["CtlFRepSrcDefinitionEst"] == "$+$66.0\\,pp"
    assert m["CtlFRepmechDefDefinitionEst"] == "$+$66.0\\,pp"
    assert m["CtlFGainRepSrcDefinitionEst"] == "$+$16.0\\,pp"
    assert m["CtlFRepSrcUntruncatedEst"].startswith("$+$6")  # 94 - 28 over 98 items
    assert "CtlFRepOrphanDefinition" not in m
    assert m["CtlWorlds"] == "3"
    # the reported quantities
    assert m["CtlProgOkRaAll"].startswith("100.0\\%") and m["CtlProgOkRatioAll"].startswith("1.00")
    assert m["CtlThinkWritesAll"] == "90" and m["CtlThinkLimit"] == "8,192"
    assert m["CtlThinkUnreadableAll"].startswith("0.0\\%")
    assert m["CtlRaLineageFalseOrphanAll"].startswith("10.0\\%")
    assert m["CtlOverActiveProjectRepairedRaOrphan"] == "6"  # worlds x checkpoints
    assert m["CtlOverActiveProjectWithdrawnRaOrphan"] == "0"
    # the truncation counts of the D0full writer, pooled, per definition and per source
    assert m["CtlFullLimit"] == "25,000" and m["CtlFullWritesAll"] == "72"
    assert (m["CtlFullWritesSrc"], m["CtlFullTruncatedAll"], m["CtlFullBySourceAll"]) == (
        "36", "24", "24",
    )  # fmt: skip
    assert m["CtlFullNewestAll"] == "0" and m["CtlFullDroppedSrc"] == "312"
    assert m["CtlFullTruncatedShareAll"].startswith("33.3\\%")
    assert (m["CtlFullTokensMedianAll"], m["CtlFullTokensMaxAll"]) == ("9,000", "24,900")
    assert m["CtlFullCloseFriendTruncatedSrc"] == "6" and m["CtlFullCloseFriendDroppedSrc"] == "72"
    assert m["CtlFullCloseFriendMessagesTruncatedAll"] == "12"
    assert m["CtlFullVipSenderEmailDroppedDef"] == "240"
    assert m["CtlFullVipSenderContactsDroppedDef"] == "0"  # the small source is given whole
    assert m["CtlFullActiveProjectTasksTruncatedAll"] == "0"
    # per definition and declared source after S_src and S_def: writes, writes cut, records left out
    for b in ("Src", "Def"):
        assert (
            m[f"CtlFullCloseFriendWrites{b}"],
            m[f"CtlFullCloseFriendMessagesTruncated{b}"],
            m[f"CtlFullCloseFriendMessagesDropped{b}"],
        ) == ("6", "6", "72")
        assert (
            m[f"CtlFullVipSenderWrites{b}"],
            m[f"CtlFullVipSenderEmailTruncated{b}"],
            m[f"CtlFullVipSenderEmailDropped{b}"],
        ) == ("6", "6", "240")
        assert (
            m[f"CtlFullVipSenderContactsTruncated{b}"],
            m[f"CtlFullVipSenderContactsDropped{b}"],
        ) == (
            "0",
            "0",
        )
        assert (
            m[f"CtlFullActiveProjectWrites{b}"],
            m[f"CtlFullActiveProjectTasksTruncated{b}"],
            m[f"CtlFullActiveProjectTasksDropped{b}"],
        ) == ("6", "0", "0")
    assert m["CtlFullWhichTruncated"] == "close\\_friend, vip\\_sender"


NOTHING_TO_SHOW = re.compile(
    r"CtlChecks?\w*"  # the checks of a run: the synthetic table has none
    r"|CtlRa\w*Free"  # the synthetic items have no free-text class
    r"|CtlRaFinalUndeclaredOfFailed\w+"  # and no failed item
    r"|Ctl[GZ]Stab\w*Definition"  # and no definition whose gold does not change
)


def test_every_estimate_macro_has_a_number_when_everything_ran(staged):
    macros = cp.numbers(staged)
    missing = sorted(k for k, v in macros.items() if v == MISSING)
    assert all(NOTHING_TO_SHOW.fullmatch(k) for k in missing), missing[:10]
    for key in ctl.contrast_keys():
        if not NOTHING_TO_SHOW.fullmatch(f"Ctl{camel(key)}"):
            assert macros[f"Ctl{camel(key)}"] != MISSING, key


def test_a_missing_cell_leaves_its_contrasts_missing(partial):
    m = cp.numbers(partial)
    assert m["CtlZUpdSrcDefinition"] == MISSING  # the zero system's D_m cell was dropped
    assert m["CtlGUpdDzeroSrcAll"] != MISSING
    assert m["CtlFRepSrcDefinition"] != MISSING  # D1 and D0full were run in `partial`
    assert "CtlZUpdSrcDefinitionEst" not in with_estimates(m)  # `with_estimates` invents none


def test_macro_values_compile_as_text(staged, tmp_path):
    if shutil.which("pdflatex") is None:
        pytest.skip("no LaTeX")
    macros = cp.numbers(staged)
    (tmp_path / "numbers.tex").write_text(
        "\n".join(_macro(k, v) for k, v in sorted(macros.items()))
    )
    lines = "".join(f"\\noindent\\texttt{{{k}}} {{\\{k}}}\\par\n" for k in sorted(macros))
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[11pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage{amsmath,newtxtext,newtxmath}",
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


def test_check_macros_mark_each_check():
    anchors = {"r": {"holdmem_D0B1_source": {"share": 0.05, "answers": 100}}}
    audit_cell = {"pass": True, "branches": {"src": {"definition": {"pass": True}}}}
    view = {
        "n_worlds": 3,
        "checks": {
            "hash": {"pass": True},
            "probe_sets": {"pass": True},
            "anchors": {"pass": False, "runs": anchors},
            "audit": {"pass": True, "cells": {"holdmem_D0B1_gold": audit_cell}},
            "scope_cited": {"pass": None, "reason": "missing"},
            "passed": 3,
            "failed": 1,
            "not_evaluated": 1,
        },
    }
    m = cp.numbers(view)
    assert m["CtlCheckAnchorPass"] == "fail" and m["CtlCheckScopeCitedPass"] == "not evaluated"
    assert m["CtlCheckAnchorWorst"].startswith("5.0\\%") and m["CtlChecksPassed"] == "3 of 5"


# ------------------------------------------------------------------------------------- sources
def test_results_sources_name_the_control_stage_and_it_is_not_staged_yet():
    assert SOURCES["ctl"] == "runs/stage5_controls"
    if not (ROOT / SOURCES["ctl"] / "analysis" / "ctl_metrics.json").exists():
        assert cp.load(ROOT, SOURCES) is None


def test_the_number_generator_hook_adds_the_control_macros():
    src = (ROOT / "src/memshift/analysis/paper_numbers.py").read_text()
    assert "controls_paper.numbers(controls_paper.load(root, sources))" in src

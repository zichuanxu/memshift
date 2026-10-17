"""Paper numbers and tables of the analyses requested by the third review round
(`analysis.ex_round3`, prereg/decisions.md 2026-10-16): read from the staged round-3 analyses, they
reproduce the published values they build on, and the committed generated files are current."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from memshift.analysis import ex_round3 as r3
from memshift.analysis import paper_numbers as pn
from memshift.analysis.ex_e1_paper import UNDEFINED
from memshift.analysis.paper_data import MISSING
from memshift.runinfo import file_sha256
from tests.test_paper_stage4 import check_tex

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
INP = r3.load(ROOT, SOURCES)
EMPTY = r3.load(ROOT, {})
NUMBERS = ROOT / "paper/generated/numbers.tex"  # read by the tests marked `paper` only
MACROS = r3.numbers(INP)
BRANCHES = ("Src", "Def", "Orphan")


def committed(name: str) -> str:
    """The value of a macro in the committed numbers.tex."""
    m = re.search(rf"\\newcommand{{\\{name}}}{{(.*)}}\n", NUMBERS.read_text())
    assert m, name
    return m.group(1)


def est(value: str) -> float:
    """The estimate of a formatted macro value, as a number (pp or percent)."""
    head = value.split(" [", 1)[0].replace("$-$", "-").replace("$+$", "")
    return float(head.replace("\\,pp", "").replace("\\%", ""))


# ------------------------------------------------------------------------------------ stage
def test_results_sources_name_the_round3_stage_and_it_is_staged():
    assert SOURCES["ex_round3"] == r3.STAGE_DIR == "runs/stage6_ex_round3"
    for key in r3.FILES:
        assert INP[key] is not None, key


def test_stage_manifest_lists_its_files_producers_and_sources():
    stage = ROOT / r3.STAGE_DIR
    man = json.loads((stage / "manifest.json").read_text())
    assert man["run_id"] == "stage6_ex_round3" and man["experiment"] == "merged-ex_round3"
    assert man["analysis_code"]["dirty"] is False
    files = {f"analysis/{name}" for name in r3.FILES.values()}
    assert set(man["files_sha256"]) == files == set(man["producers"])
    for rel, digest in man["files_sha256"].items():
        assert file_sha256(stage / rel) == digest, rel
    assert man["producers"] == r3.PRODUCERS and man["note"] == r3.NOTE
    want = [s.removeprefix("runs/") for s in r3.source_runs(SOURCES)]
    assert want[:3] == [SOURCES[k].removeprefix("runs/") for k in r3.SOURCE_KEYS]
    assert len(want) == len(set(want)) == 3 + 90 + 10  # the stages, G-b, then H-a and H-c
    assert [s["run_id"] for s in man["sources"]] == want
    for s in man["sources"]:
        assert file_sha256(ROOT / "runs" / s["run_id"] / "manifest.json") == s["manifest_sha256"]
    # writers.json names its inputs as its producer does, relative to the root
    res = json.loads((stage / "analysis" / r3.FILES["writers"]).read_text())
    assert res["inputs"] == {
        "programs": list(r3.WRITER_PROGRAMS),
        "defwriters": list(r3.WRITER_DEFWRITERS),
        "compare": [SOURCES["e1"], SOURCES["c"]],
    }
    assert res["code"] == man["analysis_code"]


def test_the_generator_hooks_call_this_block():
    numbers = (ROOT / "src/memshift/analysis/paper_numbers.py").read_text()
    tables = (ROOT / "src/memshift/analysis/paper_tables.py").read_text()
    assert "ex_round3.numbers(ex_round3.load(root, sources))" in numbers
    assert "ex_round3.tables(ex_round3.load(a.root, sources))" in tables


# ------------------------------------------------------------------------------------ macros
def test_no_stage_staged_renders_only_missing_markers():
    macros = r3.numbers(EMPTY)
    assert len(macros) > 1000
    assert set(macros) == set(MACROS)
    for name, value in macros.items():
        pn._macro(name, value)  # letters-only
        assert name.startswith("ExRThree"), name
        assert value == MISSING, (name, value)


def test_every_macro_has_a_number_when_staged():
    assert not [k for k, v in MACROS.items() if v == MISSING]
    # no other item changes after a redefinition, and a JSON arm exists for the authored D=1 only
    assert MACROS["ExRThreeNatDefShiftOtherItems"] == "0"
    assert MACROS["ExRThreeNatDefShiftOtherBxD"] == UNDEFINED
    assert "ExRThreeSRDmSrcDefinitionJsonBoneU" not in MACROS
    assert MACROS["ExRThreeSRMainSrcDefinitionDoneBoneUAbstain"] == UNDEFINED  # no gold abstain


@pytest.mark.paper
def test_store_and_reader_parts_add_up_to_the_published_interaction():
    for b in BRANCHES:
        for c in ("Definition", "Other", "All"):
            bxd = MACROS[f"ExRThreeSRMain{b}{c}BxD"]
            assert bxd == committed(f"ExClsMain{b}{c}BxD"), (b, c)
            parts = est(MACROS[f"ExRThreeSRMain{b}{c}StorePart"]) + est(
                MACROS[f"ExRThreeSRMain{b}{c}ReaderPart"]
            )
            assert abs(parts - est(bxd)) <= 0.11, (b, c)
            dm = MACROS[f"ExRThreeSRDm{b}{c}BxD"]
            assert dm == committed(f"ExClsDm{b}{c}BxD"), (b, c)
    # on definition items the interaction is stored-value repair: no reader part
    assert MACROS["ExRThreeSRMainSrcDefinitionReaderPart"] == "0.0\\,pp [$-$0.9, $+$0.9]"
    assert MACROS["ExRThreeSRMainDefDefinitionReaderPart"] == "0.0\\,pp [0.0, 0.0]"
    assert MACROS["ExRThreeSRMainDefOtherReaderPart"] == "$-$21.1\\,pp [$-$31.7, $-$10.8]"
    # the JSON arm serves the D=1 store: the same stored state, a different reader
    for b in BRANCHES:
        for c in ("Definition", "Other"):
            one, json_ = (MACROS[f"ExRThreeSRMain{b}{c}{x}BoneRStore"] for x in ("Done", "Json"))
            assert one == json_, (b, c)
    assert MACROS["ExRThreeSRMainDefDefinitionJsonBoneU"].startswith("54.4\\% [")
    # the right stores the reader lost on the other items are almost all gold abstain
    assert MACROS["ExRThreeSRMainShiftsOtherDoneBoneLost"] == "283"
    assert MACROS["ExRThreeSRMainShiftsOtherDoneBoneLostAbstain"] == "281"
    assert MACROS["ExRThreeSRMainDefOtherDoneBoneUAnswer"] == "100.0\\% [100.0, 100.0]"


@pytest.mark.paper
def test_composition_reproduces_the_attribution_of_round2():
    for b in BRANCHES:
        for p in ("Shift", "Drift"):
            for c in ("Definition", "Other", "All"):
                name = f"{b}{p}{c}"
                assert MACROS[f"ExRThreeNat{name}Items"] == committed(f"ExAttr{name}Items")
                if MACROS[f"ExRThreeNat{name}Items"] != "0":
                    assert MACROS[f"ExRThreeNat{name}BxD"] == committed(f"ExAttr{name}BxD")
    assert MACROS["ExRThreeNatSrcShiftDefinitionShare"] == "14.1\\% [9.5, 19.0]"
    assert MACROS["ExRThreeNatOrphanShiftDefinitionShare"] == "3.5\\% [1.6, 5.5]"
    # the all-item value is the share-weighted sum of the class values
    for b in ("Src", "Orphan"):
        total = sum(
            est(MACROS[f"ExRThreeNat{b}ShiftContribution{c}"]) for c in ("Definition", "Other")
        )
        assert abs(total - est(MACROS[f"ExRThreeNat{b}ShiftAllBxD"])) <= 0.11, b
    assert MACROS["ExRThreeNatAnswersAfterFork"] == "0"  # no committed run answers it
    assert INP["nat"]["natural_counterfactual"] is None


@pytest.mark.paper
def test_deletion_taxonomy_accounts_for_the_whole_loss():
    acc = INP["dm"]["B1"]["populations"]["changed_abstain"]["all"]["accounting_vs_D0"]
    shares = acc["loss_share_of_total"]
    assert abs(sum(shares[k]["est"] for k in "abcde") - 1) < 1e-9
    assert MACROS["ExRThreeDelLossShareVsDzeroA"] == "68.9\\% [61.1, 76.6]"
    assert MACROS["ExRThreeDelLossShareVsDzeroD"] == "30.5\\% [22.9, 38.2]"
    assert MACROS["ExRThreeDelLossShareVsDzeroC"] == "0.0\\% [0.0, 0.0]"
    assert MACROS["ExRThreeDelLossVsDzero"] == "35.8\\,pp [31.2, 39.9]"
    assert MACROS["ExRThreeDelItems"] == "1,365"
    # the repair contrast C1 after the deletion is the published one
    assert MACROS["ExRThreeDelCOneAll"] == committed("SynthExCOneOrphan")
    # the reader errors are as frequent at D=1, and every stored replacement is a None at D=1
    assert MACROS["ExRThreeDelDmMinusDoneD"] == "$+$1.8\\,pp [$-$2.3, $+$5.3]"
    assert MACROS["ExRThreeDelValueStored"] == MACROS["ExRThreeDelValueStoredDoneNone"] == "394"
    # task programs return the last-listed live task's due date, on deleted and on live records
    for group in ("Deleted", "Live"):
        last = MACROS[f"ExRThreeDelProgTask{group}WrongLastListed"]
        wrong = INP["dm"]["B1"]["program_by_kind"]["task_due"][group.lower()]["wrong_with_value"]
        assert int(last) == wrong > 0, group
    assert MACROS["ExRThreeDelProgTaskLiveWrong"] == "83.3\\% [71.8, 93.2]"
    # the one failing definition program: a syntax error, one request for all 20 worlds
    assert MACROS["ExRThreeDelFailDefinitionSyntaxSlots"] == "20"
    assert MACROS["ExRThreeDelFailDefinitionSyntaxRequests"] == "1"
    assert MACROS["ExRThreeDelTzeroWorkingHoursExecutable"] == "0.0\\%"


@pytest.mark.paper
def test_view_reproduces_the_registered_control_and_isolates_working_hours():
    for b in ("Src", "Def"):
        assert MACROS[f"ExRThreeView{b}DefinitionUpdateVsDm"] == committed(
            f"CtlZVsDones{b}Definition"
        )
    assert MACROS["ExRThreeViewPooledAllUpdateVsDm"] == "$-$6.6\\,pp [$-$9.2, $-$4.2]"
    assert MACROS["ExRThreeViewPooledAllStabilityVsDm"] == "$-$21.3\\,pp [$-$28.0, $-$14.8]"
    assert MACROS["ExRThreeViewPooledDefinitionStabilityVsDm"].startswith("$+$43.0\\,pp [")
    # the view's Stability edge on definition items is working hours, whose stored item fails
    assert MACROS["ExRThreeViewTypeWorkingHoursStabilityItems"] == "45"
    assert MACROS["ExRThreeViewTypeWorkingHoursStabilityLevelDm"] == "0.0\\% [0.0, 0.0]"
    assert MACROS["ExRThreeViewTypeWorkingHoursStabilityLevelView"] == "100.0\\% [100.0, 100.0]"
    without = MACROS["ExRThreeViewWithoutWorkingHoursDefinitionStabilityVsDm"]
    assert without == "$-$3.6\\,pp [$-$10.3, 0.0]"
    assert MACROS["ExRThreeViewServePooledVsDm"] == "$+$246 [$+$213, $+$289]"


# ------------------------------------------------------------------------------------ tables
LABELS = {
    "tab_main_store.tex": "tab:main",
    "tab_store_reader.tex": "tab:store-reader",
    "tab_store_reader_bxd.tex": "tab:store-reader-bxd",
    "tab_dm_deletion.tex": "tab:dm-deletion",
    "tab_view.tex": "tab:view",
}


def test_tables_are_wellformed_with_data_and_without():
    for inp in (INP, EMPTY):
        assert set(r3.tables(inp)) == set(LABELS)
        for name, tex in r3.tables(inp).items():
            check_tex(tex)
            assert tex.startswith("\\begin{table}") and f"\\label{{{LABELS[name]}}}" in tex
    for name, tex in r3.tables(EMPTY).items():
        assert MISSING in tex, name
    for name, tex in r3.tables(INP).items():
        assert MISSING not in tex, name


def _rows(tex: str) -> list[list[str]]:
    return [
        re.split(r"\s*&\s*", line[: -len(r" \\")])
        for line in tex.splitlines()
        if line.endswith(r" \\") and "&" in line and not line.startswith("&")
    ]


@pytest.mark.paper
def test_main_table_is_the_round2_table_without_the_cited_key():
    old = _rows((ROOT / "paper/generated/tab_main.tex").read_text())
    new = _rows(r3.tab_main_store(INP))
    assert len(old) == len(new) == 6
    for o, n in zip(old, new, strict=True):
        assert n == o[:-1]  # r0, r1, a0, a1 and B x D; the cited key's m1 is left out
    assert new[0][2:8] == [
        "29",
        "16.2",
        "99.1",
        "10.2",
        "65.8",
        "$+$27.3\\,{\\scriptsize [$+$21.6,\\,$+$32.6]}",
    ]


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no LaTeX")
def test_tables_compile_at_the_paper_width(tmp_path):
    names = []
    for name, tex in r3.tables(INP).items():
        (tmp_path / name).write_text(tex)
        names.append(name)
    (tmp_path / "t.tex").write_text(
        "\n".join(
            [
                r"\documentclass[10pt]{article}\usepackage[T1]{fontenc}",
                r"\usepackage[textwidth=5.5in,textheight=9in]{geometry}",
                r"\usepackage{amsmath,newtxtext,newtxmath,xcolor,booktabs}",
                r"\newcommand{\Ssrc}{S_{\mathrm{src}}}\newcommand{\Sdef}{S_{\mathrm{def}}}"
                r"\newcommand{\Sdel}{S_{\mathrm{del}}}\newcommand{\holdmem}{\textsc{HoldMem}}"
                r"\newcommand{\cref}[1]{Table~1}",
                r"\begin{document}" + "".join(rf"\input{{{n}}}" for n in names) + r"\end{document}",
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
    assert "Overfull \\hbox" not in done.stdout  # every table fits 5.5in


# ------------------------------------------------------------------------- generated files
@pytest.mark.paper
def test_committed_generated_files_are_current():
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    for name, tex in r3.tables(INP).items():
        assert (ROOT / "paper/generated" / name).read_text() == head + tex, name
    numbers = NUMBERS.read_text()
    for name, value in MACROS.items():
        assert pn._macro(name, value) in numbers, name


@pytest.mark.paper
def test_the_text_uses_defined_macros():
    sections = ROOT / "paper/sections"
    text = "\n".join(p.read_text() for p in sorted(sections.glob("*.tex")))
    used = set(re.findall(r"\\(ExRThree[A-Za-z]+)", text))
    assert used, "the text cites no round-3 macro"
    defined = set(MACROS) | {f"{k}Est" for k, v in MACROS.items() if " [" in v}
    assert used <= defined, sorted(used - defined)[:5]
    # the registered natural-branch test is reported (tests/test_round3_paper.py)
    assert "NatPending" not in text + (ROOT / "paper/main.tex").read_text()
    assert "\\input{generated/tab_main_store}" in text and "generated/tab_main}" not in text

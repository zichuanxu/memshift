"""Macros and tables of the typed-input, execution-feedback and extracted-fields experiments
(`analysis.paper_round3b`): every macro prints the analysis value it names (synthetic analyses with
the structure of round3b_typed, round3b_feedback and round3b_extract, and the committed typed-input
analysis when it is there), a block left out keeps the same macro names as placeholders and a
table of the same layout, and no output carries a verdict word."""

import json
import re
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import paper_round3b as pr
from memshift.analysis.ex_round2 import DEFINITIONS
from memshift.analysis.paper_data import MISSING
from tests.memtab_check import check_memtab

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "runs" / "stage9_round3b" / "analysis"
LABELS = {
    "tab_typed.tex": r"\label{tab:typed}",
}
# panels (a) and (b) of the programs table, whose float, caption, panel (c) and labels are in the
# appendix source
PANELS = ("tab_programs.tex",)
VERDICT = re.compile(
    r"regist|verdict|predict|threshold|\bholds?\b|\bfails?\b|exploratory|post hoc|not judged"
    r"|\b[TFW][1-3]\b",
    re.IGNORECASE,
)
DEFS = [d for d, _, _ in DEFINITIONS]
WORLDS_10 = [f"w{1000 + i}" for i in range(10)]
WORLDS_20 = [f"w{1000 + i}" for i in range(20)]


# --------------------------------------------------------------------------------- formats
def _p(x: float) -> str:
    return f"{100 * x:.1f}"


def _s(x: float) -> str:
    v = round(100 * x, 1)
    return ("$+$" if v > 0 else "$-$" if v < 0 else "") + f"{abs(v):.1f}"


def pct_iv(e: dict[str, Any]) -> str:
    """As the macros print it: the estimate alone where the printed interval has no width."""
    est, lo, hi = (_p(e[k]) for k in ("est", "lo", "hi"))
    return rf"{est}\%" if lo == est == hi else rf"{est}\% [{lo}, {hi}]"


def pp_iv(e: dict[str, Any]) -> str:
    est, lo, hi = (_s(e[k]) for k in ("est", "lo", "hi"))
    return rf"{est}\,pp" if lo == est == hi else rf"{est}\,pp [{lo}, {hi}]"


def macros(text: str) -> dict[str, str]:
    return dict(re.findall(r"\\newcommand\{\\([A-Za-z]+)\}\{(.*)\}", text))


# -------------------------------------------------------------------------------- fixtures
class Est:
    """Distinct estimates, so that a macro reading the wrong path is caught."""

    def __init__(self) -> None:
        self.c: Iterator[int] = count(1)

    def __call__(self, signed: bool = False) -> dict[str, Any]:
        k = next(self.c)
        est = ((37 * k) % 997 + 1) / 1000 * (-1 if signed and k % 2 else 1)
        lo, hi = sorted((est * 0.5, est * 1.5))
        return {"est": est, "lo": lo, "hi": hi, "width": hi - lo, "n_worlds": 10, "n_items": k}


def t_fixture() -> dict[str, Any]:
    e = Est()
    groups = [*DEFS, "definition"]

    def quantities() -> dict[str, Any]:
        out = {c: e() for c in ("typed", "bm25", "d1")}
        out.update({c: e(True) for c in ("typed_minus_bm25", "d1_minus_typed", "d1_minus_bm25")})
        return out

    def arm(writes: int, unreadable: int) -> dict[str, Any]:
        return {
            "worlds": WORLDS_10,
            "validity": {
                "typed_writes": {
                    "writes": writes,
                    "unreadable": unreadable,
                    "unreadable_share": unreadable / writes,
                    "at_limit": unreadable + 1,
                    "at_limit_share": (unreadable + 1) / writes,
                }
            },
            "t0": {g: {"n": 10, "readable": e(), "correct": e(), "at_limit": 0} for g in groups},
            **{
                part: {
                    b: {
                        g: {s: quantities() for s in ("complete", "incomplete", "all")}
                        for g in groups
                    }
                    for b in ("src", "def", "pooled")
                }
                for part in ("r", "update")
            },
        }

    return {
        "block": "T",
        "arms": {"Qwen/Qwen3.5-9B": arm(472, 9), "claude-sonnet-5-5": arm(470, 0)},
        "coverage": {d: {"complete": 20 * i, "item_times": 120} for i, d in enumerate(DEFS)},
    }


def facts_fixture() -> dict[str, Any]:
    rows = {d: {"median": 1000.5 + i, "p95": 2000 + i, "max": 3000 + i} for i, d in enumerate(DEFS)}
    rows["all"] = {"median": 14062.5, "p95": 40340, "max": 44357}
    return {"c_prompt_tokens": {"rederivations": rows}}


def _exec_block(e: Est, k: int) -> dict[str, Any]:
    return {
        "n_runs": 100 * k,
        "first_attempt": e(),
        "after_feedback": e(),
        "first_attempt_failed": 10 * k,
        "after_feedback_failed": 5 * k,
        "rounds": {"0": 90 * k, "2": 5 * k},  # no run with one round: the key is left out
        "feedback_requests": 11 * k,
        "start_kept": 3 * k,
        "first_attempt_kinds": {"syntax": 7 * k, "exception": 3 * k},
        "failed_run_kinds": {"syntax": 9 * k, "timeout": k},
        "world_data": {"requests": 11 * k, "with_record_value": k},
    }


def f_fixture() -> dict[str, Any]:
    e = Est()

    def levels() -> dict[str, Any]:
        return {"feedback": e(), "dm": e(), "difference": e(True)}

    return {
        "block": "F",
        "worlds": WORLDS_20,
        "executability": {
            "definition": {
                "post": {
                    "both": _exec_block(e, 1),
                    "per_definition": {d: _exec_block(e, 2 + i) for i, d in enumerate(DEFS)},
                },
                "all": {"both": _exec_block(e, 20)},
            },
            "other": {"post": {"both": _exec_block(e, 30)}, "all": {"both": _exec_block(e, 40)}},
        },
        "r": {b: {"definition": {"pooled": levels()}} for b in ("src", "def")},
        "bxd": {
            b: {
                c: {"Dm_feedback": e(True), "Dm": e(True), "paired": levels()}
                for c in ("definition", "other")
            }
            for b in ("src", "def", "orphan")
        },
        "deletion": {"update": {"definition": {c: levels() for c in ("B0", "B1_source")}}},
    }


def w_fixture() -> dict[str, Any]:
    e = Est()
    extraction = {}
    for i, s in enumerate((*pr.SOURCES, "pooled")):
        extraction[s] = {
            "records": 100 + i,
            "record_correct": e(),
            "record_exact": e(),
            "fields": {"title": {"n": 100, "correct": 0.5 + i / 100, "exact": 0.5}},
            "rewrite": {"missed": i, "missed_share": i / 100},
        }
    return {
        "block": "W",
        "quantities": {
            "worlds": WORLDS_10,
            "extraction": extraction,
            "r": {
                b: {"pooled": {"D1sx": e(), "D1s": e(), "difference": e(True)}}
                for b in ("src", "def")
            },
            "bxd": {
                b: {
                    c: {"extracted": e(True), "typed": e(True), "difference": e(True)}
                    for c in ("definition", "other")
                }
                for b in ("src", "def")
            },
            "attribution": {
                p: {
                    b: {
                        "item_times": 198 + j,
                        "differ": 60 + j,
                        "differ_share": 0.3,
                        "share_with_error": 0.9 + j / 100,
                    }
                    for j, b in enumerate(("src", "def"))
                }
                for p in ("B0", "B1")
            },
        },
    }


@pytest.fixture(scope="module")
def full() -> dict[str, str]:
    inp = {"t": t_fixture(), "facts": facts_fixture(), "f": f_fixture(), "w": w_fixture()}
    return pr.build(inp, ["t", "f", "w"])


@pytest.fixture(scope="module")
def placeholders() -> dict[str, str]:
    return pr.build({}, [])


# ------------------------------------------------------------------------------------ tests
def test_typed_macros_equal_the_analysis(full: dict[str, str]) -> None:
    m = macros(full["round3b_numbers.tex"])
    t = t_fixture()
    for key, wt, _ in pr.WRITERS:
        arm = t["arms"][key]
        for b, bt in pr.T_BRANCHES:
            for g, gt in pr.GROUPS:
                node = arm["r"][b][g]["all"]
                for c, ct, _ in pr.T_CELLS:
                    assert m[f"TypedIn{wt}R{ct}{bt}{gt}"] == pct_iv(node[c])
                    assert m[f"TypedIn{wt}R{ct}{bt}{gt}Est"] == rf"{_p(node[c]['est'])}\%"
                for c, ct in pr.T_DIFFS:
                    assert m[f"TypedIn{wt}R{ct}{bt}{gt}"] == pp_iv(node[c])
            for s, st in pr.SUBSETS:
                node = arm["r"][b]["definition"][s]
                assert m[f"TypedIn{wt}RTypedMinusSel{bt}All{st}"] == pp_iv(node["typed_minus_bm25"])
                assert m[f"TypedIn{wt}N{bt}All{st}"] == f"{node['typed']['n_items']:,}"
            upd = arm["update"][b]["definition"]["all"]
            assert m[f"TypedIn{wt}UpdateAuthored{bt}All"] == pct_iv(upd["d1"])
        v = arm["validity"]["typed_writes"]
        assert m[f"TypedIn{wt}Writes"] == str(v["writes"])
        assert m[f"TypedIn{wt}UnreadableShare"] == rf"{_p(v['unreadable_share'])}\%"
        assert m[f"TypedIn{wt}AtLimit"] == str(v["at_limit"])
        assert m[f"TypedIn{wt}TzeroCorrectVipSender"] == pct_iv(
            arm["t0"]["k.vip_sender"]["correct"]
        )
    assert m["TypedInWorlds"] == "10"
    # coverage: one definition, and the six summed (0 + 20 + ... + 100 of 720)
    assert m["TypedInSelCompleteCloseFriend"] == rf"{_p(20 / 120)}\%"
    assert m["TypedInSelCompleteCountAll"] == "300"
    assert m["TypedInSelItemTimesAll"] == "720"
    # request tokens: a median ending in .5 rounds half up
    assert m["TypedInTokensMedianAll"] == "14,063"
    assert m["TypedInTokensPninetyfiveAll"] == "40,340"
    assert m["TypedInTokensMaxImportantContact"] == "3,000"


def test_feedback_macros_equal_the_analysis(full: dict[str, str]) -> None:
    m = macros(full["round3b_numbers.tex"])
    f = f_fixture()
    post = f["executability"]["definition"]["post"]
    assert m["ExecFbValidWithoutAll"] == pct_iv(post["both"]["first_attempt"])
    assert m["ExecFbValidWithAllEst"] == rf"{_p(post['both']['after_feedback']['est'])}\%"
    wh = post["per_definition"]["k.working_hours"]
    assert m["ExecFbValidWithWorkingHours"] == pct_iv(wh["after_feedback"])
    assert m["ExecFbRunsWorkingHours"] == f"{wh['n_runs']:,}"
    other = f["executability"]["other"]["post"]["both"]
    assert m["ExecFbValidWithoutOtherItems"] == pct_iv(other["first_attempt"])
    assert m["ExecFbRequestsOtherItems"] == f"{other['feedback_requests']:,}"
    for b, bt in (("src", "Src"), ("def", "Def")):
        node = f["r"][b]["definition"]["pooled"]
        assert m[f"ExecFbRWithout{bt}"] == pct_iv(node["dm"])
        assert m[f"ExecFbRWith{bt}"] == pct_iv(node["feedback"])
        assert m[f"ExecFbRDiff{bt}"] == pp_iv(node["difference"])
    # the analysis gives r per branch only: no pooled macro
    assert not [name for name in m if name.startswith("ExecFbR") and "Both" in name]
    for c, ct in (("B0", "Bzero"), ("B1_source", "Bone")):
        node = f["deletion"]["update"]["definition"][c]
        assert m[f"ExecFbDelUpdateDiff{ct}"] == pp_iv(node["difference"])
    for b, bt in (("src", "Src"), ("def", "Def"), ("orphan", "Del")):
        for c, ct in (("definition", "DefItems"), ("other", "OtherItems")):
            node = f["bxd"][b][c]
            assert m[f"ExecFbBxDWithout{bt}{ct}"] == pp_iv(node["Dm"])
            assert m[f"ExecFbBxDWith{bt}{ct}"] == pp_iv(node["Dm_feedback"])
            assert m[f"ExecFbBxDDiff{bt}{ct}"] == pp_iv(node["paired"]["difference"])
    assert (m["ExecFbRoundsZero"], m["ExecFbRoundsOne"], m["ExecFbRoundsTwo"]) == ("90", "0", "5")
    assert (m["ExecFbFailedWithout"], m["ExecFbFailedWith"], m["ExecFbStartKept"]) == (
        "10",
        "5",
        "3",
    )
    assert (m["ExecFbFirstFailSyntax"], m["ExecFbFirstFailNoCode"]) == ("7", "0")
    assert (m["ExecFbRunFailTimeout"], m["ExecFbRunFailSandbox"]) == ("1", "0")
    # every feedback request: the definition and the other items over all writes (220 + 440)
    assert m["ExecFbMessages"] == "660"
    assert m["ExecFbMessagesWorldData"] == "60"
    assert m["ExecFbMessagesWorldDataShare"] == rf"{_p(60 / 660)}\%"
    assert m["ExecFbWorlds"] == "20"


def test_extract_macros_equal_the_analysis(full: dict[str, str]) -> None:
    m = macros(full["round3b_numbers.tex"])
    q = w_fixture()["quantities"]
    for b, bt in (("src", "Src"), ("def", "Def")):
        for c, ct in (("definition", "DefItems"), ("other", "OtherItems")):
            node = q["bxd"][b][c]
            assert m[f"FieldExBxDExtracted{bt}{ct}"] == pp_iv(node["extracted"])
            assert m[f"FieldExBxDTyped{bt}{ct}"] == pp_iv(node["typed"])
            assert m[f"FieldExBxDDiff{bt}{ct}Est"] == rf"{_s(node['difference']['est'])}\,pp"
        node = q["r"][b]["pooled"]
        assert m[f"FieldExRExtracted{bt}"] == pct_iv(node["D1sx"])
        assert m[f"FieldExRDiff{bt}"] == pp_iv(node["difference"])
    email = q["extraction"]["email"]
    assert m["FieldExRecordsEmail"] == str(email["records"])
    assert m["FieldExRecordCorrectEmail"] == pct_iv(email["record_correct"])
    assert m["FieldExRewriteMissedEmail"] == rf"{_p(email['rewrite']['missed_share'])}\%"
    assert m["FieldExRecordCorrectAllSources"] == pct_iv(
        q["extraction"]["pooled"]["record_correct"]
    )
    node = q["attribution"]["B1"]["def"]
    assert m["FieldExDifferBoneDef"] == str(node["differ"])
    assert m["FieldExDifferWithErrorBoneDef"] == rf"{_p(node['share_with_error'])}\%"
    assert m["FieldExWorlds"] == "10"


def test_programs_panels_pair_the_same_programs(full: dict[str, str]) -> None:
    """(a) without and with feedback, (b) on typed and on extracted fields: B x D and r on the
    definition items, the valid runs of (a), and in the notes where the two inputs differed."""
    text = full["tab_programs.tex"]
    f, q = f_fixture(), w_fixture()["quantities"]
    n, last = r"\multicolumn{1}{c}{$n$}", r"& \multicolumn{1}{c@{}}{Difference} \\"
    assert f"(a) & & {n} & Without feedback & With feedback {last}" in text
    assert f"(b) & & {n} & Typed fields & Extracted fields {last}" in text
    panel_a, panel_b = text.split("(b) & &")
    assert panel_a.count(r"$B{\times}D$ on Update (pp), definition items & ") == 1
    assert panel_a.count(r" & $\S") == 5 and panel_b.count(r" & $\S") == 4  # shifts per row
    runs = f["executability"]["definition"]["post"]["both"]
    assert rf"valid runs (\%), definition items &  & {runs['n_runs']:,} & " in panel_a
    assert "other items" not in text  # the other items' differences are in the text
    node = q["bxd"]["src"]["definition"]
    assert pr._pair(node["extracted"], sign=True) in panel_b
    notes = text.split(r"\mtnotes{", 1)[1]
    assert "(a) $B{\\times}D$ and $r$" in notes
    assert "Values differed between the two inputs" in notes


def test_a_block_left_out_keeps_the_names(
    full: dict[str, str], placeholders: dict[str, str]
) -> None:
    filled, empty = macros(full["round3b_numbers.tex"]), macros(placeholders["round3b_numbers.tex"])
    assert list(filled) == list(empty)
    assert set(empty.values()) == {MISSING}
    assert all(re.fullmatch(r"[A-Za-z]+", name) for name in filled)
    for name, label in LABELS.items():
        assert placeholders[name].count(label) == 1
        assert MISSING in placeholders[name]
    for name in PANELS:
        assert MISSING in placeholders[name] and MISSING not in full[name]
        assert full[name].count(r"\\") == placeholders[name].count(r"\\")  # the same rows
    # the takeaway sentence needs the analysis
    assert "Claude Sonnet 5.5 raises" in full["tab_typed.tex"]
    assert "Claude Sonnet 5.5 raises" not in placeholders["tab_typed.tex"]


def test_takeaways_state_the_pattern_and_leave_the_numbers_to_the_table(
    full: dict[str, str],
) -> None:
    for name in LABELS:
        caption = full[name].split(r"\caption{")[1].split(r"\label{")[0]
        first = caption.split(". ")[0]
        assert r"\%" not in first and "pp" not in first, name  # the table prints the values
    assert (
        "close to that of exact recomputation, while Qwen3.5-9B stays far below"
        in (full["tab_typed.tex"])
    )
    # one tabular each, the descriptions in the notes below it (the typed-input table has its two
    # panels as row groups of one tabular, the programs panels (a) and (b) likewise)
    for name in (*LABELS, *PANELS):
        assert full[name].count(r"\toprule") == 1, name
        assert full[name].count(r"\mtnotes{") == 1, name


def test_files_start_with_the_command_and_carry_their_labels(
    full: dict[str, str], placeholders: dict[str, str]
) -> None:
    assert set(full) == set(pr.FILES)
    for text in full.values():
        assert text.startswith("% generated by memshift.analysis.paper_round3b --blocks t,f,w;")
    for name, label in LABELS.items():
        assert full[name].count(label) == 1
        assert r"\begin{table*}[t]" in full[name] and r"\centering\mtsetup" in full[name]
        assert MISSING not in full[name]
        # one width (the paper's table style), each estimate and its interval in one centred cell
        assert full[name].count(r"\begin{mttabular*}{\linewidth}") == 1
        assert r"\mtpair{" in full[name] and "\\scriptsize" not in full[name]
    for name in PANELS:  # without a float, caption or label of its own
        assert r"\begin{mttabular}" in full[name] and r"\mtpair{" in full[name]
        assert "table*" not in full[name] and r"\label" not in full[name]
    for name, text in (*full.items(), *placeholders.items()):
        if r"\begin{table*}" in text or name in PANELS:
            check_memtab(text, panel=name in PANELS)


def test_no_verdict_words(full: dict[str, str], placeholders: dict[str, str]) -> None:
    for text in (*full.values(), *placeholders.values()):
        assert not VERDICT.findall(text)


def test_cli_with_one_block(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    for name, data in (("t.json", t_fixture()), ("f.json", f_fixture()), ("w.json", w_fixture())):
        (analysis / name).write_text(json.dumps(data))
    out = tmp_path / "generated"
    pr.main(["--analysis-dir", str(analysis), "--out", str(out), "--blocks", "t"])
    texts = {name: (out / name).read_text() for name in pr.FILES}
    for text in texts.values():
        assert text.startswith("% generated by memshift.analysis.paper_round3b --blocks t;")
    m = macros(texts["round3b_numbers.tex"])
    assert m["TypedInQwenWrites"] == "472"
    assert m["TypedInTokensMedianAll"] == MISSING  # no t_facts.json
    assert m["ExecFbValidWithAll"] == MISSING and m["FieldExWorlds"] == MISSING
    with pytest.raises(SystemExit):
        pr.main(["--analysis-dir", str(analysis), "--out", str(out), "--blocks", "t,x"])


@pytest.mark.skipif(not (ANALYSIS / "t.json").exists(), reason="the typed-input analysis")
def test_committed_typed_analysis() -> None:
    blocks = ["t"]
    files = pr.build(pr.load(ANALYSIS, blocks), blocks)
    m = macros(files["round3b_numbers.tex"])
    t = json.loads((ANALYSIS / "t.json").read_text())
    for key, wt, _ in pr.WRITERS:
        node = t["arms"][key]["r"]["pooled"]["definition"]["all"]
        assert m[f"TypedIn{wt}RTyped" + "BothAll"] == pct_iv(node["typed"])
        assert m[f"TypedIn{wt}RAuthoredMinusTypedBothAll"] == pp_iv(node["d1_minus_typed"])
        v = t["arms"][key]["validity"]["typed_writes"]
        assert m[f"TypedIn{wt}Writes"] == f"{v['writes']:,}"
    sonnet = t["arms"]["claude-sonnet-5-5"]["r"]["pooled"]["definition"]["all"]
    assert m["TypedInSonnetRTypedBothAllEst"] == rf"{_p(sonnet['typed']['est'])}\%"
    assert "Claude Sonnet 5.5 raises $r$" in files["tab_typed.tex"]
    for text in files.values():
        assert not VERDICT.findall(text)
    facts = ANALYSIS / pr.FACTS
    if facts.exists():
        tok = json.loads(facts.read_text())["c_prompt_tokens"]["rederivations"]["all"]
        assert m["TypedInTokensMaxAll"] == f"{tok['max']:,}"


@pytest.mark.skipif(
    not (ANALYSIS / "f.json").exists() or not (ANALYSIS / "w.json").exists(),
    reason="the execution-feedback and extracted-fields analyses",
)
def test_committed_programs_panels() -> None:
    """The programs panels print the committed analyses: the valid runs of the definition items
    after the fork and B x D on the definition items, without and with feedback and on typed and
    extracted fields, each with its paired difference."""
    f = json.loads((ANALYSIS / "f.json").read_text())
    w = json.loads((ANALYSIS / "w.json").read_text())
    table = pr.tab_programs(f, w)
    runs = f["executability"]["definition"]["post"]["both"]
    assert f"& {runs['n_runs']:,} & {pr._pair(runs['first_attempt'])} & " in table
    for b in ("src", "def", "orphan"):
        node = f["bxd"][b]["definition"]
        cells = (node["Dm"], node["Dm_feedback"], node["paired"]["difference"])
        assert " & ".join(pr._pair(e, sign=True) for e in cells) in table, b
    for b in ("src", "def"):
        node = w["quantities"]["bxd"][b]["definition"]
        cells = (node["typed"], node["extracted"], node["difference"])
        assert " & ".join(pr._pair(e, sign=True) for e in cells) in table, b
    assert MISSING not in table and not VERDICT.findall(table)

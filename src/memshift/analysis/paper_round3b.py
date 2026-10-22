"""Macros and tables of three experiments on the definition items, generated from their analyses:
typed input (the writer re-derives a definition item from every live record of its sources, given
as typed fields), execution feedback (the writer's programs D_m get the interpreter's error back and
may rewrite the program) and extracted fields (the programs read fields the writer extracted from
prose rewritten from the typed records).

Inputs, in --analysis-dir (a missing file renders as the MISSING marker):
  t.json        memshift.analysis.round3b_typed --runs ...   (typed input)
  t_facts.json  memshift.analysis.round3b_typed --facts --tokenizer ...   (request lengths)
  f.json        memshift.analysis.round3b_feedback   (execution feedback)
  w.json        memshift.analysis.round3b_extract   (extracted fields)
Outputs, in --out: round3b_numbers.tex (macros), tab_typed.tex, tab_feedback.tex, tab_extract.tex.
A block left out of --blocks gets the same macro names defined as the MISSING marker and a table
of the same layout and label with MISSING cells, so the paper builds either way.

Macros (letters only; X = estimate with its 95% interval in the format of paper_numbers, XEst = the
estimate alone; percent and pp with one decimal):
  TypedIn<W>R<cell><branch><group>   r on the changed definition item-times; W Qwen | Sonnet (the
        definition-item writer); cell Sel (the writer's own BM25 selection, D0B1_source) | Typed
        (D0typedB1_source) | Authored (D1B1_source) | TypedMinusSel | AuthoredMinusTyped (paired,
        pp); branch Src | Def | Both (the two pooled); group one definition (ImportantContact, ...)
        or All (the six pooled). With the suffix SelComplete | SelIncomplete (group All only): the
        item-times whose selection held every record the definition ranges over, or not.
  TypedIn<W>N<branch>All[SelComplete|SelIncomplete]   the changed definition item-times
  TypedIn<W>Update<cell><branch>All   Update on the changed definition item-times per cell
  TypedIn<W>{Writes,Unreadable,UnreadableShare,AtLimit,AtLimitShare}   the typed writes
  TypedIn<W>TzeroCorrect<group>   the typed derivation before the fork equals gold
  TypedInSel{Complete,CompleteCount,ItemTimes}<group>   the selection's coverage (All: the sum
        of the six definitions' counts)
  TypedInTokens{Median,Pninetyfive,Max}<group>   prompt tokens of the typed re-derivation
        requests (Qwen3.5-9B tokenizer; from t_facts.json)
  ExecFbValid{Without,With}<group>, ExecFb{Runs,Requests}<group>   program runs for definition
        items after the fork that end with a valid typed value without and with feedback, the runs
        and the feedback requests; group one definition, All or OtherItems (the other 42 items)
  ExecFbR{Without,With,Diff}<Src|Def>   r of D1sB1_source and D1sfbB1_source, difference (the
        analysis gives r per branch only)
  ExecFbDelUpdate{Without,With,Diff}<Bzero|Bone>   Update on definition items after a deletion
  ExecFbBxD{Without,With,Diff}<Src|Def|Del><DefItems|OtherItems>   B x D on Update (pp)
  ExecFb{Rounds<Zero|One|Two>,FailedWithout,FailedWith,StartKept}   counts (definition items
        after the fork); ExecFb{FirstFail,RunFail}<kind>: failure kinds of the first attempts
        and of every failed attempt with feedback (each program a write ran that failed, the
        rewrites included); ExecFbMessages[WorldData[Share]]: every feedback request (all
        writes, definition and other items: the sum of the two classes) and those containing a
        string value of the records
  FieldExBxD{Typed,Extracted,Diff}<Src|Def><DefItems|OtherItems>   B x D on Update (pp)
  FieldExR{Typed,Extracted,Diff}<Src|Def>   r of D1sB1_source and D1sxB1_source, difference
  FieldEx{Records,RecordCorrect,RewriteMissed}<source>   per source (AllSources: pooled)
  FieldEx{ItemTimes,Differ,DifferShare,DifferWithError}<Bzero|Bone><Src|Def>   item-times whose
        value differs between the two program inputs, and the share of those whose write read a
        record with an extraction error
  {TypedIn,ExecFb,FieldEx}Worlds   the worlds of the block

Nothing is estimated here: every number is a value of the analyses, formatted. The only arithmetic
is the sums named above and shares of two counts the analyses give.

Usage: python -m memshift.analysis.paper_round3b --analysis-dir runs/stage9_round3b/analysis
       --out paper/generated [--blocks t,f,w]
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis.ex_e1_paper import UNDEFINED
from memshift.analysis.ex_round2 import DEFINITIONS
from memshift.analysis.paper_data import MISSING, read_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, ci, pct, pp
from memshift.analysis.paper_tables import _num, _row
from memshift.holdmem.extract import SOURCES
from memshift.holdmem.program import FAILURE_KINDS

MODULE = "memshift.analysis.paper_round3b"
BLOCKS = {"t": "t.json", "f": "f.json", "w": "w.json"}
FACTS = "t_facts.json"
FILES = ("round3b_numbers.tex", "tab_typed.tex", "tab_feedback.tex", "tab_extract.tex")

WRITERS = (  # (arm in t.json, macro tag, printed name)
    ("Qwen/Qwen3.5-9B", "Qwen", "Qwen3.5-9B"),
    ("claude-sonnet-5-5", "Sonnet", "Claude Sonnet 5.5"),
)
T_CELLS = (
    ("bm25", "Sel", "selection"),
    ("typed", "Typed", "typed"),
    ("d1", "Authored", "exact"),
)
T_DIFFS = (("typed_minus_bm25", "TypedMinusSel"), ("d1_minus_typed", "AuthoredMinusTyped"))
T_BRANCHES = (("src", "Src"), ("def", "Def"), ("pooled", "Both"))  # pooled: S_src and S_def
SUBSETS = (("complete", "SelComplete"), ("incomplete", "SelIncomplete"))
GROUPS = (*((d, tag) for d, tag, _ in DEFINITIONS), ("definition", "All"))
TOKEN_STATS = (("median", "Median"), ("p95", "Pninetyfive"), ("max", "Max"))

F_BRANCHES = (("src", "Src"), ("def", "Def"), ("orphan", "Del"))
F_R_BRANCHES = F_BRANCHES[:2]  # r: after S_src and S_def, not pooled
CLASSES = (("definition", "DefItems", "definition items"), ("other", "OtherItems", "other items"))
DEL_CELLS = (("B0", "Bzero", "$B{=}0$"), ("B1_source", "Bone", "$B{=}1$"))
LEVELS = (("dm",), ("feedback",), ("difference",))  # without feedback, with it, the difference
ROUNDS = (
    ("0", "Zero", "no feedback round"),
    ("1", "One", "one feedback round"),
    ("2", "Two", "two feedback rounds"),
)
KIND_LABELS = {
    "no_code": "no code in the reply",
    "syntax": "syntax error",
    "no_derive": "no function to call",
    "sandbox": "rejected by the sandbox",
    "timeout": "timeout",
    "exception": "exception",
    "type": "value of the wrong type",
}

W_BRANCHES = (("src", "Src"), ("def", "Def"))
SOURCE_TAGS = (*((s, s.capitalize()) for s in SOURCES), ("pooled", "AllSources"))
SOURCE_LABELS = {
    "contacts": "contacts",
    "calendar": "calendar",
    "tasks": "tasks",
    "email": "e-mail",
    "messages": "chat messages",
    "notes": "notes",
    "pooled": "all sources",
}
FIELD_LABELS = {  # the record fields in plain words; the others print as they are named
    "appt_type": "appointment type",
    "duration_min": "duration",
    "from_addr": "sender address",
    "from_name": "sender name",
    "conversation_with": "conversation partner",
}
PAIRS = (("B0", "Bzero", "$B{=}0$"), ("B1", "Bone", "$B{=}1$"))
BRANCH_LABELS = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}
DM = r"$D_{\mathrm{m}}$"

Est = Mapping[str, Any] | None
Fmt = Callable[[float | None], str]


# ------------------------------------------------------------------------------------ formats
def _iv(e: Est, f: Fmt = pct) -> str:
    """Estimate with its interval in the macros' format; MISSING when absent, a dash when the
    estimate is undefined (no item-time)."""
    if not isinstance(e, Mapping):
        return MISSING
    est = e.get("est")
    if est is None or est != est:
        return UNDEFINED
    return ci(dict(e), f)


def _est(e: Est, f: Fmt = pct) -> str:
    if not isinstance(e, Mapping):
        return MISSING
    est = e.get("est")
    return UNDEFINED if est is None or est != est else f(est)


def _count(x: Any) -> str:
    return MISSING if x is None else f"{int(x):,}"


def _share(num: Any, den: Any) -> str:
    return MISSING if num is None or not den else pct(num / den)


def _fraction(x: Any) -> str:
    return MISSING if x is None else pct(float(x))


def _tokens(x: Any) -> str:
    """A token count (a median can end in .5: rounded half up)."""
    return MISSING if x is None else f"{math.floor(float(x) + 0.5):,}"


def _sum(*xs: Any) -> int | None:
    return None if any(x is None for x in xs) else sum(int(x) for x in xs)


def _n_worlds(worlds: Any) -> int | None:
    return len(worlds) if isinstance(worlds, list) else None


class _Out(dict[str, str]):
    """Macros of one block: a name defined twice is an error."""

    def __init__(self, prefix: str) -> None:
        super().__init__()
        self.prefix = prefix

    def put(self, *parts: str, value: str) -> None:
        name = "".join((self.prefix, *parts))
        if name in self:
            raise ValueError(f"macro {name} defined twice")
        self[name] = value

    def iv(self, *parts: str, e: Est, f: Fmt = pct) -> None:
        """`X` (estimate and interval) and `XEst` (the estimate)."""
        self.put(*parts, value=_iv(e, f))
        self.put(*parts, "Est", value=_est(e, f))


# ---------------------------------------------------------------------------- table cells
# The tables use the paper's table style (paper/memtab-style.tex): panels in `mttabular*` at the
# text width with notes under them (`\mtnotes`), each estimate and its interval in one centred cell
# (`\mtpair`, its widths set per column or per panel by `\mtpairwidths`), counts and shares in
# `N{template}` columns (a right-aligned value box centred under the header) and italic panel
# banners (`\mtgroup`).
def _pair(e: Est, sign: bool = False) -> str:
    """An estimate and its 95% interval in one centred cell (percent or signed pp), the estimate
    alone where the printed ends of the interval equal it (as `paper_tables.est_ci_cells`);
    MISSING or a dash when there is no node or no estimate."""
    if not isinstance(e, Mapping):
        return MISSING
    if e.get("est") is None or e["est"] != e["est"]:
        return UNDEFINED
    est, lo, hi = (_num(e[k], sign=sign) for k in ("est", "lo", "hi"))
    iv = "" if lo == est == hi else rf"[{lo},\,{hi}]"
    return rf"\mtpair{{{est}}}{{{iv}}}"


def _single(text: str) -> str:
    """A value without interval in a column of `\\mtpair` cells, aligned with the estimates."""
    return rf"\mtpair{{{text}}}{{}}"


def _boxed(text: str, template: str, edge: str = "") -> str:
    """A value in a box as wide as `template`, centred in a column of another type (`edge`:
    `@{}` at the table's right edge)."""
    return rf"\multicolumn{{1}}{{N{{{template}}}{edge}}}{{{text}}}"


def _head(text: str, last: bool = False) -> str:
    """A header over one centred cell of an `N` or `\\mtpair` column (`c@{}` at the right edge)."""
    return rf"\multicolumn{{1}}{{c{'@{}' if last else ''}}}{{{text}}}"


def _widths(est: str, iv: str) -> str:
    """The column type of a column of `\\mtpair` cells with the given widest estimate and
    interval."""
    return rf">{{\mtpairwidths{{{est}}}{{{iv}}}}}c"


def _n(e: Est) -> str:
    return _count(e.get("n_items")) if isinstance(e, Mapping) else MISSING


def _plain(x: Any) -> str:
    """A share without interval, in percent (no unit)."""
    return MISSING if x is None else _num(float(x))


def _share_cell(num: Any, den: Any) -> str:
    return MISSING if num is None or not den else _num(num / den)


def _tex(name: str) -> str:
    return name.replace("_", r"\_")


# ================================================================================ typed input
def _coverage(t: Mapping[str, Any] | None, group: str) -> tuple[int | None, int | None]:
    """(complete item-times, item-times) of the selection; All sums the six definitions."""
    cov = _get(t, "coverage")
    if group != "definition":
        return _get(cov, group, "complete"), _get(cov, group, "item_times")
    parts = [(_get(cov, d, "complete"), _get(cov, d, "item_times")) for d, _, _ in DEFINITIONS]
    return _sum(*(c for c, _ in parts)), _sum(*(n for _, n in parts))


def _typed_worlds(t: Mapping[str, Any] | None) -> int | None:
    """The worlds of the arms (each arm has the same worlds)."""
    arms = _get(t, "arms")
    sizes = [_n_worlds(_get(a, "worlds")) for a in (arms.values() if arms else ())]
    known = [s for s in sizes if s is not None]
    return max(known) if known else None


def typed_numbers(t: Mapping[str, Any] | None, facts: Mapping[str, Any] | None) -> dict[str, str]:
    out = _Out("TypedIn")
    arms = _get(t, "arms")
    out.put("Worlds", value=_count(_typed_worlds(t)))
    for key, wt, _ in WRITERS:
        arm = _get(arms, key)
        for b, bt in T_BRANCHES:
            for g, gt in GROUPS:
                node = _get(arm, "r", b, g, "all")
                for c, ct, _ in T_CELLS:
                    out.iv(wt, "R", ct, bt, gt, e=_get(node, c))
                for c, ct in T_DIFFS:
                    out.iv(wt, "R", ct, bt, gt, e=_get(node, c), f=pp)
            out.put(wt, "N", bt, "All", value=_n(_get(arm, "r", b, "definition", "all", "typed")))
            for sub, st in SUBSETS:
                node = _get(arm, "r", b, "definition", sub)
                for c, ct, _ in T_CELLS:
                    out.iv(wt, "R", ct, bt, "All", st, e=_get(node, c))
                for c, ct in T_DIFFS:
                    out.iv(wt, "R", ct, bt, "All", st, e=_get(node, c), f=pp)
                out.put(wt, "N", bt, "All", st, value=_n(_get(node, "typed")))
            for c, ct, _ in T_CELLS:
                out.iv(
                    wt, "Update", ct, bt, "All", e=_get(arm, "update", b, "definition", "all", c)
                )
        v = _get(arm, "validity", "typed_writes")
        out.put(wt, "Writes", value=_count(_get(v, "writes")))
        out.put(wt, "Unreadable", value=_count(_get(v, "unreadable")))
        out.put(wt, "UnreadableShare", value=_fraction(_get(v, "unreadable_share")))
        out.put(wt, "AtLimit", value=_count(_get(v, "at_limit")))
        out.put(wt, "AtLimitShare", value=_fraction(_get(v, "at_limit_share")))
        for g, gt in GROUPS:
            out.iv(wt, "TzeroCorrect", gt, e=_get(arm, "t0", g, "correct"))
    for g, gt in GROUPS:
        complete, item_times = _coverage(t, g)
        out.put("SelComplete", gt, value=_share(complete, item_times))
        out.put("SelCompleteCount", gt, value=_count(complete))
        out.put("SelItemTimes", gt, value=_count(item_times))
    tokens = _get(facts, "c_prompt_tokens", "rederivations")
    for g, gt in GROUPS:
        node = _get(tokens, "all" if g == "definition" else g)
        for stat, st in TOKEN_STATS:
            out.put("Tokens", st, gt, value=_tokens(_get(node, stat)))
    return out


def _typed_rows(arm: Mapping[str, Any] | None, t: Mapping[str, Any] | None) -> list[str]:
    def cells(node: Any) -> list[str]:
        return [
            _n(_get(node, "typed")),
            *(_pair(_get(node, c)) for c, _, _ in T_CELLS),
            *(_pair(_get(node, c), sign=True) for c, _ in T_DIFFS),
        ]

    rows = []
    for d, _, label in DEFINITIONS:
        complete, item_times = _coverage(t, d)
        node = _get(arm, "r", "pooled", d, "all")
        rows.append(_row(label, _share_cell(complete, item_times), *cells(node)))
    complete, item_times = _coverage(t, "definition")
    rows.append(r"\addlinespace[2pt]")
    node = _get(arm, "r", "pooled", "definition", "all")
    rows.append(_row("all six", _share_cell(complete, item_times), *cells(node)))
    for b in ("src", "def"):
        node = _get(arm, "r", b, "definition", "all")
        rows.append(_row(rf"\mtin after {BRANCH_LABELS[b]}", "", *cells(node)))
    for sub, _ in SUBSETS:
        node = _get(arm, "r", "pooled", "definition", sub)
        rows.append(_row(rf"\mtin selection {sub}", "", *cells(node)))
    return rows


def _typed_takeaway(t: Mapping[str, Any] | None) -> str:
    """The caption, the pattern the table shows (the MISSING marker without the analysis)."""
    sonnet = _get(t, "arms", "claude-sonnet-5-5", "r", "pooled", "definition", "all")
    qwen = _get(t, "arms", "Qwen/Qwen3.5-9B", "r", "pooled", "definition", "all")
    if not sonnet or not qwen:
        return MISSING
    return (
        r"Given every live record of the definition's sources as typed fields, Claude Sonnet 5.5 "
        r"raises $r$ on the changed definition item-times close to that of exact "
        r"recomputation, while Qwen3.5-9B stays far below it."
    )


# the widths of the \mtpair columns of the typed-input table, as templates of the widest estimate
# and interval: r of the selection, typed and exact cells, typed minus selection, exact minus typed
TYPED_PAIRS = (
    *[("100.0", r"[00.0,\,100.0]")] * 3,
    ("$+$00.0", r"[$+$00.0,\,$+$100.0]"),
    ("$+$100.0", r"[$+$00.0,\,$+$100.0]"),
)
TYPED_GAP = r"@{\hspace{6pt}}"  # between n and the first column of r


def tab_typed(t: Mapping[str, Any] | None) -> str:
    """The typed-input table: per writer (panels a and b) r of the selection, typed and exact
    cells per definition and pooled (the writes and request lengths are macros the text states)."""
    n = _count(_typed_worlds(t))
    arms = _get(t, "arms")
    pairs = "".join(_widths(*w) for w in TYPED_PAIRS)
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{_typed_takeaway(t)}}}",
        r"\label{tab:typed}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{2pt}",
        rf"\begin{{mttabular*}}{{\linewidth}}{{l N{{000.0}} N{{000}}{TYPED_GAP}{pairs}@{{}}}}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{$r$ (\%)} & \multicolumn{2}{c@{}}{Difference (pp)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(l){7-8}",
        r"& \multicolumn{1}{c}{\mthead{Complete}{(\%)}} & \multicolumn{1}{c}{$n$} & Selection "
        r"& Typed & Exact & Typed $-$ selection & \multicolumn{1}{c@{}}{Exact $-$ typed} \\",
    ]
    for i, (key, _, name) in enumerate(WRITERS):
        lines += [r"\midrule", rf"\mtgroup{{8}}{{({'ab'[i]}) {name}}}"]
        lines += _typed_rows(_get(arms, key), t)
    lines += [
        r"\bottomrule",
        r"\end{mttabular*}",
        rf"\mtnotes{{$r$ on the changed definition item-times after $\Ssrc$ and $\Sdef$ ({n} "
        r"worlds) when the writer re-derives the item from its own selection ($D{=}0$) or from "
        r"every live record of the definition's sources as typed fields, and with exact "
        r"recomputation ($D{=}1$). Complete: the share of item-times whose selection held every "
        r"record the definition ranges over (point estimates). In (b), Claude Sonnet 5.5 writes "
        r"the definition items.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ========================================================================= execution feedback
def _exec_node(f: Mapping[str, Any] | None, group: str) -> Any:
    post = _get(f, "executability", "definition", "post")
    if group == "definition":
        return _get(post, "both")
    if group == "other":
        return _get(f, "executability", "other", "post", "both")
    return _get(post, "per_definition", group)


def _counted(node: Any, *path: str) -> int | None:
    """A count the analysis leaves out when it is 0 (rounds, failure kinds)."""
    if not isinstance(node, Mapping):
        return None
    parent = _get(node, *path[:-1])
    return int(parent.get(path[-1], 0)) if isinstance(parent, Mapping) else None


def _messages(f: Mapping[str, Any] | None) -> tuple[int | None, int | None]:
    """(feedback requests, those containing a string value of the records) over every write of both
    item classes: the sum of the two classes."""
    parts = [_get(f, "executability", c, "all", "both", "world_data") for c, _, _ in CLASSES]
    return (
        _sum(*(_get(p, "requests") for p in parts)),
        _sum(*(_get(p, "with_record_value") for p in parts)),
    )


def feedback_numbers(f: Mapping[str, Any] | None) -> dict[str, str]:
    out = _Out("ExecFb")
    out.put("Worlds", value=_count(_n_worlds(_get(f, "worlds"))))
    for g, gt in (*GROUPS, ("other", "OtherItems")):
        node = _exec_node(f, g)
        out.iv("ValidWithout", gt, e=_get(node, "first_attempt"))
        out.iv("ValidWith", gt, e=_get(node, "after_feedback"))
        out.put("Runs", gt, value=_count(_get(node, "n_runs")))
        out.put("Requests", gt, value=_count(_get(node, "feedback_requests")))
    both = _exec_node(f, "definition")
    for r, rt, _ in ROUNDS:
        out.put("Rounds", rt, value=_count(_counted(both, "rounds", r)))
    out.put("FailedWithout", value=_count(_get(both, "first_attempt_failed")))
    out.put("FailedWith", value=_count(_get(both, "after_feedback_failed")))
    out.put("StartKept", value=_count(_get(both, "start_kept")))
    for kind in FAILURE_KINDS:
        tag = "".join(p.capitalize() for p in kind.split("_"))
        out.put("FirstFail", tag, value=_count(_counted(both, "first_attempt_kinds", kind)))
        out.put("RunFail", tag, value=_count(_counted(both, "failed_run_kinds", kind)))
    for b, bt in F_R_BRANCHES:
        node = _get(f, "r", b, "definition", "pooled")
        out.iv("RWithout", bt, e=_get(node, "dm"))
        out.iv("RWith", bt, e=_get(node, "feedback"))
        out.iv("RDiff", bt, e=_get(node, "difference"), f=pp)
    for cell, ct, _ in DEL_CELLS:
        node = _get(f, "deletion", "update", "definition", cell)
        out.iv("DelUpdateWithout", ct, e=_get(node, "dm"))
        out.iv("DelUpdateWith", ct, e=_get(node, "feedback"))
        out.iv("DelUpdateDiff", ct, e=_get(node, "difference"), f=pp)
    for b, bt in F_BRANCHES:
        for cls, clt, _ in CLASSES:
            node = _get(f, "bxd", b, cls)
            out.iv("BxDWithout", bt, clt, e=_get(node, "Dm"), f=pp)
            out.iv("BxDWith", bt, clt, e=_get(node, "Dm_feedback"), f=pp)
            out.iv("BxDDiff", bt, clt, e=_get(node, "paired", "difference"), f=pp)
    total, with_data = _messages(f)
    out.put("Messages", value=_count(total))
    out.put("MessagesWorldData", value=_count(with_data))
    out.put("MessagesWorldDataShare", value=_share(with_data, total))
    return out


def _feedback_takeaway(f: Mapping[str, Any] | None) -> str:
    """The caption's first sentence, the pattern the table shows (the plain description without
    the analysis)."""
    both = _exec_node(f, "definition")
    diffs = [_get(f, "r", b, "definition", "pooled", "difference") for b, _ in F_R_BRANCHES]
    if not isinstance(both, Mapping) or not all(isinstance(d, Mapping) for d in diffs):
        return rf"The writer's programs ({DM}) without and with execution feedback"
    return (
        rf"Execution feedback lets more of the writer's programs ({DM}) end with a valid typed "
        r"value but repairs no more definition items"
    )


NUMBER_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def _all_valid(node: Any) -> bool:
    """Every run of the node ended with a valid typed value, without feedback and with it."""
    keys = ("first_attempt", "after_feedback")
    return all(isinstance(_get(node, k), Mapping) for k in keys) and all(
        _get(node, k, x) == 1.0 for k in keys for x in ("est", "lo", "hi")
    )


def _exec_rows(f: Mapping[str, Any] | None) -> list[tuple[str, Any]]:
    """(label, node) of the valid-run rows: each definition with an invalid run, then the others
    folded into one row (two or more definitions whose every run was valid), all six and the other
    items. A folded row's node sums the runs and requests; its shares are 100.0 by construction."""
    nodes = [(label, _exec_node(f, d)) for d, _, label in DEFINITIONS]
    valid = [node for _, node in nodes if _all_valid(node)]
    if len(valid) < 2:
        rows = nodes
    else:
        one = {"est": 1.0, "lo": 1.0, "hi": 1.0}
        folded = {
            "n_runs": _sum(*(_get(n, "n_runs") for n in valid)),
            "feedback_requests": _sum(*(_get(n, "feedback_requests") for n in valid)),
            "first_attempt": one,
            "after_feedback": one,
        }
        rows = [(label, node) for label, node in nodes if not _all_valid(node)]
        rows.append((f"other {NUMBER_WORDS.get(len(valid), len(valid))} definitions", folded))
    rows.append(("all six", _exec_node(f, "definition")))
    return [*rows, ("other items", _exec_node(f, "other"))]


def tab_feedback(f: Mapping[str, Any] | None) -> str:
    n = _count(_n_worlds(_get(f, "worlds")))
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{_feedback_takeaway(f)} ({n} worlds).}}",
        r"\label{tab:feedback}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        r"\mtpairwidths{$+$00.0}{[$+$00.0,\,$+$00.0]}",  # the widths of every \mtpair cell below
        r"\begin{mttabular*}{\linewidth}{ll N{00,000} ccc@{}}",
        r"\toprule",
        r"(a) & & \multicolumn{1}{c}{$n$} & Without feedback & With feedback "
        r"& \multicolumn{1}{c@{}}{Difference} \\",
        r"\midrule",
    ]

    def contrast(label: str, sub: str, node: Any, bxd: bool = False) -> str:
        """Without feedback, with feedback, their paired difference (B x D: signed levels)."""
        keys = (("Dm",), ("Dm_feedback",), ("paired", "difference")) if bxd else LEVELS
        without, with_, diff = (_get(node, *k) for k in keys)
        return _row(
            label,
            sub,
            _n(diff),
            _pair(without, sign=bxd),
            _pair(with_, sign=bxd),
            _pair(diff, sign=True),
        )

    first = r"$r$ (\%), definition items"
    for b in ("src", "def"):
        lines.append(contrast(first, BRANCH_LABELS[b], _get(f, "r", b, "definition", "pooled")))
        first = ""
    lines.append(r"\addlinespace[2pt]")
    first = r"Update after $\Sdel$ (\%), definition items"
    for cell, _, label in DEL_CELLS:
        lines.append(contrast(first, label, _get(f, "deletion", "update", "definition", cell)))
        first = ""
    for cls, _, label in CLASSES:
        lines.append(r"\addlinespace[2pt]")
        first = rf"$B{{\times}}D$ on Update (pp), {label}"
        for b, _ in F_BRANCHES:
            lines.append(contrast(first, BRANCH_LABELS[b], _get(f, "bxd", b, cls), bxd=True))
            first = ""
    lines += [
        r"\midrule",
        r"(b) & & \multicolumn{1}{c}{Runs} & Without feedback (\%) & With feedback (\%) "
        r"& \multicolumn{1}{c@{}}{Requests} \\",
        r"\midrule",
    ]
    rows = _exec_rows(f)
    for label, node in rows:
        if label == "all six":
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            _row(
                f"{label} &",  # the second column is empty
                _count(_get(node, "n_runs")),
                _pair(_get(node, "first_attempt")),
                _pair(_get(node, "after_feedback")),
                _boxed(_count(_get(node, "feedback_requests")), "000", "@{}"),
            )
        )
    lines += [r"\end{mttabular*}\nointerlineskip"]  # one rule between the panels
    nodes = [_exec_node(f, "definition"), _exec_node(f, "other")]

    def kind_counts(kind: str) -> list[int | None]:
        return [
            _counted(node, key, kind)
            for node in nodes
            for key in ("first_attempt_kinds", "failed_run_kinds")
        ]

    shown = [k for k in FAILURE_KINDS if any(kind_counts(k)) or None in kind_counts(k)]
    unseen = [KIND_LABELS[k] for k in FAILURE_KINDS if k not in shown]
    lines += [
        r"\begin{mttabular*}{\linewidth}{l N{000}N{000} N{000}N{000}@{}}",
        r"\midrule",
        r"& \multicolumn{2}{c}{Definition items} & \multicolumn{2}{c@{}}{Other items} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        _row(
            "(c)",
            *(_head(h) for h in ("First attempts", "Failed attempts", "First attempts")),
            _head("Failed attempts", last=True),
        ),
        r"\midrule",
    ]
    for kind in shown:
        lines.append(_row(KIND_LABELS[kind], *map(_count, kind_counts(kind))))
    lines.append(r"\addlinespace[2pt]")
    cells = []
    for k, node in enumerate(nodes):
        edge = "@{}" if k == len(nodes) - 1 else ""
        counts = " / ".join(_count(_counted(node, "rounds", r)) for r, _, _ in ROUNDS[1:])
        cells.append(rf"\multicolumn{{2}}{{c{edge}}}{{{counts}}}")
    lines.append(_row("runs with one / two feedback rounds", *cells))
    none = rf" Kinds without a failed attempt: {', '.join(unseen)}." if unseen and f else ""
    lines += [
        r"\bottomrule",
        r"\end{mttabular*}",
        r"\mtnotes{(a) $r$, Update after a deletion and $B{\times}D$, without and with feedback. "
        r"(b) Runs after the fork that end with a valid typed value, the runs and the feedback "
        r"requests. (c) Failure kinds and feedback rounds after the fork, as counts. The other "
        rf"runs had no feedback round.{none}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# =========================================================================== extracted fields
def extract_numbers(w: Mapping[str, Any] | None) -> dict[str, str]:
    out = _Out("FieldEx")
    q = _get(w, "quantities")
    out.put("Worlds", value=_count(_n_worlds(_get(q, "worlds"))))
    for b, bt in W_BRANCHES:
        for cls, clt, _ in CLASSES:
            node = _get(q, "bxd", b, cls)
            out.iv("BxDTyped", bt, clt, e=_get(node, "typed"), f=pp)
            out.iv("BxDExtracted", bt, clt, e=_get(node, "extracted"), f=pp)
            out.iv("BxDDiff", bt, clt, e=_get(node, "difference"), f=pp)
        node = _get(q, "r", b, "pooled")
        out.iv("RTyped", bt, e=_get(node, "D1s"))
        out.iv("RExtracted", bt, e=_get(node, "D1sx"))
        out.iv("RDiff", bt, e=_get(node, "difference"), f=pp)
    for s, st in SOURCE_TAGS:
        node = _get(q, "extraction", s)
        out.put("Records", st, value=_count(_get(node, "records")))
        out.iv("RecordCorrect", st, e=_get(node, "record_correct"))
        out.put("RewriteMissed", st, value=_fraction(_get(node, "rewrite", "missed_share")))
    for pair, pt, _ in PAIRS:
        for b, bt in W_BRANCHES:
            node = _get(q, "attribution", pair, b)
            out.put("ItemTimes", pt, bt, value=_count(_get(node, "item_times")))
            out.put("Differ", pt, bt, value=_count(_get(node, "differ")))
            out.put("DifferShare", pt, bt, value=_fraction(_get(node, "differ_share")))
            out.put("DifferWithError", pt, bt, value=_fraction(_get(node, "share_with_error")))
    return out


def _field_name(name: str) -> str:
    """A field in words; a pooled field `source.field` with its source first."""
    source, _, field = name.rpartition(".")
    word = _tex(FIELD_LABELS.get(field, field))
    return f"{SOURCE_LABELS.get(source, source)} {word}" if source else word


def _weakest(node: Any) -> str:
    """The field with the smallest share of values extracted without an error and that share
    (the first in name order on a tie); `all` when every field of the source is at 100.0."""
    fields = _get(node, "fields")
    if not isinstance(fields, Mapping):
        return MISSING
    shares = {x: _get(fields, x, "correct") for x in sorted(fields)}
    if any(v is None for v in shares.values()) or not shares:
        return MISSING
    if {_plain(v) for v in shares.values()} == {"100.0"}:
        return "all 100.0"
    name = min(shares, key=lambda x: shares[x])
    return f"{_field_name(name)} {_plain(shares[name])}"


def _extract_takeaway(q: Mapping[str, Any] | None) -> str:
    """The caption's first sentence, the pattern the table shows (the plain description without
    the analysis)."""
    nodes = [_get(q, "bxd", b, "definition") for b, _ in W_BRANCHES]
    if not all(isinstance(x, Mapping) for x in nodes):
        return (
            rf"The writer's programs ({DM}) reading fields that the writer extracted from prose "
            r"instead of the generator's typed fields"
        )
    return (
        rf"On fields that the writer extracted from prose, the writer's programs ({DM}) keep "
        r"$B{\times}D$ on Update positive on the definition items"
    )


def _differ(q: Mapping[str, Any] | None) -> str:
    """The note on the definition item-times whose value differs between the two program inputs:
    how many per B arm and shift, and the share of them whose write read a record with an
    extraction error (once when it is the same everywhere)."""
    nodes = [(label, _get(q, "attribution", pair, b)) for pair, _, label in PAIRS
             for b, _ in W_BRANCHES]  # fmt: skip
    if q is None or not all(isinstance(n, Mapping) for _, n in nodes):
        return MISSING
    shifts = " and ".join(BRANCH_LABELS[b] for b, _ in W_BRANCHES)
    parts = []
    for i, arm in enumerate(dict.fromkeys(label for label, _ in nodes)):
        counts = " and ".join(
            f"{_count(_get(n, 'differ'))} of {_count(_get(n, 'item_times'))}"
            for label, n in nodes
            if label == arm
        )
        where = f" definition item-times after {shifts}" if i == 0 else ""
        parts.append(f"{counts}{where} at {arm}")
    shares = [rf"{_plain(_get(n, 'share_with_error'))}\%" for _, n in nodes]
    share = shares[0] if len(set(shares)) == 1 else ", ".join(shares) + ", respectively,"
    return (
        f"Values differed between the two inputs on {', and on '.join(parts)}. Of these, "
        f"{share} had read a record with an extraction error."
    )


def tab_extract(w: Mapping[str, Any] | None) -> str:
    q = _get(w, "quantities")
    n = _count(_n_worlds(_get(q, "worlds")))
    correct = _widths("100.0", r"[00.0,\,00.0]")  # the records extracted without error
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{{_extract_takeaway(q)} ({n} worlds).}}",
        r"\label{tab:extract}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular*}}{{\linewidth}}{{l N{{00,000}} {correct} N{{00.0}} l@{{}}}}",
        r"\toprule",
        r"(a) & \multicolumn{1}{c}{Records} & Without error (\%) "
        r"& \multicolumn{1}{c}{Rewrite missed (\%)} & Weakest field (\%) \\",
        r"\midrule",
    ]
    for s, _ in SOURCE_TAGS:
        node = _get(q, "extraction", s)
        if s == "pooled":
            lines.append(r"\addlinespace[2pt]")
        lines.append(
            _row(
                SOURCE_LABELS[s],
                _count(_get(node, "records")),
                _pair(_get(node, "record_correct")),
                _plain(_get(node, "rewrite", "missed_share")),
                _weakest(node),
            )
        )
    lines += [r"\end{mttabular*}\nointerlineskip"]  # one rule between the panels
    lines += [
        r"\mtpairwidths{$+$00.0}{[$+$00.0,\,$+$00.0]}",  # the widths of every \mtpair cell below
        r"\begin{mttabular*}{\linewidth}{ll N{0,000} ccc@{}}",
        r"\midrule",
        r"(b) & & \multicolumn{1}{c}{$n$} & Typed fields & Extracted fields "
        r"& \multicolumn{1}{c@{}}{Difference} \\",
        r"\midrule",
    ]
    for cls, _, label in CLASSES:
        first = rf"$B{{\times}}D$ on Update (pp), {label}"
        for b, _ in W_BRANCHES:
            node = _get(q, "bxd", b, cls)
            lines.append(
                _row(
                    first,
                    BRANCH_LABELS[b],
                    _n(_get(node, "difference")),
                    _pair(_get(node, "typed"), sign=True),
                    _pair(_get(node, "extracted"), sign=True),
                    _pair(_get(node, "difference"), sign=True),
                )
            )
            first = ""
        lines.append(r"\addlinespace[2pt]")
    first = r"$r$ (\%), definition items"
    for b, _ in W_BRANCHES:
        node = _get(q, "r", b, "pooled")
        lines.append(
            _row(
                first,
                BRANCH_LABELS[b],
                _n(_get(node, "difference")),
                _pair(_get(node, "D1s")),
                _pair(_get(node, "D1sx")),
                _pair(_get(node, "difference"), sign=True),
            )
        )
        first = ""
    lines += [
        r"\bottomrule",
        r"\end{mttabular*}",
        r"\mtnotes{(a) Per source: the distinct record contents, the share extracted with every "
        r"field right, the share of rewrites that miss a value, and the field with the fewest "
        r"values extracted right. The last two columns are point estimates. (b) $B{\times}D$ and "
        rf"$r$ with programs on typed and on extracted fields. {_differ(q)}}}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ======================================================================================= files
def load(analysis_dir: Path, blocks: Sequence[str]) -> dict[str, dict[str, Any] | None]:
    """The analyses of the requested blocks (None for a block left out or a missing file)."""
    out: dict[str, dict[str, Any] | None] = {
        b: read_json(analysis_dir / name) if b in blocks else None for b, name in BLOCKS.items()
    }
    out["facts"] = read_json(analysis_dir / FACTS) if "t" in blocks else None
    return out


def head(blocks: Sequence[str]) -> str:
    return f"% generated by {MODULE} --blocks {','.join(blocks)}; do not edit\n"


def numbers(inp: Mapping[str, Any]) -> str:
    lines = [
        "% typed input (t.json, t_facts.json)",
        *(_macro(k, v) for k, v in typed_numbers(inp.get("t"), inp.get("facts")).items()),
        "% execution feedback (f.json)",
        *(_macro(k, v) for k, v in feedback_numbers(inp.get("f")).items()),
        "% extracted fields (w.json)",
        *(_macro(k, v) for k, v in extract_numbers(inp.get("w")).items()),
    ]
    return "\n".join(lines) + "\n"


def build(inp: Mapping[str, Any], blocks: Sequence[str]) -> dict[str, str]:
    """Every output file's text, keyed by file name."""
    top = head(blocks)
    return {
        "round3b_numbers.tex": top + numbers(inp),
        "tab_typed.tex": top + tab_typed(inp.get("t")),
        "tab_feedback.tex": top + tab_feedback(inp.get("f")),
        "tab_extract.tex": top + tab_extract(inp.get("w")),
    }


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--analysis-dir", type=Path, default=Path("runs/stage9_round3b/analysis"))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    ap.add_argument("--blocks", default="t,f,w", help="comma-separated subset of t,f,w")
    a = ap.parse_args(argv)
    blocks = [b for b in a.blocks.split(",") if b]
    unknown = sorted(set(blocks) - set(BLOCKS))
    if unknown:
        ap.error(f"unknown block(s): {', '.join(unknown)}")
    blocks = [b for b in BLOCKS if b in blocks]
    a.out.mkdir(parents=True, exist_ok=True)
    for name, text in build(load(a.analysis_dir, blocks), blocks).items():
        (a.out / name).write_text(text)
        print(a.out / name)


if __name__ == "__main__":
    main()

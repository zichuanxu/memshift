"""Paper numbers, tables and the figure of the account across writers, the out-of-sample
baselines, the cited-cluster key's misses, the zero system and the post-shift JSON arm. Every macro
name starts with `Ex` and every source is a staged analysis named in results/sources.json:

  ex_account   analysis/ex_account.json (`analysis.ex_account`): the account across writers, classes
               and branches; the test across the cells
  ex_more      analysis/ex_e1_more.json (`analysis.ex_e1_more`): out-of-sample baselines, the
               twelve-to-three-month transfer, the accuracy of the items a key missed, the zero
               system after the shift
  ex_misses    analysis/ex_misses.json (`analysis.ex_misses`): why the cited-cluster key misses
  e1, rep_e1   analysis/e1_metrics.json (the post-shift JSON arm, `explore_json_arm`)

A stage that is not staged renders as the MISSING marker; a quantity that is undefined by
construction (an accuracy over no item) renders as a dash.

tab_writers      the Eq. (1) terms r0, r1 and B x D by writer, item class and branch, and the rank
                 correlation and least-squares line of B x D on the repair gap across the cells
tab_zero_after   Update and Stability of the zero system against HoldMem(1,1) and HoldMem(0,0)
tab_json_arm     the post-shift JSON arm against the s_k arm
fig_account      B x D against the repair gap r1 - r0 of the same cell, one marker per writer, class
                 and branch, with the fitted line and its bootstrap band
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.ex_e1_paper import (
    GROUP_GAP,
    PCT_BOX,
    UNDEFINED,
    _Out,
    fmt,
    mt_ci_cells,
    mt_ci_pair_cells,
    point,
)
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.analysis.paper_tables import PAIR, _num, _row

WRITER_TAGS = (
    ("main", "Main", "Qwen3.5-9B"),
    ("rep", "Rep", "Qwen3.5-9B"),
    ("gemma", "Gem", "gemma-4-12B"),
    ("llama", "Llama", "Llama-3.1-8B"),
    ("gptoss", "Gpt", "gpt-oss-20b"),
)
ACCOUNT_CLASSES = (
    ("definition", "Definition", "definition"),
    ("rule", "Field", "field"),
    ("norule", "Free", "free text"),
    ("other", "Other", "other"),
)
TERM_TAGS = (  # key, tag, formatter
    ("m", "Missed", pct),
    ("r0", "RZero", pct),
    ("r1", "ROne", pct),
    ("a0", "AZero", pct),
    ("a1", "AOne", pct),
    ("B_D0", "BDzero", pp),
    ("B_D1", "BDone", pp),
    ("D_B0", "DBzero", pp),
    ("D_B1", "DBone", pp),
    ("BxD", "BxD", pp),
    ("gap", "Gap", pp),
    ("corrected", "Corrected", pp),
)
SCHEMES = (("two_class", "Two"), ("three_class", "Three"))
PREDICTOR_TAGS = (
    ("gap_linear", "GapLinear"),
    ("class_mean", "ClassMean"),
    ("class_branch_mean", "ClassBranchMean"),
    ("grand_mean", "GrandMean"),
    ("gap_identity", "GapIdentity"),
    ("gap_and_class", "GapAndClass"),
)
HEADLINE_TAGS = PREDICTOR_TAGS[:2]
BASE_PAIRS = (("dev_to_main", "DevMain"), ("dev_to_rep", "DevRep"), ("main_to_rep", "MainRep"))
BASE_PREDICTORS = (
    ("cell", "Cell"),
    ("structural", "Structural"),
    ("literal", "Literal"),
    ("pooled_cell_mean", "Pooled"),
)
BASE_CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))
CELL_TAGS = (
    ("D0B0", "DzeroBzero"),
    ("D1B0", "DoneBzero"),
    ("D0B1_source", "DzeroBoneSource"),
    ("D1B1_source", "DoneBoneSource"),
    ("D0B1_cited", "DzeroBoneCited"),
    ("D1B1_cited", "DoneBoneCited"),
)
MISSED_TAGS = (
    ("missed_share", "Share", pct),
    ("a_missed_B1_cited", "ACited", pct),
    ("a_missed_B0", "AZeroOnMissed", pct),
    ("a_missed_B1_minus_B0", "ADiff", pp),
    ("r_invalidated_B1_cited", "RCited", pct),
    ("r_invalidated_B0", "RBzero", pct),
    ("update_B1_cited", "UpdateCited", pct),
    ("update_B0", "UpdateBzero", pct),
    ("B_effect_cited", "BEffect", pp),
)
SYSTEM_TAGS = (
    ("zero", "Zero"),
    ("rag", "Rag"),
    ("oracle", "Oracle"),
    ("holdmem_D0B0", "HoldZeroZero"),
    ("holdmem_D1B0", "HoldOneZero"),
    ("holdmem_D0B1_source", "HoldZeroOne"),
    ("holdmem_D1B1_source", "HoldOneOne"),
)
ZERO_DIFFS = (("holdmem_D1B1", "HoldOneOne"), ("holdmem_D0B0", "HoldZeroZero"))
MISS_CLASSES = (
    ("definition", "Definition", "definition"),
    ("other", "Other", "other"),
    ("rule", "Field", "field"),
    ("norule", "Free", "free text"),
    ("all", "All", "all"),
)
MISS_TAGS = (
    ("a", "A"),
    ("b", "B"),
    ("b_existing", "BExisting"),
    ("b_new", "BNew"),
    ("c", "C"),
    ("c_definition", "CDefinition"),
    ("c_left_scope", "CLeft"),
    ("c_other", "COther"),
    ("a_and_b", "AAndB"),
    ("version_changed", "Version"),
    ("cited_touched", "CitedTouched"),
)
STAGES = (("main", "Main"), ("rep", "Rep"))
MISSING_PAIR = rf"\multicolumn{{2}}{{c}}{{{MISSING}}}"  # a pair of cells not staged
JSON_CLASSES = (("definition", "definition"), ("other", "other"), ("all", "all"))


# ------------------------------------------------------------------------------------ loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs (None where a stage is not there)."""
    src = dict(sources)
    return {
        "acct": stage_json(root, src, "ex_account", "ex_account.json"),
        "more": stage_json(root, src, "ex_more", "ex_e1_more.json"),
        "miss": stage_json(root, src, "ex_misses", "ex_misses.json"),
        "main": stage_json(root, src, "e1", "e1_metrics.json"),
        "rep": stage_json(root, src, "rep_e1", "e1_metrics.json"),
    }


# ----------------------------------------------------------------------------------- formats
def snum(x: float | None, digits: int = 2) -> str:
    """A plain number with a typeset minus."""
    if x is None or x != x:
        return MISSING
    v = round(x, digits)
    return ("$-$" if v < 0 else "") + f"{abs(v):.{digits}f}"


def mae_frac(x: float | None) -> str:
    """A mean absolute error held as a fraction, in pp."""
    return MISSING if x is None or x != x else f"{100 * x:.1f}\\,pp"


def mae_pp(x: float | None) -> str:
    """A mean absolute error held in pp (`ex_account` scales its errors by 100)."""
    return MISSING if x is None or x != x else f"{x:.1f}\\,pp"


def dpp(x: float | None) -> str:
    """A signed difference held in pp."""
    if x is None or x != x:
        return MISSING
    v = round(x, 1)
    return f"{'$+$' if v > 0 else '$-$' if v < 0 else ''}{abs(v):.1f}\\,pp"


def count(x: Any) -> str:
    return MISSING if x is None else str(x)


def of(hits: Any, total: Any) -> str:
    return MISSING if hits is None or total is None else f"{hits} of {total}"


# ---------------------------------------------------------------------------------- macros
def _account_cells(inp: Mapping[str, Any], out: _Out) -> None:
    cells = _get(inp.get("acct"), "explore_account_cells") or {}
    for w, wt, _ in WRITER_TAGS:
        for b, bt in BRANCH_TAGS:
            for cls, ct, _ in ACCOUNT_CLASSES:
                node = _get(cells, w, b, cls) or {}
                n = node.get("changed_item_times")
                out.put("Acct", wt, bt, ct, "Items", value=count(n) if node else MISSING)
                for key, tag, f in TERM_TAGS:
                    out.put("Acct", wt, bt, ct, tag, value=fmt(node.get(key), f))


def _account_fit(inp: Mapping[str, Any], out: _Out) -> None:
    acct = inp.get("acct") or {}
    fit = acct.get("explore_account_fit") or {}
    out.put(
        "AcctDraws",
        value=f"{acct['explore_draws']:,}" if "explore_draws" in acct else MISSING,
    )
    for scheme, st in SCHEMES:
        f = fit.get(scheme) or {}
        classes = f.get("classes") or []
        out.put("AcctFit", st, "Cells", value=count(f.get("n_cells")) if f else MISSING)
        out.put("AcctFit", st, "Spearman", value=fmt(f.get("spearman"), snum))
        out.put("AcctFit", st, "Slope", value=fmt(f.get("slope"), snum))
        out.put("AcctFit", st, "Intercept", value=fmt(f.get("intercept"), pp))
        for cls, ct, _ in ACCOUNT_CLASSES:
            if cls not in classes and f:
                continue  # the class is not a level of this scheme
            e = _get(f, "spearman_by_class", cls)
            out.put("AcctFit", st, ct, "Spearman", value=fmt(e, snum))
        lowo = f.get("lowo") or {}
        for pred, pt in PREDICTOR_TAGS:
            out.put("AcctLowo", st, pt, "Mae", value=fmt(_get(lowo, "mae", pred), mae_pp))
            hits = _get(lowo, "cells_within_5pp", pred)
            out.put("AcctLowo", st, pt, "Hits", value=of(hits, f.get("n_cells")))
        out.put("AcctLowo", st, "Diff", "Mae", value=fmt(lowo.get("mae_gap_minus_class"), dpp))
        for cls, ct, _ in ACCOUNT_CLASSES:
            if cls not in classes and f:
                continue
            for pred, pt in HEADLINE_TAGS:
                e = _get(lowo, "by_class", cls, pred)
                out.put("AcctLowo", st, ct, pt, "Mae", value=fmt(e, mae_pp))
            e = _get(lowo, "by_class_mae_gap_minus_class", cls)
            out.put("AcctLowo", st, ct, "Diff", "Mae", value=fmt(e, dpp))
        for w, wt, _ in WRITER_TAGS:
            if w == "rep":
                continue  # R is not a fitted writer
            for pred, pt in HEADLINE_TAGS:
                e = _get(lowo, "by_writer", w, pred)
                out.put("AcctLowo", st, wt, pt, "Mae", value=fmt(e, mae_pp))
        for label, lt in (("all_fitted", "All"), ("other_lineages", "Lineage")):
            for pred, pt in PREDICTOR_TAGS:
                e = _get(f, f"heldout_{label}_mae", pred)
                out.put("AcctRHeld", lt, st, pt, "Mae", value=fmt(e, mae_pp))
            e = f.get(f"heldout_{label}_mae_gap_minus_class")
            out.put("AcctRHeld", lt, st, "Diff", "Mae", value=fmt(e, dpp))


def _baselines(inp: Mapping[str, Any], out: _Out) -> None:
    base = _get(inp.get("more"), "explore_oos_baselines") or {}
    for pair, pt in BASE_PAIRS:
        res = base.get(pair) or {}
        for pred, pred_t in BASE_PREDICTORS:
            for scope, scope_t in (*BRANCH_TAGS, ("pooled_over_branches", "Pooled")):
                for cls, ct in BASE_CLASSES:
                    if scope == "pooled_over_branches":
                        node = _get(res, scope, cls, pred) or {}
                    else:
                        node = _get(res, "branches", scope, cls, pred) or {}
                    e = node.get("mean_abs_error")
                    out.put("Base", pt, pred_t, scope_t, ct, "Mae", value=fmt(e, mae_frac))
                    out.put(
                        "Base",
                        pt,
                        pred_t,
                        scope_t,
                        ct,
                        "Hits",
                        value=of(node.get("cells_within_5pp"), node.get("n_cells")),
                    )
                    if pair == "dev_to_main" and scope != "pooled_over_branches":
                        for cell, cell_t in CELL_TAGS:
                            c = _get(node, "cells", cell) or {}
                            head = ("Base", pt, pred_t, scope_t, ct, cell_t)
                            out.put(*head, "Err", value=fmt(c.get("error"), pp))
                            out.put(
                                *head,
                                "Predicted",
                                value=point(c.get("predicted")) if c else MISSING,
                            )
                            out.put(
                                *head, "Observed", value=point(c.get("observed")) if c else MISSING
                            )
    hz = _get(inp.get("more"), "explore_horizon_transfer", "main_to_short") or {}
    for pred, pred_t in BASE_PREDICTORS[::3]:  # cell and pooled_cell_mean
        for scope, scope_t in (*BRANCH_TAGS, ("pooled_over_branches", "Pooled")):
            for cls, ct in BASE_CLASSES:
                if scope == "pooled_over_branches":
                    node = _get(hz, scope, cls, pred) or {}
                else:
                    node = _get(hz, "branches", scope, cls, pred) or {}
                out.put(
                    "Horizon",
                    pred_t,
                    scope_t,
                    ct,
                    "Mae",
                    value=fmt(node.get("mean_abs_error"), mae_frac),
                )
                out.put(
                    "Horizon",
                    pred_t,
                    scope_t,
                    ct,
                    "Hits",
                    value=of(node.get("cells_within_5pp"), node.get("n_cells")),
                )
                if scope == "pooled_over_branches":
                    continue
                for cell, cell_t in CELL_TAGS[:4]:
                    c = _get(node, "cells", cell) or {}
                    hz_head = ("Horizon", pred_t, scope_t, ct, cell_t)
                    out.put(*hz_head, "Err", value=fmt(c.get("error"), pp))
                    out.put(
                        *hz_head, "Predicted", value=point(c.get("predicted")) if c else MISSING
                    )
                    out.put(*hz_head, "Observed", value=point(c.get("observed")) if c else MISSING)


def _missed_items(inp: Mapping[str, Any], out: _Out) -> None:
    res = _get(inp.get("more"), "explore_missed_items") or {}
    for stage, st in STAGES:
        for b, bt in BRANCH_TAGS:
            for d, dt in ((0, "Dzero"), (1, "Done")):
                for cls, ct in BASE_CLASSES:
                    node = _get(res, stage, b, f"D{d}", cls) or {}
                    out.put(
                        "Missed",
                        st,
                        bt,
                        dt,
                        ct,
                        "Items",
                        value=count(node.get("changed_item_times")) if node else MISSING,
                    )
                    out.put(
                        "Missed",
                        st,
                        bt,
                        dt,
                        ct,
                        "Count",
                        value=count(node.get("missed_item_times")) if node else MISSING,
                    )
                    for key, tag, f in MISSED_TAGS:
                        out.put("Missed", st, bt, dt, ct, tag, value=fmt(node.get(key), f))


def _zero_system(inp: Mapping[str, Any], out: _Out) -> None:
    res = _get(inp.get("more"), "explore_zero_system") or {}
    for stage, st in STAGES:
        for b, bt in BRANCH_TAGS:
            for cls, ct in BASE_CLASSES:
                node = _get(res, stage, b, cls) or {}
                for _metric, mt, key in (
                    ("update", "Update", "update"),
                    ("stability", "Stab", "stability"),
                ):
                    for system, sys_t in SYSTEM_TAGS:
                        out.put(
                            "Zero", st, bt, ct, mt, sys_t, value=fmt(_get(node, key, system), pct)
                        )
                    for ref, ref_t in ZERO_DIFFS:
                        e = _get(node, "zero_minus", ref, key)
                        out.put("Zero", st, bt, ct, mt, "Diff", ref_t, value=fmt(e, pp))


def _miss_classes(inp: Mapping[str, Any], out: _Out) -> None:
    res = _get(inp.get("miss"), "explore_miss_classes") or {}
    for stage, st in STAGES:
        chk = _get(res, stage, "checks") or {}
        out.put("MissClass", st, "Worlds", value=count(_get(res, stage, "worlds")))
        out.put(
            "MissClass",
            st,
            "Verified",
            value=of(chk.get("events_hash_matches"), chk.get("world_branches")),
        )
        out.put("MissClass", st, "Missed", value=count(chk.get("missed")))
        for b, bt in BRANCH_TAGS:
            for cls, ct, _ in MISS_CLASSES:
                node = _get(res, stage, b, cls) or {}
                out.put(
                    "MissClass",
                    st,
                    bt,
                    ct,
                    "Items",
                    value=count(node.get("missed_item_times")) if node else MISSING,
                )
                for key, tag in MISS_TAGS:
                    out.put("MissClass", st, bt, ct, tag, value=fmt(node.get(key), pct))
                    c = node.get(f"{key}_count")
                    out.put(
                        "MissClass", st, bt, ct, tag, "Count", value=count(c) if node else MISSING
                    )


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every `Ex` macro of this module, from `load`."""
    out = _Out()
    for block in (
        _account_cells,
        _account_fit,
        _baselines,
        _missed_items,
        _zero_system,
        _miss_classes,
    ):
        block(inp, out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ---------------------------------------------------------------------------------- tables
def _r(e: dict[str, Any] | None) -> str:
    """An accuracy in percent, no decimals; a dash where no item was re-derived."""
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=0)


WRITER_ROWS = (  # tab_writers: class, label (the other items, then their two kinds)
    ("definition", "definition"),
    ("other", "other"),
    ("rule", r"\mtin field"),
    ("norule", r"\mtin free text"),
)


WRITER_STUDIES = {"main": "main study", "rep": "replication"}  # the second line of a stub


def tab_writers(inp: Mapping[str, Any]) -> str:
    """Appendix: r0, r1 and B x D (source-epoch binding) by writer, item class and branch (the rank
    correlation and least-squares line of B x D on the repair gap across the cells of the four
    fitted writers are in the text, from the `AcctFitTwo` macros). Qwen3.5-9B's two blocks name
    their study on the stub's second line."""
    acct = inp.get("acct")
    cells = _get(acct, "explore_account_cells") or {}
    shift = rf"N{{{PCT_BOX}}}N{{{PCT_BOX}}}{PAIR}"  # r0, r1 and B x D after one shift
    head = (
        r" & \multicolumn{1}{c}{$r_0$} & \multicolumn{1}{c}{$r_1$} "
        r"& \multicolumn{2}{c}{$B{\times}D$}"
    )
    edge = head.replace("{2}{c}", "{2}{c@{}}")  # the last shift, at the right edge
    lines = [
        r"\begin{table*}[t]",
        r"\caption{The account by writer and item class: repair correctness $r_D$ at $D$ (\%) "
        r"and $B{\times}D$ on Update (pp).}",
        r"\label{tab:writers}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}ll {f' {GROUP_GAP} '.join([shift] * 3)}@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{$\Ssrc$} & \multicolumn{4}{c}{$\Sdef$} "
        r"& \multicolumn{4}{c@{}}{$\Sdel$} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(l){11-14}",
        "Writer & Class" + head * 2 + edge + r" \\",
        r"\midrule",
    ]
    for i, (w, _, label) in enumerate(WRITER_TAGS):
        if i:
            lines.append(r"\mtgap")
        stub = [label, rf"\mtmuted{{{WRITER_STUDIES[w]}}}" if w in WRITER_STUDIES else ""]
        for k, (cls, cname) in enumerate(WRITER_ROWS):
            vals: list[str] = []
            for b, _ in BRANCH_TAGS:
                n = _get(cells, w, b, cls)
                vals += [
                    _r(_get(n, "r0")),
                    _r(_get(n, "r1")),
                    mt_ci_cells(_get(n, "BxD"), sign=True, span="c@{}" if b == "orphan" else "c"),
                ]
            lines.append(_row(stub[k] if k < len(stub) else "", cname, *vals))
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        r"\mtnotes{Field: single-field items. Free text: free-text items. Other: both. $r_0$, "
        r"$r_1$: point estimates. Qwen3.5-9B ran on 20 worlds in the main study and in the "
        r"replication, the other writers on ten.}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def _share(e: dict[str, Any] | None) -> str:
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=0)


def tab_zero_after(inp: Mapping[str, Any]) -> str:
    """Appendix: the zero system against HoldMem(1,1) and HoldMem(0,0) on Update and Stability after
    the shift, per branch and class (definition, other, all items), in the main study and R."""
    res = _get(inp.get("more"), "explore_zero_system") or {}
    share = f"N{{{PCT_BOX}}}"  # a column of whole percentages
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Update and Stability of no-store BM25 (no-store), \holdmem{}(1,1) and "
        r"\holdmem{}(0,0) by item class after each shift, and no-store BM25 minus "
        r"\holdmem{}(1,1) on Update, paired.}",
        r"\label{tab:zero-after}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        # one tabular for both stages: one column grid
        rf"\begin{{mttabular}}{{@{{}}ll {share * 3} {PAIR} {GROUP_GAP} {share * 3}@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{5}{c}{Update (\%)} & \multicolumn{3}{c@{}}{Stability (\%)} \\",
        r"\cmidrule(lr){3-7}\cmidrule(l){8-10}",
        r"Shift & Class & \multicolumn{1}{c}{No-store} & \multicolumn{1}{c}{(1,1)} "
        r"& \multicolumn{1}{c}{(0,0)} & \multicolumn{2}{c}{No-store $-$ (1,1) (pp)} "
        r"& \multicolumn{1}{c}{No-store} & \multicolumn{1}{c}{(1,1)} "
        r"& \multicolumn{1}{c@{}}{(0,0)} \\",
    ]
    for stage, title in (("main", "Main study"), ("rep", "Replication")):
        lines += [r"\midrule", rf"\mtgroup{{10}}{{{title}}}"]
        for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
            for k, cls in enumerate(("definition", "other", "all")):
                n = _get(res, stage, b, cls)
                up, st = _get(n, "update") or {}, _get(n, "stability") or {}
                cells = [
                    _share(up.get("zero")),
                    _share(up.get("holdmem_D1B1_source")),
                    _share(up.get("holdmem_D0B0")),
                    mt_ci_pair_cells(_get(n, "zero_minus", "holdmem_D1B1", "update"), sign=True),
                    _share(st.get("zero")),
                    _share(st.get("holdmem_D1B1_source")),
                    _share(st.get("holdmem_D0B0")),
                ]
                if not n:
                    cells = [*[MISSING] * 3, mt_ci_pair_cells(None), *[MISSING] * 3]
                lines.append(_row(rf"\mtin{bt}" if k == 0 else "", cls, *cells))
            if b != "orphan":
                lines.append(r"\mtgap")
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        r"\mtnotes{Update and Stability levels: point estimates.}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def tab_json_arm(inp: Mapping[str, Any]) -> str:
    """Appendix: the D=1 store served as JSON against the same store served as s_k statements, on
    the post-shift probes: main study and R side by side, per branch and class."""
    stages = ("main", "rep")
    group = rf"N{{{PCT_BOX}}}N{{{PCT_BOX}}} r@{{\,}}l r@{{\,}}l"
    head = (
        r" & \multicolumn{1}{c}{$s_k$} & \multicolumn{1}{c}{JSON} & \multicolumn{2}{c}{Update} "
        r"& \multicolumn{2}{c}{Stability}"
    )
    lines = [
        r"\begin{table*}[t]",
        r"\caption{The same $D{=}1$ store served as JSON records instead of statements ($s_k$), "
        r"in the main study and the replication.}",
        r"\label{tab:json-arm}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{mttabular}}{{@{{}}ll {group} {GROUP_GAP} {group}@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{6}{c}{Main study} & \multicolumn{6}{c@{}}{Replication} \\",
        r"\cmidrule(lr){3-8}\cmidrule(l){9-14}",
        r"& & \multicolumn{2}{c}{Update (\%)} & \multicolumn{4}{c}{JSON $-$ $s_k$ (pp)} "
        r"& \multicolumn{2}{c}{Update (\%)} & \multicolumn{4}{c@{}}{JSON $-$ $s_k$ (pp)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-10}\cmidrule(l){11-14}",
        "Shift & Class" + head + head.replace("{2}{c}{Stab", "{2}{c@{}}{Stab") + r" \\",
        r"\midrule",
    ]
    for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
        for k, (cls, cname) in enumerate(JSON_CLASSES):
            cells: list[str] = []
            for stage in stages:
                j = _get(inp.get(stage), "explore_json_arm", b, cls)
                if not j:
                    cells += [MISSING, MISSING, MISSING_PAIR, MISSING_PAIR]
                    continue
                cells += [
                    _share(_get(j, "update_sk")),
                    _share(_get(j, "update_json")),
                    mt_ci_cells(_get(j, "update_json_minus_sk"), sign=True),
                    mt_ci_cells(
                        _get(j, "stability_json_minus_sk"),
                        sign=True,
                        span="c@{}" if stage == "rep" else "c",
                    ),
                ]
            lines.append(_row(bt if k == 0 else "", cname, *cells))
        if b != "orphan":
            lines.append(r"\mtgap")
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        r"\mtnotes{Update on the changed item-times in percent, and the paired differences JSON "
        r"$-$ $s_k$ in Update and in Stability on the unchanged item-times. Update levels: point "
        r"estimates.}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "tab_writers.tex": tab_writers(inp),
        "tab_zero_after.tex": tab_zero_after(inp),
        "tab_json_arm.tex": tab_json_arm(inp),
    }


# ---------------------------------------------------------------------------------- figure
WRITER_MODELS = {  # writer of the account -> model id (its colour in `figstyle.MODELS`)
    "main": "Qwen/Qwen3.5-9B",
    "rep": "Qwen/Qwen3.5-9B",
    "gemma": "google/gemma-4-12B-it",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "gptoss": "openai/gpt-oss-20b",
}
FITTED_WRITERS = ("main", "gemma", "llama", "gptoss")  # the replication R is Qwen3.5-9B's hue
CLASS_MARKERS = {"definition": "o", "other": "^"}
ZOOM = ((-3.0, 12.0), (-3.0, 8.0))  # the inset's x and y range (pp): the crowded other items
ACCOUNT_IN = (4.5, 3.0)  # printed size (in): `figstyle.NARROW` wide
ACCOUNT_MS = 5.0  # marker size (pt); a triangle is drawn a little larger to look the same
ACCOUNT_LABEL, ACCOUNT_TICK, ACCOUNT_KEY, ACCOUNT_NOTE = 8.0, 7.2, 7.2, 7.0  # pt at print
ACCOUNT_MUTED = "#4A5059"  # the inset's tick labels
ACCOUNT_GRID = "#ECEEF1"
ACCOUNT_FIT, ACCOUNT_BAND = "#3A4250", "#DDE1E6"  # the fit (neutral slate: not a writer), band
ACCOUNT_DIAG, ACCOUNT_ZERO = "#7D8794", "#8C95A1"  # the diagonal B x D = r1 - r0, the zero line
ACCOUNT_ZOOM_FILL, ACCOUNT_ZOOM_EDGE = "#F2F4F7", "#9AA3AE"  # the framed region and its inset
ACCOUNT_NEUTRAL = "#5E6875"  # the class glyphs of the key


def _fit_cells(fit: Mapping[str, Any]) -> list[tuple[dict[str, Any], bool]]:
    """The cells of the account with a finite gap and B x D, each with whether it is held out."""
    return [
        (c, held_out)
        for key, held_out in (("cells", False), ("heldout_cells", True))
        for c in fit.get(key) or []
        if c["gap"] == c["gap"] and c["BxD"] == c["BxD"]
    ]


def _spread(ax: Any, xy: list[tuple[float, float]], size_pt: float) -> list[float]:
    """Horizontal offsets (points) for markers that would cover one another: markers whose
    centres lie closer than `size_pt` on the page form a cluster, drawn side by side around the
    cluster's centre in the order of their x."""
    import math

    pos = [ax.transData.transform(p) for p in xy]
    scale = ax.figure.dpi / 72.0
    group = list(range(len(xy)))
    for i in range(len(xy)):
        for j in range(i):
            if math.dist(pos[i], pos[j]) / scale < size_pt:
                old, new = group[i], group[j]
                group = [new if g == old else g for g in group]
    dx = [0.0] * len(xy)
    for g in set(group):
        members = sorted((i for i in range(len(xy)) if group[i] == g), key=lambda i: pos[i][0])
        for k, i in enumerate(members):
            dx[i] = (k - (len(members) - 1) / 2) * size_pt
    return dx


def _along(ax: Any, x0: float, y0: float, slope: float) -> float:
    """The angle (degrees) on the page of a line of `slope` through (x0, y0)."""
    import math

    (a, b), (c, d) = ax.transData.transform([(x0, y0), (x0 + 10, y0 + 10 * slope)])
    return math.degrees(math.atan2(d - b, c - a))


def _plain_num(x: float) -> str:
    """As `snum`, in plain text with a typographic minus."""
    v = round(x, 2)
    return ("−" if v < 0 else "") + f"{abs(v):.2f}"


def fig_account(inp: Mapping[str, Any], out: Path) -> Path:
    """B x D against the repair gap r1 - r0 of the same cell: one marker per writer (colour, the
    model colours), item class (shape: circle definition, triangle other) and branch, hollow for
    the replication (held out); the least-squares line over the fitted cells with its 95%
    bootstrap band, labelled with its slope and Spearman rho, and the dotted diagonal B x D =
    r1 - r0, each labelled along the line; an inset zooms on the crowded other items near the
    origin, its region framed on the same pale ground. The key is two rows above the axes: the
    writers, then the classes with their Spearman rho and the replication."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle
    from matplotlib.transforms import ScaledTranslation

    from memshift.analysis import figstyle as fs

    fit = _get(inp.get("acct"), "explore_account_fit", "two_class") or {}
    cells = _fit_cells(fit)
    line = fit.get("fit_line")
    slope, intercept = _get(fit, "slope", "est"), _get(fit, "intercept", "est")
    rho = _get(fit, "spearman", "est")
    rho_cls = {c: _get(fit, "spearman_by_class", c, "est") for c in ("definition", "other")}
    fitted = sum(not held_out for _, held_out in cells)
    w_mm, h_mm = ACCOUNT_IN[0] * 25.4, ACCOUNT_IN[1] * 25.4
    with fs.printed() as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM))
        ax_l, ax_b, ax_r, ax_t = 13.0, 8.4, 1.6, 11.2  # the axes box (mm), the key above it
        ax = fig.add_axes((ax_l / w_mm, ax_b / h_mm, (w_mm - ax_l - ax_r) / w_mm,
                           (h_mm - ax_b - ax_t) / h_mm))  # fmt: skip
        fs.chart_axes(ax, size=ACCOUNT_TICK + 0.5)
        ax.set_xlim(-4, 97)
        ax.set_ylim(-5, 57)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_yticks([0, 10, 20, 30, 40, 50])
        for v in ax.get_xticks():
            ax.axvline(v, color=ACCOUNT_GRID, lw=0.5, zorder=0)
        ax.set_xlabel(r"$r_1 - r_0$ of the cell (pp)", fontsize=ACCOUNT_LABEL, color=fs.INK,
                      labelpad=2)  # fmt: skip
        ax.set_ylabel(r"$B{\times}D$ on Update (pp)", fontsize=ACCOUNT_LABEL, color=fs.INK,
                      labelpad=2)  # fmt: skip
        fig.canvas.draw()  # settle the layout: the labels along the lines need the page angles

        def guides(a: Any, lo: float, hi: float, labels: bool) -> None:
            """The diagonal, the zero line and the fit with its band, on the main axes or the
            inset; `labels` names the fit and the diagonal along them."""
            a.plot([lo, hi], [lo, hi], ls=(0, (1, 1.6)), color=ACCOUNT_DIAG, lw=0.9, zorder=1)
            a.axhline(0, color=ACCOUNT_ZERO, lw=0.6, zorder=1)
            if not line or slope is None or slope != slope or intercept is None:
                return
            grid = [100 * g for g in line["grid"]]
            a.fill_between(grid, [100 * v for v in line["lo"]], [100 * v for v in line["hi"]],
                           color=ACCOUNT_BAND, lw=0, zorder=1.5)  # fmt: skip
            a.plot(grid, [100 * intercept + slope * g for g in grid], color=ACCOUNT_FIT, lw=1.1,
                   zorder=2, solid_capstyle="butt")  # fmt: skip
            if not labels:
                return
            x = 50.0  # the empty middle, below the band and right of the diagonal
            text = f"least-squares fit over the {fitted} cells:  slope {slope:.2f}"
            if rho is not None:
                text += f",  $\\rho = {rho:.2f}$"
            a.annotate(text, (x, 100 * intercept + slope * x), xytext=(0, -9.5),
                       textcoords="offset points", rotation=_along(a, x, 0, slope),
                       rotation_mode="anchor", ha="center", va="top", fontsize=ACCOUNT_NOTE,
                       color=ACCOUNT_FIT)  # fmt: skip
            x = 41.0
            a.annotate(r"$B{\times}D = r_1 - r_0$", (x, x), xytext=(2.5, -3.5),
                       textcoords="offset points", rotation=_along(a, x, x, 1.0),
                       rotation_mode="anchor", ha="center", va="top",
                       fontsize=ACCOUNT_NOTE + 0.5, color=ACCOUNT_DIAG)  # fmt: skip

        def marks(a: Any, spread: bool) -> None:
            """One marker per cell: colour = writer, shape = item class, hollow = held out (R).
            With `spread`, markers that would cover one another are drawn side by side."""
            xy = [(100 * c["gap"], 100 * c["BxD"]) for c, _ in cells]
            dxs = _spread(a, xy, ACCOUNT_MS * 0.82) if spread else [0.0] * len(xy)
            for (c, held_out), (x, y), dx in zip(cells, xy, dxs, strict=True):
                _, colour = fs.model_style(WRITER_MODELS.get(c["writer"], c["writer"]))
                marker = CLASS_MARKERS.get(c["class"], "s")
                tr = a.transData + ScaledTranslation(dx / 72, 0, a.figure.dpi_scale_trans)
                a.plot([x], [y], marker=marker, ms=ACCOUNT_MS * (1.14 if marker == "^" else 1.0),
                       mfc="white" if held_out else colour, mec=colour if held_out else "white",
                       mew=0.95 if held_out else 0.45, ls="none", zorder=4 if held_out else 3.5,
                       transform=tr)  # fmt: skip

        (zx0, zx1), (zy0, zy1) = ZOOM
        zoomed = any(zx0 <= 100 * c["gap"] <= zx1 and zy0 <= 100 * c["BxD"] <= zy1
                     for c, _ in cells)  # fmt: skip
        if zoomed:  # the framed region near the origin and its inset share one pale ground
            ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fc=ACCOUNT_ZOOM_FILL,
                                   ec="none", zorder=0.5))  # fmt: skip
        guides(ax, -5.0, 100.0, labels=True)
        marks(ax, spread=False)
        if not cells:
            ax.text(0.5, 0.5, "no cell staged", transform=ax.transAxes, ha="center",
                    fontsize=ACCOUNT_NOTE, color=fs.MUTED)  # fmt: skip
        if zoomed:
            ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False,
                                   ec=ACCOUNT_ZOOM_EDGE, lw=0.6, zorder=5))  # fmt: skip
            # the inset in the empty upper left, above the diagonal; a white margin clears the
            # grid from under its tick labels
            ax.add_patch(Rectangle((0.015, 0.545), 0.305, 0.455, transform=ax.transAxes,
                                   fc="white", ec="none", zorder=4.9))  # fmt: skip
            ins = ax.inset_axes((0.075, 0.625, 0.235, 0.355), zorder=5)
            ins.set_xlim(zx0, zx1)
            ins.set_ylim(zy0, zy1)
            ins.set_facecolor(ACCOUNT_ZOOM_FILL)
            for v in (0, 5, 10):
                ins.axvline(v, color="white", lw=0.6, zorder=0)
            guides(ins, zx0, zx1, labels=False)
            marks(ins, spread=True)
            ins.set_xticks([0, 5, 10])
            ins.set_yticks([0, 5])
            ins.tick_params(labelsize=ACCOUNT_TICK - 0.2, length=1.8, width=0.5, pad=1.2,
                            color=ACCOUNT_ZOOM_EDGE, labelcolor=ACCOUNT_MUTED)  # fmt: skip
            for spine in ins.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.6)
                spine.set_color(ACCOUNT_ZOOM_EDGE)

        # the key: the writers (hue), then the class (shape) and the replication (hollow)
        def handle(marker: str, face: str, edge: str, mew: float, ms: float = ACCOUNT_MS) -> Any:
            return Line2D([], [], marker=marker, ms=ms, mfc=face, mec=edge, mew=mew, ls="none")

        row1 = []
        for w in FITTED_WRITERS:
            label, colour = fs.model_style(WRITER_MODELS[w])
            row1.append((handle("o", colour, colour, 0.0, ACCOUNT_MS + 0.4), label))

        def rho_of(cls: str) -> str:
            r = rho_cls[cls]
            return "" if r is None else f", $\\rho = {_plain_num(r)}$"

        row2 = [
            (handle("o", ACCOUNT_NEUTRAL, "white", 0.4), "definition items" + rho_of("definition")),
            (handle("^", ACCOUNT_NEUTRAL, "white", 0.4, ACCOUNT_MS * 1.14),
             "other items" + rho_of("other")),
            (handle("o", "white", fs.BLACK, 0.95), "replication, not fitted"),
        ]  # fmt: skip
        common = {"frameon": False, "fontsize": ACCOUNT_KEY, "handlelength": 0.9,
                  "handletextpad": 0.35, "borderaxespad": 0, "labelcolor": fs.INK}  # fmt: skip
        for row, y, spacing in ((row1, 1 - 3.0 / h_mm, 1.5), (row2, 1 - 7.6 / h_mm, 1.3)):
            fig.legend([h for h, _ in row], [t for _, t in row], loc="center", ncol=len(row),
                       bbox_to_anchor=(0.53, y), columnspacing=spacing, **common)  # fmt: skip
        return fs.save_pdf(fig, out / "fig_account.pdf")

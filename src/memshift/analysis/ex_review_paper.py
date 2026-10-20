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
  ex_round3_review   analysis/forced.json: what the cells answer on the changed item-times the
               cited-cluster key missed (`tab_misses`)

A stage that is not staged renders as the MISSING marker; a quantity that is undefined by
construction (an accuracy over no item) renders as a dash.

tab_writers      the Eq. (1) terms r0, r1 and B x D by writer, item class and branch, and the rank
                 correlation and least-squares line of B x D on the repair gap across the cells
tab_misses       Update on the changes the cited-cluster key missed: the cited-cluster cell, the
                 forced re-derivation of the source-epoch cell and the append-only cell
tab_zero_after   Update and Stability of the zero system against HoldMem(1,1) and HoldMem(0,0)
tab_json_arm     the post-shift JSON arm against the s_k arm
fig_account      B x D against the repair gap r1 - r0 of the same cell, one marker per writer, class
                 and branch, with the fitted line and its bootstrap band
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.ex_e1_paper import UNDEFINED, _Out, fmt, point
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.analysis.paper_tables import (
    PAIR,
    _num,
    _row,
    ci_pair_cells,
    est_ci_cells,
    spread_columns,
)

WRITER_TAGS = (
    ("main", "Main", "Qwen3.5-9B"),
    ("rep", "Rep", "Qwen3.5-9B, repl."),
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
        "forced": stage_json(root, src, "ex_round3_review", "forced.json"),
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
    ("rule", r"\quad field"),
    ("norule", r"\quad free text"),
)


def tab_writers(inp: Mapping[str, Any]) -> str:
    """Appendix: (a) r0, r1 and B x D (source-epoch binding) by writer, item class and branch;
    (b) the rank correlation and least-squares line of B x D on the repair gap across the cells of
    the four fitted writers."""
    acct = inp.get("acct")
    cells = _get(acct, "explore_account_cells") or {}
    fit = _get(acct, "explore_account_fit", "two_class") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{The account by writer and item class: (a) repair correctness $r_D$ at $D$ "
        r"(\%) and $B{\times}D$ on Update, (b) Spearman's $\rho$ and the least-squares line of "
        r"$B{\times}D$ on $r_1{-}r_0$ across the (writer, class, shift) cells of "
        r"the four fitted writers, with the replication held out. Field: single-field items. "
        r"Free text: free-text items. Other: both. Qwen3.5-9B ran on 20 worlds in the main study "
        r"and in the replication (repl.), the other writers on ten.}",
        r"\label{tab:writers}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{tabular*}}{{\linewidth}}{{{spread_columns('l', 'l', *('r', 'r', 'P') * 3)}}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{$\Ssrc$} & \multicolumn{4}{c}{$\Sdef$} "
        r"& \multicolumn{4}{c}{$\Sdel$} \\",
        r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}\cmidrule(l){11-14}",
        "(a) writer & class" + r" & $r_0$ & $r_1$ & \multicolumn{2}{c}{$B{\times}D$}" * 3 + r" \\",
    ]
    for w, _, label in WRITER_TAGS:
        lines.append(r"\midrule")
        for k, (cls, cname) in enumerate(WRITER_ROWS):
            vals: list[str] = []
            for b, _ in BRANCH_TAGS:
                n = _get(cells, w, b, cls)
                vals += [
                    _r(_get(n, "r0")),
                    _r(_get(n, "r1")),
                    est_ci_cells(_get(n, "BxD"), sign=True),
                ]
            lines.append(_row(label if k == 0 else "", cname, *vals))
    lines += [r"\bottomrule", r"\end{tabular*}", r"\par\medskip"]
    lines += [
        rf"\begin{{tabular*}}{{\linewidth}}{{{spread_columns('l', *'PPPPP')}}}",
        r"\toprule",
        r"(b) across the cells & \multicolumn{2}{c}{$\rho$, all} "
        r"& \multicolumn{2}{c}{$\rho$, definition} & \multicolumn{2}{c}{$\rho$, other} "
        r"& \multicolumn{2}{c}{slope} & \multicolumn{2}{c}{intercept (pp)} \\",
        r"\midrule",
        _row(
            r"$B{\times}D$ on $r_1{-}r_0$",
            est_ci_cells(fit.get("spearman"), f=snum),
            est_ci_cells(_get(fit, "spearman_by_class", "definition"), f=snum),
            est_ci_cells(_get(fit, "spearman_by_class", "other"), f=snum),
            est_ci_cells(fit.get("slope"), f=snum),
            est_ci_cells(fit.get("intercept"), sign=True),
        ),
        r"\bottomrule",
        r"\end{tabular*}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _share(e: dict[str, Any] | None) -> str:
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=0)


FORCED_CELLS = ("cited", "forced", "B0")  # the cells of `tab_misses`, in its column order


def tab_misses(inp: Mapping[str, Any]) -> str:
    """Appendix: Update on the changed item-times the cited-cluster key missed, per shift, class
    and D: the cited-cluster cell (which served the stale item), the source-epoch cell (which
    re-derived the same item-times with the same request) and the append-only cell, and the gain
    of the forced re-derivation over the cited-cluster cell (`ex_round3_review`'s forced.json; the
    accuracies of the missed items of `ex_e1_more` are the cited and B=0 columns at one decimal)."""
    res = _get(inp.get("forced"), "results")
    lines = [
        r"\begin{table}[t]",
        r"\caption{Update on the changed item-times that the citation key did not "
        r"invalidate. Cited: the citation-key cell, which served the stale item. Forced: the "
        r"source-key cell, which re-derived the same item-times with the same request. "
        r"$B{=}0$: the append-only cell. Gain: forced minus cited. $n$: missed item-times.}",
        r"\label{tab:forced}\label{tab:misses}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}ll r c {PAIR} {PAIR} {PAIR} {PAIR}@{{}}}}",
        r"\toprule",
        r"& & & & \multicolumn{6}{c}{Update (\%)} & \multicolumn{2}{c}{} \\",
        r"\cmidrule(lr){5-10}",
        r"shift & class & $n$ & $D$ & \multicolumn{2}{c}{cited} & \multicolumn{2}{c}{forced} "
        r"& \multicolumn{2}{c}{$B{=}0$} & \multicolumn{2}{c}{gain (pp)} \\",
        r"\midrule",
    ]
    for i, (b, bt) in enumerate((("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$"))):
        for k, cls in enumerate(("definition", "other", "all")):
            for j, d in enumerate(("D0", "D1")):
                node = _get(res, d, b, cls)
                update = _get(node, "update") or {}
                cells = [
                    *(ci_pair_cells(update.get(c)) for c in FORCED_CELLS),
                    ci_pair_cells(_get(node, "forced_minus_cited"), sign=True),
                ]
                n = count(_get(node, "n_items"))
                lead = [bt if k == j == 0 else "", cls, n] if j == 0 else ["", "", ""]
                lines.append(_row(*lead, d[1], *cells))
        if i < 2:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_zero_after(inp: Mapping[str, Any]) -> str:
    """Appendix: the zero system against HoldMem(1,1) and HoldMem(0,0) on Update and Stability after
    the shift, per branch and class: main study (the classes; its totals are in the main text) and
    R (the classes and all items)."""
    res = _get(inp.get("more"), "explore_zero_system") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Update and Stability of no-store BM25 (no-store), \textsc{HoldMem}(1,1) and "
        r"\textsc{HoldMem}(0,0) by item class after each shift, and no-store BM25 minus "
        r"\textsc{HoldMem}(1,1) on Update, paired. The totals of the main study are in "
        r"Table~\ref{tab:overview}.}",
        r"\label{tab:zero-after}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        # one tabular for both stages: one column grid
        rf"\begin{{tabular}}{{@{{}}ll rrr {PAIR} rrr@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{5}{c}{Update (\%)} & \multicolumn{3}{c}{Stability (\%)} \\",
        r"\cmidrule(lr){3-7}\cmidrule(l){8-10}",
        r"shift & class & no-store & (1,1) & (0,0) & \multicolumn{2}{c}{no-store $-$ (1,1) (pp)} "
        r"& no-store & (1,1) & (0,0) \\",
    ]
    for stage, title, classes in (
        ("main", "main study", ("definition", "other")),
        ("rep", "replication", ("definition", "other", "all")),
    ):
        lines += [r"\midrule", rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
            for k, cls in enumerate(classes):
                n = _get(res, stage, b, cls)
                up, st = _get(n, "update") or {}, _get(n, "stability") or {}
                cells = [
                    _share(up.get("zero")),
                    _share(up.get("holdmem_D1B1_source")),
                    _share(up.get("holdmem_D0B0")),
                    ci_pair_cells(_get(n, "zero_minus", "holdmem_D1B1", "update"), sign=True),
                    _share(st.get("zero")),
                    _share(st.get("holdmem_D1B1_source")),
                    _share(st.get("holdmem_D0B0")),
                ]
                if not n:
                    cells = [*[MISSING] * 3, ci_pair_cells(None), *[MISSING] * 3]
                lines.append(_row(bt if k == 0 else "", cls, *cells))
            if b != "orphan":
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_json_arm(inp: Mapping[str, Any]) -> str:
    """Appendix: the D=1 store served as JSON against the same store served as s_k statements, on
    the post-shift probes: main study and R side by side, per branch and class."""
    stages = (("main", "main study"), ("rep", "replication"))
    group = r"rr r@{\,}l r@{\,}l"
    lines = [
        r"\begin{table}[t]",
        r"\caption{The same $D{=}1$ store served as JSON records instead of statements ($s_k$), "
        r"in the main study and the replication. Update on the changed item-times in percent, "
        r"and the paired differences JSON $-$ $s_k$ in Update and in Stability on the unchanged "
        r"item-times.}",
        r"\label{tab:json-arm}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{tabular}}{{@{{}}ll {group} {group}@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{6}{c}{main study} & \multicolumn{6}{c}{replication} \\",
        r"\cmidrule(lr){3-8}\cmidrule(l){9-14}",
        r"& & \multicolumn{2}{c}{Update (\%)} & \multicolumn{4}{c}{JSON $-$ $s_k$ (pp)} "
        r"& \multicolumn{2}{c}{Update (\%)} & \multicolumn{4}{c}{JSON $-$ $s_k$ (pp)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-8}\cmidrule(lr){9-10}\cmidrule(l){11-14}",
        "shift & class"
        + r" & $s_k$ & JSON & \multicolumn{2}{c}{Update} & \multicolumn{2}{c}{Stability}" * 2
        + r" \\",
        r"\midrule",
    ]
    for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
        for k, (cls, cname) in enumerate(JSON_CLASSES):
            cells: list[str] = []
            for stage, _ in stages:
                j = _get(inp.get(stage), "explore_json_arm", b, cls)
                if not j:
                    cells += [MISSING, MISSING, MISSING_PAIR, MISSING_PAIR]
                    continue
                cells += [
                    _share(_get(j, "update_sk")),
                    _share(_get(j, "update_json")),
                    est_ci_cells(_get(j, "update_json_minus_sk"), sign=True),
                    est_ci_cells(_get(j, "stability_json_minus_sk"), sign=True),
                ]
            lines.append(_row(bt if k == 0 else "", cname, *cells))
        if b != "orphan":
            lines.append(r"\addlinespace[3pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "tab_writers.tex": tab_writers(inp),
        "tab_misses.tex": tab_misses(inp),
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
CLASS_MARKERS = {"definition": "o", "other": "^"}
ACCOUNT_MS = 4.6  # marker size (points); a triangle is drawn a little larger to look the same
ZOOM = ((-3.0, 12.0), (-3.0, 8.0))  # the inset's x and y range (pp): the crowded other items


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


def _account_marks(ax: Any, cells: list[tuple[dict[str, Any], bool]], spread: bool) -> None:
    """One marker per cell: colour = writer, shape = item class, hollow = held out (R). With
    `spread`, markers that would cover one another are drawn side by side."""
    from matplotlib.transforms import ScaledTranslation

    from memshift.analysis import figstyle as fs

    xy = [(100 * c["gap"], 100 * c["BxD"]) for c, _ in cells]
    dxs = _spread(ax, xy, ACCOUNT_MS * 0.8) if spread else [0.0] * len(xy)
    for (c, held_out), (x, y), dx in zip(cells, xy, dxs, strict=True):
        _, colour = fs.model_style(WRITER_MODELS.get(c["writer"], c["writer"]))
        marker = CLASS_MARKERS.get(c["class"], "s")
        tr = ax.transData + ScaledTranslation(dx / 72, 0, ax.figure.dpi_scale_trans)
        # filled markers get a white rim, which keeps touching markers apart
        ax.plot([x], [y], marker=marker, ms=ACCOUNT_MS * (1.12 if marker == "^" else 1.0),
                mfc="white" if held_out else colour, mec=colour if held_out else "white",
                mew=0.8 if held_out else 0.4, ls="none", zorder=4 if held_out else 3.5,
                transform=tr)  # fmt: skip


def _along(ax: Any, x0: float, y0: float, slope: float) -> float:
    """The angle (degrees) on the page of a line of `slope` through (x0, y0)."""
    import math

    (a, b), (c, d) = ax.transData.transform([(x0, y0), (x0 + 10, y0 + 10 * slope)])
    return math.degrees(math.atan2(d - b, c - a))


def fig_account(inp: Mapping[str, Any], out: Path) -> Path:
    """B x D against the repair gap r1 - r0 of the same cell: one marker per writer (colour, the
    model colours), item class (shape: circle definition, triangle other) and branch, hollow for
    the replication (held out); the fitted line (four fitted writers) with its 95% bootstrap band
    and the diagonal B x D = r1 - r0, each labelled along the line; an inset zooms on the crowded
    other items near the origin. The writers' key is one row above the axes."""
    from matplotlib.patches import Patch, Rectangle

    from memshift.analysis import figstyle as fs

    plt = fs.plt()
    fit = _get(inp.get("acct"), "explore_account_fit", "two_class") or {}
    cells = _fit_cells(fit)
    fig, ax = plt.subplots(figsize=(fs.NARROW, 3.0))
    ax.set_xlim(-4, 97)
    ax.set_ylim(-5, 57)
    ax.set_xlabel(r"$r_1-r_0$ of the cell (pp)")
    ax.set_ylabel(r"$B{\times}D$ on Update (pp)")
    ax.grid(axis="x", visible=True)
    handles = []
    for w in ("main", "gemma", "llama", "gptoss"):  # the fitted writers; R is Qwen3.5-9B's hue
        label, colour = fs.model_style(WRITER_MODELS[w])
        handles.append(Patch(facecolor=colour, edgecolor="none", label=label))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), fontsize=8,
               handlelength=0.9, handleheight=0.9, handletextpad=0.4,
               columnspacing=1.4)  # fmt: skip
    fig.canvas.draw()  # settle the layout: the labels along the lines need the page angles

    def lines(a: Any, lo: float, hi: float, labels: bool) -> None:
        """The diagonal, the zero line and the fit with its band, on the main axes or the inset;
        `labels` names the diagonal and the fit along them."""
        a.plot([lo, hi], [lo, hi], ls=":", color=fs.GREY, lw=0.9, zorder=1)
        a.axhline(0, color=fs.GREY, lw=0.6, zorder=1)
        line = fit.get("fit_line")
        slope, intercept = _get(fit, "slope", "est"), _get(fit, "intercept", "est")
        if not line or slope is None or slope != slope:
            return
        grid = [100 * g for g in line["grid"]]
        a.fill_between(grid, [100 * v for v in line["lo"]], [100 * v for v in line["hi"]],
                       color=fs.LIGHT, lw=0, zorder=1.5)  # fmt: skip
        a.plot(grid, [100 * intercept + slope * g for g in grid], color=fs.BLACK, lw=1.1,
               zorder=2)  # fmt: skip
        if not labels:
            return
        rho = _get(fit, "spearman", "est")
        x = 50.0  # in the empty middle, below the band and right of the diagonal
        text = "least-squares fit" + (
            rf", slope {slope:.2f}, $\rho={rho:.2f}$" if rho is not None else ""
        )
        a.annotate(text, (x, 100 * intercept + slope * x), xytext=(0, -12),
                   textcoords="offset points", rotation=_along(a, x, 0, slope),
                   rotation_mode="anchor", ha="center", va="top", fontsize=8,
                   color=fs.BLACK)  # fmt: skip
        x = 40.0
        a.annotate(r"$B{\times}D=r_1-r_0$", (x, x), xytext=(3, -4), textcoords="offset points",
                   rotation=_along(a, x, x, 1.0), rotation_mode="anchor", ha="center",
                   va="top", fontsize=8, color=fs.GREY)  # fmt: skip

    lines(ax, -5.0, 100.0, labels=True)
    _account_marks(ax, cells, spread=False)
    if not cells:
        ax.text(0.5, 0.5, "no cell staged", transform=ax.transAxes, ha="center", fontsize=8,
                color=fs.GREY)  # fmt: skip
    (zx0, zx1), (zy0, zy1) = ZOOM
    if any(zx0 <= 100 * c["gap"] <= zx1 and zy0 <= 100 * c["BxD"] <= zy1 for c, _ in cells):
        # the inset sits in the empty upper left, above the diagonal, on a white ground that
        # also clears the gridlines from under its tick labels; a thin frame on the main axes
        # marks the region it shows
        ax.add_patch(Rectangle((0.02, 0.555), 0.295, 0.43, transform=ax.transAxes, fc="white",
                               ec="none", zorder=4.9))  # fmt: skip
        ins = ax.inset_axes((0.07, 0.62, 0.24, 0.36), zorder=5)
        ins.set_xlim(zx0, zx1)
        ins.set_ylim(zy0, zy1)
        lines(ins, zx0, zx1, labels=False)
        _account_marks(ins, cells, spread=True)
        ins.set_xticks([0, 5, 10])
        ins.set_yticks([0, 5])
        ins.tick_params(labelsize=8, length=2, pad=1.5)
        ins.grid(axis="x", visible=True)
        for side in ("top", "right"):
            ins.spines[side].set_visible(True)
        for spine in ins.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color(fs.GREY)
        ins.set_facecolor("white")
        ax.add_patch(Rectangle((zx0, zy0), zx1 - zx0, zy1 - zy0, fill=False, ec=fs.GREY,
                               lw=0.5, zorder=5))  # fmt: skip
    path = out / "fig_account.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path

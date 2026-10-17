"""Paper numbers, tables and the figure of the second-round exploratory analyses (prereg/
decisions.md 2026-10-01; answers to a devil's-advocate re-review). Every macro name starts with `Ex`
(exploratory) and every source is a staged analysis named in results/sources.json:

  ex_account   analysis/ex_account.json (`analysis.ex_account`): the account across writers, classes
               and branches; the test across the cells
  ex_more      analysis/ex_e1_more.json (`analysis.ex_e1_more`): out-of-sample baselines, the
               twelve-to-three-month transfer, the accuracy of the items a key missed, the zero
               system after the shift
  ex_misses    analysis/ex_misses.json (`analysis.ex_misses`): why the cited-cluster key misses
  e1, rep_e1   analysis/e1_metrics.json (the post-shift JSON arm, `explore_json_arm`)

A stage that is not staged renders as the MISSING marker; a quantity that is undefined by
construction (an accuracy over no item) renders as a dash.

tab_writers      the Eq. (1) terms r0, r1 and B x D by writer, item class and branch, and the test
                 of the account across the cells (leave-one-writer-out against the class label)
tab_baselines    out-of-sample errors of the four predictors, development to main study, to R, main
                 study to R, and the twelve-month terms against the three-month worlds
tab_misses       the classes of the changes the cited-cluster key missed (a, b, c) and what binding
                 does to the accuracy of the items a key did not invalidate
tab_zero_after   Update and Stability of the zero system against HoldMem(1,1) and HoldMem(0,0)
tab_json_arm     the post-shift JSON arm against the s_k arm
fig_account      B x D against the repair gap r1 - r0 of the same cell, one marker per writer, class
                 and branch, with the fitted line and its bootstrap band
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.ex_e1_paper import UNDEFINED, _cell, _mae, _Out, fmt, point
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.analysis.paper_tables import _num, _row, interval

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
    """Every `Ex` macro of the second-round exploratory analyses, from `load`."""
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
def _branch_head() -> str:
    return (
        r"\multicolumn{3}{c}{$\Ssrc$} & \multicolumn{3}{c}{$\Sdef$} & \multicolumn{3}{c}{$\Sdel$}"
    )


def _r(e: dict[str, Any] | None) -> str:
    """An accuracy in percent, no decimals; a dash where no item was re-derived."""
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=0)


def _mae_cell(e: dict[str, Any] | None, scale: float = 1.0) -> str:
    """A mean absolute error in pp with its interval (`paper_tables.interval`)."""
    if not e:
        return MISSING
    if e["est"] != e["est"]:
        return UNDEFINED
    return interval(*(f"{scale * e[k]:.1f}" for k in ("est", "lo", "hi")))


def _snum_cell(e: dict[str, Any] | None) -> str:
    """A plain statistic (rank correlation, slope) with its interval, typeset minus."""
    if not e:
        return MISSING
    if e.get("est") is None or e["est"] != e["est"]:
        return UNDEFINED
    return interval(*(snum(e.get(k)) for k in ("est", "lo", "hi")))


def tab_writers(inp: Mapping[str, Any]) -> str:
    """Appendix: r0, r1 and B x D (source-epoch binding) by writer, item class and branch, and the
    test of the account across the cells of the four fitted writers."""
    acct = inp.get("acct")
    cells = _get(acct, "explore_account_cells") or {}
    fit = _get(acct, "explore_account_fit", "two_class") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{The account by writer and item class (exploratory, source-epoch key). "
        r"$r_D$: accuracy (\%) of the re-derived items at $D$. $B{\times}D$: the interaction on "
        r"Update (pp, world-bootstrap 95\% interval), which with a key that misses nothing is "
        r"$(r_1{-}r_0)-(a_1{-}a_0)$, where $a_D$ is the Update of the $B{=}0$ cell. "
        r"Field: single-field items. Other: single-field and free-text items. "
        r"$\rho$: Spearman rank correlation. Writers: "
        r"main study and replication (Qwen3.5-9B, 20 worlds each), and ten worlds each for "
        r"gemma-4-12B, Llama-3.1-8B and gpt-oss-20b. Below: the test across the "
        r"(writer, class, branch) cells of the four fitted writers, with the replication held out. "
        r"It gives the leave-one-writer-out mean absolute error of the predicted $B{\times}D$ in "
        r"pp. The prediction comes from the repair gap $r_1{-}r_0$ of the same cell (least-squares "
        r"line) or from the class label alone (definition against other).}",
        r"\label{tab:writers}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}ll rrr rrr rrr@{}}",
        r"\toprule",
        rf"& & {_branch_head()} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
        r"writer & class & $r_0$ & $r_1$ & $B{\times}D$ & $r_0$ & $r_1$ & $B{\times}D$ "
        r"& $r_0$ & $r_1$ & $B{\times}D$ \\",
    ]
    for w, _, label in WRITER_TAGS:
        lines.append(r"\midrule")
        for k, (cls, _, cname) in enumerate(ACCOUNT_CLASSES):
            vals: list[str] = []
            for b, _ in BRANCH_TAGS:
                n = _get(cells, w, b, cls)
                vals += [_r(_get(n, "r0")), _r(_get(n, "r1")), _cell(_get(n, "BxD"))]
            lines.append(_row(label if k == 0 else "", cname, *vals))
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\medskip", r"\setlength{\tabcolsep}{3pt}"]
    lines += [
        r"\begin{tabular}{@{}l rrr rr@{}}",
        r"\toprule",
        r"across the cells & $\rho$, all & $\rho$, definition & $\rho$, other "
        r"& slope & intercept (pp) \\",
        r"\midrule",
        _row(
            r"$B{\times}D$ on $r_1{-}r_0$",
            _snum_cell(fit.get("spearman")),
            _snum_cell(_get(fit, "spearman_by_class", "definition")),
            _snum_cell(_get(fit, "spearman_by_class", "other")),
            _snum_cell(fit.get("slope")),
            _cell(fit.get("intercept")),
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{@{}l rrr r@{}}",
        r"\toprule",
        r"leave-one-writer-out error & all cells & definition & other & repl. cells \\",
        r"\midrule",
    ]
    for pred, label in (
        ("gap_linear", r"repair gap $r_1{-}r_0$"),
        ("class_mean", "class label"),
        ("class_branch_mean", "class and branch"),
        ("grand_mean", "no predictor (mean)"),
    ):
        lowo = fit.get("lowo") or {}
        by = lowo.get("by_class") or {}
        held = fit.get("heldout_all_fitted_mae") or {}
        lines.append(
            _row(
                label,
                _mae_cell(_get(lowo, "mae", pred)),
                _mae_cell(_get(by, "definition", pred)),
                _mae_cell(_get(by, "other", pred)),
                _mae_cell(held.get(pred)),
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_baselines(inp: Mapping[str, Any]) -> str:
    """Appendix: mean absolute error in pp (and cells within 5 pp) of each predictor, pooled over
    the three branches' cells: development to main study, to R, main study to R, and the
    twelve-month terms against the three-month worlds."""
    more = inp.get("more")
    base = _get(more, "explore_oos_baselines") or {}
    hz = _get(more, "explore_horizon_transfer", "main_to_short") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Out-of-sample error of each predictor of a cell's Update (exploratory): mean "
        r"absolute error in pp over the cells of the three branches, and the number of cells "
        r"within 5\,pp. Cell terms: each cell's Update per item class, weighted by the "
        r"class mix of the target. Structural: Eq.~(\ref{eq:decomp}) with $r$ by $D$, "
        r"$m$ by the key and $a$ from the write path. One stale accuracy: "
        r"Eq.~(\ref{eq:decomp}) with a single stale-value accuracy. Pooled mean: the fit "
        r"worlds' Update of the cell over all "
        r"items, with no class terms and no reweighting (the same number for every class). The "
        r"last rows fit the twelve-month "
        r"main study and predict the three-month worlds (four source-epoch cells per branch).}",
        r"\label{tab:baselines}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll rrrr@{}}",
        r"\toprule",
        r"fit $\to$ target & items & cell terms & structural & one stale accuracy & pooled mean \\",
        r"\midrule",
    ]
    pairs = (
        ("dev_to_main", r"development $\to$ main"),
        ("dev_to_rep", r"development $\to$ repl."),
        ("main_to_rep", r"main $\to$ repl."),
    )
    for pair, label in pairs:
        for k, (cls, cname) in enumerate(
            (("all", "all"), ("definition", "definition"), ("other", "other"))
        ):
            nodes = [_get(base, pair, "pooled_over_branches", cls, p) for p, _ in BASE_PREDICTORS]
            lines.append(_row(label if k == 0 else "", cname, *(_mae(n) for n in nodes)))
        lines.append(r"\addlinespace[2pt]")
    for k, (cls, cname) in enumerate(
        (("all", "all"), ("definition", "definition"), ("other", "other"))
    ):
        nodes = [
            _get(hz, "pooled_over_branches", cls, p) if p in ("cell", "pooled_cell_mean") else None
            for p, _ in BASE_PREDICTORS
        ]
        cells = [_mae(n) if n else UNDEFINED for n in nodes]
        if not hz:
            cells = [MISSING] * 4
        lines.append(_row(r"12 $\to$ 3 months" if k == 0 else "", cname, *cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _share(e: dict[str, Any] | None) -> str:
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=0)


def tab_misses(inp: Mapping[str, Any]) -> str:
    """Appendix: (a) the classes of the changed item-times the cited-cluster key missed, per branch
    and class: main study, and R in grey; (b) the accuracy of the items a key did not invalidate,
    B=1 cited against B=0, at each D."""
    miss = _get(inp.get("miss"), "explore_miss_classes") or {}
    more = _get(inp.get("more"), "explore_missed_items", "main") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{The changed item-times the cited-cluster key missed (exploratory). "
        r"(a) Why, main study and, in grey, the replication. Missed: the number of "
        r"missed item-times. The other columns are shares of them in percent. $a$: a record "
        r"of the item's gold "
        r"dependency set at derivation changed and the writer had not cited it. $b$: "
        r"otherwise a record entered the dependency set (new: created after the "
        r"derivation, old: existed and entered the scope). $c$ left: otherwise no "
        r"record changed and records left a time window or a narrowed scope. $a{\wedge}b$: "
        r"item-times of class $a$ into whose dependency set a record also entered. The classes "
        r"$a$, $b$, $c$ are exclusive. A change of the definition version always moves the key, "
        r"so none of the missed item-times has one. A record created after the derivation, or "
        r"leaving a scope by the clock, cannot be cited at derivation. (b) What binding does to "
        r"the accuracy (percent) of the changed items a key did not invalidate, main study. "
        r"Missed: the number of missed item-times at $D{=}1$. The $B{=}1$ cited cell on "
        r"the missed items against the $B{=}0$ cell on the same items (same) and on all changed "
        r"items (all).}",
        r"\label{tab:misses}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.4pt}",
        r"\begin{tabular}{@{}ll r rrr rrr@{}}",
        r"\toprule",
        r"(a) & & missed & $a$ & $b$ new & $b$ old & $c$ left & $a{\wedge}b$ & \\",
        r"\midrule",
    ]
    for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
        for k, (cls, _, cname) in enumerate(MISS_CLASSES[:2] + MISS_CLASSES[4:]):
            for stage, _ in STAGES:
                n = _get(miss, stage, b, cls)
                cells = [
                    count(_get(n, "missed_item_times")) if n else MISSING,
                    *(
                        _share(_get(n, key))
                        for key in (
                            "a",
                            "b_new",
                            "b_existing",
                            "c_left_scope",
                            "a_and_b",
                        )
                    ),
                ]
                if stage == "rep":
                    cells = [rf"\textcolor{{gray}}{{{c}}}" for c in cells]
                    label, cname_ = "", r"\quad\textcolor{gray}{repl.}"
                else:
                    label, cname_ = (bt if k == 0 else ""), cname
                lines.append(_row(label, cname_, *cells, ""))  # (a) leaves (b)'s last column empty
        lines.append(r"\midrule")
    lines[-1] = r"\midrule[\heavyrulewidth]"  # (b) continues the same tabular: one column grid
    lines += [
        r"(b) & & missed & \multicolumn{3}{c}{$D{=}0$} & \multicolumn{3}{c}{$D{=}1$} \\",
        r"\cmidrule(lr){4-6}\cmidrule(l){7-9}",
        r"& & & $B{=}1$ cited & $B{=}0$ same & $B{=}0$ all & $B{=}1$ cited & $B{=}0$ same "
        r"& $B{=}0$ all \\",
        r"\midrule",
    ]
    for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
        for k, (cls, cname) in enumerate(
            (("definition", "definition"), ("other", "other"), ("all", "all"))
        ):
            d0, d1 = _get(more, b, "D0", cls), _get(more, b, "D1", cls)
            cells = [count(_get(d1, "missed_item_times")) if d1 else MISSING]
            for d in (d0, d1):
                cells += [
                    _share(_get(d, key))
                    for key in ("a_missed_B1_cited", "a_missed_B0", "update_B0")
                ]
            lines.append(_row(bt if k == 0 else "", cname, *cells))
        lines.append(r"\midrule" if b != "orphan" else r"\bottomrule")
    lines += [r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_zero_after(inp: Mapping[str, Any]) -> str:
    """Appendix: the zero system against HoldMem(1,1) and HoldMem(0,0) on Update and Stability after
    the shift, per branch and class: main study and R."""
    res = _get(inp.get("more"), "explore_zero_system") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{The zero system after the shift (exploratory). Update (changed items) and "
        r"Stability (unchanged items), in percent, of the zero system (zero), "
        r"\textsc{HoldMem}(1,1) and \textsc{HoldMem}(0,0) (columns (1,1) and (0,0)). "
        r"Zero $-$ (1,1): the paired difference, zero system minus \textsc{HoldMem}(1,1) (pp, "
        r"95\% interval). Main study, then the replication.}",
        r"\label{tab:zero-after}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        # one tabular for both stages: one column grid
        r"\begin{tabular}{@{}ll rrr@{\quad}r rrr@{}}",
        r"\toprule",
        r"& & \multicolumn{4}{c}{Update} & \multicolumn{3}{c}{Stability} \\",
        r"\cmidrule(lr){3-6}\cmidrule(l){7-9}",
        r"& & zero & (1,1) & (0,0) & zero $-$ (1,1) & zero & (1,1) & (0,0) \\",
    ]
    for stage, title in (("main", "main study"), ("rep", "replication")):
        lines += [r"\midrule", rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
            for k, (cls, cname) in enumerate(
                (("definition", "definition"), ("other", "other"), ("all", "all"))
            ):
                n = _get(res, stage, b, cls)
                up, st = _get(n, "update") or {}, _get(n, "stability") or {}
                cells = [
                    _share(up.get("zero")),
                    _share(up.get("holdmem_D1B1_source")),
                    _share(up.get("holdmem_D0B0")),
                    _cell(_get(n, "zero_minus", "holdmem_D1B1", "update")),
                    _share(st.get("zero")),
                    _share(st.get("holdmem_D1B1_source")),
                    _share(st.get("holdmem_D0B0")),
                ]
                if not n:
                    cells = [MISSING] * 7
                lines.append(_row(bt if k == 0 else "", cname, *cells))
            if b != "orphan":
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_json_arm(inp: Mapping[str, Any]) -> str:
    """Appendix: the D=1 store served as JSON against the same store served as s_k statements, on
    the post-shift probes: main study and R, per branch and class."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{The post-shift JSON arm against the $s_k$ arm (exploratory). The same "
        r"$D{=}1$ store with the source-epoch key is served as JSON records or as statements "
        r"($s_k$). Update on changed items and Stability on unchanged ones, in percent, and the "
        r"paired difference JSON minus $s_k$ (pp, 95\% interval). Main study, then the "
        r"replication.}",
        r"\label{tab:json-arm}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll rr@{\quad}r r@{}}",  # one tabular for both stages: one column grid
        r"\toprule",
        r"& & $s_k$ Update & JSON Update & JSON $-$ $s_k$ & Stability, JSON $-$ $s_k$ \\",
    ]
    for stage, title in (("main", "main study"), ("rep", "replication")):
        m = inp.get(stage)
        lines += [r"\midrule", rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for b, bt in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
            for k, (cls, cname) in enumerate(JSON_CLASSES):
                j = _get(m, "explore_json_arm", b, cls)
                cells = [
                    _share(_get(j, "update_sk")),
                    _share(_get(j, "update_json")),
                    _cell(_get(j, "update_json_minus_sk")),
                    _cell(_get(j, "stability_json_minus_sk")),
                ]
                if not j:
                    cells = [MISSING] * 4
                lines.append(_row(bt if k == 0 else "", cname, *cells))
            if b != "orphan":
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "tab_writers.tex": tab_writers(inp),
        "tab_baselines.tex": tab_baselines(inp),
        "tab_misses.tex": tab_misses(inp),
        "tab_zero_after.tex": tab_zero_after(inp),
        "tab_json_arm.tex": tab_json_arm(inp),
    }


# ---------------------------------------------------------------------------------- figure
WRITER_MARKERS = {"main": "o", "rep": "o", "gemma": "s", "llama": "^", "gptoss": "D"}


def fig_account(inp: Mapping[str, Any], out: Path) -> Path:
    """B x D against the repair gap r1 - r0 of the same cell: one marker per writer, item class
    (colour) and branch; the fitted line (four fitted writers, hollow R markers are held out) with
    its 95% bootstrap band, and the diagonal B x D = r1 - r0."""
    from memshift.analysis import figstyle as fs

    plt = fs.plt()
    fit = _get(inp.get("acct"), "explore_account_fit", "two_class") or {}
    colours = {"definition": fs.VERMILION, "other": fs.BLUE}
    fig, ax = plt.subplots(figsize=(fs.TEXT * 0.72, 3.25))
    ax.plot([-5, 100], [-5, 100], ls=":", color=fs.GREY, lw=0.9, zorder=1)
    ax.axhline(0, color=fs.GREY, lw=0.6, zorder=1)
    line = fit.get("fit_line")
    slope, intercept = _get(fit, "slope", "est"), _get(fit, "intercept", "est")
    if line and slope is not None and slope == slope:
        grid = [100 * g for g in line["grid"]]
        ax.fill_between(grid, [100 * v for v in line["lo"]], [100 * v for v in line["hi"]],
                        color=fs.LIGHT, lw=0, zorder=1.5)  # fmt: skip
        ax.plot(grid, [100 * intercept + slope * g for g in grid], color=fs.BLACK, lw=1.1, zorder=2)
    for key, held_out in (("cells", False), ("heldout_cells", True)):
        for c in fit.get(key) or []:
            if c["gap"] != c["gap"] or c["BxD"] != c["BxD"]:
                continue
            colour = colours[c["class"]]
            ax.plot([100 * c["gap"]], [100 * c["BxD"]], marker=WRITER_MARKERS[c["writer"]],
                    ms=5.2, mec=colour, mew=1.2, mfc="white" if held_out else colour, ls="none",
                    zorder=4)  # fmt: skip
    rho, lowo = _get(fit, "spearman", "est"), _get(fit, "lowo", "mae")
    if rho is not None and slope is not None and lowo:
        gap, cls = _get(lowo, "gap_linear", "est"), _get(lowo, "class_mean", "est")
        # on a white box, so that no gridline runs through the note
        ax.text(0.97, 0.015,
                rf"$\rho={rho:.2f}$; slope ${slope:.2f}$" "\n"
                rf"leave-one-writer-out error: gap {gap:.1f}, class {cls:.1f}\,pp",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, color=fs.BLACK,
                zorder=5, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})  # fmt: skip
    ax.set_xlabel(r"repair gap $r_1-r_0$ of the cell (pp)")
    ax.set_ylabel(r"$B{\times}D$ on Update (pp)")
    ax.set_xlim(-5, 100)
    ax.set_ylim(-15, 62)
    ax.grid(axis="x", visible=True)
    handles = [
        plt.Line2D([], [], marker=m, ls="none", mfc="0.35", mec="0.35", ms=4.8, label=lab)
        for w, m, lab in (
            ("main", "o", "Qwen3.5-9B"), ("gemma", "s", "gemma-4-12B"),
            ("llama", "^", "Llama-3.1-8B"), ("gptoss", "D", "gpt-oss-20b"),
        )
    ]  # fmt: skip
    handles.append(plt.Line2D([], [], marker="o", ls="none", mfc="white", mec="0.35", mew=1.2,
                              ms=4.8, label="replication (not fitted)"))  # fmt: skip
    handles += [
        plt.Line2D([], [], marker="o", ls="none", mfc=colours[c], mec=colours[c],
                   ms=4.8, label=f"{c} items")
        for c in ("definition", "other")
    ]  # fmt: skip
    handles += [
        plt.Line2D([], [], color=fs.BLACK, lw=1.1, label="fit, four writers"),
        plt.Line2D([], [], color=fs.LIGHT, lw=5, label="95\\% band"),
        plt.Line2D([], [], color=fs.GREY, lw=0.9, ls=":", label=r"$B{\times}D=r_1-r_0$"),
    ]
    fig.legend(handles=handles, loc="outside right upper", ncol=1, fontsize=7.5)
    path = out / "fig_account.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path

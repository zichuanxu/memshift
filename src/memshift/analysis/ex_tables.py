"""Appendix tables for the further analyses of E5, regenerated from the committed run analyses
like the other generated tables:

tab_native_null         native systems against label-free baselines: counts by circularity, chance
                        coverage, B identification (tab:native-null); per comparison the
                        containing references and the paired differences (tab:native-diff) (E5)
A stage that has not been staged renders as MISSING in every cell that needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memshift.analysis.ex_numbers import REFERENCE_LABELS
from memshift.analysis.paper_data import MISSING, NATIVES
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_tables import PAIR, _num, _row, mt_ci_cells, mtint, notes


# --------------------------------------------------------------------------------- native null
def _pair(counts: dict[str, Any] | None, j: bool = False, muted: bool = False) -> str:
    """inside / comparisons of one set of comparisons as the two cells `k` and `/n` of a `COUNT`
    column pair (they meet at the slash), with `, j` (the discriminating ones) after `k` where `j`;
    "--" or MISSING across both. `muted`: in the grey of the baselines' reference rows."""

    def ink(text: str) -> str:
        return rf"\mtmuted{{{text}}}" if muted else text

    if not counts or not counts.get("comparisons"):
        return _span(ink("--" if counts is not None else MISSING))
    k = str(counts["inside_predicted"]) + (f", {counts['discriminating']}" if j else "")
    return f"{ink(k)} & {ink('/' + str(counts['comparisons']))}"


def _lab(system: str, res: dict[str, Any]) -> str:
    lab = _get(res, "labels", system) or {}
    if not lab:
        return MISSING
    d = lab.get("D", {}).get("label")
    b = lab.get("B", {}).get("label")
    return f"({0 if d is None else d},{'--' if b is None else b})"


# the comparisons in column order: after S_src and S_def (four), after S_del (two, from which B is
# inferred), all six
PARTS = ("out_of_sample", "label_circular", "all")
COUNT = r"r@{}l"  # a count `k/n` in two cells that meet at the slash
BASELINES = (("zero", "no-store BM25"), ("rag", "plain RAG"))  # the muted reference rows
# the reference labels of `ex_numbers` with the no-store BM25 baseline named as in the paper
TABLE_REFERENCES = {**REFERENCE_LABELS, "zero": "no-store"}
BRANCH_NAMES = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}


def _coverage(x: float | None) -> str:
    return MISSING if x is None else _num(x, 100, 0)


def _yes_no(flag: bool | None) -> str:
    return MISSING if flag is None else ("yes" if flag else "no")


def _all_discriminating(ex: dict[str, Any]) -> bool:
    """Whether every comparison inside the interval of the label cell is also outside that of the
    cell with the other B, for every system, baseline and the sums (then the table prints `k/n`
    and says so once)."""
    nodes: list[Any] = []
    for e in [*(ex.get("systems") or {}).values(), ex.get("total") or {}]:
        nodes += [_get(e, "counts", p) for p in PARTS]
        nodes += [_get(e, "baseline_counts", b, "baseline", p) for b, _ in BASELINES for p in PARTS]
    return all(n["inside_predicted"] == n["discriminating"] for n in nodes if n)


def _reference_rows(owner: dict[str, Any] | None, j: bool) -> list[str]:
    """The muted rows of no-store BM25 and plain RAG against the intervals of `owner`'s label cell
    (a system of the group, or the sums): their counts only."""
    rows = []
    for name, label in BASELINES:
        counts = [
            _pair(_get(owner, "baseline_counts", name, "baseline", p), j, muted=True)
            if owner
            else _span(MISSING)
            for p in PARTS
        ]
        rows.append(_row(rf"\mtin\mtmuted{{{label}}}", "", "&", "", *counts, ""))
    return rows


def tab_native_null(res: dict[str, Any] | None) -> str:
    """Native systems against the null baselines, from `explore_native_null` (Mem0 from stage 3,
    all six systems once block N is staged), as two tables: the counts per system with their sums
    (`tab:native-null`, the systems of one inferred label together, with no-store BM25 and plain
    RAG on the label cell's intervals as muted rows under the group, which share them), and per
    system and comparison the references that contain the native value and the paired differences
    (`tab:native-diff`, the shift printed once above its two measures; the native values are in
    the main text's overview table)."""
    res = res or {}
    ex = res.get("explore_native_null") or {}
    systems = ex.get("systems") or {}
    j = not _all_discriminating(ex)  # print the discriminating counts only where they differ
    lines = [
        r"\begin{table}[t]",
        r"\caption{Native systems against the \holdmem{} cell with their inferred label "
        r"(three-month worlds), with no-store BM25 and plain RAG on the same intervals.}",
        r"\label{tab:native-null}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}l c {PAIR} c {' '.join([COUNT] * len(PARTS))} N{{00}}@{{}}}}",
        r"\toprule",
        r"& & & & & \multicolumn{6}{c}{Inside the label cell's interval} & \\",
        r"\cmidrule(lr){6-11}",
        r"System & Label & \multicolumn{2}{c}{$P$ (\%)} & $B$ id. "
        r"& \multicolumn{2}{c}{$\Ssrc$, $\Sdef$} & \multicolumn{2}{c}{$\Sdel$} "
        r"& \multicolumn{2}{c}{All six} & \multicolumn{1}{c@{}}{Chance (\%)} \\",
        r"\midrule",
    ]
    groups: list[tuple[str, list[str], list[dict[str, Any]]]] = []  # label, rows, staged systems
    for nat in NATIVES:
        e = systems.get(nat.key) or {}
        b = e.get("b_identified") or {}
        counts = e.get("counts") or {}
        label = _lab(nat.key, res)
        if not groups or groups[-1][0] != label:
            groups.append((label, [], []))
        if e:
            groups[-1][2].append(e)
        groups[-1][1].append(
            _row(
                nat.label,
                label,
                mt_ci_cells({"est": b["rate"], "lo": b["lo"], "hi": b["hi"]})
                if b
                else _span(MISSING),
                _yes_no(b.get("identified")) if b else MISSING,
                *(_pair(counts.get(part), j) if counts else _span(MISSING) for part in PARTS),
                _coverage((e.get("chance_coverage") or {}).get("labels")),
            )
        )
    for i, (_label, rows, staged) in enumerate(groups):
        if i:
            lines.append(r"\mtgap")  # the next inferred label
        refs = {tuple(_reference_rows(e, j)) for e in staged}
        if len(refs) > 1:  # the baselines are compared with the one label cell of the group
            raise ValueError(f"the baselines differ within the label group {_label}")
        lines += rows + _reference_rows(staged[0] if staged else None, j)
    total = ex.get("total") or {}
    complete = all(n.key in systems for n in NATIVES) and bool(total)
    lines += [
        r"\midrule",
        _row(
            "total",
            "",
            "&",  # P: the two cells of its estimate and interval
            f"{total['b_identified']}/{total['systems']}" if complete else MISSING,
            *(
                _pair(_get(total, "counts", part), j) if complete else _span(MISSING)
                for part in PARTS
            ),
            _coverage(total.get("mean_labels_coverage")) if complete else MISSING,
        ),
        *_reference_rows(total if complete else None, j),
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            r"$P$: deletion persistence, the share of deleted items still answered. $B$ id.: its "
            r"interval excludes 0.5. "
            + (
                r"$k$, $j$/$n$: $k$ comparisons inside the 95\% interval of the \holdmem{} cell "
                r"with the inferred label, $j$ of them also outside that of the cell with the "
                r"other $B$. "
                if j
                else r"$k$/$n$: $k$ comparisons inside the 95\% interval of the \holdmem{} cell "
                r"with the inferred label, each also outside that of the cell with the other $B$. "
            )
            + r"$\Sdel$: the comparisons from which $B$ is inferred. Chance: share of the label "
            r"cells containing the native estimate (label permuted). Total: sums over the systems, "
            r"each baseline counted once per system it is compared with (chance: mean)."
        ),
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\caption{Native values against their references per system and comparison "
        r"(three-month worlds).}",
        r"\label{tab:native-diff}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}l l l l {PAIR} l @{{\hspace{{12pt}}}} {PAIR}@{{}}}}",
        r"\toprule",
        r"& & & & \multicolumn{3}{c}{Native $-$ cell (pp)} & \multicolumn{2}{c@{}}{} \\",
        r"\cmidrule(lr){5-7}",
        r"System, cell & \multicolumn{2}{l}{Comparison} & Contained by "
        r"& \multicolumn{2}{c}{Difference} & \multicolumn{1}{c}{Widened} "
        r"& \multicolumn{2}{c@{}}{Native $-$ no-store (pp)} \\",
        r"\midrule",
    ]
    for i, nat in enumerate(NATIVES):
        e = systems.get(nat.key)
        if i:
            lines.append(r"\mtgap")
        if not e:
            comparison = rf"\multicolumn{{2}}{{l}}{{{MISSING}}}"
            lines.append(
                _row(nat.label, comparison, MISSING, _span(MISSING), MISSING, _span(MISSING))
            )
            continue
        cell = e.get("predicted_by")
        stub = [nat.label, f"cell {TABLE_REFERENCES.get(cell or '', MISSING)}"]
        shown = None  # the shift of the comparison above
        for k, (key, c) in enumerate((e.get("comparisons") or {}).items()):
            paired = (e.get("paired_difference") or {}).get(key) or {}
            p = paired.get(cell) if cell else None
            wide = (
                mtint(_num(p["lo_inflated"], sign=True), _num(p["hi_inflated"], sign=True))
                if p and "lo_inflated" in p
                else MISSING
            )
            held = ", ".join(TABLE_REFERENCES.get(n, n) for n in c["contained_by"]) or "none"
            circ = r"$^\circ$" if c["label_circular"] else ""
            zero = paired.get("zero")
            branch, metric = key.split(":")
            lines.append(
                _row(
                    stub[k] if k < len(stub) else "",
                    BRANCH_NAMES[branch] if branch != shown else "",
                    f"{metric}{circ}",
                    held,
                    mt_ci_cells(p, sign=True) if p else _span(MISSING),
                    wide,
                    mt_ci_cells(zero, sign=True) if zero else _span("--"),
                )
            )
            shown = branch
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            r"Contained by: the references whose 95\% interval contains the native value "
            r"($(D,B)$: \holdmem{} cell, c: citation key, no-store: no-store BM25, RAG: plain "
            r"RAG, oracle: oracle context). Difference: native minus the \holdmem{} cell with the "
            r"inferred label, paired by item, with its 95\% interval. Widened: that interval "
            r"widened by the variance between repeated ingestions. Native $-$ no-store: paired by "
            r"item, with its 95\% interval. $^\circ$: after $\Sdel$, from which $B$ is inferred."
        ),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _span(text: str) -> str:
    """One value across both cells of an estimate-and-interval column pair."""
    return rf"\multicolumn{{2}}{{c}}{{{text}}}"


def build(root: Path, sources: dict[str, str], inp: dict[str, Any]) -> dict[str, str]:
    """File name -> table text for the appendix table."""
    return {
        "tab_native_null.tex": tab_native_null(inp["n"] or inp["n3"]),
    }

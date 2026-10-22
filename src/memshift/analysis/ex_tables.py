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
def _frac(counts: dict[str, Any] | None, key: str = "inside_predicted") -> str:
    if not counts or not counts.get("comparisons"):
        return "--" if counts else MISSING
    return f"{counts[key]}/{counts['comparisons']}"


def _pair(counts: dict[str, Any] | None) -> str:
    """inside, discriminating / comparisons of one set of comparisons as the two cells `i, d` and
    `n` of a `COUNT` column pair (they meet at the slash); "--" or MISSING across both."""
    if not counts or not counts.get("comparisons"):
        return _span("--" if counts is not None else MISSING)
    n = counts["comparisons"]
    return f"{counts['inside_predicted']}, {counts['discriminating']} & {n}"


def _lab(system: str, res: dict[str, Any]) -> str:
    lab = _get(res, "labels", system) or {}
    if not lab:
        return MISSING
    d = lab.get("D", {}).get("label")
    b = lab.get("B", {}).get("label")
    return f"({0 if d is None else d},{'--' if b is None else b})"


PARTS = ("all", "label_circular", "out_of_sample")
COUNT = r"r@{\,/\,}l"  # a count `k, j / n` in two cells that meet at the slash
# the header of the four comparisons after S_src and S_def on two lines (`\twoline`, defined in
# the table), so the column is as wide as its counts
OOS = r"\twoline{$\Ssrc$,}{$\Sdef$}"
# the reference labels of `ex_numbers` with the no-store BM25 baseline named as in the paper
TABLE_REFERENCES = {**REFERENCE_LABELS, "zero": "no-store"}
BRANCH_NAMES = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}


def _coverage(x: float | None) -> str:
    return MISSING if x is None else _num(x, 100, 0)


def _yes_no(flag: bool | None) -> str:
    return MISSING if flag is None else ("yes" if flag else "no")


def tab_native_null(res: dict[str, Any] | None) -> str:
    """Native systems against the null baselines, from `explore_native_null` (Mem0 from stage 3,
    all six systems once block N is staged), as two tables: the counts per system with their sums
    (`tab:native-null`, the systems of one inferred label together), and per system and comparison
    the references that contain the native value and the paired differences (`tab:native-diff`,
    the shift printed once above its two measures; the native values are in the main text's
    overview table)."""
    res = res or {}
    ex = res.get("explore_native_null") or {}
    systems = ex.get("systems") or {}
    counts_head = ("All six", r"$\Sdel$", OOS, "All six", OOS, "All six", OOS)
    lines = [
        r"\begin{table}[t]",
        r"\caption{Native systems against the \holdmem{} cell with their inferred label "
        r"(three-month worlds), with no-store BM25 and plain RAG on the same intervals.}",
        r"\label{tab:native-null}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{2.85pt}",
        r"\newcommand{\twoline}[2]{\begin{tabular}[b]{@{}c@{}}#1\\#2\end{tabular}}",
        rf"\begin{{mttabular}}{{@{{}}l c {PAIR} c {' '.join([COUNT] * len(counts_head))} "
        r"N{00} N{00}@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{$P$ (\%)} & & \multicolumn{6}{c}{Native} "
        r"& \multicolumn{4}{c}{No-store BM25} & \multicolumn{4}{c}{Plain RAG} "
        r"& \multicolumn{2}{c@{}}{Chance (\%)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){6-11}\cmidrule(lr){12-15}\cmidrule(lr){16-19}"
        r"\cmidrule(l){20-21}",
        r"System & Label & \multicolumn{2}{c}{est.} & $B$ id. & "
        + " & ".join(_span(h) for h in counts_head)
        + r" & \multicolumn{1}{c}{Labels} & \multicolumn{1}{c@{}}{Observed} \\",
        r"\midrule",
    ]
    previous = None
    for nat in NATIVES:
        e = systems.get(nat.key) or {}
        b = e.get("b_identified") or {}
        counts = e.get("counts") or {}
        bc = e.get("baseline_counts") or {}
        cov = e.get("chance_coverage") or {}
        label = _lab(nat.key, res)
        if previous is not None and label != previous:
            lines.append(r"\mtgap")  # the next inferred label
        previous = label

        def base(name: str, part: str, bc: dict[str, Any] = bc) -> str:
            return _pair(_get(bc, name, "baseline", part))

        lines.append(
            _row(
                nat.label,
                label,
                mt_ci_cells({"est": b["rate"], "lo": b["lo"], "hi": b["hi"]})
                if b
                else _span(MISSING),
                _yes_no(b.get("identified")) if b else MISSING,
                *(_pair(counts.get(part)) if counts else _span(MISSING) for part in PARTS),
                *(
                    base(name, part) if e else _span(MISSING)
                    for name in ("zero", "rag")
                    for part in ("all", "out_of_sample")
                ),
                _coverage(cov.get("labels")),
                _coverage(cov.get("observed")),
            )
        )
    total = ex.get("total") or {}
    complete = all(n.key in systems for n in NATIVES) and bool(total)
    tbc = total.get("baseline_counts") or {}
    lines += [
        r"\midrule",
        _row(
            "total",
            "",
            _span(""),  # P: the two cells of its estimate and interval
            f"{total['b_identified']}/{total['systems']}" if complete else MISSING,
            *(_pair(_get(total, "counts", part)) if complete else _span(MISSING) for part in PARTS),
            *(
                _pair(_get(tbc, name, "baseline", part)) if complete else _span(MISSING)
                for name in ("zero", "rag")
                for part in ("all", "out_of_sample")
            ),
            _coverage(total.get("mean_labels_coverage")) if complete else MISSING,
            _coverage(total.get("mean_observed_coverage")) if complete else MISSING,
        ),
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            r"$P$: deletion persistence, the share of deleted items still answered. $B$ id.: its "
            r"interval excludes 0.5. Counts $k$, $j$\,/\,$n$: $k$ comparisons inside the 95\% "
            r"interval of the \holdmem{} cell with the inferred label, $j$ of them also outside "
            r"that of the cell with the other $B$. Chance: share of the label cells containing the "
            r"native estimate (label permuted). Last row: sums (chance: means)."
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
        r"& \multicolumn{2}{c}{} & & \multicolumn{3}{c}{Native $-$ cell (pp)} "
        r"& \multicolumn{2}{c@{}}{Native $-$ no-store (pp)} \\",
        r"\cmidrule(lr){5-7}\cmidrule(l){8-9}",
        r"System, cell & \multicolumn{2}{l}{Comparison} & Contained by "
        r"& \multicolumn{2}{c}{95\% interval} & \multicolumn{1}{c}{Widened} "
        r"& \multicolumn{2}{c@{}}{95\% interval} \\",
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
        for j, (key, c) in enumerate((e.get("comparisons") or {}).items()):
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
                    stub[j] if j < len(stub) else "",
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
            r"($(D,B)$: \holdmem{} cell, c: citation key, no-store: no-store BM25, oracle: oracle "
            r"context). Differences, paired by item: native minus the \holdmem{} cell with the "
            r"inferred label, with its interval and the interval widened by the variance between "
            r"repeated ingestions, and native minus no-store BM25. $^\circ$: after $\Sdel$, from "
            r"which $B$ is inferred."
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

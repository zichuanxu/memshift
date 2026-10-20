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
from memshift.analysis.paper_tables import PAIR, _num, _row, ci_pair_cells


# --------------------------------------------------------------------------------- native null
def _frac(counts: dict[str, Any] | None, key: str = "inside_predicted") -> str:
    if not counts or not counts.get("comparisons"):
        return "--" if counts else MISSING
    return f"{counts[key]}/{counts['comparisons']}"


def _pair(counts: dict[str, Any] | None) -> str:
    """inside / discriminating of one set of comparisons as `i/n, d/n`."""
    if not counts or not counts.get("comparisons"):
        return "--" if counts is not None else MISSING
    n = counts["comparisons"]
    return f"{counts['inside_predicted']}, {counts['discriminating']}\\,/\\,{n}"


def _lab(system: str, res: dict[str, Any]) -> str:
    lab = _get(res, "labels", system) or {}
    if not lab:
        return MISSING
    d = lab.get("D", {}).get("label")
    b = lab.get("B", {}).get("label")
    return f"({0 if d is None else d},{'--' if b is None else b})"


PARTS = ("all", "label_circular", "out_of_sample")
# the header of the four comparisons after S_src and S_def on two lines, so the column is as wide
# as its counts and the table fits a 5.5in text width
OOS = r"\parbox[t]{21pt}{\centering $\Ssrc$, $\Sdef$}"
# the reference labels of `ex_numbers` with the no-store BM25 baseline named as in the paper
TABLE_REFERENCES = {**REFERENCE_LABELS, "zero": "no-store"}


def _coverage(x: float | None) -> str:
    return MISSING if x is None else _num(x, 100, 0)


def _yes_no(flag: bool | None) -> str:
    return MISSING if flag is None else ("yes" if flag else "no")


def tab_native_null(res: dict[str, Any] | None) -> str:
    """Native systems against the null baselines, from `explore_native_null` (Mem0 from stage 3,
    all six systems once block N is staged), as two tables: the counts per system with their sums
    (`tab:native-null`), and per system and comparison the references that contain the native
    value and the paired differences (`tab:native-diff`; the native values are in the main
    text's overview table)."""
    res = res or {}
    ex = res.get("explore_native_null") or {}
    systems = ex.get("systems") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Native systems against the \holdmem{} cell with their inferred label "
        r"(three-month worlds), with no-store BM25 and plain RAG on the same intervals. $P$: "
        r"deletion persistence, the share of deleted items still answered. $B$ id.: its interval "
        r"excludes 0.5. Counts $k$, $j$\,/\,$n$: $k$ comparisons inside the 95\% interval of the "
        r"\holdmem{} cell with the inferred label, $j$ of them also outside that of the cell with "
        r"the other $B$. Chance: share of the label cells containing the native estimate (label "
        r"permuted). Last row: sums (chance: means).}",
        r"\label{tab:native-null}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\begin{{tabular}}{{@{{}}l c {PAIR} c ccc cc cc cc@{{}}}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{$P$ (\%)} & & \multicolumn{3}{c}{native} "
        r"& \multicolumn{2}{c}{no-store BM25} & \multicolumn{2}{c}{plain RAG} "
        r"& \multicolumn{2}{c}{chance (\%)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){6-8}\cmidrule(lr){9-10}\cmidrule(lr){11-12}"
        r"\cmidrule(l){13-14}",
        rf"system & label & \multicolumn{{2}}{{c}}{{est.}} & $B$ id. & all six & $\Sdel$ & {OOS} "
        rf"& all six & {OOS} & all six & {OOS} & labels & observed \\",
        r"\midrule",
    ]
    for nat in NATIVES:
        e = systems.get(nat.key) or {}
        b = e.get("b_identified") or {}
        counts = e.get("counts") or {}
        bc = e.get("baseline_counts") or {}
        cov = e.get("chance_coverage") or {}

        def base(name: str, part: str, bc: dict[str, Any] = bc) -> str:
            return _pair(_get(bc, name, "baseline", part))

        lines.append(
            _row(
                nat.label,
                _lab(nat.key, res),
                ci_pair_cells({"est": b["rate"], "lo": b["lo"], "hi": b["hi"]})
                if b
                else _span(MISSING),
                _yes_no(b.get("identified")) if b else MISSING,
                *(_pair(counts.get(part)) if counts else MISSING for part in PARTS),
                *(
                    base(name, part) if e else MISSING
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
            "",
            "",  # P: the two cells of its estimate and interval
            f"{total['b_identified']}/{total['systems']}" if complete else MISSING,
            *(_pair(_get(total, "counts", part)) if complete else MISSING for part in PARTS),
            *(
                _pair(_get(tbc, name, "baseline", part)) if complete else MISSING
                for name in ("zero", "rag")
                for part in ("all", "out_of_sample")
            ),
            _coverage(total.get("mean_labels_coverage")) if complete else MISSING,
            _coverage(total.get("mean_observed_coverage")) if complete else MISSING,
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\caption{Native values against their references per system and comparison "
        r"(three-month worlds). Contained by: the references whose 95\% interval contains the "
        r"native value ($(D,B)$: \holdmem{} cell, c: citation key, no-store: no-store BM25, "
        r"oracle: oracle context). Differences, paired by item: native minus the \holdmem{} cell "
        r"with the inferred label, with its interval and the interval widened by the variance "
        r"between repeated ingestions, and native minus no-store BM25. $^\circ$: after "
        r"$\Sdel$, from which $B$ is inferred.}",
        r"\label{tab:native-diff}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}ll l {PAIR} l {PAIR}@{{}}}}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{native $-$ cell (pp)} & \multicolumn{2}{c}{native $-$ no-store "
        r"(pp)} \\",
        r"\cmidrule(lr){4-6}\cmidrule(l){7-8}",
        r"system, cell & comparison & contained by & \multicolumn{2}{c}{95\% interval} & widened "
        r"& \multicolumn{2}{c}{95\% interval} \\",
        r"\midrule",
    ]
    for i, nat in enumerate(NATIVES):
        e = systems.get(nat.key)
        if i:
            lines.append(r"\addlinespace[2pt]")
        if not e:
            lines.append(_row(nat.label, MISSING, MISSING, _span(MISSING), MISSING, _span(MISSING)))
            continue
        cell = e.get("predicted_by")
        stub = [nat.label, f"cell {TABLE_REFERENCES.get(cell or '', MISSING)}"]
        for j, (key, c) in enumerate((e.get("comparisons") or {}).items()):
            paired = (e.get("paired_difference") or {}).get(key) or {}
            p = paired.get(cell) if cell else None
            wide = (
                rf"{{\scriptsize [{_num(p['lo_inflated'], sign=True)},\,"
                rf"{_num(p['hi_inflated'], sign=True)}]}}"
                if p and "lo_inflated" in p
                else MISSING
            )
            held = ", ".join(TABLE_REFERENCES.get(n, n) for n in c["contained_by"]) or "none"
            circ = r"$^\circ$" if c["label_circular"] else ""
            zero = paired.get("zero")
            lines.append(
                _row(
                    stub[j] if j < len(stub) else "",
                    f"{_comparison_label(key)}{circ}",
                    held,
                    ci_pair_cells(p, sign=True) if p else _span(MISSING),
                    wide,
                    ci_pair_cells(zero, sign=True) if zero else _span("--"),
                )
            )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _span(text: str) -> str:
    """One value across both cells of an estimate-and-interval column pair."""
    return rf"\multicolumn{{2}}{{c}}{{{text}}}"


def _comparison_label(key: str) -> str:
    branch, metric = key.split(":")
    name = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}[branch]
    return f"{name} {metric}"


def build(root: Path, sources: dict[str, str], inp: dict[str, Any]) -> dict[str, str]:
    """File name -> table text for the appendix table."""
    return {
        "tab_native_null.tex": tab_native_null(inp["n"] or inp["n3"]),
    }

"""Appendix tables for the further analyses of E5, regenerated from the committed run analyses
like the other generated tables:

tab_native_null         native systems against their references: per system and comparison the
                        containing references and the paired differences, with each system's
                        inferred label, deletion persistence and count of comparisons inside the
                        interval of its label cell (tab:native-diff, tab:native-null) (E5)
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
# the reference labels of `ex_numbers` with the no-store BM25 baseline named as in the paper
TABLE_REFERENCES = {**REFERENCE_LABELS, "zero": "no-store"}
BRANCH_NAMES = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}


def _discriminating(systems: dict[str, Any]) -> bool:
    """Whether every comparison inside the interval of the label cell is also outside that of the
    cell with the other B, for every system (then the stub prints `k/n` and the notes say so)."""
    nodes = [_get(e, "counts", "all") for e in systems.values()]
    return all(n["inside_predicted"] == n["discriminating"] for n in nodes if n)


def _stub(nat_label: str, e: dict[str, Any], j: bool) -> list[str]:
    """The lines of a system's first column: its name, the cell with its inferred label, its
    deletion persistence P with the 95% interval (from which B is inferred), and its comparisons
    inside the interval of that cell (`k, j/n` where some are not discriminating)."""
    cell = e.get("predicted_by")
    b = e.get("b_identified") or {}
    counts = _get(e, "counts", "all") or {}
    if b:
        lo, hi = _num(b["lo"]), _num(b["hi"])
        persistence = [rf"$P$ {_num(b['rate'])}\%", rf"\mtint{{[{lo},\,{hi}]}}"]
    else:
        persistence = [MISSING, ""]
    if counts.get("comparisons"):
        k = str(counts["inside_predicted"]) + (f", {counts['discriminating']}" if j else "")
        inside = f"inside {k}/{counts['comparisons']}"
    else:
        inside = MISSING
    return [nat_label, f"cell {TABLE_REFERENCES.get(cell or '', MISSING)}", *persistence, inside]


def tab_native_null(res: dict[str, Any] | None) -> str:
    """Native systems against their references, from `explore_native_null` (Mem0 from stage 3,
    all six systems once block N is staged): per system and comparison the references that
    contain the native value and the paired differences (the shift printed once above its two
    measures; the native values are in the main text's overview table), with the system's
    inferred label, its deletion persistence and its count of comparisons inside the interval of
    the cell with that label in its first column (labels tab:native-diff and tab:native-null)."""
    res = res or {}
    ex = res.get("explore_native_null") or {}
    systems = ex.get("systems") or {}
    j = not _discriminating(systems)  # print the discriminating counts only where they differ
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Native values against their references per system and comparison, with each "
        r"system's inferred setting (three-month worlds).}",
        r"\label{tab:native-diff}\label{tab:native-null}",
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
        stub = _stub(nat.label, e, j)
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
    inside = (
        r"Inside $k$, $j$/$n$: $k$ comparisons inside the 95\% interval of the cell, $j$ of them "
        r"also outside that of the cell with the other $B$. "
        if j
        else r"Inside $k$/$n$: $k$ comparisons inside the 95\% interval of the cell, each also "
        r"outside that of the cell with the other $B$. "
    )
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            r"Cell: the \holdmem{} cell with the setting inferred from behavior. $P$: deletion "
            r"persistence, the share of deleted items still answered, with its 95\% interval. "
            + inside
            + r"Contained by: the references whose 95\% interval contains the native value "
            r"($(D,B)$: \holdmem{} cell, c: citation key, no-store: no-store BM25, RAG: plain "
            r"RAG, oracle: oracle context). Difference: native minus the \holdmem{} cell with the "
            r"inferred setting, paired by item, with its 95\% interval. Widened: that interval "
            r"widened by the variance between repeated ingestions. Native $-$ no-store: paired by "
            r"item, with its 95\% interval. $^\circ$: after $\Sdel$, from which $B$ is inferred."
        ),
        r"\end{table*}",
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

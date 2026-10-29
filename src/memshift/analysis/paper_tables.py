"""Paper tables regenerated from the committed run analyses (tables and figures come from runs/),
with compact number formats. Sources: results/sources.json.

Robustness stages (a stage that has not run renders as MISSING):
tab_replication the main study against the replication worlds, per contrast, with both estimates
                and their 95% intervals (block R)
tab_families    writer switches, reader swaps (M1, M3)
The printed captions and labels name the blocks in words (the paper's names).
Usage: python -m memshift.analysis.paper_tables [--root .] [--out paper/generated]
       [--only tab_replication.tex ...]   (write only the named files; every table is computed)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import (
    FAMILIES,
    MISSING,
    TIERS,
    load_stage4,
    replication_view,
    tier_key,
)
from memshift.analysis.paper_data import get as _get

UNDEFINED = "---"  # a fit that has no value (for example no item failed)


def _num(x: float | None, scale: float = 100.0, digits: int = 1, sign: bool = False) -> str:
    if x is None or math.isnan(x):
        return MISSING
    v = round(scale * x, digits)
    s = f"{abs(v):.{digits}f}"
    if sign and v != 0:
        s = ("$+$" if v > 0 else "$-$") + s
    elif v < 0:
        s = "$-$" + s
    return s


def interval(est: str, lo: str, hi: str) -> str:
    """The one interval format of the generated tables: the estimate, then `[lo, hi]` in
    scriptsize type with a thin space after the comma (the caller formats signs and digits)."""
    return f"{est}\\,{{\\scriptsize [{lo},\\,{hi}]}}"


def _ci(e: dict[str, Any] | None, sign: bool = False, digits: int = 1) -> str:
    if not e:
        return MISSING
    lo, hi = _num(e["lo"], sign=sign, digits=digits), _num(e["hi"], sign=sign, digits=digits)
    return interval(_num(e["est"], sign=sign, digits=digits), lo, hi)


def _minus(x: float, digits: int = 2) -> str:
    """A plain number with a typeset minus (slopes, ratios)."""
    return f"{x:.{digits}f}".replace("-", "$-$")


# ------------------------------------------------------------------------ robustness tables
# A stage that has not been staged renders as MISSING in every cell that needs it; the tables are
# generated from the same views as the stage-4 macros (`analysis.paper_data`).
REP_ROWS = (  # E1 contrast (without the granularity suffix), row label
    ("B_effect_detected_D0", r"$B$ effect, detection ($D{=}0$)"),
    ("B_effect_detected_D1", r"$B$ effect, detection ($D{=}1$)"),
    ("B_effect_update_D1", r"$B$ effect, Update ($D{=}1$)"),
    ("B_effect_update_pooled", r"$B$ effect, Update (pooled $D$)"),
    ("D_effect_missed", r"$D$ effect, missed invalidation"),
    ("D_effect_repair", r"$D$ effect, accuracy on re-derived items"),
    ("D_effect_repair_mech", r"$D$ effect, re-derived value equals gold"),
    ("BxD_interaction_update", r"$B{\times}D$ interaction, Update"),
)


def _est(
    e: dict[str, Any] | None, sign: bool = False, scale: float = 100.0, digits: int = 1
) -> str:
    return _num(e["est"], scale, digits, sign) if e else MISSING


def _row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


PAIR = r"r@{\,}l"  # the column pair of an estimate and its interval (`est_ci_cells`)


def est_ci_cells(
    e: Mapping[str, Any] | None,
    sign: bool = False,
    digits: int = 1,
    style: str = "",
    f: Callable[[float], str] | None = None,
) -> str:
    """An estimate and its 95% interval as the two cells of a column pair `PAIR`, so estimates
    align on the right and intervals (in scriptsize type) on the left, the estimate alone where
    the printed ends of the interval equal the printed estimate (every world gave the same value).
    MISSING spans both cells when the node is not there, UNDEFINED when its estimate is not a
    number. `style` opens each cell (a colour, say); `f` prints each number (default: percent or
    pp with `digits` decimals, signed if `sign`)."""
    if not e:
        return rf"\multicolumn{{2}}{{c}}{{{style}{MISSING}}}"
    if e.get("est") is None or e["est"] != e["est"]:
        return rf"\multicolumn{{2}}{{c}}{{{style}{UNDEFINED}}}"
    num = f or (lambda x: _num(x, digits=digits, sign=sign))
    est, lo, hi = (num(e[k]) for k in ("est", "lo", "hi"))
    if lo == est == hi:
        return f"{style}{est} &"
    return f"{style}{est} & {style}{{\\scriptsize [{lo},\\,{hi}]}}"


def ci_pair_cells(
    e: Mapping[str, Any] | None, sign: bool = False, f: Callable[[float], str] | None = None
) -> str:
    """`est_ci_cells` in the cell text of the robustness and further tables, where an estimate
    alone keeps a space before the next cell."""
    cells = est_ci_cells(e, sign=sign, f=f)
    return f"{cells} " if cells.endswith(" &") else cells


# The table style of the paper (paper/memtab-style.tex): an interval is secondary text (`\mtint`),
# an estimate and its interval share one centred cell (`\mtpair`) where a column pair does not fit,
# and notes under the table (`\mtnotes`) hold the definitions.
def mtint(lo: str, hi: str) -> str:
    """An interval `[lo, hi]` as the secondary text of the table style (the caller formats signs
    and digits)."""
    return rf"\mtint{{[{lo},\,{hi}]}}"


def _printed(
    e: Mapping[str, Any], sign: bool, digits: int, f: Callable[[float], str] | None
) -> tuple[str, str, str]:
    num = f or (lambda x: _num(x, digits=digits, sign=sign))
    est, lo, hi = (num(e[k]) for k in ("est", "lo", "hi"))
    return est, lo, hi


def mt_ci_cells(
    e: Mapping[str, Any] | None,
    sign: bool = False,
    digits: int = 1,
    f: Callable[[float], str] | None = None,
) -> str:
    """`ci_pair_cells` in the table style: the estimate and its interval (`mtint`) as the two cells
    of a column pair `PAIR`, the estimate alone (a space before the next cell) where the printed
    ends of the interval equal it, MISSING or UNDEFINED across both cells."""
    if not e:
        return _span(MISSING)
    if e.get("est") is None or e["est"] != e["est"]:
        return _span(UNDEFINED)
    est, lo, hi = _printed(e, sign, digits, f)
    return f"{est} & " if lo == est == hi else f"{est} & {mtint(lo, hi)}"


def mt_pair(
    e: Mapping[str, Any] | None,
    sign: bool = False,
    digits: int = 1,
    f: Callable[[float], str] | None = None,
) -> str:
    """An estimate and its interval in one cell (`\\mtpair`, its widths set by `\\mtpairwidths`
    before the tabular), the interval empty where its printed ends equal the estimate; MISSING or
    UNDEFINED alone (the caller centres it)."""
    if not e:
        return MISSING
    if e.get("est") is None or e["est"] != e["est"]:
        return UNDEFINED
    est, lo, hi = _printed(e, sign, digits, f)
    ci = "" if lo == est == hi else rf"[{lo},\,{hi}]"
    return rf"\mtpair{{{est}}}{{{ci}}}"


EMPTY_PAIR = "&"  # both cells of an estimate-and-interval column pair empty
PAIR_WIDTHS = r"\mtpairwidths{$+$00.0}{[$+$00.0,\,$+$00.0]}"  # the widest pp estimate, interval


def last_col(spec: str, last: bool) -> str:
    """A `\\multicolumn` alignment, closed with `@{}` in the table's last column."""
    return spec + "@{}" if last else spec


def centred(text: str, last: bool = False, cols: int = 1) -> str:
    """`text` (a marker or a dash) centred across `cols` columns."""
    return rf"\multicolumn{{{cols}}}{{{last_col('c', last)}}}{{{text}}}"


def notes(text: str) -> str:
    """The notes under a table of the table style."""
    return rf"\mtnotes{{{text}}}"


STUDIES = (("main study", "orig"), ("replication", "rep"))  # row label, field of a contrast row
DETECTION = ("B_effect_detected_D0", "B_effect_detected_D1")  # one row where they print alike
MISSED = "D_effect_missed"  # a line of the notes where it prints 0.0 everywhere
ZERO_ALONE = "0.0 & "  # `mt_ci_cells` of an estimate of 0.0 with an interval of zero width
# the source key's contrasts that the class split (tab:classsplit) gives for both studies
IN_CLASSSPLIT = ("B_effect_update_D1", "BxD_interaction_update")


def _rep_cells(rows: Mapping[Any, Any], key: str, g: str) -> list[list[str]]:
    """Per study (`STUDIES`) the three shifts' cells of one contrast of the replication view."""
    out = []
    for _, field in STUDIES:
        cells = []
        for branch in ("src", "def", "orphan"):
            row = rows.get((branch, f"{key}_{g}"))
            cells.append(mt_ci_cells(None if row is None else row[field], sign=True))
        out.append(cells)
    return out


def tab_replication(view: dict[str, Any]) -> str:
    """Block R: the main study above the replication worlds, every contrast that the replication
    compares, each estimate with its 95% interval (the estimate alone where every world gave the
    same value). The detection effects at D=0 and D=1 share one row where they print alike (the
    keys do not depend on D), and D's effect on missed invalidation is a line of the notes where it
    prints 0.0 in every cell. The source key's B effect on Update at D=1 and its B x D are left to
    the class split (`IN_CLASSSPLIT`), which the notes name."""
    rows = view.get("rows") or {}
    n = view.get("n_worlds") or {}
    keys = (("source", "Source key"), ("cited", "Citation key"))
    merged = bool(rows) and all(
        _rep_cells(rows, DETECTION[0], g) == _rep_cells(rows, DETECTION[1], g) for g, _ in keys
    )
    zero_missed = all(
        c == ZERO_ALONE for g, _ in keys for cells in _rep_cells(rows, MISSED, g) for c in cells
    )
    shown = []  # (contrast, row label) in table order
    for key, label in REP_ROWS:
        if merged and key == DETECTION[0]:
            shown.append((key, r"$B$ effect, detection (either $D$)"))
        elif not (merged and key == DETECTION[1]) and not (zero_missed and key == MISSED):
            shown.append((key, label))
    worlds = n.get("orig"), n.get("rep")
    each = (
        f"{worlds[0]} test worlds each"
        if worlds[0] == worlds[1]
        else (f"{worlds[0] or MISSING} and {worlds[1] or MISSING} test worlds")
    )
    gap = r"@{\hspace{14pt}}"  # between the shifts' column pairs
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Every contrast of the main experiment in the main study and in the "
        rf"replication ({each}).}}",
        r"\label{tab:replication}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}ll {PAIR} {gap} {PAIR} {gap} {PAIR}@{{}}}}",
        r"\toprule",
        r"Contrast (pp) & Study & \multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} "
        r"& \multicolumn{2}{c@{}}{$\Sdel$} \\",
    ]
    for g, title in keys:
        lines += [r"\midrule", rf"\mtgroup{{8}}{{{title}}}"]
        rows_g = [(k, lab) for k, lab in shown if not (g == "source" and k in IN_CLASSSPLIT)]
        for i, (key, label) in enumerate(rows_g):
            if i:  # the main-study and replication rows of a contrast as a pair
                lines.append(r"\addlinespace[2pt]")
            for (study, _), cells in zip(STUDIES, _rep_cells(rows, key, g), strict=True):
                lines.append(
                    _row(rf"\mtin {label}" if study == STUDIES[0][0] else "", study, *cells)
                )
    missed = (
        (
            r" $D$'s effect on missed invalidation was 0.0 with both keys after every shift in "
            r"both studies."
        )
        if zero_missed
        else ""
    )
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            r"Detection: share of the changed item-times the cell invalidated since the fork. "
            r"Value equals gold: share of the re-derived item-times whose value equals gold."
            + missed
            + r" Source key, $B$ effect on Update ($D{=}1$) and $B{\times}D$: "
            r"\cref{tab:classsplit}."
        ),
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def spread_columns(*cols: str) -> str:
    """The column spec of a `tabular*` whose columns spread over its width, so that the panels of
    one table share a width: stretchable space between columns, none inside a pair `P` (the two
    cells of `est_ci_cells`, `r@{\\,}l`)."""
    out = [r"@{\extracolsep{\fill}}"]
    for i, c in enumerate(cols):
        last = i == len(cols) - 1
        if c == "P":
            out.append(r"r@{\extracolsep{0pt}\,}l")
            out.append("@{}" if last else r"@{\extracolsep{\fill}\hspace{2\tabcolsep}}")
        else:
            out.append(c + ("@{}" if last else ""))
    return "".join(out)


def _slope_cells(s: dict[str, Any] | None, unusable: bool = False) -> str:
    """Slope with its 90% bounds as the two cells of `mt_ci_cells`; the estimate grey where
    almost no item failed (the fit is not usable)."""
    if not s:
        return _span(MISSING)
    if math.isnan(s["est"]):  # no item failed: the fit is undefined
        return _span(UNDEFINED)
    if not unusable:
        return mt_ci_cells({"est": s["est"], "lo": s["lo90"], "hi": s["hi90"]}, f=_minus)
    est, lo, hi = (_minus(s[k]) for k in ("est", "lo90", "hi90"))
    return rf"\mtmuted{{{est}}} & {mtint(lo, hi)}"


def _auroc(au: float | None) -> str:
    if au is None:
        return MISSING
    return r"\mtna" if math.isnan(au) else f"{au:.2f}"


READER_ROWS = (  # label, contrast in the reader analysis, branch
    (r"$B$ effect on Update, $\Ssrc$", "B_effect_update_pooled_source", "src"),
    (r"$B$ effect on Update, $\Sdef$", "B_effect_update_pooled_source", "def"),
    (r"$D$ effect on accuracy on re-derived items, $\Ssrc$", "D_effect_repair_source", "src"),
    (r"$D$ effect on accuracy on re-derived items, $\Sdef$", "D_effect_repair_source", "def"),
    (r"$B{\times}D$ on Update, $\Ssrc$", "BxD_interaction_update_source", "src"),
    (r"$B{\times}D$ on Update, $\Sdef$", "BxD_interaction_update_source", "def"),
    (r"$D$ effect on missed invalidation, $\Ssrc$", "D_effect_missed_source", "src"),
    (r"$D$ effect on missed invalidation, $\Sdef$", "D_effect_missed_source", "def"),
)
READER_MISSED = "D_effect_missed_source"  # rows folded into the notes where all print 0.0


def tab_families(m1: dict[str, Any] | None, readers: Mapping[str, Any]) -> str:
    """Blocks M1 and M3 as two tables: the writer switches (M1, `tab:switches`), and the other
    families as reader of the main writer's stores (M3, `tab:families`) beside Qwen3.5-9B, the
    main reader, on the same worlds (the families as writer and reader, block M2, are in the main
    text's results table). The rows of D's effect on missed invalidation are a line of the notes
    where every one of their cells prints 0.0."""
    m1 = m1 or {}
    p3 = m1.get("path3") or {}
    gap = r"@{\hspace{12pt}}"  # before the two pp column pairs
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Writer switches: how well writer agreement predicts survival of the switch, and "
        r"what the switch does to retention.}",
        r"\label{tab:switches}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}l {PAIR} N{{0.00}} {gap} {PAIR} {gap} {PAIR}@{{}}}}",
        r"\toprule",
        r"Writer switch & \multicolumn{2}{c}{Slope} & \multicolumn{1}{c}{AUROC} "
        r"& \multicolumn{2}{c}{$D$ surv. (pp)} & \multicolumn{2}{c@{}}{Retention $\Delta$ (pp)} \\",
        r"\midrule",
    ]
    caveats = set(p3.get("tier_caveats", []))
    for i, tier in enumerate(TIERS):
        if i and bool(tier.marker) != bool(TIERS[i - 1].marker):
            lines.append(r"\mtgap")  # the main study's tiers, then the other writers
        key = tier_key(m1, tier)
        cal = _get(m1, "calibration", key) or {}
        mb = _get(m1, "m1b_d_protects_switch", key) or {}
        t1 = _get(m1, "t1", key) or {}
        lines.append(
            _row(
                tier.label + (r"$^\ast$" if key in caveats else ""),
                _slope_cells(cal.get("slope"), unusable=key in caveats),
                _auroc(_get(cal, "auroc", "est")),
                mt_ci_cells(mb.get("d1_minus_d0_survival"), sign=True),
                mt_ci_cells(t1.get("delta_90"), sign=True),
            )
        )
    sl, au = p3.get("pooled_slope"), _get(p3, "pooled_auroc", "est")
    m3_worlds = {len(r.get("worlds") or []) for r in readers.values() if r} - {0}
    pairs = " ".join([PAIR] * (len(FAMILIES) + 1))
    lines += [
        r"\midrule",
        _row(
            rf"pooled, {len(p3.get('tiers', [])) or MISSING} switches",
            _slope_cells(sl),
            _auroc(au),
            EMPTY_PAIR,
            EMPTY_PAIR,
        ),
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            r"Slope: calibration slope of survival on writer agreement (90\% interval). AUROC: "
            r"point estimates. $D$ surv.: "
            r"survival of $D{=}1$ minus $D{=}0$ items. Retention $\Delta$: the change in retention "
            r"of \holdmem{}(1,1) (90\% interval). $^\ast$~Almost no item failed, so the slope "
            r"(gray) is not usable."
        ),
        r"\end{table*}",
        "",
        r"\begin{table*}[t]",
        r"\caption{Other model families as reader of the main writer's stores "
        rf"({_worlds(m3_worlds)} test worlds), beside Qwen3.5-9B on the same worlds. "
        r"\Cref{tab:main}b gives them as writer and reader.}",
        r"\label{tab:families}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{mttabular}}{{@{{}}l {pairs}@{{}}}}",
        r"\toprule",
        "As reader only (pp) & "
        + " & ".join(_span(f.label) for f in FAMILIES)
        + r" & \multicolumn{2}{c@{}}{Qwen3.5-9B} \\",
        r"\midrule",
    ]
    # every family's run reads the same stores with the main reader as well: its values are
    # those of the first staged family
    main = next((sw for f in FAMILIES if (sw := readers.get(f.key))), None) or {}
    columns = [*((readers.get(f.key) or {}, "swap") for f in FAMILIES), (main, "main")]
    reader_rows = [
        (
            label,
            name,
            [mt_ci_cells(_get(sw, b, "contrasts", name, arm), sign=True) for sw, arm in columns],
        )
        for label, name, b in READER_ROWS
    ]
    folded = all(
        c == ZERO_ALONE for _, name, cells in reader_rows if name == READER_MISSED for c in cells
    )
    for label, name, cells in reader_rows:
        if not (folded and name == READER_MISSED):
            lines.append(_row(label, *cells))
    contract_valid = (
        _get(sw, "def", "systems", "holdmem_D1B1_source", arm, "ContractValid_post")
        for sw, arm in columns
    )
    lines.append(
        _row(
            r"\holdmem{}(1,1) accuracy, $\Sdef$ (\%)",
            *(_alone(_est(cv)) for cv in contract_valid),
        )
    )
    missed_note = r"$D$'s effect on missed invalidation was 0.0 in every column. " if folded else ""
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            missed_note + r"\holdmem{}(1,1) accuracy: point estimates. Qwen3.5-9B: the main "
            r"reader, on the same worlds."
        ),
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def split_families(tex: str) -> tuple[str, str]:
    """The two tables of `tab_families`, the writer switches and the other families as reader,
    each for the subsection that cites it (tab_switches.tex, tab_families.tex)."""
    switches, families = tex.split("\n\n\\begin{table*}")
    return switches + "\n", "\\begin{table*}" + families


def _worlds(counts: set[int]) -> str:
    """The number of worlds a block ran on, the same for every family (MISSING otherwise)."""
    return str(next(iter(counts))) if len(counts) == 1 else MISSING


def _span(text: str) -> str:
    """One value across both cells of an estimate-and-interval column pair."""
    return rf"\multicolumn{{2}}{{c}}{{{text}}}"


def _alone(est: str) -> str:
    """An estimate without an interval in an estimate-and-interval column pair."""
    return f"{est} & "


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    ap.add_argument("--only", nargs="*", default=[], help="file names (tab_replication.tex, ...)")
    a = ap.parse_args()
    sources = json.loads((a.root / "results" / "sources.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    inp = load_stage4(a.root, sources)
    outputs = {
        "tab_replication.tex": tab_replication(replication_view(inp["orig"], inp["rep"])),
    }
    outputs["tab_switches.tex"], outputs["tab_families.tex"] = split_families(
        tab_families(inp["m1"], inp["m3"])
    )
    # the further analyses of the main-study outputs
    from memshift.analysis import (
        ex_e1_paper,  # lazy: it imports this module's formatters
        ex_review_paper,
        ex_tables,
    )

    outputs |= ex_e1_paper.tables(ex_e1_paper.load(a.root, sources))
    outputs |= ex_review_paper.tables(ex_review_paper.load(a.root, sources))
    outputs |= ex_tables.build(a.root, sources, inp)
    from memshift.analysis import key_audit  # lazy, as above

    outputs |= key_audit.tables(key_audit.load(a.root, sources))  # offline key audit
    from memshift.analysis import ex_round2  # lazy, as above

    outputs |= ex_round2.tables(ex_round2.load(a.root, sources))  # drift, per definition, D_m
    from memshift.analysis import ex_round3  # lazy, as above

    outputs |= ex_round3.tables(ex_round3.load(a.root, sources))  # store and reader, view
    from memshift.analysis import round3_paper  # lazy, as above

    outputs |= round3_paper.tables(round3_paper.load(a.root, sources))  # refresh, readers, ...
    unknown = sorted(set(a.only) - set(outputs))
    if unknown:
        raise SystemExit(f"unknown table file(s): {', '.join(unknown)}")
    for name, text in outputs.items():
        if a.only and name not in a.only:
            continue
        (a.out / name).write_text(head + text)
        print(a.out / name)


if __name__ == "__main__":
    main()

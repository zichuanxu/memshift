"""Paper tables regenerated from the committed run analyses (tables and figures come from runs/),
with compact number formats. Sources: results/sources.json.

Robustness stages (a stage that has not run renders as MISSING):
tab_replication the main study against the replication worlds, per contrast, with both estimates
                and their 95% intervals (block R)
tab_construct   the contrasts of the model-written derivations and program statistics (block C)
tab_families    writer switches, other families as writer and reader, reader swaps (M1-M3)
tab_rendered    the five contrasts of the other families on LLM-rendered worlds against the
                template worlds (block L)
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

from memshift.analysis.gate import path2
from memshift.analysis.paper_data import (
    FAMILIES,
    ITEM_TYPES,
    MISSING,
    TIERS,
    TYPE_GROUPS,
    in_group,
    load_stage4,
    pooled_rate,
    replication_view,
    stage_json,
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
    prints 0.0 in every cell."""
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
        r"\begin{table}[t]",
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
        for i, (key, label) in enumerate(shown):
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
        ),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


SHARE = "C2_share_of_D_effect_source"
C1 = "C1_repair_D1s_minus_D0_source"
C4 = "C4_D1s_effect_missed_source"  # a line of the notes where it prints 0.0 after every shift
DM = r"$D_{\mathrm{m}}$"  # the D1s cell as displayed: D = 1 with a model-written derivation
C_ROWS = (  # label, contrast key: repair gain, authored share, interaction, detection
    (r"accuracy on re-derived items: $D_{\mathrm{m}}-D_0$", C1),
    (r"accuracy on re-derived items: $D_1-D_{\mathrm{m}}$", "C2_repair_D1_minus_D1s_source"),
    (r"share of the $D$ effect only exact recomputation gives (\%)", SHARE),
    (rf"$B{{\times}}D$ on Update, {DM}", "C3_BxD_interaction_update_D1s_source"),
    (rf"$D$ on missed invalidation, {DM}", C4),
)
NO_SHARE = "---"  # the share of the D effect is meaningless where C1 is not positive


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


def tab_construct(c: dict[str, Any] | None) -> str:
    """Block C: the contrasts of the D1s cell, displayed as D_m, after each shift, and the
    executability of the writer's programs and their agreement with gold at t0. The share of the D
    effect (C2) is an em dash where C1 is not positive: when D_m falls below D=0 the share is not a
    share (the orphan branch gives 183%)."""
    c = c or {}
    con = c.get("contrasts") or {}
    branches = ("src", "def", "orphan")
    lines = [
        r"\begin{table}[t]",
        r"\caption{The writer's programs against exact recomputation: (a) the contrasts after "
        r"each shift, (b) the programs at $t_0$, before any shift "
        rf"({c.get('n_worlds') or MISSING} worlds).}}",
        r"\label{tab:construct}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{5pt}",
        PAIR_WIDTHS,
        r"\begin{mttabular}{@{}l N{000} ccc@{}}",
        r"\toprule",
        r"(a) Contrast (pp) & & $\Ssrc$ & $\Sdef$ & \multicolumn{1}{c@{}}{$\Sdel$} \\",
        r"\midrule",
    ]

    def one(text: str, last: bool) -> str:
        """An estimate-and-interval cell; a marker or a dash centred in its column."""
        return text if text.startswith(r"\mtpair") else centred(text, last)

    def cell(branch: str, key: str) -> str:
        last = branch == branches[-1]
        c1 = _get(con, branch, C1)
        if key == SHARE and c1 and not c1["est"] > 0:
            return centred(NO_SHARE, last)
        return one(mt_pair(_get(con, branch, key), sign=key != SHARE), last)

    zero_missed = all(cell(branch, C4) == r"\mtpair{0.0}{}" for branch in branches)
    for label, key in C_ROWS:
        if not (zero_missed and key == C4):
            lines.append(_row(f"{label} &", *(cell(branch, key) for branch in branches)))
    stats = _get(c, "synth_programs", "src", "holdmem_D1sB1_source") or {}
    pct = r"(\%)"
    lines += [
        r"\midrule",
        r"(b) Program at $t_0$ & \multicolumn{1}{c}{Items} "
        rf"& \mthead{{Executable}}{{{pct}}} & \mthead{{Agrees with gold}}{{{pct}}} "
        rf"& \multicolumn{{1}}{{c@{{}}}}{{\mthead{{Agrees, if executable}}{{{pct}}}}} \\",
        r"\midrule",
    ]
    all_ = stats.get("all")
    keys = ("exec_t0", "agree_t0", "agree_t0_given_exec")
    lines.append(
        _row(
            "all item types",
            str(all_["n"]) if all_ else MISSING,
            *(one(mt_pair((all_ or {}).get(k)), k == keys[-1]) for k in keys),
        )
    )
    for _tag, label, prefixes in TYPE_GROUPS:
        types = [t for t in ITEM_TYPES if in_group(t, prefixes) and t in stats]
        cells = [rf"\mtin {label}", str(sum(stats[t]["n"] for t in types)) if types else MISSING]
        for key in keys:
            r = pooled_rate(stats, types, key)
            cells.append(rf"\mtpair{{{_num(r)}}}{{}}" if r is not None else MISSING)
        lines.append(_row(*cells))
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            rf"{DM}: $D{{=}}1$ with the writer's programs. $D_1$, $D_0$: the $D{{=}}1$ and "
            r"$D{=}0$ cells. Share of the $D$ effect only exact recomputation gives: "
            r"$(D_1-D_{\mathrm{m}})/(D_1-D_0)$ on the accuracy on re-derived items, not defined "
            rf"({NO_SHARE}) where {DM} does not raise it over $D_0$. "
            + (
                rf"{DM}'s effect on missed invalidation was 0.0 after every shift. "
                if zero_missed
                else ""
            )
            + r"(b): shares of the items. Groups: point estimates, each the item-weighted mean of "
            r"its item types."
        ),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


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


FAMILY_ROWS = (  # label, estimate in families.json (None: the largest D effect on missed)
    (r"$D$ effect on missed invalidation, largest $|\cdot|$", None),
    (r"$B$ effect on Update, $\Ssrc$, pooled over $D$", "B_update_src"),
    (r"$D$ effect on accuracy on re-derived items, $\Ssrc$", "D_repair_src"),
    (r"$D$ effect on accuracy on re-derived items, $\Sdef$", "D_repair_def"),
    (r"$B{\times}D$ on Update, $\Sdef$", "BxD_def"),
    (r"No-store BM25 $-$ \holdmem{}(1,1), accuracy, $\Sdef$", "zero_minus_hm11_def"),
)
# a line of the notes in place of the rows of D's effect on missed invalidation where all print 0.0
MISSED_LINE = (
    r"$D$'s effect on missed invalidation, its largest absolute value over both keys and the "
    r"three shifts, was 0.0"
)


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


def _largest_paired(rendered: Mapping[str, Any] | None, which: str) -> str:
    """The largest absolute D effect on missed invalidation of the rendered-world analysis
    (`which`: rendered or template) as the cells of an estimate-and-interval pair."""
    rows = [p for p in (rendered or {}).get("paired") or [] if p.get("check") == "i"]
    if not rows:
        return _span(MISSING)
    return _alone(_num(max(abs(p[which]["est"]) for p in rows)))


def largest_missed(e1: dict[str, Any] | None) -> float | None:
    """The largest absolute D effect on missed invalidation of one E1-style analysis, over both
    keys and the three shifts (None when the analysis is not staged)."""
    if not e1 or "contrasts" not in e1:
        return None
    checks = path2(e1)["checks"].values()
    return max((abs(c["estimate"]["est"]) for c in checks), default=None)


def family_missed(root: Path, sources: Mapping[str, str]) -> dict[str, float | None]:
    """Family key -> its largest absolute D effect on missed invalidation (`largest_missed` of
    the family's staged E1 analysis, None where it is not staged)."""
    return {
        f.key: largest_missed(stage_json(root, dict(sources), f"m2_{f.key}", "e1_metrics.json"))
        for f in FAMILIES
    }


def tab_families(
    m1: dict[str, Any] | None,
    families: Mapping[str, Any],
    readers: Mapping[str, Any],
    missed: Mapping[str, float | None] | None = None,
    rendered: Mapping[str, Any] | None = None,
) -> str:
    """Blocks M1-M3 as two tables: the writer switches (M1, `tab:switches`), and the other
    families as writer and reader (M2) and as reader of the main writer's stores (M3) in one column
    grid (`tab:families`), each beside Qwen3.5-9B on the same worlds: in (a) the main writer and
    reader on the template worlds of block L (the families' worlds), in (b) the main reader.
    `missed` gives each family's largest absolute D effect on missed invalidation
    (`largest_missed` of its E1 analysis). The rows of D's effect on missed invalidation are a
    line of the notes where every one of their cells prints 0.0."""
    m1 = m1 or {}
    p3 = m1.get("path3") or {}
    missed = missed or {}
    gap = r"@{\hspace{12pt}}"  # before the two pp column pairs
    lines = [
        r"\begin{table}[t]",
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
    m2_worlds = {f.get("n_worlds") for f in families.values() if f} - {None}
    m3_worlds = {len(r.get("worlds") or []) for r in readers.values() if r} - {0}
    pairs = " ".join([PAIR] * (len(FAMILIES) + 1))

    def head(panel: str, last: str) -> str:
        return (
            f"{panel} & "
            + " & ".join(_span(f.label) for f in FAMILIES)
            + rf" & \multicolumn{{2}}{{c@{{}}}}{{{last}}} \\"
        )

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
            r"(grey) is not usable."
        ),
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\caption{Other model families as writer and reader of the main experiment (a, "
        rf"{_worlds(m2_worlds)} test worlds) and as reader of the main writer's stores (b, "
        rf"{_worlds(m3_worlds)} test worlds), beside Qwen3.5-9B on the same worlds.}}",
        r"\label{tab:families}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{3.5pt}",
        rf"\begin{{mttabular}}{{@{{}}l {pairs}@{{}}}}",
        r"\toprule",
        head("(a) As writer and reader (pp)", "Qwen3.5-9B"),
        r"\midrule",
    ]
    template = (rendered or {}).get("template_same_seeds")  # Qwen3.5-9B on the families' worlds
    folded_a = False
    for label, name in FAMILY_ROWS:
        cells = []
        for fam in FAMILIES:
            f = families.get(fam.key) or {}
            if name is None:
                largest = missed.get(fam.key)
                cells.append(_alone(_num(largest)) if f and largest is not None else _span(MISSING))
            else:
                cells.append(mt_ci_cells(_get(f, "estimates", name), sign=True))
        if name is None:
            cells.append(_largest_paired(rendered, "template"))
            if all(c == ZERO_ALONE for c in cells):
                folded_a = True
                continue
        else:
            cells.append(mt_ci_cells(_get(template, "estimates", name), sign=True))
        lines.append(_row(label, *cells))
    lines += [
        r"\midrule",
        head("(b) As reader only (pp)", "Qwen3.5-9B"),
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
    folded_b = all(
        c == ZERO_ALONE for _, name, cells in reader_rows if name == READER_MISSED for c in cells
    )
    for label, name, cells in reader_rows:
        if not (folded_b and name == READER_MISSED):
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
    if folded_a and folded_b:
        missed_note = (
            r"$D$'s effect on missed invalidation was 0.0 in every column, in (a) its "
            r"largest absolute value over both keys and the three shifts. "
        )
    elif folded_a:
        missed_note = f"(a): {MISSED_LINE} in every column. "
    elif folded_b:
        missed_note = r"(b): $D$'s effect on missed invalidation was 0.0 in every column. "
    else:
        missed_note = (
            r"$D$ effect on missed invalidation in (a): its largest absolute value "
            r"over both keys and the three shifts. "
        )
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(
            missed_note + r"\holdmem{}(1,1) accuracy: point estimates. Qwen3.5-9B: in (a) the main "
            r"writer and reader, in (b) the main reader, on the same worlds."
        ),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _worlds(counts: set[int]) -> str:
    """The number of worlds a block ran on, the same for every family (MISSING otherwise)."""
    return str(next(iter(counts))) if len(counts) == 1 else MISSING


def _span(text: str) -> str:
    """One value across both cells of an estimate-and-interval column pair."""
    return rf"\multicolumn{{2}}{{c}}{{{text}}}"


def _alone(est: str) -> str:
    """An estimate without an interval in an estimate-and-interval column pair."""
    return f"{est} & "


def tab_rendered(m: dict[str, Any] | None) -> str:
    """Block L: the five contrasts of the other families on LLM-rendered worlds beside the
    same-seed template worlds, and the paired differences (the rendering descriptives are in the
    text, from the macros of `paper_numbers`)."""
    m = m or {}
    rend, temp = m.get("rendered") or {}, m.get("template_same_seeds") or {}
    paired = {p["name"]: p for p in m.get("paired") or []}
    gap = r"@{\hspace{12pt}}"  # between the three column pairs
    lines = [
        r"\begin{table}[t]",
        r"\caption{The contrasts of Table~\ref{tab:families}a on rewritten worlds and on "
        rf"template worlds of the same seeds ({len(m.get('worlds', [])) or MISSING} worlds).}}",
        r"\label{tab:rendered}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}l {PAIR} {gap} {PAIR} {gap} {PAIR}@{{}}}}",
        r"\toprule",
        r"Contrast (pp) & \multicolumn{2}{c}{Rewritten} & \multicolumn{2}{c}{Template} "
        r"& \multicolumn{2}{c@{}}{Difference} \\",
        r"\midrule",
    ]
    folded = False
    for label, name in FAMILY_ROWS:
        if name is None:  # the largest D effect on missed invalidation: no paired difference
            largest = [_largest_paired(m, "rendered"), _largest_paired(m, "template")]
            if all(c == ZERO_ALONE for c in largest):
                folded = True
                continue
            cells = [label, *largest, centred(UNDEFINED, last=True, cols=2)]
        else:
            p = paired.get(name) or {}
            cells = [
                label,
                mt_ci_cells(_get(rend, "estimates", name), sign=True),
                mt_ci_cells(_get(temp, "estimates", name), sign=True),
                mt_ci_cells(p.get("diff"), sign=True),
            ]
        lines.append(_row(*cells))
    missed = (
        f" {MISSED_LINE} on both kinds of worlds."
        if folded
        else rf" The $D$ effect on missed invalidation has no paired difference ({UNDEFINED})."
    )
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        notes(r"Difference: rewritten minus template, paired by item." + missed),
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


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
    missed = family_missed(a.root, sources)
    outputs = {
        "tab_replication.tex": tab_replication(replication_view(inp["orig"], inp["rep"])),
        "tab_construct.tex": tab_construct(inp["c"]),
        "tab_families.tex": tab_families(inp["m1"], inp["m2"], inp["m3"], missed, inp["rendered"]),
        "tab_rendered.tex": tab_rendered(inp["rendered"]),
    }
    # the further analyses of the main-study outputs
    from memshift.analysis import (
        ex_e1_paper,  # lazy: it imports this module's formatters
        ex_review_paper,
        ex_tables,
    )

    outputs |= ex_e1_paper.tables(ex_e1_paper.load(a.root, sources))
    outputs |= ex_review_paper.tables(ex_review_paper.load(a.root, sources))
    outputs |= ex_tables.build(a.root, sources, inp)
    from memshift.analysis import controls_paper  # lazy, as above

    outputs |= controls_paper.tables(controls_paper.load(a.root, sources))  # control cells
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

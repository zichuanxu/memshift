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


def tab_replication(view: dict[str, Any]) -> str:
    """Block R: the main study (stage 2) above the replication worlds, every contrast that the
    replication compares, each estimate with its 95% interval (the estimate alone where every
    world gave the same value)."""
    rows = view.get("rows") or {}
    n = view.get("n_worlds") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Every contrast of the main experiment in the main study (main) and on the "
        rf"{n.get('rep') or MISSING} fresh test worlds of the replication (fresh). Detection: "
        r"share of the changed item-times the cell invalidated. Value equals gold: share of the "
        r"re-derived item-times whose value equals gold.}",
        r"\label{tab:replication}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}ll {PAIR} {PAIR} {PAIR}@{{}}}}",
        r"\toprule",
        r"contrast (pp) & worlds & \multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} "
        r"& \multicolumn{2}{c}{$\Sdel$} \\",
    ]
    for g, title in (("source", "source key"), ("cited", "citation key")):
        lines += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for key, label in REP_ROWS:
            for study, field in (("main", "orig"), ("fresh", "rep")):
                cells = [rf"\quad {label}" if study == "main" else "", study]
                for branch in ("src", "def", "orphan"):
                    row = rows.get((branch, f"{key}_{g}"))
                    cells.append(ci_pair_cells(None if row is None else row[field], sign=True))
                lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


SHARE = "C2_share_of_D_effect_source"
C1 = "C1_repair_D1s_minus_D0_source"
DM = r"$D_{\mathrm{m}}$"  # the D1s cell as displayed: D = 1 with a model-written derivation
C_ROWS = (  # label, contrast key: repair gain, authored share, interaction, detection
    (r"accuracy on re-derived items: $D_{\mathrm{m}}-D_0$", C1),
    (r"accuracy on re-derived items: $D_1-D_{\mathrm{m}}$", "C2_repair_D1_minus_D1s_source"),
    (r"share of the $D$ effect only exact recomputation gives (\%)", SHARE),
    (rf"$B{{\times}}D$ on Update, {DM}", "C3_BxD_interaction_update_D1s_source"),
    (rf"$D$ on missed invalidation, {DM}", "C4_D1s_effect_missed_source"),
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
    lines = [
        r"\begin{table}[t]",
        r"\caption{The writer's programs against exact recomputation: (a) the contrasts after "
        r"each shift, (b) the programs at $t_0$, before any shift "
        rf"({c.get('n_worlds') or MISSING} worlds). {DM}: $D{{=}}1$ with the writer's "
        r"programs. $D_1$, $D_0$: the $D{=}1$ and $D{=}0$ cells. Share of the $D$ effect only "
        r"exact recomputation gives: $(D_1-D_{\mathrm{m}})/(D_1-D_0)$ on the accuracy on "
        rf"re-derived items, not defined ({NO_SHARE}) where {DM} does not raise it over $D_0$. "
        r"(b) gives shares of the items in percent, each group the item-weighted mean of its "
        r"item types.}",
        r"\label{tab:construct}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l r {PAIR} {PAIR} {PAIR}@{{}}}}",
        r"\toprule",
        r"(a) contrast (pp) & & \multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} "
        r"& \multicolumn{2}{c}{$\Sdel$} \\",
        r"\midrule",
    ]

    def cell(branch: str, key: str) -> str:
        c1 = _get(con, branch, C1)
        if key == SHARE and c1 and not c1["est"] > 0:
            return rf"\multicolumn{{2}}{{c}}{{{NO_SHARE}}}"
        return est_ci_cells(_get(con, branch, key), sign=key != SHARE)

    for label, key in C_ROWS:
        lines.append(_row(label, "", *(cell(branch, key) for branch in ("src", "def", "orphan"))))
    stats = _get(c, "synth_programs", "src", "holdmem_D1sB1_source") or {}
    lines += [
        r"\midrule",
        r"(b) program at $t_0$ & items & \multicolumn{2}{c}{executable} "
        r"& \multicolumn{2}{c}{agrees with gold} & \multicolumn{2}{c}{agrees, if executable} \\",
        r"\midrule",
    ]
    all_ = stats.get("all")
    lines.append(
        _row(
            "all item types",
            str(all_["n"]) if all_ else MISSING,
            *(
                est_ci_cells((all_ or {}).get(k))
                for k in ("exec_t0", "agree_t0", "agree_t0_given_exec")
            ),
        )
    )
    for _tag, label, prefixes in TYPE_GROUPS:
        types = [t for t in ITEM_TYPES if in_group(t, prefixes) and t in stats]
        cells = [rf"\quad {label}", str(sum(stats[t]["n"] for t in types)) if types else MISSING]
        for key in ("exec_t0", "agree_t0", "agree_t0_given_exec"):
            r = pooled_rate(stats, types, key)
            cells.append((_num(r) if r is not None else MISSING) + " &")
        lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _slope_cells(s: dict[str, Any] | None, unusable: bool = False) -> str:
    """Slope with its 90% bounds as the two cells of `ci_pair_cells`; both grey where almost no
    item failed (the fit is not usable)."""
    if not s:
        return rf"\multicolumn{{2}}{{c}}{{{MISSING}}}"
    if math.isnan(s["est"]):  # no item failed: the fit is undefined
        return rf"\multicolumn{{2}}{{c}}{{{UNDEFINED}}}"
    if not unusable:
        return ci_pair_cells({"est": s["est"], "lo": s["lo90"], "hi": s["hi90"]}, f=_minus)
    est, lo, hi = (_minus(s[k]) for k in ("est", "lo90", "hi90"))
    return rf"\textcolor{{gray}}{{{est}}} & \textcolor{{gray}}{{\scriptsize [{lo},\,{hi}]}}"


def _auroc(au: float | None) -> str:
    if au is None:
        return MISSING
    return UNDEFINED if math.isnan(au) else f"{au:.2f}"


FAMILY_ROWS = (  # label, estimate in families.json (None: the largest D effect on missed)
    (r"(i) $D$ effect on missed invalidation, largest $|\cdot|$", None),
    (r"(ii) $B$ effect on Update, $\Ssrc$, pooled over $D$", "B_update_src"),
    (r"(iii) $D$ effect on accuracy on re-derived items, $\Ssrc$", "D_repair_src"),
    (r"(iii) $D$ effect on accuracy on re-derived items, $\Sdef$", "D_repair_def"),
    (r"(iv) $B{\times}D$ on Update, $\Sdef$", "BxD_def"),
    (r"(v) no-store BM25 $-$ \holdmem{}(1,1), accuracy, $\Sdef$", "zero_minus_hm11_def"),
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
) -> str:
    """Blocks M1-M3 as two tables: the writer switches (M1, `tab:switches`), and the other
    families as writer and reader (M2) and as reader of the main writer's stores beside the main
    reader (M3) in one column grid (`tab:families`). `missed` gives each family's largest absolute
    D effect on missed invalidation (`largest_missed` of its E1 analysis)."""
    m1 = m1 or {}
    p3 = m1.get("path3") or {}
    missed = missed or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Writer switches: how well writer agreement predicts survival of the switch, and "
        r"what the switch does to retention. Slope: calibration slope of survival on writer "
        r"agreement (90\% interval). $D$ surv.: survival of $D{=}1$ minus $D{=}0$ items. "
        r"Retention $\Delta$: the change in retention of \holdmem{}(1,1) (90\% interval). "
        r"$^\ast$~Almost no item failed, so the slope (grey) is not usable.}",
        r"\label{tab:switches}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l {PAIR} r {PAIR} {PAIR}@{{}}}}",
        r"\toprule",
        r"writer switch & \multicolumn{2}{c}{slope} & AUROC & \multicolumn{2}{c}{$D$ surv. (pp)} "
        r"& \multicolumn{2}{c}{retention $\Delta$ (pp)} \\",
        r"\midrule",
    ]
    caveats = set(p3.get("tier_caveats", []))
    for tier in TIERS:
        key = tier_key(m1, tier)
        cal = _get(m1, "calibration", key) or {}
        mb = _get(m1, "m1b_d_protects_switch", key) or {}
        t1 = _get(m1, "t1", key) or {}
        lines.append(
            _row(
                tier.label + (r"$^\ast$" if key in caveats else ""),
                _slope_cells(cal.get("slope"), unusable=key in caveats),
                _auroc(_get(cal, "auroc", "est")),
                ci_pair_cells(mb.get("d1_minus_d0_survival"), sign=True),
                ci_pair_cells(t1.get("delta_90"), sign=True),
            )
        )
    sl, au = p3.get("pooled_slope"), _get(p3, "pooled_auroc", "est")
    m2_worlds = {f.get("n_worlds") for f in families.values() if f} - {None}
    m3_worlds = {len(r.get("worlds") or []) for r in readers.values() if r} - {0}
    lines += [
        r"\midrule",
        _row(
            rf"pooled, {len(p3.get('tiers', [])) or MISSING} switches",
            _slope_cells(sl),
            _auroc(au),
            _span(""),
            _span(""),
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\caption{Other model families as writer and reader of the main experiment (a, "
        rf"{_worlds(m2_worlds)} test worlds) and as reader of the main writer's stores (b, "
        rf"{_worlds(m3_worlds)} test worlds). Row (i): the largest absolute $D$ effect on missed "
        r"invalidation over both keys and the three shifts. In (b) the last column is the main "
        r"reader on the same worlds.}",
        r"\label{tab:families}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l {' '.join([PAIR] * (len(FAMILIES) + 1))}@{{}}}}",
        r"\toprule",
        r"(a) as writer and reader (pp) & "
        + " & ".join(_span(f.label) for f in FAMILIES)
        + rf" & {_span('')} \\",
        r"\midrule",
    ]
    for label, name in FAMILY_ROWS:
        cells = []
        for fam in FAMILIES:
            f = families.get(fam.key) or {}
            if name is None:
                largest = missed.get(fam.key)
                cells.append(_alone(_num(largest)) if f and largest is not None else _span(MISSING))
            else:
                cells.append(ci_pair_cells(_get(f, "estimates", name), sign=True))
        lines.append(_row(label, *cells, _span("")))
    lines += [
        r"\midrule",
        r"(b) as reader only (pp) & "
        + " & ".join(_span(f.label) for f in FAMILIES)
        + rf" & {_span('Qwen3.5-9B')} \\",
        r"\midrule",
    ]
    # every family's run reads the same stores with the main reader as well: its values are
    # those of the first staged family
    main = next((sw for f in FAMILIES if (sw := readers.get(f.key))), None) or {}
    columns = [*((readers.get(f.key) or {}, "swap") for f in FAMILIES), (main, "main")]
    for label, name, b in READER_ROWS:
        cells = [
            ci_pair_cells(_get(sw, b, "contrasts", name, arm), sign=True) for sw, arm in columns
        ]
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
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
    lines = [
        r"\begin{table}[t]",
        r"\caption{The five contrasts of Table~\ref{tab:families}a on rewritten worlds and on "
        rf"template worlds of the same seeds ({len(m.get('worlds', [])) or MISSING} worlds). "
        r"Difference: "
        rf"rewritten minus template, paired by item. Row (i) has no paired difference "
        rf"({UNDEFINED}).}}",
        r"\label{tab:rendered}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{tabular}}{{@{{}}l {PAIR} {PAIR} {PAIR}@{{}}}}",
        r"\toprule",
        r"contrast (pp) & \multicolumn{2}{c}{rewritten} & \multicolumn{2}{c}{template} "
        r"& \multicolumn{2}{c}{difference} \\",
        r"\midrule",
    ]
    for label, name in FAMILY_ROWS:
        if name is None:  # row (i): the largest D effect on missed invalidation
            p2 = [p for p in paired.values() if p.get("check") == "i"]

            def largest(which: str, rows: list[dict[str, Any]] = p2) -> str:
                if not rows:
                    return _span(MISSING)
                return _alone(_num(max(abs(p[which]["est"]) for p in rows)))

            cells = [label, largest("rendered"), largest("template"), _span(UNDEFINED)]
        else:
            p = paired.get(name) or {}
            cells = [
                label,
                ci_pair_cells(_get(rend, "estimates", name), sign=True),
                ci_pair_cells(_get(temp, "estimates", name), sign=True),
                ci_pair_cells(p.get("diff"), sign=True),
            ]
        lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
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
        "tab_families.tex": tab_families(inp["m1"], inp["m2"], inp["m3"], missed),
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

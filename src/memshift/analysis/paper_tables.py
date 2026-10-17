"""Paper tables regenerated from the committed run analyses (plan §10), with compact number formats
that fit one ICML column. Sources: results/sources.json and audit/system_audit.json.

tab_gate        pre-registered falsification paths (the paper's gain, leak and agreement checks)
                and what the result supports (gate A)
tab_robust      reader swap (theta_r) and embedding swap (theta_e): contrasts and post-shift scores
tab_robust_ci   the same with 95% intervals (appendix)
tab_membership  T1-T4 for a selection of systems (the full audit is in the appendix)
Stage 4 (prereg v2 §8; a stage that has not run renders as MISSING):
tab_replication the main study against the replication worlds, per contrast, with verdict
                marks (block R)
tab_construct   C1-C4 for the writer-authored derivations and program statistics (block C)
tab_native      native systems: labels, six comparisons inside / outside, values, cost (block N)
tab_families    writer switches, other families as writer and reader, reader swaps (M1-M3)
tab_rendered    the M2 checks on LLM-rendered worlds against the template worlds (block L)
The printed captions and labels name the blocks and paths in words (the paper's names).
Usage: python -m memshift.analysis.paper_tables [--root .] [--out paper/generated]
       [--only tab_gate.tex ...]   (write only the named files; every table is still computed)
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from memshift.analysis import replication
from memshift.analysis.audit_table import NAMES, cell, factor
from memshift.analysis.e5 import DOCUMENTED_LABELS, _inferred
from memshift.analysis.paper_data import (
    FAMILIES,
    FAMILY_CHECKS,
    ITEM_TYPES,
    MISSING,
    NATIVES,
    TIERS,
    TYPE_GROUPS,
    Native,
    claim_label,
    in_group,
    label_text,
    load_stage4,
    m3_signs,
    native_results,
    pooled_rate,
    replication_view,
    synth_holds,
    tier_key,
)
from memshift.analysis.paper_data import get as _get

UNDEFINED = "---"  # a fit that has no value (for example no item failed)
MEMBERSHIP_ROWS = (
    "HoldMem (1,1)",
    "零系统（ours）",
    "Plain RAG",
    "Mem0 (self-hosted)",
    "Graphiti / Zep",
    "Letta (.af)",
    "ChatGPT memory",
    "Claude memory",
    "Gemini memory",
)


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


def _claim_test(claim: str) -> str:
    """The clause that says what tested the registered claim's quantitative prediction
    (re-review): only the agreement check (falsification path 3), which concerns writer
    switches; nothing for another claim."""
    if "quantitative" not in claim:
        return ""
    return (
        r". The only test of the claim's quantitative prediction is the agreement check, on writer "
        r"switches"
    )


def tab_gate(gate: dict[str, Any]) -> str:
    p1, p2, p3 = gate.get("path1") or {}, gate.get("path2") or {}, gate.get("path3") or {}
    checks = list((p2.get("checks") or {}).values())
    largest = max((abs(c["estimate"]["est"]) for c in checks), default=None)
    sl, au = p3.get("pooled_slope") or {}, p3.get("pooled_auroc") or {}

    def verdict(p: dict[str, Any]) -> str:
        return MISSING if "triggered" not in p else _trig(p)

    slope = (
        f"slope {interval(_minus(sl['est']), _minus(sl['lo90']), _minus(sl['hi90']))}, "
        f"AUROC {au['est']:.2f}"
        if sl and au
        else MISSING
    )
    rows = [  # the falsification paths 1-3 of the registrations, named by what each checks
        (
            "gain",
            r"$B$ effect on Update ($\Ssrc$) $\le +2$\,pp",
            _ci(p1.get("estimate"), sign=True) + r"\,pp",
            verdict(p1),
        ),
        (
            "leak",
            rf"$D$ moves missed invalidation ($\ge 2$\,pp in any of {len(checks)} contrasts)",
            rf"all {_num(largest)}\,pp" if largest == 0 else rf"max {_num(largest)}\,pp",
            verdict(p2),
        ),
        (
            "agreement",
            r"calibration slope $\le 0$ (90\% bound) or AUROC $<0.60$",
            slope,
            verdict(p3),
        ),
    ]
    claim = claim_label(gate.get("claim_level"))
    lines = [
        r"\begin{table}[t]",
        r"\caption{Registered falsification checks, decided mechanically on the "
        rf"{gate.get('n_worlds_e1', MISSING)} test worlds. A check is triggered when its "
        r"condition holds. Last row: what the registration lets the paper claim when no check "
        r"is triggered, and a triggered check restricts this claim"
        rf"{_claim_test(claim)}. "
        r"Gain check: estimate with its 95\% world-bootstrap interval. Leak check: the $D$ effect "
        r"on missed invalidation for both keys (source-epoch and cited-cluster) after each of the "
        r"three shifts. Agreement check: pooled calibration slope over the writer switches, with "
        r"90\% bounds.}",
        r"\label{tab:gate}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l >{\raggedright\arraybackslash}p{0.43\linewidth}"
        r" >{\raggedright\arraybackslash}p{0.40\linewidth}@{}}",
        r"\toprule",
        r"check & triggered if & observed (status) \\",
        r"\midrule",
    ]
    lines += [rf"{n} & {cond} & {obs} ({status})" + r" \\" for n, cond, obs, status in rows]
    lines += [
        r"\midrule",
        rf"\multicolumn{{3}}{{@{{}}l}}{{\emph{{{claim}}}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _swap(e7: dict[str, Any], key: str, b: str, what: str, name: str, arm: str) -> Any:
    entry = (e7.get(key) or {}).get(b) or {}
    if what == "contrast":
        return ((entry.get("contrasts") or {}).get(name) or {}).get(arm)
    systems = entry.get("systems") or {}
    return ((systems.get(name) or {}).get(arm) or {}).get("ContractValid_post")


def tab_robust(e7: dict[str, Any], with_ci: bool = False) -> str:
    def fmt(e: dict[str, Any] | None, sign: bool) -> str:
        if with_ci:
            return _ci(e, sign=sign)
        return _num(e["est"], sign=sign) if e else MISSING

    reader = [
        (r"$B$ effect on Update (pp)", "contrast", "B_effect_update_pooled_source", True),
        (r"$D$ effect on repair (pp)", "contrast", "D_effect_repair_source", True),
        (r"\holdmem{}(1,1)", "system", "holdmem_D1B1_source", False),
        (r"zero system", "system", "zero", False),
        (r"plain RAG", "system", "rag", False),
    ]
    n = len((e7.get("theta_r") or {}).get("worlds", [])) or MISSING
    note = (
        r" 95\% world-bootstrap intervals in brackets.}"
        if with_ci
        else r" Intervals: \cref{tab:robust-ci}.}"
    )
    env = "table" if with_ci else "memtab"  # the main-text version shares a float with membership
    lines = [
        r"\begin{table}[h]" if with_ci else r"\begin{memtab}",
        rf"\caption{{Robustness to the reader and the embedding model ({n} test worlds). "
        r"Contrasts in pp; system rows are post-shift ContractValid (\%)." + note,
        r"\label{tab:robust-ci}" if with_ci else r"\label{tab:robust}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l rr rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(l){4-5}",
        r"& main & swap & main & swap \\",
        r"\midrule",
        r"\multicolumn{5}{@{}l}{\emph{reader: Qwen3.5-9B $\to$ Ministral-3-14B}} \\",
    ]
    for label, what, name, sign in reader:
        cells = [
            fmt(_swap(e7, "theta_r", b, what, name, arm), sign)
            for b in ("src", "def")
            for arm in ("main", "swap")
        ]
        lines.append(rf"\quad {label} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"\multicolumn{5}{@{}l}{\emph{embedding: Qwen3-Embedding-0.6B $\to$ bge-m3}} \\")
    cells = [
        fmt(_swap(e7, "theta_e", b, "system", "rag", arm), False)
        for b in ("src", "def")
        for arm in ("main", "swap")
    ]
    lines.append(r"\quad plain RAG & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    return "\n".join(lines) + "\n"


def tab_membership(rows: list[dict[str, Any]]) -> str:
    by_name = {r["short_name"]: r for r in rows}
    lines = [  # a `memtab` (defined in main.tex) set next to the robustness table in one float
        r"\begin{memtab}",
        r"\caption{Membership tests for selected systems (all systems: \cref{tab:audit}).",
        r"\pass{} pass, \fail{} fail, ? undeterminable; $^\dagger$ inferred, $^\ddagger$ by",
        r"construction. D/B: part.\ = partial, t.v.\ = temporal validity only.}",
        r"\label{tab:membership}",
        r"\centering\small",
        r"\setlength{\tabcolsep}{4.5pt}",
        r"\begin{tabular}{@{}l cc cccc@{}}",
        r"\toprule",
        r"system & $D$ & $B$ & T1 & T2 & T3 & T4 \\",
        r"\midrule",
    ]
    for name in MEMBERSHIP_ROWS:
        r = by_name.get(name)
        if r is None:
            continue
        label = NAMES.get(name, name).replace("HoldMem", r"\holdmem{}")
        cells = [label, factor(r["D"]), factor(r["B"])]
        cells += [cell(r, t) for t in ("T1", "T2", "T3", "T4")]
        lines.append(" & ".join(cells) + r" \\")
        if name == "Plain RAG":
            lines.append(r"\midrule")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{memtab}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------ stage 4 tables
# Prereg v2 §8. A stage that has not been staged renders as MISSING in every cell that needs it;
# the tables are generated from the same views as the stage-4 macros (`analysis.paper_data`).
PASS, FAIL, SAME = r"\pass", r"\fail", r"$\sim$"
REP_MARKS = {
    "replicated": PASS,
    "same direction": SAME,
    "not replicated": FAIL,
    "not triggered": PASS,
    "triggered": FAIL,
}
REP_ROWS = (  # E1 contrast (without the granularity suffix), row label
    ("B_effect_detected_D0", r"$B$ effect, detection ($D{=}0$)"),
    ("B_effect_detected_D1", r"$B$ effect, detection ($D{=}1$)"),
    ("B_effect_update_D1", r"$B$ effect, Update ($D{=}1$)"),
    ("B_effect_update_pooled", r"$B$ effect, Update (pooled $D$)"),
    ("D_effect_missed", r"$D$ effect, missed invalidation"),
    ("D_effect_repair", r"$D$ effect, repair (behavioural)"),
    ("D_effect_repair_mech", r"$D$ effect, repair (mechanistic)"),
    ("BxD_interaction_update", r"$B{\times}D$ interaction, Update"),
)


def _mark(ok: bool | None) -> str:
    return MISSING if ok is None else (PASS if ok else FAIL)


def _est(
    e: dict[str, Any] | None, sign: bool = False, scale: float = 100.0, digits: int = 1
) -> str:
    return _num(e["est"], scale, digits, sign) if e else MISSING


def _row(*cells: str) -> str:
    return " & ".join(cells) + r" \\"


def _within(t1: dict[str, Any] | None) -> bool | None:
    return None if not t1 else bool(t1.get("equivalent"))


def _as_predicted(row: dict[str, Any]) -> str:
    """Marks for the main-study and the replication estimate against the pre-registered
    direction."""
    return r"\,".join(
        _mark(replication.as_predicted(row["kind"], row[stage])) for stage in ("orig", "rep")
    )


def _trig(p: dict[str, Any] | None) -> str:
    return MISSING if p is None else ("triggered" if p["triggered"] else "not triggered")


def tab_replication(view: dict[str, Any]) -> str:
    """Block R: the main study (stage 2) beside the replication worlds, every contrast of the
    v1 §4 list that has a verdict, with the verdict mark; the paths and membership tests below
    the table."""
    rows = view.get("rows") or {}
    n = view.get("n_worlds") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Replication on "
        rf"{n.get('rep') or MISSING} fresh test worlds. Registered contrasts in pp (main study: "
        rf"{n.get('orig') or MISSING} worlds, fresh: estimate on the fresh worlds). Detection: "
        r"share of the changed item-times the cell invalidated. Repair: among the re-derived "
        r"item-times, the share the reader then answers correctly (behavioural) or whose "
        r"re-derived value equals gold (mechanistic). Repl.: \pass{} replicated (same sign as the "
        r"main study, 95\% interval excludes 0), $\sim$ same direction (interval includes 0), "
        r"\fail{} not replicated. For the $D$ effect on missed invalidation \pass{} means the leak "
        r"check is not triggered. As pred.: whether the main-study and the fresh-world estimate "
        r"(in this order) lie in the registered direction, \pass{} yes and \fail{} no. The "
        r"direction is a positive point estimate (before rounding) for every effect predicted "
        r"non-zero, whatever the interval. A contrast can replicate and still be against the "
        r"prediction.}",
        r"\label{tab:replication}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",  # 13 columns in a 5.5in text width
        r"\begin{tabular}{@{}l rrcc rrcc rrcc@{}}",
        r"\toprule",
        r"& \multicolumn{4}{c}{$\Ssrc$} & \multicolumn{4}{c}{$\Sdef$} "
        r"& \multicolumn{4}{c}{$\Sdel$} \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}",
        r"contrast & main & fresh & repl. & as pred. & main & fresh & repl. & as pred. "
        r"& main & fresh & repl. & as pred. \\",
    ]
    for g, title in (("source", "source-epoch key"), ("cited", "cited-cluster key")):
        lines += [r"\midrule", rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for key, label in REP_ROWS:
            cells = [rf"\quad {label}"]
            for branch in ("src", "def", "orphan"):
                row = rows.get((branch, f"{key}_{g}"))
                if row is None:
                    cells += [MISSING] * 4
                    continue
                cells += [
                    _est(row["orig"], sign=True),
                    _est(row["rep"], sign=True),
                    REP_MARKS.get(row["verdict"], MISSING),
                    _as_predicted(row),
                ]
            lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip", r"{\footnotesize\raggedright"]
    paths, orig = view.get("paths") or {}, view.get("paths_orig") or {}
    p1 = paths.get("path1")
    checks = _get(paths, "path2", "checks") or {}
    largest = max((abs(c["estimate"]["est"]) for c in checks.values()), default=None)
    p3 = _get(paths, "path3", "pooled_slope")
    t1, t3, t4 = view.get("t1") or {}, view.get("t3") or {}, view.get("t4") or {}

    def tally(rows_: dict[str, Any]) -> str:
        return (
            MISSING
            if not rows_
            else f"{sum(r['verdict'] == 'replicated' for r in rows_.values())} of {len(rows_)}"
        )

    lines += [
        rf"Falsification checks on the fresh worlds: gain check {_trig(p1)} (main study: "
        rf"{_trig(orig.get('path1'))}, $B$ effect on Update "
        rf"{_ci(p1['estimate'], True) if p1 else MISSING}\,pp), "
        rf"leak check {_trig(paths.get('path2'))} (largest $|D$ effect on missed invalidation$|$ "
        rf"{_num(largest)}\,pp), agreement check {_trig(paths.get('path3'))} "
        rf"(pooled slope {_num(p3['est'], 1, 2) if p3 else MISSING}). "
        rf"\emph{{{claim_label(view.get('claim_level'))}}}"
        rf"{_claim_test(claim_label(view.get('claim_level')))}. "
        rf"Diagnostic tests with the main-study verdict: retention test {tally(t1)}, "
        rf"utility floor {tally(t3)}, cost ceiling {tally(t4)} replicated.",
        r"\par}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


SHARE = "C2_share_of_D_effect_source"
C1 = "C1_repair_D1s_minus_D0_source"
DM = r"$D_{\mathrm{m}}$"  # the D1s cell as displayed: D = 1 with a model-written derivation
C_ROWS = (  # label, prediction, contrast key, rule (analysis.paper_data.synth_holds)
    # the registered predictions C1-C4: repair gain, authored share, interaction, detection
    (r"repair: $D_{\mathrm{m}}-D_0$", r"$>0$", C1, "above"),
    (r"repair: $D_1-D_{\mathrm{m}}$", r"$\ge 0$", "C2_repair_D1_minus_D1s_source", "nonneg"),
    (r"authored share of the $D$ effect (\%)", "", SHARE, None),
    (
        rf"$B{{\times}}D$ on Update, {DM}",
        r"$>0$",  # registered on S_def only: the caption says so (5.5in leaves no room here)
        "C3_BxD_interaction_update_D1s_source",
        "above",
    ),
    (rf"$D$ on missed invalidation, {DM}", "$=0$", "C4_D1s_effect_missed_source", "path2"),
)
NO_SHARE = "---"  # the share of the D effect is meaningless where C1 is not positive
# prereg v2 section 4 registers C3 on S_def only (C1, C2, C4 on S_src and S_def): the S_src value of
# C3 is shown without a verdict mark, as exploratory
C_REGISTERED = {"C3_BxD_interaction_update_D1s_source": ("def",)}


def tab_construct(c: dict[str, Any] | None) -> str:
    """Block C: C1-C4 for the D1s cell, displayed as D_m (verdict marks on S_src and S_def where the
    prediction is registered there, C3 on S_def only; orphan exploratory) and the executability of
    the writer's programs and their agreement with gold at t0. The share of the D effect (C2) is an
    em dash where C1 is not positive: when D_m falls below D=0 the share is not a share (the orphan
    branch gives 183%)."""
    c = c or {}
    con = c.get("contrasts") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Model-written derivations ("
        rf"{c.get('n_worlds') or MISSING} test worlds): repair when the writer model writes the "
        rf"executable derivations instead of the authored derivation. {DM}: $D{{=}}1$ with a "
        r"model-written derivation. $D_1$: the $D{=}1$ cell. $D_0$: the $D{=}0$ cell. Upper part: "
        r"registered contrasts in pp (source-epoch key, 95\% intervals). Pred.: the "
        r"registered prediction, \pass{} it holds, \fail{} it does not. The interaction is "
        r"registered for $\Sdef$ only, so its $\Ssrc$ value, like $\Sdel$, is exploratory. "
        r"Authored share: $(D_1-D_{\mathrm{m}})/(D_1-D_0)$ on repair, the part of the $D$ effect "
        rf"the model-written derivation does not recover, not defined ({NO_SHARE}) where the "
        r"repair gain is not positive. Lower part: the writer's program at $t_0$, before any "
        r"shift (item-weighted means for the groups).}",
        r"\label{tab:construct}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}l c rc rc r@{}}",
        r"\toprule",
        r"& pred. & \multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} "
        r"& $\Sdel$ \\",
        r"\midrule",
    ]

    def cell(branch: str, key: str) -> str:
        c1 = _get(con, branch, C1)
        if key == SHARE and c1 and not c1["est"] > 0:
            return NO_SHARE
        return _ci(_get(con, branch, key), sign=key != SHARE)

    for label, pred, key, rule in C_ROWS:
        cells = [label, pred]
        for branch in ("src", "def"):
            e = _get(con, branch, key)
            judged = rule is not None and branch in C_REGISTERED.get(key, ("src", "def"))
            cells += [cell(branch, key), _mark(synth_holds(rule, e)) if rule and judged else ""]
        cells.append(cell("orphan", key))
        lines.append(_row(*cells))
    stats = _get(c, "synth_programs", "src", "holdmem_D1sB1_source") or {}
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{@{}l rrrr@{}}",
        r"\toprule",
        r"program at $t_0$ & items & executable & agrees with gold & agrees, if executable \\",
        r"\midrule",
    ]

    def rate(e: dict[str, Any] | None) -> str:
        return _ci(e) if e else MISSING

    all_ = stats.get("all")
    lines.append(
        _row(
            "all item types",
            str(all_["n"]) if all_ else MISSING,
            *(rate((all_ or {}).get(k)) for k in ("exec_t0", "agree_t0", "agree_t0_given_exec")),
        )
    )
    for _tag, label, prefixes in TYPE_GROUPS:
        types = [t for t in ITEM_TYPES if in_group(t, prefixes) and t in stats]
        cells = [rf"\quad {label}", str(sum(stats[t]["n"] for t in types)) if types else MISSING]
        for key in ("exec_t0", "agree_t0", "agree_t0_given_exec"):
            r = pooled_rate(stats, types, key)
            cells.append(_num(r) if r is not None else MISSING)
        lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def native_rows(res: dict[str, Any] | None) -> list[tuple[Native, dict[str, Any] | None]]:
    """The six expected systems (Mem0 and the five of block N), then any other system in `res`."""
    systems = (res or {}).get("systems") or {}
    extra = [Native(s, s, s) for s in sorted(systems) if s not in {n.key for n in NATIVES}]
    return [(n, systems.get(n.key)) for n in (*NATIVES, *extra)]


def inside_marks(system: dict[str, Any] | None) -> tuple[str, str]:
    """The six comparisons of one system as marks (S_src, S_def, orphan; Update then Stability)
    and their count: a filled circle inside the predicted interval, an open one outside."""
    flags = []
    for b in ("src", "def", "orphan"):
        for m in ("Update", "Stability"):
            flags.append(_get(system, "branches", b, "consistent", m))
    if not system:
        return MISSING, MISSING
    if any(f is None for f in flags):
        return "--", "--"
    marks = "".join(r"\bullet" if f else r"\circ" for f in flags)
    return f"${marks}$", f"{sum(flags)}/{len(flags)}"


def tab_native(res: dict[str, Any] | None, cost: dict[str, Any]) -> str:
    """Block N: one row per native system, with its documented and inferred (D, B) label, the six
    comparisons against the interval predicted by the white-box cell of its inferred label, three
    of the measured values and the ingestion cost per event."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Native memory systems on the short worlds. Labels $(D,B)$: documented before "
        r"any run, inferred from probes. Six comparisons (Update, Stability after $\Ssrc$, "
        r"$\Sdef$, $\Sdel$): $\bullet$ inside, $\circ$ outside the 95\% interval of the white-box "
        r"cell with the inferred label. Values: ContractValid on the measurement half (\%); cost "
        r"per event: LLM calls and prompt plus completion tokens (k).}",
        r"\label{tab:native}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}l cc cc rrr rr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{label} & \multicolumn{2}{c}{inside} & "
        r"\multicolumn{3}{c}{value (\%)} & \multicolumn{2}{c}{cost / event} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-8}\cmidrule(l){9-10}",
        r"system & doc. & inf. & six & n & $\Ssrc$ U & $\Sdef$ U & orph.\ S & calls & tok. \\",
        r"\midrule",
    ]
    for nat, system in native_rows(res):
        doc = DOCUMENTED_LABELS.get(nat.key)
        lab = _get(res, "labels", nat.key)
        inferred = _inferred(lab) if lab and lab.get("B") is not None else None
        marks, count = inside_marks(system)

        def est(b: str, m: str, system: dict[str, Any] | None = system) -> str:
            return _est(_get(system, "branches", b, "native", m))

        t = (cost.get(nat.key) or {}).get("t_i")
        tokens = None if not t else (t["prompt_tokens"] + t["completion_tokens"]) / 1000
        lines.append(
            _row(
                nat.label,
                label_text(*doc) if doc else MISSING,
                label_text(*inferred) if inferred else MISSING,
                marks,
                count,
                est("src", "Update"),
                est("def", "Update"),
                est("orphan", "Stability"),
                _num(t["llm_calls"], 1, 2) if t else MISSING,
                _num(tokens, 1, 1) if tokens is not None else MISSING,
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _slope(s: dict[str, Any] | None, unusable: bool = False) -> str:
    """Slope with its 90% bounds; grey where almost no item failed (the fit is not usable)."""
    if not s:
        return MISSING
    if math.isnan(s["est"]):  # no item failed: the fit is undefined
        return UNDEFINED
    text = interval(*(_minus(s[k]) for k in ("est", "lo90", "hi90")))
    return rf"\textcolor{{gray}}{{{text}}}" if unusable else text


def _auroc(au: float | None) -> str:
    if au is None:
        return MISSING
    return UNDEFINED if math.isnan(au) else f"{au:.2f}"


def tab_families(
    m1: dict[str, Any] | None,
    families: Mapping[str, Any],
    readers: Mapping[str, Any],
) -> str:
    """Blocks M1-M3: writer switches, other families as writer and reader, reader swaps."""
    m1 = m1 or {}
    p3 = m1.get("path3") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Other model families, \pass{} holds, \fail{} does not. (a) Writer switches "
        r"from Qwen3.5-9B: calibration slope (90\% bounds) and AUROC of writer agreement for "
        r"survival. $D$ surv.: survival of $D{=}1$ minus $D{=}0$ across the switch (pp), \pass{} "
        r"if not negative. Retention $\Delta$: retention change of \holdmem{}(1,1) (pp), \pass{} "
        r"if retention stays above the registered 97\% bar. (b) Other families as writer and "
        r"reader. The five checks are (i) leak check not triggered, (ii) $B$ effect on Update "
        r"after $\Ssrc$ above $+2$\,pp and (iii) $D$ effect on repair positive after $\Ssrc$ and "
        r"after $\Sdef$. They also include (iv) $B{\times}D$ on Update positive after $\Sdef$ and "
        r"(v) zero system below \holdmem{}(1,1) after $\Sdef$. Checks (ii)--(v) need the 95\% "
        r"interval to exclude 0. "
        r"Pattern: (ii)--(iv) all hold. (c) Other readers: contrasts that keep the main reader's "
        r"sign (of six) and the leak check for the $D$ effect on missed invalidation. The last "
        r"columns give \holdmem{}(1,1) post-$\Sdef$ ContractValid (\%) under the main and the "
        r"swapped reader. "
        r"$^\ast$~Almost no item failed, so the slope (grey) is not usable (---: undefined). The "
        r"agreement check is judged pooled.}",
        r"\label{tab:families}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{@{}l rr rc rc@{}}",
        r"\toprule",
        r"(a) writer switch & slope & AUROC & $D$ surv. & & retention $\Delta$ & \\",
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
                _slope(cal.get("slope"), unusable=key in caveats),
                _auroc(_get(cal, "auroc", "est")),
                _est(mb.get("d1_minus_d0_survival"), sign=True),
                _mark(mb.get("holds")),
                _est(t1.get("delta_90"), sign=True),
                _mark(_within(t1)),
            )
        )
    sl, au = p3.get("pooled_slope"), _get(p3, "pooled_auroc", "est")
    lines += [
        r"\midrule",
        _row(
            rf"pooled, {len(p3.get('tiers', [])) or MISSING} switches",
            _slope(sl),
            _auroc(au),
            r"\multicolumn{2}{c}{agreement check "
            + _trig(None if not p3 else {"triggered": p3["triggered"]})
            + "}",
            "",
            "",
        ),
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{@{}l cccccc c@{}}",
        r"\toprule",
        r"(b) family & (i) & (ii) & (iii) $\Ssrc$ & (iii) $\Sdef$ & (iv) & (v) & pattern \\",
        r"\midrule",
    ]
    for fam in FAMILIES:
        f = families.get(fam.key) or {}
        marks = [_mark(_get(f, "checks", key)) for key, _ in FAMILY_CHECKS]
        lines.append(_row(fam.label, *marks, _mark(f.get("pattern_holds") if f else None)))
    staged = [f for f in FAMILIES if families.get(f.key)]
    holding = (
        f"{sum(bool((families.get(f.key) or {}).get('pattern_holds')) for f in FAMILIES)} "
        f"of {len(FAMILIES)}"
        if len(staged) == len(FAMILIES)
        else MISSING
    )
    lines += [
        r"\midrule",
        rf"\multicolumn{{8}}{{@{{}}l}}{{families where (ii)--(iv) all hold: {holding}}} \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{@{}l c cc cc@{}}",
        r"\toprule",
        r"(c) reader & same sign & \multicolumn{2}{c}{leak check ($D$ on missed)} & "
        r"\multicolumn{2}{c}{\holdmem{}(1,1) post-$\Sdef$} \\",
        r"\cmidrule(lr){3-4}\cmidrule(l){5-6}",
        r"& & $\Ssrc$ & $\Sdef$ & main & swap \\",
        r"\midrule",
    ]
    for fam in FAMILIES:
        sw = readers.get(fam.key) or {}
        signs = m3_signs(sw)
        d_missed = [
            (
                replication.path2_verdict(s)
                if (s := _get(sw, b, "contrasts", "D_effect_missed_source", "swap"))
                else MISSING
            )
            for b in ("src", "def")
        ]
        cv = [
            _est(_get(sw, "def", "systems", "holdmem_D1B1_source", arm, "ContractValid_post"))
            for arm in ("main", "swap")
        ]
        lines.append(
            _row(
                fam.label,
                f"{sum(signs)}/{len(signs)}" if signs else MISSING,
                *d_missed,
                *cv,
            )
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


REND_ROWS = (  # label, check key, paired name
    (r"(i) $D$ moves missed invalidation ($\max|\cdot|$)", "i_path2_not_triggered", None),
    (r"(ii) $B$ effect on Update, $\Ssrc$", "ii_B_update_src", "B_update_src"),
    (r"(iii) $D$ effect on repair, $\Ssrc$", "iii_D_repair_src", "D_repair_src"),
    (r"(iii) $D$ effect on repair, $\Sdef$", "iii_D_repair_def", "D_repair_def"),
    (r"(iv) $B{\times}D$ on Update, $\Sdef$", "iv_BxD_def", "BxD_def"),
    (
        r"(v) zero system $-$ \holdmem{}(1,1), $\Sdef$",
        "v_zero_below_hm11_def",
        "zero_minus_hm11_def",
    ),
)


def tab_rendered(m: dict[str, Any] | None) -> str:
    """Block L: the M2 checks on LLM-rendered worlds beside the same-seed template worlds, the
    paired differences, and the rendering descriptives."""
    m = m or {}
    rend, temp = m.get("rendered") or {}, m.get("template_same_seeds") or {}
    paired = {p["name"]: p for p in m.get("paired") or []}
    lines = [
        r"\begin{table}[t]",
        r"\caption{LLM-rewritten worlds ("
        rf"{len(m.get('worlds', [])) or MISSING} worlds): the five checks of the other families "
        r"(Table~\ref{tab:families}b, effects in pp, \pass{} holds, \fail{} fails) on rewritten "
        r"and template worlds of the same seeds. Difference: rewritten minus template, paired by "
        r"item, with its 95\% interval. Check (i) is the largest absolute $D$ effect on missed "
        r"invalidation, compared with the 2\,pp of the leak check, and has no paired difference "
        rf"({UNDEFINED}).}}",
        r"\label{tab:rendered}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}l rc rc r@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{rewritten} & \multicolumn{2}{c}{template} & difference \\",
        r"\midrule",
    ]
    for label, key, name in REND_ROWS:
        if name is None:  # check (i): the largest D effect over the path-2 contrasts
            p2 = [p for p in paired.values() if p.get("check") == "i"]

            def largest(which: str, rows: list[dict[str, Any]] = p2) -> str:
                return _num(max(abs(p[which]["est"]) for p in rows)) if rows else MISSING

            cells = [label, largest("rendered"), _mark(_get(rend, "checks", key))]
            cells += [largest("template"), _mark(_get(temp, "checks", key)), UNDEFINED]
        else:
            p = paired.get(name) or {}
            cells = [
                label,
                _est(_get(rend, "estimates", name), sign=True),
                _mark(_get(rend, "checks", key)),
                _est(_get(temp, "estimates", name), sign=True),
                _mark(_get(temp, "checks", key)),
                _ci(p.get("diff"), sign=True),
            ]
        lines.append(_row(*cells))
    rd = m.get("rendering") or {}
    same = _get(m, "probe_identity", "identical")
    identical = MISSING if same is None else ("yes" if same else "no")
    tot = rd.get("total") or {}
    fid = rd.get("fidelity") or {}
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\smallskip",
        rf"{{\footnotesize\raggedright {_rewrite_note(tot)} Fidelity of the rewritten worlds "
        r"(share of gold surface forms located): "
        rf"{_num(fid.get('min_rate'))}\% at the worst world and branch. "
        rf"Probes identical to the template worlds: {identical}."
        r"\par}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _rewrite_note(tot: Mapping[str, Any]) -> str:
    """One sentence on the bodies selected for rewriting: how many, and how many needed the one
    retry or stayed template text (`rendered.json` rendering.total)."""
    if "attempted" not in tot:
        return f"Bodies selected for rewriting: {MISSING}."
    n = f"{tot['attempted']:,}"
    retried, kept = tot.get("retried_share"), tot.get("kept_template_share")
    if retried == 0 and kept == 0:
        return rf"All {n} bodies selected for rewriting were accepted on the first attempt."
    return (
        rf"Of the {n} bodies selected for rewriting, {_num(retried)}\% were retried once and "
        rf"{_num(kept)}\% kept as template."
    )


def _load(root: Path, sources: dict[str, str], group: str, name: str) -> dict[str, Any] | None:
    p = root / sources.get(group, "__missing__") / "analysis" / name
    return dict(json.loads(p.read_text())) if p.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=Path("paper/generated"))
    ap.add_argument("--only", nargs="*", default=[], help="file names (tab_gate.tex, ...)")
    a = ap.parse_args()
    sources = json.loads((a.root / "results" / "sources.json").read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    head = "% generated by memshift.analysis.paper_tables; do not edit\n"
    gate = _load(a.root, sources, "gate", "gate_a.json") or {}
    e7 = _load(a.root, sources, "e7", "e7_metrics.json") or {}
    audit = json.loads((a.root / "audit" / "system_audit.json").read_text())
    outputs = {
        "tab_gate.tex": tab_gate(gate),
        "tab_robust.tex": tab_robust(e7),
        "tab_robust_ci.tex": tab_robust(e7, with_ci=True),
        "tab_membership.tex": tab_membership(audit),
    }
    inp = load_stage4(a.root, sources)
    outputs |= {
        "tab_replication.tex": tab_replication(replication_view(inp["orig"], inp["rep"])),
        "tab_construct.tex": tab_construct(inp["c"]),
        "tab_native.tex": tab_native(native_results(inp), inp["ncost"]),
        "tab_families.tex": tab_families(inp["m1"], inp["m2"], inp["m3"]),
        "tab_rendered.tex": tab_rendered(inp["rendered"]),
    }
    # Exploratory tables (decisions.md 2026-09-30)
    from memshift.analysis import (
        ex_e1_paper,  # lazy: it imports this module's formatters
        ex_review_paper,
        ex_tables,
    )

    outputs |= ex_e1_paper.tables(ex_e1_paper.load(a.root, sources))
    outputs |= ex_review_paper.tables(ex_review_paper.load(a.root, sources))
    outputs |= ex_tables.build(a.root, sources, inp)
    from memshift.analysis import controls_paper  # lazy, as above

    outputs |= controls_paper.tables(controls_paper.load(a.root, sources))  # registered controls
    from memshift.analysis import key_audit  # lazy, as above

    outputs |= key_audit.tables(key_audit.load(a.root, sources))  # offline key audit
    from memshift.analysis import ex_round2  # lazy, as above

    outputs |= ex_round2.tables(ex_round2.load(a.root, sources))  # review round 2 (2026-10-08)
    from memshift.analysis import ex_round3  # lazy, as above

    outputs |= ex_round3.tables(ex_round3.load(a.root, sources))  # review round 3 (2026-10-16)
    from memshift.analysis import round3_paper  # lazy, as above

    outputs |= round3_paper.tables(round3_paper.load(a.root, sources))  # registered round 3
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

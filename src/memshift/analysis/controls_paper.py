"""Paper numbers and the table of the control cells (analysis in
`analysis.controls`). Every macro name starts with `Ctl`; the source is the staged analysis named
`ctl` in results/sources.json (`analysis/ctl_metrics.json`). A stage that is not staged
renders every macro and table cell as the MISSING marker.

Macro names (letters only, digits spelled as in the other blocks):

  Ctl<Contrast>               estimate and 95% interval of a contrast, `<Contrast>Est` the estimate
                              alone; the contrast keys are those of `controls.contrast_keys()`, for
                              example CtlGUpdDzeroSrcAll (Update, gold minus source, D = 0, S_src,
                              all items), CtlRRatioSrcDefinition, CtlZUpdSrcDefinition
                              The contrasts of the D0full control are CtlFRep, CtlFRepmech (D=1
                              minus D0full, S_src and S_def only, for example CtlFRepSrcDefinition)
                              and CtlFGainRep, CtlFGainUpd (the access gain); the class
                              `Untruncated` (CtlFRepSrcUntruncated) is the sensitivity analysis
                              without the truncated definitions
  CtlWorlds                   worlds
  CtlCheck..., CtlChecksPassed          the implementation checks (pass / fail / not evaluated)
  CtlProgOk..., CtlRa..., CtlOver..., CtlThink...   the quantities reported without a threshold
  CtlFull...                  the D0full notes: re-derivations, truncations and tokens given, pooled
                              (CtlFullTruncatedAll) and per branch (CtlFullTruncatedSrc), per
                              definition (CtlFullCloseFriendTruncatedSrc) and per definition and
                              declared source (CtlFullCloseFriendMessagesDroppedSrc);
                              CtlFullWhichTruncated names the definitions that were cut

tab_controls        the contrasts of the control arms of the paper (oracle, add-only, template,
                    scope, view, full access) on definition items, the other items and all items
                    for the three branches, as point estimates
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memshift.analysis import controls
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, camel, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, ci, num, pct, pp
from memshift.analysis.paper_tables import _minus, _num, _row
from memshift.gen.kappa import KAPPA_DEFS

UNDEFINED = "---"
CLASS_TAGS = (
    ("all", "All"),
    ("definition", "Definition"),
    ("rule", "Rule"),
    ("free", "Free"),
)
CHECKS = (
    ("hash", "Hash"),
    ("probe_sets", "Probes"),
    ("anchors", "Anchor"),
    ("audit", "Audit"),
    ("scope_cited", "ScopeCited"),
)
PROG_LABELS = (("all", "All"), *BRANCH_TAGS)
SD_TAGS = tuple(bt for bt in BRANCH_TAGS if bt[0] in controls.SD)  # the branches D0full ran on
FULL_LABELS = (("all", "All"), *SD_TAGS)
AUDITED = (
    ("D0B1gold", "DzeroBoneGold"),
    ("D1B1gold", "DoneBoneGold"),
    ("D1B1scope", "DoneBoneScope"),
)


def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any] | None:
    """The staged control analysis (None where it is not staged)."""
    return stage_json(root, dict(sources), "ctl", "ctl_metrics.json")


# ------------------------------------------------------------------------------------ formats
def _ok(x: Any) -> bool:
    return x is not None and x == x


def fmt(e: dict[str, Any] | None, unit: str = "pp") -> str:
    """Estimate with interval; MISSING when it is not there (not staged, or a cell missing)."""
    if not e or not _ok(e.get("est")):
        return MISSING
    return ci(e, num if unit == "ratio" else pct if unit == "pct" else pp)


def point(x: float | None, unit: str = "pct") -> str:
    """A number: a share (`pct`), a ratio, a signed pp difference or a count (`n`)."""
    if not _ok(x):
        return MISSING
    assert x is not None
    return {"ratio": num, "pct": pct, "pp": pp}.get(unit, lambda v: f"{v:,.0f}")(x)


def _word(s: str | None) -> str:
    return MISSING if s is None else s


def _pass(mark: bool | None) -> str:
    return "not evaluated" if mark is None else "pass" if mark else "fail"


class _Out(dict[str, str]):
    """Macros of this block: a name defined twice is an error."""

    def put(self, *parts: str, value: str) -> None:
        name = "".join(("Ctl", *parts))
        if name in self:
            raise ValueError(f"macro {name} defined twice")
        self[name] = value


# ------------------------------------------------------------------------------------- macros
def _contrasts(view: Mapping[str, Any] | None, out: _Out) -> None:
    for key in controls.contrast_keys():
        out.put(camel(key), value=fmt(_get(view, "contrasts", key), controls.unit_of(key)))


def _worlds(view: Mapping[str, Any] | None, out: _Out) -> None:
    out.put("Worlds", value=str(view["n_worlds"]) if view else MISSING)


def _checks(view: Mapping[str, Any] | None, out: _Out) -> None:
    checks = _get(view, "checks")
    for key, tag in CHECKS:
        node = checks.get(key) if checks else None
        out.put("Check", tag, "Pass", value=_word(_pass(node["pass"]) if node else None))
    passed = f"{checks['passed']} of {len(CHECKS)}" if checks and "passed" in checks else MISSING
    out.put("Checks", "Passed", value=passed)
    runs = (_get(checks, "anchors", "runs") or {}).values()
    anchors = [c for run in runs for c in run.values()]
    shares = [c["share"] for c in anchors if _ok(c.get("share"))]
    ctx = [c["context_share"] for c in anchors if _ok(c.get("context_share"))]
    answers = f"{sum(c['answers'] for c in anchors):,}" if anchors else MISSING
    out.put("Check", "AnchorAnswers", value=answers)
    out.put("Check", "AnchorWorst", value=pct(max(shares)) if shares else MISSING)
    out.put("Check", "AnchorContextWorst", value=pct(max(ctx)) if ctx else MISSING)
    for short, tag in AUDITED:
        cell = _get(checks, "audit", "cells", controls.CELLS[short])
        out.put("Check", "Audit", tag, value=_word(_pass(cell["pass"]) if cell else None))
    sc = _get(checks, "scope_cited") or {}
    out.put("Check", "ScopeCitedRows", value=f"{sc['rows']:,}" if "rows" in sc else MISSING)
    differ = str(sc["row_mismatches"]) if "row_mismatches" in sc else MISSING
    out.put("Check", "ScopeCitedDiffer", value=differ)


def _program_ok(rep: Any, out: _Out) -> None:
    """The share of programs that ran at t0 (deletion-aware and D_m) and their ratio."""
    for label, tag in PROG_LABELS:
        node = _get(rep, "program_t0", label) or {}
        for field, name, unit in (
            ("ra", "Ra", "pct"),
            ("d1s", "Dm", "pct"),
            ("ratio", "Ratio", "ratio"),
        ):
            out.put("ProgOk", name, tag, value=fmt(node.get(field), unit))


def _retraction_aware(rep: Any, out: _Out) -> None:
    """Lineage, failures and undeclared reads of the retraction-aware programs."""
    fields = (
        ("lineage_false", "LineageFalse"),
        ("failed", "Failed"),
        ("final_failed", "FinalFailed"),
        ("final_undeclared", "FinalUndeclared"),
        ("final_undeclared_of_failed", "FinalUndeclaredOfFailed"),
    )
    for b, bt in BRANCH_TAGS:
        for c, ct in CLASS_TAGS:
            node = _get(rep, "ra", b, c) or {}
            for field, name in fields:
                out.put("Ra", name, bt, ct, value=fmt(node.get(field), "pct") if node else MISSING)
            for field, name in (("runs", "Runs"), ("final_items", "FinalItems")):
                out.put("Ra", name, bt, ct, value=point(node.get(field), "n") if node else MISSING)


def _over_retraction(rep: Any, out: _Out) -> None:
    """Over-retraction of the task definitions (rule 3 withdraws a program that read every task)."""
    for b, bt in BRANCH_TAGS:
        for cell, ct in (("D1sB1ra", "Ra"), ("D1sB1", "Dm")):
            node = _get(rep, "over_retraction", b, cell) or {}
            for grp, gt in (("taskdef", "Task"), ("otherdef", "Other")):
                share = fmt(_get(node, "withdrawn_share", grp), "pct")
                out.put("Over", "Withdrawn", gt, ct, bt, value=share)
            for fact in controls.TASK_DEFS:
                ft = camel(fact[2:])
                f = _get(node, "task_definitions", fact) or {}
                for field, name, unit in (
                    ("repaired", "Repaired", "n"),
                    ("withdrawn", "Withdrawn", "n"),
                    ("update", "Update", "pct"),
                ):
                    out.put(
                        "Over", ft, name, ct, bt, value=point(f.get(field), unit) if f else MISSING
                    )


def _thinking(rep: Any, out: _Out) -> None:
    """The thinking writer: writes, unreadable writes and completion tokens per write."""
    out.put("Think", "Limit", value=point(_get(rep, "think", "limit_tokens"), "n"))
    for label, tag in (("all", "All"), *BRANCH_TAGS):
        node = _get(rep, "think", label) or {}
        out.put("Think", "Writes", tag, value=point(node.get("writes"), "n") if node else MISSING)
        for field, name in (("unreadable", "Unreadable"), ("at_limit", "AtLimit")):
            out.put("Think", name, tag, value=fmt(node.get(field), "pct") if node else MISSING)
        for field, name in (
            ("completion_tokens", "Tokens"),
            ("completion_tokens_readable", "TokensReadable"),
        ):
            e = node.get(field)
            out.put("Think", name, tag, value=point(e["est"], "n") if e else MISSING)
        for field, name in (
            ("completion_tokens_median", "TokensMedian"),
            ("completion_tokens_p90", "TokensNinetieth"),
        ):
            out.put("Think", name, tag, value=point(node.get(field), "n") if node else MISSING)


def _definition_names(facts: list[str]) -> str:
    """Definition ids as LaTeX text: the underscores escaped."""
    return ", ".join(f.removeprefix("k.").replace("_", r"\_") for f in facts) or "none"


def _full_access(rep: Any, out: _Out) -> None:
    """The D0full writer, from the notes of its re-derivations: how many, how many were truncated
    (and by which rule), the records left out and the tokens given, pooled and per branch; per
    definition; and per definition and declared source."""
    out.put("Full", "Limit", value=point(_get(rep, "full", "limit_tokens"), "n"))
    for label, tag in FULL_LABELS:
        node = _get(rep, "full", label) or {}
        for field, name in (
            ("rederivations", "Writes"),
            ("truncated", "Truncated"),
            ("by_source_rule", "BySource"),
            ("newest_fallback", "Newest"),
            ("dropped", "Dropped"),
            ("capped", "Capped"),
            ("tokens_median", "TokensMedian"),
            ("tokens_max", "TokensMax"),
        ):
            out.put("Full", name, tag, value=point(node.get(field), "n"))
        out.put("Full", "TruncatedShare", tag, value=fmt(node.get("truncated_share"), "pct"))
        tokens = node.get("tokens")
        out.put("Full", "Tokens", tag, value=point(tokens["est"], "n") if tokens else MISSING)
    for key, definition in KAPPA_DEFS.items():
        ft = camel(key)
        for label, tag in FULL_LABELS:
            node = _get(rep, "full", "definitions", f"k.{key}", label) or {}
            for field, name in (
                ("rederivations", "Writes"),
                ("truncated", "Truncated"),
                ("dropped", "Dropped"),
            ):
                out.put("Full", ft, name, tag, value=point(node.get(field), "n"))
            for source in definition["sources"]:
                cell = _get(node, "by_source", source) or {}
                for field, name in (("truncated", "Truncated"), ("dropped", "Dropped")):
                    out.put("Full", ft, camel(source), name, tag, value=point(cell.get(field), "n"))
    cut = _get(rep, "full", "truncated_definitions")
    out.put("Full", "WhichTruncated", value=MISSING if cut is None else _definition_names(cut))


def _reported(view: Mapping[str, Any] | None, out: _Out) -> None:
    rep = _get(view, "reported")
    _program_ok(rep, out)
    _retraction_aware(rep, out)
    _over_retraction(rep, out)
    _thinking(rep, out)
    _full_access(rep, out)


def numbers(view: Mapping[str, Any] | None) -> dict[str, str]:
    """Every `Ctl` macro of the control-cell analysis, from `load`."""
    out = _Out()
    for block in (_contrasts, _worlds, _checks, _reported):
        block(view, out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ------------------------------------------------------------------------------------- tables
GROUPS = ("definition", "other", "all")
DM = r"$D_{\mathrm{m}}$"
NOT_TAKEN = ""  # a contrast the table does not take on that item class: an empty cell
CAPTION = (
    r"Contrasts of the control cells after $\Ssrc$, $\Sdef$ and $\Sdel$ ({n} worlds, point "
    r"estimates). The full-access writer ran after $\Ssrc$ and $\Sdef$ only ({undef})."
)
COMPACT_NOTE = (
    r" The arms are those of \cref{{app:controls}}. Source, cited: the source key and the "
    r"citation key. Full access: the full-access writer. Acc.\ on re-derived: accuracy on "
    r"re-derived items. Value equals gold: the share of re-derived values that equal gold. $t_0$: "
    r"before the shift. The last two rows are ratios."
)


@dataclass(frozen=True)
class CompactRow:
    arm: str
    label: str
    cells: Mapping[str, str]  # group -> key template with {br}
    unit: str = "pp"
    source: str = "contrast"  # contrast | scope_missed
    run: tuple[str, ...] = ("src", "def", "orphan")  # the branches the row's cells were run on


SD = ("src", "def")
ROWS = (  # the control arms, each with its contrasts; the ratios last, in a block of their own
    CompactRow("oracle", r"Update, oracle $-$ source, $D{=}0$",
               {"definition": "G_upd_D0_{br}_definition", "all": "G_upd_D0_{br}_all"}),
    CompactRow("", r"\quad$D{=}1$",
               {"definition": "G_upd_D1_{br}_definition", "all": "G_upd_D1_{br}_all"}),
    CompactRow("", r"Stability, oracle $-$ source, $D{=}0$",
               {"definition": "G_stab_D0_{br}_definition", "all": "G_stab_D0_{br}_all"}),
    CompactRow("", r"\quad$D{=}1$",
               {"definition": "G_stab_D1_{br}_definition", "all": "G_stab_D1_{br}_all"}),
    CompactRow("add-only", r"Update, add-only $-$ $B{=}0$ ($D{=}1$)",
               {"definition": "R_ref_{br}_definition", "all": "R_ref_{br}_all"}),
    CompactRow("", r"\quad$B{=}1$ $-$ add-only",
               {"definition": "R_gap_{br}_definition", "all": "R_gap_{br}_all"}),
    CompactRow("template", r"Update, template $-$ $D{=}0$ ($B{=}1$)",
               {"definition": "T_upd_{br}_definition", "all": "T_upd_{br}_all"}),
    CompactRow("scope", r"Update, scope $-$ source ($D{=}1$)",
               {"definition": "K_upd_{br}_definition", "all": "K_upd_{br}_all"}),
    CompactRow("", r"Update, scope $-$ cited ($D{=}1$)",
               {"definition": "K_cited_{br}_definition", "other": "K_cited_{br}_other",
                "all": "K_cited_{br}_all"}),
    CompactRow("", r"\quad missed invalidation, scope cell (\%)",
               {"all": "scope_missed"}, "pct", "scope_missed"),
    CompactRow("view", r"Update, view $-$ no-store BM25",
               {"definition": "Z_upd_{br}_definition", "other": "Z_upd_{br}_other",
                "all": "Z_upd_{br}_all"}),
    CompactRow("", r"\quad single-field items",
               {"other": "Z_upd_{br}_rule"}),
    CompactRow("", r"\quad free-text items",
               {"other": "Z_upd_{br}_free"}),
    CompactRow("", r"Stability, view $-$ no-store BM25",
               {"definition": "Z_stab_{br}_definition", "other": "Z_stab_{br}_other",
                "all": "Z_stab_{br}_all"}),
    CompactRow("", rf"Update, view $-$ {DM}",
               {"definition": "Z_vs_D1s_{br}_definition", "all": "Z_vs_D1s_{br}_all"}),
    CompactRow("", r"Accuracy at $t_0$, view $-$ no-store BM25",
               {"definition": "Z0_upd_{br}_definition", "all": "Z0_upd_{br}_all"}),
    CompactRow("full access", r"Acc.\ on re-derived, $D{=}1$ $-$ full-access $D{=}0$",
               {"definition": "F_rep_{br}_definition", "all": "F_rep_{br}_all"}, run=SD),
    CompactRow("", r"\quad value equals gold",
               {"definition": "F_repmech_{br}_definition", "all": "F_repmech_{br}_all"},
               run=SD),
    CompactRow("add-only", r"(add-only $-$ $B{=}0$)/($B{=}1$ $-$ $B{=}0$)",
               {"definition": "R_ratio_{br}_definition"}, "ratio"),
    CompactRow("template", r"(template $-$ $D{=}0$)/($D{=}1$ $-$ $D{=}0$)",
               {"definition": "T_ratio_{br}_definition"}, "ratio"),
)  # fmt: skip


def _row_value(
    view: Mapping[str, Any] | None, row: CompactRow, group: str, br: str
) -> float | None:
    """The point value shown in one cell, or None (not staged, or the contrast is missing)."""
    if row.source == "scope_missed":
        e = _get(view, "cells", controls.CELLS["D1B1scope"], br, "all", "MissedInvalidation")
    else:
        e = _get(view, "contrasts", row.cells[group].format(br=br))
    return e["est"] if e and _ok(e.get("est")) else None


def _compact_cell(x: float | None, row: CompactRow) -> str:
    if x is None:
        return MISSING
    if row.unit == "ratio":
        return _minus(x)
    if row.unit == "pct":
        return _num(x, digits=0)
    return _num(x, sign=True)


def tab_controls(view: Mapping[str, Any] | None) -> str:
    """One row per contrast of a control arm (sub-rows for its variants), on definition items,
    the other items and all items, per branch: the arm's name on its first row, an empty cell for
    a class the contrast does not take, a dash for a branch the arm did not run on; the two ratios
    in a block of their own below the pp differences."""
    n = _get(view, "n_worlds") or MISSING
    lines = [
        r"\begin{table}[t]",
        r"\caption{" + (CAPTION + COMPACT_NOTE).format(n=n, undef=UNDEFINED) + "}",
        r"\label{tab:controls}",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}l l rrr rrr rrr@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{definition items} & \multicolumn{3}{c}{other items} "
        r"& \multicolumn{3}{c}{all items} \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(l){9-11}",
        r"arm & contrast (pp) & $\Ssrc$ & $\Sdef$ & $\Sdel$ & $\Ssrc$ & $\Sdef$ & $\Sdel$ "
        r"& $\Ssrc$ & $\Sdef$ & $\Sdel$ \\",
    ]
    blocks = (
        [row for row in ROWS if row.unit != "ratio"],
        [row for row in ROWS if row.unit == "ratio"],
    )
    for block in blocks:
        lines.append(r"\midrule")
        for k, row in enumerate(block):
            if row.arm and k > 0:
                lines.append(r"\addlinespace[2pt]")
            cells = [
                NOT_TAKEN
                if g not in row.cells
                else UNDEFINED
                if b not in row.run
                else _compact_cell(_row_value(view, row, g, b), row)
                for g in GROUPS
                for b, _ in BRANCH_TAGS
            ]
            lines.append(_row(row.arm, row.label, *cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tables(view: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        "tab_controls.tex": tab_controls(view),
    }

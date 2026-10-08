"""Paper numbers and tables of the control cells (prereg/v3_controls.md; analysis in
`analysis.controls`). Every macro name starts with `Ctl`; the source is the staged analysis named
`ctl` in paper/results_sources.json (`analysis/ctl_metrics.json`). A stage that is not staged
renders every macro and table cell as the MISSING marker; a prediction that was not judged says
so in its verdict macro.

Macro names (letters only, digits spelled as in the other blocks):

  Ctl<Contrast>               estimate and 95% interval of a contrast, `<Contrast>Est` the estimate
                              alone; the contrast keys are those of `controls.contrast_keys()`, for
                              example CtlGUpdDzeroSrcAll (Update, gold minus source, D = 0, S_src,
                              all items), CtlRRatioSrcDefinition, CtlZUpdSrcDefinition
  Ctl<Pred>Verdict            holds / fails / not judged, for the predictions G1 ... F1, with the
                              digit spelled (CtlGoneVerdict; `Z1k` is the k = 0 statement of P-Z1)
  Ctl<Pred>Hits, Total, Need, Count   met / tested / needed, and "5 of 6 contrasts"
  CtlGoneStrong, CtlTonePresentation  the two flags the registration names (yes / no)
  CtlFoneReading              the reading of the D0full control: execution / access / both / mixed
                              (`controls.Judge.f1`); its contrasts are CtlFRep, CtlFRepmech (D=1
                              minus D0full, S_src and S_def only, for example CtlFRepSrcDefinition)
                              and CtlFGainRep, CtlFGainUpd (the access gain); the class
                              `Untruncated` (CtlFRepSrcUntruncated) is the sensitivity analysis
                              without the truncated definitions
  CtlSummary<Holds|Fails|NotJudged|Total>, CtlWorlds   verdict counts, worlds
  CtlCheck..., CtlChecksPassed          the implementation checks (pass / fail / not evaluated)
  CtlProgOk..., CtlRa..., CtlOver..., CtlThink...   the quantities reported without a threshold
  CtlFull...                  the D0full notes: re-derivations, truncations and tokens given, pooled
                              (CtlFullTruncatedAll) and per branch (CtlFullTruncatedSrc), per
                              definition (CtlFullCloseFriendTruncatedSrc) and per definition and
                              declared source (CtlFullCloseFriendMessagesDroppedSrc);
                              CtlFullWhichTruncated names the definitions that were cut and
                              CtlFullAsRegistered says whether they are the registered two

tab_controls        the registered predictions, each with its contrast on definition items and on
                    all items (and on the non-definition items where the prediction says so) for
                    the three branches, and the verdict
tab_controls_full   every contrast with its interval, the reported quantities, the truncation
                    counts per definition and source, and the checks
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from memshift.analysis import controls
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, camel, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _bare, _macro, ci, num, pct, pp
from memshift.analysis.paper_tables import _minus, _num, _row, interval
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


def _tcell(e: dict[str, Any] | None, unit: str = "pp") -> str:
    """`fmt` in the interval format of the tables (`paper_tables.interval`)."""
    if not e or not _ok(e.get("est")):
        return MISSING
    if unit == "ratio":
        return interval(*(_minus(e[k]) for k in ("est", "lo", "hi")))
    f = pct if unit == "pct" else pp
    return interval(f(e["est"]), _bare(f(e["lo"])), _bare(f(e["hi"])))


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


def _verdicts(view: Mapping[str, Any] | None, out: _Out) -> None:
    for pred in controls.PREDICTIONS:
        v = _get(view, "verdicts", pred)
        tag = camel(pred)
        judged = bool(v) and v["status"] != controls.NOT_JUDGED
        status = _word(v["status"] if v else None)
        out.put(tag, "Verdict", value=status)
        for field, name in (("hits", "Hits"), ("total", "Total"), ("need", "Need")):
            out.put(tag, name, value=str(v[field]) if judged else MISSING)
        out.put(tag, "Count", value=f"{v['hits']} of {v['total']} contrasts" if judged else status)
    strong = _get(view, "verdicts", "G1", "strong")
    out.put(camel("G1"), "Strong", value=MISSING if strong is None else "yes" if strong else "no")
    dom = _get(view, "verdicts", "T1", "presentation_dominated")
    out.put(camel("T1"), "Presentation", value=MISSING if dom is None else "yes" if dom else "no")
    out.put(camel("F1"), "Reading", value=_word(_get(view, "verdicts", "F1", "reading")))
    summary = _get(view, "summary") or {}
    for key, tag in (
        (controls.HOLDS, "Holds"),
        (controls.FAILS, "Fails"),
        (controls.NOT_JUDGED, "NotJudged"),
    ):
        out.put("Summary", tag, value=str(summary[key]) if key in summary else MISSING)
    out.put("Summary", "Total", value=str(sum(summary.values())) if summary else MISSING)
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
    """P-M3: the share of programs that ran at t0 and their ratio."""
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


def _yes_no(flag: bool | None) -> str:
    return MISSING if flag is None else "yes" if flag else "no"


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
    out.put("Full", "AsRegistered", value=_yes_no(_get(rep, "full", "as_registered")))
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
    for block in (_contrasts, _verdicts, _checks, _reported):
        block(view, out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ------------------------------------------------------------------------------------- tables
PASS, FAIL = r"\pass", r"\fail"
NOT_JUDGED = "n.j."
GROUPS = ("definition", "other", "all")
DM = r"$D_{\mathrm{m}}$"
CAPTION = (
    r"Control cells (\texttt{{prereg/v3\_controls.md}}): predictions registered before running "
    r"and after the main results were known, "
    r"worlds 1000--1009 ({n} worlds), after $\Ssrc$, $\Sdef$ and on the orphan branch ($\Sdel$). "
    r"Predictions on cells that were not run were withdrawn before the first run ($D{{=}}0$ for "
    r"the add-only refresh and the scope key, $B{{=}}0$ for the template, deletion-aware and "
    r"thinking arms). The full-access control (P-F1) was run on $\Ssrc$ and $\Sdef$ only."
)
# P-Z1 at k = 0 has no ID of its own: the registration states it under P-Z1 (v3 section 3.6), and
# the analysis judges it as the separate prediction `Z1k` (`controls.PREDICTIONS`)
Z1K_NOTE = (
    r" P-Z1, $k{=}0$: the statement registered with P-Z1 for the same items before the shift "
    r"(at least $+$30\,pp; Section 3.6 of the registration), which has no ID of its own; it is "
    rf"counted as a separate prediction among the {len(controls.PREDICTIONS)}."
)
FULL_CAPTION = (
    r"Control cells (predictions registered before running, worlds 1000--1009, "
    r"{n} worlds; see Table~\ref{{tab:controls}})."
)
COMPACT_NOTE = (
    r" Arms: \emph{{oracle}}, \emph{{scope}}: the oracle key (built from the gold dependencies) "
    r"and the scope key (what each definition ranges over), against the source-epoch "
    r"(\emph{{source}}) or cited-cluster (\emph{{cited}}) key; \emph{{add-only}}: the add-only "
    r"refresh, which adds the fresh derivation without invalidating; \emph{{template}}: the "
    r"$D{{=}}0$ value in the $D{{=}}1$ template; \emph{{deletion-aware}}: model-written programs "
    r"that declare the records they read and are withdrawn when one is deleted; {dm}: $D{{=}}1$ "
    r"with a model-written derivation; \emph{{view}}: the query-time view (the zero system plus "
    r"the model-written program run on the live records); \emph{{thinking}}, "
    r"\emph{{full-access}}: a $D{{=}}0$ writer that thinks before answering, or is given every "
    r"in-scope record. Prediction IDs: P-G oracle key, P-K scope key, P-R add-only, P-T template, "
    r"P-M deletion-aware (del.-aware), P-Z view, P-D thinking, P-F full-access. "
    r"Repair: share of the re-derived items the reader answers correctly; "
    r"mechanistic repair: share whose re-derived value equals gold; $t_0$, $k{{=}}0$: before "
    r"the shift. Entries are point estimates in pp (ratios for the rows marked ratio, shares in "
    r"percent); \textbf{{bold}} entries are the ones the prediction judges; verdict: "
    r"\pass{{}}~holds, \fail{{}}~fails, n.j.~not judged, with the contrasts met of those tested, "
    r"in the order of the predictions."
)
FULL_NOTE = (
    r" Estimates in pp with 95\% intervals from the world-clustered bootstrap (2,000 draws, "
    r"seed 996)."
)


def _mark(v: Mapping[str, Any] | None) -> str:
    """The verdict mark with the contrasts met of those tested."""
    if not v:
        return MISSING
    if v["status"] == controls.NOT_JUDGED:
        return NOT_JUDGED
    mark = PASS if v["status"] == controls.HOLDS else FAIL
    return rf"{mark}\,{v['hits']}/{v['total']}"


@dataclass(frozen=True)
class CompactRow:
    pred: str
    label: str
    cells: Mapping[str, str]  # group -> key template with {br}
    judged: Mapping[str, tuple[str, ...]]  # group -> branches the prediction judges
    verdict: tuple[str, ...] = ()  # predictions whose verdict this row carries
    unit: str = "pp"
    source: str = "contrast"  # contrast | think | scope_missed | program_ratio
    run: tuple[str, ...] = ("src", "def", "orphan")  # the branches the row's cells were run on


ALLB = ("src", "def", "orphan")
SD = ("src", "def")
ROWS = (
    CompactRow("P-G1, P-G2", r"Update, oracle $-$ source, $D{=}0$",
               {"definition": "G_upd_D0_{br}_definition", "all": "G_upd_D0_{br}_all"},
               {"definition": ALLB, "all": ALLB}, ("G1", "G2")),
    CompactRow("", r"\quad$D{=}1$",
               {"definition": "G_upd_D1_{br}_definition", "all": "G_upd_D1_{br}_all"},
               {"definition": ALLB, "all": ALLB}),
    CompactRow("P-G3", r"Stability, oracle $-$ source, $D{=}0$",
               {"definition": "G_stab_D0_{br}_definition", "all": "G_stab_D0_{br}_all"},
               {"all": ALLB}, ("G3",)),
    CompactRow("", r"\quad$D{=}1$",
               {"definition": "G_stab_D1_{br}_definition", "all": "G_stab_D1_{br}_all"},
               {"all": ALLB}),
    CompactRow("P-R1", r"Update, add-only $-$ $B{=}0$ ($D{=}1$)",
               {"definition": "R_ref_{br}_definition", "all": "R_ref_{br}_all"},
               {"all": ALLB}, ("R1",)),
    CompactRow("", r"\quad$B{=}1$ $-$ add-only",
               {"definition": "R_gap_{br}_definition", "all": "R_gap_{br}_all"},
               {"all": ALLB}),
    CompactRow("P-R2", r"(add-only $-$ $B{=}0$)/($B{=}1$ $-$ $B{=}0$), ratio",
               {"definition": "R_ratio_{br}_definition"}, {"definition": SD}, ("R2",), "ratio"),
    CompactRow("P-T1, P-T3", r"Update, template $-$ $D{=}0$ ($B{=}1$)",
               {"definition": "T_upd_{br}_definition", "all": "T_upd_{br}_all"},
               {"definition": ALLB, "all": ALLB}, ("T1", "T3")),
    CompactRow("P-T2", r"(template $-$ $D{=}0$)/($D{=}1$ $-$ $D{=}0$), ratio",
               {"definition": "T_ratio_{br}_definition"}, {"definition": SD}, ("T2",), "ratio"),
    CompactRow("P-M1", r"Repair, deletion-aware $-$ $D{=}0$",
               {"definition": "M_rep_D0_{br}_definition", "all": "M_rep_D0_{br}_all"},
               {"all": ("orphan",)}, ("M1",)),
    CompactRow("", rf"\quad$-$ {DM}",
               {"definition": "M_rep_D1s_{br}_definition", "all": "M_rep_D1s_{br}_all"},
               {"all": ("orphan",)}),
    CompactRow("P-M2", rf"Update, deletion-aware $-$ {DM}",
               {"definition": "M_upd_{br}_definition", "all": "M_upd_{br}_all"},
               {"all": SD}, ("M2",)),
    CompactRow("P-M3", rf"Programs ok at $t_0$, del.-aware/{DM}, ratio",
               {"all": "program_t0"}, {"all": ALLB}, ("M3",), "ratio", "program_ratio"),
    CompactRow("P-K1", r"Update, scope $-$ source ($D{=}1$)",
               {"definition": "K_upd_{br}_definition", "all": "K_upd_{br}_all"},
               {"definition": ALLB}, ("K1",)),
    CompactRow("P-K2", r"Update, scope $-$ cited ($D{=}1$)",
               {"definition": "K_cited_{br}_definition", "other": "K_cited_{br}_other",
                "all": "K_cited_{br}_all"},
               {"other": ALLB}, ("K2",)),
    CompactRow("", r"\quad missed invalidation, scope cell (\%)",
               {"all": "scope_missed"}, {"all": ALLB}, (), "pct", "scope_missed"),
    CompactRow("P-Z1", r"Update, view $-$ zero system",
               {"definition": "Z_upd_{br}_definition", "other": "Z_upd_{br}_other",
                "all": "Z_upd_{br}_all"},
               {"definition": SD}, ("Z1",)),
    CompactRow("P-Z1, $k{=}0$", r"Accuracy at $t_0$, view $-$ zero",
               {"definition": "Z0_upd_{br}_definition", "all": "Z0_upd_{br}_all"},
               {"definition": SD}, ("Z1k",)),
    CompactRow("P-Z2", r"Update, view $-$ zero, single-field items",
               {"other": "Z_upd_{br}_rule"}, {"other": ALLB}, ("Z2",)),
    CompactRow("", r"\quad free-text items",
               {"other": "Z_upd_{br}_free"}, {"other": ALLB}),
    CompactRow("", r"\quad Stability, view $-$ zero",
               {"definition": "Z_stab_{br}_definition", "other": "Z_stab_{br}_other",
                "all": "Z_stab_{br}_all"},
               {"all": SD}),
    CompactRow("P-Z3", rf"Update, view $-$ {DM}",
               {"definition": "Z_vs_D1s_{br}_definition", "all": "Z_vs_D1s_{br}_all"},
               {"definition": SD}, ("Z3",)),
    CompactRow("P-D1", r"Repair, $D{=}1$ $-$ thinking $D{=}0$",
               {"definition": "D_rep_{br}_definition", "all": "D_rep_{br}_all"},
               {"definition": SD}, ("D1",)),
    CompactRow("", r"\quad mechanistic repair",
               {"definition": "D_repmech_{br}_definition", "all": "D_repmech_{br}_all"},
               {"definition": SD}),
    CompactRow("P-D2", r"Thinking writes unreadable (\%)",
               {"definition": "think_unreadable"}, {"definition": ALLB}, ("D2",), "pct", "think"),
    CompactRow("P-F1", r"Repair, $D{=}1$ $-$ full-access $D{=}0$",
               {"definition": "F_rep_{br}_definition", "all": "F_rep_{br}_all"},
               {"definition": SD}, ("F1",), run=SD),
    CompactRow("", r"\quad mechanistic repair",
               {"definition": "F_repmech_{br}_definition", "all": "F_repmech_{br}_all"},
               {"definition": SD}, run=SD),
)  # fmt: skip


def _row_value(
    view: Mapping[str, Any] | None, row: CompactRow, group: str, br: str
) -> float | None:
    """The point value shown in one cell, or None (not staged, or the contrast is missing)."""
    if row.source == "contrast":
        e = _get(view, "contrasts", row.cells[group].format(br=br))
    elif row.source == "program_ratio":
        e = _get(view, "reported", "program_t0", br, "ratio")
    elif row.source == "scope_missed":
        e = _get(view, "cells", controls.CELLS["D1B1scope"], br, "all", "MissedInvalidation")
    else:  # think
        e = _get(view, "reported", "think", br, "unreadable")
    return e["est"] if e and _ok(e.get("est")) else None


def _compact_cell(x: float | None, row: CompactRow, bold: bool) -> str:
    if x is None:
        return MISSING
    if row.unit == "ratio":
        text = _minus(x)
    elif row.unit == "pct":
        text = _num(x, digits=0)
    else:
        text = _num(x, sign=True)
    return rf"\textbf{{{text}}}" if bold else text


def tab_controls(view: Mapping[str, Any] | None) -> str:
    """Main text: one row per registered prediction (sub-rows for the variants of one prediction),
    definition items, non-definition items and all items per branch, and the verdict."""
    n = _get(view, "n_worlds") or MISSING
    lines = [
        r"\begin{table}[t]",
        r"\caption{" + (CAPTION + COMPACT_NOTE).format(n=n, dm=DM) + Z1K_NOTE + "}",
        r"\label{tab:controls}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}l l ccc ccc ccc r@{}}",
        r"\toprule",
        r"& & \multicolumn{3}{c}{definition items} & \multicolumn{3}{c}{non-definition} "
        r"& \multicolumn{3}{c}{all items} & \\",
        r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
        r"Pred. & contrast & $\Ssrc$ & $\Sdef$ & $\Sdel$ & $\Ssrc$ & $\Sdef$ & $\Sdel$ "
        r"& $\Ssrc$ & $\Sdef$ & $\Sdel$ & verdict \\",
        r"\midrule",
    ]
    for row in ROWS:
        cells = [
            UNDEFINED
            if g not in row.cells or b not in row.run
            else _compact_cell(_row_value(view, row, g, b), row, b in row.judged.get(g, ()))
            for g in GROUPS
            for b, _ in BRANCH_TAGS
        ]
        marks = [_mark(_get(view, "verdicts", p)) for p in row.verdict]
        lines.append(_row(row.pred, row.label, *cells, ", ".join(marks)))
    checks = _get(view, "checks")
    done = MISSING
    if checks and "passed" in checks:
        done = rf"{FAIL if checks['failed'] else PASS}\,{checks['passed']}/{len(CHECKS)}"
    lines += [
        r"\midrule",
        _row("", "Implementation checks", *[""] * 9, done),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------- the full table
TAG_LABELS = {
    "G_upd_D0": r"Update, oracle $-$ source, $D{=}0$",
    "G_upd_D1": r"Update, oracle $-$ source, $D{=}1$",
    "G_stab_D0": r"Stability, oracle $-$ source, $D{=}0$",
    "G_stab_D1": r"Stability, oracle $-$ source, $D{=}1$",
    "K_upd": r"Update, scope $-$ source, $D{=}1$",
    "K_cited": r"Update, scope $-$ cited, $D{=}1$",
    "R_ref": r"Update, add-only $-$ $B{=}0$, $D{=}1$",
    "R_b1": r"Update, $B{=}1$ $-$ $B{=}0$, $D{=}1$",
    "R_gap": r"Update, $B{=}1$ $-$ add-only, $D{=}1$",
    "R_ratio": r"(add-only $-$ $B{=}0$)/($B{=}1$ $-$ $B{=}0$)",
    "T_upd": r"Update, template $-$ $D{=}0$, $B{=}1$",
    "T_den": r"Update, $D{=}1$ $-$ $D{=}0$, $B{=}1$",
    "T_ratio": r"(template $-$ $D{=}0$)/($D{=}1$ $-$ $D{=}0$)",
    "M_rep_D0": r"Repair, deletion-aware $-$ $D{=}0$",
    "M_rep_D1s": rf"Repair, deletion-aware $-$ {DM}",
    "M_upd": rf"Update, deletion-aware $-$ {DM}",
    "O_upd": rf"Update, deletion-aware $-$ {DM}",
    "O_rep": rf"Repair, deletion-aware $-$ {DM}",
    "Z_upd": r"Update, view $-$ zero",
    "Z_stab": r"Stability, view $-$ zero",
    "Z_vs_D1s": rf"Update, view $-$ {DM}",
    "Z0_upd": r"Accuracy at $t_0$, view $-$ zero",
    "Z0_vs_D1s": rf"Accuracy at $t_0$, view $-$ {DM}",
    "D_rep": r"Repair, $D{=}1$ $-$ thinking $D{=}0$",
    "D_repmech": r"Mechanistic repair, $D{=}1$ $-$ thinking $D{=}0$",
    "D_gain_upd": r"Update, thinking $-$ $D{=}0$",
    "D_gain_rep": r"Repair, thinking $-$ $D{=}0$",
    "D_gain_repmech": r"Mechanistic repair, thinking $-$ $D{=}0$",
    "F_rep": r"Repair, $D{=}1$ $-$ full-access $D{=}0$",
    "F_repmech": r"Mechanistic repair, $D{=}1$ $-$ full-access $D{=}0$",
    "F_gain_rep": r"Repair, full-access $-$ $D{=}0$ (access gain)",
    "F_gain_upd": r"Update, full-access $-$ $D{=}0$ (access gain)",
}
CLASS_LABELS = {
    "all": "all items",
    "definition": "definition",
    "rule": "single-field items",
    "free": "free text",
    "other": "non-definition",
    "taskdef": "task definitions",
    "otherdef": "other definitions",
    "untruncated": "definition, truncated left out",
}
FULL_SECTIONS = (
    ("Oracle and scope keys (P-G1 to P-G3, P-K1, P-K2)",
     ("G_upd_D0", "G_upd_D1", "G_stab_D0", "G_stab_D1", "K_upd", "K_cited")),
    ("Add-only refresh and template presentation (P-R1, P-R2, P-T1 to P-T3)",
     ("R_ref", "R_b1", "R_gap", "R_ratio", "T_upd", "T_den", "T_ratio")),
    ("Deletion-aware programs and the thinking writer (P-M1, P-M2, P-D1, thinking gain)",
     ("M_rep_D0", "M_rep_D1s", "M_upd", "O_upd", "O_rep", "D_rep", "D_repmech",
      "D_gain_upd", "D_gain_rep", "D_gain_repmech")),
    ("Query-time view (P-Z1 to P-Z3)",
     ("Z_upd", "Z_stab", "Z_vs_D1s", "Z0_upd", "Z0_vs_D1s")),
    ("Full-access writer (P-F1, the access gain, and the sensitivity analysis without the "
     "truncated definitions close\\_friend and vip\\_sender; run on $\\Ssrc$ and $\\Sdef$ only)",
     ("F_rep", "F_repmech", "F_gain_rep", "F_gain_upd")),
)  # fmt: skip


def _cell(e: dict[str, Any] | None, unit: str = "pp") -> str:
    """`est [lo, hi]` (`paper_tables.interval`): pp with estimate and bounds signed, a ratio with
    two decimals."""
    if not e:
        return MISSING
    if not _ok(e.get("est")):
        return UNDEFINED
    if unit == "ratio":
        return interval(*(_minus(e[k]) for k in ("est", "lo", "hi")))
    return interval(*(_num(e[k], sign=True) for k in ("est", "lo", "hi")))


def _full_table(
    title: str,
    header: str,
    rows: list[str],
    label: str,
    view: Mapping[str, Any] | None,
    cols: str,
    note: str = FULL_NOTE,
) -> str:
    n = _get(view, "n_worlds") or MISSING
    caption = f"{title}. " + FULL_CAPTION.format(n=n) + note
    return "\n".join(
        [
            r"\begin{table}[t]",
            r"\caption{" + caption + "}",
            rf"\label{{tab:{label}}}",
            r"\centering\scriptsize",
            r"\setlength{\tabcolsep}{3pt}",
            rf"\begin{{tabular}}{{@{{}}{cols}@{{}}}}",
            r"\toprule",
            header,
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def _family_rows(view: Mapping[str, Any] | None, tags: tuple[str, ...]) -> list[str]:
    classes, ran = controls.families(), controls.family_branches()
    rows: list[str] = []
    for tag in tags:
        unit = "ratio" if tag.endswith("ratio") else "pp"
        for k, cls in enumerate(classes[tag]):
            cells = [
                _cell(_get(view, "contrasts", f"{tag}_{b}_{cls}"), unit)
                if b in ran[tag]
                else UNDEFINED  # a branch the contrast is not computed for
                for b, _ in BRANCH_TAGS
            ]
            rows.append(_row(TAG_LABELS[tag] if k == 0 else "", CLASS_LABELS[cls], *cells))
        rows.append(r"\addlinespace[2pt]")
    return rows


def tab_controls_full(view: Mapping[str, Any] | None) -> str:
    """Appendix: every contrast with its interval, by prediction family (one table each), then the
    pooled estimates and the quantities reported without a threshold, the truncation counts of the
    D0full cell per definition and source, and the implementation checks."""
    header = r"contrast & items & $\Ssrc$ & $\Sdef$ & $\Sdel$ \\"
    tables = [
        _full_table(
            title, header, _family_rows(view, tags), f"controls_full_{'abcde'[k]}", view, "l l ccc"
        )
        for k, (title, tags) in enumerate(FULL_SECTIONS)
    ]
    tables += [_reported_table(view), _truncation_table(view), _checks_table(view)]
    return "\n\n".join(tables) + "\n"


def _reported_table(view: Mapping[str, Any] | None) -> str:
    """Pooled contrasts and the quantities reported without a threshold."""
    rep = _get(view, "reported")
    rows: list[str] = []
    pooled = (
        ("G_pool_all", r"Update, gold $-$ source, pooled over $D$ and branch, all items"),
        ("G_pool_definition", r"Update, gold $-$ source, pooled, definition items"),
        ("Z_pool_definition", r"Update, view $-$ zero, $\Ssrc$ and $\Sdef$ pooled, definition"),
    )
    for key, label in pooled:
        rows.append(_row(label, _cell(_get(view, "contrasts", key))))
    rows.append(r"\addlinespace[2pt]")
    prog = _get(rep, "program_t0", "all") or {}
    rows += [
        _row(r"programs ok at $t_0$, deletion-aware", _tcell(prog.get("ra"), "pct")),
        _row(rf"programs ok at $t_0$, {DM}", _tcell(prog.get("d1s"), "pct")),
        _row("ratio (P-M3, at least 0.8)", _tcell(prog.get("ratio"), "ratio")),
        r"\addlinespace[2pt]",
    ]
    think = _get(rep, "think", "all") or {}
    tokens = think.get("completion_tokens")
    median = think.get("completion_tokens_median")
    p90 = think.get("completion_tokens_p90")
    rows += [
        _row(
            r"thinking writes unreadable (P-D2, at most 20\%)",
            _tcell(think.get("unreadable"), "pct"),
        ),
        _row("thinking writes at the token limit", _tcell(think.get("at_limit"), "pct")),
        _row(
            "completion tokens per thinking write, mean",
            point(tokens["est"], "n") if tokens else MISSING,
        ),
        _row(
            "median and 90th percentile",
            f"{point(median, 'n')}, {point(p90, 'n')}" if think else MISSING,
        ),
        r"\addlinespace[2pt]",
    ]
    full = _get(rep, "full", "all") or {}
    cap = point(_get(rep, "full", "limit_tokens"), "n")
    cut = _get(rep, "full", "truncated_definitions")
    registered = _yes_no(_get(rep, "full", "as_registered"))
    given = f"{point(full.get('tokens_median'), 'n')}, {point(full.get('tokens_max'), 'n')}"
    rows += [
        _row(
            "full-access writes (definition items re-derived after the shift)",
            point(full.get("rederivations"), "n"),
        ),
        _row(
            "of which truncated (records of a declared source left out)",
            point(full.get("truncated"), "n"),
        ),
        _row("share of the writes truncated", _tcell(full.get("truncated_share"), "pct")),
        _row("records left out, in total", point(full.get("dropped"), "n")),
        _row(f"approximate tokens given, median and maximum (cap {cap})", given),
        _row(
            "the definitions truncated are the two registered",
            registered + (f" ({_definition_names(cut)})" if cut else ""),
        ),
        r"\addlinespace[2pt]",
    ]
    for b, bt in BRANCH_TAGS:
        ra = _get(rep, "ra", b, "all") or {}
        rows.append(
            _row(
                f"{bt}: runs with a value although a prior read was gone",
                _tcell(ra.get("lineage_false"), "pct"),
            )
        )
        rows.append(
            _row(
                f"{bt}: final items failed by an undeclared read",
                _tcell(ra.get("final_undeclared"), "pct"),
            )
        )
    rows.append(r"\addlinespace[2pt]")
    for cell, ct in (("D1sB1ra", "deletion-aware"), ("D1sB1", DM)):
        share = _get(rep, "over_retraction", "orphan", cell, "withdrawn_share", "taskdef")
        rows.append(_row(rf"$\Sdel$: task definitions withdrawn, {ct}", _tcell(share, "pct")))
    return _full_table(
        "Reported without a threshold",
        r"quantity & all worlds \\",
        rows,
        "controls_reported",
        view,
        "l r",
    )


def _truncation_table(view: Mapping[str, Any] | None) -> str:
    """The D0full cell's truncations per definition and declared source, from the notes of its
    items: the re-derivations of the definition, those in which records of the source were left
    out, and the records left out (S_src and S_def)."""
    rep = _get(view, "reported")
    rows: list[str] = []
    for key, definition in KAPPA_DEFS.items():
        for k, source in enumerate(definition["sources"]):
            cells: list[str] = []
            for b, _ in SD_TAGS:
                node = _get(rep, "full", "definitions", f"k.{key}", b, "by_source", source) or {}
                cells += [
                    point(node.get(f), "n") for f in ("rederivations", "truncated", "dropped")
                ]
            rows.append(_row(key.replace("_", r"\_") if k == 0 else "", source, *cells))
        rows.append(r"\addlinespace[2pt]")
    header = "\n".join(
        [
            r"definition & source & \multicolumn{3}{c}{$\Ssrc$} & \multicolumn{3}{c}{$\Sdef$} \\",
            r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
            r"& & writes & cut & left out & writes & cut & left out \\",
        ]
    )
    note = (
        r" Counts over the re-derivations of the definition items after the shift, from the notes "
        r"of the full-access cell: writes, writes in which records of the source were left out "
        r"(cut), and the records left out in total."
    )
    return _full_table(
        "Truncation of the full-access writer's records per definition and source",
        header,
        rows,
        "controls_truncation",
        view,
        "l l rrr rrr",
        note,
    )


def _checks_table(view: Mapping[str, Any] | None) -> str:
    """The implementation checks of the stage, each marked pass or fail."""
    checks = _get(view, "checks") or {}
    rows: list[str] = []

    def line(label: str, node: Mapping[str, Any] | None, detail: str = "") -> None:
        result = MISSING if node is None else _pass(node.get("pass"))
        rows.append(_row(label, f"{detail} {result}".strip()))

    line("pre-shift hash assertions of the new runs", checks.get("hash"))
    line("probe sets and gold equal the published ones", checks.get("probe_sets"))
    anchors = [c for run in (_get(checks, "anchors", "runs") or {}).values() for c in run.values()]
    worst = max((c["share"] for c in anchors), default=None)
    detail = f"worst cell {pct(worst)} of answers differ," if worst is not None else ""
    line("anchor answers equal the published ones", checks.get("anchors"), detail)
    for system, cell in sorted((_get(checks, "audit", "cells") or {}).items()):
        groups = [g for b in cell["branches"].values() for g in b.values()]
        detail = f"{sum(g['pass'] for g in groups)}/{len(groups)} groups,"
        name = system.removeprefix("holdmem_").replace("_", r"\_")
        line(f"missed and false invalidation of {name} equal the audit", cell, detail)
    sc = checks.get("scope_cited")
    detail = f"{sc['row_mismatches']} of {sc['rows']:,} rows differ," if sc and "rows" in sc else ""
    line("scope cell on non-definition items equals the cited cell", sc, detail)
    rows = rows if checks else [_row("implementation checks", MISSING)]
    return _full_table(
        "Implementation checks (marked pass or fail)",
        r"check & result \\",
        rows,
        "controls_checks",
        view,
        "l r",
    )


def tables(view: Mapping[str, Any] | None) -> dict[str, str]:
    return {
        "tab_controls.tex": tab_controls(view),
        "tab_controls_full.tex": tab_controls_full(view),
    }

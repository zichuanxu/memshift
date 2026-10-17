"""Appendix tables for the exploratory analyses of the review response (decisions.md 2026-09-30),
regenerated from the committed run analyses like the other generated tables:

tab_writer_predictors   writer-switch calibration: the registered agreement rate against baselines,
                        pooled and per switch, with world and type-level intervals (E2, block M1)
tab_tests_sensitivity   T1 (net change, eps grid, by item class), T3 by item class, and which tests
                        hold by construction (E2, E3, block R)
tab_native_null         native systems against label-free baselines: containing references, counts
                        by circularity, chance coverage, B identification, paired differences (E5)
A stage that has not been staged renders as MISSING in every cell that needs it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from memshift.analysis.ex_numbers import EPS_PP, REFERENCE_LABELS, sn
from memshift.analysis.paper_data import HORIZON_TAGS, MISSING, NATIVES, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import CELL_NAMES, CLASS_NAMES
from memshift.analysis.paper_tables import _ci, _mark, _num, _row, interval

PREDICTOR_ROWS = (
    ("agree", r"(a) agreement (registered)"),
    ("new_acc", r"(b) new-writer accuracy"),
    ("old_acc", r"(c) old-writer accuracy"),
    ("cond", r"(d) P(new ok $\mid$ old ok)"),
    ("nondef", r"(e) not a definition item"),
)
NODEF_ROWS = (
    ("agree", r"(f) agreement"),
    ("new_acc", r"(b$'$) new-writer accuracy"),
    ("old_acc", r"(c$'$) old-writer accuracy"),
    ("cond", r"(d$'$) P(new ok $\mid$ old ok)"),
)
SCOPE_LABELS = {"same": "same tier", "upgrade": "upgrade (27B)", "downgrade": "downgrade"}
CELL_LABELS = {
    "holdmem_D1B1_source": r"\holdmem{}(1,1)",
    "holdmem_D0B1_source": r"\holdmem{}(0,1)",
    "holdmem_D1B0": r"\holdmem{}(1,0)",
    "holdmem_D0B0": r"\holdmem{}(0,0)",
}


def _rc(e: dict[str, Any] | None, lo: str, hi: str, digits: int = 2, signed: bool = False) -> str:
    """`est` and, in brackets, the interval between the keys `lo` and `hi`, on the raw scale."""
    if not e or e.get("est") is None:
        return MISSING
    return interval(*(sn(e.get(k), digits, signed) for k in ("est", lo, hi)))


def _interval(stat: dict[str, Any] | None, kind: str, level: str, digits: int, signed: bool) -> str:
    """Only the bracketed interval of a `{est, world: {...}, type: {...}}` stat."""
    if not stat or not stat.get(kind):
        return MISSING
    e = stat[kind]
    return (
        f"{{\\scriptsize [{sn(e.get(f'lo{level}'), digits, signed)},\\,"
        f"{sn(e.get(f'hi{level}'), digits, signed)}]}}"
    )


# ------------------------------------------------------------------------- writer predictors
def _predictor_rows(
    sc: dict[str, Any] | None, rows: tuple[tuple[str, str], ...], indent: str = r"\quad "
) -> list[str]:
    out = []
    for pred, label in rows:
        e = _get(sc or {}, "predictors", pred) or {}
        slope, auroc, diff = e.get("slope"), e.get("auroc"), e.get("auroc_minus_agree")
        est = lambda s, d, sg=False: MISSING if not s else sn(s["est"], d, sg)  # noqa: E731
        out.append(
            _row(
                indent + label,
                est(slope, 2),
                _interval(slope, "world", "90", 2, False),
                _interval(slope, "type", "90", 2, False),
                est(auroc, 3),
                _interval(auroc, "world", "95", 3, False),
                _interval(auroc, "type", "95", 3, False),
                est(diff, 3, True) if pred != "agree" else "",
                _interval(diff, "world", "95", 3, True) if pred != "agree" else "",
                _interval(diff, "type", "95", 3, True) if pred != "agree" else "",
            )
        )
    return out


def _header(title: str, sc: dict[str, Any] | None) -> str:
    n, f = (sc or {}).get("n_items"), (sc or {}).get("n_failures")
    counts = MISSING if n is None else f"{n:,} items, {f} failures"
    return rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{title}; {counts}}}}} \\"


def tab_writer_predictors(e2: dict[str, Any] | None, m1: dict[str, Any] | None) -> str:
    """Predictors (a)-(f) of survival across a writer switch, pooled and per switch."""
    w = (e2 or {}).get("explore_writer_predictors") or {}
    w1 = (m1 or {}).get("explore_writer_predictors") or {}
    sets, sets1 = w.get("sets") or {}, w1.get("sets") or {}
    lines = [
        r"\begin{table}[t]",
        r"\caption{Writer-switch calibration against baselines (exploratory, not pre-registered). "
        r"Items: $D{=}0$, $B{=}1$ items that were valid before the switch. Each predictor is a "
        r"per-type value from the disjoint calibration worlds: (a) the registered agreement rate "
        r"of the old and the new writer; (b), (c) the new and the old writer's accuracy; "
        r"(d) P(new correct given old correct); (e) 1 outside the definition class, 0 inside it; "
        r"primed rows and (f): the same on the items outside the definition class only. "
        r"Slope: calibration slope of logit P(valid after) on the logit of the predictor, 90\% "
        r"interval; AUROC, 95\% interval; $\Delta$AUROC: predictor minus (a), 95\%. Intervals: "
        r"W, worlds resampled (as registered); T, item types resampled. A slope beyond $\pm20$ "
        r"means the fit separates. Switches other than the three same-lineage ones: block M1.}",
        r"\label{tab:writer-predictors}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.2pt}",
        r"\begin{tabular}{@{}l r ll r ll r ll@{}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{slope} & \multicolumn{3}{c}{AUROC} "
        r"& \multicolumn{3}{c}{$\Delta$AUROC} \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-10}",
        r"predictor & est. & W & T & est. & W & T & est. & W & T \\",
    ]
    blocks: list[tuple[str, dict[str, Any] | None, tuple[tuple[str, str], ...]]] = [
        ("pooled, three switches", _get(sets, "all", "pooled"), PREDICTOR_ROWS),
        *[
            (SCOPE_LABELS[k], _get(sets, "all", k), PREDICTOR_ROWS)
            for k in ("same", "upgrade", "downgrade")
        ],
        ("pooled, outside the definition class", _get(sets, "nodef", "pooled"), NODEF_ROWS),
        (
            "block M1, pooled over its switches",
            _get(sets1, "all", "pooled"),
            PREDICTOR_ROWS[:3],
        ),
    ]
    for title, sc, rows in blocks:
        lines += [r"\midrule", _header(title, sc), *_predictor_rows(sc, rows)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------ tests sensitivity
def _eps_marks(grid: dict[str, Any] | None, field: str) -> list[str]:
    return [_mark(_get(grid or {}, "by_eps_pp", str(e), field)) for e in EPS_PP]


def _t1_cell_rows(e2: dict[str, Any]) -> list[str]:
    tiers = ("same", "upgrade", "downgrade")
    lines = []
    for cell in CELL_LABELS:
        entries = [_get(e2, "explore_survival_recovery", f"{t}:{cell}") for t in tiers]
        label = CELL_LABELS.get(cell, cell)
        if all(e and e.get("trivial") for e in entries):
            lines.append(
                rf"\multicolumn{{13}}{{@{{}}l}}{{{label}: no item changes in any tier "
                r"(0 gains, 0 losses); every $\varepsilon$ holds by construction} \\"
            )
            continue
        lines.append(rf"\multicolumn{{13}}{{@{{}}l}}{{\emph{{{label}}}}} \\")
        for t, sr in zip(tiers, entries, strict=True):
            grid = _get(e2, "explore_t1_eps", f"{t}:{cell}")
            cells = [
                rf"\quad {SCOPE_LABELS[t]}",
                str((sr or {}).get("gains", MISSING)),
                str((sr or {}).get("losses", MISSING)),
                _ci(_get(grid, "retention_delta_90"), sign=True),
                _ci(_get(grid, "net_90"), sign=True),
                *_eps_marks(grid, "retention_noninferior"),
                *_eps_marks(grid, "net_equivalent"),
            ]
            lines.append(_row(*cells))
    return lines


# display names of the item classes (`paper_numbers.CLASS_NAMES` tags): the structured-field class
# is "field", as in the paper's class tables; "rule" names the benchmark's D=1 extractors
CLASS_DISPLAY = {"Rule": "field", "Definition": "definition", "Freetext": "free text"}


def _class_rows(e2: dict[str, Any]) -> list[str]:
    lines = []
    for t in ("same", "upgrade", "downgrade"):
        for cls, cname in CLASS_NAMES:
            c = _get(e2, "explore_t1_by_class", f"{t}:holdmem_D1B1_source", cls) or {}
            share, tol = c.get("share_of_valid_before"), c.get("tolerated_loss_at_margin")
            lines.append(
                _row(
                    rf"{SCOPE_LABELS[t]}, {CLASS_DISPLAY[cname]}",
                    _num(share, 100, 1) if share is not None else MISSING,
                    _ci(c.get("retention_delta_90"), sign=True),
                    _ci(c.get("net_90"), sign=True),
                    _num(tol, 100, 0) if tol is not None else MISSING,
                    _mark(c.get("equivalent")) if c else MISSING,
                )
            )
    return lines


SYSTEM_COLUMNS = (
    ("holdmem_D1B1_source", r"\holdmem{}(1,1)"),
    ("rag", "RAG"),
    ("zero", "zero system"),
)


def _t3_rows(e3: dict[str, Any] | None) -> list[str]:
    lines = []
    for h, _ in HORIZON_TAGS:
        per = _get(e3 or {}, "horizons", h) or {}
        for cls, cname in CLASS_NAMES:
            cells = [rf"{h} mo., {CLASS_DISPLAY[cname]}"]
            cells.append(_ci(_get(per, "oracle", "explore_t3_by_class", cls, "contract_valid")))
            for s, _label in SYSTEM_COLUMNS:
                e = _get(per, s, "explore_t3_by_class", cls) or {}
                mark = _mark(e.get("T3_pass")) if e else MISSING
                cells.append(f"{_ci(e.get('minus_oracle'), sign=True)}\\,{mark}" if e else MISSING)
            lines.append(_row(*cells))
    return lines


def _by_construction(e2: dict[str, Any], e3: dict[str, Any] | None) -> list[tuple[str, str, str]]:
    """(test, holds by construction for, evidence) from the data where the data decide it."""
    tiers = ("same", "upgrade", "downgrade")
    b0 = [c for c, _ in CELL_NAMES if c.endswith("B0")]
    flat = [_get(e2, "explore_survival_recovery", f"{t}:{c}", "trivial") for t in tiers for c in b0]
    n0 = (
        f"{sum(f is True for f in flat)} of {len(flat)} B=0 cell-tiers without a changed item"
        if all(f is not None for f in flat)
        else MISSING
    )
    free = _get(e2, "explore_t1_by_class", "same:holdmem_D1B1_source", "norule") or {}
    tol = free.get("tolerated_loss_at_margin")
    share = free.get("share_of_valid_before")
    t1 = (
        rf"only free text depends on the writer ({_num(share, 100, 1)}\% of valid-before items, "
        rf"same tier); the margin tolerates a {_num(tol, 100, 0)}\% loss there"
        if tol is not None and share is not None
        else MISSING
    )
    slacks = [
        s
        for h, _ in HORIZON_TAGS
        for sys_ in ("holdmem_D1B1_source", "rag", "zero")
        if (s := _get(e3 or {}, "horizons", h, sys_, "explore_t4_slack")) is not None
    ]
    t4 = (
        f"serving cost at least {min(slacks):.0f} times below the ceiling for every system and "
        "horizon"
        if slacks
        else MISSING
    )
    contrib = [
        _get(
            e3 or {},
            "horizons",
            h,
            "holdmem_D1B1_source",
            "explore_t3_by_class",
            "definition",
            "contribution",
        )
        for h, _ in HORIZON_TAGS
    ]
    t3 = (
        "the definition class carries the margin: its contribution to HoldMem(1,1)'s margin is "
        + ", ".join(f"${100 * c:+.1f}$" for c in contrib)
        + r"\,pp at 1, 3, 12 months"
        if all(c is not None for c in contrib)
        else MISSING
    )
    return [
        ("T1", r"\holdmem{}(0,0), (1,0): no re-derivation", n0),
        ("T1", r"\holdmem{}(1,1), (0,1): not by construction", t1),
        ("T2", "two implementations of one data model (E6)", "round trip; not computed here"),
        ("T3", "not by construction", t3),
        ("T4", "every retrieval-based system", t4),
    ]


def tab_tests_sensitivity(
    e2: dict[str, Any] | None, e3: dict[str, Any] | None, rep_e3: dict[str, Any] | None
) -> str:
    """Two floats: T1 per cell and by item class (E2), then T3 by item class (E3, block R) and the
    tests that hold by construction."""
    e2 = e2 or {}
    eps = ", ".join(str(e) for e in EPS_PP)
    n = _get(e2, "explore_survival_recovery", "same:holdmem_D1B1_source", "n_items") or MISSING
    marks = " & ".join(str(e) for e in EPS_PP)
    lines = [
        r"\begin{table}[t]",
        r"\caption{T1 sensitivity (exploratory, not pre-registered; no registered verdict "
        r"changes). (a) Per cell and switch: items gained (wrong before, right after) and lost "
        rf"among the {n} items, the retention change ($\Delta$ = retention $-1$) and the net "
        r"change in ContractValid (valid after $-$ valid before, all items) with 90\% "
        r"world-bootstrap intervals in pp, and the verdict at "
        rf"$\varepsilon = {eps}$\,pp (the four marks of each group): retention non-inferior "
        r"(lower bound above $-\varepsilon$; for retention $\le 1$ this is the registered TOST) "
        r"and net change equivalent (inside $\pm\varepsilon$). (b) T1 by item class for "
        r"\holdmem{}(1,1): share of the valid-before items, retention and net change, the loss "
        r"within the class that the 3\,pp margin tolerates if nothing else is lost, and the "
        r"registered 3\,pp verdict on the class alone.}",
        r"\label{tab:tests-sensitivity}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
        r"\begin{tabular}{@{}l rr ll cccc cccc@{}}",
        r"\toprule",
        r"(a) & \multicolumn{2}{c}{items} & \multicolumn{2}{c}{change in pp (90\%)} "
        r"& \multicolumn{4}{c}{retention} & \multicolumn{4}{c}{net} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-9}\cmidrule(l){10-13}",
        rf"cell and switch & gained & lost & retention $\Delta$ & net & {marks} & {marks} \\",
        r"\midrule",
        *_t1_cell_rows(e2),
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{@{}l r ll r c@{}}",
        r"\toprule",
        r"(b) \holdmem{}(1,1) & share (\%) & retention $\Delta$ & net & tolerated (\%) & 3\,pp \\",
        r"\midrule",
        *_class_rows(e2),
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
        r"\begin{table}[t]",
        r"\caption{T3 by item class and the tests that hold by construction (exploratory, not "
        r"pre-registered). (c), (d) Margin against the oracle context per item class at each "
        r"horizon (pp, 95\% interval, mark at the registered 3\,pp) and the oracle's own "
        r"ContractValid (\%), on the main-study test worlds and on the replication worlds. "
        r"(e) Where a test cannot fail for a system, and the evidence.}",
        r"\label{tab:tests-sensitivity-t3}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2.4pt}",
    ]
    for label, per in (("(c) main study", e3), ("(d) replication", rep_e3)):
        lines += [
            r"\begin{tabular}{@{}l l lll@{}}",
            r"\toprule",
            rf"{label} & oracle & "
            + " & ".join(rf"{name} $-$ oracle" for _, name in SYSTEM_COLUMNS)
            + r" \\",
            r"\midrule",
            *_t3_rows(per),
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\medskip",
        ]
    lines += [
        r"\begin{tabular}{@{}l p{0.27\textwidth} p{0.5\textwidth}@{}}",
        r"\toprule",
        r"(e) test & by construction for & evidence \\",
        r"\midrule",
        *[_row(*r) for r in _by_construction(e2, e3)],
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


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
# the out-of-sample header on two lines ("out-of-" / "sample"), so the column is as wide as its
# counts and the table fits a 5.5in text width
OOS = r"\parbox[t]{21pt}{\centering out-of-sample}"
# the reference labels of `ex_numbers` with the gold-dependency context named as in the paper
TABLE_REFERENCES = {**REFERENCE_LABELS, "oracle": "gold-dep."}


def _coverage(x: float | None) -> str:
    return MISSING if x is None else _num(x, 100, 0)


def tab_native_null(res: dict[str, Any] | None) -> str:
    """Native systems against the null baselines, from `explore_native_null` (Mem0 from stage 3,
    all six systems once block N is staged), with the sums over the six systems."""
    res = res or {}
    ex = res.get("explore_native_null") or {}
    systems = ex.get("systems") or {}
    lines = [
        r"\begin{table}[tp]",  # taller than \topfraction: may need a page of its own
        r"\caption{Native systems against label-free baselines (exploratory). (a) Label: the "
        r"inferred $(D,B)$. $P$: deletion persistence, the share of deleted items still answered, "
        r"with its 95\% interval (id.\ \pass{}: it excludes the $B$ threshold 0.5). Counts as "
        r"inside, discriminating\,/\,total. Inside: inside the 95\% interval of the cell "
        r"with the inferred label. Discriminating: inside, and outside that of the cell "
        r"with the other $B$. Counts are over all six comparisons, the label-circular ones "
        r"($\Sdel$, from which $B$ is inferred) and the out-of-sample ones ($\Ssrc$, $\Sdef$). "
        r"The zero system and plain RAG are counted against the same intervals. Chance: share of "
        r"the label cells containing the native estimate (label permuted). Observed: share inside. "
        r"Last row: sums (chance: means). (b) References whose 95\% interval contains the native "
        r"estimate ((d,b): white-box cell, c: cited-cluster key, gold-dep.: gold-dependency "
        r"context) and item-paired differences in pp. Inflated: interval widened by the "
        r"repeat-ingestion variance. $^\circ$: label-circular comparison.}",
        r"\label{tab:native-null}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l c lc ccc cc cc cc@{}}",
        r"\toprule",
        r"& & \multicolumn{2}{c}{$P$ (\%)} & \multicolumn{3}{c}{native} "
        r"& \multicolumn{2}{c}{zero system} & \multicolumn{2}{c}{plain RAG} "
        r"& \multicolumn{2}{c}{chance (\%)} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}"
        r"\cmidrule(l){12-13}",
        rf"system & label & est. & id. & all six & circular & {OOS} & all six & {OOS} & all six "
        rf"& {OOS} & labels & observed \\",
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
                _ci({"est": b["rate"], "lo": b["lo"], "hi": b["hi"]}) if b else MISSING,
                _mark(b.get("identified")) if b else MISSING,
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
        r"\par\medskip",
        r"\begin{tabular}{@{}l r l ll l@{}}",
        r"\toprule",
        r"(b) comparison & native & contained by & $-$ predicted & inflated & $-$ zero \\",
        r"\midrule",
    ]
    for nat in NATIVES:
        e = systems.get(nat.key)
        if not e:
            lines.append(rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{nat.label}}}: {MISSING}}} \\")
            continue
        cell = e.get("predicted_by")
        lines.append(
            rf"\multicolumn{{6}}{{@{{}}l}}{{\emph{{{nat.label}}}, predicted by "
            rf"{TABLE_REFERENCES.get(cell or '', MISSING)}}} \\"
        )
        for key, c in (e.get("comparisons") or {}).items():
            paired = (e.get("paired_difference") or {}).get(key) or {}
            p = paired.get(cell) if cell else None
            infl = (
                _ci({"est": p["est"], "lo": p["lo_inflated"], "hi": p["hi_inflated"]}, sign=True)
                if p and "lo_inflated" in p
                else MISSING
            )
            held = ", ".join(TABLE_REFERENCES.get(n, n) for n in c["contained_by"]) or "none"
            circ = r"$^\circ$" if c["label_circular"] else ""
            lines.append(
                _row(
                    rf"\quad {_comparison_label(key)}{circ}",
                    _num(c["native"]["est"]),
                    held,
                    _ci(p, sign=True) if p else MISSING,
                    infl,
                    _ci(paired.get("zero"), sign=True) if paired.get("zero") else "--",
                )
            )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _comparison_label(key: str) -> str:
    branch, metric = key.split(":")
    name = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}[branch]
    return f"{name} {metric}"


def build(root: Path, sources: dict[str, str], inp: dict[str, Any]) -> dict[str, str]:
    """File name -> table text for the three appendix tables."""
    e2 = stage_json(root, sources, "e2", "e2_metrics.json")
    e3 = stage_json(root, sources, "e3", "e3_metrics.json")
    return {
        "tab_writer_predictors.tex": tab_writer_predictors(e2, inp["m1"]),
        "tab_tests_sensitivity.tex": tab_tests_sensitivity(e2, e3, inp["rep"]["e3"]),
        "tab_native_null.tex": tab_native_null(inp["n"] or inp["n3"]),
    }

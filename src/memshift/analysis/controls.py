"""Analysis of the control cells registered on 2026-09-30 (prereg/v3_controls.md; the dated rows of
prereg/decisions.md, the last of which, "WS2 execution scope", reduces what is run).

Exploratory status: these are registered post-review controls, not confirmatory analyses of prereg
v1 or v2. The macro prefix is `Ctl`; every table says that the predictions were registered before
running. The predictions, thresholds and the common analysis are exactly those of the registration;
a prediction that refers to a cell that was not run is withdrawn in the decision row, and is judged
here only on the contrasts that remain.

Staging (one command; run directory names are the ones the compute host wrote):

    python -m memshift.analysis.stage ctl runs/stage5_controls \\
        ctl:runs/e1_ctrl_s1000-1001_<ts> [ctl:runs/e1_ctrl_s1002-1003_<ts> ...] \\
        ctl:runs/e1_ra_s1000-1001_<ts> ... ctl:runs/e1_zerodm_s1000-1001_<ts> ... \\
        ctl:runs/e1_think_s1000-1001_<ts> ... \\
        ctl:runs/e1_full_s1000-1003_<ts> ctl:runs/e1_full_s1004-1007_<ts> \\
        ctl:runs/e1_full_s1008-1009_<ts> \\
        base:runs/stage2_e1 base:runs/stage4_c_e1

The four new runs (worlds 1000 to 1009, branches src/def/orphan, `--discriminative --no-baselines`):
cells `D0B1_source` (anchor), `D0B1_gold`, `D1B1_gold`, `D1B1_scope`, `D1B0_refresh`,
`D0tB1_source`; cells `D1sB1_source` (anchor), `D1sB1_source_ra`; the view `zero_Dm`; cells
`D0B1_source` (anchor), `D0thinkB1_source`. The D0full control (prereg v3 section 6) is three more
runs, `e1_full_s1000-1003`, `e1_full_s1004-1007` and `e1_full_s1008-1009` (worlds 1000 to 1009,
branches src and def only), with the cells `D0B1_source` (anchor) and `D0fullB1_source`; a run of
two branches is merged beside the runs of three (`controls_data`). A `ctl:` directory is a new run
with its raw JSONL (split runs of a four-world chunk each are listed one by one); a `base:`
directory is a published run of the same worlds with its raw JSONL, or a stage directory, which
expands to the source runs its manifest lists. The comparison cells missing from the new runs
(`D0B0`, `D1B0`, `D0B1_source`, `D1B1_source`, `D1B1_cited`, `D1sB1_source`, and the zero system)
come from the published runs (`D1B1_source`, the comparison cell of P-F1, from the main-study E1
runs), paired by (world, branch, fact id, checkpoint k); `analysis/ctl_metrics.json`,
`analysis/ctl_item_table.jsonl` (not committed) and `manifest.json` are written to the stage
directory. `python -m memshift.analysis.paper_numbers` and `paper_tables` then render the `Ctl`
macros and `tab_controls.tex` / `tab_controls_full.tex` (every macro is MISSING until this stage
exists).

What the stage does, in order (`controls_data` holds the first two):

  1. integrity checks that stop the stage when they fail: the pre-shift hash assertions of the new
     runs, equality of the probe sets (every system of a new run was asked what the published anchor
     was asked, and the gold of every probe is the published gold), and equality of the anchors'
     answers (a share above `controls_data.ANCHOR_TOLERANCE` of differing answers is an error);
  2. one item table over the new and the published cells (`e1.item_table`, with the gold status and
     the withdrawn-value flag added);
  3. the implementation checks that are marked pass or fail: the missed and false invalidation (and
     the re-derivations) of the gold cells, and of the scope cells on definition items, equal the
     offline audit (`e1_keys.audit`, counts and rates); the scope cell's non-definition items are
     the cited cell's;
  4. every registered contrast, as a per-item paired difference by class and branch with the
     world-clustered bootstrap of `scoring.stats` (2,000 draws, seed 996), then the verdict of each
     prediction (holds / fails / not judged, with the reason);
  5. the quantities reported without a threshold.

Readings of the registration that this module fixes (each point is one the text leaves open; every
threshold comparison is listed in `verdicts[...]["tests"]` with the value it was made on):

  * P-G1 "holds strongly": the 95% lower bound of the estimate pooled over the six (D, branch)
    contrasts on all items is above -4 pp.
  * P-R1 is judged at D = 1 on both inequalities per branch (refresh - B0 >= -1 pp and
    B1 - refresh >= -1 pp, all items), two of three branches; P-R2 is not judged when the
    denominator of either branch is below 5 pp.
  * P-T1 and P-T3 are judged at B = 1, three contrasts each; P-T2 is not judged when the D effect it
    divides by is not positive; `presentation_dominated` flags a gain of 10 pp on definition items.
  * P-M3 is the ratio of the share of retraction-aware programs that ran at t0 to that of the D1s
    programs (all branches, paired by item); P-M1 and P-M2 are not judged unless it holds.
  * P-K2: the non-definition items are the rule and free-text items; the scope cell's missed rate
    is read per branch over all items, each above 5%.
  * P-Z1 also needs the lower bound of the S_src and S_def definition items pooled above 20 pp; its
    k = 0 statement is the separate prediction `Z1k`; the free-text clause of P-Z2 (at most +3 pp)
    is read in all three branches.
  * P-D2: a thinking write is unreadable when its note has `ok` false (cut off by the token limit
    or not parseable); the share that used the whole budget is reported beside it.
  * P-F1 is judged on four point estimates, repair and mechanical repair in S_src and S_def, each
    at least 30 pp (the mechanical variant "the same way"), on the definition items. The readings
    of the registration are applied to the same four gaps: a gap of at least 30 pp reads
    "execution", one of at most 15 pp "access", one in between "both" (inclusive at both bounds).
    The reading of the control is the common one when the four agree ("execution" is the P-F1
    verdict holds) and "mixed" when they do not; the registration does not say what to do with gaps
    that disagree across branches or across the two variants, so none of them is merged into a
    reading that one of the four gaps contradicts. Reported without a threshold: the access gain
    (D0full minus D0, repair and Update), and every F contrast once more on the definition items
    without `TRUNCATED_DEFS` (`untruncated`; the two definitions the offline audit found cut), no
    verdict and no reading; and, from the notes of the D0full cell, the truncation counts per
    definition and source (`full_stats`), with whether the definitions that were cut are the two
    registered ones.
  * The error text of a failed program is kept only in `det.reason` of an item's final state, so
    undeclared-read failures are counted there (items whose last program run failed); the
    `lineage_ok` share is over all post-shift runs of the retraction-aware programs.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from typing import Any

from memshift.analysis import controls_data as cd
from memshift.analysis import e1, e1_keys
from memshift.analysis.e1_synth import _ratio_of_sums
from memshift.gen.kappa import KAPPA_DEFS
from memshift.holdmem.writer import FULL_BUDGET, THINK_MAX_TOKENS
from memshift.scoring.stats import Estimate, item_diffs, world_bootstrap

Row = e1.Row
Vec = dict[str, dict[str, float]]
STATUS = (
    "exploratory: registered post-review controls (prereg/v3_controls.md), predictions registered "
    "before running; not part of the confirmatory analyses of prereg v1 and v2"
)
BRANCHES = ("src", "def", "orphan")
SD = ("src", "def")  # the branches of the predictions that leave the orphan branch out
EPS = 1e-12
DRAWS = 2000  # bootstrap draws and seed of the registered analysis (`scoring.stats`)
SEED = 996

# short name -> system name in the runs
CELLS = {
    "D0B0": "holdmem_D0B0",
    "D1B0": "holdmem_D1B0",
    "D0B1": "holdmem_D0B1_source",
    "D1B1": "holdmem_D1B1_source",
    "D1B1cited": "holdmem_D1B1_cited",
    "D1sB1": "holdmem_D1sB1_source",
    "D0B1gold": "holdmem_D0B1_gold",
    "D1B1gold": "holdmem_D1B1_gold",
    "D0B1scope": "holdmem_D0B1_scope",
    "D1B1scope": "holdmem_D1B1_scope",
    "D1B0refresh": "holdmem_D1B0_refresh",
    "D0tB1": "holdmem_D0tB1_source",
    "D1sB1ra": "holdmem_D1sB1_source_ra",
    "D0thinkB1": "holdmem_D0thinkB1_source",
    "D0fullB1": "holdmem_D0fullB1_source",
    "zero": "zero",
    "zeroDm": "zero_Dm",
}
TASK_DEFS = tuple(f"k.{k}" for k, v in KAPPA_DEFS.items() if "tasks" in v["sources"])
# the definitions whose declared sources do not fit the D0full cap, found by the offline audit
# before the first run (decisions.md 2026-10-01): the sensitivity analysis leaves them out, and
# `full_stats` reports whether the notes of the run say the same
TRUNCATED_DEFS = ("k.close_friend", "k.vip_sender")


_class_of = cache(e1_keys.item_class)


def _in_classes(*names: str) -> Callable[[str], bool]:
    return lambda fact: _class_of(fact) in names


# item classes as predicates on a fact id: the classes of `e1.ITEM_CLASSES` (definition `k.`, rule
# = structured fields, free = `norule`), their union outside the definitions (`other`) and all; the
# task definitions (`active_project`, `overdue_task`, which a program must read every task for) and
# the other definitions, for the over-retraction of rule 3; the definition items without the ones
# D0full truncates (the sensitivity analysis of P-F1)
CLASSES: dict[str, Callable[[str], bool]] = {
    "all": lambda fact: True,
    "definition": _in_classes("definition"),
    "rule": _in_classes("rule"),
    "free": _in_classes("norule"),
    "other": _in_classes("rule", "norule"),
    "taskdef": lambda fact: fact in TASK_DEFS,
    "otherdef": lambda fact: fact.startswith("k.") and fact not in TASK_DEFS,
    "untruncated": lambda fact: _class_of(fact) == "definition" and fact not in TRUNCATED_DEFS,
}
METRICS: dict[str, tuple[str, Callable[[Row], bool]]] = {
    "update": ("cv", lambda r: bool(r["changed"])),
    "stability": ("cv", lambda r: not r["changed"]),
    "repair": ("cv", lambda r: bool(r.get("repaired_now"))),
    "repair_mech": ("repair_mech", lambda r: bool(r.get("repaired_now"))),
}


# ------------------------------------------------------------------------------ the contrasts
@dataclass(frozen=True)
class Spec:
    """A per-item paired difference `a - b` of a metric over one class and branch. `acc0` is the
    accuracy before the shift (k = 0), for the items asked at t0."""

    key: str
    a: str
    b: str
    branch: str
    metric: str
    cls: str

    @property
    def tag(self) -> str:
        """The key without its branch and class (the contrast family)."""
        return self.key.removesuffix(f"_{self.branch}_{self.cls}")


DA = ("all", "definition")
RC = ("all", "definition", "rule", "free")
FC = ("all", "definition", "untruncated")


def specs() -> list[Spec]:
    """Every contrast of the registration that the executed cells allow, in one place: the analysis
    computes exactly these and the paper generators name their macros after them."""
    out: list[Spec] = []

    def add(
        tag: str,
        a: str,
        b: str,
        metric: str,
        classes: Sequence[str],
        branches: Sequence[str] = BRANCHES,
    ) -> None:
        for br in branches:
            for cls in classes:
                out.append(Spec(f"{tag}_{br}_{cls}", a, b, br, metric, cls))

    for d in (0, 1):  # P-G1 (all items), P-G2 (definition items), P-G3 (stability)
        add(f"G_upd_D{d}", f"D{d}B1gold", f"D{d}B1", "update", DA)
        add(f"G_stab_D{d}", f"D{d}B1gold", f"D{d}B1", "stability", DA)
    add("R_ref", "D1B0refresh", "D1B0", "update", RC)  # P-R1, P-R2, refresh on orphan
    add("R_b1", "D1B1", "D1B0", "update", RC)
    add("R_gap", "D1B1", "D1B0refresh", "update", RC)
    add("T_upd", "D0tB1", "D0B1", "update", DA)  # P-T1, P-T3 (B = 1 only)
    add("T_den", "D1B1", "D0B1", "update", DA)  # P-T2 denominator
    add("M_rep_D0", "D1sB1ra", "D0B1", "repair", DA)  # P-M1 (a)
    add("M_rep_D1s", "D1sB1ra", "D1sB1", "repair", DA)  # P-M1 (b)
    add("M_upd", "D1sB1ra", "D1sB1", "update", DA)  # P-M2
    add("O_upd", "D1sB1ra", "D1sB1", "update", ("taskdef", "otherdef"))  # over-retraction
    add("O_rep", "D1sB1ra", "D1sB1", "repair", ("taskdef", "otherdef"))
    add("K_upd", "D1B1scope", "D1B1", "update", DA)  # P-K1 (D = 1 only)
    add("K_cited", "D1B1scope", "D1B1cited", "update", ("definition", "other", "all"))  # P-K2
    add("Z_upd", "zeroDm", "zero", "update", ("all", "definition", "rule", "free", "other"))
    add("Z_stab", "zeroDm", "zero", "stability", ("all", "definition", "other"))  # P-Z1, P-Z2
    add("Z_vs_D1s", "zeroDm", "D1sB1", "update", DA)  # P-Z3
    add("Z0_upd", "zeroDm", "zero", "acc0", ("all", "definition", "rule", "free"))  # k = 0
    add("Z0_vs_D1s", "zeroDm", "D1sB1", "acc0", DA)
    add("D_rep", "D1B1", "D0thinkB1", "repair", DA)  # P-D1
    add("D_repmech", "D1B1", "D0thinkB1", "repair_mech", DA)
    add("D_gain_upd", "D0thinkB1", "D0B1", "update", DA)  # the thinking gain, no threshold
    add("D_gain_rep", "D0thinkB1", "D0B1", "repair", DA)
    add("D_gain_repmech", "D0thinkB1", "D0B1", "repair_mech", DA)
    add("F_rep", "D1B1", "D0fullB1", "repair", FC, SD)  # P-F1 (S_src and S_def only)
    add("F_repmech", "D1B1", "D0fullB1", "repair_mech", FC, SD)
    add("F_gain_rep", "D0fullB1", "D0B1", "repair", FC, SD)  # the access gain, no threshold
    add("F_gain_upd", "D0fullB1", "D0B1", "update", FC, SD)
    return out


# ratio of pooled means of two contrasts: key, numerator, denominator (definition items)
RATIOS = tuple(
    (f"{tag}_ratio_{br}_definition", f"{num}_{br}_definition", f"{den}_{br}_definition")
    for tag, num, den in (("R", "R_ref", "R_b1"), ("T", "T_upd", "T_den"))
    for br in BRANCHES
)
# pooled over several contrasts (items of all of them, clustered by world)
POOLED = {
    "G_pool_all": [f"G_upd_D{d}_{br}_all" for d in (0, 1) for br in BRANCHES],
    "G_pool_definition": [f"G_upd_D{d}_{br}_definition" for d in (0, 1) for br in BRANCHES],
    "Z_pool_definition": ["Z_upd_src_definition", "Z_upd_def_definition"],
}


def contrast_keys() -> list[str]:
    return [s.key for s in specs()] + [k for k, _, _ in RATIOS] + list(POOLED)


def families() -> dict[str, list[str]]:
    """Contrast family (the key without branch and class) -> the classes it is computed for."""
    out: dict[str, list[str]] = {}
    for s in specs():
        if s.cls not in out.setdefault(s.tag, []):
            out[s.tag].append(s.cls)
    for key, _, _ in RATIOS:
        out[key.rsplit("_", 2)[0]] = ["definition"]
    return out


def family_branches() -> dict[str, tuple[str, ...]]:
    """Contrast family -> the branches it is computed for (the orphan branch is left out of the
    D0full families: the control was run on S_src and S_def only)."""
    out: dict[str, tuple[str, ...]] = {}
    for s in specs():
        out[s.tag] = tuple(b for b in BRANCHES if b in (*out.get(s.tag, ()), s.branch))
    for key, _, _ in RATIOS:
        out[key.rsplit("_", 2)[0]] = BRANCHES
    return out


def unit_of(key: str) -> str:
    return "ratio" if "_ratio_" in key else "pp"


def ratio_estimate(num: Vec, den: Vec, draws: int = DRAWS) -> Estimate:
    """Ratio of pooled paired means over the items present in both, world-clustered bootstrap:
    `e1_synth._share` with the number of draws as a parameter (the default is its estimator)."""
    by_world = {
        w: [x for k in sorted(set(num[w]) & set(den[w])) for x in (num[w][k], den[w][k])]
        for w in num
        if w in den
    }
    est = world_bootstrap(by_world, stat=_ratio_of_sums, draws=draws, seed=SEED)
    return replace(est, n_items=est.n_items // 2)  # each item was counted once per component


class Engine:
    """The contrasts of one item table, computed once each."""

    def __init__(self, table: cd.Table, draws: int = DRAWS):
        self.table = table
        self.draws = draws
        self.cache: dict[tuple[str, str, str, str], Vec] = {}
        self.by_sb: dict[tuple[str, str], list[Row]] = defaultdict(list)
        for r in table.rows:
            self.by_sb[r["system"], r["branch"]].append(r)
        self.contrasts: dict[str, Estimate] = {}
        self.diffs: dict[str, Vec] = {}

    def boot(self, by_world: dict[str, list[float]]) -> Estimate:
        return world_bootstrap(by_world, draws=self.draws, seed=SEED)

    def vec(self, cell: str, branch: str, metric: str, cls: str) -> Vec:
        key = (cell, branch, metric, cls)
        if key not in self.cache:
            self.cache[key] = self._vec(cell, branch, metric, cls)
        return self.cache[key]

    def _vec(self, cell: str, branch: str, metric: str, cls: str) -> Vec:
        system = CELLS[cell]
        pred = CLASSES[cls]
        if metric == "acc0":  # accuracy at t0 of the items asked then
            src = self.table.k0.get(system, {}).get(branch, {})
            return {w: {f"{f}@0": v for f, v in d.items() if pred(f)} for w, d in src.items()}
        field, cond = METRICS[metric]
        rows = self.by_sb.get((system, branch), [])
        return e1._vec(rows, system, branch, field, lambda r: cond(r) and pred(r["fact_id"]))

    def diff(self, spec: Spec) -> Estimate:
        a = self.vec(spec.a, spec.branch, spec.metric, spec.cls)
        b = self.vec(spec.b, spec.branch, spec.metric, spec.cls)
        diffs = item_diffs(a, b)
        self.diffs[spec.key] = diffs
        est = self.boot({w: list(d.values()) for w, d in diffs.items()})
        self.contrasts[spec.key] = est
        return est

    def pooled(self, key: str, parts: list[str]) -> Estimate:
        worlds = sorted({w for p in parts for w in self.diffs[p]})
        by_world = {
            w: [v for p in parts for v in self.diffs[p].get(w, {}).values()] for w in worlds
        }
        self.contrasts[key] = est = self.boot(by_world)
        return est

    def ratio(self, key: str, num: str, den: str) -> Estimate:
        self.contrasts[key] = est = ratio_estimate(self.diffs[num], self.diffs[den], self.draws)
        return est

    def compute_all(self) -> dict[str, Estimate]:
        for s in specs():
            self.diff(s)
        for key, num, den in RATIOS:
            self.ratio(key, num, den)
        for key, parts in POOLED.items():
            self.pooled(key, parts)
        return self.contrasts

    # ----------------------------------------------------------------------------- cell level
    def cell_rates(self) -> dict[str, Any]:
        """Per system, branch and class: Update and Stability on the asked probes and, for cells
        with items, the missed and false invalidation over all items (as `e1.cell_metrics`)."""
        out: dict[str, Any] = {}
        for system in sorted({s for s, _ in self.by_sb}):
            for br in BRANCHES:
                rs = self.by_sb.get((system, br), [])
                if not any(r["cv"] is not None for r in rs):
                    continue  # not run on this branch (the D0full cell has no orphan branch)
                detects = any("detected" in r for r in rs)
                for cls in ("all", "definition", "other"):
                    est = _cell_estimates(rs, CLASSES[cls], cls != "other", detects, self.draws)
                    out.setdefault(system, {}).setdefault(br, {})[cls] = {
                        k: v.to_json() for k, v in est.items()
                    }
        return out


def _cell_estimates(
    rs: list[Row], pred: Callable[[str], bool], accuracy: bool, detects: bool, draws: int
) -> dict[str, Estimate]:
    def rate(field: str, changed: bool) -> Estimate:
        def cond(r: Row) -> bool:
            return bool(r["changed"]) == changed and pred(r["fact_id"])

        return world_bootstrap(e1._by_world(rs, field, cond), draws=draws, seed=SEED)

    est: dict[str, Estimate] = {}
    if accuracy:
        est["Update"] = rate("cv", True)
        est["Stability"] = rate("cv", False)
    if detects:
        est["MissedInvalidation"] = e1._complement(rate("detected", True))
        est["FalseInvalidation"] = rate("detected", False)
    return est


def present(e: Estimate | None) -> float | None:
    """The estimate of a contrast, or None when the contrast could not be computed (a cell or a
    world missing)."""
    return None if e is None or e.n_worlds == 0 or math.isnan(e.est) else e.est


# ------------------------------------------------------------------------------------ verdicts
HOLDS, FAILS, NOT_JUDGED = "holds", "fails", "not judged"


def pp_text(x: float | None, unit: str = "pp") -> str:
    """A value as the reasons print it: a difference in pp, a ratio, or a share."""
    if x is None:
        return "missing"
    return (
        f"{x:.2f}"
        if unit == "ratio"
        else f"{100 * x:.1f}%"
        if unit == "share"
        else f"{100 * x:+.1f} pp"
    )


@dataclass
class Test:
    """One threshold comparison of a prediction: the contrast (or rate), its point value and
    whether it meets the threshold (None when the value is missing)."""

    key: str
    value: float | None
    ok: bool | None
    threshold: str
    unit: str = "pp"  # pp (a difference), ratio or share

    def to_json(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "ok": self.ok,
            "threshold": self.threshold,
            "unit": self.unit,
        }


def _test(
    key: str, value: float | None, check: Callable[[float], bool], threshold: str, unit: str = "pp"
) -> Test:
    return Test(key, value, None if value is None else bool(check(value)), threshold, unit)


def _at_least(key: str, value: float | None, bound: float, unit: str = "pp") -> Test:
    return _test(key, value, lambda v: v >= bound - EPS, f">= {_bound(bound, unit)}", unit)


def _at_most(key: str, value: float | None, bound: float, unit: str = "pp") -> Test:
    return _test(key, value, lambda v: v <= bound + EPS, f"<= {_bound(bound, unit)}", unit)


def _within(key: str, value: float | None, bound: float) -> Test:
    return _test(key, value, lambda v: abs(v) <= bound + EPS, f"|.| <= {_bound(bound, 'pp')}")


def _bound(bound: float, unit: str) -> str:
    return (
        f"{100 * bound:g} pp"
        if unit == "pp"
        else f"{100 * bound:g}%"
        if unit == "share"
        else f"{bound:g}"
    )


def verdict(tests: list[Test], need: int | None = None, **extra: Any) -> dict[str, Any]:
    """holds when at least `need` (default: all) of the tests are met; not judged when any value
    is missing; else fails."""
    need = len(tests) if need is None else need
    missing = [t.key for t in tests if t.ok is None]
    hits = sum(bool(t.ok) for t in tests)
    if missing:
        status, reason = NOT_JUDGED, "missing: " + ", ".join(missing[:4])
    elif hits >= need:
        status, reason = HOLDS, f"{hits} of {len(tests)} met"
    else:
        bad = ", ".join(f"{t.key} {pp_text(t.value, t.unit)}" for t in tests if not t.ok)
        status, reason = FAILS, f"{hits} of {len(tests)} met (need {need}); not met: {bad}"
    return {
        "status": status,
        "reason": reason,
        "hits": hits,
        "total": len(tests),
        "need": need,
        "tests": [t.to_json() for t in tests],
        **extra,
    }


def not_judged(reason: str, total: int = 0) -> dict[str, Any]:
    return {"status": NOT_JUDGED, "reason": reason, "hits": 0, "total": total, "need": total,
            "tests": []}  # fmt: skip


# the readings registered with P-F1 (prereg/v3_controls.md section 6): the gap between D=1 and the
# full-access D=0 cell is execution at 30 pp or more, access at 15 pp or less, both in between
EXECUTION_GAP = 0.30
ACCESS_GAP = 0.15


def gap_region(gap: float | None) -> str | None:
    """Which reading one P-F1 gap supports: execution, access or both (None when missing)."""
    if gap is None:
        return None
    return (
        "execution"
        if gap >= EXECUTION_GAP - EPS
        else "access"
        if gap <= ACCESS_GAP + EPS
        else "both"
    )


class Judge:
    """The verdicts of the registered predictions, from the contrasts and what is reported."""

    def __init__(
        self, contrasts: dict[str, Estimate], reported: dict[str, Any], rates: dict[str, Any]
    ):
        self.c = contrasts
        self.reported = reported
        self.rates = rates

    def est(self, key: str) -> float | None:
        return present(self.c.get(key))

    def lo(self, key: str) -> float | None:
        e = self.c.get(key)
        return None if present(e) is None or e is None or math.isnan(e.lo) else e.lo

    def each(self, keys: Sequence[str], make: Callable[[str, float | None], Test]) -> list[Test]:
        return [make(k, self.est(k)) for k in keys]

    # 3.1 oracle key
    def g1(self) -> dict[str, Any]:
        keys = [f"G_upd_D{d}_{br}_all" for d in (0, 1) for br in BRANCHES]
        tests = self.each(keys, lambda k, v: _at_least(k, v, -0.02))
        pooled = self.lo("G_pool_all")
        strong = None if pooled is None else pooled > -0.04
        return verdict(tests, strong=strong)

    def g2(self) -> dict[str, Any]:
        keys = [f"G_upd_D{d}_{br}_definition" for d in (0, 1) for br in BRANCHES]
        return verdict(self.each(keys, lambda k, v: _within(k, v, 0.03)))

    def g3(self) -> dict[str, Any]:
        keys = [f"G_stab_D{d}_{br}_all" for d in (0, 1) for br in BRANCHES]
        return verdict(self.each(keys, lambda k, v: _at_least(k, v, 0.0)), need=5)

    # 3.2 refresh without invalidation (D = 1 only: D0B0_refresh was not run)
    def r1(self) -> dict[str, Any]:
        tests: list[Test] = []
        for br in BRANCHES:  # Update(B0) - 0.01 <= Update(refresh) <= Update(B1) + 0.01
            lower, upper = self.est(f"R_ref_{br}_all"), self.est(f"R_gap_{br}_all")
            ok = (
                None
                if lower is None or upper is None
                else (lower >= -0.01 - EPS and upper >= -0.01 - EPS)
            )
            tests.append(
                Test(
                    f"R_ref/R_gap_{br}_all",
                    lower,
                    ok,
                    "refresh - B0 >= -1 pp and B1 - refresh >= -1 pp",
                )
            )
        return verdict(tests, need=2)

    def r2(self) -> dict[str, Any]:
        tests: list[Test] = []
        for br in ("src", "def"):
            den = self.est(f"R_b1_{br}_definition")
            if den is not None and den < 0.05:
                return not_judged(f"denominator {pp_text(den)} in {br} is below 5 pp", 2)
            ratio = self.est(f"R_ratio_{br}_definition")
            tests.append(_at_least(f"R_ratio_{br}_definition", ratio, 0.5, "ratio"))
        return verdict(tests)

    # 3.3 template presentation (B = 1 only: D0tB0 was not run)
    def t1(self) -> dict[str, Any]:
        keys = [f"T_upd_{br}_definition" for br in BRANCHES]
        big = [k for k in keys if (v := self.est(k)) is not None and v >= 0.10]
        return verdict(
            self.each(keys, lambda k, v: _within(k, v, 0.03)), presentation_dominated=bool(big)
        )

    def t2(self) -> dict[str, Any]:
        tests: list[Test] = []
        for br in ("src", "def"):
            den = self.est(f"T_den_{br}_definition")
            if den is not None and den <= 0:
                return not_judged(f"the D effect {pp_text(den)} in {br} is not positive", 2)
            tests.append(
                _at_most(
                    f"T_ratio_{br}_definition", self.est(f"T_ratio_{br}_definition"), 0.15, "ratio"
                )
            )
        return verdict(tests)

    def t3(self) -> dict[str, Any]:
        keys = [f"T_upd_{br}_all" for br in BRANCHES]
        return verdict(self.each(keys, lambda k, v: _within(k, v, 0.03)))

    # 3.4 retraction-aware programs (B = 1 only)
    def m3(self) -> dict[str, Any]:
        r = (self.reported.get("program_t0") or {}).get("all") or {}
        ratio = r.get("ratio")
        value = None if not ratio or math.isnan(ratio["est"]) else ratio["est"]
        return verdict(
            [_at_least("program_ok_t0_ratio", value, 0.8, "ratio")],
            share_ra=r.get("ra"),
            share_d1s=r.get("d1s"),
        )

    @staticmethod
    def _validity(label: str, valid: dict[str, Any]) -> dict[str, Any] | None:
        """A validity condition that does not hold makes the predictions that depend on it not
        judged (prereg/v3_controls.md P-M3, P-D2)."""
        if valid["status"] == HOLDS:
            return None
        return not_judged(f"{label} (validity) is {valid['status']}: {valid['reason']}")

    def m1(self, m3: dict[str, Any]) -> dict[str, Any]:
        if (stop := self._validity("P-M3", m3)) is not None:
            return stop
        a = _at_least("M_rep_D0_orphan_all", self.est("M_rep_D0_orphan_all"), 0.0)
        lo = self.lo("M_rep_D1s_orphan_all")
        b = _test("M_rep_D1s_orphan_all (95% lower bound)", lo, lambda v: v > 0, "> 0 pp")
        return verdict([a, b])

    def m2(self, m3: dict[str, Any]) -> dict[str, Any]:
        if (stop := self._validity("P-M3", m3)) is not None:
            return stop
        keys = [f"M_upd_{br}_all" for br in ("src", "def")]
        return verdict(self.each(keys, lambda k, v: _at_least(k, v, -0.03)))

    # 3.5 scope-aware key (D = 1 only)
    def k1(self) -> dict[str, Any]:
        keys = [f"K_upd_{br}_definition" for br in BRANCHES]
        return verdict(self.each(keys, lambda k, v: _at_least(k, v, -0.02)))

    def k2(self) -> dict[str, Any]:
        tests = self.each(
            [f"K_cited_{br}_other" for br in BRANCHES], lambda k, v: _within(k, v, 0.01)
        )
        for br in BRANCHES:
            e = ((self.rates.get("holdmem_D1B1_scope") or {}).get(br) or {}).get("all", {})
            miss = e.get("MissedInvalidation")
            value = (
                None
                if not miss or miss["n_worlds"] == 0 or math.isnan(miss["est"])
                else miss["est"]
            )
            tests.append(
                _test(f"scope_missed_{br}_all", value, lambda v: v > 0.05, "> 5%", "share")
            )
        return verdict(tests)

    # 3.6 query-time view
    def z1(self) -> dict[str, Any]:
        tests = self.each(
            [f"Z_upd_{br}_definition" for br in ("src", "def")], lambda k, v: _at_least(k, v, 0.30)
        )
        tests.append(
            _test(
                "Z_pool_definition (95% lower bound)",
                self.lo("Z_pool_definition"),
                lambda v: v > 0.20,
                "> 20 pp",
            )
        )
        return verdict(tests)

    def z1k(self) -> dict[str, Any]:
        keys = [f"Z0_upd_{br}_definition" for br in ("src", "def")]
        return verdict(self.each(keys, lambda k, v: _at_least(k, v, 0.30)))

    def z2(self) -> dict[str, Any]:
        tests = self.each(
            [f"Z_upd_{br}_rule" for br in ("src", "def")], lambda k, v: _at_least(k, v, 0.0)
        )
        tests += self.each(
            [f"Z_upd_{br}_free" for br in BRANCHES], lambda k, v: _at_most(k, v, 0.03)
        )
        tests += self.each(["Z_upd_orphan_rule"], lambda k, v: _at_most(k, v, 0.0))
        tests += self.each(
            [f"Z_stab_{br}_all" for br in ("src", "def")], lambda k, v: _at_least(k, v, -0.03)
        )
        return verdict(tests)

    def z3(self) -> dict[str, Any]:
        keys = [f"Z_vs_D1s_{br}_definition" for br in ("src", "def")]
        return verdict(self.each(keys, lambda k, v: _within(k, v, 0.10)))

    # 3.7 thinking writer (B = 1 only)
    def d2(self) -> dict[str, Any]:
        r = (self.reported.get("think") or {}).get("all") or {}
        share = r.get("unreadable")
        value = None if not share or math.isnan(share["est"]) else share["est"]
        return verdict(
            [_at_most("think_unreadable_share", value, 0.20, "share")], writes=r.get("writes")
        )

    def d1(self, d2: dict[str, Any]) -> dict[str, Any]:
        if (stop := self._validity("P-D2", d2)) is not None:
            return stop
        keys = [f"{t}_{br}_definition" for t in ("D_rep", "D_repmech") for br in ("src", "def")]
        return verdict(self.each(keys, lambda k, v: _at_least(k, v, 0.30)))

    # 6 (registered 2026-10-01) the access control: D0full against D1
    def f1(self) -> dict[str, Any]:
        keys = [f"{t}_{br}_definition" for t in ("F_rep", "F_repmech") for br in SD]
        tests = self.each(keys, lambda k, v: _at_least(k, v, 0.30))
        regions = [gap_region(t.value) for t in tests]
        common = set(regions)
        reading = None if None in common else common.pop() if len(common) == 1 else "mixed"
        return verdict(tests, reading=reading, regions=dict(zip(keys, regions, strict=True)))

    def all(self) -> dict[str, dict[str, Any]]:
        m3, d2 = self.m3(), self.d2()
        return {
            "G1": self.g1(), "G2": self.g2(), "G3": self.g3(),
            "R1": self.r1(), "R2": self.r2(),
            "T1": self.t1(), "T2": self.t2(), "T3": self.t3(),
            "M1": self.m1(m3), "M2": self.m2(m3), "M3": m3,
            "K1": self.k1(), "K2": self.k2(),
            "Z1": self.z1(), "Z1k": self.z1k(), "Z2": self.z2(), "Z3": self.z3(),
            "D1": self.d1(d2), "D2": d2, "F1": self.f1(),
        }  # fmt: skip


PREDICTIONS = ("G1", "G2", "G3", "R1", "R2", "T1", "T2", "T3", "M1", "M2", "M3", "K1", "K2",
               "Z1", "Z1k", "Z2", "Z3", "D1", "D2", "F1")  # fmt: skip
WITHDRAWN = {
    "R1": "D = 0 contrasts withdrawn (D0B0_refresh not run); judged on the three D = 1 contrasts",
    "T1": "B = 0 contrasts withdrawn (D0tB0 not run); judged on the three B = 1 contrasts",
    "T3": "B = 0 contrasts withdrawn (D0tB0 not run); judged on the three B = 1 contrasts",
    "K1": "D = 0 contrasts withdrawn (D0B1_scope not run); judged on the three D = 1 contrasts",
    "K2": "judged at D = 1 (D0B1_scope not run)",
    "M1": "B = 0 counterpart (D1sB0_ra) not run, not reported",
    "D1": "B = 0 counterpart (D0thinkB0) not run, not reported",
}


# ----------------------------------------------------------------------- reported quantities
def _mean(by_world: dict[str, list[float]]) -> dict[str, Any] | None:
    return world_bootstrap(by_world).to_json() if by_world else None


def _t0_ok(items: list[Row], system: str) -> Vec:
    """Did the item's program run at t0? world -> "branch|slot" -> 0/1."""
    out: Vec = defaultdict(dict)
    for it in items:
        if it["system"] != system:
            continue
        for h in it["history"]:
            if h["event"] == "derive" and "program" in h:
                out[it["world_id"]][f"{it['branch']}|{it['slot_id']}"] = float(h["program"]["ok"])
                break
    return dict(out)


def program_t0(items: list[Row]) -> dict[str, Any]:
    """P-M3: the share of programs that ran at t0, for the retraction-aware programs and for the
    D1s ones, over all items (per branch as well), and the ratio of the two (paired by item)."""
    ra, d1s = _t0_ok(items, CELLS["D1sB1ra"]), _t0_ok(items, CELLS["D1sB1"])
    out: dict[str, Any] = {}
    for label, prefix in (("all", ""), *((br, f"{br}|") for br in BRANCHES)):

        def sel(v: Vec, prefix: str = prefix) -> Vec:
            return {w: {k: x for k, x in d.items() if k.startswith(prefix)} for w, d in v.items()}

        a, b = sel(ra), sel(d1s)
        ratio = ratio_estimate(a, b)
        out[label] = {
            "ra": _mean({w: list(d.values()) for w, d in a.items()}),
            "d1s": _mean({w: list(d.values()) for w, d in b.items()}),
            "ratio": ratio.to_json() if ratio.n_worlds else None,
        }
    return out


def ra_stats(items: list[Row]) -> dict[str, Any]:
    """Retraction-aware programs, reported without a threshold. Per branch and class: the share of
    post-shift runs that returned a value although a prior read was gone (`lineage_ok` false), the
    share of runs that failed, and, on the final state of each item (the error text is kept only
    there, in `det.reason`, where the program's failure decided the item), the share of items
    whose last run failed and whose failure was an undeclared read."""
    acc: dict[tuple[str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for it in items:
        if it["system"] != CELLS["D1sB1ra"]:
            continue
        w, br, fact = it["world_id"], it["branch"], it["slot_id"]
        det = it.get("det") or {}
        classes = [c for c in ("all", "definition", "rule", "free") if CLASSES[c](fact)]
        for c in classes:
            a = acc[br, c]
            for h in it["history"]:
                note = h.get("program")
                if note is None or h["event"] == "derive":
                    continue
                a["runs_lineage_false"][w].append(float(note.get("lineage_ok") is False))
                a["runs_failed"][w].append(float(not note["ok"]))
            if "program_ok" in det:
                failed = not det["program_ok"]
                undeclared = failed and "undeclared read" in str(det.get("reason") or "")
                a["final_failed"][w].append(float(failed))
                a["final_undeclared"][w].append(float(undeclared))
                if failed:
                    a["final_undeclared_of_failed"][w].append(float(undeclared))
    out: dict[str, Any] = {}
    for (br, c), a in sorted(acc.items()):
        out.setdefault(br, {})[c] = {
            "runs": sum(len(v) for v in a["runs_failed"].values()),
            "lineage_false": _mean(a["runs_lineage_false"]),
            "failed": _mean(a["runs_failed"]),
            "final_items": sum(len(v) for v in a["final_failed"].values()),
            "final_failed": _mean(a["final_failed"]),
            "final_undeclared": _mean(a["final_undeclared"]),
            "final_undeclared_of_failed": _mean(a["final_undeclared_of_failed"]),
        }
    return out


def over_retraction(eng: Engine) -> dict[str, Any]:
    """Rule 3 over-retracts multi-witness items: a program that has to read every task declares
    every task, so the deletion of any task withdraws it. For the task definitions and the other
    definitions, per branch and for the retraction-aware and the D1s programs: the share of
    repaired items whose repaired value is None although the gold has an answer (a withdrawn
    value); per task definition, the same counts pooled over worlds."""
    out: dict[str, Any] = {}
    for br in BRANCHES:
        node: dict[str, Any] = {}
        for label in ("D1sB1ra", "D1sB1"):
            rs = eng.by_sb.get((CELLS[label], br), [])
            per: dict[str, Any] = {}
            for grp in ("taskdef", "otherdef"):
                pred = CLASSES[grp]
                cond = lambda r, pred=pred: (  # noqa: E731
                    bool(r.get("repaired_now"))
                    and r["gold_status"] == "answer"
                    and pred(r["fact_id"])
                )
                per[grp] = _mean(e1._by_world(rs, "repair_none", cond))
            facts: dict[str, Any] = {}
            for fact in TASK_DEFS:
                rep = [
                    r
                    for r in rs
                    if r["fact_id"] == fact
                    and r.get("repaired_now")
                    and r["gold_status"] == "answer"
                ]
                ch = [
                    r
                    for r in rs
                    if r["fact_id"] == fact and r["changed"] and r.get("cv") is not None
                ]
                facts[fact] = {
                    "repaired": len(rep),
                    "withdrawn": sum(bool(r["repair_none"]) for r in rep),
                    "update": (sum(float(r["cv"]) for r in ch) / len(ch)) if ch else None,
                    "changed_asked": len(ch),
                }
            node[label] = {"withdrawn_share": per, "task_definitions": facts}
        out[br] = node
    return out


def think_stats(items: list[Row]) -> dict[str, Any]:
    """The thinking writer's definition writes after the shift (P-D2 and the reported tokens):
    how many, the share that is unreadable (the reply was cut off by the token limit or could not
    be parsed: `note.think.ok` false), the share that used the whole budget, and the completion
    tokens per write, per branch and pooled. The unreadable share counts both causes; the
    budget share (`completion_tokens` at the limit) is the cut-off alone."""
    acc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for it in items:
        if it["system"] != CELLS["D0thinkB1"]:
            continue
        for h in it["history"]:
            note = h.get("think")
            if note is None:
                continue
            for label in (it["branch"], "all"):
                a, w = acc[label], it["world_id"]
                a["unreadable"][w].append(float(not note["ok"]))
                a["at_limit"][w].append(float(note["completion_tokens"] >= THINK_MAX_TOKENS))
                a["tokens"][w].append(float(note["completion_tokens"]))
                if note["ok"]:
                    a["tokens_readable"][w].append(float(note["completion_tokens"]))
    out: dict[str, Any] = {"limit_tokens": THINK_MAX_TOKENS}
    for label, a in sorted(acc.items()):
        tokens = sorted(x for v in a["tokens"].values() for x in v)
        out[label] = {
            "writes": len(tokens),
            "unreadable": _mean(a["unreadable"]),
            "at_limit": _mean(a["at_limit"]),
            "completion_tokens": _mean(a["tokens"]),
            "completion_tokens_readable": _mean(a["tokens_readable"]),
            "completion_tokens_median": statistics.median(tokens),
            "completion_tokens_p90": tokens[min(len(tokens) - 1, math.ceil(0.9 * len(tokens)) - 1)],
        }
    return out


def _blank_full_definition() -> dict[str, Any]:
    return {"rederivations": 0, "truncated": 0, "newest": 0, "dropped": 0, "by_source": {}}


def full_stats(items: list[Row]) -> dict[str, Any]:
    """The D0full writer's re-derivations of definition items after the shift and what each was
    given, from the `full` note of every repair event of the cell (`FullAccess.note`: `records`,
    `dropped`, `capped`, `tokens`, `truncation`, `by_source`). Per branch and pooled ("all"): the
    number of re-derivations, how many were truncated (a record of the declared sources left
    out), by the per-source rule or the newest-first fallback, the records left out, the share
    truncated (world-clustered) and the approximate tokens given against the cap. Per definition
    (`definitions`) and within it per declared source: the re-derivations, those that lost records
    of that source, the records given and left out, summed. `truncated_definitions` are the ones
    with a truncated re-derivation; `as_registered` is whether they are `TRUNCATED_DEFS`."""
    by_label: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    defs: dict[str, dict[str, dict[str, Any]]] = {}
    for it in items:
        if it["system"] != CELLS["D0fullB1"]:
            continue
        fact, w = it["slot_id"], it["world_id"]
        for h in it["history"]:
            note = h.get("full")
            if note is None:
                continue
            cut = note["dropped"] > 0
            for label in (it["branch"], "all"):
                a = by_label[label]
                a["truncated"][w].append(float(cut))
                a["tokens"][w].append(float(note["tokens"]))
                a["dropped"][w].append(float(note["dropped"]))
                a["capped"][w].append(float(note["capped"]))
                for kind in ("source", "newest"):
                    a[kind][w].append(float(note["truncation"] == kind))
                d = defs.setdefault(fact, {}).setdefault(label, _blank_full_definition())
                d["rederivations"] += 1
                d["truncated"] += cut
                d["newest"] += note["truncation"] == "newest"
                d["dropped"] += note["dropped"]
                for source, n in note["by_source"].items():
                    c = d["by_source"].setdefault(
                        source, {"rederivations": 0, "truncated": 0, "records": 0, "dropped": 0}
                    )
                    c["rederivations"] += 1
                    c["truncated"] += n["dropped"] > 0
                    c["records"] += n["records"]
                    c["dropped"] += n["dropped"]
    out: dict[str, Any] = {"limit_tokens": FULL_BUDGET}
    for label, a in sorted(by_label.items()):
        tokens = sorted(x for v in a["tokens"].values() for x in v)
        count = lambda kind, a=a: int(sum(sum(v) for v in a[kind].values()))  # noqa: E731
        out[label] = {
            "rederivations": len(tokens),
            "truncated": count("truncated"),
            "truncated_share": _mean(a["truncated"]),
            "by_source_rule": count("source"),
            "newest_fallback": count("newest"),
            "dropped": count("dropped"),
            "capped": count("capped"),
            "tokens": _mean(a["tokens"]),
            "tokens_median": statistics.median(tokens),
            "tokens_max": tokens[-1],
        }
    cut_defs = sorted(f for f, d in defs.items() if d.get("all", {}).get("truncated"))
    out["definitions"] = {f: defs[f] for f in sorted(defs)}
    out["truncated_definitions"] = cut_defs
    out["registered_truncated"] = list(TRUNCATED_DEFS)
    out["as_registered"] = bool(defs) and cut_defs == sorted(TRUNCATED_DEFS)
    return out


# ---------------------------------------------------------------------- implementation checks
AUDIT_FIELDS = ("changed", "unchanged", "changed_detected", "unchanged_detected", "rederivations")
KEY_CELLS = (
    ("D0B1gold", "gold"),
    ("D1B1gold", "gold"),
    ("D0B1scope", "scope"),
    ("D1B1scope", "scope"),
)
AuditFn = Callable[[list[int], tuple[str, ...], int], dict[str, Any]]


def _audit_counts(table: cd.Table, system: str, key: str) -> dict[str, dict[str, dict[str, int]]]:
    """The run's changed / unchanged item-checkpoints, how many of them the cell had invalidated
    (`detected`) and its re-derivations (invalidate events), per branch and class: the quantities
    `e1_keys.audit` computes from the worlds."""
    out: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: dict.fromkeys(AUDIT_FIELDS, 0))
    )
    for r in table.rows:
        if r["system"] != system or "detected" not in r:
            continue
        cls = e1_keys.item_class(r["fact_id"])
        if key == "scope" and cls != "definition":
            continue  # the other items of a scope cell are keyed by their citations
        c = out[r["branch"]][cls]
        c["changed" if r["changed"] else "unchanged"] += 1
        c["changed_detected"] += bool(r["changed"] and r["detected"])
        c["unchanged_detected"] += bool((not r["changed"]) and r["detected"])
    for it in table.items:
        if it["system"] != system:
            continue
        cls = e1_keys.item_class(it["slot_id"])
        if key == "scope" and cls != "definition":
            continue
        out[it["branch"]][cls]["rederivations"] += sum(
            h["event"] == "invalidate" for h in it["history"]
        )
    return out


def audit_checks(table: cd.Table, audit_fn: AuditFn) -> dict[str, Any]:
    """The missed and false invalidation of the gold cells, and of the scope cells on definition
    items, equal the offline audit of the keys (prereg/v3_controls.md section 1 and 2)."""
    cache: dict[tuple[tuple[int, ...], tuple[str, ...]], dict[str, Any]] = {}
    cells: dict[str, Any] = {}
    for short, key in KEY_CELLS:
        system = CELLS[short]
        worlds = frozenset(
            r["world_id"] for r in table.rows if r["system"] == system and "detected" in r
        )
        if not worlds:
            continue
        have = {r["branch"] for r in table.rows if r["system"] == system and "detected" in r}
        branches = tuple(b for b in BRANCHES if b in have)
        seeds = sorted(int(w[1:]) for w in worlds)
        ck = (tuple(seeds), branches)
        if ck not in cache:
            cache[ck] = audit_fn(seeds, branches, table.months)
        expected = cache[ck]
        counts = _audit_counts(table, system, key)
        node: dict[str, Any] = {}
        ok = True
        for br in branches:
            for cls in ("definition",) if key == "scope" else ("definition", "rule", "norule"):
                run = dict(counts.get(br, {}).get(cls) or dict.fromkeys(AUDIT_FIELDS, 0))
                aud = {
                    f: int((expected.get(br, {}).get(key, {}).get(cls) or {}).get(f, 0))
                    for f in AUDIT_FIELDS
                }
                same = run == aud
                ok = ok and same

                def rates(c: dict[str, int]) -> dict[str, float | None]:
                    ch, un = c["changed"], c["unchanged"]
                    return {
                        "missed": 1 - c["changed_detected"] / ch if ch else None,
                        "false_invalidation": c["unchanged_detected"] / un if un else None,
                    }

                node.setdefault(br, {})[cls] = {
                    "pass": same,
                    "run": run,
                    "audit": aud,
                    "run_rates": rates(run),
                    "audit_rates": rates(aud),
                }
        cells[system] = {"pass": ok, "worlds": len(worlds), "branches": node}
    passed = bool(cells) and all(c["pass"] for c in cells.values())
    return {"pass": passed if cells else None, "cells": cells}


def scope_cited_check(table: cd.Table) -> dict[str, Any]:
    """The scope cell keys the non-definition items by their citations, so on those items it is
    the cited cell: the same missed and false invalidation (counts of rows; rows that differ are
    reported)."""
    scope, cited = CELLS["D1B1scope"], CELLS["D1B1cited"]

    def detected(system: str) -> dict[tuple[str, str, str, int], tuple[bool, bool]]:
        return {
            (r["world_id"], r["branch"], r["fact_id"], r["k"]): (
                bool(r["changed"]),
                bool(r["detected"]),
            )
            for r in table.rows
            if r["system"] == system and "detected" in r and not r["fact_id"].startswith("k.")
        }

    a, b = detected(scope), detected(cited)
    if not a or not b:
        return {"pass": None, "reason": "the scope cell or the cited cell is not in the table"}
    common = a.keys() & b.keys()

    def rates(d: dict[Any, tuple[bool, bool]]) -> dict[str, float | None]:
        ch = [det for k, (c, det) in d.items() if k in common and c]
        un = [det for k, (c, det) in d.items() if k in common and not c]
        return {
            "missed": 1 - sum(ch) / len(ch) if ch else None,
            "false_invalidation": sum(un) / len(un) if un else None,
        }

    ra, rb = rates(a), rates(b)
    same = ra == rb and a.keys() == b.keys()
    return {
        "pass": same,
        "rows": len(common),
        "row_mismatches": sum(a[k] != b[k] for k in common),
        "rows_only_scope": len(a.keys() - b.keys()),
        "rows_only_cited": len(b.keys() - a.keys()),
        "scope": ra,
        "cited": rb,
    }


# ------------------------------------------------------------------------------------- driver
def evaluate(
    table: cd.Table, audit_fn: AuditFn | None = None, draws: int = DRAWS
) -> dict[str, Any]:
    """Every contrast, verdict, check and reported quantity of one merged table (`draws` is the
    registered 2,000 of the bootstrap; tests of the verdict logic use fewer)."""
    eng = Engine(table, draws)
    contrasts = eng.compute_all()
    rates = eng.cell_rates()
    reported = {
        "program_t0": program_t0(table.items),
        "ra": ra_stats(table.items),
        "over_retraction": over_retraction(eng),
        "think": think_stats(table.items),
        "full": full_stats(table.items),
    }
    verdicts = Judge(contrasts, reported, rates).all()
    checks = dict(table.checks)
    checks["audit"] = audit_checks(table, audit_fn or e1_keys.audit)
    checks["scope_cited"] = scope_cited_check(table)
    marks = [
        (checks.get(k) or {}).get("pass")
        for k in ("hash", "probe_sets", "anchors", "audit", "scope_cited")
    ]
    checks["passed"] = sum(m is True for m in marks)
    checks["failed"] = sum(m is False for m in marks)
    checks["not_evaluated"] = sum(m is None for m in marks)
    summary = {
        s: sum(v["status"] == s for v in verdicts.values()) for s in (HOLDS, FAILS, NOT_JUDGED)
    }
    return {
        "status": STATUS,
        "worlds": table.worlds,
        "n_worlds": len(table.worlds),
        "branches": [b for b in BRANCHES if any(r["branch"] == b for r in table.rows)],
        "cell_origin": {s: sorted(set(v.values())) for s, v in sorted(table.origin.items())},
        "checks": checks,
        "cells": rates,
        "contrasts": {k: v.to_json() for k, v in contrasts.items()},
        "verdicts": verdicts,
        "withdrawn": WITHDRAWN,
        "reported": reported,
        "summary": summary,
    }


def stage_sources(roles: Sequence[Sequence[str]]) -> tuple[list[Path], list[Path]]:
    """The new runs and the published runs of a `stage ctl` command line (`ctl:RUN`, `base:RUN`);
    a published stage directory expands to those of its source runs that cover the registered
    worlds."""
    by_role: dict[str, list[Path]] = {"ctl": [], "base": []}
    for role, path in roles:
        if role not in by_role:
            raise SystemExit(f"ctl sources are ctl:RUN or base:RUN, not {role}:{path}")
        by_role[role].append(Path(path))
    if not by_role["ctl"] or not by_role["base"]:
        raise SystemExit("ctl needs at least one ctl:RUN and one base:RUN")
    return by_role["ctl"], cd.expand_base(by_role["base"], set(cd.ctl_worlds()))


def analyse(
    ctl_dirs: list[Path],
    base_dirs: list[Path],
    out: Path,
    worlds: list[str] | None = None,
    tolerance: float = cd.ANCHOR_TOLERANCE,
    audit_fn: AuditFn | None = None,
) -> dict[str, Any]:
    """Stage the control analysis into `out/analysis`. A failed integrity check writes
    `ctl_checks_failed.json` (and no metrics) and raises `ControlsCheckError`."""
    adir = out / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    try:
        table = cd.build(ctl_dirs, base_dirs, worlds, tolerance)
    except cd.ControlsCheckError as err:
        path = adir / "ctl_checks_failed.json"
        path.write_text(json.dumps(err.checks, indent=1, default=str))
        raise cd.ControlsCheckError(f"{err}; diagnostics in {path}", err.checks) from err
    (adir / "ctl_checks_failed.json").unlink(missing_ok=True)
    result = evaluate(table, audit_fn)
    (adir / "ctl_metrics.json").write_text(json.dumps(result, indent=1))
    with (adir / "ctl_item_table.jsonl").open("w") as f:
        for r in table.rows:
            f.write(json.dumps(r) + "\n")
    if result["checks"]["failed"]:
        print(
            "WARNING: implementation checks failed (see checks in analysis/ctl_metrics.json): "
            + ", ".join(
                k for k in ("audit", "scope_cited") if result["checks"][k]["pass"] is False
            ),
            flush=True,
        )
    return result

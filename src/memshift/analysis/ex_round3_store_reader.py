"""Store x reader split of the repair term of Eq. (1) (exploratory; post hoc, answers the third
review round of the paper, reviewer 2, W5: a correct stored value that the reader misreads is not
a failed re-derivation). It changes no registered contrast, margin, threshold or verdict, makes no
model call and reads the raw runs the staged analyses were built from.

Eq. (1), Update = (1 - m) r + m a, is computed on the changed, asked item-times of a cell. Its
terms are answers of the reader, so `r` mixes two things: whether the re-derivation put the right
typed value in the store, and whether the retrieval and the reader then used it. Per cell and
item class this module splits every changed item-time by the state of the store at the probe time:

  stored correct   the store's state for the slot at the probe time (status and typed value) is the
                   gold (status and value), under `scoring.normalize.contract_valid`: the matcher
                   of the scorer, applied to the stored value in place of the answer. A served item
                   (status valid) states its typed value, or abstains when that value is none; an
                   item that is not served (unverifiable) leaves the slot without evidence, which
                   is the state `abstain`. The typed value is the item's `eval_value`, the value
                   the write produced (the checker's or the authored derivation's output at D=1);
                   the statement the reader sees is the writer's text and is not parsed.
  B=1 cells        one item per slot, rewritten in place at each invalidation: its state is that of
                   the last history event at or before the probe time.
  B=0 cells        old items never expire and delta items are appended: the state of a slot is the
                   newest served item's (`a_store`, the item a recency rule would use). The
                   sensitivity `a_store_any` counts a slot as correct when any served item states
                   the gold (an oracle choice among the stored statements), which bounds what a
                   reader that always picked the right statement could reach.

With d the invalidated item-times of a cell (the key detected them) and missed the rest:

  r_store = P(stored correct | d)        u = P(ContractValid | d, stored correct)
  l       = P(ContractValid | d, stored wrong)       (lucky reads of a wrong or absent store)
  r       = r_store u + (1 - r_store) l = r_store - loss + luck, with loss = r_store (1 - u) and
            luck = (1 - r_store) l; `r_net` = luck - loss = r - r_store is the net reader term
  a_store, u_a, l_a, a   the same over the missed item-times (at B=0 that is every item-time, and
            a is Update)

`u` bundles everything between a store state and the answer: the retrieval of the statement among
all served statements (BM25 over the statements, a 4,096-token budget), competing stale statements
at B=0 and the reader's reading. The tables split it by gold status (`u_answer`, `u_abstain`).

The B x D interaction on Update with a key that misses nothing is (r1 - a1) - (r0 - a0) (the paper's
Eq. (2)). With r = r_store + r_net and a = a_store + a_net it splits exactly into a store part,
(r_store1 - a_store1) - (r_store0 - a_store0), and a reader part, the same contrast of r_net and
a_net. The D effect on r and on a, and the B effect at each D, split the same way.

Cells: the main study's D0B0, D0B1, D1B0, D1B1 (source-epoch key; the D=1 arm is the benchmark's
authored derivation), the JSON arm (the D1B1 store served as JSON records: the same items, so the
same stored state, a different reader input), and block C's D_m cells (D1s: writer-authored
programs; its D=0 reference cells are block C's own). Classes follow `ex_e1_classes.CLASS_SETS`.
Intervals: world-clustered bootstrap, 2,000 draws, seed 996, resampling the worlds exactly as
`ex_e1_classes` does, so Update, r, a and the interaction reproduce its estimates; the run checks
that against the staged `e1_metrics.json` and stops on a difference.

Usage: python -m memshift.analysis.ex_round3_store_reader [--root DIR] [--out DIR]
writes OUT/store_reader.json and OUT/tab_store_reader.tex (default OUT: runs/stage6_ex_round3).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis.e1 import Row
from memshift.analysis.ex_round2 import BRANCH_LABELS
from memshift.analysis.paper_tables import _row, interval
from memshift.scoring.normalize import contract_valid

STATUS = ex.STATUS + "; post hoc, third review round (reviewer 2, W5)"
SOURCES_FILE = "results/sources.json"
MAIN_KEY, DM_KEY = "e1", "c"  # staged analyses: main study, block C
ARMS = (("D1", MAIN_KEY), ("D1s", DM_KEY))  # D=1 arm of the cell set, sources.json key
BRANCHES = ("src", "def", "orphan")
TABLE_CLASSES = ("definition", "other", "all")
CELLS = ("D0B0", "D1B0", "D0B1", "D1B1")
JSON_CELL = "D1B1json"  # the D1B1 store served as JSON records
JSON_SYSTEM = "holdmem_json_arm"
STATS = (
    "n",  # changed, asked item-times
    "d",  # ... invalidated (detected) by the key
    "ds",  # ... invalidated and stored correct
    "dsc",  # ... invalidated, stored correct and answered ContractValid
    "dwc",  # ... invalidated, stored wrong and answered ContractValid
    "ms",  # missed (not invalidated) and stored correct (newest served item)
    "msc",  # ... and answered ContractValid
    "mwc",  # missed, stored wrong, answered ContractValid
    "msa",  # missed and stored correct under the any-served-item rule
    "msac",  # ... and answered ContractValid
    "ds_abs",  # invalidated, stored correct, gold status abstain
    "dsc_abs",  # ... answered ContractValid
    "ms_abs",  # missed, stored correct, gold status abstain
    "msc_abs",  # ... answered ContractValid
)
S = {name: i for i, name in enumerate(STATS)}
Arr = np.ndarray


# ------------------------------------------------------------------------------------ loading
def source_runs(stage: Path) -> list[Path]:
    """The raw run directories a merged stage was built from (its manifest's `sources`)."""
    man = json.loads((stage / "manifest.json").read_text())
    runs = [stage.parent / s["run_id"] for s in man["sources"]]
    files = ("items.jsonl", "probes.jsonl", "answers.jsonl")
    missing = [r.name for r in runs if not all((r / f).exists() for f in files)]
    if missing:
        raise FileNotFoundError(f"{stage.name}: raw runs not found: {missing}")
    return runs


def verify_item_table(rows: Sequence[Row], stage: Path) -> int:
    """The item table rebuilt from the raw runs equals the staged one on the fields this analysis
    uses (changed, ContractValid, detected, repaired now). Returns the number of rows compared."""
    keys = ("changed", "cv", "detected", "repaired_now")
    staged: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    with (stage / "analysis" / "e1_item_table.jsonl").open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            staged[r["system"], r["world_id"], r["branch"], r["fact_id"], r["k"]] = tuple(
                r.get(k) for k in keys
            )
    bad = 0
    for r in rows:
        key = (r["system"], r["world_id"], r["branch"], r["fact_id"], r["k"])
        bad += staged.get(key) != tuple(r.get(k) for k in keys)
    if bad or len(staged) != len(rows):
        raise ValueError(f"{stage.name}: {bad} rows differ, {len(staged)} staged vs {len(rows)}")
    return len(rows)


def stored_matches(probe: Mapping[str, Any], served: bool, value: Any) -> bool:
    """Does a stored state (status and typed value) meet the gold of the probe? The scorer's
    `contract_valid` with the stored state in place of the answer: a served item with a value
    answers, anything else (no value, not served) abstains."""
    answers = served and value is not None
    return contract_valid(
        probe["vtype"],
        "answer" if answers else "abstain",
        value if answers else None,
        probe["gold_status"],
        probe["gold_value"],
    )


def stored_flags(probe: Mapping[str, Any], items: Sequence[Row]) -> tuple[bool, bool]:
    """(newest served item's state is the gold, some served item's state is the gold) for a slot
    at the probe time. No served item is the state `abstain` in both."""
    served = []
    for it in items:
        upto = [h for h in it["history"] if h["ts"] <= probe["as_of"]]
        if upto and upto[-1]["new"] == "valid":
            served.append((it["history"][0]["ts"], upto[-1]["value"]))
    if not served:
        absent = stored_matches(probe, False, None)
        return absent, absent
    served.sort(key=lambda s: s[0])
    return (
        stored_matches(probe, True, served[-1][1]),
        any(stored_matches(probe, True, v) for _, v in served),
    )


def arm_systems(arm: str) -> dict[str, tuple[str, str]]:
    """system -> (cell name, system whose store the cell serves) for the cells of one D=1 arm."""
    out = {}
    for d in (0, 1):
        for b in (0, 1):
            s = ex.cell_system(arm, d, b, "source")
            out[s] = (f"D{d}B{b}", s)
    if arm == "D1":
        out[JSON_SYSTEM] = (JSON_CELL, ex.cell_system(arm, 1, 1, "source"))
    return out


class Store:
    """The raw items and the gold of a set of runs, indexed for the stored-state lookups."""

    def __init__(self, data: Mapping[str, list[Row]]):
        self.gold = {e1._key(p): p for p in data["probes"]}
        self.slots: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
        for it in data["items"]:
            self.slots[it["world_id"], it["branch"], it["system"], it["slot_id"]].append(it)

    def flags(self, row: Row, system: str) -> tuple[bool, bool, Row]:
        """(newest, any, probe) of the item-time `row` for the store of `system`."""
        probe = self.gold[row["world_id"], row["branch"], row["fact_id"], row["k"]]
        items = self.slots[row["world_id"], row["branch"], system, row["fact_id"]]
        newest, anyone = stored_flags(probe, items)
        return newest, anyone, probe


def item_stats(det: bool, ok: bool, ok_any: bool, cv: bool, gone: bool) -> Arr:
    """The `STATS` of one changed, asked item-time."""
    d, m = float(det), float(not det)
    ok_, any_, cv_, gone_ = float(ok), float(ok_any), float(cv), float(gone)
    vec = np.zeros(len(STATS))
    vec[S["n"]] = 1.0
    vec[S["d"]] = d
    vec[S["ds"]] = d * ok_
    vec[S["dsc"]] = d * ok_ * cv_
    vec[S["dwc"]] = d * (1 - ok_) * cv_
    vec[S["ms"]] = m * ok_
    vec[S["msc"]] = m * ok_ * cv_
    vec[S["mwc"]] = m * (1 - ok_) * cv_
    vec[S["msa"]] = m * any_
    vec[S["msac"]] = m * any_ * cv_
    vec[S["ds_abs"]] = d * ok_ * gone_
    vec[S["dsc_abs"]] = d * ok_ * cv_ * gone_
    vec[S["ms_abs"]] = m * ok_ * gone_
    vec[S["msc_abs"]] = m * ok_ * cv_ * gone_
    return vec


def arm_arrays(
    store: Store, rows: Sequence[Row], arm: str, branch: str
) -> tuple[list[str], Arr, list[str]]:
    """Per-world sums of `STATS` over the changed, asked item-times: (worlds, array (worlds, cells,
    base classes, stats), cell names)."""
    systems = arm_systems(arm)
    names = [n for n in (*CELLS, JSON_CELL) if n in {c for c, _ in systems.values()}]
    d1b1 = ex.cell_system(arm, 1, 1, "source")
    detected = {
        (r["world_id"], r["fact_id"], r["k"]): bool(r.get("detected"))
        for r in rows
        if r["branch"] == branch and r["system"] == d1b1
    }
    worlds = sorted({r["world_id"] for r in rows if r["branch"] == branch})
    pos = {w: i for i, w in enumerate(worlds)}
    arr = np.zeros((len(worlds), len(names), len(ex.BASE_CLASSES), len(STATS)))
    for r in rows:
        if r["branch"] != branch or r["system"] not in systems:
            continue
        if not r["changed"] or r.get("cv") is None:
            continue
        cell, served = systems[r["system"]]
        newest, anyone, probe = store.flags(r, served)
        det = detected[r["world_id"], r["fact_id"], r["k"]] if cell == JSON_CELL else r["detected"]
        gone = probe["gold_status"] == "abstain"
        c = ex.BASE_CLASSES.index(ex.item_class(r["fact_id"]))
        arr[pos[r["world_id"]], names.index(cell), c] += item_stats(
            bool(det), newest, anyone, bool(r["cv"]), gone
        )
    return worlds, arr, names


# ------------------------------------------------------------------------------------- terms
def cell_terms(t: Arr) -> dict[str, Arr]:
    """The terms of cells from their sums t (..., len(STATS)); a term over no item-time is NaN."""
    v = {k: t[..., i] for k, i in S.items()}
    miss = v["n"] - v["d"]
    r_ok, a_ok = v["ds"], v["ms"]
    cv_d, cv_m = v["dsc"] + v["dwc"], v["msc"] + v["mwc"]
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "n": v["n"],
            "m": miss / v["n"],
            "U": (cv_d + cv_m) / v["n"],
            "r_store": r_ok / v["d"],
            "u": v["dsc"] / r_ok,
            "u_answer": (v["dsc"] - v["dsc_abs"]) / (r_ok - v["ds_abs"]),
            "u_abstain": v["dsc_abs"] / v["ds_abs"],
            "l": v["dwc"] / (v["d"] - r_ok),
            "r": cv_d / v["d"],
            "r_net": (cv_d - r_ok) / v["d"],
            "a_store": a_ok / miss,
            "u_a": v["msc"] / a_ok,
            "u_a_answer": (v["msc"] - v["msc_abs"]) / (a_ok - v["ms_abs"]),
            "u_a_abstain": v["msc_abs"] / v["ms_abs"],
            "l_a": v["mwc"] / (miss - a_ok),
            "a": cv_m / miss,
            "a_net": (cv_m - a_ok) / miss,
            "a_store_any": v["msa"] / miss,
            "u_a_any": v["msac"] / v["msa"],
            "a_net_any": (cv_m - v["msa"]) / miss,
        }


def interaction_terms(c: Mapping[str, Mapping[str, Arr]]) -> dict[str, Arr]:
    """The B x D accounting of the four cells' terms: the interaction (r1 - a1) - (r0 - a0)
    (equal to the observed one on Update when the key misses nothing), its store part and reader
    part, the B effect at each D and the D effect on r and on a, each split the same way. The
    `_any` parts take the any-served-item rule for the store of the B=0 cells."""
    r = {d: c[f"D{d}B1"] for d in (0, 1)}
    a = {d: c[f"D{d}B0"] for d in (0, 1)}
    u = {k: c[k]["U"] for k in CELLS}
    with np.errstate(divide="ignore", invalid="ignore"):
        obs = (u["D1B1"] - u["D1B0"]) - (u["D0B1"] - u["D0B0"])
        out: dict[str, Arr] = {"observed": obs}
        for d in (0, 1):  # B effect at D: r_D - a_D = store (r_store - a_store) + reader (net)
            out[f"B_effect_D{d}"] = r[d]["r"] - a[d]["a"]
            out[f"B_effect_D{d}_store"] = r[d]["r_store"] - a[d]["a_store"]
            out[f"B_effect_D{d}_reader"] = r[d]["r_net"] - a[d]["a_net"]
        out["interaction"] = out["B_effect_D1"] - out["B_effect_D0"]
        out["store"] = out["B_effect_D1_store"] - out["B_effect_D0_store"]
        out["reader"] = out["B_effect_D1_reader"] - out["B_effect_D0_reader"]
        out["gap_observed"] = obs - out["interaction"]  # 0 when the key misses nothing
        out["store_share"] = np.where(
            np.abs(out["interaction"]) > 1e-12, out["store"] / out["interaction"], np.nan
        )
        out["D_effect_r"] = r[1]["r"] - r[0]["r"]
        out["D_effect_r_store"] = r[1]["r_store"] - r[0]["r_store"]
        out["D_effect_r_reader"] = r[1]["r_net"] - r[0]["r_net"]
        out["D_effect_a"] = a[1]["a"] - a[0]["a"]
        out["D_effect_a_store"] = a[1]["a_store"] - a[0]["a_store"]
        out["D_effect_a_reader"] = a[1]["a_net"] - a[0]["a_net"]
        out["store_any"] = (r[1]["r_store"] - a[1]["a_store_any"]) - (
            r[0]["r_store"] - a[0]["a_store_any"]
        )
        out["reader_any"] = out["interaction"] - out["store_any"]
    return out


def quantities(t: Arr, names: Sequence[str]) -> dict[str, Any]:
    """Every quantity from class-summed totals t (..., cells, stats)."""
    terms = {n: cell_terms(t[..., i, :]) for i, n in enumerate(names)}
    return {"cells": terms, "interaction": interaction_terms(terms)}


def _flat(q: Mapping[str, Any], prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], Arr]:
    out: dict[tuple[str, ...], Arr] = {}
    for k, v in q.items():
        if isinstance(v, Mapping):
            out.update(_flat(v, (*prefix, k)))
        else:
            out[(*prefix, k)] = v
    return out


def _nest(flat: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, v in flat.items():
        node = out
        for p in path[:-1]:
            node = node.setdefault(p, {})
        node[path[-1]] = v
    return out


COUNTS = ("n",)  # counts, not estimates


def analyse_arm(
    store: Store, rows: Sequence[Row], arm: str, draws: int = ex.DRAWS
) -> dict[str, Any]:
    """`result[branch][class]` = {"cells": {cell: {term: estimate}}, "interaction": {...}}, all
    quantities from the same resampled worlds; `counts` are the item-times behind each cell."""
    out: dict[str, Any] = {}
    for branch in BRANCHES:
        worlds, arr, names = arm_arrays(store, rows, arm, branch)
        drawn = arr[ex.boot_index(len(worlds), draws)].sum(axis=1)
        point = arr.sum(axis=0)
        for cls in ex.CLASS_SETS:
            tot = ex.class_totals(point, cls)
            est = _flat(quantities(tot, names))
            vals = _flat(quantities(ex.class_totals(drawn, cls), names))
            n_cls = int(tot[names.index("D1B1"), S["n"]])
            res: dict[tuple[str, ...], Any] = {}
            for path, e in est.items():
                if path[-1] in COUNTS:
                    res[path] = int(e)
                else:
                    res[path] = ex.interval(
                        float(e), np.asarray(vals[path], float), len(worlds), n_cls
                    )
            counts = {n: {k: int(tot[i, S[k]]) for k in STATS} for i, n in enumerate(names)}
            node = _nest(res)
            node["counts"] = counts
            out.setdefault(branch, {})[cls] = node
    return out


# -------------------------------------------------------------------------------- self check
def check_against_stage(
    result: Mapping[str, Any], metrics: Mapping[str, Any], arm: str
) -> dict[str, Any]:
    """Update, m, r and a of the cells and the interaction against the staged
    `explore_decomposition` (same bootstrap, so equal to rounding); the JSON cell's Update against
    `explore_json_arm`. Raises on a difference above 1e-9."""
    worst, compared = 0.0, 0
    for branch in BRANCHES:
        for cls in ex.CLASS_SETS:
            mine = result[branch][cls]
            theirs = metrics["explore_decomposition"][arm][branch][cls]
            pairs = []
            for cell in CELLS:
                for term in ("U", "m", "r", "a"):
                    pairs.append((mine["cells"][cell][term], theirs["cells"]["source"][cell][term]))
            pairs.append(
                (
                    mine["interaction"]["interaction"],
                    theirs["interaction"]["source"]["corrected_formula"],
                )
            )
            pairs.append(
                (mine["interaction"]["observed"], theirs["interaction"]["source"]["observed"])
            )
            for a, b in pairs:
                for key in ("est", "lo", "hi"):
                    x, y = a[key], b[key]
                    if math.isnan(x) and math.isnan(y):
                        continue
                    worst = max(worst, abs(x - y))
                    compared += 1
            if arm == "D1":
                # the staged JSON arm resamples only the worlds that have items of the class, so
                # its intervals differ on small classes; the point estimate is the same pool
                x = mine["cells"][JSON_CELL]["U"]["est"]
                y = metrics["explore_json_arm"][branch][cls]["update_json"]["est"]
                if not (math.isnan(x) and math.isnan(y)):
                    worst = max(worst, abs(x - y))
                    compared += 1
    if worst > 1e-9:
        raise ValueError(f"{arm}: store x reader terms differ from the staged analysis by {worst}")
    return {"compared": compared, "max_abs_difference": worst}


# ------------------------------------------------------------------------------------- tables
def _pc(e: Mapping[str, Any] | None, digits: int = 1) -> str:
    if not e or e["est"] != e["est"]:
        return "---"
    return f"{100 * e['est']:.{digits}f}"


def _pp(e: Mapping[str, Any] | None, digits: int = 1) -> str:
    """`est [lo, hi]` in pp, signed, in the interval format of the generated tables."""
    if not e or e["est"] != e["est"]:
        return "---"

    def sg(x: float) -> str:
        v = round(100 * x, digits)
        s = f"{abs(v):.{digits}f}"
        return ("$+$" if v > 0 else "$-$") + s if v != 0 else s

    return interval(sg(e["est"]), sg(e["lo"]), sg(e["hi"]))


ARM_ROWS = (
    # label, arm, B=1 cell, B=0 cell
    (r"$D{=}0$", "D1", "D0B1", "D0B0"),
    (r"$D{=}1$", "D1", "D1B1", "D1B0"),
    (r"$D{=}1$, JSON", "D1", JSON_CELL, None),
    (r"$D_{\mathrm{m}}$", "D1s", "D1B1", "D1B0"),
)
# Block C's D=0 cells have the main study's counts (`analyse` checks it): one D=0 row for both.


def _gap(lines: list[str], branch_i: int, class_i: int) -> None:
    """The vertical space before the next class of a shift, or before the next shift."""
    if class_i < len(TABLE_CLASSES) - 1:
        lines.append(r"\addlinespace[1pt]")
    elif branch_i < len(BRANCHES) - 1:
        lines.append(r"\addlinespace[4pt]")


def tab_levels(res: Mapping[str, Any]) -> str:
    """Per shift and class, the store and reader terms of r (B=1) and a (B=0) at each D."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Store and reader terms of the repair term $r$ and of $a$ in Eq.~(1) (main "
        r"study and block C, 20 worlds each; source-epoch key; percent, point estimates; "
        r"exploratory). On the changed item-times, $r_{\mathrm{store}}$ ($a_{\mathrm{store}}$): "
        r"share whose stored state, status and typed value, equals gold under the scorer's "
        r"matcher, among the re-derived (missed) item-times; $u$: ContractValid given a correct "
        r"store; $l$: ContractValid given a wrong store, so $r=r_{\mathrm{store}}u+"
        r"(1{-}r_{\mathrm{store}})l$, and likewise $a$. $u$ also holds the retrieval of the "
        r"statement and the reader's reading. $m$: share of changes the key misses; at "
        r"$B{=}0$ no item is invalidated, so $a$ is Update and the store is the newest served "
        r"item. JSON: the $D{=}1$ store served as JSON records, so the same stored state. "
        r"$D_{\mathrm{m}}$: model-written derivation (block C, whose $D{=}0$ cells equal the "
        r"main study's).}",
        r"\label{tab:store-reader}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lll r rrrr rrrr@{}}",
        r"\toprule",
        r"& & & & \multicolumn{4}{c}{$B{=}1$: $r$} & \multicolumn{4}{c}{$B{=}0$: $a$} \\",
        r"\cmidrule(lr){5-8}\cmidrule(l){9-12}",
        r"& & & $m$ & $r_{\mathrm{store}}$ & $u$ & $l$ & $r$ & $a_{\mathrm{store}}$ & $u_a$ "
        r"& $l_a$ & $a$ \\",
        r"\midrule",
    ]
    for i, b in enumerate(BRANCHES):
        for k, cls in enumerate(TABLE_CLASSES):
            for j, (label, arm, c1, c0) in enumerate(ARM_ROWS):
                node = res[arm][b][cls]["cells"]
                hi = node[c1]
                lo = node[c0] if c0 else None
                lines.append(
                    _row(
                        BRANCH_LABELS[b] if (k == 0 and j == 0) else "",
                        cls if j == 0 else "",
                        label,
                        _pc(hi["m"], 0),
                        _pc(hi["r_store"]),
                        _pc(hi["u"]),
                        _pc(hi["l"]),
                        _pc(hi["r"]),
                        _pc(lo["a_store"]) if lo else "---",
                        _pc(lo["u_a"]) if lo else "---",
                        _pc(lo["l_a"]) if lo else "---",
                        _pc(lo["a"]) if lo else "---",
                    )
                )
            _gap(lines, i, k)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _share(q: Mapping[str, Any]) -> str:
    """The store part's share of the interaction, where the interaction's interval excludes 0
    (a ratio over an interaction near zero is meaningless)."""
    e = q["interaction"]
    if e["est"] != e["est"] or not (e["lo"] > 0 or e["hi"] < 0):
        return "---"
    return _pc(q["store_share"], 0)


def tab_interaction(res: Mapping[str, Any]) -> str:
    """Per shift and class, the B x D interaction on Update and its store and reader parts."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{The $B{\times}D$ interaction $(r_1{-}a_1)-(r_0{-}a_0)$ split into a store "
        r"part, the same contrast of the stored-correct shares, and a reader part, the same "
        r"contrast of the net reader terms $r-r_{\mathrm{store}}$ and $a-a_{\mathrm{store}}$ "
        r"(pp, 95\% world-bootstrap interval; source-epoch key, which misses no change; "
        r"exploratory). Store share (\%): store part over the interaction, where the "
        r"interaction's interval excludes zero. $D_{\mathrm{m}}$ replaces the authored $D{=}1$ "
        r"cell by the model-written derivation (block C).}",
        r"\label{tab:store-reader-bxd}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lll c c c r@{}}",
        r"\toprule",
        r"& & & $B{\times}D$ & store part & reader part & store share \\",
        r"\midrule",
    ]
    for i, b in enumerate(BRANCHES):
        for k, cls in enumerate(TABLE_CLASSES):
            for j, (label, arm) in enumerate(((r"$D{=}1$", "D1"), (r"$D_{\mathrm{m}}$", "D1s"))):
                q = res[arm][b][cls]["interaction"]
                lines.append(
                    _row(
                        BRANCH_LABELS[b] if (k == 0 and j == 0) else "",
                        cls if j == 0 else "",
                        label,
                        _pp(q["interaction"]),
                        _pp(q["store"]),
                        _pp(q["reader"]),
                        _share(q),
                    )
                )
            _gap(lines, i, k)
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- driver
def analyse(root: Path, out: Path, draws: int = ex.DRAWS) -> dict[str, Any]:
    sources = json.loads((root / SOURCES_FILE).read_text())
    result: dict[str, Any] = {
        "status": STATUS,
        "bootstrap": {"draws": draws, "seed": ex.SEED, "unit": "world"},
        "definitions": (
            "stored correct: the store's state for the slot at the probe time (newest served "
            "item at B=0) is the gold under scoring.normalize.contract_valid; unverifiable or "
            "no value is the state abstain; u = P(ContractValid | stored correct), l = "
            "P(ContractValid | stored wrong); r = r_store u + (1 - r_store) l"
        ),
        "arms": {"D1": "main study, authored derivation", "D1s": "block C, model-written"},
        "caveats": [
            "the stored value is the item's typed value (eval_value); the reader sees the "
            "writer's statement, which may differ (the reader corrects some wrong typed values: "
            "l > 0)",
            "u holds retrieval of the statement among all served statements, stale competitors "
            "at B=0 and the reader's reading",
            "the source-epoch key counts an item-time as invalidated when it was invalidated at "
            "or before the checkpoint, as the paper's m does; about 1% of the changed item-times "
            "were not re-derived at their own checkpoint and are counted under r",
            "an unverifiable item is not served and counts as the state abstain",
        ],
        "sources": {},
        "check": {},
        "results": {},
    }
    for arm, key in ARMS:
        stage = root / sources[key]
        runs = source_runs(stage)
        data = e1.load(*runs)
        rows = e1.item_table(data)
        n_rows = verify_item_table(rows, stage)
        res = analyse_arm(Store(data), rows, arm, draws)
        metrics = json.loads((stage / "analysis" / "e1_metrics.json").read_text())
        if draws == ex.DRAWS:
            result["check"][arm] = check_against_stage(res, metrics, arm) | {"item_rows": n_rows}
        result["sources"][arm] = {
            "stage": sources[key],
            "runs": [r.name for r in runs],
            "worlds": len({r["world_id"] for r in rows}),
        }
        result["results"][arm] = res
    main, dm = result["results"]["D1"], result["results"]["D1s"]
    result["check"]["d0_cells_equal_across_arms"] = all(
        main[b][c]["counts"][cell] == dm[b][c]["counts"][cell]
        for b in BRANCHES
        for c in ex.CLASS_SETS
        for cell in ("D0B0", "D0B1")
    )
    out.mkdir(parents=True, exist_ok=True)
    (out / "store_reader.json").write_text(json.dumps(result, indent=1))
    (out / "tab_store_reader.tex").write_text(
        "% generated by memshift.analysis.ex_round3_store_reader; draft, do not edit\n"
        + tab_levels(result["results"])
        + "\n"
        + tab_interaction(result["results"])
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "runs" / "stage6_ex_round3"
    res = analyse(args.root, out)
    print(out, json.dumps(res["check"]))


if __name__ == "__main__":
    main()

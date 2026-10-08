"""Exploratory class-stratified analyses of the E1 family (prereg/decisions.md 2026-09-30; they add
to every staged E1 analysis under `explore_*` keys and change no registered contrast or verdict).

They answer a five-seat review of the paper: the pre-registered D and B x D effects on Update are
averaged over 48 items per world, six of which are definitions whose D=1 value is the gold
evaluator. Every block below splits the same item table by item class, with the classes of
`e1.ITEM_CLASSES` (definition items `k.`; structured fields `rule`; free text `norule`; `other` is
structured plus free text) and the estimator of the paper: world-clustered paired bootstrap,
2,000 draws, seed 996 (`scoring.stats`; `boot_index` replays its resampling, so the class `all`
reproduces the registered contrasts to the last digit).

explore_class_mix       per branch and class: changed item-times, share, unique world-items
explore_class_split     per D=1 arm (`D1`: benchmark rules; `D1s`: model-written programs, block C),
                        branch, binding granularity and class: B effect on Update at D=0, D=1 and
                        pooled, D effect on Update at B=0 and B=1, B x D on Update, D effect on
                        repair (as registered, and changed items only)
explore_decomposition   Eq. (1), Update = (1 - m) r + m a over the changed item-times of each cell:
                        m missed invalidation, r accuracy of the invalidated (re-derived) items,
                        a accuracy of the items the key did not invalidate (at B=0 that is every
                        item), and r on the changed items repaired now; the identity residual;
                        the interaction as observed, as the paper states it, (m0 - m1)(r1 - r0),
                        and as the identity gives it, (r1 - a1) - (r0 - a0) with a_D the B=0
                        Update; the D effect at B=0; which terms are invariant
explore_invalidation    false invalidation, missed invalidation and invalidation precision of the
                        binding keys over all item-times (so a selective key can be told from a
                        refresh-everything one)
explore_per_definition  the same contrasts per definition type (and leave-one-type-out)
explore_json_arm        the JSON arm against the s_k arm after the shift, by class
explore_stability       ContractValid on unchanged items per cell, the B effect on it, and the net
                        B effect over all probes (changed plus unchanged)
explore_noise_floor     values and answers that differ between two ingestions of the same worlds
explore_cost_e1         tokens the E1 cells spent (run summaries)
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis.e1 import ITEM_CLASSES, Cond, Row, _diff
from memshift.scoring.stats import item_diffs

DRAWS = 2000
SEED = 996
ALPHA = 0.05
STATUS = (
    "exploratory: not part of prereg v0-v2 and changes no registered contrast, margin, threshold "
    "or verdict (prereg/decisions.md 2026-09-30)"
)
BASE_CLASSES = ("definition", "rule", "norule")
CLASS_SETS: dict[str, tuple[str, ...]] = {
    "definition": ("definition",),
    "other": ("rule", "norule"),
    "rule": ("rule",),
    "norule": ("norule",),
    "all": BASE_CLASSES,
}
GRANULARITIES = ("source", "cluster", "cited")
ARMS = ("D1", "D1s")  # the D=1 cell: benchmark-authored rules, writer-authored programs (block C)
Vec = dict[str, dict[str, float]]
Est = dict[str, float]

if set(BASE_CLASSES) != set(ITEM_CLASSES):  # the classifier this module reuses
    raise ImportError("ex_e1_classes: e1.ITEM_CLASSES changed, update BASE_CLASSES")


@cache
def item_class(fact_id: str) -> str:
    """The class of a fact id under `e1.ITEM_CLASSES` (every id is in exactly one)."""
    hits = [c for c, prefixes in ITEM_CLASSES.items() if fact_id.startswith(prefixes)]
    if len(hits) != 1:
        raise ValueError(f"{fact_id}: in {len(hits)} item classes")
    return hits[0]


def in_class(row: Row, cls: str) -> bool:
    return item_class(row["fact_id"]) in CLASS_SETS[cls]


# ------------------------------------------------------------------------------------ bootstrap
_INDEX: dict[tuple[int, int, int], np.ndarray] = {}


def boot_index(n: int, draws: int = DRAWS, seed: int = SEED) -> np.ndarray:
    """The resampling of `stats.world_bootstrap_draws` for `n` worlds: `draws` rows of `n` world
    positions, from the same generator calls in the same order."""
    key = (n, draws, seed)
    if key not in _INDEX:
        rng = random.Random(seed)
        _INDEX[key] = np.array(
            [[rng.randrange(n) for _ in range(n)] for _ in range(draws)], dtype=np.intp
        )
    return _INDEX[key]


def interval(
    est: float, vals: np.ndarray, n_worlds: int, n_items: int, alpha: float = ALPHA
) -> Est:
    """`Estimate.to_json` from the bootstrap draws of a statistic (NaN draws are dropped, as in
    `stats.world_bootstrap_draws`)."""
    v = np.sort(vals[~np.isnan(vals)])
    if not len(v):
        lo = hi = float("nan")
    else:
        lo = float(v[int(alpha / 2 * (len(v) - 1))])
        hi = float(v[int((1 - alpha / 2) * (len(v) - 1))])
    return {
        "est": est,
        "lo": lo,
        "hi": hi,
        "width": hi - lo,
        "n_worlds": n_worlds,
        "n_items": n_items,
    }


def world_estimate(
    per_world: np.ndarray,
    stat: Callable[[np.ndarray], Any],
    n_items: int = 0,
    draws: int = DRAWS,
    seed: int = SEED,
) -> Est:
    """Bootstrap of a statistic of the summed per-world components. `per_world` is (worlds,
    components); `stat` maps totals (..., components) to the statistic (...), so it runs on the
    point total and on every draw at once."""
    n = per_world.shape[0]
    if n == 0:
        return interval(float("nan"), np.array([]), 0, 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        est = float(stat(per_world.sum(axis=0)))
        vals = np.asarray(stat(per_world[boot_index(n, draws, seed)].sum(axis=1)), dtype=float)
    return interval(est, vals, n, n_items)


def mean_estimate(by_world: Mapping[str, Sequence[float]], draws: int = DRAWS) -> Est:
    """`stats.world_bootstrap` of a pooled mean: worlds without items are dropped."""
    groups = [by_world[w] for w in sorted(by_world) if len(by_world[w])]
    per_world = np.array([[sum(g), len(g)] for g in groups], dtype=float).reshape(-1, 2)
    return world_estimate(
        per_world, lambda t: t[..., 0] / t[..., 1], int(per_world[:, 1].sum()), draws
    )


def paired_estimate(x: Vec, y: Vec, draws: int = DRAWS) -> Est:
    """`stats.paired_diff`: mean of x[item] - y[item] over the items in both, by world."""
    return mean_estimate({w: list(d.values()) for w, d in item_diffs(x, y).items()}, draws)


# -------------------------------------------------------------------------------------- tables
class Table:
    """An E1 item table indexed by (system, branch)."""

    def __init__(self, rows: Iterable[Row]):
        self.cells: dict[tuple[str, str], list[Row]] = defaultdict(list)
        for r in rows:
            self.cells[r["system"], r["branch"]].append(r)
        self.systems = {s for s, _ in self.cells}
        self.branches = sorted({b for _, b in self.cells})
        self.worlds = sorted({r["world_id"] for rs in self.cells.values() for r in rs})

    def rows(self, system: str, branch: str) -> list[Row]:
        return self.cells.get((system, branch), [])

    def vec(self, system: str, branch: str, field: str, cond: Cond) -> Vec:
        """`e1._vec`: world -> "fact@k" -> value, for the rows that meet `cond` and have `field`."""
        out: Vec = defaultdict(dict)
        for r in self.rows(system, branch):
            if cond(r) and r.get(field) is not None:
                out[r["world_id"]][f"{r['fact_id']}@{r['k']}"] = float(r[field])
        return out

    def arms(self) -> list[str]:
        """The D=1 cells present with the D=0 reference and both B=0 and source-epoch B=1."""
        if "holdmem_D0B0" not in self.systems or "holdmem_D0B1_source" not in self.systems:
            return []
        return [a for a in ARMS if {f"holdmem_{a}B0", f"holdmem_{a}B1_source"} <= self.systems]

    def granularities(self, arm: str = "D1") -> list[str]:
        return [
            g
            for g in GRANULARITIES
            if f"holdmem_D0B1_{g}" in self.systems and f"holdmem_{arm}B1_{g}" in self.systems
        ]


def cell_system(arm: str, d: int, b: int, g: str) -> str:
    """The system of cell (D, B) at granularity `g`; `arm` names the D=1 cell."""
    head = "holdmem_D0" if d == 0 else f"holdmem_{arm}"
    return f"{head}B0" if b == 0 else f"{head}B1_{g}"


def _changed(cls: str) -> Cond:
    return lambda r: bool(r["changed"]) and in_class(r, cls)


def _repaired(cls: str, changed_only: bool) -> Cond:
    return lambda r: (
        bool(r.get("repaired_now"))
        and in_class(r, cls)
        and (bool(r["changed"]) or not changed_only)
    )


def _unchanged(cls: str) -> Cond:
    return lambda r: not r["changed"] and in_class(r, cls)


def _asked(cls: str) -> Cond:
    return lambda r: in_class(r, cls)


def _fact(fact_id: str) -> Cond:
    return lambda r: bool(r["changed"]) and r["fact_id"] == fact_id


def _definitions_but(fact_id: str) -> Cond:
    return lambda r: bool(r["changed"]) and in_class(r, "definition") and r["fact_id"] != fact_id


def vec_mean(v: Vec, draws: int = DRAWS) -> Est:
    """`stats.world_bootstrap` of the pooled mean of a `Table.vec`."""
    return mean_estimate({w: list(d.values()) for w, d in v.items()}, draws)


# ------------------------------------------------------------------------------------ class mix
def class_mix(tab: Table) -> dict[str, Any]:
    """Per branch, the changed item-times of each class (the items Update is computed on), their
    share, and the world-items (world, slot) behind them: a slot changes at several checkpoints."""
    out: dict[str, Any] = {}
    for b in tab.branches:
        system = "holdmem_D0B0" if ("holdmem_D0B0", b) in tab.cells else sorted(tab.systems)[0]
        ch = [r for r in tab.rows(system, b) if r["changed"] and r.get("cv") is not None]
        total = len(ch)
        slots_all = {(r["world_id"], r["fact_id"]) for r in ch}
        res: dict[str, Any] = {}
        for cls, members in CLASS_SETS.items():
            rs = [r for r in ch if item_class(r["fact_id"]) in members]
            slots = {(r["world_id"], r["fact_id"]) for r in rs}
            per_world: dict[str, int] = defaultdict(int)
            for r in rs:
                per_world[r["world_id"]] += 1
            res[cls] = {
                "changed_item_times": len(rs),
                "share": len(rs) / total if total else float("nan"),
                "world_items": len(slots),
                "world_item_share": len(slots) / len(slots_all) if slots_all else float("nan"),
                "worlds": len(per_world),
            }
        out[b] = {"reference_system": system, "changed_item_times": total, "classes": res}
    return out


# ---------------------------------------------------------------------------------- class split
def class_contrasts(
    tab: Table, branch: str, arm: str, g: str, cls: str, draws: int = DRAWS
) -> dict[str, Est]:
    """The contrasts of the E1 list on the items of one class, for one D=1 arm and granularity."""
    ch = _changed(cls)

    def upd(d: int, b: int) -> Vec:
        return tab.vec(cell_system(arm, d, b, g), branch, "cv", ch)

    def est(x: Vec, y: Vec) -> Est:
        return paired_estimate(x, y, draws)

    u = {(d, b): upd(d, b) for d in (0, 1) for b in (0, 1)}
    pooled: dict[int, Vec] = {}
    for b in (0, 1):
        pooled[b] = defaultdict(dict)
        for d in (0, 1):
            for w, v in u[d, b].items():
                pooled[b][w].update({f"{k}|D{d}": x for k, x in v.items()})
    out = {
        "B_effect_update_D0": est(u[0, 1], u[0, 0]),
        "B_effect_update_D1": est(u[1, 1], u[1, 0]),
        "B_effect_update_pooled": est(pooled[1], pooled[0]),
        "D_effect_update_B0": est(u[1, 0], u[0, 0]),
        "D_effect_update_B1": est(u[1, 1], u[0, 1]),
        "BxD_interaction_update": est(_diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0])),
    }
    d1, d0 = cell_system(arm, 1, 1, g), cell_system(arm, 0, 1, g)
    for key, changed_only in (("D_effect_repair", False), ("D_effect_repair_changed", True)):
        rc = _repaired(cls, changed_only)
        out[key] = est(tab.vec(d1, branch, "cv", rc), tab.vec(d0, branch, "cv", rc))
    return out


def class_split(tab: Table, draws: int = DRAWS) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in tab.arms():
        grans = tab.granularities(arm)
        out[arm] = {
            b: {
                g: {cls: class_contrasts(tab, b, arm, g, cls, draws) for cls in CLASS_SETS}
                for g in grans
            }
            for b in tab.branches
        }
    return out


# ---------------------------------------------------------------------------- Eq. (1) decomposition
STATS = ("n", "cv", "det", "cvdet", "rep", "cvrep")  # per-cell sums over changed, asked item-times
CELLS = ("D0B0", "D1B0", "D0B1", "D1B1")


def cell_arrays(
    tab: Table, branch: str, arm: str, grans: Sequence[str]
) -> tuple[list[str], np.ndarray, list[str]]:
    """Per-world sums of the decomposition over the changed item-times: (worlds, array (worlds,
    cells, classes, stats), cell names). Cells are D0B0, D1B0 and, for each granularity g, D0B1_g
    and D1B1_g; classes follow BASE_CLASSES; stats follow STATS. Only rows with a ContractValid
    answer count (every changed item is asked)."""
    names = ["D0B0", "D1B0"] + [f"D{d}B1_{g}" for g in grans for d in (0, 1)]
    systems = [
        cell_system(arm, int(n[1]), int(n[3]), n.split("_")[1] if "_" in n else "source")
        for n in names
    ]
    worlds = sorted({r["world_id"] for r in tab.rows(systems[0], branch)})
    pos = {w: i for i, w in enumerate(worlds)}
    arr = np.zeros((len(worlds), len(names), len(BASE_CLASSES), len(STATS)))
    for c, system in enumerate(systems):
        for r in tab.rows(system, branch):
            if not r["changed"] or r.get("cv") is None or r["world_id"] not in pos:
                continue
            cv = float(r["cv"])
            det = float(bool(r.get("detected")))
            rep = float(bool(r.get("repaired_now")))
            arr[pos[r["world_id"]], c, BASE_CLASSES.index(item_class(r["fact_id"]))] += (
                1.0,
                cv,
                det,
                cv * det,
                rep,
                cv * rep,
            )
    return worlds, arr, names


def class_totals(t: np.ndarray, cls: str) -> np.ndarray:
    """(..., cells, classes, stats) -> (..., cells, stats) summed over the class set."""
    idx = [BASE_CLASSES.index(c) for c in CLASS_SETS[cls]]
    return np.asarray(t[..., idx, :].sum(axis=-2))


def cell_terms(t: np.ndarray) -> dict[str, np.ndarray]:
    """Eq. (1) terms of cells from their sums t (..., 6): Update U, missed m, accuracy r of the
    invalidated (detected) items, accuracy a of the others, r on the items repaired now, and the
    identity residual U - [(1 - m) r + m a]."""
    n, cv, det, cvdet, rep, cvrep = (t[..., i] for i in range(len(STATS)))
    miss, cvmiss = n - det, cv - cvdet
    with np.errstate(divide="ignore", invalid="ignore"):
        u, m = cv / n, miss / n
        r, a, rr = cvdet / det, cvmiss / miss, cvrep / rep
        part_r = np.where(det > 0, (det / n) * r, 0.0)
        part_a = np.where(miss > 0, (miss / n) * a, 0.0)
    return {
        "n": n,
        "miss": miss,
        "U": u,
        "m": m,
        "r": r,
        "a": a,
        "r_rep": rr,
        "resid": u - (part_r + part_a),
    }


def _recomposed(c: Mapping[str, np.ndarray]) -> np.ndarray:
    """(1 - m) r + m a from the terms of a cell; a term whose weight is 0 is absent."""
    with np.errstate(invalid="ignore"):
        rec = (1 - c["m"]) * np.nan_to_num(c["r"]) + c["m"] * np.nan_to_num(c["a"])
    return np.asarray(np.where(c["n"] > 0, rec, np.nan))


def interaction_terms(c: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The B x D accounting of one granularity from the four cells' terms (`cell_terms`):
    the interaction as observed, as the paper states it, (m0 - m1)(r1 - r0), and as the identity
    gives it, (r1 - a1) - (r0 - a0) with a_D the B=0 Update (at B=0 every item is missed, so a is
    the Update), which drops the missed terms of the B=1 cells; the exact recomposition from all
    terms (equal to the observed interaction by the identity); and the main effects."""
    with np.errstate(divide="ignore", invalid="ignore"):
        u = {k: c[k]["U"] for k in CELLS}
        obs = (u["D1B1"] - u["D1B0"]) - (u["D0B1"] - u["D0B0"])
        m0 = (c["D0B0"]["miss"] + c["D1B0"]["miss"]) / (c["D0B0"]["n"] + c["D1B0"]["n"])
        m1 = (c["D0B1"]["miss"] + c["D1B1"]["miss"]) / (c["D0B1"]["n"] + c["D1B1"]["n"])
        r0, r1 = c["D0B1"]["r"], c["D1B1"]["r"]
        paper = (m0 - m1) * (r1 - r0)
        corrected = (r1 - u["D1B0"]) - (r0 - u["D0B0"])
        rec = {k: _recomposed(c[k]) for k in CELLS}
        exact = (rec["D1B1"] - rec["D1B0"]) - (rec["D0B1"] - rec["D0B0"])
        return {
            "observed": obs,
            "paper_formula": paper,
            "corrected_formula": corrected,
            "exact_from_terms": exact,
            "gap_paper": obs - paper,
            "gap_corrected": obs - corrected,
            "m0": m0,
            "m1": m1,
            "B_effect_D0": u["D0B1"] - u["D0B0"],
            "B_effect_D1": u["D1B1"] - u["D1B0"],
            "D_effect_B0": u["D1B0"] - u["D0B0"],
            "D_effect_B1": u["D1B1"] - u["D0B1"],
            "D_effect_missed": c["D1B1"]["m"] - c["D0B1"]["m"],
            "D_effect_a_B0": c["D1B0"]["a"] - c["D0B0"]["a"],
            "D_effect_a_B1": c["D1B1"]["a"] - c["D0B1"]["a"],
        }


def decomposition_quantities(
    t: np.ndarray, names: Sequence[str], grans: Sequence[str]
) -> dict[str, dict[str, Any]]:
    """Every decomposition quantity from class-summed totals t (..., cells, stats): the terms of
    each cell, the interaction accounting per granularity, and r and m compared across
    granularities (to see which terms are granularity- or D-free)."""
    pos = {n: i for i, n in enumerate(names)}
    terms = {n: cell_terms(t[..., i, :]) for n, i in pos.items()}
    out: dict[str, dict[str, Any]] = {"cells": {}, "interaction": {}, "invariance": {}}
    for g in grans:
        cg = {
            "D0B0": terms["D0B0"],
            "D1B0": terms["D1B0"],
            "D0B1": terms[f"D0B1_{g}"],
            "D1B1": terms[f"D1B1_{g}"],
        }
        out["cells"][g] = cg
        out["interaction"][g] = interaction_terms(cg)
    if len(grans) > 1 and "source" in grans:
        for g in (x for x in grans if x != "source"):
            out["invariance"][g] = {
                f"r_source_minus_{g}_D{d}": terms[f"D{d}B1_source"]["r"] - terms[f"D{d}B1_{g}"]["r"]
                for d in (0, 1)
            } | {
                f"m_source_minus_{g}_D{d}": terms[f"D{d}B1_source"]["m"] - terms[f"D{d}B1_{g}"]["m"]
                for d in (0, 1)
            }
    return out


def _flatten(
    q: Mapping[str, Any], prefix: tuple[str, ...] = ()
) -> dict[tuple[str, ...], np.ndarray]:
    out: dict[tuple[str, ...], np.ndarray] = {}
    for k, v in q.items():
        if isinstance(v, Mapping):
            out.update(_flatten(v, (*prefix, k)))
        else:
            out[(*prefix, k)] = v
    return out


def _nest(flat: Mapping[tuple[str, ...], Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for path, v in flat.items():
        d = out
        for p in path[:-1]:
            d = d.setdefault(p, {})
        d[path[-1]] = v
    return out


COUNT_TERMS = ("n", "miss")  # counts, not estimates: no interval


def decomposition(tab: Table, draws: int = DRAWS) -> dict[str, Any]:
    """`explore_decomposition[arm][branch][class]`: estimates with world-bootstrap intervals of
    every quantity of `decomposition_quantities`, all from the same resampled worlds. Under
    `cells[g][cell]` the terms of each cell, under `interaction[g]` the B x D accounting, under
    `invariance[g]` source-epoch against the finer key; `identity_max_abs_residual` is the largest
    residual of Update - [(1 - m) r + m a] over the cells of the class."""
    out: dict[str, Any] = {}
    for arm in tab.arms():
        grans = tab.granularities(arm)
        for b in tab.branches:
            worlds, arr, names = cell_arrays(tab, b, arm, grans)
            if not worlds or not grans:
                continue
            drawn = arr[boot_index(len(worlds), draws)].sum(axis=1)
            point = arr.sum(axis=0)
            for cls in CLASS_SETS:
                tot = class_totals(point, cls)
                est = _flatten(decomposition_quantities(tot, names, grans))
                vals = _flatten(decomposition_quantities(class_totals(drawn, cls), names, grans))
                n_cls = int(tot[names.index(f"D1B1_{grans[0]}"), 0])
                res: dict[tuple[str, ...], Any] = {}
                for path, e in est.items():
                    if path[-1] in COUNT_TERMS:
                        res[path] = int(e)
                    else:
                        res[path] = interval(
                            float(e), np.asarray(vals[path], float), len(worlds), n_cls
                        )
                resid = [
                    abs(float(e)) for p_, e in est.items() if p_[-1] == "resid" and np.isfinite(e)
                ]
                node = _nest(res)
                node["identity_max_abs_residual"] = max(resid, default=float("nan"))
                out.setdefault(arm, {}).setdefault(b, {})[cls] = node
    return out


# ------------------------------------------------------------------------------- invalidation
INV_STATS = ("n_ch", "det_ch", "n_un", "det_un")


def invalidation(tab: Table, draws: int = DRAWS) -> dict[str, Any]:
    """False and missed invalidation and invalidation precision of each B=1 key over all
    item-times (asked or not): precision is the share of invalidated item-times whose gold
    changed, which for a key that invalidates everything is the base rate of change."""
    out: dict[str, Any] = {}
    for arm in tab.arms():
        for b in tab.branches:
            res: dict[str, Any] = {}
            for g in tab.granularities(arm):
                for d in (0, 1):
                    system = cell_system(arm, d, 1, g)
                    rows = [r for r in tab.rows(system, b) if r.get("detected") is not None]
                    worlds = sorted({r["world_id"] for r in rows})
                    pos = {w: i for i, w in enumerate(worlds)}
                    arr = np.zeros((len(worlds), len(BASE_CLASSES), len(INV_STATS)))
                    for r in rows:
                        det = float(bool(r["detected"]))
                        ch = bool(r["changed"])
                        arr[pos[r["world_id"]], BASE_CLASSES.index(item_class(r["fact_id"]))] += (
                            float(ch),
                            det * ch,
                            float(not ch),
                            det * (not ch),
                        )
                    for cls in CLASS_SETS:
                        idx = [BASE_CLASSES.index(c) for c in CLASS_SETS[cls]]
                        pw = arr[:, idx, :].sum(axis=1)
                        res.setdefault(g, {}).setdefault(cls, {})[f"D{d}"] = _inv_metrics(pw, draws)
            if res:
                out.setdefault(arm, {})[b] = res
    return out


def _inv_metrics(per_world: np.ndarray, draws: int) -> dict[str, Est]:
    def col(t: np.ndarray, i: int) -> Any:
        return t[..., i]

    stats: dict[str, Callable[[np.ndarray], Any]] = {
        "false_invalidation": lambda t: col(t, 3) / col(t, 2),
        "missed_invalidation": lambda t: 1 - col(t, 1) / col(t, 0),
        "precision": lambda t: (col(t, 1)) / (col(t, 1) + col(t, 3)),
        "base_rate_of_change": lambda t: col(t, 0) / (col(t, 0) + col(t, 2)),
        "invalidated_share": lambda t: (col(t, 1) + col(t, 3)) / (col(t, 0) + col(t, 2)),
    }
    n = int(per_world[:, 0].sum() + per_world[:, 2].sum())
    return {k: world_estimate(per_world, f, n, draws) for k, f in stats.items()}


# ------------------------------------------------------------------------------ per definition
def definition_types(tab: Table) -> list[str]:
    return sorted(
        {
            r["fact_id"]
            for rs in tab.cells.values()
            for r in rs
            if r["fact_id"].startswith(ITEM_CLASSES["definition"])
        }
    )


def _type_contrasts(
    tab: Table, branch: str, arm: str, g: str, cond: Cond, draws: int
) -> dict[str, Est]:
    u = {
        (d, b): tab.vec(cell_system(arm, d, b, g), branch, "cv", cond)
        for d in (0, 1)
        for b in (0, 1)
    }
    return {
        "B_effect_update_D0": paired_estimate(u[0, 1], u[0, 0], draws),
        "B_effect_update_D1": paired_estimate(u[1, 1], u[1, 0], draws),
        "D_effect_update_B0": paired_estimate(u[1, 0], u[0, 0], draws),
        "D_effect_update_B1": paired_estimate(u[1, 1], u[0, 1], draws),
        "BxD_interaction_update": paired_estimate(
            _diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0]), draws
        ),
    }


def per_definition(tab: Table, draws: int = DRAWS) -> dict[str, Any]:
    """B, D and B x D on Update per definition type at each granularity present (`type`), the same
    contrasts over the definition items without that type (`without_type`: leave-one-type-out),
    and the Update of every cell and of the zero system, plain RAG and the oracle where the run has
    them (`update`). Six types: the interval is over worlds, the types are fixed."""
    out: dict[str, Any] = {}
    for arm in tab.arms():
        grans = tab.granularities(arm)
        for b in tab.branches:
            for t in definition_types(tab):
                only, rest = _fact(t), _definitions_but(t)
                node = out.setdefault(arm, {}).setdefault(b, {}).setdefault(t, {})
                node["type"] = {g: _type_contrasts(tab, b, arm, g, only, draws) for g in grans}
                node["without_type"] = {
                    g: _type_contrasts(tab, b, arm, g, rest, draws) for g in grans
                }
                g0 = grans[0]
                node["update"] = {
                    f"D{d}B{bb}": vec_mean(
                        tab.vec(cell_system(arm, d, bb, g0), b, "cv", only), draws
                    )
                    for d in (0, 1)
                    for bb in (0, 1)
                } | {
                    s: vec_mean(tab.vec(s, b, "cv", only), draws)
                    for s in ("zero", "rag", "oracle")
                    if (s, b) in tab.cells
                }
    return out


# ----------------------------------------------------------------- JSON arm and stability
def json_arm(tab: Table, draws: int = DRAWS) -> dict[str, Any]:
    """The JSON arm (the D1B1 source-epoch store served as JSON) against the same store served as
    s_k statements, on the post-shift probes: paired differences on changed items (Update), on
    unchanged items (Stability) and on all asked probes, by class. The paper reports only the
    pre-shift difference (`pre_intervention.json_minus_sk`)."""
    sk, js = "holdmem_D1B1_source", "holdmem_json_arm"
    if not {sk, js} <= tab.systems:
        return {}
    out: dict[str, Any] = {}
    for b in tab.branches:
        res: dict[str, Any] = {}
        for cls in CLASS_SETS:
            ch = _changed(cls)
            un, al = _unchanged(cls), _asked(cls)
            res[cls] = {
                "update_sk": vec_mean(tab.vec(sk, b, "cv", ch), draws),
                "update_json": vec_mean(tab.vec(js, b, "cv", ch), draws),
                "update_json_minus_sk": paired_estimate(
                    tab.vec(js, b, "cv", ch), tab.vec(sk, b, "cv", ch), draws
                ),
                "stability_json_minus_sk": paired_estimate(
                    tab.vec(js, b, "cv", un), tab.vec(sk, b, "cv", un), draws
                ),
                "all_probes_json_minus_sk": paired_estimate(
                    tab.vec(js, b, "cv", al), tab.vec(sk, b, "cv", al), draws
                ),
            }
        out[b] = res
    return out


def stability(tab: Table, draws: int = DRAWS) -> dict[str, Any]:
    """ContractValid on unchanged items (Stability) per cell and class, the B effect on it, and
    the net B effect over all asked probes (changed plus unchanged; in the discriminative design
    the asked unchanged items equal the changed ones in number, so the pool is half and half)."""
    out: dict[str, Any] = {}
    for b in tab.branches:
        res: dict[str, Any] = {}
        systems = sorted(s for s in tab.systems if (s, b) in tab.cells)
        for cls in CLASS_SETS:
            un, al = _unchanged(cls), _asked(cls)
            node: dict[str, Any] = {
                "stability": {s: vec_mean(tab.vec(s, b, "cv", un), draws) for s in systems}
            }
            for arm in tab.arms():
                for g in tab.granularities(arm):
                    blk: dict[str, Any] = {}
                    for d in (0, 1):
                        b1, b0 = cell_system(arm, d, 1, g), cell_system(arm, d, 0, g)
                        blk[f"B_effect_stability_D{d}"] = paired_estimate(
                            tab.vec(b1, b, "cv", un), tab.vec(b0, b, "cv", un), draws
                        )
                        blk[f"net_B_effect_all_probes_D{d}"] = paired_estimate(
                            tab.vec(b1, b, "cv", al), tab.vec(b0, b, "cv", al), draws
                        )
                    node.setdefault(arm, {})[g] = blk
            res[cls] = node
        out[b] = res
    return out


# ----------------------------------------------------------------- run summaries (raw runs)
def _summaries(run_dirs: Sequence[Path]) -> list[dict[str, Any]]:
    out = []
    for d in run_dirs:
        p = Path(d) / "world_summaries.jsonl"
        if p.exists():
            out += [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    return out


def noise_floor(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Second ingestion of the same world (experiments.e1.noise_floor: identical requests with
    another seed, before any shift): items whose written value differs, and pre-shift probes whose
    ContractValid differs. Counted over world-branches (`by_branch`, `total`), and over worlds
    (`world_level`): the three branches of a world share the pre-shift world and run the same
    second-ingestion requests, so their counts repeat one another (almost always exactly) and a
    world counts once there, from the first branch that reports it."""
    by_branch: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    by_world: dict[Any, Mapping[str, Any]] = {}
    for w in summaries:
        for b, v in (w.get("branches") or {}).items():
            nf = v.get("noise_floor")
            if not nf:
                continue
            by_world.setdefault(w.get("seed"), nf)
            acc = by_branch[b]
            acc["world_branches"] += 1
            for k in ("items", "values_differ", "answers_differ"):
                acc[k] += nf[k]
            acc["cv_first_sum"] += nf["cv_first"]
            acc["cv_second_sum"] += nf["cv_second"]
    if not by_branch:
        return {}

    def pack(acc: Mapping[str, float]) -> dict[str, float]:
        n = acc["world_branches"]
        return {
            "world_branches": int(n),
            "items": int(acc["items"]),
            "values_differ": int(acc["values_differ"]),
            "values_differ_share": acc["values_differ"] / acc["items"],
            "answers_differ": int(acc["answers_differ"]),
            "answers_differ_share": acc["answers_differ"] / acc["items"],
            "cv_first_mean": acc["cv_first_sum"] / n,
            "cv_second_mean": acc["cv_second_sum"] / n,
        }

    total: dict[str, float] = defaultdict(float)
    for acc in by_branch.values():
        for k, v in acc.items():
            total[k] += v
    world = {
        k: sum(nf[k] for nf in by_world.values())
        for k in ("items", "values_differ", "answers_differ")
    }
    return {
        "by_branch": {b: pack(a) for b, a in sorted(by_branch.items())},
        "total": pack(total),
        "world_level": {
            "worlds": len(by_world),
            "items": int(world["items"]),
            "values_differ": int(world["values_differ"]),
            "values_differ_share": world["values_differ"] / world["items"],
            "answers_differ": int(world["answers_differ"]),
            "answers_differ_share": world["answers_differ"] / world["items"],
        },
    }


def cost_e1(summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Tokens each E1 cell spent per world-branch (run summaries): ingestion, maintenance (the
    re-derivations at the checkpoints) and serving."""
    acc: dict[tuple[str, str], dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for w in summaries:
        for b, v in (w.get("branches") or {}).items():
            for system, c in (v.get("costs") or {}).items():
                a = acc[b, system]
                a["world_branches"] += 1
                for k in ("ingest_tokens", "maintain_tokens", "serve_tokens"):
                    a[k] += c.get(k, 0)
    out: dict[str, Any] = {}
    for (b, system), a in sorted(acc.items()):
        n = a["world_branches"]
        out.setdefault(b, {})[system] = {
            "world_branches": int(n),
            **{
                f"{k}_per_world_branch": a[k] / n
                for k in ("ingest_tokens", "maintain_tokens", "serve_tokens")
            },
        }
    return out


# ----------------------------------------------------------------------------------- entry point
def analyse_rows(
    rows: Iterable[Row], run_dirs: Sequence[Path] = (), draws: int = DRAWS
) -> dict[str, Any]:
    """Every `explore_*` block of this module for one E1 item table (and, for the noise floor and
    the token counts, the raw runs' summaries when they are at hand)."""
    tab = Table(rows)
    out: dict[str, Any] = {
        "explore_class_mix": class_mix(tab),
        "explore_class_split": class_split(tab, draws),
        "explore_decomposition": decomposition(tab, draws),
        "explore_invalidation": invalidation(tab, draws),
        "explore_per_definition": per_definition(tab, draws),
        "explore_json_arm": json_arm(tab, draws),
        "explore_stability": stability(tab, draws),
    }
    summaries = _summaries(run_dirs)
    for key, block in (
        ("explore_noise_floor", noise_floor(summaries)),
        ("explore_cost_e1", cost_e1(summaries)),
    ):
        if block:
            out[key] = block
    out["explore_status"] = STATUS
    return out

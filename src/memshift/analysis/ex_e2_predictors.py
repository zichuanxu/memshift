"""Exploratory baselines and intervals for the writer-switch calibration (decisions.md 2026-09-30;
not pre-registered, no registered verdict changes; keys start with `explore_`).

Path 3 of prereg v1 regresses survival on the per-type agreement rate of the old and the new writer
(`e2.analyse`: D = 0, B = 1 items valid before the switch, intervals from resampled worlds with the
item types held fixed). Added here, for the pooled items of every tier present and for each tier
(so the cross-family switches of block M1 are covered once staged, no change needed):
  predictors, each scored by the calibration slope of logit P(survive) on logit(x) and by AUROC:
    agree     the registered per-type agreement rate (a);
    new_acc   the new writer's per-type accuracy on the calibration set (b);
    old_acc   the old writer's per-type accuracy (c);
    cond      P(new correct | old correct) per type, the plug-in for survival itself (d); a type
              on which the old writer is never right takes the new writer's accuracy (independence);
    nondef    1 for an item outside the definition class and 0 inside it (e);
  the same predictors (a)-(d) on the items outside the definition class only ("nodef"; (f) is agree
    there), so that each baseline is compared on the same items;
  intervals: the registered world bootstrap (same seed, resampling order and quantile rule as
    `e2.boot`, so the agree rows reproduce path 3) and a type-level cluster bootstrap that
    resamples item types (all items of a drawn type, across tiers when pooled) with the same
    seed; AUROC minus agree with both;
  leave-one-type-out estimates, and failure counts per type and tier.
The bootstrap statistics are computed from count tables (items grouped by cluster and predictor
value), which gives the same values as `e2.slope_on_logit` and `e2.auroc` on the expanded items.
"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis.e1 import ITEM_CLASSES
from memshift.analysis.e2 import _logit
from memshift.experiments.common import read_jsonl

Row = dict[str, Any]
PREDICTORS = ("agree", "new_acc", "old_acc", "cond", "nondef")
NODEF_PREDICTORS = ("agree", "new_acc", "old_acc", "cond")
POOLED = "pooled"
DEFINITION_PREFIXES = ITEM_CLASSES["definition"]
QUANTILES = (("lo95", 0.025), ("lo90", 0.05), ("hi90", 0.95), ("hi95", 0.975))


@dataclass(frozen=True)
class Point:
    tier: str
    world: str
    type: str
    definition: bool
    y: float  # survived: valid after the switch
    x: dict[str, float]  # predictor -> value


def type_rates(cal: list[Row]) -> dict[str, dict[str, float]]:
    """Per item type from one tier's calibration rows: agreement and both writers' accuracies."""
    by: dict[str, list[Row]] = defaultdict(list)
    for c in cal:
        by[c["type"]].append(c)
    out = {}
    for t, v in sorted(by.items()):
        n = len(v)
        old = sum(bool(c["old_correct"]) for c in v)
        new = sum(bool(c["new_correct"]) for c in v)
        both = sum(bool(c["old_correct"] and c["new_correct"]) for c in v)
        out[t] = {
            "agree": sum(bool(c["agree"]) for c in v) / n,
            "new_acc": new / n,
            "old_acc": old / n,
            "cond": both / old if old else new / n,
        }
    return out


def load_calibration(run_dir: Path, tiers: list[str]) -> dict[str, list[Row]]:
    """The calibration rows of every tier that has a `calibration_<tier>.jsonl` in `run_dir`."""
    return {
        t: read_jsonl(run_dir / f"calibration_{t}.jsonl")
        for t in tiers
        if (run_dir / f"calibration_{t}.jsonl").exists()
    }


def collect_points(rows: list[Row], cal: dict[str, list[Row]]) -> list[Point]:
    """The D = 0, B = 1 items valid before the switch (path 3's items) of the tiers in `cal`, each
    with the predictor values of its type."""
    rates = {t: type_rates(c) for t, c in cal.items()}
    out = []
    for r in rows:
        ok = r["D"] == 0 and r["B"] == 1 and r["valid_before"] and r["predicted"] is not None
        if not ok or r["tier"] not in rates or r["type"] not in rates[r["tier"]]:
            continue
        definition = str(r.get("slot_id", "")).startswith(DEFINITION_PREFIXES)
        x = {**rates[r["tier"]][r["type"]], "nondef": 0.0 if definition else 1.0}
        x["agree"] = float(r["predicted"])  # the registered value
        out.append(
            Point(r["tier"], r["world_id"], r["type"], definition, float(r["valid_after"]), x)
        )
    return out


# ------------------------------------------------------------------------ statistics from counts
def auroc_counts(pos: np.ndarray, neg: np.ndarray) -> np.ndarray:
    """AUROC (Mann-Whitney, ties half) per row of (B, K) survivor / failure counts over ascending
    predictor values; nan where a class is empty."""
    sp, sn = pos.sum(axis=1), neg.sum(axis=1)
    below = np.cumsum(neg, axis=1) - neg
    wins = (pos * (below + 0.5 * neg)).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = wins / (sp * sn)
    out[(sp == 0) | (sn == 0)] = math.nan
    return np.asarray(out)


def logistic_counts(
    x: np.ndarray, pos: np.ndarray, tot: np.ndarray, iters: int = 50
) -> tuple[np.ndarray, np.ndarray]:
    """Intercept and slope of logit P(y) = a + b x (Newton-Raphson with the ridge and iteration
    rule of `e2.logistic_fit`) for each row of (B, K) counts of survivors `pos` and items `tot` at
    the predictor values `x`; nan where x takes fewer than two values or y one."""
    n = pos.shape[0]
    a, b = np.zeros(n), np.zeros(n)
    active = np.ones(n, dtype=bool)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(a[:, None] + b[:, None] * x[None, :])))
            w = tot * (p * (1 - p) + 1e-9)
            r = pos - tot * p
            g0, g1 = r.sum(axis=1), (r * x).sum(axis=1)
            h00, h01 = w.sum(axis=1) + 1e-6, (w * x).sum(axis=1)
            h11 = (w * x * x).sum(axis=1) + 1e-6
            det = h00 * h11 - h01 * h01
            s0 = np.where(active, (h11 * g0 - h01 * g1) / det, 0.0)
            s1 = np.where(active, (h00 * g1 - h01 * g0) / det, 0.0)
            a, b = a + s0, b + s1
            active &= np.maximum(np.abs(s0), np.abs(s1)) >= 1e-8
            if not active.any():
                break
    distinct = (tot > 0).sum(axis=1)
    sp, st = pos.sum(axis=1), tot.sum(axis=1)
    bad = (distinct < 2) | (sp == 0) | (sp == st)
    a[bad] = math.nan
    b[bad] = math.nan
    return a, b


def multiplicities(n_clusters: int, draws: int, seed: int) -> np.ndarray:
    """(draws, n_clusters) times each cluster is drawn; the draw order is that of `e2.boot` (one
    `randrange(n_clusters)` per cluster per draw from `Random(seed)`)."""
    rng = random.Random(seed)
    m = np.zeros((draws, n_clusters))
    for d in range(draws):
        for _ in range(n_clusters):
            m[d, rng.randrange(n_clusters)] += 1
    return m


class _Tables:
    """Count tables of one predictor over the clusters of a scope."""

    def __init__(self, cluster: np.ndarray, y: np.ndarray, x: np.ndarray, n_clusters: int) -> None:
        ux, ix = np.unique(x, return_inverse=True)
        self.pos_raw = np.zeros((n_clusters, len(ux)))
        self.neg_raw = np.zeros((n_clusters, len(ux)))
        np.add.at(self.pos_raw, (cluster, ix), y)
        np.add.at(self.neg_raw, (cluster, ix), 1 - y)
        lx = np.array([_logit(v) for v in x])
        self.logit_x, il = np.unique(lx, return_inverse=True)
        self.pos_logit = np.zeros((n_clusters, len(self.logit_x)))
        self.tot_logit = np.zeros((n_clusters, len(self.logit_x)))
        np.add.at(self.pos_logit, (cluster, il), y)
        np.add.at(self.tot_logit, (cluster, il), 1.0)

    def stats(self, mult: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """AUROC, intercept and slope for each row of `mult` (cluster multiplicities)."""
        auc = auroc_counts(mult @ self.pos_raw, mult @ self.neg_raw)
        a, b = logistic_counts(self.logit_x, mult @ self.pos_logit, mult @ self.tot_logit)
        return auc, a, b


def _interval(draws: np.ndarray) -> dict[str, Any]:
    vals = np.sort(draws[~np.isnan(draws)])
    out: dict[str, Any] = {"draws": int(len(vals))}
    for name, q in QUANTILES:
        out[name] = float(vals[int(q * (len(vals) - 1))]) if len(vals) else math.nan
    return out


def _cluster_key(cluster: str) -> Any:
    return (lambda p: p.world) if cluster == "world" else (lambda p: p.type)


def _tables(
    points: list[Point], predictors: tuple[str, ...], cluster: str
) -> tuple[list[str], dict[str, _Tables]]:
    key = _cluster_key(cluster)
    names = sorted({key(p) for p in points})
    index = {c: i for i, c in enumerate(names)}
    ci = np.array([index[key(p)] for p in points])
    y = np.array([p.y for p in points])
    tables = {
        pred: _Tables(ci, y, np.array([p.x[pred] for p in points]), len(names))
        for pred in predictors
    }
    return names, tables


def _scope(
    points: list[Point], predictors: tuple[str, ...], draws: int, seed: int
) -> dict[str, Any]:
    """Slope and AUROC of each predictor on `points` with world and type bootstrap intervals."""
    res: dict[str, Any] = {
        "n_items": len(points),
        "n_failures": int(sum(1 - p.y for p in points)),
        "n_worlds": len({p.world for p in points}),
        "n_types": len({p.type for p in points}),
        "predictors": {},
    }
    if not points:
        return res
    per: dict[str, dict[str, Any]] = {p: {} for p in predictors}
    for cluster in ("world", "type"):
        names, tables = _tables(points, predictors, cluster)
        one = np.ones((1, len(names)))
        many = multiplicities(len(names), draws, seed)
        base = tables["agree"].stats(many)[0]
        for pred in predictors:
            est = tables[pred].stats(one)
            boot = tables[pred].stats(many)
            entry = per[pred]
            entry.setdefault("slope", {"est": float(est[2][0]), "intercept": float(est[1][0])})
            entry.setdefault("auroc", {"est": float(est[0][0])})
            entry["slope"][cluster] = _interval(boot[2])
            entry["auroc"][cluster] = _interval(boot[0])
            if pred != "agree":
                entry.setdefault("auroc_minus_agree", {"est": math.nan})[cluster] = _interval(
                    boot[0] - base
                )
    agree_auc = per["agree"]["auroc"]["est"]
    for pred in predictors:
        if pred != "agree":
            per[pred]["auroc_minus_agree"]["est"] = per[pred]["auroc"]["est"] - agree_auc
    res["predictors"] = per
    return res


def _leave_one_type_out(points: list[Point], predictors: tuple[str, ...]) -> dict[str, Any]:
    if not points:
        return {"types": {}, "range": {}}
    names, tables = _tables(points, predictors, "type")
    drop = 1 - np.eye(len(names))
    stats = {pred: tables[pred].stats(drop) for pred in predictors}
    out: dict[str, Any] = {"types": {}, "range": {}}
    for i, t in enumerate(names):
        mine = [p for p in points if p.type == t]
        out["types"][t] = {
            "n_dropped": len(mine),
            "failures_dropped": int(sum(1 - p.y for p in mine)),
            "predictors": {
                pred: {"slope": float(stats[pred][2][i]), "auroc": float(stats[pred][0][i])}
                for pred in predictors
            },
        }
    for pred in predictors:
        rng: dict[str, Any] = {}
        for stat, col in (("slope", 2), ("auroc", 0)):
            vals = [
                (float(v), names[i]) for i, v in enumerate(stats[pred][col]) if not math.isnan(v)
            ]
            if vals:
                lo, hi = min(vals), max(vals)
                rng.update(
                    {f"{stat}_min": lo[0], f"{stat}_min_type": lo[1]}
                    | {f"{stat}_max": hi[0], f"{stat}_max_type": hi[1]}
                )
        out["range"][pred] = rng
    return out


def failures(points: list[Point]) -> dict[str, Any]:
    """Items and failures (valid before, not after) per item type, per scope and overall."""
    out: dict[str, Any] = {}
    scopes = {POOLED: points} | {
        t: [p for p in points if p.tier == t] for t in sorted({p.tier for p in points})
    }
    for scope, pts in scopes.items():
        per: dict[str, dict[str, Any]] = {}
        for p in pts:
            e = per.setdefault(p.type, {"n": 0, "failures": 0, "definition": p.definition})
            e["n"] += 1
            e["failures"] += int(p.y == 0)
        out[scope] = dict(sorted(per.items()))
    cells = [(t, ty, e) for t in sorted({p.tier for p in points}) for ty, e in out[t].items()]
    pooled = out[POOLED]
    ranked = sorted(pooled, key=lambda t: (-pooled[t]["failures"], t))
    total = sum(e["failures"] for e in pooled.values())
    top = ranked[:2] if total else []
    out["summary"] = {
        "n_items": len(points),
        "n_failures": total,
        "definition_failures": sum(e["failures"] for e in pooled.values() if e["definition"]),
        "top_two_types": top,
        "top_two_failures": sum(pooled[t]["failures"] for t in top),
        "type_tier_cells": len(cells),
        "type_tier_cells_with_failures": sum(1 for *_, e in cells if e["failures"]),
        "types": len(pooled),
    }
    return out


def analyse(run_dir: Path, rows: list[Row], draws: int = 2000, seed: int = 996) -> dict[str, Any]:
    """The `explore_writer_*` and `explore_failures_by_type` fields for the E2 survival rows."""
    tiers = sorted({r["tier"] for r in rows})
    cal = load_calibration(run_dir, tiers)
    points = collect_points(rows, cal)
    nodef_points = [p for p in points if not p.definition]
    scopes = [(POOLED, points)] + [
        (t, [p for p in points if p.tier == t])
        for t in sorted(cal)
        if any(p.tier == t for p in points)
    ]
    predictors: dict[str, Any] = {"all": {}, "nodef": {}}
    loto: dict[str, Any] = {}
    for scope, pts in scopes:
        predictors["all"][scope] = _scope(pts, PREDICTORS, draws, seed)
        predictors["nodef"][scope] = _scope(
            [p for p in pts if not p.definition], NODEF_PREDICTORS, draws, seed
        )
        loto[scope] = _leave_one_type_out(pts, PREDICTORS)
    return {
        "explore_writer_predictors": {
            "tiers": [t for t, _ in scopes[1:]],
            "tiers_without_calibration": [t for t in tiers if t not in cal],
            "draws": draws,
            "seed": seed,
            "n_items_nodef": len(nodef_points),
            "sets": predictors,
        },
        "explore_writer_loto": loto,
        "explore_failures_by_type": failures(points),
    }

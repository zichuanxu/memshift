"""E2 analysis (plan §2.3, §2.4 path 3, §6 E2).

Retention per (tier, cell) = P(valid_after | valid_before).
Calibration (D=0, B=1, items valid before the switch): survival y = valid_after is regressed on
the per-type agreement rate p from the disjoint calibration set, logit(P(y)) = a + b logit(p);
b is the calibration slope (b > 0: agreement predicts survival; b = 1: calibrated). AUROC of p for
y uses the Mann-Whitney statistic with ties counted half. Intervals: world-level bootstrap.
D=1 survival is compared with the new writer's checker pass rate on the calibration set.

prereg v1 §5 (T1): per tier, delta = Retention(HoldMem(1,1), source epoch) - 1 with a 90% world
bootstrap interval; equivalent when it lies inside +/- T1_MARGIN (TOST, alpha = 0.05).
`explore_pooled_curve` is the pooled logistic fit of path 3 (intercept, slope and predicted survival
on a grid of agreement rates, with a pointwise 90% world-bootstrap band) for the calibration figure.
Exploratory: `explore_t1_cells` applies the T1 computation to every cell of the tier (per tier and
system: delta, margin, equivalent), e.g. HoldMem(0,1) at source-epoch binding.
prereg v1 §3 (path 3): the primary decision pools the D=0, B=1 items of all tiers present; each
tier is also reported, and a tier whose slope estimate is <= 0 is listed as a caveat.
Exploratory (decisions.md 2026-09-30; `ex_e2_predictors`, `ex_tests`): `explore_writer_predictors`
scores the registered agreement rate against baselines (the writers' per-type accuracy, P(new
correct | old correct), a definition-item indicator) with world and type-level intervals,
`explore_writer_loto` and `explore_failures_by_type` give leave-one-type-out estimates and failure
counts, and `explore_survival_recovery`, `explore_t1_eps` and `explore_t1_by_class` add survival,
recovery, the net change, the verdict at several margins and T1 by item class.
"""

from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import ex_tests
from memshift.analysis.e1 import ITEM_CLASSES
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import paired_diff, tost, world_bootstrap

Row = dict[str, Any]
Stat = Callable[[list[float], list[float]], float]

T1_MARGIN = 0.03  # prereg v1 §5: epsilon of plan §4.6
T1_SYSTEM = "holdmem_D1B1_source"


def _logit(p: float) -> float:
    p = min(max(p, 0.01), 0.99)
    return math.log(p / (1 - p))


def logistic_fit(x: list[float], y: list[float], iters: int = 50) -> tuple[float, float]:
    """Intercept a and slope b of logit(P(y)) = a + b x by Newton-Raphson; nan when undefined."""
    if len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan, math.nan
    xa = np.column_stack([np.ones(len(x)), np.array(x)])
    ya = np.array(y, dtype=float)
    beta = np.zeros(2)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(xa @ beta)))
        w = p * (1 - p) + 1e-9
        grad = xa.T @ (ya - p)
        hess = (xa * w[:, None]).T @ xa + 1e-6 * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.max(np.abs(step)) < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def logistic_slope(x: list[float], y: list[float], iters: int = 50) -> float:
    """Slope b of logit(P(y)) = a + b x by Newton-Raphson; nan when undefined."""
    return logistic_fit(x, y, iters)[1]


def auroc(score: list[float], y: list[float]) -> float:
    pos = [s for s, t in zip(score, y, strict=True) if t]
    neg = [s for s, t in zip(score, y, strict=True) if not t]
    if not pos or not neg:
        return math.nan
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def slope_on_logit(x: list[float], y: list[float]) -> float:
    return logistic_slope([_logit(v) for v in x], y)


def boot(
    by_world: dict[str, list[tuple[float, float]]], fn: Stat, draws: int = 2000, seed: int = 996
) -> dict[str, float]:
    worlds = sorted(by_world)
    pts = [p for w in worlds for p in by_world[w]]
    est = fn([p[0] for p in pts], [p[1] for p in pts]) if pts else math.nan
    rng = random.Random(seed)
    vals = []
    for _ in range(draws if worlds else 0):
        sample = [p for _ in worlds for p in by_world[worlds[rng.randrange(len(worlds))]]]
        v = fn([p[0] for p in sample], [p[1] for p in sample])
        if not math.isnan(v):
            vals.append(v)
    vals.sort()
    res: dict[str, float] = {"est": est, "n_items": len(pts), "n_worlds": len(worlds)}
    for name, q in (("lo95", 0.025), ("lo90", 0.05), ("hi90", 0.95), ("hi95", 0.975)):
        res[name] = vals[int(q * (len(vals) - 1))] if vals else math.nan
    return res


CURVE_GRID = [round(0.20 + 0.01 * i, 2) for i in range(80)]  # agreement 0.20 .. 0.99


def _survival(a: float, b: float, p: float) -> float:
    return 1 / (1 + math.exp(-(a + b * _logit(p))))


def calibration_curve(
    by_world: dict[str, list[tuple[float, float]]],
    grid: list[float] = CURVE_GRID,
    draws: int = 2000,
    seed: int = 996,
) -> dict[str, Any]:
    """The fitted calibration curve behind `slope_on_logit` on the same points: predicted survival
    sigmoid(a + b logit(p)) on `grid` with its pointwise 90% band. The band resamples worlds
    exactly as `boot` does (same seed and draw order), so it and the 90% interval of the slope
    come from the same resamples."""
    worlds = sorted(by_world)
    pts = [p for w in worlds for p in by_world[w]]
    a, b = (
        logistic_fit([_logit(p[0]) for p in pts], [p[1] for p in pts]) if pts else (math.nan,) * 2
    )
    curve = [_survival(a, b, g) if not math.isnan(b) else math.nan for g in grid]
    rng = random.Random(seed)
    draws_curve: list[list[float]] = []
    for _ in range(draws if worlds else 0):
        sample = [p for _ in worlds for p in by_world[worlds[rng.randrange(len(worlds))]]]
        da, db = logistic_fit([_logit(p[0]) for p in sample], [p[1] for p in sample])
        if not math.isnan(db):
            draws_curve.append([_survival(da, db, g) for g in grid])

    def band(q: float) -> list[float]:
        if not draws_curve:
            return [math.nan] * len(grid)
        return [
            sorted(c[i] for c in draws_curve)[int(q * (len(draws_curve) - 1))]
            for i in range(len(grid))
        ]

    return {
        "grid": grid,
        "est": curve,
        "lo90": band(0.05),
        "hi90": band(0.95),
        "intercept": a,
        "slope": b,
        "n_items": len(pts),
        "n_worlds": len(worlds),
    }


def _agreement(cal: list[Row]) -> dict[str, dict[str, float]]:
    by: dict[str, list[Row]] = defaultdict(list)
    for c in cal:
        by[c["type"]].append(c)
    return {
        t: {
            "n": len(v),
            "agree": sum(c["agree"] for c in v) / len(v),
            "old_correct": sum(c["old_correct"] for c in v) / len(v),
            "new_correct": sum(c["new_correct"] for c in v) / len(v),
        }
        for t, v in sorted(by.items())
    }


def analyse(run_dir: Path) -> dict[str, Any]:
    rows = read_jsonl(run_dir / "survival.jsonl")
    out: dict[str, Any] = {"retention": {}, "calibration": {}, "d1_vs_checker": {}, "t1": {}}
    pooled: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for tier in sorted({r["tier"] for r in rows}):
        tr = [r for r in rows if r["tier"] == tier]
        for system in sorted({r["system"] for r in tr}):
            by_w: dict[str, list[float]] = defaultdict(list)
            for r in tr:
                if r["system"] == system and r["valid_before"]:
                    by_w[r["world_id"]].append(float(r["valid_after"]))
            out["retention"][f"{tier}:{system}"] = world_bootstrap(by_w).to_json()
            # Exploratory (decisions.md 2026-09-28): retention by item class, 90% intervals
            for cls, prefixes in ITEM_CLASSES.items():
                by_c: dict[str, list[float]] = defaultdict(list)
                for r in tr:
                    ok = r["system"] == system and r["valid_before"]
                    if ok and str(r.get("slot_id", "")).startswith(prefixes):
                        by_c[r["world_id"]].append(float(r["valid_after"]))
                est = world_bootstrap(by_c, alpha=0.10).to_json()
                out.setdefault("explore_retention_by_class", {})[f"{tier}:{system}:{cls}"] = est
            delta = world_bootstrap({w: [v - 1 for v in xs] for w, xs in by_w.items()}, alpha=0.10)
            t1 = {
                "delta_90": delta.to_json(),
                "margin": T1_MARGIN,
                "equivalent": tost(delta, T1_MARGIN),
            }
            if system == T1_SYSTEM:
                out["t1"][tier] = t1
            # Exploratory (not pre-registered): the same test for every cell of the tier (the
            # pre-registered one included), so that the paper can say which of D and B a layer
            # needs to satisfy T1. B = 0 cells keep retention 1 by construction and pass
            # trivially (v1 §4: not "updated" but "unchanged").
            out.setdefault("explore_t1_cells", {}).setdefault(tier, {})[system] = t1
        pts: dict[str, list[tuple[float, float]]] = defaultdict(list)
        for r in tr:
            if r["D"] == 0 and r["B"] == 1 and r["valid_before"] and r["predicted"] is not None:
                pts[r["world_id"]].append((float(r["predicted"]), float(r["valid_after"])))
                pooled[r["world_id"]].append((float(r["predicted"]), float(r["valid_after"])))
        per_type: dict[str, list[Row]] = defaultdict(list)
        for r in tr:
            if r["D"] == 0 and r["B"] == 1 and r["valid_before"] and r["predicted"] is not None:
                per_type[r["type"]].append(r)
        out["calibration"][tier] = {
            "slope": boot(pts, slope_on_logit),
            "auroc": boot(pts, auroc),
            "per_type": {
                t: {
                    "predicted": v[0]["predicted"],
                    "observed": sum(r["valid_after"] for r in v) / len(v),
                    "n": len(v),
                }
                for t, v in sorted(per_type.items())
            },
        }
        cal_path = run_dir / f"calibration_{tier}.jsonl"
        if cal_path.exists():
            cal = read_jsonl(cal_path)
            d1 = [r for r in tr if r["D"] == 1 and r["B"] == 1 and r["valid_before"]]
            out["d1_vs_checker"][tier] = {
                "checker_pass_rate_calibration": sum(c["new_check_pass"] for c in cal) / len(cal),
                "new_writer_correct_calibration": sum(c["new_correct"] for c in cal) / len(cal),
                "d1_b1_survival": sum(r["valid_after"] for r in d1) / max(1, len(d1)),
                "agreement_by_type": _agreement(cal),
            }
    # M1-b (prereg v2 §2): per switch, survival of D=1 minus D=0 at source-epoch binding, paired
    # by item among items valid before the switch in both cells; holds when the estimate is >= 0
    # (decisions.md 2026-09-29)
    for tier in sorted({r["tier"] for r in rows}):
        surv: dict[int, dict[str, dict[str, float]]] = {0: defaultdict(dict), 1: defaultdict(dict)}
        for r in rows:
            if r["tier"] == tier and r["B"] == 1 and r["valid_before"]:
                surv[r["D"]][r["world_id"]][r["slot_id"]] = float(r["valid_after"])
        est = paired_diff(surv[1], surv[0]).to_json()
        out.setdefault("m1b_d_protects_switch", {})[tier] = {
            "d1_minus_d0_survival": est,
            "holds": bool(est["est"] >= 0) if est["n_items"] else None,
        }
    out["path3_quantitative_prediction_holds"] = {
        t: bool(c["slope"]["lo90"] > 0 and c["auroc"]["est"] >= 0.60)
        for t, c in out["calibration"].items()
        if not math.isnan(c["slope"]["est"])
    }
    tiers = sorted(out["calibration"])
    slope, auc = boot(pooled, slope_on_logit), boot(pooled, auroc)
    # Exploratory (plot support, not a statistic): the pooled fit whose slope and AUROC path 3 uses
    out["explore_pooled_curve"] = calibration_curve(pooled)
    out["path3"] = {
        "tiers": tiers,
        "pooled_slope": slope,
        "pooled_auroc": auc,
        "triggered": not (slope["lo90"] > 0 and auc["est"] >= 0.60),
        "tier_caveats": [t for t in tiers if not out["calibration"][t]["slope"]["est"] > 0],
    }
    # Exploratory (decisions.md 2026-09-30): writer-switch predictors with baselines and type-level
    # intervals, T1 survival / recovery / net change, the eps grid and T1 by item class
    from memshift.analysis import ex_e2_predictors

    out.update(ex_e2_predictors.analyse(run_dir, rows))
    out.update(ex_tests.t1_sensitivity(rows, T1_MARGIN))
    d = run_dir / "analysis"
    d.mkdir(exist_ok=True)
    (d / "e2_metrics.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    import sys

    res = analyse(Path(sys.argv[1]))
    keys = ("t1", "path3", "path3_quantitative_prediction_holds")
    print(json.dumps({k: res[k] for k in keys}, indent=1, default=str))

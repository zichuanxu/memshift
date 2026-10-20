"""Further analyses of the E1 family: out-of-sample baselines, transfer across horizons, the items
a key missed and the zero system. Every key starts with `explore_`. Inputs are staged analyses:
`role:STAGE_DIR`, each with analysis/e1_item_table.jsonl.

explore_oos_baselines   The out-of-sample test of Eq. (1) (`ex_e1_cross.oos`) with a baseline
                        that has no class terms: the pooled development cell mean, each cell's
                        Update over all classes of the fit worlds, with no class terms and no
                        class-mix reweighting. Predictors, per cell and class set (definition,
                        other, all):
                          cell              each cell's Update per class, weighted by the class
                                            mix of the target worlds (the paper's "cell" terms)
                          structural        Eq. (1) with r by D, m by the key, a the B=0 Update
                                            (write path) or the accuracy of the missed items
                          literal           Eq. (1) with one stale-value accuracy for every cell
                          pooled_cell_mean  the baseline: the fit worlds' Update of the cell over
                                            all item classes
                        for dev_to_main, dev_to_rep and main_to_rep (six cells per branch)
explore_horizon_transfer  the same from the twelve-month main study to the three-month worlds:
                        the four source-epoch cells, predictors cell and pooled_cell_mean; errors
                        per branch, class and cell with world-bootstrap intervals
explore_missed_items    what binding does to the items a key did not invalidate, per branch
                        and class at D=0 and D=1: the accuracy of the changed items the cited
                        cluster key missed (B=1) against the Update of the B=0 cell on all changed
                        items and on the same missed items; the cited cell's Update against B=0
explore_zero_system     Update and Stability of the zero system, plain RAG, the oracle context and
                        the HoldMem cells after the shift, per branch and class, and the paired
                        differences of the zero system against HoldMem(1,1) and HoldMem(0,0)

Usage: python -m memshift.analysis.ex_e1_more OUT dev:STAGE devorphan:STAGE main:STAGE rep:STAGE
           short:STAGE [gemma:STAGE llama:STAGE gptoss:STAGE]
Writes OUT/analysis/ex_e1_more.json and OUT/manifest.json.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_e1_cross as cross
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import read_jsonl

Est = ex.Est
DRAWS = ex.DRAWS
SEED = ex.SEED
WITHIN_PP = cross.WITHIN_PP
CELLS6 = cross.FIT_CELLS  # D0B0, D1B0, D0B1_source, D1B1_source, D0B1_cited, D1B1_cited
CELLS4 = CELLS6[:4]
CLASSES = cross.OOS_CLASSES  # definition, other, all
PREDICTOR_LABELS = {
    "cell": "class terms: each cell's Update per class, weighted by the target's class mix",
    "structural": "Eq. (1) with r by D, m by the key, a the B=0 Update or the missed accuracy",
    "literal": "Eq. (1) with one stale-value accuracy for every cell",
    "pooled_cell_mean": "each cell's Update over all classes of the fit worlds (no class terms, "
    "no class-mix reweighting)",
}
PAIRS = {
    "dev_to_main": ("dev", "main"),
    "dev_to_rep": ("dev", "rep"),
    "main_to_rep": ("main", "rep"),
}
HORIZON_PAIR = ("main", "short")
SYSTEMS = (
    "zero",
    "rag",
    "oracle",
    "holdmem_D0B0",
    "holdmem_D1B0",
    "holdmem_D0B1_source",
    "holdmem_D1B1_source",
)
MISSED_CLASSES = ("definition", "other", "all")


# --------------------------------------------------------------------------------- transfer
def _predict(tf: np.ndarray, cells: Sequence[str], predictors: Sequence[str]) -> Any:
    """Predicted Update per predictor and cell and base class (..., classes) from fit totals
    tf (..., cells, classes, stats)."""
    out: dict[str, dict[str, np.ndarray]] = {}
    if len(cells) == len(CELLS6):
        full = cross.predict(tf)
        out |= {p: full[p] for p in predictors if p in full}
    else:
        out["cell"] = {c: ex.cell_terms(tf[..., i, :, :])["U"] for i, c in enumerate(cells)}
    if "pooled_cell_mean" in predictors:
        pooled = {}
        for i, c in enumerate(cells):
            tot = tf[..., i, :, :].sum(axis=-2)  # (..., stats): all classes of the fit worlds
            with np.errstate(divide="ignore", invalid="ignore"):
                u = tot[..., 1] / tot[..., 0]
            pooled[c] = np.broadcast_to(u[..., None], (*u.shape, len(ex.BASE_CLASSES)))
        out["pooled_cell_mean"] = pooled
    return out


def _observed(tt: np.ndarray, cells: Sequence[str], cls: str) -> dict[str, np.ndarray]:
    tot = ex.class_totals(tt, cls)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {c: tot[..., i, 1] / tot[..., i, 0] for i, c in enumerate(cells)}


def _errors(
    fit: tuple[np.ndarray, np.ndarray],
    target: tuple[np.ndarray, np.ndarray],
    cells: Sequence[str],
    predictors: Sequence[str],
) -> dict[tuple[str, str, str, str], tuple[Any, Any]]:
    """(label, class, predictor, cell) -> (predicted, observed) for the point totals and for the
    bootstrap draws (`fit` and `target` are (point totals, draw totals))."""
    out: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
    for label, (tf, tt) in (("point", (fit[0], target[0])), ("draws", (fit[1], target[1]))):
        preds = _predict(tf, cells, predictors)
        w = tt[..., 0, :, 0]  # changed item-times per base class (the D0B0 cell)
        for cls in CLASSES[:3]:
            obs = _observed(tt, cells, cls)
            for p in predictors:
                for c in cells:
                    out[label, cls, p, c] = (cross._combine(preds[p][c], w, cls), obs[c])
    return out


def _summarise(
    q: Mapping[tuple[str, str, str, str], tuple[Any, Any]],
    cells: Sequence[str],
    predictors: Sequence[str],
    n_worlds: int,
    n_items: Mapping[str, int],
) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for cls in CLASSES[:3]:
        res[cls] = {}
        for p in predictors:
            node: dict[str, Any] = {"cells": {}}
            pts, drs = [], []
            for c in cells:
                pr, ob = q["point", cls, p, c]
                prd, obd = q["draws", cls, p, c]
                err = np.asarray(obd - prd, float)
                node["cells"][c] = {
                    "predicted": float(pr),
                    "observed": float(ob),
                    "error": ex.interval(float(ob - pr), err, n_worlds, n_items[cls]),
                }
                pts.append(abs(float(ob - pr)))
                drs.append(np.abs(err))
            node["mean_abs_error"] = ex.interval(
                float(np.nanmean(pts)), np.nanmean(np.stack(drs), axis=0), n_worlds, n_items[cls]
            )
            node["cells_within_5pp"] = int(np.sum(np.array(pts) <= WITHIN_PP))
            node["n_cells"] = len(cells)
            res[cls][p] = node
    return res


def transfer(
    fit: ex.Table,
    target: ex.Table,
    cells: Sequence[str],
    predictors: Sequence[str],
    draws: int = DRAWS,
) -> dict[str, Any]:
    """Terms fitted on `fit`, Update predicted on `target`, per branch and pooled over the
    branches. The fit worlds and the target worlds are resampled, as in `ex_e1_cross.oos`."""
    grans = ["source"] if len(cells) == len(CELLS4) else list(cross.GRANS)
    out: dict[str, Any] = {"branches": {}}
    pooled: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    pooled_pt: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_target = 0
    for b in sorted(set(fit.branches) & set(target.branches)):
        fw, fa, _ = ex.cell_arrays(fit, b, "D1", grans)
        tw, ta, _ = ex.cell_arrays(target, b, "D1", grans)
        if not fw or not tw:
            continue
        pt_f, pt_t = fa.sum(axis=0), ta.sum(axis=0)
        dr_f = fa[ex.boot_index(len(fw), draws, SEED + 1)].sum(axis=1)
        dr_t = ta[ex.boot_index(len(tw), draws, SEED)].sum(axis=1)
        q = _errors((pt_f, dr_f), (pt_t, dr_t), cells, predictors)
        n_items = {cls: int(ex.class_totals(pt_t, cls)[0, 0]) for cls in CLASSES[:3]}
        out["branches"][b] = _summarise(q, cells, predictors, len(tw), n_items)
        n_target = len(tw)
        for cls in CLASSES[:3]:
            for p in predictors:
                for c in cells:
                    pr, ob = q["point", cls, p, c]
                    prd, obd = q["draws", cls, p, c]
                    pooled_pt[cls, p].append(abs(float(ob - pr)))
                    pooled[cls, p].append(np.abs(np.asarray(obd - prd, float)))
    if len(out["branches"]) == 3:  # the same worlds are resampled in every branch
        out["pooled_over_branches"] = {}
        for (cls, p), pts in pooled_pt.items():
            out["pooled_over_branches"].setdefault(cls, {})[p] = {
                "mean_abs_error": ex.interval(
                    float(np.nanmean(pts)),
                    np.nanmean(np.stack(pooled[cls, p]), axis=0),
                    n_target,
                    0,
                ),
                "cells_within_5pp": int(np.sum(np.array(pts) <= WITHIN_PP)),
                "n_cells": len(pts),
            }
    return out


def oos_baselines(tables: Mapping[str, ex.Table], draws: int = DRAWS) -> dict[str, Any]:
    """`explore_oos_baselines`: the three fit/target pairs with the four predictors."""
    res: dict[str, Any] = {
        "predictors": PREDICTOR_LABELS,
        "cells": list(CELLS6),
        "within_pp": 100 * WITHIN_PP,
    }
    for name, (fit, target) in PAIRS.items():
        if fit in tables and target in tables:
            res[name] = transfer(
                tables[fit], tables[target], CELLS6, tuple(PREDICTOR_LABELS), draws
            )
    return res


def horizon_transfer(tables: Mapping[str, ex.Table], draws: int = DRAWS) -> dict[str, Any]:
    """`explore_horizon_transfer`: twelve-month terms against the three-month worlds."""
    fit, target = HORIZON_PAIR
    if fit not in tables or target not in tables:
        return {}
    preds = ("cell", "pooled_cell_mean")
    return {
        "predictors": {p: PREDICTOR_LABELS[p] for p in preds},
        "cells": list(CELLS4),
        "within_pp": 100 * WITHIN_PP,
        "main_to_short": transfer(tables[fit], tables[target], CELLS4, preds, draws),
    }


# ------------------------------------------------------------------------------ missed items
MISS_STATS = (
    "n",  # changed item-times of the class at the cited cell
    "n_missed",  # of which the cited key did not invalidate
    "cv_missed",  # B=1 cited: ContractValid summed over the missed ones
    "cv_b0_missed",  # B=0: the same item-times
    "cv_detected",  # B=1 cited: summed over the invalidated ones
    "cv_b0_detected",  # B=0: the same item-times
    "cv_cited",  # B=1 cited: over all changed item-times
    "cv_b0",  # B=0: over all changed item-times
)


def _miss_arrays(tab: ex.Table, branch: str, d: int) -> tuple[list[str], np.ndarray]:
    """Per-world sums of `MISS_STATS` by base class for one branch and D."""
    cited, b0 = f"holdmem_D{d}B1_cited", f"holdmem_D{d}B0"
    base = {
        (r["world_id"], r["fact_id"], r["k"]): r
        for r in tab.rows(b0, branch)
        if r["changed"] and r.get("cv") is not None
    }
    worlds = sorted({r["world_id"] for r in tab.rows(cited, branch)})
    pos = {w: i for i, w in enumerate(worlds)}
    arr = np.zeros((len(worlds), len(ex.BASE_CLASSES), len(MISS_STATS)))
    for r in tab.rows(cited, branch):
        key = (r["world_id"], r["fact_id"], r["k"])
        if not r["changed"] or r.get("cv") is None or key not in base:
            continue
        cv, cv0 = float(r["cv"]), float(base[key]["cv"])
        miss = not r.get("detected")
        arr[pos[r["world_id"]], ex.BASE_CLASSES.index(ex.item_class(r["fact_id"]))] += (
            1.0,
            float(miss),
            cv * miss,
            cv0 * miss,
            cv * (not miss),
            cv0 * (not miss),
            cv,
            cv0,
        )
    return worlds, arr


def _miss_stats(t: np.ndarray) -> dict[str, np.ndarray]:
    n, n_miss, cv_miss, cv0_miss, cv_det, cv0_det, cv_cited, cv0 = (
        t[..., i] for i in range(len(MISS_STATS))
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "missed_share": n_miss / n,
            "a_missed_B1_cited": cv_miss / n_miss,
            "a_missed_B0": cv0_miss / n_miss,
            "a_missed_B1_minus_B0": (cv_miss - cv0_miss) / n_miss,
            "r_invalidated_B1_cited": cv_det / (n - n_miss),
            "r_invalidated_B0": cv0_det / (n - n_miss),
            "update_B1_cited": cv_cited / n,
            "update_B0": cv0 / n,
            "B_effect_cited": (cv_cited - cv0) / n,
        }


def missed_items(tab: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """`explore_missed_items[branch][D<d>][class]`: estimates with world-bootstrap intervals; a
    class with no missed item has undefined missed-item accuracies (NaN)."""
    out: dict[str, Any] = {}
    for b in tab.branches:
        for d in (0, 1):
            if f"holdmem_D{d}B1_cited" not in tab.systems or f"holdmem_D{d}B0" not in tab.systems:
                continue
            worlds, arr = _miss_arrays(tab, b, d)
            if not worlds:
                continue
            drawn = arr[ex.boot_index(len(worlds), draws)].sum(axis=1)
            for cls in MISSED_CLASSES:
                idx = [ex.BASE_CLASSES.index(c) for c in ex.CLASS_SETS[cls]]
                pt = arr.sum(axis=0)[idx].sum(axis=0)
                dr = drawn[:, idx, :].sum(axis=1)
                est, vals = _miss_stats(pt), _miss_stats(dr)
                n = int(pt[0])
                node: dict[str, Any] = {"changed_item_times": n, "missed_item_times": int(pt[1])}
                for k, v in est.items():
                    node[k] = ex.interval(float(v), np.asarray(vals[k], float), len(worlds), n)
                out.setdefault(b, {}).setdefault(f"D{d}", {})[cls] = node
    return out


# ------------------------------------------------------------------------------- zero system
def zero_system(tab: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """`explore_zero_system[branch][class]`: Update (changed items) and Stability (unchanged
    items) of each post-shift system, and the paired differences of the zero system against
    HoldMem(1,1) and HoldMem(0,0) on the same item-times."""
    out: dict[str, Any] = {}
    present = [s for s in SYSTEMS if s in tab.systems]
    zero, hm11, hm00 = "zero", "holdmem_D1B1_source", "holdmem_D0B0"
    for b in tab.branches:
        for cls in ex.CLASS_SETS:
            ch, un = ex._changed(cls), ex._unchanged(cls)
            node: dict[str, Any] = {"update": {}, "stability": {}, "zero_minus": {}}
            for s in present:
                node["update"][s] = ex.vec_mean(tab.vec(s, b, "cv", ch), draws)
                node["stability"][s] = ex.vec_mean(tab.vec(s, b, "cv", un), draws)
            for label, ref in (("holdmem_D1B1", hm11), ("holdmem_D0B0", hm00)):
                if zero in present and ref in present:
                    node["zero_minus"][label] = {
                        "update": ex.paired_estimate(
                            tab.vec(zero, b, "cv", ch), tab.vec(ref, b, "cv", ch), draws
                        ),
                        "stability": ex.paired_estimate(
                            tab.vec(zero, b, "cv", un), tab.vec(ref, b, "cv", un), draws
                        ),
                    }
            out.setdefault(b, {})[cls] = node
    return out


# ------------------------------------------------------------------------------------ driver
def analyse_tables(tables: Mapping[str, ex.Table], draws: int = DRAWS) -> dict[str, Any]:
    res: dict[str, Any] = {
        "explore_status": ex.STATUS,
        "explore_draws": draws,
        "explore_seed": SEED,
    }
    res["explore_oos_baselines"] = oos_baselines(tables, draws)
    if hl := horizon_transfer(tables, draws):
        res["explore_horizon_transfer"] = hl
    res["explore_missed_items"] = {
        k: missed_items(t, draws) for k, t in tables.items() if k in ("main", "rep")
    }
    res["explore_zero_system"] = {
        k: zero_system(t, draws)
        for k, t in tables.items()
        if k in ("main", "rep", "gemma", "llama", "gptoss")
    }
    return res


def load_tables(stages: Mapping[str, Sequence[Path]]) -> dict[str, ex.Table]:
    tables: dict[str, ex.Table] = {}
    dev = [*stages.get("dev", []), *stages.get("devorphan", [])]
    if dev:
        tables["dev"] = ex.Table(
            [r for d in dev for r in read_jsonl(d / "analysis" / "e1_item_table.jsonl")]
        )
    for role in ("main", "rep", "short", "gemma", "llama", "gptoss"):
        if stages.get(role):
            tables[role] = ex.Table(
                read_jsonl(stages[role][0] / "analysis" / "e1_item_table.jsonl")
            )
    return tables


def main() -> None:
    out, *specs = sys.argv[1:]
    stages: dict[str, list[Path]] = {}
    for s in specs:
        role, path = s.split(":", 1)
        stages.setdefault(role, []).append(Path(path))
    if not stages:
        raise SystemExit(__doc__)
    res = analyse_tables(load_tables(stages))
    o = Path(out)
    (o / "analysis").mkdir(parents=True, exist_ok=True)
    (o / "analysis" / "ex_e1_more.json").write_text(json.dumps(res, indent=1))
    write_stage_manifest(o, "ex_e1_more", [p for ps in stages.values() for p in ps], {})
    print(json.dumps({k: list(v) if isinstance(v, dict) else v for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()

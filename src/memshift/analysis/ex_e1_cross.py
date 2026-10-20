"""Analyses of the E1 family that need more than one stage. Inputs are staged analyses:
`role:STAGE_DIR`, with analysis/e1_metrics.json and, where an item table is needed,
analysis/e1_item_table.jsonl.

explore_oos         Out-of-sample test of Eq. (1). The terms m, r, a of each cell and class are
                    fitted on one set of worlds (the ten development worlds; also the twenty test
                    worlds) and Update is predicted on another set from those terms and its class
                    mix:
                      cell        the fitted Update of the cell (what the terms give cell by cell)
                      structural  Eq. (1) with r fixed by D (source-epoch cells), m by the key and a
                                  the B=0 Update for B=0 (the write path) and the accuracy of the
                                  missed items for B=1
                      literal     Eq. (1) as the paper states it: one stale-value accuracy a for
                                  every cell, r by D, m by the key
                    errors (observed - predicted) and their mean absolute value over the six cells,
                    with intervals from resampling the fit worlds and the target worlds. The
                    interval of a mean absolute error is the percentile interval of |error| over
                    the draws: it lies above the point value where errors are near zero, so the
                    signed-error intervals per cell are the ones to read.
explore_replication Which of the 21 source-epoch contrasts are identities by construction and how
                    many of the rest replicate; the class-stratified replication on R; Holm on
                    the contrasts that are not detection identities, exact bootstrap p, 10,000
                    draws.
explore_short       The short (three-month) worlds: the B effect at each D predicted from the
                    twelve-month terms and that study's class mix against what the short worlds
                    show.
explore_cost        Maintenance against serving tokens of the E3 cost model (an account of the
                    maintenance path, not an execution), per horizon and system, against the zero
                    system.
explore_noise_floor Values and answers that differ between two ingestions, pooled over stages.

Usage: python -m memshift.analysis.ex_e1_cross OUT dev:STAGE devorphan:STAGE main:STAGE rep:STAGE
           short:STAGE e3:STAGE e3rep:STAGE noisepool:STAGE [noisepool:STAGE ...] [noise:STAGE ...]
(`noisepool` stages are pooled in the noise floor, `noise` stages are listed only.)
Writes OUT/analysis/ex_e1.json and OUT/manifest.json.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import multiplicity, rendered, replication
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import world_bootstrap_draws

Est = ex.Est
DRAWS = ex.DRAWS
SEED = ex.SEED
HOLM_DRAWS = 10_000  # the exact-p check of the multiplicity analysis uses five times the draws
FIT_CELLS = ("D0B0", "D1B0", "D0B1_source", "D1B1_source", "D0B1_cited", "D1B1_cited")
GRANS = ("source", "cited")
PREDICTORS = ("cell", "structural", "literal")
OOS_CLASSES = ("definition", "other", "all")
WITHIN_PP = 0.05  # an error within 5 pp is counted as a hit

CONTRAST_ORDER = (
    "B_effect_update_D0",
    "B_effect_update_D1",
    "B_effect_update_pooled",
    "D_effect_update_B0",
    "D_effect_update_B1",
    "BxD_interaction_update",
    "D_effect_repair",
    "D_effect_repair_changed",
)
MEMBERSHIP = {"rule": ("rule",), "norule": ("norule",)}


def _safe(weight: np.ndarray, x: np.ndarray) -> np.ndarray:
    """weight * x, with 0 where the weight is 0 (x may be NaN there: the term is absent)."""
    return np.asarray(np.where(weight == 0, 0.0, weight * x))


# ---------------------------------------------------------------------------- out-of-sample
def _fit_terms(t: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Terms of the six cells from totals t (..., cells, classes, stats): each (..., classes)."""
    return {n: ex.cell_terms(t[..., i, :, :]) for i, n in enumerate(FIT_CELLS)}


def predict(t: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Predicted Update per predictor, cell and base class (..., classes) from the fit totals."""
    tm = _fit_terms(t)
    cited = [FIT_CELLS.index(c) for c in ("D0B1_cited", "D1B1_cited")]
    source = [FIT_CELLS.index(c) for c in ("D0B1_source", "D1B1_source")]
    with np.errstate(divide="ignore", invalid="ignore"):
        miss = sum(t[..., i, :, 0] - t[..., i, :, 2] for i in cited)
        wrong = sum(t[..., i, :, 1] - t[..., i, :, 3] for i in cited)
        # a: accuracy of the items the cited key missed; a class it missed nothing of borrows the
        # accuracy over all classes (the stale value is one constant in Eq. 1)
        stale = np.where(
            miss > 0,
            wrong / miss,
            wrong.sum(axis=-1, keepdims=True) / miss.sum(axis=-1, keepdims=True),
        )
        m_src = sum(tm[FIT_CELLS[i]]["miss"] for i in source) / sum(
            tm[FIT_CELLS[i]]["n"] for i in source
        )
        m_cit = sum(tm[FIT_CELLS[i]]["miss"] for i in cited) / sum(
            tm[FIT_CELLS[i]]["n"] for i in cited
        )

        def eq1(m: Any, r: Any, a: Any) -> np.ndarray:
            return np.asarray(_safe(1 - m, r) + _safe(m, a))

        out: dict[str, dict[str, np.ndarray]] = {p: {} for p in PREDICTORS}
        for cell in FIT_CELLS:
            out["cell"][cell] = tm[cell]["U"]
        for d in (0, 1):
            r_d = tm[f"D{d}B1_source"]["r"]
            for p in ("structural", "literal"):
                out[p][f"D{d}B1_source"] = eq1(m_src, r_d, stale)
                out[p][f"D{d}B1_cited"] = eq1(m_cit, r_d, stale)
            out["structural"][f"D{d}B0"] = tm[f"D{d}B0"]["U"]  # a at B=0: the write path
            out["literal"][f"D{d}B0"] = stale  # one stale-value accuracy for every cell
    return out


def _combine(pred: np.ndarray, w: np.ndarray, cls: str) -> np.ndarray:
    """Predicted Update of a class set: the base classes' predictions weighted by `w` (the number
    of changed item-times of each base class in the world set whose mix is used)."""
    idx = [ex.BASE_CLASSES.index(c) for c in ex.CLASS_SETS[cls]]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(
            sum(_safe(w[..., i], pred[..., i]) for i in idx) / sum(w[..., i] for i in idx)
        )


def _observed(t: np.ndarray, cls: str) -> dict[str, np.ndarray]:
    tot = ex.class_totals(t, cls)
    with np.errstate(divide="ignore", invalid="ignore"):
        return {n: tot[..., i, 1] / tot[..., i, 0] for i, n in enumerate(FIT_CELLS)}


def oos_branch(
    fit: tuple[list[str], np.ndarray], target: tuple[list[str], np.ndarray], draws: int
) -> dict[str, Any]:
    """One branch: terms fitted on `fit`, Update predicted on `target` (worlds, array (worlds,
    cells, classes, stats) of `ex.cell_arrays` with the cells FIT_CELLS)."""
    (fw, fa), (tw, ta) = fit, target
    point_f, point_t = fa.sum(axis=0), ta.sum(axis=0)
    draw_f = fa[ex.boot_index(len(fw), draws, SEED + 1)].sum(axis=1)
    draw_t = ta[ex.boot_index(len(tw), draws, SEED)].sum(axis=1)
    res: dict[str, Any] = {}
    quant: dict[tuple[str, str, str, str], tuple[Any, Any]] = {}
    for label, (tf, tt) in {"point": (point_f, point_t), "draws": (draw_f, draw_t)}.items():
        preds = predict(tf)
        w_target = tt[..., 0, :, 0]  # changed item-times per base class (D0B0 cell)
        w_fit = tf[..., 0, :, 0]
        for cls in OOS_CLASSES:
            obs = _observed(tt, cls)
            for p in PREDICTORS:
                for cell in FIT_CELLS:
                    pr = _combine(preds[p][cell], w_target, cls)
                    quant[label, cls, p, cell] = (pr, obs[cell])
                    if cls == "all":  # the fit worlds' own mix, to see what the class mix adds
                        quant[label, "all_fit_mix", p, cell] = (
                            _combine(preds[p][cell], w_fit, cls),
                            obs[cell],
                        )
    n_items = {cls: int(ex.class_totals(point_t, cls)[0, 0]) for cls in OOS_CLASSES}
    n_items["all_fit_mix"] = n_items["all"]
    for cls in (*OOS_CLASSES, "all_fit_mix"):
        res[cls] = {}
        for p in PREDICTORS:
            cells: dict[str, Any] = {}
            errs_point, errs_draw = [], []
            for cell in FIT_CELLS:
                pr, ob = quant["point", cls, p, cell]
                prd, obd = quant["draws", cls, p, cell]
                err = obd - prd
                cells[cell] = {
                    "predicted": float(pr),
                    "observed": float(ob),
                    "error": ex.interval(
                        float(ob - pr), np.asarray(err, float), len(tw), n_items[cls]
                    ),
                    "abs_error": ex.interval(
                        float(abs(ob - pr)), np.abs(np.asarray(err, float)), len(tw), n_items[cls]
                    ),
                }
                errs_point.append(abs(ob - pr))
                errs_draw.append(np.abs(err))
            mae_draw = np.nanmean(np.stack(errs_draw), axis=0) if errs_draw else np.array([])
            res[cls][p] = {
                "cells": cells,
                "mean_abs_error": ex.interval(
                    float(np.nanmean(errs_point)), mae_draw, len(tw), n_items[cls]
                ),
                "cells_within_5pp": int(np.sum(np.array(errs_point) <= WITHIN_PP)),
                "cells_error_excludes_zero": sum(
                    c["error"]["lo"] > 0 or c["error"]["hi"] < 0 for c in cells.values()
                ),
            }
    return res


def oos(fit: ex.Table, target: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """Out-of-sample test of Eq. (1) per branch, and pooled over the branches (the same worlds are
    resampled in every branch, so the pooled interval respects the clustering)."""
    out: dict[str, Any] = {"branches": {}, "class_mix": {}}
    arrays: dict[str, tuple[tuple[list[str], np.ndarray], tuple[list[str], np.ndarray]]] = {}
    for b in sorted(set(fit.branches) & set(target.branches)):
        fw, fa, _ = ex.cell_arrays(fit, b, "D1", GRANS)
        tw, ta, _ = ex.cell_arrays(target, b, "D1", GRANS)
        if not fw or not tw:
            continue
        arrays[b] = ((fw, fa), (tw, ta))
        out["branches"][b] = oos_branch((fw, fa), (tw, ta), draws)
        out["class_mix"][b] = {
            "fit": ex.class_mix(fit)[b]["classes"],
            "target": ex.class_mix(target)[b]["classes"],
        }
    same_worlds = len({tuple(v[0][0]) for v in arrays.values()}) == 1 and (
        len({tuple(v[1][0]) for v in arrays.values()}) == 1
    )
    if arrays and same_worlds:
        out["pooled_over_branches"] = _oos_pooled(arrays, draws)
    return out


def _oos_pooled(
    arrays: Mapping[str, tuple[tuple[list[str], np.ndarray], tuple[list[str], np.ndarray]]],
    draws: int,
) -> dict[str, Any]:
    """Mean absolute error over branches and the six cells, per predictor and class set."""
    acc: dict[tuple[str, str], list[np.ndarray]] = {}
    acc_point: dict[tuple[str, str], list[float]] = {}
    for (fw, fa), (tw, ta) in arrays.values():
        tf, tt = fa.sum(axis=0), ta.sum(axis=0)
        df = fa[ex.boot_index(len(fw), draws, SEED + 1)].sum(axis=1)
        dt = ta[ex.boot_index(len(tw), draws, SEED)].sum(axis=1)
        for label, (f_, t_) in {"point": (tf, tt), "draws": (df, dt)}.items():
            preds = predict(f_)
            w = t_[..., 0, :, 0]
            for cls in OOS_CLASSES:
                obs = _observed(t_, cls)
                for p in PREDICTORS:
                    for cell in FIT_CELLS:
                        err = np.abs(obs[cell] - _combine(preds[p][cell], w, cls))
                        if label == "point":
                            acc_point.setdefault((cls, p), []).append(float(err))
                        else:
                            acc.setdefault((cls, p), []).append(np.asarray(err, float))
    out: dict[str, Any] = {}
    n_worlds = len(next(iter(arrays.values()))[1][0])
    for (cls, p), pts in acc_point.items():
        out.setdefault(cls, {})[p] = {
            "mean_abs_error": ex.interval(
                float(np.nanmean(pts)), np.nanmean(np.stack(acc[cls, p]), axis=0), n_worlds, 0
            ),
            "cells_within_5pp": int(np.sum(np.array(pts) <= WITHIN_PP)),
            "n_cells": len(pts),
        }
    return out


# --------------------------------------------------------------------------- short worlds
def short_prediction(main: ex.Table, short: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """The B effect at each D on the short worlds, predicted from the twelve-month terms: for a
    class, B|D = (1 - m) r_D + m a - a_D with r_D and the B=0 Update a_D of the main study (m is
    the source-epoch key's missed rate, 0), weighted by the short worlds' class mix. Reported with
    what the short worlds show, the prediction at the main study's own mix (its observed B effect),
    and the terms that moved."""
    out: dict[str, Any] = {}
    names = ("D0B0", "D1B0", "D0B1_source", "D1B1_source")
    for b in sorted(set(main.branches) & set(short.branches)):
        mw, ma, _ = ex.cell_arrays(main, b, "D1", ["source"])
        sw, sa, _ = ex.cell_arrays(short, b, "D1", ["source"])
        if not mw or not sw:
            continue
        df = ma[ex.boot_index(len(mw), draws, SEED + 1)].sum(axis=1)
        dt = sa[ex.boot_index(len(sw), draws, SEED)].sum(axis=1)
        pt = (ma.sum(axis=0), sa.sum(axis=0))
        res: dict[str, Any] = {}
        for cls in OOS_CLASSES:
            entry: dict[str, Any] = {}
            for d in (0, 1):
                vals: dict[str, tuple[Any, ...]] = {}
                for label, (tf, tt) in {"point": pt, "draws": (df, dt)}.items():
                    tm = {n: ex.cell_terms(tf[..., i, :, :]) for i, n in enumerate(names)}
                    m = tm[f"D{d}B1_source"]["m"]
                    with np.errstate(divide="ignore", invalid="ignore"):
                        rec = _safe(1 - m, tm[f"D{d}B1_source"]["r"]) + _safe(
                            m, tm[f"D{d}B1_source"]["a"]
                        )
                        pred_base = rec - tm[f"D{d}B0"]["U"]
                        w_short, w_main = tt[..., 0, :, 0], tf[..., 0, :, 0]
                        pred = _combine(pred_base, w_short, cls)
                        pred_main_mix = _combine(pred_base, w_main, cls)
                        tot = ex.class_totals(tt, cls)
                        u1 = tot[..., 2 + d, 1] / tot[..., 2 + d, 0]
                        u0 = tot[..., d, 1] / tot[..., d, 0]
                    vals[label] = (pred, u1 - u0, pred_main_mix)
                n_items = int(ex.class_totals(pt[1], cls)[0, 0])
                p_, o_, pm_ = vals["point"]
                pd_, od_, pmd_ = vals["draws"]
                entry[f"D{d}"] = {
                    "predicted": ex.interval(float(p_), np.asarray(pd_, float), len(sw), n_items),
                    "observed": ex.interval(float(o_), np.asarray(od_, float), len(sw), n_items),
                    "observed_minus_predicted": ex.interval(
                        float(o_ - p_), np.asarray(od_ - pd_, float), len(sw), n_items
                    ),
                    "predicted_at_main_mix": ex.interval(
                        float(pm_), np.asarray(pmd_, float), len(mw), n_items
                    ),
                }
            res[cls] = entry
        res["terms"] = _short_terms(ma, sa, names, draws, len(mw), len(sw))
        out[b] = res
    return out


def _short_terms(
    ma: np.ndarray, sa: np.ndarray, names: Sequence[str], draws: int, n_main: int, n_short: int
) -> dict[str, Any]:
    """The Eq. (1) terms that can move between the regimes, per class: r and the B=0 Update a at
    each D, in the main study and the short worlds, and the difference (short - main)."""
    dm = ma[ex.boot_index(n_main, draws, SEED + 1)].sum(axis=1)
    ds = sa[ex.boot_index(n_short, draws, SEED)].sum(axis=1)
    out: dict[str, Any] = {}
    for cls in OOS_CLASSES:
        node: dict[str, Any] = {}
        for label, arr_m, arr_s in (("point", ma.sum(axis=0), sa.sum(axis=0)), ("draws", dm, ds)):
            tm = {
                n: ex.cell_terms(ex.class_totals(arr_m, cls)[..., i, :])
                for i, n in enumerate(names)
            }
            ts = {
                n: ex.cell_terms(ex.class_totals(arr_s, cls)[..., i, :])
                for i, n in enumerate(names)
            }
            for d in (0, 1):
                for term, cell in (("r", f"D{d}B1_source"), ("a_B0", f"D{d}B0")):
                    key = "r" if term == "r" else "U"
                    node.setdefault(f"{term}_D{d}", {})[label] = (
                        tm[cell][key],
                        ts[cell][key],
                    )
        out[cls] = {
            k: {
                "main": ex.interval(
                    float(v["point"][0]), np.asarray(v["draws"][0], float), n_main, 0
                ),
                "short": ex.interval(
                    float(v["point"][1]), np.asarray(v["draws"][1], float), n_short, 0
                ),
                "short_minus_main": ex.interval(
                    float(v["point"][1] - v["point"][0]),
                    np.asarray(v["draws"][1] - v["draws"][0], float),
                    n_short,
                    0,
                ),
            }
            for k, v in node.items()
        }
    return out


# ----------------------------------------------------------------------------- replication
NON_CONTRAST = "B_effect_detected_"  # B on detection: 0 or 100 by construction
D_EFFECT_ON_REPAIR = ("D_effect_repair_source", "D_effect_repair_mech_source")
UPDATE_KEYS = ("B_effect_update_D1_source", "B_effect_update_pooled_source")


def source_epoch_accounting(orig: Mapping[str, Any], rep: Mapping[str, Any]) -> dict[str, Any]:
    """The 21 source-epoch contrasts predicted non-zero (3 branches x 7): which are identities by
    construction (B on detection: a B=0 cell never invalidates and a source-epoch key invalidates
    whenever a declared source changed), how the rest replicate, and how many independent numbers
    the Update contrasts hold (the pooled B effect and the interaction are functions of the B
    effect at D=0 and D=1)."""
    rows = [
        r
        for r in replication.contrast_rows(dict(orig), dict(rep))
        if r["kind"] == "effect" and r["contrast"].endswith("_source")
    ]
    listed = []
    for r in rows:
        identity = r["contrast"].startswith(NON_CONTRAST)
        listed.append(
            {
                "branch": r["branch"],
                "contrast": r["contrast"],
                "identity_by_construction": identity,
                "orig": r["orig"],
                "rep": r["rep"],
                "replication": r["replication"],
            }
        )
    rest = [r for r in listed if not r["identity_by_construction"]]

    def tally(rs: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        return {
            v: sum(r["replication"] == v for r in rs)
            for v in ("replicated", "same direction", "not replicated")
        }

    return {
        "n_contrasts": len(listed),
        "n_identities": len(listed) - len(rest),
        "n_non_identity": len(rest),
        "replication_all": tally(listed),
        "replication_non_identity": tally(rest),
        "n_d_effect_on_repair": sum(r["contrast"] in D_EFFECT_ON_REPAIR for r in rest),
        "n_update_contrasts": sum(r["contrast"] not in D_EFFECT_ON_REPAIR for r in rest),
        "independent_update_estimates": 2 * len({r["branch"] for r in rest}),
        "contrasts": listed,
    }


def class_replication(orig: Mapping[str, Any], rep: Mapping[str, Any]) -> dict[str, Any]:
    """The class-stratified contrasts of the main study against R: R's estimate and interval for
    the definition and the non-definition items, with the sign rule of
    `replication.replication_class` (in the direction of the main study's estimate) and whether
    each interval excludes 0."""
    out: dict[str, Any] = {"rows": [], "summary": {}}
    for arm in orig.get("explore_class_split") or {}:
        for b, grans in orig["explore_class_split"][arm].items():
            for g, classes in grans.items():
                for cls in ("definition", "other"):
                    for key in CONTRAST_ORDER:
                        o = (classes.get(cls) or {}).get(key)
                        r = (
                            (
                                (rep.get("explore_class_split") or {})
                                .get(arm, {})
                                .get(b, {})
                                .get(g, {})
                            )
                            .get(cls, {})
                            .get(key)
                        )
                        if not o or not r:
                            continue
                        out["rows"].append(
                            {
                                "arm": arm,
                                "branch": b,
                                "granularity": g,
                                "class": cls,
                                "contrast": key,
                                "orig": o,
                                "rep": r,
                                "replication": replication.replication_class(o, r),
                                "same_sign": (o["est"] >= 0) == (r["est"] >= 0),
                                "orig_excludes_zero": o["lo"] > 0 or o["hi"] < 0,
                                "rep_excludes_zero": r["lo"] > 0 or r["hi"] < 0,
                            }
                        )
    src = [r for r in out["rows"] if r["arm"] == "D1" and r["granularity"] == "source"]
    for cls in ("definition", "other"):
        rs = [r for r in src if r["class"] == cls]
        out["summary"][cls] = {
            "n_contrasts": len(rs),
            "rep_same_sign": sum(r["same_sign"] for r in rs),
            "rep_excludes_zero": sum(r["rep_excludes_zero"] for r in rs),
            "orig_excludes_zero": sum(r["orig_excludes_zero"] for r in rs),
            "replicated": sum(r["replication"] == "replicated" for r in rs),
        }
    return out


def holm_family(
    rows: Sequence[Mapping[str, Any]], metrics: Mapping[str, Any], draws: int = HOLM_DRAWS
) -> dict[str, Any]:
    """Holm-Bonferroni on the contrasts predicted non-zero that are not detection identities
    (`replication.PREDICTED_NONZERO` without B on detection), with one-sided bootstrap p-values
    from `draws` world resamples (seed 996): p = (1 + #{draws with mean <= 0}) / (1 + draws), so
    that the smallest p is 1 / (1 + draws) and the smallest adjusted p is m / (1 + draws)."""
    vectors = rendered.contrast_vectors([dict(r) for r in rows])
    listed = []
    for b, cons in metrics["contrasts"].items():
        for key in cons:
            if not key.startswith(replication.PREDICTED_NONZERO) or key.startswith(NON_CONTRAST):
                continue
            by_world = {w: list(d.values()) for w, d in vectors[b, key].items()}
            est, vals = world_bootstrap_draws(by_world, draws=draws, seed=SEED)
            against = sum(v <= 0 for v in vals)
            listed.append(
                {
                    "branch": b,
                    "contrast": key,
                    "est": est.est,
                    "lo": est.lo,
                    "hi": est.hi,
                    "draws_at_or_below_zero": against,
                    "p": multiplicity.one_sided_p(vals) if vals else 1.0,
                }
            )
    adj, reject = multiplicity.holm([r["p"] for r in listed])
    bh, bh_reject = multiplicity.benjamini_hochberg([r["p"] for r in listed])
    for r, a, h, q, k in zip(listed, adj, reject, bh, bh_reject, strict=True):
        r.update({"holm_p": a, "survives_holm": h, "bh_q": q, "survives_bh": k})
    fails = [r for r in listed if not r["survives_holm"]]
    return {
        "family_size": len(listed),
        "draws": draws,
        "smallest_possible_p": 1 / (1 + draws),
        "smallest_possible_holm_p": len(listed) / (1 + draws),
        "n_survive_holm": sum(r["survives_holm"] for r in listed),
        "n_survive_bh": sum(r["survives_bh"] for r in listed),
        "n_draws_below_floor_zero": sum(r["draws_at_or_below_zero"] == 0 for r in listed),
        "failing": [f"{r['branch']}:{r['contrast']}" for r in fails],
        "largest_holm_p_among_survivors": max(
            (r["holm_p"] for r in listed if r["survives_holm"]), default=None
        ),
        "contrasts": listed,
    }


# ------------------------------------------------------------------------------ cost model
COST_SYSTEMS = ("zero", "rag", "holdmem_D1B1_source")
COST_LABEL = (
    "cost model, not an execution: the maintenance path is replayed on the world log and its "
    "tokens counted (analysis.e3)"
)


def cost_model(e3: Mapping[str, Any]) -> dict[str, Any]:
    """Tokens per world-year of the E3 cost model: ingestion, maintenance and serving of the zero
    system, plain RAG and HoldMem(1,1), each as a multiple of the zero system's total; maintenance
    tokens per query and per counted re-derivation."""
    out: dict[str, Any] = {"label": COST_LABEL, "horizons": {}}
    for h, per in (e3.get("horizons") or {}).items():
        zero = per.get("zero") or {}
        zc = zero.get("cost_mean") or {}
        ztotal = sum(zc.get(k, 0) for k in ("ingest_tokens", "maintain_tokens", "serve_tokens"))
        node: dict[str, Any] = {}
        for s in COST_SYSTEMS:
            e = per.get(s) or {}
            c = e.get("cost_mean")
            if not c:
                continue
            queries = c["serve_tokens"] / e["serve_tokens_per_query"]
            total = c["ingest_tokens"] + c["maintain_tokens"] + c["serve_tokens"]
            node[s] = {
                "ingest_tokens": c["ingest_tokens"],
                "maintain_tokens": c["maintain_tokens"],
                "serve_tokens": c["serve_tokens"],
                "total_tokens": total,
                "total_over_zero": total / ztotal if ztotal else float("nan"),
                "maintain_over_serve": c["maintain_tokens"] / c["serve_tokens"],
                "queries": queries,
                "maintain_tokens_per_query": c["maintain_tokens"] / queries,
                "serve_tokens_per_query": e["serve_tokens_per_query"],
                "maintain_writes_counted": c.get("maintain_writes_counted"),
                "tokens_per_counted_write": (
                    c["maintain_tokens"] / c["maintain_writes_counted"]
                    if c.get("maintain_writes_counted")
                    else None
                ),
                "contract_valid": (e.get("contract_valid") or {}).get("est"),
            }
        out["horizons"][h] = node
    return out


# ------------------------------------------------------------------------------ noise floor
def pooled_noise_floor(
    metrics: Mapping[str, Mapping[str, Any]], pooled: Sequence[str] = ()
) -> dict[str, Any]:
    """Values and answers that differ between two ingestions, per stage, and pooled over the
    stages named in `pooled` (`metrics`: stage name -> e1_metrics.json), over world-branches
    (`total`) and over worlds (`world_level`, see `ex_e1_classes.noise_floor`). A stage that
    re-analyses runs another stage holds (block C re-uses the test worlds' writer runs) is listed
    but not pooled, so that no world counts twice."""
    out: dict[str, Any] = {"stages": {}}
    keys = ("world_branches", "items", "values_differ", "answers_differ")
    tot = {k: 0 for k in keys}
    world = {k: 0 for k in ("worlds", "items", "values_differ", "answers_differ")}
    for name, m in metrics.items():
        nf = m.get("explore_noise_floor") or {}
        if not nf.get("total"):
            continue
        out["stages"][name] = {**nf["total"], "world_level": nf.get("world_level")}
        if name in pooled:
            for k in tot:
                tot[k] += nf["total"][k]
            for k in world:
                world[k] += (nf.get("world_level") or {}).get(k, 0)
    if tot["items"]:
        out["pooled"] = {
            "stages": [n for n in metrics if n in pooled and n in out["stages"]],
            **tot,
            "values_differ_share": tot["values_differ"] / tot["items"],
            "answers_differ_share": tot["answers_differ"] / tot["items"],
            "world_level": {
                **world,
                "values_differ_share": world["values_differ"] / world["items"]
                if world["items"]
                else None,
                "answers_differ_share": world["answers_differ"] / world["items"]
                if world["items"]
                else None,
            },
        }
    return out


# ---------------------------------------------------------------------------------- driver
def _load_table(stage: Path) -> ex.Table:
    return ex.Table(read_jsonl(stage / "analysis" / "e1_item_table.jsonl"))


def _load_json(stage: Path, name: str) -> dict[str, Any]:
    data: dict[str, Any] = json.loads((stage / "analysis" / name).read_text())
    return data


def analyse(stages: Mapping[str, list[Path]], out: Path, draws: int = DRAWS) -> dict[str, Any]:
    """`stages`: role -> stage directories (dev, devorphan, main, rep, short, e3, e3rep, noise)."""

    def one(role: str) -> Path | None:
        return stages[role][0] if stages.get(role) else None

    res: dict[str, Any] = {"status": ex.STATUS}
    dev_dirs = [*stages.get("dev", []), *stages.get("devorphan", [])]
    main, rep, short = one("main"), one("rep"), one("short")
    tables: dict[str, ex.Table] = {}
    if dev_dirs:
        tables["dev"] = ex.Table(
            [r for d in dev_dirs for r in read_jsonl(d / "analysis" / "e1_item_table.jsonl")]
        )
    for role, d in (("main", main), ("rep", rep), ("short", short)):
        if d is not None:
            tables[role] = _load_table(d)
    if "dev" in tables and "main" in tables:
        res["explore_oos"] = {
            "dev_to_main": oos(tables["dev"], tables["main"], draws),
            **({"dev_to_rep": oos(tables["dev"], tables["rep"], draws)} if "rep" in tables else {}),
            **(
                {"main_to_rep": oos(tables["main"], tables["rep"], draws)}
                if "rep" in tables
                else {}
            ),
        }
    if main and rep:
        om, rm = _load_json(main, "e1_metrics.json"), _load_json(rep, "e1_metrics.json")
        res["explore_replication"] = {
            "source_epoch": source_epoch_accounting(om, rm),
            "class_split": class_replication(om, rm),
            "holm_non_detection": {
                "main": holm_family(
                    read_jsonl(main / "analysis" / "e1_item_table.jsonl"), om, HOLM_DRAWS
                ),
                "rep": holm_family(
                    read_jsonl(rep / "analysis" / "e1_item_table.jsonl"), rm, HOLM_DRAWS
                ),
            },
        }
    if "main" in tables and "short" in tables:
        res["explore_short"] = short_prediction(tables["main"], tables["short"], draws)
    costs = {
        label: cost_model(_load_json(d, "e3_metrics.json"))
        for label, role in (("main", "e3"), ("rep", "e3rep"))
        if (d := one(role)) is not None
    }
    if costs:
        res["explore_cost"] = costs
    pool_dirs = stages.get("noisepool", [])
    noise = {
        d.name: _load_json(d, "e1_metrics.json") for d in [*stages.get("noise", []), *pool_dirs]
    }
    if noise:
        res["explore_noise_floor"] = pooled_noise_floor(noise, [d.name for d in pool_dirs])
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "ex_e1.json").write_text(json.dumps(res, indent=1))
    write_stage_manifest(out, "ex_e1", [p for ps in stages.values() for p in ps], {})
    return res


def main() -> None:
    out, *specs = sys.argv[1:]
    stages: dict[str, list[Path]] = {}
    for s in specs:
        role, path = s.split(":", 1)
        stages.setdefault(role, []).append(Path(path))
    if not stages:
        raise SystemExit(__doc__)
    res = analyse(stages, Path(out))
    print(json.dumps({k: list(v) if isinstance(v, dict) else v for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()

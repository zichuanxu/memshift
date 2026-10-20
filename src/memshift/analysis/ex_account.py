"""Test of the account across writers and item classes. Every key starts with `explore_`.

The account says binding is worth what the writer's own re-derivation leaves open: with the key
invalidating nearly everything, the B x D interaction on Update is (r1 - a1) - (r0 - a0), where r_D
is the accuracy of the re-derived items at each D and a_D the Update of the B=0 cell. A class
split of one writer cannot tell "the class decides" from "r0 of that writer decides", so this
module computes the Eq. (1) accounting for every writer (Qwen3.5-9B: main study and R; gemma,
Llama, gpt-oss: block M2, ten worlds each), item class and branch at source-epoch binding, and
tests the account across the cells:

explore_account_cells   per writer, branch and class (definition; structured field `rule`; free
                        text `norule`; `other` is both): Update of the four cells; m (missed
                        invalidation of the source-epoch key), r0, r1 (accuracy of the invalidated
                        items at D=0 and D=1), a0, a1 (the B=0 Updates); B|D0, B|D1, D|B0, D|B1,
                        B x D and the repair gap r1 - r0; each with world-bootstrap intervals
                        (2,000 draws, seed 996, the draws of the staged decompositions)
explore_account_fit     across all (writer, class, branch) cells of the four fitted writers (the
                        main study, gemma, Llama, gpt-oss; R is not fitted): the observed B x D
                        against the repair gap r1 - r0 of the same cell: Spearman correlation and
                        the least-squares line (slope, intercept), with intervals from a bootstrap
                        over worlds within each writer (independent across writers); leave-one-
                        writer-out: each writer's cells predicted from a fit on the other three,
                        mean absolute error, for the gap, for the class label alone (definition
                        against other: the mean of the training cells of the class) and for
                        further reference predictors; the same for R's cells predicted from the
                        four fitted writers and from the three that are not of its lineage

The identity behind the test: at source-epoch binding m is about 0, so B x D is (r1 - r0) - D|B0,
and a fit of B x D on r1 - r0 tests whether D|B0 (the D effect of the write path) is regular across
cells, not the identity itself. The `corrected` predictor (the identity) is listed as a check.

Usage: python -m memshift.analysis.ex_account OUT main:STAGE rep:STAGE gemma:STAGE llama:STAGE
           gptoss:STAGE
Writes OUT/analysis/ex_account.json and OUT/manifest.json.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as sps

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import read_jsonl

DRAWS = ex.DRAWS
SEED = ex.SEED
BRANCHES = ("src", "def", "orphan")
WRITERS = (  # key, label
    ("main", "Qwen3.5-9B"),
    ("rep", "Qwen3.5-9B (R)"),
    ("gemma", "gemma-4-12B"),
    ("llama", "Llama-3.1-8B"),
    ("gptoss", "gpt-oss-20b"),
)
FITTED = ("main", "gemma", "llama", "gptoss")  # R re-draws the worlds of the main study's writer
LINEAGE = {"rep": "main"}
CELL_CLASSES = ("definition", "rule", "norule", "other")
FIT_CLASSES = {
    "two_class": ("definition", "other"),
    "three_class": ("definition", "rule", "norule"),
}
PREDICTORS = {
    "gap_linear": "B x D from a least-squares line in r1 - r0, fitted on the training cells",
    "class_mean": "mean B x D of the training cells of the same class (definition against other)",
    "class_branch_mean": "mean B x D of the training cells of the same class and branch",
    "grand_mean": "mean B x D of all training cells (no predictor)",
    "gap_identity": "B x D = r1 - r0 (the account without the D effect of the write path)",
    "gap_and_class": "least-squares on r1 - r0 and the class indicator (definition against rest)",
    "corrected": "the identity (r1 - a1) - (r0 - a0) on the cell itself: a check, not a prediction",
}
HEADLINE = ("gap_linear", "class_mean")
Est = ex.Est
Quantities = dict[str, np.ndarray]
WriterArrays = dict[str, dict[str, tuple[Quantities, Quantities, int]]]


# ---------------------------------------------------------------------------------- the cells
def _quantities(t: np.ndarray, names: Sequence[str]) -> dict[str, np.ndarray]:
    """The terms and contrasts of one class from totals t (..., cells, stats), source epoch."""
    q = ex._flatten(ex.decomposition_quantities(t, names, ["source"]))

    def cell(c: str, term: str) -> np.ndarray:
        return np.asarray(q[("cells", "source", c, term)], float)

    def inter(k: str) -> np.ndarray:
        return np.asarray(q[("interaction", "source", k)], float)

    r0, r1 = cell("D0B1", "r"), cell("D1B1", "r")
    return {
        "n": cell("D1B1", "n"),
        "U_D0B0": cell("D0B0", "U"),
        "U_D1B0": cell("D1B0", "U"),
        "U_D0B1": cell("D0B1", "U"),
        "U_D1B1": cell("D1B1", "U"),
        "m": cell("D1B1", "m"),
        "m_D0": cell("D0B1", "m"),
        "r0": r0,
        "r1": r1,
        "a0": cell("D0B0", "U"),
        "a1": cell("D1B0", "U"),
        "B_D0": inter("B_effect_D0"),
        "B_D1": inter("B_effect_D1"),
        "D_B0": inter("D_effect_B0"),
        "D_B1": inter("D_effect_B1"),
        "BxD": inter("observed"),
        "gap": r1 - r0,
        "corrected": inter("corrected_formula"),
    }


def writer_arrays(tab: ex.Table, draws: int = DRAWS, seed: int = SEED) -> WriterArrays:
    """branch -> class -> (point quantities, draw quantities, worlds) for one writer; the draws
    resample the writer's worlds with `seed`."""
    out: WriterArrays = {}
    for b in BRANCHES:
        if b not in tab.branches:
            continue
        worlds, arr, names = ex.cell_arrays(tab, b, "D1", ["source"])
        if not worlds:
            continue
        drawn = arr[ex.boot_index(len(worlds), draws, seed)].sum(axis=1)
        point = arr.sum(axis=0)
        per_class: dict[str, tuple[Quantities, Quantities, int]] = {}
        for cls in CELL_CLASSES:
            pt = ex.class_totals(point, cls)
            dr = ex.class_totals(drawn, cls)
            per_class[cls] = (_quantities(pt, names), _quantities(dr, names), len(worlds))
        out[b] = per_class
    return out


def _est(point: Any, draws: np.ndarray, n_worlds: int, n_items: int) -> Est:
    return ex.interval(float(point), np.asarray(draws, float), n_worlds, n_items)


def cell_nodes(tab: ex.Table, draws: int = DRAWS) -> dict[str, dict[str, Any]]:
    """`explore_account_cells[writer]` for one table: branch -> class -> node; the intervals use
    the draws of the staged decompositions (`ex_e1_classes.boot_index`, seed 996)."""
    out: dict[str, dict[str, Any]] = {}
    for b, per_class in writer_arrays(tab, draws, SEED).items():
        for cls, (pt, dr, n_worlds) in per_class.items():
            n = int(pt["n"])
            node: dict[str, Any] = {"changed_item_times": n}
            for k, v in pt.items():
                if k == "n":
                    continue
                node[k] = _est(v, dr[k], n_worlds, n)
            out.setdefault(b, {})[cls] = node
    return out


# ------------------------------------------------------------------------------------ the fit
def _ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(slope, intercept) of the least-squares line; NaN when x does not vary."""
    if len(x) < 3 or np.ptp(x) == 0:
        return float("nan"), float("nan")
    slope, intercept = np.polyfit(x, y, 1)
    return float(slope), float(intercept)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman correlation; NaN with fewer than three cells or when x or y does not vary."""
    if len(x) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        return float("nan")
    return float(sps.spearmanr(x, y).statistic)


def _predict(
    name: str,
    tr: Mapping[str, np.ndarray],
    te: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Predictions of the test cells by predictor `name` from the training cells; both are dicts
    of equal-length arrays: x (gap), y (B x D), corr (identity), cls, br (codes)."""
    n = len(te["x"])
    if name == "gap_identity":
        return np.asarray(te["x"], float)
    if name == "corrected":
        return np.asarray(te["corr"], float)
    if not len(tr["y"]):
        return np.full(n, np.nan)
    if name == "gap_linear":
        s, i = _ols(tr["x"], tr["y"])
        return np.asarray(i + s * te["x"])
    if name == "grand_mean":
        return np.full(n, float(np.mean(tr["y"])))
    if name == "class_mean":
        return np.array(
            [
                np.mean(tr["y"][tr["cls"] == c]) if np.any(tr["cls"] == c) else np.nan
                for c in te["cls"]
            ]
        )
    if name == "class_branch_mean":
        out = []
        for c, b in zip(te["cls"], te["br"], strict=True):
            sel = (tr["cls"] == c) & (tr["br"] == b)
            sel = sel if sel.any() else tr["cls"] == c
            out.append(np.mean(tr["y"][sel]) if sel.any() else np.nan)
        return np.array(out)
    if name == "gap_and_class":
        if len(tr["y"]) < 4:
            return np.full(n, np.nan)
        design = np.column_stack([np.ones(len(tr["x"])), tr["x"], tr["cls"] == 0])
        coef, *_ = np.linalg.lstsq(design, tr["y"], rcond=None)
        return np.asarray(coef[0] + coef[1] * te["x"] + coef[2] * (te["cls"] == 0))
    raise ValueError(name)


def _subset(cells: Mapping[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    return {k: v[keep] for k, v in cells.items()}


def fit_statistics(
    cells: Mapping[str, np.ndarray], held_out: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Every statistic of the test for one set of cell values: `cells` holds the fitted writers'
    cells (x, y, corr, cls, br, wr codes), `held_out` the cells of R. Cells with an undefined x or
    y are dropped."""
    ok = np.isfinite(cells["x"]) & np.isfinite(cells["y"])
    c = _subset(cells, ok)
    res: dict[str, Any] = {"n_cells": int(ok.sum())}
    if len(c["x"]) < 3:
        return res
    slope, intercept = _ols(c["x"], c["y"])
    res |= {"spearman": _spearman(c["x"], c["y"]), "slope": slope, "intercept": intercept}
    abs_err: dict[str, np.ndarray] = {}
    for name in PREDICTORS:
        pred = np.full(len(c["y"]), np.nan)
        for w in np.unique(c["wr"]):
            test = c["wr"] == w
            pred[test] = _predict(name, _subset(c, ~test), _subset(c, test))
        abs_err[name] = np.abs(c["y"] - pred)
    res["lowo_mae"] = {k: float(np.nanmean(v)) for k, v in abs_err.items()}
    res["lowo_mae_by_writer"] = {
        int(w): {k: float(np.nanmean(v[c["wr"] == w])) for k, v in abs_err.items()}
        for w in np.unique(c["wr"])
    }
    res["lowo_cells_within_5pp"] = {k: int(np.sum(v <= 0.05)) for k, v in abs_err.items()}
    res["lowo_mae_by_class"] = {
        int(k): {n: float(np.nanmean(v[c["cls"] == k])) for n, v in abs_err.items()}
        for k in np.unique(c["cls"])
    }
    res["spearman_by_class"] = {
        int(k): _spearman(c["x"][c["cls"] == k], c["y"][c["cls"] == k]) for k in np.unique(c["cls"])
    }
    hk = np.isfinite(held_out["x"]) & np.isfinite(held_out["y"])
    h = _subset(held_out, hk)
    if len(h["x"]):
        for label, train in (
            ("all_fitted", c),
            ("other_lineages", _subset(c, c["wr"] != 0)),  # writer code 0 is the main study
        ):
            res[f"heldout_mae_{label}"] = {
                name: float(np.nanmean(np.abs(h["y"] - _predict(name, train, h))))
                for name in PREDICTORS
            }
    return res


def _cells_of(
    arrays: Mapping[str, WriterArrays],
    writers: Sequence[str],
    classes: Sequence[str],
    which: str,
) -> tuple[dict[str, np.ndarray], list[tuple[str, str, str]]]:
    """The cells of `writers` over branch x class as arrays: x (repair gap), y (B x D), corr (the
    identity), and the integer codes cls, br, wr. `which` is "point" (values are (cells,)) or
    "draws" (values are (draws, cells); the codes stay (cells,))."""
    xs, ys, corr, cls, br, wr = [], [], [], [], [], []
    index: list[tuple[str, str, str]] = []
    for wi, w in enumerate(writers):
        for b in BRANCHES:
            for ci, c in enumerate(classes):
                node = (arrays[w].get(b) or {}).get(c)
                if node is None:
                    continue
                q = node[0] if which == "point" else node[1]
                xs.append(q["gap"])
                ys.append(q["BxD"])
                corr.append(q["corrected"])
                cls.append(ci)
                br.append(BRANCHES.index(b))
                wr.append(wi)
                index.append((w, c, b))
    join = np.array if which == "point" else (lambda v: np.stack(v, axis=-1))
    cells = {"x": join(xs), "y": join(ys), "corr": join(corr)}
    cells |= {"cls": np.array(cls), "br": np.array(br), "wr": np.array(wr)}
    return cells, index


def _at(cells: Mapping[str, np.ndarray], d: int) -> dict[str, np.ndarray]:
    """Draw d of a dict of draw arrays (the code arrays are the same in every draw)."""
    return {k: (v[d] if v.ndim == 2 else v) for k, v in cells.items()}


def fit_across_cells(
    arrays: Mapping[str, WriterArrays], classes: Sequence[str], draws: int
) -> dict[str, Any]:
    """The test for one class scheme: point values and the bootstrap over worlds within writer."""
    fit_pt, index = _cells_of(arrays, FITTED, classes, "point")
    rep_pt, rep_index = _cells_of(arrays, ("rep",), classes, "point")
    rep_pt["wr"] = np.full(len(rep_pt["x"]), -1)
    fit_dr, _ = _cells_of(arrays, FITTED, classes, "draws")
    rep_dr, _ = _cells_of(arrays, ("rep",), classes, "draws")
    rep_dr["wr"] = np.full(len(rep_pt["x"]), -1)
    point = fit_statistics(fit_pt, rep_pt)
    boots = [fit_statistics(_at(fit_dr, d), _at(rep_dr, d)) for d in range(draws)]

    def est(getter: Any, scale: float = 1.0) -> Est | None:
        p = getter(point)
        if p is None:
            return None
        vals = np.array([getter(b) if getter(b) is not None else np.nan for b in boots], float)
        return ex.interval(scale * float(p), scale * vals, sum_worlds(arrays), point["n_cells"])

    out: dict[str, Any] = {
        "classes": list(classes),
        "n_cells": point["n_cells"],
        "cells": [
            {
                "writer": w,
                "class": c,
                "branch": b,
                "gap": float(x),
                "BxD": float(y),
                "corrected": float(k),
            }
            for (w, c, b), x, y, k in zip(
                index, fit_pt["x"], fit_pt["y"], fit_pt["corr"], strict=True
            )
        ],
        "heldout_cells": [
            {"writer": w, "class": c, "branch": b, "gap": float(x), "BxD": float(y)}
            for (w, c, b), x, y in zip(rep_index, rep_pt["x"], rep_pt["y"], strict=True)
        ],
        "spearman": est(lambda s: s.get("spearman")),
        "spearman_by_class": {
            classes[k]: est(lambda s, k=k: s["spearman_by_class"][k])
            for k in point["spearman_by_class"]
        },
        "slope": est(lambda s: s.get("slope")),
        "intercept": est(lambda s: s.get("intercept")),
        "lowo": {
            "mae": {n: est(lambda s, n=n: s["lowo_mae"][n], 100.0) for n in PREDICTORS},
            "cells_within_5pp": point["lowo_cells_within_5pp"],
            "mae_gap_minus_class": est(
                lambda s: s["lowo_mae"]["gap_linear"] - s["lowo_mae"]["class_mean"], 100.0
            ),
            "by_class": {
                classes[k]: {
                    n: est(lambda s, k=k, n=n: s["lowo_mae_by_class"][k][n], 100.0)
                    for n in PREDICTORS
                }
                for k in point["lowo_mae_by_class"]
            },
            "by_class_mae_gap_minus_class": {
                classes[k]: est(
                    lambda s, k=k: (
                        s["lowo_mae_by_class"][k]["gap_linear"]
                        - s["lowo_mae_by_class"][k]["class_mean"]
                    ),
                    100.0,
                )
                for k in point["lowo_mae_by_class"]
            },
            "by_writer": {
                FITTED[w]: {
                    n: est(lambda s, w=w, n=n: s["lowo_mae_by_writer"][w][n], 100.0)
                    for n in PREDICTORS
                }
                for w in point["lowo_mae_by_writer"]
            },
        },
    }
    for label in ("all_fitted", "other_lineages"):
        if f"heldout_mae_{label}" in point:
            out[f"heldout_{label}_mae"] = {
                n: est(lambda s, n=n, label=label: s[f"heldout_mae_{label}"][n], 100.0)
                for n in PREDICTORS
            }
            out[f"heldout_{label}_mae_gap_minus_class"] = est(
                lambda s, label=label: (
                    s[f"heldout_mae_{label}"]["gap_linear"]
                    - s[f"heldout_mae_{label}"]["class_mean"]
                ),
                100.0,
            )
    out["fit_line"] = _line_band(fit_pt["x"], fit_dr, boots)
    return out


def sum_worlds(arrays: Mapping[str, WriterArrays]) -> int:
    return sum(next(iter(arrays[w].values()))["definition"][2] for w in FITTED if arrays.get(w))


def _line_band(
    x_point: np.ndarray, fit_dr: Mapping[str, np.ndarray], boots: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The fitted line and its 95% bootstrap band on a grid of the repair gap (pp)."""
    lo, hi = float(np.nanmin(x_point)), float(np.nanmax(x_point))
    grid = np.linspace(lo - 0.05, hi + 0.05, 25)
    lines = np.array(
        [
            b["intercept"] + b["slope"] * grid
            for b in boots
            if b.get("slope") is not None and np.isfinite(b["slope"])
        ]
    )
    return {
        "grid": grid.tolist(),
        "lo": np.percentile(lines, 2.5, axis=0).tolist(),
        "hi": np.percentile(lines, 97.5, axis=0).tolist(),
    }


# ------------------------------------------------------------------------------------ driver
def analyse_tables(tables: Mapping[str, ex.Table], draws: int = DRAWS) -> dict[str, Any]:
    """Both blocks from the writers' item tables (writer key -> table)."""
    keys = [k for k, _ in WRITERS if k in tables]
    cells = {w: cell_nodes(tables[w], draws) for w in keys}
    res: dict[str, Any] = {
        "explore_status": ex.STATUS,
        "explore_draws": draws,
        "explore_seed": SEED,
        "explore_account_writers": {
            k: {
                "label": lab,
                "worlds": len(tables[k].worlds),
                "fitted": k in FITTED,
                "resampling_seed": SEED + i,
            }
            for i, (k, lab) in enumerate(WRITERS)
            if k in tables
        },
        "explore_account_cells": cells,
        "explore_account_predictors": PREDICTORS,
        "explore_account_headline": list(HEADLINE),
    }
    if not set(FITTED) | {"rep"} <= set(tables):
        return res
    arrays = {
        k: writer_arrays(tables[k], draws, SEED + i)
        for i, (k, _) in enumerate(WRITERS)
        if k in tables
    }
    res["explore_account_fit"] = {
        name: fit_across_cells(arrays, classes, draws) for name, classes in FIT_CLASSES.items()
    }
    return res


def main() -> None:
    out, *specs = sys.argv[1:]
    stages = dict(s.split(":", 1) for s in specs)
    if not stages:
        raise SystemExit(__doc__)
    tables = {
        k: ex.Table(read_jsonl(Path(p) / "analysis" / "e1_item_table.jsonl"))
        for k, p in stages.items()
    }
    res = analyse_tables(tables)
    o = Path(out)
    (o / "analysis").mkdir(parents=True, exist_ok=True)
    (o / "analysis" / "ex_account.json").write_text(json.dumps(res, indent=1))
    write_stage_manifest(o, "ex_account", [Path(p) for p in stages.values()], {})
    fit = res.get("explore_account_fit", {}).get("two_class", {})
    print(json.dumps({k: fit.get(k) for k in ("n_cells", "spearman", "slope", "intercept")}))


if __name__ == "__main__":
    main()

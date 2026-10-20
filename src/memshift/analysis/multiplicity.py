"""Multiplicity check on the directional E1 contrasts.

The main analysis reports the E1 contrasts without a multiplicity correction. This module adds one
as a robustness check. The family is every contrast of a stage's
`e1_metrics.json` predicted non-zero (`replication.PREDICTED_NONZERO`: the B effects on detection
and on Update, the D effects on repair, the B x D interaction), over all branches and both
granularities, so the family size is the same for every stage with the same cells. Every one of them
is predicted positive (B raises detection and Update, D raises repair, the interaction is
predicted > 0), the direction `predicted_sign` returns.

For each contrast the per-world item differences are taken exactly as `e1.contrasts` builds them
(`rendered.contrast_vectors`, the same code path with the item pairs kept), and bootstrapped with
the main analysis' resampling (`stats.world_bootstrap_draws`: world-clustered, 2,000 draws, seed
996). The recomputed estimate and 95% interval must equal the ones in `e1_metrics.json`; otherwise
the analysis stops. The one-sided p-value is read off the same bootstrap distribution: for a
positive prediction p = (1 + #{draws with mean <= 0}) / (1 + draws), mirrored for a negative one
(the +1 keeps p away from 0 for a finite number of draws, so the smallest p is 1 / 2,001).

Holm-Bonferroni (familywise alpha = 0.05) and Benjamini-Hochberg (q = 0.05, secondary) are applied
per stage. The bootstrap p-values inherit the main analysis' assumptions; contrasts that share
cells (the two detection contrasts, which coincide because detection does not depend on D; the
pooled and the D=1 B effect on Update) are positively dependent, which Holm allows and BH does for
positive dependence, but the correction treats each key as its own test.

Usage: python -m memshift.analysis.multiplicity OUT STAGE_DIR [STAGE_DIR ...]
(stage directories with analysis/e1_metrics.json and analysis/e1_item_table.jsonl)
Writes OUT/analysis/multiplicity.json and OUT/manifest.json.
"""

from __future__ import annotations

import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import rendered
from memshift.analysis.replication import PREDICTED_NONZERO
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import world_bootstrap_draws

ALPHA = 0.05  # Holm: familywise error rate
Q = 0.05  # Benjamini-Hochberg: false discovery rate
DRAWS = 2000  # the main analysis' bootstrap (`scoring.stats`)
SEED = 996
MATCH_TOL = 1e-12  # recomputed vs stored estimate and interval; they agree exactly on the data
STATUS = (
    "additional analysis: Holm and Benjamini-Hochberg over the contrasts predicted non-zero "
    "(replication.PREDICTED_NONZERO)"
)

Row = dict[str, Any]


def predicted_sign(key: str) -> int:
    """+1: every contrast predicted non-zero is predicted positive."""
    if not key.startswith(PREDICTED_NONZERO):
        raise ValueError(f"{key}: not a contrast predicted non-zero")
    return 1


def one_sided_p(draws: Sequence[float], sign: int = 1) -> float:
    """One-sided bootstrap p-value in the predicted direction: (1 + #{mean <= 0}) / (1 + draws) for
    a positive prediction, (1 + #{mean >= 0}) / (1 + draws) for a negative one."""
    if sign not in (1, -1):
        raise ValueError(f"sign must be +1 or -1, got {sign}")
    against = sum(d * sign <= 0 for d in draws)
    return (1 + against) / (1 + len(draws))


def holm(ps: Sequence[float], alpha: float = ALPHA) -> tuple[list[float], list[bool]]:
    """Holm-Bonferroni step-down: with p_(1) <= ... <= p_(m), the adjusted p_(i) is
    max_{j <= i} min(1, (m - j + 1) p_(j)); a hypothesis is rejected when its adjusted p <= alpha,
    which is the rule "reject p_(1), p_(2), ... until the first p_(i) > alpha / (m - i + 1)". Tied
    p-values get the same adjusted value. Returned in the order of `ps`."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * ps[i]))
        adjusted[i] = running
    return adjusted, [a <= alpha for a in adjusted]


def benjamini_hochberg(ps: Sequence[float], q: float = Q) -> tuple[list[float], list[bool]]:
    """Benjamini-Hochberg step-up: with p_(1) <= ... <= p_(m), the adjusted value (q-value) of
    p_(i) is min_{j >= i} min(1, m p_(j) / j); a hypothesis is rejected when it is <= q, which is
    the rule "reject p_(1), ..., p_(k) for the largest k with p_(k) <= k q / m". Tied p-values get
    the same adjusted value. Returned in the order of `ps`."""
    m = len(ps)
    order = sorted(range(m), key=lambda i: ps[i], reverse=True)
    adjusted = [0.0] * m
    running = 1.0
    for back, i in enumerate(order):
        running = min(running, min(1.0, m * ps[i] / (m - back)))
        adjusted[i] = running
    return adjusted, [a <= q for a in adjusted]


def _deviation(a: float, b: float) -> float:
    """|a - b|; 0 when both are NaN, infinite when only one is."""
    if math.isnan(a) or math.isnan(b):
        return 0.0 if math.isnan(a) and math.isnan(b) else math.inf
    return abs(a - b)


def _excludes_zero(lo: float, hi: float, sign: int) -> bool:
    """The 95% interval excludes 0 on the predicted side (`sign` = +1: lo > 0; -1: hi < 0)."""
    return lo > 0 if sign > 0 else hi < 0


def contrast_rows(
    rows: list[Row], metrics: dict[str, Any], stage: str = ""
) -> tuple[list[Row], float]:
    """One row per contrast of `metrics` predicted non-zero (branch, then the metrics' key order):
    the recomputed estimate with its bootstrap distribution's one-sided p-value, and the largest
    deviation of a recomputed estimate or interval bound from the one in `metrics`. Raises when
    a deviation exceeds `MATCH_TOL` (or is not a number where the stored value is)."""
    vectors = rendered.contrast_vectors(rows)
    out: list[Row] = []
    worst = 0.0
    for branch, contrasts in metrics["contrasts"].items():
        for key, stored in contrasts.items():
            if not key.startswith(PREDICTED_NONZERO):
                continue
            if (branch, key) not in vectors:
                raise ValueError(f"{stage} {branch}:{key}: the item table does not build it")
            by_world = {w: list(d.values()) for w, d in vectors[branch, key].items()}
            est, draws = world_bootstrap_draws(by_world, draws=DRAWS, seed=SEED)
            got = {"est": est.est, "lo": est.lo, "hi": est.hi}
            dev = max(_deviation(got[f], stored[f]) for f in got)
            if not dev <= MATCH_TOL:
                raise ValueError(
                    f"{stage} {branch}:{key}: recomputed {got} differs from e1_metrics.json "
                    f"{ {f: stored[f] for f in got} }"
                )
            worst = max(worst, dev)
            sign = predicted_sign(key)
            out.append(
                {
                    "branch": branch,
                    "key": key,
                    "predicted_sign": sign,
                    **got,
                    "n_worlds": est.n_worlds,
                    "n_items": est.n_items,
                    # no draws (no world has items): nothing can be rejected, p = 1
                    "p": one_sided_p(draws, sign) if draws else 1.0,
                    "interval_excludes_zero": _excludes_zero(est.lo, est.hi, sign),
                }
            )
    return out, worst


def correct(rows: list[Row]) -> list[Row]:
    """Add Holm and Benjamini-Hochberg to the rows of one stage (one family)."""
    ps = [r["p"] for r in rows]
    holm_p, holm_reject = holm(ps)
    bh_q, bh_reject = benjamini_hochberg(ps)
    return [
        {**r, "holm_p": hp, "survives_holm": hr, "bh_q": bq, "survives_bh": br}
        for r, hp, hr, bq, br in zip(rows, holm_p, holm_reject, bh_q, bh_reject, strict=True)
    ]


def _name(r: Row) -> str:
    return f"{r['branch']}:{r['key']}"


def summarise(rows: list[Row]) -> dict[str, Any]:
    """Counts of one stage's family, and where the correction changes the interval reading."""
    survivors = [r for r in rows if r["survives_holm"]]
    rest = [r for r in rows if not r["survives_holm"]]
    return {
        "n_contrasts": len(rows),
        "n_interval_excludes_zero": sum(r["interval_excludes_zero"] for r in rows),
        # the mirrored interval [-hi, -lo] excludes 0 on the predicted side
        "n_interval_excludes_zero_opposite": sum(
            _excludes_zero(-r["hi"], -r["lo"], r["predicted_sign"]) for r in rows
        ),
        "n_survive_holm": len(survivors),
        "n_survive_bh": sum(r["survives_bh"] for r in rows),
        "largest_holm_p_among_survivors": max((r["holm_p"] for r in survivors), default=None),
        "smallest_holm_p_among_nonsurvivors": min((r["holm_p"] for r in rest), default=None),
        "interval_excludes_zero_but_not_holm": [
            _name(r) for r in rows if r["interval_excludes_zero"] and not r["survives_holm"]
        ],
        "holm_but_interval_includes_zero": [
            _name(r) for r in rows if r["survives_holm"] and not r["interval_excludes_zero"]
        ],
    }


def analyse_stage(stage: Path) -> dict[str, Any]:
    """The family of one stage directory (analysis/e1_metrics.json and e1_item_table.jsonl)."""
    adir = stage / "analysis"
    metrics = json.loads((adir / "e1_metrics.json").read_text())
    table = read_jsonl(adir / "e1_item_table.jsonl")
    rows, worst = contrast_rows(table, metrics, stage.name)
    rows = correct(rows)
    return {
        "n_worlds": metrics.get("n_worlds"),
        "recomputed_vs_e1_metrics": {"max_abs_deviation": worst, "tolerance": MATCH_TOL},
        "summary": summarise(rows),
        "contrasts": rows,
    }


def analyse(stages: Sequence[Path], out: Path) -> dict[str, Any]:
    """Analyse each stage as its own family; results go to OUT/analysis/multiplicity.json."""
    names = [s.name for s in stages]
    if len(set(names)) != len(names):
        raise ValueError(f"stage directories must have distinct names: {names}")
    res = {
        "status": STATUS,
        "method": {
            "family": "contrasts of e1_metrics.json with a key starting with one of "
            f"{list(PREDICTED_NONZERO)}, all branches, one family per stage",
            "predicted_sign": "+1 for every contrast in the family",
            "p_value": "one-sided bootstrap over worlds, (1 + #{draws with mean <= 0}) / "
            "(1 + draws); the draws are those of the interval in e1_metrics.json",
            "bootstrap": {"draws": DRAWS, "seed": SEED},
            "holm": {"familywise_alpha": ALPHA},
            "benjamini_hochberg": {"q": Q},
        },
        "stages": {s.name: analyse_stage(s) for s in stages},
    }
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "multiplicity.json").write_text(json.dumps(res, indent=1))
    write_stage_manifest(out, "multiplicity", list(stages), {})
    return res


def main() -> None:
    out, *stages = sys.argv[1:]
    if not stages:
        raise SystemExit(
            "usage: python -m memshift.analysis.multiplicity OUT STAGE_DIR [STAGE_DIR ...]"
        )
    res = analyse([Path(s) for s in stages], Path(out))
    for name, stage in res["stages"].items():
        print(name, json.dumps(stage["summary"], indent=1))


if __name__ == "__main__":
    main()

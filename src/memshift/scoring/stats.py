"""Statistics (plan §6 "统计"): the unit is a per-item paired difference, clustered by world; the
confidence interval is a world-level bootstrap. Equivalence uses TOST on the world-level mean
difference with the bootstrap distribution (stage 2 only; stage 1 reports estimates and widths).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Estimate:
    est: float
    lo: float
    hi: float
    n_worlds: int
    n_items: int

    @property
    def width(self) -> float:
        return self.hi - self.lo

    def to_json(self) -> dict[str, float]:
        return {
            "est": self.est,
            "lo": self.lo,
            "hi": self.hi,
            "width": self.width,
            "n_worlds": self.n_worlds,
            "n_items": self.n_items,
        }


def pooled_mean(groups: Sequence[Sequence[float]]) -> float:
    n = sum(len(g) for g in groups)
    return sum(sum(g) for g in groups) / n if n else math.nan


def world_bootstrap_draws(
    by_world: Mapping[str, Sequence[float]],
    stat: Callable[[Sequence[Sequence[float]]], float] = pooled_mean,
    draws: int = 2000,
    seed: int = 996,
    alpha: float = 0.05,
) -> tuple[Estimate, list[float]]:
    """`world_bootstrap` with the sorted bootstrap distribution it took the interval from (the
    draws of `stat` that are not NaN; empty when there are no worlds)."""
    groups = [list(by_world[w]) for w in sorted(by_world) if len(by_world[w])]
    n_items = sum(len(g) for g in groups)
    if not groups:
        return Estimate(math.nan, math.nan, math.nan, 0, 0), []
    est = stat(groups)
    rng = random.Random(seed)
    vals = sorted(
        v
        for v in (stat([groups[rng.randrange(len(groups))] for _ in groups]) for _ in range(draws))
        if not math.isnan(v)
    )
    if not vals:
        return Estimate(est, math.nan, math.nan, len(groups), n_items), vals
    lo = vals[int(alpha / 2 * (len(vals) - 1))]
    hi = vals[int((1 - alpha / 2) * (len(vals) - 1))]
    return Estimate(est, lo, hi, len(groups), n_items), vals


def world_bootstrap(
    by_world: Mapping[str, Sequence[float]],
    stat: Callable[[Sequence[Sequence[float]]], float] = pooled_mean,
    draws: int = 2000,
    seed: int = 996,
    alpha: float = 0.05,
) -> Estimate:
    """Resample worlds with replacement; `stat` maps a list of per-world item lists to a number."""
    return world_bootstrap_draws(by_world, stat, draws, seed, alpha)[0]


def item_diffs(
    a: Mapping[str, Mapping[str, float]], b: Mapping[str, Mapping[str, float]]
) -> dict[str, dict[str, float]]:
    """Per world, a[item] - b[item] over the items present in both (item keys in sorted order)."""
    return {w: {k: a[w][k] - b[w][k] for k in sorted(set(a[w]) & set(b[w]))} for w in a if w in b}


def paired_diff(
    a: Mapping[str, Mapping[str, float]],
    b: Mapping[str, Mapping[str, float]],
    draws: int = 2000,
    seed: int = 996,
    alpha: float = 0.05,
) -> Estimate:
    """Mean of a[item] - b[item] over items present in both, clustered by world.

    `a` and `b` map world -> item key -> value (0/1 outcomes or rates).
    """
    diffs = {w: list(d.values()) for w, d in item_diffs(a, b).items()}
    return world_bootstrap(diffs, draws=draws, seed=seed, alpha=alpha)


def tost(est: Estimate, margin: float) -> bool:
    """Equivalent within +/- margin when the (1 - 2 alpha) interval lies inside it.

    Callers pass an Estimate computed with alpha = 0.10 for a 5% TOST.
    """
    return -margin < est.lo and est.hi < margin


def required_worlds(width_now: float, n_now: int, target_width: float) -> int:
    """Worlds needed for a target CI width, assuming width ~ 1/sqrt(n)."""
    if width_now <= 0 or target_width <= 0 or math.isnan(width_now):
        return n_now
    return max(n_now, math.ceil(n_now * (width_now / target_width) ** 2))

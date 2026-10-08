"""Exploratory sensitivity of the membership tests T1 and T3 (decisions.md 2026-09-30; not
pre-registered, no registered verdict changes; every key this module adds starts with `explore_`).

T1 (E2, `t1_sensitivity`, per tier and cell). The registered T1 is the 90% world-bootstrap interval
of Retention - 1, Retention = P(valid after | valid before), inside +/- 3 pp. Retention ignores
items that are wrong before the switch and right after it, and Retention <= 1 makes the upper bound
automatic, so T1 is a non-inferiority test of a conditional statistic. Added here, for every cell
of every tier present (so the cross-family switches of block M1 are covered once staged):
  explore_survival_recovery  survival P(valid after | valid before), recovery P(valid after |
                             invalid before), valid-before and valid-after rates, and the NET
                             unconditional change in ContractValid (valid after - valid before over
                             all items), each with a 90% world-bootstrap interval, plus the gain and
                             loss counts; `trivial` marks a cell in which no item changed (losses =
                             gains = 0, the by-construction pass of a layer that never re-derives);
  explore_t1_eps             the verdict at eps in {1, 2, 3, 5} pp for retention and for the net
                             change, each as non-inferiority (90% lower bound above -eps) and as
                             equivalence (TOST, both bounds inside +/- eps);
  explore_t1_by_class        retention and the net change per item class (`e1.ITEM_CLASSES`), with
                             the class's share of the valid-before items and the largest loss
                             within the class that the margin tolerates if nothing else is lost.
T3 (E3, `t3_by_class`, per horizon and system): the margin against the oracle context per item
class, the oracle's own ContractValid per class, each class's contribution to the pooled margin and
the verdict at several eps. `t4_slack` gives the factor by which a system's serving cost could
grow before T4 fails.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from memshift.analysis.e1 import ITEM_CLASSES
from memshift.scoring.stats import Estimate, paired_diff, tost, world_bootstrap

Row = dict[str, Any]
EPS_PP = (1, 2, 3, 5)  # margins of the sensitivity grid, percentage points
ALPHA = 0.10  # T1's interval: 90% world bootstrap (TOST at 5%)


def _est(by_world: Mapping[str, list[float]]) -> Estimate:
    return world_bootstrap(by_world, alpha=ALPHA)


def _verdicts(delta: Estimate, net: Estimate, eps_pp: tuple[int, ...] = EPS_PP) -> dict[str, Any]:
    """Per eps (pp): non-inferiority (lower bound > -eps) and TOST equivalence of the retention
    delta (Retention - 1) and of the net unconditional change."""
    out: dict[str, Any] = {}
    for e in eps_pp:
        m = e / 100
        out[str(e)] = {
            "retention_noninferior": bool(-m < delta.lo),
            "retention_equivalent": tost(delta, m),
            "net_noninferior": bool(-m < net.lo),
            "net_equivalent": tost(net, m),
        }
    return out


def _by_world(rows: list[Row], value: Any, cond: Any = None) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if cond is None or cond(r):
            out[r["world_id"]].append(float(value(r)))
    return out


def survival_recovery(rows: list[Row]) -> dict[str, Any]:
    """Survival, recovery and the net change of one cell (rows of one tier and system)."""
    before = lambda r: r["valid_before"]  # noqa: E731
    after = lambda r: r["valid_after"]  # noqa: E731
    gains = sum(bool(not r["valid_before"] and r["valid_after"]) for r in rows)
    losses = sum(bool(r["valid_before"] and not r["valid_after"]) for r in rows)
    n_before = sum(bool(r["valid_before"]) for r in rows)
    return {
        "n_items": len(rows),
        "n_valid_before": n_before,
        "valid_before": _est(_by_world(rows, before)).to_json(),
        "valid_after": _est(_by_world(rows, after)).to_json(),
        "survival": _est(_by_world(rows, after, before)).to_json(),
        "recovery": _est(_by_world(rows, after, lambda r: not r["valid_before"])).to_json(),
        "net": _est(
            _by_world(rows, lambda r: int(r["valid_after"]) - int(r["valid_before"]))
        ).to_json(),
        "gains": gains,
        "losses": losses,
        "stay_valid": n_before - losses,
        "stay_invalid": len(rows) - n_before - gains,
        "trivial": gains == 0 and losses == 0,
    }


def _cell_entries(rows: list[Row], margin: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """(survival/recovery entry, eps grid) of one cell."""
    sr = survival_recovery(rows)
    delta = _est(_by_world(rows, lambda r: int(r["valid_after"]) - 1, lambda r: r["valid_before"]))
    net = _est(_by_world(rows, lambda r: int(r["valid_after"]) - int(r["valid_before"])))
    sr["retention_delta"] = delta.to_json()
    grid = {
        "margin": margin,
        "retention_delta_90": delta.to_json(),
        "net_90": net.to_json(),
        "by_eps_pp": _verdicts(delta, net),
    }
    return sr, grid


def _class_entries(rows: list[Row], margin: float) -> dict[str, Any]:
    n_before = sum(bool(r["valid_before"]) for r in rows)
    out: dict[str, Any] = {}
    for cls, prefixes in ITEM_CLASSES.items():
        rs = [r for r in rows if str(r.get("slot_id", "")).startswith(prefixes)]
        nb = sum(bool(r["valid_before"]) for r in rs)
        share = nb / n_before if n_before else math.nan
        delta = _est(
            _by_world(rs, lambda r: int(r["valid_after"]) - 1, lambda r: r["valid_before"])
        )
        net = _est(_by_world(rs, lambda r: int(r["valid_after"]) - int(r["valid_before"])))
        out[cls] = {
            "n_items": len(rs),
            "n_valid_before": nb,
            "share_of_valid_before": share,
            "retention_delta_90": delta.to_json(),
            "net_90": net.to_json(),
            "equivalent": tost(delta, margin),
            "noninferior": bool(-margin < delta.lo),
            "by_eps_pp": _verdicts(delta, net),
            # loss l within the class moves the pooled delta by -share * l: the loss the margin
            # tolerates if nothing else is lost
            "tolerated_loss_at_margin": min(1.0, margin / share) if share and share > 0 else None,
        }
    return out


def t1_sensitivity(rows: list[Row], margin: float) -> dict[str, Any]:
    """The three `explore_*` T1 fields for every (tier, system) of the E2 survival rows."""
    sr: dict[str, Any] = {}
    eps: dict[str, Any] = {}
    cls: dict[str, Any] = {}
    for tier in sorted({r["tier"] for r in rows}):
        tr = [r for r in rows if r["tier"] == tier]
        for system in sorted({r["system"] for r in tr}):
            cell = [r for r in tr if r["system"] == system]
            sr[f"{tier}:{system}"], eps[f"{tier}:{system}"] = _cell_entries(cell, margin)
            cls[f"{tier}:{system}"] = _class_entries(cell, margin)
    return {
        "explore_survival_recovery": sr,
        "explore_t1_eps": eps,
        "explore_t1_by_class": cls,
    }


def t3_by_class(
    cv: Mapping[str, Mapping[str, Mapping[str, float]]],
    epsilon: float,
    eps_pp: tuple[int, ...] = EPS_PP,
) -> dict[str, dict[str, Any]]:
    """System -> item class -> the T3 margin on that class.

    `cv` maps system -> world -> fact id -> ContractValid at one horizon. For a system other than
    the oracle: `minus_oracle` (95% world-bootstrap interval, the interval of the registered T3),
    `T3_pass` (estimate >= -epsilon, the registered rule), the same rule at each eps in `eps_pp`,
    and `contribution`, the class's share of the paired items times its margin (the classes
    partition the items, so the contributions add up to the pooled margin). Every system also
    carries its own `contract_valid` per class; the oracle's is the ceiling T3 is measured from.
    """
    out: dict[str, dict[str, Any]] = {s: {} for s in cv}
    oracle = cv.get("oracle")
    paired = {
        s: sum(len(set(per[w]) & set(oracle[w])) for w in per if w in oracle)
        for s, per in cv.items()
        if oracle is not None
    }
    for cls, prefixes in ITEM_CLASSES.items():
        kept = {
            s: {w: {f: x for f, x in v.items() if f.startswith(prefixes)} for w, v in per.items()}
            for s, per in cv.items()
        }
        for s in cv:
            entry: dict[str, Any] = {
                "contract_valid": world_bootstrap(
                    {w: list(v.values()) for w, v in kept[s].items()}
                ).to_json()
            }
            if oracle is not None and s != "oracle":
                d = paired_diff(kept[s], kept["oracle"])
                total = paired[s]
                entry.update(
                    {
                        "minus_oracle": d.to_json(),
                        "T3_pass": bool(d.est >= -epsilon),
                        "T3_pass_at_eps_pp": {str(e): bool(d.est >= -e / 100) for e in eps_pp},
                        "share_of_items": d.n_items / total if total else math.nan,
                        "contribution": d.est * d.n_items / total if total else math.nan,
                    }
                )
            out[s][cls] = entry
    return out


def t4_slack(serve_ratio: float | None, ceiling: float) -> float | None:
    """How many times larger the serving cost per query could be before T4 fails."""
    return ceiling / serve_ratio if serve_ratio else None

"""Block C of prereg v2 §4: the construct check with writer-authored executable derivations (D1s).

The D1s cell repairs like D=1, but its replacement values (and, for definition items, its check)
come from a program the writer model writes, not from benchmark-authored rules or the gold
evaluator. On the same item table as the E1 contrasts this module reports the pre-registered list:
  C1  RepairCorrectness, D1s - D0 at B=1 source epoch (S_src and S_def), prediction > 0
  C2  D1 - D1s, and its share of the original D effect (D1 - D0): the part of the D effect that
      depends on benchmark-authored rules, prediction >= 0
  C3  B x D interaction on Update, [B effect | D1s] - [B effect | D0], prediction > 0 on S_def
  C4  D effect on missed invalidation with D1s as the D=1 arm, prediction 0 (D is not in the key)
plus program statistics per item type: executable rate, and agreement of the program value with
gold before the shift (t0). Contrasts on the cited-cluster cells are outside the list and carry the
`explore_` prefix. Estimates are the world-clustered paired bootstrap of the other E1 contrasts.

Exploratory (not on the list): C1 and C2 split by item class at source-epoch binding, with the
classes of `e1.ITEM_CLASSES` (structured fields `rule`, evaluated definitions `definition`, free
text `norule`) and the same repaired items as in the D-effect class split of the E1 contrasts:
`explore_C1_repair_D1s_minus_D0_source_<class>` and `explore_C2_repair_D1_minus_D1s_source_<class>`.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from memshift.analysis.e1 import ITEM_CLASSES, Cond, Row, _diff, _repair_matches, _repaired_in, _vec
from memshift.scoring.stats import Estimate, paired_diff, world_bootstrap

SYNTH_GRANULARITIES = ("source", "cited")


def synth_granularities(rows: list[Row]) -> list[str]:
    present = {r["system"] for r in rows}
    return [g for g in SYNTH_GRANULARITIES if f"holdmem_D1sB1_{g}" in present]


def is_synth(system: str) -> bool:
    return system.startswith("holdmem_D1sB")


def _ratio_of_sums(groups: Sequence[Sequence[float]]) -> float:
    """Numerators and denominators are interleaved per item: sum(num) / sum(den)."""
    num = sum(sum(g[0::2]) for g in groups)
    den = sum(sum(g[1::2]) for g in groups)
    return num / den if den else math.nan


def _share(num: dict[str, dict[str, float]], den: dict[str, dict[str, float]]) -> Estimate:
    """Ratio of pooled paired means over the items present in both, world-clustered bootstrap."""
    by_world = {
        w: [x for k in sorted(set(num[w]) & set(den[w])) for x in (num[w][k], den[w][k])]
        for w in num
        if w in den
    }
    est = world_bootstrap(by_world, stat=_ratio_of_sums)
    return replace(est, n_items=est.n_items // 2)  # each item was counted once per component


def contrasts(rows: list[Row], branch: str) -> dict[str, Estimate]:
    """C1-C4 for one branch; a contrast is skipped when a cell it needs is not in the run."""
    ch: Cond = lambda r: r["changed"]  # noqa: E731
    rep: Cond = lambda r: bool(r.get("repaired_now"))  # noqa: E731
    present = {r["system"] for r in rows}
    out: dict[str, Estimate] = {}
    for g in synth_granularities(rows):
        tag = "" if g == "source" else "explore_"
        d0, d1, d1s = f"holdmem_D0B1_{g}", f"holdmem_D1B1_{g}", f"holdmem_D1sB1_{g}"
        cv_s = _vec(rows, d1s, branch, "cv", rep)
        if d0 in present:
            cv_0 = _vec(rows, d0, branch, "cv", rep)
            out[f"{tag}C1_repair_D1s_minus_D0_{g}"] = paired_diff(cv_s, cv_0)
            out[f"{tag}C4_D1s_effect_missed_{g}"] = paired_diff(
                _vec(rows, d0, branch, "detected", ch), _vec(rows, d1s, branch, "detected", ch)
            )
        if d1 in present:
            cv_1 = _vec(rows, d1, branch, "cv", rep)
            out[f"{tag}C2_repair_D1_minus_D1s_{g}"] = paired_diff(cv_1, cv_s)
            if d0 in present:
                out[f"{tag}C2_share_of_D_effect_{g}"] = _share(_diff(cv_1, cv_s), _diff(cv_1, cv_0))
        if g == "source":  # exploratory class split of C1 and C2, over the items each cell repaired
            for cls, prefixes in ITEM_CLASSES.items():
                rc = _repaired_in(prefixes)
                cls_s = _vec(rows, d1s, branch, "cv", rc)
                if d0 in present:
                    out[f"explore_C1_repair_D1s_minus_D0_{g}_{cls}"] = paired_diff(
                        cls_s, _vec(rows, d0, branch, "cv", rc)
                    )
                if d1 in present:
                    out[f"explore_C2_repair_D1_minus_D1s_{g}_{cls}"] = paired_diff(
                        _vec(rows, d1, branch, "cv", rc), cls_s
                    )
        if {d0, "holdmem_D0B0", "holdmem_D1sB0"} <= present:
            u = {
                (d, b): _vec(rows, name, branch, "cv", ch)
                for (d, b), name in {
                    (1, 1): d1s,
                    (1, 0): "holdmem_D1sB0",
                    (0, 1): d0,
                    (0, 0): "holdmem_D0B0",
                }.items()
            }
            out[f"{tag}C3_BxD_interaction_update_D1s_{g}"] = paired_diff(
                _diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0])
            )
    return out


def item_type(fact_id: str) -> str:
    """The item type of experiments.e2.item_type, from a fact id: user attributes keep their id,
    contact attributes are `contact.<field>`, definitions `kappa.<id>`, the rest are per kind."""
    prefix = fact_id.split(".")[0]
    if prefix == "c":
        return f"contact.{fact_id.split('.')[-1]}"
    if prefix == "k":
        return f"kappa.{fact_id[2:]}"
    return {"s": "series", "a": "appointment", "t": "task_due"}.get(prefix, fact_id)


def _rate(by_world: dict[str, list[float]]) -> dict[str, float] | None:
    return world_bootstrap(by_world).to_json() if by_world else None


def program_stats(data: dict[str, list[Row]]) -> dict[str, Any]:
    """Per branch, D1s system and item type: how often the writer's program ran (exec) and how
    often its value at t0, before any shift, equals gold (agree; also among the executable ones).
    `exec_post` is the executable rate of the programs of later writes (repairs and appends, which
    include the re-requests under a new definition). Rates are per-item indicators pooled over
    worlds with the world bootstrap; `n` counts the t0 items."""
    gold0 = {
        (p["world_id"], p["branch"], p["fact_id"]): p
        for p in data["probes"]
        if int(p["tags"].get("k", 0)) == 0
    }
    cells: dict[tuple[str, str, str], dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for it in data["items"]:
        if not is_synth(it["system"]):
            continue
        w, b, slot = it["world_id"], it["branch"], it["slot_id"]
        for event in it["history"]:
            note = event.get("program")
            if note is None:
                continue
            for kind in (item_type(slot), "all"):
                acc = cells[(b, it["system"], kind)]
                if event["event"] != "derive":
                    acc["exec_post"][w].append(float(note["ok"]))
                    continue
                agree = bool(note["ok"]) and _repair_matches(gold0[w, b, slot], note["value"])
                acc["exec_t0"][w].append(float(note["ok"]))
                acc["agree_t0"][w].append(float(agree))
                if note["ok"]:
                    acc["agree_t0_given_exec"][w].append(float(agree))
    out: dict[str, Any] = {}
    for (b, system, kind), acc in sorted(cells.items()):
        out.setdefault(b, {}).setdefault(system, {})[kind] = {
            "n": sum(len(v) for v in acc["exec_t0"].values()),
            **{
                m: _rate(acc[m])
                for m in ("exec_t0", "agree_t0", "agree_t0_given_exec", "exec_post")
            },
        }
    return out

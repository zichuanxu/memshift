"""E1 analysis: item-paired detection and repair metrics per cell.

For every (world, branch, slot, checkpoint k) the gold at k is compared with the gold at t0:
  changed   gold value or status differs from t0
  detected  the cell invalidated the slot's original item in (t0, t_k] (an upsert cell: invalidated
            or superseded, `detection_events`)
MissedInvalidation = P(not detected | changed); FalseInvalidation = P(detected | unchanged);
RepairCorrectness  = P(probe ContractValid at k | item repaired at k) (behavioural, primary) and
                     P(repaired value matches gold at k) (mechanistic, secondary);
Update / Stability = ContractValid on changed / unchanged items (discriminative probes: all
changed items plus an equal-size seeded sample of unchanged ones per world-branch-k).
Contrasts are world-clustered paired differences with bootstrap intervals.

`_nofb` cells keep the D=1 checker but disable its programmatic fallback, isolating checking from
executable extraction in the repair contrasts.

`ex_e1_classes` adds class-stratified contrasts, the Eq. (1) decomposition and other splits of this
item table under `explore_*` keys; nothing above changes.

When D1s cells are present (D=1 with writer-authored programs instead of benchmark-authored rules),
`e1_synth` adds C1-C4 to the contrasts and per-type program statistics under `synth_programs`; runs
without D1s cells analyse exactly as before.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

from memshift.experiments.common import read_jsonl
from memshift.scoring.normalize import values_match
from memshift.scoring.stats import Estimate, paired_diff, world_bootstrap

Row = dict[str, Any]
Cond = Callable[[Row], bool]
# By fact-id prefix, following holdmem.checker.fallback's actual coverage (tests/test_nofb.py):
# "rule" and "definition" items have a programmatic extractor, "norule" items return NO_RULE.
ITEM_CLASSES = {
    "rule": ("c.", "s.", "t.", "a.", "u.timezone"),  # structured fields; timezone from settings
    "definition": ("k.",),  # evaluated person definitions
    "norule": ("u.employer", "u.manager", "u.home_city", "u.diet", "u.commute"),  # free text
}

HM_CELLS = (
    "holdmem_D0B0",
    "holdmem_D1B0",
    "holdmem_D0B1_source",
    "holdmem_D1B1_source",
    "holdmem_D0B1_cluster",
    "holdmem_D1B1_cluster",
    "holdmem_D0B1_cited",
    "holdmem_D1B1_cited",
    # D=1 no-fallback cells (checker-only)
    "holdmem_D1B0_nofb",
    "holdmem_D1B1_source_nofb",
    "holdmem_D1B1_cluster_nofb",
    "holdmem_D1B1_cited_nofb",
    # D1s: writer-authored programs
    "holdmem_D1sB0",
    "holdmem_D1sB1_source",
    "holdmem_D1sB1_cited",
    # control cells: oracle and scope keys, refresh without invalidation, template presentation,
    # thinking writer
    "holdmem_D0B1_gold",
    "holdmem_D1B1_gold",
    "holdmem_D0B1_scope",
    "holdmem_D1B1_scope",
    "holdmem_D0B0_refresh",
    "holdmem_D1B0_refresh",
    "holdmem_D0tB0",
    "holdmem_D0tB1_source",
    "holdmem_D0thinkB0",
    "holdmem_D0thinkB1_source",
    "holdmem_D1sB0_ra",
    "holdmem_D1sB1_source_ra",
    "holdmem_D0fullB1_source",
    # slot-keyed upsert cells and the D1s refresh cell
    "holdmem_D0B0_upsert",
    "holdmem_D1B0_upsert",
    "holdmem_D1sB0_upsert",
    "holdmem_D1sB0_refresh",
)


def detection_events(system: str) -> tuple[str, ...]:
    """History events that count as detection in a cell: `invalidate`; in an upsert cell also
    `supersede`, the retirement of a slot's older items by its new delta item."""
    return ("invalidate", "supersede") if "_upsert" in system else ("invalidate",)


def _key(r: Row) -> tuple[str, str, str, int]:
    return (r["world_id"], r["branch"], r["fact_id"], int(r["tags"].get("k", 0)))


def load(*run_dirs: Path) -> dict[str, list[Row]]:
    """Merge one or more E1 runs (e.g. two halves of the dev set and a supplement)."""
    data: dict[str, list[Row]] = {"probes": [], "answers": [], "items": []}
    for d in run_dirs:
        for key in data:
            data[key] += read_jsonl(d / f"{key}.jsonl")
    return data


def granularities(rows: list[Row]) -> list[str]:
    present = {r["system"] for r in rows}
    return [g for g in ("source", "cluster", "cited") if f"holdmem_D1B1_{g}" in present]


def _repair_matches(p: Row, value: Any) -> bool:
    if p["gold_status"] == "abstain":
        return value is None
    return p["gold_status"] == "answer" and values_match(p["vtype"], value, p["gold_value"])


def item_table(data: dict[str, list[Row]]) -> list[Row]:
    """One row per (world, branch, system, slot, k>0) with changed/detected/cv/repair flags."""
    gold = {_key(p): p for p in data["probes"]}
    cv = {(a["system"], *_key(a)): a["contract_valid"] for a in data["answers"]}
    systems = sorted({s for (s, *_rest) in cv})
    hist: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for it in data["items"]:
        hist[(it["world_id"], it["branch"], it["system"], it["slot_id"])].extend(it["history"])
    rows = []
    for (w, b, slot, k), p in sorted(gold.items()):
        if k == 0:
            continue
        p0 = gold[(w, b, slot, 0)]
        changed = (p["gold_value"], p["gold_status"]) != (p0["gold_value"], p0["gold_status"])
        for system in systems:
            row: Row = {
                "world_id": w,
                "branch": b,
                "system": system,
                "fact_id": slot,
                "vtype": p["vtype"],
                "k": k,
                "changed": changed,
                "cv": cv.get((system, w, b, slot, k)),
                "cv_pre": cv.get((system, w, b, slot, 0)),
            }
            events = hist.get((w, b, system, slot))
            if events is not None:
                upto = [h for h in events if h["ts"] <= p["as_of"]]
                row["detected"] = any(h["event"] in detection_events(system) for h in upto)
                now = [h for h in upto if h["event"] == "repair" and h["ts"] == p["as_of"]]
                row["repaired_now"] = bool(now)
                if now:
                    row["repair_mech"] = _repair_matches(p, now[-1]["value"])
            rows.append(row)
    return rows


def _by_world(rows: list[Row], field: str, cond: Cond) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if cond(r) and r.get(field) is not None:
            out[r["world_id"]].append(float(r[field]))
    return out


def discriminative(rows: list[Row], seed: int = 0) -> list[Row]:
    """All changed rows plus an equal-size seeded sample of unchanged rows per (w, b, k, system)."""
    groups: dict[tuple[Any, ...], list[Row]] = defaultdict(list)
    for r in rows:
        groups[(r["world_id"], r["branch"], r["k"], r["system"])].append(r)
    out = []
    for key, rs in sorted(groups.items()):
        asked = [r for r in rs if r.get("cv") is not None]
        ch = [r for r in asked if r["changed"]]
        un = sorted((r for r in asked if not r["changed"]), key=lambda r: r["fact_id"])
        rng = random.Random(f"{seed}|{key[0]}|{key[1]}|{key[2]}")
        out += ch + rng.sample(un, min(len(ch), len(un)))
    return out


def _complement(e: Estimate) -> Estimate:
    return Estimate(1 - e.est, 1 - e.hi, 1 - e.lo, e.n_worlds, e.n_items)


def cell_metrics(rows: list[Row], system: str, branch: str) -> dict[str, Any]:
    rs = [r for r in rows if r["system"] == system and r["branch"] == branch]
    res: dict[str, Estimate] = {}
    if rs and "detected" in rs[0]:
        res["MissedInvalidation"] = _complement(
            world_bootstrap(_by_world(rs, "detected", lambda r: r["changed"]))
        )
        res["FalseInvalidation"] = world_bootstrap(
            _by_world(rs, "detected", lambda r: not r["changed"])
        )
        rep: Cond = lambda r: bool(r.get("repaired_now"))  # noqa: E731
        rep_ch: Cond = lambda r: bool(r.get("repaired_now")) and r["changed"]  # noqa: E731
        res["RepairCorrectness"] = world_bootstrap(_by_world(rs, "cv", rep))
        res["RepairCorrectness_mech"] = world_bootstrap(_by_world(rs, "repair_mech", rep))
        res["RepairCorrectness_changed"] = world_bootstrap(_by_world(rs, "cv", rep_ch))
    disc = discriminative(rs)
    res["Update"] = world_bootstrap(_by_world(disc, "cv", lambda r: r["changed"]))
    res["Stability"] = world_bootstrap(_by_world(disc, "cv", lambda r: not r["changed"]))
    res["ContractValid_post"] = world_bootstrap(_by_world(rs, "cv", lambda r: True))
    return {k: v.to_json() for k, v in res.items()}


def _vec(
    rows: list[Row], system: str, branch: str, field: str, cond: Cond
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["system"] == system and r["branch"] == branch and cond(r) and r.get(field) is not None:
            out[r["world_id"]][f"{r['fact_id']}@{r['k']}"] = float(r[field])
    return out


def _diff(
    x: dict[str, dict[str, float]], y: dict[str, dict[str, float]]
) -> dict[str, dict[str, float]]:
    """Per-item x - y over the items present in both, per world."""
    return {w: {k: x[w][k] - y[w][k] for k in x[w].keys() & y.get(w, {}).keys()} for w in x}


def _repaired_in(prefixes: tuple[str, ...]) -> Cond:
    def cond(r: Row) -> bool:
        return bool(r.get("repaired_now")) and str(r["fact_id"]).startswith(prefixes)

    return cond


def contrasts(
    rows: list[Row], branch: str, paired_diff: Callable[..., Any] = paired_diff
) -> dict[str, Any]:
    """The E1 contrasts, world-clustered paired differences.

    `paired_diff` may be replaced by a function that keeps its two per-world item maps (the return
    value only needs `to_json()`): `analysis.rendered` takes each contrast's item-level values that
    way, so it pairs the same contrast across two conditions without restating its definition.
    """
    ch: Cond = lambda r: r["changed"]  # noqa: E731
    un: Cond = lambda r: not r["changed"]  # noqa: E731
    rep: Cond = lambda r: bool(r.get("repaired_now"))  # noqa: E731
    out: dict[str, Estimate] = {}
    grans = granularities(rows)
    for g in grans:
        for d in (0, 1):
            b1, b0 = f"holdmem_D{d}B1_{g}", f"holdmem_D{d}B0"
            out[f"B_effect_detected_D{d}_{g}"] = paired_diff(
                _vec(rows, b1, branch, "detected", ch), _vec(rows, b0, branch, "detected", ch)
            )
            out[f"B_effect_update_D{d}_{g}"] = paired_diff(
                _vec(rows, b1, branch, "cv", ch), _vec(rows, b0, branch, "cv", ch)
            )
        d1, d0 = f"holdmem_D1B1_{g}", f"holdmem_D0B1_{g}"
        out[f"D_effect_missed_{g}"] = paired_diff(
            _vec(rows, d0, branch, "detected", ch), _vec(rows, d1, branch, "detected", ch)
        )
        out[f"D_effect_false_inval_{g}"] = paired_diff(
            _vec(rows, d1, branch, "detected", un), _vec(rows, d0, branch, "detected", un)
        )
        out[f"D_effect_repair_{g}"] = paired_diff(
            _vec(rows, d1, branch, "cv", rep), _vec(rows, d0, branch, "cv", rep)
        )
        out[f"D_effect_repair_mech_{g}"] = paired_diff(
            _vec(rows, d1, branch, "repair_mech", rep), _vec(rows, d0, branch, "repair_mech", rep)
        )
    for g in grans:
        # B effect on Update, pooled over D
        a: dict[str, dict[str, float]] = defaultdict(dict)
        b: dict[str, dict[str, float]] = defaultdict(dict)
        for d in (0, 1):
            for w, v in _vec(rows, f"holdmem_D{d}B1_{g}", branch, "cv", ch).items():
                a[w].update({f"{k}|D{d}": x for k, x in v.items()})
            for w, v in _vec(rows, f"holdmem_D{d}B0", branch, "cv", ch).items():
                b[w].update({f"{k}|D{d}": x for k, x in v.items()})
        out[f"B_effect_update_pooled_{g}"] = paired_diff(a, b)
        # B x D interaction on Update, [B effect | D=1] - [B effect | D=0]
        u = {
            (d, bb): _vec(
                rows, f"holdmem_D{d}B1_{g}" if bb else f"holdmem_D{d}B0", branch, "cv", ch
            )
            for d in (0, 1)
            for bb in (0, 1)
        }
        out[f"BxD_interaction_update_{g}"] = paired_diff(
            _diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0])
        )
        rep_ch: Cond = lambda r: bool(r.get("repaired_now")) and r["changed"]  # noqa: E731
        out[f"D_effect_repair_changed_{g}"] = paired_diff(
            _vec(rows, f"holdmem_D1B1_{g}", branch, "cv", rep_ch),
            _vec(rows, f"holdmem_D0B1_{g}", branch, "cv", rep_ch),
        )
    # D effect on repair by item class.
    # D=1's programmatic extraction computes rule-covered structured items and definition items
    # exactly as gold is defined; free-text user attributes have no rule (checker only).
    for g in grans:
        for cls, prefixes in ITEM_CLASSES.items():
            rc = _repaired_in(prefixes)
            out[f"explore_D_effect_repair_{g}_{cls}"] = paired_diff(
                _vec(rows, f"holdmem_D1B1_{g}", branch, "cv", rc),
                _vec(rows, f"holdmem_D0B1_{g}", branch, "cv", rc),
            )
    # The D=1 fallback vs checker-only (_nofb) split, isolating the programmatic fallback's
    # contribution from the checker's, only reported when a run includes the _nofb cell.
    systems_present = {r["system"] for r in rows}
    for g in grans:
        d1, d1_nofb, d0 = f"holdmem_D1B1_{g}", f"holdmem_D1B1_{g}_nofb", f"holdmem_D0B1_{g}"
        if d1_nofb not in systems_present:
            continue
        out[f"explore_fallback_effect_repair_{g}"] = paired_diff(
            _vec(rows, d1, branch, "cv", rep), _vec(rows, d1_nofb, branch, "cv", rep)
        )
        out[f"explore_check_only_effect_repair_{g}"] = paired_diff(
            _vec(rows, d1_nofb, branch, "cv", rep), _vec(rows, d0, branch, "cv", rep)
        )
        for cls, prefixes in ITEM_CLASSES.items():
            rc = _repaired_in(prefixes)
            out[f"explore_fallback_effect_repair_{g}_{cls}"] = paired_diff(
                _vec(rows, d1, branch, "cv", rc), _vec(rows, d1_nofb, branch, "cv", rc)
            )
            out[f"explore_check_only_effect_repair_{g}_{cls}"] = paired_diff(
                _vec(rows, d1_nofb, branch, "cv", rc), _vec(rows, d0, branch, "cv", rc)
            )
    for fine in (g for g in grans if g != "source"):
        suffix = "" if fine == "cluster" else f"_{fine}"
        for d in (0, 1):
            out[f"granularity_false_inval_D{d}{suffix}"] = paired_diff(
                _vec(rows, f"holdmem_D{d}B1_source", branch, "detected", un),
                _vec(rows, f"holdmem_D{d}B1_{fine}", branch, "detected", un),
            )
            out[f"granularity_missed_D{d}{suffix}"] = paired_diff(
                _vec(rows, f"holdmem_D{d}B1_source", branch, "detected", ch),
                _vec(rows, f"holdmem_D{d}B1_{fine}", branch, "detected", ch),
            )
    from memshift.analysis import e1_synth  # lazy: e1_synth builds on this module's helpers

    out.update(e1_synth.contrasts(rows, branch))
    return {k: v.to_json() for k, v in out.items()}


def pre_intervention(data: dict[str, list[Row]]) -> dict[str, Any]:
    """H_E: JSON arm vs s_k cells before the shift; the six cells must coincide."""
    by_sys: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for a in data["answers"]:
        if int(a["tags"].get("k", 0)) == 0:
            key = f"{a['branch']}:{a['fact_id']}"
            by_sys[a["system"]][a["world_id"]][key] = float(a["contract_valid"])
    out: dict[str, Any] = {
        s: world_bootstrap({w: list(v.values()) for w, v in d.items()}).to_json()
        for s, d in sorted(by_sys.items())
    }
    if "holdmem_json_arm" in by_sys and "holdmem_D1B1_source" in by_sys:
        out["json_minus_sk"] = paired_diff(
            by_sys["holdmem_json_arm"], by_sys["holdmem_D1B1_source"]
        ).to_json()
    cells = [c for c in HM_CELLS if c in by_sys]
    out["cells_identical_answers"] = all(by_sys[c] == by_sys[cells[0]] for c in cells)
    return out


def analyse(*run_dirs: Path, out: Path | None = None) -> dict[str, Any]:
    """Analyse one or more merged runs; results go to (out or the first run)/analysis."""
    data = load(*run_dirs)
    rows = item_table(data)
    branches = sorted({r["branch"] for r in rows})
    systems = sorted({r["system"] for r in rows})
    result = {
        "pre_intervention": pre_intervention(data),
        "cells": {b: {s: cell_metrics(rows, s, b) for s in systems} for b in branches},
        "contrasts": {b: contrasts(rows, b) for b in branches},
        "n_worlds": len({r["world_id"] for r in rows}),
    }
    from memshift.analysis import e1_synth  # lazy, see `contrasts`

    if any(e1_synth.is_synth(s) for s in systems):
        result["synth_programs"] = e1_synth.program_stats(data)
    from memshift.analysis import ex_e1_classes  # lazy, see `contrasts`

    result.update(ex_e1_classes.analyse_rows(rows, run_dirs))  # every key explore_*
    adir = (out or run_dirs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "e1_metrics.json").write_text(json.dumps(result, indent=1))
    with (adir / "e1_item_table.jsonl").open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return result


if __name__ == "__main__":
    import sys

    # python -m memshift.analysis.e1 OUT_DIR RUN_DIR [RUN_DIR ...]
    res = analyse(*(Path(p) for p in sys.argv[2:]), out=Path(sys.argv[1]))
    print(json.dumps(res["contrasts"], indent=1)[:4000])

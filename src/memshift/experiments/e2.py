"""E2: writer-model switch S_wr and per-item agreement as a prediction of survival.

Tiers (all same lineage): same tier Qwen3.5-9B -> MiMo-V2.6-Distill-Qwen-9B; upgrade Qwen3.5-9B ->
Qwen3.8-27B; downgrade = the reverse of the upgrade. Switches to other model families from the main
writer are given as `--tiers cross:<model>` and labelled `cross_<model slug>`.

Test items: the dev worlds' slots at t0 = t_f - 1 min, sources frozen (gold unchanged, so the
same probes are asked before and after). Four cells (D x B, source-epoch binding) are built
with the old writer; `set_writer(new)` then invalidates every B=1 item and re-derives it.
Retention = P(ContractValid after | ContractValid before), per cell.

Calibration items: disjoint worlds (their own seeds) at three as-of times; both writers derive
each slot independently; an item agrees when both derivations are correct or both wrong
(typed match against gold). Per item type the agreement rate is the predictor of per-item
survival of D=0, B=1 items; we report the calibration slope and AUROC.

`warm(model, ...)` precomputes every writer call of a model into the response cache, so that a
model needing a coordination window (Qwen3.8-27B, TP=2) is served once, in one batch.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memshift.experiments.common import (
    MAIN_MODEL,
    answer_probes,
    client,
    new_run_dir,
    slug,
    write_jsonl,
)
from memshift.gen.gold import gold_at
from memshift.gen.model import Slot, World
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.gen.world import add_months
from memshift.holdmem.checker import check
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import derive
from memshift.llm import LLMClient
from memshift.probes import make_probes
from memshift.runinfo import world_record, write_env, write_manifest
from memshift.scoring.normalize import values_match

MIMO = "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B"
BIG = "Qwen/Qwen3.8-27B"
TIERS = {"same": (MAIN_MODEL, MIMO), "upgrade": (MAIN_MODEL, BIG), "downgrade": (BIG, MAIN_MODEL)}
CROSS_PREFIX = "cross:"  # other model families: "cross:<model>" switches the main writer
CELLS = [CellConfig(d, b, "source") for d in (0, 1) for b in (0, 1)]
CAL_MONTHS = (3, 6, 9)


def tier_models(tier: str) -> tuple[str, str]:
    """(old writer, new writer) for a named same-lineage tier or a cross-family token."""
    if tier.startswith(CROSS_PREFIX):
        return MAIN_MODEL, tier[len(CROSS_PREFIX) :]
    return TIERS[tier]


def tier_name(tier: str) -> str:
    """File-safe tier label: 'cross:meta-llama/Llama-3.1-8B-Instruct' ->
    'cross_meta-llama_Llama-3.1-8B-Instruct'; named tiers are unchanged."""
    if tier.startswith(CROSS_PREFIX):
        return "cross_" + slug(tier[len(CROSS_PREFIX) :])
    return tier


def item_type(slot: Slot) -> str:
    if slot.kind == "user_attr":
        return slot.slot_id
    if slot.kind == "contact_attr":
        return "contact." + slot.slot_id.split(".")[-1]
    if slot.kind == "kappa":
        return f"kappa.{slot.kappa_id}"
    return slot.kind


def correct(slot: Slot, value: Any, gold_value: Any, gold_status: str) -> bool:
    if gold_status == "abstain":
        return value is None
    return value is not None and values_match(slot.vtype, value, gold_value)


def test_times(world: World) -> list[datetime]:
    return [world.fork - timedelta(minutes=1)]


def cal_times(world: World) -> list[datetime]:
    return [add_months(world.start, m) - timedelta(minutes=1) for m in CAL_MONTHS]


def warm(model: str, seeds: list[int], cal_seeds: list[int], run_dir: Path) -> dict[str, Any]:
    """Derive every (slot, time) of the test and calibration items with `model` (fills cache)."""
    llm = client(model, run_dir)
    jobs: list[tuple[World, SourceLog, Slot, datetime]] = []
    plan = [(s, test_times) for s in seeds] + [(s, cal_times) for s in cal_seeds]
    for seed, times in plan:
        world = generate_branch(seed, "natural")
        log = SourceLog(world.events)
        for t in times(world):
            log.snapshot(t)
            jobs += [(world, log, slot, t) for slot in world.slots]
    with ThreadPoolExecutor(max_workers=llm.max_workers) as ex:
        list(ex.map(lambda j: derive(llm, j[0], j[1], j[2], j[3]), jobs))
    return {"model": model, "writes": len(jobs), "usage": llm.usage}


def calibration(
    old: LLMClient, new: LLMClient, cal_seeds: list[int]
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    jobs: list[tuple[World, SourceLog, Slot, datetime]] = []
    for seed in cal_seeds:
        world = generate_branch(seed, "natural")
        log = SourceLog(world.events)
        for t in cal_times(world):
            log.snapshot(t)
            jobs += [(world, log, slot, t) for slot in world.slots]

    def one(job: tuple[World, SourceLog, Slot, datetime]) -> dict[str, Any]:
        world, log, slot, t = job
        g = gold_at(world, log, slot, t)
        a = derive(old, world, log, slot, t)
        b = derive(new, world, log, slot, t)
        ca = correct(slot, a.value, g.value, g.status)
        cb = correct(slot, b.value, g.value, g.status)
        return {
            "world_id": world.world_id,
            "slot_id": slot.slot_id,
            "type": item_type(slot),
            "t": str(t),
            "old_correct": ca,
            "new_correct": cb,
            "agree": ca == cb,
            "new_check_pass": check(world, log, slot, t, b.value, b.citations).ok,
        }

    with ThreadPoolExecutor(max_workers=48) as ex:
        rows: list[dict[str, Any]] = list(ex.map(one, jobs))
    rates: dict[str, list[bool]] = {}
    for r in rows:
        rates.setdefault(r["type"], []).append(r["agree"])
    return rows, {k: sum(v) / len(v) for k, v in rates.items()}


def run_tier(token: str, seeds: list[int], cal_seeds: list[int], run_dir: Path) -> dict[str, Any]:
    old_model, new_model = tier_models(token)
    tier = tier_name(token)
    old, new = client(old_model, run_dir), client(new_model, run_dir)
    reader = client(MAIN_MODEL, run_dir)
    cal_rows, agree = calibration(old, new, cal_seeds)
    write_jsonl(run_dir / f"calibration_{tier}.jsonl", cal_rows)
    worlds = []
    rows = []
    for seed in seeds:
        world = generate_branch(seed, "natural")
        log = SourceLog(world.events)
        worlds.append(world_record(world))
        t0 = test_times(world)[0]
        probes = make_probes(world, log, [t0], tags={"tier": tier})
        by_slot = {s.slot_id: s for s in world.slots}
        for cfg in CELLS:
            hm = HoldMem(world, log, cfg, old, old_model)
            hm.build(t0)
            before, _ = answer_probes(hm.name, hm.serve, probes, reader, {"phase": "before"})
            hm.set_writer(new, new_model)
            stats = hm.maintain(t0)
            after, _ = answer_probes(hm.name, hm.serve, probes, reader, {"phase": "after"})
            write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in before + after), "a")
            for b, a in zip(before, after, strict=True):
                slot = by_slot[b.fact_id]
                item = hm.items[hm.by_slot[slot.slot_id][-1]]
                rows.append(
                    {
                        "tier": tier,
                        "world_id": world.world_id,
                        "system": hm.name,
                        "D": cfg.D,
                        "B": cfg.B,
                        "slot_id": slot.slot_id,
                        "type": item_type(slot),
                        "valid_before": b.contract_valid,
                        "valid_after": a.contract_valid,
                        "predicted": agree.get(item_type(slot)),
                        "check": (item.det or {}).get("check"),
                        "invalidated": stats["invalidated"],
                    }
                )
    write_jsonl(run_dir / "survival.jsonl", rows, "a")
    return {"tier": tier, "agreement_by_type": agree, "worlds": worlds}


def run(
    seeds: list[int],
    cal_seeds: list[int],
    tiers: list[str],
    name: str = "e2",
    out: Path | None = None,
) -> Path:
    run_dir = out or new_run_dir(name)
    summaries = [run_tier(t, seeds, cal_seeds, run_dir) for t in tiers]
    (run_dir / "tiers.json").write_text(json.dumps(summaries, indent=1, default=str))
    writer_models = sorted({m for t in tiers for m in tier_models(t)})
    models = {f"writer_{i}": m for i, m in enumerate(writer_models)}
    models["reader"] = MAIN_MODEL
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E2",
        [w for s in summaries for w in s["worlds"]],
        models,
        {
            "seeds": seeds,
            "calibration_seeds": cal_seeds,
            "calibration_months": list(CAL_MONTHS),
            "tiers": {tier_name(t): tier_models(t) for t in tiers},
            "cells": [c.name for c in CELLS],
        },
    )
    return run_dir


if __name__ == "__main__":
    import argparse

    from memshift.experiments.e1 import parse_seeds

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--cal-seeds", required=True)
    ap.add_argument("--tiers", default="same")
    ap.add_argument("--warm", default="", help="Only warm the cache for this writer model.")
    ap.add_argument("--name", default="e2")
    a = ap.parse_args()
    seeds, cal = parse_seeds(a.seeds), parse_seeds(a.cal_seeds)
    if a.warm:
        d = new_run_dir(f"{a.name}_warm")
        res = warm(a.warm, seeds, cal, d)
        (d / "warm.json").write_text(json.dumps(res, indent=1))
        write_manifest(d, "E2-warm", [], {"writer": a.warm}, {"seeds": seeds, "cal_seeds": cal})
        print(d)
    else:
        print(run(seeds, cal, a.tiers.split(","), a.name))

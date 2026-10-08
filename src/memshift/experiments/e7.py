"""E7 (plan §6): failure analysis.

conflict   unresolvable source conflicts (gen.shifts.apply_conflict): AbstainAccuracy of the
           HoldMem cells (source-epoch binding) and the zero system on the conflicting items one
           week after the fork, with the same number of non-conflicting contact items as controls.
cons       consent revocation S_cons: the revoked agent asks every item one week after the fork;
           ScopeLeak = P(answered | gold blocked), BlockAccuracy = P(blocked | gold blocked), for
           HoldMem B=0 / B=1 and the zero system (which filters sources but cannot say "blocked").
implicit   implicit definitions: if the person's definition changes but is never stated, no
           system can see the change; the survival ceiling is P(value under v1 == gold under v2)
           over kappa items, computed from gold alone (no model calls).
The S2 read-set analysis (missed invalidation of scope-predicate items under object-cluster keys)
and the reader swap (theta_r) are computed from E1-style runs by `memshift.analysis.e7`.
"""

from __future__ import annotations

import json
import random
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import ZeroSystem
from memshift.experiments.common import (
    MAIN_MODEL,
    Answer,
    answer_probes,
    client,
    new_run_dir,
    write_jsonl,
)
from memshift.experiments.e1 import _par, parse_seeds
from memshift.gen.gold import gold_at
from memshift.gen.kappa import evaluate
from memshift.gen.model import World
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.llm import LLMClient
from memshift.probes import Probe, make_probe
from memshift.runinfo import world_record, write_env, write_manifest
from memshift.scoring.normalize import values_match

CELLS = [CellConfig(d, b, "source") for d in (0, 1) for b in (0, 1)]


def _systems(world: World, log: SourceLog, llm: LLMClient) -> dict[str, Any]:
    t0 = world.fork - timedelta(minutes=1)
    cells = {c.name: HoldMem(world, log, c, llm, MAIN_MODEL) for c in CELLS}
    _par(partial(HoldMem.build, t=t0), list(cells.values()))
    _par(partial(HoldMem.maintain, t=world.fork + timedelta(weeks=1)), list(cells.values()))
    return {**cells, "zero": ZeroSystem(world, log)}


def _answer(item: tuple[str, Any], probes: list[Probe], llm: LLMClient) -> list[Answer]:
    return answer_probes(item[0], item[1].serve, probes, llm)[0]


def _ask(world: World, log: SourceLog, probes: list[Probe], llm: LLMClient) -> list[Answer]:
    systems = _systems(world, log, llm)
    work = partial(_answer, probes=probes, llm=llm)
    return [a for ans in _par(work, list(systems.items()), 4) for a in ans]


def run_conflict(seed: int, run_dir: Path, llm: LLMClient) -> dict[str, Any]:
    world = generate_branch(seed, "conflict")
    log = SourceLog(world.events)
    t1 = world.fork + timedelta(weeks=1)
    targets = set(world.params["shift_targets"])
    pool = sorted(s.slot_id for s in world.slots if s.kind == "contact_attr")
    controls = random.Random(seed).sample([s for s in pool if s not in targets], len(targets))
    probes = [
        make_probe(
            world, log, world.slot(sid), t1, tags={"e7": "conflict", "conflict": sid in targets}
        )
        for sid in sorted(targets) + controls
    ]
    answers = _ask(world, log, probes, llm)
    write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
    return {"seed": seed, "world": world_record(world), "branch": "conflict"}


def run_cons(seed: int, run_dir: Path, llm: LLMClient) -> dict[str, Any]:
    world = generate_branch(seed, "cons")
    log = SourceLog(world.events)
    agent, scope = world.params["shift_targets"]
    t1 = world.fork + timedelta(weeks=1)
    probes = [
        make_probe(world, log, s, t1, agent_id=agent, tags={"e7": "cons"}) for s in world.slots
    ]
    answers = _ask(world, log, probes, llm)
    write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
    return {"seed": seed, "world": world_record(world), "branch": "cons", "revoked": [agent, scope]}


def implicit_ceiling(seeds: list[int]) -> dict[str, Any]:
    """P(v1 value == v2 gold) for kappa items one week after the fork (no model calls)."""
    rows = []
    for seed in seeds:
        world = generate_branch(seed, "def")
        log = SourceLog(world.events)
        t1 = world.fork + timedelta(weeks=1)
        snap = log.snapshot(t1)
        for slot in world.slots:
            if not slot.kappa_id or world.kappa_version(slot.kappa_id, t1) != "v2":
                continue
            v1, _ = evaluate(slot.kappa_id, snap, t1, "v1")
            gold = gold_at(world, log, slot, t1)
            rows.append(
                {
                    "world_id": world.world_id,
                    "kappa": slot.kappa_id,
                    "survives": bool(values_match(slot.vtype, v1, gold.value)),
                }
            )
    by: dict[str, list[bool]] = {}
    for r in rows:
        by.setdefault(str(r["kappa"]), []).append(bool(r["survives"]))
    return {
        "items": len(rows),
        "ceiling": sum(bool(r["survives"]) for r in rows) / max(1, len(rows)),
        "by_kappa": {k: sum(v) / len(v) for k, v in sorted(by.items())},
    }


def run(seeds: list[int], parts: list[str], name: str = "e7", out: Path | None = None) -> Path:
    run_dir = out or new_run_dir(name)
    llm = client(MAIN_MODEL, run_dir)
    worlds: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}
    if "conflict" in parts:
        worlds += _par(lambda s: run_conflict(s, run_dir, llm), seeds, 3)
    if "cons" in parts:
        worlds += _par(lambda s: run_cons(s, run_dir, llm), seeds, 3)
    if "implicit" in parts:
        summary["implicit"] = implicit_ceiling(seeds)
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "analysis" / "e7_summary.json").write_text(json.dumps(summary, indent=1))
    write_jsonl(run_dir / "world_summaries.jsonl", worlds)
    models = {"writer": MAIN_MODEL, "reader": MAIN_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E7",
        [w["world"] for w in worlds],
        models,
        {"seeds": seeds, "parts": parts, "cells": [c.name for c in CELLS]},
    )
    return run_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--parts", default="conflict,cons,implicit")
    ap.add_argument("--name", default="e7")
    a = ap.parse_args()
    print(run(parse_seeds(a.seeds), a.parts.split(","), a.name))

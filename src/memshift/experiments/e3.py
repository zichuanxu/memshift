"""E3: zero system, plain RAG and cost at horizons of 1, 3 and 12 months.

Static ContractValid: at each horizon h (t_h = start + h months - 1 min, natural branch) every
slot is probed against the zero system, plain RAG, HoldMem (1,1) (source-epoch binding, built cold
at t_h with the full D=1 write path) and the oracle. Update / Stability after S_src and S_def for
the same systems come from E1.

Five-part cost per system over a horizon, at q = 20 queries/day:
  ingest     HoldMem: tokens of the first derivation of every slot; RAG: embedding tokens of
             every chunk of the live corpus; zero: 0.
  maintain   HoldMem: (# daily-batch invalidations under source-epoch keys, counted by replaying
             keys without any model call) x (measured mean tokens per write); RAG: re-embedding of
             inserted and updated objects; zero: 0.
  serve      q x days x measured mean reader tokens per probe.
  verify     HoldMem D=1 checker CPU seconds, scaled by the write count.
  storage    bytes of the derived store (HoldMem items; RAG vectors); zero: 0.
The full re-read reference is the approx-token count of every live source object at t_h, per
query. Maintenance counts are "counted x measured unit cost", labelled as such in the report.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import CHUNK_TOKENS, Oracle, PlainRAG, ZeroSystem, chunk
from memshift.experiments.common import (
    EMBED_MODEL,
    MAIN_MODEL,
    Answer,
    answer_probes,
    client,
    new_run_dir,
    write_jsonl,
)
from memshift.experiments.e1 import _par, embed_client, parse_seeds
from memshift.gen.model import World
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.gen.world import add_months
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.llm import LLMClient
from memshift.probes import Probe, make_probes
from memshift.retrieval import Doc
from memshift.runinfo import world_record, write_env, write_manifest
from memshift.textutil import approx_tokens

HORIZONS = (1, 3, 12)
QUERIES_PER_DAY = 20


def horizon_time(world: World, months: int) -> datetime:
    return add_months(world.start, months) - timedelta(minutes=1)


def daily_invalidations(world: World, log: SourceLog, t_end: datetime) -> int:
    """Invalidations of source-epoch keyed items with one maintenance batch per day."""
    count = 0
    prev = log.source_epochs(world.start)
    kprev = {
        s.slot_id: world.kappa_version(s.kappa_id, world.start) for s in world.slots if s.kappa_id
    }
    day = world.start + timedelta(days=1)
    while day <= t_end:
        ep = log.source_epochs(day)
        changed = {s for s in ep if ep.get(s) != prev.get(s)}
        for slot in world.slots:
            bump = bool(changed & set(slot.sources))
            if slot.kappa_id:
                kv = world.kappa_version(slot.kappa_id, day)
                bump = bump or kv != kprev[slot.slot_id]
                kprev[slot.slot_id] = kv
            count += bump
        prev = ep
        day += timedelta(days=1)
    return count


def changed_object_tokens(log: SourceLog, t0: datetime, t1: datetime) -> int:
    """Embedding tokens to re-embed objects inserted or updated in (t0, t1]."""
    return sum(
        approx_tokens(e.payload.get("text", "")) for e in log.between(t0, t1) if e.op != "delete"
    )


def _answer(item: tuple[str, Any], probes: list[Probe], llm: LLMClient) -> list[Answer]:
    return answer_probes(item[0], item[1].serve, probes, llm)[0]


def run_world(seed: int, run_dir: Path) -> dict[str, Any]:
    llm = client(MAIN_MODEL, run_dir)
    embed = embed_client()
    world = generate_branch(seed, "natural")
    log = SourceLog(world.events)
    out: dict[str, Any] = {"seed": seed, "world": world_record(world), "horizons": {}}
    for h in HORIZONS:
        t = horizon_time(world, h)
        days = (t - world.start).days + 1
        hm = HoldMem(world, log, CellConfig(1, 1, "source"), llm, MAIN_MODEL)
        hm.build(t, apply_fallback=True)
        systems: dict[str, Any] = {
            hm.name: hm,
            "zero": ZeroSystem(world, log),
            "oracle": Oracle(world, log),
        }
        if embed is not None:
            systems["rag"] = PlainRAG(world, log, embed)
        probes = make_probes(world, log, [t], tags={"horizon_months": h})
        answers: list[Answer] = []
        for ans in _par(partial(_answer, probes=probes, llm=llm), list(systems.items()), 4):
            answers += ans
        write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
        write_jsonl(run_dir / "probes.jsonl", (p.to_json() for p in probes), mode="a")

        snap = log.snapshot(t)
        writes = len(world.slots)
        tokens_per_write = hm.cost["ingest_tokens"] / writes
        inval = daily_invalidations(world, log, t)
        per_probe = {
            name: sum(a.prompt_tokens + a.completion_tokens for a in answers if a.system == name)
            / max(1, sum(a.system == name for a in answers))
            for name in systems
        }
        queries = QUERIES_PER_DAY * days
        costs: dict[str, dict[str, float]] = {
            hm.name: {
                "ingest_tokens": hm.cost["ingest_tokens"],
                "maintain_tokens": inval * tokens_per_write,
                "maintain_writes_counted": inval,
                "serve_tokens": queries * per_probe[hm.name],
                "verify_cpu_s": hm.cost["verify_cpu_s"] * (1 + inval / writes),
                "storage_bytes": hm.cost["storage_bytes"],
            },
            "zero": {
                "ingest_tokens": 0,
                "maintain_tokens": 0,
                "serve_tokens": queries * per_probe["zero"],
                "verify_cpu_s": 0.0,
                "storage_bytes": 0,
            },
        }
        if "rag" in systems:
            corpus = [
                c
                for o in snap.all()
                for c in chunk(Doc(o.object_id, o.text, o.updated, o.revision), CHUNK_TOKENS)
            ]
            costs["rag"] = {
                "ingest_tokens": sum(approx_tokens(c.text) for c in corpus),
                "maintain_tokens": changed_object_tokens(log, world.start, t),
                "serve_tokens": queries * per_probe["rag"],
                "verify_cpu_s": 0.0,
                "storage_bytes": systems["rag"].cost["storage_bytes"],
            }
        out["horizons"][h] = {
            "t": str(t),
            "days": days,
            "queries": queries,
            "full_reread_tokens_per_query": sum(approx_tokens(o.text) for o in snap.all()),
            "reader_tokens_per_probe": per_probe,
            "costs": costs,
            "contract_valid": {
                name: sum(a.contract_valid for a in answers if a.system == name) / len(probes)
                for name in systems
            },
        }
    return out


def run(seeds: list[int], name: str = "e3", out: Path | None = None) -> Path:
    run_dir = out or new_run_dir(name)
    results = _par(lambda s: run_world(s, run_dir), seeds, 3)
    write_jsonl(run_dir / "world_results.jsonl", results)
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "analysis" / "e3_raw.json").write_text(json.dumps(results, indent=1, default=str))
    models = {"writer": MAIN_MODEL, "reader": MAIN_MODEL, "embedding": EMBED_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E3",
        [r["world"] for r in results],
        models,
        {
            "seeds": seeds,
            "horizons_months": list(HORIZONS),
            "queries_per_day": QUERIES_PER_DAY,
            "holdmem_cell": "D1B1_source, cold build with full D=1 path",
            "maintenance_cost": "daily-batch invalidations counted by key replay x measured tokens",
        },
    )
    return run_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--name", default="e3")
    a = ap.parse_args()
    print(run(parse_seeds(a.seeds), a.name))

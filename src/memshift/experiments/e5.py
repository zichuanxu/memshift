"""E5: native systems (Mem0, Graphiti, A-Mem, MemOS, LightRAG, HippoRAG 2, Cognee) under S_src
and S_def, the orphan test for black-box B inference, and cost.

Per world and native system: the natural prefix up to t0 = t_f - 1 min is ingested once and the
store is snapshotted, so that the prefix is ingested once per world. Every branch (src, def, orphan)
restores the snapshot and ingests that branch's events after t0 incrementally, probing every slot at
t_f + {1, 2, 4, 8} weeks (orphan: 1 week only). All probes are also asked at t0 (static
ContractValid). The shared reader, template and 4k budget apply. White-box references for the same
worlds and branches come from `memshift.experiments.e1` (cells at source-epoch binding).

`--measure` ingests one world's prefix only and reports native t_i (LLM calls, tokens and wall
seconds per attempted event), which sets the number of worlds the compute budget allows;
`--measure-events N` caps the measured prefix at its first N events. Each world's result is
appended to `world_results.jsonl` as soon as it finishes.

Split runs (same systems, settings, events and probes). `--phase prefix`
ingests the prefix, snapshots it and answers the t0 probes; `--phase branch --prefix-run RUN`
restores that run's snapshot and runs the branches given by `--branches` (one per container lets a
world's three branches run in parallel; they are independent after the snapshot). `--resume-from
RUN` copies the prefix store of an interrupted run of the same world and system into this run
before ingesting: the store's ledger resumes ingestion after the last recorded unit, and units that
failed stay failed. Usage counters of the interrupted part are not recovered, so a resumed run's
prefix cost is not reported (native cost per event comes from the `--measure` runs); its ingested
and failed units are counted from the store (ledger, failures.jsonl).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from memshift.adapters.native import (
    FAILURES_FILE,
    STORE_MARKER,
    GraphitiSystem,
    Mem0System,
    NativeSystem,
    copy_store,
)
from memshift.adapters.native_amem import AMemSystem
from memshift.adapters.native_cognee import CogneeSystem
from memshift.adapters.native_hipporag import HippoRAGSystem
from memshift.adapters.native_lightrag import LightRAGSystem
from memshift.adapters.native_memos import MemOSSystem
from memshift.experiments.common import (
    EMBED_MODEL,
    MAIN_MODEL,
    answer_probes,
    client,
    endpoints,
    new_run_dir,
    world_params,
    write_jsonl,
)
from memshift.experiments.e1 import parse_seeds
from memshift.gen.model import World
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.probes import make_probes
from memshift.runinfo import REPO, world_record, write_env, write_manifest

NATIVE: dict[str, type[NativeSystem]] = {
    "mem0": Mem0System,
    "graphiti": GraphitiSystem,
    "amem": AMemSystem,
    "cognee": CogneeSystem,
    "hipporag": HippoRAGSystem,
    "lightrag": LightRAGSystem,
    "memos": MemOSSystem,
}
OFFSETS = {"src": (1, 2, 4, 8), "def": (1, 2, 4, 8), "orphan": (1,)}
PHASES = ("all", "prefix", "branch")
NATIVE_LLM = "MEMSHIFT_NATIVE_LLM"  # endpoints key for the long-context server (>= 32k tokens)


GRAPHITI_BATCH = 1  # >1: Graphiti's documented bulk path (same extraction and invalidation)
# >1: the documented bulk ingestion of HippoRAG 2 (`index(list)`) and LightRAG (`ainsert(list)`,
# with as many documents in parallel), and concurrent MemOS extractions with adds kept in event
# order. Batches hold distinct objects in time order and never cross a probe time. A-Mem, Cognee
# and Mem0 are unaffected.
NATIVE_BATCH = 1
ON_ERROR = "raise"  # "skip": a unit whose ingestion fails is recorded (failures.jsonl) and skipped


def _make(kind: str, world: World, store: Path, llm_url: str, embed_url: str) -> NativeSystem:
    kw: dict[str, Any] = {"batch_size": GRAPHITI_BATCH} if kind == "graphiti" else {}
    if NATIVE_BATCH > 1 and kind in ("hipporag", "lightrag"):
        kw["batch_size"] = NATIVE_BATCH
    if NATIVE_BATCH > 1 and kind == "lightrag":
        # documents in parallel and, so that the server's KV cache is not oversubscribed, as many
        # concurrent LLM calls per store (library default 4)
        kw["options"] = {"max_parallel_insert": NATIVE_BATCH, "llm_model_max_async": NATIVE_BATCH}
    if NATIVE_BATCH > 1 and kind == "memos":
        kw["workers"] = NATIVE_BATCH
    kw["on_error"] = ON_ERROR
    return NATIVE[kind](
        world, store, llm_url, MAIN_MODEL, embed_url, EMBED_MODEL, world.world_id, **kw
    )


def _run_store(run: str, root: Path) -> Path:
    """The store directory of world and system `root.name` inside the earlier run `run`."""
    return REPO / "runs" / run / "stores" / root.name


def store_totals(system: NativeSystem) -> dict[str, int]:
    """Units recorded in the store itself: ledger entries and failure records."""
    f = system.store_dir / FAILURES_FILE
    failed = len(f.read_text(encoding="utf-8").splitlines()) if f.is_file() else 0
    return {"ingested": len(system.ledger), "failed": failed}


def run_world(
    seed: int,
    kind: str,
    run_dir: Path,
    params: dict[str, Any] | None,
    measure_only: bool = False,
    measure_events: int | None = None,
    phase: str = "all",
    resume_from: str | None = None,
    prefix_run: str | None = None,
    branches: tuple[str, ...] = tuple(OFFSETS),
) -> dict[str, Any]:
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    if (phase == "branch") != (prefix_run is not None):
        raise ValueError("--prefix-run is required by, and only by, --phase branch")
    eps = endpoints()
    llm_url = eps.get(NATIVE_LLM, eps[MAIN_MODEL]).split("|")[0]
    embed_url = eps[EMBED_MODEL]
    reader = client(MAIN_MODEL, run_dir)
    natural = generate_branch(seed, "natural", params)
    nlog = SourceLog(natural.events)
    t0 = natural.fork - timedelta(minutes=1)
    root = run_dir / "stores" / f"{natural.world_id}_{kind}"
    out: dict[str, Any] = {"seed": seed, "system": kind, "world": world_record(natural)}
    answers: list[Any] = []
    probe_rows: list[dict[str, Any]] = []
    if phase == "branch":
        assert prefix_run is not None
        snapshot = _run_store(prefix_run, root) / "snapshot"
        if not (snapshot / STORE_MARKER).is_file():
            raise FileNotFoundError(f"no prefix snapshot in {snapshot}")
        out.update({"phase": phase, "prefix_run": prefix_run})
    else:
        if resume_from:
            copy_store(_run_store(resume_from, root) / "prefix", root / "prefix")
        prefix = _make(kind, natural, root / "prefix", llm_url, embed_url)
        prefix_events = nlog.upto(t0)
        if measure_only and measure_events:
            prefix_events = prefix_events[:measure_events]
        ingest_stats = prefix.ingest(prefix_events, "ingest")
        attempted = ingest_stats["events"] - ingest_stats["skipped"]
        out.update(
            {
                "prefix_events": len(prefix_events),
                "prefix_ingest": ingest_stats,
                # includes units that failed under on_error="skip"
                "prefix_stats": dict(prefix.stats),
                "t_i": {
                    k: v / max(1, attempted)
                    for k, v in ingest_stats.items()
                    if isinstance(v, int | float)
                },
                "describe": prefix.describe(),
                "identifiable": prefix.identifiable,
            }
        )
        if resume_from:
            out["prefix_stats"].update(store_totals(prefix))
            out.update({"resumed_from": resume_from, "t_i": None})
        if phase != "all":
            out["phase"] = phase
        if measure_only:
            prefix.close()
            return out
        snapshot = root / "snapshot"
        prefix.snapshot(snapshot)
        pre = make_probes(natural, nlog, [t0], tags={"phase": "pre", "k": 0})
        answers += answer_probes(kind, prefix.serve, pre, reader)[0]
        prefix.close()
        probe_rows += [p.to_json() for p in pre]
    for branch in branches if phase != "prefix" else ():
        offsets = OFFSETS[branch]
        world = generate_branch(seed, branch, params)
        log = SourceLog(world.events)
        system = _make(kind, world, root / branch, llm_url, embed_url)
        system.restore(snapshot)
        out.setdefault("identifiable", system.identifiable)
        prev = t0
        for k in offsets:
            t = world.fork + timedelta(weeks=k)
            system.ingest(log.between(prev, t), "maintain")
            probes = make_probes(world, log, [t], tags={"phase": "post", "k": k})
            answers += answer_probes(kind, system.serve, probes, reader)[0]
            probe_rows += [p.to_json() for p in probes]
            prev = t
        out.setdefault("branches", {})[branch] = {"cost": system.cost, "stats": system.stats}
        system.close()
    write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
    write_jsonl(run_dir / "probes.jsonl", probe_rows, mode="a")
    return out


def run(
    seeds: list[int],
    systems: list[str],
    name: str = "e5",
    params: dict[str, Any] | None = None,
    measure_only: bool = False,
    measure_events: int | None = None,
    phase: str = "all",
    resume_from: str | None = None,
    prefix_run: str | None = None,
    branches: tuple[str, ...] = tuple(OFFSETS),
) -> Path:
    if (resume_from or prefix_run) and len(seeds) * len(systems) != 1:
        raise ValueError("--resume-from and --prefix-run take one world and one system")
    run_dir = new_run_dir(name)
    results = []
    for k in systems:
        for s in seeds:
            results.append(
                run_world(
                    s,
                    k,
                    run_dir,
                    params,
                    measure_only,
                    measure_events,
                    phase,
                    resume_from,
                    prefix_run,
                    branches,
                )
            )
            write_jsonl(run_dir / "world_results.jsonl", results[-1:], mode="a")
    keys = ("seed", "system", "prefix_events", "t_i", "identifiable")
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "analysis" / "e5_native.json").write_text(
        json.dumps([{k: r.get(k) for k in keys} for r in results], indent=1, default=str)
    )
    models = {"native_llm": MAIN_MODEL, "reader": MAIN_MODEL, "embedding": EMBED_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E5-measure" if measure_only else "E5",
        [r["world"] for r in results],
        models,
        {
            "seeds": seeds,
            "systems": systems,
            "world_params": params or {},
            "offsets": OFFSETS,
            "graphiti_batch": GRAPHITI_BATCH,
            "native_batch": NATIVE_BATCH,
            "measure_events": measure_events,
            "on_error": ON_ERROR,
            "phase": phase,
            "resume_from": resume_from,
            "prefix_run": prefix_run,
            "branches": list(branches),
        },
    )
    return run_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--systems", default="mem0,graphiti")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--measure-events", type=int, default=0, help="cap the measured prefix")
    ap.add_argument("--name", default="e5")
    ap.add_argument("--graphiti-batch", type=int, default=1, help="episodes per bulk add")
    ap.add_argument(
        "--native-batch", type=int, default=1, help="HippoRAG / LightRAG / MemOS bulk ingestion"
    )
    ap.add_argument("--on-error", choices=("raise", "skip"), default="raise")
    ap.add_argument("--phase", choices=PHASES, default="all")
    ap.add_argument("--resume-from", default=None, help="run name of an interrupted run")
    ap.add_argument("--prefix-run", default=None, help="run name of a --phase prefix run")
    ap.add_argument("--branches", default=",".join(OFFSETS), help="with --phase branch")
    a = ap.parse_args()
    GRAPHITI_BATCH = max(1, a.graphiti_batch)
    NATIVE_BATCH = max(1, a.native_batch)
    ON_ERROR = a.on_error
    wp = world_params(a.months)
    print(
        run(
            parse_seeds(a.seeds),
            a.systems.split(","),
            a.name,
            wp,
            a.measure,
            a.measure_events or None,
            a.phase,
            a.resume_from,
            a.prefix_run,
            tuple(a.branches.split(",")),
        )
    )

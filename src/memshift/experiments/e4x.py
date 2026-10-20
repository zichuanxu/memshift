"""E4 extension: official sync semantics for all six
sources, {ideal, official L = 5 min} x B in {0, 1} (D = 1) x {HoldMem, zero system},
discriminative probes.

Per world (S_src branch; test worlds 1000-1004):
  streams   ideal (every event once, at its own time; lag recorded as 0) and official at L = 5 min:
            every source through its own official-sync profile (calsync.sim.PROFILES: 15-min
            polls, 50-item pages, per-source cursor TTL / invalidation / tombstones; Slack-style
            messages never deliver edits or deletions of messages already fetched), one server and
            adapter per source. Sync seed world.seed * 7919 + 4 as in E4, so the calendar stream
            equals E4's official stream at the same lag; other sources' seed strings name them.
  systems   per stream: HoldMem D=1 B=0, HoldMem D=1 B=1 (source-epoch key) and the zero system,
            on the delivered world with all slots; HoldMem is built at t0 = t_f - 1 min.
  events    true change events in (t_f, t_f + 8 weeks] that touch at least one slot (below),
            a seeded sample per source (seed string names world seed and source). Calendar, the
            replication of E4: up to 10 per world, stratified as in E4 (series,
            appointment and settings changes first, then meetings). Other sources (descriptive):
            up to 2 each, shift-tagged events first, then updates / deletes, then inserts.
  probes    a slot is touched by event e when e.object_id is in the dependency set of the true
            world's gold at e.ts or at e.ts - 1 s. At as_of = e.ts + {10, 60} min: every touched
            slot plus an equal number of untouched slots (seeded sample per event, the same at
            both deltas); (slot, time) pairs are deduplicated, touched pairs first. Gold from the
            true world.
  timeline  walking the probe times in order, each HoldMem maintains at a probe time whenever its
            delivered log changed since its last maintenance (as in E4); the zero system reads the
            raw delivered objects at as_of.
Tags on every answer: stream, mode, lag, delta, system, B, source (of the triggering event),
touched, synced (the triggering object is identical in the delivered and the true log at as_of).
Outputs: probes.jsonl; answers.jsonl; divergence.jsonl (per stream, source and probe time);
polls.jsonl (official polls that delivered changes or ran a full sync, per source);
world_summaries.jsonl; manifest.json; env.json.

LLM calls per world (writer and reader are both MAIN_MODEL; cache hits count as calls). With
  S   slots (all of them; 48 in the generator's worlds),
  K   streams = 2, two HoldMem cells per stream,
  P   probes,
  M_s probe times at which stream s delivered >= 1 event since the previous maintenance
      (M_s <= T, the number of distinct probe times, T <= 2 deltas x (10 + 5 x 2) events = 40),
  writer <= sum_s [S (1 + M_s) for B=0 + S (1 + M_s) for B=1] = 2 S sum_s (1 + M_s)
           (build derives every slot; a B=0 maintenance derives the slots with a non-empty
           in-scope delta, a B=1 maintenance re-derives the slots whose source-epoch key changed;
           both at most S),
  reader = 3 K P = 6 P (three systems per stream; no probe is blocked),
  total  <= 2 S sum_s (1 + M_s) + 6 P  <=  4 S (1 + T) + 6 P.
Each event contributes at most 2 x min(2 h_e, S) probes (h_e touched slots), so P <= S T. With
S = 48 and T <= 40 the bound is 7,872 + 6 P <= 7,872 + 11,520 = 19,392 per world. The first form
(with the realised M_s) is written per world as `llm_calls_expected`, next to the client's
`llm_calls`.
"""

from __future__ import annotations

import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import ZeroSystem
from memshift.calsync.sim import ALL_SOURCES, PROFILES, SyncStats, delivered_world_all, divergence
from memshift.experiments.common import (
    MAIN_MODEL,
    Timer,
    answer_probes,
    client,
    new_run_dir,
    write_jsonl,
)
from memshift.experiments.e1 import _par, parse_seeds
from memshift.experiments.e4 import _cached, run_holdmem
from memshift.gen.gold import Gold, gold_at
from memshift.gen.model import Event, Slot, World, fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, ServeResult
from memshift.llm import LLMClient
from memshift.probes import Probe, make_probe
from memshift.runinfo import world_record, write_env, write_manifest

BRANCH = "src"
MODES = ("ideal", "official")
LAG = 5
DELTAS = (10, 60)
B_VALUES = (0, 1)
MAX_EVENTS_PER_SOURCE = 2
MAX_CALENDAR_EVENTS = 10  # the calendar replication, sampled as in E4
WINDOW_WEEKS = 8
_WRITE_LOCK = threading.Lock()


def _stratum(e: Event) -> int:
    if e.source == "calendar":
        return 0 if e.payload.get("kind") != "meeting" else 1
    if e.tag.startswith("shift"):
        return 0
    return 1 if e.op in ("update", "delete") else 2


def touched_slots(world: World, log: SourceLog, ev: Event) -> tuple[list[Slot], dict[str, Gold]]:
    """Slots whose true gold depends on ev's object just before or at ev.ts; gold before ev."""
    before = {s.slot_id: gold_at(world, log, s, ev.ts - timedelta(seconds=1)) for s in world.slots}
    hit = [
        s
        for s in world.slots
        if _depends(before[s.slot_id], ev.object_id)
        or _depends(gold_at(world, log, s, ev.ts), ev.object_id)
    ]
    return hit, before


def sample_events(world: World, log: SourceLog) -> list[Event]:
    """Per source, a seeded sample of true changes after the fork that touch at least one slot;
    within a source, strata are exhausted in order (`_stratum`), each stratum in seeded order."""
    window = log.between(world.fork, world.fork + timedelta(weeks=WINDOW_WEEKS))
    chosen: list[Event] = []
    for src in ALL_SOURCES:
        cap = MAX_CALENDAR_EVENTS if src == "calendar" else MAX_EVENTS_PER_SOURCE
        rng = random.Random(f"e4x|{world.seed}|{src}")
        picked: list[Event] = []
        for k in range(3):
            stratum = [e for e in window if e.source == src and _stratum(e) == k]
            rng.shuffle(stratum)
            for e in stratum:
                if len(picked) == cap:
                    break
                if touched_slots(world, log, e)[0]:
                    picked.append(e)
        chosen += picked
    return sorted(chosen, key=lambda e: (e.ts, e.event_id))


def _depends(g: Gold, object_id: str) -> bool:
    return any(oid == object_id for oid, _ in g.deps)


def probe_plan(world: World, log: SourceLog) -> tuple[list[Probe], list[Event]]:
    """Discriminative probes after the sampled true events; returns (probes, sampled events)."""
    events = sample_events(world, log)
    plan: list[tuple[Event, list[Slot], list[Slot], dict[str, Gold]]] = []
    for ev in events:
        hit, before = touched_slots(world, log, ev)
        rest = [s for s in world.slots if s not in hit]
        rng = random.Random(f"e4x-untouched|{world.seed}|{ev.event_id}")
        plan.append((ev, hit, rng.sample(rest, min(len(hit), len(rest))), before))
    probes: list[Probe] = []
    seen: set[tuple[str, datetime]] = set()
    for touched in (True, False):
        for ev, hit, control, before in plan:
            for d in DELTAS:
                t = ev.ts + timedelta(minutes=d)
                for slot in hit if touched else control:
                    if (slot.slot_id, t) in seen:
                        continue
                    seen.add((slot.slot_id, t))
                    p = make_probe(world, log, slot, t, tags={"delta": d})
                    g0 = before[slot.slot_id]
                    p.tags |= {
                        "source": ev.source,
                        "touched": touched,
                        "event_id": ev.event_id,
                        "event_op": ev.op,
                        "event_tag": ev.tag,
                        "object_id": ev.object_id,
                        "gold_changed": (p.gold_value, p.gold_status) != (g0.value, g0.status),
                    }
                    probes.append(p)
    probes.sort(key=lambda p: (p.as_of, p.probe_id))
    return probes, events


def _synced(p: Probe, true_log: SourceLog, dlog: SourceLog) -> bool:
    oid = p.tags["object_id"]
    a, b = true_log.snapshot(p.as_of).get(oid), dlog.snapshot(p.as_of).get(oid)
    return (a.payload if a else None) == (b.payload if b else None)


def run_world(
    seed: int,
    run_dir: Path,
    llm: LLMClient,
    branch: str = BRANCH,
    world_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timer = Timer()
    calls0 = llm.usage["calls"]
    world = generate_branch(seed, branch, world_params)
    true_log = SourceLog(world.events)
    probes, events = probe_plan(world, true_log)
    t0 = world.fork - timedelta(minutes=1)
    until = max((p.as_of for p in probes), default=t0)
    sync_seed = world.seed * 7919 + 4

    streams: list[tuple[str, int, World, dict[str, SyncStats]]] = []
    for mode in MODES:
        lag = 0 if mode == "ideal" else LAG
        dw, stats = delivered_world_all(world, mode, lag, sync_seed, until=until)
        streams.append((mode, lag, dw, stats))

    jobs = [(i, b) for i in range(len(streams)) for b in B_VALUES]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {
            (i, b): ex.submit(run_holdmem, streams[i][2], b, llm, t0, probes) for i, b in jobs
        }
        cells = {k: f.result() for k, f in futures.items()}

    answers, div_rows, poll_rows = [], [], []
    stream_summaries: dict[str, Any] = {}
    expected_writer = 0
    times = sorted({p.as_of for p in probes})
    for i, (mode, lag, sw, stats) in enumerate(streams):
        key = f"{mode}_L{lag:g}"
        dlog = SourceLog(sw.events)
        tagged = [replace(p, tags={**p.tags, "synced": _synced(p, true_log, dlog)}) for p in probes]
        systems: dict[str, tuple[int | None, dict[str, ServeResult]]] = {
            CellConfig(1, b, "source").name: (b, cells[(i, b)][0]) for b in B_VALUES
        }
        zero = ZeroSystem(sw, dlog)
        systems["zero"] = (None, {p.probe_id: zero.serve(p) for p in tagged})
        for name, (b, served) in systems.items():
            ans, _ = answer_probes(
                name,
                _cached(served),
                tagged,
                llm,
                tags={"stream": key, "mode": mode, "lag": lag, "system": name, "B": b},
            )
            answers += ans
        for t in times:
            snap = true_log.snapshot(t)
            div_rows += [
                {
                    "world_id": world.world_id,
                    "stream": key,
                    "mode": mode,
                    "lag": lag,
                    "source": src,
                    "as_of": fmt_ts(t),
                    "divergence": divergence(true_log, dlog, t, src),
                    "n_true_live": len(snap.by_source(src)),
                }
                for src in ALL_SOURCES
            ]
        poll_rows += [
            {"world_id": world.world_id, "stream": key, "lag": lag, "source": src, **r.to_json()}
            for src, st in stats.items()
            for r in st.polls
            if r.n_changes or r.full
        ]
        maintained = {f"B{b}": cells[(i, b)][1] for b in B_VALUES}
        expected_writer += sum(len(sw.slots) * (1 + m) for m in maintained.values())
        stream_summaries[key] = {
            "sources": {src: st.summary() for src, st in stats.items()},
            "maintained": maintained,
        }

    with _WRITE_LOCK:
        write_jsonl(run_dir / "probes.jsonl", (p.to_json() for p in probes), mode="a")
        write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
        write_jsonl(run_dir / "divergence.jsonl", div_rows, mode="a")
        write_jsonl(run_dir / "polls.jsonl", poll_rows, mode="a")
    return {
        "seed": seed,
        "world": world_record(world),
        "n_slots": len(world.slots),
        "events": [
            {
                "event_id": e.event_id,
                "source": e.source,
                "op": e.op,
                "tag": e.tag,
                "ts": fmt_ts(e.ts),
            }
            for e in events
        ],
        "n_probes": len(probes),
        "n_probes_touched": sum(bool(p.tags["touched"]) for p in probes),
        "n_probes_by_source": {
            src: sum(p.tags["source"] == src for p in probes) for src in ALL_SOURCES
        },
        "n_probe_times": len(times),
        "n_gold_changed": sum(bool(p.tags["gold_changed"]) for p in probes),
        "streams": stream_summaries,
        "llm_calls_expected": expected_writer + 3 * len(streams) * len(probes),
        "llm_calls": llm.usage["calls"] - calls0,
        "seconds": round(timer.lap(), 1),
    }


def run(
    seeds: list[int],
    name: str = "e4x",
    out: Path | None = None,
    llm: LLMClient | None = None,
    branch: str = BRANCH,
    parallel_worlds: int = 3,
    world_params: dict[str, Any] | None = None,
) -> Path:
    """Worlds run `parallel_worlds` at a time, each with its own client so that its call count is
    exact; a client passed in (tests) is shared, so its worlds run one at a time."""
    run_dir = out or new_run_dir(name)

    def one(seed: int) -> dict[str, Any]:
        s = run_world(seed, run_dir, llm or client(MAIN_MODEL, run_dir), branch, world_params)
        with _WRITE_LOCK:
            write_jsonl(run_dir / "world_summaries.jsonl", [s], mode="a")
        return s

    summaries = _par(one, seeds, parallel_worlds if llm is None else 1)
    models = {"writer": MAIN_MODEL, "reader": MAIN_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E4X",
        [s["world"] for s in summaries],
        models,
        {
            "seeds": seeds,
            "branch": branch,
            "world_params": world_params or {},
            "sources": list(ALL_SOURCES),
            "modes": list(MODES),
            "lag_minutes": LAG,
            "profiles": {src: PROFILES[src].to_json() for src in ALL_SOURCES},
            "sync_seed": "world.seed * 7919 + 4; calendar seeded as in E4, other sources' "
            "phase and invalidation seed strings include the source name",
            "cells": [CellConfig(1, b, "source").name for b in B_VALUES],
            "systems": [*(CellConfig(1, b, "source").name for b in B_VALUES), "zero"],
            "max_events_per_source": MAX_EVENTS_PER_SOURCE,
            "event_strata": ["shift-tagged", "update/delete", "insert"],
            "window_weeks": WINDOW_WEEKS,
            "probe_deltas_minutes": list(DELTAS),
            "touched": "event.object_id in true gold deps at event.ts or event.ts - 1 s",
            "probes": "every touched slot + equal-size seeded sample of untouched slots per "
            "event (same at both deltas); (slot, time) deduplicated, touched first",
            "t0": "fork - 1 minute",
            "maintain": "at probe times where the delivered log changed since last maintenance",
            "reader_budget_tokens": 4096,
        },
        {"llm_calls_expected": sum(s["llm_calls_expected"] for s in summaries)},
    )
    return run_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--name", default="e4x")
    ap.add_argument("--parallel-worlds", type=int, default=3)
    ap.add_argument("--no-analyse", action="store_true")
    a = ap.parse_args()
    run_dir = run(parse_seeds(a.seeds), a.name, parallel_worlds=a.parallel_worlds)
    if not a.no_analyse:
        from memshift.analysis.e4x import analyse

        analyse(run_dir)
    print(run_dir)

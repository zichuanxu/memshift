"""E4 (plan §6 E4, §2.5 H_S): calendar sync semantics, {ideal, official} x {B} x {HoldMem, zero}.

Per world (S_src branch, calendar source only):
  streams   ideal (every calendar event once, at its own time; lag recorded as 0) and official at
            L in {1, 5, 30} min (calsync.sim: 15-min polls, sync-token paging, 410 -> full resync,
            token TTL 7 days, per-poll 410 probability 0.001). All lags of a world share the poll
            phase and the 410 draws.
  systems   per stream: HoldMem D=1 B=0, HoldMem D=1 B=1 (source-epoch key) and the zero system,
            each built on a world whose events are that stream's calendar events and whose slots
            are the calendar slots (sources only calendar, series, appointments, u.timezone,
            kappa important_contact / working_hours).
  timeline  HoldMem is built at t0 = t_f - 1 min from the delivered log. Probe times: for up to
            10 true calendar events in (t_f, t_f + 8 weeks] (seeded sample stratified: series,
            appointment and settings changes first, then meetings), event.ts + {2, 10,
            60} min, probing the calendar slots the event touches; (slot, time) pairs are
            deduplicated. Walking the probe times in order, HoldMem maintains at a probe time
            whenever its delivered log changed since its last maintenance; otherwise maintenance
            would be a no-op for B=1 and an empty-delta writer pass for B=0, and is skipped.
  gold      from the true world and its full log. Probes whose gold depends on a non-calendar
            object are dropped (working_hours under v1 is read from a note).
Outputs: probes.jsonl; answers.jsonl (tags mode, lag, delta, system, B, synced = the triggering
object is identical in the delivered and true logs at as_of); divergence.jsonl (per stream and
probe time); polls.jsonl (official polls that delivered changes or ran a full sync);
world_summaries.jsonl; manifest.json; env.json.

LLM calls per world (writer and reader are both MAIN_MODEL; cache hits count as calls). With
  S   calendar slots (9 in the generator's worlds),
  K   streams = 1 + len(LAGS) = 4,
  P   probes kept,
  M_s probe times at which stream s delivered >= 1 calendar event since the previous maintenance
      (M_s <= T, the number of distinct probe times, T <= 3 * 10 = 30),
  writer = sum_s [S (1 + M_s) for B=0 + S (1 + M_s) for B=1] = 2 S sum_s (1 + M_s)
           (build derives every slot; a B=0 maintenance derives every slot's delta; a B=1
           maintenance re-derives every slot, since each calendar slot's source-epoch key holds
           the calendar epoch),
  reader = 3 K P (three systems per stream; no probe is blocked),
  total  = 2 S sum_s (1 + M_s) + 12 P  <=  8 S (1 + T) + 12 P.
With S = 9, T <= 30, P <= 60 the bound is 2,232 + 720 = 2,952 per world. The formula's
upper bound is written per world as `llm_calls_expected` (B=0 skips empty deltas), next to
the client's `llm_calls`.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import ZeroSystem
from memshift.calsync.sim import (
    GONE_PROB,
    PAGE_SIZE,
    POLL_MINUTES,
    SOURCE,
    TOKEN_TTL_DAYS,
    SyncStats,
    delivered_world,
    divergence,
)
from memshift.experiments.common import (
    MAIN_MODEL,
    Timer,
    answer_probes,
    client,
    new_run_dir,
    write_jsonl,
)
from memshift.gen.gold import gold_at
from memshift.gen.model import Event, Slot, World, fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem, ServeResult
from memshift.llm import LLMClient
from memshift.probes import Probe, make_probe
from memshift.runinfo import world_record, write_env, write_manifest

BRANCH = "src"
MODES = ("ideal", "official")
LAGS = (1, 5, 30)
DELTAS = (2, 10, 60)
B_VALUES = (0, 1)
MAX_EVENTS = 10
WINDOW_WEEKS = 8
CAL_KAPPAS = ("important_contact", "working_hours")


def calendar_slots(world: World) -> list[Slot]:
    return [
        s
        for s in world.slots
        if s.sources == (SOURCE,)
        or s.kind in ("series", "appointment")
        or s.slot_id == "u.timezone"
        or s.kappa_id in CAL_KAPPAS
    ]


def touched(event: Event, slots: list[Slot]) -> list[Slot]:
    """Calendar slots whose gold scope contains the event's object."""
    kind = event.payload.get("kind")
    return [
        s
        for s in slots
        if (kind == "settings" and s.slot_id == "u.timezone")
        or (kind == "series" and s.kind == "series" and s.subject == event.object_id)
        or (
            kind == "appointment"
            and s.kind == "appointment"
            and s.subject == event.payload.get("appt_type")
        )
        or (kind == "meeting" and s.kappa_id in CAL_KAPPAS)
    ]


def sample_events(world: World, log: SourceLog) -> list[Event]:
    """Seeded sample of true calendar changes, stratified so that series, appointment and
    settings changes come first (one-off meetings dominate the stream and only touch the
    important-contact definition); meetings fill the remaining places."""
    t_end = world.fork + timedelta(weeks=WINDOW_WEEKS)
    cal = [e for e in log.between(world.fork, t_end) if e.source == SOURCE]
    rng = random.Random(f"e4|{world.seed}")
    other = [e for e in cal if e.payload.get("kind") != "meeting"]
    meetings = [e for e in cal if e.payload.get("kind") == "meeting"]
    chosen = rng.sample(other, min(MAX_EVENTS, len(other)))
    chosen += rng.sample(meetings, min(MAX_EVENTS - len(chosen), len(meetings)))
    return sorted(chosen, key=lambda e: (e.ts, e.event_id))


def _grounded(log: SourceLog, p: Probe) -> bool:
    snap = log.snapshot(p.as_of)
    return all((o := snap.get(oid)) is not None and o.source == SOURCE for oid, _ in p.deps)


def probe_plan(world: World, log: SourceLog, slots: list[Slot]) -> tuple[list[Probe], int]:
    """Probes after sampled true calendar events; returns (probes, dropped as non-calendar)."""
    probes: list[Probe] = []
    seen: set[tuple[str, datetime]] = set()
    dropped = 0
    for ev in sample_events(world, log):
        before = ev.ts - timedelta(minutes=1)
        for slot in touched(ev, slots):
            g0 = gold_at(world, log, slot, before)
            for d in DELTAS:
                t = ev.ts + timedelta(minutes=d)
                if (slot.slot_id, t) in seen:
                    continue
                seen.add((slot.slot_id, t))
                p = make_probe(world, log, slot, t, tags={"delta": d})
                if not _grounded(log, p):
                    dropped += 1
                    continue
                p.tags |= {
                    "event_id": ev.event_id,
                    "event_kind": ev.payload.get("kind"),
                    "event_op": ev.op,
                    "object_id": ev.object_id,
                    "gold_changed": (p.gold_value, p.gold_status) != (g0.value, g0.status),
                }
                probes.append(p)
    probes.sort(key=lambda p: (p.as_of, p.probe_id))
    return probes, dropped


def calendar_world(world: World, slots: list[Slot]) -> World:
    return replace(world, events=[e for e in world.events if e.source == SOURCE], slots=slots)


def by_time(probes: list[Probe]) -> list[tuple[datetime, list[Probe]]]:
    out: dict[datetime, list[Probe]] = {}
    for p in probes:
        out.setdefault(p.as_of, []).append(p)
    return sorted(out.items())


def run_holdmem(
    sw: World, b: int, writer: LLMClient, t0: datetime, probes: list[Probe]
) -> tuple[dict[str, ServeResult], int]:
    """Build at t0, then maintain (when the delivered log changed) and serve at each probe time."""
    log = SourceLog(sw.events)
    hm = HoldMem(sw, log, CellConfig(1, b, "source"), writer, MAIN_MODEL)
    hm.workers = max(1, len(sw.slots))
    hm.build(t0)
    served: dict[str, ServeResult] = {}
    maintained = 0
    for t, group in by_time(probes):
        assert hm.last_t is not None
        if log.between(hm.last_t, t):
            hm.maintain(t)
            maintained += 1
        for p in group:
            served[p.probe_id] = hm.serve(p)
    return served, maintained


def _cached(served: dict[str, ServeResult]) -> Callable[[Probe], ServeResult]:
    return lambda p: served[p.probe_id]


def _stream_key(mode: str, lag: float) -> str:
    return f"{mode}_L{lag:g}"


def run_world(seed: int, run_dir: Path, llm: LLMClient, branch: str = BRANCH) -> dict[str, Any]:
    timer = Timer()
    calls0 = llm.usage["calls"]
    world = generate_branch(seed, branch)
    true_log = SourceLog(world.events)
    slots = calendar_slots(world)
    probes, dropped = probe_plan(world, true_log, slots)
    t0 = world.fork - timedelta(minutes=1)
    until = max((p.as_of for p in probes), default=t0)
    sync_seed = world.seed * 7919 + 4

    streams: list[tuple[str, int, World, SyncStats]] = []
    for mode in MODES:
        for lag in (0,) if mode == "ideal" else LAGS:
            dw, stats = delivered_world(
                world,
                mode,
                lag,
                POLL_MINUTES,
                sync_seed,
                token_ttl_days=TOKEN_TTL_DAYS,
                gone_prob=GONE_PROB,
                page_size=PAGE_SIZE,
                until=until,
            )
            streams.append((mode, lag, calendar_world(dw, slots), stats))

    jobs = [(i, b) for i in range(len(streams)) for b in B_VALUES]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        futures = {
            (i, b): ex.submit(run_holdmem, streams[i][2], b, llm, t0, probes) for i, b in jobs
        }
        cells = {k: f.result() for k, f in futures.items()}

    answers, div_rows, poll_rows = [], [], []
    stream_summaries: dict[str, Any] = {}
    expected_writer = 0
    for i, (mode, lag, sw, stats) in enumerate(streams):
        key = _stream_key(mode, lag)
        dlog = SourceLog(sw.events)
        synced_probes = []
        for p in probes:
            oid = p.tags["object_id"]
            a, b_ = true_log.snapshot(p.as_of).get(oid), dlog.snapshot(p.as_of).get(oid)
            same = (a.payload if a else None) == (b_.payload if b_ else None)
            synced_probes.append(replace(p, tags={**p.tags, "synced": same}))
        systems: dict[str, tuple[int | None, dict[str, ServeResult]]] = {
            CellConfig(1, b, "source").name: (b, cells[(i, b)][0]) for b in B_VALUES
        }
        zero = ZeroSystem(sw, dlog)
        systems["zero"] = (None, {p.probe_id: zero.serve(p) for p in synced_probes})
        for name, (b, served) in systems.items():
            ans, _ = answer_probes(
                name,
                _cached(served),
                synced_probes,
                llm,
                tags={"mode": mode, "lag": lag, "system": name, "B": b},
            )
            answers += ans
        for t, delta in sorted({(p.as_of, p.tags["delta"]) for p in probes}):
            div_rows.append(
                {
                    "world_id": world.world_id,
                    "mode": mode,
                    "lag": lag,
                    "as_of": fmt_ts(t),
                    "delta": delta,
                    "divergence": divergence(true_log, dlog, t),
                    "n_true_live": len(true_log.snapshot(t).by_source(SOURCE)),
                }
            )
        poll_rows += [
            {"world_id": world.world_id, "lag": lag, **r.to_json()}
            for r in stats.polls
            if r.n_changes or r.full
        ]
        maintained = {f"B{b}": cells[(i, b)][1] for b in B_VALUES}
        expected_writer += sum(len(slots) * (1 + m) for m in maintained.values())
        stream_summaries[key] = {**stats.summary(), "maintained": maintained}

    write_jsonl(run_dir / "probes.jsonl", (p.to_json() for p in probes), mode="a")
    write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
    write_jsonl(run_dir / "divergence.jsonl", div_rows, mode="a")
    write_jsonl(run_dir / "polls.jsonl", poll_rows, mode="a")
    return {
        "seed": seed,
        "world": world_record(world),
        "n_calendar_slots": len(slots),
        "n_probes": len(probes),
        "n_probe_times": len(by_time(probes)),
        "n_dropped_non_calendar": dropped,
        "n_gold_changed": sum(bool(p.tags["gold_changed"]) for p in probes),
        "streams": stream_summaries,
        "llm_calls_expected": expected_writer + 3 * len(streams) * len(probes),
        "llm_calls": llm.usage["calls"] - calls0,
        "seconds": round(timer.lap(), 1),
    }


def run(
    seeds: list[int],
    name: str = "e4",
    out: Path | None = None,
    llm: LLMClient | None = None,
    branch: str = BRANCH,
) -> Path:
    run_dir = out or new_run_dir(name)
    llm = llm or client(MAIN_MODEL, run_dir)
    summaries = []
    for seed in seeds:
        s = run_world(seed, run_dir, llm, branch)
        summaries.append(s)
        write_jsonl(run_dir / "world_summaries.jsonl", [s], mode="a")
    models = {"writer": MAIN_MODEL, "reader": MAIN_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E4",
        [s["world"] for s in summaries],
        models,
        {
            "seeds": seeds,
            "branch": branch,
            "source": SOURCE,
            "modes": list(MODES),
            "lags_minutes": list(LAGS),
            "poll_minutes": POLL_MINUTES,
            "token_ttl_days": TOKEN_TTL_DAYS,
            "gone_prob_per_poll": GONE_PROB,
            "page_size": PAGE_SIZE,
            "sync_seed": "world.seed * 7919 + 4 (shared across lags); poll phase from it",
            "cells": [CellConfig(1, b, "source").name for b in B_VALUES],
            "systems": [*(CellConfig(1, b, "source").name for b in B_VALUES), "zero"],
            "calendar_slots": "sources == (calendar,) | series | appointment | u.timezone | "
            f"kappa in {list(CAL_KAPPAS)}",
            "probe_deltas_minutes": list(DELTAS),
            "max_events_per_world": MAX_EVENTS,
            "window_weeks": WINDOW_WEEKS,
            "t0": "fork - 1 minute",
            "maintain": "at probe times where the delivered log changed since last maintenance",
            "probe_filter": "gold deps all calendar objects",
            "reader_budget_tokens": 4096,
        },
        {"llm_calls_expected": sum(s["llm_calls_expected"] for s in summaries)},
    )
    return run_dir

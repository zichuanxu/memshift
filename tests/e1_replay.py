"""Offline replay of E1 cells on one small world, for the cell tests: one recording fake writer per
cell, a build at t0 and maintenance at the given weeks after the fork."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from memshift.experiments.common import world_params
from memshift.gen.model import World
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.probes import Probe, make_probes
from memshift.reader import context_hash
from tests.fakes import RecordingLLM

THINK_BODY = {"chat_template_kwargs": {"enable_thinking": True}}


@dataclass
class Replay:
    world: World
    log: SourceLog
    cells: dict[str, HoldMem]
    llms: dict[str, Any]  # cell -> its recording writer
    think_llms: dict[str, Any]  # cell -> its recording thinking writer (think cells only)
    t0: datetime
    stats: dict[str, list[dict[str, int]]]  # cell -> maintenance stats per checkpoint
    contexts_t0: dict[str, dict[str, str]]  # cell -> probe id -> context hash, before any shift

    def at(self, weeks: int) -> datetime:
        return self.world.fork + timedelta(weeks=weeks)

    def probes(self, t: datetime) -> list[Probe]:
        return make_probes(self.world, self.log, [t])

    def contexts(self, name: str, t: datetime) -> dict[str, str]:
        return {p.probe_id: context_hash(self.cells[name].serve(p).blocks) for p in self.probes(t)}

    def keys(self, name: str, purpose: str | None = None) -> set[str]:
        keys: set[str] = self.llms[name].keys(purpose)
        return keys


def replay(
    cfgs: list[CellConfig],
    branch: str = "def",
    seed: int = 5,
    weeks: tuple[int, ...] = (1, 2),
    months: int = 3,
    make_llm: Callable[[], Any] = RecordingLLM,
) -> Replay:
    world = generate_branch(seed, branch, world_params(months))
    log = SourceLog(world.events)
    t0 = world.fork - timedelta(minutes=1)
    llms: dict[str, Any] = {c.name: make_llm() for c in cfgs}
    think_llms: dict[str, Any] = {
        c.name: RecordingLLM(extra_body={**RecordingLLM().extra_body, **THINK_BODY})
        for c in cfgs
        if c.think
    }
    cells: dict[str, HoldMem] = {}
    for c in cfgs:
        cells[c.name] = HoldMem(
            world, log, c, llms[c.name], "fake/writer-a", think_writer=think_llms.get(c.name)
        )
        cells[c.name].workers = 4
        cells[c.name].build(t0)
    pre = make_probes(world, log, [t0])
    contexts_t0 = {
        name: {p.probe_id: context_hash(hm.serve(p).blocks) for p in pre}
        for name, hm in cells.items()
    }
    stats: dict[str, list[dict[str, int]]] = {c.name: [] for c in cfgs}
    for k in weeks:
        for name, hm in cells.items():
            stats[name].append(hm.maintain(world.fork + timedelta(weeks=k)))
    return Replay(world, log, cells, llms, think_llms, t0, stats, contexts_t0)


def digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()

"""
探针生成：点查询，带 as-of 时间与 gold；分叉后 k = 1, 2, 4, 8。负责：研究员 1。

Probe schema (plan §4.3): `{probe_id, fact_id, as_of, question_text, gold_value, gold_status,
deps, kappa_version, agent_id}`. Gold status is `abstain` when the fact has no current value and
`blocked` when the asking agent lacks consent for a source the fact depends on.

A definition item's question ends with "Definition in effect: <statement>" of the version in effect
at the probe time. `unnamed_defs=True` (prereg v4 block Q) leaves that sentence out; the question
keeps "under the user's current definition", and the probe ids, gold and dependencies are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memshift.gen.gold import gold_at
from memshift.gen.model import Slot, World, fmt_ts
from memshift.gen.state import SourceLog


@dataclass
class Probe:
    probe_id: str
    world_id: str
    branch: str
    fact_id: str
    vtype: str
    as_of: datetime
    question_text: str
    gold_value: Any
    gold_status: str
    deps: list[tuple[str, int]]
    kappa_version: str | None
    agent_id: str = "assistant"
    tags: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "probe_id": self.probe_id,
            "world_id": self.world_id,
            "branch": self.branch,
            "fact_id": self.fact_id,
            "vtype": self.vtype,
            "as_of": fmt_ts(self.as_of),
            "question_text": self.question_text,
            "gold_value": self.gold_value,
            "gold_status": self.gold_status,
            "deps": [list(d) for d in self.deps],
            "kappa_version": self.kappa_version,
            "agent_id": self.agent_id,
            "tags": self.tags,
        }


def question_text(world: World, slot: Slot, t: datetime, unnamed_defs: bool = False) -> str:
    q = f"As of {t:%Y-%m-%d %H:%M}: {slot.question}"
    if slot.kappa_id and not unnamed_defs:
        ver = world.kappa_version(slot.kappa_id, t)
        q += f" Definition in effect: {world.kappa_defs[slot.kappa_id][ver].statement}"
    return q


def make_probe(
    world: World,
    log: SourceLog,
    slot: Slot,
    t: datetime,
    agent_id: str = "assistant",
    tags: dict[str, Any] | None = None,
    unnamed_defs: bool = False,
) -> Probe:
    g = gold_at(world, log, slot, t)
    status = g.status
    snap = log.snapshot(t)
    dep_sources = {o.source for oid, _ in g.deps if (o := snap.get(oid)) is not None}
    if not dep_sources:
        dep_sources = set(slot.sources[:1])
    if not dep_sources <= world.granted(agent_id, t):
        status = "blocked"
    suffix = "".join(f".{v}" for v in (tags or {}).values())
    return Probe(
        probe_id=f"{world.world_id}.{world.branch}.{slot.slot_id}.{t:%Y%m%d%H%M}.{agent_id}{suffix}",
        world_id=world.world_id,
        branch=world.branch,
        fact_id=slot.slot_id,
        vtype=slot.vtype,
        as_of=t,
        question_text=question_text(world, slot, t, unnamed_defs),
        gold_value=g.value,
        gold_status=status,
        deps=g.deps,
        kappa_version=g.kappa_version,
        agent_id=agent_id,
        tags=dict(tags or {}),
    )


def make_probes(
    world: World,
    log: SourceLog,
    times: list[datetime],
    slots: list[Slot] | None = None,
    agent_id: str = "assistant",
    tags: dict[str, Any] | None = None,
    unnamed_defs: bool = False,
) -> list[Probe]:
    return [
        make_probe(world, log, s, t, agent_id, tags, unnamed_defs)
        for t in times
        for s in (slots or world.slots)
    ]

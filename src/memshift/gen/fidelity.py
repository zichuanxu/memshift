"""Fidelity checker: every gold fact must be locatable in the text of its dependency objects.

Follows the "Ground Truth First" practice (2607.21962) of verifying planted facts after rendering.
Sampled weekly plus at the fork; a slot-time passes when the gold value's surface form appears in
the text of at least one positive dependency (every member, for set values). Definition-derived
values without a textual surface form (kappa working hours v2) pass when their dependency set is
non-empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from memshift.gen import pools
from memshift.gen.gold import gold_at
from memshift.gen.model import Slot, World
from memshift.gen.state import SourceLog


def surface_forms(slot: Slot, value: Any) -> list[list[str]]:
    """Alternatives (outer) of conjunctions (inner) that must appear in dependency text."""
    v = str(value)
    if slot.vtype == "diet":
        return [[pools.DIET_PHRASES[v]], [f"Current diet: {v}"]]
    if slot.vtype == "commute":
        return [[pools.COMMUTE_PHRASES[v]]]
    if slot.vtype == "weekly":
        day, time = v.split()
        return [[day, time]]
    return [[v]]


@dataclass
class FidelityReport:
    checked: int = 0
    located: int = 0
    failures: list[dict[str, Any]] = field(default_factory=list)

    @property
    def rate(self) -> float:
        return self.located / self.checked if self.checked else 1.0


def check_world(world: World, log: SourceLog | None = None) -> FidelityReport:
    log = log or SourceLog(world.events)
    times: list[datetime] = []
    t = world.start + timedelta(days=7)
    while t <= world.end:
        times.append(t)
        t += timedelta(days=7)
    times.append(world.fork + timedelta(hours=6))
    rep = FidelityReport()
    for slot in world.slots:
        for t in times:
            g = gold_at(world, log, slot, t)
            if g.status != "answer":
                continue
            rep.checked += 1
            snap = log.snapshot(t)
            texts = [o.text for oid, _ in g.deps if (o := snap.get(oid)) is not None]
            if isinstance(g.value, list):
                blob = "\n".join(texts)
                ok = all(str(m) in blob for m in g.value)
            elif slot.kind == "kappa" and slot.vtype == "timerange":
                ok = bool(texts)
            else:
                ok = any(
                    all(part in text for part in conj)
                    for conj in surface_forms(slot, g.value)
                    for text in texts
                )
            if ok:
                rep.located += 1
            elif len(rep.failures) < 50:
                rep.failures.append({"slot": slot.slot_id, "t": str(t), "value": g.value})
    return rep

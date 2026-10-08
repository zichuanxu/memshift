"""Gold value, status and dependency set of every slot at any time t (plan §4.2 "gold 事实").

`deps = (positive object ids with revisions, scope predicate, negative deps)`. Script-level user
attributes come from timelines; everything structured is recomputed from source state, so gold
and sources cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memshift.gen.kappa import SCOPE_PREDICATES, evaluate
from memshift.gen.model import Slot, World
from memshift.gen.state import SourceLog


@dataclass
class Gold:
    value: Any  # None means "no current value" -> gold status abstain
    status: str  # answer | abstain
    deps: list[tuple[str, int]] = field(default_factory=list)
    scope: str = ""
    negative: list[str] = field(default_factory=list)
    kappa_version: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "status": self.status,
            "deps": [list(d) for d in self.deps],
            "scope": self.scope,
            "negative": self.negative,
            "kappa_version": self.kappa_version,
        }


def _mk(value: Any, deps: list[tuple[str, int]], scope: str, neg: list[str] | None = None) -> Gold:
    return Gold(value, "abstain" if value is None else "answer", deps, scope, neg or [])


def _with_revs(log: SourceLog, t: datetime, ids: list[str]) -> list[tuple[str, int]]:
    snap = log.snapshot(t)
    return [(oid, o.revision) for oid in ids if (o := snap.get(oid)) is not None]


def gold_at(world: World, log: SourceLog, slot: Slot, t: datetime) -> Gold:
    snap = log.snapshot(t)
    conflict = world.params.get("conflicts", {}).get(slot.slot_id)
    if conflict is not None and t >= datetime.fromisoformat(conflict["ts"]):
        # unresolvable conflict between sources (E7): the only correct status is abstain
        return Gold(
            None, "abstain", _with_revs(log, t, conflict["objects"]), "unresolvable conflict", []
        )
    if slot.kind == "user_attr":
        value, deps = None, []
        for pt in world.timelines.get(slot.slot_id, []):
            if pt.ts <= t:
                value, deps = pt.value, pt.deps
        return _mk(value, _with_revs(log, t, deps), f"script timeline {slot.slot_id}")
    if slot.kind == "contact_attr":
        card = snap.get(f"contacts:{slot.subject}")
        if card is None:
            return _mk(None, [], "contact card")
        field_ = slot.slot_id.split(".")[-1]
        return _mk(card.payload[field_], [(card.object_id, card.revision)], "contact card")
    if slot.kind in ("series", "task_due"):
        assert slot.subject is not None
        obj = snap.get(slot.subject)
        scope = "recurring series object" if slot.kind == "series" else "task object"
        if obj is None:
            return _mk(None, [], scope, [slot.subject] if slot.subject in snap.tombstones else [])
        value = (
            f"{obj.payload['weekday']} {obj.payload['time']}"
            if slot.kind == "series"
            else obj.payload["due"]
        )
        return _mk(value, [(obj.object_id, obj.revision)], scope)
    if slot.kind == "appointment":
        now = t.strftime("%Y-%m-%d %H:%M")
        upcoming = [
            o
            for o in snap.by_source("calendar")
            if o.payload.get("kind") == "appointment"
            and o.payload["appt_type"] == slot.subject
            and o.payload["start"] > now
        ]
        scope = f"calendar appointments of type {slot.subject} with start > as_of"
        if not upcoming:
            return _mk(None, [], scope)
        nxt = min(upcoming, key=lambda o: (o.payload["start"], o.object_id))
        return _mk(nxt.payload["start"], [(nxt.object_id, nxt.revision)], scope)
    if slot.kind == "kappa":
        assert slot.kappa_id is not None
        version = world.kappa_version(slot.kappa_id, t)
        value, ids = evaluate(slot.kappa_id, snap, t, version)
        g = _mk(value, _with_revs(log, t, ids), SCOPE_PREDICATES[slot.kappa_id])
        g.kappa_version = version
        return g
    raise ValueError(slot.kind)

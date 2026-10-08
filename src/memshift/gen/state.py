"""Replay a world's event stream into source state as of any time t.

Used by gold computation, by every system under test (they read sources as of t), and by
HoldMem's binding keys (source epochs and object-cluster epochs).
"""

from __future__ import annotations

import bisect
import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memshift.gen.model import Event


@dataclass(frozen=True)
class ObjState:
    object_id: str
    source: str
    revision: int
    payload: dict[str, Any]
    created: datetime
    updated: datetime

    @property
    def text(self) -> str:
        return str(self.payload.get("text", ""))

    @property
    def cluster(self) -> str:
        return str(self.payload.get("cluster", self.object_id))


class Snapshot:
    """Live (non-deleted) objects at a point in time, plus tombstones."""

    def __init__(self, t: datetime, objects: dict[str, ObjState], tombstones: dict[str, int]):
        self.t = t
        self.objects = objects
        self.tombstones = tombstones

    def get(self, object_id: str) -> ObjState | None:
        return self.objects.get(object_id)

    def by_source(self, *sources: str) -> list[ObjState]:
        return [o for o in self.objects.values() if o.source in sources]

    def all(self) -> list[ObjState]:
        return list(self.objects.values())


_LOG_IDS = itertools.count()


class SourceLog:
    """Ordered event log with cached snapshots and epoch counters. `uid` is unique per instance
    for the lifetime of the process (safe as a cache key, unlike id())."""

    def __init__(self, events: list[Event]):
        self.uid = next(_LOG_IDS)
        self.events = sorted(events, key=lambda e: (e.ts, e.event_id))
        self._ts = [e.ts for e in self.events]
        self._cache: dict[datetime, Snapshot] = {}

    def upto(self, t: datetime) -> list[Event]:
        return self.events[: bisect.bisect_right(self._ts, t)]

    def between(self, t0: datetime, t1: datetime) -> list[Event]:
        """Events with t0 < ts <= t1."""
        return self.events[bisect.bisect_right(self._ts, t0) : bisect.bisect_right(self._ts, t1)]

    def snapshot(self, t: datetime) -> Snapshot:
        if t in self._cache:
            return self._cache[t]
        objects: dict[str, ObjState] = {}
        created: dict[str, datetime] = {}
        tomb: dict[str, int] = {}
        for e in self.upto(t):
            if e.op == "delete":
                objects.pop(e.object_id, None)
                tomb[e.object_id] = e.revision
                continue
            created.setdefault(e.object_id, e.ts)
            tomb.pop(e.object_id, None)
            objects[e.object_id] = ObjState(
                e.object_id, e.source, e.revision, e.payload, created[e.object_id], e.ts
            )
        snap = Snapshot(t, objects, tomb)
        if len(self._cache) > 64:
            self._cache.clear()
        self._cache[t] = snap
        return snap

    def source_epochs(self, t: datetime) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.upto(t):
            out[e.source] = out.get(e.source, 0) + 1
        return out

    def cluster_epochs(self, t: datetime) -> dict[str, int]:
        out: dict[str, int] = {}
        for e in self.upto(t):
            c = str(e.payload.get("cluster", e.object_id))
            out[c] = out.get(c, 0) + 1
        return out

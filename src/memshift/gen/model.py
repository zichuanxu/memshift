"""Data model of a MemShift world (plan §4.2).

A world is a person's 12-month multi-source event stream plus everything needed to score it:
fact slots (the knowledge items), per-slot gold timelines for script-level facts, versioned
definitions (kappa), consent state per agent, and the fork point. Source objects carry an
`object_id`, a monotone `revision` and tombstones for deletes; gold for structured facts is
recomputed from source state, so gold and sources cannot drift apart.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any

SOURCES = ("email", "calendar", "tasks", "contacts", "notes", "messages")
TIME_FMT = "%Y-%m-%dT%H:%M"


def fmt_ts(t: datetime) -> str:
    return t.strftime(TIME_FMT)


def parse_ts(s: str) -> datetime:
    return datetime.strptime(s, TIME_FMT)


def days(n: float) -> timedelta:
    return timedelta(days=n)


@dataclass(frozen=True)
class Event:
    """One source event, shaped like an official connector delta."""

    event_id: str
    ts: datetime
    source: str
    object_id: str
    revision: int
    op: str  # insert | update | delete
    payload: dict[str, Any]  # full object state after the op (last state for delete)
    tag: str = "natural"  # natural | shift:<name>

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["ts"] = fmt_ts(self.ts)
        return d


@dataclass(frozen=True)
class Slot:
    """A knowledge item k: one point-queryable fact about the person."""

    slot_id: str
    kind: str  # user_attr | contact_attr | series | appointment | task_due | kappa
    vtype: str  # value type used by the scorer
    question: str
    description: str
    query: str  # retrieval terms for write-time and serve-time lookup
    sources: tuple[str, ...]  # declared scope at source-epoch granularity
    subject: str | None = None
    kappa_id: str | None = None
    structured: bool = False  # a programmatic extractor exists (D=1 fallback)


@dataclass(frozen=True)
class KappaDef:
    kappa_id: str
    version: str
    statement: str


@dataclass
class TimelinePoint:
    """Script-level gold for user attributes: value from `ts` on, evidenced by `deps`."""

    ts: datetime
    value: Any
    deps: list[str]


@dataclass
class Person:
    pid: str
    name: str
    email: str
    relation: str  # colleague | friend | family | provider | other


@dataclass
class World:
    world_id: str
    seed: int
    start: datetime
    end: datetime
    fork: datetime
    user_name: str
    people: dict[str, Person]
    slots: list[Slot]
    events: list[Event]
    timelines: dict[str, list[TimelinePoint]]
    kappa_defs: dict[str, dict[str, KappaDef]]  # kappa_id -> version -> def
    kappa_log: list[tuple[datetime, str, str]]  # (ts, kappa_id, version)
    agents: dict[str, list[str]]  # agent_id -> granted sources at start
    consent_log: list[tuple[datetime, str, str, str]] = field(default_factory=list)
    branch: str = "natural"
    params: dict[str, Any] = field(default_factory=dict)

    def slot(self, slot_id: str) -> Slot:
        for s in self.slots:
            if s.slot_id == slot_id:
                return s
        raise KeyError(slot_id)

    def kappa_version(self, kappa_id: str, t: datetime) -> str:
        version = "v1"
        for ts, kid, ver in sorted(self.kappa_log):
            if kid == kappa_id and ts <= t:
                version = ver
        return version

    def granted(self, agent_id: str, t: datetime) -> set[str]:
        scopes = set(self.agents.get(agent_id, []))
        for ts, aid, scope, state in sorted(self.consent_log):
            if aid == agent_id and ts <= t:
                if state == "revoked":
                    scopes.discard(scope)
                else:
                    scopes.add(scope)
        return scopes

"""Versioned personal definitions kappa with deterministic evaluators (plan §4.2 "定义 κ").

Each definition has v1 and v2. A definition change bumps the version while source bytes stay the
same; gold is recomputed by the evaluator. Evaluators read only structured fields of source
objects (calendar, contacts, tasks, message and email headers, the working-hours note), which is
also what HoldMem's D=1 checker has access to: a person who has codified a definition can
re-evaluate it mechanically. This is the D=1 extreme by construction, reported as such.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any

from memshift.gen.state import ObjState, Snapshot

KAPPA_DEFS: dict[str, dict[str, Any]] = {
    "important_contact": {
        "vtype": "set_person",
        "question": "Who are the user's important contacts under the user's current definition? "
        "List full names.",
        "description": "the set of the user's important contacts (full names)",
        "query": "calendar meeting attendees sync coffee",
        "sources": ["calendar"],
        "v1": "An important contact is a person who attended at least 5 of the user's one-off "
        "calendar meetings (not recurring series) in the last 90 days.",
        "v2": "An important contact is a person who attended at least 2 of the user's one-off "
        "calendar meetings (not recurring series) in the last 30 days.",
    },
    "close_friend": {
        "vtype": "set_person",
        "question": "Who are the user's close friends under the user's current definition? List "
        "full names.",
        "description": "the set of the user's close friends (full names)",
        "query": "chat message conversation friend family",
        "sources": ["messages"],
        "v1": "A close friend is a contact with at least 6 chat messages exchanged with the user "
        "in the last 30 days.",
        "v2": "A close friend is a contact with at least 4 chat messages exchanged with the user "
        "in the last 14 days.",
    },
    "active_project": {
        "vtype": "set_project",
        "question": "Which projects are the user's active projects under the user's current "
        "definition? List project names.",
        "description": "the set of the user's active projects (project names)",
        "query": "task project open status due",
        "sources": ["tasks"],
        "v1": "An active project is a project with at least one open task.",
        "v2": "An active project is a project with at least one task created or updated in the "
        "last 14 days (any status).",
    },
    "working_hours": {
        "vtype": "timerange",
        "question": "What are the user's working hours under the user's current definition? "
        "Answer as HH:MM-HH:MM.",
        "description": "the user's working hours as HH:MM-HH:MM",
        "query": "working hours note meetings work",
        "sources": ["notes", "calendar"],
        "v1": "Working hours are the hours declared in the user's most recent 'Working hours' "
        "note.",
        "v2": "Working hours run from the earliest start to the latest end of the user's one-off "
        "work meetings in the last 30 days, rounded outward to whole hours.",
    },
    "vip_sender": {
        "vtype": "set_person",
        "question": "Who are the user's VIP email senders under the user's current definition? "
        "List full names.",
        "description": "the set of the user's VIP email senders (full names)",
        "query": "email from sender colleague",
        "sources": ["email", "contacts"],
        "v1": "A VIP sender is a contact who sent the user at least 4 emails in the last 30 days.",
        "v2": "A VIP sender is a contact who sent the user at least 3 emails in the last 30 days, "
        "or a starred contact who sent at least 1 email in that period.",
    },
    "overdue_task": {
        "vtype": "set_task",
        "question": "Which tasks are overdue under the user's current definition? List task "
        "titles.",
        "description": "the set of the user's overdue tasks (task titles)",
        "query": "task due open overdue priority",
        "sources": ["tasks"],
        "v1": "An overdue task is an open task whose due date is before today.",
        "v2": "An overdue task is an open, high-priority task whose due date is before today.",
    },
}

SCOPE_PREDICATES = {
    "important_contact": "one-off meetings with start in [as_of-90d (v1) / -30d (v2), as_of]",
    "close_friend": "chat messages with sent in [as_of-30d (v1) / -14d (v2), as_of]",
    "active_project": "all live tasks",
    "working_hours": "note 'Working hours' (v1) / work meetings in [as_of-30d, as_of] (v2)",
    "vip_sender": "emails from contacts with sent in [as_of-30d, as_of]; contact cards",
    "overdue_task": "open tasks with due < as_of",
}

_WH_RE = re.compile(r"My working hours: (\d\d:\d\d)-(\d\d:\d\d)")
_FMT = "%Y-%m-%d %H:%M"
Result = tuple[Any, list[str]]


def _dt(s: str) -> datetime:
    return datetime.strptime(s, _FMT)


def _cards(snap: Snapshot) -> dict[str, dict[str, Any]]:
    """name -> card payload for live contact cards."""
    return {o.payload["name"]: o.payload for o in snap.by_source("contacts")}


def _meetings(
    snap: Snapshot, t: datetime, window: int, category: str | None = None
) -> list[ObjState]:
    lo = t - timedelta(days=window)
    out = []
    for o in snap.by_source("calendar"):
        p = o.payload
        if p.get("kind") != "meeting" or (category and p.get("category") != category):
            continue
        if lo <= _dt(p["start"]) <= t:
            out.append(o)
    return out


def _pick(
    counts: Counter[str], deps: dict[str, list[str]], keep: Callable[[str, int], bool]
) -> Result:
    chosen = sorted(n for n, c in counts.items() if keep(n, c))
    return chosen, sorted({d for n in chosen for d in deps[n]})


def important_contact(snap: Snapshot, t: datetime, version: str) -> Result:
    counts: Counter[str] = Counter()
    deps: dict[str, list[str]] = {}
    for o in _meetings(snap, t, 90 if version == "v1" else 30):
        for name in o.payload["attendees"]:
            counts[name] += 1
            deps.setdefault(name, []).append(o.object_id)
    return _pick(counts, deps, lambda n, c: c >= (5 if version == "v1" else 2))


def close_friend(snap: Snapshot, t: datetime, version: str) -> Result:
    lo = t - timedelta(days=30 if version == "v1" else 14)
    counts: Counter[str] = Counter()
    deps: dict[str, list[str]] = {}
    for o in snap.by_source("messages"):
        if lo <= _dt(o.payload["sent"]) <= t:
            name = o.payload["conversation_with"]
            counts[name] += 1
            deps.setdefault(name, []).append(o.object_id)
    return _pick(counts, deps, lambda n, c: c >= (6 if version == "v1" else 4))


def active_project(snap: Snapshot, t: datetime, version: str) -> Result:
    lo = t - timedelta(days=14)
    chosen: dict[str, list[str]] = {}
    for o in snap.by_source("tasks"):
        p = o.payload
        ok = p["status"] == "open" if version == "v1" else _dt(p["updated"]) >= lo
        if ok:
            chosen.setdefault(p["project"], []).append(o.object_id)
    return sorted(chosen), sorted({d for v in chosen.values() for d in v})


def working_hours(snap: Snapshot, t: datetime, version: str) -> Result:
    if version == "v1":
        notes = [o for o in snap.by_source("notes") if o.payload["title"] == "Working hours"]
        if not notes:
            return None, []
        note = max(notes, key=lambda o: o.updated)
        m = _WH_RE.search(note.payload["body"])
        return (f"{m.group(1)}-{m.group(2)}", [note.object_id]) if m else (None, [])
    meetings = _meetings(snap, t, 30, category="work")
    if not meetings:
        return None, []
    h0 = min(_dt(o.payload["start"]).hour for o in meetings)
    h1 = max(
        _dt(o.payload["end"]).hour + (1 if _dt(o.payload["end"]).minute else 0) for o in meetings
    )
    return f"{h0:02d}:00-{h1:02d}:00", sorted(o.object_id for o in meetings)


def vip_sender(snap: Snapshot, t: datetime, version: str) -> Result:
    lo = t - timedelta(days=30)
    cards = _cards(snap)
    by_mail = {c["email"]: c for c in cards.values()}
    counts: Counter[str] = Counter()
    deps: dict[str, list[str]] = {}
    for o in snap.by_source("email"):
        card = by_mail.get(o.payload["from_addr"])
        if card and lo <= _dt(o.payload["sent"]) <= t:
            counts[card["name"]] += 1
            deps.setdefault(card["name"], []).append(o.object_id)
    if version == "v1":
        return _pick(counts, deps, lambda n, c: c >= 4)
    return _pick(counts, deps, lambda n, c: c >= 3 or bool(cards.get(n, {}).get("starred")))


def overdue_task(snap: Snapshot, t: datetime, version: str) -> Result:
    today = t.strftime("%Y-%m-%d")
    chosen: dict[str, str] = {}
    for o in snap.by_source("tasks"):
        p = o.payload
        if (
            p["status"] == "open"
            and p["due"] < today
            and (version == "v1" or p["priority"] == "high")
        ):
            chosen[p["title"]] = o.object_id
    return sorted(chosen), sorted(chosen.values())


EVALUATORS: dict[str, Callable[[Snapshot, datetime, str], Result]] = {
    "important_contact": important_contact,
    "close_friend": close_friend,
    "active_project": active_project,
    "working_hours": working_hours,
    "vip_sender": vip_sender,
    "overdue_task": overdue_task,
}


def evaluate(kappa_id: str, snap: Snapshot, t: datetime, version: str) -> Result:
    return EVALUATORS[kappa_id](snap, t, version)


# Scope ranges: the executable form of SCOPE_PREDICATES. The objects a definition ranges over at t
# (object id, revision), i.e. the records the predicate admits, not the records that support the
# value: a value depends on nothing outside its range, and an in-range change (an object enters,
# leaves or is revised, including by a window sliding with t) may change it. Used by the
# scope-aware key of the E1 cell `B1_scope`; they read no evaluator output.
Members = list[tuple[str, int]]


def _members(objs: Iterable[ObjState]) -> Members:
    return sorted((o.object_id, o.revision) for o in objs)


def _scope_important_contact(snap: Snapshot, t: datetime, version: str) -> Members:
    return _members(_meetings(snap, t, 90 if version == "v1" else 30))


def _scope_close_friend(snap: Snapshot, t: datetime, version: str) -> Members:
    lo = t - timedelta(days=30 if version == "v1" else 14)
    return _members(o for o in snap.by_source("messages") if lo <= _dt(o.payload["sent"]) <= t)


def _scope_active_project(snap: Snapshot, t: datetime, version: str) -> Members:
    if version == "v1":
        return _members(snap.by_source("tasks"))
    lo = t - timedelta(days=14)
    return _members(o for o in snap.by_source("tasks") if _dt(o.payload["updated"]) >= lo)


def _scope_working_hours(snap: Snapshot, t: datetime, version: str) -> Members:
    if version == "v1":
        return _members(o for o in snap.by_source("notes") if o.payload["title"] == "Working hours")
    return _members(_meetings(snap, t, 30, category="work"))


def _scope_vip_sender(snap: Snapshot, t: datetime, version: str) -> Members:
    lo = t - timedelta(days=30)
    cards = snap.by_source("contacts")
    addresses = {o.payload["email"] for o in cards}
    mails = (
        o
        for o in snap.by_source("email")
        if o.payload["from_addr"] in addresses and lo <= _dt(o.payload["sent"]) <= t
    )
    return _members([*cards, *mails])


def _scope_overdue_task(snap: Snapshot, t: datetime, version: str) -> Members:
    today = t.strftime("%Y-%m-%d")
    return _members(
        o
        for o in snap.by_source("tasks")
        if o.payload["status"] == "open" and o.payload["due"] < today
    )


SCOPE_RANGES: dict[str, Callable[[Snapshot, datetime, str], Members]] = {
    "important_contact": _scope_important_contact,
    "close_friend": _scope_close_friend,
    "active_project": _scope_active_project,
    "working_hours": _scope_working_hours,
    "vip_sender": _scope_vip_sender,
    "overdue_task": _scope_overdue_task,
}


def scope_members(kappa_id: str, snap: Snapshot, t: datetime, version: str) -> Members:
    """The (object id, revision) of every object the definition ranges over at t (version given)."""
    return SCOPE_RANGES[kappa_id](snap, t, version)

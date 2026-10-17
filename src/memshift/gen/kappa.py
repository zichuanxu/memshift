"""Versioned personal definitions kappa with deterministic evaluators (plan §4.2 "定义 κ").

Each definition has v1 and v2. A definition change bumps the version while source bytes stay the
same; gold is recomputed by the evaluator. Evaluators read only structured fields of source
objects (calendar, contacts, tasks, message and email headers, the working-hours note), which is
also what HoldMem's D=1 checker has access to: a person who has codified a definition can
re-evaluate it mechanically. This is the D=1 extreme by construction, reported as such.

Held-out definitions (prereg v4_round3 §G-a). Besides the six fixed definitions (`KAPPA_DEFS`), a
grammar samples definitions as (source, predicate over records, aggregate, window, threshold):

- predicate (`GRAMMAR`): a filter over the records of one source, optionally joined with contact
  cards, with a past window in days (`due_ahead`: a window ahead of today). Every (source,
  predicate) pair differs from the fixed six's (`FIXED_PAIRS`).
- aggregate: `count` (the people or projects with at least `threshold` admitted records),
  `members` (the titles of the admitted tasks) or `span` (earliest start to latest end of the
  admitted meetings, rounded outward to whole hours).
- version 2 changes either the threshold or the window (one of the two, never both); `members`
  and `span` have no threshold, so their v2 changes the window.

A draw (`heldout_draw(seed)`) is six definitions with distinct predicates, sampled from
`random.Random(seed)`: predicate uniformly among those not yet in the draw, then aggregate,
window, threshold and the v2 change uniformly from the predicate's options. Each candidate is
checked by a rejection rule computed offline from gold only (no model) before it joins the draw:

  For each version separately, compute the candidate's gold on worlds 1000-1009 (natural branch,
  default parameters; the def branch has the same events) at the four post-fork checkpoints
  fork + 1, 2, 4, 8 weeks. Reject the candidate if, for either version, its gold is constant across
  the four checkpoints in more than half the worlds (6 or more of 10), or empty or abstain (no
  value, or the empty set) in more than half of the item-checkpoints (21 or more of 40). A rejected
  candidate is discarded and the next one is drawn from the same random stream, so the draw stays
  a function of the seed.

A held-out definition's id is `h<seed>_<name>`, so `evaluate` and `scope_members` resolve it from
the id alone (`heldout_spec`). Worlds take a draw through the world parameter
`kappa_set: "heldout:<seed>"` (`kappa_set_defs`, `gen.world.to_world`).
"""

from __future__ import annotations

import argparse
import random
import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import cache
from typing import Any

from memshift.gen.state import ObjState, Snapshot, SourceLog

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


def _hour_span(meetings: list[ObjState]) -> Result:
    """Earliest start to latest end of the meetings, rounded outward to whole hours."""
    if not meetings:
        return None, []
    h0 = min(_dt(o.payload["start"]).hour for o in meetings)
    h1 = max(
        _dt(o.payload["end"]).hour + (1 if _dt(o.payload["end"]).minute else 0) for o in meetings
    )
    return f"{h0:02d}:00-{h1:02d}:00", sorted(o.object_id for o in meetings)


def working_hours(snap: Snapshot, t: datetime, version: str) -> Result:
    if version == "v1":
        notes = [o for o in snap.by_source("notes") if o.payload["title"] == "Working hours"]
        if not notes:
            return None, []
        note = max(notes, key=lambda o: o.updated)
        m = _WH_RE.search(note.payload["body"])
        return (f"{m.group(1)}-{m.group(2)}", [note.object_id]) if m else (None, [])
    return _hour_span(_meetings(snap, t, 30, category="work"))


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
    """Gold value and supporting object ids of a definition (fixed or held-out) at t."""
    if kappa_id in EVALUATORS:
        return EVALUATORS[kappa_id](snap, t, version)
    return evaluate_spec(heldout_spec(kappa_id, version), snap, t)


def scope_predicate(kappa_id: str) -> str:
    """The English scope predicate of a definition (both versions), recorded with its gold."""
    if kappa_id in SCOPE_PREDICATES:
        return SCOPE_PREDICATES[kappa_id]
    d = heldout_def(kappa_id)
    pred = GRAMMAR[d.v1.predicate]
    return f"{pred.records}; window {d.v1.window}d (v1) / {d.v2.window}d (v2)"


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
    if kappa_id in SCOPE_RANGES:
        return SCOPE_RANGES[kappa_id](snap, t, version)
    return scope_spec(heldout_spec(kappa_id, version), snap, t)


# ---------------------------------------------------------------- held-out definitions (G-a)
# Records admitted by a predicate at t for a window, and the contact cards it consulted (a join).
Admitted = tuple[list[ObjState], list[ObjState]]


def _hour(o: ObjState, key: str) -> int:
    return _dt(o.payload[key]).hour


def _recent(snap: Snapshot, source: str, key: str, t: datetime, window: int) -> list[ObjState]:
    lo = t - timedelta(days=window)
    return [o for o in snap.by_source(source) if lo <= _dt(o.payload[key]) <= t]


def _social_meeting(snap: Snapshot, t: datetime, window: int) -> Admitted:
    return _meetings(snap, t, window, category="social"), []


def _group_meeting(snap: Snapshot, t: datetime, window: int) -> Admitted:
    return [o for o in _meetings(snap, t, window) if len(o.payload["attendees"]) >= 2], []


def _morning_meeting(snap: Snapshot, t: datetime, window: int) -> Admitted:
    return [o for o in _meetings(snap, t, window) if _hour(o, "start") < 10], []


def _long_meeting(snap: Snapshot, t: datetime, window: int) -> Admitted:
    hour = timedelta(minutes=60)
    return [
        o
        for o in _meetings(snap, t, window)
        if _dt(o.payload["end"]) - _dt(o.payload["start"]) >= hour
    ], []


def _sent_message(snap: Snapshot, t: datetime, window: int) -> Admitted:
    msgs = _recent(snap, "messages", "sent", t, window)
    return [o for o in msgs if o.payload["sender"] != o.payload["conversation_with"]], []


def _received_message(snap: Snapshot, t: datetime, window: int) -> Admitted:
    msgs = _recent(snap, "messages", "sent", t, window)
    return [o for o in msgs if o.payload["sender"] == o.payload["conversation_with"]], []


def _evening_message(snap: Snapshot, t: datetime, window: int) -> Admitted:
    return [o for o in _recent(snap, "messages", "sent", t, window) if _hour(o, "sent") >= 18], []


def _family_message(snap: Snapshot, t: datetime, window: int) -> Admitted:
    family = [o for o in snap.by_source("contacts") if o.payload["relation"] == "family"]
    names = {o.payload["name"] for o in family}
    msgs = _recent(snap, "messages", "sent", t, window)
    return [o for o in msgs if o.payload["conversation_with"] in names], family


def _after_hours_email(snap: Snapshot, t: datetime, window: int) -> Admitted:
    cards = snap.by_source("contacts")
    addresses = {o.payload["email"] for o in cards}
    mails = _recent(snap, "email", "sent", t, window)
    return [
        o for o in mails if o.payload["from_addr"] in addresses and _hour(o, "sent") >= 17
    ], cards


def _done_task(snap: Snapshot, t: datetime, window: int) -> Admitted:
    tasks = _recent(snap, "tasks", "updated", t, window)
    return [o for o in tasks if o.payload["status"] == "done"], []


def _due_ahead(snap: Snapshot, t: datetime, window: int) -> Admitted:
    today = t.strftime("%Y-%m-%d")
    until = (t + timedelta(days=window)).strftime("%Y-%m-%d")
    return [
        o
        for o in snap.by_source("tasks")
        if o.payload["status"] == "open" and today <= o.payload["due"] <= until
    ], []


def _high_priority_task(snap: Snapshot, t: datetime, window: int) -> Admitted:
    tasks = _recent(snap, "tasks", "updated", t, window)
    return [o for o in tasks if o.payload["priority"] == "high"], []


def _attendees(o: ObjState) -> list[str]:
    return list(o.payload["attendees"])


def _partner(o: ObjState) -> list[str]:
    return [o.payload["conversation_with"]]


def _sender(o: ObjState) -> list[str]:
    # admitted emails come from a contact card's address, and card names never change, so the
    # sender name is the card's
    return [o.payload["from_name"]]


def _project(o: ObjState) -> list[str]:
    return [o.payload["project"]]


def _title(o: ObjState) -> list[str]:
    return [o.payload["title"]]


@dataclass(frozen=True)
class Form:
    """One aggregate over a predicate's records, with the text a definition item needs.

    `statement` is formatted with k (threshold), w (window, days), and s and were (agreement
    with k)."""

    name: str
    plural: str  # the defined term, plural ("social regulars")
    vtype: str
    statement: str
    query: str
    key: Callable[[ObjState], list[str]] | None = None  # group or member key (count, members)


@dataclass(frozen=True)
class Predicate:
    source: str
    sources: tuple[str, ...]  # declared sources: the source and any joined one
    records: str  # what it admits, in words (the scope predicate)
    admit: Callable[[Snapshot, datetime, int], Admitted]
    windows: tuple[int, ...]
    thresholds: tuple[int, ...]
    forms: dict[str, Form]  # aggregate -> form


_MEET_Q = "calendar meeting attendees"
GRAMMAR: dict[str, Predicate] = {
    "social_meeting": Predicate(
        "calendar",
        ("calendar",),
        "one-off social meetings (category social) with start in [as_of-window, as_of]",
        _social_meeting,
        (14, 21, 30, 45, 60, 90),
        (1, 2, 3, 4),
        {
            "count": Form(
                "social_regular",
                "social regulars",
                "set_person",
                "A social regular is a person who attended at least {k} of the user's one-off "
                "social calendar meetings (category social, not recurring series) in the last {w} "
                "days.",
                f"{_MEET_Q} social coffee friend",
                _attendees,
            ),
            "span": Form(
                "social_hours",
                "social hours",
                "timerange",
                "Social hours run from the earliest start to the latest end of the user's one-off "
                "social calendar meetings (category social) in the last {w} days, rounded outward "
                "to whole hours.",
                "calendar meeting social coffee time",
            ),
        },
    ),
    "group_meeting": Predicate(
        "calendar",
        ("calendar",),
        "one-off meetings with two or more attendees with start in [as_of-window, as_of]",
        _group_meeting,
        (30, 45, 60, 90),
        (1, 2, 3),
        {
            "count": Form(
                "team_regular",
                "team regulars",
                "set_person",
                "A team regular is a person who attended at least {k} of the user's one-off "
                "calendar meetings with two or more attendees (not recurring series) in the last "
                "{w} days.",
                f"{_MEET_Q} sync team",
                _attendees,
            ),
        },
    ),
    "morning_meeting": Predicate(
        "calendar",
        ("calendar",),
        "one-off meetings starting before 10:00 with start in [as_of-window, as_of]",
        _morning_meeting,
        (30, 45, 60, 90),
        (1, 2, 3),
        {
            "count": Form(
                "morning_contact",
                "morning contacts",
                "set_person",
                "A morning contact is a person who attended at least {k} of the user's one-off "
                "calendar meetings starting before 10:00 (not recurring series) in the last {w} "
                "days.",
                f"{_MEET_Q} morning early sync",
                _attendees,
            ),
        },
    ),
    "long_meeting": Predicate(
        "calendar",
        ("calendar",),
        "one-off meetings lasting 60 minutes or more with start in [as_of-window, as_of]",
        _long_meeting,
        (30, 45, 60, 90),
        (1, 2, 3, 4),
        {
            "count": Form(
                "deep_dive_partner",
                "deep-dive partners",
                "set_person",
                "A deep-dive partner is a person who attended at least {k} of the user's one-off "
                "calendar meetings lasting 60 minutes or more (not recurring series) in the last "
                "{w} days.",
                f"{_MEET_Q} long sync",
                _attendees,
            ),
        },
    ),
    "sent_message": Predicate(
        "messages",
        ("messages",),
        "chat messages sent by the user with sent in [as_of-window, as_of]",
        _sent_message,
        (14, 21, 30, 60),
        (2, 3, 4, 5, 6, 8),
        {
            "count": Form(
                "frequent_recipient",
                "frequent recipients",
                "set_person",
                "A frequent recipient is a contact to whom the user sent at least {k} chat "
                "message{s} in the last {w} days.",
                "chat message conversation sent friend family",
                _partner,
            ),
        },
    ),
    "received_message": Predicate(
        "messages",
        ("messages",),
        "chat messages sent to the user with sent in [as_of-window, as_of]",
        _received_message,
        (14, 21, 30, 60),
        (2, 3, 4, 5, 6, 8),
        {
            "count": Form(
                "frequent_texter",
                "frequent texters",
                "set_person",
                "A frequent texter is a contact who sent the user at least {k} chat message{s} in "
                "the last {w} days.",
                "chat message conversation from friend family",
                _partner,
            ),
        },
    ),
    "evening_message": Predicate(
        "messages",
        ("messages",),
        "chat messages sent at or after 18:00 with sent in [as_of-window, as_of]",
        _evening_message,
        (14, 21, 30, 60),
        (1, 2, 3, 4),
        {
            "count": Form(
                "evening_chat_partner",
                "evening chat partners",
                "set_person",
                "An evening chat partner is a contact with at least {k} chat message{s} exchanged "
                "with the user at or after 18:00 in the last {w} days.",
                "chat message conversation evening friend family",
                _partner,
            ),
        },
    ),
    "family_message": Predicate(
        "messages",
        ("messages", "contacts"),
        "chat messages with family contacts with sent in [as_of-window, as_of]; family cards",
        _family_message,
        (14, 21, 30, 60),
        (2, 3, 4, 6, 8),
        {
            "count": Form(
                "close_relative",
                "close relatives",
                "set_person",
                "A close relative is a family contact (relationship family on the contact card) "
                "with at least {k} chat message{s} exchanged with the user in the last {w} days.",
                "chat message conversation family relationship contact",
                _partner,
            ),
        },
    ),
    "after_hours_email": Predicate(
        "email",
        ("email", "contacts"),
        "emails from contacts sent at or after 17:00 with sent in [as_of-window, as_of]; contact "
        "cards",
        _after_hours_email,
        (30, 45, 60, 90),
        (1, 2, 3),
        {
            "count": Form(
                "after_hours_sender",
                "after-hours senders",
                "set_person",
                "An after-hours sender is a contact who sent the user at least {k} email{s} at or "
                "after 17:00 in the last {w} days.",
                "email from sender colleague evening",
                _sender,
            ),
        },
    ),
    "done_task": Predicate(
        "tasks",
        ("tasks",),
        "tasks with status done and updated in [as_of-window, as_of]",
        _done_task,
        (7, 14, 21, 30, 45, 60),
        (1, 2, 3),
        {
            "members": Form(
                "recently_completed_task",
                "recently completed tasks",
                "set_task",
                "A recently completed task is a task with status done that was last updated in "
                "the last {w} days.",
                "task done completed status project",
                _title,
            ),
            "count": Form(
                "productive_project",
                "productive projects",
                "set_project",
                "A productive project is a project with at least {k} task{s} with status done "
                "that {were} last updated in the last {w} days.",
                "task project done completed status",
                _project,
            ),
        },
    ),
    "due_ahead": Predicate(
        "tasks",
        ("tasks",),
        "open tasks with as_of <= due <= as_of+window",
        _due_ahead,
        (7, 10, 14, 21, 30),
        (1, 2, 3),
        {
            "members": Form(
                "upcoming_task",
                "upcoming tasks",
                "set_task",
                "An upcoming task is an open task whose due date is between today and {w} days "
                "from today (inclusive).",
                "task due open upcoming",
                _title,
            ),
            "count": Form(
                "busy_project",
                "busy projects",
                "set_project",
                "A busy project is a project with at least {k} open task{s} whose due date is "
                "between today and {w} days from today (inclusive).",
                "task project due open upcoming",
                _project,
            ),
        },
    ),
    "high_priority_task": Predicate(
        "tasks",
        ("tasks",),
        "high-priority tasks (any status) with updated in [as_of-window, as_of]",
        _high_priority_task,
        (7, 14, 21, 30, 45, 60),
        (1, 2),
        {
            "members": Form(
                "hot_task",
                "hot tasks",
                "set_task",
                "A hot task is a high-priority task (any status) that was last updated in the "
                "last {w} days.",
                "task priority high updated",
                _title,
            ),
            "count": Form(
                "priority_project",
                "priority projects",
                "set_project",
                "A priority project is a project with at least {k} high-priority task{s} (any "
                "status) that {were} last updated in the last {w} days.",
                "task project priority high updated",
                _project,
            ),
        },
    ),
}

# The (source, predicate) pairs of the six fixed definitions, both versions, in the grammar's
# vocabulary. The grammar's pairs are disjoint from these (tested).
FIXED_PAIRS = frozenset(
    {
        ("calendar", "one_off_meeting"),  # important_contact
        ("calendar", "work_meeting"),  # working_hours v2
        ("notes", "working_hours_note"),  # working_hours v1
        ("messages", "message"),  # close_friend
        ("tasks", "open_task"),  # active_project v1
        ("tasks", "updated_task"),  # active_project v2
        ("tasks", "overdue_task"),  # overdue_task v1
        ("tasks", "overdue_high_priority_task"),  # overdue_task v2
        ("email", "email_from_contact"),  # vip_sender
        ("contacts", "starred_contact"),  # vip_sender v2
    }
)


@dataclass(frozen=True)
class Spec:
    """One version of a held-out definition: (source, predicate, aggregate, window, threshold).
    `threshold` is 0 for aggregates without one (members, span)."""

    source: str
    predicate: str
    aggregate: str
    window: int
    threshold: int

    @property
    def form(self) -> Form:
        return GRAMMAR[self.predicate].forms[self.aggregate]

    def statement(self) -> str:
        k = self.threshold
        one = k == 1
        return self.form.statement.format(
            k=k, w=self.window, s="" if one else "s", were="was" if one else "were"
        )


def evaluate_spec(spec: Spec, snap: Snapshot, t: datetime) -> Result:
    """The generic evaluator: gold value and supporting object ids of a held-out definition."""
    records, _ = GRAMMAR[spec.predicate].admit(snap, t, spec.window)
    if spec.aggregate == "span":
        return _hour_span(records)
    key = spec.form.key
    assert key is not None
    if spec.aggregate == "members":
        chosen: dict[str, str] = {}
        for o in records:
            for name in key(o):
                chosen[name] = o.object_id
        return sorted(chosen), sorted(set(chosen.values()))
    counts: Counter[str] = Counter()
    deps: dict[str, list[str]] = {}
    for o in records:
        for name in key(o):
            counts[name] += 1
            deps.setdefault(name, []).append(o.object_id)
    return _pick(counts, deps, lambda n, c: c >= spec.threshold)


def scope_spec(spec: Spec, snap: Snapshot, t: datetime) -> Members:
    """The scope range of a held-out definition: admitted records and the cards it consulted."""
    records, cards = GRAMMAR[spec.predicate].admit(snap, t, spec.window)
    return _members([*cards, *records])


@dataclass(frozen=True)
class Sanity:
    """Gold of one candidate on the rejection worlds, per version: worlds (of 10) whose gold is
    constant across the four post-fork checkpoints, item-checkpoints (of 40) that are empty or
    abstain; and item-checkpoints where v1 and v2 gold differ."""

    constant: tuple[int, int]
    empty: tuple[int, int]
    v1_ne_v2: int
    n_worlds: int
    n_points: int

    def rejected(self) -> str | None:
        for i, ver in enumerate(("v1", "v2")):
            if self.constant[i] > self.n_worlds // 2:
                return f"{ver} constant in {self.constant[i]}/{self.n_worlds} worlds"
            if self.empty[i] > self.n_points // 2:
                return f"{ver} empty in {self.empty[i]}/{self.n_points} item-checkpoints"
        return None

    def summary(self) -> str:
        return (
            f"v1 constant {self.constant[0]}/{self.n_worlds} worlds, empty "
            f"{self.empty[0]}/{self.n_points}; v2 constant {self.constant[1]}/{self.n_worlds}, "
            f"empty {self.empty[1]}/{self.n_points}; v1 != v2 at {self.v1_ne_v2}/{self.n_points}"
        )


@dataclass(frozen=True)
class HeldoutDef:
    kappa_id: str
    v1: Spec
    v2: Spec
    changed: str  # threshold | window
    sanity: Sanity

    @property
    def form(self) -> Form:
        return self.v1.form

    def as_kappa_def(self) -> dict[str, Any]:
        """The definition in the shape of a `KAPPA_DEFS` entry."""
        f = self.form
        noun = {
            "set_person": ("Who are", "List full names.", "full names"),
            "set_project": ("Which projects are", "List project names.", "project names"),
            "set_task": ("Which tasks are", "List task titles.", "task titles"),
            "timerange": ("What are", "Answer as HH:MM-HH:MM.", ""),
        }
        ask, answer, unit = noun[f.vtype]
        description = (
            f"the user's {f.plural} as HH:MM-HH:MM"
            if f.vtype == "timerange"
            else f"the set of the user's {f.plural} ({unit})"
        )
        return {
            "vtype": f.vtype,
            "question": f"{ask} the user's {f.plural} under the user's current definition? "
            f"{answer}",
            "description": description,
            "query": f.query,
            "sources": list(GRAMMAR[self.v1.predicate].sources),
            "v1": self.v1.statement(),
            "v2": self.v2.statement(),
        }


@dataclass
class Draw:
    seed: int
    defs: list[HeldoutDef]
    rejected: list[tuple[Spec, Spec, str]] = field(default_factory=list)


REJECTION_WORLDS = tuple(range(1000, 1010))
REJECTION_WEEKS = (1, 2, 4, 8)
DRAW_SIZE = 6
MAX_CANDIDATES = 500


@cache
def _rejection_logs() -> tuple[tuple[SourceLog, tuple[datetime, ...]], ...]:
    # imported here: gen.world imports this module
    from memshift.gen.world import generate_world

    out = []
    for seed in REJECTION_WORLDS:
        world = generate_world(seed)
        times = tuple(world.fork + timedelta(weeks=k) for k in REJECTION_WEEKS)
        out.append((SourceLog(world.events), times))
    return tuple(out)


def _empty(value: Any) -> bool:
    return value is None or value == []


def sanity(v1: Spec, v2: Spec) -> Sanity:
    """The rejection statistics of a candidate (see the module docstring)."""
    constant, empty, ne = [0, 0], [0, 0], 0
    logs = _rejection_logs()
    for log, times in logs:
        values: list[list[Any]] = [[], []]
        for t in times:
            snap = log.snapshot(t)
            for i, spec in enumerate((v1, v2)):
                values[i].append(evaluate_spec(spec, snap, t)[0])
        for i in (0, 1):
            constant[i] += all(v == values[i][0] for v in values[i])
            empty[i] += sum(_empty(v) for v in values[i])
        ne += sum(a != b for a, b in zip(values[0], values[1], strict=True))
    n_points = len(logs) * len(REJECTION_WEEKS)
    return Sanity((constant[0], constant[1]), (empty[0], empty[1]), ne, len(logs), n_points)


def _candidate(rng: random.Random, used: set[str]) -> tuple[Spec, Spec, str]:
    pname = rng.choice(sorted(set(GRAMMAR) - used))
    pred = GRAMMAR[pname]
    aggregate = rng.choice(sorted(pred.forms))
    window = rng.choice(pred.windows)
    threshold = rng.choice(pred.thresholds) if aggregate == "count" else 0
    changed = rng.choice(("threshold", "window")) if aggregate == "count" else "window"
    if changed == "threshold":
        w2, k2 = window, rng.choice([k for k in pred.thresholds if k != threshold])
    else:
        w2, k2 = rng.choice([w for w in pred.windows if w != window]), threshold
    v1 = Spec(pred.source, pname, aggregate, window, threshold)
    return v1, Spec(pred.source, pname, aggregate, w2, k2), changed


@cache
def heldout_draw(seed: int) -> Draw:
    """Six held-out definitions with distinct predicates, a function of `seed` (see the module
    docstring)."""
    rng = random.Random(seed)
    draw = Draw(seed, [])
    used: set[str] = set()
    for _ in range(MAX_CANDIDATES):
        v1, v2, changed = _candidate(rng, used)
        stats = sanity(v1, v2)
        reason = stats.rejected()
        if reason is not None:
            draw.rejected.append((v1, v2, reason))
            continue
        used.add(v1.predicate)
        draw.defs.append(HeldoutDef(f"h{seed}_{v1.form.name}", v1, v2, changed, stats))
        if len(draw.defs) == DRAW_SIZE:
            return draw
    raise RuntimeError(f"heldout:{seed}: no {DRAW_SIZE} definitions in {MAX_CANDIDATES} draws")


_HELDOUT_ID = re.compile(r"h(\d+)_[a-z_]+")


def heldout_def(kappa_id: str) -> HeldoutDef:
    m = _HELDOUT_ID.fullmatch(kappa_id)
    if m is None:
        raise KeyError(kappa_id)
    for d in heldout_draw(int(m.group(1))).defs:
        if d.kappa_id == kappa_id:
            return d
    raise KeyError(kappa_id)


def heldout_spec(kappa_id: str, version: str) -> Spec:
    d = heldout_def(kappa_id)
    return d.v1 if version == "v1" else d.v2


def kappa_set_defs(kappa_set: str | None) -> dict[str, dict[str, Any]]:
    """The six definitions of a world: `KAPPA_DEFS` for None or "fixed", a draw for
    "heldout:<seed>"."""
    if kappa_set is None or kappa_set == "fixed":
        return KAPPA_DEFS
    m = re.fullmatch(r"heldout:(\d+)", kappa_set)
    if m is None:
        raise ValueError(f"unknown kappa_set {kappa_set!r}")
    return {d.kappa_id: d.as_kappa_def() for d in heldout_draw(int(m.group(1))).defs}


def _spec_text(spec: Spec) -> str:
    k = f", threshold {spec.threshold}" if spec.aggregate == "count" else ""
    return f"{spec.source} / {spec.predicate} / {spec.aggregate}, window {spec.window}d{k}"


def list_heldout(seeds: Sequence[int]) -> str:
    worlds = f"{REJECTION_WORLDS[0]}-{REJECTION_WORLDS[-1]}"
    weeks = ", ".join(map(str, REJECTION_WEEKS))
    lines = [
        f"Held-out definitions. Gold sanity on worlds {worlds} (natural branch), checkpoints "
        f"fork + {weeks} weeks; rejection: for either version, constant in more than half the "
        f"worlds or empty/abstain in more than half the item-checkpoints."
    ]
    for seed in seeds:
        draw = heldout_draw(seed)
        n = len(draw.rejected)
        lines += ["", f"heldout:{seed} ({n} candidate{'' if n == 1 else 's'} rejected)"]
        for d in draw.defs:
            spec = d.as_kappa_def()
            lines += [
                f"  {d.kappa_id}  [{_spec_text(d.v1)}; v2 changes the {d.changed}]",
                f"    vtype {spec['vtype']}, sources {', '.join(spec['sources'])}",
                f"    v1: {spec['v1']}",
                f"    v2: {spec['v2']}",
                f"    question: {spec['question']}",
                f"    gold: {d.sanity.summary()}",
            ]
        for v1, v2, reason in draw.rejected:
            change = "threshold" if v1.threshold != v2.threshold else "window"
            lines.append(f"  rejected: {_spec_text(v1)}; v2 {change} -> {reason}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="held-out definitions (prereg v4_round3 §G-a)")
    ap.add_argument("--list-heldout", type=int, nargs="+", metavar="SEED", required=True)
    print(list_heldout(ap.parse_args().list_heldout))

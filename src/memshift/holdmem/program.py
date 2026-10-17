"""Writer-authored executable derivations for the D1s cell (prereg v2 §4, construct check).

In the D=1 cell a failed check falls back to rules written by the benchmark authors, and for
definition items both the checker and the fallback call the gold evaluator `gen.kappa.evaluate`.
D1s replaces both uses of benchmark-authored logic by a program the writer model writes itself:
every item write sends one more writer request asking for `def derive(records, as_of)` over the
structured records of the item's declared sources, and the program runs in the restricted
interpreter of `sandbox.py`.

The request depends only on the item description, its value type, the definition statement in
effect (definition items) and the declared sources, never on world data or on the time beyond the
definition version, so identical items across worlds and times share one cached response. A source
change re-runs the same program on the new records; a definition change is a new request.

Program sets from other writers (prereg v4 block G-b) send the same request text through another
client: another model, a sampling temperature (none for a Claude model, see
`experiments.e1.program_extra_body`), and a sample index that the client's body carries as
`memshift_sample` (part of the cache key, removed before the HTTP request), so each sample has its
own cache entry.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from concurrent.futures import Future
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from memshift.gen.model import Slot, World
from memshift.gen.state import Snapshot, SourceLog
from memshift.holdmem.checker import CheckResult
from memshift.holdmem.sandbox import (
    BUILTIN_FUNCTIONS,
    EXCEPTION_CLASSES,
    LOCATOR_FIELDS,
    METADATA_FIELDS,
    TIMEOUT_S,
    run_source,
    run_source_ra,
)
from memshift.holdmem.writer import VALUE_FORMAT
from memshift.llm import LLMClient, LLMResult
from memshift.scoring.normalize import values_match
from memshift.textutil import sha256_text

PROGRAM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}

SYSTEM = (
    "You maintain a personal knowledge store for one person (called 'the user'). You write small, "
    "deterministic Python functions that compute one knowledge item from the user's structured "
    "source records."
)

COMMON_FIELDS = (
    "Every record is a dict. Besides the source-specific keys below it has: object_id (str), "
    'source (str), revision (int), updated (str "YYYY-MM-DD HH:MM", time of the last change), '
    "cluster (str, grouping id) and text (str, the record rendered as plain text)."
)

SOURCE_FIELDS = {
    "contacts": (
        "contacts (one record per contact card): name (str, full name), email (str), "
        "phone (str), employer (str, company name), city (str), "
        "relation (str: colleague | friend | family | other), starred (bool)"
    ),
    "calendar": (
        "calendar (four kinds of record, told apart by the str key `kind`):\n"
        '  kind == "settings": timezone (str, IANA time zone name)\n'
        '  kind == "series" (recurring event): title (str), weekday (str, e.g. "Tuesday"), '
        'time (str "HH:MM", start time), duration_min (int), attendees (list of str, full '
        "names), category (str: work | personal)\n"
        '  kind == "appointment": appt_type (str, e.g. "dentist"), provider (str), title (str), '
        'start (str "YYYY-MM-DD HH:MM"), duration_min (int)\n'
        '  kind == "meeting" (one-off event): title (str), start (str "YYYY-MM-DD HH:MM"), '
        'end (str "YYYY-MM-DD HH:MM"), attendees (list of str, full names), '
        "category (str: work | social)"
    ),
    "tasks": (
        "tasks (one record per task): title (str), project (str), "
        'due (str "YYYY-MM-DD"), status (str: open | done), '
        "priority (str: low | medium | high)"
    ),
    "email": (
        "email (one record per message): from_name (str), from_addr (str), to (str), "
        'subject (str), body (str), sent (str "YYYY-MM-DD HH:MM")'
    ),
    "messages": (
        "messages (one record per chat message): conversation_with (str, full name of the other "
        'person), contact (str, key of that person, e.g. "p03"), sender (str, full name), '
        'body (str), sent (str "YYYY-MM-DD HH:MM")'
    ),
    "notes": "notes (one record per note): title (str), body (str)",
}

TOOLBOX = (
    "Available to the function: the builtins "
    + ", ".join(BUILTIN_FUNCTIONS)
    + "; the exception classes "
    + ", ".join(EXCEPTION_CLASSES)
    + "; `datetime` and `timedelta` from the datetime module; and `parse_ts(s)`, which parses "
    '"YYYY-MM-DD HH:MM" (also "YYYY-MM-DDTHH:MM" and "YYYY-MM-DD") into a datetime. '
    "There are no imports and no file or network access. The wall clock is not available: "
    "compute times relative to `as_of`. Only plain functions, loops, comprehensions and lambdas "
    "are allowed (no classes, `with`, `yield`, `global` or `nonlocal`; `except` must name an "
    "exception class), attributes starting with an underscore and names starting with a double "
    f"underscore are not allowed, and the function must finish within {TIMEOUT_S:g} seconds."
)

RETURN_SET = (
    "It must return a Python list of strings in the value format above (an empty list if there "
    "are none)."
)
RETURN_SCALAR = (
    "It must return a Python string in the value format above, or None if the records do not "
    "establish a current value."
)

TASK = (
    "Task: write a Python function `def derive(records, as_of):` that computes the value of the "
    'knowledge item as of `as_of` (a string "YYYY-MM-DD HH:MM") from `records`, the live '
    "records of the declared sources at that time (grouped by source in the order listed above, "
    "oldest `updated` first within a source). {returns} {toolbox} "
    'Reply with JSON only: {{"code": "<the Python source of the function and any helpers>"}}.'
)


TASK_RA = (
    "Task: write a Python function `def derive(records, as_of, prior_reads):` that computes the "
    'value of the knowledge item as of `as_of` (a string "YYYY-MM-DD HH:MM") from `records`, the '
    "live records of the declared sources at that time (grouped by source in the order listed "
    "above, oldest `updated` first within a source). `prior_reads` is the list of object ids that "
    "earlier runs of this function declared as read for this item (an empty list on the first "
    "run). {returns} {reads} {toolbox} "
    'Reply with JSON only: {{"code": "<the Python source of the function and any helpers>"}}.'
)

RETURN_RA_SET = (
    'It must return a dict {"value": <list of strings in the value format above, [] if there '
    'are none; None when the value is withdrawn, see below>, "reads": <list of object ids>}.'
)
RETURN_RA_SCALAR = (
    'It must return a dict {"value": <string in the value format above, or None if the records '
    'do not establish a current value>, "reads": <list of object ids>}.'
)

READS_RA = (
    "`reads` lists the object_id of every record whose fields the function reads to compute the "
    "value. A record that is not listed may not be read: using a field of it is an error. "
    "{locators} The value is withdrawn with the evidence it was computed from: if any id in "
    "`prior_reads` does not belong to a record in `records` (that record was deleted), the "
    "function must return None as the value, and it must not use other records in place of a "
    "missing one."
)


def locators_text(sources: tuple[str, ...]) -> str:
    """The fields a retraction-aware program may read on any record (fixed text, no world data)."""
    per_source = "; ".join(f"{s}: {', '.join(LOCATOR_FIELDS[s])}" for s in sources)
    return (
        "Only these fields can be read on any record without listing it, to find the relevant "
        f"records: {', '.join(METADATA_FIELDS)}, and, per source, {per_source}."
    )


def fields_text(sources: tuple[str, ...]) -> str:
    """The fixed description of the records of the declared sources (no world data)."""
    return "\n".join([COMMON_FIELDS, *(SOURCE_FIELDS[s] for s in sources)])


# prereg v4b block V: the six fixed definitions as the user would state them, with the same sources,
# thresholds, windows and conditions as their `KAPPA_DEFS` statements (fixed in the registration)
USER_WORDS: dict[str, dict[str, str]] = {
    "important_contact": {
        "v1": "Count someone as an important contact if they've been at five or more of my "
        "one-off meetings in the past 90 days. Recurring series don't count.",
        "v2": "Count someone as an important contact if they've been at two or more of my "
        "one-off meetings in the past 30 days. Recurring series don't count.",
    },
    "close_friend": {
        "v1": "Anyone I've had six or more chat messages with in the past 30 days, theirs and "
        "mine together, is a close friend.",
        "v2": "Anyone I've had four or more chat messages with in the past 14 days, theirs and "
        "mine together, is a close friend.",
    },
    "active_project": {
        "v1": "A project is active as long as it still has a task I haven't finished.",
        "v2": "A project is active if any of its tasks, finished or not, was created or changed "
        "in the past 14 days.",
    },
    "working_hours": {
        "v1": "My working hours are whatever my latest 'Working hours' note says.",
        "v2": "Go by my one-off work meetings in the past 30 days: from the earliest start to "
        "the latest end, widened to whole hours.",
    },
    "vip_sender": {
        "v1": "Treat a contact as a VIP sender if they've emailed me four or more times in the "
        "past 30 days.",
        "v2": "Treat a contact as a VIP sender if they've emailed me three or more times in the "
        "past 30 days, or at least once if I've starred them.",
    },
    "overdue_task": {
        "v1": "A task is overdue if I haven't finished it and its due date is before today.",
        "v2": "Only high-priority tasks count as overdue: not finished, and due before today.",
    },
}


def program_request(
    world: World, slot: Slot, t: datetime, ra: bool = False, user_words: bool = False
) -> dict[str, Any]:
    """Writer request for the item's program. `t` only selects the definition version. `ra`: the
    retraction-aware variant (a different request, so it never collides with the D1s one).
    `user_words` (prereg v4b block V): the definition in effect is stated as in `USER_WORDS`."""
    lines = [
        f"Knowledge item: {slot.description}",
        f"Value format: {VALUE_FORMAT[slot.vtype]}",
    ]
    if slot.kappa_id:
        ver = world.kappa_version(slot.kappa_id, t)
        stmt = world.kappa_defs[slot.kappa_id][ver].statement
        if user_words:
            if slot.kappa_id not in USER_WORDS:
                raise ValueError(f"no user-words statement for {slot.kappa_id} (fixed six only)")
            stmt = USER_WORDS[slot.kappa_id][ver]
        lines.append(f"Definition in effect ({slot.kappa_id} {ver}): {stmt}")
    is_set = slot.vtype.startswith("set_")
    if ra:
        task = TASK_RA.format(
            returns=RETURN_RA_SET if is_set else RETURN_RA_SCALAR,
            reads=READS_RA.format(locators=locators_text(slot.sources)),
            toolbox=TOOLBOX,
        )
    else:
        task = TASK.format(returns=RETURN_SET if is_set else RETURN_SCALAR, toolbox=TOOLBOX)
    lines += [
        "",
        f"Declared sources: {', '.join(slot.sources)}",
        fields_text(slot.sources),
        "",
        task,
    ]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "max_tokens": 1500,
        "schema": PROGRAM_SCHEMA,
        "purpose": "program",
    }


# ---------------------------------------------------------------------------- records
def _source_records(snap: Snapshot, source: str) -> list[dict[str, Any]]:
    objs = sorted(snap.by_source(source), key=lambda o: (o.updated, o.object_id))
    return [
        {
            "object_id": o.object_id,
            "source": o.source,
            "revision": o.revision,
            "updated": o.updated.strftime("%Y-%m-%d %H:%M"),
            **o.payload,
        }
        for o in objs
    ]


def program_records(log: SourceLog, slot: Slot, t: datetime) -> list[dict[str, Any]]:
    """Live objects of the slot's declared sources at t, as dicts (see `fields_text`)."""
    snap = log.snapshot(t)
    return [r for source in slot.sources for r in _source_records(snap, source)]


class _Memo[T]:
    """Bounded memo with single flight: concurrent callers of one key wait for one computation."""

    def __init__(self, size: int):
        self.size = size
        self._futures: OrderedDict[Hashable, Future[T]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable, make: Callable[[], T]) -> T:
        with self._lock:
            fut = self._futures.get(key)
            owner = fut is None
            if fut is None:
                fut = Future()
                self._futures[key] = fut
                if len(self._futures) > self.size:
                    self._futures.popitem(last=False)
        if owner:
            try:
                fut.set_result(make())
            except Exception as exc:
                fut.set_exception(exc)
                with self._lock:
                    self._futures.pop(key, None)
                raise
        return fut.result()


# The email source alone is megabytes, and the three D1s cells of a branch write the same items.
_JSON = _Memo[str](32)
_RUNS = _Memo[tuple[bool, Any, str, float]](8192)
_RUNS_RA = _Memo[tuple[bool, Any, list[str], str, float]](8192)


def _source_json(log: SourceLog, source: str, t: datetime) -> str:
    """Comma-joined JSON of one source's records, built once per (log, t, source)."""
    return _JSON.get(
        (log.uid, t, source),
        lambda: ",".join(json.dumps(r) for r in _source_records(log.snapshot(t), source)),
    )


def records_json(log: SourceLog, slot: Slot, t: datetime) -> str:
    parts = (_source_json(log, source, t) for source in slot.sources)
    return "[" + ",".join(p for p in parts if p) + "]"


# ---------------------------------------------------------------------------- running
def validate_value(vtype: str, value: Any) -> tuple[bool, Any, str]:
    """Type the program output against the item's value type: set_* -> list of str (returned
    de-duplicated and sorted, so that a set item has one byte form); others -> str or None."""
    if vtype.startswith("set_"):
        if isinstance(value, list) and all(isinstance(v, str) for v in value):
            return True, sorted(set(value)), ""
        return False, None, f"type mismatch: {vtype} needs a list of str"
    if value is None or isinstance(value, str):
        return True, value, ""
    return False, None, f"type mismatch: {vtype} needs a str or None"


def _typed(vtype: str, ok: bool, value: Any, error: str) -> tuple[bool, Any, str]:
    return validate_value(vtype, value) if ok else (False, None, error)


def run_program(
    code: str, records: list[dict[str, Any]], as_of: str, vtype: str
) -> tuple[bool, Any, str]:
    """Run a program on explicit records: (ok, value, error). Errors, timeouts and values of the
    wrong type are not ok."""
    return _typed(vtype, *run_source(code, json.dumps(records), as_of))


def extract_code(res: LLMResult) -> str | None:
    """The `code` string of a program response, without a markdown fence if the model added one."""
    data = res.json()
    if not isinstance(data, dict) or not isinstance(data.get("code"), str):
        return None
    code: str = data["code"]
    fenced = re.fullmatch(r"\s*```[A-Za-z0-9_]*\n(.*?)\n?```\s*", code, re.S)
    return fenced.group(1) if fenced else code


@dataclass
class ProgramOutcome:
    ok: bool
    value: Any
    error: str
    key: str  # cache key of the program request
    prompt_tokens: int
    completion_tokens: int
    run_s: float  # wall time of the sandboxed run
    reads: list[str] | None = None  # retraction-aware runs: the ids this run declared
    lineage: list[str] | None = None  # ... and all ids any run of the item declared so far
    lineage_ok: bool | None = None  # ... False: a prior read is gone and the value is not None


def _run_cached(code: str, log: SourceLog, slot: Slot, t: datetime) -> tuple[bool, Any, str, float]:
    """Sandboxed run of `code` on the records live at t: (ok, raw value, error, wall seconds).
    A run is a pure function of the program and the records, so cells that share the program and
    the log at t share the run (and its recorded cost)."""

    def make() -> tuple[bool, Any, str, float]:
        start = time.perf_counter()
        ok, value, error = run_source(code, records_json(log, slot, t), f"{t:%Y-%m-%d %H:%M}")
        return ok, value, error, time.perf_counter() - start

    return _RUNS.get((sha256_text(code), log.uid, t, slot.sources), make)


def synthesise(
    llm: LLMClient,
    world: World,
    log: SourceLog,
    slot: Slot,
    t: datetime,
    user_words: bool = False,
) -> ProgramOutcome:
    """Ask the writer for the item's program and run it on the records live at t. `user_words`:
    see `program_request`."""
    res = llm.complete(**program_request(world, slot, t, user_words=user_words))
    usage = (res.prompt_tokens, res.completion_tokens)
    code = extract_code(res)
    if code is None:
        return ProgramOutcome(False, None, "no code in the response", res.key, *usage, 0.0)
    ok, value, error, run_s = _run_cached(code, log, slot, t)
    return ProgramOutcome(*_typed(slot.vtype, ok, value, error), res.key, *usage, run_s)


def _run_cached_ra(
    code: str, log: SourceLog, slot: Slot, t: datetime, prior: tuple[str, ...]
) -> tuple[bool, Any, list[str], str, float]:
    """`_run_cached` for retraction-aware programs, which also depend on the item's prior reads."""

    def make() -> tuple[bool, Any, list[str], str, float]:
        start = time.perf_counter()
        ok, value, reads, error = run_source_ra(
            code, records_json(log, slot, t), f"{t:%Y-%m-%d %H:%M}", list(prior)
        )
        return ok, value, reads, error, time.perf_counter() - start

    return _RUNS_RA.get((sha256_text(code), log.uid, t, slot.sources, prior), make)


def synthesise_ra(
    llm: LLMClient, world: World, log: SourceLog, slot: Slot, t: datetime, prior: list[str]
) -> ProgramOutcome:
    """`synthesise` for the retraction-aware program of the item, which gets the ids earlier runs
    declared (`prior`, the item's lineage)."""
    res = llm.complete(**program_request(world, slot, t, ra=True))
    usage = (res.prompt_tokens, res.completion_tokens)
    code = extract_code(res)
    if code is None:
        error = "no code in the response"
        return ProgramOutcome(False, None, error, res.key, *usage, 0.0, [], sorted(prior), None)
    ok, raw, reads, error, run_s = _run_cached_ra(code, log, slot, t, tuple(prior))
    # a withdrawn value is None for every type, also for the set types that have [] for "none"
    ok, value, error = (
        (True, None, "") if ok and raw is None else _typed(slot.vtype, ok, raw, error)
    )
    lineage = sorted({*prior, *reads})  # a failed run declares nothing and keeps the lineage
    lineage_ok = None
    if ok:
        live = log.snapshot(t)
        gone = any(live.get(oid) is None for oid in prior)
        lineage_ok = not (gone and value is not None)
    return ProgramOutcome(ok, value, error, res.key, *usage, run_s, reads, lineage, lineage_ok)


def program_check(
    vtype: str, writer_ok: bool, writer_value: Any, prog: ProgramOutcome
) -> CheckResult:
    """Definition items: the writer's value passes when it equals the writer's own program output;
    an unreadable writer output or a failing program does not pass."""
    if not prog.ok:
        return CheckResult(False, f"program failed: {prog.error}")
    if not writer_ok:
        return CheckResult(False, "writer output unreadable")
    if values_match(vtype, writer_value, prog.value):
        return CheckResult(True, "matches the writer's program")
    return CheckResult(False, "writer value differs from its program's output")

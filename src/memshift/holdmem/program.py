"""Writer-authored executable derivations for the D1s cell.

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

Program sets from other writers send the same request text through another
client: another model, a sampling temperature (none for a Claude model, see
`experiments.e1.program_extra_body`), and a sample index that the client's body carries as
`memshift_sample` (part of the cache key, removed before the HTTP request), so each sample has its
own cache entry.

Execution feedback (the D1sfb cells; `synthesise_feedback`): the program
request is D1s's, and a program that fails on the records of a write (no code in the reply, syntax
error, no `derive`, sandbox violation, exception, timeout, or a value `validate_value` rejects) is
returned to the writer: the program request's messages, the failed reply exactly as the client
returned it, and a fixed user turn (`FEEDBACK`) with the interpreter's message, or the type message
and a repr of the value, cut to `FEEDBACK_CHARS` characters; same schema and output limit. At most
`FEEDBACK_ROUNDS` such requests per write, each showing only the reply that failed last. A feedback
request holds world data only where the interpreter's message or the value does (a KeyError naming a
key, a value of the wrong type); each write records whether it did (`world_data`). A corrected
program is kept for the item and its program request and runs first at the item's later writes,
where a new failure is fed back again; a request that is new for the server is only ever a feedback
request.
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
from typing import Any, Protocol

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


# The six fixed definitions as the user would state them, with the same sources, thresholds,
# windows and conditions as their `KAPPA_DEFS` statements
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
    `user_words`: the definition in effect is stated as in `USER_WORDS`."""
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
    feedback: dict[str, Any] | None = None  # D1sfb: what the feedback loop did at this write


class RecordView(Protocol):
    """Another form of the live records a program reads (fields extracted from prose,
    `holdmem.extract.RecordExtractor`): the JSON text of the record list, and a tag that
    keeps runs on it apart from runs on the store's records."""

    tag: str

    def records_json(self, log: SourceLog, slot: Slot, t: datetime) -> str: ...


def _run_cached(
    code: str, log: SourceLog, slot: Slot, t: datetime, view: RecordView | None = None
) -> tuple[bool, Any, str, float]:
    """Sandboxed run of `code` on the records live at t: (ok, raw value, error, wall seconds).
    A run is a pure function of the program and the records, so cells that share the program and
    the log at t share the run (and its recorded cost). With `view`, the program reads the view's
    records (built before the clock starts: the view may send model requests)."""

    def make() -> tuple[bool, Any, str, float]:
        data = None if view is None else view.records_json(log, slot, t)
        start = time.perf_counter()
        records = records_json(log, slot, t) if data is None else data
        ok, value, error = run_source(code, records, f"{t:%Y-%m-%d %H:%M}")
        return ok, value, error, time.perf_counter() - start

    key: tuple[Hashable, ...] = (sha256_text(code), log.uid, t, slot.sources)
    return _RUNS.get(key if view is None else (*key, view.tag), make)


def synthesise(
    llm: LLMClient,
    world: World,
    log: SourceLog,
    slot: Slot,
    t: datetime,
    user_words: bool = False,
    view: RecordView | None = None,
) -> ProgramOutcome:
    """Ask the writer for the item's program and run it on the records live at t. `user_words`:
    see `program_request`. `view`: the program reads these records instead (the request is the
    same)."""
    res = llm.complete(**program_request(world, slot, t, user_words=user_words))
    usage = (res.prompt_tokens, res.completion_tokens)
    code = extract_code(res)
    if code is None:
        return ProgramOutcome(False, None, "no code in the response", res.key, *usage, 0.0)
    ok, value, error, run_s = _run_cached(code, log, slot, t, view)
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


# ---------------------------------------------------------------------------- execution feedback
# Execution feedback (the D1sfb cells): a program that fails on the records of a write is sent
# back to the writer with the interpreter's own message. The conversation of one feedback request is
# the program request's messages, the failed reply as the client returned it, and FEEDBACK. Nothing
# in it comes from the gold or from an evaluator: only the failure the restricted interpreter or
# `validate_value` reported, cut to FEEDBACK_CHARS characters.
FEEDBACK_ROUNDS = 2
FEEDBACK_CHARS = 600
NO_CODE = "no code in the response"
FEEDBACK = (
    "Your function failed in the restricted interpreter on the live records of the declared "
    "sources: {failure}\n"
    "Write a corrected function. "
    'Reply with JSON only, in the same format: {{"code": "<the Python source of the function and '
    'any helpers>"}}.'
)
FAILURE_KINDS = ("no_code", "syntax", "no_derive", "sandbox", "timeout", "exception", "type")


def failure_kind(error: str) -> str:
    """The class of a failed program run, from its error message (`sandbox.check_source`,
    `sandbox._run`, `validate_value`)."""
    if error == NO_CODE:
        return "no_code"
    if error.startswith("syntax error"):
        return "syntax"
    if error.startswith(("no top-level function derive", "derive must take")):
        return "no_derive"
    if error.startswith(("disallowed syntax", "forbidden ", "program too long", "sandbox crashed")):
        return "sandbox"
    if error == "timeout":
        return "timeout"
    if error.startswith("type mismatch"):
        return "type"
    return "exception"


def failure_text(error: str, raw: Any = None) -> str:
    """What the feedback request says failed: the interpreter's message (and, for a value of the
    wrong type, a repr of the value), cut to FEEDBACK_CHARS characters."""
    kind = failure_kind(error)
    if kind == "no_code":
        text = 'the reply is not a JSON object with a "code" string'
    elif kind == "timeout":
        text = f"timeout: the function did not finish within {TIMEOUT_S:g} seconds"
    elif kind == "type" and raw is not None:
        text = f"{error}; it returned {raw!r}"
    else:
        text = error
    return text[:FEEDBACK_CHARS]


def feedback_request(
    request: dict[str, Any], reply: str, error: str, raw: Any = None
) -> dict[str, Any]:
    """The request that returns a failed program to its writer: the program request's messages,
    the failed reply exactly as returned, and FEEDBACK; schema and output limit unchanged."""
    turn = FEEDBACK.format(failure=failure_text(error, raw))
    return {
        **request,
        "messages": [
            *request["messages"],
            {"role": "assistant", "content": reply},
            {"role": "user", "content": turn},
        ],
        "purpose": "program_feedback",
    }


def _world_values(records: list[dict[str, Any]], prompt: str) -> set[str]:
    """The string field values of the records (3 characters or more) that the request does not
    already state: what counts as world data in a feedback text."""
    out: set[str] = set()
    for r in records:
        for v in r.values():
            items = v if isinstance(v, list) else [v]
            out |= {s for s in items if isinstance(s, str) and len(s) >= 3 and s not in prompt}
    return out


def _attempt(
    reply: str, log: SourceLog, slot: Slot, t: datetime
) -> tuple[bool, Any, str, Any, float]:
    """Run the program of one reply on the records live at t: (ok, typed value, error, raw value,
    wall seconds)."""
    code = extract_code(LLMResult(reply, 0, 0, 0.0, True, "", ""))
    if code is None:
        return False, None, NO_CODE, None, 0.0
    ok, raw, error, run_s = _run_cached(code, log, slot, t)
    return (*_typed(slot.vtype, ok, raw, error), raw if ok else None, run_s)


def synthesise_feedback(
    llm: LLMClient,
    world: World,
    log: SourceLog,
    slot: Slot,
    t: datetime,
    kept: dict[tuple[str, str], str],
    user_words: bool = False,
) -> ProgramOutcome:
    """`synthesise` with execution feedback (the D1sfb cells). The program request is D1s's (a
    cache hit of the D1s runs). The program that runs first is the one kept for the item and that
    request (`kept`, the reply of the last program that ran without failure, updated in place), else
    D1s's own. While it fails and fewer than FEEDBACK_ROUNDS feedback requests were sent at this
    write, the failed reply goes back to the writer (`feedback_request`) and the reply's program
    runs on the same records. The first program that runs without failure is used and kept; if all
    fail, the outcome is a failure, as D1s's on a failing program. `feedback` records the write:
    which program ran first ("program" or "kept"), whether D1s's own program ran without failure
    and its failure kind, the feedback requests sent, the failure kind of each failed run, the
    final status, and per feedback request whether its text holds a string value of the records
    (`world_data`)."""
    request = program_request(world, slot, t, user_words=user_words)
    res = llm.complete(**request)
    prompt_tokens, completion_tokens = res.prompt_tokens, res.completion_tokens
    ok, value, error, raw, run_s = _attempt(res.text, log, slot, t)
    d1s_ok, d1s_kind = ok, None if ok else failure_kind(error)
    reply = kept.get((slot.slot_id, res.key))
    start = "kept" if reply is not None else "program"
    if reply is None:
        reply = res.text
    else:
        ok, value, error, raw, more = _attempt(reply, log, slot, t)
        run_s += more
    kinds: list[str] = []
    world_data: list[bool] = []
    values: set[str] | None = None
    while not ok and len(world_data) < FEEDBACK_ROUNDS:
        kinds.append(failure_kind(error))
        if values is None:
            prompt = request["messages"][-1]["content"]
            values = _world_values(program_records(log, slot, t), prompt)
        failure = failure_text(error, raw)
        world_data.append(any(v in failure for v in values))
        fb = llm.complete(**feedback_request(request, reply, error, raw))
        prompt_tokens += fb.prompt_tokens
        completion_tokens += fb.completion_tokens
        reply = fb.text
        ok, value, error, raw, more = _attempt(reply, log, slot, t)
        run_s += more
    if not ok:
        kinds.append(failure_kind(error))
    elif world_data:  # a corrected program: kept for the item's later writes
        kept[(slot.slot_id, res.key)] = reply
    note = {
        "start": start,
        "d1s_ok": d1s_ok,
        "d1s_kind": d1s_kind,
        "rounds": len(world_data),
        "kinds": kinds,
        "ok": ok,
        "world_data": world_data,
    }
    return ProgramOutcome(
        ok,
        value,
        error,
        res.key,
        prompt_tokens,
        completion_tokens,
        run_s,
        feedback=note,
    )


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

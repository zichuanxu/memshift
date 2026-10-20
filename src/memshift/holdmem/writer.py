"""HoldMem write path, step 1: the writer model theta_w turns in-scope source records
into one knowledge-item string s_k with cited object ids and quoted source text.

The writer returns JSON {value, statement, citations}. s_k is rendered from the statement and the
citations only; the typed `value` is kept solely for the D=1 sidecar (`det.parse`). A D=0 cell
therefore stores exactly the same s_k bytes as a D=1 cell, without the parse.

The records of a request are the BM25-selected in-scope records (`records_for`, 6,000 tokens),
except for the D0full control, whose definition items are re-derived from every live record of
the item's declared sources (`full_records`), and the D0typed control, whose
definition items are re-derived from the typed records a D_m program reads (`typed_records`: the
records of `holdmem.program.program_records` as one columnar JSON table per declared source,
without the rendered text and the bodies of e-mail and chat messages, never truncated). Its
request is `writer_request` with the record block replaced by the typed block (`TypedInput.text`);
everything else in the request is the same.

The definition-item writer (another model, which reasons before it answers)
sends the same request with its own output limit (`max_tokens`); a reply cut off by that limit is
unreadable, as for a thinking request (`derive(reasoning=True)`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memshift.gen.model import Slot, World
from memshift.gen.state import SourceLog
from memshift.llm import LLMClient, LLMResult
from memshift.retrieval import Doc, cached_index, pack
from memshift.textutil import approx_tokens, truncate_tokens

WRITER_BUDGET = 6000
WRITER_DOC_CAP = 600
# D0full: the writer of a definition item gets every live record of the item's declared sources, not
# a BM25 selection of 6,000 tokens; above FULL_BUDGET tokens (`textutil.approx_tokens` over the
# rendered records, as every budget here) the largest source is cut to its most recent records (see
# `full_records`). A cap of 28,000 tokens would suit a 32,768-token context, but
# `approx_tokens` undercounts the serving tokenizer on dates and digits: on the 3,789 writer
# requests of eight replication worlds (both branches) with a recorded `prompt_tokens`, the server
# counted 1.11-1.15 tokens per approximate token for prompts of calendar or message records (up to
# 1.19 for tasks, 0.67-0.72 for e-mail). 28,000 approximate tokens of messages are then
# 31,000-32,200 real ones, which with the request's own text (about 500 tokens) and its 700 output
# tokens does not always fit. 25,000 leave about 1,800 tokens of room at the highest ratio measured.
FULL_BUDGET = 25_000
# D0think: the request of a thinking writer has room for the reasoning (6,000 tokens of records and
# 8,192 of output fit the 16k context of the smaller serving configuration) and carries no JSON
# schema (a grammar that has to open with `{` leaves the model no room to think on a server without
# a reasoning parser); it names the answer's keys in the prompt instead, and the answer is parsed
# from the reply as every other writer output is. THINK_MAX_TOKENS is the default output limit; the
# E1 flag `--think-max-tokens` sets another one (32,768 in the long-thinking runs, on a
# 40,960-token server context), which only changes `max_tokens` in the request.
THINK_MAX_TOKENS = 8192
# finish reasons of a reply cut off by its output limit (OpenAI-compatible, Anthropic)
CUT_OFF = ("length", "max_tokens")
THINK_FORMAT = (
    'JSON format: {"value": ..., "statement": "...", "citations": [{"object_id": "...", '
    '"revision": 1, "quote": "..."}]}'
)

WRITER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "value": {
            "anyOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
                {"type": "null"},
            ]
        },
        "statement": {"type": "string"},
        "citations": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "object_id": {"type": "string"},
                    "revision": {"type": "integer"},
                    "quote": {"type": "string"},
                },
                "required": ["object_id", "revision", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["value", "statement", "citations"],
    "additionalProperties": False,
}

SYSTEM = (
    "You maintain a personal knowledge store for one person (called 'the user'). You write one "
    "knowledge item at a time from the user's source records, citing the records you used."
)

TASK = (
    "Task: determine the value of the knowledge item as of the date above, relying on the most "
    "recent applicable evidence. `value` must be only the value in the format given above (not a "
    "sentence); if the records do not establish a current value, set value to null. Write "
    "`statement` as one or two self-contained sentences stating the value (name the item and the "
    "date). Give up to 6 citations, each with the record id, revision and a short quote (at most "
    "25 words) copied exactly from that record. Reply with JSON only."
)

VALUE_FORMAT = {
    "org": 'the company name only, e.g. "Acme Corp"',
    "person": "the person's full name only",
    "city": "the city name only",
    "tz": 'an IANA time zone name only, e.g. "Europe/Paris"',
    "diet": "one of: omnivore, vegetarian, vegan, pescatarian, gluten-free",
    "commute": "one of: bike, train, car, bus, walk",
    "weekly": 'weekday and 24-hour start time, e.g. "Tuesday 14:30"',
    "datetime": '"YYYY-MM-DD HH:MM" (24-hour)',
    "date": '"YYYY-MM-DD"',
    "timerange": '"HH:MM-HH:MM" (24-hour)',
    "set_person": "a JSON array of full names ([] if none)",
    "set_project": "a JSON array of project names ([] if none)",
    "set_task": "a JSON array of task titles ([] if none)",
}


@dataclass
class Citation:
    object_id: str
    revision: int
    quote: str


@dataclass
class WriterOutput:
    value: Any
    statement: str
    citations: list[Citation] = field(default_factory=list)
    ok: bool = True  # False when the model output could not be parsed
    readset: list[tuple[str, int]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # D0full / D0typed: what the writer was given (`FullAccess.note`, `TypedInput.note`)
    access: dict[str, Any] | None = None
    finish_reason: str = ""  # of the model reply; empty when no request was sent


def records_for(
    log: SourceLog, slot: Slot, t: datetime, since: datetime | None = None
) -> list[tuple[Doc, str]]:
    """In-scope records for a slot at t (declared sources). `since` restricts to the delta."""

    def docs() -> list[Doc]:
        objs = log.snapshot(t).by_source(*slot.sources)
        if since is not None:
            objs = [o for o in objs if o.updated > since]
        return [Doc(o.object_id, o.text, o.updated, o.revision) for o in objs]

    index = cached_index(("records", log.uid, t, slot.sources, since), docs)
    ranked = [d for d, s in index.rank(f"{slot.description} {slot.query}") if s > 0]
    return pack(ranked, WRITER_BUDGET, WRITER_DOC_CAP)


def record_block(d: Doc, text: str) -> str:
    """One record as the writer request shows it."""
    return f"<record id={d.doc_id} rev={d.revision}>\n{text}\n</record>"


@dataclass(frozen=True)
class FullAccess:
    """The records a D0full writer is given for one definition item (`full_records`)."""

    records: list[tuple[Doc, str]]  # oldest first
    dropped: int  # in-scope records left out
    capped: int  # in-scope records (given or dropped) whose text the per-document cap cut
    tokens: int  # `approx_tokens` of the rendered records that are given
    truncation: str  # "none" | "source" (the largest source cut) | "newest" (newest-first fallback)
    by_source: dict[
        str, dict[str, int]
    ]  # declared source -> {"records": given, "dropped": left out}

    def note(self) -> dict[str, Any]:
        """The record kept in the item's notes, one entry per re-derivation."""
        return {
            "records": len(self.records),
            "dropped": self.dropped,
            "capped": self.capped,
            "tokens": self.tokens,
            "truncation": self.truncation,
            "by_source": {s: dict(v) for s, v in sorted(self.by_source.items())},
        }


def _newest_run(costs: list[tuple[int, int]], room: int) -> set[int]:
    """Of (index, cost) pairs in time order, the indices of the newest run that fits `room`: the
    first record (from the newest) that does not fit ends the run."""
    kept: set[int] = set()
    used = 0
    for i, cost in reversed(costs):
        if used + cost > room:
            break
        kept.add(i)
        used += cost
    return kept


def full_records(log: SourceLog, slot: Slot, t: datetime, budget: int | None = None) -> FullAccess:
    """Every live record of the slot's declared sources at t, for the D0full writer.

    "In scope" is the writer's own notion (`records_for`; `Slot.sources` is the item's "declared
    scope" in `gen.model`, the scope of the source-epoch key): all records of the declared sources,
    structured fields rendered as the record text, each cut to `WRITER_DOC_CAP` tokens as in every
    writer request. It is not the definition's scope range (`gen.kappa.scope_members`): that range
    applies the definition's own predicate (its time window, a task's status and due date) to the
    structured fields, which for some definitions is the answer's support set, so the writer would
    be handed the filtering step of the derivation, and the control would measure access plus part
    of the execution. The D=1 checker and the D1s programs read the same records (all live records
    of the declared sources) and apply the predicate themselves. No evaluator output, gold or
    definition predicate is read.

    Records are ordered by time (`updated`, then id). The cost of a record is `approx_tokens` of its
    rendered block; `budget` defaults to `FULL_BUDGET`. Truncation is per source:
      - the records fit: every record is given (`truncation` "none");
      - they do not, and every source but the largest (by cost) fits whole: those sources are given
        whole and the largest one is cut to its newest records that fit what is left ("source").
        A cut is always a run of the newest records without gaps: the first record, from the
        newest, that does not fit ends it;
      - the sources other than the largest already exceed the budget: the newest-first rule over
        all records together ("newest"), counted apart.
    Without the per-source rule, the oldest records would go first, and a small source that is
    older than a large one (the contact cards against e-mail) would go with them.
    """
    budget = FULL_BUDGET if budget is None else budget
    objs = sorted(log.snapshot(t).by_source(*slot.sources), key=lambda o: (o.updated, o.object_id))
    rows: list[tuple[str, Doc, str, int]] = []  # (source, record, text, cost) in time order
    capped = 0
    for o in objs:
        d = Doc(o.object_id, o.text, o.updated, o.revision)
        text = truncate_tokens(d.text, WRITER_DOC_CAP)
        capped += text != d.text
        rows.append((o.source, d, text, approx_tokens(record_block(d, text))))
    by_cost: dict[str, int] = {}
    for source, _, _, cost in rows:
        by_cost[source] = by_cost.get(source, 0) + cost
    total = sum(by_cost.values())
    truncation = "none"
    keep = set(range(len(rows)))
    if total > budget:
        largest = max(by_cost, key=lambda s: (by_cost[s], s))
        others = total - by_cost[largest]
        if others <= budget:
            truncation = "source"
            cut = _newest_run(
                [(i, r[3]) for i, r in enumerate(rows) if r[0] == largest], budget - others
            )
            keep = {i for i, r in enumerate(rows) if r[0] != largest} | cut
        else:
            truncation = "newest"
            keep = _newest_run([(i, r[3]) for i, r in enumerate(rows)], budget)
    kept = [r for i, r in enumerate(rows) if i in keep]
    by_source = {s: {"records": 0, "dropped": 0} for s in slot.sources}
    for i, r in enumerate(rows):
        by_source[r[0]]["records" if i in keep else "dropped"] += 1
    return FullAccess(
        [(r[1], r[2]) for r in kept],
        len(rows) - len(kept),
        capped,
        sum(r[3] for r in kept),
        truncation,
        by_source,
    )


# D0typed: the typed records of a definition item, one table per declared source. A
# table names its source once and lists each record as an array of its field values in the fixed
# order of `TYPED_FIELDS` (the record metadata, then the source's fields in the order
# `holdmem.program.SOURCE_FIELDS` describes them); a field a record does not have is null (the
# calendar's four kinds share one table). This is the "ColumnsNoBody" layout whose sizes
# `analysis.ex_round3_review` (sizes.json) measured, with one table per source instead of one per
# calendar kind, so that every record of a source is in one table and the order of
# `program_records` (oldest `updated` first within a source) is kept. The field names are written
# once per table instead of once per record, about a quarter fewer tokens than the dict layout on
# the largest inputs (stage 8: 44,109 against 57,002 for close friends, 47,326 against 61,489 for
# VIP senders, maxima over 20 worlds), and the nulls of the calendar union cost a token each.
# Dropped: `source` (the table's), the rendered `text` of every record and the `body` of e-mail and
# chat messages; the note bodies stay (working_hours v1 reads the latest "Working hours" note).
TYPED_META = ("object_id", "revision", "updated", "cluster")
TYPED_FIELDS: dict[str, tuple[str, ...]] = {
    "contacts": (
        *TYPED_META,
        *("name", "email", "phone", "employer", "city", "relation", "starred"),
    ),
    "calendar": (
        *(*TYPED_META, "kind", "timezone", "title", "weekday", "time", "duration_min"),
        *("attendees", "category", "appt_type", "provider", "start", "end"),
    ),
    "tasks": (*TYPED_META, "title", "project", "due", "status", "priority"),
    "email": (*TYPED_META, "from_name", "from_addr", "to", "subject", "sent"),
    "messages": (*TYPED_META, "conversation_with", "contact", "sender", "sent"),
    "notes": (*TYPED_META, "title", "body"),
}
TYPED_DROPPED: dict[str, tuple[str, ...]] = {"email": ("body",), "messages": ("body",)}
TYPED_HEAD = (
    "Source records, as typed fields: one table per declared source, one JSON object per line "
    '{"source": ..., "fields": [...], "rows": [[...], ...]}, each row one record with its values '
    "in the order of `fields` (null where the record has no such field), oldest `updated` first. "
    "The rendered text of the records and the bodies of e-mail and chat messages are omitted."
)
TYPED_COMMON = (
    "Every table has the fields object_id (str), revision (int), updated (str "
    '"YYYY-MM-DD HH:MM", time of the last change) and cluster (str, grouping id), then the '
    "source-specific fields below."
)


def typed_fields_text(sources: tuple[str, ...]) -> str:
    """`holdmem.program.fields_text` for the typed tables: the common fields without `source` and
    `text`, and the source lines without the omitted bodies (fixed text, no world data)."""
    from memshift.holdmem.program import SOURCE_FIELDS  # program imports this module

    lines = [TYPED_COMMON]
    for s in sources:
        text = SOURCE_FIELDS[s]
        for f in TYPED_DROPPED.get(s, ()):
            cut = text.replace(f"{f} (str), ", "")
            if cut == text:
                raise ValueError(f"{s}: the field description has no {f!r} to omit")
            text = cut
        lines.append(text)
    return "\n".join(lines)


def _compact(o: object) -> str:
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class TypedInput:
    """The typed records a D0typed writer is given for one definition item (`typed_records`)."""

    sources: tuple[str, ...]
    tables: list[dict[str, Any]]  # one per declared source, in declared order
    readset: list[tuple[str, int]]  # (object_id, revision) of every row, in table order

    def block(self) -> str:
        """The JSON of the tables, one compact table per line."""
        return "\n".join(_compact(tab) for tab in self.tables)

    def text(self) -> str:
        """What replaces the record block of the writer request."""
        return "\n".join([TYPED_HEAD, typed_fields_text(self.sources), self.block()])

    def note(self) -> dict[str, Any]:
        """The record kept in the item's notes, one entry per re-derivation: the records given
        per source, the characters and `approx_tokens` of the JSON block (no cap, so nothing is
        dropped)."""
        block = self.block()
        return {
            "records": len(self.readset),
            "by_source": {tab["source"]: len(tab["rows"]) for tab in self.tables},
            "chars": len(block),
            "tokens": approx_tokens(block),
        }


def typed_records(log: SourceLog, slot: Slot, t: datetime) -> TypedInput:
    """Every live record of the slot's declared sources at t, as the D_m program reads them
    (`holdmem.program.program_records`, same records, same order), in the columnar layout of
    `TYPED_FIELDS`. A field outside `TYPED_FIELDS` would be kept after the fixed ones (sorted);
    the generated sources have none. No evaluator output, gold or definition predicate is read."""
    from memshift.holdmem.program import program_records  # program imports this module

    rows: dict[str, list[dict[str, Any]]] = {s: [] for s in slot.sources}
    for r in program_records(log, slot, t):
        rows[r["source"]].append(r)
    tables = []
    readset = []
    for s in slot.sources:
        drop = {"source", "text", *TYPED_DROPPED.get(s, ())}
        extra = sorted({k for r in rows[s] for k in r} - drop - set(TYPED_FIELDS[s]))
        fields = [*TYPED_FIELDS[s], *extra]
        table_rows = [[r.get(f) for f in fields] for r in rows[s]]
        tables.append({"source": s, "fields": fields, "rows": table_rows})
        readset += [(str(r["object_id"]), int(r["revision"])) for r in rows[s]]
    return TypedInput(tuple(slot.sources), tables, readset)


def writer_request(
    world: World,
    slot: Slot,
    t: datetime,
    records: list[tuple[Doc, str]],
    delta_since: datetime | None = None,
    think: bool = False,
    max_tokens: int | None = None,
    typed: TypedInput | None = None,
) -> dict[str, Any]:
    """`max_tokens`: the output limit; by default THINK_MAX_TOKENS for a thinking request, else 700.
    The request text does not depend on it. `typed` (D0typed): the typed block replaces the record
    block (`records` is then empty); every other line is the same."""
    lines = [
        f"Knowledge item: {slot.description}",
        f"Value format: {VALUE_FORMAT[slot.vtype]}",
        f"As of: {t:%Y-%m-%d %H:%M}",
    ]
    if slot.kappa_id:
        ver = world.kappa_version(slot.kappa_id, t)
        stmt = world.kappa_defs[slot.kappa_id][ver].statement
        lines.append(f"Definition in effect ({slot.kappa_id} {ver}): {stmt}")
    if delta_since is not None:
        lines.append(
            f"The records below are only those added or changed since {delta_since:%Y-%m-%d %H:%M}."
        )
    if typed is not None:
        lines += ["", typed.text()]
    else:
        lines += ["", "Source records (each starts with its id and revision):"]
        lines += [record_block(d, text) for d, text in records]
        if not records:
            lines.append("(no records)")
    lines += ["", TASK]
    if think:
        lines.append(THINK_FORMAT)
    if max_tokens is None:
        max_tokens = THINK_MAX_TOKENS if think else 700
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "max_tokens": max_tokens,
        "schema": None if think else WRITER_SCHEMA,
        "purpose": "write",
    }


def parse_writer(res: LLMResult, records: list[tuple[Doc, str]]) -> WriterOutput:
    data = res.json()
    readset = [(d.doc_id, d.revision) for d, _ in records]
    usage = (res.prompt_tokens, res.completion_tokens)
    if not isinstance(data, dict) or "statement" not in data:
        return WriterOutput(None, "", [], False, readset, *usage)
    cits = []
    for c in data.get("citations") or []:
        if isinstance(c, dict) and "object_id" in c:
            try:
                rev = int(c.get("revision", 0))
            except (TypeError, ValueError):
                rev = 0
            cits.append(Citation(str(c["object_id"]), rev, str(c.get("quote", ""))[:300]))
    statement = str(data.get("statement", "")).strip()
    return WriterOutput(data.get("value"), statement, cits, True, readset, *usage)


def render_sk(slot: Slot, t: datetime, out: WriterOutput) -> str:
    head = f"[{slot.description} | written {t:%Y-%m-%d}]"
    if not out.ok:
        return f"{head} (writer output unreadable)"
    ev = "; ".join(f'{c.object_id} r{c.revision}: "{c.quote}"' for c in out.citations[:6])
    return f"{head} {out.statement}" + (f" Evidence: {ev}" if ev else "")


def derive(
    llm: LLMClient,
    world: World,
    log: SourceLog,
    slot: Slot,
    t: datetime,
    since: datetime | None = None,
    think: bool = False,
    full: bool = False,
    max_tokens: int | None = None,
    reasoning: bool = False,
    typed: bool = False,
) -> WriterOutput:
    """`full`: every in-scope record (D0full, `full_records`) instead of the BM25 selection.
    `typed`: the typed records instead (D0typed, `typed_records`); the read set is every record
    given and `access` its `TypedInput.note`. `max_tokens`: the output limit (`writer_request`).
    `reasoning`: the writer reasons before it answers (the definition-item writer), so, as for a
    thinking request, a reply cut off by the output limit is unreadable."""
    if typed and (full or since is not None):
        raise ValueError("typed input is a full re-derivation: not with full or a delta")
    given = typed_records(log, slot, t) if typed else None
    access = full_records(log, slot, t) if full else None
    if given is not None:
        records: list[tuple[Doc, str]] = []
    else:
        records = access.records if access is not None else records_for(log, slot, t, since)
    if since is not None and not records:
        # append-only delta with no new in-scope records: nothing to derive, no model call
        return WriterOutput(None, "", [], True, [], 0, 0)
    res = llm.complete(**writer_request(world, slot, t, records, since, think, max_tokens, given))
    if (think or reasoning) and res.finish_reason in CUT_OFF:
        # the reasoning used the whole budget: whatever braces are left in it are not an answer
        readset = [(d.doc_id, d.revision) for d, _ in records]
        usage = (res.prompt_tokens, res.completion_tokens)
        out = WriterOutput(None, "", [], False, readset, *usage, finish_reason=res.finish_reason)
    else:
        out = parse_writer(res, records)
        out.finish_reason = res.finish_reason
        if access is not None:
            out.access = access.note()
    if given is not None:
        out.readset, out.access = list(given.readset), given.note()
    return out

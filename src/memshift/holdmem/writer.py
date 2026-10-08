"""HoldMem write path, step 1 (plan §5.2): the writer model theta_w turns in-scope source records
into one knowledge-item string s_k with cited object ids and quoted source text.

The writer returns JSON {value, statement, citations}. s_k is rendered from the statement and the
citations only; the typed `value` is kept solely for the D=1 sidecar (`det.parse`). A D=0 cell
therefore stores exactly the same s_k bytes as a D=1 cell, without the parse.

The records of a request are the BM25-selected in-scope records (`records_for`, 6,000 tokens),
except for the D0full control (prereg v3 §6), whose definition items are re-derived from every live
record of the item's declared sources (`full_records`).
"""

from __future__ import annotations

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
# `full_records`). The registered cap is 28,000 tokens for a 32,768-token context, but
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
# from the reply as every other writer output is.
THINK_MAX_TOKENS = 8192
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
    access: dict[str, Any] | None = None  # D0full: what the writer was given (`FullAccess.note`)


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


def writer_request(
    world: World,
    slot: Slot,
    t: datetime,
    records: list[tuple[Doc, str]],
    delta_since: datetime | None = None,
    think: bool = False,
) -> dict[str, Any]:
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
    lines += ["", "Source records (each starts with its id and revision):"]
    lines += [record_block(d, text) for d, text in records]
    if not records:
        lines.append("(no records)")
    lines += ["", TASK]
    if think:
        lines.append(THINK_FORMAT)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": "\n".join(lines)},
        ],
        "max_tokens": THINK_MAX_TOKENS if think else 700,
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
) -> WriterOutput:
    """`full`: every in-scope record (D0full, `full_records`) instead of the BM25 selection."""
    access = full_records(log, slot, t) if full else None
    records = access.records if access is not None else records_for(log, slot, t, since)
    if since is not None and not records:
        # append-only delta with no new in-scope records: nothing to derive, no model call
        return WriterOutput(None, "", [], True, [], 0, 0)
    res = llm.complete(**writer_request(world, slot, t, records, since, think))
    if think and res.finish_reason == "length":
        # the reasoning used the whole budget: whatever braces are left in it are not an answer
        readset = [(d.doc_id, d.revision) for d, _ in records]
        return WriterOutput(None, "", [], False, readset, res.prompt_tokens, res.completion_tokens)
    out = parse_writer(res, records)
    if access is not None:
        out.access = access.note()
    return out

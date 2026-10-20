"""Records rewritten as prose and their fields extracted back (cells D1sxB0 and D1sxB1_source).

The D_m programs (`holdmem.program`) read the generator's typed fields. A memory that holds text
would first have to extract them, and the benchmark's record text is a per-field template
(`gen.render`) from which extraction is trivial. Here every record revision a D1sx program reads
is first rewritten as plain prose from its typed fields, with every value kept verbatim, and the
writer then extracts the typed fields back from the prose; the D1sx programs run on the extracted
records. The writer, the checker, the keys, the stores and the reader of every cell see the
original records, so only the programs' input changes.

Rewrite (`rewrite_request`). One request per record revision to the rewriting model (Qwen3.5-9B
whatever the writer, `experiments.e1.REWRITE_MODEL`; temperature 0, the served default body,
reasoning off). It shows the kind of record and its values (`facts`), each as the exact string that
must appear in the prose: names, addresses, phone numbers, titles, places, dates and times in the
record's own format, the attendees one by one, the subject and the body word for word; durations
as "<n> minutes"; enumerations and booleans as the fixed words of `FIXED` ("starred" / "not
starred", "high priority", "done", "recurring", "one-off", ...), which are marked as fixed wording.
The reply is JSON {"text": ...}: one to three plain sentences, no JSON, lists or "field: value"
lines. Metadata (object_id, source, revision, updated, cluster) is not rewritten: the store keeps
it, and the typed `updated` of tasks and notes is that metadata.

Check (`check_text`, mechanical). Every value is contained in the text, exactly and
case-sensitively after runs of whitespace are collapsed on both sides; a fixed word occurs as a
word, case-insensitively ("starred" not preceded by "not "); a duration occurs as its number
followed by "min" ("45 minutes", "45-minute"); and outside the verbatim subject and body the text
neither opens with a JSON bracket nor has two or more "field: value" lines or list lines. A failing
text (or a reply that is not a JSON object with a non-empty "text", or was cut off) is requested
once more with the problems listed. The second text is kept even if it fails, and counted
(`missed`); it is never replaced by a template, which would make extraction trivial again. When
neither attempt gives a text, the record has no prose, no extraction request is sent and its fields
are null (`no_text`).

Extraction (`extract_request`). One request per text to the writer (the client of the writer
requests, at their settings): the source's field description exactly as the program requests show
it (`program.SOURCE_FIELDS`, without the metadata and `text` fields), the text, and a JSON schema
per source (`extraction_schema`) with the types of `SOURCE_FIELDS`: every field nullable (a text
may lack a value the rewrite dropped), enumerations restricted to their values, calendar records
a union of four objects, one per kind: `kind` (that one value) first, then the fields of that kind
in the order of the field description. (A flat object with the fields of all four kinds makes the
writer write null for the fields of a kind that come after another kind's null fields, so that
no meeting keeps its start or end.) A reply
that was cut off or is not a JSON object gives null fields (`extract_failed`); a value of the wrong
JSON type becomes null. The extracted record is the store's metadata, the extracted fields and
`text` = the prose.

Background. Newsletter digests (`gen.llm_render.is_background`: about 6,400 tokens of filler each,
308 per world, from senders who are not contacts) are not rewritten: their typed records go to the
programs unchanged and are counted (`background`), as in the rewritten worlds. Rewriting
them verbatim would mean about two million generated tokens per world in each of the two steps, for
records that carry no fact any item depends on.

Keys and cost. Both requests depend on the source and the typed fields only (no object id,
revision, time, branch or cell), so a revision the branches share (everything before the fork, and
any identical content) costs one rewrite and one extraction per world, and identical content shares
one request. A `RecordExtractor` serves the cells of one world: it memoises each record by that
content with a single flight (concurrent callers of one record wait for one computation) and works
through the records a program run needs on a shared thread pool before the run. Every request goes
through the response cache. A failed model call raises; nothing is kept silently.

Accounting (experimenter-side, never served). A record's `errors` are the fields whose extracted
value differs from the typed value after `scoring.normalize.norm_str` (attendees as sets, numbers
and booleans by value; a calendar record also by `kind`), its `inexact` fields those that differ
after runs of whitespace are collapsed (attendees as sorted lists). Each D1sx program run records
its input (`input_note`: records, background, missed rewrites, failed extractions, records with an
error and the erroneous fields) in the item history; the run writes one row per (world, branch,
record revision) that a D1sx program read (`rows`, extractions.jsonl) and a per-source report per
branch (`report`, world_summaries.jsonl).
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from memshift.gen.llm_render import DIGEST_CLUSTER_PREFIX, clean_body
from memshift.gen.model import Slot
from memshift.gen.state import ObjState, SourceLog
from memshift.holdmem.program import SOURCE_FIELDS, _Memo
from memshift.llm import LLMResult
from memshift.scoring.normalize import norm_str
from memshift.textutil import approx_tokens, sha256_json

MAX_NEW_TOKENS = 4096
RETRY_MARK = "Your previous text was rejected"
METADATA_KEYS = ("cluster", "updated", "text")  # payload keys the store keeps, not rewritten

# field types of `program.SOURCE_FIELDS`, in its order
STR, INT, BOOL, LIST, ENUM = "str", "int", "bool", "list", "enum"
FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "contacts": (
        ("name", STR),
        ("email", STR),
        ("phone", STR),
        ("employer", STR),
        ("city", STR),
        ("relation", ENUM),
        ("starred", BOOL),
    ),
    "tasks": (("title", STR), ("project", STR), ("due", STR), ("status", ENUM), ("priority", ENUM)),
    "email": (
        ("from_name", STR),
        ("from_addr", STR),
        ("to", STR),
        ("subject", STR),
        ("body", STR),
        ("sent", STR),
    ),
    "messages": (
        ("conversation_with", STR),
        ("contact", STR),
        ("sender", STR),
        ("body", STR),
        ("sent", STR),
    ),
    "notes": (("title", STR), ("body", STR)),
}
CALENDAR_KINDS: dict[str, tuple[tuple[str, str], ...]] = {
    "settings": (("timezone", STR),),
    "series": (
        ("title", STR),
        ("weekday", STR),
        ("time", STR),
        ("duration_min", INT),
        ("attendees", LIST),
        ("category", ENUM),
    ),
    "appointment": (
        ("appt_type", STR),
        ("provider", STR),
        ("title", STR),
        ("start", STR),
        ("duration_min", INT),
    ),
    "meeting": (
        ("title", STR),
        ("start", STR),
        ("end", STR),
        ("attendees", LIST),
        ("category", ENUM),
    ),
}
ENUMS: dict[str, tuple[str, ...]] = {
    "relation": ("colleague", "friend", "family", "other"),
    "status": ("open", "done"),
    "priority": ("low", "medium", "high"),
    "category": ("work", "personal", "social"),  # series: work | personal; meeting: work | social
}
SOURCES = ("contacts", "calendar", "tasks", "email", "messages", "notes")

# the words an enumeration or boolean value is written with in the prose
FIXED: dict[str, dict[Any, str]] = {
    "kind": {
        "settings": "calendar settings",
        "series": "recurring",
        "appointment": "appointment",
        "meeting": "one-off",
    },
    "relation": {
        "colleague": "colleague",
        "friend": "friend",
        "family": "family member",
        "other": "acquaintance",
    },
    "starred": {True: "starred", False: "not starred"},
    "status": {"open": "open", "done": "done"},
    "priority": {"low": "low priority", "medium": "medium priority", "high": "high priority"},
    "category": {"work": "work", "personal": "personal", "social": "social"},
}
VERBATIM = ("subject", "body")  # values the prose carries word for word, possibly over lines

RECORD_KIND = {
    "contacts": "a contact card from the user's address book",
    "calendar": "an entry in the user's calendar",
    "tasks": "a task from the user's task list",
    "email": "an email in the user's mailbox",
    "messages": "a message from one of the user's chat conversations",
    "notes": "one of the user's notes",
}
LABELS = {
    "kind": "kind of entry",
    "name": "name",
    "email": "email address",
    "phone": "phone number",
    "employer": "company",
    "city": "city",
    "relation": "relationship",
    "starred": "starred",
    "timezone": "time zone",
    "title": "title",
    "weekday": "weekday",
    "time": "start time",
    "duration_min": "duration",
    "attendees": "attendee",
    "category": "category",
    "appt_type": "type of appointment",
    "provider": "provider",
    "start": "start",
    "end": "end",
    "project": "project",
    "due": "due date",
    "status": "status",
    "priority": "priority",
    "from_name": "sender's name",
    "from_addr": "sender's address",
    "to": "recipient",
    "subject": "subject",
    "body": "body",
    "sent": "sent",
    "conversation_with": "conversation with",
    "contact": "contact key",
    "sender": "written by",
}

REWRITE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"text": {"type": "string"}},
    "required": ["text"],
    "additionalProperties": False,
}
REWRITE_SYSTEM = (
    "You rewrite structured records from a person's address book, calendar, task list, email, "
    "chat history and notes as short plain prose, keeping every value exactly as it is given."
)
REWRITE_RULES = (
    "- Keep every value exactly as written between the tags, character for character: names, "
    "addresses, phone numbers, titles, places, dates and times in the format given, and numbers.\n"
    "- Put the subject and the body into your text word for word; you may put them in quotation "
    "marks.\n"
    "- Where a value is marked as fixed wording, use exactly those words.\n"
    "- Do not add any facts, names, dates, times or numbers.\n"
    '- Plain sentences only: no JSON, no lists, no "field: value" lines.'
)
EXTRACT_SYSTEM = (
    "You maintain a personal knowledge store for one person (called 'the user'). You read a short "
    "text about one of the user's records and fill in the record's fields."
)
EXTRACT_TASK = (
    "Copy names, addresses, titles, subjects and bodies exactly as the text gives them, write "
    "dates and times in the formats above, and use null for a value the text does not give. "
    "Reply with JSON only."
)
CALENDAR_NOTE = "Give the record's `kind` first, then the fields of that kind."


# ------------------------------------------------------------------------------------- records
def is_background(o: ObjState) -> bool:
    """Newsletter digests (`gen.llm_render.is_background`), kept typed."""
    return o.source == "email" and str(o.payload.get("cluster", "")).startswith(
        DIGEST_CLUSTER_PREFIX
    )


def kind_fields(source: str, kind: Any = None) -> tuple[tuple[str, str], ...]:
    """The typed fields of a record of `source` (of the calendar kind `kind`)."""
    if source == "calendar":
        return (("kind", ENUM), *CALENDAR_KINDS[kind]) if kind in CALENDAR_KINDS else ()
    return FIELDS[source]


def all_fields(source: str) -> tuple[tuple[str, str], ...]:
    """Every field a record of `source` can have (calendar: `kind` and the union of the kinds)."""
    if source != "calendar":
        return FIELDS[source]
    out: dict[str, str] = {"kind": ENUM}
    for fs in CALENDAR_KINDS.values():
        for name, typ in fs:
            out.setdefault(name, typ)
    return tuple(out.items())


def typed_fields(o: ObjState) -> dict[str, Any]:
    """The record's typed fields: its payload without the metadata the store keeps."""
    typed = {k: v for k, v in o.payload.items() if k not in METADATA_KEYS}
    expected = {name for name, _ in kind_fields(o.source, typed.get("kind"))}
    if set(typed) != expected:
        raise ValueError(f"{o.object_id}: fields {sorted(typed)} are not {sorted(expected)}")
    return typed


def typed_record(o: ObjState) -> dict[str, Any]:
    """The record as `program.program_records` gives it (the D1s programs' input)."""
    return {
        "object_id": o.object_id,
        "source": o.source,
        "revision": o.revision,
        "updated": o.updated.strftime("%Y-%m-%d %H:%M"),
        **o.payload,
    }


def content_key(source: str, typed: Mapping[str, Any]) -> str:
    return sha256_json([source, dict(sorted(typed.items()))])


# --------------------------------------------------------------------------------------- facts
@dataclass(frozen=True)
class Fact:
    """A string the prose must carry: a value ("value"), the fixed words of an enumeration or a
    boolean ("fixed"), or a duration ("minutes", the text "<n> minutes")."""

    field: str
    text: str
    how: str = "value"


def facts(source: str, typed: Mapping[str, Any]) -> list[Fact]:
    """The values of a record as the rewrite request lists them, in `SOURCE_FIELDS` order."""
    out: list[Fact] = []
    for name, typ in kind_fields(source, typed.get("kind")):
        value = typed[name]
        if name in FIXED:
            out.append(Fact(name, FIXED[name][value], "fixed"))
        elif name == "duration_min":
            out.append(Fact(name, f"{int(value)} minutes", "minutes"))
        elif typ == LIST:
            out += [Fact(name, str(v)) for v in value]
        else:
            out.append(Fact(name, str(value)))
    return out


def _ws(s: str) -> str:
    return " ".join(s.split())


def present(f: Fact, text: str) -> bool:
    """Does the text carry the fact (module docstring, Check)?"""
    flat = _ws(text)
    if f.how == "value":
        return _ws(f.text) in flat
    if f.how == "minutes":
        n = re.escape(f.text.split()[0])
        return re.search(rf"(?<![\d.]){n}[- ]?min", flat, re.I) is not None
    if f.field == "starred" and f.text == "starred":
        return re.search(r"(?<!not )\bstarred\b", flat, re.I) is not None
    return re.search(rf"(?<!\w){re.escape(f.text)}(?!\w)", flat, re.I) is not None


_FIELD_LINE = re.compile(r"^\s*[A-Za-z][A-Za-z' ]{0,30}:\s*\S", re.M)
_LIST_LINE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+\S", re.M)


def _without(text: str, values: Sequence[str]) -> str:
    """The text with each verbatim value (matched across any whitespace) cut out."""
    for v in values:
        words = v.split()
        if words:
            text = re.sub(r"\s+".join(map(re.escape, words)), " ", text)
    return text


def format_problem(text: str, verbatim: Sequence[str]) -> bool:
    """JSON, a list or "field: value" lines outside the verbatim subject and body."""
    rest = _without(text, verbatim)
    return (
        rest.lstrip().startswith(("{", "["))
        or len(_FIELD_LINE.findall(rest)) >= 2
        or len(_LIST_LINE.findall(rest)) >= 2
    )


Issue = tuple[str, str]  # (kind, detail); kinds: output, missing, format


def check_text(text: str | None, fs: Sequence[Fact]) -> list[Issue]:
    """Mechanical check of a rewritten text (empty list = accepted)."""
    if text is None:
        return [("output", "")]
    issues: list[Issue] = []
    if missing := [f for f in fs if not present(f, text)]:
        issues.append(("missing", ", ".join(f"<value>{f.text}</value>" for f in missing)))
    if format_problem(text, [f.text for f in fs if f.field in VERBATIM]):
        issues.append(("format", ""))
    return issues


_ISSUE_TEXT = {
    "output": 'The reply was not a JSON object with a non-empty "text" (or it was cut off).',
    "missing": "These values are missing or not written exactly: {}",
    "format": 'The text contains JSON, a list or "field: value" lines; write plain sentences.',
}


# ------------------------------------------------------------------------------------ requests
def rewrite_prompt(source: str, fs: Sequence[Fact], issues: Sequence[Issue] = ()) -> str:
    def line(f: Fact) -> str:
        mark = " (fixed wording)" if f.how == "fixed" else ""
        return f"- {LABELS[f.field]}{mark}: <value>{f.text}</value>"

    lines = [
        f"Rewrite the record below ({RECORD_KIND[source]}) as one to three plain English "
        "sentences, the way the user might describe it in their own words.",
        "",
        "Rules:",
        REWRITE_RULES,
        "",
        "Values:",
        *(line(f) for f in fs),
        "",
    ]
    if issues:
        lines += [f"{RETRY_MARK}:"]
        lines += [f"- {_ISSUE_TEXT[k].format(d)}" for k, d in issues]
        lines += ["Write the text again and fix these problems.", ""]
    lines.append('Reply with JSON of the form {"text": "..."} containing only your text.')
    return "\n".join(lines)


def rewrite_request(
    source: str, typed: Mapping[str, Any], issues: Sequence[Issue] = ()
) -> dict[str, Any]:
    fs = facts(source, typed)
    size = approx_tokens(" ".join(f.text for f in fs))
    return {
        "messages": [
            {"role": "system", "content": REWRITE_SYSTEM},
            {"role": "user", "content": rewrite_prompt(source, fs, issues)},
        ],
        "max_tokens": min(MAX_NEW_TOKENS, 2 * size + 256),
        "schema": REWRITE_SCHEMA,
        "purpose": "rewrite",
    }


def parse_rewrite(res: LLMResult) -> str | None:
    if res.finish_reason == "length":
        return None
    data = res.json()
    text = data.get("text") if isinstance(data, dict) else None
    if not isinstance(text, str):
        return None
    return clean_body(text) or None


_JSON_TYPES: dict[str, dict[str, Any]] = {
    STR: {"type": "string"},
    INT: {"type": "integer"},
    BOOL: {"type": "boolean"},
    LIST: {"type": "array", "items": {"type": "string"}},
}


def _object_schema(fields: Sequence[tuple[str, str]], kind: str | None = None) -> dict[str, Any]:
    props: dict[str, Any] = {} if kind is None else {"kind": {"enum": [kind]}}
    for name, typ in fields:
        t = {"enum": list(ENUMS[name])} if typ == ENUM else _JSON_TYPES[typ]
        props[name] = {"anyOf": [t, {"type": "null"}]}
    return {
        "type": "object",
        "properties": props,
        "required": list(props),
        "additionalProperties": False,
    }


def extraction_schema(source: str) -> dict[str, Any]:
    if source == "calendar":
        return {"anyOf": [_object_schema(fs, kind) for kind, fs in CALENDAR_KINDS.items()]}
    return _object_schema(FIELDS[source])


def extract_prompt(source: str, text: str) -> str:
    lines = [
        f"The text below describes one record from the user's {source} ({RECORD_KIND[source]}). "
        "Fill in the record's fields from the text.",
        "",
        "Fields:",
        SOURCE_FIELDS[source],
        "",
        EXTRACT_TASK,
    ]
    if source == "calendar":
        lines.append(CALENDAR_NOTE)
    lines += ["", "Text:", f"<text>\n{text}\n</text>"]
    return "\n".join(lines)


def extract_request(source: str, text: str) -> dict[str, Any]:
    return {
        "messages": [
            {"role": "system", "content": EXTRACT_SYSTEM},
            {"role": "user", "content": extract_prompt(source, text)},
        ],
        "max_tokens": min(MAX_NEW_TOKENS, 2 * approx_tokens(text) + 256),
        "schema": extraction_schema(source),
        "purpose": "extract",
    }


def _typed_ok(typ: str, name: str, v: Any) -> bool:
    if typ == STR:
        return isinstance(v, str)
    if typ == INT:
        return isinstance(v, int) and not isinstance(v, bool)
    if typ == BOOL:
        return isinstance(v, bool)
    if typ == LIST:
        return isinstance(v, list) and all(isinstance(x, str) for x in v)
    return v in (CALENDAR_KINDS if name == "kind" else ENUMS[name])


def null_fields(source: str) -> dict[str, Any]:
    """The fields of a record whose extraction failed (a calendar record of unknown kind)."""
    return {name: None for name, _ in (FIELDS.get(source) or (("kind", ENUM),))}


def parse_extraction(source: str, res: LLMResult) -> dict[str, Any] | None:
    """The extracted fields in `SOURCE_FIELDS` order (calendar: those of the extracted kind); a
    value of the wrong type is None. None when the reply was cut off or is not a JSON object."""
    data = res.json()
    if res.finish_reason == "length" or not isinstance(data, dict):
        return None
    if source == "calendar":
        kind = data.get("kind")
        if kind not in CALENDAR_KINDS:
            out: dict[str, Any] = {"kind": None}
            for name, typ in all_fields(source)[1:]:
                v = data.get(name)
                if v is not None and _typed_ok(typ, name, v):
                    out[name] = v
            return out
        fields = kind_fields(source, kind)
    else:
        fields = FIELDS[source]
    return {
        name: (v if (v := data.get(name)) is not None and _typed_ok(typ, name, v) else None)
        for name, typ in fields
    }


# ---------------------------------------------------------------------------------- comparison
def _same(typ: str, a: Any, b: Any, exact: bool) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if typ == LIST:
        if not isinstance(b, list):
            return False
        if exact:
            return sorted(_ws(str(x)) for x in a) == sorted(_ws(str(x)) for x in b)
        return {norm_str(x) for x in a} == {norm_str(x) for x in b}
    if typ in (INT, BOOL):
        return type(a) is type(b) and a == b
    if not isinstance(b, str):
        return False
    return _ws(a) == _ws(b) if exact else norm_str(a) == norm_str(b)


def field_errors(
    source: str, typed: Mapping[str, Any], extracted: Mapping[str, Any] | None
) -> tuple[list[str], list[str]]:
    """(fields that differ after normalisation, fields that differ after whitespace collapsing),
    over the typed record's fields (module docstring, Accounting)."""
    errors: list[str] = []
    inexact: list[str] = []
    got = extracted or {}
    for name, typ in kind_fields(source, typed.get("kind")):
        if not _same(typ, typed[name], got.get(name), exact=False):
            errors.append(name)
        if not _same(typ, typed[name], got.get(name), exact=True):
            inexact.append(name)
    return errors, inexact


# ---------------------------------------------------------------------------------- extraction
@dataclass
class Extraction:
    """One record content through rewrite, check and extraction."""

    source: str
    typed: dict[str, Any]
    prose: str | None
    first_issues: list[Issue]
    retried: bool
    final_issues: list[Issue]
    fields: dict[str, Any] | None  # extracted; None when the extraction failed or no prose
    errors: list[str]
    inexact: list[str]
    keys: dict[str, list[str]] = field(default_factory=dict)  # purpose -> cache keys
    usage: dict[str, int] = field(default_factory=dict)  # prompt / completion tokens

    @property
    def extract_ok(self) -> bool:
        return self.fields is not None

    def record(self, o: ObjState) -> dict[str, Any]:
        """The record a D1sx program reads: store metadata, extracted fields, the prose."""
        return {
            "object_id": o.object_id,
            "source": o.source,
            "revision": o.revision,
            "updated": o.updated.strftime("%Y-%m-%d %H:%M"),
            **(self.fields if self.fields is not None else null_fields(o.source)),
            "cluster": o.cluster,
            "text": self.prose or "",
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "typed": self.typed,
            "prose": self.prose,
            "rewrite": {
                "first_issues": [list(i) for i in self.first_issues],
                "retried": self.retried,
                "final_issues": [list(i) for i in self.final_issues],
            },
            "extracted": self.fields,
            "extract_ok": self.extract_ok,
            "errors": self.errors,
            "inexact": self.inexact,
            "keys": self.keys,
            "usage": self.usage,
        }


class ChatClient(Protocol):
    model: str

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult: ...


def run_extraction(
    rewriter: ChatClient, extractor: ChatClient, source: str, typed: Mapping[str, Any]
) -> Extraction:
    """Rewrite (and at most one retry), check and extract one record content."""
    typed = dict(typed)
    fs = facts(source, typed)
    keys: dict[str, list[str]] = defaultdict(list)
    usage = Counter[str]()

    def send(client: ChatClient, req: dict[str, Any]) -> LLMResult:
        res = client.complete(**req)
        keys[req["purpose"]].append(res.key)
        usage["prompt_tokens"] += res.prompt_tokens
        usage["completion_tokens"] += res.completion_tokens
        return res

    text = parse_rewrite(send(rewriter, rewrite_request(source, typed)))
    first = check_text(text, fs)
    final, retried = first, False
    if first:
        retried = True
        again = parse_rewrite(send(rewriter, rewrite_request(source, typed, first)))
        if again is not None or text is None:
            text, final = again, check_text(again, fs)
    fields = None
    if text is not None:
        fields = parse_extraction(source, send(extractor, extract_request(source, text)))
    errors, inexact = field_errors(source, typed, fields)
    return Extraction(
        source, typed, text, first, retried, final, fields, errors, inexact, dict(keys), dict(usage)
    )


_IDS = itertools.count()


class RecordExtractor:
    """The extracted view of the live records for the D1sx cells of one world (both branches):
    `records_json` is what a D1sx program reads (`program.RecordView`), `input_note` what its item
    records about that input, `rows` and `report` what the run writes per branch."""

    def __init__(self, rewriter: ChatClient, extractor: ChatClient, workers: int = 64):
        self.uid = next(_IDS)
        self.tag = f"extracted:{self.uid}"  # keeps program runs on these records apart
        self.rewriter = rewriter
        self.extractor = extractor
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="extract")
        self._futures: dict[str, Future[Extraction]] = {}
        self._used: dict[int, dict[tuple[str, int], tuple[str, str]]] = defaultdict(dict)
        self._lock = threading.Lock()
        self._json = _Memo[str](128)
        self._notes = _Memo[dict[str, Any]](4096)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def _submit(self, log: SourceLog, o: ObjState) -> Future[Extraction] | None:
        """Start (or join) the extraction of a live record; None for a background record."""
        if is_background(o):
            with self._lock:
                self._used[log.uid][(o.object_id, o.revision)] = (o.source, "")
            return None
        typed = typed_fields(o)
        key = content_key(o.source, typed)
        with self._lock:
            self._used[log.uid][(o.object_id, o.revision)] = (o.source, key)
            fut = self._futures.get(key)
            if fut is None:
                fut = self._pool.submit(
                    run_extraction, self.rewriter, self.extractor, o.source, typed
                )
                self._futures[key] = fut
        return fut

    def _live(self, log: SourceLog, sources: tuple[str, ...], t: datetime) -> list[ObjState]:
        snap = log.snapshot(t)
        return [
            o
            for s in sources
            for o in sorted(snap.by_source(s), key=lambda o: (o.updated, o.object_id))
        ]

    def prepare(
        self, log: SourceLog, sources: tuple[str, ...], t: datetime
    ) -> list[tuple[ObjState, Extraction | None]]:
        """Every live record of `sources` at t with its extraction (None: background), in the
        order of `program.program_records`; extracts the records not done yet, in parallel."""
        objs = self._live(log, sources, t)
        futures = [self._submit(log, o) for o in objs]
        return [(o, None if f is None else f.result()) for o, f in zip(objs, futures, strict=True)]

    def _source_json(self, log: SourceLog, source: str, t: datetime) -> str:
        def make() -> str:
            pairs = self.prepare(log, (source,), t)
            return ",".join(
                json.dumps(typed_record(o) if ex is None else ex.record(o)) for o, ex in pairs
            )

        return self._json.get((log.uid, t, source), make)

    def records_json(self, log: SourceLog, slot: Slot, t: datetime) -> str:
        """The JSON a D1sx program reads: `program.records_json` with the extracted records."""
        for s in slot.sources:  # start every source's records before waiting on any
            for o in self._live(log, (s,), t):
                self._submit(log, o)
        parts = (self._source_json(log, s, t) for s in slot.sources)
        return "[" + ",".join(p for p in parts if p) + "]"

    def input_note(self, log: SourceLog, slot: Slot, t: datetime) -> dict[str, Any]:
        """What a D1sx program read at t: records, background records kept typed, rewrites that
        still missed a value, failed extractions, records with an extraction error and the
        erroneous fields (`source.field`)."""

        def make() -> dict[str, Any]:
            pairs = self.prepare(log, slot.sources, t)
            exs = [ex for _, ex in pairs if ex is not None]
            return {
                "records": len(pairs),
                "background": len(pairs) - len(exs),
                "rewrite_missed": sum(bool(ex.final_issues) for ex in exs),
                "extract_failed": sum(not ex.extract_ok for ex in exs),
                "with_error": sum(bool(ex.errors) for ex in exs),
                "error_fields": sorted({f"{ex.source}.{f}" for ex in exs for f in ex.errors}),
            }

        return self._notes.get((log.uid, t, slot.sources), make)

    def _done(self, log: SourceLog) -> list[tuple[tuple[str, int], str, Extraction | None]]:
        with self._lock:
            used = sorted(self._used.get(log.uid, {}).items())
            futures = dict(self._futures)
        return [(oid_rev, src, futures[k].result() if k else None) for oid_rev, (src, k) in used]

    def rows(self, log: SourceLog, world_id: str, branch: str) -> list[dict[str, Any]]:
        """One row per record revision a D1sx program read on this log (background: none)."""
        head = {"world_id": world_id, "branch": branch}
        return [
            {**head, "object_id": oid, "revision": rev, **ex.to_json()}
            for (oid, rev), _, ex in self._done(log)
            if ex is not None
        ]

    def report(self, log: SourceLog) -> dict[str, Any]:
        """Per source, over the record revisions read on this log."""
        out: dict[str, dict[str, Any]] = {}
        for _, source, ex in self._done(log):
            c = out.setdefault(
                source,
                {
                    "records": 0,
                    "background": 0,
                    "rewritten_first": 0,
                    "rewritten_retry": 0,
                    "missed": 0,
                    "no_text": 0,
                    "first_failures": {},
                    "extract_failed": 0,
                    "with_error": 0,
                },
            )
            c["records"] += 1
            if ex is None:
                c["background"] += 1
                continue
            for kind in sorted({k for k, _ in ex.first_issues}):
                c["first_failures"][kind] = c["first_failures"].get(kind, 0) + 1
            if ex.prose is None:
                c["no_text"] += 1
            elif ex.final_issues:
                c["missed"] += 1
            else:
                c["rewritten_retry" if ex.retried else "rewritten_first"] += 1
            c["extract_failed"] += not ex.extract_ok
            c["with_error"] += bool(ex.errors)
        return dict(sorted(out.items()))

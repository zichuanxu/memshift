"""Records rewritten as prose and their fields extracted back (`holdmem.extract`): the field
tables against the generator and the program requests, the facts and fixed words, the mechanical
check, the one retry and the counted misses, extraction parsing per source, the comparison, and the
extractor's sharing and record view (fake model, `tests.extract_fakes`)."""

import json
from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from memshift.experiments.common import world_params
from memshift.gen.shifts import generate_branch
from memshift.gen.state import ObjState, SourceLog
from memshift.holdmem import extract as ex
from memshift.holdmem.program import SOURCE_FIELDS, records_json
from memshift.llm import LLMResult
from tests.extract_fakes import ExtractLLM
from tests.fakes import RecordingLLM

T = datetime(2026, 3, 1, 12, 0)


def obj(source, payload, oid="x:1", rev=1, updated=T):
    full = {**payload, "cluster": f"c:{oid}", "text": "template text"}
    return ObjState(oid, source, rev, full, updated, updated)


CONTACT = {
    "name": "Ana Lopez",
    "email": "ana.lopez@acme.example",
    "phone": "+1-555-123-4567",
    "employer": "Acme Corp",
    "city": "Lisbon",
    "relation": "family",
    "starred": False,
}
MEETING = {
    "kind": "meeting",
    "title": "Budget review",
    "start": "2026-02-03 10:00",
    "end": "2026-02-03 11:00",
    "attendees": ["Ana Lopez", "Ben Okafor"],
    "category": "work",
}
SERIES = {
    "kind": "series",
    "title": "Team sync",
    "weekday": "Tuesday",
    "time": "14:30",
    "duration_min": 30,
    "attendees": ["Ana Lopez"],
    "category": "work",
}
TASK = {
    "title": "File taxes",
    "project": "Home",
    "due": "2026-03-04",
    "status": "open",
    "priority": "high",
}
MESSAGE = {
    "conversation_with": "Ana Lopez",
    "contact": "p03",
    "sender": "Ana Lopez",
    "body": "See you at 7!",
    "sent": "2026-02-03 18:05",
}
EMAIL = {
    "from_name": "Ben Okafor",
    "from_addr": "ben.okafor@acme.example",
    "to": "Kim Doe",
    "subject": "Quick question",
    "body": "Hi Kim,\n\nCan you send the draft?\n\nBest,\nBen",
    "sent": "2026-02-03 09:15",
}
NOTE = {
    "title": "Working hours",
    "body": "My working hours: 09:00-17:00. No meetings before 09:30.",
}
SAMPLES = {
    "contacts": CONTACT,
    "calendar": MEETING,
    "tasks": TASK,
    "messages": MESSAGE,
    "email": EMAIL,
    "notes": NOTE,
}


@pytest.fixture(scope="module")
def worlds():
    return [generate_branch(5, b, world_params(3)) for b in ("src", "def", "orphan")]


# ---------------------------------------------------------------------------- tables
def test_field_tables_match_the_generator_payloads(worlds):
    seen = set()
    for w in worlds:
        for e in w.events:
            if e.op == "delete":
                continue
            o = ObjState(e.object_id, e.source, e.revision, e.payload, e.ts, e.ts)
            typed = ex.typed_fields(o)  # raises on a field outside the tables
            seen.add((e.source, typed.get("kind")))
            for name, typ in ex.kind_fields(e.source, typed.get("kind")):
                if name in ex.ENUMS:
                    assert typed[name] in ex.ENUMS[name]
                assert ex._typed_ok(typ, name, typed[name]), (e.source, name)
    assert {s for s, _ in seen} == set(ex.SOURCES)
    assert {k for s, k in seen if s == "calendar"} == set(ex.CALENDAR_KINDS)


def test_field_tables_are_the_fields_the_program_requests_describe():
    for source in ex.SOURCES:
        text = SOURCE_FIELDS[source]
        for name, typ in ex.all_fields(source):
            assert f"{name} (" in text or f"`{name}`" in text, (source, name)
            if typ == ex.ENUM and name != "kind":
                kinds = ex.ENUMS[name] if name != "category" else ("work", "personal")
                assert all(v in text for v in kinds), (source, name)
        assert ex.FIXED["kind"].keys() == ex.CALENDAR_KINDS.keys()
    for name, words in ex.FIXED.items():
        assert len(set(words.values())) == len(words), name


# ---------------------------------------------------------------------------- facts and check
def test_facts_use_fixed_words_and_list_every_attendee():
    fs = ex.facts("contacts", CONTACT)
    assert [(f.field, f.text, f.how) for f in fs][-2:] == [
        ("relation", "family member", "fixed"),
        ("starred", "not starred", "fixed"),
    ]
    fs = ex.facts("calendar", MEETING)
    assert fs[0] == ex.Fact("kind", "one-off", "fixed")
    assert [f.text for f in fs if f.field == "attendees"] == ["Ana Lopez", "Ben Okafor"]
    assert ex.Fact("duration_min", "30 minutes", "minutes") in ex.facts("calendar", SERIES)
    assert [f.text for f in ex.facts("tasks", TASK)][-2:] == ["open", "high priority"]


def test_values_must_be_verbatim_and_fixed_words_whole_words():
    fs = ex.facts("contacts", CONTACT)
    good = (
        "Ana Lopez (ana.lopez@acme.example, +1-555-123-4567) works at Acme Corp in Lisbon. "
        "She is a Family Member of mine and is not starred."
    )
    assert ex.check_text(good, fs) == []
    lower = good.replace("Ana Lopez", "ana lopez")
    assert ex.check_text(lower, fs) == [("missing", "<value>Ana Lopez</value>")]
    starred = ex.facts("contacts", {**CONTACT, "starred": True})
    assert ex.check_text(good, starred) == [("missing", "<value>starred</value>")]
    assert ex.check_text(good.replace("not starred", "Starred"), starred) == []
    assert ex.check_text(None, fs) == [("output", "")]


def test_durations_bodies_and_format():
    fs = ex.facts("calendar", SERIES)
    base = "A recurring work event, Team sync, every Tuesday at 14:30 with Ana Lopez"
    assert ex.check_text(base + ", for 30 minutes.", fs) == []
    assert ex.check_text(base + ", a 30-minute slot.", fs) == []
    assert ex.check_text(base + ", 30.", fs)[0][0] == "missing"
    efs = ex.facts("email", EMAIL)
    head = "Ben Okafor (ben.okafor@acme.example) wrote to Kim Doe on 2026-02-03 09:15"
    flat = f'{head}, subject "Quick question": "Hi Kim, Can you send the draft? Best, Ben".'
    assert ex.check_text(flat, efs) == []  # the body's whitespace is collapsed
    lines = "Sender: Ben Okafor\nAddress: ben.okafor@acme.example\n" + flat
    assert ex.check_text(lines, efs) == [("format", "")]
    assert ex.check_text("{" + flat + "}", efs) == [("format", "")]
    # "field: value" lines inside the verbatim body do not count
    nfs = ex.facts("notes", {"title": "Plan", "body": "Monday: gym\nTuesday: pool\nWed: rest"})
    note = 'My note "Plan" says:\nMonday: gym\nTuesday: pool\nWed: rest'
    assert ex.check_text(note, nfs) == []


def test_rewrite_request_lists_the_values_not_the_template():
    req = ex.rewrite_request("contacts", CONTACT)
    prompt = req["messages"][-1]["content"]
    assert "- relationship (fixed wording): <value>family member</value>" in prompt
    assert "- starred (fixed wording): <value>not starred</value>" in prompt
    assert "template text" not in prompt and "Contact card:" not in prompt
    assert req["schema"] == ex.REWRITE_SCHEMA and req["purpose"] == "rewrite"
    retry = ex.rewrite_request("contacts", CONTACT, [("missing", "<value>Lisbon</value>")])
    assert ex.RETRY_MARK in retry["messages"][-1]["content"]
    assert "<value>Lisbon</value>" in retry["messages"][-1]["content"].split(ex.RETRY_MARK)[1]


def test_requests_depend_on_the_content_only():
    a = obj("tasks", TASK, "tasks:t001", 1, T)
    b = obj("tasks", TASK, "tasks:t009", 4, T + timedelta(days=30))
    ta, tb = ex.typed_fields(a), ex.typed_fields(b)
    assert ex.rewrite_request("tasks", ta) == ex.rewrite_request("tasks", tb)
    assert ex.content_key("tasks", ta) == ex.content_key("tasks", tb)
    assert ex.content_key("tasks", ta) != ex.content_key("tasks", {**TASK, "status": "done"})


# ---------------------------------------------------------------------------- one retry
def test_a_miss_is_retried_once_naming_the_missing_value():
    llm = ExtractLLM(drop=["contact"])
    out = ex.run_extraction(llm, llm, "messages", MESSAGE)
    rewrites = [b for _, p, b in llm.requests if p == "rewrite"]
    assert len(rewrites) == 2 and out.retried
    assert out.first_issues == [("missing", "<value>p03</value>")]
    retry = rewrites[1]["messages"][-1]["content"]
    assert retry.split(ex.RETRY_MARK)[1].count("<value>p03</value>") == 1
    assert out.final_issues == [] and out.errors == [] and out.fields == MESSAGE


def test_a_second_miss_is_kept_and_counted_never_replaced_by_a_template():
    llm = ExtractLLM(drop=["city"], drop_always=True)
    out = ex.run_extraction(llm, llm, "contacts", CONTACT)
    assert [p for _, p, _ in llm.requests] == ["rewrite", "rewrite", "extract"]
    assert out.final_issues == [("missing", "<value>Lisbon</value>")]
    assert out.prose is not None and "Lisbon" not in out.prose
    assert out.errors == ["city"] and out.fields is not None and out.fields["city"] is None
    lines = ExtractLLM(fields_lines=True)
    fixed = ex.run_extraction(lines, lines, "contacts", CONTACT)
    assert fixed.first_issues == [("format", "")] and fixed.final_issues == []


class GarbleRewrite(RecordingLLM):
    def complete(self, messages, max_tokens=512, schema=None, purpose=""):
        key, body = self.request_key(messages, max_tokens, schema)
        self.requests.append((key, purpose, body))
        return LLMResult('{"note": "no text"}', 10, 5, 0.0, False, "stop", key)


def test_no_text_sends_no_extraction_and_gives_null_fields():
    llm = GarbleRewrite()
    out = ex.run_extraction(llm, llm, "tasks", TASK)
    assert [p for _, p, _ in llm.requests] == ["rewrite", "rewrite"]
    assert out.prose is None and out.fields is None and not out.extract_ok
    assert out.final_issues == [("output", "")]
    assert out.errors == [n for n, _ in ex.FIELDS["tasks"]]
    rec = out.record(obj("tasks", TASK, "tasks:t001"))
    assert rec["text"] == "" and all(rec[n] is None for n, _ in ex.FIELDS["tasks"])
    assert rec["updated"] == "2026-03-01 12:00" and rec["cluster"] == "c:tasks:t001"


# ---------------------------------------------------------------------------- extraction
@pytest.mark.parametrize("source", ex.SOURCES)
def test_extraction_round_trip_per_source(source):
    typed = SAMPLES[source]
    llm = ExtractLLM()
    out = ex.run_extraction(llm, llm, source, typed)
    assert out.fields == typed and out.errors == [] and out.inexact == []
    req = next(b for _, p, b in llm.requests if p == "extract")
    prompt = req["messages"][-1]["content"]
    assert SOURCE_FIELDS[source] in prompt and out.prose in prompt
    assert ("Give the record's `kind`" in prompt) == (source == "calendar")
    schema = req["response_format"]["json_schema"]["schema"]
    assert schema == ex.extraction_schema(source)
    if source != "calendar":
        assert schema["required"] == [n for n, _ in ex.FIELDS[source]]


def test_calendar_schema_is_one_object_per_kind():
    branches = ex.extraction_schema("calendar")["anyOf"]
    assert [b["properties"]["kind"] for b in branches] == [{"enum": [k]} for k in ex.CALENDAR_KINDS]
    assert [b["required"] for b in branches] == [
        ["kind", *(n for n, _ in fs)] for fs in ex.CALENDAR_KINDS.values()
    ]
    assert all(b["additionalProperties"] is False for b in branches)


def test_extraction_errors_and_types():
    llm = ExtractLLM(corrupt={("messages", "sender"): "Kim Doe", ("messages", "contact"): 3})
    out = ex.run_extraction(llm, llm, "messages", MESSAGE)
    assert out.fields is not None and out.fields["contact"] is None  # wrong type -> null
    assert out.errors == ["contact", "sender"]
    garbled = ExtractLLM(garble=["notes"])
    bad = ex.run_extraction(garbled, garbled, "notes", NOTE)
    assert bad.fields is None and bad.errors == ["title", "body"]


def _res(text, finish="stop"):
    return LLMResult(text, 1, 1, 0.0, False, finish, "k")


def test_calendar_keeps_the_fields_of_the_extracted_kind():
    flat = {n: None for n, _ in ex.all_fields("calendar")}
    reply = {**flat, **SERIES, "start": "2026-02-03 10:00"}
    assert ex.parse_extraction("calendar", _res(json.dumps(reply))) == SERIES
    wrong = {**flat, **MEETING, "kind": "series"}
    got = ex.parse_extraction("calendar", _res(json.dumps(wrong)))
    errors, _ = ex.field_errors("calendar", MEETING, got)
    assert errors == ["kind", "start", "end"]
    assert ex.parse_extraction("calendar", _res(json.dumps(reply), "length")) is None
    unknown = ex.parse_extraction("calendar", _res(json.dumps({**reply, "kind": "party"})))
    assert unknown is not None and unknown["kind"] is None and unknown["title"] == "Team sync"


def test_errors_are_after_normalisation_inexact_after_whitespace():
    got = {
        **MEETING,
        "title": "budget review",
        "attendees": ["Ben Okafor", "Ana Lopez"],
        "start": "2026-02-03  10:00",
    }
    assert ex.field_errors("calendar", MEETING, got) == ([], ["title"])
    assert ex.field_errors("calendar", SERIES, {**SERIES, "duration_min": "30"}) == (
        ["duration_min"],
        ["duration_min"],
    )
    assert ex.field_errors("contacts", CONTACT, {**CONTACT, "starred": None})[0] == ["starred"]


# ---------------------------------------------------------------------------- the extractor
def _first_digest(log, t):
    return next(o for o in log.snapshot(t).by_source("email") if ex.is_background(o))


def test_extractor_view_matches_the_store_but_for_text_and_shares_requests(worlds):
    w_src, w_def = worlds[0], worlds[1]
    logs = [SourceLog(w_src.events), SourceLog(w_def.events)]
    llm = ExtractLLM()
    extractor = ex.RecordExtractor(llm, llm, workers=8)
    t0 = w_src.fork - timedelta(minutes=1)
    slot = w_src.slot("k.vip_sender")  # email and contacts
    try:
        got = json.loads(extractor.records_json(logs[0], slot, t0))
        want = json.loads(records_json(logs[0], slot, t0))
        assert len(got) == len(want)
        for g, r in zip(got, want, strict=True):
            assert {k: v for k, v in g.items() if k != "text"} == {
                k: v for k, v in r.items() if k != "text"
            }
            assert (g["text"] == r["text"]) == str(g.get("cluster", "")).startswith(
                ex.DIGEST_CLUSTER_PREFIX
            )
        sent = len(llm.requests)
        # the other branch: the records before the fork are shared, no new request for them
        extractor.records_json(logs[1], slot, t0)
        assert len(llm.requests) == sent
        digest = _first_digest(logs[0], t0)
        assert all(digest.payload["body"] not in json.dumps(b) for _, _, b in llm.requests)
        note = extractor.input_note(logs[0], slot, t0)
        assert note["records"] == len(want) and note["background"] > 0
        assert note["with_error"] == 0 and note["error_fields"] == []
        rows = extractor.rows(logs[0], w_src.world_id, "src")
        assert len(rows) == note["records"] - note["background"]
        rep = extractor.report(logs[0])
        assert rep["email"]["background"] == note["background"]
        assert sum(c["records"] for c in rep.values()) == note["records"]
    finally:
        extractor.close()


def test_identical_content_is_rewritten_once(worlds):
    w = worlds[1]
    log = SourceLog(w.events)
    t0 = w.fork - timedelta(minutes=1)
    o = log.snapshot(t0).by_source("tasks")[0]
    twin = replace(o, object_id="tasks:twin", revision=7)
    llm = ExtractLLM()
    extractor = ex.RecordExtractor(llm, llm, workers=4)
    try:
        a = extractor._submit(log, o)
        b = extractor._submit(log, twin)
        assert a is b and a is not None
        a.result()
        assert [p for _, p, _ in llm.requests] == ["rewrite", "extract"]
    finally:
        extractor.close()

"""Sandbox and request of the writer-authored programs (cell D1s)."""

import json
import re
from datetime import timedelta

import pytest

from memshift.experiments.common import world_params
from memshift.gen.kappa import evaluate
from memshift.gen.model import SOURCES
from memshift.gen.model import parse_ts as gen_parse_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem import sandbox
from memshift.holdmem.program import (
    COMMON_FIELDS,
    SOURCE_FIELDS,
    ProgramOutcome,
    extract_code,
    fields_text,
    program_check,
    program_records,
    program_request,
    records_json,
    run_program,
    validate_value,
)
from memshift.llm import LLMResult

DERIVE = "def derive(records, as_of):\n"
AS_OF = "2026-01-15 12:00"


def code(body: str) -> str:
    return DERIVE + "".join(f"    {line}\n" for line in body.splitlines())


@pytest.fixture(scope="module")
def short_world():
    w = generate_branch(11, "def", world_params(3))
    return w, SourceLog(w.events)


# ---------------------------------------------------------------------------- whitelist
@pytest.mark.parametrize(
    "source, why",
    [
        ("import os\n" + code("return None"), "Import"),
        ("from os import path\n" + code("return None"), "ImportFrom"),
        (code("return ().__class__"), "__class__"),
        (code("return __builtins__"), "__builtins__"),
        (code("return records[0].__len__()"), "__len__"),
        (code("return records._private"), "_private"),
        (code("return open('x')"), "open"),
        (code("return eval('1')"), "eval"),
        (code("exec('x = 1')"), "exec"),
        (code("return compile('1', 'x', 'eval')"), "compile"),
        (code("return globals()"), "globals"),
        (code("return locals()"), "locals"),
        (code("return vars()"), "vars"),
        (code("return getattr(records, 'x')"), "getattr"),
        (code("setattr(records, 'x', 1)"), "setattr"),
        (code("delattr(records, 'x')"), "delattr"),
        (code("return input()"), "input"),
        (code("breakpoint()"), "breakpoint"),
        (code("return __import__('os')"), "__import__"),
        (code("f = getattr\nreturn f"), "getattr"),
        (code("return (x for x in []).gi_frame"), "gi_frame"),
        (code("return datetime.now()"), "now"),
        (code("return datetime.today()"), "today"),
        (code("return str.mro()"), "mro"),
        ("class A:\n    pass\n" + code("return None"), "ClassDef"),
        (code("with records as r:\n    pass"), "With"),
        (code("global x\nreturn x"), "Global"),
        (code("yield 1"), "Yield"),
        (code("try:\n    return 1\nexcept:\n    return 2"), "bare except"),
        ("x = 1\n", "no top-level function derive"),
        ("def derive(records):\n    return None\n", "exactly"),
        ("def derive(records, as_of, extra):\n    return None\n", "exactly"),
        ("def derive(*records):\n    return None\n", "exactly"),
        (code("return ("), "syntax error"),
        (code("return None\n" + "#" * sandbox.MAX_CODE_CHARS), "too long"),
    ],
)
def test_source_is_rejected(source, why):
    error = sandbox.check_source(source)
    assert error is not None and why in error
    ok, value, err = run_program(source, [], AS_OF, "city")
    assert not ok and value is None and why in err


def test_source_that_should_pass_the_whitelist():
    assert sandbox.check_source(code("return None")) is None
    assert sandbox.check_source(code("for _ in range(2):\n    pass\nreturn None")) is None
    assert sandbox.check_source(code("return sorted(records, key=lambda r: r['updated'])")) is None
    helper = "def helper(x):\n    return [y for y in x]\n"
    assert sandbox.check_source(helper + code("return helper([])")) is None
    assert (
        sandbox.check_source(code("try:\n    return int('x')\nexcept ValueError:\n    return 2"))
        is None
    )
    fstring = 'return f"{as_of[:4]}"'
    assert sandbox.check_source(code(fstring)) is None


def test_every_builtin_of_the_whitelist_exists_and_the_dangerous_ones_are_absent():
    import builtins

    assert all(hasattr(builtins, n) for n in sandbox.BUILTIN_FUNCTIONS + sandbox.EXCEPTION_CLASSES)
    assert not set(sandbox.BUILTIN_FUNCTIONS) & sandbox.FORBIDDEN_NAMES
    assert {
        "open",
        "eval",
        "exec",
        "compile",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
        "__import__",
        "input",
        "breakpoint",
    } <= sandbox.FORBIDDEN_NAMES


# ---------------------------------------------------------------------------- execution
def test_program_runs_on_records_with_the_provided_names():
    body = (
        "now = parse_ts(as_of)\n"
        "lo = now - timedelta(days=30)\n"
        "seen = [r['who'] for r in records if lo <= datetime.strptime(r['sent'], "
        "'%Y-%m-%d %H:%M') <= now]\n"
        "return sorted(set(seen))"
    )
    records = [
        {"who": "Ann Lee", "sent": "2026-01-10 09:00"},
        {"who": "Bo Chen", "sent": "2025-11-01 09:00"},
        {"who": "Ann Lee", "sent": "2026-01-14 10:30"},
    ]
    assert run_program(code(body), records, AS_OF, "set_person") == (True, ["Ann Lee"], "")


def test_datetime_methods_work_but_nothing_else_can_be_imported():
    body = (
        "d = datetime.strptime(as_of, '%Y-%m-%d %H:%M') + timedelta(hours=3)\n"
        "return d.strftime('%A %H:%M') + ' ' + f'{d:%Y}' + ' ' + d.isoformat()"
    )
    assert run_program(code(body), [], AS_OF, "city") == (
        True,
        "Thursday 15:00 2026 2026-01-15T15:00:00",
        "",
    )
    assert sandbox._restricted_import("time").strftime  # the two imports datetime needs
    assert sandbox._restricted_import("_strptime")
    for name in ("os", "sys", "subprocess", "builtins", "datetime"):
        with pytest.raises(ImportError):
            sandbox._restricted_import(name)
    with pytest.raises(ImportError):
        sandbox._restricted_import("time", level=1)


def test_importing_the_provided_datetime_names_is_a_no_op():
    body = "return (datetime.strptime(as_of, '%Y-%m-%d %H:%M') + timedelta(days=1)).strftime('%A')"
    for head in (
        "from datetime import datetime, timedelta\n",
        "from datetime import timedelta\n",
        "from datetime import datetime as datetime\n",
    ):
        assert run_program(head + code(body), [], AS_OF, "city") == (True, "Friday", "")
    inner = code("from datetime import datetime\n" + body)  # inside the function too
    assert run_program(inner, [], AS_OF, "city") == (True, "Friday", "")
    for head, why in (
        ("import datetime\n", "Import"),
        ("from datetime import date\n", "ImportFrom"),
        ("from datetime import datetime as dt\n", "ImportFrom"),
        ("from datetime import datetime, date\n", "ImportFrom"),
        ("from datetime import *\n", "ImportFrom"),
        ("from time import time\n", "ImportFrom"),
    ):
        assert sandbox.check_source(head + code(body)) == f"disallowed syntax: {why}"


def test_program_output_is_typed_against_the_value_type():
    ok = run_program(code("return 'Paris'"), [], AS_OF, "city")
    assert ok == (True, "Paris", "")
    assert run_program(code("return None"), [], AS_OF, "city") == (True, None, "")
    for body, vtype in [
        ("return 3", "city"),
        ("return ['Paris']", "city"),
        ("return {'Paris'}", "city"),
        ("return ('a', 'b')", "set_person"),
        ("return 'Ann'", "set_person"),
        ("return None", "set_person"),
        ("return ['Ann', 3]", "set_person"),
        ("return [['Ann']]", "set_person"),
    ]:
        ok, value, error = run_program(code(body), [], AS_OF, vtype)
        assert not ok and value is None and "type mismatch" in error, body


def test_set_output_is_deduplicated_and_sorted():
    assert run_program(code("return ['b', 'a', 'b']"), [], AS_OF, "set_task") == (
        True,
        ["a", "b"],
        "",
    )
    assert validate_value("set_task", []) == (True, [], "")
    assert validate_value("date", "2026-01-01") == (True, "2026-01-01", "")
    assert validate_value("date", 20260101)[0] is False


def test_runtime_errors_are_reported_not_raised():
    ok, value, error = run_program(code("raise ValueError('boom')"), [], AS_OF, "city")
    assert (ok, value) == (False, None) and "ValueError: boom" in error
    ok, _, error = run_program(code("return print('x')"), [], AS_OF, "city")
    assert not ok and "NameError" in error  # print is not a whitelisted builtin
    ok, _, error = run_program(code("return records[0]['x']"), [], AS_OF, "city")
    assert not ok and "IndexError" in error
    ok, _, error = run_program(code("return 1 / 0"), [], AS_OF, "city")
    assert not ok and "ZeroDivisionError" in error
    ok, _, error = run_program(code("return ''.join(['a']).nonexistent"), [], AS_OF, "city")
    assert not ok and "AttributeError" in error


def test_timeout_is_a_failure_even_when_the_program_swallows_exceptions():
    loop = code("while True:\n    pass")
    assert run_program(loop, [], AS_OF, "city") == (False, None, "timeout")
    swallow = code("while True:\n    try:\n        pass\n    except Exception:\n        pass")
    assert run_program(swallow, [], AS_OF, "city") == (False, None, "timeout")


def test_runs_are_isolated_from_each_other_and_from_the_parent():
    first = code("records.append(1)\nreturn 'a'")
    assert run_program(first, [], AS_OF, "city")[0]
    # each run gets fresh records and a fresh namespace
    second = code("return str(len(records))")
    assert run_program(second, [], AS_OF, "city") == (True, "0", "")
    # the fixed hash seed makes iteration order over sets of strings reproducible
    order = code("return ','.join(list({'alpha', 'beta', 'gamma', 'delta'}))")
    assert run_program(order, [], AS_OF, "city") == run_program(order, [], AS_OF, "city")


def test_parse_ts_reads_record_times_and_agrees_with_the_generator_format():
    assert sandbox.parse_ts("2026-01-15 12:30").minute == 30
    assert sandbox.parse_ts("2026-01-15T12:30") == gen_parse_ts("2026-01-15T12:30")
    assert sandbox.parse_ts("2026-01-15").hour == 0
    with pytest.raises(ValueError):
        sandbox.parse_ts("15 January 2026")


# ---------------------------------------------------------------------------- against the gold
CLOSE_FRIEND = """
now = parse_ts(as_of)
lo = now - timedelta(days={days})
counts = {{}}
for r in records:
    if r['source'] == 'messages' and lo <= parse_ts(r['sent']) <= now:
        counts[r['conversation_with']] = counts.get(r['conversation_with'], 0) + 1
return sorted(n for n, c in counts.items() if c >= {need})
"""
OVERDUE = """
today = as_of[:10]
return sorted(
    r['title'] for r in records
    if r['status'] == 'open' and r['due'] < today and (r['priority'] == 'high' or {v1})
)
"""


def test_a_correct_program_gives_the_evaluators_set(short_world):
    w, log = short_world
    checked = 0
    for t in (w.fork - timedelta(minutes=1), w.fork + timedelta(weeks=1), w.end):
        for version, (days, need) in {"v1": (30, 6), "v2": (14, 4)}.items():
            programs = {
                "close_friend": CLOSE_FRIEND.format(days=days, need=need),
                "overdue_task": OVERDUE.format(v1=str(version == "v1")),
            }
            for kid, body in programs.items():
                slot = w.slot(f"k.{kid}")
                ok, value, error = run_program(
                    code(body), program_records(log, slot, t), f"{t:%Y-%m-%d %H:%M}", slot.vtype
                )
                expected, _ = evaluate(kid, log.snapshot(t), t, version)
                assert ok, error
                assert value == sorted(expected), (kid, version, t)
                checked += 1
    assert checked == 12


def test_records_carry_ids_revision_updated_and_the_payload(short_world):
    w, log = short_world
    t = w.fork
    slot = w.slot("k.working_hours")  # sources: notes, calendar
    records = program_records(log, slot, t)
    assert {r["source"] for r in records} == set(slot.sources)
    assert [r["source"] for r in records] == sorted(
        (r["source"] for r in records), key=slot.sources.index
    )
    snap = log.snapshot(t)
    for r in records:
        o = snap.get(r["object_id"])
        assert o is not None and r["revision"] == o.revision
        assert r["updated"] == o.updated.strftime("%Y-%m-%d %H:%M")
        assert r["text"] == o.text and all(r[k] == v for k, v in o.payload.items())
    for source in slot.sources:
        stamps = [r["updated"] for r in records if r["source"] == source]
        assert stamps == sorted(stamps)
    # the cached JSON path equals the plain path
    assert json.loads(records_json(log, slot, t)) == records


def test_field_text_lists_exactly_the_payload_keys_of_every_source(short_world):
    w, _ = short_world
    written = {}
    for e in w.events:
        written.setdefault(e.source, set()).update(e.payload)
    everywhere = {"cluster", "text", "updated"}  # documented once, for every record
    for source in SOURCES:
        documented = set(re.findall(r"(\w+) \((?:str|int|bool|list)", SOURCE_FIELDS[source]))
        assert documented == written[source] - everywhere - {"kind"}, source
    assert "`kind`" in SOURCE_FIELDS["calendar"]
    common = COMMON_FIELDS
    assert all(f"{name} (" in common for name in ("object_id", "source", "revision", "updated"))
    assert "cluster (" in common and "text (" in common
    assert fields_text(("tasks", "notes")).count("(one record per") == 2


# ---------------------------------------------------------------------------- the request
def test_request_depends_only_on_item_type_definition_and_sources():
    a = generate_branch(11, "src", world_params(3))
    b = generate_branch(12, "src", world_params(3))
    t0 = a.fork - timedelta(minutes=1)
    for slot_id in ("k.close_friend", "k.vip_sender", "u.timezone", "u.diet", "u.employer"):
        assert program_request(a, a.slot(slot_id), t0) == program_request(b, b.slot(slot_id), t0)
        assert program_request(a, a.slot(slot_id), t0) == program_request(
            a, a.slot(slot_id), a.fork + timedelta(weeks=8)
        )  # no shift on this branch: same definition, same request


def test_request_changes_with_the_definition_version(short_world):
    w, _ = short_world
    t0, t1 = w.fork - timedelta(minutes=1), w.fork + timedelta(weeks=1)
    changed = [
        s
        for s in w.slots
        if s.kappa_id and w.kappa_version(s.kappa_id, t0) != w.kappa_version(s.kappa_id, t1)
    ]
    assert changed
    for s in changed:
        r0, r1 = program_request(w, s, t0), program_request(w, s, t1)
        assert r0 != r1
        stmt = w.kappa_defs[s.kappa_id][w.kappa_version(s.kappa_id, t1)].statement
        assert stmt in r1["messages"][1]["content"] and stmt not in r0["messages"][1]["content"]
    plain = next(s for s in w.slots if s.kappa_id is None)
    assert program_request(w, plain, t0) == program_request(w, plain, t1)


def test_request_shape_and_content(short_world):
    w, _ = short_world
    t = w.fork
    slot = w.slot("k.vip_sender")
    req = program_request(w, slot, t)
    assert req["purpose"] == "program" and req["schema"]["required"] == ["code"]
    system, user = (m["content"] for m in req["messages"])
    assert "Knowledge item: " + slot.description in user
    assert "Value format: a JSON array of full names" in user
    assert "Definition in effect (vip_sender v1): " in user
    assert "def derive(records, as_of)" in user and "list of strings" in user
    for source in slot.sources:
        assert source in user
    assert "notes (one record per note)" not in user  # only the declared sources are described
    scalar = program_request(w, w.slot("u.timezone"), t)["messages"][1]["content"]
    assert "Definition in effect" not in scalar and "or None if the records do not" in scalar
    # nothing of the world in the request: no person, company or city names
    everything = system + user + scalar
    assert w.user_name not in everything
    assert all(p.name not in everything for p in w.people.values())


# ---------------------------------------------------------------------------- response handling
def result(text: str) -> LLMResult:
    return LLMResult(text, 1, 1, 0.0, False, "stop", "k")


def test_extract_code():
    assert extract_code(result(json.dumps({"code": "def derive(r, a):\n    return None\n"}))) == (
        "def derive(r, a):\n    return None\n"
    )
    fenced = json.dumps({"code": "```python\ndef derive(r, a):\n    return None\n```"})
    assert extract_code(result(fenced)) == "def derive(r, a):\n    return None"
    assert extract_code(result("not json")) is None
    assert extract_code(result(json.dumps({"code": 3}))) is None
    assert extract_code(result(json.dumps(["code"]))) is None


def test_program_check_for_definition_items():
    def outcome(ok, value, error=""):
        return ProgramOutcome(ok, value, error, "key", 1, 1, 0.0)

    same = program_check("set_person", True, ["b", "A"], outcome(True, ["a", "b"]))
    assert same.ok  # set comparison is case- and order-insensitive, as in scoring
    assert not program_check("set_person", True, ["a"], outcome(True, ["a", "b"])).ok
    assert not program_check("set_person", False, None, outcome(True, ["a"])).ok
    failed = program_check("set_person", True, ["a"], outcome(False, None, "timeout"))
    assert not failed.ok and "timeout" in failed.reason
    assert program_check("timerange", True, None, outcome(True, None)).ok
    assert program_check("timerange", True, "09:00-17:00", outcome(True, "09:00-17:00")).ok
    assert not program_check("timerange", True, "09:00-17:00", outcome(True, None)).ok

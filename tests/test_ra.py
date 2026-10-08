"""Retraction-aware programs (the `_ra` cells): the sandbox enforces the declared reads, the request
is a new one, and the cell hands each item's lineage to the program that replaces it."""

import json
from datetime import timedelta

import pytest

from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_cell
from memshift.gen.gold import gold_at
from memshift.gen.model import fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem import sandbox
from memshift.holdmem.program import (
    locators_text,
    program_request,
    records_json,
    synthesise_ra,
)
from memshift.holdmem.system import CellConfig
from memshift.llm import LLMResult
from tests.e1_replay import replay
from tests.fakes import RecordingProgramLLM
from tests.test_program import code as legacy_code

AS_OF = "2026-01-15 12:00"
HEAD = "def derive(records, as_of, prior_reads):\n"


def ra_code(body: str) -> str:
    return HEAD + "".join(f"    {line}\n" for line in body.splitlines())


def rec(oid, source, **fields):
    return {
        "object_id": oid,
        "source": source,
        "revision": 1,
        "updated": "2026-01-01 09:00",
        "cluster": oid,
        **fields,
    }


RECORDS = [
    rec("contacts:p1", "contacts", name="Ann Lee", city="Rome", employer="Acme", text="Ann"),
    rec("contacts:p2", "contacts", name="Bo Chen", city="Oslo", employer="Bolt", text="Bo"),
    rec("email:e1", "email", from_addr="a@x", sent="2026-01-02 10:00", body="I moved to Paris"),
]


def run_ra(body, records=RECORDS, prior=()):
    return sandbox.run_source_ra(ra_code(body), json.dumps(records), AS_OF, list(prior))


FIND_ANN = "ann = [r for r in records if r['source'] == 'contacts' and r['name'] == 'Ann Lee']\n"


# ---------------------------------------------------------------------------- the sandbox
def test_a_program_that_declares_what_it_reads_runs():
    ok, value, reads, error = run_ra(
        FIND_ANN + "return {'value': ann[0]['city'], 'reads': [ann[0]['object_id']]}"
    )
    assert (ok, value, reads, error) == (True, "Rome", ["contacts:p1"], "")


def test_locating_a_record_reads_no_content_but_using_it_does():
    # the name and the identity fields of every card can be read to find the right one
    scan = "[r['name'] for r in records if r['source'] == 'contacts']\n"
    assert run_ra(scan + "return {'value': None, 'reads': []}")[0] is True
    ok, _, _, error = run_ra(FIND_ANN + "return {'value': ann[0]['city'], 'reads': []}")
    assert not ok and error == "undeclared read: contacts:p1"


def test_a_record_that_is_not_declared_cannot_be_read():
    body = FIND_ANN + "return {'value': records[2]['body'], 'reads': [ann[0]['object_id']]}"
    ok, value, reads, error = run_ra(body)
    assert (ok, value, reads) == (False, None, []) and error == "undeclared read: email:e1"


@pytest.mark.parametrize(
    "read",
    [
        "records[1].get('city')",
        "list(records[1].items())",
        "list(records[1].values())",
        "dict(records[1])",
        "{**records[1]}",
        "records[1].copy()",
        "records[1]['employer']",
    ],
)
def test_every_way_of_reading_a_content_field_is_tracked(read):
    ok, _, _, error = run_ra(f"_x = {read}\nreturn {{'value': None, 'reads': []}}")
    assert not ok and error == "undeclared read: contacts:p2", read
    declared = f"_x = {read}\nreturn {{'value': None, 'reads': ['contacts:p2']}}"
    assert run_ra(declared)[0] is True, read


def test_the_field_names_of_a_record_are_free():
    body = (
        "k = [list(r) for r in records]\nreturn {'value': str('city' in records[0]), 'reads': []}"
    )
    assert run_ra(body)[:2] == (True, "True")


def test_the_answer_must_be_a_value_and_a_list_of_ids():
    for bad in (
        "return 'Rome'",
        "return {'value': 'Rome'}",
        "return {'value': 'Rome', 'reads': [], 'extra': 1}",
        "return {'value': 'Rome', 'reads': 'contacts:p1'}",
        "return {'value': 'Rome', 'reads': [1]}",
        "return {'value': 3, 'reads': []}",
    ):
        ok, _, _, error = run_ra(bad)
        assert not ok and error.startswith("type mismatch"), bad


def test_the_signature_follows_the_mode():
    assert sandbox.check_source(ra_code("return {'value': None, 'reads': []}"), ra=True) is None
    assert "prior_reads" in sandbox.check_source(legacy_code("return None"), ra=True)
    assert (
        sandbox.check_source(ra_code("return None")) == "derive must take exactly (records, as_of)"
    )
    # the legacy runner is what it was: two arguments, a plain value
    assert sandbox.run_source(legacy_code("return records[1]['city']"), json.dumps(RECORDS), AS_OF)[
        :2
    ] == (True, "Oslo")


def test_the_program_receives_the_prior_reads_and_is_still_timed():
    out = run_ra("return {'value': ', '.join(prior_reads), 'reads': []}", prior=["a", "b"])
    assert out[:2] == (True, "a, b")
    ok, _, _, error = sandbox.run_source_ra(
        ra_code("while True:\n    pass"), "[]", AS_OF, [], timeout=0.5
    )
    assert not ok and error == "timeout"


# ---------------------------------------------------------------------------- the request
def test_the_ra_request_is_a_new_request_with_fixed_text():
    a = generate_branch(11, "def", world_params(3))
    b = generate_branch(5, "def", world_params(3))
    slot_a, slot_b = a.slot("k.close_friend"), b.slot("k.close_friend")
    plain = program_request(a, slot_a, a.fork)
    ra = program_request(a, slot_a, a.fork, ra=True)
    assert plain == program_request(a, slot_a, a.fork, ra=False)  # the D1s request is untouched
    assert ra != plain and ra["purpose"] == "program" and ra["schema"] == plain["schema"]
    text = ra["messages"][-1]["content"]
    assert "def derive(records, as_of, prior_reads)" in text and "`reads`" in text
    assert "deleted" in text and "must return None" in text
    assert locators_text(slot_a.sources) in text
    assert "messages: conversation_with, contact" in text and "contacts: name" not in text
    # no world data: the same item and definition version give the same request in any world
    before = a.fork - timedelta(minutes=1)  # definition v1 in both
    assert program_request(a, slot_a, before, ra=True) == program_request(
        b, slot_b, b.fork - timedelta(minutes=1), ra=True
    )
    scalar = program_request(a, a.slot("u.timezone"), a.fork, ra=True)["messages"][-1]["content"]
    assert "string in the value format above, or None" in scalar


# ---------------------------------------------------------------------------- the cell
ORPHAN = generate_branch(11, "orphan", world_params(3))
TARGETS = set(ORPHAN.params["shift_targets"])
CONTACT_SLOTS = [s for s in ORPHAN.slots if s.kind == "contact_attr"]
CELL = CellConfig(1, 1, "source", synth=True, ra=True)
CELL0 = CellConfig(1, 0, synth=True, ra=True)
LEGACY = CellConfig(1, 1, "source", synth=True)

RA_SCALAR = "return {'value': None, 'reads': []}"
RA_SET = "return {'value': [], 'reads': []}"


def contact_program(slot, find_missing):
    name = ORPHAN.people[slot.subject].name
    field = slot.slot_id.split(".")[-1]
    return ra_code(
        f"live = {{r['object_id'] for r in records}}\n"
        f"cards = [r for r in records if r['source'] == 'contacts' and r['name'] == {name!r}]\n"
        + find_missing
        + f"return {{'value': cards[0]['{field}'], 'reads': [cards[0]['object_id']]}}"
    )


AWARE = (
    "if any(i not in live for i in prior_reads):\n    return {'value': None, 'reads': []}\n"
    "if not cards:\n    return {'value': None, 'reads': []}\n"
)
# substitutes another card for the missing one, and declares it
UNAWARE = "if not cards:\n    cards = [r for r in records if r['source'] == 'contacts'][:1]\n"
# substitutes another card without declaring it
UNDECLARED = (
    "if not cards:\n    return {'value': [r for r in records if r['source'] == 'contacts']"
    "[0]['city'], 'reads': []}\n"
)


def llm_with(find_missing):
    by_item = {s.description: contact_program(s, find_missing) for s in CONTACT_SLOTS}

    def default(prompt):
        if "prior_reads" not in prompt:
            return None  # a legacy (D1s) request: the fake program
        for description, program in by_item.items():
            if f"Knowledge item: {description}\n" in prompt:
                return program
        return ra_code(RA_SET if "[] if there are none" in prompt else RA_SCALAR)

    return lambda: RecordingProgramLLM(default=default)


@pytest.fixture(scope="module")
def aware():
    return replay([CELL, CELL0, LEGACY, CellConfig(0, 1, "source")], "orphan", 11, (1, 2),
                  make_llm=llm_with(AWARE))  # fmt: skip


def card(slot):
    return f"contacts:{slot.subject}"


def test_tokens_and_names():
    assert CELL.name == "holdmem_D1sB1_source_ra" and CELL0.name == "holdmem_D1sB0_ra"
    assert parse_cell("D1sB1_source_ra") == CELL and parse_cell("D1sB0_ra") == CELL0
    assert parse_cell("D1sB1_cited_ra") == CellConfig(1, 1, "cited", synth=True, ra=True)
    with pytest.raises(ValueError):
        CellConfig(1, 1, "source", ra=True)  # the retraction-aware programs are D1s programs
    with pytest.raises(ValueError):
        parse_cell("D1B1_source_ra")


def test_the_writer_requests_are_those_of_the_d1s_cell_and_the_program_requests_are_new(aware):
    ra, legacy = aware.llms[CELL.name], aware.llms[LEGACY.name]
    assert ra.keys("write") == legacy.keys("write")
    assert ra.keys("program") and ra.keys("program").isdisjoint(legacy.keys("program"))
    # one program request per item (and definition version), however often it is re-derived
    n_items = len(aware.world.slots)
    assert len(ra.keys("program")) == n_items
    assert aware.llms[CELL0.name].keys("program") == ra.keys("program")
    assert len(ra.requests) > len(ra.keys())  # re-derivations repeat requests the cache serves


def test_pre_shift_bytes_are_those_of_every_other_cell(aware):
    first = aware.contexts_t0[CELL.name]
    assert first and all(ctx == first for ctx in aware.contexts_t0.values())


def test_a_program_that_returns_no_value_for_a_retracted_record_repairs_the_orphan(aware):
    hm = aware.cells[CELL.name]
    w, log, t = aware.world, aware.log, aware.at(1)
    n_target = n_kept = 0
    for slot in CONTACT_SLOTS:
        item = hm.items[f"{slot.slot_id}#0"]
        assert item.det is not None and card(slot) in item.det["program_lineage"]
        gold = gold_at(w, log, slot, aware.at(2))
        if card(slot) in TARGETS:
            n_target += 1
            assert gold.status == "abstain"
            assert item.eval_value is None and item.det["program_lineage_ok"] is True
            assert "no current value" in item.s_k and item.status == "valid"
        else:
            n_kept += 1
            assert item.eval_value == gold.value and item.det["program_lineage_ok"] is True
    assert n_target > 0 and n_kept > 0
    assert log.snapshot(t).get(card(next(s for s in CONTACT_SLOTS if card(s) in TARGETS))) is None


def test_the_lineage_outlives_the_run_that_returned_no_value(aware):
    """The program returned reads [] once its record was gone; the layer keeps the id it declared
    at t0 and hands it to the next run, so the item stays withdrawn."""
    hm = aware.cells[CELL.name]
    for slot in (s for s in CONTACT_SLOTS if card(s) in TARGETS):
        item = hm.items[f"{slot.slot_id}#0"]
        assert item.det["program_lineage"] == [card(slot)]
        assert item.det["program_n_reads"] == 0  # the last run declared nothing
        notes = [h["program"] for h in item.history if h["event"] in ("derive", "repair")]
        assert notes[0]["value"] is not None
        assert [n["value"] for n in notes[1:]] == [None, None]  # k=1, k=2
        assert [n["n_reads"] for n in notes] == [1, 0, 0]
        assert all(n["lineage_ok"] is not False for n in notes)


def test_a_program_that_substitutes_another_record_is_measured_not_stopped():
    rep = replay([CELL], "orphan", 11, (1,), make_llm=llm_with(UNAWARE))
    hm = rep.cells[CELL.name]
    for slot in (s for s in CONTACT_SLOTS if card(s) in TARGETS):
        item = hm.items[f"{slot.slot_id}#0"]
        assert item.det["program_lineage_ok"] is False  # a prior read is gone, a value came back
        assert item.eval_value is not None and "no current value" not in item.s_k
        assert card(slot) in item.det["program_lineage"] and len(item.det["program_lineage"]) == 2
    kept = next(s for s in CONTACT_SLOTS if card(s) not in TARGETS)
    assert hm.items[f"{kept.slot_id}#0"].det["program_lineage_ok"] is True


def test_a_program_that_reads_a_record_it_did_not_declare_fails_the_item():
    rep = replay([CELL], "orphan", 11, (1,), make_llm=llm_with(UNDECLARED))
    hm = rep.cells[CELL.name]
    for slot in (s for s in CONTACT_SLOTS if card(s) in TARGETS):
        item = hm.items[f"{slot.slot_id}#0"]
        assert item.status == "unverifiable" and item.det["program_ok"] is False
        assert "program failed: undeclared read" in item.det["reason"]
        assert item.det["program_lineage"] == [card(slot)]  # a failed run keeps the lineage


def test_the_b0_cell_passes_the_lineage_of_the_latest_item(aware):
    hm = aware.cells[CELL0.name]
    slot = next(s for s in CONTACT_SLOTS if card(s) in TARGETS)
    ids = hm.by_slot[slot.slot_id]
    assert len(ids) > 1
    lineages = [hm.items[i].det["program_lineage"] for i in ids]
    assert all(card(slot) in lin for lin in lineages) and ids[0].endswith("#0")
    for i in ids[1:]:  # an appended item comes from a run that was given the lineage before it
        assert hm.items[i].history[0]["event"] == "append"
        assert hm.items[i].det["program_lineage_ok"] is True
    # nothing was invalidated: the old item is served beside whatever was appended
    assert not any(h["event"] == "invalidate" for i in ids for h in hm.items[i].history)


def test_synthesise_ra_reports_a_response_without_code():
    w = ORPHAN
    log = SourceLog(w.events)

    class NoCode(RecordingProgramLLM):
        def complete(self, messages, max_tokens=512, schema=None, purpose=""):
            return LLMResult("not json", 5, 5, 0.0, False, "stop", "k")

    out = synthesise_ra(NoCode(), w, log, w.slots[0], w.fork, ["x:1"])
    assert not out.ok and out.error == "no code in the response"
    assert out.lineage == ["x:1"] and out.reads == [] and out.lineage_ok is None


def test_programs_run_on_the_same_live_records_as_the_d1s_programs():
    w = ORPHAN
    log = SourceLog(w.events)
    slot = CONTACT_SLOTS[0]
    t = w.fork + timedelta(weeks=1)
    text = records_json(log, slot, t)
    assert fmt_ts(t) and text.startswith("[") and '"object_id"' in text
    # the deleted card is simply absent from the live records
    deleted = next(s for s in CONTACT_SLOTS if card(s) in TARGETS)
    assert card(deleted) not in records_json(log, deleted, t)
    assert card(deleted) in records_json(log, deleted, w.fork - timedelta(minutes=1))


def test_a_withdrawn_value_is_none_for_a_set_item_too():
    w = ORPHAN
    log = SourceLog(w.events)
    slot = w.slot("k.close_friend")
    withdrawn = ra_code("return {'value': None, 'reads': []}")
    llm = RecordingProgramLLM(default=lambda prompt: withdrawn)
    out = synthesise_ra(llm, w, log, slot, w.fork, ["messages:m1"])
    assert out.ok and out.value is None and out.lineage == ["messages:m1"]
    assert out.lineage_ok is True  # the prior read is gone (there is no such record) and no value
    empty = ra_code("return {'value': [], 'reads': []}")
    llm = RecordingProgramLLM(default=lambda prompt: empty)
    # the empty set is a value, not a withdrawal
    assert synthesise_ra(llm, w, log, slot, w.fork, ["messages:m1"]).lineage_ok is False
    llm = RecordingProgramLLM(default=lambda prompt: ra_code("return {'value': 'x', 'reads': []}"))
    bad = synthesise_ra(llm, w, log, slot, w.fork, [])
    assert not bad.ok and "set_person needs a list" in bad.error

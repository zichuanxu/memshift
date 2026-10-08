"""The D1s cell (prereg v2 §4): D=1 whose replacement values are the writer's own programs."""

import json
from datetime import timedelta

import pytest

from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_cell
from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.llm import LLMResult
from memshift.probes import make_probes
from memshift.reader import context_hash
from memshift.textutil import sha256_json
from tests.fakes import FakeLLM
from tests.test_program import CLOSE_FRIEND, code

ACTIVE_V1 = "return sorted({r['project'] for r in records if r['status'] == 'open'})"
ACTIVE_V2 = (
    "lo = parse_ts(as_of) - timedelta(days=14)\n"
    "return sorted({r['project'] for r in records if parse_ts(r['updated']) >= lo})"
)
BROKEN = "import os\ndef derive(records, as_of):\n    return None\n"
D1S = [CellConfig(1, 0, synth=True), CellConfig(1, 1, "source", synth=True)]


class ProgramLLM(FakeLLM):
    """FakeLLM whose program answers are chosen by a marker in the request text."""

    def __init__(self, programs=None, default=None):
        super().__init__()
        self.programs = programs or {}
        self.default = default

    def complete(self, messages, max_tokens=512, schema=None, purpose=""):
        if purpose == "program":
            prompt = messages[-1]["content"]
            src = next((c for marker, c in self.programs.items() if marker in prompt), self.default)
            if src is not None:
                self.usage["calls"] += 1
                key = sha256_json(list(messages))
                return LLMResult(json.dumps({"code": src}), 100, 20, 0.0, False, "stop", key)
        return super().complete(messages, max_tokens, schema, purpose)


CORRECT = {
    "(active_project v1)": code(ACTIVE_V1),
    "(active_project v2)": code(ACTIVE_V2),
    "(close_friend v1)": code(CLOSE_FRIEND.format(days=30, need=6)),
}


@pytest.fixture(scope="module")
def world_log():
    w = generate_branch(11, "def", world_params(3))
    return w, SourceLog(w.events)


def _t0(w):
    return w.fork - timedelta(minutes=1)


def _built(world, log, cfg, llm=None):
    hm = HoldMem(world, log, cfg, llm or FakeLLM(), "fake/writer-a")
    hm.workers = 8
    hm.build(_t0(world))
    return hm


def _last(hm, slot):
    return hm.items[hm.by_slot[slot.slot_id][-1]]


def test_names_and_validation():
    assert CellConfig(1, 0, synth=True).name == "holdmem_D1sB0"
    assert CellConfig(1, 1, "source", synth=True).name == "holdmem_D1sB1_source"
    assert CellConfig(1, 1, "cited", synth=True).name == "holdmem_D1sB1_cited"
    assert CellConfig(1, 1, "cited").name == "holdmem_D1B1_cited"  # unchanged
    with pytest.raises(ValueError):
        CellConfig(0, 1, "source", synth=True)
    with pytest.raises(ValueError):
        CellConfig(1, 1, "source", fallback=False, synth=True)


def test_parse_cell_tokens():
    assert parse_cell("D1sB0") == CellConfig(1, 0, synth=True)
    assert parse_cell("D1sB1_source") == CellConfig(1, 1, "source", synth=True)
    assert parse_cell("D1sB1_cited") == CellConfig(1, 1, "cited", synth=True)
    assert parse_cell("D1sB1") == CellConfig(1, 1, "source", synth=True)
    assert parse_cell("D1B1_source") == CellConfig(1, 1, "source")
    assert parse_cell("D1B0_nofb") == CellConfig(1, 0, "source", fallback=False)
    assert parse_cell("D1B1_cited_nofb") == CellConfig(1, 1, "cited", fallback=False)
    for bad in ("D1s", "D2sX", "D1sB1_", "holdmem_D1sB0", "D1sB1_source_x"):
        with pytest.raises(ValueError):
            parse_cell(bad)
    with pytest.raises(ValueError):
        parse_cell("D1sB1_source_nofb")  # no checker-only variant of D1s
    with pytest.raises(ValueError):
        parse_cell("D0sB0")


def test_pre_intervention_bytes_identical_across_all_cells(world_log):
    w, log = world_log
    cfgs = [
        CellConfig(d, b, g) for d in (0, 1) for b, g in ((0, "source"), (1, "source"), (1, "cited"))
    ]
    cfgs += [
        CellConfig(1, b, g, synth=True) for b, g in ((0, "source"), (1, "source"), (1, "cited"))
    ]
    cells = [_built(w, log, c, ProgramLLM(CORRECT)) for c in cfgs]
    for p in make_probes(w, log, [_t0(w)]):
        assert len({context_hash(hm.serve(p).blocks) for hm in cells}) == 1, p.fact_id
    for hm in cells:
        for item in hm.items.values():
            assert item.status == "valid"  # nothing is substituted or dropped before the shift
            det = item.det or {}
            assert ("program_ok" in det) == hm.cfg.synth, hm.name
            if hm.cfg.synth:
                assert det["check"] in ("pass", "fail")
                assert set(det) >= {"parse", "spans", "check", "reason", "program_value"}
                assert det["program_key"] and det["program_ok"]


def test_kappa_check_is_agreement_with_the_writers_own_program(world_log):
    w, log = world_log
    hm = _built(w, log, D1S[0], ProgramLLM(CORRECT))
    item = _last(hm, w.slot("k.close_friend"))
    det = item.det
    assert det is not None
    gold = gold_at(w, log, w.slot("k.close_friend"), _t0(w)).value
    assert det["program_ok"] and det["program_value"] == gold  # a correct program
    assert det["parse"] == "X"  # the fake writer's own value
    assert (
        det["check"] == "fail" and det["reason"] == "writer value differs from its program's output"
    )
    assert det["spans"] == []  # the check does not look at citations for definition items


def test_no_benchmark_authored_logic_is_reached(world_log, monkeypatch):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)

    def boom(*args, **kwargs):
        raise AssertionError("benchmark-authored logic was called")

    monkeypatch.setattr("memshift.holdmem.checker.evaluate", boom)
    monkeypatch.setattr("memshift.holdmem.system.fallback", boom)
    for cfg in D1S:
        hm = _built(w, log, cfg)
        hm.maintain(t)  # runs the whole repair path
    with pytest.raises(AssertionError):
        _built(w, log, CellConfig(1, 1, "source"))  # the D=1 checker calls the gold evaluator


def test_program_output_replaces_the_value_after_the_shift(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    llm = ProgramLLM(CORRECT)
    hm = _built(w, log, CellConfig(1, 1, "source", synth=True), llm)
    before = {s.slot_id: _last(hm, s).det["program_key"] for s in w.slots}
    stats = hm.maintain(t)
    assert stats["invalidated"] > 0
    seen = set()
    for slot_id in ("k.active_project", "k.close_friend"):
        slot = w.slot(slot_id)
        item = _last(hm, slot)
        events = [h["event"] for h in item.history]
        if "repair" not in events:
            continue
        seen.add(slot_id)
        gold = gold_at(w, log, slot, t)
        assert item.status == "valid" and item.det["check"] == "program"
        assert item.eval_value == gold.value and item.det["parse"] == gold.value
        assert item.det["program_ok"] and item.det["program_value"] == gold.value
        assert "computed by the writer's own program" in item.s_k
        assert "programmatic extraction" not in item.s_k
        assert item.refs == [] and item.det["spans"] == []
        assert item.history[-1]["program"] == {"ok": True, "value": gold.value}
        assert item.history[0]["program"]["ok"]  # the t0 program is kept in the history
        version = w.kappa_version(slot.kappa_id, t)
        assert f"{slot.kappa_id} {version}: " in item.s_k
        # a source change re-runs the same request (a cache hit); a new definition is a new one
        same = w.kappa_version(slot.kappa_id, t) == w.kappa_version(slot.kappa_id, _t0(w))
        assert (item.det["program_key"] == before[slot_id]) == same
    assert seen == {"k.active_project", "k.close_friend"}
    assert w.kappa_version("active_project", t) == "v2"  # the def shift is in force


def test_failing_programs_make_repaired_items_unverifiable_not_wrong(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    hm = _built(w, log, CellConfig(1, 1, "source", synth=True), ProgramLLM(default=BROKEN))
    assert all(i.status == "valid" for i in hm.items.values())  # the shared snapshot is intact
    assert not any(i.det["program_ok"] for i in hm.items.values())
    stats = hm.maintain(t)
    assert stats["unverifiable"] > 0
    for slot in w.slots:
        item = _last(hm, slot)
        if not any(h["event"] == "repair" for h in item.history):
            continue
        assert item.det["program_ok"] is False and item.det["program_value"] is None
        if item.status == "unverifiable":
            assert item.det["check"] == "unverifiable"
            assert "program failed" in item.det["reason"]
            assert "Import" in item.det["reason"]
        else:
            assert item.det["check"] == "pass"  # the span check passed, the program was not needed
    unserved = [i for i in hm.items.values() if i.status == "unverifiable"]
    assert unserved
    for p in make_probes(w, log, [t]):
        served = hm.serve(p).blocks
        assert not any(i.s_k in served for i in unserved)  # an unverifiable item is never served


def test_a_response_without_code_is_a_program_failure(world_log):
    w, log = world_log

    class NoCode(FakeLLM):
        def complete(self, messages, max_tokens=512, schema=None, purpose=""):
            if purpose == "program":
                return LLMResult("not json", 7, 3, 0.0, False, "stop", "k")
            return super().complete(messages, max_tokens, schema, purpose)

    hm = _built(w, log, D1S[0], NoCode())
    det = _last(hm, w.slot("k.close_friend")).det
    assert det["program_ok"] is False and det["check"] == "fail"
    assert det["reason"] == "program failed: no code in the response"


def test_detection_is_the_same_as_in_the_d1_cell(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    for b in (0, 1):
        d1 = _built(w, log, CellConfig(1, b, "source"))
        d1s = _built(w, log, CellConfig(1, b, "source", synth=True))
        assert d1.maintain(t)["invalidated"] == d1s.maintain(t)["invalidated"]
        for slot in w.slots:
            a, b_ = d1.by_slot[slot.slot_id], d1s.by_slot[slot.slot_id]
            assert len(a) == len(b_), slot.slot_id
            for ia, ib in zip(a, b_, strict=True):
                x, y = d1.items[ia], d1s.items[ib]
                assert x.bind == y.bind and x.readset == y.readset
                assert [h["event"] for h in x.history] == [h["event"] for h in y.history]
                assert not any("program" in h for h in x.history)
                assert all("program" in h for h in y.history if h["event"] != "invalidate")


def test_program_request_is_one_more_writer_call_per_item_write(world_log):
    w, log = world_log
    d1, d1s = FakeLLM(), FakeLLM()
    a = _built(w, log, CellConfig(1, 0), d1)
    b = _built(w, log, CellConfig(1, 0, synth=True), d1s)
    n = len(w.slots)
    assert d1.usage["calls"] == n and d1s.usage["calls"] == 2 * n
    assert b.cost["ingest_tokens"] == 2 * a.cost["ingest_tokens"] == n * 240
    assert b.cost["verify_cpu_s"] > 0
    t = w.fork + timedelta(weeks=1)
    a.maintain(t)
    b.maintain(t)
    assert b.cost["maintain_tokens"] == 2 * a.cost["maintain_tokens"] > 0


def test_append_only_cell_keeps_the_writer_and_program_side_by_side(world_log):
    w, log = world_log
    hm = _built(w, log, D1S[0], ProgramLLM(CORRECT))
    stats = hm.maintain(w.fork + timedelta(weeks=1))
    assert stats["invalidated"] == 0 and stats["appended"] > 0
    for slot in w.slots:
        ids = hm.by_slot[slot.slot_id]
        for item_id in ids[1:]:  # appended items come from the delta and carry their own program
            item = hm.items[item_id]
            assert item.history[0]["event"] == "append" and "program" in item.history[0]
            assert item.det["program_key"]

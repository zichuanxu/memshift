from datetime import datetime, timedelta

import pytest

from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import check, fallback
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import Citation
from memshift.probes import make_probes
from memshift.reader import context_hash
from memshift.retrieval import Doc, pack
from memshift.textutil import approx_tokens
from tests.fakes import FakeLLM

CELLS = [
    CellConfig(d, b, g) for d in (0, 1) for b, g in ((0, "source"), (1, "source"), (1, "cluster"))
]


@pytest.fixture(scope="module")
def world_log():
    w = generate_branch(11, "src")
    return w, SourceLog(w.events)


def _built(world, log, cfg, llm=None):
    hm = HoldMem(world, log, cfg, llm or FakeLLM(), "fake/writer-a")
    hm.workers = 1
    hm.build(world.fork)
    return hm


def test_pre_intervention_bytes_identical(world_log):
    w, log = world_log
    cells = [_built(w, log, c) for c in CELLS]
    for p in make_probes(w, log, [w.fork]):
        hashes = {context_hash(hm.serve(p).blocks) for hm in cells}
        assert len(hashes) == 1, p.fact_id


def test_b1_invalidates_b0_does_not(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    b1 = _built(w, log, CellConfig(1, 1, "source"))
    b0 = _built(w, log, CellConfig(1, 0, "source"))
    s1 = b1.maintain(t)
    s0 = b0.maintain(t)
    assert s1["invalidated"] > 0 and s1["repaired"] == s1["invalidated"]
    assert s0["invalidated"] == 0
    assert all(i.history[0]["event"] in ("derive", "append") for i in b0.items.values())
    assert not any(h["event"] == "invalidate" for i in b0.items.values() for h in i.history)


def test_cluster_granularity_invalidates_less(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    src = _built(w, log, CellConfig(1, 1, "source")).maintain(t)
    clu = _built(w, log, CellConfig(1, 1, "cluster")).maintain(t)
    assert clu["invalidated"] <= src["invalidated"]


def test_writer_switch_invalidates_every_b1_item(world_log):
    w, log = world_log
    hm = _built(w, log, CellConfig(0, 1, "source"))
    hm.set_writer(FakeLLM("fake/writer-b"), "fake/writer-b")
    assert hm.maintain(w.fork)["invalidated"] == len(w.slots)
    b0 = _built(w, log, CellConfig(0, 0, "source"))
    b0.set_writer(FakeLLM("fake/writer-b"), "fake/writer-b")
    assert b0.maintain(w.fork)["invalidated"] == 0


def test_d1_repair_uses_fallback_on_structured_slots(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    hm = _built(w, log, CellConfig(1, 1, "source"))
    hm.maintain(t)
    repaired = 0
    for slot in w.slots:
        item = hm.items[hm.by_slot[slot.slot_id][-1]]
        if not any(h["event"] == "repair" for h in item.history):
            continue
        repaired += 1
        if slot.structured:
            assert item.det is not None and item.det["check"] == "fallback"
            assert item.eval_value == gold_at(w, log, slot, t).value
        else:
            assert item.status == "unverifiable"
    assert repaired > 0


def test_d0_never_parses(world_log):
    w, log = world_log
    hm = _built(w, log, CellConfig(0, 1, "source"))
    hm.maintain(w.fork + timedelta(weeks=1))
    assert all(i.det is None for i in hm.items.values())


def test_checker_accepts_real_quote_and_rejects_fabrication(world_log):
    w, log = world_log
    slot = next(s for s in w.slots if s.kind == "contact_attr" and s.slot_id.endswith(".city"))
    card = log.snapshot(w.fork).get(f"contacts:{slot.subject}")
    assert card is not None
    city = card.payload["city"]
    good = [Citation(card.object_id, card.revision, f"City: {city}")]
    assert check(w, log, slot, w.fork, city, good).ok
    bad = [Citation(card.object_id, card.revision, "City: Atlantis")]
    assert not check(w, log, slot, w.fork, city, bad).ok
    stale = [Citation(card.object_id, card.revision + 5, f"City: {city}")]
    assert not check(w, log, slot, w.fork, city, stale).ok
    fb = fallback(w, log, slot, w.fork)
    assert isinstance(fb, tuple) and fb[0] == city


def test_pack_respects_budget():
    docs = [Doc(f"d{i}", "word " * 300, datetime(2026, 1, 1 + i)) for i in range(20)]
    chosen = pack(docs, budget=1000, per_doc_cap=200)
    assert sum(approx_tokens(t) + 8 for _, t in chosen) <= 1000
    assert [d.ts for d, _ in chosen] == sorted(d.ts for d, _ in chosen)


def test_locate_quote_accepts_skipped_lines_and_ellipses():
    from memshift.holdmem.checker import locate_quote

    text = "Contact card: Ann Lee\nEmail: a@x\nPhone: +1\nCompany: Acme\nCity: Boston"
    assert locate_quote("Contact card: Ann Lee\nCompany: Acme", text) is not None
    assert locate_quote("Company: Acme ... City: Boston", text) is not None
    assert locate_quote("City: Boston\nCompany: Acme", text) is None  # order matters
    assert locate_quote("City: Denver", text) is None


def test_cited_granularity_detection_is_shared_by_both_d_arms(world_log):
    w, log = world_log
    t = w.fork + timedelta(weeks=1)
    d0 = _built(w, log, CellConfig(0, 1, "cited"))
    d1 = _built(w, log, CellConfig(1, 1, "cited"))
    assert d0.maintain(t)["invalidated"] == d1.maintain(t)["invalidated"]
    for slot in w.slots:
        h0 = [h["event"] for h in d0.items[f"{slot.slot_id}#0"].history]
        h1 = [h["event"] for h in d1.items[f"{slot.slot_id}#0"].history]
        assert h0 == h1, slot.slot_id


def test_empty_delta_makes_no_writer_call(world_log):
    w, log = world_log
    llm = FakeLLM()
    hm = _built(w, log, CellConfig(0, 0, "source"), llm)
    calls = llm.usage["calls"]
    assert hm.maintain(hm.last_t)["appended"] == 0  # delta (t, t] is empty
    assert llm.usage["calls"] == calls

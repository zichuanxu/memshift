"""zero_Dm, the query-time executable view: the zero system plus the writer's D1s program run on
the live records at the probe's time. The program requests are those of the D1s cells."""

from datetime import timedelta

import pytest

from memshift.adapters.baselines import ZeroDm, ZeroSystem
from memshift.experiments.common import world_params
from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import render_fallback_sk
from memshift.holdmem.system import (
    PROGRAM_ORIGIN,
    READER_BUDGET,
    READER_DOC_CAP,
    CellConfig,
    HoldMem,
    definition_text,
)
from memshift.probes import make_probe, make_probes
from memshift.retrieval import pack
from memshift.textutil import approx_tokens
from tests.fakes import RecordingProgramLLM
from tests.test_program import CLOSE_FRIEND, code
from tests.test_synth import ACTIVE_V1, ACTIVE_V2, BROKEN

PROGRAMS = {
    "(active_project v1)": code(ACTIVE_V1),
    "(active_project v2)": code(ACTIVE_V2),
    "(close_friend v1)": code(CLOSE_FRIEND.format(days=30, need=6)),
    "(close_friend v2)": code(CLOSE_FRIEND.format(days=14, need=4)),
}
D1S = CellConfig(1, 1, "source", synth=True)


@pytest.fixture(scope="module")
def world_log():
    w = generate_branch(11, "def", world_params(3))
    return w, SourceLog(w.events)


def _times(w):
    return [w.fork - timedelta(minutes=1), w.fork + timedelta(weeks=1)]


def test_the_view_is_the_programs_value_in_the_d1_template_ahead_of_the_evidence(world_log):
    w, log = world_log
    zero = ZeroDm(w, log, RecordingProgramLLM(PROGRAMS))
    plain = ZeroSystem(w, log)
    slot = w.slot("k.close_friend")
    for t in _times(w):
        probe = make_probe(w, log, slot, t)
        gold = gold_at(w, log, slot, t)
        res = zero.serve(probe)
        expected = render_fallback_sk(
            slot, t, gold.value, [], definition_text(w, slot, t), origin=PROGRAM_ORIGIN
        )
        assert res.blocks[0] == expected  # the program is right (it is the gold rule)
        assert "under the definition in effect" in res.blocks[0]
        # the evidence is the zero system's BM25 ranking, packed into the budget the view leaves
        left = READER_BUDGET - (approx_tokens(expected) + 8)
        ranked = plain.ranked(probe)
        assert res.blocks[1:] == [text for _, text in pack(ranked, left, READER_DOC_CAP)]
        assert res.blocks[1:] and not res.blocked


def test_the_view_follows_the_live_records_with_no_maintenance(world_log):
    """No stored state: the same object answers at the next time from the records then live."""
    w, log = world_log
    zero = ZeroDm(w, log, RecordingProgramLLM(PROGRAMS))
    slot = w.slot("k.active_project")
    views = []
    for t in _times(w):
        probe = make_probe(w, log, slot, t)
        views.append(zero.serve(probe).blocks[0])
        assert views[-1] == render_fallback_sk(
            slot,
            t,
            gold_at(w, log, slot, t).value,
            [],
            definition_text(w, slot, t),
            origin=PROGRAM_ORIGIN,
        )
    assert views[0] != views[1]
    assert not hasattr(zero, "items") and zero.cost["storage_bytes"] == 0
    assert zero.cost["verify_cpu_s"] > 0


def test_program_requests_are_those_of_the_d1s_cells(world_log):
    w, log = world_log
    t0, t1 = _times(w)
    d1s_llm = RecordingProgramLLM(PROGRAMS)
    d1s = HoldMem(w, log, D1S, d1s_llm, "fake/writer-a")
    d1s.workers = 4
    d1s.build(t0)
    d1s.maintain(t1)
    zero_llm = RecordingProgramLLM(PROGRAMS)
    zero = ZeroDm(w, log, zero_llm)
    for t in (t0, t1):
        for probe in make_probes(w, log, [t]):
            zero.serve(probe)
    asked = zero_llm.keys("program")
    assert asked and asked <= d1s_llm.keys("program")  # every program is a response-cache hit
    assert zero_llm.keys("write") == set()  # the view sends no writer request
    # a program is authored once per item and definition version, not once per query
    n_programs = len(w.slots) + sum(1 for s in w.slots if s.kappa_id)
    assert zero.cost["ingest_tokens"] <= 120 * n_programs


def test_the_view_shares_the_evidence_budget(world_log):
    w, log = world_log
    zero = ZeroDm(w, log, RecordingProgramLLM(PROGRAMS))
    n_views = 0
    for t in _times(w):
        for probe in make_probes(w, log, [t]):
            blocks = zero.serve(probe).blocks
            assert sum(approx_tokens(b) + 8 for b in blocks) <= READER_BUDGET, probe.fact_id
            n_views += any(PROGRAM_ORIGIN in b for b in blocks)
    assert n_views > 0


def test_an_item_with_a_failing_program_is_served_as_the_zero_system_serves_it(world_log):
    w, log = world_log
    zero = ZeroDm(w, log, RecordingProgramLLM(default=BROKEN))
    plain = ZeroSystem(w, log)
    for probe in make_probes(w, log, _times(w)):
        res = zero.serve(probe)
        assert res.blocks == plain.serve(probe).blocks, probe.fact_id
        assert res.items[0]["view"]["ok"] is False


def test_no_view_for_sources_the_asking_agent_may_not_read():
    w = generate_branch(11, "cons", world_params(3))
    log = SourceLog(w.events)
    _, agent, scope, _ = w.consent_log[0]
    t = w.fork + timedelta(weeks=1)
    slot = next(s for s in w.slots if scope in s.sources)
    zero = ZeroDm(w, log, RecordingProgramLLM())
    revoked = make_probe(w, log, slot, t, agent_id=agent)
    assert scope not in w.granted(agent, t)
    assert zero.view(revoked)[0] is None
    assert zero.view(make_probe(w, log, slot, t))[0] is not None  # an agent that may read it

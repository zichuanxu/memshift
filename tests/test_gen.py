from datetime import timedelta

import pytest

from memshift.gen.fidelity import check_world
from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.runinfo import world_record


@pytest.fixture(scope="module")
def worlds():
    return {b: generate_branch(7, b) for b in ("natural", "src", "def", "cons")}


def test_generation_is_deterministic():
    a = world_record(generate_branch(3, "natural"))
    b = world_record(generate_branch(3, "natural"))
    assert a["events_sha256"] == b["events_sha256"]
    assert a["events_sha256"] != world_record(generate_branch(4, "natural"))["events_sha256"]


def test_world_scale_matches_plan(worlds):
    w = worlds["natural"]
    assert 1300 <= len(w.events) <= 2600  # 1,500-2,500 events per world (calibrated)
    assert {e.source for e in w.events} == {
        "email",
        "calendar",
        "tasks",
        "contacts",
        "notes",
        "messages",
    }
    kinds = {s.kind for s in w.slots}
    assert kinds == {"user_attr", "contact_attr", "series", "appointment", "task_due", "kappa"}


def test_revisions_monotone_and_tombstones(worlds):
    last: dict[str, int] = {}
    for e in worlds["src"].events:
        assert e.revision == last.get(e.object_id, 0) + 1
        last[e.object_id] = e.revision
    assert any(e.op == "delete" for e in worlds["natural"].events)


def test_branches_share_natural_events(worlds):
    nat = [(e.object_id, e.ts) for e in worlds["natural"].events]
    for b in ("src", "def", "cons"):
        assert [(e.object_id, e.ts) for e in worlds[b].events if e.tag == "natural"] == nat


def test_fidelity_all_gold_located(worlds):
    for b in ("natural", "src"):
        rep = check_world(worlds[b])
        assert rep.checked > 1000 and rep.rate == 1.0, rep.failures[:3]


def test_src_shift_changes_gold(worlds):
    w = worlds["src"]
    log = SourceLog(w.events)
    before, after = w.fork - timedelta(minutes=1), w.fork + timedelta(days=7)
    changed = {
        s.slot_id for s in w.slots if gold_at(w, log, s, before) != gold_at(w, log, s, after)
    }
    targets = set(w.params["shift_targets"])
    assert len(changed & targets) >= len(targets) // 2


def test_def_shift_bumps_half_the_kappa(worlds):
    w = worlds["def"]
    upgraded = [k for k in w.kappa_defs if w.kappa_version(k, w.fork) == "v2"]
    assert len(upgraded) == len(w.kappa_defs) // 2
    assert all(w.kappa_version(k, w.fork - timedelta(minutes=1)) == "v1" for k in w.kappa_defs)


def test_cons_shift_revokes_scope(worlds):
    w = worlds["cons"]
    agent, scope = w.params["shift_targets"]
    assert scope in w.granted(agent, w.fork - timedelta(minutes=1))
    assert scope not in w.granted(agent, w.fork)


def test_conflict_branch_makes_gold_abstain():
    w = generate_branch(2, "conflict")
    log = SourceLog(w.events)
    targets = w.params["shift_targets"]
    assert len(targets) == 4
    for sid in targets:
        s = w.slot(sid)
        assert gold_at(w, log, s, w.fork - timedelta(minutes=1)).status == "answer"
        g = gold_at(w, log, s, w.fork + timedelta(days=1))
        assert g.status == "abstain" and len(g.deps) == 2

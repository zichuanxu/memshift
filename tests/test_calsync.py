from datetime import datetime, timedelta

import pytest

from memshift.calsync.sim import (
    CalendarServer,
    PreconditionFailedError,
    SyncAdapter,
    delivered_world,
    divergence,
    simulate,
)
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog

POLL = 15
DAYS = 60


@pytest.fixture(scope="module")
def world():
    return generate_branch(3, "natural")


def _cal(events):
    return [e for e in events if e.source == "calendar"]


def _official(world, lag, **kw):
    until = world.start + timedelta(days=DAYS)
    return delivered_world(world, "official", lag, POLL, 17, until=until, **kw)


def _appt(start):
    return {
        "kind": "appointment",
        "appt_type": "dentist",
        "provider": "Bright Dental",
        "title": "Bright Dental (dentist)",
        "start": start,
        "duration_min": 45,
        "cluster": "calendar:ev1",
    }


def test_ideal_mode_is_identity(world):
    dw, stats = delivered_world(world, "ideal")
    assert dw.events == world.events
    assert _cal(dw.events) == _cal(world.events)
    assert stats.polls == [] and stats.deliveries == []


@pytest.mark.parametrize("lag", [1, 5, 30])
def test_official_delay_between_lag_and_lag_plus_poll(world, lag):
    dw, stats = _official(world, lag)
    first = {}
    for d in stats.deliveries:
        if d.server_revision is not None and d.committed is not None:
            key = (d.object_id, d.server_revision)
            if key not in first or d.delivered < first[key][0]:
                first[key] = (d.delivered, d.committed)
    assert first
    for delivered, committed in first.values():
        assert timedelta(minutes=lag) <= delivered - committed <= timedelta(minutes=lag + POLL)
    horizon = world.start + timedelta(days=DAYS) - timedelta(minutes=lag + POLL)
    for e in _cal(world.events):
        if e.ts <= horizon:
            assert any(
                o == e.object_id and r >= e.revision and dl - e.ts <= timedelta(minutes=lag + POLL)
                for (o, r), (dl, _) in first.items()
            ), e.event_id
    assert all(e.source != "calendar" for e in dw.events if e.event_id.startswith("ev"))


def test_gone_forces_full_resync_that_restores_true_state(world):
    until = world.start + timedelta(days=20)
    events, stats = simulate(
        _cal(world.events), world.start, until, lag_minutes=0, gone_prob=1.0, page_size=7
    )
    true_log, got = SourceLog(_cal(world.events)), SourceLog(events)
    assert not stats.polls[0].gone and stats.polls[0].full
    assert all(p.gone and p.full for p in stats.polls[1:])
    assert any(p.pages > 1 for p in stats.polls)
    for p in stats.polls[:: len(stats.polls) // 25]:
        assert divergence(true_log, got, p.time) == 0.0


def test_expired_token_is_gone_and_resync_deletes_vanished_objects():
    t = datetime(2026, 3, 2, 9, 0)
    server = CalendarServer(lag_minutes=0, token_ttl_days=7, gone_prob=0.0)
    for i in range(3):
        server.commit(f"calendar:ev{i}", "insert", {"kind": "x", "n": i}, t)
    adapter = SyncAdapter(server)
    assert adapter.poll(t).full
    later = t + timedelta(days=1)
    server.commit("calendar:ev0", "delete", {"kind": "x", "n": 0}, later)
    server.commit("calendar:ev1", "update", {"kind": "x", "n": 11}, later)
    rec = adapter.poll(t + timedelta(days=8))
    assert rec.gone and rec.full
    state = SourceLog(adapter.events).snapshot(t + timedelta(days=8))
    assert {o: s.payload for o, s in state.objects.items()} == {
        "calendar:ev1": {"kind": "x", "n": 11},
        "calendar:ev2": {"kind": "x", "n": 2},
    }
    assert adapter.events[-1].op == "delete" and adapter.events[-1].object_id == "calendar:ev0"


def test_conditional_write_412_on_stale_etag():
    t = datetime(2026, 3, 2, 9, 0)
    server = CalendarServer(lag_minutes=5, gone_prob=0.0)
    server.commit("calendar:ev1", "insert", _appt("2026-03-10 10:00"), t)
    adapter = SyncAdapter(server)
    adapter.poll(t + timedelta(minutes=5))
    assert adapter.mirror["calendar:ev1"]["etag"] == "calendar:ev1:1"

    other = t + timedelta(minutes=20)
    server.commit("calendar:ev1", "update", _appt("2026-03-11 10:00"), other)
    with pytest.raises(PreconditionFailedError):
        server.update("calendar:ev1", {"start": "2026-03-12 10:00"}, "calendar:ev1:1", now=other)
    with pytest.raises(PreconditionFailedError):
        adapter.write("calendar:ev1", {"start": "2026-03-12 10:00"}, other + timedelta(minutes=1))
    item = adapter.write(
        "calendar:ev1", {"start": "2026-03-12 10:00"}, other + timedelta(minutes=6)
    )
    assert item["etag"] == "calendar:ev1:3"
    assert "2026-03-12 10:00" in item["payload"]["text"]


def test_divergence_zero_in_ideal_and_positive_inside_lag_window(world):
    true_log = SourceLog(world.events)
    ideal, _ = delivered_world(world, "ideal")
    ideal_log = SourceLog(ideal.events)
    official, _ = _official(world, 30)
    off_log = SourceLog(official.events)
    horizon = world.start + timedelta(days=DAYS)
    evs = [e for e in _cal(world.events) if e.ts <= horizon]
    assert evs
    for e in evs:
        assert divergence(true_log, ideal_log, e.ts) == 0.0
    changed = [e for e in evs if e.op in ("insert", "delete")]
    assert changed
    for e in changed:
        assert divergence(true_log, off_log, e.ts) > 0.0
        assert divergence(true_log, off_log, e.ts + timedelta(minutes=29)) > 0.0


def test_delivered_revisions_monotone_per_object(world):
    dw, stats = _official(world, 5)
    assert sum(p.gone for p in stats.polls) > 0
    last = {}
    for e in sorted(_cal(dw.events), key=lambda e: (e.ts, e.event_id)):
        assert e.revision == last.get(e.object_id, 0) + 1, e.event_id
        last[e.object_id] = e.revision

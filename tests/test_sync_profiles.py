from dataclasses import replace
from datetime import datetime, timedelta

import pytest

from memshift.calsync.sim import (
    ALL_SOURCES,
    CALENDAR,
    CONTACTS,
    EMAIL,
    MESSAGES,
    NOTES,
    PROFILES,
    TASKS,
    SyncAdapter,
    SyncServer,
    delivered_world,
    delivered_world_all,
    divergence,
    simulate_source,
    source_seeds,
)
from memshift.gen.model import Event
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog

T = datetime(2026, 3, 2, 9, 0)
LAG = 5
POLL = 15


@pytest.fixture(scope="module")
def world():
    return generate_branch(3, "natural")


def _ev(n, ts, src, obj, rev, op, payload):
    return Event(f"e{n}", ts, src, f"{src}:{obj}", rev, op, payload)


def test_profiles_cover_every_source():
    assert tuple(PROFILES) == ALL_SOURCES
    assert all(p.poll_minutes == POLL and p.page_size == 50 for p in PROFILES.values())
    # only the calendar invalidates cursors at random; only messages hides edits and deletions
    assert [s for s, p in PROFILES.items() if p.gone_prob > 0] == ["calendar"]
    assert [s for s, p in PROFILES.items() if p.incremental == "created"] == ["messages"]


def test_messages_edits_and_deletes_of_delivered_messages_are_invisible():
    src = "messages"
    evs = [
        _ev(1, T, src, "m1", 1, "insert", {"body": "v1"}),
        _ev(2, T + timedelta(hours=1), src, "m1", 2, "update", {"body": "v1 edited"}),
        _ev(3, T + timedelta(hours=2), src, "m1", 3, "delete", {"body": "v1 edited"}),
        _ev(4, T + timedelta(hours=2), src, "m2", 1, "insert", {"body": "v2"}),
        _ev(5, T + timedelta(hours=2, minutes=1), src, "m2", 2, "update", {"body": "v2 edited"}),
        _ev(6, T + timedelta(hours=3), src, "m3", 1, "insert", {"body": "v3"}),
        _ev(7, T + timedelta(hours=3, minutes=1), src, "m3", 2, "delete", {"body": "v3"}),
    ]
    end = T + timedelta(hours=5)
    delivered, stats = simulate_source(MESSAGES, evs, T - timedelta(hours=1), end, lag_minutes=LAG)
    # m1 once, as first fetched; m2 in the state it had when first fetched; m3 never
    assert [(e.object_id, e.op, e.payload) for e in delivered] == [
        ("messages:m1", "insert", {"body": "v1"}),
        ("messages:m2", "insert", {"body": "v2 edited"}),
    ]
    assert not any(p.gone for p in stats.polls)
    state = SourceLog(delivered).snapshot(end)
    assert state.get("messages:m1") is not None
    assert divergence(SourceLog(evs), SourceLog(delivered), end, src) == 0.5


@pytest.mark.parametrize("profile", [TASKS, CONTACTS, NOTES, EMAIL, CALENDAR])
def test_deletion_delivered_as_tombstone_within_lag_plus_poll(profile):
    src = profile.source
    gone = T + timedelta(hours=2, minutes=7)
    evs = [
        _ev(1, T, src, "a", 1, "insert", {"n": 1}),
        _ev(2, T, src, "b", 1, "insert", {"n": 2}),
        _ev(3, T + timedelta(hours=1), src, "b", 2, "update", {"n": 3}),
        _ev(4, gone, src, "a", 2, "delete", {"n": 1}),
    ]
    end = T + timedelta(hours=4)
    delivered, stats = simulate_source(
        replace(profile, gone_prob=0.0),
        evs,
        T - timedelta(hours=1),
        end,
        lag_minutes=LAG,
        phase_minutes=4,
    )
    dels = [d for d in stats.deliveries if d.op == "delete"]
    assert len(dels) == 1 and dels[0].object_id == f"{src}:a" and not dels[0].resync
    assert dels[0].committed == gone
    assert timedelta(minutes=LAG) <= dels[0].delivered - gone <= timedelta(minutes=LAG + POLL)
    assert [p.full for p in stats.polls] == [True] + [False] * (len(stats.polls) - 1)
    state = SourceLog(delivered).snapshot(end)
    assert {o: s.payload for o, s in state.objects.items()} == {f"{src}:b": {"n": 3}}
    assert divergence(SourceLog(evs), SourceLog(delivered), end, src) == 0.0


@pytest.mark.parametrize(
    "profile, expires",
    [(CONTACTS, True), (EMAIL, True), (TASKS, False), (NOTES, False), (MESSAGES, False)],
)
def test_cursor_expiry_follows_the_profile_ttl(profile, expires):
    src = profile.source
    server = SyncServer(profile, lag_minutes=0)
    server.commit(f"{src}:a", "insert", {"n": 1}, T)
    adapter = SyncAdapter(server)
    assert adapter.poll(T).full
    # every poll refreshes the cursor, so a 7-day TTL only bites after a 7-day gap
    assert not adapter.poll(T + timedelta(days=7)).gone
    rec = adapter.poll(T + timedelta(days=14, minutes=1))
    assert rec.gone == expires and rec.full == expires


def test_update_renders_the_objects_own_source():
    server = SyncServer(TASKS, lag_minutes=0)
    payload = {
        "title": "File taxes",
        "project": "Admin",
        "due": "2026-03-10",
        "status": "open",
        "priority": "high",
    }
    server.commit("tasks:t1", "insert", payload, T)
    item = server.update("tasks:t1", {"due": "2026-03-12"}, "tasks:t1:1", now=T)
    assert item["payload"]["text"].startswith("Task: File taxes")
    assert "Due: 2026-03-12" in item["payload"]["text"]


def test_calendar_through_all_sources_matches_the_calendar_only_path(world):
    until = world.start + timedelta(days=60)
    one, st1 = delivered_world(world, "official", LAG, POLL, 17, until=until)
    only, st = delivered_world_all(world, "official", LAG, 17, ("calendar",), until=until)
    assert only.events == one.events
    assert st["calendar"].polls == st1.polls and st["calendar"].deliveries == st1.deliveries
    assert st["calendar"].summary() == st1.summary()
    allw, sts = delivered_world_all(world, "official", LAG, 17, until=until)
    cal = [e for e in allw.events if e.source == "calendar"]
    assert cal == [e for e in one.events if e.source == "calendar"]
    assert sts["calendar"].summary() == st1.summary()
    assert set(sts) == set(ALL_SOURCES)


def test_all_sources_official_replaces_every_source_and_ideal_is_identity(world):
    until = world.start + timedelta(days=30)
    ideal, _ = delivered_world_all(world, "ideal", 0, 17)
    assert ideal.events == world.events
    dw, stats = delivered_world_all(world, "official", LAG, 17, until=until)
    ids = [e.event_id for e in dw.events]
    assert len(ids) == len(set(ids))
    assert all(e.event_id.startswith(("sync", f"{e.source}.sync")) for e in dw.events)
    assert all(stats[s].summary()["n_delivered"] > 0 for s in ALL_SOURCES)
    true_log, ideal_log, off_log = (SourceLog(w.events) for w in (world, ideal, dw))
    t = until - timedelta(hours=1)
    for src in ALL_SOURCES:
        assert divergence(true_log, ideal_log, t, src) == 0.0
    assert off_log.snapshot(t).by_source("email")
    # calendar keeps the stage-1 seeding; every other adapter's seed strings name its source
    assert source_seeds("calendar", 17) == (17, "phase|17")
    for src in (s for s in ALL_SOURCES if s != "calendar"):
        assert all(src in str(part) for part in source_seeds(src, 17))

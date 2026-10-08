"""Source servers with official incremental-sync semantics and a compliant polling client
(plan §4.2 "同步语义": sync-token paging, clear-and-rebuild after an invalid token, deletions in
the incremental response, ETag `If-Match` conditional replacement, read-after-write lag L).

SyncProfile      the incremental-sync contract of one source's official API (PROFILES: one per
                 generator source; each profile documents the endpoint it models).
SyncServer       append-only log of committed changes; etag = "{object_id}:{revision}".
                 `list` without a sync token returns every live object (full sync); with one it
                 returns what the profile's cursor reports since the token's log position:
                 "changes" = every object changed since then, collapsed to its latest visible
                 state, deletions as tombstones; "created" = only objects first committed since
                 then, in their latest visible state (edits and deletions of objects created
                 earlier are never reported). Results are paged; the last page carries
                 `next_sync_token`. A token older than the profile's TTL, or hit by the seeded
                 per-poll invalidation draw, raises GoneError (the client must full-sync). A change
                 committed at t is visible to `list`/`get` only from t + L. `update` compares
                 `If-Match` with the committed (not the visible) etag and raises
                 PreconditionFailedError (412) when they differ.
CalendarServer   SyncServer with the calendar profile and the stage-1 E4 constructor.
SyncAdapter      polls every `poll_minutes`, follows pages to the sync token and, on an invalid
                 token, clears its mirror and runs a full sync in the same poll. It emits the
                 delivered event stream: each change becomes an Event stamped with the poll time at
                 which the client first saw it and numbered by a per-object revision counter owned
                 by the data layer, which survives resyncs. A full sync re-emits every live object
                 as an insert and deletes mirrored objects that are no longer listed.
delivered_world  the world with its calendar replaced by the delivered stream ("official") or left
                 as is ("ideal": every event once, at its own time). Stage-1 E4.
delivered_world_all
                 the same for every listed source, each through its own profile, server and
                 adapter; calendar keeps delivered_world's seeding, so its delivered events and
                 stats are identical to the calendar-only path for the same seed.
divergence       fraction of live objects of one source whose presence or payload differs.
"""

from __future__ import annotations

import bisect
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta
from typing import Any

from memshift.gen.model import SOURCES, Event, World, fmt_ts
from memshift.gen.render import render_text
from memshift.gen.state import SourceLog

SOURCE = "calendar"
ALL_SOURCES = SOURCES
MODES = ("ideal", "official")
POLL_MINUTES = 15
PAGE_SIZE = 50
TOKEN_TTL_DAYS = 7.0
GONE_PROB = 0.001
_TOKEN_TS = "%Y%m%dT%H%M"

Item = dict[str, Any]


@dataclass(frozen=True)
class SyncProfile:
    """The incremental-sync contract of one source's official API, as the simulator models it.

    Every profile is polled by the same compliant client (every `poll_minutes`, `page_size` items
    per page, read-after-write lag L set per run). A client that polls every 15 minutes refreshes
    its cursor on every poll, so a TTL of days never fires in practice; the only resyncs come from
    `gone_prob`. That is left to emerge from the simulation, not special-cased.
    """

    source: str
    api: str  # endpoint and cursor parameter modelled
    incremental: str  # "changes" | "created" (see SyncServer._items)
    tombstone: str | None  # how an incremental response marks a deletion; None: never reported
    invalid_cursor: str  # what the API answers for an expired / invalidated cursor
    token_ttl_days: float | None = TOKEN_TTL_DAYS  # None: cursors never expire
    gone_prob: float = 0.0  # per incremental poll: server invalidates the cursor (full resync)
    page_size: int = PAGE_SIZE
    poll_minutes: int = POLL_MINUTES

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


# Google Calendar API `events.list` with `syncToken` (nextSyncToken on the last page): changes
# since the token, deletions as `status: "cancelled"`; an expired or invalidated token returns
# HTTP 410 GONE and the client must clear its store and run a full sync. TTL, 410 probability
# and page size are the stage-1 E4 parameters.
CALENDAR = SyncProfile(
    source="calendar",
    api="Google Calendar API events.list(syncToken)",
    incremental="changes",
    tombstone='status: "cancelled"',
    invalid_cursor="HTTP 410 GONE -> clear and full sync",
    token_ttl_days=TOKEN_TTL_DAYS,
    gone_prob=GONE_PROB,
)
# People API `people.connections.list` with `requestSyncToken=true`, then `syncToken`: sync
# tokens expire after 7 days (EXPIRED_SYNC_TOKEN -> full sync without a token); deleted contacts
# appear in incremental responses with `metadata.deleted = true`. No documented random
# invalidation.
CONTACTS = SyncProfile(
    source="contacts",
    api="People API people.connections.list(requestSyncToken, syncToken)",
    incremental="changes",
    tombstone="metadata.deleted = true",
    invalid_cursor="EXPIRED_SYNC_TOKEN -> full sync",
    token_ttl_days=7.0,
)
# Gmail `users.history.list` with `startHistoryId`: history records (messagesAdded,
# messagesDeleted, ...) after the id; history is typically available for at least a week, and an
# out-of-date id returns HTTP 404, after which the client runs a full sync (messages.list).
EMAIL = SyncProfile(
    source="email",
    api="Gmail users.history.list(startHistoryId)",
    incremental="changes",
    tombstone="messagesDeleted",
    invalid_cursor="HTTP 404 -> full sync",
    token_ttl_days=7.0,
)
# Google Tasks `tasks.list` with `updatedMin`, `showDeleted=true`, `showHidden=true`: tasks
# modified at or after updatedMin, deleted tasks with `deleted = true`. The client sets updatedMin
# to the latest `updated` it has seen (not to its own clock) and de-duplicates equal timestamps
# by etag, so with a lag common to all changes the cursor is a log position. updatedMin has no
# expiry.
TASKS = SyncProfile(
    source="tasks",
    api="Google Tasks tasks.list(updatedMin, showDeleted=true, showHidden=true)",
    incremental="changes",
    tombstone="deleted = true",
    invalid_cursor="none (updatedMin does not expire)",
    token_ttl_days=None,
)
# Google Drive `changes.list` with `pageToken` (first token from `changes.getStartPageToken`;
# newStartPageToken on the last page): removals as `removed = true`; page tokens do not expire.
NOTES = SyncProfile(
    source="notes",
    api="Google Drive changes.list(pageToken from changes.getStartPageToken)",
    incremental="changes",
    tombstone="removed = true",
    invalid_cursor="none (page tokens do not expire)",
    token_ttl_days=None,
)
# Slack `conversations.history` with `oldest = <latest ts seen>`: a polling client only receives
# messages posted after `oldest`, in their current state; edits and deletions of messages it has
# already fetched are not visible (they arrive only through the Events API, which a polling
# client does not use).
MESSAGES = SyncProfile(
    source="messages",
    api="Slack conversations.history(oldest = latest ts seen)",
    incremental="created",
    tombstone=None,
    invalid_cursor="none (oldest is a timestamp)",
    token_ttl_days=None,
)
PROFILES: dict[str, SyncProfile] = {
    p.source: p for p in (EMAIL, CALENDAR, TASKS, CONTACTS, NOTES, MESSAGES)
}


class SyncError(Exception):
    status = 0


class GoneError(SyncError):
    """HTTP 410: the sync token is no longer valid; the client must run a full sync."""

    status = 410


class PreconditionFailedError(SyncError):
    """HTTP 412: `If-Match` does not match the object's current etag."""

    status = 412


class NotFoundError(SyncError):
    status = 404


def etag(object_id: str, revision: int) -> str:
    return f"{object_id}:{revision}"


@dataclass(frozen=True)
class Change:
    seq: int
    committed: datetime
    object_id: str
    revision: int
    op: str
    payload: dict[str, Any]
    tag: str = "natural"

    def item(self) -> Item:
        cancelled = self.op == "delete"
        return {
            "id": self.object_id,
            "etag": etag(self.object_id, self.revision),
            "revision": self.revision,
            "status": "cancelled" if cancelled else "confirmed",
            "updated": self.committed,
            "payload": {} if cancelled else dict(self.payload),
            "tag": self.tag,
        }


class SyncServer:
    def __init__(self, profile: SyncProfile, lag_minutes: float = 0, seed: int | str = 0):
        self.profile = profile
        self.lag = timedelta(minutes=lag_minutes)
        ttl = profile.token_ttl_days
        self.token_ttl = None if ttl is None else timedelta(days=ttl)
        self.gone_prob = profile.gone_prob
        self.page_size = profile.page_size
        self.rng = random.Random(seed)
        self.log: list[Change] = []
        self._committed: list[datetime] = []
        self._latest: dict[str, Change] = {}

    def commit(
        self, object_id: str, op: str, payload: dict[str, Any], ts: datetime, tag: str = "natural"
    ) -> Change:
        """Record a change made by any client (the user's other devices, or `update`)."""
        if self._committed and ts < self._committed[-1]:
            raise ValueError(f"commit at {ts} precedes the last commit {self._committed[-1]}")
        prev = self._latest.get(object_id)
        ch = Change(
            len(self.log), ts, object_id, (prev.revision if prev else 0) + 1, op, dict(payload), tag
        )
        self.log.append(ch)
        self._committed.append(ts)
        self._latest[object_id] = ch
        return ch

    def commit_event(self, e: Event) -> Change:
        if e.source != self.profile.source:
            raise ValueError(f"{e.event_id}: {e.source} event on the {self.profile.source} server")
        return self.commit(e.object_id, e.op, e.payload, e.ts, e.tag)

    def horizon(self, now: datetime) -> int:
        """Number of changes visible at `now`; a log prefix, since every change has the same lag."""
        return bisect.bisect_right(self._committed, now - self.lag)

    def _items(self, kind: str, pos: int, horizon: int) -> list[Item]:
        changes = self.log[pos:horizon]
        created_only = kind == "delta" and self.profile.incremental == "created"
        if created_only:
            # only objects whose first commit lies after the cursor
            new = {ch.object_id for ch in changes if ch.revision == 1}
            changes = [ch for ch in changes if ch.object_id in new]
        last: dict[str, Change] = {}
        for ch in changes:
            if (kind == "full" or created_only) and ch.op == "delete":
                last.pop(ch.object_id, None)
            else:
                last[ch.object_id] = ch
        if kind == "full":
            return [last[oid].item() for oid in sorted(last)]
        return [ch.item() for ch in sorted(last.values(), key=lambda c: c.seq)]

    def _position(self, sync_token: str, now: datetime) -> int:
        draw = self.rng.random()
        try:
            _, pos, issued = sync_token.split(":")
            position, issued_at = int(pos), datetime.strptime(issued, _TOKEN_TS)
        except ValueError as exc:
            raise GoneError(sync_token) from exc
        expired = self.token_ttl is not None and now - issued_at > self.token_ttl
        if expired or draw < self.gone_prob:
            raise GoneError(sync_token)
        return position

    def list(
        self, sync_token: str | None = None, page_token: str | None = None, *, now: datetime
    ) -> dict[str, Any]:
        """One page of a full (no token) or incremental (sync token) listing at `now`.

        The invalidation draw happens once per incremental listing (its first page), i.e. once per
        poll of a client that holds a token.
        """
        if page_token is not None:
            _, kind, pos_s, hor_s, off_s = page_token.split(":")
            pos, horizon, offset = int(pos_s), int(hor_s), int(off_s)
        elif sync_token is not None:
            kind, pos, offset = "delta", self._position(sync_token, now), 0
            horizon = self.horizon(now)
        else:
            kind, pos, horizon, offset = "full", 0, self.horizon(now), 0
        items = self._items(kind, pos, horizon)
        end = offset + self.page_size
        out: dict[str, Any] = {"items": items[offset:end]}
        if end < len(items):
            out["next_page_token"] = f"page:{kind}:{pos}:{horizon}:{end}"
        else:
            out["next_sync_token"] = f"sync:{horizon}:{now:{_TOKEN_TS}}"
        return out

    def get(self, object_id: str, *, now: datetime) -> Item:
        """Read one object as visible at `now` (subject to the read-after-write lag)."""
        for ch in reversed(self.log[: self.horizon(now)]):
            if ch.object_id == object_id:
                if ch.op == "delete":
                    break
                return ch.item()
        raise NotFoundError(object_id)

    def update(self, object_id: str, body: dict[str, Any], if_match: str, *, now: datetime) -> Item:
        """Conditional replacement: apply `body` only if `if_match` is the current etag."""
        cur = self._latest.get(object_id)
        if cur is None or cur.op == "delete":
            raise NotFoundError(object_id)
        current = etag(object_id, cur.revision)
        if if_match != current:
            raise PreconditionFailedError(f"{object_id}: If-Match {if_match} != {current}")
        payload = {**cur.payload, **body}
        payload["text"] = render_text(self.profile.source, payload)
        return self.commit(object_id, "update", payload, now, "write").item()


class CalendarServer(SyncServer):
    """The calendar server with the stage-1 E4 constructor."""

    def __init__(
        self,
        lag_minutes: float = 0,
        token_ttl_days: float = TOKEN_TTL_DAYS,
        gone_prob: float = GONE_PROB,
        page_size: int = PAGE_SIZE,
        seed: int | str = 0,
    ):
        profile = replace(
            CALENDAR, token_ttl_days=token_ttl_days, gone_prob=gone_prob, page_size=page_size
        )
        super().__init__(profile, lag_minutes, seed)


@dataclass(frozen=True)
class PollRecord:
    time: datetime
    pages: int
    gone: bool
    full: bool
    n_changes: int

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["time"] = fmt_ts(self.time)
        return d


@dataclass(frozen=True)
class Delivery:
    """One emitted event; `server_revision`/`committed` are None for a resync-inferred delete."""

    object_id: str
    op: str
    revision: int
    delivered: datetime
    server_revision: int | None
    committed: datetime | None
    resync: bool


class SyncAdapter:
    def __init__(
        self, server: SyncServer, poll_minutes: int = POLL_MINUTES, event_prefix: str = "sync"
    ):
        self.server = server
        self.poll_minutes = poll_minutes
        self.event_prefix = event_prefix
        self.sync_token: str | None = None
        self.mirror: dict[str, Item] = {}
        self.revisions: dict[str, int] = {}
        self.events: list[Event] = []
        self.polls: list[PollRecord] = []
        self.deliveries: list[Delivery] = []

    def _fetch(self, sync_token: str | None, now: datetime) -> tuple[list[Item], int]:
        items: list[Item] = []
        pages = 0
        page: str | None = None
        while True:
            resp = self.server.list(sync_token, page, now=now)
            pages += 1
            items += resp["items"]
            page = resp.get("next_page_token")
            if page is None:
                self.sync_token = resp["next_sync_token"]
                return items, pages

    def poll(self, now: datetime) -> PollRecord:
        full, gone = self.sync_token is None, False
        try:
            items, pages = self._fetch(self.sync_token, now)
        except GoneError:
            full, gone = True, True
            self.sync_token = None
            items, pages = self._fetch(None, now)
        n0 = len(self.events)
        if full:
            self._resync(items, now)
        else:
            for item in items:
                self._apply(item, now)
        rec = PollRecord(now, pages, gone, full, len(self.events) - n0)
        self.polls.append(rec)
        return rec

    def _resync(self, items: list[Item], now: datetime) -> None:
        old, self.mirror = self.mirror, {}
        for item in items:
            self.mirror[item["id"]] = item
            self._emit(item["id"], "insert", item["payload"], item["tag"], now, item, True)
        for oid in sorted(set(old) - set(self.mirror)):
            self._emit(oid, "delete", old[oid]["payload"], old[oid]["tag"], now, None, True)

    def _apply(self, item: Item, now: datetime) -> None:
        oid = item["id"]
        held = self.mirror.get(oid)
        if item["status"] == "cancelled":
            if held is not None:
                del self.mirror[oid]
                self._emit(oid, "delete", held["payload"], item["tag"], now, item, False)
            return
        if held is not None and held["etag"] == item["etag"]:
            return
        self.mirror[oid] = item
        op = "insert" if held is None else "update"
        self._emit(oid, op, item["payload"], item["tag"], now, item, False)

    def _emit(
        self,
        oid: str,
        op: str,
        payload: dict[str, Any],
        tag: str,
        now: datetime,
        item: Item | None,
        resync: bool,
    ) -> None:
        rev = self.revisions.get(oid, 0) + 1
        self.revisions[oid] = rev
        n = len(self.events) + 1
        src = self.server.profile.source
        eid = f"{self.event_prefix}{n:06d}"
        self.events.append(Event(eid, now, src, oid, rev, op, dict(payload), tag))
        self.deliveries.append(
            Delivery(
                oid,
                op,
                rev,
                now,
                item["revision"] if item else None,
                item["updated"] if item else None,
                resync,
            )
        )

    def write(self, object_id: str, body: dict[str, Any], now: datetime, retries: int = 1) -> Item:
        """Conditional write with the mirrored etag; on 412 re-read the object and retry.

        The re-read is itself subject to the read-after-write lag, so a retry inside the lag
        window can fail again; the last PreconditionFailedError is raised.
        """
        held = self.mirror.get(object_id)
        tag = held["etag"] if held else self.server.get(object_id, now=now)["etag"]
        for attempt in range(retries + 1):
            try:
                return self.server.update(object_id, body, tag, now=now)
            except PreconditionFailedError:
                if attempt == retries:
                    raise
                tag = self.server.get(object_id, now=now)["etag"]
        raise AssertionError("unreachable")


@dataclass
class SyncStats:
    mode: str
    lag_minutes: float
    poll_minutes: int
    phase_minutes: int
    polls: list[PollRecord] = field(default_factory=list)
    deliveries: list[Delivery] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        delays = [
            (d.delivered - d.committed).total_seconds() / 60
            for d in self.deliveries
            if d.committed is not None and not d.resync
        ]
        return {
            "mode": self.mode,
            "lag_minutes": self.lag_minutes,
            "poll_minutes": self.poll_minutes,
            "phase_minutes": self.phase_minutes,
            "n_polls": len(self.polls),
            "n_gone": sum(p.gone for p in self.polls),
            "n_full": sum(p.full for p in self.polls),
            "pages": sum(p.pages for p in self.polls),
            "n_delivered": len(self.deliveries),
            "n_resync_delivered": sum(d.resync for d in self.deliveries),
            "delay_minutes_mean": sum(delays) / len(delays) if delays else None,
            "delay_minutes_max": max(delays) if delays else None,
        }


def simulate(
    events: Iterable[Event],
    start: datetime,
    until: datetime,
    *,
    lag_minutes: float,
    poll_minutes: int = POLL_MINUTES,
    phase_minutes: int = 0,
    token_ttl_days: float = TOKEN_TTL_DAYS,
    gone_prob: float = GONE_PROB,
    page_size: int = PAGE_SIZE,
    seed: int | str = 0,
) -> tuple[list[Event], SyncStats]:
    """Calendar events through the calendar profile (stage-1 E4 parameters as arguments)."""
    profile = replace(
        CALENDAR,
        token_ttl_days=token_ttl_days,
        gone_prob=gone_prob,
        page_size=page_size,
        poll_minutes=poll_minutes,
    )
    return simulate_source(
        profile,
        events,
        start,
        until,
        lag_minutes=lag_minutes,
        phase_minutes=phase_minutes,
        seed=seed,
    )


def simulate_source(
    profile: SyncProfile,
    events: Iterable[Event],
    start: datetime,
    until: datetime,
    *,
    lag_minutes: float,
    phase_minutes: int = 0,
    seed: int | str = 0,
    event_prefix: str = "sync",
) -> tuple[list[Event], SyncStats]:
    """Commit true events as the clock reaches them and poll at start + phase + k * poll."""
    server = SyncServer(profile, lag_minutes, seed)
    adapter = SyncAdapter(server, profile.poll_minutes, event_prefix)
    todo = sorted(events, key=lambda e: (e.ts, e.event_id))
    i = 0
    t = start + timedelta(minutes=phase_minutes)
    step = timedelta(minutes=profile.poll_minutes)
    while t <= until:
        while i < len(todo) and todo[i].ts <= t:
            server.commit_event(todo[i])
            i += 1
        adapter.poll(t)
        t += step
    stats = SyncStats(
        "official",
        lag_minutes,
        profile.poll_minutes,
        phase_minutes,
        adapter.polls,
        adapter.deliveries,
    )
    return adapter.events, stats


def delivered_world(
    world: World,
    mode: str,
    lag_minutes: float = 5,
    poll_minutes: int = POLL_MINUTES,
    seed: int = 0,
    *,
    token_ttl_days: float = TOKEN_TTL_DAYS,
    gone_prob: float = GONE_PROB,
    page_size: int = PAGE_SIZE,
    until: datetime | None = None,
) -> tuple[World, SyncStats]:
    """`world` with its calendar events replaced by what the sync client delivers.

    The poll phase is drawn from `seed` alone, so every lag of one world shares the same poll
    times and the same 410 draws (common random numbers across lags).
    """
    if mode not in MODES:
        raise ValueError(mode)
    sync = {"mode": mode, "lag_minutes": lag_minutes, "poll_minutes": poll_minutes}
    if mode == "ideal":
        ideal = replace(world, events=list(world.events), params={**world.params, "sync": sync})
        return ideal, SyncStats("ideal", 0, 0, 0)
    phase = random.Random(f"phase|{seed}").randrange(poll_minutes)
    delivered, stats = simulate(
        (e for e in world.events if e.source == SOURCE),
        world.start,
        until or world.end,
        lag_minutes=lag_minutes,
        poll_minutes=poll_minutes,
        phase_minutes=phase,
        token_ttl_days=token_ttl_days,
        gone_prob=gone_prob,
        page_size=page_size,
        seed=seed,
    )
    events = sorted([e for e in world.events if e.source != SOURCE] + delivered, key=lambda e: e.ts)
    sync |= {
        "phase_minutes": phase,
        "token_ttl_days": token_ttl_days,
        "gone_prob": gone_prob,
        "page_size": page_size,
        "seed": seed,
    }
    return replace(world, events=events, params={**world.params, "sync": sync}), stats


def source_seeds(source: str, seed: int) -> tuple[int | str, str]:
    """(server RNG seed for the invalidation draws, poll-phase seed) of one source's adapter.

    Calendar keeps delivered_world's seeding; every other source's strings include its name.
    """
    if source == SOURCE:
        return seed, f"phase|{seed}"
    return f"{source}|{seed}", f"phase|{source}|{seed}"


def delivered_world_all(
    world: World,
    mode: str,
    lag_minutes: float = 5,
    seed: int = 0,
    sources: Iterable[str] = ALL_SOURCES,
    *,
    until: datetime | None = None,
    profiles: Mapping[str, SyncProfile] = PROFILES,
) -> tuple[World, dict[str, SyncStats]]:
    """`world` with every listed source's events replaced by what its own sync client delivers.

    Each source has its own server and adapter (profile, poll phase, invalidation draws). The
    calendar's delivered events and stats equal delivered_world's for the same seed; other
    sources' event ids carry the source name so ids stay unique across adapters.
    """
    if mode not in MODES:
        raise ValueError(mode)
    srcs = list(sources)
    sync: dict[str, Any] = {"mode": mode, "lag_minutes": lag_minutes, "sources": srcs}
    if mode == "ideal":
        ideal = replace(world, events=list(world.events), params={**world.params, "sync": sync})
        return ideal, {s: SyncStats("ideal", 0, 0, 0) for s in srcs}
    stats: dict[str, SyncStats] = {}
    profs: dict[str, Any] = {}
    delivered: list[Event] = []
    for src in srcs:
        profile = profiles[src]
        server_seed, phase_seed = source_seeds(src, seed)
        phase = random.Random(phase_seed).randrange(profile.poll_minutes)
        events, stats[src] = simulate_source(
            profile,
            (e for e in world.events if e.source == src),
            world.start,
            until or world.end,
            lag_minutes=lag_minutes,
            phase_minutes=phase,
            seed=server_seed,
            event_prefix="sync" if src == SOURCE else f"{src}.sync",
        )
        delivered += events
        profs[src] = {"phase_minutes": phase, **profile.to_json()}
    sync |= {"seed": seed, "profiles": profs}
    events = sorted(
        [e for e in world.events if e.source not in srcs] + delivered, key=lambda e: e.ts
    )
    return replace(world, events=events, params={**world.params, "sync": sync}), stats


def divergence(
    true_log: SourceLog, delivered_log: SourceLog, t: datetime, source: str = SOURCE
) -> float:
    """Share of `source` objects live in either log at t whose presence or payload differs."""
    a = {o.object_id: o.payload for o in true_log.snapshot(t).by_source(source)}
    b = {o.object_id: o.payload for o in delivered_log.snapshot(t).by_source(source)}
    ids = a.keys() | b.keys()
    return sum(a.get(i) != b.get(i) for i in ids) / len(ids) if ids else 0.0

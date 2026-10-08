"""Shift scripts applied at the fork point t_f (plan §4.2 "shift 脚本").

S_src: source revisions that change ~20% of slots (updates, deletes, new objects entering scope).
S_def: half of the kappa definitions move to v2; source bytes unchanged.
S_cons: one agent loses one scope.
S_wr and S_impl act on the system, not the world, and live in the experiment drivers.

Each shift draws from its own random stream, so the natural events of every branch are identical.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from memshift.gen import pools
from memshift.gen.model import World
from memshift.gen.world import Builder, build, to_world

BRANCHES = ("natural", "src", "def", "cons", "conflict", "orphan")


def inference_half(slot_id: str) -> bool:
    """E5: the fixed half of knowledge items used for black-box (D,B) inference; the other half
    measures carryover (plan §6 E5 (2)). Deterministic, independent of the world."""
    return sum(slot_id.encode()) % 2 == 0


def _upcoming_appointment(b: Builder, atype: str, t: datetime) -> str | None:
    state: dict[str, dict[str, Any]] = {}
    for r in sorted(b.raw, key=lambda r: (r.ts, r.seq)):
        if r.ts > t or r.source != "calendar":
            continue
        if r.op == "insert":
            state[r.object_id] = dict(r.patch)
        elif r.op == "update" and r.object_id in state:
            state[r.object_id].update(r.patch)
        elif r.op == "delete":
            state.pop(r.object_id, None)
    now = t.strftime("%Y-%m-%d %H:%M")
    ups = [
        (p["start"], oid)
        for oid, p in state.items()
        if p.get("kind") == "appointment" and p["appt_type"] == atype and p["start"] > now
    ]
    return min(ups)[1] if ups else None


def apply_src(b: Builder) -> list[str]:
    """Inject source revisions at the fork. Returns the targeted slot ids."""
    rng = random.Random(b.seed * 7919 + 1)
    b.rng = rng
    b.tag = "shift:src"
    slots = sorted(b.slots, key=lambda s: s.slot_id)
    k = max(1, round(float(b.p["src_fraction"]) * len(slots)))
    chosen = sorted(rng.sample(slots, k), key=lambda s: s.slot_id)
    done: set[str] = set()
    for i, slot in enumerate(chosen):
        ts = b.fork + timedelta(minutes=10 + 7 * i)
        sid = slot.slot_id
        if sid in done:
            continue
        if sid == "u.employer":
            b.job_change(0, ts=ts)
            done.add("u.manager")
        elif sid == "u.manager":
            b.reorg(0, ts=ts)
        elif sid == "u.home_city":
            b.move(0, ts=ts)
            done.add("u.timezone")
        elif sid == "u.timezone":
            current = b.current("u.timezone", ts)
            b.set_timezone(ts, rng.choice([tz for _, tz in pools.CITIES if tz != current]))
        elif sid == "u.diet":
            b.diet_change(0, ts=ts)
        elif sid == "u.commute":
            b.commute_change(0, ts=ts)
        elif slot.kind == "contact_attr":
            assert slot.subject is not None
            b.change_contact(slot.subject, sid.split(".")[-1], ts)
        elif slot.kind == "series":
            assert slot.subject is not None
            if b.alive(slot.subject, ts):
                if rng.random() < 0.3:
                    b.cancel_series(slot.subject, 0, ts=ts)
                else:
                    b.reschedule_series(slot.subject, 0, ts=ts)
        elif slot.kind == "appointment":
            _shift_appointment(b, rng, str(slot.subject), ts)
        elif slot.kind == "task_due":
            assert slot.subject is not None
            if b.alive(slot.subject, ts):
                due = (ts + timedelta(days=rng.randint(3, 30))).strftime("%Y-%m-%d")
                b.emit(ts, "tasks", slot.subject, "update", {"due": due})
        elif slot.kind == "kappa":
            _shift_kappa(b, rng, str(slot.kappa_id), ts)
        done.add(sid)
    b.tag = "natural"
    return [s.slot_id for s in chosen]


def _shift_appointment(b: Builder, rng: random.Random, atype: str, ts: datetime) -> None:
    provider = dict(pools.APPOINTMENT_TYPES)[atype]
    oid = _upcoming_appointment(b, atype, ts)
    r = rng.random()
    if oid is None or r < 0.3:
        start = (ts + timedelta(days=rng.randint(3, 20))).replace(
            hour=rng.choice((9, 11, 14)), minute=0
        )
        b.book_appointment(atype, provider, ts, start)
    elif r < 0.7:
        start = (ts + timedelta(days=rng.randint(21, 50))).replace(
            hour=rng.choice((10, 15)), minute=30
        )
        b.move_appointment(oid, atype, provider, ts, start)
    else:
        b.cancel_appointment(oid, atype, provider, ts, ts)


def _shift_kappa(b: Builder, rng: random.Random, kid: str, ts: datetime) -> None:
    day = (ts - b.start).days
    if kid == "important_contact":
        pid = rng.choice(b.friends + b.colleagues_at(day))
        for d in (1, 2, 3):
            b.meeting(ts + timedelta(days=d, hours=10), [pid], False, 45)
    elif kid == "close_friend":
        pid = rng.choice(b.friends + b.family)
        for h in range(7):
            b.message(ts + timedelta(hours=3 * h + 1), pid, h % 2 == 0, rng.choice(pools.SMALLTALK))
    elif kid == "active_project":
        oid = f"tasks:t{900 + b.nid('shift_task'):03d}"
        b.emit(
            ts,
            "tasks",
            oid,
            "insert",
            {
                "title": rng.choice(b.task_titles[40:]),
                "project": rng.choice([p for p in pools.PROJECTS if p not in b.projects]),
                "due": (ts + timedelta(days=20)).strftime("%Y-%m-%d"),
                "status": "open",
                "priority": "medium",
                "cluster": oid,
            },
        )
    elif kid == "working_hours":
        b.update_working_hours(ts)
        colleagues = b.colleagues_at(day)
        if colleagues:
            b.meeting((ts + timedelta(days=1)).replace(hour=7), [rng.choice(colleagues)], True, 60)
    elif kid == "vip_sender":
        colleagues = b.colleagues_at(day)
        if colleagues:
            pid = rng.choice(colleagues)
            for d in range(4):
                b.work_email(ts + timedelta(days=d, hours=9), pid)
    elif kid == "overdue_task":
        open_tasks = [oid for oid, _ in b.task_ids if b.alive(oid, ts)]
        if open_tasks:
            due = (ts - timedelta(days=2)).strftime("%Y-%m-%d")
            b.emit(ts, "tasks", rng.choice(open_tasks), "update", {"due": due, "status": "open"})


def apply_conflict(b: Builder, n: int = 4) -> dict[str, dict[str, Any]]:
    """E7: at the fork, a contact card and a same-day note from that contact disagree about the
    contact's city or employer. Neither source is newer or more authoritative, so the gold status
    of those items becomes `abstain` (see gold.gold_at)."""
    rng = random.Random(b.seed * 7919 + 4)
    b.rng = rng
    b.tag = "shift:conflict"
    slots = sorted((s for s in b.slots if s.kind == "contact_attr"), key=lambda s: s.slot_id)
    conflicts: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    for i, slot in enumerate(rng.sample(slots, len(slots))):
        if len(conflicts) >= n or slot.subject in used:
            continue
        assert slot.subject is not None
        pid, field_ = slot.subject, slot.slot_id.split(".")[-1]
        used.add(pid)
        ts = b.fork + timedelta(minutes=20 + 9 * i)
        card = b.cards[pid]
        if field_ == "city":
            pool = [c for c in [*b.other_cities, b.home_city] if c != card["city"]]
            a, other = rng.sample(pool, 2)
            line = f"Quick update: I'm living in {other} now."
        else:
            pool = [o for o in b.orgs[2:] if o != card["employer"]]
            a, other = rng.sample(pool, 2)
            line = f"Quick update: I work at {other} now."
        card_oid = b.update_card(ts, pid, **{field_: a})
        person = b.people[pid]
        if person.relation in ("friend", "family"):
            msg = b.message(ts + timedelta(minutes=1), pid, False, line)
        else:
            msg = b.email(ts + timedelta(minutes=1), person.name, person.email, "Update", line)
        conflicts[slot.slot_id] = {
            "ts": (ts + timedelta(minutes=1)).isoformat(),
            "objects": [card_oid, msg],
            "values": [a, other],
        }
    b.tag = "natural"
    return conflicts


def apply_orphan(b: Builder) -> list[str]:
    """E5 B probe: at the fork, delete the source object behind every structured inference-half
    item (recurring series, task, contact card). Gold for those items becomes abstain because it
    is recomputed from source state; a layer that keeps serving the old value missed the change."""
    b.tag = "shift:orphan"
    targets: list[str] = []
    for i, slot in enumerate(sorted(b.slots, key=lambda s: s.slot_id)):
        if not inference_half(slot.slot_id) or slot.kind not in (
            "series",
            "task_due",
            "contact_attr",
        ):
            continue
        oid = str(slot.subject) if slot.kind != "contact_attr" else f"contacts:{slot.subject}"
        ts = b.fork + timedelta(minutes=10 + 3 * i)
        if oid not in {t for t in targets} and b.alive(oid, ts):
            b.emit(ts, oid.split(":")[0], oid, "delete", {})
            targets.append(oid)
    b.tag = "natural"
    return targets


def apply_def(world: World) -> list[str]:
    """Half of the kappa definitions move to v2 at the fork."""
    rng = random.Random(world.seed * 7919 + 2)
    kids = sorted(world.kappa_defs)
    chosen = sorted(rng.sample(kids, len(kids) // 2))
    for kid in chosen:
        world.kappa_log.append((world.fork, kid, "v2"))
    return chosen


def apply_cons(world: World) -> tuple[str, str]:
    """One agent loses one of its scopes at the fork."""
    rng = random.Random(world.seed * 7919 + 3)
    agent = rng.choice(["scheduler", "concierge"])
    scope = rng.choice(world.agents[agent])
    world.consent_log.append((world.fork, agent, scope, "revoked"))
    return agent, scope


def generate_branch(seed: int, branch: str, params: dict[str, Any] | None = None) -> World:
    if branch not in BRANCHES:
        raise ValueError(branch)
    b = build(seed, params)
    targeted: Any = None
    conflicts: dict[str, dict[str, Any]] = {}
    if branch == "src":
        targeted = apply_src(b)
    elif branch == "conflict":
        conflicts = apply_conflict(b)
        targeted = sorted(conflicts)
    elif branch == "orphan":
        targeted = apply_orphan(b)
    world = to_world(b, branch)
    if conflicts:
        world.params = {**world.params, "conflicts": conflicts}
    if branch == "def":
        targeted = apply_def(world)
    elif branch == "cons":
        targeted = list(apply_cons(world))
    world.params = {**world.params, "shift_targets": targeted}
    return world

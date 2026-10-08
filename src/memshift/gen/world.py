"""Seeded, script-first world generator (plan §4.2).

Gold changes are scripted first; every change is then rendered into one or more source objects
(template rendering, no LLM), so each gold fact is locatable in text by construction and the
fidelity checker (gen/fidelity.py) confirms it. Background traffic (newsletters, small talk,
work email, one-off meetings) provides realistic volume and lexical distractors.

The natural stream is generated first from `Random(seed)`. Shifts (S_src, S_def, S_cons) are
applied afterwards from a separate random stream (gen/shifts.py), so every branch of a world
shares the same natural events byte for byte.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from memshift.gen import pools
from memshift.gen.kappa import KAPPA_DEFS
from memshift.gen.model import Event, KappaDef, Person, Slot, TimelinePoint, World
from memshift.gen.render import render_text

DEFAULT_PARAMS: dict[str, Any] = {
    "months": 12,
    "fork_month": 9,
    "n_contacts": 30,
    "n_tracked_contacts": 12,
    "n_tracked_tasks": 6,
    "target_tokens": 2_000_000,
    "newsletter_per_topic_per_month": 4,
    "src_fraction": 0.2,
}

START = datetime(2026, 1, 1, 0, 0)

AGENTS = {
    "assistant": ["email", "calendar", "tasks", "contacts", "notes", "messages"],
    "scheduler": ["calendar", "tasks"],
    "concierge": ["email", "calendar", "contacts"],
}


def add_months(t: datetime, months: int) -> datetime:
    y, m = divmod(t.month - 1 + months, 12)
    return t.replace(year=t.year + y, month=m + 1)


def poisson(rng: random.Random, lam: float) -> int:
    limit = math.exp(-lam)
    k, prod = 0, rng.random()
    while prod > limit:
        k += 1
        prod *= rng.random()
    return k


@dataclass
class Raw:
    ts: datetime
    seq: int
    source: str
    object_id: str
    op: str
    patch: dict[str, Any]
    tag: str


class Builder:
    """Holds the script state while the natural stream (and later shifts) are generated."""

    def __init__(self, seed: int, params: dict[str, Any]):
        self.seed = seed
        self.p = {**DEFAULT_PARAMS, **params}
        self.rng = random.Random(seed)
        self.start = START
        self.end = add_months(START, int(self.p["months"]))
        self.fork = add_months(START, int(self.p["fork_month"]))
        self.raw: list[Raw] = []
        self._seq = 0
        self._ids: dict[str, int] = {}
        self.people: dict[str, Person] = {}
        self.cards: dict[str, dict[str, Any]] = {}
        self.timelines: dict[str, list[TimelinePoint]] = {}
        self.slots: list[Slot] = []
        self.threads = 0
        self.tag = "natural"
        self.used_first: set[str] = set()
        self.user_name = ""
        self.user_first = ""

    # ------------------------------------------------------------------ helpers
    def nid(self, prefix: str) -> int:
        self._ids[prefix] = self._ids.get(prefix, 0) + 1
        return self._ids[prefix]

    def emit(self, ts: datetime, source: str, oid: str, op: str, patch: dict[str, Any]) -> str:
        self._seq += 1
        self.raw.append(Raw(ts, self._seq, source, oid, op, dict(patch), self.tag))
        return oid

    def at(self, day: float, hour: int | None = None, minute: int | None = None) -> datetime:
        base = self.start + timedelta(days=int(day))
        h = self.rng.randint(8, 21) if hour is None else hour
        m = self.rng.choice((0, 5, 10, 15, 20, 30, 40, 45, 50)) if minute is None else minute
        return base.replace(hour=h, minute=m)

    def total_days(self) -> int:
        return (self.end - self.start).days

    def rday(self, lo: int, hi: int) -> int:
        """randint(lo, hi) for a 12-month world; lower bounds scale down for shorter worlds."""
        n = self.total_days()
        if n < 365:
            lo = int(lo * n / 365)
            hi = max(lo, min(hi, n - 1))
        return self.rng.randint(lo, hi)

    def new_name(self) -> tuple[str, str]:
        while True:
            first = self.rng.choice(pools.FIRST_NAMES)
            last = self.rng.choice(pools.LAST_NAMES)
            if first not in self.used_first:
                self.used_first.add(first)
                return first, last

    @staticmethod
    def domain(org: str) -> str:
        return org.lower().split()[0] + ".example"

    def email(
        self,
        ts: datetime,
        sender: str,
        addr: str,
        subject: str,
        body: str,
        thread: str | None = None,
    ) -> str:
        oid = f"email:e{self.nid('email'):05d}"
        if thread is None:
            self.threads += 1
            thread = f"thread:{self.threads:05d}"
        return self.emit(
            ts,
            "email",
            oid,
            "insert",
            {
                "from_name": sender,
                "from_addr": addr,
                "to": self.user_name,
                "subject": subject,
                "body": body,
                "cluster": thread,
                "sent": ts.strftime("%Y-%m-%d %H:%M"),
            },
        )

    def message(self, ts: datetime, pid: str, from_me: bool, body: str) -> str:
        oid = f"messages:m{self.nid('msg'):05d}"
        other = self.people[pid].name
        return self.emit(
            ts,
            "messages",
            oid,
            "insert",
            {
                "conversation_with": other,
                "contact": pid,
                "sender": self.user_name if from_me else other,
                "body": body,
                "cluster": f"conv:{pid}",
                "sent": ts.strftime("%Y-%m-%d %H:%M"),
            },
        )

    def note(self, ts: datetime, title: str, body: str, oid: str | None = None) -> str:
        if oid is None:
            oid = f"notes:n{self.nid('note'):03d}"
            return self.emit(ts, "notes", oid, "insert", {"title": title, "body": body})
        return self.emit(ts, "notes", oid, "update", {"title": title, "body": body})

    def add_person(
        self, relation: str, employer: str, city: str, day: float = 0, ts: datetime | None = None
    ) -> str:
        first, last = self.new_name()
        pid = f"p{len(self.people) + 1:02d}"
        name = f"{first} {last}"
        mail = f"{first.lower()}.{last.lower()}@{self.domain(employer)}"
        self.people[pid] = Person(pid, name, mail, relation)
        card = {
            "name": name,
            "email": mail,
            "phone": f"+1-555-{self.rng.randint(100, 999)}-{self.rng.randint(1000, 9999)}",
            "employer": employer,
            "city": city,
            "relation": relation,
            "starred": self.rng.random() < 0.25,
            "cluster": f"contacts:{pid}",
        }
        self.cards[pid] = card
        self.emit(ts or self.at(day, 8, 0), "contacts", f"contacts:{pid}", "insert", card)
        return pid

    def update_card(self, ts: datetime, pid: str, **fields: Any) -> str:
        self.cards[pid].update(fields)
        return self.emit(ts, "contacts", f"contacts:{pid}", "update", fields)

    def set_timeline(self, slot_id: str, ts: datetime, value: Any, deps: list[str]) -> None:
        self.timelines.setdefault(slot_id, []).append(TimelinePoint(ts, value, deps))

    def current(self, slot_id: str, ts: datetime) -> Any:
        val = None
        for pt in sorted(self.timelines.get(slot_id, []), key=lambda p: p.ts):
            if pt.ts <= ts:
                val = pt.value
        return val

    def employer_at(self, day: int) -> str:
        return str(self.current("u.employer", self.start + timedelta(days=day)))

    def colleagues_at(self, day: int) -> list[str]:
        emp = self.employer_at(day)
        t = self.start + timedelta(days=day)
        return [
            p
            for p, c in self.cards.items()
            if self.people[p].relation == "colleague"
            and c["employer"] == emp
            and self.joined.get(p, self.start) <= t
        ]

    # ------------------------------------------------------------------ script
    def build_natural(self) -> None:
        rng = self.rng
        first, last = self.new_name()
        self.user_name = f"{first} {last}"
        self.user_first = first
        orgs = list(pools.ORGS)
        rng.shuffle(orgs)
        self.orgs = orgs
        cities = list(pools.CITIES)
        rng.shuffle(cities)
        self.home_city = cities[0][0]
        self.other_cities = [c for c, _ in cities[1:]]
        self.employer0 = orgs[0]
        self.joined: dict[str, datetime] = {}

        colleagues = [
            self.add_person("colleague", self.employer0, self.home_city) for _ in range(8)
        ]
        for _ in range(2):
            colleagues.append(self.add_person("colleague", orgs[1], rng.choice(self.other_cities)))
        near = [self.home_city, *self.other_cities[:4]]
        self.friends = [
            self.add_person("friend", rng.choice(orgs[2:]), rng.choice(near)) for _ in range(9)
        ]
        self.family = [
            self.add_person("family", rng.choice(orgs[2:]), rng.choice(self.other_cities[:5]))
            for _ in range(4)
        ]
        self.others = [
            self.add_person("other", rng.choice(orgs[2:]), rng.choice(self.other_cities))
            for _ in range(int(self.p["n_contacts"]) - 23)
        ]
        self.colleagues0 = colleagues[:8]

        self.emit(
            self.at(0, 7, 0),
            "calendar",
            "calendar:settings",
            "insert",
            {
                "kind": "settings",
                "timezone": pools.CITY_TZ[self.home_city],
                "cluster": "calendar:settings",
            },
        )
        self.set_timeline(
            "u.timezone", self.at(0, 7, 0), pools.CITY_TZ[self.home_city], ["calendar:settings"]
        )

        self._user_facts()
        self._series()
        self._appointments()
        self._tasks()
        self._contact_changes()
        self._meetings()
        self._messages()
        self._work_emails()
        self._notes()
        self._newsletters()
        self._slots()

    # user attributes -----------------------------------------------------------
    def _user_facts(self) -> None:
        rng = self.rng
        n = self.total_days()
        e0 = self.employer0
        m0 = self.colleagues0[0]
        t_emp = self.at(rng.randint(2, 6), 9)
        ev = self.email(
            t_emp,
            f"{e0} People Team",
            f"people@{self.domain(e0)}",
            f"Benefits enrollment at {e0}",
            f"Hi {self.user_first},\n\nAs an employee of {e0}, you can update your benefits "
            f"elections until the end of the month. Your manager {self.people[m0].name} has been "
            f"copied for visibility.\n\nThanks,\n{e0} People Team",
        )
        self.set_timeline("u.employer", t_emp, e0, [ev])
        self.set_timeline("u.manager", t_emp, self.people[m0].name, [ev])
        self.employer_now, self.manager_now = e0, m0

        jc_day = self.rday(60, n - 40) if rng.random() < 0.6 else None
        if rng.random() < 0.45:
            lo, hi = (20, jc_day - 20) if jc_day else (20, n - 20)
            if hi > lo:
                self.reorg(rng.randint(lo, hi))
        if jc_day:
            self.job_change(jc_day)
            if rng.random() < 0.3 and jc_day + 30 < n - 10:
                self.reorg(self.rday(jc_day + 30, n - 10))

        street = rng.choice(pools.STREETS)
        num = rng.randint(10, 480)
        t_home = self.at(rng.randint(1, 8))
        self.home_note = self.note(
            t_home,
            "Home info",
            f"Home address: {num} {street}, {self.home_city}.\n"
            f"Building code 4{rng.randint(100, 999)}. Trash day is Thursday.",
        )
        self.set_timeline("u.home_city", t_home, self.home_city, [self.home_note])
        if rng.random() < 0.5:
            self.move(self.rday(40, n - 10))

        diet = rng.choice(pools.DIETS)
        t_diet = self.at(rng.randint(3, 20))
        msg = self.message(
            t_diet,
            rng.choice(self.friends),
            True,
            f"Just so you know for Friday's dinner: I'm {pools.DIET_PHRASES[diet]}.",
        )
        t_meal = self.at(rng.randint(1, 10))
        self.meal_note = self.note(
            t_meal, "Meal plan", f"Current diet: {diet}. Groceries on Sundays."
        )
        self.set_timeline("u.diet", min(t_diet, t_meal), diet, [msg, self.meal_note])
        if rng.random() < 0.5:
            self.diet_change(self.rday(40, n - 5))

        commute = rng.choice(pools.COMMUTES)
        t_comm = self.at(rng.randint(2, 15))
        msg = self.message(
            t_comm,
            rng.choice(self.friends),
            True,
            f"These days I'm {pools.COMMUTE_PHRASES[commute]}, takes about "
            f"{rng.randint(15, 45)} minutes.",
        )
        self.set_timeline("u.commute", t_comm, commute, [msg])
        for _ in range(rng.choice((0, 1, 1, 2))):
            self.commute_change(self.rday(30, n - 5))

    def reorg(self, day: int, ts: datetime | None = None) -> None:
        emp = self.employer_now
        t = ts or self.at(day, 9)
        pool = [
            p
            for p, c in self.cards.items()
            if c["employer"] == emp
            and self.people[p].relation == "colleague"
            and p != self.manager_now
            and self.joined.get(p, self.start) <= t
        ]
        if not pool:
            return
        new_m = self.rng.choice(pool)
        ev = self.email(
            t,
            f"{emp} People Team",
            f"people@{self.domain(emp)}",
            "Org update",
            f"Hi team,\n\nStarting {t:%B %d, %Y}, {self.user_first} will report to "
            f"{self.people[new_m].name}. {self.people[self.manager_now].name} is moving to a new "
            f"role within {emp}.\n\nBest,\n{emp} People Team",
        )
        self.set_timeline("u.manager", t, self.people[new_m].name, [ev])
        self.manager_now = new_m

    def job_change(self, day: int, ts: datetime | None = None) -> None:
        rng = self.rng
        old = self.employer_now
        choices = [o for o in (self.orgs[1], *self.orgs[2:8]) if o != old]
        new = rng.choice(choices)
        t = ts or self.at(day, 9)
        if ts is None:
            self.email(
                self.at(max(day - 14, 1), 17),
                self.people[self.manager_now].name,
                self.people[self.manager_now].email,
                "Re: next steps",
                f"Thanks for letting me know, {self.user_first}. We'll miss you here at {old}. "
                f"Let's plan the handover before your last day.",
            )
            other = self.orgs[12]
            self.email(
                self.at(max(day - 30, 1)),
                f"Recruiting at {other}",
                f"talent@{self.domain(other)}",
                f"Opportunities at {other}",
                f"Hi {self.user_first}, we'd love to tell you about open roles at {other}. "
                f"Would you have 20 minutes next week?",
            )
        new_colleagues = []
        for _ in range(4):
            pid = self.add_person("colleague", new, self.home_city, ts=t - timedelta(minutes=5))
            self.joined[pid] = t
            new_colleagues.append(pid)
        mgr = new_colleagues[0]
        ev = self.email(
            t,
            f"{new} People Team",
            f"people@{self.domain(new)}",
            f"Welcome to {new}!",
            f"Hi {self.user_first},\n\nWelcome to {new}! We're thrilled you're joining us. Your "
            f"first day is {t:%A, %B %d, %Y}. Your manager will be {self.people[mgr].name}, who "
            f"will meet you at reception at 9:30.\n\nSee you soon,\n{new} People Team",
        )
        msg = self.message(
            t + timedelta(days=rng.randint(1, 4), hours=8),
            rng.choice(self.friends),
            True,
            f"First week at {new} done! My new manager {self.people[mgr].name.split()[0]} "
            f"seems great.",
        )
        self.set_timeline("u.employer", t, new, [ev, msg])
        self.set_timeline("u.manager", t, self.people[mgr].name, [ev])
        self.employer_now, self.manager_now = new, mgr

    def move(self, day: int, ts: datetime | None = None) -> None:
        rng = self.rng
        city = rng.choice([c for c in self.other_cities[:6] if c != self.home_city])
        street = rng.choice(pools.STREETS)
        num = rng.randint(10, 480)
        t = ts or self.at(day, 11)
        ev = self.email(
            t,
            "Keystone Leasing",
            "leasing@keystone.example",
            "Lease confirmation",
            f"Dear {self.user_name},\n\nThis confirms your lease at {num} {street}, {city}, "
            f"beginning {t:%B %d, %Y}. Keys can be picked up at our office from 10am.\n\n"
            f"Keystone Leasing",
        )
        self.note(
            t + timedelta(hours=2),
            "Home info",
            f"Home address: {num} {street}, {city}.\nMoved in {t:%B %d, %Y}.",
            oid=self.home_note,
        )
        msg = self.message(
            t + timedelta(days=rng.randint(1, 5)),
            rng.choice(self.family),
            True,
            f"We're all moved in to the new place in {city}!",
        )
        self.set_timeline("u.home_city", t, city, [ev, self.home_note, msg])
        old_tz = self.current("u.timezone", t)
        tz = pools.CITY_TZ[city]
        if tz != old_tz:
            t2 = t + timedelta(hours=3)
            self.emit(t2, "calendar", "calendar:settings", "update", {"timezone": tz})
            self.set_timeline("u.timezone", t2, tz, ["calendar:settings"])
        self.home_city = city

    def set_timezone(self, ts: datetime, tz: str) -> None:
        self.emit(ts, "calendar", "calendar:settings", "update", {"timezone": tz})
        self.set_timeline("u.timezone", ts, tz, ["calendar:settings"])

    def diet_change(self, day: int, ts: datetime | None = None) -> None:
        t = ts or self.at(day)
        old = self.current("u.diet", t)
        diet = self.rng.choice([d for d in pools.DIETS if d != old])
        msg = self.message(
            t,
            self.rng.choice(self.friends),
            True,
            f"Heads up for our next dinner: I've switched and I'm {pools.DIET_PHRASES[diet]} now.",
        )
        self.note(
            t + timedelta(hours=1),
            "Meal plan",
            f"Current diet: {diet}. Groceries on Sundays.",
            oid=self.meal_note,
        )
        self.set_timeline("u.diet", t, diet, [msg, self.meal_note])

    def commute_change(self, day: int, ts: datetime | None = None) -> None:
        t = ts or self.at(day)
        old = self.current("u.commute", t)
        mode = self.rng.choice([c for c in pools.COMMUTES if c != old])
        msg = self.message(
            t,
            self.rng.choice(self.friends),
            True,
            f"Small life update: starting this week I'm {pools.COMMUTE_PHRASES[mode]} instead.",
        )
        self.set_timeline("u.commute", t, mode, [msg])

    # recurring calendar series ---------------------------------------------------
    def _series(self) -> None:
        rng = self.rng
        n = self.total_days()
        chosen = [pools.SERIES[0], pools.SERIES[1], *rng.sample(pools.SERIES[2:], 2)]
        self.series_ids: list[tuple[str, str, str, list[str]]] = []
        for title, cat in chosen:
            sid = f"calendar:series_{title.lower().replace(' ', '_')}"
            wd = rng.randint(0, 4) if cat == "work" else rng.randint(0, 6)
            hh = rng.choice((9, 10, 11, 14, 15, 16)) if cat == "work" else rng.choice((7, 18, 19))
            att = rng.sample(self.colleagues0, 3) if cat == "work" else rng.sample(self.friends, 2)
            self.emit(
                self.at(rng.randint(0, 10), 8),
                "calendar",
                sid,
                "insert",
                {
                    "kind": "series",
                    "title": title,
                    "weekday": pools.WEEKDAYS[wd],
                    "time": f"{hh:02d}:{rng.choice((0, 30)):02d}",
                    "duration_min": 30 if cat == "work" else 60,
                    "attendees": [self.people[a].name for a in att],
                    "category": cat,
                    "cluster": sid,
                },
            )
            self.series_ids.append((sid, title, cat, att))
            for _ in range(rng.choice((0, 1, 1, 2))):
                self.reschedule_series(sid, self.rday(20, n - 5))
            if rng.random() < 0.15:
                self.cancel_series(sid, rng.randint(n // 2, n - 3))

    def _series_info(self, sid: str) -> tuple[str, str, list[str]]:
        for s, title, cat, att in self.series_ids:
            if s == sid:
                return title, cat, att
        raise KeyError(sid)

    def reschedule_series(self, sid: str, day: int, ts: datetime | None = None) -> None:
        rng = self.rng
        title, cat, att = self._series_info(sid)
        wd2 = rng.randint(0, 4) if cat == "work" else rng.randint(0, 6)
        hours = (9, 10, 11, 13, 14, 15, 16) if cat == "work" else (7, 17, 18, 19, 20)
        t2 = f"{rng.choice(hours):02d}:{rng.choice((0, 30)):02d}"
        t = ts or self.at(day, 10)
        self.emit(t, "calendar", sid, "update", {"weekday": pools.WEEKDAYS[wd2], "time": t2})
        if rng.random() < 0.6:
            who = self.people[att[0]]
            self.email(
                t + timedelta(minutes=5),
                who.name,
                who.email,
                f"{title} moving",
                f"Hi all, from now on {title} will be on {pools.WEEKDAYS[wd2]}s at {t2}. "
                f"Calendar invite updated.",
            )

    def cancel_series(self, sid: str, day: int, ts: datetime | None = None) -> None:
        title, _, att = self._series_info(sid)
        t = ts or self.at(day, 12)
        self.emit(t, "calendar", sid, "delete", {})
        who = self.people[att[0]]
        self.email(
            t + timedelta(minutes=10),
            who.name,
            who.email,
            f"{title} cancelled",
            f"Hi all, {title} is cancelled going forward. Thanks for joining!",
        )

    # appointments ------------------------------------------------------------------
    def _appointments(self) -> None:
        rng = self.rng
        n = self.total_days()
        for atype, provider in pools.APPOINTMENT_TYPES:
            day = rng.randint(5, 40)
            while day < n - 3:
                book = max(1, day - rng.randint(7, 25))
                start = self.at(day, rng.choice((8, 9, 10, 11, 14, 15, 16)))
                oid = self.book_appointment(atype, provider, self.at(book, 10), start)
                r = rng.random()
                if r < 0.25 and day - book > 3:
                    day = day + rng.randint(2, 10)
                    new_start = self.at(day, rng.choice((9, 13, 16)))
                    self.move_appointment(
                        oid, atype, provider, self.at(rng.randint(book + 1, day - 1), 12), new_start
                    )
                elif r < 0.35 and day - book > 3:
                    self.cancel_appointment(
                        oid, atype, provider, self.at(rng.randint(book + 1, day - 1), 12), start
                    )
                day += rng.randint(45, 120) if atype == "dentist" else rng.randint(30, 80)

    @staticmethod
    def _appt_addr(provider: str) -> str:
        return f"appointments@{provider.lower().split()[0]}.example"

    def book_appointment(self, atype: str, provider: str, ts: datetime, start: datetime) -> str:
        oid = f"calendar:ev{self.nid('ev'):04d}"
        self.emit(
            ts,
            "calendar",
            oid,
            "insert",
            {
                "kind": "appointment",
                "appt_type": atype,
                "provider": provider,
                "title": f"{provider} ({atype})",
                "start": start.strftime("%Y-%m-%d %H:%M"),
                "duration_min": 45,
                "cluster": oid,
            },
        )
        self.email(
            ts + timedelta(minutes=5),
            provider,
            self._appt_addr(provider),
            "Appointment confirmation",
            f"Dear {self.user_name}, your {atype} appointment at {provider} is confirmed for "
            f"{start:%A, %B %d, %Y at %H:%M}. Reply to reschedule.",
        )
        return oid

    def move_appointment(
        self, oid: str, atype: str, provider: str, ts: datetime, new_start: datetime
    ) -> None:
        self.emit(ts, "calendar", oid, "update", {"start": new_start.strftime("%Y-%m-%d %H:%M")})
        self.email(
            ts + timedelta(minutes=3),
            provider,
            self._appt_addr(provider),
            "Appointment rescheduled",
            f"Your {atype} appointment has been moved to {new_start:%A, %B %d, %Y at %H:%M}.",
        )

    def cancel_appointment(
        self, oid: str, atype: str, provider: str, ts: datetime, start: datetime
    ) -> None:
        self.emit(ts, "calendar", oid, "delete", {})
        self.email(
            ts + timedelta(minutes=3),
            provider,
            self._appt_addr(provider),
            "Appointment cancelled",
            f"Your {atype} appointment on {start:%B %d} has been cancelled.",
        )

    # tasks ------------------------------------------------------------------------
    def _tasks(self) -> None:
        rng = self.rng
        n = self.total_days()
        projects = rng.sample(pools.PROJECTS, 5)
        self.projects = projects
        titles = [f"{v} {o}" for v in pools.TASK_VERBS for o in pools.TASK_OBJECTS]
        rng.shuffle(titles)
        self.task_titles = titles
        self.task_ids: list[tuple[str, str]] = []
        for i in range(40):
            created = rng.randint(0, n - 10)
            oid = f"tasks:t{i + 1:03d}"
            proj = projects[min(len(projects) - 1, int(created / n * len(projects) + rng.random()))]
            due = self.start + timedelta(days=created + rng.randint(5, 45))
            self.emit(
                self.at(created, 9),
                "tasks",
                oid,
                "insert",
                {
                    "title": titles[i],
                    "project": proj,
                    "due": due.strftime("%Y-%m-%d"),
                    "status": "open",
                    "priority": rng.choice(("low", "medium", "high")),
                    "cluster": oid,
                },
            )
            self.task_ids.append((oid, titles[i]))
            day = created
            for _ in range(rng.choice((0, 1, 1, 2))):
                day += rng.randint(2, 20)
                if day >= n:
                    break
                due = due + timedelta(days=rng.randint(3, 21))
                self.emit(
                    self.at(day, 11), "tasks", oid, "update", {"due": due.strftime("%Y-%m-%d")}
                )
            done_day = day + rng.randint(3, 40)
            if rng.random() < 0.85 and done_day < n:
                self.emit(self.at(done_day, 17), "tasks", oid, "update", {"status": "done"})
            elif rng.random() < 0.15 and done_day < n:
                self.emit(self.at(done_day, 17), "tasks", oid, "delete", {})

    # contact attribute changes ------------------------------------------------------
    def _contact_changes(self) -> None:
        rng = self.rng
        n = self.total_days()
        pool = self.friends + self.family + self.others + self.colleagues0[4:]
        self.tracked_contacts = rng.sample(pool, int(self.p["n_tracked_contacts"]))
        for pid in self.tracked_contacts:
            for field_ in ("employer", "city"):
                if rng.random() < 0.35:
                    self.change_contact(pid, field_, self.at(self.rday(10, n - 5), 10))

    def change_contact(self, pid: str, field_: str, ts: datetime) -> None:
        rng = self.rng
        card = self.cards[pid]
        if field_ == "employer":
            val = rng.choice([o for o in self.orgs[2:] if o != card["employer"]])
            line = f"Big news: I've started a new job at {val}!"
        else:
            val = rng.choice([c for c in [*self.other_cities, self.home_city] if c != card["city"]])
            line = f"Guess what, I moved to {val} last weekend!"
        self.update_card(ts, pid, **{field_: val})
        if rng.random() < 0.6:
            person = self.people[pid]
            if person.relation in ("friend", "family"):
                self.message(ts + timedelta(hours=1), pid, False, line)
            else:
                self.email(ts + timedelta(hours=1), person.name, person.email, "Update", line)

    # one-off meetings (important_contact, working_hours) ------------------------------
    def meeting(self, start: datetime, pids: list[str], work: bool, dur: int) -> str:
        rng = self.rng
        oid = f"calendar:ev{self.nid('ev'):04d}"
        first = self.people[pids[0]].name.split()[0]
        title = f"{rng.choice(pools.PROJECTS)} sync" if work else f"Coffee with {first}"
        self.emit(
            start - timedelta(days=rng.randint(1, 6)),
            "calendar",
            oid,
            "insert",
            {
                "kind": "meeting",
                "title": title,
                "start": start.strftime("%Y-%m-%d %H:%M"),
                "end": (start + timedelta(minutes=dur)).strftime("%Y-%m-%d %H:%M"),
                "attendees": [self.people[x].name for x in pids],
                "category": "work" if work else "social",
                "cluster": oid,
            },
        )
        return oid

    def _meetings(self) -> None:
        rng = self.rng
        n = self.total_days()
        social = self.friends + self.family
        for q in range(0, n, 30):
            colleagues = self.colleagues_at(q)
            hot = rng.sample(colleagues, min(3, len(colleagues))) + rng.sample(social, 2)
            for pid in colleagues + social:
                rate = 1.8 if pid in hot else 0.3
                for _ in range(poisson(rng, rate)):
                    day = q + rng.randint(0, 29)
                    if day >= n:
                        continue
                    work = self.people[pid].relation == "colleague"
                    hh = (
                        rng.choice((8, 9, 10, 11, 13, 14, 15, 16, 17))
                        if work
                        else rng.choice((12, 18, 19))
                    )
                    pids = [pid]
                    if work and rng.random() < 0.4:
                        extra = rng.choice(colleagues)
                        if extra != pid:
                            pids.append(extra)
                    self.meeting(
                        self.at(day, hh, rng.choice((0, 30))), pids, work, rng.choice((30, 45, 60))
                    )

    # messages with friends and family (close_friend) ----------------------------------
    def _messages(self) -> None:
        rng = self.rng
        n = self.total_days()
        for q in range(0, n, 30):
            hot = rng.sample(self.friends + self.family, 3)
            for pid in self.friends + self.family:
                rate = 10.0 if pid in hot else 2.5
                for _ in range(poisson(rng, rate)):
                    day = q + rng.randint(0, 29)
                    if day < n:
                        self.message(
                            self.at(day), pid, rng.random() < 0.5, rng.choice(pools.SMALLTALK)
                        )

    # work email (vip_sender) --------------------------------------------------------
    def work_email(self, ts: datetime, pid: str) -> str:
        who = self.people[pid]
        body = " ".join(self.rng.sample(pools.WORK_EMAIL_LINES, 3))
        subject = self.rng.choice(
            ("Quick question", "Notes", "Follow-up", "Draft for review", "Re: planning")
        )
        return self.email(
            ts,
            who.name,
            who.email,
            subject,
            f"Hi {self.user_first},\n\n{body}\n\nBest,\n{who.name.split()[0]}",
        )

    def _work_emails(self) -> None:
        rng = self.rng
        n = self.total_days()
        for q in range(0, n, 30):
            colleagues = self.colleagues_at(q)
            hot = rng.sample(colleagues, min(2, len(colleagues)))
            for pid in colleagues:
                rate = 5.0 if pid in hot else 2.0
                for _ in range(poisson(rng, rate)):
                    day = q + rng.randint(0, 29)
                    if day < n:
                        self.work_email(self.at(day, rng.randint(8, 18)), pid)

    def _notes(self) -> None:
        rng = self.rng
        n = self.total_days()
        h0 = rng.choice((8, 9))
        self.wh_note = self.note(
            self.at(1, 20),
            "Working hours",
            f"My working hours: {h0:02d}:00-{h0 + 8:02d}:00. No meetings before {h0:02d}:30 if "
            f"possible.",
        )
        for _ in range(rng.choice((1, 2))):
            self.update_working_hours(self.at(self.rday(30, n - 5), 21))
        for _ in range(12):
            self.note(
                self.at(rng.randint(0, n - 1)),
                rng.choice(("Ideas", "To remember", "List")),
                " ".join(rng.sample(pools.NOTE_LINES, 3)),
            )

    def update_working_hours(self, ts: datetime) -> None:
        rng = self.rng
        h = rng.choice((7, 8, 9, 10))
        self.note(
            ts,
            "Working hours",
            f"My working hours: {h:02d}:00-{h + rng.choice((8, 9)):02d}:00. Updated after talking "
            f"with my manager.",
            oid=self.wh_note,
        )

    def _newsletters(self) -> None:
        rng = self.rng
        n = self.total_days()
        per_month = float(self.p["newsletter_per_topic_per_month"])
        per_topic = int(per_month * n / 30)
        count = len(pools.NEWSLETTER_TOPICS) * per_topic
        # ~17 tokens per filler sentence; newsletters carry ~90% of the target volume
        sentences = max(20, int(float(self.p["target_tokens"]) * 0.9 / max(count, 1) / 17))
        for topic in pools.NEWSLETTER_TOPICS:
            bank = pools.FILLER[topic]
            for k in range(per_topic):
                day = min(n - 1, int(k * 30 / per_month) + rng.randint(0, 3))
                parts = [
                    rng.choice(bank).format(
                        org=rng.choice(self.orgs),
                        city=rng.choice(pools.CITIES)[0],
                        street=rng.choice(pools.STREETS),
                    )
                    for _ in range(sentences)
                ]
                paras = [" ".join(parts[i : i + 6]) for i in range(0, len(parts), 6)]
                self.email(
                    self.at(day, 6, 30),
                    f"The {topic.title()} Digest",
                    f"digest@{topic}news.example",
                    f"{topic.title()} Digest #{k + 1}",
                    "\n\n".join(paras),
                    thread=f"thread:digest-{topic}",
                )

    # slots ------------------------------------------------------------------------
    def _slots(self) -> None:
        s = self.slots
        s.append(
            Slot(
                "u.employer",
                "user_attr",
                "org",
                "Which company is the user currently employed by?",
                "the company the user currently works for",
                "employer job company welcome first day works joined",
                ("email", "messages", "notes"),
            )
        )
        s.append(
            Slot(
                "u.manager",
                "user_attr",
                "person",
                "Who is the user's current manager (full name)?",
                "the person the user currently reports to (their manager)",
                "manager report to org update welcome",
                ("email", "messages"),
            )
        )
        s.append(
            Slot(
                "u.home_city",
                "user_attr",
                "city",
                "In which city does the user currently live?",
                "the city where the user currently lives",
                "home address lease moved city apartment",
                ("email", "notes", "messages"),
            )
        )
        s.append(
            Slot(
                "u.timezone",
                "user_attr",
                "tz",
                "What is the user's current home time zone (IANA name, e.g. Europe/Paris)?",
                "the user's current home time zone as an IANA name",
                "time zone timezone calendar settings",
                ("calendar",),
                structured=True,
            )
        )
        s.append(
            Slot(
                "u.diet",
                "user_attr",
                "diet",
                "What is the user's current diet? Answer with one of: omnivore, vegetarian, vegan,"
                " pescatarian, gluten-free.",
                "the user's current diet (omnivore, vegetarian, vegan, pescatarian or gluten-free)",
                "diet vegetarian vegan pescatarian gluten meal plan dinner",
                ("messages", "notes"),
            )
        )
        s.append(
            Slot(
                "u.commute",
                "user_attr",
                "commute",
                "How does the user currently commute to work? Answer with one of: bike, train, "
                "car, bus, walk.",
                "how the user currently commutes to work (bike, train, car, bus or walk)",
                "commute biking train driving bus walking work",
                ("messages",),
            )
        )
        for pid in self.tracked_contacts:
            name = self.people[pid].name
            s.append(
                Slot(
                    f"c.{pid}.employer",
                    "contact_attr",
                    "org",
                    f"Which company does {name} currently work for?",
                    f"the company {name} currently works for",
                    f"{name} company employer job",
                    ("contacts", "email", "messages"),
                    subject=pid,
                    structured=True,
                )
            )
            s.append(
                Slot(
                    f"c.{pid}.city",
                    "contact_attr",
                    "city",
                    f"In which city does {name} currently live?",
                    f"the city where {name} currently lives",
                    f"{name} city moved lives",
                    ("contacts", "email", "messages"),
                    subject=pid,
                    structured=True,
                )
            )
        for sid, title, _, _ in self.series_ids:
            s.append(
                Slot(
                    f"s.{sid.split(':')[-1]}",
                    "series",
                    "weekly",
                    f"On which weekday and at what time does the user's recurring '{title}' take "
                    f"place?",
                    f"the weekday and start time of the recurring calendar event '{title}'",
                    f"{title} recurring every weekly",
                    ("calendar", "email"),
                    subject=sid,
                    structured=True,
                )
            )
        for atype, provider in pools.APPOINTMENT_TYPES:
            s.append(
                Slot(
                    f"a.{atype}",
                    "appointment",
                    "datetime",
                    f"When is the user's next {atype} appointment (date and time)?",
                    f"the date and time of the user's next upcoming {atype} appointment",
                    f"{atype} appointment {provider} confirmed rescheduled",
                    ("calendar", "email"),
                    subject=atype,
                    structured=True,
                )
            )
        alive = [(oid, title) for oid, title in self.task_ids if self.alive(oid, self.fork)]
        k = min(int(self.p["n_tracked_tasks"]), len(alive))
        for oid, title in sorted(self.rng.sample(alive, k)):
            s.append(
                Slot(
                    f"t.{oid.split(':')[-1]}",
                    "task_due",
                    "date",
                    f"What is the due date of the task '{title}'?",
                    f"the due date of the task '{title}'",
                    f"{title} task due",
                    ("tasks",),
                    subject=oid,
                    structured=True,
                )
            )
        for kid, spec in KAPPA_DEFS.items():
            s.append(
                Slot(
                    f"k.{kid}",
                    "kappa",
                    str(spec["vtype"]),
                    str(spec["question"]),
                    str(spec["description"]),
                    str(spec["query"]),
                    tuple(spec["sources"]),
                    kappa_id=kid,
                    structured=True,
                )
            )

    def alive(self, oid: str, t: datetime) -> bool:
        alive = False
        for r in sorted(self.raw, key=lambda r: (r.ts, r.seq)):
            if r.object_id == oid and r.ts <= t:
                alive = r.op != "delete"
        return alive

    # ------------------------------------------------------------------ materialise
    def materialise(self) -> list[Event]:
        state: dict[str, dict[str, Any]] = {}
        revs: dict[str, int] = {}
        events: list[Event] = []
        for r in sorted(self.raw, key=lambda r: (r.ts, r.seq)):
            if r.op == "insert":
                payload = dict(r.patch)
            elif r.object_id not in state:
                continue  # update or delete of an object that no longer exists
            elif r.op == "update":
                payload = {**state[r.object_id], **r.patch}
            else:
                payload = dict(state[r.object_id])
            revs[r.object_id] = revs.get(r.object_id, 0) + 1
            if r.source in ("tasks", "notes"):
                payload["updated"] = r.ts.strftime("%Y-%m-%d %H:%M")
            payload.setdefault("cluster", r.object_id)
            payload["text"] = render_text(r.source, payload)
            if r.op == "delete":
                state.pop(r.object_id, None)
            else:
                state[r.object_id] = payload
            events.append(
                Event(
                    f"ev{len(events) + 1:06d}",
                    r.ts,
                    r.source,
                    r.object_id,
                    revs[r.object_id],
                    r.op,
                    payload,
                    r.tag,
                )
            )
        return events


def build(seed: int, params: dict[str, Any] | None = None) -> Builder:
    b = Builder(seed, params or {})
    b.build_natural()
    return b


def to_world(b: Builder, branch: str = "natural") -> World:
    kappa_defs = {
        kid: {v: KappaDef(kid, v, str(spec[v])) for v in ("v1", "v2")}
        for kid, spec in KAPPA_DEFS.items()
    }
    return World(
        world_id=f"w{b.seed:04d}",
        seed=b.seed,
        start=b.start,
        end=b.end,
        fork=b.fork,
        user_name=b.user_name,
        people=b.people,
        slots=b.slots,
        events=b.materialise(),
        timelines={k: sorted(v, key=lambda p: p.ts) for k, v in b.timelines.items()},
        kappa_defs=kappa_defs,
        kappa_log=[(b.start, kid, "v1") for kid in KAPPA_DEFS],
        agents={k: list(v) for k, v in AGENTS.items()},
        branch=branch,
        params=dict(b.p),
    )


def generate_world(seed: int, params: dict[str, Any] | None = None) -> World:
    return to_world(build(seed, params))

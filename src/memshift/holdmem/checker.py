"""HoldMem write path, step 2: the D=1 deterministic checker and programmatic fallback.

check(parse, spans, sources, kappa): every citation must name a live object at its current
revision and quote text that occurs in it; the parsed value must be supported by the quotes
(typed surface match); upcoming-event values must lie after the as-of time; definition-dependent
values must equal the definition's evaluator output. On failure, structured sources (contacts,
calendar, tasks, calendar settings, kappa evaluators) are re-read by rule; if no rule exists the
item is `unverifiable`.

Honest limitation: the checker verifies support, not recency. A value quoted from an older but
still-live record passes, which is why D=1 repair is not perfect on unstructured facts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from memshift.gen import pools
from memshift.gen.kappa import evaluate
from memshift.gen.model import Slot, World
from memshift.gen.state import SourceLog
from memshift.holdmem.writer import Citation
from memshift.scoring.normalize import (
    SYNONYMS,
    as_list,
    norm_str,
    parse_date,
    parse_time,
    values_match,
)

_WS = re.compile(r"\s+")
_WEEKDAY_RE = re.compile(r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)")


def _squash(s: str) -> str:
    return _WS.sub(" ", s).strip().lower()


def _has_phrase(phrase: str, text: str) -> bool:
    """Whole-word containment of a normalised phrase in normalised text."""
    p = norm_str(phrase)
    return bool(p) and re.search(rf"(^| ){re.escape(p)}( |$)", norm_str(text)) is not None


_SEGMENT_SPLIT = re.compile(r"\n+|\.\.\.|\u2026")


def locate_quote(quote: str, text: str) -> tuple[int, int] | None:
    """Find a quote in text: each line / ellipsis-separated segment verbatim (modulo whitespace
    and case) and in order. Returns the (start, end) of the matched region in squashed text."""
    body = _squash(text)
    segments = [_squash(s) for s in _SEGMENT_SPLIT.split(quote) if s.strip()]
    if not segments:
        return None
    pos = start = -1
    for seg in segments:
        found = body.find(seg, pos + 1 if pos >= 0 else 0)
        if found < 0:
            return None
        if start < 0:
            start = found
        pos = found + len(seg) - 1
    return start, pos + 1


@dataclass
class CheckResult:
    ok: bool
    reason: str
    spans: list[dict[str, Any]] = field(default_factory=list)


def _enum_supported(vtype: str, v: str, text: str) -> bool:
    canon = [c for c, words in SYNONYMS[vtype].items() if any(_has_phrase(w, v) for w in words)]
    if len(canon) != 1:
        return False
    phrases = list(SYNONYMS[vtype][canon[0]])
    if vtype == "diet":
        phrases.append(pools.DIET_PHRASES[canon[0]])
    return any(_has_phrase(p, text) for p in phrases)


def _surface_ok(slot: Slot, value: Any, text: str) -> bool:
    """Is `value` (typed) visibly supported by `text`?"""
    if slot.vtype.startswith("set_"):
        return all(_has_phrase(v, text) for v in as_list(value))
    v = value if isinstance(value, str) else ", ".join(map(str, value))
    if slot.vtype in ("diet", "commute"):
        return _enum_supported(slot.vtype, v, text)
    if slot.vtype == "weekly":
        m = _WEEKDAY_RE.search(v.lower())
        tm = parse_time(v)
        return bool(m and tm and m.group(1) in text.lower() and f"{tm[0]:02d}:{tm[1]:02d}" in text)
    if slot.vtype in ("datetime", "date"):
        d = parse_date(v)
        if d is None:
            return False
        forms = [d.isoformat(), f"{d:%B} {d.day:02d}, {d.year}", f"{d:%B} {d.day}, {d.year}"]
        if not any(f in text for f in forms):
            return False
        if slot.vtype == "datetime":
            tm = parse_time(v[10:] if len(v) > 10 else v)
            return tm is not None and f"{tm[0]:02d}:{tm[1]:02d}" in text
        return True
    if slot.vtype == "timerange":
        times = re.findall(r"\d{1,2}:\d{2}", v)
        return len(times) == 2 and all(x in text for x in times)
    return _has_phrase(v, text)


def check(
    world: World, log: SourceLog, slot: Slot, t: datetime, value: Any, cits: list[Citation]
) -> CheckResult:
    snap = log.snapshot(t)
    if value is None or value == "":
        return CheckResult(False, "null value")
    if not cits:
        return CheckResult(False, "no citations")
    spans: list[dict[str, Any]] = []
    quotes: list[str] = []
    for c in cits:
        obj = snap.get(c.object_id)
        if obj is None:
            return CheckResult(False, f"cited object missing: {c.object_id}")
        if obj.revision != c.revision:
            return CheckResult(False, f"stale revision: {c.object_id} r{c.revision}")
        located = locate_quote(c.quote, obj.text)
        if located is None:
            return CheckResult(False, f"quote not found in {c.object_id}")
        spans.append(
            {
                "object_id": c.object_id,
                "revision": c.revision,
                "start": located[0],
                "end": located[1],
            }
        )
        quotes.append(c.quote)
    if slot.kind == "kappa":
        assert slot.kappa_id is not None
        expected, _ = evaluate(slot.kappa_id, snap, t, world.kappa_version(slot.kappa_id, t))
        if not values_match(slot.vtype, value, expected):
            return CheckResult(False, "definition check failed", spans)
        return CheckResult(True, "definition check passed", spans)
    if slot.kind == "appointment":
        d = parse_date(str(value))
        if d is None or d.isoformat() < t.strftime("%Y-%m-%d"):
            return CheckResult(False, "appointment not upcoming", spans)
    if not any(_surface_ok(slot, value, q) for q in quotes):
        return CheckResult(False, "value not supported by quotes", spans)
    return CheckResult(True, "spans verified", spans)


NO_RULE = "no-rule"
FallbackResult = tuple[Any, list[tuple[str, int]]]


def fallback(world: World, log: SourceLog, slot: Slot, t: datetime) -> FallbackResult | str:
    """Programmatic extraction from structured sources; NO_RULE when no rule exists."""
    snap = log.snapshot(t)
    if slot.slot_id == "u.timezone":
        o = snap.get("calendar:settings")
        return (o.payload["timezone"], [(o.object_id, o.revision)]) if o else (None, [])
    if slot.kind == "contact_attr":
        o = snap.get(f"contacts:{slot.subject}")
        if o is None:
            return None, []
        return o.payload[slot.slot_id.split(".")[-1]], [(o.object_id, o.revision)]
    if slot.kind == "series":
        o = snap.get(str(slot.subject))
        if o is None:
            return None, []
        return f"{o.payload['weekday']} {o.payload['time']}", [(o.object_id, o.revision)]
    if slot.kind == "task_due":
        o = snap.get(str(slot.subject))
        return (o.payload["due"], [(o.object_id, o.revision)]) if o else (None, [])
    if slot.kind == "appointment":
        now = t.strftime("%Y-%m-%d %H:%M")
        ups = sorted(
            (o.payload["start"], o.object_id, o.revision)
            for o in snap.by_source("calendar")
            if o.payload.get("kind") == "appointment"
            and o.payload["appt_type"] == slot.subject
            and o.payload["start"] > now
        )
        return (ups[0][0], [(ups[0][1], ups[0][2])]) if ups else (None, [])
    if slot.kind == "kappa":
        assert slot.kappa_id is not None
        value, ids = evaluate(slot.kappa_id, snap, t, world.kappa_version(slot.kappa_id, t))
        return value, [(i, snap.objects[i].revision) for i in ids[:6] if i in snap.objects]
    return NO_RULE


def render_fallback_sk(
    slot: Slot,
    t: datetime,
    value: Any,
    refs: list[tuple[str, int]],
    definition: str | None = None,
    origin: str = "programmatic extraction",
) -> str:
    """s_k for a value produced by the rule-based extractor; for definition items it names the
    definition version it was computed under (without it readers cannot tell a stale set from a
    current one and abstain). `origin` says who computed it (D1s: the writer's own program)."""
    head = f"[{slot.description} | written {t:%Y-%m-%d}]"
    under = f" under the definition in effect ({definition})" if definition else ""
    if value is None:
        body = f"As of {t:%Y-%m-%d}, there is no current value for {slot.description}{under}."
    elif isinstance(value, list):
        listed = ", ".join(value) if value else "none"
        body = f"As of {t:%Y-%m-%d}, {slot.description}{under}: {listed}."
    else:
        body = f"As of {t:%Y-%m-%d}, {slot.description}{under}: {value}."
    src = "; ".join(f"{o} r{r}" for o, r in refs[:6])
    return f"{head} {body} ({origin} from {src or 'structured sources'})"

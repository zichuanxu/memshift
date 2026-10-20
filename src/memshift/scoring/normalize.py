"""Typed answer matching: no LLM judge.

Dates, times and weekdays are parsed and compared as values; entity strings are compared after
standardisation (case, punctuation, whitespace), allowing at most three extra words around the
gold phrase ("at Northwind Labs"); enumerations use a fixed synonym table; sets compare as sets.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Any

WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday")
MONTHS = [
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
]
_MON_ALT = "|".join(MONTHS) + "|[a-z]{3}"

SYNONYMS = {
    "diet": {
        "omnivore": ("omnivore", "omnivorous", "eats everything"),
        "vegetarian": ("vegetarian", "veggie"),
        "vegan": ("vegan", "plant-based", "plant based"),
        "pescatarian": ("pescatarian", "pescetarian"),
        "gluten-free": ("gluten-free", "gluten free", "glutenfree"),
    },
    "commute": {
        "bike": ("bike", "biking", "bicycle", "cycling", "cycles", "cycle"),
        "train": ("train", "rail", "subway", "metro"),
        "car": ("car", "driving", "drives", "drive"),
        "bus": ("bus",),
        "walk": ("walk", "walking", "walks", "on foot"),
    },
}


def norm_str(s: Any) -> str:
    s = str(s).lower().replace("’", "'")
    s = re.sub(r"[^a-z0-9/_+\- ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return re.sub(r"^(the|a|an) ", "", s)


def _phrase_match(answer: Any, gold: Any, max_extra: int = 3) -> bool:
    a, g = norm_str(answer), norm_str(gold)
    if not g:
        return False
    if a == g:
        return True
    if re.search(rf"(^| ){re.escape(g)}( |$)", a):
        return len(a.split()) - len(g.split()) <= max_extra
    return False


def _enum(kind: str, answer: Any) -> str | None:
    a = norm_str(answer)
    hits = {
        canon
        for canon, words in SYNONYMS[kind].items()
        for w in words
        if re.search(rf"(^| ){re.escape(w)}( |$)", a)
    }
    return hits.pop() if len(hits) == 1 else None


_TIME_RE = re.compile(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?", re.I)


def parse_time(s: str) -> tuple[int, int] | None:
    for m in _TIME_RE.finditer(s):
        h, mnt, ampm = int(m.group(1)), m.group(2), m.group(3)
        if mnt is None and ampm is None:
            continue
        mm = int(mnt) if mnt else 0
        if ampm:
            ampm = ampm.lower().replace(".", "")
            if ampm == "pm" and h < 12:
                h += 12
            if ampm == "am" and h == 12:
                h = 0
        if 0 <= h < 24 and 0 <= mm < 60:
            return h, mm
    return None


def _month(tok: str) -> int | None:
    for i, name in enumerate(MONTHS):
        if tok in (name, name[:3]):
            return i + 1
    return None


def _mkdate(y: str, m: int | None, d: str) -> date | None:
    if m is None:
        return None
    try:
        return date(int(y), m, int(d))
    except ValueError:
        return None


def parse_date(s: str) -> date | None:
    m = re.search(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b", s)
    if m:
        return _mkdate(m.group(1), int(m.group(2)), m.group(3))
    low = s.lower()
    m = re.search(rf"\b({_MON_ALT})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})", low)
    if m:
        return _mkdate(m.group(3), _month(m.group(1)), m.group(2))
    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MON_ALT})\.?,?\s+(\d{{4}})", low)
    if m:
        return _mkdate(m.group(3), _month(m.group(2)), m.group(1))
    return None


def _strip_date(s: str) -> str:
    s = re.sub(r"\b\d{4}-\d{1,2}-\d{1,2}\b", " ", s)
    return re.sub(r"\b\d{1,2}(st|nd|rd|th)?,?\s+\d{4}\b", " ", s)


def parse_weekday(s: str) -> int | None:
    low = s.lower()
    hits = {i for i, d in enumerate(WEEKDAYS) if re.search(rf"\b({d}|{d[:3]})s?\b", low)}
    return hits.pop() if len(hits) == 1 else None


def as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value if str(v).strip()]
    s = str(value).strip()
    if norm_str(s) in ("", "none", "nobody", "no one", "empty", "n a"):
        return []
    return [p.strip() for p in re.split(r",|;|\n|\band\b|&", s) if p.strip()]


def values_match(vtype: str, answer: Any, gold: Any) -> bool:
    if gold is None:
        return answer is None or as_list(answer) == []
    if vtype.startswith("set_"):
        return sorted({norm_str(x) for x in as_list(answer)}) == sorted({norm_str(x) for x in gold})
    if isinstance(answer, list):
        answer = answer[0] if len(answer) == 1 else ", ".join(map(str, answer))
    if answer is None:
        return False
    a = str(answer)
    g = str(gold)
    if vtype in ("diet", "commute"):
        return _enum(vtype, a) == g
    if vtype == "weekly":
        gd, gt = g.split()
        return parse_weekday(a) == WEEKDAYS.index(gd.lower()) and parse_time(a) == parse_time(gt)
    if vtype == "datetime":
        return parse_date(a) == parse_date(g) and parse_time(_strip_date(a)) == parse_time(g[11:])
    if vtype == "date":
        return parse_date(a) == parse_date(g)
    if vtype == "timerange":
        times = [(int(h), int(m)) for h, m in re.findall(r"(\d{1,2}):(\d{2})", a)]
        return times == [(int(h), int(m)) for h, m in re.findall(r"(\d{1,2}):(\d{2})", g)]
    if vtype == "tz":
        return norm_str(g) in norm_str(a).split()
    return _phrase_match(a, g)


def contract_valid(vtype: str, status: str, answer: Any, gold_status: str, gold_value: Any) -> bool:
    """ContractValid = status matches gold status and, for answers, the value matches."""
    if status != gold_status:
        return False
    if gold_status != "answer":
        return True
    return values_match(vtype, answer, gold_value)

"""Fake answers to the block W requests (`holdmem.extract`): rewrites and extractions."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Sequence
from typing import Any

from memshift.holdmem.extract import (
    CALENDAR_KINDS,
    FIXED,
    LABELS,
    LIST,
    RECORD_KIND,
    RETRY_MARK,
    kind_fields,
)
from memshift.llm import LLMResult
from tests.fakes import RecordingProgramLLM

_LINE = re.compile(r"^- (.+?)(?: \(fixed wording\))?: <value>(.*?)</value>$", re.S | re.M)
_KIND = re.compile(r"below \((.*?)\) as one to three")
_TEXT = re.compile(r"<text>\n(.*)\n</text>", re.S)
_FIELD_OF = {label: name for name, label in LABELS.items()}
_SOURCE_OF = {kind: source for source, kind in RECORD_KIND.items()}


def parse_values(prompt: str) -> tuple[str, list[tuple[str, str]]]:
    """(source, [(field, shown text)]) of a rewrite prompt."""
    m = _KIND.search(prompt)
    assert m is not None
    values = prompt.split("\nValues:\n", 1)[1].split(f"\n\n{RETRY_MARK}", 1)[0]
    values = values.rsplit("\n\nReply with JSON", 1)[0]
    return _SOURCE_OF[m.group(1)], [
        (_FIELD_OF[label], text) for label, text in _LINE.findall(values)
    ]


def typed_from(source: str, pairs: Sequence[tuple[str, str]]) -> dict[str, Any]:
    """The typed fields a rewrite prompt's values stand for (fixed words and durations undone)."""
    out: dict[str, Any] = {}
    for name, text in pairs:
        if name in FIXED:
            out[name] = next(v for v, words in FIXED[name].items() if words == text)
        elif name == "duration_min":
            out[name] = int(text.split()[0])
        elif name == "attendees":
            out.setdefault(name, []).append(text)
        else:
            out[name] = text
    for name, typ in kind_fields(source, out.get("kind")):
        if typ == LIST:
            out.setdefault(name, [])
    return out


class ExtractLLM(RecordingProgramLLM):
    """A recording writer (`tests.fakes`) that also answers the block W requests. A rewrite joins
    the listed values into one line ("; "), which passes the check, and remembers the typed fields
    behind it; an extraction returns those fields. `drop`: fields left out of first-attempt
    rewrites (on every attempt with `drop_always`); `fields_lines`: first attempts are "field:
    value" lines; `corrupt`: (source, field) -> the value an extraction gives instead; `garble`:
    sources whose extraction reply is not JSON."""

    def __init__(
        self,
        programs: dict[str, str] | None = None,
        drop: Sequence[str] = (),
        drop_always: bool = False,
        fields_lines: bool = False,
        corrupt: dict[tuple[str, str], Any] | None = None,
        garble: Sequence[str] = (),
    ):
        super().__init__(programs)
        self.drop = set(drop)
        self.drop_always = drop_always
        self.fields_lines = fields_lines
        self.corrupt = corrupt or {}
        self.garble = set(garble)
        self.table: dict[str, tuple[str, dict[str, Any]]] = {}
        self._table_lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        if purpose not in ("rewrite", "extract"):
            return super().complete(messages, max_tokens, schema, purpose)
        key, body = self.request_key(messages, max_tokens, schema)
        with self._rec_lock:
            self.requests.append((key, purpose, body))
        prompt = messages[-1]["content"]
        text = self._rewrite(prompt) if purpose == "rewrite" else self._extract(prompt)
        return LLMResult(text, 100, 20, 0.0, False, "stop", key)

    def _rewrite(self, prompt: str) -> str:
        source, pairs = parse_values(prompt)
        retry = RETRY_MARK in prompt
        shown = [(n, t) for n, t in pairs if retry and not self.drop_always or n not in self.drop]
        if self.fields_lines and not retry:
            prose = "\n".join(f"{LABELS[n]}: {t}" for n, t in shown)
        else:
            prose = "; ".join(t for _, t in shown) + "."
        with self._table_lock:
            self.table[prose] = (source, typed_from(source, shown))
        return json.dumps({"text": prose})

    def _extract(self, prompt: str) -> str:
        m = _TEXT.search(prompt)
        assert m is not None
        with self._table_lock:
            source, typed = self.table[m.group(1)]
        if source in self.garble:
            return "no JSON here"
        assert source != "calendar" or typed["kind"] in CALENDAR_KINDS
        # calendar: the branch of the extracted kind, `kind` and that kind's fields
        fields = kind_fields(source, typed.get("kind"))
        got = {n: self.corrupt.get((source, n), typed.get(n)) for n, _ in fields}
        return json.dumps(got)

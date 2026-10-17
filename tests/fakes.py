"""Deterministic stand-in for the serving engine, for unit tests without a GPU."""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from memshift.gen.llm_render import RETRY_MARK
from memshift.llm import LLMClient, LLMResult, default_extra_body, is_claude
from memshift.textutil import sha256_json

_REC = re.compile(r"<record id=(\S+) rev=(\d+)>\n(.*?)\n</record>", re.S)
_ORIGINAL = re.compile(r"<original>\n(.*?)\n</original>", re.S)
_ANCHOR = re.compile(r"<anchor>(.*?)</anchor>", re.S)
_MEDIUM = re.compile(r"medium \((\w+)\)")
_PREFIX = {"email": "Hi,\n\n", "chat": "Hey! ", "note": "Note to self: "}


SET_PROGRAM = "def derive(records, as_of):\n    return []\n"
SCALAR_PROGRAM = (
    "def derive(records, as_of):\n    return records[0]['source'] if records else None\n"
)
# the retraction-aware variants (request text with `prior_reads`): they read nothing but the source
RA_SET_PROGRAM = "def derive(records, as_of, prior_reads):\n    return {'value': [], 'reads': []}\n"
RA_SCALAR_PROGRAM = (
    "def derive(records, as_of, prior_reads):\n"
    "    return {'value': records[0]['source'] if records else None, 'reads': []}\n"
)


class FakeLLM:
    """Writer requests: cite the first record, claim value 'X'. Program requests: a valid program
    (an empty list for set items, else the first record's source). Reader requests: abstain.
    Render requests: prefix the original with a few neutral words (anchors, digits and names
    survive); `render_mode` changes that: anchors_only (a worst-case renderer that keeps nothing but
    the anchors), or a fault: drop_anchor (removes the first anchor), add_number, add_date,
    add_name, add_city, bad_json, cut_off. With `fix_on_retry` a fault happens on first attempts
    only (a retry prompt carries RETRY_MARK)."""

    def __init__(
        self, model: str = "fake/writer-a", render_mode: str = "ok", fix_on_retry: bool = False
    ):
        self.model = model
        self.render_mode = render_mode
        self.fix_on_retry = fix_on_retry
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}

    def _render(self, prompt: str) -> tuple[str, str]:
        original = _ORIGINAL.search(prompt).group(1)  # type: ignore[union-attr]
        anchors = _ANCHOR.findall(prompt)
        medium = _MEDIUM.search(prompt).group(1)  # type: ignore[union-attr]
        body = _PREFIX[medium] + original
        mode = "ok" if self.fix_on_retry and RETRY_MARK in prompt else self.render_mode
        if mode == "anchors_only":
            body = " ".join(anchors) or "Noted."
        elif mode == "drop_anchor" and anchors:
            body = body.replace(anchors[0], "something")
        elif mode == "add_number":
            body += " Call 4417."
        elif mode == "add_date":
            body += " See you on Saturday, July 19."
        elif mode == "add_name":
            body += " Say hi to Castellano."
        elif mode == "add_city":
            body += " We flew over Seattle."
        if mode == "bad_json":
            return "not json", "stop"
        if mode == "cut_off":
            return json.dumps({"body": body}), "length"
        return json.dumps({"body": body}), "stop"

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        self.usage["calls"] += 1
        prompt = messages[-1]["content"]
        if purpose == "render":
            text, finish = self._render(prompt)
            return LLMResult(text, 100, 20, 0.0, False, finish, "k")
        if purpose == "write":
            recs = _REC.findall(prompt)
            cits = []
            if recs:
                oid, rev, text = recs[0]
                quote = " ".join(text.split()[:5])
                cits = [{"object_id": oid, "revision": int(rev), "quote": quote}]
            body: dict[str, Any] = {
                "value": "X",
                "statement": f"The value is X ({self.model}).",
                "citations": cits,
            }
        elif purpose == "program":
            if "prior_reads" in prompt:
                is_set = "[] if there are none" in prompt
                body = {"code": RA_SET_PROGRAM if is_set else RA_SCALAR_PROGRAM}
            else:
                body = {
                    "code": SET_PROGRAM if "Python list of strings" in prompt else SCALAR_PROGRAM
                }
        else:
            body = {"status": "abstain", "value": None}
        return LLMResult(json.dumps(body), 100, 20, 0.0, False, "stop", "k")

    def map(self, requests: Sequence[dict[str, Any]]) -> list[LLMResult]:
        return [self.complete(**r) for r in requests]


class RecordingLLM(FakeLLM):
    """FakeLLM that records every request under the response-cache key a real `LLMClient` with the
    same model and extra body would give it (`LLMClient.request_body`, then a SHA256 of the body),
    so that tests can assert which requests two cells share (cache hits) and which are new."""

    def __init__(
        self,
        model: str = "fake/writer-a",
        extra_body: dict[str, Any] | None = None,
        **kw: Any,
    ):
        super().__init__(model, **kw)
        self.extra_body = default_extra_body(model) if extra_body is None else extra_body
        self.requests: list[tuple[str, str, dict[str, Any]]] = []  # (key, purpose, body)
        self._rec_lock = threading.Lock()

    def request_key(
        self, messages: Sequence[dict[str, str]], max_tokens: int, schema: dict[str, Any] | None
    ) -> tuple[str, dict[str, Any]]:
        # a client without HTTP or cache: `request_body` reads only the model, extras and API, so
        # a Claude model gets the Messages API body it would send
        shape = LLMClient.__new__(LLMClient)
        shape.model, shape.extra_body = self.model, self.extra_body
        shape.api = "anthropic" if is_claude(self.model) else "openai"
        body = shape.request_body(messages, max_tokens, schema)
        return sha256_json(body), body

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        key, body = self.request_key(messages, max_tokens, schema)
        with self._rec_lock:
            self.requests.append((key, purpose, body))
        return replace(super().complete(messages, max_tokens, schema, purpose), key=key)

    def keys(self, purpose: str | None = None) -> set[str]:
        return {k for k, p, _ in self.requests if purpose is None or p == purpose}


class RecordingProgramLLM(RecordingLLM):
    """RecordingLLM whose program answers are chosen by a marker in the request text: the first
    entry of `programs` whose marker occurs in the prompt, else `default` (a program, or a function
    of the prompt), else the FakeLLM program."""

    def __init__(
        self,
        programs: dict[str, str] | None = None,
        default: str | Callable[[str], str | None] | None = None,
    ):
        super().__init__()
        self.programs = programs or {}
        self.default = default

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        if purpose == "program":
            prompt = messages[-1]["content"]
            src = next((c for marker, c in self.programs.items() if marker in prompt), None)
            if src is None:
                src = self.default(prompt) if callable(self.default) else self.default
            if src is not None:
                key, body = self.request_key(messages, max_tokens, schema)
                with self._rec_lock:
                    self.requests.append((key, purpose, body))
                return LLMResult(json.dumps({"code": src}), 100, 20, 0.0, False, "stop", key)
        return super().complete(messages, max_tokens, schema, purpose)

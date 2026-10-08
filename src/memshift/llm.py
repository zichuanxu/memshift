"""OpenAI-compatible chat client with a persistent response cache and per-call usage log.

`base_url` may list several replicas of the same model separated by "|"; requests are spread
round-robin. The cache key is the SHA256 of the full request (model, messages, sampling, schema;
not the replica), so identical requests from different cells share one response: this is how the
four HoldMem cells are instantiated from the same writer outputs (plan §5.1) and how reruns are
replayable. HTTP errors are retried at most three times and then raised; a failed call never
becomes a scored answer.
"""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import httpx

from memshift.textutil import sha256_json

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool
    finish_reason: str
    key: str

    def json(self) -> Any:
        return parse_json(self.text)


def parse_json(text: str) -> Any:
    text = _THINK_RE.sub("", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def default_extra_body(model: str) -> dict[str, Any]:
    """Request extras that turn reasoning off (or to its minimum) for each served family: the
    Qwen-lineage (incl. MiMo) and Gemma 4 templates read `enable_thinking` (both default to
    thinking off in Gemma 4); gpt-oss always reasons and takes an effort level; Llama 3.1 has no
    reasoning mode; Mistral models are served in Mistral's own tokenizer mode, which has no chat
    template to pass kwargs to."""
    if model.startswith(("mistralai/", "meta-llama/")):
        return {}
    if model.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low"}
    return {"chat_template_kwargs": {"enable_thinking": False}}


def think_extra_body(model: str) -> dict[str, Any]:
    """`default_extra_body` with thinking switched on, for the families whose chat template has an
    `enable_thinking` switch (the Qwen lineage, Gemma 4); the body, and so the response-cache key,
    differs from every non-thinking request."""
    body = default_extra_body(model)
    if body.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError(f"{model} has no enable_thinking switch")
    return {**body, "chat_template_kwargs": {"enable_thinking": True}}


class ResponseCache:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False, timeout=60)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, model TEXT, response TEXT)"
        )
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def put(self, key: str, model: str, response: dict[str, Any]) -> None:
        with self._lock:
            self._db.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, model, json.dumps(response))
            )
            self._db.commit()


class LLMClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        cache_path: Path,
        max_workers: int = 32,
        extra_body: dict[str, Any] | None = None,
        timeout: float = 600.0,
        log_path: Path | None = None,
    ):
        self.base_urls = [u.rstrip("/") for u in base_url.split("|") if u]
        self._rr = 0
        self.model = model
        self.cache = ResponseCache(cache_path)
        self.max_workers = max_workers
        self.extra_body = default_extra_body(model) if extra_body is None else extra_body
        self.log_path = log_path
        self._log_lock = threading.Lock()
        self._http = httpx.Client(timeout=timeout)
        self.usage = {"calls": 0, "cached": 0, "prompt_tokens": 0, "completion_tokens": 0}
        self._inflight: dict[str, Future[LLMResult]] = {}

    def request_body(
        self, messages: Sequence[dict[str, str]], max_tokens: int, schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "temperature": 0.0,
            "max_tokens": max_tokens,
            **self.extra_body,
        }
        if schema is not None:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            }
        return body

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        body = self.request_body(messages, max_tokens, schema)
        key = sha256_json(body)
        hit = self.cache.get(key)
        if hit is not None:
            res = LLMResult(**{**hit, "cached": True, "key": key})
        else:
            res = self._single_flight(body, key)
        self._account(res, purpose)
        return res

    def _single_flight(self, body: dict[str, Any], key: str) -> LLMResult:
        """Identical concurrent requests wait for the one already in flight."""
        with self._log_lock:
            fut = self._inflight.get(key)
            owner = fut is None
            if fut is None:
                fut = Future()
                self._inflight[key] = fut
        if not owner:
            return replace(fut.result(), cached=True)
        try:
            res = self._post(body, key)
            self.cache.put(key, self.model, {k: v for k, v in asdict(res).items() if k != "key"})
            fut.set_result(res)
            return res
        except Exception as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._log_lock:
                self._inflight.pop(key, None)

    def _post(self, body: dict[str, Any], key: str) -> LLMResult:
        last: Exception | None = None
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = self._http.post(f"{self._next_url()}/chat/completions", json=body)
                if r.status_code >= 500:
                    raise httpx.HTTPStatusError(str(r.status_code), request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return LLMResult(
                    text=_THINK_RE.sub("", choice["message"].get("content") or "").strip(),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    latency_s=time.perf_counter() - t0,
                    cached=False,
                    finish_reason=str(choice.get("finish_reason")),
                    key=key,
                )
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                last = exc
                if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                    raise
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"LLM request failed after retries: {last}")

    def _next_url(self) -> str:
        with self._log_lock:
            self._rr = (self._rr + 1) % len(self.base_urls)
            return self.base_urls[self._rr]

    def _account(self, res: LLMResult, purpose: str) -> None:
        with self._log_lock:
            self.usage["calls"] += 1
            self.usage["cached"] += int(res.cached)
            self.usage["prompt_tokens"] += res.prompt_tokens
            self.usage["completion_tokens"] += res.completion_tokens
            if self.log_path is not None:
                rec = {
                    "key": res.key,
                    "model": self.model,
                    "purpose": purpose,
                    "cached": res.cached,
                    "prompt_tokens": res.prompt_tokens,
                    "completion_tokens": res.completion_tokens,
                    "latency_s": round(res.latency_s, 3),
                    "finish_reason": res.finish_reason,
                }
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")

    def map(self, requests: Sequence[dict[str, Any]]) -> list[LLMResult]:
        """Run many `complete(**req)` calls concurrently, preserving order."""
        if not requests:
            return []
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda req: self.complete(**req), requests))

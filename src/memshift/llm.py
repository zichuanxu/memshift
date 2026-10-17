"""OpenAI-compatible chat client with a persistent response cache and per-call usage log.

`base_url` may list several replicas of the same model separated by "|"; requests are spread
round-robin. The cache key is the SHA256 of the full request (model, messages, sampling, schema;
not the replica), so identical requests from different cells share one response: this is how the
four HoldMem cells are instantiated from the same writer outputs (plan §5.1) and how reruns are
replayable. HTTP errors are retried at most three times and then raised; a failed call never
becomes a scored answer.

Model ids starting with `claude-` go to Anthropic's Messages API instead (key from CLAUDE_API_KEY,
else ANTHROPIC_API_KEY). The request is translated (system messages become `system`, a schema
becomes `output_config.format`) and its key is the SHA256 of that Anthropic body. The result is the
same `LLMResult`: the text blocks without thinking blocks, all input tokens, the output tokens
(thinking included), and `stop_reason` as `finish_reason` with `max_tokens` mapped to "length".
Rate limits (429), overload (529), 5xx and connection errors are retried with backoff, up to
`CLAUDE_ATTEMPTS` attempts; each client has at most `api_concurrency` requests in flight
(MEMSHIFT_CLAUDE_CONCURRENCY, default 8). The cache row of an API response also keeps the response
id, served model, `stop_reason`, raw `usage` and time.

Top-level request keys starting with `memshift_` (e.g. `memshift_sample`) enter the cache key but
are removed before the request is sent, on both paths. On the Anthropic path `seed` (which E1's
second ingestion adds to every writer's extras) is handled the same way, and a `temperature` or
`top_p` raises ValueError before the cache is read: the 5.5 models reject any sampling parameter
(prereg/decisions.md, 2026-10-16), so a request that carries one is a mistake, not a setting.

Batch collection (`MEMSHIFT_ANTHROPIC_COLLECT=<file>`): an Anthropic request that misses the cache
is not sent; its key, model and wire body are appended to the file and the caller gets an empty,
uncached result with finish reason "collected". Such a run is void. `memshift.claude_batch` sends
the collected requests through the Message Batches API and writes each result into the cache as the
online path would, so a rerun without the variable finds every response in the cache.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import re
import sqlite3
import threading
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from memshift.textutil import sha256_json

_THINK_RE = re.compile(r"<think>.*?</think>", re.S)
_LOG = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com"
ANTHROPIC_VERSION = "2023-06-01"
CLAUDE_CONCURRENCY = 8
CLAUDE_ATTEMPTS = 8
_RETRY_STATUS = frozenset({429, 500, 502, 503, 504, 529})
# a monthly spend cap also answers 429, without retry-after, until the month ends: not retried
_SPEND_CAP = "enforced_spend_limit_reached"
# stop reasons that cut the reply off: the output limit, or the context window
_LENGTH_STOPS = frozenset({"max_tokens", "model_context_window_exceeded"})
# JSON-schema keywords that Anthropic's structured outputs reject (as is `minItems` above 1); the
# official SDKs move them into the description, and so does `structured_schema`
_UNSUPPORTED_SCHEMA_KEYS = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "maxItems",
    }
)
# request keys the Messages API does not take but callers may set on any writer client: they stay in
# the cache key and are removed before sending (like `memshift_*`); sampling keys raise instead
_ANTHROPIC_UNSENT = frozenset({"seed"})
_ANTHROPIC_REFUSED = ("temperature", "top_p")
COLLECT_ENV = "MEMSHIFT_ANTHROPIC_COLLECT"
_COLLECT_LOCK = threading.Lock()
_COLLECTED: set[str] = set()
_FIRST_CALL: dict[str, str] = {}
_FIRST_CALL_LOCK = threading.Lock()


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


_CACHED_FIELDS = frozenset(f.name for f in fields(LLMResult)) - {"key"}


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


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def anthropic_key() -> str:
    key = os.environ.get("CLAUDE_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError("set CLAUDE_API_KEY (or ANTHROPIC_API_KEY) to call Anthropic models")
    return key


def anthropic_first_call(model: str) -> str | None:
    """UTC time of the first request this process sent to the API for `model` (None if every
    response so far came from the cache)."""
    with _FIRST_CALL_LOCK:
        return _FIRST_CALL.get(model)


def _note_first_call(model: str) -> None:
    with _FIRST_CALL_LOCK:
        _FIRST_CALL.setdefault(model, datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"))


def structured_schema(schema: Any) -> Any:
    """`schema` in the subset that Anthropic's structured outputs accept: rejected constraints move
    into the description as JSON, so the model still reads them but the grammar does not enforce
    them."""
    if isinstance(schema, list):
        return [structured_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: dict[str, Any] = {}
    moved: dict[str, Any] = {}
    for k, v in schema.items():
        if k in _UNSUPPORTED_SCHEMA_KEYS or (k == "minItems" and isinstance(v, int) and v > 1):
            moved[k] = v
        elif k in ("properties", "$defs", "definitions") and isinstance(v, dict):
            out[k] = {name: structured_schema(sub) for name, sub in v.items()}
        else:
            out[k] = structured_schema(v)
    if moved:
        note = json.dumps(moved, sort_keys=True)
        out["description"] = f"{out['description']} {note}" if "description" in out else note
    return out


def _wire(body: dict[str, Any]) -> dict[str, Any]:
    """The body as sent: `memshift_*` keys only namespace the cache key."""
    return {k: v for k, v in body.items() if not k.startswith("memshift_")}


def _error_body(r: httpx.Response) -> dict[str, Any]:
    """The `error` object of an API error response ({} if the body is not the documented JSON)."""
    try:
        err = r.json().get("error")
    except (json.JSONDecodeError, AttributeError):
        return {}
    return err if isinstance(err, dict) else {}


def anthropic_message_result(
    data: dict[str, Any], key: str, latency: float
) -> tuple[LLMResult, dict[str, Any]]:
    """The `LLMResult` of a Messages API response (online or a batch result's `message`) and the
    API metadata its cache row keeps."""
    usage = data.get("usage") or {}
    stop = str(data.get("stop_reason"))
    text = "".join(
        str(b.get("text", "")) for b in data.get("content") or [] if b.get("type") == "text"
    )
    prompt = sum(
        int(usage.get(k) or 0)
        for k in ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
    )
    meta = {
        "id": data.get("id"),
        "model": data.get("model"),
        "stop_reason": stop,
        "usage": usage,
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    res = LLMResult(
        text=text.strip(),
        prompt_tokens=prompt,
        completion_tokens=int(usage.get("output_tokens") or 0),
        latency_s=latency,
        cached=False,
        finish_reason="length" if stop in _LENGTH_STOPS else stop,
        key=key,
    )
    return res, meta


class AnthropicError(RuntimeError):
    def __init__(self, status: int, kind: str, message: str, request_id: str | None):
        super().__init__(f"Anthropic API {status} {kind}: {message} (request {request_id})")
        self.status = status
        self.kind = kind
        self.request_id = request_id


def default_extra_body(model: str) -> dict[str, Any]:
    """Request extras that turn reasoning off (or to its minimum) for each served family: the
    Qwen-lineage (incl. MiMo) and Gemma 4 templates read `enable_thinking` (both default to
    thinking off in Gemma 4); gpt-oss always reasons and takes an effort level; Llama 3.1 has no
    reasoning mode; Mistral models are served in Mistral's own tokenizer mode, which has no chat
    template to pass kwargs to. Anthropic models get no extras, so the API's defaults apply: on the
    5.5 models that is adaptive thinking, and the API rejects `thinking.type` "enabled" (a token
    budget), "disabled" and any non-default temperature; callers set `thinking` / `output_config`
    explicitly."""
    if is_claude(model) or model.startswith(("mistralai/", "meta-llama/")):
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

    def put(self, key: str, model: str, response: dict[str, Any]) -> dict[str, Any]:
        """Store the reply unless the key already holds one; return the stored reply. The first
        reply stored wins, so every process and client that posted the same request concurrently
        uses the same reply (tests/test_llm_cache_race.py)."""
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO cache VALUES (?,?,?)", (key, model, json.dumps(response))
            )
            self._db.commit()
            row = self._db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
        kept: dict[str, Any] = json.loads(row[0])
        return kept


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
        api_concurrency: int | None = None,
    ):
        self.api = "anthropic" if is_claude(model) else "openai"
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
        slots = api_concurrency or int(
            os.environ.get("MEMSHIFT_CLAUDE_CONCURRENCY") or CLAUDE_CONCURRENCY
        )
        self._slots = threading.BoundedSemaphore(slots)
        self._api_meta: dict[str, dict[str, Any]] = {}

    def request_body(
        self, messages: Sequence[dict[str, str]], max_tokens: int, schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        if getattr(self, "api", "openai") == "anthropic":  # stand-ins without .api use OpenAI
            return self._anthropic_body(messages, max_tokens, schema)
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

    def _anthropic_body(
        self, messages: Sequence[dict[str, str]], max_tokens: int, schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Messages API body: system messages joined into `system`; `extra_body` (e.g.
        `thinking`, `output_config.effort`) passed through; a schema becomes `output_config.format`.
        A `temperature` or `top_p` raises ValueError (module docstring)."""
        system = [m["content"] for m in messages if m["role"] == "system"]
        body: dict[str, Any] = {"model": self.model, "max_tokens": max_tokens}
        if system:
            body["system"] = "\n\n".join(system)
        body["messages"] = [
            {"role": m["role"], "content": m["content"]} for m in messages if m["role"] != "system"
        ]
        body.update(self.extra_body)
        refused = [k for k in _ANTHROPIC_REFUSED if k in body]
        if refused:
            raise ValueError(
                f"{self.model}: the API rejects {' and '.join(refused)} on the Claude 5.5 models "
                "(prereg/decisions.md, 2026-10-16); remove it from the client's extra_body"
            )
        if schema is not None:
            fmt = {"type": "json_schema", "schema": structured_schema(schema)}
            body["output_config"] = {**body.get("output_config", {}), "format": fmt}
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
            row = {k: v for k, v in hit.items() if k in _CACHED_FIELDS}
            res = LLMResult(**{**row, "cached": True, "key": key})
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
            if getattr(self, "api", "openai") == "anthropic" and os.environ.get(COLLECT_ENV):
                res = self._collect(body, key)  # never cached
                fut.set_result(res)
                return res
            res = self._post(body, key)
            row = {k: v for k, v in asdict(res).items() if k != "key"}
            with self._log_lock:
                meta = self._api_meta.pop(key, None)
            if meta is not None:
                row["api"] = meta
            kept = self.cache.put(key, self.model, row)
            if json.dumps(kept, sort_keys=True) != json.dumps(row, sort_keys=True):
                fields_ = {k: v for k, v in kept.items() if k in _CACHED_FIELDS}
                res = LLMResult(**{**fields_, "cached": True, "key": key})
            fut.set_result(res)
            return res
        except Exception as exc:
            fut.set_exception(exc)
            raise
        finally:
            with self._log_lock:
                self._inflight.pop(key, None)

    def _post(self, body: dict[str, Any], key: str) -> LLMResult:
        if getattr(self, "api", "openai") == "anthropic":  # stand-ins without .api use OpenAI
            return self._post_anthropic(body, key)
        last: Exception | None = None
        for attempt in range(4):
            t0 = time.perf_counter()
            try:
                r = self._http.post(f"{self._next_url()}/chat/completions", json=_wire(body))
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

    def _post_anthropic(self, body: dict[str, Any], key: str) -> LLMResult:
        headers = {
            "x-api-key": anthropic_key(),
            "anthropic-version": ANTHROPIC_VERSION,
            "content-type": "application/json",
        }
        wire = {k: v for k, v in _wire(body).items() if k not in _ANTHROPIC_UNSENT}
        last = ""
        for attempt in range(1, CLAUDE_ATTEMPTS + 1):
            wait = min(60.0, 2.0**attempt) * random.uniform(0.5, 1.0)
            t0 = time.perf_counter()
            try:
                with self._slots:
                    _note_first_call(self.model)
                    r = self._http.post(
                        f"{self._next_url()}/v1/messages", json=wire, headers=headers
                    )
            except httpx.TransportError as exc:
                last = type(exc).__name__
                self._retry_later(last, None, attempt, wait)
                continue
            if r.status_code == 200:
                return self._anthropic_result(r.json(), key, time.perf_counter() - t0)
            err = _error_body(r)
            kind = str(err.get("type", "unknown"))
            rid = r.headers.get("request-id")
            spend_cap = (err.get("details") or {}).get("error_code") == _SPEND_CAP
            if r.status_code not in _RETRY_STATUS or spend_cap:
                raise AnthropicError(r.status_code, kind, str(err.get("message", "")), rid)
            with contextlib.suppress(ValueError):
                wait = max(wait, float(r.headers.get("retry-after", "")))
            last = f"HTTP {r.status_code} {kind}"
            self._retry_later(last, rid, attempt, wait)
        raise RuntimeError(f"Anthropic request failed after {CLAUDE_ATTEMPTS} attempts: {last}")

    def _retry_later(self, why: str, request_id: str | None, attempt: int, wait: float) -> None:
        _LOG.warning(
            "%s: %s (request %s), attempt %d/%d, retry in %.0fs",
            *(self.model, why, request_id, attempt, CLAUDE_ATTEMPTS, wait),
        )
        time.sleep(wait)

    def _anthropic_result(self, data: dict[str, Any], key: str, latency: float) -> LLMResult:
        res, meta = anthropic_message_result(data, key, latency)
        with self._log_lock:
            self._api_meta[key] = meta
        return res

    def _collect(self, body: dict[str, Any], key: str) -> LLMResult:
        """Batch collection (module docstring): record the request instead of sending it."""
        path = Path(os.environ[COLLECT_ENV])
        wire = {k: v for k, v in _wire(body).items() if k not in _ANTHROPIC_UNSENT}
        with _COLLECT_LOCK:
            if key not in _COLLECTED:
                _COLLECTED.add(key)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps({"key": key, "model": self.model, "params": wire}) + "\n")
        return LLMResult("", 0, 0, 0.0, False, "collected", key)

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

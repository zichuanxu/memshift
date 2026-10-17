import json
import logging
import threading

import httpx
import pytest

from memshift import llm as llm_mod
from memshift.llm import AnthropicError, LLMClient, structured_schema
from memshift.textutil import sha256_json

KEY = "sk-test-not-a-real-key"
MSGS = [{"role": "system", "content": "S"}, {"role": "user", "content": "hi"}]
PROGRAM_SCHEMA = {
    "type": "object",
    "properties": {"code": {"type": "string"}},
    "required": ["code"],
    "additionalProperties": False,
}


def reply(text='{"ok": 1}', stop="end_turn", **usage):
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "sig"},
            {"type": "redacted_thinking", "data": "opaque"},
            {"type": "text", "text": text},
        ],
        "stop_reason": stop,
        "usage": {"input_tokens": 10, "output_tokens": 7, **usage},
    }


def claude(tmp_path, handler, model="claude-sonnet-5-5", **kw):
    c = LLMClient(llm_mod.ANTHROPIC_URL, model, tmp_path / f"{model}.sqlite", **kw)
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", KEY)
    monkeypatch.delenv("MEMSHIFT_CLAUDE_CONCURRENCY", raising=False)
    monkeypatch.setattr(llm_mod.time, "sleep", lambda s: None)


def test_existing_cache_keys_unchanged(tmp_path):
    qwen = LLMClient("http://unused", "Qwen/Qwen3.5-9B", tmp_path / "q.sqlite")
    assert sha256_json(qwen.request_body(MSGS, 8, None)) == (
        "7e35a2c8fdcb101ed52934bd33e31195efe0c19f0f83e7eb7f5b6d107629b86c"
    )
    assert sha256_json(qwen.request_body(MSGS, 64, PROGRAM_SCHEMA)) == (
        "01962594f02aa4156ad85155ecf734337a9dcf699b0569ad561a2ddab349ab5f"
    )


def test_memshift_keys_enter_the_key_but_are_not_sent(tmp_path):
    sent = []

    def vllm(request):
        sent.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "{}"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    def anthropic(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply())

    for i, (url, model, handler, sampling) in enumerate(
        [
            ("http://vllm", "Qwen/Qwen3.5-9B", vllm, {"temperature": 0.7}),
            (llm_mod.ANTHROPIC_URL, "claude-x", anthropic, {}),  # Claude: no temperature
        ]
    ):
        plain = LLMClient(url, model, tmp_path / f"{i}.sqlite")
        extra = {**plain.extra_body, **sampling, "memshift_sample": 2}
        sampled = LLMClient(url, model, tmp_path / f"{i}.sqlite", extra_body=extra)
        for c in (plain, sampled):
            c._http = httpx.Client(transport=httpx.MockTransport(handler))
        a, b = plain.complete(MSGS, 8), sampled.complete(MSGS, 8)
        assert a.key != b.key and not b.cached
        assert sampled.request_body(MSGS, 8, None)["memshift_sample"] == 2
        assert sampled.complete(MSGS, 8).cached
    assert len(sent) == 4
    assert not any(k.startswith("memshift_") for body in sent for k in body)
    assert sent[1]["temperature"] == 0.7 and sent[2] == sent[3]


def test_anthropic_path_drops_seed_and_refuses_sampling(tmp_path):
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=reply())

    plain = claude(tmp_path, handler)
    seeded = claude(tmp_path, handler, extra_body={"seed": 1})  # E1's second ingestion
    a, b = plain.complete(MSGS, 8), seeded.complete(MSGS, 8)
    assert a.key != b.key  # the seed still names its own cache entry
    assert b.key == sha256_json({**plain.request_body(MSGS, 8, None), "seed": 1})
    assert len(sent) == 2 and sent[0] == sent[1] and "seed" not in sent[1]
    for extra in ({"temperature": 0.0}, {"temperature": 1.0}, {"top_p": 0.9}):
        bad = claude(tmp_path, handler, extra_body=extra)
        with pytest.raises(ValueError, match=next(iter(extra))):
            bad.complete(MSGS, 8)
    assert len(sent) == 2  # refused before the cache or the API
    qwen = LLMClient("http://unused", "Qwen/Qwen3.5-9B", tmp_path / "q.sqlite")
    extra = {**qwen.extra_body, "seed": 1, "top_p": 0.9}
    seeded_qwen = LLMClient("http://unused", qwen.model, tmp_path / "q.sqlite", extra_body=extra)
    body = seeded_qwen.request_body(MSGS, 8, None)  # served models: unchanged
    assert body["seed"] == 1 and body["top_p"] == 0.9 and body["temperature"] == 0.0


def test_anthropic_request_shape(tmp_path):
    c = LLMClient(llm_mod.ANTHROPIC_URL, "claude-sonnet-5-5", tmp_path / "c.sqlite")
    assert c.extra_body == {}  # the API's defaults: no thinking field, no temperature
    body = c.request_body(MSGS + [{"role": "system", "content": "T"}], 64, None)
    assert body == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 64,
        "system": "S\n\nT",
        "messages": [{"role": "user", "content": "hi"}],
    }
    think = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    t = LLMClient(llm_mod.ANTHROPIC_URL, "claude-x", tmp_path / "t.sqlite", extra_body=think)
    body = t.request_body(MSGS, 17_000, PROGRAM_SCHEMA)
    assert body["thinking"] == {"type": "adaptive"}
    assert body["output_config"] == {
        "effort": "high",
        "format": {"type": "json_schema", "schema": PROGRAM_SCHEMA},
    }
    assert think["output_config"] == {"effort": "high"}  # the client's extras are not mutated
    assert "temperature" not in body and "response_format" not in body


def test_structured_schema_moves_rejected_constraints():
    schema = {
        "type": "object",
        "properties": {
            "maxItems": {"type": "string", "maxLength": 5},
            "citations": {"type": "array", "maxItems": 6, "minItems": 1, "items": {}},
            "n": {"type": "integer", "minimum": 0, "description": "count"},
        },
        "required": ["citations"],
        "additionalProperties": False,
    }
    out = structured_schema(schema)
    assert out["properties"]["maxItems"] == {"type": "string", "description": '{"maxLength": 5}'}
    assert out["properties"]["citations"] == {
        "type": "array",
        "minItems": 1,
        "items": {},
        "description": '{"maxItems": 6}',
    }
    assert out["properties"]["n"]["description"] == 'count {"minimum": 0}'
    assert structured_schema(PROGRAM_SCHEMA) == PROGRAM_SCHEMA


def test_anthropic_response_and_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(llm_mod, "_FIRST_CALL", {})
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(
            200,
            json=reply(
                '  {"code": "x"} ',
                stop="max_tokens",
                cache_read_input_tokens=5,
                cache_creation_input_tokens=0,
            ),
        )

    log = tmp_path / "calls.jsonl"
    c = claude(tmp_path, handler, log_path=log)
    res = c.complete(MSGS, 64, PROGRAM_SCHEMA, purpose="program")
    assert (res.text, res.prompt_tokens, res.completion_tokens) == ('{"code": "x"}', 15, 7)
    assert res.finish_reason == "length" and not res.cached
    assert res.json() == {"code": "x"}
    req = seen[0]
    assert req.url == "https://api.anthropic.com/v1/messages"
    assert req.headers["x-api-key"] == KEY
    assert req.headers["anthropic-version"] == "2023-06-01"
    row = c.cache.get(res.key)
    assert row["api"]["stop_reason"] == "max_tokens" and row["api"]["usage"]["output_tokens"] == 7
    assert llm_mod.anthropic_first_call("claude-sonnet-5-5") is not None
    monkeypatch.setattr(llm_mod, "_FIRST_CALL", {})
    again = c.complete(MSGS, 64, PROGRAM_SCHEMA)
    assert again.cached and again.text == res.text and len(seen) == 1
    assert llm_mod.anthropic_first_call("claude-sonnet-5-5") is None  # a cache hit sends nothing
    assert KEY not in log.read_text()


def test_end_turn_passes_through(tmp_path):
    c = claude(tmp_path, lambda r: httpx.Response(200, json=reply()))
    assert c.complete(MSGS, 8).finish_reason == "end_turn"


def test_retries_rate_limit_overload_and_connection_errors(tmp_path, monkeypatch, caplog):
    script = iter(["connect", 529, 429, 500, 200])
    waits = []
    monkeypatch.setattr(llm_mod.time, "sleep", waits.append)

    def handler(request):
        step = next(script)
        if step == "connect":
            raise httpx.ConnectError("refused", request=request)
        if step == 200:
            return httpx.Response(200, json=reply())
        kind = {529: "overloaded_error", 429: "rate_limit_error", 500: "api_error"}[step]
        headers = {"retry-after": "30", "request-id": "req_1"} if step == 429 else {}
        body = {"type": "error", "error": {"type": kind, "message": "busy"}}
        return httpx.Response(step, json=body, headers=headers)

    c = claude(tmp_path, handler)
    with caplog.at_level(logging.WARNING, logger="memshift.llm"):
        res = c.complete(MSGS, 8)
    assert res.text == '{"ok": 1}'
    assert len(waits) == 4 and waits[2] >= 30  # retry-after is honoured
    assert "overloaded_error" in caplog.text and "req_1" in caplog.text
    assert KEY not in caplog.text


def test_client_errors_and_spend_cap_are_not_retried(tmp_path):
    calls = []

    def bad_request(request):
        calls.append(1)
        body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "thinking.type.enabled"},
            "request_id": "req_2",
        }
        return httpx.Response(400, json=body, headers={"request-id": "req_2"})

    with pytest.raises(AnthropicError) as err:
        claude(tmp_path, bad_request).complete(MSGS, 8)
    assert err.value.status == 400 and "thinking.type.enabled" in str(err.value)
    assert KEY not in str(err.value) and len(calls) == 1

    def spend_cap(request):
        calls.append(1)
        error = {
            "type": "rate_limit_error",
            "message": "usage limits",
            "details": {"error_code": "enforced_spend_limit_reached"},
        }
        return httpx.Response(429, json={"type": "error", "error": error})

    with pytest.raises(AnthropicError):
        claude(tmp_path, spend_cap, model="claude-y").complete(MSGS, 8)
    assert len(calls) == 2


def test_retries_are_bounded(tmp_path):
    calls = []

    def overloaded(request):
        calls.append(1)
        return httpx.Response(529, json={"error": {"type": "overloaded_error"}})

    with pytest.raises(RuntimeError, match="after 8 attempts"):
        claude(tmp_path, overloaded).complete(MSGS, 8)
    assert len(calls) == llm_mod.CLAUDE_ATTEMPTS


def test_missing_key(tmp_path, monkeypatch):
    monkeypatch.delenv("CLAUDE_API_KEY")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="CLAUDE_API_KEY"):
        claude(tmp_path, lambda r: httpx.Response(200, json=reply())).complete(MSGS, 8)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    assert claude(tmp_path, lambda r: httpx.Response(200, json=reply())).complete(MSGS, 8)


@pytest.mark.parametrize("limit, env", [(3, None), (None, "2"), (None, None)])
def test_concurrency_limit(tmp_path, monkeypatch, limit, env):
    if env:
        monkeypatch.setenv("MEMSHIFT_CLAUDE_CONCURRENCY", env)
    expected = limit or int(env or llm_mod.CLAUDE_CONCURRENCY)
    lock = threading.Lock()
    state = {"now": 0, "peak": 0}

    def handler(request):
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        threading.Event().wait(0.1)  # time.sleep is stubbed out by the fixture
        with lock:
            state["now"] -= 1
        return httpx.Response(200, json=reply())

    c = claude(tmp_path, handler, max_workers=24, api_concurrency=limit)
    reqs = [{"messages": [{"role": "user", "content": str(i)}], "max_tokens": 8} for i in range(24)]
    assert len(c.map(reqs)) == 24
    assert state["peak"] == expected


def test_common_client_and_cache_path(tmp_path, monkeypatch):
    from memshift.experiments.common import cache_path, client

    monkeypatch.delenv("MEMSHIFT_ENDPOINTS", raising=False)
    monkeypatch.delenv("MEMSHIFT_CACHE_TAGS", raising=False)
    monkeypatch.setattr("memshift.experiments.common.REPO", tmp_path)
    assert cache_path("claude-sonnet-5-5").name == "anthropic_claude-sonnet-5-5.sqlite"
    assert cache_path("org/Model-1").name == "org_Model-1.sqlite"
    c = client("claude-sonnet-5-5", None)
    assert c.api == "anthropic" and c.base_urls == ["https://api.anthropic.com"]
    with pytest.raises(SystemExit):
        client("org/Model-1", None)


def test_manifest_records_the_api(tmp_path, monkeypatch):
    from memshift.runinfo import write_manifest

    monkeypatch.setitem(llm_mod._FIRST_CALL, "claude-opus-5-5", "2026-10-16T00:00:00Z")
    run = tmp_path / "run_z"
    run.mkdir()
    m = write_manifest(run, "test", [], {"writer": "claude-opus-5-5", "reader": "org/Model-1"}, {})
    assert m["models"]["writer"] == {
        "api": "anthropic",
        "model": "claude-opus-5-5",
        "anthropic_version": "2023-06-01",
        "first_call_utc": "2026-10-16T00:00:00Z",
    }
    assert set(m["models"]["reader"]) == {"model", "revision", "image_id"}

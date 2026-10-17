"""Batch collection in `memshift.llm` and the Message Batches round trip in
`memshift.claude_batch` (prereg v4b block Y): a collecting run sends nothing and caches nothing;
batch results land in the cache with the online path's row, so a rerun is all cache hits."""

import json
from pathlib import Path

import httpx
import pytest

from memshift import claude_batch
from memshift.experiments.common import cache_path
from memshift.llm import COLLECT_ENV, LLMClient, ResponseCache

SONNET = "claude-sonnet-5-5"
BODY = {"thinking": {"type": "between_tools"}}
MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q?"}]


def message(text: str = '{"status": "answer", "value": "x"}') -> dict:
    return {
        "id": "msg_1",
        "model": SONNET,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def make_client(tmp: Path, handler) -> LLMClient:
    c = LLMClient(
        "https://api.anthropic.com",
        SONNET,
        cache_path=tmp / cache_path(SONNET).name,
        extra_body=dict(BODY),
    )
    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return c


def refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request may be sent: {request.url}")


@pytest.fixture(autouse=True)
def key(monkeypatch):
    monkeypatch.setenv("CLAUDE_API_KEY", "test-key")
    monkeypatch.setattr("memshift.llm._COLLECTED", set())


def test_a_collecting_client_sends_and_caches_nothing(tmp_path, monkeypatch):
    out = tmp_path / "collect.jsonl"
    monkeypatch.setenv(COLLECT_ENV, str(out))
    c = make_client(tmp_path, refuse)
    a = c.complete(MESSAGES, max_tokens=256, purpose="serve")
    b = c.complete(MESSAGES, max_tokens=256, purpose="serve")
    assert (a.text, a.finish_reason, a.cached) == ("", "collected", False)
    assert b.key == a.key
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["key"] == a.key and rows[0]["model"] == SONNET
    assert rows[0]["params"] == c.request_body(MESSAGES, 256, None)
    assert c.cache.get(a.key) is None


def batch_api(results: list[dict]):
    created = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["x-api-key"] == "test-key"
        if request.method == "POST" and request.url.path == "/v1/messages/batches":
            created.append(json.loads(request.content))
            return httpx.Response(200, json={"id": "msgbatch_1", "created_at": "t"})
        if request.url.path == "/v1/messages/batches/msgbatch_1":
            return httpx.Response(
                200,
                json={
                    "processing_status": "ended",
                    "request_counts": {"succeeded": len(results)},
                    "results_url": "https://api.anthropic.com/results/msgbatch_1",
                },
            )
        if request.url.path == "/results/msgbatch_1":
            return httpx.Response(200, text="\n".join(json.dumps(r) for r in results))
        raise AssertionError(request.url)

    return handler, created


def test_batch_results_are_cached_like_online_responses(tmp_path, monkeypatch):
    out = tmp_path / "collect.jsonl"
    monkeypatch.setenv(COLLECT_ENV, str(out))
    q2 = [MESSAGES[0], {"role": "user", "content": "other?"}]
    k1 = make_client(tmp_path, refuse).complete(MESSAGES, max_tokens=256).key
    k2 = make_client(tmp_path, refuse).complete(q2, max_tokens=256).key
    monkeypatch.delenv(COLLECT_ENV)
    results = [
        {"custom_id": k1, "result": {"type": "succeeded", "message": message()}},
        {"custom_id": k2, "result": {"type": "errored", "error": {"type": "overloaded"}}},
    ]
    handler, created = batch_api(results)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    state = tmp_path / "state.json"
    claude_batch.submit([out, out], state, http=http, cache_dir=tmp_path)
    (req,) = created
    assert sorted(r["custom_id"] for r in req["requests"]) == sorted([k1, k2])
    claude_batch.collect(state, http=http, cache_dir=tmp_path)
    b = json.loads(state.read_text())["batches"][0]
    assert b["outcome"] == {"succeeded": 1, "errored": 1}
    assert b["usd"] == pytest.approx((100 * 1.0 + 10 * 5.0) / 1e6)
    # the rerun: the succeeded request is a cache hit, the errored one goes online
    online = []

    def api(request: httpx.Request) -> httpx.Response:
        online.append(json.loads(request.content))
        return httpx.Response(200, json=message('{"status": "abstain", "value": null}'))

    c = make_client(tmp_path, api)
    hit = c.complete(MESSAGES, max_tokens=256)
    assert hit.cached and hit.text == '{"status": "answer", "value": "x"}'
    miss = c.complete(q2, max_tokens=256)
    assert not miss.cached and len(online) == 1
    cache = ResponseCache(tmp_path / cache_path(SONNET).name)
    batch_row, online_row = cache.get(k1), cache.get(k2)
    assert batch_row["api"].pop("batch_id") == "msgbatch_1"
    for row in (batch_row, online_row):
        row["api"].pop("utc")
        row.pop("latency_s")
        row.pop("text")
    assert batch_row == online_row
    # nothing is resubmitted once cached
    created.clear()
    claude_batch.submit([out], tmp_path / "state2.json", http=http, cache_dir=tmp_path)
    assert created == []


def test_chunks_respect_the_limits():
    reqs = [{"custom_id": str(i), "params": {"x": "y" * 10}} for i in range(7)]
    assert [len(c) for c in claude_batch.chunks(reqs, max_requests=3)] == [3, 3, 1]
    size = len(json.dumps(reqs[0]))
    assert [len(c) for c in claude_batch.chunks(reqs, max_bytes=2 * size)] == [2, 2, 2, 1]

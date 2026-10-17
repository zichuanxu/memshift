"""The exact bodies the E1 Claude clients send to Anthropic (prereg v4 blocks H-a and G-b as amended
on 2026-10-16, prereg/decisions.md), with a mocked HTTP transport: the clients are built as E1
builds them (`common.client` with `defwriter_settings` / `program_extra_body`), and the requests
go through the HoldMem writer and program paths (`writer.derive`, `program.synthesise`)."""

import json
from typing import Any

import httpx
import pytest

from memshift.experiments.common import client, world_params
from memshift.experiments.e1 import defwriter_settings, program_extra_body
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem import program, writer
from memshift.holdmem.program import PROGRAM_SCHEMA, program_request, synthesise
from memshift.holdmem.writer import WRITER_SCHEMA, derive, records_for, writer_request
from memshift.llm import LLMClient, structured_schema
from memshift.textutil import sha256_json

SONNET, OPUS = "claude-sonnet-5-5", "claude-opus-5-5"
KEY = "sk-test-not-a-real-key"
WRITE = '{"value": "x", "statement": "The item is x.", "citations": []}'
PROGRAM = json.dumps({"code": "def derive(records, as_of):\n    return None\n"})


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("CLAUDE_API_KEY", KEY)
    monkeypatch.delenv("MEMSHIFT_ENDPOINTS", raising=False)
    monkeypatch.delenv("MEMSHIFT_CACHE_TAGS", raising=False)
    monkeypatch.setattr("memshift.experiments.common.REPO", tmp_path)


@pytest.fixture(scope="module")
def site() -> tuple[Any, ...]:
    """A world, its log, a definition item and a time after the shift."""
    world = generate_branch(5, "def", world_params(3))
    slot = next(s for s in world.slots if s.kappa_id)
    return world, SourceLog(world.events), slot, world.fork


def mocked(c: LLMClient, text: str, stop: str = "end_turn") -> list[dict[str, Any]]:
    """Replace the client's transport; returns the list the sent JSON bodies go to."""
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url == "https://api.anthropic.com/v1/messages"
        assert request.headers["x-api-key"] == KEY
        sent.append(json.loads(request.content))
        content = [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": text},
        ]
        usage = {"input_tokens": 100, "output_tokens": 20}
        return httpx.Response(
            200,
            json={
                "id": "m",
                "model": c.model,
                "content": content,
                "stop_reason": stop,
                "usage": usage,
            },
        )

    c._http = httpx.Client(transport=httpx.MockTransport(handler))
    return sent


def test_a_defwriter_definition_write(site):
    world, log, slot, t = site
    extra, limit = defwriter_settings(SONNET)
    c = client(SONNET, None, extra_body=extra, timeout=1800.0)  # as e1.run_world builds it
    assert c._http.timeout == httpx.Timeout(1800.0)
    sent = mocked(c, WRITE)
    out = derive(c, world, log, slot, t, max_tokens=limit, reasoning=True)
    assert out.ok and out.value == "x" and out.finish_reason == "end_turn"
    user = writer_request(world, slot, t, records_for(log, slot, t))["messages"][1]["content"]
    assert sent == [
        {
            "model": SONNET,
            "max_tokens": 17000,
            "system": writer.SYSTEM,
            "messages": [{"role": "user", "content": user}],
            "thinking": {"type": "adaptive"},
            "output_config": {
                "effort": "high",
                "format": {"type": "json_schema", "schema": structured_schema(WRITER_SCHEMA)},
            },
        }
    ]


def test_a_defwriter_reply_cut_off_at_the_limit_is_unreadable(site):
    world, log, slot, t = site
    extra, limit = defwriter_settings(SONNET)
    c = client(SONNET, None, extra_body=extra, timeout=1800.0)
    mocked(c, WRITE[:20], stop="max_tokens")
    out = derive(c, world, log, slot, t, max_tokens=limit, reasoning=True)
    assert not out.ok and out.finish_reason in writer.CUT_OFF


def program_body(world: Any, slot: Any, t: Any, model: str, **extra: Any) -> dict[str, Any]:
    user = program_request(world, slot, t)["messages"][1]["content"]
    return {
        "model": model,
        "max_tokens": 1500,
        "system": program.SYSTEM,
        "messages": [{"role": "user", "content": user}],
        **extra,
    }


def test_b_sonnet_program_request_with_sample_index_2(site):
    world, log, slot, t = site
    c = client(SONNET, None, extra_body=program_extra_body(SONNET, 0.0, 2))
    sent = mocked(c, PROGRAM)
    prog = synthesise(c, world, log, slot, t)
    fmt = {"format": {"type": "json_schema", "schema": PROGRAM_SCHEMA}}
    body = program_body(
        world, slot, t, SONNET, thinking={"type": "between_tools"}, output_config=fmt
    )
    assert sent == [body]
    # the sample index names the cache entry and is not sent
    assert prog.key == sha256_json({**body, "memshift_sample": 2}) != sha256_json(body)
    other = client(SONNET, None, extra_body=program_extra_body(SONNET, 0.0, 0))
    sent_0 = mocked(other, PROGRAM)
    assert synthesise(other, world, log, slot, t).key != prog.key and sent_0 == sent


def test_c_opus_program_request(site):
    world, log, slot, t = site
    c = client(OPUS, None, extra_body=program_extra_body(OPUS))
    sent = mocked(c, PROGRAM)
    synthesise(c, world, log, slot, t)
    fmt = {"effort": "low", "format": {"type": "json_schema", "schema": PROGRAM_SCHEMA}}
    assert sent == [program_body(world, slot, t, OPUS, output_config=fmt)]

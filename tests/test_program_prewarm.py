"""`experiments.program_prewarm` (round-3 swap phase): it sends exactly the definition items'
program requests that a G-b / V replay sends (the same response-cache keys), and a replay whose
program requests are all cache hits never contacts the program model's endpoint, so it runs while
only the reader is served."""

import json
from typing import Any

import httpx
import pytest

from memshift.experiments import common
from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import parse_cells, program_extra_body
from memshift.experiments.e1 import run as run_e1
from memshift.experiments.program_prewarm import planned_requests, prewarm
from memshift.llm import LLMClient, LLMResult
from tests.fakes import SCALAR_PROGRAM, SET_PROGRAM, FakeLLM, RecordingLLM

PROGRAM_MODEL = "fake/program-writer"
TEMPERATURE, SAMPLE = 0.7, 1


def replay(out: Any, seeds: list[int], branches: tuple[str, ...], **kw: Any) -> Any:
    """A G-b replay (block G-b's cells and flags) on small worlds."""
    cells, _ = parse_cells("D1sB0,D1sB1_source")
    return run_e1(
        seeds,
        "gb",
        out=out,
        cells_cfg=cells,
        baselines=False,
        branches=branches,
        world_params=world_params(3),
        discriminative=True,
        post_probes="definition",
        program_model=PROGRAM_MODEL,
        program_temperature=TEMPERATURE,
        program_sample=SAMPLE,
        **kw,
    )


def warm(out: Any, seeds: list[int], branches: tuple[str, ...], **kw: Any) -> dict[str, Any]:
    return prewarm(
        seeds,
        branches,
        PROGRAM_MODEL,
        TEMPERATURE,
        SAMPLE,
        world_params=world_params(3),
        out=out,
        **kw,
    )


@pytest.mark.parametrize(
    "seeds, branches, user_words",
    [([5, 6], ("src", "def", "orphan"), False), ([5], ("def",), True)],
)
def test_prewarm_sends_the_program_requests_of_the_replay(
    tmp_path, monkeypatch, seeds, branches, user_words
):
    made: list[RecordingLLM] = []

    def recording(model, run_dir, workers=48, **kw):
        llm = RecordingLLM(model, extra_body=kw.get("extra_body"))
        made.append(llm)
        return llm

    def replay_client(model, run_dir, workers=48, **kw):
        return recording(model, run_dir, **kw) if model == PROGRAM_MODEL else FakeLLM(model)

    monkeypatch.setattr("memshift.experiments.e1.client", replay_client)
    (tmp_path / "replay").mkdir()
    replay(tmp_path / "replay", seeds, branches, program_user_words=user_words)
    replayed = set().union(*(llm.keys("program") for llm in made))
    assert all(llm.model == PROGRAM_MODEL for llm in made) and len(made) == len(seeds)

    made.clear()
    monkeypatch.setattr("memshift.experiments.program_prewarm.client", recording)
    (tmp_path / "warm").mkdir()
    summary = warm(tmp_path / "warm", seeds, branches, program_user_words=user_words)
    (warmer,) = made
    assert warmer.extra_body == program_extra_body(PROGRAM_MODEL, TEMPERATURE, SAMPLE)
    sent = [k for k, _, _ in warmer.requests]
    assert set(sent) == replayed  # the same cache keys
    assert len(sent) == len(set(sent)) == summary["requests"]  # each request once
    # both definition versions are among them (S_def moves definitions), each item at t0
    plan, _ = planned_requests(seeds, branches, world_params(3), user_words)
    assert {p.version for p in plan} == {"v1", "v2"}
    assert {p.slot_id for p in plan if p.version == "v1"} == {p.slot_id for p in plan}
    texts = [b["messages"][-1]["content"] for _, _, b in warmer.requests]
    assert all(("Definition in effect" in t) for t in texts)
    rows = json.loads((tmp_path / "warm" / "prewarm.json").read_text())["rows"]
    assert {r["key"] for r in rows} == replayed


def served_program(self: LLMClient, body: dict[str, Any], key: str) -> LLMResult:
    """What the served program model answers (a valid program, as `tests.fakes.FakeLLM`)."""
    prompt = body["messages"][-1]["content"]
    code = SET_PROGRAM if "Python list of strings" in prompt else SCALAR_PROGRAM
    return LLMResult(json.dumps({"code": code}), 100, 20, 0.0, False, "stop", key)


def test_a_replay_on_a_warm_cache_never_contacts_the_program_endpoint(tmp_path, monkeypatch):
    seeds, branches = [5], ("src", "def")
    # the real client, its sqlite response cache under tmp_path, an endpoint nothing serves
    monkeypatch.setattr("memshift.experiments.common.REPO", tmp_path)
    monkeypatch.setenv("MEMSHIFT_ENDPOINTS", json.dumps({PROGRAM_MODEL: "http://127.0.0.1:9/v1"}))
    monkeypatch.delenv("MEMSHIFT_CACHE_TAGS", raising=False)
    monkeypatch.delenv("MEMSHIFT_HTTP_TIMEOUT", raising=False)

    # 1. the family phase: the program model is served
    monkeypatch.setattr(LLMClient, "_post", served_program)
    for sub in ("warm", "warm_again", "replay"):
        (tmp_path / sub).mkdir()
    first = warm(tmp_path / "warm", seeds, branches)
    assert first["requests"] > 0
    assert (first["sent"], first["hit"], first["no_code"]) == (first["requests"], 0, 0)
    again = warm(tmp_path / "warm_again", seeds, branches)  # a rerun (after a pause) sends nothing
    assert (again["sent"], again["hit"]) == (0, again["requests"])
    warmed = {r["key"] for r in read_jsonl(tmp_path / "warm" / "calls.jsonl")}

    # 2. the reader phase: nothing serves the program model, and any HTTP request fails the test
    def unserved(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the program model's endpoint was contacted")

    monkeypatch.setattr(LLMClient, "_post", unserved)
    monkeypatch.setattr(httpx.Client, "send", unserved)

    def replay_client(model, run_dir, workers=48, **kw):
        if model == PROGRAM_MODEL:
            return common.client(model, run_dir, workers, **kw)  # what e1 builds
        return FakeLLM(model)

    monkeypatch.setattr("memshift.experiments.e1.client", replay_client)
    out = replay(tmp_path / "replay", seeds, branches)
    calls = [c for c in read_jsonl(out / "calls.jsonl") if c["model"] == PROGRAM_MODEL]
    assert calls and all(c["cached"] for c in calls)
    assert {c["key"] for c in calls} == warmed
    manifest = json.loads((out / "manifest.json").read_text())  # serving metadata from files
    assert manifest["models"]["program_writer"]["model"] == PROGRAM_MODEL
    (summary,) = [s for s in read_jsonl(out / "world_summaries.jsonl") if s["seed"] == seeds[0]]
    assert summary["program_llm_usage"]["calls"] == summary["program_llm_usage"]["cached"] > 0

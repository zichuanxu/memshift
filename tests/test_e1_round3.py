"""The run-level writer flags of round 3: the thinking writer's output limit
(`--think-max-tokens`), the definition-item writer (`--defwriter`) and the program sets
(`--program-model`, `--program-temperature`, `--program-sample`). Each test
replays cells on one small world with recording fake writers, which give the request bodies of
every client. That the defaults change no existing request is pinned in test_e1_round3_pins.py."""

import json
import re
from datetime import timedelta
from typing import Any

import pytest

from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import (
    DEFWRITER_BODY,
    DEFWRITER_MAX_TOKENS,
    defwriter_settings,
    parse_cells,
    program_extra_body,
    separate_program_client,
)
from memshift.experiments.e1 import run as run_e1
from memshift.gen.kappa import KAPPA_DEFS
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.program import PROGRAM_SCHEMA, program_request
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import THINK_MAX_TOKENS, WRITER_SCHEMA, derive
from memshift.llm import LLMResult, default_extra_body, structured_schema
from memshift.probes import make_probes
from memshift.reader import context_hash
from tests.e1_replay import THINK_BODY
from tests.fakes import FakeLLM, RecordingLLM

WEEKS = (1, 2)
KAPPA_DESCRIPTIONS = {spec["description"] for spec in KAPPA_DEFS.values()}
SONNET = "claude-sonnet-5-5"
OPUS = "claude-opus-5-5"
# what the API must never receive from the Claude clients (fixed Claude request settings)
NOT_SENT = {"temperature", "top_p", "seed", "response_format", "chat_template_kwargs"}


def replay_with(cfg: CellConfig, branch: str = "def", **kw: Any) -> dict[str, Any]:
    """Build one cell at t0 and maintain it at WEEKS; `kw` goes to HoldMem. Returns the cell, its
    recording clients, its maintenance stats and its contexts at t0."""
    world = generate_branch(5, branch, world_params(3))
    log = SourceLog(world.events)
    writer: Any = RecordingLLM()
    think: Any = None
    if cfg.think:
        think = RecordingLLM(extra_body={**RecordingLLM().extra_body, **THINK_BODY})
    hm = HoldMem(world, log, cfg, writer, "fake/writer-a", think, **kw)
    hm.workers = 4
    t0 = world.fork - timedelta(minutes=1)
    hm.build(t0)
    contexts = {p.probe_id: context_hash(hm.serve(p).blocks) for p in make_probes(world, log, [t0])}
    stats = [hm.maintain(world.fork + timedelta(weeks=k)) for k in WEEKS]
    return {
        "world": world,
        "hm": hm,
        "writer": writer,
        "think": think,
        "stats": stats,
        "contexts_t0": contexts,
    }


def requests(llm: Any, purpose: str | None = None) -> list[dict[str, Any]]:
    """The request bodies a recording client sent, in a fixed order."""
    out = [b for _, p, b in llm.requests if purpose is None or p == purpose]
    return sorted(out, key=lambda b: json.dumps(b, sort_keys=True))


def request_text(body: dict[str, Any]) -> str:
    """The text of a request body, OpenAI-style or Messages API (system messages in `system`)."""
    system = [body["system"]] if "system" in body else []
    return json.dumps(system + [m["content"] for m in body["messages"]])


def site(body: dict[str, Any]) -> tuple[str, str]:
    """(item description, as-of time) of a writer request body."""
    text = body["messages"][-1]["content"]
    item = re.search(r"Knowledge item: (.*)", text)
    as_of = re.search(r"As of: (.*)", text)
    assert item and as_of
    return item.group(1), as_of.group(1)


def after_shift_definition_write(body: dict[str, Any], world: Any) -> bool:
    item, as_of = site(body)
    return item in KAPPA_DESCRIPTIONS and as_of >= f"{world.fork:%Y-%m-%d %H:%M}"


# ---------------------------------------------------------------------------- --think-max-tokens
@pytest.fixture(scope="module")
def think_runs():
    cfg = CellConfig(0, 1, "source", think=True)
    return replay_with(cfg), replay_with(cfg, think_max_tokens=32768)


def test_the_think_limit_changes_only_max_tokens_of_the_thinking_requests(think_runs):
    default, big = think_runs
    assert default["hm"].think_max_tokens == THINK_MAX_TOKENS == 8192
    a, b = requests(default["think"]), requests(big["think"])
    assert a and len(a) == len(b)
    assert all(x["max_tokens"] == THINK_MAX_TOKENS for x in a)
    assert all(y["max_tokens"] == 32768 for y in b)
    assert [{**y, "max_tokens": THINK_MAX_TOKENS} for y in b] == a
    # the normal writer's requests do not move
    assert requests(default["writer"]) == requests(big["writer"])
    assert all(x["max_tokens"] in (700,) for x in requests(default["writer"], "write"))


def test_run_records_the_think_limit_and_gives_the_client_time(tmp_path, monkeypatch):
    made = []

    def fake_client(model, run_dir, workers=48, **kw):
        made.append((model, kw.get("extra_body"), kw.get("timeout")))
        return FakeLLM(model)

    monkeypatch.setattr("memshift.experiments.e1.client", fake_client)
    cells, _ = parse_cells("D0B1_source,D0thinkB1_source")
    out = run_e1(
        [5],
        "think",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
        think_max_tokens=32768,
    )
    params = json.loads((out / "manifest.json").read_text())["params"]
    assert params["think_writer"]["max_tokens"] == 32768
    think = [m for m in made if m[1] == {"chat_template_kwargs": {"enable_thinking": True}}]
    assert think and think[0][2] == 1800.0 * 4


# ---------------------------------------------------------------------------- --defwriter
DEF_CELLS = [
    CellConfig(0, 0),
    CellConfig(0, 1, "source"),
    CellConfig(1, 0),
    CellConfig(1, 1, "source"),
    CellConfig(1, 0, synth=True),
    CellConfig(1, 1, "source", synth=True),
]


def sonnet() -> Any:
    return RecordingLLM(SONNET, extra_body=DEFWRITER_BODY)


@pytest.fixture(scope="module")
def def_runs():
    """cell name -> (the cell without a definition-item writer, the cell with one)."""
    return {
        c.name: (
            replay_with(c),
            replay_with(c, def_writer=sonnet(), def_max_tokens=DEFWRITER_MAX_TOKENS),
        )
        for c in DEF_CELLS
    }


def test_defwriter_settings():
    adaptive = {"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}}
    assert defwriter_settings(SONNET) == (adaptive, 17000)
    assert defwriter_settings(OPUS) == (adaptive, DEFWRITER_MAX_TOKENS)
    defwriter_settings(SONNET)[0]["output_config"]["effort"] = "low"  # a copy, not the constant
    assert adaptive == DEFWRITER_BODY
    model = "Qwen/Qwen3.5-9B"
    assert defwriter_settings(model) == (default_extra_body(model), None)


def test_the_defwriter_writes_exactly_the_definition_items_after_the_shift(def_runs):
    for name, (plain, run) in def_runs.items():
        world, sent = run["world"], run["hm"].def_writer.requests
        assert sent, name
        moved = [
            b for b in requests(plain["writer"], "write") if after_shift_definition_write(b, world)
        ]
        assert moved, name
        for _, purpose, body in sent:
            assert purpose == "write" and after_shift_definition_write(body, world)
            assert body["model"] == SONNET and body["thinking"] == {"type": "adaptive"}
            assert body["max_tokens"] == DEFWRITER_MAX_TOKENS
            assert body["output_config"] == {
                "effort": "high",
                "format": {"type": "json_schema", "schema": structured_schema(WRITER_SCHEMA)},
            }
            assert not NOT_SENT & body.keys()
        # the same request text as the main writer's
        assert sorted(request_text(b) for _, _, b in sent) == sorted(
            request_text(b) for b in moved
        ), name
        # every other request (programs included) is the main writer's, unchanged
        rest = [
            b
            for b in requests(plain["writer"])
            if "As of: " not in b["messages"][-1]["content"]
            or not after_shift_definition_write(b, world)
        ]
        assert requests(run["writer"]) == rest, name
        assert run["contexts_t0"] == plain["contexts_t0"], name  # the pre-shift snapshot


def test_defwriter_items_record_the_write_and_other_items_do_not(def_runs):
    for name, (plain, run) in def_runs.items():
        hm, ref = run["hm"], plain["hm"]
        assert hm.items.keys() == ref.items.keys(), name
        n_def = 0
        for item in hm.items.values():
            slot = hm.world.slot(item.slot_id)
            if slot.kappa_id and item.written > hm.world.fork:
                n_def += 1
                assert item.note == {
                    "defwriter": {
                        "model": SONNET,
                        "ok": True,
                        "completion_tokens": 20,
                        "at_limit": False,
                    }
                }
                assert item.theta_w == "fake/writer-a"  # keys keep the main writer's id
            else:
                assert item.note == {} and "note" not in item.to_json()
                assert item.s_k == ref.items[item.item_id].s_k
        assert n_def > 0, name
        sent = len(hm.def_writer.requests)
        assert hm.def_stats == {"writes": sent, "unreadable": 0, "at_limit": 0}
        assert run["stats"] == plain["stats"], name


class CutLLM(RecordingLLM):
    """A recording writer whose replies are cut off by the output limit (Anthropic's reason)."""

    def complete(self, messages, max_tokens=512, schema=None, purpose=""):
        res = super().complete(messages, max_tokens, schema, purpose)
        return LLMResult(res.text[:-5], 10, max_tokens, 0.0, False, "max_tokens", res.key)


def test_a_defwriter_reply_cut_off_by_the_limit_is_unreadable_and_counted():
    w = generate_branch(5, "def", world_params(3))
    log = SourceLog(w.events)
    slot = next(s for s in w.slots if s.kappa_id)
    cut = derive(CutLLM(), w, log, slot, w.fork, max_tokens=17000, reasoning=True)
    assert not cut.ok and cut.value is None and cut.finish_reason == "max_tokens"
    assert cut.completion_tokens == 17000
    for cfg in (CellConfig(0, 1, "source"), CellConfig(0, 0)):
        run = replay_with(cfg, def_writer=CutLLM(SONNET), def_max_tokens=DEFWRITER_MAX_TOKENS)
        hm = run["hm"]
        sent = len(hm.def_writer.requests)
        assert sent > 0
        assert hm.def_stats == {"writes": sent, "unreadable": sent, "at_limit": sent}
        noted = [i.note["defwriter"] for i in hm.items.values() if "defwriter" in i.note]
        if cfg.B:  # a B=1 repair stores the unreadable write and records it
            assert noted and all(not n["ok"] and n["at_limit"] for n in noted)
            assert all(n["completion_tokens"] == DEFWRITER_MAX_TOKENS for n in noted)
        else:  # B=0 drops an unreadable delta write; only the counts keep it
            assert noted == []


def no_think_body(model: str) -> dict[str, Any]:
    raise AssertionError(f"think_extra_body({model}) reached")


def test_run_builds_the_defwriter_client_and_records_it(tmp_path, monkeypatch):
    made = []

    def fake_client(model, run_dir, workers=48, **kw):
        made.append((model, kw.get("extra_body"), kw.get("timeout")))
        return FakeLLM(model)

    monkeypatch.setattr("memshift.experiments.e1.client", fake_client)
    monkeypatch.setattr("memshift.experiments.e1.think_extra_body", no_think_body)
    cells, _ = parse_cells("D0B0,D1sB1_source")
    out = run_e1(
        [5],
        "defwriter",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
        defwriter=SONNET,
    )
    assert (SONNET, DEFWRITER_BODY, 1800.0) in made
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["defwriter"]["model"] == SONNET
    assert manifest["params"]["defwriter"]["extra_body"] == DEFWRITER_BODY
    assert manifest["params"]["defwriter"]["max_tokens"] == DEFWRITER_MAX_TOKENS
    assert manifest["models"]["defwriter"]["model"] == SONNET
    (summary,) = read_jsonl(out / "world_summaries.jsonl")
    stats = summary["branches"]["def"]["defwriter"]
    assert set(stats) == {c.name for c in cells} and all(s["writes"] > 0 for s in stats.values())
    assert summary["defwriter_llm_usage"]["calls"] > 0
    noted = [r for r in read_jsonl(out / "items.jsonl") if "defwriter" in r.get("note", {})]
    assert noted and all(r["slot_id"].startswith("k.") for r in noted)
    with pytest.raises(ValueError, match="D0think"):
        run_e1([5], out=tmp_path, cells_cfg=parse_cells("D0thinkB0")[0], defwriter=SONNET)


# ---------------------------------------------------------------------------- program sets
PROGRAM_CELLS = [CellConfig(1, 0, synth=True), CellConfig(1, 1, "source", synth=True)]


def definition_program(body: dict[str, Any]) -> bool:
    item = re.search(r"Knowledge item: (.*)", body["messages"][-1]["content"])
    return item is not None and item.group(1) in KAPPA_DESCRIPTIONS


def test_program_extra_body_and_when_a_separate_client_is_needed():
    qwen = "Qwen/Qwen3.5-9B"
    assert program_extra_body(qwen) == default_extra_body(qwen)
    assert program_extra_body(qwen, 0.7, 2) == {
        **default_extra_body(qwen),
        "temperature": 0.7,
        "memshift_sample": 2,
    }
    # Claude: no up-front thinking (Sonnet), the least thinking (Opus); never a temperature
    assert program_extra_body(SONNET) == {"thinking": {"type": "between_tools"}}
    assert program_extra_body(SONNET, 0.0, 0) == {
        "thinking": {"type": "between_tools"},
        "memshift_sample": 0,
    }
    assert program_extra_body(OPUS, sample=2) == {
        "output_config": {"effort": "low"},
        "memshift_sample": 2,
    }
    for model in (SONNET, OPUS):
        with pytest.raises(ValueError, match="no temperature"):
            program_extra_body(model, 0.7, 0)
    with pytest.raises(ValueError, match="no program-writer settings"):
        program_extra_body("claude-other-1")
    assert not separate_program_client(qwen, qwen, 0.0, None)
    assert separate_program_client(SONNET, qwen, 0.0, None)
    assert separate_program_client(qwen, qwen, 0.7, None)
    assert separate_program_client(qwen, qwen, 0.0, 0)


def test_a_sample_index_gives_each_sample_its_own_cache_key():
    w = generate_branch(5, "def", world_params(3))
    slot = next(s for s in w.slots if s.kappa_id)
    request = program_request(w, slot, w.fork)

    def key(**kw):
        llm = RecordingLLM(extra_body=program_extra_body("fake/writer-a", **kw))
        return llm.request_key(request["messages"], request["max_tokens"], request["schema"])

    (k0, plain), (k1, s1), (k2, s2) = key(), key(temperature=0.7, sample=1), key(temperature=0.7)
    assert len({k0, k1, k2, key(temperature=0.7, sample=2)[0]}) == 4
    assert key(temperature=0.7, sample=1)[0] == k1  # the same sample is a cache hit
    assert plain["temperature"] == 0.0 and s1["temperature"] == s2["temperature"] == 0.7
    assert s1["memshift_sample"] == 1 and "memshift_sample" not in s2
    assert {k: v for k, v in s1.items() if k not in ("temperature", "memshift_sample")} == {
        k: v for k, v in plain.items() if k != "temperature"
    }


@pytest.fixture(scope="module")
def program_runs():
    """cell name -> (the cell, the cell whose definition programs come from another client)."""
    body = program_extra_body(SONNET, sample=2)
    return {
        c.name: (replay_with(c), replay_with(c, program_writer=RecordingLLM(SONNET, body)))
        for c in PROGRAM_CELLS
    }


def test_only_the_definition_items_programs_move_to_the_program_client(program_runs):
    for name, (plain, run) in program_runs.items():
        sent = run["hm"].program_writer.requests
        assert sent, name
        moved = [b for b in requests(plain["writer"], "program") if definition_program(b)]
        for _, purpose, body in sent:
            assert purpose == "program" and definition_program(body)
            assert body["model"] == SONNET and body["thinking"] == {"type": "between_tools"}
            assert body["output_config"] == {
                "format": {"type": "json_schema", "schema": PROGRAM_SCHEMA}
            }
            assert body["memshift_sample"] == 2 and not NOT_SENT & body.keys()
        assert sorted(request_text(b) for _, _, b in sent) == sorted(
            request_text(b) for b in moved
        ), name
        rest = [
            b
            for _, p, b in plain["writer"].requests
            if not (p == "program" and definition_program(b))
        ]
        assert requests(run["writer"]) == sorted(rest, key=lambda b: json.dumps(b, sort_keys=True))
        assert all(not definition_program(b) for b in requests(run["writer"], "program"))
        assert run["contexts_t0"] == plain["contexts_t0"], name


def test_run_builds_the_program_client_only_off_the_defaults(tmp_path, monkeypatch):
    made = []

    def fake_client(model, run_dir, workers=48, **kw):
        made.append((model, kw.get("extra_body")))
        return FakeLLM(model)

    monkeypatch.setattr("memshift.experiments.e1.client", fake_client)
    monkeypatch.setattr("memshift.experiments.e1.think_extra_body", no_think_body)
    cells, _ = parse_cells("D0B0,D1sB1_source")
    common = {
        "cells_cfg": cells,
        "baselines": False,
        "branches": ("def",),
        "world_params": world_params(3),
        "discriminative": True,
    }
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    out = run_e1([5], "programs", out=tmp_path / "a", **common)
    assert made == [("Qwen/Qwen3.5-9B", None)]  # the writer only
    params = json.loads((out / "manifest.json").read_text())["params"]
    assert params["program_writer"] == {
        "model": "Qwen/Qwen3.5-9B",
        "temperature": 0.0,
        "sample": None,
        "extra_body": None,
        "items": "definition items (other items: the writer)",
        "user_words": False,
    }
    made.clear()
    out = run_e1(
        [5],
        "programs",
        out=tmp_path / "b",
        program_model=SONNET,
        program_sample=1,
        **common,
    )
    body = {"thinking": {"type": "between_tools"}, "memshift_sample": 1}
    assert (SONNET, body) in made
    manifest = json.loads((out / "manifest.json").read_text())
    pw = manifest["params"]["program_writer"]
    assert (pw["model"], pw["temperature"], pw["sample"], pw["extra_body"]) == (
        SONNET,
        0.0,
        1,
        body,
    )
    assert manifest["models"]["program_writer"]["model"] == SONNET
    (summary,) = read_jsonl(out / "world_summaries.jsonl")
    assert summary["program_llm_usage"]["calls"] > 0
    # --program-temperature with a Claude program model is refused before anything runs
    made.clear()
    for model in (SONNET, OPUS):
        with pytest.raises(ValueError, match="no temperature"):
            run_e1([5], out=tmp_path, program_model=model, program_temperature=0.7, **common)
    assert made == []

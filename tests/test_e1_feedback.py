"""The execution-feedback cells `D1sfbB0` / `D1sfbB1_source`
(`holdmem.program.synthesise_feedback`). The loop is tested with a fake writer whose programs fail
in each way the interpreter reports; the cells are replayed on one small world against D1s, whose
requests they must repeat byte for byte; and a fake-LLM E1 run with the new cells passes the
pre-intervention hash assertion."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import timedelta
from typing import Any

import pytest

from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import parse_cell, parse_cells
from memshift.experiments.e1 import run as run_e1
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.program import (
    FEEDBACK_CHARS,
    FEEDBACK_ROUNDS,
    PROGRAM_SCHEMA,
    failure_kind,
    failure_text,
    program_request,
    synthesise,
    synthesise_feedback,
)
from memshift.holdmem.system import CellConfig
from memshift.llm import LLMResult
from tests.e1_replay import Replay, replay
from tests.fakes import SCALAR_PROGRAM, SET_PROGRAM, FakeLLM, RecordingLLM

D1SB0, D1SSRC = CellConfig(1, 0, synth=True), CellConfig(1, 1, "source", synth=True)
FBB0 = CellConfig(1, 0, synth=True, feedback=True)
FBSRC = CellConfig(1, 1, "source", synth=True, feedback=True)

SYNTAX = "def derive(records, as_of):\n    return [\n"
RAISES = "def derive(records, as_of):\n    return records[0]['no_such_field']\n"
WRONG_TYPE = "def derive(records, as_of):\n    return 7 if False else [1, 2]\n"
NO_DERIVE = "def compute(records, as_of):\n    return []\n"
SET_OK = "def derive(records, as_of):\n    return ['b', 'a', 'a']\n"


def reply(code: str) -> str:
    return json.dumps({"code": code})


class FeedbackLLM(RecordingLLM):
    """Program requests: `first`. Feedback requests: `fixes[failed code]`, else `default`; other
    requests as RecordingLLM (writer: value 'X', reader: abstain)."""

    def __init__(self, first: str, fixes: dict[str, str] | None = None, default: str = SET_OK):
        super().__init__()
        self.first, self.fixes, self.default = first, fixes or {}, default

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        if purpose not in ("program", "program_feedback"):
            return super().complete(messages, max_tokens, schema, purpose)
        key, body = self.request_key(messages, max_tokens, schema)
        with self._rec_lock:
            self.requests.append((key, purpose, body))
        return LLMResult(reply(self.program(messages, purpose)), 100, 20, 0.0, False, "stop", key)

    def program(self, messages: Sequence[dict[str, str]], purpose: str) -> str:
        if purpose == "program":
            return self.first
        failed = json.loads(messages[-2]["content"])["code"]
        return self.fixes.get(failed, self.default)


@pytest.fixture(scope="module")
def small():
    world = generate_branch(5, "def", world_params(3))
    log = SourceLog(world.events)
    slot = next(s for s in world.slots if s.vtype.startswith("set_") and s.kappa_id)
    return world, log, slot, world.fork - timedelta(days=7)


# ---------------------------------------------------------------------------- tokens and config
def test_tokens_names_and_validation():
    assert parse_cell("D1sfbB0") == FBB0 and FBB0.name == "holdmem_D1sfbB0"
    assert parse_cell("D1sfbB1_source") == FBSRC and FBSRC.name == "holdmem_D1sfbB1_source"
    for token in ("D1sB0", "D1sB1_source", "D1sB0_ra", "D1sB0_upsert", "D1sB0_refresh"):
        assert not parse_cell(token).feedback
        assert parse_cell(token).name == f"holdmem_{token}"
    assert parse_cells("D0B0,D1sfbB0,D1sfbB1_source")[0] == [CellConfig(0, 0), FBB0, FBSRC]
    with pytest.raises(ValueError, match="feedback"):
        CellConfig(1, 0, feedback=True)  # not a D1s cell
    with pytest.raises(ValueError, match="feedback"):
        CellConfig(1, 0, synth=True, ra=True, feedback=True)
    with pytest.raises(ValueError):
        parse_cell("D0sfbB0")
    with pytest.raises(ValueError, match="feedback"):
        parse_cell("D1sfbB0_ra")


# ---------------------------------------------------------------------------- the loop
def test_failure_kinds_and_text():
    assert failure_kind("no code in the response") == "no_code"
    assert failure_kind("syntax error: invalid syntax (<unknown>, line 2)") == "syntax"
    assert failure_kind("no top-level function derive") == "no_derive"
    assert failure_kind("derive must take exactly (records, as_of)") == "no_derive"
    assert failure_kind("disallowed syntax: Import") == "sandbox"
    assert failure_kind("forbidden name: __import__") == "sandbox"
    assert failure_kind("timeout") == "timeout"
    assert failure_kind("KeyError: 'x'") == "exception"
    assert failure_kind("type mismatch: set_person needs a list of str") == "type"
    long = failure_text("type mismatch: set_person needs a list of str", ["n" * 50] * 40)
    assert len(long) == FEEDBACK_CHARS and long.startswith("type mismatch: set_person needs")
    assert "2 seconds" in failure_text("timeout")


def test_a_program_that_runs_sends_no_feedback_and_is_d1s(small):
    world, log, slot, t = small
    llm, plain = FeedbackLLM(SET_OK), FeedbackLLM(SET_OK)
    kept: dict[tuple[str, str], str] = {}
    out = synthesise_feedback(llm, world, log, slot, t, kept)
    ref = synthesise(plain, world, log, slot, t)
    assert (out.ok, out.value, out.error, out.key) == (ref.ok, ref.value, ref.error, ref.key)
    assert out.value == ["a", "b"] and not kept
    assert [p for _, p, _ in llm.requests] == ["program"] and llm.keys() == plain.keys()
    assert out.feedback == {
        "start": "program",
        "d1s_ok": True,
        "d1s_kind": None,
        "rounds": 0,
        "kinds": [],
        "ok": True,
        "world_data": [],
    }


def test_a_syntax_error_corrected_on_round_1(small):
    world, log, slot, t = small
    llm = FeedbackLLM(SYNTAX, {SYNTAX: SET_OK})
    kept: dict[tuple[str, str], str] = {}
    out = synthesise_feedback(llm, world, log, slot, t, kept)
    assert out.ok and out.value == ["a", "b"]
    fb = out.feedback
    assert fb is not None
    assert (fb["d1s_ok"], fb["d1s_kind"], fb["rounds"], fb["kinds"]) == (
        False,
        "syntax",
        1,
        ["syntax"],
    )
    assert fb["world_data"] == [False]
    # the request: program messages, the failed reply as returned, the fixed feedback turn
    request = program_request(world, slot, t)
    (_, _, body), (_, purpose, fb_body) = llm.requests
    assert purpose == "program_feedback"
    assert fb_body["messages"][:2] == request["messages"] == body["messages"]
    assert fb_body["messages"][2] == {"role": "assistant", "content": reply(SYNTAX)}
    turn = fb_body["messages"][3]
    assert turn["role"] == "user" and turn["content"].count("syntax error") == 1
    assert {k: v for k, v in fb_body.items() if k != "messages"} == {
        k: v for k, v in body.items() if k != "messages"
    }
    assert fb_body["max_tokens"] == request["max_tokens"] == 1500
    assert fb_body["response_format"]["json_schema"]["schema"] == PROGRAM_SCHEMA
    assert out.key == llm.requests[0][0]  # the D1s program key
    assert list(kept.values()) == [reply(SET_OK)]


def test_an_exception_corrected_on_round_2(small):
    world, log, slot, t = small
    llm = FeedbackLLM(RAISES, {RAISES: NO_DERIVE, NO_DERIVE: SET_OK})
    out = synthesise_feedback(llm, world, log, slot, t, {})
    assert out.ok and out.feedback is not None
    assert out.feedback["kinds"] == ["exception", "no_derive"] and out.feedback["rounds"] == 2
    turns = [b["messages"][-1]["content"] for _, p, b in llm.requests if p == "program_feedback"]
    assert "KeyError: 'no_such_field'" in turns[0] and "no top-level function derive" in turns[1]
    # round 2 shows only the reply that failed last
    last = [b for _, p, b in llm.requests if p == "program_feedback"][-1]["messages"]
    assert len(last) == 4 and last[2]["content"] == reply(NO_DERIVE)


def test_a_wrong_type_never_corrected_ends_as_d1s_on_a_failing_program(small):
    world, log, slot, t = small
    llm = FeedbackLLM(WRONG_TYPE, default=WRONG_TYPE)
    kept: dict[tuple[str, str], str] = {}
    out = synthesise_feedback(llm, world, log, slot, t, kept)
    ref = synthesise(FeedbackLLM(WRONG_TYPE), world, log, slot, t)
    assert not out.ok and (out.value, out.error) == (ref.value, ref.error)
    assert out.feedback is not None
    assert out.feedback["rounds"] == FEEDBACK_ROUNDS == 2
    assert out.feedback["kinds"] == ["type"] * 3 and not out.feedback["ok"]
    turn = llm.requests[1][2]["messages"][-1]["content"]
    assert "needs a list of str; it returned [1, 2]" in turn
    assert not kept
    assert (out.prompt_tokens, out.completion_tokens) == (300, 60)  # 1 program + 2 feedback


def test_a_corrected_program_is_kept_and_rechecked_at_later_writes(small):
    world, log, slot, t = small
    llm = FeedbackLLM(SYNTAX, {SYNTAX: SET_OK})
    kept: dict[tuple[str, str], str] = {}
    synthesise_feedback(llm, world, log, slot, t, kept)
    later = synthesise_feedback(llm, world, log, slot, t + timedelta(days=6), kept)
    assert later.ok and later.feedback is not None
    assert (later.feedback["start"], later.feedback["rounds"]) == ("kept", 0)
    assert (later.feedback["d1s_ok"], later.feedback["d1s_kind"]) == (False, "syntax")
    assert [p for _, p, _ in llm.requests] == ["program", "program_feedback", "program"]
    # a kept program that fails on new records goes back again, starting from that program
    kept[next(iter(kept))] = reply(RAISES)
    again = synthesise_feedback(llm, world, log, slot, t, kept)
    assert again.ok and again.feedback is not None
    assert again.feedback["start"] == "kept" and again.feedback["kinds"] == ["exception"]
    assert llm.requests[-1][2]["messages"][2]["content"] == reply(RAISES)
    assert next(iter(kept.values())) == reply(SET_OK)


# ---------------------------------------------------------------------------- the cells
class CellLLM(FeedbackLLM):
    """Definition items: a first program with a syntax error, corrected by the first feedback
    request; other items: the FakeLLM programs, which run."""

    def __init__(self) -> None:
        super().__init__(SYNTAX)

    def program(self, messages: Sequence[dict[str, str]], purpose: str) -> str:
        prompt = messages[1]["content"]
        is_set = "Python list of strings" in prompt
        if purpose == "program_feedback":
            return SET_OK if is_set else SCALAR_PROGRAM
        if "Definition in effect" in prompt:
            return SYNTAX
        return SET_PROGRAM if is_set else SCALAR_PROGRAM


@pytest.fixture(scope="module")
def rep() -> Replay:
    return replay([D1SB0, D1SSRC, FBB0, FBSRC], weeks=(1, 2, 4), make_llm=CellLLM)


def _sent(rep: Replay, cfg: CellConfig, purposes: tuple[str, ...]) -> list[str]:
    return sorted(repr(b) for _, p, b in rep.llms[cfg.name].requests if p in purposes)


def test_the_feedback_cells_send_the_d1s_requests_and_only_feedback_requests_besides(rep):
    for fb, plain in ((FBB0, D1SB0), (FBSRC, D1SSRC)):
        shared = ("write", "program")
        assert _sent(rep, fb, shared) == _sent(rep, plain, shared), fb.name
        assert {p for _, p, _ in rep.llms[plain.name].requests} == set(shared)
        assert _sent(rep, fb, ("program_feedback",)), fb.name


def test_pre_shift_contexts_are_those_of_the_d1s_cells(rep):
    first = rep.contexts_t0[D1SB0.name]
    for name, ctx in rep.contexts_t0.items():
        assert ctx == first, name


def test_feedback_repairs_the_definition_items_and_is_noted_per_write(rep):
    for fb, plain in ((FBB0, D1SB0), (FBSRC, D1SSRC)):
        hm, ref = rep.cells[fb.name], rep.cells[plain.name]
        assert hm.items.keys() == ref.items.keys()
        n_first = n_kept = 0
        for item_id, item in hm.items.items():
            slot = rep.world.slot(item.slot_id)
            for ev, ref_ev in zip(item.history, ref.items[item_id].history, strict=True):
                if "program" not in ev:
                    continue
                note = ev["feedback"]
                assert note["d1s_ok"] == ref_ev["program"]["ok"]
                if slot.kappa_id:
                    assert not note["d1s_ok"] and note["ok"] and ev["program"]["ok"]
                    n_first += note["start"] == "program"
                    n_kept += note["start"] == "kept"
                    assert note["rounds"] == (1 if note["start"] == "program" else 0)
                else:
                    assert note == {**note, "rounds": 0, "kinds": [], "start": "program"}
                    assert ev["program"] == ref_ev["program"]
        assert n_first and n_kept, fb.name
        n_feedback = sum(p == "program_feedback" for _, p, _ in rep.llms[fb.name].requests)
        assert n_feedback == n_first


# ---------------------------------------------------------------------------- an E1 run
def test_e1_run_with_the_feedback_cells(tmp_path, monkeypatch):
    def fake_client(model: str, run_dir: Any, workers: int = 48, **kw: Any) -> FakeLLM:
        return CellLLM()

    monkeypatch.setattr("memshift.experiments.e1.client", fake_client)
    cells, _ = parse_cells("D0B0,D0B1_source,D1sB0,D1sB1_source,D1sfbB0,D1sfbB1_source")
    out = run_e1(
        [5],
        "fb",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def", "orphan"),
        world_params=world_params(3),
        discriminative=True,
    )
    for branch in ("def", "orphan"):
        assert json.loads((out / f"hash_check_w0005_{branch}.json").read_text())["ok"]
    params = json.loads((out / "manifest.json").read_text())["params"]
    assert params["program_feedback"] == {
        "rounds": 2,
        "failure_chars": 600,
        "kept": "per cell, item and program request; re-checked at every write",
    }
    rows = list(read_jsonl(out / "items.jsonl"))
    notes = [r for r in rows if r["system"].startswith("holdmem_D1sfb")]
    assert notes and all("feedback" in r["note"] for r in notes)
    assert not any("note" in r for r in rows if not r["system"].startswith("holdmem_D1sfb"))

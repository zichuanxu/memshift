"""Round 3, Claude runs: the user-words program requests (`--program-user-words`,
`holdmem.program.USER_WORDS`) and the Claude reader settings (`--reader claude-sonnet-5-5`,
`CLAUDE_READER_BODY`). The defaults change no request."""

import json
from datetime import timedelta

import pytest

from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import CLAUDE_READER_BODY, parse_cells, reader_extra_body
from memshift.experiments.e1 import run as run_e1
from memshift.gen.kappa import KAPPA_DEFS
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.program import USER_WORDS, program_request
from memshift.holdmem.system import CellConfig, HoldMem
from tests.fakes import FakeLLM, RecordingLLM


@pytest.fixture(scope="module")
def world():
    return generate_branch(5, "def", world_params(3))


def test_user_words_cover_the_six_fixed_definitions_and_both_versions():
    assert set(USER_WORDS) == set(KAPPA_DEFS)
    for kid, spec in KAPPA_DEFS.items():
        assert set(USER_WORDS[kid]) == {"v1", "v2"}
        for ver in ("v1", "v2"):
            assert USER_WORDS[kid][ver] != spec[ver]


def test_user_words_change_only_the_definition_line_of_definition_programs(world):
    t0 = world.fork - timedelta(minutes=1)
    t1 = world.fork + timedelta(weeks=1)
    for slot in world.slots:
        for t in (t0, t1):
            plain = program_request(world, slot, t)
            words = program_request(world, slot, t, user_words=True)
            if not slot.kappa_id:
                assert words == plain
                continue
            ver = world.kappa_version(slot.kappa_id, t)
            a = plain["messages"][-1]["content"].splitlines()
            b = words["messages"][-1]["content"].splitlines()
            diff = [(x, y) for x, y in zip(a, b, strict=True) if x != y]
            assert diff == [
                (
                    f"Definition in effect ({slot.kappa_id} {ver}): "
                    + world.kappa_defs[slot.kappa_id][ver].statement,
                    f"Definition in effect ({slot.kappa_id} {ver}): "
                    + USER_WORDS[slot.kappa_id][ver],
                )
            ]
            assert {k: v for k, v in words.items() if k != "messages"} == {
                k: v for k, v in plain.items() if k != "messages"
            }


def test_holdmem_sends_user_words_only_for_definition_programs(world):
    log = SourceLog(world.events)
    t0 = world.fork - timedelta(minutes=1)
    sent = {}
    for words in (False, True):
        writer = RecordingLLM()
        hm = HoldMem(
            world, log, CellConfig(1, 0, synth=True), writer, "fake/w", program_user_words=words
        )
        hm.workers = 4
        hm.build(t0)
        sent[words] = sorted(
            b["messages"][-1]["content"] for _, p, b in writer.requests if p == "program"
        )
    changed = [(a, b) for a, b in zip(sent[False], sent[True], strict=True) if a != b]
    assert len(changed) == sum(1 for s in world.slots if s.kappa_id)
    assert all("Definition in effect" in a for a, _ in changed)
    with pytest.raises(ValueError, match="_ra"):
        HoldMem(
            world,
            log,
            CellConfig(1, 0, synth=True, ra=True),
            RecordingLLM(),
            "w",
            program_user_words=True,
        )


def test_reader_extra_body():
    assert reader_extra_body("claude-sonnet-5-5") == {"thinking": {"type": "between_tools"}}
    assert reader_extra_body("claude-sonnet-5-5") is not CLAUDE_READER_BODY["claude-sonnet-5-5"]
    assert reader_extra_body("Qwen/Qwen3.5-9B") is None
    with pytest.raises(ValueError, match="no reader settings"):
        reader_extra_body("claude-opus-5-5")


def test_post_probes_definition_answers_the_same_definition_items(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memshift.experiments.e1.client", lambda model, run_dir, workers=48, **kw: FakeLLM(model)
    )
    cells, _ = parse_cells("D0B0,D1sB1_source")
    common = {
        "cells_cfg": cells,
        "baselines": False,
        "branches": ("def",),
        "world_params": world_params(3),
        "discriminative": True,
    }
    out = {}
    for mode in ("all", "definition"):
        (tmp_path / mode).mkdir()
        d = run_e1([5], "pp", out=tmp_path / mode, post_probes=mode, **common)
        assert json.loads((d / "manifest.json").read_text())["params"]["post_probes"] == mode
        probes = {p["probe_id"]: p for p in read_jsonl(d / "probes.jsonl")}
        out[mode] = (
            {(a["system"], a["probe_id"]): a for a in read_jsonl(d / "answers.jsonl")},
            probes,
        )
    (full, probes), (defs, _) = out["all"], out["definition"]

    def definition_after_fork(pid: str) -> bool:
        p = probes[pid]
        return bool(p["tags"]["k"] == 0 or p["fact_id"].startswith("k."))

    assert set(defs) == {k for k in full if definition_after_fork(k[1])}
    assert any(not definition_after_fork(pid) for _, pid in full)  # something was dropped
    assert any(probes[pid]["tags"]["k"] > 0 for _, pid in defs)
    for k, a in defs.items():
        assert (a["value"], a["context_sha256"]) == (full[k]["value"], full[k]["context_sha256"])

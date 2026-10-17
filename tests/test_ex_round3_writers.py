"""The axes of the writer figure (`analysis.ex_round3_writers`, exploratory): the programs'
agreement with gold at the fork on hand-built items, and the staged result."""

import json
from pathlib import Path

import pytest

from memshift.analysis import ex_round3_writers as ew

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "results/sources.json").read_text())
STAGED = ROOT / SOURCES["ex_round3"] / "analysis" / "writers.json"
DM = "holdmem_D1sB1_source"


def probe(world: str, branch: str, fact: str, gold: list[str] | None, k: int = 0) -> dict:
    return {
        "world_id": world,
        "branch": branch,
        "fact_id": fact,
        "tags": {"k": k},
        "vtype": "set_person",
        "gold_status": "answer" if gold is not None else "abstain",
        "gold_value": gold,
    }


def item(world: str, branch: str, slot: str, notes: list, system: str = DM) -> dict:
    history = [
        {"event": ev, "ts": f"2026-10-0{i + 1}", **({"program": n} if n is not None else {})}
        for i, (ev, n) in enumerate(notes)
    ]
    return {
        "world_id": world,
        "branch": branch,
        "system": system,
        "slot_id": slot,
        "history": history,
    }


def test_program_t0_scores_the_first_derive_of_each_definition_slot():
    probes = [
        probe("w1", "def", "k.a", ["Ann", "Bo"]),
        probe("w1", "def", "k.b", ["Cy"]),
        probe("w1", "def", "k.c", None),
        probe("w1", "def", "k.a", ["Ann"], k=1),  # a later checkpoint: not the t0 gold
        probe("w2", "src", "k.a", ["Dee"]),
    ]
    items = [
        item("w1", "def", "k.a", [("derive", {"ok": True, "value": ["bo", "Ann"]}),
                                  ("repair", {"ok": True, "value": ["Ann"]})]),
        item("w1", "def", "k.b", [("derive", {"ok": False, "value": ["Cy"]})]),
        item("w1", "def", "k.c", [("invalidate", None), ("derive", {"ok": True, "value": None})]),
        item("w2", "src", "k.a", [("derive", {"ok": True, "value": ["Eve"]})]),
        item("w2", "src", "c.p01.city", [("derive", {"ok": True, "value": "Oslo"})]),
        item("w2", "src", "k.a", [("derive", {"ok": True, "value": ["Dee"]})],
             system="holdmem_D1sB0"),
    ]  # fmt: skip
    out = ew.program_t0(items, probes)
    assert out["agree"] == {
        "w1": {"def|k.a": 1.0, "def|k.b": 0.0, "def|k.c": 1.0},
        "w2": {"src|k.a": 0.0},
    }  # a set compares unordered; a program that did not run is wrong; abstain matches None
    assert out["exec"] == {
        "w1": {"def|k.a": 1.0, "def|k.b": 0.0, "def|k.c": 1.0},
        "w2": {"src|k.a": 1.0},
    }
    assert ew._branch(out["agree"], "src") == {"w2": {"src|k.a": 0.0}}
    assert ew._branch(out["agree"], None) == out["agree"]


def test_analyse_without_runs_is_empty():
    res = ew.analyse([], [], [])
    assert res["program_sets"] == {"sets": {}}
    assert res["definition_writers"] == {"writers": {}}
    assert res["status"].startswith("exploratory")


@pytest.mark.skipif(not STAGED.exists(), reason="writers.json is not staged")
def test_staged_result_names_its_inputs_and_covers_every_set_and_branch():
    res = json.loads(STAGED.read_text())
    assert "/Users/" not in STAGED.read_text()
    assert all(p.startswith("runs/") for p in res["inputs"]["programs"])
    for sid, node in res["program_sets"]["sets"].items():
        assert sid.split("#")[0] == node["writer"]
        for b in ("src", "def", "orphan", "all"):
            e = node["agreement_t0"][b]
            assert 0.0 <= e["est"] <= 1.0 and e["n_worlds"] == 20
            assert e["est"] <= node["executable_t0"][b]["est"] + 1e-12
    writers = res["definition_writers"]["writers"]
    assert set(writers) == {"claude-sonnet-5-5", "claude-opus-5-5"}

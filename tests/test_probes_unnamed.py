"""Probes that do not name the definition version (`--probe-unnamed-defs`)."""

import json
from datetime import timedelta

from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import parse_cells
from memshift.experiments.e1 import run as run_e1
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.probes import make_probes
from tests.fakes import FakeLLM

SENTENCE = " Definition in effect: "


def test_unnamed_probes_drop_only_the_definition_sentence():
    w = generate_branch(5, "def", world_params(3))
    log = SourceLog(w.events)
    times = [w.fork - timedelta(minutes=1), w.fork + timedelta(weeks=8)]  # both versions
    named = make_probes(w, log, times)
    assert [p.to_json() for p in named] == [
        p.to_json() for p in make_probes(w, log, times, unnamed_defs=False)
    ]  # the default is the named probe
    unnamed = make_probes(w, log, times, unnamed_defs=True)
    assert len(named) == len(unnamed)
    n_kappa = 0
    for a, b in zip(named, unnamed, strict=True):
        slot = w.slot(a.fact_id)
        ja, jb = a.to_json(), b.to_json()
        assert {k: v for k, v in ja.items() if k != "question_text"} == {
            k: v for k, v in jb.items() if k != "question_text"
        }
        if slot.kappa_id:
            n_kappa += 1
            head, _, stmt = a.question_text.partition(SENTENCE)
            assert stmt and b.question_text == head
            assert SENTENCE.strip() not in b.question_text
            assert b.question_text.endswith(slot.question)
        else:
            assert a.question_text == b.question_text
    assert n_kappa > 0


def test_the_driver_asks_unnamed_probes_in_every_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memshift.experiments.e1.client", lambda model, run_dir, workers=48, **kw: FakeLLM(model)
    )
    cells, views = parse_cells("D0B0,D1B1_source,D1sB0")
    out = run_e1(
        [5],
        "unnamed",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
        views=views,
        probe_unnamed_defs=True,
    )
    check = json.loads(next(out.glob("hash_check_*_def.json")).read_text())
    assert check["ok"] and check["probes"] > 0
    probes = read_jsonl(out / "probes.jsonl")
    kappa = [p for p in probes if p["fact_id"].startswith("k.")]
    assert kappa and not any(SENTENCE in p["question_text"] for p in kappa)
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["params"]["probe_unnamed_defs"] is True
    systems = {r["system"] for r in read_jsonl(out / "answers.jsonl")}
    assert systems == {c.name for c in cells} | {"holdmem_json_arm"}

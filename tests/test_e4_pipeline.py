import json

from memshift.analysis.e4 import analyse
from memshift.experiments import e4
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from tests.fakes import FakeLLM


def test_probe_plan_is_calendar_grounded():
    w = generate_branch(0, "natural")
    log = SourceLog(w.events)
    slots = e4.calendar_slots(w)
    probes, dropped = e4.probe_plan(w, log, slots)
    assert probes and dropped > 0
    assert {s.slot_id for s in slots} >= {"u.timezone", "k.important_contact"}
    assert len({p.probe_id for p in probes}) == len(probes)
    for p in probes:
        assert p.tags["delta"] in e4.DELTAS and "gold_changed" in p.tags
        assert all(oid.startswith("calendar:") for oid, _ in p.deps)
        assert w.fork < p.as_of


def test_e4_run_matches_expected_llm_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(e4, "MAX_EVENTS", 3)
    monkeypatch.setattr(e4, "LAGS", (30,))
    llm = FakeLLM()
    run_dir = e4.run([1], out=tmp_path, llm=llm)
    summary = json.loads((run_dir / "world_summaries.jsonl").read_text())
    # B=0 maintenance skips slots with an empty delta, so the formula is an upper bound
    assert summary["llm_calls"] == llm.usage["calls"] <= summary["llm_calls_expected"]
    answers = [json.loads(line) for line in (run_dir / "answers.jsonl").open()]
    assert {(a["tags"]["mode"], a["tags"]["lag"]) for a in answers} == {
        ("ideal", 0),
        ("official", 30),
    }
    assert {a["system"] for a in answers} == {"holdmem_D1B0", "holdmem_D1B1_source", "zero"}
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["params"]["lags_minutes"] == [30]
    metrics = analyse(run_dir)
    assert "interaction_B_all" in metrics["contrasts"]["L30"]
    assert metrics["divergence"]["ideal_L0"]["all"]["est"] == 0.0

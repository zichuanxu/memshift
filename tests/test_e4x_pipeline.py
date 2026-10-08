import json
from collections import Counter
from datetime import timedelta

from memshift.analysis.e4x import analyse
from memshift.calsync.sim import ALL_SOURCES
from memshift.experiments import e4x
from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from tests.fakes import FakeLLM

SHORT = {"months": 3, "fork_month": 2, "target_tokens": 300_000}
TAGS = {"stream", "mode", "lag", "delta", "system", "B", "source", "touched", "synced"}


def test_probe_plan_is_stratified_and_discriminative():
    w = generate_branch(1000, "src")
    log = SourceLog(w.events)
    probes, events = e4x.probe_plan(w, log)
    per_source = Counter(e.source for e in events)
    assert set(per_source) <= set(ALL_SOURCES) and "calendar" in per_source
    for src, n in per_source.items():
        cap = e4x.MAX_CALENDAR_EVENTS if src == "calendar" else e4x.MAX_EVENTS_PER_SOURCE
        assert n <= cap
    # every sampled event touches at least one slot
    assert all(e4x.touched_slots(w, log, e)[0] for e in events)
    window = log.between(w.fork, w.fork + timedelta(weeks=e4x.WINDOW_WEEKS))
    for src in ALL_SOURCES:
        cap = e4x.MAX_CALENDAR_EVENTS if src == "calendar" else e4x.MAX_EVENTS_PER_SOURCE
        # strata are exhausted in order: an eligible event of an earlier stratum is never
        # skipped while a later stratum is used
        eligible = [e for e in window if e.source == src and e4x.touched_slots(w, log, e)[0]]
        picked = [e for e in events if e.source == src]
        first = Counter(e4x._stratum(e) for e in eligible)
        got = Counter(e4x._stratum(e) for e in picked)
        for k in sorted(first):
            assert got[k] == min(first[k], cap - sum(got[j] for j in range(k)))
    assert len({(p.fact_id, p.as_of) for p in probes}) == len(probes)
    by_event = {e.event_id: e for e in events}
    touched = Counter(p.tags["event_id"] for p in probes if p.tags["touched"])
    control = Counter(p.tags["event_id"] for p in probes if not p.tags["touched"])
    assert touched and all(control[k] <= touched[k] for k in control)
    for p in probes:
        ev = by_event[p.tags["event_id"]]
        assert p.tags["delta"] in e4x.DELTAS and p.tags["source"] == ev.source
        assert p.as_of == ev.ts + timedelta(minutes=p.tags["delta"])
        slot = w.slot(p.fact_id)
        deps = {
            oid
            for t in (ev.ts - timedelta(seconds=1), ev.ts)
            for oid, _ in gold_at(w, log, slot, t).deps
        }
        assert (ev.object_id in deps) == p.tags["touched"]


def test_e4x_run_outputs_tags_and_ideal_divergence(tmp_path, monkeypatch):
    monkeypatch.setattr(e4x, "MAX_EVENTS_PER_SOURCE", 1)
    llm = FakeLLM()
    run_dir = e4x.run([2], out=tmp_path, llm=llm, world_params=SHORT)
    for name in ("probes", "answers", "divergence", "polls", "world_summaries"):
        assert (run_dir / f"{name}.jsonl").stat().st_size > 0
    assert (run_dir / "manifest.json").exists() and (run_dir / "env.json").exists()
    summary = json.loads((run_dir / "world_summaries.jsonl").read_text())
    # B=0 skips empty deltas and B=1 unchanged keys, so the formula is an upper bound
    assert summary["llm_calls"] == llm.usage["calls"] <= summary["llm_calls_expected"]
    assert set(summary["streams"]["official_L5"]["sources"]) == set(ALL_SOURCES)

    answers = [json.loads(line) for line in (run_dir / "answers.jsonl").open()]
    assert {a["tags"]["stream"] for a in answers} == {"ideal_L0", "official_L5"}
    assert {a["system"] for a in answers} == {"holdmem_D1B0", "holdmem_D1B1_source", "zero"}
    assert all(a["tags"].keys() >= TAGS for a in answers)
    assert {a["tags"]["touched"] for a in answers} == {True, False}
    assert all(a["tags"]["synced"] for a in answers if a["tags"]["mode"] == "ideal")
    assert len(answers) == 6 * summary["n_probes"]

    div = [json.loads(line) for line in (run_dir / "divergence.jsonl").open()]
    assert {r["source"] for r in div} == set(ALL_SOURCES)
    assert all(r["divergence"] == 0.0 for r in div if r["mode"] == "ideal")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["params"]["lag_minutes"] == e4x.LAG
    assert manifest["params"]["profiles"]["messages"]["incremental"] == "created"

    metrics = analyse(run_dir)
    assert (run_dir / "analysis" / "e4x_metrics.json").exists()
    contrasts = metrics["contrasts"]["L5"]
    assert "interaction_B" in contrasts["all"]["pooled"]
    assert "interaction_B" in contrasts["touched"]["pooled"]
    assert metrics["divergence"]["ideal_L0"]["pooled"]["est"] == 0.0


def test_e4x_analysis_interaction_is_per_source_and_paired(tmp_path):
    """Calendar: official hurts B=0 only (interaction +1); email: no drop (interaction 0)."""
    systems = (("holdmem_D1B0", 0), ("holdmem_D1B1_source", 1), ("zero", None))
    rows, div = [], []
    for w in ("w1", "w2"):
        for src in ("calendar", "email"):
            for mode, lag in (("ideal", 0), ("official", 5)):
                stream = f"{mode}_L{lag}"
                base = {"stream": stream, "mode": mode, "lag": lag, "source": src}
                for k in range(2):
                    for system, b in systems:
                        rows.append(
                            {
                                "world_id": w,
                                "probe_id": f"{w}.{src}.{k}",
                                "system": system,
                                "contract_valid": not (
                                    src == "calendar" and mode == "official" and b == 0
                                ),
                                "tags": {**base, "B": b, "touched": k == 0},
                            }
                        )
                div.append({"world_id": w, **base, "divergence": float(mode == "official")})
    for name, data in (("answers", rows), ("divergence", div)):
        (tmp_path / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in data))
    m = analyse(tmp_path)
    c = m["contrasts"]["L5"]
    assert c["all"]["calendar"]["interaction_B"]["est"] == 1.0
    assert c["all"]["calendar"]["drop_holdmem_D1B0"]["est"] == -1.0
    assert c["all"]["email"]["interaction_B"]["est"] == 0.0
    assert c["all"]["pooled"]["interaction_B"]["est"] == 0.5
    assert c["touched"]["calendar"]["interaction_B"]["n_items"] == 2
    assert m["primary"]["L5"] == c["all"]["calendar"]["interaction_B"]
    assert m["cells"]["official_L5:holdmem_D1B0"]["all"]["calendar"]["est"] == 0.0
    assert m["divergence"]["official_L5"]["pooled"]["est"] == 1.0
    assert m["n_probes_by_source"]["calendar"] == {"all": 4, "touched": 2}

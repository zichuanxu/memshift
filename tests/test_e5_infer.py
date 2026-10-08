import math

from memshift.analysis.e5_infer import cohen_kappa, label, probe_samples


def test_cohen_kappa():
    assert cohen_kappa([(0, 0), (1, 1), (0, 0), (1, 1)]) == 1.0
    # one D=1 cell labelled 0 among four cells: po = 0.75, pe = 0.5
    assert cohen_kappa([(0, 0), (1, 0), (0, 0), (1, 1)]) == 0.5
    assert math.isnan(cohen_kappa([(1, 1), (1, 1)]))


def test_label_thresholds_and_identifiability():
    persist = {"w0": [1.0] * 5 + [0.0] * 20}
    assert label(persist, "B", structural=False)["label"] == 1  # P = 0.2 <= 0.5
    assert label({"w0": [1.0] * 25}, "B", structural=False)["label"] == 0
    exact = {"w0": [1.0] * 25}
    assert label(exact, "D", structural=True)["label"] == 1
    assert label(exact, "D", structural=False)["label"] == 0  # no typed value + spans export
    assert label({"w0": [1.0] * 5}, "D", structural=True)["label"] is None  # n < 20
    assert label(exact, "B", structural=False, identifiable=False)["label"] is None


def _probe(branch, fact, k, status, value):
    return {
        "world_id": "w0",
        "branch": branch,
        "fact_id": fact,
        "gold_status": status,
        "gold_value": value,
        "tags": {"k": k},
    }


def _answer(branch, fact, k, status, value, gold_status, gold_value, cv):
    return {
        "system": "s",
        "world_id": "w0",
        "branch": branch,
        "fact_id": fact,
        "vtype": "org",
        "status": status,
        "value": value,
        "gold_status": gold_status,
        "gold_value": gold_value,
        "contract_valid": cv,
        "tags": {"k": k},
    }


def test_probe_samples_orphan_persistence_and_repair():
    fact = "c.p11.employer"  # structured and in the inference half
    probes = [
        _probe("orphan", fact, 0, "answer", "Acme"),
        _probe("src", fact, 0, "answer", "Acme"),
    ]
    answers = [
        # orphaned item still answered with the old value -> persisted
        _answer("orphan", fact, 1, "answer", "Acme", "abstain", None, False),
        # new value after a source revision, correct -> counts toward R
        _answer("src", fact, 1, "answer", "Borealis", "answer", "Borealis", True),
        # old value kept after the revision -> not a repair, excluded from R
        _answer("src", fact, 2, "answer", "Acme", "answer", "Borealis", False),
    ]
    s = probe_samples({"probes": probes, "answers": answers, "items": []})["s"]
    assert s["B"]["w0"] == [1.0]
    assert s["D"]["w0"] == [1.0]


def test_carry_uses_measurement_half_only():
    from memshift.analysis.e5 import _carry
    from memshift.gen.shifts import inference_half

    meas = next(f for f in ("c.p01.city", "c.p02.city", "c.p03.city") if not inference_half(f))
    infer = next(f for f in ("c.p01.city", "c.p02.city", "c.p03.city") if inference_half(f))

    def row(fact, changed, cv):
        return {
            "system": "mem0",
            "branch": "src",
            "world_id": "w0",
            "fact_id": fact,
            "changed": changed,
            "cv": cv,
        }

    rows = [row(meas, True, True), row(meas, False, False), row(infer, True, False)]
    c = _carry(rows, "mem0", "src", {"w0"})
    assert c["Update"]["est"] == 1.0 and c["Update"]["n_items"] == 1
    assert c["Stability"]["est"] == 0.0


def test_world_params_short_and_default():
    from memshift.experiments.common import world_params

    assert world_params(12) is None
    assert world_params(3) == {"months": 3, "fork_month": 2, "target_tokens": 500_000}


def test_e5_cost_counts_in_flight_unit_when_run_stopped(tmp_path):
    import json

    from memshift.analysis.e5_cost import analyse

    store = tmp_path / "run" / "stores" / "w1_graphiti" / "prefix"
    store.mkdir(parents=True)
    (store / "memshift_native.json").write_text(json.dumps({"system": "graphiti"}))
    (store / "ledger.jsonl").write_text("{}\n" * 3)
    usage = {"llm_calls": 24, "prompt_tokens": 400, "completion_tokens": 40}
    (store / "state.json").write_text(json.dumps({"usage": usage, "cost": {}}))
    g = analyse(tmp_path / "run", out=tmp_path / "out")["by_system"]["graphiti"]
    assert g["attempted"] == 4 and not g["finished"]  # no world_results.jsonl: one unit in flight
    assert g["t_i"]["llm_calls"] == 6.0 and g["t_i"]["embed_calls"] == 0.0


def test_noise_floor_pairs_shared_worlds_and_failure_rate():
    from memshift.analysis.e5 import failures, noise_floor
    from memshift.gen.shifts import inference_half

    meas = next(f for f in ("c.p01.city", "c.p02.city", "c.p03.city") if not inference_half(f))
    probes = [
        {**_probe("src", meas, k, "answer", v), "vtype": "city"}
        for k, v in ((0, "Oslo"), (1, "Rome"))
    ]
    first = [_answer("src", meas, 1, "answer", "Rome", "answer", "Rome", True)]
    only_once = {**first[0], "world_id": "w9"}  # ingested once: not part of the noise floor
    second = [_answer("src", meas, 1, "answer", "Oslo", "answer", "Rome", False)]
    nf = noise_floor([*first, only_once], second, probes)
    s = nf["systems"]["s"]["src"]
    assert nf["worlds"] == ["w0"]
    assert s["answers"] == 1 and s["share_differ"] == 1.0
    assert (s["cv_first"], s["cv_second"]) == (1.0, 0.0)
    assert s["diff"] == {"Update": -1.0}  # no unchanged item on the measurement half

    results = [
        {
            "system": "mem0",
            "prefix_stats": {"ingested": 8, "failed": 1},
            "branches": {"src": {"stats": {"ingested": 1, "failed": 0}}},
        }
    ]
    f = failures(results)["mem0"]
    assert (f["ingested"], f["failed"], f["rate"]) == (9, 1, 0.1)


def test_e5_cost_pools_stores_of_one_system(tmp_path):
    import json

    from memshift.analysis.e5_cost import analyse

    run = tmp_path / "run"
    for world, calls, units in (("w1", 10, 5), ("w2", 30, 15)):
        store = run / "stores" / f"{world}_mem0" / "prefix"
        store.mkdir(parents=True)
        (store / "memshift_native.json").write_text(json.dumps({"system": "mem0"}))
        (store / "ledger.jsonl").write_text("{}\n" * units)
        (store / "state.json").write_text(json.dumps({"usage": {"llm_calls": calls}}))
    (run / "world_results.jsonl").write_text("{}\n")
    m = analyse(run, out=tmp_path / "out")["by_system"]["mem0"]
    assert (m["stores"], m["attempted"], m["finished"]) == (2, 20, True)
    assert m["t_i"]["llm_calls"] == 2.0  # (10 + 30) / (5 + 15)


def test_probe_samples_native_runs_probe_t0_on_the_natural_world():
    fact = "c.p11.employer"
    probes = [_probe("natural", fact, 0, "answer", "Acme")]  # no per-branch t0 probe
    answers = [
        _answer("orphan", fact, 1, "answer", "Acme", "abstain", None, False),
        _answer("src", fact, 1, "answer", "Borealis", "answer", "Borealis", True),
    ]
    s = probe_samples({"probes": probes, "answers": answers, "items": []})["s"]
    assert s["B"]["w0"] == [1.0] and s["D"]["w0"] == [1.0]


def test_e5_macros_labels_rates_failures_and_noise():
    from memshift.analysis.paper_numbers import MISSING, _macro, e5_numbers

    est = {"est": 0.5, "lo": 0.4, "hi": 0.6, "n_items": 10}
    m = {
        "labels": {"mem0": {"B": {"label": 0, "rate": 0.9, "n": 30}, "D": {"label": 0}}},
        "failures": {"mem0": {"ingested": 99, "failed": 1, "rate": 0.01}},
        "systems": {
            "mem0": {
                "worlds": ["w1", "w2"],
                "branches": {
                    "src": {
                        "native": {"Update": est},
                        "white_box": {"Update": est},
                        "consistent": {"Update": True},
                        "deviation": {"Update": -0.05},
                    }
                },
            }
        },
        "noise_floor": {
            "worlds": ["w1"],
            "systems": {"mem0": {"src": {"share_differ": 0.2, "diff": {"Update": 0.01}}}},
        },
    }
    out = e5_numbers(m)
    assert (out["EFiveMemB"], out["EFiveMemBRate"], out["EFiveMemBN"]) == ("0", "90.0\\%", "30")
    assert out["EFiveMemUpdateConsistentSrc"] == "inside"
    assert out["EFiveMemUpdateDevSrc"] == "$-$5.0\\,pp"
    assert (out["EFiveMemFailRate"], out["EFiveMemUnits"]) == ("1.00\\%", "100")
    assert (out["EFiveMemNoiseDifferSrc"], out["EFiveMemNoiseWorlds"]) == ("20.0\\%", "1")
    assert out["EFiveGraphitiB"] == MISSING
    for k, v in out.items():
        _macro(k, v)  # letters-only names


def test_membership_t3_against_oracle_and_t4_ratio(monkeypatch):
    from memshift.analysis import e5

    monkeypatch.setattr(
        e5, "full_reread_tokens", lambda worlds, params: dict.fromkeys(worlds, 1000)
    )

    def ans(system, branch, fact, cv, tokens=100):
        return {
            "system": system,
            "world_id": "w0",
            "branch": branch,
            "fact_id": fact,
            "contract_valid": cv,
            "prompt_tokens": tokens,
            "completion_tokens": 0,
            "tags": {"k": 0},
        }

    natives = [ans("mem0", "natural", "a", True), ans("mem0", "natural", "b", False)]
    base = [ans("oracle", "src", "a", True, 900), ans("oracle", "src", "b", True, 900)]
    base.append({**ans("oracle", "src", "a", False), "tags": {"k": 1}})  # post-shift: ignored
    m = e5.membership(natives, base, None)
    assert m["mem0"]["minus_oracle"]["est"] == -0.5 and m["mem0"]["T3_pass"] is False
    assert m["mem0"]["serve_ratio_to_full_reread"] == 0.1 and m["mem0"]["T4_pass"] is True
    assert m["oracle"]["T4_pass"] is False and "T3_pass" not in m["oracle"]
    assert e5.membership(natives, [], None) == {}  # no oracle: T3 not determinable


def test_silent_failures_counted_from_stderr(tmp_path):
    import json

    from memshift.analysis.e5 import PARSE_ERROR, silent_failures

    run = tmp_path / "run"
    run.mkdir()
    results = [{"system": "mem0", "prefix_stats": {"ingested": 10, "failed": 0}}]
    (run / "world_results.jsonl").write_text(json.dumps(results[0]) + "\n")
    (run / "stderr.txt").write_text(f"{PARSE_ERROR}: Expecting ','\nother line\n{PARSE_ERROR}: x\n")
    s = silent_failures([run, tmp_path / "no_log"], results)["mem0"]
    assert (s["runs"], s["parse_errors"], s["rate"]) == (1, 2, 0.2)


def test_stage4_predicted_cell_and_summary():
    from memshift.analysis.e5 import predicted_cell, stage4_summary

    assert predicted_cell("lightrag", 0, 1) == "holdmem_D0B1_cited"
    assert predicted_cell("amem", 0, 1) == "holdmem_D0B1_source"
    assert predicted_cell("mem0", 0, 0) == "holdmem_D0B0"
    assert predicted_cell("hipporag", 1, 1) == "holdmem_D1B1_cited"  # the inferred D is followed
    assert predicted_cell("amem", 1, 0) == "holdmem_D1B0"
    assert predicted_cell("mem0", 0, None) is None
    assert predicted_cell("mem0", None, 0) is None
    from memshift.analysis.e5 import _inferred

    assert _inferred({"D": {"label": None}, "B": {"label": 1}}) == (0, 1)  # D=0 by structure
    assert _inferred({"B": {"label": None}}) == (0, None)

    def branch(inside: list[bool], stab: float) -> dict:
        return {
            "native": {"Update": {"est": 0.5}, "Stability": {"est": stab}},
            "consistent": dict(zip(("Update", "Stability"), inside, strict=True)),
        }

    res = {
        "labels": {
            "mem0": {"B": {"label": 0}},
            "lightrag": {"B": {"label": 1}},
        },
        "systems": {
            "mem0": {
                "branches": {
                    "src": branch([True, True], 0.7),
                    "def": branch([True, False], 0.7),
                    "orphan": branch([True, True], 0.4),
                }
            },
            "lightrag": {
                "branches": {
                    "src": branch([False, False], 0.9),
                    "def": branch([True, False], 0.9),
                    "orphan": branch([True, True], 0.8),
                }
            },
        },
    }
    s = stage4_summary(res)
    assert s["n1_label_agreement"] == {"lightrag": True}  # N-1 counts stage-4 systems only
    assert s["n1_holds"] is False  # 1 of the needed 4
    assert s["n2_per_system"]["mem0"] == {"inside": 5, "comparisons": 6, "n2_holds": True}
    assert s["n2_per_system"]["lightrag"]["n2_holds"] is False
    assert s["n3_share_inside"] == 8 / 12 and s["n3_holds"]
    assert s["n4_holds"] is True

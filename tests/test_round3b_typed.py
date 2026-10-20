"""`analysis.round3b_typed` (block T) on a synthetic fixture: a fake-writer E1 run of the three
cells on two small worlds, whose ContractValid values are then set to known patterns, so that
every r, Update and contrast has a known value."""

import json
import shutil
import sqlite3
from dataclasses import replace
from datetime import timedelta

import pytest

from memshift.analysis import round3b_typed as rt
from memshift.experiments.common import MAIN_MODEL, read_jsonl, world_params, write_jsonl
from memshift.experiments.e1 import parse_cells
from memshift.experiments.e1 import run as run_e1
from memshift.gen.kappa import scope_members
from memshift.gen.model import parse_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.program import program_request
from memshift.holdmem.writer import records_for
from tests.fakes import FakeLLM

SEEDS = [5, 6]
CV = {rt.TYPED: 1.0, rt.BM25: 0.0, rt.D1: 1.0}  # on the changed definition item-times


@pytest.fixture(scope="module")
def fixture(tmp_path_factory):
    root = tmp_path_factory.mktemp("t")
    (root / "new").mkdir()
    mp = pytest.MonkeyPatch()
    mp.setattr("memshift.experiments.e1.client", lambda model, run_dir, **kw: FakeLLM(model))
    try:
        cells, _ = parse_cells("D0B1_source,D1B1_source,D0typedB1_source")
        out = run_e1(
            SEEDS,
            "e1_t_fake",
            out=root / "new",
            cells_cfg=cells,
            baselines=False,
            branches=("src", "def"),
            world_params=world_params(3),
            discriminative=True,
            parallel_worlds=1,
        )
    finally:
        mp.undo()
    committed = root / "committed"
    shutil.copytree(out, committed)
    # known values: on changed definition item-times the typed and D=1 cells are right, BM25 wrong
    probes = {p["probe_id"]: p for p in read_jsonl(out / "probes.jsonl")}
    rows = read_jsonl(out / "answers.jsonl")
    for a in rows:
        if a["system"] in CV and a["fact_id"].startswith("k."):
            a["contract_valid"] = CV[a["system"]]
    (out / "answers.jsonl").unlink()
    write_jsonl(out / "answers.jsonl", rows)
    # the committed copy has the anchors' (changed) answers too, so the pairing holds; like the
    # H-a runs, it also had D1s cells with another program writer, which the anchors never read
    (committed / "answers.jsonl").unlink()
    write_jsonl(committed / "answers.jsonl", [a for a in rows if a["system"] in rt.ANCHORS])
    man = json.loads((committed / "manifest.json").read_text())
    man["params"]["program_writer"] = {"model": "claude-sonnet-5-5", "temperature": 0.0}
    (committed / "manifest.json").write_text(json.dumps(man))
    return root, out, committed, probes


def complete_all(out, value: bool):
    rows = read_jsonl(out / "probes.jsonl")
    return {
        (p["world_id"], p["branch"], p["fact_id"].removeprefix("k."), p["tags"]["k"]): value
        for p in rows
        if p["fact_id"].startswith("k.") and p["tags"]["k"]
    }


def test_levels_and_contrasts_have_the_known_values(fixture):
    _, out, committed, _ = fixture
    res = rt.analyse([out], [committed], complete=complete_all(out, True), draws=200)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    assert res["pairing"]["anchor_answers_compared"] > 0
    (mine,), _ = rt.rr.read_runs([out])
    (theirs,), _ = rt.rr.read_runs([committed])
    assert mine.variant != theirs.variant and rt.anchor_group(mine) == rt.anchor_group(theirs)
    (arm,) = res["arms"].values()
    assert arm["writer"] == MAIN_MODEL and arm["defwriter"] is None
    assert arm["worlds"] == ["w0005", "w0006"]
    for b in rt.BRANCHES:
        for q in ("r", "update"):
            node = arm[q][b]["definition"]["all"]
            if not node["typed"]["n_items"]:
                continue
            assert node["typed"]["est"] == 1.0 and node["bm25"]["est"] == 0.0
            assert node["d1"]["est"] == 1.0
            assert node["typed_minus_bm25"]["est"] == 1.0 and node["d1_minus_typed"]["est"] == 0.0
            assert node["d1_minus_bm25"]["est"] == 1.0
            # the three cells answer the same item-times
            assert node["typed"]["n_items"] == node["bm25"]["n_items"] == node["d1"]["n_items"]
        # every subset: all complete, so "incomplete" is empty
        inc = arm["r"][b]["definition"]["incomplete"]["typed"]
        assert inc["n_items"] == 0
    pooled = arm["update"]["pooled"]["definition"]["all"]["typed"]["n_items"]
    parts = sum(arm["update"][b]["definition"]["all"]["typed"]["n_items"] for b in ("src", "def"))
    assert pooled == parts > 0
    per_def = sum(arm["update"]["pooled"][g]["all"]["typed"]["n_items"] for g in rt.DEFINITIONS)
    assert per_def == pooled
    assert arm["validity"]["typed_writes"]["writes"] > 0
    assert arm["validity"]["typed_writes"]["unreadable"] == 0
    t0 = arm["t0"]["definition"]
    assert t0["n"] == 6 * len(SEEDS) and t0["readable"]["est"] == 1.0
    assert res["coverage"]["k.close_friend"]["item_times"] > 0
    assert rt.summary(res).startswith("block T: pairing ok")


def test_the_split_follows_the_coverage_map(fixture):
    _, out, _, _ = fixture
    res = rt.analyse([out], complete=complete_all(out, False), draws=200)
    (arm,) = res["arms"].values()
    node = arm["r"]["pooled"]["definition"]
    assert node["complete"]["typed"]["n_items"] == 0
    assert node["incomplete"]["typed"]["n_items"] == node["all"]["typed"]["n_items"] > 0
    assert not res["pairing"]["detail"]["runs"][out.name]  # no committed runs: hash checks only


def test_a_changed_anchor_answer_breaks_the_pairing(fixture, tmp_path):
    _, out, committed, _ = fixture
    bad = tmp_path / committed.name
    shutil.copytree(committed, bad)
    rows = read_jsonl(bad / "answers.jsonl")
    i = next(i for i, a in enumerate(rows) if a["system"] == rt.BM25)
    rows[i]["contract_valid"] = 1.0 - float(rows[i]["contract_valid"] or 0.0)
    (bad / "answers.jsonl").unlink()
    write_jsonl(bad / "answers.jsonl", rows)
    res = rt.analyse([out], [bad], complete=complete_all(out, True), draws=50)
    assert not res["pairing"]["ok"] and res["pairing"]["counts"]["anchor_answers_differ"] == 1


def test_the_coverage_split_is_the_stage_8_one(fixture):
    """`scope_complete` on the runs' worlds: complete iff the BM25 selection holds the scope."""
    _, out, _, _ = fixture
    runs, _ = rt.rr.read_runs([out])
    complete = rt.scope_complete(runs)
    man = json.loads((out / "manifest.json").read_text())
    probe_times = {
        (p["world_id"], p["tags"]["k"]): p["as_of"] for p in read_jsonl(out / "probes.jsonl")
    }
    assert {(w, b) for w, b, _, _ in complete} >= {
        (w["world_id"], w["branch"]) for w in man["worlds"]
    }
    world = generate_branch(5, "def", world_params(3))
    log = SourceLog(world.events)
    seen = set()
    for slot in (s for s in world.slots if s.kappa_id):
        for k in (1, 8):
            t = parse_ts(probe_times["w0005", k])
            assert t == world.fork + timedelta(weeks=k)
            ver = world.kappa_version(slot.kappa_id, t)
            scope = {oid for oid, _ in scope_members(slot.kappa_id, log.snapshot(t), t, ver)}
            sel = {d.doc_id for d, _ in records_for(log, slot, t)}
            assert complete["w0005", "def", slot.kappa_id, k] == (scope <= sel)
            seen.add(scope <= sel)
    assert seen == {True, False}


def test_the_cpu_facts_on_a_small_world(tmp_path):
    """Gold is invariant under the omitted fields; a committed program that reads a message body
    is caught, one that does not agrees; the request count follows the source-epoch key."""
    world = generate_branch(5, "src", world_params(3))
    cache = tmp_path / "qwen.sqlite"
    db = sqlite3.connect(cache)
    db.execute("CREATE TABLE cache (key TEXT PRIMARY KEY, model TEXT, response TEXT)")
    programs = {
        "close_friend": "def derive(records, as_of):\n    return [r['body'] for r in records][:1]",
        "overdue_task": "def derive(records, as_of):\n    return [r['title'] for r in records][:1]",
    }
    for slot in (s for s in world.slots if s.kappa_id in programs):
        fixed = replace(world, kappa_log=[(world.start, slot.kappa_id, "v1")])
        key = rt.program_key(MAIN_MODEL, program_request(fixed, slot, world.start))
        text = json.dumps({"code": programs[str(slot.kappa_id)]})
        db.execute(
            "INSERT INTO cache VALUES (?,?,?)", (key, MAIN_MODEL, json.dumps({"text": text}))
        )
    db.commit()
    db.close()
    res = rt.facts([5], program_cache=cache, months=3)
    gold = res["a_gold_invariant"]
    assert gold["identical"] and gold["evaluations"] == 2 * gold["item_times"] == 2 * 2 * 5 * 6
    progs = res["a_programs"]["by_definition_version"]
    assert progs["k.close_friend v1"]["differ"] == 10  # reads a dropped body: caught every time
    assert progs["k.overdue_task v1"]["same"] == 10 and "differ" not in progs["k.overdue_task v1"]
    assert progs["k.vip_sender v2"] == {"no_program": 10}
    req = res["b_requests"]
    assert req["t0_diagnostic"] == 12
    assert req["rederivations"] == sum(req["per_definition_branch"].values())
    assert 0 < req["rederivations"] <= 2 * 6 * 4 and "c_prompt_tokens" not in res

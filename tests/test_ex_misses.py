"""Why the cited-cluster key misses (`analysis.ex_misses`): the classes are checked on a generated
world against what each kind of change does to the gold dependency set, the stage driver runs end
to end on a run directory written from that world, and the committed analysis holds the shares the
paper reads."""

import json
from datetime import timedelta
from pathlib import Path

import pytest

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_misses as mm
from memshift.experiments.e1 import OFFSETS_WEEKS
from memshift.gen.gold import gold_at
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.runinfo import world_record

ROOT = Path(__file__).resolve().parents[1]
STAGE = ROOT / "runs/stage5_ex_misses/analysis/ex_misses.json"
EXCLUSIVE = ("a", "b", "c")


def _changed(world, log, slot, k):
    t0 = world.fork - timedelta(minutes=1)
    g0, gk = (
        gold_at(world, log, slot, t0),
        gold_at(world, log, slot, world.fork + timedelta(weeks=k)),
    )
    return (g0.value, g0.status) != (gk.value, gk.status)


@pytest.fixture(scope="module")
def src():
    world = generate_branch(1000, "src", None)
    return world, SourceLog(world.events)


@pytest.fixture(scope="module")
def clock():
    world = generate_branch(1002, "src", None)  # its next appointment passes within the horizon
    return world, SourceLog(world.events)


@pytest.fixture(scope="module")
def orphan():
    world = generate_branch(1000, "orphan", None)
    return world, SourceLog(world.events)


def _flags(world, log, cited_of=lambda slot, t0: set()):
    """classify() for every changed (slot, k) of a world."""
    t0 = world.fork - timedelta(minutes=1)
    out = {}
    for slot in world.slots:
        cited = cited_of(slot, t0)
        for k in OFFSETS_WEEKS:
            if _changed(world, log, slot, k):
                out[slot.slot_id, k] = mm.classify(world, log, slot, k, cited)
    return out


def test_classes_partition_the_changes(src, orphan):
    for world, log in (src, orphan):
        flags = _flags(world, log)
        assert flags
        for f in flags.values():
            assert sum(f[c] for c in EXCLUSIVE) == 1
            assert f["b"] == (f["b_existing"] or f["b_new"]) and not (
                f["b_existing"] and f["b_new"]
            )
            assert f["c"] == (f["c_definition"] or f["c_left_scope"] or f["c_other"])
            assert f["c_definition"] + f["c_left_scope"] + f["c_other"] == f["c"]
            assert not f["a_and_b"] or f["a"]
            assert not f["cited_touched"]  # nothing is cited here


def test_a_revised_record_nobody_cited_is_class_a_and_a_cited_one_is_not(src):
    world, log = src
    t0 = world.fork - timedelta(minutes=1)
    deps = {s.slot_id: dict(gold_at(world, log, s, t0).deps) for s in world.slots}
    uncited = _flags(world, log)
    everything = _flags(world, log, lambda slot, _t: set(deps[slot.slot_id]))
    cards = [key for key, f in uncited.items() if f["a"]]
    assert cards, "the source shift revises records the derivation depended on"
    for key in cards:
        assert not everything[key]["a"] and everything[key]["cited_touched"]
    # citing every dependency of t0 removes class a altogether, and only class a
    assert not any(f["a"] for f in everything.values())
    for key, f in uncited.items():
        if not f["a"]:
            assert f["b"] == everything[key]["b"] and f["c"] == everything[key]["c"]


def test_a_record_inserted_after_the_derivation_is_class_b_new(src):
    world, log = src
    flags = _flags(world, log)
    attrs = {k for k in flags if k[0].startswith("u.") and flags[k]["b_new"]}
    assert attrs, "a user attribute that changes is backed by a record created after t0"
    for slot_id, k in attrs:
        assert not flags[slot_id, k]["a"] and not flags[slot_id, k]["c"]


def test_time_passing_moves_the_window_without_any_record_changing(clock):
    world, log = clock
    flags = _flags(world, log)
    clock = [k for k, f in flags.items() if f["c_left_scope"]]
    assert clock, "records leave a time window although none changed"
    kinds = {world.slot(s).kind for s, _ in clock}
    assert kinds <= {"appointment", "kappa"}
    for key in clock:
        assert not flags[key]["a"] and not flags[key]["b"] and not flags[key]["version_changed"]


def test_a_definition_version_change_is_flagged():
    world = generate_branch(1000, "def", None)
    log = SourceLog(world.events)
    flags = _flags(world, log)
    moved = [k for k, f in flags.items() if f["version_changed"]]
    assert moved and all(world.slot(s).kappa_id for s, _ in moved)
    # the key carries the definition version, so these are not missed in a run: here they show
    # the flag only, and the record-level class is whatever the records did
    assert all(sum(flags[k][c] for c in EXCLUSIVE) == 1 for k in moved)


def test_a_deleted_record_is_class_a_unless_cited(orphan):
    world, log = orphan
    t0 = world.fork - timedelta(minutes=1)
    flags = _flags(world, log)
    deleted = [
        (s.slot_id, k)
        for s in world.slots
        for k in OFFSETS_WEEKS
        if (g := gold_at(world, log, s, world.fork + timedelta(weeks=k))).status == "abstain"
        and gold_at(world, log, s, t0).status == "answer"
        and g.negative
    ]
    assert deleted, "the orphan branch deletes records"
    assert all(flags[key]["a"] for key in deleted)


# ------------------------------------------------------------------------------ stage driver
def test_citations_come_from_the_t0_items_of_the_writer_cell(tmp_path):
    lines = [
        {"system": mm.WRITER_CELL, "item_id": "u.a#0", "world_id": "w1", "branch": "src",
         "slot_id": "u.a", "refs": [["email:e1", 1], ["email:e2", 1]]},
        {"system": mm.WRITER_CELL, "item_id": "u.a#1", "world_id": "w1", "branch": "src",
         "slot_id": "u.a", "refs": [["email:e9", 1]]},
        {"system": "holdmem_D0B1_cited", "item_id": "u.a#0", "world_id": "w1", "branch": "src",
         "slot_id": "u.a", "refs": [["email:e7", 1]]},
    ]  # fmt: skip
    (tmp_path / "items.jsonl").write_text("\n".join(json.dumps(r) for r in lines) + "\n")
    assert mm.citations(tmp_path) == {("w1", "src", "u.a"): {"email:e1", "email:e2"}}


def _stage(tmp_path, seed=1000, branches=("src", "def", "orphan")):
    """A run directory and a stage written from generated worlds: the cited key misses every
    changed item-time of the odd slots, the other slots are detected."""
    run, stage = tmp_path / "run", tmp_path / "stage"
    run.mkdir()
    (stage / "analysis").mkdir(parents=True)
    worlds, items, rows = [], [], []
    for b in branches:
        world = generate_branch(seed, b, None)
        log = SourceLog(world.events)
        worlds.append(world_record(world))
        t0 = world.fork - timedelta(minutes=1)
        for i, slot in enumerate(world.slots):
            cited = [[oid, rev] for oid, rev in gold_at(world, log, slot, t0).deps[:2]]
            items.append({"system": mm.WRITER_CELL, "item_id": f"{slot.slot_id}#0",
                          "world_id": f"w{seed}", "branch": b, "slot_id": slot.slot_id,
                          "refs": cited})  # fmt: skip
            for k in OFFSETS_WEEKS:
                changed = _changed(world, log, slot, k)
                for system in (mm.CITED, mm.CITED_D0):
                    rows.append({"world_id": f"w{seed}", "branch": b, "system": system,
                                 "fact_id": slot.slot_id, "vtype": "city", "k": k,
                                 "changed": changed, "cv": True, "detected": i % 2 == 0,
                                 "repaired_now": False})  # fmt: skip
    (run / "items.jsonl").write_text("\n".join(json.dumps(r) for r in items) + "\n")
    (run / "manifest.json").write_text(json.dumps({"worlds": worlds}))
    (stage / "analysis" / "e1_item_table.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n"
    )
    (stage / "manifest.json").write_text(json.dumps({"sources": [{"run_id": "run"}]}))
    return stage, tmp_path


def test_stage_driver_end_to_end(tmp_path):
    stage, runs = _stage(tmp_path)
    assert mm.stage_runs(stage, runs) == [runs / "run"]
    table = ex.Table(
        json.loads(line)
        for line in (stage / "analysis" / "e1_item_table.jsonl").read_text().splitlines()
    )
    res = mm.classify_stage(table, mm.stage_runs(stage, runs), draws=20)
    chk = res["checks"]
    assert chk["world_branches"] == 3 and chk["events_hash_matches"] == 3
    assert chk["missed"] > 0 and chk["missed_same_at_D0"] == chk["missed"]
    total = 0
    for b in ("src", "def", "orphan"):
        for cls in ex.CLASS_SETS:
            n = res[b][cls]
            total += n["missed_item_times"] if cls == "all" else 0
            if n["missed_item_times"]:
                # the three classes add up, the sub-classes add up to their class
                assert sum(n[f"{c}_count"] for c in EXCLUSIVE) == n["missed_item_times"]
                assert n["b_existing_count"] + n["b_new_count"] == n["b_count"]
                assert (
                    n["c_definition_count"] + n["c_left_scope_count"] + n["c_other_count"]
                    == (n["c_count"])
                )
                assert n["a"]["est"] + n["b"]["est"] + n["c"]["est"] == pytest.approx(1)
        parts = sum(res[b][c]["missed_item_times"] for c in ("definition", "rule", "norule"))
        assert parts == res[b]["all"]["missed_item_times"]
    assert total == chk["missed"]


def test_a_changed_world_fails_the_hash_check(tmp_path):
    stage, runs = _stage(tmp_path, branches=("src",))
    man = json.loads((runs / "run" / "manifest.json").read_text())
    man["worlds"][0]["events_sha256"] = "0" * 64
    (runs / "run" / "manifest.json").write_text(json.dumps(man))
    table = ex.Table(
        json.loads(line)
        for line in (stage / "analysis" / "e1_item_table.jsonl").read_text().splitlines()
    )
    res = mm.classify_stage(table, mm.stage_runs(stage, runs), draws=10)
    assert res["checks"]["events_hash_matches"] == 0 and res["checks"]["world_branches"] == 1


# ------------------------------------------------------------------- the committed analysis
committed = pytest.mark.skipif(not STAGE.exists(), reason="stage not staged")


@committed
def test_committed_misses_are_counts_that_add_up_and_verified_worlds():
    res = json.loads(STAGE.read_text())["explore_miss_classes"]
    assert set(res) == {"main", "rep"}
    for role, r in res.items():
        chk = r["checks"]
        assert chk["events_hash_matches"] == chk["world_branches"] and chk["missed"] > 0
        assert chk["missed_same_at_D0"] == chk["missed"]
        for b in ("src", "def", "orphan"):
            for cls, n in r[b].items():
                assert sum(n[f"{c}_count"] for c in EXCLUSIVE) == n["missed_item_times"], (
                    role,
                    b,
                    cls,
                )
                assert n["cited_touched_count"] == 0, (role, b, cls)  # a revision moves the cluster
    main = res["main"]
    # the missed changes are 174, 116 and 136 item-times (14.9% and 20.8% of the changed ones)
    assert [main[b]["all"]["missed_item_times"] for b in ("src", "def", "orphan")] == [
        174,
        116,
        136,
    ]


@committed
def test_committed_misses_are_mostly_not_uncited_dependencies():
    main = json.loads(STAGE.read_text())["explore_miss_classes"]["main"]
    for b in ("src", "def", "orphan"):
        a = main[b]["all"]["a"]["est"]
        assert a < 0.10  # a perfect t0 read set would have caught at most this share
        assert main[b]["all"]["b"]["est"] + main[b]["all"]["c"]["est"] > 0.90

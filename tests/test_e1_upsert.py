"""The slot-keyed upsert cells (`D0B0_upsert`, `D1B0_upsert`, `D1sB0_upsert`)
and the refresh cell with D1s programs (`D1sB0_refresh`). Each test replays the cells on one small
world with recording fake writers (`tests/e1_replay.py`)."""

import pytest

from memshift.experiments.e1 import parse_cell
from memshift.gen.model import fmt_ts
from memshift.holdmem.system import CellConfig
from tests.e1_replay import Replay, replay

D0B0, D1B0, D1SB0 = CellConfig(0, 0), CellConfig(1, 0), CellConfig(1, 0, synth=True)
D0UP, D1UP = CellConfig(0, 0, upsert=True), CellConfig(1, 0, upsert=True)
D1SUP = CellConfig(1, 0, synth=True, upsert=True)
D1SSRC = CellConfig(1, 1, "source", synth=True)
D1SREFRESH = CellConfig(1, 0, synth=True, refresh=True)
PAIRS = ((D0UP, D0B0), (D1UP, D1B0), (D1SUP, D1SB0))
ALL = [D0B0, D1B0, D1SB0, D0UP, D1UP, D1SUP, D1SSRC, D1SREFRESH]
WEEKS = (1, 2, 4, 8)


@pytest.fixture(scope="module")
def rep() -> Replay:
    return replay(ALL, branch="def", weeks=WEEKS)


def bodies(rep, name):
    """Every request of a cell, as (purpose, cache key), in a fixed order (repeats kept)."""
    return sorted((p, k) for k, p, _ in rep.llms[name].requests)


def test_names_tokens_and_validation():
    assert D0UP.name == "holdmem_D0B0_upsert" and D1UP.name == "holdmem_D1B0_upsert"
    assert D1SUP.name == "holdmem_D1sB0_upsert"
    assert D1SREFRESH.name == "holdmem_D1sB0_refresh"
    for token, cfg in {
        "D0B0_upsert": D0UP,
        "D1B0_upsert": D1UP,
        "D1sB0_upsert": D1SUP,
        "D1sB0_refresh": D1SREFRESH,
    }.items():
        assert parse_cell(token) == cfg, token
        assert parse_cell(token).name == f"holdmem_{token}"
    assert parse_cell("D1B0_upsert_nofb") == CellConfig(1, 0, fallback=False, upsert=True)
    assert parse_cell("D1B0_upsert_nofb").name == "holdmem_D1B0_upsert_nofb"
    for bad in ("D0B1_source_upsert", "D1sB1_source_upsert", "D0B0_refresh_upsert", "D0B0upsert"):
        with pytest.raises(ValueError):
            parse_cell(bad)
    with pytest.raises(ValueError):
        CellConfig(0, 1, "source", upsert=True)  # an upsert keeps no key
    with pytest.raises(ValueError):
        CellConfig(0, 0, refresh=True, upsert=True)


def test_upsert_sends_exactly_the_requests_of_the_append_only_cell(rep):
    for cfg, plain in PAIRS:
        assert rep.llms[cfg.name].requests
        assert bodies(rep, cfg.name) == bodies(rep, plain.name), cfg.name
        same = sorted(repr(b) for _, _, b in rep.llms[cfg.name].requests)
        assert same == sorted(repr(b) for _, _, b in rep.llms[plain.name].requests)


def test_pre_shift_contexts_are_identical_across_the_cells(rep):
    first = rep.contexts_t0[D0B0.name]
    assert first
    for name, ctx in rep.contexts_t0.items():
        assert ctx == first, name


def test_a_new_item_retires_the_older_items_of_its_slot_and_nothing_else(rep):
    for cfg, plain in PAIRS:
        up, ref = rep.cells[cfg.name], rep.cells[plain.name]
        assert up.items.keys() == ref.items.keys()
        assert up.by_slot == ref.by_slot
        superseded = kept = 0
        for slot in rep.world.slots:
            ids = up.by_slot[slot.slot_id]
            for n, item_id in enumerate(ids):
                item, plain_item = up.items[item_id], ref.items[item_id]
                assert item.s_k == plain_item.s_k and item.eval_value == plain_item.eval_value
                events = [h for h in item.history if h["event"] == "supersede"]
                if n == len(ids) - 1:
                    # the newest item of the slot stays as the append-only cell wrote it
                    assert item.status == plain_item.status and not events
                    kept += len(ids) == 1
                    continue
                # an older item is retired once, when the next item of the slot arrives
                assert plain_item.status in ("valid", "unverifiable")
                assert item.status == "invalid"
                assert events == [
                    {
                        "ts": fmt_ts(up.items[ids[n + 1]].written),
                        "event": "supersede",
                        "old": plain_item.status,
                        "new": "invalid",
                        "value": None,
                    }
                ]
                superseded += 1
        assert superseded > 0 and kept > 0, cfg.name  # both cases occur on this world
        assert sum(s["superseded"] for s in rep.stats[cfg.name]) == superseded
        assert [s["appended"] for s in rep.stats[cfg.name]] == [
            s["appended"] for s in rep.stats[plain.name]
        ]
        assert all("superseded" not in s for s in rep.stats[plain.name])


def test_upsert_serves_only_the_newest_item_of_a_slot(rep):
    t = rep.at(WEEKS[-1])
    for cfg, plain in PAIRS:
        up, ref = rep.cells[cfg.name], rep.cells[plain.name]
        slot = max(rep.world.slots, key=lambda s: len(up.by_slot[s.slot_id]))
        ids = up.by_slot[slot.slot_id]
        assert len(ids) > 1
        valid = [i for i in ids if up.items[i].status == "valid"]
        assert valid in ([], [ids[-1]])
        (probe,) = [p for p in rep.probes(t) if p.fact_id == slot.slot_id]
        blocks = up.serve(probe).blocks
        for item_id in ids[:-1]:
            assert up.items[item_id].s_k not in blocks
        # the append-only cell keeps the older derivations servable beside the new one
        assert all(ref.items[i].status in ("valid", "unverifiable") for i in ids)


def test_d1s_refresh_runs_with_the_source_epoch_requests_and_never_invalidates(rep):
    hm = rep.cells[D1SREFRESH.name]
    assert bodies(rep, D1SREFRESH.name) == bodies(rep, D1SSRC.name)
    assert {p for p, _ in bodies(rep, D1SREFRESH.name)} == {"write", "program"}
    assert sum(s["appended"] for s in rep.stats[D1SREFRESH.name]) > 0
    assert [s["appended"] for s in rep.stats[D1SREFRESH.name]] == [
        s["repaired"] for s in rep.stats[D1SSRC.name]
    ]
    assert not any(h["event"] == "invalidate" for i in hm.items.values() for h in i.history)
    kappa = next(s for s in rep.world.slots if s.kappa_id)
    item = hm.items[hm.by_slot[kappa.slot_id][-1]]
    assert item.det is not None and "program_ok" in item.det  # the D1s path, not the fallback

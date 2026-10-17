"""Where a slot-keyed upsert can act (`analysis.ex_round3_upsert`, exploratory): the split of the
changed item-times by whether the delta yielded a new item for the slot, the parts' levels, their
contributions to binding's effect and the answers the upsert moved, on small hand-built inputs
(no model, no committed run)."""

import math

from memshift.analysis import ex_round3_upsert as eu

FORK = "2026-09-30T23:59"
K1, K2 = "2026-10-07T23:59", "2026-10-14T23:59"


def test_yielded_needs_an_item_after_the_fork_and_by_the_checkpoint():
    assert eu.yielded([FORK, K1], FORK, K1)  # written at the checkpoint
    assert not eu.yielded([FORK], FORK, K2)  # only the pre-shift snapshot
    assert not eu.yielded([FORK, K2], FORK, K1)  # written after this checkpoint
    assert eu.yielded([K1], FORK, K2)


def test_times_and_supersede_counts_come_from_the_items():
    items = [
        {
            "system": "holdmem_D1B0_upsert",
            "world_id": "w1",
            "branch": "src",
            "slot_id": "k.alpha",
            "written": FORK,
            "history": [{"ts": FORK, "event": "derive"}, {"ts": K1, "event": "supersede"}],
        },
        {
            "system": "holdmem_D1B0_upsert",
            "world_id": "w1",
            "branch": "src",
            "slot_id": "u.employer",
            "written": K1,
            "history": [{"ts": FORK, "event": "supersede"}],  # at the fork: not after it
        },
    ]
    times = eu.written_times(items)
    assert times[("holdmem_D1B0_upsert", "w1", "src", "k.alpha")] == [FORK]
    asof = {("w1", "src", 0): FORK, ("w1", "src", 1): K1}
    counts = eu.supersede_counts(items, asof)
    assert counts == {"holdmem_D1B0_upsert": {"src": {"definition": 1, "other": 0}}}


def test_split_levels_contributions_and_moved_answers():
    # w1: two item-times where the delta yielded an item (B0 already right on one) and two where
    # it yielded none (only B1 right); w2: one of each. The upsert moves one answer up.
    x = {"w1": {"a@1": 1.0, "b@1": 1.0, "c@1": 0.0, "d@1": 0.0}, "w2": {"a@1": 1.0, "c@1": 0.0}}
    b0 = {"w1": {"a@1": 1.0, "b@1": 0.0, "c@1": 0.0, "d@1": 0.0}, "w2": {"a@1": 1.0, "c@1": 0.0}}
    b1 = {"w1": {"a@1": 1.0, "b@1": 1.0, "c@1": 1.0, "d@1": 1.0}, "w2": {"a@1": 1.0, "c@1": 1.0}}
    flag = {
        "w1": {"a@1": True, "b@1": True, "c@1": False, "d@1": False},
        "w2": {"a@1": True, "c@1": False},
    }
    out = eu.split(x, b0, b1, flag, draws=50)
    assert out["n_items"] == 6
    y, n = out["parts"]["yielded"], out["parts"]["none"]
    assert (y["n_items"], n["n_items"]) == (3, 3) and y["share"] == n["share"] == 0.5
    assert math.isclose(y["update"]["B0"]["est"], 2 / 3)
    assert math.isclose(y["update"]["upsert"]["est"], 1.0)
    assert n["update"]["B0"]["est"] == n["update"]["upsert"]["est"] == 0.0
    assert n["update"]["B1"]["est"] == 1.0
    assert y["upsert_differs"] == {"above_B0": 1, "below_B0": 0}
    assert n["upsert_differs"] == {"above_B0": 0, "below_B0": 0}
    # binding's effect (B1 - B0 over all six) is 4/6, of which the no-yield part carries 3/6
    assert math.isclose(out["B_effect"], 4 / 6)
    assert math.isclose(n["contribution_to_B_effect"], 3 / 6)
    assert math.isclose(y["contribution_to_B_effect"] + n["contribution_to_B_effect"], 4 / 6)


def test_split_keeps_only_item_times_every_cell_answered_and_an_empty_part():
    x = {"w1": {"a@1": 0.0, "z@1": 1.0}}  # z: B=0 and B=1 did not answer it
    b0 = {"w1": {"a@1": 0.0}}
    b1 = {"w1": {"a@1": 1.0}}
    out = eu.split(x, b0, b1, {"w1": {"a@1": False, "z@1": True}}, draws=50)
    assert out["n_items"] == 1
    assert out["parts"]["yielded"]["n_items"] == 0
    assert math.isnan(out["parts"]["yielded"]["update"]["B0"]["est"])
    assert out["parts"]["none"]["contribution_to_B_effect"] == 1.0

"""The offline key audit agrees with what the cells do when they run."""

import pytest

from memshift.analysis.e1_keys import audit, item_class
from memshift.holdmem.system import CellConfig
from tests.e1_replay import replay
from tests.test_e1_controls import events

WEEKS = (1, 2, 4, 8)


@pytest.fixture(scope="module")
def result():
    return audit([5], ("def", "orphan"), months=3)


def test_the_keys_miss_nothing_and_the_oracle_falsely_invalidates_far_less(result):
    for branch in ("def", "orphan"):
        for key, by_class in result[branch].items():
            for cls, r in by_class.items():
                assert r["missed"] in (0.0, None), (branch, key, cls)
        source, gold = result[branch]["source"], result[branch]["gold"]
        for cls in ("rule", "norule"):
            assert source[cls]["false_invalidation"] > 0.9
            assert gold[cls]["false_invalidation"] < 0.5 * source[cls]["false_invalidation"]
            assert gold[cls]["rederivations"] < source[cls]["rederivations"]
        assert set(result[branch]["scope"]) == {"definition"}  # the other items are keyed by cites


def test_the_audit_counts_the_invalidations_of_the_cells_that_run():
    cfgs = [CellConfig(0, 1, "source"), CellConfig(0, 1, "gold"), CellConfig(0, 1, "scope")]
    rep = replay(cfgs, "def", 5, WEEKS)
    res = audit([5], ("def",), months=3)["def"]
    for cfg in cfgs:
        hm = rep.cells[cfg.name]
        n = {"definition": 0, "rule": 0, "norule": 0}
        for slot in rep.world.slots:
            if cfg.granularity == "scope" and not slot.kappa_id:
                continue
            n[item_class(slot.slot_id)] += sum(
                ev == "invalidate" for ev, _ in events(hm, slot.slot_id)
            )
        for cls, rate in res[cfg.granularity].items():
            assert rate["rederivations"] == n[cls], (cfg.name, cls)

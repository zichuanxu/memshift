"""Held-out definitions (prereg v4_round3 §G-a): grammar, seeded draws with the gold-only rejection
rule, and worlds that swap the six fixed definitions for a draw while keeping everything else."""

from dataclasses import asdict
from datetime import timedelta

import pytest

from memshift.gen.gold import gold_at
from memshift.gen.io import world_meta
from memshift.gen.kappa import (
    FIXED_PAIRS,
    GRAMMAR,
    KAPPA_DEFS,
    evaluate,
    evaluate_spec,
    heldout_def,
    heldout_draw,
    heldout_spec,
    kappa_set_defs,
    list_heldout,
    scope_members,
)
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import fallback
from memshift.holdmem.program import SOURCE_FIELDS
from memshift.holdmem.writer import VALUE_FORMAT
from memshift.probes import make_probes
from memshift.textutil import sha256_json

BRANCHES = ("natural", "src", "def", "orphan")
WEEKS = (1, 2, 4, 8)

# world 1000 with the fixed definitions, recorded before held-out definitions existed: events,
# slots, world record, probes (questions, gold, deps, versions) and gold at fork + 1, 2, 4, 8 weeks
FIXED_1000 = {
    "natural": ("433c7312796b0587", "d8043f14d4625538", "1600c61a970d1d3c", "fe7fb5fc90732520",
                "b0329da461038f3c"),
    "src": ("df724790d6af5b22", "d8043f14d4625538", "6e7d0201338d1a1a", "1fbbf188c69ba708",
            "f00cada2c0fc8059"),
    "def": ("433c7312796b0587", "d8043f14d4625538", "022d2422ff3d320d", "2290fa1483e3c7fa",
            "61058093222780b4"),
    "orphan": ("c357e8df05252063", "d8043f14d4625538", "a56bc4d337dac4bd", "8ec9efbae78c218b",
               "c946ab67aa79e9b2"),
}  # fmt: skip

# the registered draws (listed in the decision log before any model call)
DRAWS = {
    4001: (
        [
            "h4001_priority_project",
            "h4001_social_hours",
            "h4001_recently_completed_task",
            "h4001_evening_chat_partner",
            "h4001_after_hours_sender",
            "h4001_team_regular",
        ],
        "f3cfde08d333951d106129f14b59e84451c35a98d31718537aa9122bb051d78c",
    ),
    4002: (
        [
            "h4002_frequent_recipient",
            "h4002_deep_dive_partner",
            "h4002_after_hours_sender",
            "h4002_productive_project",
            "h4002_team_regular",
            "h4002_priority_project",
        ],
        "c45292d1db354157efaa14a757b2baff3b47621b71759811d08f055ca8fc42a6",
    ),
    4003: (
        [
            "h4003_frequent_recipient",
            "h4003_social_hours",
            "h4003_upcoming_task",
            "h4003_recently_completed_task",
            "h4003_hot_task",
            "h4003_close_relative",
        ],
        "0300f9f1f255fd8eab46c74c0b9be79ddc995ea5f7d16c24b45539fe0cd9d24a",
    ),
}


def _times(world):
    return [world.fork + timedelta(weeks=k) for k in WEEKS]


@pytest.fixture(scope="module")
def fixed():
    return {b: generate_branch(1000, b) for b in BRANCHES}


@pytest.fixture(scope="module")
def heldout():
    return {b: generate_branch(1000, b, {"kappa_set": "heldout:4001"}) for b in BRANCHES}


def test_fixed_set_worlds_unchanged(fixed):
    for b, w in fixed.items():
        log = SourceLog(w.events)
        times = _times(w)
        got = (
            sha256_json([e.to_json() for e in w.events])[:16],
            sha256_json([asdict(s) for s in w.slots])[:16],
            sha256_json(world_meta(w))[:16],
            sha256_json([p.to_json() for p in make_probes(w, log, times)])[:16],
            sha256_json([gold_at(w, log, s, t).to_json() for t in times for s in w.slots])[:16],
        )
        assert got == FIXED_1000[b], b
    assert kappa_set_defs(None) is KAPPA_DEFS and kappa_set_defs("fixed") is KAPPA_DEFS


def test_grammar_is_disjoint_from_the_fixed_definitions():
    pairs = {(p.source, name) for name, p in GRAMMAR.items()}
    assert len(pairs) == len(GRAMMAR) and not pairs & FIXED_PAIRS
    for p in GRAMMAR.values():
        assert p.source == p.sources[0] and set(p.sources) <= set(SOURCE_FIELDS)
        assert len(p.windows) >= 2 and len(p.thresholds) >= 2
        for aggregate, form in p.forms.items():
            assert aggregate in ("count", "members", "span")
            assert form.vtype in VALUE_FORMAT
            assert (form.key is None) == (aggregate == "span")


@pytest.mark.parametrize("seed", sorted(DRAWS))
def test_registered_draws(seed):
    ids, digest = DRAWS[seed]
    draw = heldout_draw(seed)
    assert [d.kappa_id for d in draw.defs] == ids
    rows = [[d.kappa_id, asdict(d.v1), asdict(d.v2), d.changed] for d in draw.defs]
    assert sha256_json(rows) == digest
    assert len({d.v1.predicate for d in draw.defs}) == 6
    for d in draw.defs:
        assert d.sanity.rejected() is None
        assert d.v1.predicate == d.v2.predicate and d.v1.aggregate == d.v2.aggregate
        moved = {
            "window": d.v1.window != d.v2.window,
            "threshold": d.v1.threshold != d.v2.threshold,
        }
        assert moved == {c: c == d.changed for c in moved}
        spec = d.as_kappa_def()
        for ver, s in (("v1", d.v1), ("v2", d.v2)):
            assert f" {s.window} days" in spec[ver]
            assert s.aggregate != "count" or f"at least {s.threshold} " in spec[ver]
        assert heldout_def(d.kappa_id) is d and heldout_spec(d.kappa_id, "v2") == d.v2
    assert all(reason for _, _, reason in draw.rejected)


def test_heldout_worlds_change_only_the_definition_items(fixed, heldout):
    draw = heldout_draw(4001)
    ids = [d.kappa_id for d in draw.defs]
    for b in BRANCHES:
        fw, hw = fixed[b], heldout[b]
        assert [e.to_json() for e in hw.events] == [e.to_json() for e in fw.events]
        assert [s.kind for s in hw.slots] == [s.kind for s in fw.slots]
        assert [s.kappa_id for s in hw.slots if s.kind == "kappa"] == ids
        others = [s for s in fw.slots if s.kind != "kappa"]
        assert [s for s in hw.slots if s.kind != "kappa"] == others
        assert sorted(hw.kappa_defs) == sorted(ids) and hw.params["kappa_set"] == "heldout:4001"
        times = _times(fw)
        assert [p.to_json() for p in make_probes(hw, SourceLog(hw.events), times, others)] == [
            p.to_json() for p in make_probes(fw, SourceLog(fw.events), times, others)
        ]
    moved = heldout["def"].params["shift_targets"]
    assert len(moved) == 3 and set(moved) <= set(ids)
    assert all(heldout["def"].kappa_version(k, heldout["def"].fork) == "v2" for k in moved)


def test_gold_and_authored_derivation_share_the_generic_evaluator(heldout):
    for w in heldout.values():
        log = SourceLog(w.events)
        for slot in (s for s in w.slots if s.kind == "kappa"):
            assert slot.kappa_id is not None
            for t in _times(w):
                snap = log.snapshot(t)
                ver = w.kappa_version(slot.kappa_id, t)
                g = gold_at(w, log, slot, t)
                value, ids = evaluate_spec(heldout_spec(slot.kappa_id, ver), snap, t)
                assert (g.value, [d[0] for d in g.deps]) == (value, ids)
                assert evaluate(slot.kappa_id, snap, t, ver) == (value, ids)
                fb = fallback(w, log, slot, t)
                assert not isinstance(fb, str) and fb[0] == g.value
                scope = {oid for oid, _ in scope_members(slot.kappa_id, snap, t, ver)}
                assert set(ids) <= scope


def test_kappa_set_errors_and_listing():
    with pytest.raises(ValueError):
        kappa_set_defs("heldout")
    with pytest.raises(KeyError):
        heldout_def("h4001_no_such_definition")
    with pytest.raises(KeyError):
        heldout_spec("no_such_definition", "v1")
    out = list_heldout([4001])
    assert "heldout:4001" in out and all(i in out for i in DRAWS[4001][0])

"""The E1 control cells: oracle and scope keys, refresh without invalidation, template presentation
and the thinking writer. Each test replays the cells on one small world with recording fake writers
(`tests/e1_replay.py`), which gives the request set of every cell as response-cache keys."""

import re
from datetime import timedelta

import pytest

from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_cell, parse_cells
from memshift.gen.gold import gold_at
from memshift.gen.kappa import KAPPA_DEFS, SCOPE_PREDICATES, SCOPE_RANGES, scope_members
from memshift.gen.model import fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import render_fallback_sk
from memshift.holdmem.system import CellConfig, HoldMem, _as_text, definition_text
from memshift.holdmem.writer import THINK_FORMAT, THINK_MAX_TOKENS, derive, writer_request
from memshift.llm import LLMResult, default_extra_body, think_extra_body
from tests.e1_replay import Replay, replay
from tests.fakes import FakeLLM

D0B0, D1B0 = CellConfig(0, 0), CellConfig(1, 0)
D0SRC, D1SRC = CellConfig(0, 1, "source"), CellConfig(1, 1, "source")
D0CITED = CellConfig(0, 1, "cited")
D0GOLD, D1GOLD = CellConfig(0, 1, "gold"), CellConfig(1, 1, "gold")
D0SCOPE, D1SCOPE = CellConfig(0, 1, "scope"), CellConfig(1, 1, "scope")
D0REFRESH, D1REFRESH = CellConfig(0, 0, refresh=True), CellConfig(1, 0, refresh=True)
D0T0, D0T1 = CellConfig(0, 0, present=True), CellConfig(0, 1, "source", present=True)
D0THINK0, D0THINK1 = CellConfig(0, 0, think=True), CellConfig(0, 1, "source", think=True)
ALL = [
    D0B0, D1B0, D0SRC, D1SRC, D0CITED, D0GOLD, D1GOLD, D0SCOPE, D1SCOPE,
    D0REFRESH, D1REFRESH, D0T0, D0T1, D0THINK0, D0THINK1,
]  # fmt: skip
WEEKS = (1, 2, 4, 8)
KAPPA_DESCRIPTIONS = {spec["description"] for spec in KAPPA_DEFS.values()}


@pytest.fixture(scope="module")
def rep() -> Replay:
    return replay(ALL, branch="def", weeks=WEEKS)


def site(body):
    """(item description, as-of time, is a delta request) of a writer request body."""
    text = body["messages"][-1]["content"]
    item = re.search(r"Knowledge item: (.*)", text)
    as_of = re.search(r"As of: (.*)", text)
    assert item and as_of
    return item.group(1), as_of.group(1), "only those added or changed since" in text


def sites(rep, name):
    return {site(b) for _, p, b in rep.llms[name].requests if p == "write"}


def events(hm, slot_id):
    return [(h["event"], h["ts"]) for i in hm.by_slot[slot_id] for h in hm.items[i].history]


# ---------------------------------------------------------------------------- configuration
def test_names_validation_and_tokens():
    assert D0GOLD.name == "holdmem_D0B1_gold" and D1GOLD.name == "holdmem_D1B1_gold"
    assert D0SCOPE.name == "holdmem_D0B1_scope" and D1SCOPE.name == "holdmem_D1B1_scope"
    assert D0REFRESH.name == "holdmem_D0B0_refresh" and D1REFRESH.name == "holdmem_D1B0_refresh"
    assert D0T0.name == "holdmem_D0tB0" and D0T1.name == "holdmem_D0tB1_source"
    assert D0THINK0.name == "holdmem_D0thinkB0"
    assert D0THINK1.name == "holdmem_D0thinkB1_source"
    assert len({c.name for c in ALL}) == len(ALL)
    for token, cfg in {
        "D0B1_gold": D0GOLD,
        "D1B1_gold": D1GOLD,
        "D0B1_scope": D0SCOPE,
        "D0B0_refresh": D0REFRESH,
        "D1B0_refresh": D1REFRESH,
        "D0tB0": D0T0,
        "D0tB0_source": D0T0,
        "D0tB1_source": D0T1,
        "D0tB1": D0T1,
        "D0thinkB0": D0THINK0,
        "D0thinkB1_source": D0THINK1,
    }.items():
        assert parse_cell(token) == cfg, token
        assert parse_cell(token).name == cfg.name
    # the tokens that existed keep their meaning
    assert parse_cell("D1B1_source_nofb") == CellConfig(1, 1, "source", fallback=False)
    assert parse_cell("D1sB1_cited") == CellConfig(1, 1, "cited", synth=True)
    for bad in ("D1B1_golden", "D1B1_refresh", "D0B0_refresh_nofb", "D1tB0", "D1thinkB1_source"):
        with pytest.raises(ValueError):
            parse_cell(bad)
    with pytest.raises(ValueError):
        CellConfig(0, 1, "cited", refresh=True)  # refresh is the B=0 cell on the source-epoch key
    with pytest.raises(ValueError):
        CellConfig(1, 0, present=True)
    with pytest.raises(ValueError):
        CellConfig(0, 0, present=True, think=True)
    with pytest.raises(ValueError):
        CellConfig(0, 1, "weekly")


def test_views_are_split_from_cells():
    cells, views = parse_cells("D0B0,zero_Dm,D1B1_gold")
    assert cells == [D0B0, D1GOLD] and views == ("zero_Dm",)
    assert parse_cells("zero_Dm") == ([], ("zero_Dm",))
    assert parse_cells("D0B0") == ([D0B0], ())


def test_a_think_cell_needs_a_think_writer(rep):
    with pytest.raises(ValueError):
        HoldMem(rep.world, rep.log, D0THINK1, FakeLLM(), "fake/writer-a")


# ---------------------------------------------------------------------------- shared invariants
def test_pre_shift_contexts_are_identical_across_every_cell(rep):
    first = rep.contexts_t0[D0B0.name]
    assert first
    for name, ctx in rep.contexts_t0.items():
        assert ctx == first, name


def test_control_cells_send_no_request_the_cache_could_not_serve(rep):
    """Every writer request of a control cell is one of an existing cell, except the listed few."""
    existing = {name: rep.keys(name, "write") for name in (D0B0.name, D0SRC.name, D0CITED.name)}
    pool = set().union(*existing.values())
    for cfg in (D0REFRESH, D1REFRESH, D0T0, D0T1):  # no new request at all
        assert rep.keys(cfg.name, "write") <= pool, cfg.name
    for cfg in (D0GOLD, D1GOLD, D0SCOPE, D1SCOPE):  # see the tests of the two keys
        assert len(rep.keys(cfg.name, "write") - pool) <= 3, cfg.name


# ---------------------------------------------------------------------------- B1_gold
def _gold_key(rep, slot, t):
    g = gold_at(rep.world, rep.log, slot, t)
    kv = rep.world.kappa_version(slot.kappa_id, t) if slot.kappa_id else None
    return sorted(g.deps), kv


def test_gold_key_invalidates_exactly_the_items_whose_dependencies_changed(rep):
    for cfg in (D0GOLD, D1GOLD):
        hm = rep.cells[cfg.name]
        n_invalidated = 0
        for slot in rep.world.slots:
            invalidated = {ts for ev, ts in events(hm, f"{slot.slot_id}") if ev == "invalidate"}
            last = rep.t0
            for k in WEEKS:
                t = rep.at(k)
                changed = _gold_key(rep, slot, t) != _gold_key(rep, slot, last)
                assert (fmt_ts(t) in invalidated) == changed, (cfg.name, slot.slot_id, k)
                if changed:
                    last = t
                    n_invalidated += 1
        assert n_invalidated == sum(s["invalidated"] for s in rep.stats[cfg.name])
        assert 0 < n_invalidated < len(rep.world.slots) * len(WEEKS) / 2  # not a refresh-everything


def test_gold_key_misses_no_changed_item_and_falsely_invalidates_far_fewer(rep):
    def rates(name):
        missed = changed = false = unchanged = 0
        for slot in rep.world.slots:
            g0 = gold_at(rep.world, rep.log, slot, rep.t0)
            hist = events(rep.cells[name], slot.slot_id)
            for k in WEEKS:
                t = rep.at(k)
                g = gold_at(rep.world, rep.log, slot, t)
                detected = any(ev == "invalidate" and ts <= fmt_ts(t) for ev, ts in hist)
                if (g.value, g.status) != (g0.value, g0.status):
                    changed += 1
                    missed += not detected
                else:
                    unchanged += 1
                    false += detected
        return missed / changed, false / unchanged

    miss_gold, fi_gold = rates(D0GOLD.name)
    miss_src, fi_src = rates(D0SRC.name)
    assert miss_gold == 0.0 and miss_src == 0.0
    assert fi_src > 0.9 and fi_gold < 0.5 * fi_src


def test_gold_detection_does_not_depend_on_d(rep):
    for slot in rep.world.slots:
        assert events(rep.cells[D0GOLD.name], slot.slot_id) == events(
            rep.cells[D1GOLD.name], slot.slot_id
        ), slot.slot_id
    assert [s["invalidated"] for s in rep.stats[D0GOLD.name]] == [
        s["invalidated"] for s in rep.stats[D1GOLD.name]
    ]


def test_gold_rederivations_are_source_epoch_requests_except_where_no_source_changed(rep):
    source = sites(rep, D0SRC.name)
    for cfg in (D0GOLD, D1GOLD):
        for request in sites(rep, cfg.name):
            if request not in source:
                # a dependency that changed with no source event (the clock moved a window)
                item, as_of, _ = request
                slot = next(s for s in rep.world.slots if s.description == item)
                t = next(rep.at(k) for k in WEEKS if f"{rep.at(k):%Y-%m-%d %H:%M}" == as_of)
                ev = events(rep.cells[D0SRC.name], slot.slot_id)
                assert ("invalidate", fmt_ts(t)) not in ev, (item, as_of)
    assert 0 < len(sites(rep, D0GOLD.name) & source) < len(source)


def test_only_the_gold_cell_reads_gold(rep, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("gold was read to build a key")

    monkeypatch.setattr("memshift.holdmem.system.gold_at", boom)
    t = rep.at(1)
    for cfg in (D0SRC, D1SRC, D0CITED, D0SCOPE, D1SCOPE, D0REFRESH):
        hm = rep.cells[cfg.name]
        for slot in rep.world.slots:
            hm.key_at(slot, hm.items[f"{slot.slot_id}#0"].readset_clusters, t)


def test_gold_key_shape(rep):
    hm = rep.cells[D1GOLD.name]
    for slot in rep.world.slots:
        bind = hm.items[f"{slot.slot_id}#0"].bind
        assert bind["granularity"] == "gold" and set(bind["scope"]) == {"deps", "n"}
        assert bind["theta_w"] == "fake/writer-a"
        assert (bind["kappa_version"] is not None) == bool(slot.kappa_id)


# ---------------------------------------------------------------------------- B0_refresh
def test_refresh_sends_exactly_the_source_epoch_requests(rep):
    assert rep.keys(D0REFRESH.name) == rep.keys(D0SRC.name)
    assert rep.keys(D1REFRESH.name) == rep.keys(D1SRC.name)
    assert sites(rep, D0REFRESH.name) == sites(rep, D0SRC.name)
    assert not any(delta for _, _, delta in sites(rep, D0REFRESH.name))  # full in-scope records


def test_refresh_appends_a_fresh_item_and_never_invalidates(rep):
    for cfg, src in ((D0REFRESH, D0SRC), (D1REFRESH, D1SRC)):
        hm = rep.cells[cfg.name]
        assert not any(h["event"] == "invalidate" for i in hm.items.values() for h in i.history)
        assert all(s["invalidated"] == 0 and s["repaired"] == 0 for s in rep.stats[cfg.name])
        # as many refreshes as the source-epoch cell has repairs, each one more item in the slot
        assert [s["appended"] for s in rep.stats[cfg.name]] == [
            s["repaired"] for s in rep.stats[src.name]
        ]
        for slot in rep.world.slots:
            ids = hm.by_slot[slot.slot_id]
            assert len(ids) == 1 + sum(
                1 for h in rep.cells[src.name].items[f"{slot.slot_id}#0"].history
                if h["event"] == "repair"
            )  # fmt: skip
            assert [hm.items[i].history[0]["event"] for i in ids] == ["derive"] + ["append"] * (
                len(ids) - 1
            )
            assert all(hm.items[i].status in ("valid", "unverifiable") for i in ids)


def test_refresh_serves_the_old_derivation_beside_the_new_one(rep):
    hm, src = rep.cells[D0REFRESH.name], rep.cells[D0SRC.name]

    def repairs(slot):
        return sum(h["event"] == "repair" for h in src.items[f"{slot.slot_id}#0"].history)

    slot = max(rep.world.slots, key=repairs)
    assert repairs(slot) == len(WEEKS) and len(hm.by_slot[slot.slot_id]) == 1 + len(WEEKS)
    (probe,) = [p for p in rep.probes(rep.at(WEEKS[-1])) if p.fact_id == slot.slot_id]
    blocks = hm.serve(probe).blocks
    for item_id in hm.by_slot[slot.slot_id]:
        assert hm.items[item_id].s_k in blocks, item_id
    # the source-epoch cell keeps one item per slot and serves only its newest derivation
    assert src.by_slot[slot.slot_id] == [f"{slot.slot_id}#0"]
    assert src.items[f"{slot.slot_id}#0"].s_k in src.serve(probe).blocks


def test_refresh_keeps_the_fallback_path_of_d1(rep):
    hm = rep.cells[D1REFRESH.name]
    structured = next(s for s in rep.world.slots if s.structured and not s.kappa_id)
    new = hm.items[hm.by_slot[structured.slot_id][-1]]
    assert new.det is not None and new.det["check"] in ("fallback", "pass")
    kappa = next(s for s in rep.world.slots if s.kappa_id)
    item = hm.items[hm.by_slot[kappa.slot_id][-1]]
    assert item.det is not None and item.det["check"] == "fallback"  # the fake writer's value fails
    assert "under the definition in effect" in item.s_k


# ---------------------------------------------------------------------------- D0t
def test_d0t_sends_exactly_the_d0_requests(rep):
    assert rep.keys(D0T0.name) == rep.keys(D0B0.name)
    assert rep.keys(D0T1.name) == rep.keys(D0SRC.name)
    assert sites(rep, D0T0.name) == sites(rep, D0B0.name)


def test_d0t_keeps_the_d0_values_and_has_no_checker(rep):
    for t_cfg, d0_cfg in ((D0T0, D0B0), (D0T1, D0SRC)):
        t_cell, d0 = rep.cells[t_cfg.name], rep.cells[d0_cfg.name]
        assert t_cell.items.keys() == d0.items.keys()
        for item_id, item in t_cell.items.items():
            ref = d0.items[item_id]
            assert item.det is None and item.status == ref.status == "valid"
            assert item.eval_value == ref.eval_value and item.refs == ref.refs
            assert item.bind == ref.bind  # binding is D0's, presentation is the only change
            assert item.readset == ref.readset


def test_d0t_renders_items_written_after_the_shift_in_the_d1_template(rep):
    for cfg, d0_cfg in ((D0T0, D0B0), (D0T1, D0SRC)):
        t_cell, d0 = rep.cells[cfg.name], rep.cells[d0_cfg.name]
        n_template = 0
        for item_id, item in t_cell.items.items():
            slot = rep.world.slot(item.slot_id)
            built = item.history[0]["event"] == "derive" and item.written == rep.t0
            if built:  # the pre-shift snapshot is shared
                assert item.s_k == d0.items[item_id].s_k
                continue
            n_template += 1
            expected = render_fallback_sk(
                slot,
                item.written,
                _as_text(item.eval_value),
                item.refs,
                definition_text(rep.world, slot, item.written),
            )
            assert item.s_k == expected
            assert item.s_k != d0.items[item_id].s_k
            assert ("under the definition in effect" in item.s_k) == bool(slot.kappa_id)
            assert "programmatic extraction from" in item.s_k
        assert n_template > 0


def test_d0t_repairs_what_d0_repairs_at_the_same_times(rep):
    for slot in rep.world.slots:
        assert events(rep.cells[D0T1.name], slot.slot_id) == events(
            rep.cells[D0SRC.name], slot.slot_id
        )
    assert rep.stats[D0T0.name] == rep.stats[D0B0.name]
    assert rep.stats[D0T1.name] == rep.stats[D0SRC.name]


def test_d0t_template_handles_a_value_the_writer_could_not_give(rep):
    slot = rep.world.slots[0]
    assert _as_text(None) is None and _as_text(["a", 3]) == ["a", "3"] and _as_text(3) == "3"
    text = render_fallback_sk(slot, rep.t0, None, [("x:1", 1)], None)
    assert "no current value" in text


# ---------------------------------------------------------------------------- B1_scope
def test_scope_key_definition_items_by_scope_others_by_cited_clusters(rep):
    scope, cited = rep.cells[D0SCOPE.name], rep.cells[D0CITED.name]
    for slot in rep.world.slots:
        item = scope.items[f"{slot.slot_id}#0"]
        if slot.kappa_id:
            assert item.bind["granularity"] == "scope"
            assert set(item.bind["scope"]) == {"members", "n"}
        else:
            assert item.bind["granularity"] == "cited"
            assert item.bind == cited.items[f"{slot.slot_id}#0"].bind
            assert events(scope, slot.slot_id) == events(cited, slot.slot_id), slot.slot_id


def test_scope_detection_does_not_depend_on_d(rep):
    for slot in rep.world.slots:
        assert events(rep.cells[D0SCOPE.name], slot.slot_id) == events(
            rep.cells[D1SCOPE.name], slot.slot_id
        ), slot.slot_id


def _scope_key(rep, slot, t):
    kv = rep.world.kappa_version(slot.kappa_id, t)
    return scope_members(slot.kappa_id, rep.log.snapshot(t), t, kv), kv


def test_scope_key_invalidates_exactly_when_the_scope_or_the_version_moves(rep):
    hm = rep.cells[D0SCOPE.name]
    for slot in (s for s in rep.world.slots if s.kappa_id):
        invalidated = {ts for ev, ts in events(hm, slot.slot_id) if ev == "invalidate"}
        last = rep.t0
        for k in WEEKS:
            t = rep.at(k)
            changed = _scope_key(rep, slot, t) != _scope_key(rep, slot, last)
            assert (fmt_ts(t) in invalidated) == changed, (slot.slot_id, k)
            if changed:
                last = t


def test_scope_key_is_not_built_from_the_extractor(rep, monkeypatch):
    import memshift.gen.kappa as kappa

    def boom(*args, **kwargs):
        raise AssertionError("an evaluator was called to build a key")

    monkeypatch.setattr(kappa, "evaluate", boom)
    monkeypatch.setattr("memshift.holdmem.system.gold_at", boom)
    monkeypatch.setattr("memshift.holdmem.checker.evaluate", boom)
    monkeypatch.setattr(kappa, "EVALUATORS", {k: boom for k in kappa.EVALUATORS})
    hm = rep.cells[D1SCOPE.name]
    for slot in rep.world.slots:
        assert hm.key_at(slot, [], rep.at(2)) is not None


def test_scope_rederivations_are_source_epoch_requests_except_where_no_source_changed(rep):
    source = sites(rep, D0SRC.name)
    for cfg in (D0SCOPE, D1SCOPE):
        for request in sites(rep, cfg.name) - source:
            item, as_of, _ = request
            slot = next(s for s in rep.world.slots if s.description == item)
            t = next(rep.at(k) for k in WEEKS if f"{rep.at(k):%Y-%m-%d %H:%M}" == as_of)
            assert ("invalidate", fmt_ts(t)) not in events(rep.cells[D0SRC.name], slot.slot_id)


@pytest.mark.parametrize("seed", [5, 11])
@pytest.mark.parametrize("branch", ["src", "def", "orphan"])
def test_gold_and_scope_keys_miss_no_value_change_on_a_three_day_grid(seed, branch):
    w = generate_branch(seed, branch, world_params(3))
    log = SourceLog(w.events)
    grid = [w.fork - timedelta(minutes=1)] + [w.fork + timedelta(days=d) for d in range(3, 60, 3)]
    changes = gold_false = scope_false = 0
    for slot in w.slots:
        prev = None
        for t in grid:
            g = gold_at(w, log, slot, t)
            kv = w.kappa_version(slot.kappa_id, t) if slot.kappa_id else None
            members = (
                scope_members(slot.kappa_id, log.snapshot(t), t, kv) if slot.kappa_id else None
            )
            cur = ((g.value, g.status), (sorted(g.deps), kv), (members, kv))
            if prev is not None:
                value_changed = cur[0] != prev[0]
                changes += value_changed
                if value_changed:
                    assert cur[1] != prev[1], ("gold", slot.slot_id, t)
                    if slot.kappa_id:
                        assert cur[2] != prev[2], ("scope", slot.slot_id, t)
                gold_false += (not value_changed) and cur[1] != prev[1]
                scope_false += bool(slot.kappa_id) and (not value_changed) and cur[2] != prev[2]
            prev = cur
    assert changes > 0
    assert gold_false + scope_false > 0  # the keys are supersets of the changes, not exact


def test_scope_ranges_cover_every_definition():
    assert set(SCOPE_RANGES) == set(KAPPA_DEFS) == set(SCOPE_PREDICATES)


# ---------------------------------------------------------------------------- D0think
def test_think_request_body():
    w = generate_branch(5, "def", world_params(3))
    slot = next(s for s in w.slots if s.kappa_id)
    plain = writer_request(w, slot, w.fork, [])
    think = writer_request(w, slot, w.fork, [], think=True)
    assert plain["max_tokens"] == 700 and plain["schema"] is not None
    assert think["max_tokens"] == THINK_MAX_TOKENS and think["schema"] is None
    assert think["messages"][-1]["content"].endswith(THINK_FORMAT)
    assert think["messages"][-1]["content"].startswith(plain["messages"][-1]["content"])
    assert think_extra_body("Qwen/Qwen3.5-9B") == {
        "chat_template_kwargs": {"enable_thinking": True}
    }
    assert default_extra_body("Qwen/Qwen3.5-9B") == {
        "chat_template_kwargs": {"enable_thinking": False}
    }
    for model in ("openai/gpt-oss-20b", "mistralai/Ministral-3-14B-Instruct-2512-BF16"):
        with pytest.raises(ValueError):
            think_extra_body(model)


def test_think_cells_think_only_about_definition_items_after_the_shift(rep):
    for cfg, plain in ((D0THINK1, D0SRC), (D0THINK0, D0B0)):
        think_llm, normal = rep.think_llms[cfg.name], rep.llms[cfg.name]
        assert think_llm.requests
        for _, purpose, body in think_llm.requests:
            item, as_of, _ = site(body)
            assert purpose == "write" and item in KAPPA_DESCRIPTIONS
            assert body["chat_template_kwargs"] == {"enable_thinking": True}
            assert "response_format" not in body and body["max_tokens"] == THINK_MAX_TOKENS
            assert as_of >= f"{rep.world.fork:%Y-%m-%d %H:%M}"  # the shared snapshot is not thought
        # no collision: a thinking request is never a request of any non-thinking client
        others = set().union(*(rep.keys(c.name) for c in ALL))
        assert think_llm.keys().isdisjoint(others)
        # the normal client sends the D0 requests minus the definition items' repairs
        kappa_repairs = {
            k
            for k, p, b in rep.llms[plain.name].requests
            if site(b)[0] in KAPPA_DESCRIPTIONS and site(b)[1] >= f"{rep.world.fork:%Y-%m-%d %H:%M}"
        }
        assert normal.keys() == rep.keys(plain.name) - kappa_repairs
        assert len(think_llm.requests) <= len(KAPPA_DESCRIPTIONS) * len(WEEKS)


def test_think_items_record_their_reasoning_and_keep_the_pre_shift_bytes(rep):
    hm, plain = rep.cells[D0THINK1.name], rep.cells[D0SRC.name]
    for slot in rep.world.slots:
        item = hm.items[f"{slot.slot_id}#0"]
        first = item.history[0]
        assert first["event"] == "derive" and "think" not in first  # built by the normal writer
        later = item.history[1:]
        assert all(("think" in h) == bool(slot.kappa_id) for h in later if h["event"] == "repair")
        if slot.kappa_id:
            assert item.note == {"think": {"ok": True, "completion_tokens": 20}}
            assert item.to_json()["note"]["think"]["ok"]
        else:
            assert item.note == {} and "note" not in item.to_json()
            assert item.s_k == plain.items[f"{slot.slot_id}#0"].s_k
    assert rep.stats[D0THINK1.name] == rep.stats[D0SRC.name]


def test_think_answer_cut_off_by_the_token_limit_is_unreadable():
    w = generate_branch(5, "def", world_params(3))
    log = SourceLog(w.events)
    slot = next(s for s in w.slots if s.kappa_id)

    class Cut(FakeLLM):
        def complete(self, messages, max_tokens=512, schema=None, purpose=""):
            text = '<think>maybe {"value": "X", "statement": "s", "citations": []}'
            return LLMResult(text, 10, max_tokens, 0.0, False, "length", "k")

    out = derive(Cut(), w, log, slot, w.fork, think=True)
    assert not out.ok and out.value is None and out.completion_tokens == THINK_MAX_TOKENS
    normal = derive(Cut(), w, log, slot, w.fork)  # not a thinking request: parsed as always
    assert normal.ok and normal.value == "X"

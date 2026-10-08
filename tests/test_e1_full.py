"""The D0full control (prereg v3 section 6): the D=0, B=1 source-epoch cell whose definition items
are re-derived after the shift from every live record of their declared sources instead of a BM25
selection of 6,000 tokens. Each test replays D0B1_source and D0fullB1_source on one small world
with recording fake writers (`tests/e1_replay.py`), which gives the request set of each cell as
response-cache keys."""

import json
import re
from datetime import datetime, timedelta

import pytest

import memshift.holdmem.writer as writer_module
from memshift.analysis.e1 import analyse
from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import parse_cell, parse_cells, run_world
from memshift.experiments.e1 import run as run_e1
from memshift.gen.kappa import KAPPA_DEFS
from memshift.gen.model import fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import (
    FULL_BUDGET,
    WRITER_DOC_CAP,
    FullAccess,
    full_records,
    record_block,
)
from memshift.retrieval import Doc
from memshift.textutil import approx_tokens, truncate_tokens
from tests.e1_replay import Replay, digest, replay
from tests.fakes import FakeLLM

D0SRC = CellConfig(0, 1, "source")
D0FULL = CellConfig(0, 1, "source", full=True)
WEEKS = (1, 2, 4, 8)
KAPPA_DESCRIPTIONS = {spec["description"] for spec in KAPPA_DEFS.values()}
RECORD = re.compile(r"<record id=(\S+) rev=(\d+)>\n(.*?)\n</record>", re.S)
TINY_BUDGET = 700  # approximate tokens: about nine records of the 3-month world


def replay_with_budget(budget: int | None) -> Replay:
    """Replay both cells with `writer.FULL_BUDGET` set to `budget` (None: the registered cap)."""
    saved = writer_module.FULL_BUDGET
    writer_module.FULL_BUDGET = saved if budget is None else budget
    try:
        return replay([D0SRC, D0FULL], branch="def", weeks=WEEKS)
    finally:
        writer_module.FULL_BUDGET = saved


@pytest.fixture(scope="module")
def rep() -> Replay:
    """The cap as registered (the 3-month world's e-mail already exceeds it)."""
    return replay_with_budget(None)


@pytest.fixture(scope="module")
def every() -> Replay:
    """No cap: every in-scope record."""
    return replay_with_budget(10**9)


@pytest.fixture(scope="module")
def cut() -> Replay:
    """A budget far below the records of a definition, so that most definition items are cut."""
    return replay_with_budget(TINY_BUDGET)


def site(body):
    """(item description, as-of time) of a writer request body."""
    text = body["messages"][-1]["content"]
    item, as_of = re.search(r"Knowledge item: (.*)", text), re.search(r"As of: (.*)", text)
    assert item and as_of
    return item.group(1), datetime.strptime(as_of.group(1), "%Y-%m-%d %H:%M")


def records(body) -> list[tuple[str, int, str]]:
    return [(i, int(r), text) for i, r, text in RECORD.findall(body["messages"][-1]["content"])]


def definition_requests(rep: Replay, name: str) -> dict[tuple[str, datetime], dict]:
    """(item, as-of) -> request body of a cell's definition-item writes after the shift."""
    out = {}
    for _, purpose, body in rep.llms[name].requests:
        item, as_of = site(body)
        if purpose == "write" and item in KAPPA_DESCRIPTIONS and as_of >= rep.world.fork:
            out[(item, as_of)] = body
    return out


def in_scope(rep: Replay, slot, t):
    """Every live record of the slot's declared sources at t, oldest first."""
    objs = rep.log.snapshot(t).by_source(*slot.sources)
    return sorted(objs, key=lambda o: (o.updated, o.object_id))


def kappa_slot(rep: Replay, description: str):
    return next(s for s in rep.world.slots if s.description == description)


def events(hm, slot_id):
    return [(h["event"], h["ts"]) for i in hm.by_slot[slot_id] for h in hm.items[i].history]


# ---------------------------------------------------------------------------- configuration
def test_name_token_and_validation():
    assert D0FULL.name == "holdmem_D0fullB1_source"
    assert parse_cell("D0fullB1_source") == D0FULL and parse_cell("D0fullB1") == D0FULL
    assert parse_cells("D0B1_source,D0fullB1_source") == ([D0SRC, D0FULL], ())
    # the tokens that existed keep their meaning
    assert parse_cell("D0thinkB1_source") == CellConfig(0, 1, "source", think=True)
    assert parse_cell("D0tB1_source") == CellConfig(0, 1, "source", present=True)
    assert parse_cell("D1sB1_source") == CellConfig(1, 1, "source", synth=True)
    for bad in (
        "D1fullB1_source",  # D=0 only
        "D0fullB0",  # B=1 only
        "D0fullB1_cited",  # source-epoch key only
        "D0fullB1_gold",
        "D0fullB1_source_refresh",
        "D0fullB1_source_nofb",
    ):
        with pytest.raises(ValueError):
            parse_cell(bad)
    with pytest.raises(ValueError):
        CellConfig(0, 1, "source", full=True, present=True)
    with pytest.raises(ValueError):
        CellConfig(0, 1, "source", full=True, think=True)


def test_full_budget_fits_the_serving_context():
    """28,000 approximate tokens do not fit the 32,768-token server (see `writer.FULL_BUDGET`); the
    cap used leaves room for the request's own text and its 700 output tokens at the highest
    tokens-per-approximate-token ratio measured on the recorded requests (1.19)."""
    prompt_text = 600  # approximate tokens of everything but the records (measured: 330-340)
    assert (FULL_BUDGET + prompt_text) * 1.19 + 700 <= 32_768
    assert (28_000 + prompt_text) * 1.19 + 700 > 32_768


# ---------------------------------------------------------------------------- the requests
def test_definition_items_after_the_shift_get_every_in_scope_record(every):
    rep = every
    full, src = definition_requests(rep, D0FULL.name), definition_requests(rep, D0SRC.name)
    assert set(full) == set(src) and len(full) > 0
    strictly_more = 0
    for (item, t), body in full.items():
        slot = kappa_slot(rep, item)
        expected = [
            (o.object_id, o.revision, truncate_tokens(o.text, WRITER_DOC_CAP))
            for o in in_scope(rep, slot, t)
        ]
        assert records(body) == expected, (item, t)  # all of them, oldest first
        assert body["max_tokens"] == 700 and body["response_format"]  # the writer's JSON schema
        ids_src = {i for i, _, _ in records(src[(item, t)])}
        assert ids_src <= {i for i, _, _ in expected}  # the BM25 selection is a subset
        strictly_more += len(expected) > len(ids_src)
    assert strictly_more > 0  # the selection left records out: the control changes something


def test_the_request_differs_from_d0_only_in_its_records(rep):
    full, src = definition_requests(rep, D0FULL.name), definition_requests(rep, D0SRC.name)
    head = lambda body: RECORD.split(body["messages"][-1]["content"])[0]  # noqa: E731
    tail = lambda body: body["messages"][-1]["content"].rsplit("</record>", 1)[-1]  # noqa: E731
    for key, body in full.items():
        other = src[key]
        assert body["messages"][0] == other["messages"][0]  # same system message
        assert head(body) == head(other) and tail(body) == tail(other)
        assert {k: v for k, v in body.items() if k != "messages"} == {
            k: v for k, v in other.items() if k != "messages"
        }


def test_other_items_and_pre_shift_requests_are_those_of_d0_source(rep):
    src, full = rep.keys(D0SRC.name, "write"), rep.keys(D0FULL.name, "write")
    assert rep.keys(D0SRC.name) == src and rep.keys(D0FULL.name) == full  # writer requests only
    by_key = {k: b for n in (D0SRC, D0FULL) for k, _, b in rep.llms[n.name].requests}
    new, gone = full - src, src - full
    assert new and gone
    for key in new | gone:  # everything that differs is a definition item written after the shift
        item, as_of = site(by_key[key])
        assert item in KAPPA_DESCRIPTIONS and as_of >= rep.world.fork, (item, as_of)
    pre = lambda keys: {k for k in keys if site(by_key[k])[1] < rep.world.fork}  # noqa: E731
    assert pre(src) == pre(full) and pre(src)
    assert len(new) == len(gone)  # one full request per definition request it replaces


def test_the_pre_shift_snapshot_is_the_same_bytes(rep):
    assert rep.contexts_t0[D0FULL.name] == rep.contexts_t0[D0SRC.name]
    src, full = rep.cells[D0SRC.name], rep.cells[D0FULL.name]
    for slot in rep.world.slots:
        a, b = src.items[f"{slot.slot_id}#0"], full.items[f"{slot.slot_id}#0"]
        first = lambda item: item.history[0]  # noqa: E731
        assert first(a) == first(b) and first(b)["event"] == "derive" and "full" not in first(b)


def test_detection_and_keys_are_those_of_d0_source(rep):
    src, full = rep.cells[D0SRC.name], rep.cells[D0FULL.name]
    assert rep.stats[D0FULL.name] == rep.stats[D0SRC.name]
    for slot in rep.world.slots:
        assert events(full, slot.slot_id) == events(src, slot.slot_id), slot.slot_id
        assert full.items[f"{slot.slot_id}#0"].bind == src.items[f"{slot.slot_id}#0"].bind
    assert sum(s["repaired"] for s in rep.stats[D0FULL.name]) > 0


def test_other_items_are_written_exactly_as_in_d0_source(rep):
    src, full = rep.cells[D0SRC.name], rep.cells[D0FULL.name]
    for slot in rep.world.slots:
        a, b = src.items[f"{slot.slot_id}#0"], full.items[f"{slot.slot_id}#0"]
        if not slot.kappa_id:
            assert (a.s_k, a.readset, a.refs, a.eval_value) == (
                b.s_k,
                b.readset,
                b.refs,
                b.eval_value,
            )
            assert a.history == b.history and not b.note and "note" not in b.to_json()
        else:
            assert len(b.readset) >= len(a.readset)
    assert any(
        len(full.items[f"{s.slot_id}#0"].readset) > len(src.items[f"{s.slot_id}#0"].readset)
        for s in rep.world.slots
        if s.kappa_id
    )


def test_the_cell_reads_no_evaluator_gold_or_scope_range(monkeypatch):
    world = generate_branch(5, "def", world_params(3))
    log = SourceLog(world.events)

    def boom(*args, **kwargs):
        raise AssertionError("the benchmark's own logic was read by the D0full write path")

    import memshift.gen.kappa as kappa

    monkeypatch.setattr(kappa, "evaluate", boom)
    monkeypatch.setattr(kappa, "EVALUATORS", {k: boom for k in kappa.EVALUATORS})
    monkeypatch.setattr(kappa, "SCOPE_RANGES", {k: boom for k in kappa.SCOPE_RANGES})
    for name in ("gold_at", "scope_members"):
        monkeypatch.setattr(f"memshift.holdmem.system.{name}", boom)
    monkeypatch.setattr("memshift.holdmem.checker.evaluate", boom)
    hm = HoldMem(world, log, D0FULL, FakeLLM(), "fake/writer-a")
    hm.workers = 4
    hm.build(world.fork - timedelta(minutes=1))
    for k in (1, 2):
        hm.maintain(world.fork + timedelta(weeks=k))
    assert any("full" in h for i in hm.items.values() for h in i.history)


# ---------------------------------------------------------------------------- the record of access
def cost_of(o) -> int:
    text = truncate_tokens(o.text, WRITER_DOC_CAP)
    return approx_tokens(record_block(Doc(o.object_id, text, o.updated, o.revision), text))


def assert_selection(everything, given: list[str], budget: int, truncation: str) -> None:
    """`given` (record ids, in the order sent) is what the per-source rule says for `everything`
    (every in-scope record, oldest first): whole when it fits; else the largest source cut to its
    newest run and the others whole; else, when the others alone exceed the budget, the newest run
    of all records together. `truncation` names the case."""
    cost = {o.object_id: cost_of(o) for o in everything}
    ids = {o.object_id for o in everything}
    assert given == [o.object_id for o in everything if o.object_id in set(given)]  # time order
    assert set(given) <= ids and len(set(given)) == len(given)
    used = sum(cost[i] for i in given)
    total = sum(cost.values())
    if total <= budget:
        assert truncation == "none" and set(given) == ids
        return
    assert used <= budget
    by_source: dict[str, int] = {}
    for o in everything:
        by_source[o.source] = by_source.get(o.source, 0) + cost[o.object_id]
    largest = max(by_source, key=lambda s: (by_source[s], s))
    others = total - by_source[largest]
    if others <= budget:
        assert truncation == "source"
        assert all(o.object_id in given for o in everything if o.source != largest)
        mine = [o for o in everything if o.source == largest]
        kept = [o for o in mine if o.object_id in given]
        assert len(kept) < len(mine) and kept == mine[len(mine) - len(kept) :]  # the newest run
        assert (
            used + cost[mine[len(mine) - len(kept) - 1].object_id] > budget
        )  # the next did not fit
    else:
        assert truncation == "newest"
        n = len(given)
        assert given == [o.object_id for o in everything][len(everything) - n :]  # the newest run
        assert n < len(everything)
        assert used + cost[everything[len(everything) - n - 1].object_id] > budget


def assert_note(note: dict, everything, given: list[str], budget: int) -> None:
    by_source = note["by_source"]
    assert note["records"] == len(given) and note["records"] + note["dropped"] == len(everything)
    assert note["tokens"] == sum(cost_of(o) for o in everything if o.object_id in set(given))
    assert note["tokens"] <= budget
    assert note["capped"] == sum(
        truncate_tokens(o.text, WRITER_DOC_CAP) != o.text for o in everything
    )
    for source, n in by_source.items():
        mine = [o for o in everything if o.source == source]
        assert n["records"] == sum(o.object_id in set(given) for o in mine)
        assert n["records"] + n["dropped"] == len(mine)
    assert sum(n["dropped"] for n in by_source.values()) == note["dropped"]
    assert note["truncation"] in ("none", "source", "newest")
    assert (note["dropped"] > 0) == (note["truncation"] != "none")


def check_replay(replay_: Replay, budget: int) -> set[str]:
    """Every definition request of the full cell and its item note follow the rule at `budget`;
    returns the cases seen."""
    hm = replay_.cells[D0FULL.name]
    seen = set()
    for (item, t), body in definition_requests(replay_, D0FULL.name).items():
        slot = kappa_slot(replay_, item)
        everything = in_scope(replay_, slot, t)
        given = [i for i, _, _ in records(body)]
        note = next(
            h["full"]
            for i in hm.by_slot[slot.slot_id]
            for h in hm.items[i].history
            if h["event"] == "repair" and h["ts"] == fmt_ts(t)
        )
        assert_selection(everything, given, budget, note["truncation"])
        assert_note(note, everything, given, budget)
        seen.add(note["truncation"])
    return seen


def test_the_registered_cap_cuts_only_what_exceeds_it_and_the_small_sources_stay_whole(rep):
    seen = check_replay(rep, FULL_BUDGET)
    assert seen == {"none", "source"}  # e-mail is cut in a 3-month world, tasks are not


def test_a_tight_budget_cuts_the_largest_source_or_falls_back_to_the_newest_records(cut):
    seen = check_replay(cut, TINY_BUDGET)
    assert seen == {"source", "newest"}  # the fallback is its own case
    hm = cut.cells[D0FULL.name]
    for slot in (s for s in cut.world.slots if s.kappa_id):
        item = hm.items[f"{slot.slot_id}#0"]
        rows = [h for h in item.history if "full" in h]
        assert item.note == {"full": rows[-1]["full"]}  # the item keeps its latest re-derivation
        assert item.to_json()["note"] == item.note
        assert all(h["event"] == "repair" for h in rows)
    assert hm.cost["maintain_tokens"] > 0


def test_full_records_follow_the_rule_at_every_budget():
    world = generate_branch(5, "def", world_params(3))
    log = SourceLog(world.events)
    t = world.fork + timedelta(weeks=8)
    seen = set()
    for slot in (s for s in world.slots if s.kappa_id):
        everything = sorted(
            log.snapshot(t).by_source(*slot.sources), key=lambda o: (o.updated, o.object_id)
        )
        for budget in (0, 1, 60, 300, 700, 1_500, 3_000, 6_000, 25_000, 10**9):
            acc = full_records(log, slot, t, budget)
            given = [d.doc_id for d, _ in acc.records]
            assert_selection(everything, given, budget, acc.truncation)
            assert_note(acc.note(), everything, given, budget)
            assert set(acc.by_source) == set(slot.sources)
            seen.add(acc.truncation)
        assert full_records(log, slot, t, 10**9).dropped == 0
        assert full_records(log, slot, t, 0).records == []
        assert isinstance(full_records(log, slot, t), FullAccess)  # the default cap
    assert seen == {"none", "source", "newest"}


def test_the_fallback_is_counted_apart_and_needs_the_other_sources_to_exceed_the_budget():
    world = generate_branch(5, "def", world_params(3))
    log = SourceLog(world.events)
    t = world.fork + timedelta(weeks=8)
    slot = next(s for s in world.slots if s.kappa_id == "vip_sender")
    cards = sum(cost_of(o) for o in log.snapshot(t).by_source("contacts"))
    assert full_records(log, slot, t, cards - 1).truncation == "newest"
    whole = full_records(log, slot, t, cards)  # the cards exactly fit: the e-mail gets no room
    assert whole.truncation == "source" and whole.by_source["email"]["records"] == 0
    assert whole.by_source["contacts"]["dropped"] == 0
    assert full_records(log, slot, t, cards - 1).by_source["contacts"]["dropped"] > 0


def test_vip_sender_keeps_all_contact_cards_and_cuts_only_the_emails(rep):
    hm = rep.cells[D0FULL.name]
    slot = next(s for s in rep.world.slots if s.kappa_id == "vip_sender")
    rows = [h["full"] for i in hm.by_slot[slot.slot_id] for h in hm.items[i].history if "full" in h]
    assert rows
    for (item, t), body in definition_requests(rep, D0FULL.name).items():
        if item != slot.description:
            continue
        cards = [o.object_id for o in rep.log.snapshot(t).by_source("contacts")]
        given = {i for i, _, _ in records(body)}
        assert cards and set(cards) <= given  # every contact card is in the request
    for n in rows:
        assert n["truncation"] == "source"
        assert n["by_source"]["contacts"]["dropped"] == 0
        assert (
            n["by_source"]["email"]["dropped"] > 0
            and n["dropped"] == n["by_source"]["email"]["dropped"]
        )


def test_vip_sender_keeps_all_contact_cards_in_a_12_month_world():
    world = generate_branch(1000, "def", world_params(12))
    log = SourceLog(world.events)
    slot = next(s for s in world.slots if s.kappa_id == "vip_sender")
    for k in WEEKS:
        t = world.fork + timedelta(weeks=k)
        acc = full_records(log, slot, t)
        assert acc.truncation == "source" and acc.tokens <= FULL_BUDGET
        assert acc.by_source["contacts"] == {
            "records": len(log.snapshot(t).by_source("contacts")),
            "dropped": 0,
        }
        assert (
            acc.by_source["email"]["dropped"] > 0
            and acc.dropped == acc.by_source["email"]["dropped"]
        )
        newest = max(log.snapshot(t).by_source("email"), key=lambda o: (o.updated, o.object_id))
        assert newest.object_id in {d.doc_id for d, _ in acc.records}  # the newest e-mail is given


def test_a_record_longer_than_the_document_cap_is_cut_and_counted():
    world = generate_branch(5, "def", world_params(3))
    log = SourceLog(world.events)
    slot = next(s for s in world.slots if s.kappa_id == "vip_sender")
    acc = full_records(log, slot, world.fork)
    long_text = sum(
        approx_tokens(o.text) > WRITER_DOC_CAP
        for o in log.snapshot(world.fork).by_source(*slot.sources)
    )
    assert acc.capped == long_text
    assert all(approx_tokens(text) <= WRITER_DOC_CAP for _, text in acc.records)


# ---------------------------------------------------------------------------- the driver
@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("full")
    cells, _ = parse_cells("D0B1_source,D0fullB1_source,D1B1_source")
    summary = run_world(
        5,
        out,
        branches=("def",),
        baselines=True,
        writer=FakeLLM(),
        cells_cfg=cells,
        world_params=world_params(3),
        discriminative=True,
    )
    return out, summary, cells


def test_the_driver_asserts_the_pre_shift_hash_across_the_two_cells(run):
    out, summary, cells = run
    assert summary["branches"]["def"]["hash_ok"]
    check = json.loads(next(out.glob("hash_check_*_def.json")).read_text())
    assert check["ok"] and check["mismatches"] == [] and check["probes"] > 0
    assert {r["system"] for r in read_jsonl(out / "answers.jsonl")} >= {c.name for c in cells}


def test_the_items_file_carries_the_access_record_and_the_analysis_reads_the_cell(run):
    out, _, cells = run
    rows = [r for r in read_jsonl(out / "items.jsonl") if r["system"] == D0FULL.name]
    definition = [r for r in rows if r["slot_id"].startswith("k.")]
    assert definition and all(r["note"]["full"]["records"] > 0 for r in definition)
    assert not any("note" in r for r in rows if not r["slot_id"].startswith("k."))
    metrics = analyse(out)["cells"]["def"]
    assert metrics[D0FULL.name]["Update"]["n_items"] > 0
    assert metrics[D0FULL.name]["MissedInvalidation"] == metrics[D0SRC.name]["MissedInvalidation"]


def test_the_manifest_records_the_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memshift.experiments.e1.client", lambda model, run_dir, **kw: FakeLLM(model)
    )
    cells, _ = parse_cells("D0B1_source,D0fullB1_source")
    out = run_e1(
        [5],
        "full",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
    )
    params = json.loads((out / "manifest.json").read_text())["params"]
    assert params["full_writer"] == {"budget_tokens": FULL_BUDGET, "doc_cap_tokens": WRITER_DOC_CAP}
    assert "holdmem_D0fullB1_source" in params["cells"]
    (tmp_path / "plain").mkdir()
    plain = run_e1(
        [5],
        "plain",
        out=tmp_path / "plain",
        cells_cfg=[D0SRC],
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
    )
    assert "full_writer" not in json.loads((plain / "manifest.json").read_text())["params"]


# ---------------------------------------------------------------------------- pin
PIN = {
    "requests": "95bd05e3caa1963f31846c13da07b000289b450a378cedd709a2f9bab0d8d5c3",
    "items": "d25239371e912d46528a480d1d6e7269a430ca29e9e6e60796c4b2895a760d51",
}


def test_the_cell_is_pinned(rep):
    """The requests and item rows of the cell on the small world: a change here means a cached run
    of the cell would no longer be a hit, or the cell behaves differently from the registered
    one."""
    requests = sorted(rep.llms[D0FULL.name].keys())
    items = [i.to_json() for i in rep.cells[D0FULL.name].items.values()]
    assert digest(requests) == PIN["requests"]
    assert digest(items) == PIN["items"]
    assert fmt_ts(rep.t0)

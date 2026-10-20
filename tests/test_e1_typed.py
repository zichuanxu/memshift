"""The D0typed control: the D=0, B=1 source-epoch cell whose definition items
are re-derived after the shift from the typed records a D_m program reads (columnar JSON, without
the rendered text and the bodies of e-mail and chat messages, never truncated) instead of a BM25
selection of 6,000 tokens. The cell tests replay D0B1_source and D0typedB1_source on one small world
with recording fake writers (`tests/e1_replay.py`); the driver tests run `e1.run_world` with fake
writers, and a collecting Claude client checks what the Sonnet arm would send to the batch API."""

import json
import re
from datetime import datetime, timedelta

import httpx
import pytest

from memshift.analysis.e1 import analyse
from memshift.experiments.common import cache_path, read_jsonl, world_params
from memshift.experiments.e1 import (
    DEFWRITER_BODY,
    DEFWRITER_MAX_TOKENS,
    parse_cell,
    parse_cells,
    run_world,
)
from memshift.experiments.e1 import run as run_e1
from memshift.gen.kappa import KAPPA_DEFS
from memshift.gen.model import fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem import writer
from memshift.holdmem.program import fields_text, program_records
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import (
    TYPED_FIELDS,
    TYPED_HEAD,
    derive,
    typed_fields_text,
    typed_records,
    writer_request,
)
from memshift.llm import COLLECT_ENV, LLMClient
from memshift.textutil import approx_tokens
from tests.e1_replay import Replay, digest, replay
from tests.fakes import FakeLLM, RecordingLLM

D0SRC = CellConfig(0, 1, "source")
D0TYPED = CellConfig(0, 1, "source", typed=True)
WEEKS = (1, 2, 4, 8)
KAPPA_DESCRIPTIONS = {spec["description"] for spec in KAPPA_DEFS.values()}
SONNET = "claude-sonnet-5-5"
REPLY = '{"value": "X", "statement": "The value is X.", "citations": []}'


@pytest.fixture(scope="module")
def rep() -> Replay:
    return replay([D0SRC, D0TYPED], branch="def", weeks=WEEKS)


def site(body):
    """(item description, as-of time) of a writer request body."""
    text = body["messages"][-1]["content"]
    item, as_of = re.search(r"Knowledge item: (.*)", text), re.search(r"As of: (.*)", text)
    assert item and as_of
    return item.group(1), datetime.strptime(as_of.group(1), "%Y-%m-%d %H:%M")


def definition_requests(rep: Replay, name: str) -> dict[tuple[str, datetime], dict]:
    """(item, as-of) -> request body of a cell's definition-item writes after the shift."""
    out = {}
    for _, purpose, body in rep.llms[name].requests:
        item, as_of = site(body)
        if purpose == "write" and item in KAPPA_DESCRIPTIONS and as_of >= rep.world.fork:
            out[(item, as_of)] = body
    return out


def kappa_slot(world, description: str):
    return next(s for s in world.slots if s.description == description)


def events(hm, slot_id):
    return [(h["event"], h["ts"]) for i in hm.by_slot[slot_id] for h in hm.items[i].history]


def tables_of(body) -> list[dict]:
    """The typed tables of a request (the JSON lines between the field description and TASK)."""
    text = body["messages"][-1]["content"]
    return [json.loads(line) for line in text.splitlines() if line.startswith('{"source":')]


# ---------------------------------------------------------------------------- configuration
def test_name_token_and_validation():
    assert D0TYPED.name == "holdmem_D0typedB1_source"
    assert parse_cell("D0typedB1_source") == D0TYPED and parse_cell("D0typedB1") == D0TYPED
    assert parse_cells("D0B1_source,D1B1_source,D0typedB1_source") == (
        [D0SRC, CellConfig(1, 1, "source"), D0TYPED],
        (),
    )
    # the tokens that existed keep their meaning
    assert parse_cell("D0tB1_source") == CellConfig(0, 1, "source", present=True)
    assert parse_cell("D0thinkB1_source") == CellConfig(0, 1, "source", think=True)
    assert parse_cell("D0fullB1_source") == CellConfig(0, 1, "source", full=True)
    assert parse_cell("D1sB1_source") == CellConfig(1, 1, "source", synth=True)
    for bad in (
        "D1typedB1_source",  # D=0 only
        "D0typedB0",  # B=1 only
        "D0typedB1_cited",  # source-epoch key only
        "D0typedB1_scope",
        "D0typedB1_source_refresh",
        "D0typedB1_source_nofb",
        "D0typedB0_upsert",
    ):
        with pytest.raises(ValueError):
            parse_cell(bad)
    for other in ("present", "think", "full"):
        with pytest.raises(ValueError):
            CellConfig(0, 1, "source", typed=True, **{other: True})


# ---------------------------------------------------------------------------- the typed input
@pytest.fixture(scope="module")
def small():
    world = generate_branch(5, "def", world_params(3))
    return world, SourceLog(world.events)


def test_the_typed_input_is_every_program_record_in_order_without_text_or_message_bodies(small):
    world, log = small
    seen = set()
    for k in (0, *WEEKS):
        t = world.fork + timedelta(weeks=k)
        for slot in (s for s in world.slots if s.kappa_id):
            recs = program_records(log, slot, t)
            given = typed_records(log, slot, t)
            assert [tab["source"] for tab in given.tables] == list(slot.sources)
            # every record, in program_records order, no cap
            assert given.readset == [(r["object_id"], r["revision"]) for r in recs]
            rows = [
                dict(zip(tab["fields"], row, strict=True))
                for tab in given.tables
                for row in tab["rows"]
            ]
            assert len(rows) == len(recs)
            for r, row in zip(recs, rows, strict=True):
                s = r["source"]
                kept = {f: v for f, v in r.items() if f not in ("source", "text")}
                if s in ("email", "messages"):
                    assert r["body"] and "body" not in row  # the message bodies are dropped
                    kept.pop("body")
                assert {f: v for f, v in row.items() if v is not None} == {
                    f: v for f, v in kept.items() if v is not None
                }
                assert set(row) - set(kept) == {f for f, v in row.items() if v is None}
                if s == "notes":
                    assert row["body"] == r["body"]  # note bodies stay (working_hours v1)
                seen.add(s)
            for tab in given.tables:
                assert tab["fields"] == list(TYPED_FIELDS[tab["source"]])  # no field beyond them
                assert "text" not in tab["fields"] and "source" not in tab["fields"]
            note = given.note()
            assert note["records"] == len(recs) and note["chars"] == len(given.block())
            assert note["tokens"] == approx_tokens(given.block())
            assert note["by_source"] == {
                s: sum(r["source"] == s for r in recs) for s in slot.sources
            }
    assert seen == {"calendar", "messages", "tasks", "notes", "email", "contacts"}


def test_the_field_description_omits_exactly_the_dropped_fields():
    every = ("contacts", "calendar", "tasks", "email", "messages", "notes")
    text = typed_fields_text(every)
    plain = fields_text(every)
    assert "body (str)" in text  # the notes
    assert text.count("body (str)") == plain.count("body (str)") - 2
    assert "text (str" not in text and "source (str)" not in text
    for line in plain.splitlines()[1:]:
        assert line.replace("body (str), ", "") in text.splitlines()
    for s, fields in TYPED_FIELDS.items():  # every column is described
        for f in fields[4:]:
            text = typed_fields_text((s,))
            assert f"{f} (" in text or f"`{f}`" in text, (s, f)


def test_the_typed_request_differs_from_d0_only_in_its_record_block(rep):
    typed, src = definition_requests(rep, D0TYPED.name), definition_requests(rep, D0SRC.name)
    assert set(typed) == set(src) and typed
    for key, body in typed.items():
        other = src[key]
        mine, theirs = body["messages"][-1]["content"], other["messages"][-1]["content"]
        head = "\n\nSource records"
        assert mine.split(head)[0] == theirs.split(head)[0]  # item, format, as-of, definition
        assert mine.rsplit("\n\n", 1)[1] == theirs.rsplit("\n\n", 1)[1] == writer.TASK
        assert body["messages"][0] == other["messages"][0]  # same system message
        assert {k: v for k, v in body.items() if k != "messages"} == {
            k: v for k, v in other.items() if k != "messages"
        }  # max_tokens 700, the writer's JSON schema
        assert body["max_tokens"] == 700 and body["response_format"]
        assert TYPED_HEAD in mine and "<record id=" not in mine
        slot = kappa_slot(rep.world, key[0])
        given = typed_records(rep.log, slot, key[1])
        assert tables_of(body) == given.tables
        assert mine.split(head)[1].startswith(TYPED_HEAD.removeprefix("Source records"))


def test_other_items_and_pre_shift_requests_are_those_of_d0_source(rep):
    src, typed = rep.keys(D0SRC.name, "write"), rep.keys(D0TYPED.name, "write")
    assert rep.keys(D0SRC.name) == src and rep.keys(D0TYPED.name) == typed  # writer requests only
    by_key = {k: b for n in (D0SRC, D0TYPED) for k, _, b in rep.llms[n.name].requests}
    new, gone = typed - src, src - typed
    assert new and gone and len(new) == len(gone)
    for key in new | gone:  # everything that differs is a definition item written after the shift
        item, as_of = site(by_key[key])
        assert item in KAPPA_DESCRIPTIONS and as_of >= rep.world.fork, (item, as_of)
    for key in new:
        assert TYPED_HEAD in by_key[key]["messages"][-1]["content"]
    pre = lambda keys: {k for k in keys if site(by_key[k])[1] < rep.world.fork}  # noqa: E731
    assert pre(src) == pre(typed) and pre(src)


def test_the_pre_shift_snapshot_detection_and_keys_are_those_of_d0_source(rep):
    assert rep.contexts_t0[D0TYPED.name] == rep.contexts_t0[D0SRC.name]
    src, typed = rep.cells[D0SRC.name], rep.cells[D0TYPED.name]
    assert rep.stats[D0TYPED.name] == rep.stats[D0SRC.name]
    assert sum(s["repaired"] for s in rep.stats[D0TYPED.name]) > 0
    for slot in rep.world.slots:
        a, b = src.items[f"{slot.slot_id}#0"], typed.items[f"{slot.slot_id}#0"]
        assert a.history[0] == b.history[0] and b.history[0]["event"] == "derive"
        assert events(typed, slot.slot_id) == events(src, slot.slot_id), slot.slot_id
        assert a.bind == b.bind
        if not slot.kappa_id:
            assert (a.s_k, a.readset, a.refs, a.eval_value, a.history) == (
                b.s_k,
                b.readset,
                b.refs,
                b.eval_value,
                b.history,
            )
            assert not b.note and "note" not in b.to_json()


def test_each_typed_write_records_what_was_given_and_its_outcome(rep):
    hm = rep.cells[D0TYPED.name]
    n = 0
    for slot in (s for s in rep.world.slots if s.kappa_id):
        item = hm.items[f"{slot.slot_id}#0"]
        rows = [h for h in item.history if "typed" in h]
        assert all(h["event"] == "repair" for h in rows)
        for h in rows:
            t = datetime.strptime(h["ts"], "%Y-%m-%dT%H:%M")
            given = typed_records(rep.log, slot, t)
            assert h["typed"] == {
                **given.note(),
                "ok": True,
                "at_limit": False,
                "completion_tokens": 20,
            }
            n += 1
        if rows:
            assert item.note == {"typed": rows[-1]["typed"]} and item.to_json()["note"] == item.note
            # the read set is every record given
            t = datetime.strptime(rows[-1]["ts"], "%Y-%m-%dT%H:%M")
            assert item.readset == typed_records(rep.log, slot, t).readset
    assert n == len(definition_requests(rep, D0TYPED.name))


def test_the_cell_reads_no_evaluator_gold_or_scope_range(monkeypatch, small):
    world, log = small

    def boom(*args, **kwargs):
        raise AssertionError("the benchmark's own logic was read by the D0typed write path")

    import memshift.gen.kappa as kappa

    monkeypatch.setattr(kappa, "evaluate", boom)
    monkeypatch.setattr(kappa, "EVALUATORS", {k: boom for k in kappa.EVALUATORS})
    monkeypatch.setattr(kappa, "SCOPE_RANGES", {k: boom for k in kappa.SCOPE_RANGES})
    for name in ("gold_at", "scope_members"):
        monkeypatch.setattr(f"memshift.holdmem.system.{name}", boom)
    monkeypatch.setattr("memshift.holdmem.checker.evaluate", boom)
    hm = HoldMem(world, SourceLog(world.events), D0TYPED, FakeLLM(), "fake/writer-a")
    hm.workers = 4
    hm.build(world.fork - timedelta(minutes=1))
    for k in (1, 2):
        hm.maintain(world.fork + timedelta(weeks=k))
    assert any("typed" in h for i in hm.items.values() for h in i.history)


def test_typed_input_refuses_a_delta_or_the_full_control(small):
    world, log = small
    slot = next(s for s in world.slots if s.kappa_id)
    with pytest.raises(ValueError):
        derive(FakeLLM(), world, log, slot, world.fork, since=world.start, typed=True)
    with pytest.raises(ValueError):
        derive(FakeLLM(), world, log, slot, world.fork, full=True, typed=True)


def test_the_request_without_typed_input_is_unchanged(small):
    """`writer_request` and `derive` without `typed` give the request they gave before."""
    world, log = small
    slot = next(s for s in world.slots if s.kappa_id)
    t = world.fork + timedelta(weeks=1)
    recs = writer.records_for(log, slot, t)
    a = writer_request(world, slot, t, recs)
    b = writer_request(world, slot, t, recs, typed=None)
    assert a == b and "Source records (each starts with its id and revision):" in str(a)


# ---------------------------------------------------------------------------- the driver
CELLS = "D0B1_source,D1B1_source,D0typedB1_source"


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("typed")
    cells, _ = parse_cells(CELLS)
    summary = run_world(
        5,
        out,
        branches=("src", "def"),
        baselines=False,
        writer=FakeLLM(),
        cells_cfg=cells,
        world_params=world_params(3),
        discriminative=True,
    )
    return out, summary, cells


def test_the_driver_asserts_the_pre_shift_hash_and_writes_the_t0_diagnostic(run):
    out, summary, cells = run
    for b in ("src", "def"):
        assert summary["branches"][b]["hash_ok"]
        check = json.loads(next(out.glob(f"hash_check_*_{b}.json")).read_text())
        assert check["ok"] and check["mismatches"] == [] and check["probes"] > 0
    answered = {r["system"] for r in read_jsonl(out / "answers.jsonl")}
    assert answered == {c.name for c in cells} | {"holdmem_json_arm"}
    rows = read_jsonl(out / "typed_t0.jsonl")
    world = generate_branch(5, "def", world_params(3))
    defs = [s.slot_id for s in world.slots if s.kappa_id]
    assert sorted((r["branch"], r["fact_id"]) for r in rows) == sorted(
        (b, f) for b in ("src", "def") for f in defs
    )
    for r in rows:
        assert r["model"] == "fake/writer-a" and r["ok"] and r["value"] == "X"
        assert r["correct"] is False  # the fake writer's "X" is never a definition's value
        assert r["given"]["records"] > 0 and r["as_of"] == fmt_ts(world.fork - timedelta(minutes=1))


def test_the_items_file_carries_the_typed_record_and_the_analysis_reads_the_cell(run):
    out, _, _ = run
    rows = [r for r in read_jsonl(out / "items.jsonl") if r["system"] == D0TYPED.name]
    definition = [r for r in rows if r["slot_id"].startswith("k.")]
    assert any("typed" in r.get("note", {}) for r in definition)
    assert not any("note" in r for r in rows if not r["slot_id"].startswith("k."))
    metrics = analyse(out)["cells"]["def"]
    assert metrics[D0TYPED.name]["Update"]["n_items"] > 0
    assert metrics[D0TYPED.name]["MissedInvalidation"] == metrics[D0SRC.name]["MissedInvalidation"]


def test_the_manifest_records_the_typed_input(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "memshift.experiments.e1.client", lambda model, run_dir, **kw: FakeLLM(model)
    )
    cells, _ = parse_cells("D0B1_source,D0typedB1_source")
    out = run_e1(
        [5],
        "typed",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
    )
    params = json.loads((out / "manifest.json").read_text())["params"]
    assert params["typed_writer"]["omitted"] == {
        "all": ["source", "text"],
        "email": ["body"],
        "messages": ["body"],
    }
    assert "holdmem_D0typedB1_source" in params["cells"] and "full_writer" not in params
    assert (out / "typed_t0.jsonl").exists()


# ---------------------------------------------------------------------------- the Sonnet arm
def run_cells(out, spec: str, def_writer, writer_=None):
    """One branch of the small world with `def_writer` as the definition-item writer (H-a)."""
    cells, _ = parse_cells(spec)
    return run_world(
        5,
        out,
        branches=("def",),
        baselines=False,
        writer=writer_ or RecordingLLM(),
        cells_cfg=cells,
        world_params=world_params(3),
        discriminative=True,
        defwriter=SONNET,
        def_writer=def_writer,
    )


def test_with_a_definition_writer_the_typed_cell_adds_only_typed_requests(tmp_path):
    """H-a design: the typed re-derivations and the t0 diagnostic are the definition writer's, with
    its output limit; every other definition-writer request is one the H-a cells send, and the
    served writer sends no typed request."""
    world = generate_branch(5, "def", world_params(3))
    ha, main_ha = RecordingLLM(SONNET, extra_body=DEFWRITER_BODY), RecordingLLM()
    tt, main_tt = RecordingLLM(SONNET, extra_body=DEFWRITER_BODY), RecordingLLM()
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run_cells(tmp_path / "a", "D0B1_source,D1B1_source", ha, main_ha)
    run_cells(tmp_path / "b", CELLS, tt, main_tt)
    new = tt.keys() - ha.keys()
    assert ha.keys() <= tt.keys() and new
    bodies = {k: b for k, _, b in tt.requests}
    assert all(TYPED_HEAD in bodies[k]["messages"][-1]["content"] for k in new)
    assert all(TYPED_HEAD not in bodies[k]["messages"][-1]["content"] for k in ha.keys(None))
    assert all(bodies[k]["max_tokens"] == DEFWRITER_MAX_TOKENS for k in new)
    t0 = {k for k in new if site(bodies[k])[1] < world.fork}  # the diagnostic
    assert len(t0) == sum(1 for s in world.slots if s.kappa_id)
    assert main_ha.keys() <= main_tt.keys()
    assert not any(
        TYPED_HEAD in b["messages"][-1]["content"] for _, p, b in main_tt.requests if p == "write"
    )


def message(text: str) -> dict:
    return {
        "id": "msg_1",
        "model": SONNET,
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": text}],
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }


def refuse(request: httpx.Request) -> httpx.Response:
    raise AssertionError(f"no request may be sent: {request.url}")


def test_a_collecting_sonnet_run_records_exactly_the_typed_requests(tmp_path, monkeypatch):
    """The Sonnet arm's batch flow on the small world: with the H-a definition writes in the cache,
    a collecting run sends nothing and records only the typed requests (re-derivations and t0
    diagnostic), whose writes it treats as unreadable; once their results are in the cache, the
    final run sends nothing."""
    monkeypatch.setenv("CLAUDE_API_KEY", "test-key")
    monkeypatch.setattr("memshift.llm._COLLECTED", set())
    cache = tmp_path / cache_path(SONNET).name

    def sonnet(handler) -> LLMClient:
        c = LLMClient(
            "https://api.anthropic.com", SONNET, cache_path=cache, extra_body=dict(DEFWRITER_BODY)
        )
        c._http = httpx.Client(transport=httpx.MockTransport(handler))
        return c

    sent: list[dict] = []

    def api(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=message(REPLY))

    for d in ("ha", "collect", "final"):
        (tmp_path / d).mkdir()
    run_cells(tmp_path / "ha", "D0B1_source,D1B1_source", sonnet(api))  # the committed H-a run
    assert sent
    collected = tmp_path / "collected.jsonl"
    monkeypatch.setenv(COLLECT_ENV, str(collected))
    c = sonnet(refuse)
    summary = run_cells(tmp_path / "collect", CELLS, c)
    rows = read_jsonl(collected)
    assert rows and all(TYPED_HEAD in r["params"]["messages"][-1]["content"] for r in rows)
    assert all(r["model"] == SONNET and r["params"]["max_tokens"] == 17000 for r in rows)
    items = read_jsonl(tmp_path / "collect" / "items.jsonl")
    notes = [h["typed"] for i in items for h in i["history"] if "typed" in h]
    t0 = read_jsonl(tmp_path / "collect" / "typed_t0.jsonl")
    assert len(rows) == len(notes) + len(t0) and notes and t0  # one distinct request each
    # a collected request is an unreadable write: the run completes, its typed items are void
    assert not any(n["ok"] for n in notes) and not any(r["ok"] for r in t0)
    stats = summary["branches"]["def"]["defwriter"][D0TYPED.name]
    assert stats["writes"] == stats["unreadable"] == len(notes)
    for name in ("holdmem_D0B1_source", "holdmem_D1B1_source"):
        assert summary["branches"]["def"]["defwriter"][name]["unreadable"] == 0  # cache hits
    # the batch results go into the cache (`claude_batch.collect` writes the online row)
    from dataclasses import asdict

    from memshift.llm import ResponseCache, anthropic_message_result

    store = ResponseCache(cache)
    for r in rows:
        res, meta = anthropic_message_result(message(REPLY), r["key"], 0.0)
        store.put(
            r["key"], SONNET, {**{k: v for k, v in asdict(res).items() if k != "key"}, "api": meta}
        )
    monkeypatch.delenv(COLLECT_ENV)
    final = sonnet(refuse)
    summary = run_cells(tmp_path / "final", CELLS, final)
    assert final.usage["calls"] == final.usage["cached"] > 0
    stats = summary["branches"]["def"]["defwriter"][D0TYPED.name]
    assert stats["writes"] == len(notes) and stats["unreadable"] == 0


# ---------------------------------------------------------------------------- pin
PIN = {
    "requests": "f21f056a492cd1c18a724ed0541ab73f4684e47ec60251e21422e3e78c2b32d8",
    "items": "55b6bce0b0f316152330a2570c95675ed3fab76bc7735c2f927599df9b6fc7f9",
}


def test_the_cell_is_pinned(rep):
    """The requests and item rows of the cell on the small world: a change here means a cached run
    of the cell would no longer be a hit, or the cell behaves differently from the one that was
    run."""
    requests = sorted(rep.llms[D0TYPED.name].keys())
    items = [i.to_json() for i in rep.cells[D0TYPED.name].items.values()]
    assert digest(requests) == PIN["requests"]
    assert digest(items) == PIN["items"]

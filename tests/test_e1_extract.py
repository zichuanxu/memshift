"""The D1sx cells (`--cells D1sxB0,D1sxB1_source`): D1s whose programs read
fields extracted from prose. They send exactly the D1s writer and program requests, change only the
programs' input, record that input, and a fake-model E1 run with them passes the pre-intervention
hash assertion and writes the extraction rows."""

import json
from datetime import timedelta

import pytest

from memshift.analysis.round3b_extract import block_w
from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import parse_cell, parse_cells
from memshift.experiments.e1 import run as run_e1
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.extract import RecordExtractor
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.probes import make_probes
from memshift.reader import context_hash
from tests.extract_fakes import ExtractLLM
from tests.test_program import CLOSE_FRIEND, code

D1S = [CellConfig(1, 0, synth=True), CellConfig(1, 1, "source", synth=True)]
D1SX = [
    CellConfig(1, 0, synth=True, extracted=True),
    CellConfig(1, 1, "source", synth=True, extracted=True),
]
WEEKS = (1, 2)
# the close-friend program reads `conversation_with` and `sent`: extraction errors there move it
PROGRAMS = {"(close_friend v1)": code(CLOSE_FRIEND.format(days=30, need=2))}


def test_names_tokens_and_validation():
    assert D1SX[0].name == "holdmem_D1sxB0"
    assert D1SX[1].name == "holdmem_D1sxB1_source"
    assert parse_cell("D1sxB0") == D1SX[0]
    assert parse_cell("D1sxB1_source") == D1SX[1]
    assert parse_cell("D1sxB1") == D1SX[1]
    # the tokens that existed keep their meaning
    assert parse_cell("D1sB0") == D1S[0] and not parse_cell("D1sB0").extracted
    assert parse_cell("D1sB1_source") == D1S[1]
    assert parse_cells("D0B0,D1sxB0,zero_Dm") == ([CellConfig(0, 0), D1SX[0]], ("zero_Dm",))
    for bad in ("D0sxB0", "D1sxB0_ra", "D1sxB1_source_nofb"):
        with pytest.raises(ValueError):
            parse_cell(bad)
    with pytest.raises(ValueError, match="variant of the D1s programs"):
        CellConfig(1, 0, extracted=True)
    with pytest.raises(ValueError, match="variant of the D1s programs"):
        CellConfig(1, 0, synth=True, ra=True, extracted=True)


@pytest.fixture(scope="module")
def world_log():
    w = generate_branch(5, "def", world_params(3))
    return w, SourceLog(w.events)


def test_a_d1sx_cell_needs_an_extractor(world_log):
    w, log = world_log
    with pytest.raises(ValueError, match="extractor"):
        HoldMem(w, log, D1SX[0], ExtractLLM(), "fake/writer-a")


def _replay(world, log, cfgs, make_llm):
    """Build and maintain the cells (one recording writer each, one extractor per cell, as a run
    shares one per world); returns cells, writers, contexts per checkpoint."""
    t0 = world.fork - timedelta(minutes=1)
    cells, llms, extractors = {}, {}, []
    for c in cfgs:
        llms[c.name] = make_llm()
        extractor = None
        if c.extracted:
            extractor = RecordExtractor(llms[c.name], llms[c.name], workers=8)
            extractors.append(extractor)
        cells[c.name] = HoldMem(world, log, c, llms[c.name], "fake/writer-a", extractor=extractor)
        cells[c.name].workers = 8
        cells[c.name].build(t0)
    contexts = {}
    for k in (0, *WEEKS):
        t = t0 if k == 0 else world.fork + timedelta(weeks=k)
        if k:
            for hm in cells.values():
                hm.maintain(t)
        for name, hm in cells.items():
            for p in make_probes(world, log, [t]):
                contexts[name, k, p.probe_id] = context_hash(hm.serve(p).blocks)
    for e in extractors:
        e.close()
    return cells, llms, contexts


@pytest.fixture(scope="module")
def perfect(world_log):
    w, log = world_log
    return _replay(w, log, D1S + D1SX, lambda: ExtractLLM(PROGRAMS))


@pytest.fixture(scope="module")
def faulty(world_log):
    """An extractor that attributes every chat message to one person."""
    w, log = world_log
    corrupt = {("messages", "conversation_with"): "Ana Lopez"}
    return _replay(w, log, D1S + D1SX, lambda: ExtractLLM(PROGRAMS, corrupt=corrupt))


def _values(hm, slot_id):
    return [
        (h["event"], h["ts"], h.get("value"))
        for i in hm.by_slot[slot_id]
        for h in hm.items[i].history
    ]


def test_d1sx_sends_exactly_the_d1s_writer_and_program_requests(perfect):
    _, llms, _ = perfect
    for s, x in zip(D1S, D1SX, strict=True):
        for purpose in ("write", "program"):
            assert llms[x.name].keys(purpose) == llms[s.name].keys(purpose), (x.name, purpose)
        assert not llms[s.name].keys("rewrite") and not llms[s.name].keys("extract")
        assert llms[x.name].keys("rewrite") and llms[x.name].keys("extract")


def test_pre_shift_contexts_are_identical(perfect, faulty):
    for _, _, contexts in (perfect, faulty):
        at_t0: dict[str, set[str]] = {}
        for (_, k, pid), h in contexts.items():
            if k == 0:
                at_t0.setdefault(pid, set()).add(h)
        assert at_t0 and all(len(hs) == 1 for hs in at_t0.values())


def test_perfect_extraction_gives_the_d1s_values(perfect, world_log):
    cells, _, _ = perfect
    w, _ = world_log
    for s, x in zip(D1S, D1SX, strict=True):
        for slot in w.slots:
            assert _values(cells[x.name], slot.slot_id) == _values(cells[s.name], slot.slot_id)


def test_extraction_errors_move_the_program_and_are_recorded(faulty, world_log):
    cells, _, _ = faulty
    w, _ = world_log
    s, x = cells[D1S[1].name], cells[D1SX[1].name]
    item = x.items["k.close_friend#0"]
    note = item.history[0]["program"]["input"]
    assert note["error_fields"] == ["messages.conversation_with"] and note["with_error"] > 0
    t0 = w.fork - timedelta(minutes=1)
    assert note["records"] == len(SourceLog(w.events).snapshot(t0).by_source("messages"))
    assert item.history[0]["program"]["key"] == item.det["program_key"]
    writes = [h for h in item.history if "program" in h]
    assert item.det["program_input"] == writes[-1]["program"]["input"]
    assert x.items["k.close_friend#0"].det["program_value"] in ([], ["Ana Lopez"])
    assert s.items["k.close_friend#0"].det["program_value"] != ["Ana Lopez"]
    # an item whose sources were extracted without error is the D1s item
    assert x.items["k.overdue_task#0"].history[0]["program"]["input"]["with_error"] == 0
    assert _values(x, "k.overdue_task") == _values(s, "k.overdue_task")
    # the D1s cells record no input
    assert all("input" not in h.get("program", {}) for i in s.items.values() for h in i.history)


def test_fake_e1_run_with_the_block_w_cells(tmp_path, monkeypatch):
    llm = ExtractLLM(PROGRAMS)
    monkeypatch.setattr(
        "memshift.experiments.e1.client", lambda model, run_dir, workers=48, **kw: llm
    )
    cells, _ = parse_cells("D0B0,D0B1_source,D1sB0,D1sB1_source,D1sxB0,D1sxB1_source")
    d = run_e1(
        [5],
        "w",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("src", "def"),
        world_params=world_params(3),
        discriminative=True,
    )
    for b in ("src", "def"):
        assert json.loads((d / f"hash_check_w0005_{b}.json").read_text())["ok"]
    man = json.loads((d / "manifest.json").read_text())
    assert man["params"]["extraction"]["rewriter"] == "Qwen/Qwen3.5-9B"
    assert "holdmem_D1sxB1_source" in man["params"]["cells"]
    rows = read_jsonl(d / "extractions.jsonl")
    assert {r["branch"] for r in rows} == {"src", "def"}
    assert all(r["errors"] == [] and r["extract_ok"] for r in rows)
    summary = read_jsonl(d / "world_summaries.jsonl")[0]
    for b in ("src", "def"):
        rep = summary["branches"][b]["extraction"]
        assert set(rep) == {"calendar", "contacts", "email", "messages", "notes", "tasks"}
        assert sum(c["records"] - c["background"] for c in rep.values()) == sum(
            r["branch"] == b for r in rows
        )
    # one rewrite and one extraction per record content in the world, both branches together
    keys = {(r["source"], json.dumps(r["typed"], sort_keys=True)) for r in rows}
    assert len(llm.keys("rewrite")) == len(keys) == len(llm.keys("extract"))
    answers = read_jsonl(d / "answers.jsonl")
    assert {a["system"] for a in answers} == {c.name for c in cells}
    # block W's analysis reads the run: a perfect extraction leaves D_m as it was
    res = block_w([d], draws=50)
    assert res["pairing"]["ok"] and res["quantities"]["worlds"] == ["w0005"]
    assert res["quantities"]["extraction"]["pooled"]["record_correct"]["est"] == 1.0
    for b in ("src", "def"):
        assert res["quantities"]["attribution"]["B1"][b]["differ"] == 0
        assert res["quantities"]["attribution"]["B1"][b]["item_times"] > 0
        diff = res["quantities"]["bxd"][b]["definition"]["difference"]
        assert diff["n_items"] == 0 or diff["est"] == 0.0

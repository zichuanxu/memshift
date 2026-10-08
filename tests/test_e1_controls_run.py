"""The E1 driver with the control cells and the query-time view, offline with the fake writer: the
pre-shift hash assertion covers every cell, the answers of every system are written and the
analysis reads them."""

import json

import pytest

from memshift.analysis.e1 import analyse
from memshift.experiments.common import read_jsonl, world_params
from memshift.experiments.e1 import parse_cells, run_world
from memshift.experiments.e1 import run as run_e1
from memshift.holdmem.system import HoldMem
from tests.fakes import FakeLLM

TOKENS = (
    "D0B0,D0B1_source,D1B1_source,D0B1_gold,D1B1_gold,D0B1_scope,D1B0_refresh,D0tB1_source,"
    "D0thinkB1_source,D1sB1_source_ra,D1sB0_ra,zero_Dm"
)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = tmp_path_factory.mktemp("controls")
    cells, views = parse_cells(TOKENS)
    think = FakeLLM()
    summary = run_world(
        5,
        out,
        branches=("def",),
        baselines=True,
        writer=FakeLLM(),
        cells_cfg=cells,
        world_params=world_params(3),
        discriminative=True,
        views=views,
        think_writer=think,
    )
    return out, summary, cells, think


def test_the_hash_assertion_covers_every_cell(run):
    out, summary, cells, _ = run
    assert summary["branches"]["def"]["hash_ok"]
    check = json.loads(next(out.glob("hash_check_*_def.json")).read_text())
    assert check["ok"] and check["mismatches"] == [] and check["probes"] > 0
    # the assertion compares the contexts of the cells it was given: every control cell is one
    systems = {r["system"] for r in read_jsonl(out / "answers.jsonl")}
    assert {c.name for c in cells} <= systems
    assert {"zero_Dm", "zero", "oracle"} <= systems


def test_a_failing_hash_assertion_stops_the_run(tmp_path, monkeypatch):
    """Change the pre-shift bytes of a control cell only: the driver must raise, not continue."""
    cells, _ = parse_cells("D0B0,D0tB1_source")
    real_build = HoldMem.build

    def build(self, t, apply_fallback=False):
        real_build(self, t, apply_fallback)
        if self.cfg.present:
            for item in self.items.values():
                item.s_k += " (changed)"

    monkeypatch.setattr(HoldMem, "build", build)
    with pytest.raises(RuntimeError, match="hash assertion"):
        run_world(
            5,
            tmp_path,
            branches=("def",),
            baselines=False,
            writer=FakeLLM(),
            cells_cfg=cells,
            world_params=world_params(3),
        )
    check = json.loads(next(tmp_path.glob("hash_check_*_def.json")).read_text())
    assert not check["ok"] and check["mismatches"]


def test_the_thinking_writer_is_used_and_reported(run):
    out, summary, _, think = run
    assert think.usage["calls"] > 0
    assert summary["think_llm_usage"]["calls"] == think.usage["calls"]
    think_items = [
        r
        for r in read_jsonl(out / "items.jsonl")
        if r["system"] == "holdmem_D0thinkB1_source" and r["slot_id"].startswith("k.")
    ]
    assert think_items and all(r["note"]["think"]["ok"] for r in think_items)


def test_the_analysis_reads_the_control_cells(run):
    out, _, cells, _ = run
    res = analyse(out)
    assert res["pre_intervention"]["cells_identical_answers"]
    metrics = res["cells"]["def"]
    for c in cells:
        assert metrics[c.name]["Update"]["n_items"] > 0, c.name
    assert metrics["zero_Dm"]["Update"]["n_items"] > 0
    # B=0 and refresh detect nothing, the oracle key and source-epoch both detect the changed items
    assert metrics["holdmem_D0B0"]["MissedInvalidation"]["est"] == 1.0
    assert metrics["holdmem_D1B0_refresh"]["MissedInvalidation"]["est"] == 1.0
    assert metrics["holdmem_D0B1_gold"]["MissedInvalidation"]["est"] == 0.0
    assert metrics["holdmem_D0B1_source"]["MissedInvalidation"]["est"] == 0.0
    fi = lambda c: metrics[c]["FalseInvalidation"]["est"]  # noqa: E731
    assert fi("holdmem_D0B1_gold") < fi("holdmem_D0B1_source")
    # detection is the key's, not D's
    assert (
        metrics["holdmem_D1B1_gold"]["FalseInvalidation"]
        == metrics["holdmem_D0B1_gold"]["FalseInvalidation"]
    )


def test_run_writes_the_views_and_the_thinking_writer_into_the_manifest(tmp_path, monkeypatch):
    made = []

    def fake_client(model, run_dir, workers=48, **kw):
        made.append((model, kw.get("extra_body")))
        return FakeLLM(model)

    monkeypatch.setattr("memshift.experiments.e1.client", fake_client)
    cells, views = parse_cells("D0B0,D0thinkB1_source,zero_Dm")
    out = run_e1(
        [5],
        "controls",
        out=tmp_path,
        cells_cfg=cells,
        baselines=False,
        branches=("def",),
        world_params=world_params(3),
        discriminative=True,
        views=views,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    flat = json.dumps(manifest)
    assert "holdmem_D0thinkB1_source" in flat and "zero_Dm" in flat and "think_writer" in flat
    assert ("Qwen/Qwen3.5-9B", {"chat_template_kwargs": {"enable_thinking": True}}) in made
    assert {r["system"] for r in read_jsonl(out / "answers.jsonl")} >= {"zero_Dm"}

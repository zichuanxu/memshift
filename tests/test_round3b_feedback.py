"""`analysis.round3b_feedback` (execution feedback) on a synthetic fixture: fake-LLM E1 runs on
two small worlds, an F run with the six cells of `F_CELLS` and a "committed" run with the D0, D1
and D_m cells. The fake writer's definition programs have a syntax error that the first feedback
request corrects (tests/test_e1_feedback.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import round3b_feedback as fb
from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_cells
from memshift.experiments.e1 import run as run_e1
from tests.test_e1_feedback import CellLLM

F_CELLS = "D0B0,D0B1_source,D1sB0,D1sB1_source,D1sfbB0,D1sfbB1_source"
COMMITTED = "D0B0,D1B0,D0B1_source,D1B1_source,D1sB0,D1sB1_source"


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    mp = pytest.MonkeyPatch()
    mp.setattr("memshift.experiments.e1.client", lambda *a, **kw: CellLLM())
    out = []
    try:
        for name, spec in (("f", F_CELLS), ("committed", COMMITTED)):
            cells, _ = parse_cells(spec)
            out.append(
                run_e1(
                    [5, 6],
                    name,
                    out=tmp_path_factory.mktemp(name),
                    cells_cfg=cells,
                    baselines=False,
                    branches=("src", "def", "orphan"),
                    world_params=world_params(3),
                    discriminative=True,
                    parallel_worlds=2,
                )
            )
    finally:
        mp.undo()
    return out[0], out[1]


@pytest.fixture(scope="module")
def res(runs: tuple[Path, Path]) -> dict[str, Any]:
    return fb.analyse([runs[0]], [runs[1]], draws=200)


def test_the_shared_cells_reproduce_the_committed_answers(res):
    assert "error" not in res, res.get("error")
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    assert res["pairing"]["anchor_answers_compared"] > 0
    assert res["worlds"] == ["w0005", "w0006"]


def test_executability_first_attempt_and_after_feedback(res):
    ex = res["executability"]
    assert ex["checks"] == {"d1s_differs": 0, "d1s_missing": 0}
    for phase in fb.PHASES:
        d = ex["definition"][phase]["both"]
        assert d["n_runs"] > 0 and d["first_attempt"]["est"] == 0.0
        assert d["after_feedback"]["est"] == 1.0 and d["after_feedback_failed"] == 0
        assert d["first_attempt_kinds"] == {"syntax": d["n_runs"]}
        assert d["failed_run_kinds"] == {"syntax": d["feedback_requests"]}
        assert set(d["rounds"]) <= {"0", "1"} and d["start_kept"] == d["rounds"].get("0", 0)
        assert d["world_data"] == {"requests": d["feedback_requests"], "with_record_value": 0}
        assert set(d) <= set(ex["definition"][phase]["holdmem_D1sfbB0"])
        assert all(k.startswith("k.") for k in ex["definition"][phase]["per_definition"])
        o = ex["other"][phase]["both"]
        assert o["first_attempt"]["est"] == o["after_feedback"]["est"] == 1.0
        assert o["feedback_requests"] == 0
    assert ex["definition"]["all"]["both"]["n_runs"] > ex["definition"]["post"]["both"]["n_runs"]


def test_contrasts_are_present_for_every_branch_and_class(res):
    for b in fb.SHIFTS:
        for cls in fb.CLASSES:
            node = res["r"][b][cls]["pooled"]
            assert set(node) == {"feedback", "dm", "difference"}
    for b in fb.BRANCHES:
        for cls in fb.CLASSES:
            node = res["bxd"][b][cls]
            assert set(node) == {"Dm_feedback", "Dm", "paired"}
    upd = res["deletion"]["update"]
    assert set(upd["definition"]) == {"B0", "B1_source"}


def test_the_deletion_accounting_reuses_the_stage_6_measure(res, runs):
    loss = res["deletion"]["loss"]
    assert loss["B1"]["d1_rows"] == "compare"
    blk = loss["B1"]["all"]
    assert {"Dm_feedback", "Dm", "loss_vs_D0_paired"} <= set(blk)
    assert "loss_total" in blk["Dm"]["accounting_vs_D0"] and "accounting_vs_D1" in blk["Dm"]
    # without the committed D=1 rows the D0 rows stand in and only the D=0 accounting is kept
    alone = fb.analyse([runs[0]], [], draws=50)["deletion"]["loss"]["B1"]
    assert alone["d1_rows"] == "stand_in_D0"
    assert "accounting_vs_D1" not in alone["all"]["Dm"]
    a = alone["all"]["Dm"]["accounting_vs_D0"]["loss_total"]["est"]
    b = blk["Dm"]["accounting_vs_D0"]["loss_total"]["est"]
    assert a == b or (math.isnan(a) and math.isnan(b))


def test_cli_writes_the_result(runs, tmp_path):
    out = tmp_path / "f.json"
    fb.main(["--runs", str(runs[0]), "--compare", str(runs[1]), "--draws", "50", "--out", str(out)])
    assert json.loads(out.read_text())["block"] == "F"

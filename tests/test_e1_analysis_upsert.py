"""The E1 analysis of the cells of prereg v4 block U: the upsert cells (`D0B0_upsert`,
`D1B0_upsert`, `D1sB0_upsert`) and `D1sB0_refresh` join the pre-intervention identity check, and
an upsert cell's `supersede` events count as detection for that cell only. The item rows, cell
metrics, contrasts and pre-intervention results of the existing cells are pinned on one run; the
digest was taken on the analysis code before the change (commit 85ed9e1)."""

from pathlib import Path
from typing import Any

import pytest

from memshift.analysis.e1 import HM_CELLS, analyse, detection_events, item_table, load
from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_cells, run_world
from memshift.textutil import sha256_json
from tests.fakes import FakeLLM

EXISTING = ("D0B0", "D1B0", "D0B1_source", "D1B1_source", "D1sB0", "D1sB1_source", "D1B0_refresh")
NEW = ("D0B0_upsert", "D1B0_upsert", "D1sB0_upsert", "D1sB0_refresh")
UPSERT = tuple(f"holdmem_{t}" for t in NEW if t.endswith("_upsert"))
PIN = "79d059cc7263e7cca825d0e5847ba8e7ab277991178fce63b1489ee565111ebc"


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("e1_upsert")
    cells, _ = parse_cells(",".join(EXISTING + NEW))
    summary = run_world(
        5,
        d,
        branches=("def",),
        baselines=False,
        writer=FakeLLM(),
        cells_cfg=cells,
        world_params=world_params(3),
    )
    assert summary["branches"]["def"]["hash_ok"]
    return d


@pytest.fixture(scope="module")
def result(run, tmp_path_factory) -> dict[str, Any]:
    return analyse(run, out=tmp_path_factory.mktemp("e1_upsert_analysis"))


def existing(result: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """What the analysis reports for the existing cells."""
    old = {f"holdmem_{t}" for t in EXISTING}
    return {
        "rows": [r for r in rows if r["system"] in old],
        "cells": {b: {s: m for s, m in c.items() if s in old} for b, c in result["cells"].items()},
        "contrasts": result["contrasts"],
        "pre_intervention": {
            k: v for k, v in result["pre_intervention"].items() if k != "cells_identical_answers"
        },
    }


def test_existing_cells_are_byte_identical(run, result):
    got = existing(result, item_table(load(run)))
    assert got["rows"] and set(got["cells"]["def"]) == {f"holdmem_{t}" for t in EXISTING}
    assert got["contrasts"]["def"]
    assert sha256_json(got) == PIN


def test_the_new_cells_are_in_the_pre_intervention_check(result):
    assert {f"holdmem_{t}" for t in NEW} <= set(HM_CELLS)
    assert {f"holdmem_{t}" for t in NEW} <= set(result["pre_intervention"])
    assert result["pre_intervention"]["cells_identical_answers"]


def test_only_upsert_cells_count_supersede():
    for name in UPSERT + ("holdmem_D1B0_upsert_nofb",):
        assert detection_events(name) == ("invalidate", "supersede")
    for name in [f"holdmem_{t}" for t in EXISTING] + ["holdmem_D1sB0_refresh"]:
        assert detection_events(name) == ("invalidate",)


def test_an_upsert_cell_detects_by_supersede(run, result):
    data = load(run)
    as_of = {
        (p["world_id"], p["branch"], p["fact_id"], int(p["tags"].get("k", 0))): p["as_of"]
        for p in data["probes"]
    }
    superseded: dict[tuple[str, str, str, str], list[str]] = {}
    for it in data["items"]:
        ts = [h["ts"] for h in it["history"] if h["event"] == "supersede"]
        superseded.setdefault((it["world_id"], it["branch"], it["system"], it["slot_id"]), [])
        superseded[(it["world_id"], it["branch"], it["system"], it["slot_id"])] += ts
    n = 0
    for r in item_table(data):
        if "detected" not in r:
            continue
        ts = superseded[(r["world_id"], r["branch"], r["system"], r["fact_id"])]
        if r["system"] in UPSERT:
            t = as_of[(r["world_id"], r["branch"], r["fact_id"], r["k"])]
            assert r["detected"] == any(x <= t for x in ts), r
            n += r["detected"]
        else:
            assert not ts  # only an upsert cell supersedes
    assert n > 0
    cells = result["cells"]["def"]
    for name in UPSERT:
        assert cells[name]["MissedInvalidation"]["est"] < 1.0, name
    assert cells["holdmem_D0B0"]["MissedInvalidation"]["est"] == 1.0
    assert cells["holdmem_D1sB0_refresh"]["MissedInvalidation"]["est"] == 1.0

from datetime import timedelta

import pytest

from memshift.analysis.e1 import ITEM_CLASSES, analyse
from memshift.experiments.e1 import parse_cell, run_world
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import NO_RULE, fallback
from memshift.holdmem.system import CellConfig, HoldMem
from tests.fakes import FakeLLM


def test_nofb_names_and_validation():
    assert CellConfig(1, 1, "source", fallback=False).name == "holdmem_D1B1_source_nofb"
    assert CellConfig(1, 0, fallback=False).name == "holdmem_D1B0_nofb"
    assert CellConfig(1, 1, "cited").name == "holdmem_D1B1_cited"
    assert CellConfig(0, 0).name == "holdmem_D0B0"
    with pytest.raises(ValueError):
        CellConfig(0, 1, "source", fallback=False)


def test_parse_cell():
    assert parse_cell("D0B0") == CellConfig(0, 0, "source")
    assert parse_cell("D1B1_cited") == CellConfig(1, 1, "cited")
    assert parse_cell("D1B1_source_nofb") == CellConfig(1, 1, "source", fallback=False)
    assert parse_cell("D1B0_nofb") == CellConfig(1, 0, "source", fallback=False)


def test_item_classes_follow_fallback_coverage():
    w = generate_branch(1000, "natural", None)
    log = SourceLog(w.events)
    for slot in w.slots:
        hits = [c for c, prefixes in ITEM_CLASSES.items() if slot.slot_id.startswith(prefixes)]
        assert len(hits) == 1, slot.slot_id
        has_rule = fallback(w, log, slot, w.fork) != NO_RULE
        assert has_rule == (hits[0] in ("rule", "definition")), slot.slot_id


def test_nofb_repair_never_substitutes_fallback():
    w = generate_branch(11, "src")
    log = SourceLog(w.events)
    t = w.fork + timedelta(weeks=1)
    hm = HoldMem(w, log, CellConfig(1, 1, "source", fallback=False), FakeLLM(), "fake/writer-a")
    hm.workers = 1
    hm.build(w.fork)
    hm.maintain(t)
    failed = 0
    for item in hm.items.values():
        assert item.det is None or item.det["check"] != "fallback"
        if item.det is not None and item.det["check"] == "unverifiable":
            failed += 1
            assert item.status == "unverifiable"
    assert failed > 0


def test_nofb_contrasts_only_with_nofb_cell(tmp_path):
    cells = [CellConfig(0, 1, "source"), CellConfig(1, 1, "source")]
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    run_world(5, tmp_path / "a", branches=("src",), baselines=False, writer=FakeLLM(),
              cells_cfg=cells)  # fmt: skip
    assert (
        "explore_fallback_effect_repair_source" not in analyse(tmp_path / "a")["contrasts"]["src"]
    )
    run_world(5, tmp_path / "b", branches=("src",), baselines=False, writer=FakeLLM(),
              cells_cfg=[*cells, CellConfig(1, 1, "source", fallback=False)])  # fmt: skip
    res = analyse(tmp_path / "b")
    c = res["contrasts"]["src"]
    for key in ("explore_fallback_effect_repair_source", "explore_check_only_effect_repair_source"):
        assert key in c
        assert all(f"{key}_{cls}" in c for cls in ITEM_CLASSES)
    assert "holdmem_D1B1_source_nofb" in res["cells"]["src"]

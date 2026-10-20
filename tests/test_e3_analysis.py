"""E3 analysis (T3, and T3 without the definition items) on synthetic answers."""

import json
from pathlib import Path

import pytest

from memshift.analysis import e3

WORLDS = ("w1", "w2", "w3", "w4")


def _answers(systems: dict[str, tuple[float, float]]) -> list[dict]:
    """One row per (system, world, item) at a single horizon; `systems` maps a system to its
    ContractValid rate on definition items and on the other items (10 items of each per world)."""
    rows = []
    for system, (def_rate, other_rate) in systems.items():
        for w in WORLDS:
            for prefix, rate in (("k.item", def_rate), ("c.item", other_rate)):
                for i in range(10):
                    rows.append(
                        {
                            "system": system,
                            "world_id": w,
                            "fact_id": f"{prefix}{i}",
                            "contract_valid": int(i < rate * 10),
                            "tags": {"horizon_months": 12},
                        }
                    )
    return rows


def _analyse(tmp_path: Path, rows: list[dict]) -> dict:
    (tmp_path / "answers.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (tmp_path / "world_results.jsonl").write_text("")
    per: dict = e3.analyse(tmp_path)["horizons"][12]
    return per


def test_t3_without_definition_items_changes_the_outcome_when_the_oracle_is_weak_there(tmp_path):
    # the oracle is poor on definition items (30%); HoldMem computes them (100%) and is worse than
    # the oracle on the rest (80% against 90%); the zero system is poor on both
    per = _analyse(
        tmp_path,
        _answers({"oracle": (0.3, 0.9), "holdmem_D1B1_source": (1.0, 0.8), "zero": (0.3, 0.6)}),
    )
    hold, zero = per["holdmem_D1B1_source"], per["zero"]
    # all items: (1.0 + 0.8) / 2 - (0.3 + 0.9) / 2 = +0.3, passes
    assert hold["minus_oracle"]["est"] == pytest.approx(0.3) and hold["T3_pass"]
    # without definition items: 0.8 - 0.9 = -0.1, fails the 3 pp floor
    nd = hold["explore_no_definition"]
    assert nd["minus_oracle"]["est"] == pytest.approx(-0.1) and not nd["T3_pass"]
    assert nd["contract_valid"]["est"] == pytest.approx(0.8)
    assert nd["contract_valid"]["n_items"] == 4 * 10  # the ten definition items per world are out
    assert per["oracle"]["explore_no_definition"]["contract_valid"]["est"] == pytest.approx(0.9)
    assert "T3_pass" not in per["oracle"]["explore_no_definition"]
    assert zero["explore_no_definition"]["minus_oracle"]["est"] == pytest.approx(-0.3)
    assert not zero["explore_no_definition"]["T3_pass"]
    assert zero["minus_oracle"]["est"] == pytest.approx(-0.15) and not zero["T3_pass"]


def test_existing_fields_are_unchanged_by_the_explore_field(tmp_path):
    per = _analyse(tmp_path, _answers({"oracle": (0.5, 0.9), "zero": (0.5, 0.9)}))
    z = per["zero"]
    assert z["minus_oracle"]["est"] == 0.0 and z["T3_pass"] is True
    assert z["contract_valid"]["est"] == pytest.approx(0.7)
    assert set(z["explore_by_class"]) == {"rule", "definition", "norule"}
    assert z["explore_no_definition"]["T3_pass"] is True


def test_no_oracle_means_no_t3(tmp_path):
    per = _analyse(tmp_path, _answers({"zero": (0.5, 0.9)}))
    nd = per["zero"]["explore_no_definition"]
    assert "minus_oracle" not in nd and "T3_pass" not in nd
    assert nd["contract_valid"]["est"] == pytest.approx(0.9)

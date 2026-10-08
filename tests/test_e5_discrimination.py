"""E5 native discriminability (exploratory): the native estimate against the interval of the
white-box cell of the other B label, the rule that counts a comparison as discriminating, and the
wiring into `e5.analyse` and `stage4_summary` on synthetic runs."""

import json
from pathlib import Path

import pytest

from memshift.analysis import e5, e5_infer
from memshift.gen.shifts import inference_half

WORLDS = tuple(f"w{i}" for i in range(6))
CHANGED = next(f for f in ("c.p01.city", "c.p02.city", "c.p03.city") if not inference_half(f))
UNCHANGED = next(f for f in ("c.p04.city", "c.p05.city", "c.p06.city") if not inference_half(f))


def test_other_cell_is_the_other_b_label_with_the_same_d():
    assert e5.other_cell("mem0", 0, 0) == "holdmem_D0B1_source"
    assert e5.other_cell("mem0", 0, 1) == "holdmem_D0B0"
    assert e5.other_cell("lightrag", 0, 1) == "holdmem_D0B0"
    assert e5.other_cell("lightrag", 0, 0) == "holdmem_D0B1_cited"  # document-level binding
    assert e5.other_cell("hipporag", 1, 0) == "holdmem_D1B1_cited"
    assert e5.other_cell("amem", 1, 1) == "holdmem_D1B0"
    assert e5.other_cell("mem0", 0, None) is None and e5.other_cell("mem0", None, 0) is None
    for system in ("mem0", "lightrag", "amem"):
        for d in (0, 1):
            for b in (0, 1):
                assert e5.other_cell(system, d, b) != e5.predicted_cell(system, d, b)


def _est(est: float, lo: float, hi: float) -> dict:
    return {"est": est, "lo": lo, "hi": hi, "n_worlds": 5, "n_items": 50}


def test_discrimination_rule_four_cases():
    native = {m: _est(0.5, 0.4, 0.6) for m in ("Update", "Stability")}
    other = {"Update": _est(0.9, 0.8, 1.0), "Stability": _est(0.5, 0.45, 0.55)}
    got = e5.discrimination(
        native, other, {"Update": True, "Stability": True}, ["Update", "Stability"]
    )
    # inside the predicted interval and outside the other label's: discriminates; inside both: not
    assert got["Update"] == {"inside_predicted": True, "inside_other": False, "discriminates": True}
    assert got["Stability"] == {
        "inside_predicted": True,
        "inside_other": True,
        "discriminates": False,
    }
    # outside the predicted interval never discriminates, whatever the other label says
    got = e5.discrimination(
        native, other, {"Update": False, "Stability": False}, ["Update", "Stability"]
    )
    assert not got["Update"]["discriminates"] and got["Update"]["inside_other"] is False
    assert not got["Stability"]["discriminates"] and got["Stability"]["inside_other"] is True
    # interval ends are inside; a metric that is not asked for is not reported
    edge = e5.discrimination(native, {"Update": _est(0.9, 0.5, 1.0)}, {"Update": True}, ["Update"])
    assert edge["Update"]["inside_other"] is True and list(edge) == ["Update"]


def _rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def _probe(w: str, fact: str, k: int, value: str) -> dict:
    return {
        "world_id": w,
        "branch": "src",
        "fact_id": fact,
        "vtype": "org",
        "gold_status": "answer",
        "gold_value": value,
        "as_of": f"2026-01-0{k + 1}",
        "tags": {"k": k},
    }


def _answer(system: str, w: str, fact: str, k: int, cv: bool) -> dict:
    return {
        "system": system,
        "world_id": w,
        "branch": "src",
        "fact_id": fact,
        "vtype": "org",
        "contract_valid": cv,
        "tags": {"k": k},
    }


def _synthetic_runs(
    tmp_path: Path,
    native_cv: dict[str, bool],
    system: str = "mem0",
    cells: tuple[str, str] = ("holdmem_D0B0", "holdmem_D0B1_source"),
) -> tuple[Path, Path]:
    """`CHANGED` changes value at k = 1, `UNCHANGED` does not. holdmem_D0B0 (the cell that predicts
    a B = 0 native system) never updates and stays right on the unchanged item; holdmem_D0B1_source
    (the other label) updates and forgets the unchanged item. `native_cv` gives the native
    system's ContractValid on the changed and the unchanged item at k = 1."""
    no_update, update = cells  # the cell that never updates and the one that updates
    cells_cv = {
        no_update: {"changed": False, "unchanged": True},
        update: {"changed": True, "unchanged": False},
        system: {"changed": native_cv["changed"], "unchanged": native_cv["unchanged"]},
    }
    probes, answers = [], []
    for w in WORLDS:
        for fact, new in ((CHANGED, "b"), (UNCHANGED, "a")):
            probes += [_probe(w, fact, 0, "a"), _probe(w, fact, 1, new)]
        for name, cv in cells_cv.items():
            for fact, key in ((CHANGED, "changed"), (UNCHANGED, "unchanged")):
                answers.append(_answer(name, w, fact, 1, cv[key]))
    native, white = tmp_path / "native", tmp_path / "white"
    native.mkdir()
    white.mkdir()
    natives = [a for a in answers if a["system"] == system]
    _rows(native / "answers.jsonl", natives)
    _rows(native / "probes.jsonl", [])
    _rows(native / "world_results.jsonl", [{"system": system}])
    _rows(white / "answers.jsonl", [a for a in answers if a["system"] != system])
    _rows(white / "probes.jsonl", probes)
    _rows(white / "items.jsonl", [])
    return native, white


@pytest.fixture
def b0_label(monkeypatch):
    lab = {"B": {"label": 0, "rate": 0.9, "n": 30}, "D": {"label": None, "rate": 1.0, "n": 0}}
    monkeypatch.setattr(e5_infer, "infer", lambda *a, **k: {"mem0": lab})
    monkeypatch.setattr(e5, "exploratory", lambda *a, **k: {})


def test_analyse_places_the_native_value_against_the_other_labels_interval(tmp_path, b0_label):
    # the native system does not update (Update 0, like B = 0) and keeps the unchanged item
    native, white = _synthetic_runs(tmp_path, {"changed": False, "unchanged": True})
    res = e5.analyse([native], [white], tmp_path / "out")
    per = res["systems"]["mem0"]
    assert (
        per["predicted_by"] == "holdmem_D0B0" and per["other_label_cell"] == "holdmem_D0B1_source"
    )
    src = per["branches"]["src"]
    assert src["consistent"] == {"Update": True, "Stability": True}  # the existing fields
    assert src["white_box_other"]["Update"]["est"] == 1.0
    assert src["white_box_other"]["Stability"]["est"] == 0.0
    assert src["discrimination"] == {
        "Update": {"inside_predicted": True, "inside_other": False, "discriminates": True},
        "Stability": {"inside_predicted": True, "inside_other": False, "discriminates": True},
    }
    # branches without data have no comparisons
    assert per["branches"]["def"]["discrimination"] == {}
    disc = res["stage4"]["discriminability"]
    assert disc["per_system"]["mem0"] == {
        "comparisons": 2,
        "inside_predicted": 2,
        "inside_other": 0,
        "discriminating": 2,
    }
    assert disc["total"] == disc["per_system"]["mem0"]


def test_a_native_system_that_looks_like_the_other_label_does_not_discriminate(tmp_path, b0_label):
    # updates the changed item and forgets the unchanged one: exactly the other label's cell
    native, white = _synthetic_runs(tmp_path, {"changed": True, "unchanged": False})
    res = e5.analyse([native], [white], tmp_path / "out")
    src = res["systems"]["mem0"]["branches"]["src"]
    assert src["consistent"] == {"Update": False, "Stability": False}
    assert src["discrimination"]["Update"] == {
        "inside_predicted": False,
        "inside_other": True,
        "discriminates": False,
    }
    assert res["stage4"]["discriminability"]["total"] == {
        "comparisons": 2,
        "inside_predicted": 0,
        "inside_other": 2,
        "discriminating": 0,
    }


def test_document_level_system_is_compared_at_cited_cluster_binding(tmp_path, monkeypatch):
    """A document-level system with B = 1 is predicted by the cited-cluster cell and its other
    label is HoldMem(0,0); with B = 0 the other label is the cited-cluster cell."""
    monkeypatch.setattr(e5, "exploratory", lambda *a, **k: {})
    cells = ("holdmem_D0B0", "holdmem_D0B1_cited")  # (never updates, updates)
    for b, native_cv, want in (
        (1, {"changed": True, "unchanged": False}, ("holdmem_D0B1_cited", "holdmem_D0B0")),
        (0, {"changed": False, "unchanged": True}, ("holdmem_D0B0", "holdmem_D0B1_cited")),
    ):
        lab = {"B": {"label": b, "rate": 0.5, "n": 30}, "D": {"label": None, "rate": 1.0, "n": 0}}
        monkeypatch.setattr(e5_infer, "infer", lambda *a, lab=lab, **k: {"lightrag": lab})
        run = tmp_path / f"b{b}"
        run.mkdir()
        native, white = _synthetic_runs(run, native_cv, "lightrag", cells)
        res = e5.analyse([native], [white], run / "out")
        per = res["systems"]["lightrag"]
        assert (per["predicted_by"], per["other_label_cell"]) == want
        # the native system behaves like its own label's cell and unlike the other one, so both
        # comparisons discriminate
        src = per["branches"]["src"]
        assert src["consistent"] == {"Update": True, "Stability": True}
        assert all(d["discriminates"] for d in src["discrimination"].values())
        assert res["stage4"]["discriminability"]["per_system"]["lightrag"]["discriminating"] == 2


def test_the_other_cell_may_be_missing_from_the_white_box_runs(tmp_path, b0_label):
    native, white = _synthetic_runs(tmp_path, {"changed": False, "unchanged": True})
    kept = [
        a
        for a in map(json.loads, (white / "answers.jsonl").read_text().splitlines())
        if a["system"] != "holdmem_D0B1_source"
    ]
    _rows(white / "answers.jsonl", kept)
    res = e5.analyse([native], [white], tmp_path / "out")
    src = res["systems"]["mem0"]["branches"]["src"]
    assert src["consistent"] == {"Update": True, "Stability": True}  # unchanged
    assert src["discrimination"] == {}  # nothing to place it against
    assert res["stage4"]["discriminability"]["total"]["comparisons"] == 0

"""Block W's analysis (`analysis.round3b_extract`) on small hand-built runs: extraction accuracy,
r and B x D with D_m on extracted against typed fields, the attribution of differing values to
extraction errors (with the programs read from a response cache), the threshold comparisons and
the pairing that voids a run. No model, no committed run."""

import json
from pathlib import Path
from typing import Any

import pytest

from memshift.analysis import round3b_extract as w3
from memshift.llm import ResponseCache
from tests.test_round3_blocks import WORLDS, const, on, write_run

# Update on the changed item-times (k.alpha and k.beta at k = 1, 2): D_m on typed fields repairs
# every definition item, on extracted fields only k.alpha -> B x D 1.0 typed, 0.5 extracted
NEW = {
    "holdmem_D0B0": const(False),
    "holdmem_D0B1_source": const(False),
    "holdmem_D1sB0": const(False),
    "holdmem_D1sB1_source": const(True),
    "holdmem_D1sxB0": const(False),
    "holdmem_D1sxB1_source": on("k.alpha", "c.p1.city", "t.t001"),
}
COMMITTED = {k: v for k, v in NEW.items() if "D1sx" not in k}
CODE_KEY = "a" * 64
CODE = "def derive(records, as_of):\n    return sorted({r['sender'] for r in records})\n"


def _ts(k: int) -> str:
    return f"2026-10-{k + 1:02d}T00:00"


def _history(values: list[Any], input_: dict[str, Any] | None) -> list[dict[str, Any]]:
    """derive at k = 0, then invalidate and repair at k = 1, 2, with these values."""
    prog: dict[str, Any] = {"ok": True, "key": CODE_KEY}
    if input_ is not None:
        prog["input"] = input_
    out = [{"ts": _ts(0), "event": "derive", "value": values[0], "program": prog}]
    for k in (1, 2):
        out.append({"ts": _ts(k), "event": "invalidate", "value": None})
        out.append({"ts": _ts(k), "event": "repair", "value": values[k], "program": prog})
    return out


def _items(d: Path) -> None:
    clean = {"records": 3, "with_error": 0, "error_fields": []}
    bad = {"records": 3, "with_error": 1, "error_fields": ["messages.sender"]}
    rows = []
    for w in WORLDS:
        for b in ("src", "def", "orphan"):
            for fact in ("k.alpha", "k.beta"):
                for system in ("holdmem_D1sB1_source", "holdmem_D1sxB1_source"):
                    extracted = "D1sx" in system
                    values: list[Any] = ["v0", "v1", "v2"]
                    note = None
                    if extracted:
                        note = bad if fact == "k.beta" else clean
                        if fact == "k.beta":
                            values = ["v0", "wrong", "v2"]  # differs from D1s at k = 1 only
                    rows.append(
                        {
                            "world_id": w,
                            "branch": b,
                            "system": system,
                            "item_id": f"{fact}#0",
                            "slot_id": fact,
                            "history": _history(values, note),
                        }
                    )
    with (d / "items.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def _extraction_rows(d: Path) -> None:
    typed = {
        "conversation_with": "Ana Lopez",
        "contact": "p03",
        "sender": "Ana Lopez",
        "body": "Hi!",
        "sent": "2026-02-03 18:05",
    }
    rows = []
    for i, w in enumerate(WORLDS):
        wrong = {**typed, "sender": "Kim Doe"} if i == 0 else {**typed, "body": "hi"}
        for b in ("src", "def"):  # both branches read the same revisions: counted once
            for oid, extracted in (("messages:m1", typed), ("messages:m2", wrong)):
                content = {**typed, "body": f"{typed['body']} {oid}"}
                got = {**extracted, "body": f"{extracted['body']} {oid}"}
                rows.append(
                    {
                        "world_id": w,
                        "branch": b,
                        "object_id": oid,
                        "revision": 1,
                        "source": "messages",
                        "typed": content,
                        "prose": "Ana Lopez wrote ...",
                        "rewrite": {
                            "first_issues": [["missing", "x"]] if oid.endswith("2") else [],
                            "retried": oid.endswith("2"),
                            "final_issues": [],
                        },
                        "extracted": got,
                        "extract_ok": True,
                    }
                )
    with (d / "extractions.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


@pytest.fixture(autouse=True)
def fixture_worlds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(w3, "W_WORLDS", WORLDS)


@pytest.fixture
def runs(tmp_path: Path) -> tuple[Path, Path]:
    new = write_run(tmp_path / "e1_w_qwen_s1000-1003_20261020T000000Z", NEW, "2026-10-20")
    _items(new)
    _extraction_rows(new)
    committed = write_run(tmp_path / "e1_test_s1000-1003_20260928T202242Z", COMMITTED, "2026-09")
    return new, committed


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    path = tmp_path / "cache.sqlite"
    ResponseCache(path).put(CODE_KEY, "fake", {"text": json.dumps({"code": CODE})})
    return path


def test_extraction_accuracy_per_source_and_field(runs):
    res = w3.block_w([runs[0]], draws=200)
    m = res["quantities"]["extraction"]["messages"]
    assert m["records"] == 2 * len(WORLDS)  # two revisions per world, both branches once
    # m2: a wrong sender in one world; elsewhere "Hi!" against "hi", the same after
    # normalisation but not exact
    assert m["record_correct"]["est"] == pytest.approx(7 / 8)
    assert m["record_exact"]["est"] == pytest.approx(0.5)
    assert m["fields"]["sender"]["correct"] == pytest.approx(7 / 8)
    assert m["fields"]["body"] == {"n": 8, "correct": 1.0, "exact": pytest.approx(5 / 8)}
    assert m["rewrite"]["first_attempt_failed"] == pytest.approx(0.5)
    assert m["rewrite"]["first_failures"] == {"missing": 4} and m["rewrite"]["missed"] == 0
    assert m["with_error"] == 1
    pooled = res["quantities"]["extraction"]["pooled"]
    assert pooled["records"] == 8 and "messages.sender" in pooled["fields"]


def test_r_and_bxd_extracted_against_typed(runs):
    q = w3.block_w([runs[0]], draws=200)["quantities"]
    for b in ("src", "def"):
        r = q["r"][b]
        assert r["pooled"]["D1s"]["est"] == 1.0 and r["pooled"]["D1sx"]["est"] == 0.5
        assert r["pooled"]["difference"]["est"] == -0.5
        assert r["k.alpha"]["difference"]["est"] == 0.0
        assert r["k.beta"]["difference"]["est"] == -1.0
        assert r["detection_differs"] == 0
        d = q["bxd"][b]["definition"]
        assert d["typed"]["est"] == 1.0 and d["extracted"]["est"] == 0.5
        assert d["difference"]["est"] == -0.5
        assert q["bxd"][b]["other"]["difference"]["est"] == 0.0


def test_attribution_with_and_without_the_program_cache(runs, cache):
    q = w3.block_w([runs[0]], draws=200)["quantities"]["attribution"]
    b1 = q["B1"]["src"]
    # definition item-times answered after the fork: 2 facts x 2 checkpoints x 4 worlds
    assert b1["item_times"] == 16 and b1["differ"] == 4
    assert b1["differ_with_error"] == 4 and b1["share_with_error"] == 1.0
    assert b1["by_definition"]["k.beta"]["differ"] == 4
    assert "differ_with_error_in_named_field" not in b1
    assert q["B0"]["src"]["item_times"] == 0 and q["B0"]["src"]["no_write"] == 16
    named = w3.block_w([runs[0]], response_cache=cache, draws=200)["quantities"]["attribution"]
    assert named["B1"]["def"]["differ_with_error_in_named_field"] == 4
    assert named["B1"]["def"]["share_with_error_in_named_field"] == 1.0
    assert named["B1"]["def"]["programs_missing_in_cache"] == 0


def test_threshold_tests_and_pairing(runs, tmp_path):
    new, committed = runs
    res = w3.block_w([new], [committed], draws=200)
    assert res["pairing"]["ok"], res["pairing"]["counts"]
    assert res["pairing"]["anchor_answers_compared"] > 0
    for name in ("W1", "W2"):
        tests = res["threshold_tests"][name]["tests"]
        assert len(tests) == 2 and all(t["meets_threshold"] for t in tests), name
    assert "block W" in w3.summary(res)
    # an anchor that answers differently voids the run
    moved = {**COMMITTED, "holdmem_D1sB0": const(True)}
    other = write_run(tmp_path / "e1_test_moved_20260928T202243Z", moved, "2026-09")
    void = w3.block_w([new], [other], draws=200)
    assert not void["pairing"]["ok"] and void["pairing"]["counts"]["anchor_answers_differ"] > 0
    assert all(p["tests"] for p in void["threshold_tests"].values())  # still reported


def test_missing_worlds_are_listed(runs, monkeypatch):
    monkeypatch.setattr(w3, "W_WORLDS", (*WORLDS, "w1009"))
    res = w3.block_w([runs[0]], draws=200)
    for p in res["threshold_tests"].values():
        assert p["worlds_missing"] == ["w1009"] and all(t["meets_threshold"] for t in p["tests"])


def test_cli_writes_the_result(runs, tmp_path, capsys):
    out = tmp_path / "w.json"
    w3.main(["--runs", str(runs[0]), "--draws", "100", "--out", str(out)])
    assert json.loads(out.read_text())["block"] == "W"
    assert "W1" in capsys.readouterr().out

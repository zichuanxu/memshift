"""A synthetic multi-system E5 fixture shared by the tests of the native null baselines: three
native systems (two add-path systems and one document-level system predicted by the cited-cluster
cell), the six white-box cells, the zero / RAG / oracle baselines on all three branches, a second
ingestion of two worlds and orphan probes that give each system its persistence rate."""

import json
from pathlib import Path
from typing import Any

from memshift.analysis import e5_infer
from memshift.gen.shifts import inference_half

WORLDS = tuple(f"w{i}" for i in range(6))
MEASURE = [
    f
    for i in range(1, 80)
    if not inference_half(f := f"c.p{i:02d}.city")  # the half that measures carryover
]
CHANGED, UNCHANGED = MEASURE[:6], MEASURE[6:12]
ELIGIBLE = [
    f
    for i in range(1, 80)
    if inference_half(f := f"c.q{i:02d}.city")  # the half that infers labels (structured, "c.")
][:6]
BRANCHES = ("src", "def", "orphan")
# ContractValid rates (changed items, unchanged items); the same on every branch
RATES: dict[str, tuple[float, float]] = {
    "holdmem_D0B0": (0.3, 0.7),
    "holdmem_D1B0": (0.8, 0.7),
    "holdmem_D0B1_source": (0.8, 1.0),
    "holdmem_D1B1_source": (1.0, 1.0),
    "holdmem_D0B1_cited": (0.7, 0.9),
    "holdmem_D1B1_cited": (1.0, 0.9),
    "mem0": (0.3, 0.7),  # looks like (0,0)
    "amem": (0.6, 0.6),  # between the cells
    "lightrag": (0.7, 0.9),  # looks like (0,1), cited cluster
    "zero": (0.3, 0.7),  # the label-free baselines
    "rag": (0.4, 0.6),
    "oracle": (0.9, 0.9),
}
NATIVE = ("mem0", "amem", "lightrag")
BASELINES = ("zero", "rag", "oracle")
LABELS = {"mem0": 0, "amem": 0, "lightrag": 1}  # B label (D is 0 by export structure)
PERSIST = {"mem0": 18, "amem": 32, "lightrag": 4}  # of 36 orphan probes answered with the old value


def _cv(system: str, changed: bool, w: int, f: int) -> bool:
    rate = RATES[system][0 if changed else 1]
    return (w * 3 + f) % 10 < round(rate * 10)


def _probe(
    w: str, branch: str, fact: str, k: int, status: str, value: str | None
) -> dict[str, Any]:
    return {
        "world_id": w,
        "branch": branch,
        "fact_id": fact,
        "vtype": "org",
        "gold_status": status,
        "gold_value": value,
        "as_of": f"2026-01-0{k + 1}",
        "tags": {"k": k},
    }


def _answer(system: str, p: dict[str, Any], cv: bool, value: str | None = "a") -> dict[str, Any]:
    return {
        "system": system,
        "world_id": p["world_id"],
        "branch": p["branch"],
        "fact_id": p["fact_id"],
        "vtype": "org",
        "status": "answer" if value is not None else "abstain",
        "value": value,
        "gold_status": p["gold_status"],
        "gold_value": p["gold_value"],
        "contract_valid": cv,
        "tags": p["tags"],
    }


def _probes() -> list[dict[str, Any]]:
    out = []
    for w in WORLDS:
        for b in BRANCHES:
            for fact in CHANGED + UNCHANGED + ELIGIBLE:
                out.append(_probe(w, b, fact, 0, "answer", "a"))
                if fact in CHANGED:
                    out.append(_probe(w, b, fact, 1, "answer", "b"))
                elif fact in ELIGIBLE and b == "orphan":
                    out.append(_probe(w, b, fact, 1, "abstain", None))
                else:
                    out.append(_probe(w, b, fact, 1, "answer", "a"))
    return out


def _measurement_answers(system: str, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in probes:
        if p["tags"]["k"] != 1 or p["fact_id"] in ELIGIBLE:
            continue
        changed = p["fact_id"] in CHANGED
        facts = CHANGED if changed else UNCHANGED
        out.append(
            _answer(
                system,
                p,
                _cv(system, changed, WORLDS.index(p["world_id"]), facts.index(p["fact_id"])),
            )
        )
    return out


def _orphan_answers(system: str, probes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Orphan probes: the system keeps the old value for `PERSIST[system]` of the 36 items."""
    out = []
    n = 0
    for p in probes:
        if p["branch"] == "orphan" and p["tags"]["k"] == 1 and p["fact_id"] in ELIGIBLE:
            keeps = n < PERSIST[system]
            n += 1
            out.append(_answer(system, p, not keeps, "a" if keeps else None))
    return out


def _write(path: Path, name: str, rows: list[dict[str, Any]]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))


def _second_ingestion(first: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """A second ingestion of w0 and w1: the first two changed items of each world answer wrong."""
    out = []
    for a in first:
        if a["world_id"] in ("w0", "w1") and a["fact_id"] not in ELIGIBLE:
            b = dict(a)
            if a["fact_id"] in CHANGED[:2]:
                b["contract_valid"] = False
            out.append(b)
    return out


def build_runs(tmp_path: Path, monkeypatch: Any) -> tuple[Path, Path, Path, Path]:
    """(native dir, white-box dir, noise dir, baseline dir) and the labels patched in."""
    probes = _probes()
    white = tmp_path / "white"
    _write(white, "probes.jsonl", probes)
    _write(white, "items.jsonl", [])
    _write(
        white,
        "answers.jsonl",
        [a for s in RATES if s.startswith("holdmem_") for a in _measurement_answers(s, probes)],
    )
    native = tmp_path / "native"
    answers = [
        a for s in NATIVE for a in _measurement_answers(s, probes) + _orphan_answers(s, probes)
    ]
    _write(native, "answers.jsonl", answers)
    _write(native, "probes.jsonl", [])
    _write(native, "world_results.jsonl", [{"system": s} for s in NATIVE])
    (native / "manifest.json").write_text(json.dumps({"params": {"world_params": {}}}))
    noise = tmp_path / "noise"
    _write(noise, "answers.jsonl", _second_ingestion([a for a in answers if a["system"] in NATIVE]))
    base = tmp_path / "base"
    _write(base, "answers.jsonl", [a for s in BASELINES for a in _measurement_answers(s, probes)])
    _write(base, "probes.jsonl", probes)
    lab = {
        s: {
            "B": {"label": LABELS[s], "rate": 0.5, "n": 36},
            "D": {"label": None, "rate": 1.0, "n": 0},
        }
        for s in NATIVE
    }
    monkeypatch.setattr(e5_infer, "infer", lambda *a, **k: lab)
    return native, white, noise, base

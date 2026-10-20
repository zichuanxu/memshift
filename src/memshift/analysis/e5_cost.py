"""Native ingestion cost t_i from the stores of E5 measure runs.

A store keeps its usage counters (`state.json`, saved after every ingest call, also when it fails),
its ledger of ingested units and, under on_error="skip", its failures. Per system:
t_i = counter / attempted units, attempted = ingested + failed, plus the unit in flight when a run
stopped on an error (a run without `world_results.jsonl` did not finish). Wall seconds per unit run
from the store's creation (marker file) to its last counter save; they depend on how loaded the
shared servers were and are reported as measured, not as a property of the system. Several stores of
one system (one per world) are pooled: summed counters over summed attempted units.

Usage: python -m memshift.analysis.stage e5_cost OUT RUN_DIR [RUN_DIR ...]
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COUNTERS = ("llm_calls", "prompt_tokens", "completion_tokens", "embed_calls", "embed_tokens")


def _lines(path: Path) -> int:
    return sum(1 for line in path.open() if line.strip()) if path.exists() else 0


def store_cost(store: Path, finished: bool) -> dict[str, Any]:
    marker = json.loads((store / "memshift_native.json").read_text())
    usage = json.loads((store / "state.json").read_text())["usage"]
    ingested = _lines(store / "ledger.jsonl")
    failed = _lines(store / "failures.jsonl")
    attempted = ingested + failed + (0 if finished else 1)
    wall = (store / "state.json").stat().st_mtime - (store / "memshift_native.json").stat().st_mtime
    return {
        "system": marker["system"],
        "store": str(store),
        "finished": finished,
        "ingested": ingested,
        "failed": failed,
        "attempted": attempted,
        "usage": {k: usage.get(k, 0) for k in COUNTERS},
        "wall_s": wall,
        "t_i": {k: usage.get(k, 0) / max(1, attempted) for k in COUNTERS},
        "wall_s_per_unit": wall / max(1, attempted),
    }


def pool(rows: list[dict[str, Any]]) -> dict[str, Any]:
    attempted = sum(r["attempted"] for r in rows)
    return {
        "system": rows[0]["system"],
        "stores": len(rows),
        "finished": all(r["finished"] for r in rows),
        **{k: sum(r[k] for r in rows) for k in ("ingested", "failed", "attempted")},
        "t_i": {k: sum(r["usage"][k] for r in rows) / max(1, attempted) for k in COUNTERS},
        "wall_s_per_unit": sum(r["wall_s"] for r in rows) / max(1, attempted),
    }


def analyse(*run_dirs: Path, out: Path | None = None) -> dict[str, Any]:
    rows = []
    for d in run_dirs:
        finished = (d / "world_results.jsonl").exists()
        for store in sorted(d.glob("stores/*/prefix")):
            if (store / "state.json").exists():
                rows.append({"run": d.name, **store_cost(store, finished)})
    systems = sorted({r["system"] for r in rows})
    by_system = {s: pool([r for r in rows if r["system"] == s]) for s in systems}
    res = {"stores": rows, "by_system": by_system}
    adir = (out or run_dirs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "e5_cost.json").write_text(json.dumps(res, indent=1))
    return res

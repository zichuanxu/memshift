"""Offline estimate of the model calls an E1 invocation makes, without a model or a GPU.

    python -m memshift.experiments.e1_estimate --seeds 1000-1001 --branches src,def,orphan \\
        --cells D0B1_gold,D1B1_gold,D0B0_refresh,zero_Dm

Every cell's writes are replayed with a stand-in model that only records requests (writer and
program requests do not depend on any model output, so the record is exact). Cells go through the
same build at t0 and maintenance at the fork + {1, 2, 4, 8} weeks as in `e1.run_world`. A request is
a response-cache key (`llm.LLMClient.request_body`), so the number of distinct requests is the
number of calls on a cold cache, and the distinct requests that no reference cell sends are the
calls the cache of the reference cells cannot serve. Per cell and world:

  writer_new / program_new   distinct requests that no reference cell sends (the default
                             reference: the six cells of `e1.CELLS`, the two D1s cells of block C)
  think                      requests of the thinking writer (all new: their body differs)
  reader_upper               an upper bound on new reader calls: the asked probes after the fork
                             (a cell's contexts are identical to its reference's before the shift,
                             so those reader calls are cache hits); zero_Dm: every asked probe
  view_runs                  zero_Dm: sandbox executions, one per probe with a runnable program
  full_access                D0full: what its writer is given at each re-derivation of a definition
                             item (records, records dropped by the cap `writer.FULL_BUDGET`, record
                             and prompt tokens by `approx_tokens`), and in how many the definition's
                             scope range (`gen.kappa`, read here for the report only) is not
                             fully given. The cell's own code reads none of that

Program requests of world-independent items (user attributes, appointments, definitions) are shared
between worlds, so the sum over worlds of `program_new` overstates a multi-world run that shares one
cache; the first world shows the upper bound.
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from statistics import median
from typing import Any
from unittest import mock

from memshift.adapters.baselines import ZeroDm
from memshift.experiments.common import world_params
from memshift.experiments.e1 import (
    BRANCHES,
    CELLS,
    OFFSETS_WEEKS,
    discriminative_probes,
    parse_cells,
    parse_seeds,
)
from memshift.gen.kappa import scope_members
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import full_records, writer_request
from memshift.llm import LLMClient, LLMResult
from memshift.probes import make_probes
from memshift.textutil import approx_tokens, sha256_json

MODEL = "estimate/writer"
REFERENCE = [*CELLS, CellConfig(1, 0, synth=True), CellConfig(1, 1, "source", synth=True)]
_WRITER_REPLY: dict[str, Any] = {"value": None, "statement": "", "citations": []}
_PROGRAM = "def derive(records, as_of):\n    return None\n"
_PROGRAM_RA = "def derive(records, as_of, prior_reads):\n    return {'value': None, 'reads': []}\n"


class Recorder:
    """Stands in for `LLMClient`: answers every request with a valid empty reply and records the
    response-cache key it would have had."""

    def __init__(self, extra_body: dict[str, Any] | None = None):
        self.model = MODEL
        self.api = "openai"  # LLMClient.request_body branches on it
        self.extra_body = extra_body or {}
        self.requests: list[tuple[str, str]] = []  # (cache key, purpose)
        self._lock = threading.Lock()

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        body = LLMClient.request_body(self, messages, max_tokens, schema)  # type: ignore[arg-type]
        key = sha256_json(body)
        with self._lock:
            self.requests.append((key, purpose))
        if purpose == "program":
            code = _PROGRAM_RA if "prior_reads" in messages[-1]["content"] else _PROGRAM
            text = json.dumps({"code": code})
        else:
            text = json.dumps(_WRITER_REPLY)
        return LLMResult(text, 0, 0, 0.0, False, "stop", key)

    def keys(self, purpose: str) -> set[str]:
        return {k for k, p in self.requests if p == purpose}


def _replay(
    world: Any, log: SourceLog, cfgs: list[CellConfig]
) -> dict[str, tuple[Recorder, Recorder | None, HoldMem]]:
    """Build and maintain each cell once: its writer recorder, its thinking one (or None) and the
    cell."""
    t0 = world.fork - timedelta(minutes=1)
    out: dict[str, tuple[Recorder, Recorder | None, HoldMem]] = {}
    for cfg in cfgs:
        writer = Recorder()
        think = Recorder({"think": True}) if cfg.think else None
        hm = HoldMem(world, log, cfg, writer, MODEL, think_writer=think)  # type: ignore[arg-type]
        hm.build(t0)
        for k in OFFSETS_WEEKS:
            hm.maintain(world.fork + timedelta(weeks=k))
        out[cfg.name] = (writer, think, hm)
    return out


def full_access_rows(world: Any, log: SourceLog, hm: HoldMem) -> list[dict[str, Any]]:
    """One row per re-derivation of a definition item by a D0full cell: what its writer was given
    (the cell's own record, `Item.note`), the approximate tokens of the whole request, and how many
    of the records the definition ranges over at that time were not given (`range_missing`). The
    range is read only here, to say whether a truncation cut into it."""
    rows = []
    for item in hm.items.values():
        slot = world.slot(item.slot_id)
        for h in (h for h in item.history if "full" in h):
            t = datetime.fromisoformat(h["ts"])
            access = full_records(log, slot, t)
            given = {(d.doc_id, d.revision) for d, _ in access.records}
            in_range = scope_members(
                slot.kappa_id, log.snapshot(t), t, world.kappa_version(slot.kappa_id, t)
            )
            prompt = sum(
                approx_tokens(m["content"])
                for m in writer_request(world, slot, t, access.records)["messages"]
            )
            rows.append(
                {
                    "item": slot.kappa_id,
                    "t": h["ts"],
                    **h["full"],
                    "prompt_tokens": prompt,
                    "range_records": len(in_range),
                    "range_missing": sum(m not in given for m in in_range),
                }
            )
    return rows


def _spread(values: list[int]) -> dict[str, float]:
    return {"min": min(values), "median": median(values), "max": max(values)}


def summarise_full_access(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How often the cap bites (`truncated`; `newest_first`: the fallback rule, the sources other
    than the largest exceeding the cap), from which sources, and how large the requests are (one
    entry for all rows, one per definition)."""

    def block(part: list[dict[str, Any]]) -> dict[str, Any]:
        dropped_from = sorted({s for r in part for s, v in r["by_source"].items() if v["dropped"]})
        return {
            "rederivations": len(part),
            "truncated": sum(r["dropped"] > 0 for r in part),
            "newest_first": sum(r["truncation"] == "newest" for r in part),
            "range_incomplete": sum(r["range_missing"] > 0 for r in part),
            "dropped_by_source": {
                s: _spread([r["by_source"].get(s, {"dropped": 0})["dropped"] for r in part])
                for s in dropped_from
            },
            "records": _spread([r["records"] for r in part]),
            "dropped": _spread([r["dropped"] for r in part]),
            "record_tokens": _spread([r["tokens"] for r in part]),
            "prompt_tokens": _spread([r["prompt_tokens"] for r in part]),
        }

    by_item = {
        k: block([r for r in rows if r["item"] == k]) for k in sorted({r["item"] for r in rows})
    }
    return {**block(rows), "by_item": by_item} if rows else {}


@dataclass
class Requests:
    """What a cell (or the reference) sends on one branch, as response-cache keys."""

    write: set[str] = field(default_factory=set)
    program: set[str] = field(default_factory=set)
    think: set[str] = field(default_factory=set)
    reader: int = 0  # asked probes after the fork (zero_Dm: all asked probes)
    view_runs: int = 0
    full: list[dict[str, Any]] = field(default_factory=list)  # D0full: `full_access_rows`

    def merge(self, other: Requests) -> None:
        self.write |= other.write
        self.program |= other.program
        self.think |= other.think
        self.reader += other.reader
        self.view_runs += other.view_runs
        self.full += other.full


def estimate_branch(
    seed: int,
    branch: str,
    cells: list[CellConfig],
    views: tuple[str, ...],
    months: int,
    reference: list[CellConfig],
) -> dict[str, Requests]:
    """The requests of each cell on one branch; the reference cells together: `reference`."""
    world = generate_branch(seed, branch, world_params(months))
    log = SourceLog(world.events)
    t0 = world.fork - timedelta(minutes=1)
    pre = make_probes(world, log, [t0])
    post = [
        p
        for k in OFFSETS_WEEKS
        for p in discriminative_probes(
            make_probes(world, log, [world.fork + timedelta(weeks=k)]),
            pre,
            f"{world.world_id}|{branch}|{k}",
        )
    ]
    out: dict[str, Requests] = {}
    # the sandbox runs are a function of the program and the records, not of the requests
    stub: Any = mock.patch("memshift.holdmem.program._run_cached", lambda *a: (True, None, "", 0.0))
    stub_ra: Any = mock.patch(
        "memshift.holdmem.program._run_cached_ra", lambda *a: (True, None, [], "", 0.0)
    )
    with stub, stub_ra:
        ref = Requests()
        for writer, _, _ in _replay(world, log, reference).values():
            ref.write |= writer.keys("write")
            ref.program |= writer.keys("program")
        out["reference"] = ref
        for name, (writer, think, hm) in _replay(world, log, cells).items():
            out[name] = Requests(
                writer.keys("write"),
                writer.keys("program"),
                think.keys("write") if think else set(),
                len(post),
                full=(
                    [{"branch": branch, **r} for r in full_access_rows(world, log, hm)]
                    if hm.cfg.full
                    else []
                ),
            )
        for view in views:
            recorder = Recorder()
            zero = ZeroDm(world, log, recorder)  # type: ignore[arg-type]
            probes = [*pre, *post]
            for p in probes:
                zero.view(p)
            out[view] = Requests(set(), recorder.keys("program"), set(), len(probes), len(probes))
    return out


def estimate(
    seeds: list[int],
    branches: tuple[str, ...],
    cells: list[CellConfig],
    views: tuple[str, ...],
    months: int = 12,
    reference: list[CellConfig] | None = None,
) -> dict[str, Any]:
    """Per world: the calls of each cell on a cold cache shared by the branches (the branches of a
    world share every request whose records are the same, so requests are united, not added)."""
    ref = REFERENCE if reference is None else reference
    per_world: dict[str, dict[str, dict[str, Any]]] = {}
    for seed in seeds:
        merged: dict[str, Requests] = {}
        for branch in branches:
            for name, req in estimate_branch(seed, branch, cells, views, months, ref).items():
                merged.setdefault(name, Requests()).merge(req)
        base = merged.pop("reference")
        per_world[f"seed{seed}"] = {
            name: {
                "writer_distinct": len(req.write),
                "writer_new": len(req.write - base.write),
                "program_distinct": len(req.program),
                "program_new": len(req.program - base.program),
                "think": len(req.think),
                "reader_upper": req.reader,
                **({"view_runs": req.view_runs} if name in views else {}),
                **({"full_access": summarise_full_access(req.full)} if req.full else {}),
            }
            for name, req in merged.items()
        }
    return {"branches": list(branches), "months": months, "per_world": per_world}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--branches", default=",".join(BRANCHES))
    ap.add_argument("--cells", required=True, help="as for e1 --cells")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--reference", default="", help="cells already cached; default: CELLS + D1s")
    a = ap.parse_args()
    cfgs, view_names = parse_cells(a.cells)
    ref_cfgs = parse_cells(a.reference)[0] if a.reference else None
    result = estimate(
        parse_seeds(a.seeds),
        tuple(a.branches.split(",")),
        cfgs,
        view_names,
        a.months,
        ref_cfgs,
    )
    print(json.dumps(result, indent=1))

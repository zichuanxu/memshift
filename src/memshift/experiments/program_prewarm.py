"""Pre-warm a program set of prereg v4 block G-b (or v4b block V): send the definition items' D1s
program requests that an E1 replay of the same worlds and branches sends to the program model, so
that the replay later finds every one of them in the response cache. GPU1 holds one model at a time
and the program writer and the reader cannot be served together: the program model is served for
this command only, and the replay runs later with only the reader served.

    python -m memshift.experiments.program_prewarm --program-model M --program-temperature T \\
        --program-sample I --seeds 1000-1019 --branches src,def,orphan [--program-user-words]

The flags mean what they mean for `experiments.e1`. Requests go through the client
`e1.run_world` builds for the definition items' programs, `common.client(M, run_dir,
extra_body=e1.program_extra_body(M, T, I))`, so their bodies, and so their response-cache keys and
cache file, are the replay's. When the flags are e1's defaults (the writer, 0, no sample) the
replay's writer client answers these requests with the same body.

Which requests. A program request depends on the item and on the definition version in effect at
the write time, nothing else (`holdmem.program.program_request`). In every D1s cell every item is
written at t0 (fork - 1 minute). In `D1sB1_source` the definition version is part of the key, so at
a checkpoint (fork + `e1.OFFSETS_WEEKS` weeks) whose version differs from the one of the item's
last write the item is re-derived and its program requested. The program requests of a replay with
`D1sB1_source` (G-b and V replay `D1sB0,D1sB1_source`) are thus the requests of every definition
item at t0 and at each checkpoint, deduplicated; `D1sB0` sends a subset of them. Equal requests are
sent once.

The replay after a pre-warm does not contact the program model: `LLMClient` opens no connection
when it is built, a cache hit sends nothing (`LLMClient.complete`), and the manifest and env.json
read serving metadata from the files in `logs/serving/` (`runinfo.serving_info`). The program model
still needs an entry in MEMSHIFT_ENDPOINTS (`common.client`); a request that misses the cache is
then sent there and fails while the model is not served, so an incomplete pre-warm fails the
replay instead of changing it.

Outputs, in a run directory `runs/<name>_<stamp>/`: `calls.jsonl` (the client's call log),
`prewarm.json` (the counts and, per request: cache key, item, definition version, item-times that
need it, whether the cache already held it, finish reason, whether the reply holds code) and
`manifest.json`. A reply without code is an answer (the program set's own output) and is counted,
not retried. An HTTP failure ends the command with an error; the requests answered before it are
cached, so a rerun sends only the rest.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from memshift.experiments.common import client, new_run_dir, slug
from memshift.experiments.e1 import (
    BRANCHES,
    OFFSETS_WEEKS,
    check_program_temperature,
    parse_seeds,
    program_extra_body,
)
from memshift.gen.model import World
from memshift.gen.shifts import generate_branch
from memshift.holdmem.program import extract_code, program_request
from memshift.holdmem.writer import CUT_OFF
from memshift.runinfo import world_record, write_env, write_manifest
from memshift.textutil import sha256_json


@dataclass
class Planned:
    """One distinct program request and the item-times of the replay that send it."""

    request: dict[str, Any]
    slot_id: str
    kappa_id: str
    version: str
    sites: list[str] = field(default_factory=list)  # "w1000/def/k1"


def write_times(world: World) -> list[tuple[int, datetime]]:
    """(k, time) of t0 and of the checkpoints of `e1.run_world`."""
    return [
        (0, world.fork - timedelta(minutes=1)),
        *((k, world.fork + timedelta(weeks=k)) for k in OFFSETS_WEEKS),
    ]


def planned_requests(
    seeds: Sequence[int],
    branches: Sequence[str],
    world_params: dict[str, Any] | None = None,
    user_words: bool = False,
) -> tuple[list[Planned], list[dict[str, Any]]]:
    """The distinct definition-item program requests of a replay of `seeds` x `branches` (module
    docstring), in first-seen order, and the replay's world records."""
    plan: dict[str, Planned] = {}
    worlds = []
    for seed in seeds:
        for branch in branches:
            world = generate_branch(seed, branch, world_params)
            worlds.append(world_record(world))
            for k, t in write_times(world):
                for slot in world.slots:
                    if not slot.kappa_id:
                        continue
                    req = program_request(world, slot, t, user_words=user_words)
                    ver = world.kappa_version(slot.kappa_id, t)
                    entry = plan.setdefault(
                        sha256_json(req), Planned(req, slot.slot_id, slot.kappa_id, ver)
                    )
                    entry.sites.append(f"{world.world_id}/{branch}/k{k}")
    return list(plan.values()), worlds


def prewarm(
    seeds: Sequence[int],
    branches: Sequence[str],
    program_model: str,
    program_temperature: float = 0.0,
    program_sample: int | None = None,
    program_user_words: bool = False,
    world_params: dict[str, Any] | None = None,
    name: str = "",
    out: Path | None = None,
) -> dict[str, Any]:
    """Send the planned requests through the replay's program client; returns the summary that
    `prewarm.json` holds (without the per-request rows)."""
    check_program_temperature(program_model, program_temperature)
    body = program_extra_body(program_model, program_temperature, program_sample)
    plan, worlds = planned_requests(seeds, branches, world_params, program_user_words)
    run_dir = out or new_run_dir(name or default_name(program_model, program_sample))
    llm = client(program_model, run_dir, extra_body=body)
    results = llm.map([p.request for p in plan])
    rows = [
        {
            "key": res.key,
            "slot_id": p.slot_id,
            "kappa_id": p.kappa_id,
            "version": p.version,
            "item_times": len(p.sites),
            "first": p.sites[0],
            "cached": res.cached,
            "finish_reason": res.finish_reason,
            "has_code": extract_code(res) is not None,
        }
        for p, res in zip(plan, results, strict=True)
    ]
    summary: dict[str, Any] = {
        "run_dir": str(run_dir),
        "program_model": program_model,
        "temperature": program_temperature,
        "sample": program_sample,
        "user_words": program_user_words,
        "extra_body": body,
        "seeds": list(seeds),
        "branches": list(branches),
        "requests": len(rows),
        "sent": sum(not r["cached"] for r in rows),
        "hit": sum(r["cached"] for r in rows),
        "no_code": sum(not r["has_code"] for r in rows),
        "cut_off": sum(r["finish_reason"] in CUT_OFF for r in rows),
    }
    (run_dir / "prewarm.json").write_text(json.dumps({**summary, "rows": rows}, indent=1))
    models = {"program_writer": program_model}
    write_env(run_dir, models)
    params = {k: summary[k] for k in ("temperature", "sample", "user_words", "extra_body")}
    params |= {
        "seeds": list(seeds),
        "branches": list(branches),
        "world_params": world_params or {},
        "times": "fork - 1 minute; fork + " + ", ".join(map(str, OFFSETS_WEEKS)) + " weeks",
        "items": "definition items, D1s program requests",
    }
    write_manifest(run_dir, "program_prewarm", worlds, models, params)
    return summary


def default_name(program_model: str, program_sample: int | None) -> str:
    sample = "" if program_sample is None else f"_p{program_sample}"
    return f"prewarm_{slug(program_model)}{sample}"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", required=True, help="as for e1, e.g. 1000-1019")
    ap.add_argument("--branches", default=",".join(BRANCHES))
    ap.add_argument("--program-model", required=True)
    ap.add_argument("--program-temperature", type=float, default=0.0)
    ap.add_argument("--program-sample", type=int, default=None)
    ap.add_argument("--program-user-words", action="store_true")
    ap.add_argument(
        "--name", default="", help="run directory prefix (default prewarm_<model>_p<I>)"
    )
    return ap


if __name__ == "__main__":
    ap = build_parser()
    a = ap.parse_args()
    try:
        check_program_temperature(a.program_model, a.program_temperature)
    except ValueError as exc:
        ap.error(str(exc))
    print(
        json.dumps(
            prewarm(
                parse_seeds(a.seeds),
                tuple(a.branches.split(",")),
                a.program_model,
                a.program_temperature,
                a.program_sample,
                a.program_user_words,
                name=a.name,
            )
        )
    )

"""Stage 0.4 smoke run (plan §7): one world (3 months), HoldMem (1,1) ingest, 50 probes.

Measures t_i (per ingested event: writer tokens and wall-clock) and t_q (per probe: reader tokens
and wall-clock), plus the D=1 checker verdict mix (plan §12: `unverifiable` > 20% triggers the
fallback-heavy alternative). The zero system answers the same probes as a reference point.
"""

from __future__ import annotations

import json
import random
from datetime import timedelta
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import ZeroSystem
from memshift.experiments.common import (
    MAIN_MODEL,
    Answer,
    Timer,
    answer_probes,
    client,
    new_run_dir,
    write_jsonl,
)
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.gen.world import add_months
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.probes import make_probes
from memshift.runinfo import world_record, write_env, write_manifest

SMOKE_PARAMS: dict[str, Any] = {"months": 3, "fork_month": 2, "target_tokens": 500_000}


def _per_probe(ans: list[Answer], secs: float) -> dict[str, float]:
    n = max(len(ans), 1)
    return {
        "n": len(ans),
        "contract_valid": sum(a.contract_valid for a in ans) / n,
        "prompt_tokens_mean": sum(a.prompt_tokens for a in ans) / n,
        "completion_tokens_mean": sum(a.completion_tokens for a in ans) / n,
        "latency_s_mean": sum(a.latency_s for a in ans) / n,
        "wall_s_per_probe_at_concurrency": secs / n,
    }


def _verdicts(hm: HoldMem) -> dict[str, int]:
    """Checker verdicts over every write (initial derivations and repairs)."""
    out: dict[str, int] = {}
    for item in hm.items.values():
        verdict = (item.det or {}).get("check", "none")
        out[verdict] = out.get(verdict, 0) + 1
    return out


def run(seed: int = 0, n_probes: int = 50, out: Path | None = None) -> Path:
    run_dir = out or new_run_dir("smoke")
    llm = client(MAIN_MODEL, run_dir)
    world = generate_branch(seed, "natural", SMOKE_PARAMS)
    log = SourceLog(world.events)
    t_build = add_months(world.start, 1)
    checkpoints = [t_build + timedelta(weeks=k) for k in range(1, 9)]
    t_end = checkpoints[-1]

    hm = HoldMem(world, log, CellConfig(1, 1, "source"), llm, MAIN_MODEL)
    timer = Timer()
    hm.build(t_build)
    build_s = timer.lap()
    build_verdicts = _verdicts(hm)
    maint: list[dict[str, Any]] = []
    for t in checkpoints:
        events = len(log.between(hm.last_t or t_build, t))
        before = hm.cost["maintain_tokens"]
        stats = hm.maintain(t)
        maint.append(
            {
                "t": str(t),
                "events": events,
                "seconds": round(timer.lap(), 2),
                "tokens": hm.cost["maintain_tokens"] - before,
                **stats,
            }
        )
    events_window = sum(m["events"] for m in maint)
    n_writes = len(world.slots) + sum(m["repaired"] for m in maint)

    rng = random.Random(seed)
    probes = make_probes(world, log, [t_end - timedelta(days=14), t_end])
    probes = sorted(rng.sample(probes, min(n_probes, len(probes))), key=lambda p: p.probe_id)
    timer.lap()
    hm_answers, _ = answer_probes(hm.name, hm.serve, probes, llm)
    hm_q_s = timer.lap()
    zero = ZeroSystem(world, log)
    zero_answers, _ = answer_probes("zero", zero.serve, probes, llm)
    zero_q_s = timer.lap()

    write_jsonl(run_dir / "probes.jsonl", (p.to_json() for p in probes))
    write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in hm_answers + zero_answers))
    write_jsonl(run_dir / "items.jsonl", (i.to_json() for i in hm.items.values()))

    summary = {
        "world": world_record(world),
        "n_slots": len(world.slots),
        "build": {
            "t": str(t_build),
            "seconds": round(build_s, 2),
            "tokens": hm.cost["ingest_tokens"],
            "writes": len(world.slots),
            "d1_check_verdicts": build_verdicts,
        },
        "maintenance": maint,
        "t_i": {
            "events": events_window,
            "writer_tokens_per_event": hm.cost["maintain_tokens"] / max(events_window, 1),
            "wall_s_per_event": sum(m["seconds"] for m in maint) / max(events_window, 1),
            "writer_tokens_per_write": (hm.cost["ingest_tokens"] + hm.cost["maintain_tokens"])
            / max(n_writes, 1),
            "writes": n_writes,
        },
        "t_q": {
            "holdmem": _per_probe(hm_answers, hm_q_s),
            "zero": _per_probe(zero_answers, zero_q_s),
        },
        "d1_final_check_verdicts": _verdicts(hm),
        "cost": hm.cost,
        "llm_usage": llm.usage,
    }
    analysis = run_dir / "analysis"
    analysis.mkdir(exist_ok=True)
    (analysis / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    models = {"writer": MAIN_MODEL, "reader": MAIN_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "smoke-0.4",
        [world_record(world)],
        models,
        {
            "world_params": SMOKE_PARAMS,
            "seed": seed,
            "n_probes": n_probes,
            "cell": hm.name,
            "build_at": str(t_build),
            "checkpoints": [str(t) for t in checkpoints],
        },
    )
    return run_dir

"""Write a generated world to disk: JSONL + manifest + SHA256 (plan §4.2 "规模与种子").

worlds/<world_id>/<branch>/events.jsonl   one Event per line (see model.Event.to_json)
worlds/<world_id>/<branch>/world.json     slots, timelines, kappa, agents, consent, params
worlds/<world_id>/<branch>/fidelity.json  fidelity checker report
worlds/<world_id>/<branch>/manifest.json  seed, branch, counts, SHA256 of the files above
Generation is deterministic (seed -> world), so these files are a record, not an input.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from memshift.gen.fidelity import check_world
from memshift.gen.model import World, fmt_ts
from memshift.runinfo import file_sha256


def world_meta(world: World) -> dict[str, Any]:
    return {
        "world_id": world.world_id,
        "seed": world.seed,
        "branch": world.branch,
        "start": fmt_ts(world.start),
        "end": fmt_ts(world.end),
        "fork": fmt_ts(world.fork),
        "user_name": world.user_name,
        "people": {k: asdict(v) for k, v in world.people.items()},
        "slots": [asdict(s) for s in world.slots],
        "timelines": {
            k: [{"ts": fmt_ts(p.ts), "value": p.value, "deps": p.deps} for p in pts]
            for k, pts in world.timelines.items()
        },
        "kappa_defs": {
            k: {v: d.statement for v, d in vs.items()} for k, vs in world.kappa_defs.items()
        },
        "kappa_log": [[fmt_ts(t), k, v] for t, k, v in world.kappa_log],
        "agents": world.agents,
        "consent_log": [[fmt_ts(t), a, s, st] for t, a, s, st in world.consent_log],
        "params": world.params,
    }


def write_world(world: World, root: Path) -> Path:
    d = root / world.world_id / world.branch
    d.mkdir(parents=True, exist_ok=True)
    with (d / "events.jsonl").open("w", encoding="utf-8") as f:
        for e in world.events:
            f.write(json.dumps(e.to_json(), ensure_ascii=False, sort_keys=True) + "\n")
    (d / "world.json").write_text(json.dumps(world_meta(world), indent=1, ensure_ascii=False))
    rep = check_world(world)
    fidelity = {
        "checked": rep.checked,
        "located": rep.located,
        "rate": rep.rate,
        "failures": rep.failures,
    }
    (d / "fidelity.json").write_text(json.dumps(fidelity, indent=1))
    manifest = {
        "world_id": world.world_id,
        "seed": world.seed,
        "branch": world.branch,
        "n_events": len(world.events),
        "n_slots": len(world.slots),
        "fidelity_rate": rep.rate,
        "files_sha256": {
            p.name: file_sha256(p) for p in sorted(d.iterdir()) if p.name != "manifest.json"
        },
    }
    (d / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return d

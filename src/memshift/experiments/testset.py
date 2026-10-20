"""Test-set generation record: every branch of every test world is generated from
its seed, its event SHA256 goes into the manifest, and the fidelity checker (every gold fact is
locatable in the text of its dependencies, `memshift.gen.fidelity`) runs on each branch.

Experiments regenerate worlds from seeds; this run fixes which bytes those seeds produce at the
frozen commit, so later runs can be checked against it (same `events_sha256`).
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from memshift.experiments.common import new_run_dir
from memshift.experiments.e1 import parse_seeds
from memshift.gen.fidelity import check_world
from memshift.gen.shifts import BRANCHES, generate_branch
from memshift.runinfo import world_record, write_env, write_manifest


def one(job: tuple[int, str]) -> dict[str, Any]:
    seed, branch = job
    world = generate_branch(seed, branch)
    rep = check_world(world)
    return {
        "record": world_record(world),
        "fidelity": {"checked": rep.checked, "located": rep.located, "failures": rep.failures},
    }


def run(seeds: list[int], name: str = "testset", workers: int = 8) -> Path:
    run_dir = new_run_dir(name)
    jobs = [(s, b) for s in seeds for b in BRANCHES]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(one, jobs))
    checked = sum(r["fidelity"]["checked"] for r in results)
    located = sum(r["fidelity"]["located"] for r in results)
    summary = {
        "seeds": seeds,
        "branches": list(BRANCHES),
        "worlds": len(results),
        "fidelity_checked": checked,
        "fidelity_located": located,
        "fidelity_rate": located / checked if checked else float("nan"),
        "per_world": [
            {
                "world_id": r["record"]["world_id"],
                "branch": r["record"]["branch"],
                "events_sha256": r["record"]["events_sha256"],
                **r["fidelity"],
            }
            for r in results
        ],
    }
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "analysis" / "testset.json").write_text(json.dumps(summary, indent=1))
    write_env(run_dir, {})
    write_manifest(run_dir, "testset", [r["record"] for r in results], {}, {"seeds": seeds})
    return run_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--name", default="testset")
    a = ap.parse_args()
    print(run(parse_seeds(a.seeds), a.name))

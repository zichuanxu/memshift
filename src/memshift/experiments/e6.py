"""E6 (plan §3.2 T2, §5.5, §6 E6): the leave test through Portable Memory (.mem).

Per world and branch (src, def): HoldMem (1,1) with source-epoch binding is built at
t0 = t_f - 1 min and exported twice, with the derivation record (custom record kind, which the
format round-trips verbatim) and without it (standard fact records only). Each bundle is imported
by a second, independent HoldMem implementation (`mem_export.import_to_holdmem`). All stores then
receive the branch's source changes and are maintained and probed at t_f + {1, 2, 4, 8} weeks.
Outputs follow E1 (probes, answers, items with histories), so `memshift.analysis.e1` computes
MissedInvalidation of the imported stores; ExportFidelity = ContractValid(import) /
ContractValid(native) at t0 and after the shift. `carriers.json` records which extension carriers
survive a spec-compliant round trip (`mem_export.ext_survival`).
"""

from __future__ import annotations

import json
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

from memshift.adapters.mem_export import export_holdmem, ext_survival, import_to_holdmem
from memshift.experiments.common import (
    MAIN_MODEL,
    Answer,
    answer_probes,
    client,
    new_run_dir,
    write_jsonl,
)
from memshift.experiments.e1 import OFFSETS_WEEKS, _par, parse_seeds
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.llm import LLMClient
from memshift.probes import Probe, make_probes
from memshift.runinfo import world_record, write_env, write_manifest

CELL = CellConfig(1, 1, "source")


def _answer(item: tuple[str, Any], probes: list[Probe], llm: LLMClient) -> list[Answer]:
    return answer_probes(item[0], item[1].serve, probes, llm)[0]


def _mem0_arms(world: Any, log: SourceLog, bdir: Path, run_dir: Path, t0: Any) -> dict[str, Any]:
    """Mem0 as importer: each bundle enters through Mem0's raw add path, later source events
    through Mem0's own extraction path (plan §6 E6 "导入方：Mem0 自身导入路径")."""
    from memshift.adapters.mem_export import import_to_mem0
    from memshift.experiments.common import EMBED_MODEL, endpoints
    from memshift.experiments.e5 import NATIVE_LLM

    eps = endpoints()
    llm_url = eps.get(NATIVE_LLM, eps[MAIN_MODEL]).split("|")[0]
    arms: dict[str, Any] = {}
    for tag in ("with_ext", "no_ext"):
        store = run_dir / "stores" / f"{world.world_id}_{world.branch}_mem0_{tag}"
        arms[f"mem0_imported_{tag}"] = import_to_mem0(
            bdir / f"{tag}.mem",
            world,
            store,
            llm_url,
            MAIN_MODEL,
            eps[EMBED_MODEL],
            EMBED_MODEL,
            world.world_id,
        )
    return arms


def run_world(seed: int, run_dir: Path, llm: LLMClient, mem0: bool = False) -> dict[str, Any]:
    out: dict[str, Any] = {"seed": seed, "worlds": [], "carriers": None}
    for branch in ("src", "def"):
        world = generate_branch(seed, branch)
        log = SourceLog(world.events)
        out["worlds"].append(world_record(world))
        t0 = world.fork - timedelta(minutes=1)
        native = HoldMem(world, log, CELL, llm, MAIN_MODEL)
        native.build(t0)
        bdir = run_dir / "bundles" / f"{world.world_id}_{branch}"
        export_holdmem(native, bdir / "with_ext.mem", with_ext=True, carriers=("kind",))
        export_holdmem(native, bdir / "no_ext.mem", with_ext=False)
        if out["carriers"] is None:
            every = ("kind", "episode", "fact")
            export_holdmem(native, bdir / "all_carriers.mem", with_ext=True, carriers=every)
            out["carriers"] = ext_survival(bdir / "all_carriers.mem")
        stores: dict[str, HoldMem] = {"native_holdmem_D1B1_source": native}
        for tag in ("with_ext", "no_ext"):
            imp = import_to_holdmem(bdir / f"{tag}.mem", world, log, CELL, llm, MAIN_MODEL, t0)
            stores[f"imported_{tag}"] = imp
        natives = _mem0_arms(world, log, bdir, run_dir, t0) if mem0 else {}
        answers: list[Answer] = []
        probe_rows = []
        prev = t0
        for k in (0, *OFFSETS_WEEKS):
            if k == 0:
                probes = make_probes(world, log, [t0], tags={"phase": "pre", "k": 0})
            else:
                t = world.fork + timedelta(weeks=k)
                _par(partial(HoldMem.maintain, t=t), list(stores.values()))
                for system in natives.values():
                    system.ingest(log.between(prev, t), "maintain")
                prev = t
                probes = make_probes(world, log, [t], tags={"phase": "post", "k": k})
            probe_rows += [p.to_json() for p in probes]
            servers = [*stores.items(), *natives.items()]
            for ans in _par(partial(_answer, probes=probes, llm=llm), servers, 3):
                answers += ans
        for system in natives.values():
            system.close()
        write_jsonl(run_dir / "probes.jsonl", probe_rows, mode="a")
        write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
        write_jsonl(
            run_dir / "items.jsonl",
            (
                {"world_id": world.world_id, "branch": branch, "system": name, **i.to_json()}
                for name, hm in stores.items()
                for i in hm.items.values()
            ),
            mode="a",
        )
    return out


def run(seeds: list[int], name: str = "e6", mem0: bool = False) -> Path:
    run_dir = new_run_dir(name)
    llm = client(MAIN_MODEL, run_dir)
    # Mem0's embedded Qdrant is driven from one thread per store: worlds run sequentially then
    results = (
        [run_world(s, run_dir, llm, mem0) for s in seeds]
        if mem0
        else _par(lambda s: run_world(s, run_dir, llm), seeds, 3)
    )
    (run_dir / "analysis").mkdir(exist_ok=True)
    (run_dir / "analysis" / "carriers.json").write_text(
        json.dumps(results[0]["carriers"], indent=1, default=str)
    )
    models = {"writer": MAIN_MODEL, "reader": MAIN_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E6",
        [w for r in results for w in r["worlds"]],
        models,
        {
            "seeds": seeds,
            "cell": CELL.name,
            "format": "portable-memory 0.3.0 (format 1.1)",
            "mem0_arms": mem0,
        },
    )
    return run_dir


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--name", default="e6")
    ap.add_argument("--mem0", action="store_true", help="also import into Mem0")
    a = ap.parse_args()
    print(run(parse_seeds(a.seeds), a.name, a.mem0))

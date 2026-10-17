"""Block X of prereg/v4_round3.md: diagnosis and intervention on Mem0.

    python -m memshift.experiments.x_mem0 --seeds 1000 [--branches src,def,orphan]
        [--arms mem0,mem0_det,mem0_rep] [--prefix-policy auto|resume|rebuild] [--e5-run RUN]
    python -m memshift.experiments.x_mem0 --seeds 1000-1009 --dry-run [--e5-root DIR]

Mem0 as in E5 (`experiments.e5`: mem0ai 2.2.1, the same configuration, LLM Qwen3.5-9B with thinking
off, Qwen3-Embedding-0.6B, on_error "skip", which here never covers an infrastructure error: those
are retried and then stop the run, `native_mem0x.RetryPolicy`, `--request-timeout`, `--attempts`),
short worlds (`world_params(3)`), branches src, def and
orphan, the probes, reader and answer path of E5 (every item at the fork + {1, 2, 4, 8} weeks,
orphan 1 week). Arms (`adapters.native_mem0x`): `mem0` (native, with the event log), `mem0_det`
(source-keyed invalidation; its own store, restored from the same prefix snapshot) and `mem0_rep`
(the program view on definition items, served from the `mem0` arm's store, so its effect is the
view alone). The view's programs are the cached D1s ones: before any model call the run stops if a
program request it will send is not in the response cache (`uncached_programs`, `CachedPrograms`).

Prefix. Per world the E5 run of that world (`runs/e5_mem0_s<seed>_*`, or `--e5-run`) is checked
(`prefix_checks`): its snapshot exists, the store settings recorded in its marker equal this
adapter's, the installed mem0ai is the version E5 pinned, Mem0's configuration is E5's, on_error
and the world (events hash, world parameters) are E5's, and the LLM and embedding servers have
E5's model, revision, engine image and engine arguments and at least E5's context length
(`runinfo.serving_info`, against E5's manifest and env; a longer context changes no response to a
request that fits, and every E5 request fitted). If every check holds the snapshot is restored
(`prefix.source = "e5"`; prefix units before the fork have no event log, which detection does not
need, but their ledger is kept), else the prefix is ingested again with the log (`"rebuilt"`);
`--prefix-policy resume` aborts instead of rebuilding, `rebuild` never resumes. The t0 probes are
answered from the prefix as in E5; for a resumed prefix they are the anchor: `anchor` counts how
many contexts and answers equal E5's committed ones.

Outputs as E5 (`probes.jsonl`, `answers.jsonl`, `world_results.jsonl`, stores under
`stores/<world>_mem0/{prefix,snapshot,<branch>,<branch>_det}` with `ledger.jsonl` and
`events.jsonl`), plus each branch's t0 probes (gold only, not asked: the reference for "changed"),
`fork_deps.jsonl` (each item's gold dependency set at the fork, for detection) and `views.jsonl`
(the program view of each `mem0_rep` definition probe). `analysis.x_mem0` reads them.

`--dry-run` calls no model: it counts the units, searches, reader calls and program requests of a
full run from the generator, and prices them with E5's measured per-event costs of the same world
and branch (`e5_costs`), for a resumed and for a rebuilt prefix.
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import ZeroDm
from memshift.adapters.native import LEDGER_FILE, STORE_MARKER, _read_marker
from memshift.adapters.native_mem0x import (
    MEM0_VERSION,
    LoggedMem0,
    Mem0Det,
    Mem0Rep,
    RetryPolicy,
    mem0_version,
    read_events,
)
from memshift.experiments.common import (
    EMBED_MODEL,
    MAIN_MODEL,
    answer_probes,
    client,
    endpoints,
    new_run_dir,
    read_jsonl,
    world_params,
    write_jsonl,
)
from memshift.experiments.e1 import parse_seeds
from memshift.experiments.e5 import NATIVE_LLM, OFFSETS
from memshift.gen.gold import gold_at
from memshift.gen.model import World, fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.program import program_request
from memshift.llm import LLMClient, LLMResult
from memshift.probes import make_probes
from memshift.runinfo import REPO, serving_info, world_record, write_env, write_manifest
from memshift.textutil import sha256_json

ARMS = ("mem0", "mem0_det", "mem0_rep")
BRANCHES = ("src", "def", "orphan")
POLICIES = ("auto", "resume", "rebuild")
MONTHS = 3
ON_ERROR = "skip"  # E5's Mem0 runs (manifest params.on_error); never infrastructure errors
RETRY = (
    RetryPolicy()
)  # model-server requests (Mem0's and the reader's): --request-timeout, --attempts
# Mem0System.config() without paths, URLs and keys, as at E5's code commit (ef70d0f; unchanged)
E5_MEM0_CONFIG: dict[str, Any] = {
    "llm": {"provider": "openai", "config": {"model": MAIN_MODEL, "temperature": 0.0}},
    "embedder": {"provider": "openai", "config": {"model": EMBED_MODEL}},
    "vector_store": {
        "provider": "qdrant",
        "config": {"collection_name": "memshift", "embedding_model_dims": 1024, "on_disk": True},
    },
}
_STORE_ONLY = ("openai_base_url", "api_key", "path")
SERVING_KEYS = ("model", "revision", "image_id", "extra_args")  # equal to E5's
SERVER_ROLES = {"llm": ("native_llm", MAIN_MODEL), "embedder": ("embedding", EMBED_MODEL)}


def config_fingerprint(cfg: dict[str, Any]) -> dict[str, Any]:
    """The parts of a Mem0 `MemoryConfig` dict that do not depend on the store or the host."""
    out: dict[str, Any] = {}
    for part in ("llm", "embedder", "vector_store"):
        conf = {k: v for k, v in cfg[part]["config"].items() if k not in _STORE_ONLY}
        out[part] = {"provider": cfg[part]["provider"], "config": conf}
    return out


def e5_run_dir(seed: int, root: Path | None = None) -> Path | None:
    """The E5 Mem0 run of one world (not the noise-floor repeat)."""
    found = sorted(p for p in (root or REPO / "runs").glob(f"e5_mem0_s{seed}_*") if p.is_dir())
    if len(found) > 1:
        raise ValueError(f"more than one E5 Mem0 run for seed {seed}: {[p.name for p in found]}")
    return found[0] if found else None


def _check(ok: bool, e5: Any, now: Any) -> dict[str, Any]:
    return {"ok": bool(ok), "e5": e5, "now": now}


def prefix_checks(
    e5_dir: Path,
    system: LoggedMem0,
    natural: World,
    params: dict[str, Any] | None,
    serving: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Whether E5's prefix snapshot of this world was built with exactly this configuration."""
    store = e5_dir / "stores" / f"{natural.world_id}_mem0" / "snapshot"
    manifest = json.loads((e5_dir / "manifest.json").read_text())
    env_path = e5_dir / "env.json"
    env = json.loads(env_path.read_text()) if env_path.is_file() else {}
    has_store = (store / STORE_MARKER).is_file() and (store / LEDGER_FILE).is_file()
    marker = _read_marker(store) if has_store else None
    worlds = [w for w in manifest.get("worlds", []) if w.get("world_id") == natural.world_id]
    sha_e5 = worlds[0].get("events_sha256") if worlds else None
    sha_now = world_record(natural)["events_sha256"]
    e5_params = manifest.get("params", {})
    checks = {
        "store": _check(has_store, str(store), None),
        "settings": _check(marker == system.describe(), marker, system.describe()),
        "mem0_version": _check(mem0_version() == MEM0_VERSION, MEM0_VERSION, mem0_version()),
        "mem0_config": _check(
            config_fingerprint(system.config()) == E5_MEM0_CONFIG,
            E5_MEM0_CONFIG,
            config_fingerprint(system.config()),
        ),
        "on_error": _check(e5_params.get("on_error") == ON_ERROR, e5_params.get("on_error"), None),
        "world": _check(
            sha_e5 == sha_now and (e5_params.get("world_params") or None) == params,
            {"events_sha256": sha_e5, "world_params": e5_params.get("world_params")},
            {"events_sha256": sha_now, "world_params": params},
        ),
    }
    checks["on_error"]["now"] = ON_ERROR
    serving = serving_info() if serving is None else serving
    for role, (e5_role, model) in SERVER_ROLES.items():
        m = manifest.get("models", {}).get(e5_role, {})
        e = env.get("models", {}).get(e5_role, {})
        e5 = {**{k: m.get(k) for k in SERVING_KEYS[:3]}, "extra_args": e.get("extra_args")}
        rec = serving.get(model) or {}
        # a serving record names its model `served_model`; E5's manifest calls it `model`
        rec = {"model": rec.get("served_model", rec.get("model")), **rec} if rec else {}
        now = {k: rec.get(k) for k in SERVING_KEYS} if rec else None
        checks[role] = _check(now == e5, e5, now)
        ctx_e5, ctx = e.get("max_model_len"), rec.get("max_model_len")
        ok = isinstance(ctx, int) and isinstance(ctx_e5, int) and ctx >= ctx_e5
        checks[f"{role}_context"] = _check(ok, ctx_e5, ctx)
    return checks


def prefix_decision(
    policy: str, e5_dir: Path | None, checks: dict[str, dict[str, Any]] | None
) -> str:
    """`"e5"` (restore E5's snapshot) or `"rebuilt"`; `resume` raises when it is not possible."""
    if policy not in POLICIES:
        raise ValueError(f"prefix policy must be one of {POLICIES}")
    match = e5_dir is not None and checks is not None and all(c["ok"] for c in checks.values())
    if policy == "resume" and not match:
        bad = sorted(k for k, c in (checks or {}).items() if not c["ok"]) or ["no E5 run"]
        raise SystemExit(f"--prefix-policy resume: E5 prefix does not match ({', '.join(bad)})")
    return "e5" if match and policy != "rebuild" else "rebuilt"


def anchor(answers: list[Any], e5_dir: Path | None) -> dict[str, Any] | None:
    """The t0 answers against E5's committed ones (same probe ids): identical contexts and
    identical answers (status and value)."""
    path = e5_dir / "answers.jsonl" if e5_dir else None
    if path is None or not path.is_file():
        return None
    ref = {a["probe_id"]: a for a in read_jsonl(path) if a["system"] == "mem0"}
    pairs = [(a, ref[a.probe_id]) for a in answers if a.probe_id in ref]
    return {
        "probes": len(answers),
        "compared": len(pairs),
        "context_identical": sum(a.context_sha256 == r["context_sha256"] for a, r in pairs),
        "answer_identical": sum((a.status, a.value) == (r["status"], r["value"]) for a, r in pairs),
    }


def _make(cls: type[LoggedMem0], world: World, store: Path, urls: tuple[str, str]) -> LoggedMem0:
    return cls(
        world,
        store,
        urls[0],
        MAIN_MODEL,
        urls[1],
        EMBED_MODEL,
        world.world_id,
        on_error=ON_ERROR,
        retry=RETRY,
    )


def event_counts(system: LoggedMem0) -> dict[str, int]:
    """Logged events of a store by kind, origin and scope (e.g. `add/mem0/extracted/ADD`)."""
    out: dict[str, int] = {}
    for line in read_events(system.events_path):
        for ev in line.get("events") or []:
            k = f"{line.get('kind')}/{ev.get('origin')}/{ev.get('scope')}/{ev.get('event')}"
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def units_without_events(system: LoggedMem0, before: set[str]) -> list[str]:
    """Units the store ingested after `before` with no `add` line in its log (a crash between the
    ledger line and the log line); detection cannot see such units."""
    logged = {str(x.get("key")) for x in read_events(system.events_path) if x.get("kind") == "add"}
    return sorted(set(system.ledger.entries) - before - logged)


def program_key(llm: LLMClient, world: World, slot: Any, t: Any) -> str:
    """The response-cache key of the item's D1s program request at time t."""
    req = program_request(world, slot, t)
    return sha256_json(llm.request_body(req["messages"], req["max_tokens"], req["schema"]))


def uncached_programs(llm: LLMClient, world: World, times: list[Any]) -> list[str]:
    """Definition items (with their version) whose D1s program is not in the response cache."""
    return sorted(
        {
            f"{s.slot_id}@{world.kappa_version(s.kappa_id, t)}"
            for s in world.slots
            if s.kappa_id
            for t in times
            if llm.cache.get(program_key(llm, world, s, t)) is None
        }
    )


class CachedPrograms:
    """The program view's writer: the cached D1s programs only. A request that is not in the
    response cache stops the run instead of writing a new program."""

    def __init__(self, llm: LLMClient):
        self.llm, self.model = llm, llm.model

    def complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        if self.llm.cache.get(sha256_json(self.llm.request_body(messages, max_tokens, schema))):
            return self.llm.complete(messages, max_tokens, schema, purpose)
        raise SystemExit("mem0_rep: a D1s program request is not in the response cache")


def fork_deps(world: World, log: SourceLog) -> list[dict[str, Any]]:
    """Each item's gold dependency set at the fork (records as [object_id, revision])."""
    return [
        {
            "world_id": world.world_id,
            "branch": world.branch,
            "fact_id": s.slot_id,
            "fork": fmt_ts(world.fork),
            "deps": [list(d) for d in gold_at(world, log, s, world.fork).deps],
        }
        for s in world.slots
    ]


def run_world(
    seed: int,
    run_dir: Path,
    params: dict[str, Any] | None,
    branches: tuple[str, ...] = BRANCHES,
    arms: tuple[str, ...] = ARMS,
    policy: str = "auto",
    e5_dir: Path | None = None,
) -> dict[str, Any]:
    eps = endpoints()
    urls = (eps.get(NATIVE_LLM, eps[MAIN_MODEL]).split("|")[0], eps[EMBED_MODEL])
    reader = client(MAIN_MODEL, run_dir, timeout=RETRY.timeout_s)
    natural = generate_branch(seed, "natural", params)
    nlog = SourceLog(natural.events)
    t0 = natural.fork - timedelta(minutes=1)
    if "mem0_rep" in arms:  # before any model call: every program the view needs is cached
        missing = sorted(
            {
                m
                for b in branches
                for w in [generate_branch(seed, b, params)]
                for m in uncached_programs(
                    reader, w, [w.fork + timedelta(weeks=k) for k in OFFSETS[b]]
                )
            }
        )
        if missing:
            raise SystemExit(f"mem0_rep: D1s programs not in the response cache: {missing}")
    root = run_dir / "stores" / f"{natural.world_id}_mem0"
    out: dict[str, Any] = {"seed": seed, "world": world_record(natural), "arms": list(arms)}
    # ------------------------------------------------------------------ prefix
    prefix = _make(LoggedMem0, natural, root / "prefix", urls)
    checks = prefix_checks(e5_dir, prefix, natural, params) if e5_dir else None
    source = prefix_decision(policy, e5_dir, checks)
    if source == "e5":
        assert e5_dir is not None
        prefix.restore(e5_dir / "stores" / root.name / "snapshot")
    stats = prefix.ingest(nlog.upto(t0), "ingest")  # resumed: only units E5 never recorded
    out["prefix"] = {
        "source": source,
        "e5_run": e5_dir.name if e5_dir else None,
        "policy": policy,
        "checks": checks,
        "ingest": stats,
        "units": len(prefix.ledger),
        "failed": len(prefix.failed_keys()),
    }
    snapshot = root / "snapshot"
    prefix.snapshot(snapshot)
    snap_keys = set(prefix.ledger.entries)
    pre = make_probes(natural, nlog, [t0], tags={"phase": "pre", "k": 0})
    pre_answers = answer_probes("mem0", prefix.serve, pre, reader)[0]
    prefix.close()
    if source == "e5":
        out["anchor"] = anchor(pre_answers, e5_dir)
    answers: list[Any] = list(pre_answers)
    probe_rows = [p.to_json() for p in pre]
    deps_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    # ------------------------------------------------------------------ branches
    for branch in branches:
        world = generate_branch(seed, branch, params)
        log = SourceLog(world.events)
        probe_rows += [  # gold at t0 on this branch: the reference for "changed" (not asked)
            p.to_json() for p in make_probes(world, log, [t0], tags={"phase": "pre", "k": 0})
        ]
        deps_rows += fork_deps(world, log)
        stores: dict[str, LoggedMem0] = {}
        if {"mem0", "mem0_rep"} & set(arms):
            stores["mem0"] = _make(LoggedMem0, world, root / branch, urls)
        if "mem0_det" in arms:
            stores["mem0_det"] = _make(Mem0Det, world, root / f"{branch}_det", urls)
        for s in stores.values():
            s.restore(snapshot)
        view = ZeroDm(world, log, CachedPrograms(reader))  # type: ignore[arg-type]
        rep = Mem0Rep(stores["mem0"], view) if "mem0_rep" in arms else None
        serve = {"mem0": stores["mem0"].serve if "mem0" in stores else None}
        serve["mem0_det"] = stores["mem0_det"].serve if "mem0_det" in stores else None
        serve["mem0_rep"] = rep.serve if rep else None
        prev = t0
        for k in OFFSETS[branch]:
            t = world.fork + timedelta(weeks=k)
            events = log.between(prev, t)
            for s in stores.values():
                s.ingest(events, "maintain")
            probes = make_probes(world, log, [t], tags={"phase": "post", "k": k})
            for arm in (a for a in ARMS if a in arms):  # mem0 before mem0_rep: shared requests
                answers += answer_probes(arm, serve[arm], probes, reader)[0]  # type: ignore[arg-type]
            probe_rows += [p.to_json() for p in probes]
            prev = t
        res: dict[str, Any] = {}
        for name, s in stores.items():
            res[name] = {
                "cost": s.cost,
                "stats": s.stats,
                "events": event_counts(s),
                "units_without_events": units_without_events(s, snap_keys),
            }
            s.close()
        if rep is not None:
            view_rows += [{"world_id": world.world_id, "branch": branch, **v} for v in rep.views]
            res["mem0_rep"] = {"views": len(rep.views), "view_ok": sum(_ok(v) for v in rep.views)}
        out.setdefault("branches", {})[branch] = res
    write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
    write_jsonl(run_dir / "probes.jsonl", probe_rows, mode="a")
    write_jsonl(run_dir / "fork_deps.jsonl", deps_rows, mode="a")
    write_jsonl(run_dir / "views.jsonl", view_rows, mode="a")
    return out


def _ok(v: dict[str, Any]) -> bool:
    return bool((v.get("view") or {}).get("ok"))


def run(
    seeds: list[int],
    name: str = "x_mem0",
    branches: tuple[str, ...] = BRANCHES,
    arms: tuple[str, ...] = ARMS,
    policy: str = "auto",
    e5_run: str | None = None,
) -> Path:
    if e5_run and len(seeds) != 1:
        raise ValueError("--e5-run takes one world")
    params = world_params(MONTHS)
    run_dir = new_run_dir(name)
    results = []
    for s in seeds:
        e5_dir = REPO / "runs" / e5_run if e5_run else e5_run_dir(s)
        results.append(run_world(s, run_dir, params, branches, arms, policy, e5_dir))
        write_jsonl(run_dir / "world_results.jsonl", results[-1:], mode="a")
    models = {"native_llm": MAIN_MODEL, "reader": MAIN_MODEL, "embedding": EMBED_MODEL}
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "X-mem0",
        [r["world"] for r in results],
        models,
        {
            "seeds": seeds,
            "branches": list(branches),
            "arms": list(arms),
            "world_params": params,
            "offsets": OFFSETS,
            "on_error": ON_ERROR,
            "retry": vars(RETRY),
            "prefix_policy": policy,
            "prefix_sources": {r["world"]["world_id"]: r["prefix"]["source"] for r in results},
            "mem0_version": mem0_version(),
        },
    )
    return run_dir


# ---------------------------------------------------------------------- dry run
USAGE = ("llm_calls", "prompt_tokens", "completion_tokens", "embed_calls", "embed_tokens")


def e5_costs(e5_dir: Path) -> dict[str, Any]:
    """E5's measured costs of one world: per prefix event (`t_i`), per branch event (the branch
    store's usage minus the snapshot's, less the searches of the branch's probes), per search
    (serve tokens per probe) and per reader answer."""
    r = read_jsonl(e5_dir / "world_results.jsonl")[0]
    store = e5_dir / "stores" / f"{r['world']['world_id']}_mem0"
    snap = json.loads((store / "snapshot" / "state.json").read_text())
    probes = [p for p in read_jsonl(e5_dir / "probes.jsonl") if p["tags"].get("k", 0)]
    answers = [
        a
        for a in read_jsonl(e5_dir / "answers.jsonl")
        if a["system"] == "mem0" and a["status"] != "blocked"
    ]
    out: dict[str, Any] = {
        "run": e5_dir.name,
        "prefix_event": {k: r["t_i"][k] for k in (*USAGE, "wall_s")},
        "reader_answer": {
            "prompt_tokens": sum(a["prompt_tokens"] for a in answers) / max(1, len(answers)),
            "completion_tokens": sum(a["completion_tokens"] for a in answers)
            / max(1, len(answers)),
        },
        "branch_event": {},
    }
    serve_tokens, searches = 0.0, 0
    for b, br in r["branches"].items():
        st = json.loads((store / b / "state.json").read_text())
        n = br["stats"]["ingested"]
        n_probes = sum(p["branch"] == b for p in probes)
        d = {k: st["usage"][k] - snap["usage"][k] for k in USAGE}
        served = st["cost"]["serve_tokens"] - snap["cost"]["serve_tokens"]
        d["embed_calls"] -= n_probes
        d["embed_tokens"] -= served
        out["branch_event"][b] = {k: v / max(1, n) for k, v in d.items()}
        out["branch_event"][b]["wall_s"] = br["stats"]["ingest_wall_s"] / max(1, n)
        serve_tokens += served
        searches += n_probes
    out["search"] = {"embed_calls": 1.0, "embed_tokens": serve_tokens / max(1, searches)}
    return out


def program_keys(world: World, log: SourceLog, times: list[Any]) -> set[str]:
    """Response-cache keys of the D1s program requests the view sends for the definition items
    at these times (a program request depends only on the item and its definition version)."""
    from memshift.experiments.e1_estimate import Recorder

    rec = Recorder()
    view = ZeroDm(world, log, rec)  # type: ignore[arg-type]
    for slot in (s for s in world.slots if s.kappa_id):
        for t in times:
            view.view(next(iter(make_probes(world, log, [t], [slot]))))
    return rec.keys("program")


def _scale(per: dict[str, float], n: float) -> dict[str, float]:
    return {k: v * n for k, v in per.items()}


def _add(acc: dict[str, float], part: dict[str, float]) -> None:
    for k, v in part.items():
        acc[k] = acc.get(k, 0.0) + v


def dry_run(
    seeds: list[int],
    branches: tuple[str, ...] = BRANCHES,
    arms: tuple[str, ...] = ARMS,
    e5_root: Path | None = None,
) -> dict[str, Any]:
    """Counts and E5-priced costs of a full run, without any model call (see the module doc)."""
    params = world_params(MONTHS)
    all_prog: set[str] = set()
    all_long: set[str] = set()  # what the 12-month D1s cells of the same seeds send (E1, block C)
    per_world: dict[str, Any] = {}
    total: dict[str, dict[str, float]] = {"resumed": {}, "rebuilt": {}}
    for seed in seeds:
        e5_dir = e5_run_dir(seed, e5_root)
        if e5_dir is None:
            raise SystemExit(f"no E5 Mem0 run for seed {seed} under {e5_root or REPO / 'runs'}")
        cost = e5_costs(e5_dir)
        natural = generate_branch(seed, "natural", params)
        nlog = SourceLog(natural.events)
        t0 = natural.fork - timedelta(minutes=1)
        n_prefix = len(nlog.upto(t0))
        ingesting = int(bool({"mem0", "mem0_rep"} & set(arms))) + int("mem0_det" in arms)
        searching = ingesting
        w: dict[str, Any] = {"e5_run": e5_dir.name, "prefix_events": n_prefix, "branches": {}}
        ingest: dict[str, float] = {}
        reader_calls = len(make_probes(natural, nlog, [t0]))  # t0 answers (anchor)
        searches = reader_calls
        prog: set[str] = set()
        for b in branches:
            world = generate_branch(seed, b, params)
            log = SourceLog(world.events)
            times = [world.fork + timedelta(weeks=k) for k in OFFSETS[b]]
            n_ev = len(log.between(t0, times[-1]))
            probes = make_probes(world, log, times)
            asked = [p for p in probes if p.gold_status != "blocked"]
            defs = [p for p in asked if p.fact_id.startswith("k.")]
            calls = len(asked) * ingesting + (len(defs) if "mem0_rep" in arms else 0)
            reader_calls += calls
            searches += len(probes) * searching
            _add(ingest, _scale(cost["branch_event"][b], n_ev * ingesting))
            if "mem0_rep" in arms:
                prog |= program_keys(world, log, times)
            w["branches"][b] = {
                "events": n_ev,
                "units": n_ev * ingesting,
                "probes": len(probes),
                "reader_calls_upper": calls,
                "definition_probes": len(defs),
            }
        long_keys: set[str] = set()
        if "mem0_rep" in arms:
            for b in ("natural", "def"):
                lw = generate_branch(seed, b, None)
                ll = SourceLog(lw.events)
                lt = [lw.fork - timedelta(minutes=1)]
                lt += [lw.fork + timedelta(weeks=k) for k in OFFSETS["def"]]
                long_keys |= program_keys(lw, ll, lt)
        all_prog |= prog
        all_long |= long_keys
        wall_s = ingest.pop("wall_s", 0.0)
        reader = _scale(cost["reader_answer"], reader_calls)
        serve = _scale(cost["search"], searches)
        prefix = _scale(cost["prefix_event"], n_prefix)
        prefix_wall = prefix.pop("wall_s")
        w.update(
            {
                "ingest_branches": ingest,
                "ingest_wall_hours_sequential": wall_s / 3600,
                "prefix_if_rebuilt": prefix,
                "prefix_wall_hours_if_rebuilt": prefix_wall / 3600,
                "searches": serve,
                "reader_calls_upper": reader_calls,
                "reader_tokens_upper": reader,
                "program_requests": len(prog),
                "program_requests_not_in_12_month_d1s": len(prog - long_keys) if prog else 0,
                "e5_costs": cost,
            }
        )
        per_world[natural.world_id] = w
        for scenario in total:
            acc = total[scenario]
            _add(acc, {f"ingest_{k}": v for k, v in ingest.items()})
            if scenario == "rebuilt":
                _add(acc, {f"ingest_{k}": v for k, v in prefix.items()})
            _add(
                acc, {"reader_calls": reader_calls, **{f"reader_{k}": v for k, v in reader.items()}}
            )
            _add(acc, {f"serve_{k}": v for k, v in serve.items()})
            _add(acc, {"wall_hours_sequential": (wall_s + prefix_wall * (scenario == "rebuilt"))})
    for acc in total.values():
        acc["wall_hours_sequential"] = acc.get("wall_hours_sequential", 0.0) / 3600
    return {
        "seeds": seeds,
        "branches": list(branches),
        "arms": list(arms),
        "world_params": params,
        "mem0_version_installed": mem0_version(),
        "note": "prices: E5's measured per-event costs of the same world and branch (mem0_det "
        "priced as native Mem0); reader calls are an upper bound (mem0_rep's other items and a "
        "matching anchor are cache hits); a resumed prefix costs no LLM call",
        "total": total,
        "program_requests": {
            "distinct": len(all_prog),
            "not_in_12_month_d1s": len(all_prog - all_long),
        },
        "per_world": per_world,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--branches", default=",".join(BRANCHES))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--prefix-policy", choices=POLICIES, default="auto")
    ap.add_argument("--e5-run", default=None, help="E5 run name (default: runs/e5_mem0_s<seed>_*)")
    ap.add_argument("--name", default="x_mem0")
    ap.add_argument("--dry-run", action="store_true", help="count and price, no model call")
    ap.add_argument("--e5-root", default=None, help="--dry-run: directory of the E5 runs")
    ap.add_argument("--request-timeout", type=float, default=RETRY.timeout_s, help="seconds")
    ap.add_argument("--attempts", type=int, default=RETRY.attempts, help="per request")
    a = ap.parse_args()
    RETRY = RetryPolicy(timeout_s=a.request_timeout, attempts=max(1, a.attempts))
    br = tuple(a.branches.split(","))
    arm_list = tuple(a.arms.split(","))
    if not set(br) <= set(BRANCHES) or not set(arm_list) <= set(ARMS):
        raise SystemExit(f"branches must be in {BRANCHES}, arms in {ARMS}")
    if a.dry_run:
        root = Path(a.e5_root) if a.e5_root else None
        print(json.dumps(dry_run(parse_seeds(a.seeds), br, arm_list, root), indent=1))
    else:
        print(run(parse_seeds(a.seeds), a.name, br, arm_list, a.prefix_policy, a.e5_run))

"""E1 (plan §6): detection / repair dissociation, white box.

Per world and per branch (S_src, S_def): six HoldMem cells, D in {0,1} x {B=0, B=1 source-epoch,
B=1 object-cluster}, all instantiated at t0 = t_f - 1 min from the same writer outputs; a
hash assertion checks that every probe's rendered context is identical across cells before the
shift. After the shift, cells are maintained and probed at t_f + {1, 2, 4, 8} weeks. The JSON
mismatch arm (D=1, B=1 source-epoch rendered as parse JSON), the zero system, plain RAG and the
oracle answer the same probes. Outputs: probes.jsonl, answers.jsonl, items.jsonl, hash_check_*,
maintenance.jsonl, world_summaries.jsonl, manifest.json, env.json.

Prereg v2 §4 adds the D1s cells (`--cells D1sB0,D1sB1_source,D1sB1_cited`): D=1 whose replacement
values come from a program the writer model writes, not from benchmark-authored rules. They ride
on the same writer requests (cache hits), add one program request per item write, and are covered
by the same pre-intervention hash assertion.

Control cells and arms (prereg/decisions.md, registered before their runs; each writes from the
same writer requests as an existing cell wherever its design allows, so the response cache serves
it): `D0B1_gold` / `D1B1_gold` (oracle dependency key), `D0B1_scope` / `D1B1_scope` (scope-aware
key), `D0B0_refresh` / `D1B0_refresh` (refresh without invalidation), `D0tB0` / `D0tB1_source`
(D=0 values in the D=1 template), `D0thinkB0` / `D0thinkB1_source` (thinking writer for definition
items after the shift; needs new writer requests), `D1sB1_source_ra` / `D1sB0_ra` (D1s with
retraction-aware programs; new program requests), `D0fullB1_source` (D=0 whose definition items are
re-derived after the shift from every live record of their declared sources, not a BM25 selection;
new writer requests for those items only, every other request and the pre-shift snapshot are
`D0B1_source`'s) and, through the same flag, the query-time view `zero_Dm` (the zero system plus the
D1s program's value, run on the live records per probe).
Prereg v4 block U adds the slot-keyed upsert cells `D0B0_upsert` / `D1B0_upsert` / `D1sB0_upsert`
(the requests of `D0B0` / `D1B0` / `D1sB0`; a new delta item retires the slot's older items) and
runs `D1sB0_refresh` (D1s programs, refresh without invalidation).
Every HoldMem cell is covered by the pre-intervention hash assertion.

`--render MODEL` (prereg v2 block L): every branch's world is passed through
gen/llm_render.py before the source log is built, so all systems read LLM-written free text; the
per-branch render report goes into world_summaries.jsonl and the renderer into the manifest.

`--post-probes definition` (prereg v4 blocks G-a, G-b, U5 and v4b block V, judged on definition
items; decision log 2026-10-16): after the fork only the definition items among the discriminative
draw are answered, the same items an `all` run answers; the t0 probes are always all answered.
`--probe-unnamed-defs` (prereg v4 block Q): the definition items' probes leave out "Definition in
effect: <statement>" (`probes.make_probes`). Every cell gets the same probes, so the hash assertion
still holds; only reader requests change.

`--think-max-tokens N` (prereg v4 block H-b): the output limit of the D0think writer's requests.
`--defwriter MODEL` (block H-a): in every cell of the run, the definition items written after the
shift come from MODEL with the same request text (a Claude model with adaptive thinking, see
`defwriter_settings`); the pre-shift snapshot and every other item are the main writer's.
`--program-model MODEL`, `--program-temperature T`, `--program-sample I` (block G-b): the D1s
program requests of the definition items go to MODEL at temperature T, with `memshift_sample: I` in
the body (`program_extra_body`); at the defaults (the writer, 0, none) nothing changes. A Claude
program model takes no temperature (`--program-temperature` is refused with it): its samples
differ only by the sample index.
"""

from __future__ import annotations

import copy
import json
import random
import re
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

from memshift.adapters.baselines import EmbedClient, Oracle, PlainRAG, ZeroDm, ZeroSystem
from memshift.experiments.common import (
    EMBED_MODEL,
    MAIN_MODEL,
    Answer,
    Timer,
    answer_probes,
    client,
    endpoints,
    new_run_dir,
    slug,
    world_params,
    write_jsonl,
)
from memshift.gen.llm_render import render_world_llm
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem
from memshift.holdmem.writer import FULL_BUDGET, THINK_MAX_TOKENS, WRITER_DOC_CAP
from memshift.llm import LLMClient, default_extra_body, think_extra_body
from memshift.probes import Probe, make_probes
from memshift.reader import context_hash
from memshift.runinfo import REPO, world_record, write_env, write_manifest

CELLS = [
    CellConfig(d, b, g) for d in (0, 1) for b, g in ((0, "source"), (1, "source"), (1, "cited"))
]
JSON_ARM_OF = "holdmem_D1B1_source"
VIEWS = ("zero_Dm",)  # query-time systems, asked through the same `--cells` flag
OFFSETS_WEEKS = (1, 2, 4, 8)
BRANCHES = ("src", "def")
# prereg v4 blocks H-a and G-b as amended on 2026-10-16 (prereg/decisions.md): the API rejects a
# thinking budget and any temperature on the Claude 5.5 models. A Claude definition-item writer
# thinks adaptively at effort "high" inside a 17,000-token output limit; a Claude program writer
# answers without up-front thinking (Sonnet 5.5) or at the least thinking the model allows (Opus
# 5.5). All at the API's default sampling.
DEFWRITER_BODY: dict[str, Any] = {
    "thinking": {"type": "adaptive"},
    "output_config": {"effort": "high"},
}
DEFWRITER_MAX_TOKENS = 17000
CLAUDE_PROGRAM_BODY: dict[str, dict[str, Any]] = {
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}},
    "claude-opus-5-5": {"output_config": {"effort": "low"}},
}


# prereg v4b block Y: a Claude reader answers without thinking first, as the served readers read
# with thinking off (the 5.5 models reject `thinking.type` "disabled")
CLAUDE_READER_BODY: dict[str, dict[str, Any]] = {
    "claude-sonnet-5-5": {"thinking": {"type": "between_tools"}},
}


def reader_extra_body(model: str) -> dict[str, Any] | None:
    """Extras of a swapped-in reader's requests: a Claude reader's registered settings (ValueError
    for a Claude model without any); for a served model None, so the client's default applies."""
    if not is_claude(model):
        return None
    if model not in CLAUDE_READER_BODY:
        raise ValueError(f"no registered reader settings for {model}")
    return copy.deepcopy(CLAUDE_READER_BODY[model])


def is_claude(model: str) -> bool:
    return model.startswith("claude-")


def defwriter_settings(model: str) -> tuple[dict[str, Any], int | None]:
    """(extra body, output limit) of the definition-item writer `--defwriter MODEL`: adaptive
    thinking at effort "high" for Claude models; another served model gets its default body and
    the writer request's own limit."""
    if is_claude(model):
        return copy.deepcopy(DEFWRITER_BODY), DEFWRITER_MAX_TOKENS
    return default_extra_body(model), None


def check_program_temperature(model: str, temperature: float) -> None:
    """A Claude program model takes no temperature (the API rejects any on the 5.5 models)."""
    if temperature and is_claude(model):
        raise ValueError(
            f"--program-temperature {temperature} with {model}: Claude models take no temperature "
            "(prereg/decisions.md, 2026-10-16); their samples differ by --program-sample only"
        )


def program_extra_body(
    model: str, temperature: float = 0.0, sample: int | None = None
) -> dict[str, Any]:
    """Body extras of the definition items' program requests (prereg v4 block G-b): the model's
    default (reasoning off) and the sampling temperature when it is not 0 (it replaces the body's
    0.0); for a Claude model, its registered settings (`CLAUDE_PROGRAM_BODY`) and no temperature
    (ValueError if one is given). With a sample index, `memshift_sample`, which the LLM client
    keeps in the cache key and removes from the HTTP request, so each sample has its own cache
    entry."""
    if is_claude(model):
        check_program_temperature(model, temperature)
        if model not in CLAUDE_PROGRAM_BODY:
            raise ValueError(f"no registered program-writer settings for {model}")
        body = copy.deepcopy(CLAUDE_PROGRAM_BODY[model])
    else:
        body = default_extra_body(model)
        if temperature:
            body = {**body, "temperature": temperature}
    if sample is not None:
        body = {**body, "memshift_sample": sample}
    return body


def separate_program_client(
    model: str, writer_model: str, temperature: float, sample: int | None
) -> bool:
    """Do the program flags differ from the defaults (the writer, temperature 0, no sample)?"""
    return model != writer_model or bool(temperature) or sample is not None


_WRITE_LOCK = threading.Lock()


def _par[T, R](fn: Callable[[T], R], items: list[T], workers: int = 8) -> list[R]:
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as ex:
        return list(ex.map(fn, items))


def discriminative_probes(probes: list[Probe], pre: list[Probe], key: str) -> list[Probe]:
    """All items whose gold changed since t0 plus an equal-size seeded sample of unchanged items
    (plan §4.3). Selection depends on gold only, never on any system."""
    before = {p.fact_id: (p.gold_value, p.gold_status) for p in pre}
    changed = [p for p in probes if (p.gold_value, p.gold_status) != before.get(p.fact_id)]
    unchanged = [p for p in probes if (p.gold_value, p.gold_status) == before.get(p.fact_id)]
    rng = random.Random(key)
    keep = rng.sample(unchanged, min(len(changed), len(unchanged)))
    return sorted(changed + keep, key=lambda p: p.fact_id)


def _answer(server: tuple[str, Any], probes: list[Probe], reader: Any) -> list[Answer]:
    return answer_probes(server[0], server[1], probes, reader)[0]


def noise_floor(
    world: Any,
    log: SourceLog,
    t0: Any,
    pre: list[Probe],
    ref: HoldMem,
    writer: Any,
    run_dir: Path,
    writer_model: str = MAIN_MODEL,
) -> dict[str, Any]:
    """Second ingestion of the same world (plan §4.2): identical requests with an ingest seed,
    which only changes the cache key and batch composition (temperature 0). Reports the share of
    items whose written value differs and the ContractValid difference on the pre-shift probes.
    """
    if not isinstance(writer, LLMClient):
        return {}
    second = client(
        writer_model, run_dir, extra_body={**default_extra_body(writer_model), "seed": 1}
    )
    hm = HoldMem(world, log, CellConfig(0, 1, "source"), second, writer_model)
    hm.build(t0)
    a = answer_probes("noise_a", ref.serve, pre, writer)[0]
    b = answer_probes("noise_b", hm.serve, pre, writer)[0]
    diff_values = sum(
        ref.items[f"{s.slot_id}#0"].eval_value != hm.items[f"{s.slot_id}#0"].eval_value
        for s in world.slots
    )
    return {
        "items": len(world.slots),
        "values_differ": diff_values,
        "cv_first": sum(x.contract_valid for x in a) / len(a),
        "cv_second": sum(x.contract_valid for x in b) / len(b),
        "answers_differ": sum(
            x.contract_valid != y.contract_valid for x, y in zip(a, b, strict=True)
        ),
    }


def embed_client(model: str = EMBED_MODEL) -> EmbedClient | None:
    eps = endpoints()
    if model not in eps:
        return None
    cache = REPO / "cache" / "emb" / f"{slug(model)}.sqlite"
    return EmbedClient(eps[model], model, cache)


def run_world(
    seed: int,
    run_dir: Path,
    branches: tuple[str, ...] = BRANCHES,
    baselines: bool = True,
    writer: Any = None,
    cells_cfg: list[CellConfig] | None = None,
    world_params: dict[str, Any] | None = None,
    discriminative: bool = False,
    reader_model: str | None = None,
    embed_model: str | None = None,
    render_model: str | None = None,
    renderer: Any = None,
    writer_model: str | None = None,
    views: tuple[str, ...] = (),
    think_writer: Any = None,
    probe_unnamed_defs: bool = False,
    think_max_tokens: int = THINK_MAX_TOKENS,
    defwriter: str | None = None,
    def_writer: Any = None,
    program_model: str | None = None,
    program_temperature: float = 0.0,
    program_sample: int | None = None,
    program_writer: Any = None,
    program_user_words: bool = False,
    post_probes: str = "all",
) -> dict[str, Any]:
    # stage 4 M2: another family as writer (and, by default, reader)
    wm = writer_model or MAIN_MODEL
    writer = writer or client(wm, run_dir)
    if think_writer is None and any(c.think for c in (CELLS if cells_cfg is None else cells_cfg)):
        # D0think cells; a thinking write can take up to `think_max_tokens` tokens (8,192 by
        # default, with 1,800 s to finish; a larger limit gets proportionally more time)
        timeout = 1800.0 * max(1.0, think_max_tokens / THINK_MAX_TOKENS)
        think_writer = client(wm, run_dir, extra_body=think_extra_body(wm), timeout=timeout)
    # prereg v4 H-a: the definition items written after the shift, in every cell, by `defwriter`
    def_body, def_max_tokens = defwriter_settings(defwriter) if defwriter else ({}, None)
    if def_writer is None and defwriter:
        def_writer = client(defwriter, run_dir, extra_body=def_body, timeout=1800.0)
    # prereg v4 G-b: the definition items' D1s programs from another model, temperature or sample
    pm = program_model or wm
    if (
        program_writer is None
        and any(c.synth for c in (CELLS if cells_cfg is None else cells_cfg))
        and separate_program_client(pm, wm, program_temperature, program_sample)
    ):
        body = program_extra_body(pm, program_temperature, program_sample)
        program_writer = client(pm, run_dir, extra_body=body)
    renderer = renderer or (client(render_model, run_dir) if render_model else None)
    # E7 theta_r: another reader over the same (cached) writer outputs
    reader = (
        client(reader_model, run_dir, extra_body=reader_extra_body(reader_model))
        if reader_model
        else writer
    )
    embed = embed_client(embed_model or EMBED_MODEL) if baselines else None  # E7 theta_e: swap
    summary: dict[str, Any] = {"seed": seed, "branches": {}}
    worlds = []
    for branch in branches:
        timer = Timer()
        world = generate_branch(seed, branch, world_params)
        render_report = None
        if renderer is not None:
            world, rendered = render_world_llm(world, renderer)
            render_report = rendered.to_json()
        log = SourceLog(world.events)
        worlds.append(world_record(world))
        t0 = world.fork - timedelta(minutes=1)
        cfgs = CELLS if cells_cfg is None else cells_cfg
        cells = {
            c.name: HoldMem(
                world,
                log,
                c,
                writer,
                wm,
                think_writer,
                think_max_tokens=think_max_tokens,
                def_writer=def_writer,
                def_max_tokens=def_max_tokens,
                program_writer=program_writer,
                program_user_words=program_user_words,
            )
            for c in cfgs
        }
        _par(partial(HoldMem.build, t=t0), list(cells.values()))
        pre = make_probes(
            world, log, [t0], tags={"phase": "pre", "k": 0}, unnamed_defs=probe_unnamed_defs
        )
        mismatches = []
        for p in pre:
            hashes = {name: context_hash(hm.serve(p).blocks) for name, hm in cells.items()}
            if len(set(hashes.values())) > 1:
                mismatches.append({"probe_id": p.probe_id, "hashes": hashes})
        hash_ok = not mismatches
        (run_dir / f"hash_check_{world.world_id}_{branch}.json").write_text(
            json.dumps({"probes": len(pre), "mismatches": mismatches, "ok": hash_ok}, indent=1)
        )
        if not hash_ok:
            raise RuntimeError(f"pre-intervention hash assertion failed: {world.world_id} {branch}")
        noise = (
            noise_floor(world, log, t0, pre, cells["holdmem_D0B1_source"], writer, run_dir, wm)
            if "holdmem_D0B1_source" in cells
            else {}
        )

        systems: dict[str, Any] = dict(cells)
        if baselines:
            systems["zero"] = ZeroSystem(world, log)
            systems["oracle"] = Oracle(world, log)
            if embed is not None:
                systems["rag"] = PlainRAG(world, log, embed)
        for view in views:
            systems[view] = ZeroDm(world, log, writer)

        answers = []
        maint_rows = []
        probe_rows = [p.to_json() for p in pre]
        for k in (0, *OFFSETS_WEEKS):
            if k == 0:
                probes = pre
            else:
                t = world.fork + timedelta(weeks=k)
                stats_all = _par(partial(HoldMem.maintain, t=t), list(cells.values()))
                for name, stats in zip(cells, stats_all, strict=True):
                    maint_rows.append(
                        {
                            "world_id": world.world_id,
                            "branch": branch,
                            "system": name,
                            "k": k,
                            **stats,
                        }
                    )
                probes = make_probes(
                    world, log, [t], tags={"phase": "post", "k": k}, unnamed_defs=probe_unnamed_defs
                )
                asked = (
                    discriminative_probes(probes, pre, f"{world.world_id}|{branch}|{k}")
                    if discriminative
                    else probes
                )
                if post_probes == "definition":  # after the discriminative draw: same items
                    asked = [p for p in asked if p.fact_id.startswith("k.")]
                ids = {p.probe_id for p in asked}
                # gold of every item is recorded (detection metrics use all items); only the
                # asked probes are answered
                probe_rows += [{**p.to_json(), "asked": p.probe_id in ids} for p in probes]
                probes = asked
            servers: list[tuple[str, Any]] = [(n, s.serve) for n, s in systems.items()]
            if JSON_ARM_OF in cells:
                arm = partial(cells[JSON_ARM_OF].serve, render="json")
                servers.append(("holdmem_json_arm", arm))
            for ans in _par(partial(_answer, probes=probes, reader=reader), servers, 4):
                answers += ans
        with _WRITE_LOCK:
            write_jsonl(run_dir / "probes.jsonl", probe_rows, mode="a")
            write_jsonl(run_dir / "answers.jsonl", (a.to_json() for a in answers), mode="a")
            write_jsonl(run_dir / "maintenance.jsonl", maint_rows, mode="a")
            write_jsonl(
                run_dir / "items.jsonl",
                (
                    {"world_id": world.world_id, "branch": branch, "system": name, **i.to_json()}
                    for name, hm in cells.items()
                    for i in hm.items.values()
                ),
                mode="a",
            )
        summary["branches"][branch] = {
            "seconds": round(timer.lap(), 1),
            "hash_ok": hash_ok,
            "noise_floor": noise,
            "costs": {name: s.cost for name, s in systems.items()},
            **({"render": render_report} if render_report is not None else {}),
            **(
                {"defwriter": {name: dict(hm.def_stats) for name, hm in cells.items()}}
                if def_writer is not None
                else {}
            ),
        }
    summary["worlds"] = worlds
    summary["llm_usage"] = writer.usage
    if think_writer is not None:
        summary["think_llm_usage"] = think_writer.usage
    if def_writer is not None:
        summary["defwriter_llm_usage"] = def_writer.usage
    if program_writer is not None:
        summary["program_llm_usage"] = program_writer.usage
    return summary


def run(
    seeds: list[int],
    name: str = "e1",
    out: Path | None = None,
    parallel_worlds: int = 3,
    cells_cfg: list[CellConfig] | None = None,
    baselines: bool = True,
    branches: tuple[str, ...] = BRANCHES,
    world_params: dict[str, Any] | None = None,
    discriminative: bool = False,
    reader_model: str | None = None,
    embed_model: str | None = None,
    render_model: str | None = None,
    writer_model: str | None = None,
    views: tuple[str, ...] = (),
    probe_unnamed_defs: bool = False,
    think_max_tokens: int = THINK_MAX_TOKENS,
    defwriter: str | None = None,
    program_model: str | None = None,
    program_temperature: float = 0.0,
    program_sample: int | None = None,
    program_user_words: bool = False,
    post_probes: str = "all",
) -> Path:
    cells = CELLS if cells_cfg is None else cells_cfg
    if defwriter and any(c.think for c in cells):
        raise ValueError("--defwriter and the D0think cells write the same items: run them apart")
    wm = writer_model or MAIN_MODEL
    check_program_temperature(program_model or wm, program_temperature)
    run_dir = out or new_run_dir(name)

    def one(seed: int) -> dict[str, Any]:
        s = run_world(
            seed,
            run_dir,
            branches=branches,
            baselines=baselines,
            cells_cfg=cells,
            world_params=world_params,
            discriminative=discriminative,
            reader_model=reader_model,
            embed_model=embed_model,
            render_model=render_model,
            writer_model=wm,
            views=views,
            probe_unnamed_defs=probe_unnamed_defs,
            think_max_tokens=think_max_tokens,
            defwriter=defwriter,
            program_model=program_model,
            program_temperature=program_temperature,
            program_sample=program_sample,
            program_user_words=program_user_words,
            post_probes=post_probes,
        )
        with _WRITE_LOCK:
            write_jsonl(run_dir / "world_summaries.jsonl", [s], mode="a")
        return s

    summaries = _par(one, seeds, parallel_worlds)
    pm = program_model or wm
    separate = separate_program_client(pm, wm, program_temperature, program_sample)
    models = {
        "writer": wm,
        "reader": reader_model or wm,
        "embedding": embed_model or EMBED_MODEL,
        **({"renderer": render_model} if render_model else {}),
        **({"defwriter": defwriter} if defwriter else {}),
        **({"program_writer": pm} if pm != wm else {}),
    }
    write_env(run_dir, models)
    write_manifest(
        run_dir,
        "E1",
        [w for s in summaries for w in s["worlds"]],
        models,
        {
            "seeds": seeds,
            "branches": list(branches),
            "world_params": world_params or {},
            "render_model": render_model,
            "probes_after_fork": "discriminative" if discriminative else "all",
            "cells": [c.name for c in cells],
            "views": list(views),
            "baselines": baselines,
            "json_arm_of": JSON_ARM_OF,
            "offsets_weeks": list(OFFSETS_WEEKS),
            "t0": "fork - 1 minute",
            "reader_budget_tokens": 4096,
            "probe_unnamed_defs": probe_unnamed_defs,
            "post_probes": post_probes,
            **(
                {
                    "think_writer": {
                        "extra_body": think_extra_body(wm),
                        "max_tokens": think_max_tokens,
                    }
                }
                if any(c.think for c in cells)
                else {}
            ),
            **(
                {"full_writer": {"budget_tokens": FULL_BUDGET, "doc_cap_tokens": WRITER_DOC_CAP}}
                if any(c.full for c in cells)
                else {}
            ),
            **(
                {"reader": {"model": reader_model, "extra_body": reader_extra_body(reader_model)}}
                if reader_model and is_claude(reader_model)
                else {}
            ),
            **(
                {
                    "defwriter": {
                        "model": defwriter,
                        "extra_body": defwriter_settings(defwriter)[0],
                        "max_tokens": defwriter_settings(defwriter)[1],
                        "items": "definition items written after the shift, every cell",
                    }
                }
                if defwriter
                else {}
            ),
            **(
                {
                    "program_writer": {
                        "model": pm,
                        "temperature": program_temperature,
                        "sample": program_sample,
                        "extra_body": (
                            program_extra_body(pm, program_temperature, program_sample)
                            if separate
                            else None
                        ),
                        "items": "definition items (other items: the writer)",
                        "user_words": program_user_words,
                    }
                }
                if any(c.synth for c in cells)
                else {}
            ),
        },
    )
    return run_dir


def parse_cell(token: str) -> CellConfig:
    """Parse one --cells token, e.g. 'D1B1_source', 'D1B0', 'D1B1_cited_nofb', 'D1B0_nofb'; the
    writer-authored derivation cells of prereg v2 §4 are 'D1sB0', 'D1sB1_source', 'D1sB1_cited'.
    The control cells: 'D0B1_gold', 'D1B1_scope' (keys), 'D0B0_refresh' (refresh without
    invalidation), 'D0tB0', 'D0tB1_source' (template presentation), 'D0thinkB1_source',
    'D0fullB1_source' (every in-scope record for the definition items), the retraction-aware
    programs 'D1sB1_source_ra', 'D1sB0_ra', and the slot-keyed upsert cells of prereg v4 block U
    'D0B0_upsert', 'D1B0_upsert', 'D1sB0_upsert' (refresh with D1s programs: 'D1sB0_refresh')."""
    body = token.removesuffix("_nofb")
    upsert = body.endswith("_upsert")
    rest = body.removesuffix("_upsert")
    refresh, ra = rest.endswith("_refresh"), rest.endswith("_ra")
    m = re.fullmatch(
        r"D(\d)(s|t|think|full)?B(\d)(?:_([a-z]+))?",
        rest.removesuffix("_refresh").removesuffix("_ra"),
    )
    if m is None:
        raise ValueError(f"bad cell token {token!r}")
    d, variant, b, granularity = m.groups()
    return CellConfig(
        int(d),
        int(b),
        granularity or "source",
        fallback=body == token,
        synth=variant == "s",
        refresh=refresh,
        present=variant == "t",
        think=variant == "think",
        full=variant == "full",
        ra=ra,
        upsert=upsert,
    )


def parse_cells(spec: str) -> tuple[list[CellConfig], tuple[str, ...]]:
    """Split a --cells value into HoldMem cells and query-time views (`VIEWS`)."""
    tokens = [t for t in spec.split(",") if t]
    views = tuple(t for t in tokens if t in VIEWS)
    return [parse_cell(t) for t in tokens if t not in VIEWS], views


def parse_seeds(spec: str) -> list[int]:
    """'0-9' -> [0..9]; '0,3,5' -> [0, 3, 5]."""
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            out += list(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--name", default="e1")
    ap.add_argument("--parallel-worlds", type=int, default=3)
    ap.add_argument("--granularities", default="", help="e.g. 'cited': only B=1 cells, D in {0,1}")
    ap.add_argument("--no-baselines", action="store_true")
    ap.add_argument("--discriminative", action="store_true")
    ap.add_argument("--branches", default=",".join(BRANCHES))
    ap.add_argument(
        "--cells",
        default="",
        help="e.g. 'D0B0,D1B1_cited,D1sB1_source,D0B1_gold,D0fullB1_source,zero_Dm'; 'none': "
        "baselines only",
    )
    ap.add_argument("--writer", default="", help="writer model (stage 4 M2); default: main")
    ap.add_argument("--reader", default="", help="reader model (E7 theta_r); default: writer")
    ap.add_argument("--months", type=int, default=12, help="short worlds (E5 white-box reference)")
    ap.add_argument(
        "--kappa-set", default="", help='"heldout:<seed>": held-out definitions (v4_round3 §G-a)'
    )
    ap.add_argument("--embed", default="", help="embedding model for RAG (E7 theta_e)")
    ap.add_argument(
        "--render", default="", help="LLM-render the free-text bodies with this model (block L)"
    )
    ap.add_argument(
        "--probe-unnamed-defs",
        action="store_true",
        help="definition-item probes without 'Definition in effect: ...' (prereg v4 block Q)",
    )
    ap.add_argument(
        "--think-max-tokens",
        type=int,
        default=THINK_MAX_TOKENS,
        help="output limit of the D0think writer's requests (prereg v4 block H-b: 32768)",
    )
    ap.add_argument(
        "--defwriter",
        default="",
        help="model that writes the definition items after the shift in every cell, e.g. "
        "'claude-sonnet-5-5' (prereg v4 block H-a); default: the writer",
    )
    ap.add_argument(
        "--program-model",
        default="",
        help="model that writes the definition items' D1s programs (prereg v4 block G-b); "
        "default: the writer",
    )
    ap.add_argument(
        "--program-temperature",
        type=float,
        default=0.0,
        help="sampling temperature of those program requests (G-b: 0.7 for served models; "
        "refused for a Claude model); default 0",
    )
    ap.add_argument(
        "--post-probes",
        choices=("all", "definition"),
        default="all",
        help="probes answered after the fork: all asked items, or only the definition items "
        "among them (blocks judged on definition items; the t0 probes are always all asked)",
    )
    ap.add_argument(
        "--program-user-words",
        action="store_true",
        help="the definition items' D1s program requests state the definition in the user's "
        "words (prereg v4b block V, `holdmem.program.USER_WORDS`)",
    )
    ap.add_argument(
        "--program-sample",
        type=int,
        default=None,
        help="sample index I: adds memshift_sample=I to those program requests (own cache key)",
    )
    a = ap.parse_args()
    try:
        check_program_temperature(a.program_model or a.writer or MAIN_MODEL, a.program_temperature)
    except ValueError as exc:
        ap.error(str(exc))
    cfg = (
        [CellConfig(d, 1, g) for g in a.granularities.split(",") for d in (0, 1)]
        if a.granularities
        else None
    )
    views: tuple[str, ...] = ()
    if a.cells == "none":
        cfg = []
    elif a.cells:
        cfg, views = parse_cells(a.cells)
    print(
        run(
            parse_seeds(a.seeds),
            a.name,
            parallel_worlds=a.parallel_worlds,
            cells_cfg=cfg,
            baselines=not a.no_baselines,
            discriminative=a.discriminative,
            branches=tuple(a.branches.split(",")),
            reader_model=a.reader or None,
            world_params=world_params(a.months, a.kappa_set or None),
            embed_model=a.embed or None,
            render_model=a.render or None,
            writer_model=a.writer or None,
            views=views,
            probe_unnamed_defs=a.probe_unnamed_defs,
            think_max_tokens=a.think_max_tokens,
            defwriter=a.defwriter or None,
            program_model=a.program_model or None,
            program_temperature=a.program_temperature,
            program_sample=a.program_sample,
            program_user_words=a.program_user_words,
            post_probes=a.post_probes,
        )
    )

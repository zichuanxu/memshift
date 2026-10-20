"""Block T (E1): complete typed input for the D=0 writer. Quantities of the runs with the cells
D0B1_source, D1B1_source and D0typedB1_source (`holdmem.writer.typed_records`); nothing here
calls a model or applies a threshold.

    python -m memshift.analysis.round3b_typed --runs RUN [RUN ...] [--compare RUN_OR_STAGE ...] \\
        [--response-cache SQLITE] [--no-coverage] --out FILE.json
    python -m memshift.analysis.round3b_typed --facts [--seeds 1000-1009] [--tokenizer JSON] \\
        [--program-cache QWEN_SQLITE] [--workers N] --out FILE.json

Arms. The runs are merged per variant (`round3_blocks.by_variant`: writer, reader, definition
writer, ...), so the Qwen3.5-9B arm (no definition writer) and the Claude Sonnet 5.5 arm
(`--defwriter claude-sonnet-5-5`, H-a settings) are separate arms of one result. Within an arm,
the newest complete run of each (cell, world, branch) is used (`round3_blocks.Merged`).

Quantities, per arm, on the changed definition item-times (gold value or status differs from t0,
`e1.item_table`; fact ids `k.*`) after S_src, after S_def and the two pooled, per definition and
the six pooled:
  r        r of Eq. (1) per cell: ContractValid of the changed item-times the source-epoch key
           invalidated (`round3_blocks.r0`'s definition, applied to each cell: D0typedB1_source
           "typed", D0B1_source "bm25", D1B1_source "d1"); the three cells share the key, so they
           re-derive the same item-times. The paired differences typed - bm25, d1 - typed and
           d1 - bm25 are per item-time over the item-times present in both cells.
  update   Update (ContractValid on the changed item-times) of the three cells, and the D effect
           at B=1: d1 - typed and d1 - bm25 (and typed - bm25), paired per item-time.
  each again split by the scope completeness of the D=0 writer's BM25 selection (the stage-8 split
  of `ex_round3_review.coverage`: per definition item-time after the shift, complete when the
  writer's 6,000-token selection, `holdmem.writer.records_for`, holds every record the definition
  ranges over, `gen.kappa.scope_members`), recomputed from the seeded generator for the runs'
  worlds (whose events must hash to the runs' manifests); `--no-coverage` leaves the split out.
Validity, per arm: the typed cell's writes (item notes `typed`: unreadable replies, replies cut
off at the output limit, completion tokens), the definition-item writer's per-cell counts
(`def_stats` of the run summaries) and, with a definition writer, the H-a validity of its distinct
write requests (`round3_blocks.h_validity`, calls.jsonl and the response cache when
`--response-cache` names it). The t0 diagnostic (`typed_t0.jsonl`): per definition, the share of
typed derivations at t0 that are readable and that match gold (one row per world and definition;
the src and def branches send the same request at t0 and the first row is kept).

Integrity (`round3_blocks.pairing`, reported under `pairing`): every pre-shift hash assertion
exists and passed; with `--compare`, the committed runs of the same variant up to the D_m program
settings, which change no request of these cells (`anchor_group`: e1_test_* for the Qwen arm, the
block H-a runs, which also had D_m cells with Sonnet programs, for the Sonnet arm) are the
reference: the probe keys and gold match, and the answers of D0B1_source, D1B1_source and the
JSON arm equal the committed ones at every checkpoint (all their requests are cache hits of those
runs).

CPU facts before any run (`--facts`; the seeded generator only, worlds 1000-1009, S_src and S_def,
t0 and the four checkpoints): (a) the gold of the six definitions, both versions at every
item-time, computed from snapshots whose records lack the fields the typed input omits (`text`
everywhere, `body` of e-mail and messages) equals the gold from the full records (the evaluators
of `gen.kappa` read none of them); with `--program-cache` (the Qwen3.5-9B response cache), the
committed D_m programs of the main study (`holdmem.program`, both versions of each definition) run
on the stripped and on the full records of every item-time and their results are compared; (b)
the typed requests a run sends, on the source-epoch schedule (the key moved) and at t0, per
definition and branch, and how many are distinct; (c) with `--tokenizer` (the Qwen3.5-9B
tokenizer.json, read offline; needs the `tokenizers` package, `uv run --with tokenizers`), the
prompt tokens of each typed request, system and user message in the chat template (its markers
and the empty thinking block): median, 95th percentile, maximum and sum per definition, and the
sum over distinct requests.

Estimators: those of `round3_blocks` and `ex_e1_classes` (world-clustered bootstrap, 2,000
draws, seed 996, percentile 95% interval; per-item paired differences over the item-times present
in both cells). Pooled branches prefix the item keys with the branch (`round3_blocks._pool`).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_round2 as r2
from memshift.analysis import ex_round3_review as rv
from memshift.analysis import round3_blocks as rr
from memshift.experiments.common import MAIN_MODEL, world_params
from memshift.experiments.e1 import OFFSETS_WEEKS, parse_seeds
from memshift.runinfo import code_state, file_sha256

Row = e1.Row
Vec = ex.Vec
Est = ex.Est
Cond = e1.Cond
CovKey = rv.CovKey

STATUS = (
    "block T (typed input): quantities only; reads the runs, "
    "the committed runs they pair with and the seeded generator, and calls no model"
)
UNITS = rr.UNITS
DRAWS = rr.DRAWS
TYPED = "holdmem_D0typedB1_source"
BM25 = rr.ARMS["D0"][1]  # holdmem_D0B1_source
D1 = rr.ARMS["D1"][1]  # holdmem_D1B1_source
JSON_ARM = "holdmem_json_arm"
CELLS = {"typed": TYPED, "bm25": BM25, "d1": D1}
CONTRASTS = (
    ("typed_minus_bm25", "typed", "bm25"),
    ("d1_minus_typed", "d1", "typed"),
    ("d1_minus_bm25", "d1", "bm25"),
)
BRANCHES = ("src", "def", "pooled")
DEFINITIONS = tuple(f for f, _, _ in r2.DEFINITIONS)
GROUPS = (*DEFINITIONS, "definition")  # each definition, then the six pooled
SUBSETS = ("complete", "incomplete", "all")
ANCHORS = frozenset({BM25, D1, JSON_ARM})


def _scope(run: rr.Run, system: str) -> str | None:
    """Anchors: every cell but the typed one, at every checkpoint."""
    return "all" if system in ANCHORS else None


def anchor_group(run: rr.Run) -> str:
    """What changes the anchors' requests: the run's variant without the D_m program settings
    (program writer, user words), which only the D1s cells read."""
    v = json.loads(run.variant)
    return json.dumps({k: x for k, x in v.items() if k not in ("program", "user_words")})


# ================================================================================== coverage
def scope_complete(runs: Sequence[rr.Run]) -> dict[CovKey, bool]:
    """The stage-8 coverage split (`ex_round3_review._scope_coverage`) for the runs' worlds:
    (world, branch, kappa id, k) -> the BM25 selection holds every scope record."""
    times: dict[str, dict[int, str]] = defaultdict(dict)
    worlds: dict[str, tuple[int, dict[str, Any]]] = {}
    hashes: dict[tuple[str, str], str] = {}
    for run in runs:
        for w in run.manifest["worlds"]:
            worlds[w["world_id"]] = (int(w["seed"]), run.world_params)
            hashes[w["world_id"], w["branch"]] = str(w["events_sha256"])
        for p in run.probes:
            k = rr._k(p)
            if times[p["world_id"]].setdefault(k, p["as_of"]) != p["as_of"]:
                raise ValueError(f"{p['world_id']} k={k}: branches probe at different times")
    st = rv._Stage([r.path for r in runs], {}, dict(times), worlds, hashes)
    cov, _ = rv._scope_coverage(st, [])
    return {key: n == got for key, (n, got) in cov.items()}


def _subset(complete: Mapping[CovKey, bool] | None, sub: str) -> Cond:
    if sub == "all":
        return lambda r: True
    assert complete is not None

    def covered(r: Row) -> bool:
        key = (r["world_id"], r["branch"], str(r["fact_id"]).removeprefix("k."), int(r["k"]))
        return complete[key]

    return covered if sub == "complete" else lambda r: not covered(r)


# ================================================================================ quantities
def _group(group: str) -> Cond:
    if group == "definition":
        return lambda r: ex.in_class(r, "definition")
    return lambda r: r["fact_id"] == group


def _both(*conds: Cond) -> Cond:
    return lambda r: all(c(r) for c in conds)


def _vec(tab: ex.Table, system: str, branch: str, cond: Cond) -> Vec:
    """ContractValid of `system` on the rows that meet `cond` (pooled: keys prefixed by branch)."""
    if branch != "pooled":
        return tab.vec(system, branch, "cv", cond)
    return rr._pool({b: tab.vec(system, b, "cv", cond) for b in rr.SHIFTS})


def _estimates(vecs: Mapping[str, Vec], draws: int) -> dict[str, Any]:
    """Levels of the three cells and their paired differences."""
    out: dict[str, Any] = {name: ex.vec_mean(v, draws) for name, v in vecs.items()}
    for key, x, y in CONTRASTS:
        out[key] = ex.paired_estimate(vecs[x], vecs[y], draws)
    return out


def arm_quantities(
    tab: ex.Table, complete: Mapping[CovKey, bool] | None, draws: int = DRAWS
) -> dict[str, Any]:
    """r and Update of the three cells and their contrasts, per branch, definition and subset."""
    subsets = SUBSETS if complete is not None else ("all",)
    r: dict[str, Any] = {}
    update: dict[str, Any] = {}
    for b in BRANCHES:
        for g in GROUPS:
            for sub in subsets:
                base = _both(rr.changed("definition"), _group(g), _subset(complete, sub))
                det = _both(base, lambda row: bool(row.get("detected")))
                rv_ = {name: _vec(tab, cell, b, det) for name, cell in CELLS.items()}
                uv = {name: _vec(tab, cell, b, base) for name, cell in CELLS.items()}
                r.setdefault(b, {}).setdefault(g, {})[sub] = _estimates(rv_, draws)
                update.setdefault(b, {}).setdefault(g, {})[sub] = _estimates(uv, draws)
    return {"r": r, "update": update}


def typed_validity(runs: Sequence[rr.Run]) -> dict[str, Any]:
    """The typed cell's writes after the shift, from its item notes."""
    writes = unreadable = cut = 0
    tokens: list[int] = []
    by_def: dict[str, dict[str, int]] = defaultdict(lambda: {"writes": 0, "unreadable": 0})
    for run in runs:
        for it in run.items:
            if it["system"] != TYPED:
                continue
            for h in it["history"]:
                note = h.get("typed")
                if note is None:
                    continue
                writes += 1
                unreadable += not note["ok"]
                cut += bool(note["at_limit"])
                tokens.append(int(note["completion_tokens"]))
                by_def[it["slot_id"]]["writes"] += 1
                by_def[it["slot_id"]]["unreadable"] += not note["ok"]
    return {
        "writes": writes,
        "unreadable": unreadable,
        "unreadable_share": unreadable / writes if writes else None,
        "at_limit": cut,
        "at_limit_share": cut / writes if writes else None,
        "completion_tokens_mean": sum(tokens) / writes if writes else None,
        "completion_tokens_max": max(tokens) if tokens else None,
        "by_definition": dict(sorted(by_def.items())),
    }


def cell_def_stats(runs: Sequence[rr.Run]) -> dict[str, dict[str, int]]:
    """The definition-item writer's per-cell counts (`HoldMem.def_stats`) in the run summaries."""
    out: dict[str, dict[str, int]] = defaultdict(lambda: {"writes": 0, "unreadable": 0})
    for run in runs:
        for s in rr._rows(run.path / "world_summaries.jsonl"):
            for node in (s.get("branches") or {}).values():
                for cell, st in (node.get("defwriter") or {}).items():
                    for k in ("writes", "unreadable", "at_limit"):
                        out[cell][k] = out[cell].get(k, 0) + int(st.get(k, 0))
    return dict(sorted(out.items()))


def t0_rows(runs: Sequence[rr.Run]) -> list[Row]:
    """The t0 diagnostic, one row per (world, definition): the first row of the runs' files."""
    seen: dict[tuple[str, str], Row] = {}
    for run in sorted(runs, key=lambda r: (r.created, r.name), reverse=True):  # newest first
        for row in rr._rows(run.path / "typed_t0.jsonl"):
            seen.setdefault((row["world_id"], row["fact_id"]), row)
    return [seen[k] for k in sorted(seen)]


def t0_quantities(rows: Sequence[Row], draws: int = DRAWS) -> dict[str, Any]:
    """Per definition and pooled: readable and correct shares, world bootstrap."""
    out: dict[str, Any] = {}
    for g in GROUPS:
        sel = [r for r in rows if g == "definition" or r["fact_id"] == g]
        ok: dict[str, list[float]] = defaultdict(list)
        correct: dict[str, list[float]] = defaultdict(list)
        for r in sel:
            p = {
                "gold_status": r["gold_status"],
                "gold_value": r["gold_value"],
                "vtype": r["vtype"],
            }
            ok[r["world_id"]].append(float(r["ok"]))
            correct[r["world_id"]].append(
                float(bool(r["ok"]) and e1._repair_matches(p, r["value"]))
            )
        out[g] = {
            "n": len(sel),
            "readable": ex.mean_estimate(ok, draws),
            "correct": ex.mean_estimate(correct, draws),
            "at_limit": sum(bool(r["at_limit"]) for r in sel),
        }
    return out


# ===================================================================================== block
def analyse(
    runs: Sequence[Path],
    compare: Sequence[Path] = (),
    response_cache: Path | None = None,
    coverage: bool = True,
    draws: int = DRAWS,
    complete: Mapping[CovKey, bool] | None = None,
) -> dict[str, Any]:
    """Every quantity of the module docstring. `complete` overrides the coverage split (tests)."""
    keep = {TYPED, BM25, D1}.__contains__
    new, skipped = rr.read_runs(runs, items=keep)
    need = {*ANCHORS, "holdmem_D0B0"}.__contains__
    cmp_runs, skipped_c = rr.read_runs(rr.expand(compare), need) if compare else ([], [])
    newm = rr.Merged(new, rr.by_variant)
    cmpm = rr.Merged(cmp_runs, anchor_group) if cmp_runs else None
    pair = rr.pairing(
        rr.Merged(new, anchor_group),
        cmpm,
        [*skipped, *skipped_c],
        _scope if cmpm else None,
        anchor_group,
    )
    if complete is None and coverage and new:
        complete = scope_complete(new)
    arms: dict[str, Any] = {}
    for var in newm.groups():
        vruns = newm.runs_of(var)
        head = vruns[0]
        tab = newm.table(var)
        name = head.defwriter or head.writer
        node: dict[str, Any] = {
            "variant": json.loads(var),
            "writer": head.writer,
            "defwriter": head.defwriter,
            "runs": sorted(r.name for r in vruns),
            "worlds": sorted(rr.cell_worlds(tab, CELLS.values())),
            "validity": {
                "typed_writes": typed_validity(vruns),
                "cell_def_stats": cell_def_stats(vruns),
            },
            "t0": t0_quantities(t0_rows(vruns), draws),
            **arm_quantities(tab, complete, draws),
        }
        if head.defwriter:
            paths = [r.path for r in vruns]
            node["validity"]["defwriter"] = rr.h_validity(paths, head.defwriter, response_cache)
        arms[name] = node
    cov_counts: dict[str, Any] | None = None
    if complete is not None:
        by_def: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for (_w, _b, kid, _k), ok in complete.items():
            by_def[f"k.{kid}"][0] += int(ok)
            by_def[f"k.{kid}"][1] += 1
        cov_counts = {k: {"complete": c, "item_times": n} for k, (c, n) in sorted(by_def.items())}
    return {
        "block": "T",
        "status": STATUS,
        "units": UNITS,
        "inputs": {"runs": [str(p) for p in runs], "compare": [str(p) for p in compare]},
        "cells": CELLS,
        "pairing": pair,
        "coverage": cov_counts,
        "arms": arms,
        "code": code_state(),
    }


def summary(res: Mapping[str, Any]) -> str:
    pair = res["pairing"]
    bad = {k: v for k, v in pair.get("counts", {}).items() if v}
    lines = [
        f"block T: pairing {'ok' if pair.get('ok') else 'FAILED ' + json.dumps(bad)}"
        f" ({pair.get('anchor_answers_compared', 0)} anchor answers compared)"
    ]
    fmt: Callable[[Mapping[str, Any] | None], str] = rr._iv
    for name, node in res["arms"].items():
        v = node["validity"]["typed_writes"]
        lines.append(
            f"  {name}: typed writes {v['writes']}, unreadable {v['unreadable']}, "
            f"cut off {v['at_limit']}"
        )
        for b in BRANCHES:
            r = node["r"][b]["definition"]["all"]
            lines.append(
                f"    {b:6} r typed {rr._pc(r['typed']['est'])} bm25 {rr._pc(r['bm25']['est'])} "
                f"d1 {rr._pc(r['d1']['est'])}  d1-typed {fmt(r['d1_minus_typed'])}"
            )
    return "\n".join(lines)


# ===================================================================================== facts
FACT_SEEDS = tuple(range(1000, 1010))
STRIPPED = {"email": ("body",), "messages": ("body",)}  # and `text` everywhere (`TYPED_DROPPED`)


def stripped_snapshot(snap: Any) -> Any:
    """The snapshot with the fields the typed input omits removed from every payload."""
    from memshift.gen.state import Snapshot

    objs = {}
    for oid, o in snap.objects.items():
        drop = {"text", *STRIPPED.get(o.source, ())}
        objs[oid] = replace(o, payload={k: v for k, v in o.payload.items() if k not in drop})
    return Snapshot(snap.t, objs, snap.tombstones)


def strip_records(recs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """`program_records` without the fields the typed input omits."""
    out = []
    for r in recs:
        drop = {"text", *STRIPPED.get(str(r["source"]), ())}
        out.append({k: v for k, v in r.items() if k not in drop})
    return out


def chat_text(req: Mapping[str, Any]) -> str:
    """A writer request in the Qwen3.5-9B chat template with reasoning off, as served."""
    sys_, user = req["messages"][0]["content"], req["messages"][1]["content"]
    return (
        f"<|im_start|>system\n{sys_}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def program_key(model: str, req: Mapping[str, Any]) -> str:
    """The response-cache key of a served model's program request (`LLMClient.request_body`
    with the model's default extras; it reads only the model, the extras and the API)."""
    from memshift.llm import LLMClient, default_extra_body
    from memshift.textutil import sha256_json

    shape = LLMClient.__new__(LLMClient)
    shape.model, shape.extra_body, shape.api = model, default_extra_body(model), "openai"
    return sha256_json(shape.request_body(req["messages"], req["max_tokens"], req["schema"]))


def committed_programs(world: Any, cache: Path) -> dict[tuple[str, str], str | None]:
    """(kappa id, version) -> the code of the main study's D_m program (Qwen3.5-9B, the writer's
    default request), None when the cache lacks it. The request depends on the world and the
    time only through the definition version in effect, which a copy of the world with that
    version from its start fixes."""
    from memshift.holdmem.program import extract_code, program_request
    from memshift.llm import LLMResult

    db = sqlite3.connect(f"file:{cache}?mode=ro", uri=True)
    out: dict[tuple[str, str], str | None] = {}
    try:
        for slot in (s for s in world.slots if s.kappa_id):
            kid = str(slot.kappa_id)
            for ver in ("v1", "v2"):
                fixed = replace(world, kappa_log=[(world.start, kid, ver)])
                key = program_key(MAIN_MODEL, program_request(fixed, slot, world.start))
                row = db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
                text = str(json.loads(row[0]).get("text") or "") if row else None
                res = LLMResult(text or "", 0, 0, 0.0, True, "", key)
                out[kid, ver] = extract_code(res) if text is not None else None
    finally:
        db.close()
    return out


def _facts_seed(
    seed: int, tokenizer: Path | None, program_cache: Path | None, months: int = 12
) -> dict[str, Any]:
    """The facts of one seed, S_src and S_def (see the module docstring)."""
    from memshift.gen.kappa import EVALUATORS
    from memshift.gen.shifts import generate_branch
    from memshift.gen.state import SourceLog
    from memshift.holdmem.program import program_records, run_program
    from memshift.holdmem.system import CellConfig, HoldMem
    from memshift.holdmem.writer import typed_records, writer_request

    count: Callable[[str], int] | None = None
    if tokenizer is not None:
        from tokenizers import Tokenizer

        tok = Tokenizer.from_file(str(tokenizer))
        count = lambda s: len(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    gold: dict[str, Any] = {"evaluations": 0, "item_times": 0, "differ": []}
    progs: dict[str, Counter[str]] = defaultdict(Counter)
    reqs: list[dict[str, Any]] = []
    codes: dict[tuple[str, str], str | None] = {}
    for b in rr.SHIFTS:
        world = generate_branch(seed, b, world_params(months))
        log = SourceLog(world.events)
        if program_cache is not None and not codes:
            codes = committed_programs(world, program_cache)
        # only the key is used: the source-epoch schedule of the typed cell (no writer call)
        cell = HoldMem(world, log, CellConfig(0, 1, "source", typed=True), None, "")  # type: ignore[arg-type]
        t0 = world.fork - timedelta(minutes=1)
        slots = [s for s in world.slots if s.kappa_id]
        bind = {s.slot_id: cell.key_at(s, [], t0) for s in slots}
        for k in (0, *OFFSETS_WEEKS):
            t = t0 if k == 0 else world.fork + timedelta(weeks=k)
            snap = log.snapshot(t)
            bare = stripped_snapshot(snap)
            for slot in slots:
                kid = str(slot.kappa_id)
                gold["item_times"] += 1
                for ver in ("v1", "v2"):
                    gold["evaluations"] += 1
                    if EVALUATORS[kid](snap, t, ver) != EVALUATORS[kid](bare, t, ver):
                        gold["differ"].append([f"w{seed}", b, kid, ver, k])
                if program_cache is not None:
                    recs = program_records(log, slot, t)
                    as_of = f"{t:%Y-%m-%d %H:%M}"
                    for ver in ("v1", "v2"):
                        code = codes.get((kid, ver))
                        tag = f"k.{kid} {ver}"
                        if code is None:
                            progs[tag]["no_program"] += 1
                            continue
                        full = run_program(code, recs, as_of, slot.vtype)
                        same = full == run_program(code, strip_records(recs), as_of, slot.vtype)
                        progs[tag]["same" if same else "differ"] += 1
                        progs[tag]["runs" if full[0] else "fails_on_full_records"] += 1
                if k:
                    key = cell.key_at(slot, [], t)
                    if key == bind[slot.slot_id]:
                        continue  # the typed cell does not re-derive the item here
                    bind[slot.slot_id] = key
                req = writer_request(world, slot, t, [], typed=typed_records(log, slot, t))
                reqs.append(
                    {
                        "branch": b,
                        "fact_id": slot.slot_id,
                        "k": k,
                        "sha": hashlib.sha256(json.dumps(req, sort_keys=True).encode()).hexdigest(),
                        "tokens": count(chat_text(req)) if count is not None else None,
                    }
                )
    return {"gold": gold, "programs": {k: dict(v) for k, v in progs.items()}, "requests": reqs}


def _sizes(xs: Sequence[int]) -> dict[str, Any]:
    v = sorted(xs)
    if not v:
        return {"n": 0}
    return {
        "n": len(v),
        "median": float(statistics.median(v)),
        "p95": v[int(0.95 * (len(v) - 1))],
        "max": v[-1],
        "sum": sum(v),
    }


def facts(
    seeds: Sequence[int] = FACT_SEEDS,
    tokenizer: Path | None = None,
    program_cache: Path | None = None,
    workers: int = 1,
    months: int = 12,
) -> dict[str, Any]:
    """The CPU facts of the module docstring, before any run (with `workers` > 1, one process per
    seed). `months`: the world length (the worlds of the main study have 12)."""
    n = len(seeds)
    if workers > 1:
        with ProcessPoolExecutor(workers) as pool:
            args = ([tokenizer] * n, [program_cache] * n, [months] * n)
            parts = list(pool.map(_facts_seed, seeds, *args))
    else:
        parts = [_facts_seed(s, tokenizer, program_cache, months) for s in seeds]
    differ = [d for p in parts for d in p["gold"]["differ"]]
    progs: dict[str, Counter[str]] = defaultdict(Counter)
    for p in parts:
        for tag, c in p["programs"].items():
            progs[tag].update(c)
    reqs = [r for p in parts for r in p["requests"]]
    post = [r for r in reqs if r["k"]]
    at_t0 = [r for r in reqs if not r["k"]]
    out: dict[str, Any] = {
        "status": STATUS,
        "worlds": [f"w{s}" for s in seeds],
        "world_months": months,
        "branches": list(rr.SHIFTS),
        "checkpoints_weeks_after_fork": list(OFFSETS_WEEKS),
        "omitted": {"all": ["text"], **{s: list(f) for s, f in STRIPPED.items()}},
        "a_gold_invariant": {
            "evaluations": sum(p["gold"]["evaluations"] for p in parts),
            "item_times": sum(p["gold"]["item_times"] for p in parts),
            "differ": differ,
            "identical": not differ,
        },
        "a_programs": None,
        "b_requests": {
            "rederivations": len(post),
            "rederivations_distinct": len({r["sha"] for r in post}),
            "per_definition_branch": dict(
                sorted(Counter(f"{r['fact_id']} {r['branch']}" for r in post).items())
            ),
            "t0_diagnostic": len(at_t0),
            "t0_distinct": len({r["sha"] for r in at_t0}),
            "distinct_all": len({r["sha"] for r in reqs}),
        },
        "code": code_state(),
    }
    if program_cache is not None:
        out["a_programs"] = {
            "cache": program_cache.name,
            "by_definition_version": {k: dict(v) for k, v in sorted(progs.items())},
            "same": sum(c["same"] for c in progs.values()),
            "differ": sum(c["differ"] for c in progs.values()),
        }
    if tokenizer is not None:
        by_def: dict[str, list[int]] = defaultdict(list)
        for r in post:
            by_def[r["fact_id"]].append(int(r["tokens"]))
        distinct = {r["sha"]: int(r["tokens"]) for r in reqs}
        out["c_prompt_tokens"] = {
            "tokenizer": {"model": MAIN_MODEL, "sha256": file_sha256(tokenizer)},
            "rederivations": {
                **{k: _sizes(v) for k, v in sorted(by_def.items())},
                "all": _sizes([int(r["tokens"]) for r in post]),
            },
            "t0_diagnostic": _sizes([int(r["tokens"]) for r in at_t0]),
            "distinct_sum": sum(distinct.values()),
        }
    return out


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 2)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs", nargs="+", default=[], type=Path)
    ap.add_argument("--compare", nargs="*", default=[], type=Path)
    ap.add_argument("--response-cache", type=Path, default=None)
    ap.add_argument("--no-coverage", action="store_true")
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--facts", action="store_true", help="the CPU facts before any run")
    ap.add_argument("--seeds", default="1000-1009")
    ap.add_argument("--tokenizer", type=Path, default=None, help="Qwen3.5-9B tokenizer.json")
    ap.add_argument("--program-cache", type=Path, default=None, help="Qwen3.5-9B response cache")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.facts:
        res = facts(parse_seeds(a.seeds), a.tokenizer, a.program_cache, a.workers)
    elif a.runs:
        res = analyse(a.runs, a.compare, a.response_cache, not a.no_coverage, a.draws)
    else:
        ap.error("give --runs, or --facts")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    if not a.facts:
        print(summary(res))


if __name__ == "__main__":
    main()

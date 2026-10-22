"""What part of the B x D interaction on Update is attributable to the shift, against the natural
branch (no model call). Two parts, written to one JSON file
(`runs/stage6_ex_round3/natural.json` by default):

composition  The all-item "by the shift" and "drift only" B x D of `analysis.ex_round2`
             (tab:cause) are pooled means over item-times of two classes, definition items and
             the other 42 items, so each is the class-share-weighted sum of the class B x D. Per
             branch and part: the class shares, the class B x D, each class's contribution
             (share x class B x D, with its own world-bootstrap interval), the identity residual,
             and the all-item B x D re-weighted to the branch's pooled class mix (the mix of the
             all-item contrasts of `e1.contrasts`: shift and drift together). A small all-item
             value that comes from a large definition effect on few item-times shows as a
             definition contribution that exceeds it, balanced by a near-zero or slightly negative
             other-class contribution. The class values are the ones of the paper
             (`ex_round2._contrasts`, asserted equal to the last digit).
natural      Whether any committed run answers the natural branch (no shift) of the test worlds in
             the four cells D0B0, D0B1_source, D1B0, D1B1_source at the checkpoints the shifted
             branches are probed at (fork + 1, 2, 4, 8 weeks). The audit lists every run with
             natural-branch HoldMem answers, the cells and times it covers, and how many answers
             are at a post-fork checkpoint. If there are none, the plan for the run that would
             supply them: the `experiments.e1` command, the probes it asks (no `--discriminative`:
             that flag selects by the natural branch's own drift and misses most of the item-times
             a shift changed), the writer requests the response cache of the shifted-branch runs
             cannot serve (a replay with a stand-in model, as `experiments.e1_estimate`) and an
             upper bound on reader calls and prompt tokens (cache hits across branches cannot be
             known without the stored outputs). With `--natural-run RUN` (a finished natural run)
             the counterfactual itself: per shifted branch, part (changed by the shift, drift
             only) and class, B x D in the shifted branch minus B x D on the same (world, item,
             checkpoint) in the natural branch, paired per item, 2,000 world-bootstrap draws,
             seed 996 (the estimator of `analysis.ex_e1_classes`).

The labels (changed by the shift or by drift only) are those of `ex_round2`: a changed item-time
is changed by the shift when its gold differs from the gold of the generator's natural branch at
the same time.

Usage: python -m memshift.analysis.ex_round3_natural [--stage runs/stage2_e1] [--out PATH]
           [--natural-run RUN_DIR] [--no-estimate] [--check-published] [--workers N]
`--check-published` also reruns `ex_round2._attribution` and asserts the counts and class values
equal it (adds about a minute).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_round2 as r2
from memshift.analysis.e1 import Cond, Row, _diff
from memshift.experiments import e1_estimate
from memshift.experiments.e1 import OFFSETS_WEEKS, discriminative_probes, parse_cell
from memshift.gen.shifts import BRANCHES as GEN_BRANCHES
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.llm import LLMResult
from memshift.probes import make_probes
from memshift.runinfo import REPO, code_state
from memshift.textutil import approx_tokens

STATUS = (
    "additional analysis: the shifted branches against the natural branch; changes no contrast, "
    "margin or threshold of the main analysis"
)
BRANCHES = ("src", "def", "orphan")  # the shifted branches of the main study
PARTS = ("shift", "drift")  # changed by the shift; the same change happens without it
CLASSES = ("definition", "other")
PLAN_CELLS = ("D0B0", "D0B1_source", "D1B0", "D1B1_source")
JSON_ARM = "holdmem_json_arm"  # `experiments.e1` asks it whenever D1B1_source is a cell
STAGE = "runs/stage2_e1"
OUT = "runs/stage6_ex_round3/natural.json"
RATE_TOKENS_PER_HOUR = (30e6, 60e6)  # prompt tokens per A100-hour: the slow and the fast reading


def gpu_hours(tokens: float) -> dict[str, float]:
    """A100-hours of prefill at each assumed rate (an extrapolation, not a measurement)."""
    return {f"at_{int(r / 1e6)}M_tokens_per_hour": tokens / r for r in RATE_TOKENS_PER_HOUR}


Vec = ex.Vec
Est = ex.Est
Key = r2.Key


# --------------------------------------------------------------------------------------- labels
@dataclass
class Labelled:
    """The changed item-times of a staged E1 analysis, each labelled "shift" or "drift"."""

    cause: dict[Key, str]
    times: dict[str, dict[int, str]]  # world -> checkpoint k -> as_of
    worlds: dict[str, tuple[int, dict[str, Any]]]  # world -> (seed, world params)
    runs: list[Path]


def label_item_times(stage: Path) -> Labelled:
    """`ex_round2._attribution`'s labelling of the changed item-times: the gold of the branch at
    the checkpoint differs from its gold at t0 (changed), and from the gold of the generator's
    natural branch at the same time (by the shift) or not (drift only). The generator must
    reproduce the committed gold where the shift cannot act."""
    runs = r2._source_runs(stage)
    if runs is None:
        raise SystemExit(f"{stage}: no manifest with sources")
    probes: dict[Key, Row] = {}
    times: dict[str, dict[int, str]] = defaultdict(dict)
    worlds: dict[str, tuple[int, dict[str, Any]]] = {}
    for run in runs:
        man = json.loads((run / "manifest.json").read_text())
        params = dict((man.get("params") or {}).get("world_params") or {})
        for w in man["worlds"]:
            worlds[w["world_id"]] = (int(w["seed"]), params)
        for p in r2._jsonl(run / "probes.jsonl"):
            k = int(p["tags"]["k"])
            probes[p["world_id"], p["branch"], p["fact_id"], k] = p
            if times[p["world_id"]].setdefault(k, p["as_of"]) != p["as_of"]:
                raise ValueError(f"{p['world_id']} k={k}: branches probe at different times")
    natural: dict[tuple[str, str, int], str] = {}
    for wid, (seed, params) in sorted(worlds.items()):
        for (fact, k), g in r2._natural_gold(seed, params, times[wid]).items():
            natural[wid, fact, k] = g
    for (w, b, f, k), p in probes.items():
        moved = p["kappa_version"] != probes[w, b, f, 0]["kappa_version"]
        if (k == 0 or (b == "def" and not moved)) and r2._gold(p) != natural[w, f, k]:
            raise ValueError(f"natural branch does not reproduce the committed gold: {w} {b} {f}")
    cause: dict[Key, str] = {}
    for (w, b, f, k), p in sorted(probes.items()):
        if k == 0 or r2._gold(p) == r2._gold(probes[w, b, f, 0]):
            continue
        cause[w, b, f, k] = "shift" if r2._gold(p) != natural[w, f, k] else "drift"
    return Labelled(cause, dict(times), worlds, runs)


# ---------------------------------------------------------------------------------- composition
def _quad(tab: ex.Table, branch: str, cond: Cond) -> dict[tuple[int, int], Vec]:
    """Contract validity of the four cells (D, B) at the source-epoch key, rows that meet `cond`."""
    return {
        (d, b): tab.vec(ex.cell_system("D1", d, b, "source"), branch, "cv", cond)
        for d in (0, 1)
        for b in (0, 1)
    }


def _double_diff(u: Mapping[tuple[int, int], Vec]) -> Vec:
    """Per item (B at D=1) - (B at D=0), over the items present in all four cells."""
    return _diff(_diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0]))


def _components(by_class: Mapping[str, Vec], worlds: Sequence[str]) -> np.ndarray:
    """Per world: sum and count of the double differences of the definition and other items."""
    rows = []
    for w in worlds:
        row: list[float] = []
        for cls in CLASSES:
            v = by_class[cls].get(w, {})
            row += [sum(v.values()), float(len(v))]
        rows.append(row)
    return np.array(rows, dtype=float)


def _est(arr: np.ndarray, stat: Callable[[np.ndarray], Any], n_items: float) -> Est:
    return ex.world_estimate(arr, stat, int(n_items))


def _mixed(w: float) -> Callable[[np.ndarray], Any]:
    """The all-item B x D at definition share `w`: w x definition mean + (1 - w) x other mean."""
    return lambda t: w * t[..., 0] / t[..., 1] + (1 - w) * t[..., 2] / t[..., 3]


def composition(
    tab: ex.Table, branch: str, cause: Mapping[Key, str], pooled_share: float | None = None
) -> dict[str, Any]:
    """Per part: item counts, class shares, class B x D, class contributions and the all-item
    B x D re-weighted to `pooled_share` (the definition share of the shift and drift item-times
    together; computed here when not given). Components: [sum, n] of the definition items, then of
    the other items; `all` = the pooled mean = the share-weighted sum of the class means."""
    per: dict[str, np.ndarray] = {}
    for part in PARTS:
        by_class = {
            cls: _double_diff(_quad(tab, branch, r2._part(cls, branch, cause, part)))
            for cls in (*CLASSES, "all")
        }
        worlds = sorted(w for w, d in by_class["all"].items() if d)
        per[part] = _components(by_class, worlds)
        if int(per[part][:, 1].sum() + per[part][:, 3].sum()) != sum(
            len(d) for d in by_class["all"].values()
        ):
            raise ValueError(f"{branch}/{part}: the classes do not partition the items")
    n_def = sum(float(per[p][:, 1].sum()) for p in PARTS)
    n_all = n_def + sum(float(per[p][:, 3].sum()) for p in PARTS)
    w_star = n_def / n_all if pooled_share is None else pooled_share
    all_item = r2._contrasts(tab, branch, ex._changed("all"))["BxD_interaction_update"]
    out: dict[str, Any] = {
        "pooled_definition_share": w_star,
        # the all-item B x D of `e1.contrasts` (shift and drift item-times together) and the
        # item-weighted mean of the two parts, which it must equal
        "all_item": all_item,
        "parts": {},
    }
    for part in PARTS:
        a = per[part]
        nd, no = float(a[:, 1].sum()), float(a[:, 3].sum())
        # a class's own estimate resamples the worlds that have items of the class, as the paper's
        a_def, a_oth = a[a[:, 1] > 0], a[a[:, 3] > 0]
        node: dict[str, Any] = {
            "n": {"definition": int(nd), "other": int(no), "all": int(nd + no)},
            "definition_share": _est(a, lambda t: t[..., 1] / (t[..., 1] + t[..., 3]), nd + no),
            "class": {
                "definition": _est(a_def, lambda t: t[..., 0] / t[..., 1], nd),
                "all": _est(
                    a, lambda t: (t[..., 0] + t[..., 2]) / (t[..., 1] + t[..., 3]), nd + no
                ),
            },
            "contribution": {
                "definition": _est(a, lambda t: t[..., 0] / (t[..., 1] + t[..., 3]), nd + no),
            },
        }
        if no:
            node["class"]["other"] = _est(a_oth, lambda t: t[..., 2] / t[..., 3], no)
            node["contribution"]["other"] = _est(
                a, lambda t: t[..., 2] / (t[..., 1] + t[..., 3]), nd + no
            )
            node["at_pooled_mix"] = _est(a, _mixed(w_star), nd + no)
        total = sum(c["est"] for c in node["contribution"].values())
        node["identity_residual"] = node["class"]["all"]["est"] - total
        out["parts"][part] = node
    parts = out["parts"]
    mean_of_parts = sum(parts[p]["n"]["all"] * parts[p]["class"]["all"]["est"] for p in PARTS)
    if abs(mean_of_parts / n_all - all_item["est"]) > 1e-12:
        raise ValueError(f"{branch}: the parts do not average to the all-item B x D")
    return out


def check_against_paper(tab: ex.Table, branch: str, cause: Mapping[Key, str], comp: Any) -> None:
    """The class and all-item B x D must equal `ex_round2._contrasts` (the table of the paper) to
    the last digit, interval ends included."""
    for part in PARTS:
        for cls in (*CLASSES, "all"):
            ref = r2._contrasts(tab, branch, r2._part(cls, branch, cause, part))
            ref_est = ref["BxD_interaction_update"]
            mine = comp["parts"][part]["class"].get(cls)
            if mine is None:
                if ref_est["n_items"]:
                    raise ValueError(f"{branch}/{part}/{cls}: class missing")
                continue
            for key in ("est", "lo", "hi", "n_items"):
                if abs(mine[key] - ref_est[key]) > 1e-12:
                    raise ValueError(f"{branch}/{part}/{cls}: {key} differs from ex_round2")


# -------------------------------------------------------------------------------- availability
def natural_availability(runs_dir: Path, times: Mapping[str, Mapping[int, str]]) -> dict[str, Any]:
    """Every run with natural-branch HoldMem answers, and the E1 runs' branches.
    `answers_at_postfork_checkpoints` counts the natural answers asked at fork + k weeks (k in 1,
    2, 4, 8) of their world."""
    checkpoints = {w: {t for k, t in ks.items() if k > 0} for w, ks in times.items()}
    e1_branches: set[str] = set()
    e1_runs = 0
    found: dict[tuple[str, tuple[str, ...], tuple[str, ...]], list[str]] = defaultdict(list)
    answers_at_checkpoint = 0
    for man in sorted(runs_dir.glob("*/manifest.json")):
        m = json.loads(man.read_text())
        branches = {w.get("branch") for w in m.get("worlds") or []}
        if m.get("experiment") == "E1":
            e1_runs += 1
            e1_branches |= {str(b) for b in branches | set(m["params"].get("branches") or [])}
        if "natural" not in branches or not (man.parent / "answers.jsonl").exists():
            continue
        cells: set[str] = set()
        asked: set[str] = set()
        hits = 0
        with (man.parent / "answers.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if '"natural"' not in line or "holdmem_D" not in line:
                    continue
                a = json.loads(line)
                if a["branch"] != "natural" or not str(a["system"]).startswith("holdmem_D"):
                    continue
                cells.add(a["system"])
                asked.add(a["as_of"])
                hits += a["as_of"] in checkpoints.get(a["world_id"], ())
        if cells:
            found[m["experiment"], tuple(sorted(cells)), tuple(sorted(asked))].append(
                man.parent.name
            )
            answers_at_checkpoint += hits
    return {
        "e1_runs": e1_runs,
        "e1_branches": sorted(e1_branches),
        "natural_holdmem_runs": [
            {"experiment": e, "cells": list(c), "as_of": list(t), "runs": r}
            for (e, c, t), r in sorted(found.items())
        ],
        "postfork_checkpoints": sorted({t for ts in checkpoints.values() for t in ts}),
        "answers_at_postfork_checkpoints": answers_at_checkpoint,
        "available": answers_at_checkpoint > 0,
    }


def def_branch_vs_natural(runs_dir: Path, lab: Labelled, tab: ex.Table) -> dict[str, Any]:
    """S_def moves the definitions' version at the fork and nothing else, so its event log is the
    natural branch's (equal `events_sha256` in the manifests of the 12-month E2 and E3 runs) and,
    on the non-definition items, its probes are what the natural branch would be probed with. It
    is not a natural run: the store holds the redefined definition items, which can change a
    reader's context, and it was probed on the changed item-times plus an equal-size sample. The
    count: item-times S_src and S_del change, of non-definition items, that S_def's probes cover in
    all four cells."""
    natural_sha: dict[str, set[str]] = defaultdict(set)
    for man in sorted(runs_dir.glob("e[23]_*/manifest.json")):
        for w in json.loads(man.read_text())["worlds"]:
            if w["branch"] == "natural" and w["world_id"] in lab.worlds:
                natural_sha[w["world_id"]].add(w["events_sha256"])
    equal = total = 0
    for run in lab.runs:
        for w in json.loads((run / "manifest.json").read_text())["worlds"]:
            if w["branch"] == "def":
                total += 1
                equal += natural_sha.get(w["world_id"]) == {w["events_sha256"]}
    asked: set[tuple[str, str, int]] | None = None
    for b01 in (0, 1):
        for d01 in (0, 1):
            system = ex.cell_system("D1", d01, b01, "source")
            cur = {
                (r["world_id"], r["fact_id"], r["k"])
                for r in tab.rows(system, "def")
                if r["cv"] is not None
            }
            asked = cur if asked is None else asked & cur
    assert asked is not None
    coverage: dict[str, Any] = {}
    for b in ("src", "orphan"):
        for part in PARTS:
            need = {
                (w, f, k)
                for (w, bb, f, k), c in lab.cause.items()
                if bb == b and c == part and ex.item_class(f) != "definition"
            }
            coverage[f"{b}_{part}_other"] = {"item_times": len(need), "covered": len(need & asked)}
    return {
        "def_worlds_with_events_identical_to_natural": equal,
        "def_worlds": total,
        "other_item_times_covered_by_def_probes": coverage,
    }


# ------------------------------------------------------------------------------------ run plan
class _SizedRecorder(e1_estimate.Recorder):
    """`e1_estimate.Recorder` that also keeps the approximate prompt tokens of each request."""

    def __init__(self, extra_body: dict[str, Any] | None = None):
        super().__init__(extra_body)
        self.tokens: dict[str, int] = {}

    def complete(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 512,
        schema: dict[str, Any] | None = None,
        purpose: str = "",
    ) -> LLMResult:
        res = super().complete(messages, max_tokens, schema, purpose)
        with self._lock:
            self.tokens[res.key] = sum(approx_tokens(m["content"]) for m in messages)
        return res


def _requests(seed: int, branch: str) -> tuple[set[str], dict[str, int], Any]:
    """The distinct writer and program requests of the four plan cells on one branch of a world
    (response-cache keys), their approximate prompt tokens, and the world."""
    world = generate_branch(seed, branch, None)
    log = SourceLog(world.events)
    cfgs = [parse_cell(c) for c in PLAN_CELLS]
    with mock.patch.object(e1_estimate, "Recorder", _SizedRecorder):
        replay = e1_estimate._replay(world, log, cfgs)
    keys: set[str] = set()
    tokens: dict[str, int] = {}
    for writer, _, _ in replay.values():
        assert isinstance(writer, _SizedRecorder)
        keys |= {k for k, _ in writer.requests}
        tokens |= writer.tokens
    return keys, tokens, (world, log)


def plan_world(seed: int) -> dict[str, Any]:
    """One world: the writer requests of the natural branch that no shifted branch sends, and the
    probes `--discriminative` would ask on the natural branch."""
    shifted: set[str] = set()
    for b in BRANCHES:
        shifted |= _requests(seed, b)[0]
    keys, tokens, (world, log) = _requests(seed, "natural")
    new = keys - shifted
    t0 = world.fork - timedelta(minutes=1)
    pre = make_probes(world, log, [t0])
    selected: list[list[Any]] = []
    for k in OFFSETS_WEEKS:
        probes = make_probes(world, log, [world.fork + timedelta(weeks=k)])
        for p in discriminative_probes(probes, pre, f"{world.world_id}|natural|{k}"):
            selected.append([p.fact_id, k])
    return {
        "world_id": world.world_id,
        "slots": len(world.slots),
        "writer_distinct": len(keys),
        "writer_new": len(new),
        "writer_new_tokens": sum(tokens[k] for k in new),
        "discriminative": selected,
    }


def reader_prompt_tokens(runs: Sequence[Path], systems: Sequence[str]) -> dict[str, Any]:
    """Mean prompt tokens of the reader calls after the fork, per system, in the published runs."""
    acc: dict[str, list[int]] = {s: [0, 0] for s in systems}
    for run in runs:
        with (run / "answers.jsonl").open(encoding="utf-8") as f:
            for line in f:
                if "holdmem_" not in line:
                    continue
                a = json.loads(line)
                if a["system"] in acc and int(a["tags"]["k"]) > 0:
                    acc[a["system"]][0] += a["prompt_tokens"]
                    acc[a["system"]][1] += 1
    return {
        s: {"calls": n, "mean_prompt_tokens": t / n if n else None} for s, (t, n) in acc.items()
    }


def writer_tokens_per_write(runs: Sequence[Path]) -> float:
    """Mean prompt and completion tokens of one writer call, from the t0 ingestion of the plan
    cells in the published runs (`HoldMem.cost`)."""
    total, count = 0.0, 0
    for run in runs:
        for line in (run / "world_summaries.jsonl").read_text().splitlines():
            for node in json.loads(line)["branches"].values():
                for c in PLAN_CELLS:
                    total += node["costs"][f"holdmem_{c}"]["ingest_tokens"]
                    count += 1
    return total / (count * 48)


def run_plan(lab: Labelled, workers: int, estimate: bool) -> dict[str, Any]:
    """The natural-branch run that would supply the missing answers, with its cost."""
    seeds = sorted(seed for seed, _ in lab.worlds.values())
    spec = f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else None
    cells = ",".join(PLAN_CELLS)
    args = f"--branches natural --no-baselines --cells {cells} --name e1_natural"
    plan: dict[str, Any] = {
        "natural_is_a_generator_branch": "natural" in GEN_BRANCHES,
        "command": (
            f"python -m memshift.experiments.e1 --seeds {spec} {args}"
            if spec
            else f"python -m memshift.experiments.e1 --seeds {','.join(map(str, seeds))} {args}"
        ),
        "runner": (
            "RUN_MODULE=memshift.experiments.e1 scripts/run.sh --seeds 1000-1003 ... "
            "(chunks of four worlds, a distinct --name per chunk, as for the control runs)"
        ),
        "environment": "MEMSHIFT_ENDPOINTS: the main model only (writer and reader), no embedding",
        "no_discriminative_flag": (
            "`--discriminative` keeps the natural branch's own drift-changed item-times plus an "
            "equal-size sample of its unchanged ones, so it misses most item-times that only a "
            "shift changed; without it every slot is probed at each checkpoint"
        ),
        "checkpoints_weeks": list(OFFSETS_WEEKS),
        "cells": [f"holdmem_{c}" for c in PLAN_CELLS],
        "reader_systems": [f"holdmem_{c}" for c in PLAN_CELLS] + [JSON_ARM],
        "pre_shift": "k = 0 reads and writes are the published runs' (same t0 world): cache hits",
    }
    needed: dict[str, set[tuple[str, str, int]]] = defaultdict(set)
    for (w, b, f, k), _ in lab.cause.items():
        needed[b].add((w, f, k))
    union = set().union(*needed.values())
    plan["needed_item_times"] = {
        **{b: len(needed[b]) for b in BRANCHES},
        "union": len(union),
        "worlds": len(lab.worlds),
    }
    if not estimate:
        return plan
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(seeds)))) as pool:
        rows = list(pool.map(plan_world, seeds))
    slots = {r["slots"] for r in rows}
    if len(slots) != 1:
        raise ValueError(f"worlds differ in slots: {sorted(slots)}")
    n_slots = slots.pop()
    per_cell = n_slots * len(OFFSETS_WEEKS)
    systems = plan["reader_systems"]
    tok = reader_prompt_tokens(lab.runs, systems)
    mean_tok = {s: tok[s]["mean_prompt_tokens"] for s in systems}
    reader_calls = len(rows) * per_cell * len(systems)
    reader_tokens = len(rows) * per_cell * sum(mean_tok.values())
    union_calls = len(union) * len(systems)
    union_tokens = len(union) * sum(mean_tok.values())
    covered = sum(1 for r in rows for f, k in r["discriminative"] if (r["world_id"], f, k) in union)
    write_tok = writer_tokens_per_write(lab.runs)
    writer_new = sum(r["writer_new"] for r in rows)
    writer_tokens = sum(r["writer_new_tokens"] for r in rows)
    plan["estimate"] = {
        "kind": (
            "replay with a stand-in model (no model call); an upper bound on reader calls: "
            "cache hits across branches need the stored outputs"
        ),
        "slots_per_world": n_slots,
        "asked_per_system_world": per_cell,
        "reader_calls_upper": reader_calls,
        "reader_calls_per_world": len(systems) * per_cell,
        "mean_prompt_tokens": mean_tok,
        "reader_prompt_tokens_upper": reader_tokens,
        "gpu_hours_reader": gpu_hours(reader_tokens),
        "writer_requests_natural_distinct": sum(r["writer_distinct"] for r in rows),
        "writer_requests_new": writer_new,
        "writer_new_approx_prompt_tokens": writer_tokens,
        "writer_mean_tokens_per_write_published": write_tok,
        "writer_new_tokens_at_published_mean": writer_new * write_tok,
        "writer_new_per_world": {r["world_id"]: r["writer_new"] for r in rows},
        "gpu_hours_writer_new": gpu_hours(writer_new * write_tok),
        "cold_cache_extra": {
            "note": "added if the response cache of the published runs is not on the host: every "
            "distinct writer request, and the k = 0 reads, are then made again",
            "writer_requests": sum(r["writer_distinct"] for r in rows),
            "writer_prompt_tokens": sum(r["writer_distinct"] for r in rows) * write_tok,
            "k0_reader_calls": len(rows) * n_slots * len(systems),
            "k0_reader_prompt_tokens": len(rows) * n_slots * sum(mean_tok.values()),
        },
        "needed_subset": {
            "note": "only the item-times a shifted branch changes; needs a probe-key filter that "
            "`experiments.e1` does not have (not run, not changed)",
            "reader_calls_upper": union_calls,
            "reader_prompt_tokens_upper": union_tokens,
            "gpu_hours_reader": gpu_hours(union_tokens),
        },
        "discriminative_flag": {
            "asked_item_times": sum(len(r["discriminative"]) for r in rows),
            "needed_item_times_covered": covered,
            "needed_item_times": len(union),
            "coverage": covered / len(union) if union else None,
        },
    }
    return plan


# ------------------------------------------------------------------------ natural counterfactual
def read_natural_cv(run: Path) -> dict[str, Vec]:
    """System -> world -> "fact@k" -> contract validity of the natural-branch answers of a run
    after the fork (the keys of `ex.Table.vec`)."""
    out: dict[str, Vec] = defaultdict(lambda: defaultdict(dict))
    with (run / "answers.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if '"natural"' not in line:
                continue
            a = json.loads(line)
            k = int((a.get("tags") or {}).get("k", 0))
            if a["branch"] == "natural" and k > 0:
                out[a["system"]][a["world_id"]][f"{a['fact_id']}@{k}"] = float(a["contract_valid"])
    return out


def paired_vs_natural(
    tab: ex.Table,
    nat: Mapping[str, Vec],
    branch: str,
    cause: Mapping[Key, str],
    part: str,
    cls: str,
) -> dict[str, Any]:
    """B x D (contract validity, source-epoch key, D=1 authored) on the item-times of one part and
    class in the shifted branch, on the same (world, item, checkpoint) in the natural branch, and
    their paired difference, over the item-times both have in all four cells."""
    xb = _double_diff(_quad(tab, branch, r2._part(cls, branch, cause, part)))
    xn = _double_diff(
        {(d, b): nat.get(ex.cell_system("D1", d, b, "source"), {}) for d in (0, 1) for b in (0, 1)}
    )
    both = {w: sorted(xb[w].keys() & xn.get(w, {}).keys()) for w in xb}
    n_branch = sum(len(v) for v in xb.values())
    n_both = sum(len(v) for v in both.values())

    def est(f: Callable[[str, str], float]) -> Est:
        return ex.mean_estimate({w: [f(w, i) for i in ids] for w, ids in both.items() if ids})

    return {
        "item_times_in_branch": n_branch,
        "item_times_with_natural": n_both,
        "shifted_branch": est(lambda w, i: xb[w][i]),
        "natural": est(lambda w, i: xn[w][i]),
        "difference": est(lambda w, i: xb[w][i] - xn[w][i]),
    }


def natural_counterfactual(tab: ex.Table, lab: Labelled, run: Path) -> dict[str, Any]:
    nat = read_natural_cv(run)
    return {
        "natural_run": run.name,
        "branches": {
            b: {
                part: {
                    cls: paired_vs_natural(tab, nat, b, lab.cause, part, cls)
                    for cls in (*CLASSES, "all")
                }
                for part in PARTS
            }
            for b in BRANCHES
        },
    }


# ------------------------------------------------------------------------------------------ main
def check_counts(lab: Labelled, root: Path, stage: str) -> None:
    """The labels equal `ex_round2._attribution`'s counts (slow: it labels the item-times again)."""
    attr = r2._attribution(str(root.resolve()), stage)
    assert attr is not None
    for b in BRANCHES:
        classes = attr["branches"][b]["classes"]
        for part in PARTS:
            for cls in CLASSES:
                mine = sum(
                    1
                    for (_, bb, f, _), c in lab.cause.items()
                    if bb == b
                    and c == part
                    and (ex.item_class(f) == "definition") == (cls == "definition")
                )
                if mine != classes.get(cls, {}).get(part, 0):
                    raise ValueError(f"{b}/{part}/{cls}: {mine} item-times differ from ex_round2")


def run(
    root: Path,
    stage: str,
    out: Path,
    natural_run: Path | None,
    estimate: bool,
    check_published: bool,
    workers: int,
) -> dict[str, Any]:
    lab = label_item_times(root / stage)
    tab = ex.Table(r2._jsonl(root / stage / "analysis" / "e1_item_table.jsonl"))
    if check_published:
        check_counts(lab, root, stage)
    comp: dict[str, Any] = {}
    for b in BRANCHES:
        comp[b] = composition(tab, b, lab.cause)
        check_against_paper(tab, b, lab.cause, comp[b])
    avail = natural_availability(root / "runs", lab.times)
    result: dict[str, Any] = {
        "status": STATUS,
        "question": "incremental B x D attributable to the shift, relative to the natural branch",
        "stage": stage,
        "sources": [r.name for r in lab.runs],
        "n_worlds": len(lab.worlds),
        "units": "fractions (multiply by 100 for percentage points); 95% world-bootstrap "
        "intervals, 2,000 draws, seed 996",
        "composition": comp,
        "natural_availability": avail,
        "def_branch_as_natural": def_branch_vs_natural(root / "runs", lab, tab),
        "code": code_state(),
    }
    if natural_run is not None:
        result["natural_counterfactual"] = natural_counterfactual(tab, lab, natural_run)
    else:
        result["natural_counterfactual"] = None
        if avail["available"]:
            result["note"] = "natural answers at the checkpoints exist: pass --natural-run RUN"
        else:
            result["run_plan"] = run_plan(lab, workers, estimate)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1, default=str) + "\n")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--root", default=str(REPO))
    ap.add_argument("--stage", default=STAGE, help="the merged main-study E1 stage")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--natural-run", default="", help="a finished natural-branch E1 run directory")
    ap.add_argument("--no-estimate", action="store_true", help="skip the replay of the run plan")
    ap.add_argument("--check-published", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    root = Path(a.root)
    res = run(
        root,
        a.stage,
        root / a.out,
        Path(a.natural_run) if a.natural_run else None,
        not a.no_estimate,
        a.check_published,
        a.workers,
    )
    print(
        root / a.out,
        "natural answers at the post-fork checkpoints:",
        res["natural_availability"]["answers_at_postfork_checkpoints"],
    )


if __name__ == "__main__":
    main()

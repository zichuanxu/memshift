"""Binding against the add-only refresh, B x D by the cause of a change, clock-only changes, forced
re-derivation, evidence coverage, program input sizes and the estimates the design fixes. They
read committed run outputs, the seeded world generator and, for the token counts, the Qwen3.5-9B
tokenizer offline; they add no run and call no model. The stage is `ex_round3_review` in
results/sources.json (runs/stage8_ex_round3_review/analysis/, seven files; the stage manifest lists
the producers and the source runs; `python -m memshift.analysis.ex_round3_review --tokenizer
TOKENIZER_JSON` reruns them all, about 30 minutes):

  refresh.json   binding against the add-only refresh (`B0_refresh`, the block U runs, which
                 re-derives every slot whose source-epoch key moved and keeps the old item) at D=0,
                 the authored D=1 and D_m, on the changed definition, other and all item-times of
                 each shift: Update of B=1 (source-epoch key) minus the refresh, paired per
                 item-time; B x D with the refresh as the B arm; B x D with B=1 minus that with the
                 refresh. The levels and the two gains over B=0 are those of block U (tab_refresh_
                 upsert); the B=1 B x D is the class split's
  cause.json     B x D on Update with the authored D=1 and with D_m (block C), on the changed
                 item-times changed by the shift, by drift only (the labels of `ex_round2`, via
                 `ex_round3_natural.label_item_times`) and both (pooled), per shift and class; and
                 the item-times the cited-cluster key did not invalidate (D1B1_cited) by the same
                 split. The authored-D=1 parts are tab_redef_only's, the pooled values the class
                 split's
  clock.json     changes that come from the clock alone. Per changed item-time (gold differs from
                 t0), its last change is the latest interval between consecutive checkpoints (t0,
                 fork + 1, 2, 4, 8 weeks) in which its gold moved; that change is record-driven
                 when a record of the gold dependencies was revised, deleted or created or the
                 definition version moved, else clock-only (an overdue status, a window that
                 expires). A clock-only change is caught when a record of the item's declared
                 sources arrived in the interval anyway (the source-epoch key counts events per
                 source, so it moved for an unrelated reason), else silent (the key cannot see it).
                 Counts per item-time and per interval (the generator's gold, which must reproduce
                 every committed change flag, on worlds whose events hash to the committed
                 manifests); HoldMem(1,1)'s wrong answers on changed definition item-times and those
                 that are silent clock changes the key did not invalidate after the change; m and r
                 of Eq. (1) (`ex_e1_classes.decomposition`) with detection redefined relative to
                 the last change (an invalidation after the interval's start, up to the
                 checkpoint), beside detection since t0; main study and replication
  forced.json    forced re-derivation on the cited-cluster key's misses: on the changed item-times
                 that D{0,1}B1_cited did not invalidate, its Update against that of
                 D{0,1}B1_source, which re-derived the same item-times with the same writer request
                 (what a forced re-derivation would serve), and of B=0
  coverage.json  evidence coverage of the D=0 writer. Per definition item-time after the shift
                 (20 test worlds), the records the definition ranges over (`gen.kappa.
                 scope_members`) and how many of them the writer's 6,000-token BM25 selection
                 (`holdmem.writer.records_for`) holds (complete: all). r0 and r1 (r of Eq. (1) in
                 D0B1_source and D1B1_source) on changed definition item-times and B x D on
                 definition items split by complete or incomplete scope, for Qwen3.5-9B (20
                 worlds, and worlds 1000-1009), Claude Sonnet 5.5 and Claude Opus 5.5 (the block H
                 runs, worlds 1000-1009). The full-access control (D0fullB1_source, the control
                 stage's e1_full runs): its re-derivations per definition, the share truncated at
                 its 25,000-token limit (from the runs' notes) and the share whose writer request is
                 the D=0 request (`holdmem.writer.writer_request` on both record sets)
  sizes.json     the complete typed records a D_m program receives (`holdmem.program.
                 program_records`) per definition item-time after the shift, in Qwen3.5-9B tokens
                 (the model's tokenizer.json, read offline; its SHA256 is recorded), each record
                 serialised as compact JSON and counted alone, plus one token per separator and two
                 for the brackets: the dicts without the rendered `text` (Dicts), the same without
                 the bodies of e-mail and messages (DictsNoBody), and one table per source and kind
                 with the column names once and each record as an array, without those bodies
                 (ColumnsNoBody). Median, maximum and the share that fits 31,000 tokens (a
                 32,768-token context less the request text and the answer), per definition, for
                 the six main definitions (20 worlds) and the 18 held-out ones (worlds 1000-1009).
                 It needs the `tokenizers` package, which is not a project dependency (`uv run
                 --with tokenizers python -m ...`)
  fixed.json     the estimates that the design fixes: the source-epoch key does not depend on
                 D, so every estimate of executability's effect on missed invalidation and C4 are
                 exactly 0, and an import without the keys misses every change by definition
                 (+100 pp); each row has staged estimates, and every one has that value with a
                 zero-width interval

Estimators: those of `ex_e1_classes` and `round3_blocks` (world-clustered bootstrap, 2,000
draws, seed 996; per-item paired differences over the item-times present in every cell used).

Macros (letters only, prefix ExRev; a stage that is not staged renders as the MISSING marker, a
quantity over no item-time as a dash):
  ExRevRefresh<arm><branch><class>...      arm Dzero, Done, Dm: Items, BoneMinusRefresh; for Done
                                           and Dm also BxDRefresh and BxDBoneMinusRefresh
  ExRevCause<arm><branch><part><class>...  arm Main (authored D=1) or Dm; part Pooled, Shift,
                                           Drift: Items, BxD
  ExRevCited<branch><part><class>...       Items, Missed, MissShare (D1B1_cited)
  ExRevClock<study><group><class>...       study Main or Rep, group a branch or Shifts (the three
                                           together): Changed, ClockOnly, ClockOnlyShare,
                                           ClockCaught, ClockSilent (item-times); per interval
                                           DefinitionChanges, DefinitionChangesClockOnly,
                                           DefinitionChangesClockSilent; ErrorsOneOne,
                                           ErrorsOneOneClock, ErrorsOneOneItems (HoldMem(1,1) on
                                           changed definition item-times)
  ExRevClock<study><branch><class><cell>MissedLast / RepairLast   cell DzeroBone, DoneBone
  ExRevForced<D><branch><class>...         D Dzero or Done: Items, Cited, Forced, Bzero, Gain
  ExRevCov<definition>...                  ScopeShare, CompleteShare, ItemTimes
  ExRevCov<writer><branch><subset>...      writer Qwen, QwenTen, Sonnet, Opus; subset Complete or
                                           Incomplete: RZero, ROne, ItemsZero, ItemsOne, DoneBxD,
                                           DoneItems, DmBxD, DmItems
  ExRevFull<definition or All>...          Rederived, TruncatedShare, SameRequestShare
  ExRevSize<set><definition><variant>...   set Main or Heldout (definition tag with the draw:
                                           One, Two, Three); variant Dicts, DictsNoBody,
                                           ColumnsNoBody: Median, Max, Fit; ...ItemTimes; and
                                           <definition> All for the set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_round2 as r2
from memshift.analysis import round3_blocks as rr
from memshift.analysis import round3_paper as r3p
from memshift.analysis.e1 import Cond, Row, _diff
from memshift.analysis.ex_e1_paper import UNDEFINED, _Out, fmt
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, camel, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.experiments.common import MAIN_MODEL, world_params
from memshift.experiments.e1 import OFFSETS_WEEKS
from memshift.runinfo import code_state, file_sha256, world_record

STATUS = (
    "additional analysis: reads committed run outputs and the seeded generator only, adds no run "
    "and calls no model"
)
UNITS = "fractions (x 100 for percent or pp); 95% world-clustered bootstrap, 2,000 draws, seed 996"
STAGE_KEY = "ex_round3_review"
STAGE_DIR = "runs/stage8_ex_round3_review"
FILES = {
    "refresh": "refresh.json",
    "cause": "cause.json",
    "clock": "clock.json",
    "forced": "forced.json",
    "coverage": "coverage.json",
    "sizes": "sizes.json",
    "fixed": "fixed.json",
}
TOKENIZER_ARG = "QWEN3_5_9B_TOKENIZER_JSON"  # the producer's placeholder for a local file
PRODUCERS = {
    f"analysis/{name}": f"python -m memshift.analysis.ex_round3_review --part {part}"
    + (f" --tokenizer {TOKENIZER_ARG}" if part == "sizes" else "")
    for part, name in FILES.items()
}
NOTE = (
    "additional analysis, computed from the raw runs of the source stages, the block U refresh "
    "runs, the block H runs, the staged analyses the fixed rows read and the seeded generator; no "
    "model call. sizes.json counts tokens with the Qwen3.5-9B tokenizer.json read offline (its "
    "SHA256 is in the file)"
)
# the stages the analyses read (results/sources.json keys); then the raw runs outside them
SOURCE_KEYS = ("e1", "c", "rep_e1", "ctl", "gate", "l", "e6", "m2_gemma", "m2_llama", "m2_gptoss")
REFRESH_RUNS = tuple(r for r in r3p.RUNS["u"] if r.startswith("runs/e1_u_refresh_"))
H_RUNS = {h.key.removeprefix("h_"): h.runs for h in r3p.H_WRITERS}  # sonnet, opus
BRANCHES = rr.BRANCHES
CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))
K_ORDER = (0, *OFFSETS_WEEKS)  # the probe checkpoints: t0, then weeks after the fork
DEFINITIONS = tuple((f.removeprefix("k."), tag) for f, tag, _ in r2.DEFINITIONS)
Vec = ex.Vec
Est = ex.Est


def source_runs(sources: Mapping[str, str]) -> list[str]:
    """The stage's sources: the stages of `SOURCE_KEYS`, then the refresh and block H runs."""
    stages = [sources[k] for k in SOURCE_KEYS]
    return list(dict.fromkeys([*stages, *REFRESH_RUNS, *(r for rs in H_RUNS.values() for r in rs)]))


def _mean(v: Vec) -> Est:
    return ex.vec_mean(v)


def _paired(x: Vec, y: Vec) -> Est:
    return ex.paired_estimate(x, y)


def _n(v: Mapping[str, Mapping[str, float]]) -> int:
    return sum(len(d) for d in v.values())


def _rows_cond(*conds: Cond) -> Cond:
    return lambda r: all(c(r) for c in conds)


def _in_class(cls: str) -> Cond:
    return lambda r: ex.in_class(r, cls)


def _changed_asked(cls: str) -> Cond:
    """Changed item-times of a class with a ContractValid answer (the rows Eq. (1) counts)."""
    return lambda r: bool(r["changed"]) and r.get("cv") is not None and ex.in_class(r, cls)


# ----------------------------------------------------------------------------------- stages
@dataclass
class _Stage:
    """The probes, checkpoint times and worlds of the raw runs a merged stage lists."""

    runs: list[Path]
    probes: dict[r2.Key, Row]
    times: dict[str, dict[int, str]]  # world -> k -> as_of
    worlds: dict[str, tuple[int, dict[str, Any]]]  # world -> (seed, world params)
    hashes: dict[tuple[str, str], str]  # (world, branch) -> events_sha256 of the run manifest


def _read_stage(stage: Path) -> _Stage:
    runs = r2._source_runs(stage)
    if runs is None:
        raise SystemExit(f"{stage}: no manifest with sources")
    probes: dict[r2.Key, Row] = {}
    times: dict[str, dict[int, str]] = defaultdict(dict)
    worlds: dict[str, tuple[int, dict[str, Any]]] = {}
    hashes: dict[tuple[str, str], str] = {}
    for run in runs:
        man = json.loads((run / "manifest.json").read_text())
        params = dict((man.get("params") or {}).get("world_params") or {})
        for w in man["worlds"]:
            worlds[w["world_id"]] = (int(w["seed"]), params)
            hashes[w["world_id"], w["branch"]] = str(w["events_sha256"])
        for p in r2._jsonl(run / "probes.jsonl"):
            k = int(p["tags"]["k"])
            probes[p["world_id"], p["branch"], p["fact_id"], k] = p
            if times[p["world_id"]].setdefault(k, p["as_of"]) != p["as_of"]:
                raise ValueError(f"{p['world_id']} k={k}: branches probe at different times")
    return _Stage(runs, probes, dict(times), worlds, hashes)


def _world(seed: int, branch: str, params: Mapping[str, Any] | None, want: str | None) -> Any:
    """The generator's branch of a test world; `want`: the committed events SHA256 it must have."""
    from memshift.gen.shifts import generate_branch

    world = generate_branch(seed, branch, dict(params or {}) or None)
    if want is not None and world_record(world)["events_sha256"] != want:
        raise ValueError(f"seed {seed} {branch}: the generator does not reproduce the run's world")
    return world


def _parse(ts: str) -> datetime:
    from memshift.gen.model import parse_ts

    return parse_ts(ts)


# ================================================================================== refresh
def refresh(root: Path) -> dict[str, Any]:
    """Binding against the add-only refresh per shift, class and D arm (see the docstring)."""
    new, skipped = rr.read_runs([root / r for r in REFRESH_RUNS])
    need = set(rr.ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = rr.read_runs(
        rr.expand([root / c for c in r3p.COMPARE]), need.__contains__
    )
    if skipped or skipped_c:
        raise SystemExit(f"incomplete runs: {[*skipped, *skipped_c]}")
    newm, cmpm = rr.Merged(new, rr.by_family), rr.Merged(cmp_runs, rr.by_family)
    (fam,) = [f for f in newm.groups() if rr._family_label(f) == "main"]
    tn, tc = newm.table(fam), cmpm.table(fam)
    results: dict[str, Any] = {}
    for b in BRANCHES:
        for cls, _ in CLASSES:
            cond = rr.changed(cls)
            vec = {
                arm: {
                    "B0": tc.vec(lo, b, "cv", cond),
                    "B1": tc.vec(hi, b, "cv", cond),
                    "refresh": tn.vec(rr.u_cell(arm, "refresh"), b, "cv", cond),
                }
                for arm, (lo, hi) in rr.ARMS.items()
            }
            node: dict[str, Any] = {}
            for arm, v in vec.items():
                keys = rr._common(v["B0"], v["refresh"], v["B1"])
                z, x, o = (rr._restrict(v[k], keys) for k in ("B0", "refresh", "B1"))
                q: dict[str, Any] = {
                    "n_items": rr._n_items(keys),
                    "update": {"B0": _mean(z), "refresh": _mean(x), "B1": _mean(o)},
                    "B1_minus_B0": _paired(o, z),
                    "refresh_minus_B0": _paired(x, z),
                    "B1_minus_refresh": _paired(o, x),
                }
                if arm != "D0":
                    d0 = vec["D0"]
                    q["BxD_B1"] = _mean(rr.double_diff(v["B1"], v["B0"], d0["B1"], d0["B0"]))
                    q["BxD_refresh"] = _mean(
                        rr.double_diff(v["refresh"], v["B0"], d0["refresh"], d0["B0"])
                    )
                    q["BxD_B1_minus_refresh"] = _mean(
                        rr.double_diff(v["B1"], v["refresh"], d0["B1"], d0["refresh"])
                    )
                node[arm] = q
            results.setdefault(b, {})[cls] = node
    return {
        "status": STATUS,
        "units": UNITS,
        "runs": list(REFRESH_RUNS),
        "compare": list(r3p.COMPARE),
        "cells": {arm: [*cells, rr.u_cell(arm, "refresh")] for arm, cells in rr.ARMS.items()},
        "worlds": sorted(set(tn.worlds) & set(tc.worlds)),
        "results": results,
        "code": code_state(),
    }


# ==================================================================================== cause
CAUSE_ARMS = (("D1", "e1"), ("D1s", "c"))  # the D=1 arm and the stage whose item table has it
PARTS = ("pooled", "shift", "drift")
CITED_CELLS = ("holdmem_D0B1_cited", "holdmem_D1B1_cited")


def arm_contrasts(tab: ex.Table, branch: str, cond: Cond, arm: str = "D1") -> dict[str, Est]:
    """`ex_round2._contrasts` (the estimator of tab_redef_only) with the D=1 arm as a parameter:
    B effect at D=0 and D=1 and B x D on Update, source-epoch key."""
    u = {
        (d, b): tab.vec(ex.cell_system(arm, d, b, "source"), branch, "cv", cond)
        for d in (0, 1)
        for b in (0, 1)
    }
    return {
        "B_effect_update_D0": ex.paired_estimate(u[0, 1], u[0, 0]),
        "B_effect_update_D1": ex.paired_estimate(u[1, 1], u[1, 0]),
        "BxD_interaction_update": ex.paired_estimate(
            _diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0])
        ),
    }


def _part_cond(part: str, cls: str, branch: str, cause: Mapping[r2.Key, str]) -> Cond:
    if part == "pooled":
        return lambda r: bool(r["changed"]) and ex.in_class(r, cls)
    return r2._part(cls, branch, cause, part)


def _check_labels(tab: ex.Table, branch: str, cause: Mapping[r2.Key, str], name: str) -> None:
    """The item table and the probes must agree on the changed item-times (as `ex_round2`)."""
    changed = {
        (r["world_id"], r["fact_id"], r["k"])
        for r in tab.rows("holdmem_D0B0", branch)
        if r["changed"]
    }
    if changed != {(w, f, k) for (w, b, f, k) in cause if b == branch}:
        raise ValueError(f"{name} {branch}: the item table and the probes disagree")


def cause(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """B x D by cause for the authored D=1 and D_m; the cited-cluster key's misses by cause."""
    from memshift.analysis import ex_round3_natural as nat

    lab = nat.label_item_times(root / sources["e1"])
    tables = {
        arm: ex.Table(r2._jsonl(root / sources[key] / "analysis" / "e1_item_table.jsonl"))
        for arm, key in CAUSE_ARMS
    }
    contrasts: dict[str, Any] = {}
    for arm, tab in tables.items():
        for b in BRANCHES:
            _check_labels(tab, b, lab.cause, arm)
            for part in PARTS:
                for cls, _ in CLASSES:
                    cond = _part_cond(part, cls, b, lab.cause)
                    n = sum(1 for r in tab.rows("holdmem_D0B0", b) if cond(r))
                    node = {"n": n, **arm_contrasts(tab, b, cond, arm)}
                    contrasts.setdefault(arm, {}).setdefault(b, {}).setdefault(part, {})[cls] = node
    tab = tables["D1"]
    cited: dict[str, Any] = {}
    for b in BRANCHES:
        for part in PARTS:
            for cls, _ in CLASSES:
                cond = _rows_cond(
                    _part_cond(part, cls, b, lab.cause), lambda r: r["cv"] is not None
                )
                flags = [tab.vec(c, b, "detected", cond) for c in CITED_CELLS]
                miss = {w: {k: 1.0 - x for k, x in d.items()} for w, d in flags[1].items()}
                cited.setdefault(b, {}).setdefault(part, {})[cls] = {
                    "n": _n(miss),
                    "missed": int(round(sum(sum(d.values()) for d in miss.values()))),
                    "miss_share": _mean(miss),
                    "same_at_D0": flags[0] == flags[1],  # the key's detection does not use D
                }
    return {
        "status": STATUS,
        "units": UNITS,
        "stages": {arm: sources[key] for arm, key in CAUSE_ARMS},
        "labels": "ex_round3_natural.label_item_times on the probes of the e1 stage's runs",
        "n_worlds": len(lab.worlds),
        "contrasts": contrasts,
        "cited": cited,
        "cited_cell": CITED_CELLS[1],
        "code": code_state(),
    }


# ==================================================================================== clock
STUDIES = (("main", "e1"), ("rep", "rep_e1"))
LABELS = ("record", "clock_caught", "clock_silent")
ONE_ONE = "holdmem_D1B1_source"


def _audit_changes(st: _Stage) -> dict[tuple[str, str, str, int], str]:
    """(world, branch, fact, i) -> the cause of the gold change between checkpoints i - 1 and i
    (`LABELS`), for every interval in which the gold moved. The generator's gold must move on
    exactly the intervals where the committed probes' gold moves."""
    from memshift.gen.gold import gold_at
    from memshift.gen.state import SourceLog

    out: dict[tuple[str, str, str, int], str] = {}
    for w, (seed, params) in sorted(st.worlds.items()):
        ts = [_parse(st.times[w][k]) for k in K_ORDER]
        for b in BRANCHES:
            world = _world(seed, b, params, st.hashes.get((w, b)))
            log = SourceLog(world.events)
            snaps = [log.snapshot(t) for t in ts]
            for slot in world.slots:
                golds = [gold_at(world, log, slot, t) for t in ts]
                pr = [r2._gold(st.probes[w, b, slot.slot_id, k]) for k in K_ORDER]
                for i in range(1, len(K_ORDER)):
                    a, c = golds[i - 1], golds[i]
                    moved = (a.value, a.status) != (c.value, c.status)
                    if moved != (pr[i] != pr[i - 1]):
                        raise ValueError(f"{w} {b} {slot.slot_id} k={K_ORDER[i]}: gold differs")
                    if not moved:
                        continue
                    deps_a, deps_c = dict(a.deps), dict(c.deps)
                    touched = any(
                        (s := snaps[i].get(o)) is None or s.revision != rev
                        for o, rev in deps_a.items()
                    )
                    created = any(snaps[i - 1].get(o) is None for o in deps_c)
                    kv = bool(slot.kappa_id) and world.kappa_version(
                        slot.kappa_id, ts[i - 1]
                    ) != world.kappa_version(slot.kappa_id, ts[i])
                    event = any(e.source in slot.sources for e in log.between(ts[i - 1], ts[i]))
                    if touched or created or kv:
                        out[w, b, slot.slot_id, i] = "record"
                    else:
                        out[w, b, slot.slot_id, i] = "clock_caught" if event else "clock_silent"
    return out


def _last_change(st: _Stage) -> dict[r2.Key, int]:
    """(world, branch, fact, k) of each changed item-time -> the index i of the last interval
    (checkpoints i - 1, i) in which its gold moved."""
    out: dict[r2.Key, int] = {}
    for (w, b, f, k), p in st.probes.items():
        if k == 0 or r2._gold(p) == r2._gold(st.probes[w, b, f, 0]):
            continue
        pr = [r2._gold(st.probes[w, b, f, kk]) for kk in K_ORDER]
        i = K_ORDER.index(k)
        out[w, b, f, k] = max(j for j in range(1, i + 1) if pr[j] != pr[j - 1])
    return out


def _history(data: Mapping[str, list[Row]]) -> dict[tuple[str, str, str, str], list[Row]]:
    hist: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for it in data["items"]:
        hist[(it["world_id"], it["branch"], it["system"], it["slot_id"])].extend(it["history"])
    return hist


def _detected_last(
    rows: Sequence[Row],
    data: Mapping[str, list[Row]],
    st: _Stage,
    last: Mapping[r2.Key, int],
) -> list[Row]:
    """The item table with `detected` of every changed row redefined: the cell invalidated the
    item after the start of the interval in which its gold last moved, up to the checkpoint."""
    hist = _history(data)
    out = []
    for r in rows:
        key = (r["world_id"], r["branch"], r["fact_id"], int(r["k"]))
        events = hist.get((r["world_id"], r["branch"], r["system"], r["fact_id"]))
        if not r["changed"] or events is None:
            out.append(r)
            continue
        since = st.times[r["world_id"]][K_ORDER[last[key] - 1]]
        upto = st.times[r["world_id"]][int(r["k"])]
        kinds = e1.detection_events(r["system"])
        hit = any(h["event"] in kinds and since < h["ts"] <= upto for h in events)
        out.append({**r, "detected": hit})
    return out


def _terms(dec: Mapping[str, Any], b: str, cls: str, cell: str) -> dict[str, Any]:
    c = _get(dec, "D1", b, cls, "cells", "source", cell) or {}
    return {k: c.get(k) for k in ("n", "miss", "m", "r")}


def _indicator_share(flags: Mapping[str, list[float]]) -> Est:
    return ex.mean_estimate(flags)


def clock(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Clock-only changes, HoldMem(1,1)'s staleness errors and m, r with detection relative to
    the last change (see the docstring), main study and replication."""
    res: dict[str, Any] = {"status": STATUS, "units": UNITS, "stages": {}}
    for study, stage_key in STUDIES:
        st = _read_stage(root / sources[stage_key])
        audit = _audit_changes(st)
        last = _last_change(st)
        data = e1.load(*st.runs)
        rows = e1.item_table(data)
        rows_last = _detected_last(rows, data, st, last)
        # per item-time and per interval, by branch (and the three together) and class
        item_flags: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(dict)
        counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for (w, b, f, _k), i in sorted(last.items()):
            label = audit[w, b, f, i]
            for g in (b, "shifts"):
                for cls, _ in CLASSES:
                    if not ex.in_class({"fact_id": f}, cls):
                        continue
                    counts[g, cls]["changed"] += 1
                    counts[g, cls][label] += 1
                    item_flags[g, cls].setdefault(w, []).append(float(label != "record"))
        changes: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        for (_w, b, f, _i), label in audit.items():
            for g in (b, "shifts"):
                for cls, _ in CLASSES:
                    if ex.in_class({"fact_id": f}, cls):
                        changes[g, cls]["changes"] += 1
                        changes[g, cls][label] += 1
        groups = [*BRANCHES, "shifts"]
        item_times = {
            g: {
                cls: {
                    **{k: int(counts[g, cls][k]) for k in ("changed", *LABELS)},
                    "clock_only_share": _indicator_share(item_flags[g, cls]),
                }
                for cls, _ in CLASSES
            }
            for g in groups
        }
        intervals = {
            g: {
                cls: {k: int(changes[g, cls][k]) for k in ("changes", *LABELS)}
                for cls, _ in CLASSES
            }
            for g in groups
        }
        # HoldMem(1,1)'s wrong answers on changed definition item-times
        by_key = {(r["world_id"], r["branch"], r["fact_id"], r["k"], r["system"]): r for r in rows}
        errors: dict[str, Any] = {
            g: {"items": 0, "wrong": 0, "clock": 0, "cases": []} for g in groups
        }
        for r in rows_last:
            if r["system"] != ONE_ONE or not r["changed"] or r.get("cv") is None:
                continue
            if not ex.in_class(r, "definition"):
                continue
            key = (r["world_id"], r["branch"], r["fact_id"], int(r["k"]))
            label = audit[(*key[:3], last[key])]
            stale = label == "clock_silent" and not r["detected"]
            for g in (r["branch"], "shifts"):
                errors[g]["items"] += 1
                if not r["cv"]:
                    errors[g]["wrong"] += 1
                    errors[g]["clock"] += int(stale)
                    if g == "shifts":
                        errors[g]["cases"].append(
                            {
                                "world": key[0],
                                "branch": key[1],
                                "fact": key[2],
                                "k": key[3],
                                "last_change": label,
                                "detected_since_t0": bool(by_key[(*key, ONE_ONE)]["detected"]),
                                "detected_after_last_change": bool(r["detected"]),
                                "repaired_now": bool(r.get("repaired_now")),
                                "clock_staleness": stale,
                            }
                        )
        # m and r of Eq. (1): detection since t0 and detection relative to the last change
        dec = {"since_t0": ex.decomposition(ex.Table(rows))}
        dec["after_last_change"] = ex.decomposition(ex.Table(rows_last))
        terms = {
            b: {
                cls: {
                    cell: {d: _terms(dec[d], b, cls, cell) for d in dec}
                    for cell in ("D0B1", "D1B1")
                }
                for cls, _ in CLASSES
            }
            for b in BRANCHES
        }
        res["stages"][study] = sources[stage_key]
        res[study] = {
            "n_worlds": len(st.worlds),
            "world_branches_hashed": len(st.hashes),
            "item_times": item_times,
            "intervals": intervals,
            "errors_one_one": errors,
            "terms": terms,
        }
    res["code"] = code_state()
    return res


# =================================================================================== forced
def forced(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Update of the cited-cluster cell, of the source-epoch cell and of B=0 on the changed
    item-times the cited-cluster key did not invalidate."""
    tab = ex.Table(r2._jsonl(root / sources["e1"] / "analysis" / "e1_item_table.jsonl"))
    results: dict[str, Any] = {}
    for d in (0, 1):
        for b in BRANCHES:
            for cls, _ in CLASSES:
                ch = _changed_asked(cls)
                missed = _rows_cond(ch, lambda r: not r.get("detected"))
                cit = tab.vec(f"holdmem_D{d}B1_cited", b, "cv", missed)
                src = tab.vec(f"holdmem_D{d}B1_source", b, "cv", ch)
                b0 = tab.vec(f"holdmem_D{d}B0", b, "cv", ch)
                keys = rr._common(cit, src, b0)
                c, s, z = (rr._restrict(v, keys) for v in (cit, src, b0))
                results.setdefault(f"D{d}", {}).setdefault(b, {})[cls] = {
                    "n_items": rr._n_items(keys),
                    "update": {"cited": _mean(c), "forced": _mean(s), "B0": _mean(z)},
                    "forced_minus_cited": _paired(s, c),
                    "cited_minus_B0": _paired(c, z),
                }
    return {
        "status": STATUS,
        "units": UNITS,
        "stage": sources["e1"],
        "results": results,
        "code": code_state(),
    }


# ================================================================================= coverage
COV_WRITERS = ("qwen", "qwen10", "sonnet", "opus")
SUBSETS = ("complete", "incomplete", "all")
TEN = frozenset(f"w{s}" for s in range(1000, 1010))
FULL_CELL = "holdmem_D0fullB1_source"
CovKey = tuple[str, str, str, int]  # (world, branch, kappa id, k)


def _full_writes(root: Path, sources: Mapping[str, str]) -> tuple[list[Path], list[Row]]:
    """The full-access control's re-derivations of definition items (from its runs' notes)."""
    ctl = json.loads((root / sources["ctl"] / "analysis" / "ctl_metrics.json").read_text())
    runs = [root / "runs" / r for r in ctl["cell_origin"][FULL_CELL]]
    out = []
    for run in runs:
        for it in r2._jsonl(run / "items.jsonl", lambda line: FULL_CELL in line):
            if it["system"] != FULL_CELL or not it["slot_id"].startswith("k."):
                continue
            for h in it["history"]:
                if h["event"] == "repair" and "full" in h:
                    out.append(
                        {
                            "world_id": it["world_id"],
                            "branch": it["branch"],
                            "fact_id": it["slot_id"],
                            "ts": h["ts"],
                            "truncation": h["full"]["truncation"],
                        }
                    )
    return runs, out


def _scope_coverage(
    st: _Stage, full: Sequence[Row]
) -> tuple[dict[CovKey, tuple[int, int]], dict[int, bool]]:
    """Per definition item-time after the shift: (scope records, of them in the writer's
    selection); per full-access re-derivation (index in `full`): its request equals the D=0
    request. The truncation the runs noted must be the one `full_records` gives."""
    from memshift.gen.kappa import scope_members
    from memshift.gen.state import SourceLog
    from memshift.holdmem.writer import full_records, records_for, writer_request

    cov: dict[CovKey, tuple[int, int]] = {}
    same: dict[int, bool] = {}
    todo: dict[tuple[str, str], list[int]] = defaultdict(list)
    for i, f in enumerate(full):
        todo[f["world_id"], f["branch"]].append(i)
    for w, (seed, params) in sorted(st.worlds.items()):
        for b in BRANCHES:
            world = _world(seed, b, params, st.hashes.get((w, b)))
            log = SourceLog(world.events)
            for k in OFFSETS_WEEKS:
                t = _parse(st.times[w][k])
                snap = log.snapshot(t)
                for slot in world.slots:
                    if not slot.kappa_id:
                        continue
                    ver = world.kappa_version(slot.kappa_id, t)
                    scope = {oid for oid, _ in scope_members(slot.kappa_id, snap, t, ver)}
                    sel = {d.doc_id for d, _ in records_for(log, slot, t)}
                    cov[w, b, slot.kappa_id, k] = (len(scope), len(scope & sel))
            for i in todo.pop((w, b), []):
                slot = world.slot(full[i]["fact_id"])
                t = _parse(full[i]["ts"])
                acc = full_records(log, slot, t)
                if acc.truncation != full[i]["truncation"]:
                    raise ValueError(f"{w} {b} {slot.slot_id} {t}: truncation differs from the run")
                same[i] = writer_request(world, slot, t, records_for(log, slot, t)) == (
                    writer_request(world, slot, t, acc.records)
                )
    if todo:
        raise ValueError(f"full-access writes outside the stage's worlds: {sorted(todo)[:3]}")
    return cov, same


def _ratio(per_world: Mapping[str, tuple[float, float]], n_items: int) -> Est:
    arr = np.array([per_world[w] for w in sorted(per_world)], dtype=float).reshape(-1, 2)
    return ex.world_estimate(arr, lambda t: t[..., 0] / t[..., 1], n_items)


def _writer_tables(root: Path, sources: Mapping[str, str]) -> dict[str, dict[str, ex.Table]]:
    """Per writer, the item table of each D=1 arm (D1 authored, Dm model-written)."""
    main = ex.Table(r2._jsonl(root / sources["e1"] / "analysis" / "e1_item_table.jsonl"))
    synth = ex.Table(r2._jsonl(root / sources["c"] / "analysis" / "e1_item_table.jsonl"))
    out = {"qwen": {"D1": main, "Dm": synth}, "qwen10": {"D1": main, "Dm": synth}}
    keep = {*rr.ARMS["D0"], rr.ARMS["D1"][1]}
    for name, runs in H_RUNS.items():
        got, skipped = rr.read_runs([root / r for r in runs], items=keep.__contains__)
        if skipped:
            raise SystemExit(f"incomplete runs: {skipped}")
        m = rr.Merged(got, rr.by_variant)
        (group,) = m.groups()
        tab = m.table(group)
        out[name] = {"D1": tab, "Dm": tab}
    return out


def _in_worlds(worlds: Collection[str] | None) -> Cond:
    return lambda r: worlds is None or r["world_id"] in worlds


def coverage(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Evidence coverage of the D=0 writer, r and B x D by coverage, and the full-access control
    (see the docstring)."""
    st = _read_stage(root / sources["e1"])
    full_runs, full = _full_writes(root, sources)
    cov, same = _scope_coverage(st, full)
    complete = {key: n == got for key, (n, got) in cov.items()}

    def covered(r: Row) -> bool:
        return complete[r["world_id"], r["branch"], r["fact_id"].removeprefix("k."), int(r["k"])]

    subset: dict[str, Cond] = {
        "complete": covered,
        "incomplete": lambda r: not covered(r),
        "all": lambda r: True,
    }
    scope: dict[str, Any] = {}
    for kid, _ in [*DEFINITIONS, ("all", "All")]:
        per: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
        flags: dict[str, list[float]] = defaultdict(list)
        n = 0
        for (w, _b, k_id, _k), (n_scope, n_in) in cov.items():
            if kid not in ("all", k_id):
                continue
            per[w][0] += n_in
            per[w][1] += n_scope
            flags[w].append(float(n_scope == n_in))
            n += 1
        scope[kid] = {
            "item_times": n,
            "scope_records": int(sum(v[1] for v in per.values())),
            "in_selection": int(sum(v[0] for v in per.values())),
            "scope_share": _ratio({w: (v[0], v[1]) for w, v in per.items()}, n),
            "complete_share": ex.mean_estimate(flags),
        }
    tables = _writer_tables(root, sources)
    r_terms: dict[str, Any] = {}
    bxd: dict[str, Any] = {}
    for name in COV_WRITERS:
        in_worlds = _in_worlds(None if name == "qwen" else TEN)
        tabs = tables[name]
        for b in BRANCHES:
            for sub in SUBSETS:
                base = _rows_cond(rr.changed("definition"), subset[sub], in_worlds)
                det = _rows_cond(base, lambda r: bool(r.get("detected")))
                node = {}
                for d, cell in (("r0", rr.ARMS["D0"][1]), ("r1", rr.ARMS["D1"][1])):
                    v = tabs["D1"].vec(cell, b, "cv", det)
                    node[d] = _mean(v)
                r_terms.setdefault(name, {}).setdefault(b, {})[sub] = node
                for arm in ("D1", "Dm"):
                    e = rr.bxd(rr.getter(tabs[arm], b, base), arm)
                    bxd.setdefault(name, {}).setdefault(b, {}).setdefault(arm, {})[sub] = e
    full_by: dict[str, Any] = {}
    for kid, _ in [*DEFINITIONS, ("all", "All")]:
        trunc: dict[str, list[float]] = defaultdict(list)
        eq: dict[str, list[float]] = defaultdict(list)
        for i, f in enumerate(full):
            if kid not in ("all", f["fact_id"].removeprefix("k.")):
                continue
            trunc[f["world_id"]].append(float(f["truncation"] != "none"))
            eq[f["world_id"]].append(float(same[i]))
        full_by[kid] = {
            "rederived": sum(len(v) for v in trunc.values()),
            "truncated": int(sum(sum(v) for v in trunc.values())),
            "truncated_share": ex.mean_estimate(trunc),
            "same_request": int(sum(sum(v) for v in eq.values())),
            "same_request_share": ex.mean_estimate(eq),
            "truncation_kinds": dict(
                Counter(
                    f["truncation"] for f in full if kid in ("all", f["fact_id"].removeprefix("k."))
                )
            ),
        }
    return {
        "status": STATUS,
        "units": UNITS,
        "stage": sources["e1"],
        "writers": {
            "qwen": "Qwen3.5-9B, the e1 stage (D1) and the c stage (Dm), worlds 1000-1019",
            "qwen10": "the same, worlds 1000-1009",
            **{name: list(runs) for name, runs in H_RUNS.items()},
        },
        "full_access_runs": [p.name for p in full_runs],
        "n_world_branches_hashed": len(st.hashes),
        "scope": scope,
        "r": r_terms,
        "bxd": bxd,
        "full_access": full_by,
        "code": code_state(),
    }


# ==================================================================================== sizes
FIT_TOKENS = 31_000  # 32,768 less about 1,100 tokens of request text and the 700-token answer
SIZE_VARIANTS = (
    ("dicts", "Dicts"),
    ("dicts_no_body", "DictsNoBody"),
    ("columns_no_body", "ColumnsNoBody"),
)
BODY_SOURCES = ("email", "messages")  # their free-text bodies are left out of *_no_body
HELDOUT_SEEDS = (4001, 4002, 4003)
SIZE_WORLDS = {"main": range(1000, 1020), "heldout": range(1000, 1010)}


def _compact(o: object) -> str:
    return json.dumps(o, separators=(",", ":"), ensure_ascii=False)


class _Counter:
    """Qwen3.5-9B tokens of a string, cached by variant and content."""

    def __init__(self, path: Path):
        from tokenizers import Tokenizer

        self.tok = Tokenizer.from_file(str(path))
        self.cache: dict[tuple[str, bytes], int] = {}

    def __call__(self, s: str, variant: str = "") -> int:
        key = (variant, hashlib.sha1(s.encode()).digest())
        if key not in self.cache:
            self.cache[key] = len(self.tok.encode(s, add_special_tokens=False).ids)
        return self.cache[key]


def typed_sizes(
    records: Sequence[Mapping[str, Any]], ntok: Callable[[str, str], int]
) -> dict[str, int]:
    """Tokens of the complete typed records in the three serialisations (see the docstring)."""
    sizes = dict.fromkeys(("dicts", "dicts_no_body", "columns_no_body"), 2)
    tables: dict[str, list[str]] = {}
    for r in records:
        b = {k: v for k, v in r.items() if k != "text"}
        c = {k: v for k, v in b.items() if not (k == "body" and r.get("source") in BODY_SOURCES)}
        sizes["dicts"] += ntok(_compact(b), "B") + 1
        sizes["dicts_no_body"] += ntok(_compact(c), "C") + 1
        cols = tables.setdefault(f"{r['source']}|{r.get('kind', '')}", list(b))
        cols += [k for k in b if k not in cols]
        sizes["columns_no_body"] += ntok(_compact([c.get(k) for k in cols if k in c]), "E") + 1
    for cols in tables.values():
        sizes["columns_no_body"] += ntok(_compact(cols), "") + 4
    return sizes


def _summary(xs: Sequence[int]) -> dict[str, Any]:
    return {
        "n": len(xs),
        "median": float(statistics.median(xs)) if xs else None,
        "max": max(xs) if xs else None,
        "fit": sum(x <= FIT_TOKENS for x in xs) / len(xs) if xs else None,
    }


def sizes(root: Path, sources: Mapping[str, str], tokenizer: Path) -> dict[str, Any]:
    """Typed-input sizes per definition after the shift, main and held-out definitions."""
    from memshift.gen.state import SourceLog
    from memshift.holdmem.program import program_records

    count = _Counter(tokenizer)
    main = _read_stage(root / sources["e1"])
    rows: list[dict[str, Any]] = []
    for kset, seeds in (
        (None, SIZE_WORLDS["main"]),
        *((f"heldout:{s}", SIZE_WORLDS["heldout"]) for s in HELDOUT_SEEDS),
    ):
        for seed in seeds:
            for b in BRANCHES:
                want = main.hashes.get((f"w{seed}", b)) if kset is None else None
                world = _world(seed, b, world_params(12, kset), want)
                log = SourceLog(world.events)
                for k in OFFSETS_WEEKS:
                    t = world.fork + timedelta(weeks=k)
                    if kset is None and main.times[f"w{seed}"][k] != f"{t:%Y-%m-%dT%H:%M}":
                        raise ValueError(f"w{seed} k={k}: not the probe time of the main study")
                    for slot in world.slots:
                        if not slot.kappa_id:
                            continue
                        recs = program_records(log, slot, t)
                        rows.append(
                            {
                                "set": "main" if kset is None else "heldout",
                                "kappa_id": slot.kappa_id,
                                "sources": list(slot.sources),
                                "records": len(recs),
                                **typed_sizes(recs, count),
                            }
                        )
    out: dict[str, Any] = {}
    for s in ("main", "heldout"):
        sel = [r for r in rows if r["set"] == s]
        defs: dict[str, Any] = {}
        for kid in [*dict.fromkeys(r["kappa_id"] for r in sel), "all"]:
            rs = [r for r in sel if kid in ("all", r["kappa_id"])]
            defs[kid] = {
                "item_times": len(rs),
                "sources": sorted({x for r in rs for x in r["sources"]}),
                "records_median": float(statistics.median(r["records"] for r in rs)),
                **{v: _summary([r[v] for r in rs]) for v, _ in SIZE_VARIANTS},
            }
        out[s] = {"worlds": [f"w{x}" for x in SIZE_WORLDS[s]], "definitions": defs}
    return {
        "status": STATUS,
        "tokenizer": {
            "model": MAIN_MODEL,
            "file": "tokenizer.json",
            "sha256": file_sha256(tokenizer),
        },
        "fit_tokens": FIT_TOKENS,
        "checkpoints_weeks_after_fork": list(OFFSETS_WEEKS),
        "kappa_sets": {"main": "fixed", "heldout": [f"heldout:{s}" for s in HELDOUT_SEEDS]},
        **out,
        "code": code_state(),
    }


# ==================================================================================== fixed
@dataclass(frozen=True)
class Fixed:
    """A set of estimates that the design fixes, named as the analysis records it."""

    where: str  # where the row was shown: "figure" or "table"
    group: str  # the table's group (empty for the figure)
    label: str
    expected: float
    reason: str


FIXED = (
    Fixed(
        "figure",
        "",
        r"Leak check: executability's effect on missed invalidation, both keys",
        0.0,
        "the keys do not depend on D, so D's effect on missed invalidation is 0",
    ),
    Fixed(
        "figure",
        "",
        "Model-written derivations' effect on missed invalidation",
        0.0,
        "C4: the source-epoch key does not depend on D_m",
    ),
    Fixed(
        "table",
        "Other model families as writer and reader",
        "Leak check",
        0.0,
        "as the main study's leak check, for each other family",
    ),
    Fixed(
        "table",
        "Rewritten worlds",
        "Leak check (largest effect)",
        0.0,
        "as the main study's leak check, on the rewritten worlds",
    ),
    Fixed(
        "table",
        "Replication on fresh worlds",
        "Leak check (largest effect)",
        0.0,
        "as the main study's leak check, on the fresh worlds",
    ),
    Fixed(
        "table",
        "Sync and export",
        "Import without the keys, missed invalidation minus the original",
        1.0,
        "without the keys nothing is invalidated, and the source-epoch key misses no change here",
    ),
)


def _fixed_estimates(root: Path, sources: Mapping[str, str]) -> list[list[Mapping[str, Any]]]:
    """The staged estimates behind each row of `FIXED`, in its order."""
    from memshift.analysis.gate import path2

    def path2_estimates(key: str) -> list[Mapping[str, Any]]:
        m = stage_json(root, dict(sources), key, "e1_metrics.json") or {"contrasts": {}}
        return [c["estimate"] for c in path2(m)["checks"].values()]

    c = stage_json(root, dict(sources), "c", "e1_metrics.json") or {}
    rend = stage_json(root, dict(sources), "l", "rendered.json") or {}
    e6 = stage_json(root, dict(sources), "e6", "e6_t2.json") or {}
    missed = _get(e6, "imports", "imported_no_ext", "diff_90", "missed")
    return [
        path2_estimates("e1"),
        [_get(c, "contrasts", b, "C4_D1s_effect_missed_source") for b in ("src", "def")],
        [e for f in ("m2_gemma", "m2_llama", "m2_gptoss") for e in path2_estimates(f)],
        [p["rendered"] for p in rend.get("paired") or [] if p.get("check") == "i"],
        path2_estimates("rep_e1"),
        [missed] if missed else [],
    ]


def fixed(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Which rows of `FIXED` the design fixes, checked on the staged estimates: a row is found
    when its estimates are staged."""
    rows = []
    for row, ests in zip(FIXED, _fixed_estimates(root, sources), strict=True):
        vals = [[e.get("est"), e.get("lo"), e.get("hi")] for e in ests if e]
        found = bool(vals)
        exact = bool(vals) and all(
            v is not None and abs(float(v) - row.expected) < 1e-12 for vs in vals for v in vs
        )
        rows.append(
            {
                "where": row.where,
                "group": row.group,
                "label": row.label,
                "reason": row.reason,
                "expected": row.expected,
                "estimates": len(vals),
                "row_found": found,
                "all_estimates_fixed": exact,
                "fixed": found and exact,
            }
        )
    return {
        "status": STATUS,
        "rows": rows,
        "n_fixed": sum(r["fixed"] for r in rows),
        "code": code_state(),
    }


# ================================================================================== loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs (None where the stage or a file is not there)."""
    src = dict(sources)
    return {key: stage_json(root, src, STAGE_KEY, name) for key, name in FILES.items()}


# =================================================================================== macros
def _count(x: Any) -> str:
    return MISSING if x is None else f"{int(x):,}"


def _est(parent: Mapping[str, Any] | None, key: str, f: Callable[[float | None], str] = pct) -> str:
    """An estimate with its interval: MISSING without the stage, a dash over no item-time."""
    if parent is None:
        return MISSING
    e = parent.get(key)
    return UNDEFINED if not e else fmt(e, f)


def _num(parent: Mapping[str, Any] | None, key: str) -> str:
    if parent is None:
        return MISSING
    x = parent.get(key)
    return UNDEFINED if x is None else f"{int(x):,}"


def _sum(parent: Mapping[str, Any] | None, *keys: str) -> str:
    """The sum of counts under `parent` (a count it does not list is 0)."""
    return MISSING if parent is None else f"{sum(int(parent.get(k) or 0) for k in keys):,}"


def _items(parent: Mapping[str, Any] | None, key: str) -> str:
    """The item-times an estimate under `parent` is over (0 when it is over none)."""
    if parent is None:
        return MISSING
    return f"{int((parent.get(key) or {}).get('n_items') or 0):,}"


def _node(res: Mapping[str, Any] | None, *keys: str) -> dict[str, Any] | None:
    """The node at `keys`: None without the stage, {} when the stage lacks it."""
    if res is None:
        return None
    node = _get(res, *keys)
    return dict(node) if isinstance(node, Mapping) else {}


def _refresh_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for arm, at in (("D0", "Dzero"), ("D1", "Done"), ("Dm", "Dm")):
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                node = _node(res, "results", b, cls, arm)
                head = ("Rev", "Refresh", at, bt, ct)
                out.put(*head, "Items", value=_num(node, "n_items"))
                out.put(*head, "BoneMinusRefresh", value=_est(node, "B1_minus_refresh", pp))
                if arm != "D0":
                    out.put(*head, "BxDRefresh", value=_est(node, "BxD_refresh", pp))
                    out.put(
                        *head, "BxDBoneMinusRefresh", value=_est(node, "BxD_B1_minus_refresh", pp)
                    )


PART_TAGS = (("pooled", "Pooled"), ("shift", "Shift"), ("drift", "Drift"))


def _cause_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for arm, at in (("D1", "Main"), ("D1s", "Dm")):
        for b, bt in BRANCH_TAGS:
            for part, pt in PART_TAGS:
                for cls, ct in CLASSES:
                    node = _node(res, "contrasts", arm, b, part, cls)
                    out.put("Rev", "Cause", at, bt, pt, ct, "Items", value=_num(node, "n"))
                    bxd = _est(node, "BxD_interaction_update", pp)
                    out.put("Rev", "Cause", at, bt, pt, ct, "BxD", value=bxd)
    for b, bt in BRANCH_TAGS:
        for part, pt in PART_TAGS:
            for cls, ct in CLASSES:
                node = _node(res, "cited", b, part, cls)
                out.put("Rev", "Cited", bt, pt, ct, "Items", value=_num(node, "n"))
                out.put("Rev", "Cited", bt, pt, ct, "Missed", value=_num(node, "missed"))
                out.put("Rev", "Cited", bt, pt, ct, "MissShare", value=_est(node, "miss_share"))


STUDY_TAGS = (("main", "Main"), ("rep", "Rep"))
GROUP_TAGS = (*BRANCH_TAGS, ("shifts", "Shifts"))


def _clock_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for study, st in STUDY_TAGS:
        for g, gt in GROUP_TAGS:
            for cls, ct in CLASSES:
                node = _node(res, study, "item_times", g, cls)
                head = ("Rev", "Clock", st, gt, ct)
                out.put(*head, "Changed", value=_sum(node, "changed"))
                out.put(*head, "ClockOnly", value=_sum(node, "clock_caught", "clock_silent"))
                out.put(*head, "ClockOnlyShare", value=_est(node, "clock_only_share"))
                out.put(*head, "ClockCaught", value=_sum(node, "clock_caught"))
                out.put(*head, "ClockSilent", value=_sum(node, "clock_silent"))
            node = _node(res, study, "intervals", g, "definition")
            head = ("Rev", "Clock", st, gt, "Definition")
            out.put(*head, "Changes", value=_sum(node, "changes"))
            out.put(*head, "ChangesClockOnly", value=_sum(node, "clock_caught", "clock_silent"))
            out.put(*head, "ChangesClockSilent", value=_sum(node, "clock_silent"))
            err = _node(res, study, "errors_one_one", g)
            out.put("Rev", "Clock", st, gt, "ErrorsOneOne", value=_num(err, "wrong"))
            out.put("Rev", "Clock", st, gt, "ErrorsOneOneClock", value=_num(err, "clock"))
            out.put("Rev", "Clock", st, gt, "ErrorsOneOneItems", value=_num(err, "items"))
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                for cell, cellt in (("D0B1", "DzeroBone"), ("D1B1", "DoneBone")):
                    node = _node(res, study, "terms", b, cls, cell, "after_last_change")
                    out.put("Rev", "Clock", st, bt, ct, cellt, "MissedLast", value=_est(node, "m"))
                    out.put("Rev", "Clock", st, bt, ct, cellt, "RepairLast", value=_est(node, "r"))


def _forced_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for d, dt in (("D0", "Dzero"), ("D1", "Done")):
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES[:2]:
                node = _node(res, "results", d, b, cls)
                upd = None if node is None else node.get("update") or {}
                head = ("Rev", "Forced", dt, bt, ct)
                out.put(*head, "Items", value=_num(node, "n_items"))
                out.put(*head, "Cited", value=_est(upd, "cited"))
                out.put(*head, "Forced", value=_est(upd, "forced"))
                out.put(*head, "Bzero", value=_est(upd, "B0"))
                out.put(*head, "Gain", value=_est(node, "forced_minus_cited", pp))


WRITER_TAGS = (("qwen", "Qwen"), ("qwen10", "QwenTen"), ("sonnet", "Sonnet"), ("opus", "Opus"))
SUBSET_TAGS = (("complete", "Complete"), ("incomplete", "Incomplete"))


def _coverage_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for kid, dt in DEFINITIONS:
        node = _node(res, "scope", kid)
        out.put("Rev", "Cov", dt, "ScopeShare", value=_est(node, "scope_share"))
        out.put("Rev", "Cov", dt, "CompleteShare", value=_est(node, "complete_share"))
        out.put("Rev", "Cov", dt, "ItemTimes", value=_num(node, "item_times"))
    for w, wt in WRITER_TAGS:
        for b, bt in BRANCH_TAGS:
            for s, stag in SUBSET_TAGS:
                node = _node(res, "r", w, b, s)
                head = ("Rev", "Cov", wt, bt, stag)
                for d, dtag in (("r0", "Zero"), ("r1", "One")):
                    out.put(*head, f"R{dtag}", value=_est(node, d))
                    out.put(*head, f"Items{dtag}", value=_items(node, d))
                for arm, armt in (("D1", "Done"), ("Dm", "Dm")):
                    arm_node = _node(res, "bxd", w, b, arm)
                    out.put(*head, armt, "BxD", value=_est(arm_node, s, pp))
                    out.put(*head, armt, "Items", value=_items(arm_node, s))
    for kid, dt in [*DEFINITIONS, ("all", "All")]:
        node = _node(res, "full_access", kid)
        out.put("Rev", "Full", dt, "Rederived", value=_num(node, "rederived"))
        out.put("Rev", "Full", dt, "TruncatedShare", value=_est(node, "truncated_share"))
        out.put("Rev", "Full", dt, "SameRequestShare", value=_est(node, "same_request_share"))


DRAW_TAGS = ("One", "Two", "Three")


def size_definitions() -> list[tuple[str, str, str]]:
    """(set, kappa id, macro tag) of every definition the sizes cover, and the set totals."""
    from memshift.gen.kappa import heldout_draw

    out = [("main", kid, tag) for kid, tag in DEFINITIONS] + [("main", "all", "All")]
    for seed, draw in zip(HELDOUT_SEEDS, DRAW_TAGS, strict=True):
        for d in heldout_draw(seed).defs:
            out.append(("heldout", d.kappa_id, draw + camel(d.kappa_id.split("_", 1)[1])))
    return [*out, ("heldout", "all", "All")]


def _tokens(x: Any) -> str:
    return f"{x:,.0f}"


def _size_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for s, kid, tag in size_definitions():
        st = s.capitalize()
        node = _node(res, s, "definitions", kid)
        out.put("Rev", "Size", st, tag, "ItemTimes", value=_num(node, "item_times"))
        for v, vt in SIZE_VARIANTS:
            e = None if node is None else node.get(v) or {}
            for key, kt, f in (
                ("median", "Median", _tokens),
                ("max", "Max", _tokens),
                ("fit", "Fit", pct),
            ):
                x = None if e is None else e.get(key)
                value = MISSING if e is None else UNDEFINED if x is None else f(x)
                out.put("Rev", "Size", st, tag, vt, kt, value=value)


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every macro of this block, from `load`."""
    out = _Out()
    _refresh_numbers(inp.get("refresh"), out)
    _cause_numbers(inp.get("cause"), out)
    _clock_numbers(inp.get("clock"), out)
    _forced_numbers(inp.get("forced"), out)
    _coverage_numbers(inp.get("coverage"), out)
    _size_numbers(inp.get("sizes"), out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ================================================================================== staging
def produce(root: Path, part: str, tokenizer: Path | None = None) -> dict[str, Any]:
    sources = json.loads((root / "results" / "sources.json").read_text())
    if part == "refresh":
        return refresh(root)
    if part == "sizes":
        if tokenizer is None:
            raise SystemExit("sizes needs --tokenizer (the Qwen3.5-9B tokenizer.json)")
        return sizes(root, sources, tokenizer)
    fn: dict[str, Callable[[Path, Mapping[str, str]], dict[str, Any]]] = {
        "cause": cause,
        "clock": clock,
        "forced": forced,
        "coverage": coverage,
        "fixed": fixed,
    }
    return fn[part](root, sources)


def stage(root: Path, out: Path, parts: Iterable[str], tokenizer: Path | None) -> None:
    """Write the analyses of `parts` into OUT/analysis and the stage manifest (over every file
    there, with the producers of `PRODUCERS`)."""
    from memshift.analysis.stage import write_stage_manifest

    sources = json.loads((root / "results" / "sources.json").read_text())
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for part in parts:
        res = produce(root, part, tokenizer)
        (analysis / FILES[part]).write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(part, "->", analysis / FILES[part], flush=True)
    srcs = [root / s for s in source_runs(sources)]
    write_stage_manifest(out, STAGE_KEY, srcs, {"producers": PRODUCERS, "note": NOTE})


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage the review analyses (see the docstring).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None, help=f"default: ROOT/{STAGE_DIR}")
    ap.add_argument("--part", choices=sorted(FILES), action="append", default=None)
    ap.add_argument("--tokenizer", type=Path, default=None, help="Qwen3.5-9B tokenizer.json")
    a = ap.parse_args()
    stage(a.root, a.out or a.root / STAGE_DIR, a.part or list(FILES), a.tokenizer)


if __name__ == "__main__":
    main()

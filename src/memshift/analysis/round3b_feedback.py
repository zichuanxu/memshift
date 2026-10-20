"""E2 (block F): the main writer's programs with execution feedback. D_m with feedback
(cells D1sfbB0, D1sfbB1_source) against D_m (D1sB0, D1sB1_source) and D=0 (D0B0, D0B1_source),
Qwen3.5-9B, worlds 1000-1019, branches S_src, S_def and S_del. Nothing here calls a model.

    python -m memshift.analysis.round3b_feedback --runs RUN [RUN ...] \\
        [--compare RUN_OR_STAGE ...] --out FILE.json

Inputs. The F runs (`--cells D0B0,D0B1_source,D1sB0,D1sB1_source,D1sfbB0,D1sfbB1_source`,
`--discriminative --no-baselines`). `--compare` takes the committed main-study runs (the stage
directory stage4_c_e1 expands to them): the F runs' shared cells must reproduce their answers at
every checkpoint (`round3_blocks.pairing`, as for blocks U and Q; an F run's D_m cells send the
committed requests, so they are anchors too), and the deletion accounting takes its D=1 cells
(D1B0, D1B1_source) from them. Runs and copies of a cell are merged as in `round3_blocks` (the
newest complete run owns a cell).

Quantities. Units are fractions (x 100 for percent or pp); intervals are the 95% world-clustered
bootstrap of the other E1 contrasts (2,000 draws, seed 996, `ex_e1_classes`). Every block is given
for the six definition items (`definition`) and for the other 42 items (`other`); the thresholds
apply to the definition items, and the other items are reported without thresholds.
  executability  one row per program run of a D1sfb cell (an item write; its item history carries
                 the `feedback` note of `holdmem.program.synthesise_feedback`): the share that ends
                 with a valid typed value at the first attempt (D_m's own program on the same
                 records, `d1s_ok`) and after feedback (`ok`), per definition and pooled, for the
                 writes after the shift (`post`, whose outcome the cell uses) and for all writes
                 (`all`, with the pre-shift build, whose outcome is recorded only), per cell and
                 both cells; the feedback requests per write; the failure kinds of the first
                 attempts and of every failed run; the writes that start from a kept program; the
                 feedback requests whose text holds a string value of the records (`world_data`),
                 a string field value of 3 characters or more that the program request does not
                 already state (a name, a time, a title). The check
                 `d1s_differs` counts the writes at which the D_m cell's own program outcome is not
                 `d1s_ok` (0 when the two cells write alike).
  r              ContractValid of the changed item-times that the source-epoch key invalidated in
                 the B=1 cell (the population of `round3_blocks.r0`, whose keys are those of the
                 writer and the same in both cells), after S_src and S_def: D1sfbB1_source,
                 D1sB1_source and their paired difference, pooled and per definition.
  bxd            B x D on Update (the changed item-times) with D_m with feedback and with D_m as the
                 D=1 arm (`round3_blocks.double_diff`), and the paired per-item difference of
                 the double differences, after S_src, S_def and S_del.
  deletion       S_del (branch orphan): Update of each D1sfb cell, its D_m cell and the paired
                 difference; and the D_m deletion-loss measure of `ex_round3_dm_deletion` (stage 6):
                 at B=1 (and B=0) on the changed item-times whose record was deleted, the share that
                 D=0 answered and D_m did not (`accounting_vs_D0.loss_total`) with its split by
                 class, with D_m := D1sfb and D_m := D1s, and the paired difference of the loss
                 indicator. Without D=1 rows (no `--compare` holding them) the D0 rows stand in for
                 D1 and only the accounting against D=0 is kept (`d1_rows` says which).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_round3_dm_deletion as dm
from memshift.analysis import round3_blocks as r3
from memshift.runinfo import code_state

Row = r3.Row
Vec = ex.Vec
Est = ex.Est

STATUS = "block F (execution feedback): quantities only"
UNITS = r3.UNITS
DRAWS = r3.DRAWS
D0 = r3.ARMS["D0"]
DM = r3.ARMS["Dm"]
FB = ("holdmem_D1sfbB0", "holdmem_D1sfbB1_source")
D1 = r3.ARMS["D1"]
CELLS = (*D0, *DM, *FB)
CLASSES = ("definition", "other")
BRANCHES = r3.BRANCHES
SHIFTS = r3.SHIFTS
DELETION = "orphan"
DELETED = ("changed_abstain", "all")  # the population of the stage-6 deletion accounting
PHASES = ("post", "all")
# D_m with feedback as the D_m of `ex_round3_dm_deletion` (its HEAD names the cells by prefix)
AS_DM = {f: d for f, d in zip(FB, DM, strict=True)}


def _class(slot: str) -> str:
    return "definition" if ex.item_class(slot) == "definition" else "other"


def _restrict(v: Vec, keys: Mapping[str, set[str]]) -> Vec:
    return r3._restrict(v, keys)


def _paired(x: Vec, y: Vec, draws: int) -> dict[str, Est]:
    """Each vector's mean on the items in both, and the paired difference x - y."""
    keys = r3._common(x, y)
    xs, ys = _restrict(x, keys), _restrict(y, keys)
    return {
        "feedback": ex.vec_mean(xs, draws),
        "dm": ex.vec_mean(ys, draws),
        "difference": ex.paired_estimate(xs, ys, draws),
    }


# ============================================================================= executability
def program_runs(items: Iterable[Row]) -> list[Row]:
    """One row per program run of a D1s or D1sfb cell (the history events with a program note)."""
    out: list[Row] = []
    for it in items:
        system = str(it["system"])
        if not system.startswith("holdmem_D1s"):
            continue
        for ev in it["history"]:
            if "program" not in ev:
                continue
            out.append(
                {
                    "system": system,
                    "world_id": it["world_id"],
                    "branch": it["branch"],
                    "slot": it["slot_id"],
                    "ts": ev["ts"],
                    "event": ev["event"],
                    "ok": bool(ev["program"]["ok"]),
                    "feedback": ev.get("feedback"),
                }
            )
    return out


def _share(rows: Sequence[Row], field: str, draws: int) -> Est:
    v: Vec = defaultdict(dict)
    for r in rows:
        v[r["world_id"]][f"{r['system']}|{r['branch']}|{r['slot']}|{r['ts']}"] = float(r[field])
    return ex.vec_mean(v, draws)


def _exec_block(rows: Sequence[Row], draws: int) -> dict[str, Any]:
    first_kinds = Counter(str(r["d1s_kind"]) for r in rows if not r["d1s_ok"])
    run_kinds = Counter(k for r in rows for k in r["kinds"])
    world_data = [x for r in rows for x in r["world_data"]]
    return {
        "n_runs": len(rows),
        "first_attempt": _share(rows, "d1s_ok", draws),
        "after_feedback": _share(rows, "ok", draws),
        "first_attempt_failed": sum(not r["d1s_ok"] for r in rows),
        "after_feedback_failed": sum(not r["ok"] for r in rows),
        "rounds": dict(sorted(Counter(str(r["rounds"]) for r in rows).items())),
        "feedback_requests": sum(r["rounds"] for r in rows),
        "start_kept": sum(r["start"] == "kept" for r in rows),
        "first_attempt_kinds": dict(sorted(first_kinds.items())),
        "failed_run_kinds": dict(sorted(run_kinds.items())),
        "world_data": {"requests": len(world_data), "with_record_value": sum(world_data)},
    }


def executability(items: Sequence[Row], t_fork: Mapping[tuple[str, str], str], draws: int) -> Any:
    """The executability block of the module docstring. `t_fork` maps (world, branch) to the
    pre-shift build's time (the writes after it are `post`)."""
    runs = program_runs(items)
    fb_rows = []
    for r in runs:
        if r["system"] in FB and r["feedback"] is not None:
            pre = r["ts"] <= t_fork.get((r["world_id"], r["branch"]), "")
            fb_rows.append({**r, **r["feedback"], "post": not pre, "cls": _class(r["slot"])})
    own = {
        (r["system"], r["world_id"], r["branch"], r["slot"], r["ts"]): r["ok"]
        for r in runs
        if r["system"] in DM
    }
    differs = missing = 0
    for r in fb_rows:
        mine = own.get((AS_DM[r["system"]], r["world_id"], r["branch"], r["slot"], r["ts"]))
        missing += mine is None
        differs += mine is not None and mine != r["d1s_ok"]
    out: dict[str, Any] = {"checks": {"d1s_differs": differs, "d1s_missing": missing}}
    for cls in CLASSES:
        node: dict[str, Any] = {}
        for phase in PHASES:
            rows = [r for r in fb_rows if r["cls"] == cls and (phase == "all" or r["post"])]
            by_cell = {"both": rows, **{c: [r for r in rows if r["system"] == c] for c in FB}}
            node[phase] = {c: _exec_block(rs, draws) for c, rs in by_cell.items()}
            if cls == "definition":
                node[phase]["per_definition"] = {
                    s: _exec_block([r for r in rows if r["slot"] == s], draws)
                    for s in sorted({r["slot"] for r in rows})
                }
        out[cls] = node
    return out


def fork_times(data: Mapping[str, list[Row]]) -> dict[tuple[str, str], str]:
    """(world, branch) -> the time of the pre-shift probes (the build's time, t0)."""
    out: dict[tuple[str, str], str] = {}
    for p in data["probes"]:
        if int((p.get("tags") or {}).get("k", 0)) == 0:
            out[p["world_id"], p["branch"]] = str(p["as_of"])
    return out


# ======================================================================================= r
def _detected_changed(cls: str) -> e1.Cond:
    return lambda r: bool(r["changed"]) and bool(r.get("detected")) and ex.in_class(r, cls)


def r_block(tab: ex.Table, draws: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in SHIFTS:
        for cls in CLASSES:
            fb = tab.vec(FB[1], b, "cv", _detected_changed(cls))
            dmv = tab.vec(DM[1], b, "cv", _detected_changed(cls))
            node: dict[str, Any] = {"pooled": _paired(fb, dmv, draws)}
            if cls == "definition":
                fb_f, dm_f = r3._by_fact(fb), r3._by_fact(dmv)
                node["per_definition"] = {
                    f: _paired(fb_f[f], dm_f.get(f, {}), draws) for f in sorted(fb_f)
                }
            out.setdefault(b, {})[cls] = node
    return out


# ===================================================================================== B x D
def bxd_block(tab: ex.Table, draws: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for b in BRANCHES:
        for cls in CLASSES:
            get = r3.getter(tab, b, r3.changed(cls))
            dd_fb = r3.double_diff(get(FB[1]), get(FB[0]), get(D0[1]), get(D0[0]))
            dd_dm = r3.double_diff(get(DM[1]), get(DM[0]), get(D0[1]), get(D0[0]))
            node = _paired(dd_fb, dd_dm, draws)
            out.setdefault(b, {})[cls] = {
                "Dm_feedback": ex.vec_mean(dd_fb, draws),
                "Dm": ex.vec_mean(dd_dm, draws),
                "paired": node,
            }
    return out


# ================================================================================== deletion
def update_block(tab: ex.Table, draws: int) -> dict[str, Any]:
    """S_del: Update of each D1sfb cell against its D_m cell, paired per item."""
    out: dict[str, Any] = {}
    for cls in CLASSES:
        for fb, dmc in AS_DM.items():
            fbv = tab.vec(fb, DELETION, "cv", r3.changed(cls))
            dmv = tab.vec(dmc, DELETION, "cv", r3.changed(cls))
            out.setdefault(cls, {})[fb.removeprefix("holdmem_D1sfb")] = _paired(fbv, dmv, draws)
    return out


def deletion_raw(
    data: Mapping[str, list[Row]], d1: Mapping[str, list[Row]] | None, dm_cells: Sequence[str]
) -> tuple[dm.Raw, str]:
    """The S_del rows in the shape of `ex_round3_dm_deletion.Raw`, with `dm_cells` (D1s or D1sfb)
    as its D_m cells and the D=1 rows from `d1` (else the D0 rows stand in for them)."""
    rename = dict(zip(dm_cells, DM, strict=True))
    raw = dm.Raw({}, {}, defaultdict(list), {}, [])
    for p in data["probes"]:
        if p["branch"] == DELETION:
            raw.probes[p["world_id"], p["fact_id"], int(p["tags"]["k"])] = p
    d1_systems = set(D1) & {a["system"] for a in (d1 or {}).get("answers", [])}
    source = "compare" if d1_systems == set(D1) else "stand_in_D0"
    keep = {*D0, *dm_cells}

    def add(rows: Mapping[str, list[Row]], systems: Mapping[str, str]) -> None:
        for a in rows["answers"]:
            if a["branch"] == DELETION and a["system"] in systems:
                k = int(a["tags"]["k"])
                raw.answers[systems[a["system"]], a["world_id"], a["fact_id"], k] = a
        for it in rows["items"]:
            if it["branch"] == DELETION and it["system"] in systems:
                raw.items[systems[it["system"]], it["world_id"], it["slot_id"]].append(it)

    add(data, {s: rename.get(s, s) for s in keep})
    if source == "compare":
        assert d1 is not None
        add(d1, {s: s for s in D1})
    else:
        add(data, dict(zip(D0, D1, strict=True)))
    return raw, source


def _loss(joint: dm.Joint, keys: Sequence[dm.Key]) -> Vec:
    return dm.vec(joint, keys, dm.moves("D0", True, False))


def deletion_loss(
    data: Mapping[str, list[Row]], d1: Mapping[str, list[Row]] | None, draws: int
) -> dict[str, Any]:
    """`ex_round3_dm_deletion`'s accounting with D_m := D1sfb and D_m := D1s on the same rows."""
    out: dict[str, Any] = {}
    for b in (1, 0):
        joints: dict[str, dm.Joint] = {}
        node: dict[str, Any] = {}
        for name, cells in (("Dm_feedback", FB), ("Dm", DM)):
            raw, source = deletion_raw(data, d1, cells)
            joint, checks = dm.records(raw, b)
            joints[name] = joint
            node["d1_rows"] = source
            node.setdefault("checks", {})[name] = checks
            for cs in ("all", "definition", "other"):
                keys = dm.select(joint, DELETED[0], cs)
                if not keys:
                    continue
                blk = dm.block(joint, keys, draws)
                if source != "compare":  # the stand-in D1 rows are D0's: drop what reads them
                    blk = {k: v for k, v in blk.items() if not k.endswith("D1")}
                    for part in ("share", "counts"):
                        blk[part] = {c: v for c, v in blk[part].items() if c != "D1"}
                node.setdefault(cs, {})[name] = blk
        for cs in ("all", "definition", "other"):
            keys = sorted(
                set(dm.select(joints["Dm_feedback"], DELETED[0], cs))
                & set(dm.select(joints["Dm"], DELETED[0], cs))
            )
            if keys:
                node.setdefault(cs, {})["loss_vs_D0_paired"] = ex.paired_estimate(
                    _loss(joints["Dm_feedback"], keys), _loss(joints["Dm"], keys), draws
                )
        out[f"B{b}"] = node
    return out


# ======================================================================================= run
def analyse(runs: Sequence[Path], compare: Sequence[Path], draws: int = DRAWS) -> dict[str, Any]:
    new, skipped = r3.read_runs(runs, items=lambda s: True)
    newm = r3.Merged(new, r3.by_family)
    cmpm: r3.Merged | None = None
    skipped_c: list[str] = []
    if compare:
        want = {*CELLS, *D1}
        cmp_runs, skipped_c = r3.read_runs(r3.expand(compare), want.__contains__, want.__contains__)
        cmpm = r3.Merged(cmp_runs, r3.by_family)
    pair = r3.pairing(newm, cmpm, [*skipped, *skipped_c], lambda run, system: "all")
    res: dict[str, Any] = {
        "block": "F",
        "status": STATUS,
        "units": UNITS,
        "inputs": {"runs": [str(p) for p in runs], "compare": [str(p) for p in compare]},
        "pairing": pair,
        "cells": {"Dm_feedback": list(FB), "Dm": list(DM), "D0": list(D0)},
        "code": code_state(),
    }
    fams = newm.groups()
    if len(fams) != 1:
        res["error"] = f"expected the F runs of one writer family, got {len(fams)}"
        return res
    fam = fams[0]
    data = newm.data(fam)
    tab = ex.Table(e1.item_table(data))
    missing = [c for c in CELLS if c not in tab.systems]
    if missing:
        res["error"] = f"cells missing from the runs: {', '.join(missing)}"
        return res
    d1 = cmpm.data(fam) if cmpm is not None and fam in cmpm.groups() else None
    res["worlds"] = sorted(r3.cell_worlds(tab, CELLS))
    res["executability"] = executability(data["items"], fork_times(data), draws)
    res["r"] = r_block(tab, draws)
    res["bxd"] = bxd_block(tab, draws)
    res["deletion"] = {
        "update": update_block(tab, draws),
        "loss": deletion_loss(data, d1, draws),
    }
    return res


def summary(res: Mapping[str, Any]) -> str:
    pair = res["pairing"]
    bad = {k: v for k, v in pair.get("counts", {}).items() if v}
    lines = [
        f"block F: pairing {'ok' if pair.get('ok') else 'FAILED ' + json.dumps(bad)}"
        f" ({pair.get('anchor_answers_compared', 0)} anchor answers compared)"
    ]
    if "error" in res:
        return "\n".join([*lines, f"  {res['error']}"])
    ex_def = res["executability"]["definition"]["post"]["both"]
    lines.append(
        "  executability (definition, after the shift): first "
        f"{r3._iv(ex_def['first_attempt'], r3._pc)}, after feedback "
        f"{r3._iv(ex_def['after_feedback'], r3._pc)}"
        f", {ex_def['feedback_requests']} feedback requests over {ex_def['n_runs']} runs"
    )
    for b, node in res["r"].items():
        d = node["definition"]["pooled"]
        lines.append(f"  r {b:6} definition: difference {r3._iv(d['difference'])}")
    for b, node in res["bxd"].items():
        d = node["definition"]
        lines.append(
            f"  B x D {b:6} definition: feedback {r3._iv(d['Dm_feedback'])}, D_m {r3._iv(d['Dm'])}"
            f", difference {r3._iv(d['paired']['difference'])}"
        )
    loss = res["deletion"]["loss"]["B1"].get("all", {}).get("loss_vs_D0_paired")
    lines.append(f"  S_del B=1 loss against D=0, feedback minus D_m: {r3._iv(loss)}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 2)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--compare", nargs="*", default=[], type=Path)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = analyse(a.runs, a.compare, a.draws)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(summary(res))
    if not res["pairing"].get("ok"):
        print("PAIRING CHECK FAILED (see pairing.counts)", file=sys.stderr)


if __name__ == "__main__":
    main()

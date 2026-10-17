"""The query-time view against the stored model-written derivation on every item class (exploratory;
post hoc, answers the third review round of the paper, reviewer 1, Q3: "how does stored D_m with
binding compare with the query-time view on all items, given the same programs?"). It changes no
registered contrast, margin, threshold or verdict and makes no model call.

The registered control P-Z3 (prereg/v3_controls.md, section 3.6) compares the view with the stored
derivation on Update of definition items only. This module puts the same cells side by side on the
other items too, on Stability, and against the stored derivation without binding:

  view     `zero_Dm`: the zero system (BM25 over the live records the agent may read) plus, for the
           probed item, the value of the writer's D_m program (the D1s request, a response-cache
           hit of block C) run on the live records at the probe's time; nothing is stored
  DmB1     `holdmem_D1sB1_source`: HoldMem with the model-written derivation and binding
           (source-epoch key), block C
  DmB0     `holdmem_D1sB0`: the same derivation without binding (append-only store), block C
  zero     the zero system (main study)
  D1B1     `holdmem_D1B1_source`, HoldMem(1,1) with the benchmark-authored derivation (main
           study); levels only, as a reference

Worlds 1000-1009, branches src, def and orphan (S_src, S_def, S_del), the discriminative probes of
the control runs: every cell was asked the same item-times (`check.probe_sets`). The four cells of
the control stage come from its item table (`ctl_item_table.jsonl`, the merge of
`analysis.controls_data`, which checked hashes, probe sets and anchors), DmB0 from block C's item
table; both files are checked against the SHA256 their stage manifest records, and block C's DmB1
rows must equal the control stage's (the same source runs). Metrics: `update` (ContractValid on the
changed item-times), `stability` (on the unchanged ones), `all_probes` (both); classes follow
`ex_e1_classes.CLASS_SETS` (`other` is structured fields plus free text); `pooled` takes the item-
times of the three branches together, clustered by world. Every estimate is the world-clustered
paired bootstrap of the paper (2,000 draws, seed 996, `ex_e1_classes.paired_estimate`), so the
contrasts the control stage already holds (Z_upd, Z_stab, Z_vs_D1s and the cell rates) are
reproduced and checked to 1e-9; the run stops on a difference.

Costs (`costs`), from the raw runs when they are at hand: the reader's tokens per query (prompt plus
completion of every answer, `answers.jsonl`) on the same post-shift probes, paired; for the view,
the program requests replayed per query (`calls.jsonl`, purpose `program`) and the run summaries'
`ingest_tokens` (one program per item and definition version) and `verify_cpu_s` (the sandbox runs
at query time); for the stored cells, the run summaries' ingest and maintenance tokens and sandbox
seconds per world-branch. The summaries' `serve_tokens` field is 0 for every cell (not recorded
there), so serving cost is read from the answers.

Usage: python -m memshift.analysis.ex_round3_view [--root DIR] [--out DIR]
writes OUT/view.json and OUT/tab_view.tex (default OUT: runs/stage6_ex_round3).
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import controls_data as cd
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis.e1 import Cond, Row
from memshift.analysis.ex_e1_paper import UNDEFINED, _cell
from memshift.analysis.ex_review_paper import _share
from memshift.analysis.ex_round2 import BRANCH_LABELS
from memshift.analysis.paper_data import MISSING
from memshift.analysis.paper_tables import _row
from memshift.runinfo import file_sha256

STATUS = ex.STATUS + "; post hoc, third review round (reviewer 1, Q3)"
SOURCES_FILE = "results/sources.json"
CTL_KEY, DM_KEY = "ctl", "c"  # results/sources.json keys: the control stage, block C
CTL_TABLE = "analysis/ctl_item_table.jsonl"
DM_TABLE = "analysis/e1_item_table.jsonl"
SYSTEMS = {
    "view": "zero_Dm",
    "DmB1": "holdmem_D1sB1_source",
    "DmB0": "holdmem_D1sB0",
    "zero": "zero",
    "D1B1": "holdmem_D1B1_source",
}
FROM_DM = ("DmB0",)  # from block C's item table; the others from the control stage's
CONTRASTS = (  # name, a, b: the paired difference a - b
    ("view_minus_DmB1", "view", "DmB1"),
    ("view_minus_zero", "view", "zero"),
    ("DmB1_minus_zero", "DmB1", "zero"),
    ("view_minus_DmB0", "view", "DmB0"),
)
METRICS: dict[str, Callable[[str], Cond]] = {
    "update": ex._changed,
    "stability": ex._unchanged,
    "all_probes": ex._asked,
}
BRANCHES = ("src", "def", "orphan")
POOLED = "pooled"
GROUPS = (*BRANCHES, POOLED)
CLASSES = ("definition", "other", "all")
COMPARED = ("changed", "cv", "detected", "repaired_now")  # DmB1 rows of both stages
TOL = 1e-9


# ------------------------------------------------------------------------------------ loading
def _lines(path: Path) -> Iterable[Row]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def read_table(stage: Path, rel: str, systems: set[str], worlds: set[str]) -> list[Row]:
    """The rows of a staged item table for `systems` and `worlds`, after checking the file against
    the SHA256 its stage manifest records."""
    path = stage / rel
    want = json.loads((stage / "manifest.json").read_text())["files_sha256"][rel]
    got = file_sha256(path)
    if got != want:
        raise ValueError(f"{path}: sha256 {got} differs from the manifest's {want}")
    return [r for r in _lines(path) if r["system"] in systems and r["world_id"] in worlds]


def _key(r: Row) -> tuple[str, str, str, int]:
    return (r["world_id"], r["branch"], r["fact_id"], int(r["k"]))


def check_dm_b1(ctl_rows: Sequence[Row], dm_rows: Sequence[Row]) -> int:
    """Block C's DmB1 rows equal the control stage's on `COMPARED` (the control stage took the cell
    from the same source runs). Returns the number of rows compared."""
    system = SYSTEMS["DmB1"]
    a = {_key(r): tuple(r.get(f) for f in COMPARED) for r in ctl_rows if r["system"] == system}
    b = {_key(r): tuple(r.get(f) for f in COMPARED) for r in dm_rows if r["system"] == system}
    bad = sum(a.get(k) != v for k, v in b.items())
    if bad or set(a) != set(b):
        raise ValueError(f"DmB1: {bad} rows differ between the stages ({len(a)} vs {len(b)} rows)")
    return len(a)


def check_probe_sets(rows: Sequence[Row]) -> dict[str, Any]:
    """Every cell was asked the same item-times, with the same changed flag (the view's set)."""
    asked: dict[str, dict[tuple[str, str, str, int], bool]] = defaultdict(dict)
    for r in rows:
        if r.get("cv") is not None:
            asked[r["system"]][_key(r)] = bool(r["changed"])
    ref = asked[SYSTEMS["view"]]
    for name, system in SYSTEMS.items():
        if asked[system] != ref:
            raise ValueError(f"{name}: asked item-times differ from the view's")
    counts: dict[str, dict[str, int]] = {}
    for (_, b, _, _), ch in ref.items():
        node = counts.setdefault(b, {"changed": 0, "unchanged": 0})
        node["changed" if ch else "unchanged"] += 1
    return {"equal": True, "cells": len(SYSTEMS), "asked_item_times": counts}


# ---------------------------------------------------------------------------------- estimates
def vec(tab: ex.Table, name: str, group: str, cond: Cond) -> ex.Vec:
    """`Table.vec` of ContractValid; `pooled` joins the branches (item keys prefixed by branch)."""
    system = SYSTEMS[name]
    if group != POOLED:
        return tab.vec(system, group, "cv", cond)
    out: ex.Vec = defaultdict(dict)
    for b in BRANCHES:
        for w, d in tab.vec(system, b, "cv", cond).items():
            out[w].update({f"{b}|{k}": x for k, x in d.items()})
    return out


def reading(e: Mapping[str, Any]) -> str:
    """Which side a paired difference a - b favours at the 95% level."""
    if e["est"] != e["est"]:
        return "undefined"
    if e["lo"] > 0:
        return "a higher"
    if e["hi"] < 0:
        return "b higher"
    return "interval includes 0"


def estimates(tab: ex.Table, draws: int = ex.DRAWS) -> dict[str, Any]:
    """`levels[group][class][metric][cell]` and `contrasts[group][class][metric][name]`, each with
    its world-bootstrap interval, and the reading of every contrast."""
    levels: dict[str, Any] = {}
    contrasts: dict[str, Any] = {}
    for g in GROUPS:
        for cls in CLASSES:
            for metric, make in METRICS.items():
                cond = make(cls)
                v = {name: vec(tab, name, g, cond) for name in SYSTEMS}
                levels.setdefault(g, {}).setdefault(cls, {})[metric] = {
                    name: ex.vec_mean(x, draws) for name, x in v.items()
                }
                node = contrasts.setdefault(g, {}).setdefault(cls, {}).setdefault(metric, {})
                for cname, a, b in CONTRASTS:
                    e = ex.paired_estimate(v[a], v[b], draws)
                    node[cname] = e | {"reading": reading(e)}
    return {"levels": levels, "contrasts": contrasts}


def _of_type(cond: Cond, fact: str, keep: bool) -> Cond:
    return lambda r: cond(r) and (r["fact_id"] == fact) == keep


def per_definition_type(tab: ex.Table, draws: int = ex.DRAWS) -> dict[str, Any]:
    """Per definition type, pooled over the branches: the cells' Update and Stability on that type
    (`levels`), view - DmB1 and view - zero on it, and view - DmB1 on the definition items and on
    all items without it (`without_type`, leave-one-type-out). Six types: the interval is over
    worlds, the types are fixed."""
    out: dict[str, Any] = {}
    for t in ex.definition_types(tab):
        for metric in ("update", "stability"):
            only = _of_type(METRICS[metric]("definition"), t, True)
            v = {name: vec(tab, name, POOLED, only) for name in ("view", "DmB1", "DmB0", "zero")}
            rest = {
                cls: {
                    name: vec(tab, name, POOLED, _of_type(METRICS[metric](cls), t, False))
                    for name in ("view", "DmB1")
                }
                for cls in ("definition", "all")
            }
            out.setdefault(t, {})[metric] = {
                "levels": {name: ex.vec_mean(x, draws) for name, x in v.items()},
                "view_minus_DmB1": ex.paired_estimate(v["view"], v["DmB1"], draws),
                "view_minus_zero": ex.paired_estimate(v["view"], v["zero"], draws),
                "without_type": {
                    cls: {"view_minus_DmB1": ex.paired_estimate(r["view"], r["DmB1"], draws)}
                    for cls, r in rest.items()
                },
            }
    return out


def summary(contrasts: Mapping[str, Any]) -> dict[str, Any]:
    """Per contrast and metric, the (group, class) cells whose interval excludes 0, by side."""
    out: dict[str, Any] = {}
    for cname, a, b in CONTRASTS:
        for metric in METRICS:
            node: dict[str, list[str]] = {f"{a} higher": [], f"{b} higher": [], "neither": []}
            for g in GROUPS:
                for cls in CLASSES:
                    r = contrasts[g][cls][metric][cname]["reading"]
                    side = {"a higher": f"{a} higher", "b higher": f"{b} higher"}.get(r, "neither")
                    node[side].append(f"{g}/{cls}")
            out.setdefault(cname, {})[metric] = node
    return out


# -------------------------------------------------------------------------------- self check
def _close(mine: Mapping[str, Any], theirs: Mapping[str, Any]) -> float:
    """Largest difference of est, lo, hi (NaN equal to NaN); counts must agree exactly."""
    if (mine["n_worlds"], mine["n_items"]) != (theirs["n_worlds"], theirs["n_items"]):
        return math.inf
    worst = 0.0
    for k in ("est", "lo", "hi"):
        x, y = float(mine[k]), float(theirs[k])
        if not (math.isnan(x) and math.isnan(y)):
            worst = max(worst, abs(x - y))
    return worst


def check_against_ctl(res: Mapping[str, Any], metrics: Mapping[str, Any]) -> dict[str, Any]:
    """The contrasts and cell rates the control stage holds, reproduced: Z_upd and Z_stab (view -
    zero), Z_vs_D1s (view - DmB1, Update) and the Update and Stability of the four cells it has.
    Raises on a difference above `TOL`."""
    pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    con, lev = res["contrasts"], res["levels"]
    for b in BRANCHES:
        for cls in CLASSES:
            for tag, metric, cname, classes in (
                ("Z_upd", "update", "view_minus_zero", CLASSES),
                ("Z_stab", "stability", "view_minus_zero", CLASSES),
                ("Z_vs_D1s", "update", "view_minus_DmB1", ("all", "definition")),
            ):
                if cls in classes:
                    key = f"{tag}_{b}_{cls}"
                    pairs.append((key, con[b][cls][metric][cname], metrics["contrasts"][key]))
            if cls == "other":  # the control stage's cell rates have no accuracy for `other`
                continue
            for name in ("view", "DmB1", "zero", "D1B1"):
                for metric, label in (("update", "Update"), ("stability", "Stability")):
                    theirs = metrics["cells"][SYSTEMS[name]][b][cls][label]
                    pairs.append((f"{name}_{b}_{cls}_{label}", lev[b][cls][metric][name], theirs))
    diffs = {key: _close(m, t) for key, m, t in pairs}
    worst = max(diffs.values())
    if worst > TOL:
        bad = sorted(k for k, d in diffs.items() if d > TOL)
        raise ValueError(f"differs from the control stage by {worst}: {bad[:5]}")
    return {"compared": len(pairs), "max_abs_difference": worst}


# ------------------------------------------------------------------------------------- costs
def cost_runs(
    ctl_metrics: Mapping[str, Any], dm_stage: Path, worlds: set[str]
) -> dict[str, list[str]]:
    """The raw runs each cell's answers come from: the control stage's `cell_origin`, and for DmB0
    the block C source runs of the worlds."""
    out = {n: list(ctl_metrics["cell_origin"][s]) for n, s in SYSTEMS.items() if n not in FROM_DM}
    man = json.loads((dm_stage / "manifest.json").read_text())
    dm = [s["run_id"] for s in man["sources"] if set(s["worlds"]) & worlds]
    for n in FROM_DM:
        out[n] = dm
    return out


def load_tokens(
    runs: Path, by_cell: Mapping[str, Sequence[str]], worlds: set[str]
) -> dict[tuple[str, str, str, str, int], tuple[int, int, bool]]:
    """(cell, world, branch, fact, k) -> (prompt tokens, completion tokens, ContractValid) of every
    answer of the cells in their runs."""
    out: dict[tuple[str, str, str, str, int], tuple[int, int, bool]] = {}
    for name, run_ids in by_cell.items():
        system = SYSTEMS[name]
        for rid in run_ids:
            for a in _lines(runs / rid / "answers.jsonl"):
                if a["system"] != system or a["world_id"] not in worlds:
                    continue
                k = int(a["tags"].get("k", 0))
                key = (name, a["world_id"], a["branch"], a["fact_id"], k)
                out[key] = (
                    int(a["prompt_tokens"]),
                    int(a["completion_tokens"]),
                    a["contract_valid"],
                )
    return out


def token_vecs(
    rows: Sequence[Row], tokens: Mapping[tuple[str, str, str, str, int], tuple[int, int, bool]]
) -> tuple[dict[tuple[str, str, str], ex.Vec], int]:
    """Per (cell, group, part) the tokens of every asked post-shift item-time (`part`: prompt,
    completion, total), and the number of staged rows whose raw answer has another ContractValid
    (raises when an asked item-time has no answer)."""
    names = {s: n for n, s in SYSTEMS.items()}
    out: dict[tuple[str, str, str], ex.Vec] = defaultdict(lambda: defaultdict(dict))
    mismatched = 0
    for r in rows:
        if r.get("cv") is None:
            continue
        name = names[r["system"]]
        got = tokens.get((name, *_key(r)))
        if got is None:
            raise ValueError(f"{name}: no answer for {_key(r)}")
        p, c, cv = got
        mismatched += bool(cv) != bool(r["cv"])
        item = f"{r['fact_id']}@{r['k']}"
        for part, x in (("prompt", p), ("completion", c), ("total", p + c)):
            out[name, r["branch"], part][r["world_id"]][item] = float(x)
            out[name, POOLED, part][r["world_id"]][f"{r['branch']}|{item}"] = float(x)
    return out, mismatched


def summaries(runs: Path, by_cell: Mapping[str, Sequence[str]], worlds: set[str]) -> dict[str, Any]:
    """Per cell and branch, the run summaries' write-side costs per world-branch (mean over the
    worlds): ingest and maintenance tokens, sandbox seconds, stored bytes."""
    fields = ("ingest_tokens", "maintain_tokens", "serve_tokens", "verify_cpu_s", "storage_bytes")
    out: dict[str, Any] = {}
    for name, run_ids in by_cell.items():
        acc: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for rid in run_ids:
            for w in _lines(runs / rid / "world_summaries.jsonl"):
                if cd.world_id(int(w["seed"])) not in worlds:
                    continue
                for b, v in (w.get("branches") or {}).items():
                    c = (v.get("costs") or {}).get(SYSTEMS[name])
                    if c is None:
                        continue
                    acc[b]["world_branches"] += 1
                    for f in fields:
                        acc[b][f] += float(c.get(f, 0))
        out[name] = {
            b: {"world_branches": int(a["world_branches"])}
            | {f"{f}_per_world_branch": a[f] / a["world_branches"] for f in fields}
            for b, a in sorted(acc.items())
        }
    return out


def program_calls(runs: Path, run_ids: Sequence[str], worlds: set[str]) -> dict[str, Any]:
    """The view's program requests as replayed (one per answer, `calls.jsonl`): calls, the share
    served from the response cache, tokens per call and per query (answers of the view, k = 0
    included)."""
    calls = cached = tokens = answers = 0
    for rid in run_ids:
        for c in _lines(runs / rid / "calls.jsonl"):
            if c["purpose"] == "program":
                calls += 1
                cached += bool(c["cached"])
                tokens += int(c["prompt_tokens"]) + int(c["completion_tokens"])
        answers += sum(
            a["system"] == SYSTEMS["view"] and a["world_id"] in worlds
            for a in _lines(runs / rid / "answers.jsonl")
        )
    return {
        "program_calls": calls,
        "cached_share": cached / calls if calls else math.nan,
        "tokens_per_call": tokens / calls if calls else math.nan,
        "answers": answers,
        "program_tokens_per_query": tokens / answers if answers else math.nan,
    }


def costs(
    root: Path, rows: Sequence[Row], ctl_metrics: Mapping[str, Any], dm_stage: Path, draws: int
) -> dict[str, Any]:
    """The `costs` block (see the module docstring); {"available": False, ...} without raw runs."""
    worlds = set(cd.ctl_worlds())
    runs = root / "runs"
    by_cell = cost_runs(ctl_metrics, dm_stage, worlds)
    missing = sorted(
        {r for ids in by_cell.values() for r in ids if not (runs / r / "answers.jsonl").exists()}
    )
    if missing:
        return {"available": False, "missing_runs": missing}
    tokens = load_tokens(runs, by_cell, worlds)
    tv, mismatched = token_vecs(rows, tokens)
    if mismatched:
        raise ValueError(f"{mismatched} raw answers disagree with the staged ContractValid")
    per_query: dict[str, Any] = {}
    for g in GROUPS:
        for part in ("prompt", "completion", "total"):
            node = per_query.setdefault(g, {}).setdefault(part, {})
            node["mean"] = {n: ex.vec_mean(tv[n, g, part], draws) for n in SYSTEMS}
            node["paired"] = {
                cname: ex.paired_estimate(tv[a, g, part], tv[b, g, part], draws)
                for cname, a, b in CONTRASTS
            }
    return {
        "available": True,
        "runs": by_cell,
        "answers_match_staged_cv": True,
        "unit": "tokens per answered post-shift probe (reader prompt plus completion)",
        "serving_per_query": per_query,
        "view_program_requests": program_calls(runs, by_cell["view"], worlds),
        "write_side_per_world_branch": summaries(runs, by_cell, worlds),
    }


# ------------------------------------------------------------------------------------- table
TABLE_CELLS = ("view", "DmB1", "DmB0", "zero")
TABLE_CONTRASTS = ("view_minus_DmB1", "view_minus_zero", "DmB1_minus_zero")
GROUP_LABELS = {**BRANCH_LABELS, POOLED: "pooled"}
PANELS = (
    ("update", "Update (changed item-times)"),
    ("stability", "Stability (unchanged item-times)"),
)


def _count(e: Mapping[str, Any] | None) -> str:
    return MISSING if not e else (str(int(e["n_items"])) if e["n_items"] else UNDEFINED)


def tab_view(res: Mapping[str, Any]) -> str:
    """Per panel (Update, Stability), group and class: the four cells and the three paired
    differences."""
    lev, con = res["levels"], res["contrasts"]
    ncol = 3 + len(TABLE_CELLS) + len(TABLE_CONTRASTS)
    lines = [
        r"\begin{table}[t]",
        r"\caption{The query-time view against the stored model-written derivation (exploratory; "
        r"post hoc). Update (changed item-times) and Stability (unchanged item-times), percent, of "
        r"the view (the zero system plus the writer's program run on the live records at the "
        r"probe's time; nothing stored), $D_{\mathrm{m}}$ (\textsc{HoldMem} with the same "
        r"model-written derivation stored and bound by the source-epoch key), the same without "
        r"binding ($B{=}0$), and the zero system, on the same probes of worlds 1000--1009 "
        r"(10 worlds); paired differences in pp with 95\% world-bootstrap intervals. $n$: paired "
        r"item-times. Pooled: the item-times of $\Ssrc$, $\Sdef$ and $\Sdel$ together, clustered "
        r"by world.}",
        r"\label{tab:view}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}ll r rrrr@{\quad}rrr@{}}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{percent} & \multicolumn{3}{c}{paired difference (pp)} \\",
        r"\cmidrule(lr){4-7}\cmidrule(l){8-10}",
        r"& & $n$ & view & $D_{\mathrm{m}}$ & $B{=}0$ & zero "
        r"& view $-$ $D_{\mathrm{m}}$ & view $-$ zero & $D_{\mathrm{m}}$ $-$ zero \\",
    ]
    for metric, title in PANELS:
        lines += [r"\midrule", rf"\multicolumn{{{ncol}}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for i, g in enumerate(GROUPS):
            for k, cls in enumerate(CLASSES):
                lv, cn = lev[g][cls][metric], con[g][cls][metric]
                cells = [
                    _count(cn["view_minus_DmB1"]),
                    *(_share(lv[n]) for n in TABLE_CELLS),
                    *(_cell(cn[c]) for c in TABLE_CONTRASTS),
                ]
                lines.append(_row(GROUP_LABELS[g] if k == 0 else "", cls, *cells))
            if i < len(GROUPS) - 1:
                lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------- driver
def analyse(root: Path, out: Path, draws: int = ex.DRAWS) -> dict[str, Any]:
    sources = json.loads((root / SOURCES_FILE).read_text())
    ctl_stage, dm_stage = root / sources[CTL_KEY], root / sources[DM_KEY]
    worlds = set(cd.ctl_worlds())
    ctl_systems = {s for n, s in SYSTEMS.items() if n not in FROM_DM}
    dm_systems = {SYSTEMS[n] for n in FROM_DM} | {SYSTEMS["DmB1"]}
    ctl_rows = read_table(ctl_stage, CTL_TABLE, ctl_systems, worlds)
    dm_rows = read_table(dm_stage, DM_TABLE, dm_systems, worlds)
    n_dm_b1 = check_dm_b1(ctl_rows, dm_rows)
    rows = ctl_rows + [r for r in dm_rows if r["system"] != SYSTEMS["DmB1"]]
    tab = ex.Table(rows)
    if tab.worlds != sorted(worlds) or tab.branches != sorted(BRANCHES):
        raise ValueError(f"worlds {tab.worlds} or branches {tab.branches} are not the scope")
    probe_sets = check_probe_sets(rows)
    res = estimates(tab, draws)
    ctl_metrics = json.loads((ctl_stage / "analysis" / "ctl_metrics.json").read_text())
    check: dict[str, Any] = {
        "sha256": "staged item tables equal their manifests",
        "DmB1_rows_equal_across_stages": n_dm_b1,
        "probe_sets": probe_sets,
    }
    if draws == ex.DRAWS:
        check["reproduces_control_stage"] = check_against_ctl(res, ctl_metrics)
    result: dict[str, Any] = {
        "status": STATUS,
        "question": (
            "on all items, does storing the model-written derivation with binding beat running "
            "the same programs at query time, and where?"
        ),
        "bootstrap": {"draws": draws, "seed": ex.SEED, "unit": "world"},
        "cells": SYSTEMS,
        "sources": {
            "control_stage": sources[CTL_KEY],
            "block_c_stage": sources[DM_KEY],
            "worlds": sorted(worlds),
            "from_block_c": [SYSTEMS[n] for n in FROM_DM],
        },
        "check": check,
        "levels": res["levels"],
        "contrasts": res["contrasts"],
        "summary": summary(res["contrasts"]),
        "per_definition_type": per_definition_type(tab, draws),
        "costs": costs(root, rows, ctl_metrics, dm_stage, draws),
        "caveats": [
            "probes coincide with the maintenance checkpoints, so the stored cell's value is "
            "fresh at every probe: this compares a stored value with its program, not "
            "write-time with query-time freshness",
            "the view requests the program for the definition version in force at the probe, a "
            "response-cache hit of block C's write-time requests: the programs are the same",
            "the view serves an item whose program fails, or whose sources the agent may not "
            "read, as the zero system does",
            "Stability of definition items rests on few item-times (see n_items), above all "
            "after S_def, and on few definition types (see per_definition_type): an item the "
            "stored cell cannot serve (its program fails at re-derivation) is still answered by "
            "the view's zero-system fallback",
        ],
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "view.json").write_text(json.dumps(result, indent=1))
    (out / "tab_view.tex").write_text(
        "% generated by memshift.analysis.ex_round3_view; draft, do not edit\n" + tab_view(res)
    )
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    out = args.out or args.root / "runs" / "stage6_ex_round3"
    res = analyse(args.root, out)
    print(out, json.dumps(res["check"]))


if __name__ == "__main__":
    main()

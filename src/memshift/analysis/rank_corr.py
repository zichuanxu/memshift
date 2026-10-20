"""Does the ranking of systems by "knowledge update" (Update after S_src) decide their ranking under
the other shifts? A descriptive analysis with no threshold.

Over every system measured on the short worlds (1000-1014): the native systems of the E5 runs, the
short-world baselines (oracle, zero system, RAG) and the white-box HoldMem cells of the short-world
runs, each system gets
  Update on S_src, Update on S_def, and Stability after the orphan branch,
computed exactly as the E5 analysis does (`analysis.e5.carry_samples`: measurement-half items,
ContractValid on changed / unchanged items, pooled per world) and Spearman rank correlations
(average ranks for ties) between the S_src Update ranking and each of the other two. The interval is
a world-level bootstrap (2,000 draws, seed 996): resample worlds, recompute every system's metrics
on the resampled worlds, then the correlation. A system enters a correlation when it has both of
its metrics; the short-world baselines were run on the src branch only, so they appear in the table
but not in the correlations unless S_def and orphan runs of them are supplied. `holdmem_json_arm`
(a serving-format ablation of one cell) is left out.

The same correlations are also computed within three subsets (`SUBSETS`), each with its own world
bootstrap: the white-box HoldMem cells, the native systems, and the native systems with no-store
BM25 (`zero`) and plain RAG.

Inputs are raw run directories, as `analysis.e5` takes them (fetch them from the compute host):
  --e5         native E5 runs (answers.jsonl, probes.jsonl)
  --e1         white-box E1 runs on the same short worlds (their t0 probes give the gold of every
               branch the native runs need)
  --baselines  E1-format baseline runs (oracle / zero / RAG)
  --seeds      worlds to use (default 1000-1014)

Usage: python -m memshift.analysis.rank_corr OUT --e5 RUN [RUN ...] --e1 RUN [RUN ...]
                                             [--baselines RUN ...] [--seeds 1000-1014]
Writes OUT/analysis/rank_corr.json and rank_corr.md.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import e1, e5
from memshift.experiments.common import read_jsonl
from memshift.experiments.e1 import parse_seeds
from memshift.scoring.stats import world_bootstrap

Row = dict[str, Any]
Samples = dict[str, dict[str, dict[str, list[float]]]]  # system -> metric -> world -> values

# metric name -> (branch, E5 carryover metric)
METRICS = {
    "update_src": ("src", "Update"),
    "update_def": ("def", "Update"),
    "stability_orphan": ("orphan", "Stability"),
}
PAIRS = (("update_src", "update_def"), ("update_src", "stability_orphan"))
BASELINES = frozenset({"oracle", "zero", "rag"})
NO_STORE = frozenset({"zero", "rag"})  # no-store BM25 and plain RAG: the baselines with no store
EXCLUDED = frozenset({"holdmem_json_arm"})
SUBSETS = ("holdmem", "native", "native_baselines")
STATUS = "additional analysis: descriptive, no threshold"


def average_ranks(values: Sequence[float]) -> list[float]:
    """1-based ranks in ascending order, ties sharing the mean of their positions."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        for t in range(i, j + 1):
            ranks[order[t]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Pearson correlation of the average ranks; nan for fewer than 3 pairs or a constant input."""
    xs, ys = [round(v, 12) for v in x], [round(v, 12) for v in y]
    if len(xs) < 3 or len(set(xs)) < 2 or len(set(ys)) < 2:
        return math.nan
    return float(np.corrcoef(average_ranks(xs), average_ranks(ys))[0, 1])


def load_rows(
    e5_runs: list[Path], e1_runs: list[Path], baseline_runs: list[Path], worlds: set[str]
) -> tuple[list[Row], set[str]]:
    """The E1-format item table of all runs merged, restricted to `worlds`, and the names of the
    native systems."""
    data: dict[str, list[Row]] = {"probes": [], "answers": [], "items": []}
    native: set[str] = set()
    for d in e5_runs:
        answers = read_jsonl(d / "answers.jsonl")
        native |= {a["system"] for a in answers}
        data["answers"] += answers
        data["probes"] += read_jsonl(d / "probes.jsonl")
    for d in [*e1_runs, *baseline_runs]:
        for key in ("probes", "answers"):
            data[key] += read_jsonl(d / f"{key}.jsonl")
    data = {k: [r for r in v if r["world_id"] in worlds] for k, v in data.items()}
    return e1.item_table(data), native


def measure(rows: list[Row], worlds: set[str]) -> Samples:
    """Per system and metric, the world -> ContractValid values of the E5 definition."""
    grouped: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        grouped[r["system"], r["branch"]].append(r)
    out: Samples = {}
    for system in sorted({s for s, _ in grouped} - EXCLUDED):
        per: dict[str, dict[str, list[float]]] = {}
        for name, (branch, metric) in METRICS.items():
            by = e5.carry_samples(grouped[system, branch], system, branch, worlds)[metric]
            if by:
                per[name] = by
        if per:
            out[system] = per
    return out


def _encode(samples: Samples, systems: list[str], worlds: list[str]) -> dict[str, list[float]]:
    """One flat vector per world: (sum, count) of every (system, metric), in `systems` x METRICS
    order, so a world resample is a resample of these vectors and the metrics are recomputed from
    the summed pairs (`world_bootstrap`'s `stat` receives the resampled vectors)."""
    out: dict[str, list[float]] = {}
    for w in worlds:
        vec: list[float] = []
        for s in systems:
            for m in METRICS:
                vals = samples[s].get(m, {}).get(w, [])
                vec += [sum(vals), float(len(vals))]
        out[w] = vec
    return out


def _rho_stat(
    systems: list[str], left: str, right: str
) -> Callable[[Sequence[Sequence[float]]], float]:
    """`world_bootstrap` statistic over `_encode`d worlds: Spearman rho between two metrics."""
    slot = {
        (s, m): 2 * (i * len(METRICS) + j)
        for i, s in enumerate(systems)
        for j, m in enumerate(METRICS)
    }

    def stat(groups: Sequence[Sequence[float]]) -> float:
        total = np.asarray(groups, dtype=float).sum(axis=0)
        xs, ys = [], []
        for s in systems:
            ix, iy = slot[s, left], slot[s, right]
            if total[ix + 1] > 0 and total[iy + 1] > 0:
                xs.append(total[ix] / total[ix + 1])
                ys.append(total[iy] / total[iy + 1])
        return spearman(xs, ys)

    return stat


def kind(system: str, native: set[str]) -> str:
    return "native" if system in native else "baseline" if system in BASELINES else "white-box"


def subsets(systems: Iterable[str], native: set[str]) -> dict[str, list[str]]:
    """The systems of each subset in `SUBSETS`: the white-box HoldMem cells, the native systems,
    and the native systems with no-store BM25 and plain RAG."""
    names = sorted(systems)
    return {
        "holdmem": [s for s in names if kind(s, native) == "white-box"],
        "native": [s for s in names if s in native],
        "native_baselines": [s for s in names if s in native or s in NO_STORE],
    }


def correlations(samples: Samples, among: Sequence[str] | None = None) -> dict[str, Any]:
    """The Spearman correlations of `PAIRS` over the systems `among` (default: every system), with
    a world bootstrap over the worlds these systems were measured on."""
    systems = sorted(samples if among is None else among)
    worlds = sorted({w for s in systems for by in samples[s].values() for w in by})
    by_world = _encode(samples, systems, worlds)
    out: dict[str, Any] = {}
    for left, right in PAIRS:
        est = world_bootstrap(by_world, stat=_rho_stat(systems, left, right))
        both = [s for s in systems if left in samples[s] and right in samples[s]]
        out[f"{left}_vs_{right}"] = {
            "rho": est.est,
            "lo": est.lo,
            "hi": est.hi,
            "n_worlds": est.n_worlds,
            "n_systems": len(both),
            "systems": both,
            "left_out": [s for s in systems if s not in both],
        }
    return out


def system_rows(samples: Samples, native: set[str]) -> list[Row]:
    """One row per system: kind, each metric's estimate (world bootstrap, as E5 reports it) and its
    rank (1 = highest) among the systems that have it."""
    values = {
        m: {s: world_bootstrap(samples[s][m]) for s in samples if m in samples[s]} for m in METRICS
    }
    ranks = {}
    for m, ests in values.items():
        names = sorted(ests)
        asc = average_ranks([round(ests[s].est, 12) for s in names])
        ranks[m] = {s: len(names) + 1 - r for s, r in zip(names, asc, strict=True)}
    rows = []
    for s in sorted(samples):
        row: Row = {"system": s, "kind": kind(s, native)}
        for m in METRICS:
            row[m] = values[m][s].to_json() if s in values[m] else None
            row[f"rank_{m}"] = ranks[m].get(s)
        rows.append(row)
    return sorted(rows, key=lambda r: ("native", "baseline", "white-box").index(r["kind"]))


def _pct(e: dict[str, Any] | None) -> str:
    return (
        "n/a" if e is None else f"{100 * e['est']:.1f} [{100 * e['lo']:.1f}, {100 * e['hi']:.1f}]"
    )


def table(res: dict[str, Any]) -> str:
    lines = [
        f"Rank correlation across systems ({STATUS})",
        "",
        "| system | kind | Update S_src | Update S_def | Stability orphan "
        "| rank S_src | rank S_def | rank orphan |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in res["systems"]:
        ranks = [r[f"rank_{m}"] for m in METRICS]
        lines.append(
            f"| {r['system']} | {r['kind']} | "
            + " | ".join(_pct(r[m]) for m in METRICS)
            + " | "
            + " | ".join("n/a" if k is None else f"{k:g}" for k in ranks)
            + " |"
        )
    lines += ["", "Ranks: 1 = highest, among the systems that have the metric.", ""]
    groups = [("all systems", res["correlations"])]
    groups += [(f"subset {s}", c) for s, c in (res.get("subsets") or {}).items()]
    for group, corr in groups:
        for name, c in corr.items():
            left, right = name.split("_vs_")
            left_out = (
                f"; left out (metric missing): {', '.join(c['left_out'])}" if c["left_out"] else ""
            )
            lines.append(
                f"Spearman rho({left}, {right}), {group} = {c['rho']:.2f} "
                f"[{c['lo']:.2f}, {c['hi']:.2f}] over {c['n_systems']} systems, "
                f"{c['n_worlds']} worlds{left_out}"
            )
    return "\n".join(lines) + "\n"


def analyse(
    e5_runs: list[Path],
    e1_runs: list[Path],
    out: Path,
    baseline_runs: list[Path] | None = None,
    seeds: list[int] | None = None,
) -> dict[str, Any]:
    worlds = {f"w{s:04d}" for s in (seeds if seeds is not None else range(1000, 1015))}
    rows, native = load_rows(e5_runs, e1_runs, baseline_runs or [], worlds)
    samples = measure(rows, worlds)
    res = {
        "status": STATUS,
        "definitions": "Update / Stability as analysis.e5.carry_samples: measurement-half items, "
        "changed / unchanged, pooled per world; interval: world bootstrap",
        "worlds": sorted({w for s in samples.values() for by in s.values() for w in by}),
        "excluded_systems": sorted(EXCLUDED),
        "systems": system_rows(samples, native),
        "correlations": correlations(samples),
        "subsets": {
            name: correlations(samples, among) for name, among in subsets(samples, native).items()
        },
    }
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "rank_corr.json").write_text(json.dumps(res, indent=1))
    (out / "analysis" / "rank_corr.md").write_text(table(res))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--e5", nargs="+", required=True)
    ap.add_argument("--e1", nargs="+", required=True)
    ap.add_argument("--baselines", nargs="*", default=[])
    ap.add_argument("--seeds", default="1000-1014")
    a = ap.parse_args()
    r = analyse(
        [Path(p) for p in a.e5],
        [Path(p) for p in a.e1],
        Path(a.out),
        [Path(p) for p in a.baselines],
        parse_seeds(a.seeds),
    )
    print(table(r))

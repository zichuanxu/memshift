"""Block L (prereg/v2.md §5): the M2 checks on LLM-rendered worlds, and their paired difference
from the same-seed template worlds.

(a) `families.family_checks` on the rendered E1 run, the same call block M2 makes: checks (i)-(v)
    of prereg v2 §2 and the pattern verdict (ii)-(iv). The same call on the template test data,
    restricted to the rendered run's worlds, gives the reference.
(b) For every contrast behind (ii)-(v), and each D effect on missed invalidation (path 2, every
    granularity and branch present in both), the difference rendered minus template. Both conditions
    ask identical probes (rendering changes free text, not gold), so every contrast is paired at the
    item level: an item's contrast value (its cell-vs-cell difference, or double difference for the
    interaction) in the rendered world minus the same item's value in the template world, pooled
    over items, world-clustered bootstrap (2,000 draws, seed 996). On the items both conditions
    share this equals the rendered estimate minus the template estimate (`diff`). The contrast
    estimated on each world in both conditions and their difference are listed per world
    (`per_world`), with the equal-weight mean of those per-world differences and its bootstrap
    interval as the alternative reading (`diff_world_mean`; equal to `diff` when the worlds have
    equally many items, and it is not what the checks use). A descriptive table gives
    ContractValid rendered minus template per branch and cell, paired at the probe level (world,
    branch, k, cell, fact), before the shift, after it (k > 0 pooled) and at each k.
(c) Rendering descriptives read from the rendered run: per source kind, the share of bodies kept as
    template, the events retried, and the fidelity of the rendered worlds (must be 1.0).
    `experiments.e1` writes them per world and branch as `RenderReport.to_json()` under
    `branches[branch]["render"]` in `world_summaries.jsonl` (a raw-run file, not committed), and the
    totals only (`model`, `sources`, `skipped_background`, `kept_template`) as `worlds[i]["render"]`
    in `manifest.json`; the manifest is the fallback when no world summaries are at hand.

An input is an E1 stage directory (`analysis/e1_item_table.jsonl`, as `analysis.families` takes) or
a raw E1 run directory (probes / answers / items .jsonl).

Usage: python -m memshift.analysis.rendered RENDERED TEMPLATE OUT [--runs RUN_DIR ...]
Writes OUT/analysis/rendered.json. --runs: raw rendered run directories for (c); default RENDERED.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, NamedTuple

from memshift.analysis import e1, families
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import item_diffs, paired_diff, world_bootstrap

Row = dict[str, Any]
Vec = dict[str, dict[str, float]]  # world -> item -> value

# family_checks estimate name -> (check of prereg v2 §2, branch, e1.contrasts key)
FAMILY_CONTRASTS: dict[str, tuple[str | None, str, str]] = {
    "B_update_src": ("ii", "src", "B_effect_update_pooled_source"),
    "D_repair_src": ("iii", "src", "D_effect_repair_source"),
    "D_repair_def": ("iii", "def", "D_effect_repair_source"),
    "BxD_def": ("iv", "def", "BxD_interaction_update_source"),
    "BxD_src": (None, "src", "BxD_interaction_update_source"),  # reported, not a check
}
ZERO_MINUS_HM = "zero_minus_hm11_def"  # check (v)
COUNT_FIELDS = (
    "events",
    "skipped_background",
    "rendered",
    "retried",
    "kept_template",
    "anchored_events",
    "anchors",
    "outside_length_band",
)


class _Pair(NamedTuple):
    """The two per-world item maps of one contrast, as `e1.contrasts` hands them to paired_diff."""

    a: Vec
    b: Vec

    def to_json(self) -> _Pair:
        return self


def load_rows(path: Path) -> list[Row]:
    """The E1 item table of a stage directory or of a raw run directory."""
    table = path / "analysis" / "e1_item_table.jsonl"
    if table.exists():
        return read_jsonl(table)
    if (path / "answers.jsonl").exists():
        return e1.item_table(e1.load(path))
    raise FileNotFoundError(f"{path}: neither analysis/e1_item_table.jsonl nor answers.jsonl")


def summarise(rows: list[Row]) -> dict[str, Any]:
    """The part of `e1.analyse`'s result that `families.family_checks` reads."""
    branches = sorted({r["branch"] for r in rows})
    return {
        "contrasts": {b: e1.contrasts(rows, b) for b in branches},
        "n_worlds": len({r["world_id"] for r in rows}),
    }


def contrast_vectors(rows: list[Row]) -> dict[tuple[str, str], Vec]:
    """(branch, contrast key) -> world -> item -> contrast value, for every contrast of
    `e1.contrasts` that is a paired difference (a[item] - b[item] over the items in both)."""
    out: dict[tuple[str, str], Vec] = {}
    for branch in sorted({r["branch"] for r in rows}):
        for key, pair in e1.contrasts(rows, branch, paired_diff=_Pair).items():
            if isinstance(pair, _Pair):
                out[(branch, key)] = item_diffs(pair.a, pair.b)
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired_row(vr: Vec, vt: Vec, per_world: bool = True, **labels: Any) -> Row:
    """Rendered minus template for one contrast's (or one cell's) item values: the pooled means over
    the items both conditions have, their item-paired difference with its world bootstrap interval,
    and (optionally) the same per world with the equal-weight mean over worlds."""
    common = item_diffs(vr, vt)
    kept = {w: list(d) for w, d in common.items() if d}
    row: Row = {
        **labels,
        "mean_rendered": _mean([vr[w][k] for w, ks in kept.items() for k in ks]),
        "mean_template": _mean([vt[w][k] for w, ks in kept.items() for k in ks]),
        "diff": paired_diff(vr, vt).to_json(),
        "items": {
            "rendered": sum(len(d) for d in vr.values()),
            "template": sum(len(d) for d in vt.values()),
            "common": sum(len(ks) for ks in kept.values()),
        },
    }
    if per_world:
        row["per_world"] = {
            w: {
                "rendered": _mean([vr[w][k] for k in ks]),
                "template": _mean([vt[w][k] for k in ks]),
                "diff": _mean([common[w][k] for k in ks]),
                "n_items": len(ks),
            }
            for w, ks in kept.items()
        }
        row["diff_world_mean"] = world_bootstrap(
            {w: [v["diff"]] for w, v in row["per_world"].items()}
        ).to_json()
    return row


def contrast_rows(
    rows_r: list[Row], rows_t: list[Row], sum_r: dict[str, Any], sum_t: dict[str, Any]
) -> list[Row]:
    """One row per contrast behind checks (ii)-(v) and per path-2 contrast: the estimate in each
    condition (`e1.contrasts`, all of the condition's items) and the paired difference."""
    vec_r, vec_t = contrast_vectors(rows_r), contrast_vectors(rows_t)
    wanted = [(n, c, b, k) for n, (c, b, k) in FAMILY_CONTRASTS.items()]
    wanted += [
        (f"{b}:{k}", "i", b, k)
        for (b, k) in sorted(vec_r)
        if k.startswith("D_effect_missed_") and (b, k) in vec_t
    ]
    out = [
        paired_row(
            vec_r[branch, key],
            vec_t[branch, key],
            name=name,
            check=check,
            branch=branch,
            contrast=key,
            rendered=sum_r["contrasts"][branch][key],
            template=sum_t["contrasts"][branch][key],
        )
        for name, check, branch, key in wanted
        if (branch, key) in vec_r and (branch, key) in vec_t
    ]
    zr, hr = families.zero_hm_vectors(rows_r)
    zt, ht = families.zero_hm_vectors(rows_t)
    if zr and hr and zt and ht:
        out.append(
            paired_row(
                item_diffs(zr, hr),
                item_diffs(zt, ht),
                name=ZERO_MINUS_HM,
                check="v",
                branch="def",
                contrast="ContractValid of zero minus HoldMem(1,1), k > 0",
                rendered=paired_diff(zr, hr).to_json(),
                template=paired_diff(zt, ht).to_json(),
            )
        )
    return out


def _cv_vectors(rows: list[Row]) -> dict[tuple[str, str, str], Vec]:
    """(branch, system, period) -> world -> probe -> ContractValid. Periods: "pre" (k = 0), "post"
    (k > 0 pooled; probe = fact@k) and "k<k>" (probe = fact)."""
    vec: dict[tuple[str, str, str], Vec] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        b, s, w, fact = r["branch"], r["system"], r["world_id"], r["fact_id"]
        if r.get("cv") is not None:
            vec[b, s, "post"][w][f"{fact}@{r['k']}"] = float(r["cv"])
            vec[b, s, f"k{r['k']}"][w][fact] = float(r["cv"])
        if r.get("cv_pre") is not None:
            vec[b, s, "pre"][w][fact] = float(r["cv_pre"])
    return vec


def cell_rows(rows_r: list[Row], rows_t: list[Row]) -> list[Row]:
    """ContractValid rendered minus template per (branch, cell, period), paired at the probe level
    (descriptive: no verdict attaches to it)."""
    vr, vt = _cv_vectors(rows_r), _cv_vectors(rows_t)
    rows = [
        paired_row(vr[key], vt[key], per_world=False, branch=key[0], system=key[1], period=key[2])
        for key in sorted(set(vr) & set(vt))
    ]
    return [r for r in rows if r["items"]["common"]]


def probe_identity(rows_r: list[Row], rows_t: list[Row]) -> dict[str, Any]:
    """Whether the two conditions ask the same probes with the same gold change: for every
    (world, branch, fact, k) the item table's `changed` flag and value type must agree."""

    def gold(rows: list[Row]) -> dict[tuple[str, str, str, int], tuple[bool, str]]:
        return {
            (r["world_id"], r["branch"], r["fact_id"], int(r["k"])): (r["changed"], r["vtype"])
            for r in rows
        }

    gr, gt = gold(rows_r), gold(rows_t)
    common = gr.keys() & gt.keys()
    mismatch = sum(gr[k] != gt[k] for k in common)
    return {
        "probes_rendered": len(gr),
        "probes_template": len(gt),
        "probes_common": len(common),
        "only_rendered": len(gr.keys() - gt.keys()),
        "only_template": len(gt.keys() - gr.keys()),
        "changed_or_vtype_mismatch": mismatch,
        "identical": bool(common) and not mismatch and gr.keys() == gt.keys(),
    }


def compare_checks(fam_r: dict[str, Any], fam_t: dict[str, Any]) -> dict[str, Any]:
    checks = {
        k: {"rendered": v, "template": fam_t["checks"][k], "agree": v == fam_t["checks"][k]}
        for k, v in fam_r["checks"].items()
    }
    return {
        "checks": checks,
        "pattern_holds": {"rendered": fam_r["pattern_holds"], "template": fam_t["pattern_holds"]},
    }


# ------------------------------------------------------------------------------ (c) descriptives
def _source_totals(counts: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] = {f: sum(c[f] for c in counts) for f in COUNT_FIELDS}
    attempted = total["events"] - total["skipped_background"]
    total["attempted"] = attempted
    total["kept_template_share"] = total["kept_template"] / attempted if attempted else 0.0
    total["retried_share"] = total["retried"] / attempted if attempted else 0.0
    failures: dict[str, int] = defaultdict(int)
    for c in counts:
        for kind, n in c["first_failures"].items():
            failures[kind] += n
    total["first_failures"] = dict(sorted(failures.items()))
    return total


def _from_summaries(run_dirs: list[Path], worlds: set[str] | None) -> dict[str, Any] | None:
    reports: list[tuple[str, str, dict[str, Any]]] = []
    for d in run_dirs:
        path = d / "world_summaries.jsonl"
        if not path.exists():
            continue
        for row in read_jsonl(path):
            ids = {w["branch"]: w["world_id"] for w in row.get("worlds", [])}
            for branch, b in row["branches"].items():
                wid = ids.get(branch, f"w{row['seed']:04d}")
                if "render" in b and (worlds is None or wid in worlds):
                    reports.append((wid, branch, b["render"]))
    if not reports:
        return None
    reports.sort(key=lambda t: (t[0], t[1]))
    sources = sorted({s for _, _, r in reports for s in r["sources"]})
    usage: dict[str, int] = defaultdict(int)
    for _, _, r in reports:
        for k, v in r["usage"].items():
            usage[k] += v
    by_world_branch = []
    for w, b, r in reports:
        t = _source_totals(list(r["sources"].values()))
        by_world_branch.append(
            {
                "world_id": w,
                "branch": b,
                "attempted": t["attempted"],
                "retried": t["retried"],
                "kept_template": t["kept_template"],
                "kept_template_share": t["kept_template_share"],
                "fidelity_rate": r["fidelity"]["rate"],
            }
        )
    fid = [(w, b, r["fidelity"]) for w, b, r in reports]
    return {
        "source": "world_summaries.jsonl",
        "models": sorted({r["model"] for _, _, r in reports}),
        "world_branches": len(reports),
        "note": "counts are summed over world x branch renderings: every branch renders its own "
        "copy of the natural events",
        "per_source": {
            s: _source_totals([r["sources"][s] for _, _, r in reports if s in r["sources"]])
            for s in sources
        },
        "total": _source_totals([c for _, _, r in reports for c in r["sources"].values()]),
        "fidelity": {
            "checked": sum(f["checked"] for _, _, f in fid),
            "min_rate": min(f["rate"] for _, _, f in fid),
            "min_rate_template": min(f["rate_template"] for _, _, f in fid),
            "all_one": all(f["rate"] == 1.0 for _, _, f in fid),
            "below_one": [f"{w}/{b}" for w, b, f in fid if f["rate"] < 1.0],
        },
        "requests": sum(r["requests"] for _, _, r in reports),
        "usage": dict(sorted(usage.items())),
        "by_world_branch": by_world_branch,
    }


def _from_manifests(run_dirs: list[Path], worlds: set[str] | None) -> dict[str, Any] | None:
    recs = []
    for d in run_dirs:
        path = d / "manifest.json"
        if not path.exists():
            continue
        for w in json.loads(path.read_text()).get("worlds", []):
            if "render" in w and (worlds is None or w["world_id"] in worlds):
                recs.append(w)
    if not recs:
        return None
    return {
        "source": "manifest.json",
        "models": sorted({w["render"]["model"] for w in recs}),
        "world_branches": len(recs),
        "note": "the manifest keeps totals only; per source kind, retries and fidelity are in "
        "world_summaries.jsonl of the raw run (pass it with --runs)",
        "per_source": None,
        "total": {
            "skipped_background": sum(w["render"]["skipped_background"] for w in recs),
            "kept_template": sum(w["render"]["kept_template"] for w in recs),
        },
        "fidelity": None,
        "by_world_branch": [
            {
                "world_id": w["world_id"],
                "branch": w["branch"],
                "kept_template": w["render"]["kept_template"],
            }
            for w in sorted(recs, key=lambda w: (w["world_id"], w["branch"]))
        ],
    }


def render_descriptives(run_dirs: list[Path], worlds: set[str] | None = None) -> dict[str, Any]:
    """Rendering descriptives of raw rendered runs, restricted to `worlds` when given."""
    found = _from_summaries(run_dirs, worlds) or _from_manifests(run_dirs, worlds)
    if found is None:
        return {
            "source": None,
            "note": "no world_summaries.jsonl or manifest render records in "
            + ", ".join(map(str, run_dirs))
            + " (pass the raw rendered run directories with --runs)",
        }
    return found


# ------------------------------------------------------------------------------------------ main
def analyse(
    rendered: Path, template: Path, out: Path, runs: list[Path] | None = None
) -> dict[str, Any]:
    rows_r = load_rows(rendered)
    worlds = sorted({r["world_id"] for r in rows_r})
    rendered_worlds = set(worlds)
    rows_t = [r for r in load_rows(template) if r["world_id"] in rendered_worlds]
    template_worlds = {r["world_id"] for r in rows_t}
    sum_r, sum_t = summarise(rows_r), summarise(rows_t)
    fam_r, fam_t = families.family_checks(sum_r, rows_r), families.family_checks(sum_t, rows_t)
    res = {
        "prereg": "v2 section 5 (block L); checks (i)-(v) of section 2 (M2)",
        "worlds": worlds,
        "worlds_missing_in_template": [w for w in worlds if w not in template_worlds],
        "rendered": fam_r,
        "template_same_seeds": fam_t,
        "check_comparison": compare_checks(fam_r, fam_t),
        "paired": contrast_rows(rows_r, rows_t, sum_r, sum_t),
        "cells": cell_rows(rows_r, rows_t),
        "probe_identity": probe_identity(rows_r, rows_t),
        "rendering": render_descriptives(runs if runs else [rendered], set(worlds)),
    }
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "rendered.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("rendered")
    ap.add_argument("template")
    ap.add_argument("out")
    ap.add_argument("--runs", nargs="*", default=[])
    a = ap.parse_args()
    r = analyse(Path(a.rendered), Path(a.template), Path(a.out), [Path(p) for p in a.runs])
    print(
        json.dumps(
            {
                "worlds": len(r["worlds"]),
                "rendered_checks": r["rendered"]["checks"],
                "rendered_pattern_holds": r["rendered"]["pattern_holds"],
                "template_pattern_holds": r["template_same_seeds"]["pattern_holds"],
                "diff_rendered_minus_template": {
                    p["name"]: round(p["diff"]["est"], 4) for p in r["paired"]
                },
                "probes_identical": r["probe_identity"]["identical"],
                "fidelity": (r["rendering"].get("fidelity") or {}).get("min_rate"),
            },
            indent=1,
        )
    )

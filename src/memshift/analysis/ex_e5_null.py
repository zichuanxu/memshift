"""Exploratory null baselines for the native-system test (decisions.md 2026-09-30; not
pre-registered, no registered verdict changes; stored by `e5.analyse` as `explore_native_null`).

The registered criterion (prereg v2 §3, N-2 / N-3) counts a comparison "inside" when the native
estimate lies in the 95% interval of the white-box cell with the inferred (D, B) label, and
`e5.stage4_summary` adds "discriminating" (inside, and outside the other B label's interval). It has
no null: a system that stores nothing could be inside as often. For every native system in the
analysis (so block N's six systems need no change), on the six comparisons (Update and Stability
after S_src, S_def and the orphan branch; a comparison exists where the system has measured items):
  contained_by      which reference intervals contain the native estimate: every white-box cell
                    present (source-epoch and cited-cluster), and the zero system, plain RAG and the
                    oracle context where they were run on the same worlds and branch;
  counts            comparisons inside the predicted interval, inside the other label's interval and
                    discriminating, over all six, over the label-circular ones (the orphan branch,
                    from which B is inferred) and over the out-of-sample ones (S_src, S_def);
  baseline_counts   the same counts for each label-free baseline's own estimate placed against this
                    system's predicted and other intervals (what a system with no state would
                    score under the same rule), beside the native counts on the same comparisons;
  chance_coverage   the share of reference intervals that contain the native estimate, i.e. the
                    coverage expected when the label is permuted: over the four (D, B) label cells
                    at the system's binding granularity, and over those plus the zero system and
                    RAG;
  paired_difference the item-paired native - reference difference (reference: the predicted cell and
                    the zero and RAG baselines), world-clustered, with its 95% interval and an
                    interval inflated by the repeat-ingestion variance of the noise-floor runs;
  b_identified      the orphan persistence P with its world-bootstrap interval; B is unidentified
                    when the interval straddles `e5_infer.B_THRESHOLD`.
Inflation: a second, independent ingestion of a few worlds gives, per system, branch and metric,
the per-world difference d_w between the two ingestions (measurement-half items present in both);
the variance of one ingestion's per-world mean is taken as mean(d_w^2) / 2 (the mean difference is
zero by exchangeability), the variance this adds to a pooled estimate over worlds with n_w items is
that times sum((n_w / N)^2), and each half-width h of the interval becomes sqrt(h^2 + (1.96 se)^2).
The world bootstrap already contains one draw of the ingestion noise, so this is conservative.
The interval comparisons (`contained_by`, `counts`) use each system's own measured items; the paired
difference uses the items both were asked. They differ where the two were asked at different
checkpoints: Mem0 answered the orphan branch at k = 1 only, the white-box cells at k = 1, 2, 4, 8.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from memshift.analysis import e1, e5, e5_infer
from memshift.experiments.common import read_jsonl
from memshift.gen.shifts import inference_half
from memshift.scoring.stats import paired_diff, world_bootstrap

Row = dict[str, Any]
METRICS = ("Update", "Stability")
CIRCULAR_BRANCHES = frozenset({"orphan"})  # B is inferred from the orphan probes
BASELINES = ("zero", "rag", "oracle")
NULL_BASELINES = ("zero", "rag")  # the label-free references of the null comparison
WHITE_REFS = (*e5_infer.WHITE_BOX, "holdmem_D0B1_cited", "holdmem_D1B1_cited")
Z95 = 1.959964


def _index(rows: Iterable[Row]) -> dict[tuple[str, str], list[Row]]:
    idx: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        if r["cv"] is not None:
            idx[(r["system"], r["branch"])].append(r)
    return idx


def baseline_rows(white: dict[str, list[Row]], baseline_runs: list[Path] | None) -> list[Row]:
    """Item table of the baseline runs (zero, RAG, oracle), gold from the white-box probes and the
    baselines' own probes; empty without baseline runs."""
    answers: list[Row] = []
    probes: list[Row] = []
    for d in baseline_runs or []:
        answers += read_jsonl(d / "answers.jsonl")
        if (d / "probes.jsonl").exists():
            probes += read_jsonl(d / "probes.jsonl")
    if not answers:
        return []
    return e1.item_table({"probes": white["probes"] + probes, "answers": answers, "items": []})


def items(
    idx: dict[tuple[str, str], list[Row]],
    system: str,
    branch: str,
    worlds: Iterable[str],
    metric: str,
) -> dict[str, dict[str, float]]:
    """World -> item key -> ContractValid of the measurement-half items behind one metric."""
    keep = set(worlds)
    changed = metric == "Update"
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for r in idx.get((system, branch), ()):
        if r["world_id"] in keep and r["changed"] == changed and not inference_half(r["fact_id"]):
            out[r["world_id"]][f"{r['fact_id']}@{r['k']}"] = float(r["cv"])
    return out


class Intervals:
    """95% world-bootstrap intervals of (system, branch, world set), cached."""

    def __init__(self, idx: dict[tuple[str, str], list[Row]]) -> None:
        self.idx = idx
        self.cache: dict[tuple[str, str, frozenset[str]], dict[str, dict[str, float]]] = {}

    def get(self, system: str, branch: str, worlds: Iterable[str]) -> dict[str, dict[str, float]]:
        key = (system, branch, frozenset(worlds))
        if key not in self.cache:
            samples = e5.carry_samples(
                self.idx.get((system, branch), []), system, branch, set(key[2])
            )
            self.cache[key] = {m: world_bootstrap(by).to_json() for m, by in samples.items()}
        return self.cache[key]

    def present(self, system: str, branch: str, worlds: Iterable[str], metric: str) -> bool:
        return bool(self.get(system, branch, worlds)[metric]["n_items"])


def contains(iv: dict[str, float], value: float) -> bool:
    return bool(iv["lo"] <= value <= iv["hi"])


# ------------------------------------------------------------------------------ repeat ingestion
def ingestion_variance(
    first: dict[str, dict[str, float]], second: dict[str, dict[str, float]]
) -> dict[str, float] | None:
    """Variance of one ingestion's per-world mean from two ingestions of the same worlds: per
    world the difference of the means over the items both have, then mean(d_w^2) / 2."""
    diffs = []
    for w in sorted(first.keys() & second.keys()):
        keys = first[w].keys() & second[w].keys()
        if keys:
            diffs.append(sum(second[w][k] - first[w][k] for k in keys) / len(keys))
    if not diffs:
        return None
    var = sum(d * d for d in diffs) / len(diffs) / 2
    return {"variance": var, "sd": math.sqrt(var), "worlds": len(diffs)}


def noise_variances(
    idx: dict[tuple[str, str], list[Row]],
    white: dict[str, list[Row]],
    noise_runs: list[Path] | None,
) -> dict[tuple[str, str, str], dict[str, float]]:
    """(system, branch, metric) -> `ingestion_variance`, from the noise-floor runs: the second
    ingestion's answers against the first's on the worlds both have."""
    second = [a for d in noise_runs or [] for a in read_jsonl(d / "answers.jsonl")]
    if not second:
        return {}
    second_idx = _index(e1.item_table({"probes": white["probes"], "answers": second, "items": []}))
    out: dict[tuple[str, str, str], dict[str, float]] = {}
    for system in sorted({a["system"] for a in second}):
        worlds = {a["world_id"] for a in second if a["system"] == system}
        for branch in e5.BRANCHES:
            for m in METRICS:
                v = ingestion_variance(
                    items(idx, system, branch, worlds, m),
                    items(second_idx, system, branch, worlds, m),
                )
                if v is not None:
                    out[(system, branch, m)] = v
    return out


def paired(
    native: dict[str, dict[str, float]],
    ref: dict[str, dict[str, float]],
    noise: dict[str, float] | None,
) -> dict[str, Any] | None:
    """Item-paired native - reference difference, world-clustered, with the noise-inflated
    interval when the noise floor covers the system; None without paired items."""
    d = paired_diff(native, ref)
    if not d.n_items:
        return None
    out = d.to_json()
    out["excludes_zero"] = bool(d.lo > 0 or d.hi < 0)
    if noise is not None:
        n = {w: len(native[w].keys() & ref[w].keys()) for w in native if w in ref}
        total = sum(n.values())
        se = math.sqrt(noise["variance"] * sum((c / total) ** 2 for c in n.values()))
        lo = d.est - math.hypot(d.est - d.lo, Z95 * se)
        hi = d.est + math.hypot(d.hi - d.est, Z95 * se)
        out.update(
            {
                "ingestion_sd": noise["sd"],
                "ingestion_worlds": noise["worlds"],
                "ingestion_se": se,
                "lo_inflated": lo,
                "hi_inflated": hi,
                "excludes_zero_inflated": bool(lo > 0 or hi < 0),
            }
        )
    return out


# ------------------------------------------------------------------------------------ B labels
def b_identification(samples: dict[str, list[float]], label: int | None) -> dict[str, Any]:
    """Orphan persistence P with its 95% world-bootstrap interval; B = 1 if P <= threshold. The
    label is unidentified when the interval straddles the threshold or the probe had too few
    samples (`label` None)."""
    est = world_bootstrap(samples)
    thr = e5_infer.B_THRESHOLD
    straddles = bool(est.n_items and est.lo <= thr <= est.hi)
    return {
        "label": label,
        "rate": est.est,
        "lo": est.lo,
        "hi": est.hi,
        "n": est.n_items,
        "n_worlds": est.n_worlds,
        "threshold": thr,
        "straddles": straddles,
        "identified": bool(label is not None and est.n_items and not straddles),
    }


# ------------------------------------------------------------------------------------ one system
def _counts(flags: list[dict[str, bool]]) -> dict[str, int]:
    return {
        "comparisons": len(flags),
        "inside_predicted": sum(f["inside_predicted"] for f in flags),
        "inside_other": sum(f["inside_other"] for f in flags),
        "discriminating": sum(f["discriminates"] for f in flags),
    }


def _split(flags: list[tuple[str, dict[str, bool]]]) -> dict[str, dict[str, int]]:
    return {
        "all": _counts([f for _, f in flags]),
        "label_circular": _counts([f for b, f in flags if b in CIRCULAR_BRANCHES]),
        "out_of_sample": _counts([f for b, f in flags if b not in CIRCULAR_BRANCHES]),
    }


def label_cells(system: str) -> list[str]:
    """The four (D, B) label cells, at the binding granularity the system is compared at."""
    cells = [e5.predicted_cell(system, d, b) for d in (0, 1) for b in (0, 1)]
    return [c for c in cells if c is not None]


def system_entry(
    system: str,
    res: dict[str, Any],
    iv: Intervals,
    idx: dict[tuple[str, str], list[Row]],
    noise: dict[tuple[str, str, str], dict[str, float]],
    b_samples: dict[str, list[float]],
) -> dict[str, Any]:
    per = res["systems"][system]
    worlds = per["worlds"]
    cell, other = per.get("predicted_by"), per.get("other_label_cell")
    refs_white = [
        c
        for c in WHITE_REFS
        if any(iv.present(c, b, worlds, m) for b in e5.BRANCHES for m in METRICS)
    ]
    refs_base = [
        c
        for c in BASELINES
        if any(iv.present(c, b, worlds, m) for b in e5.BRANCHES for m in METRICS)
    ]
    comparisons: dict[str, Any] = {}
    flags: list[tuple[str, dict[str, bool]]] = []
    base_flags: dict[str, list[tuple[str, dict[str, bool]]]] = {b: [] for b in refs_base}
    native_on: dict[str, list[tuple[str, dict[str, bool]]]] = {b: [] for b in refs_base}
    label_share: dict[str, list[bool]] = {c: [] for c in label_cells(system)}
    share_all: list[float] = []
    paired_out: dict[str, Any] = {}
    for branch in e5.BRANCHES:
        br = per["branches"][branch]
        for m in METRICS:
            nat = br["native"][m]
            if not nat["n_items"]:
                continue
            key = f"{branch}:{m}"
            cons = (br.get("consistent") or {}).get(m)
            disc = (br.get("discrimination") or {}).get(m)
            refs: dict[str, Any] = {}
            for name in (*refs_white, *refs_base):
                r = iv.get(name, branch, worlds)[m]
                if r["n_items"]:
                    refs[name] = {
                        "kind": "baseline" if name in BASELINES else "white_box",
                        "est": r["est"],
                        "lo": r["lo"],
                        "hi": r["hi"],
                        "n_items": r["n_items"],
                        "contains": contains(r, nat["est"]),
                    }
            held = [n for n, r in refs.items() if r["contains"]]
            comparisons[key] = {
                "branch": branch,
                "metric": m,
                "label_circular": branch in CIRCULAR_BRANCHES,
                "native": {k: nat[k] for k in ("est", "lo", "hi", "n_items")},
                "references": refs,
                "contained_by": held,
                "n_contained": len(held),
                "n_references": len(refs),
                "inside_predicted": cons,
                "discriminates": None if disc is None else disc["discriminates"],
            }
            if disc is not None:
                flags.append((branch, disc))
            for c in label_share:
                if c in refs:
                    label_share[c].append(refs[c]["contains"])
            cells_here = [n for n in (*label_cells(system), *NULL_BASELINES) if n in refs]
            if cells_here:
                share_all.append(sum(refs[n]["contains"] for n in cells_here) / len(cells_here))
            # the label-free baselines under this system's label
            pred, alt = br.get("white_box", {}).get(m), br.get("white_box_other", {}).get(m)
            if pred and alt and pred["n_items"] and alt["n_items"]:
                for b in refs_base:
                    if b in refs:
                        inside = contains(pred, refs[b]["est"])
                        inside_alt = contains(alt, refs[b]["est"])
                        f = {
                            "inside_predicted": inside,
                            "inside_other": inside_alt,
                            "discriminates": inside and not inside_alt,
                        }
                        base_flags[b].append((branch, f))
                        if disc is not None:
                            native_on[b].append((branch, disc))
            # item-paired differences to the predicted cell and to the zero / RAG baselines
            nat_items = items(idx, system, branch, worlds, m)
            for ref in ([cell] if cell else []) + [b for b in NULL_BASELINES if b in refs_base]:
                p = paired(
                    nat_items, items(idx, ref, branch, worlds, m), noise.get((system, branch, m))
                )
                if p is not None:
                    paired_out.setdefault(key, {})[ref] = p
    coverage_cells = {c: (sum(v) / len(v) if v else None) for c, v in label_share.items()}
    defined = [v for v in coverage_cells.values() if v is not None]
    pairs = [x for v in label_share.values() for x in v]
    return {
        "predicted_by": cell,
        "other_label_cell": other,
        "b_identified": b_identification(b_samples, res["labels"][system]["B"]["label"]),
        "white_box_references": refs_white,
        "baseline_references": refs_base,
        "comparisons": comparisons,
        "counts": _split(flags),
        "baseline_counts": {
            b: {"baseline": _split(base_flags[b]), "native_same_comparisons": _split(native_on[b])}
            for b in refs_base
        },
        "chance_coverage": {
            "label_cells": coverage_cells,
            "labels": sum(defined) / len(defined) if defined else None,
            "labels_pooled": sum(pairs) / len(pairs) if pairs else None,
            "references": sum(share_all) / len(share_all) if share_all else None,
            "observed": _observed(per),
        },
        "paired_difference": paired_out,
    }


def _observed(per: dict[str, Any]) -> float | None:
    """Share of the comparisons with a predicted interval that lie inside it (`consistent`)."""
    flags = [ok for br in per["branches"].values() for ok in (br.get("consistent") or {}).values()]
    return sum(flags) / len(flags) if flags else None


def _total(entries: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Counts summed over the systems, and the pooled coverage."""
    keys = ("comparisons", "inside_predicted", "inside_other", "discriminating")
    out: dict[str, Any] = {
        "counts": {
            part: {k: sum(e["counts"][part][k] for e in entries.values()) for k in keys}
            for part in ("all", "label_circular", "out_of_sample")
        },
        "baseline_counts": {},
        "b_identified": sum(bool(e["b_identified"]["identified"]) for e in entries.values()),
        "systems": len(entries),
    }
    for b in BASELINES:
        have = [e for e in entries.values() if b in e["baseline_counts"]]
        if have:
            out["baseline_counts"][b] = {
                which: {
                    part: {
                        k: sum(e["baseline_counts"][b][which][part][k] for e in have) for k in keys
                    }
                    for part in ("all", "label_circular", "out_of_sample")
                }
                for which in ("baseline", "native_same_comparisons")
            }
    for name in ("labels", "references", "observed"):
        vals = [
            e["chance_coverage"][name]
            for e in entries.values()
            if e["chance_coverage"][name] is not None
        ]
        out[f"mean_{name}_coverage"] = sum(vals) / len(vals) if vals else None
    return out


def analyse(
    res: dict[str, Any],
    rows: list[Row],
    natives: dict[str, list[Row]],
    white: dict[str, list[Row]],
    noise_runs: list[Path] | None = None,
    baseline_runs: list[Path] | None = None,
) -> dict[str, Any]:
    """`explore_native_null` for every native system of `res` (`e5.analyse`'s result so far)."""
    idx = _index(rows + baseline_rows(white, baseline_runs))
    iv = Intervals(idx)
    noise = noise_variances(idx, white, noise_runs)
    samples = e5_infer.probe_samples(
        {
            "probes": white["probes"] + natives["probes"],
            "answers": natives["answers"],
            "items": [],
        }
    )
    entries = {
        s: system_entry(s, res, iv, idx, noise, samples.get(s, {}).get("B", {}))
        for s in sorted(res["systems"])
    }
    return {
        "definitions": {
            "inside_predicted": "native estimate inside the 95% interval of the predicted cell",
            "discriminating": "inside the predicted cell's interval, outside the other label's",
            "label_circular": "comparisons on the branches B is inferred from (circular_branches)",
            "chance_coverage": "share of the four label cells' 95% intervals that contain the "
            "native estimate, i.e. the coverage when the label is permuted",
            "inflated": "interval half-widths widened in quadrature by 1.96 x the standard error "
            "from the repeat-ingestion variance",
        },
        "circular_branches": sorted(CIRCULAR_BRANCHES),
        "systems": entries,
        "total": _total(entries),
    }

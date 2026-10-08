"""E5 analysis (plan §6 E5, docs/E5_PROTOCOL.md §3): the sufficient-statistic test on native
systems.

1. Labels: each native system's (D, B) from `analysis.e5_infer` on the inference half. B uses the
   orphan probe (validated, kappa_B = 1.0 on the white-box cells). D inference did not reach the
   kappa bar (decisions.md), so D is described only; for the prediction D comes from the export
   structure (no typed value with located spans -> D = 0).
2. Carryover on the measurement half (items outside the inference half): Update (changed items)
   and Stability (unchanged items) after S_src, S_def and the orphan branch, world bootstrap.
3. Prediction: the white-box HoldMem cell with the same (D, B) at source-epoch binding, on the
   same worlds, branches and items (E1 test runs); its 95% world-bootstrap interval is the
   predicted interval. A native estimate inside it is consistent with the sufficient statistic;
   outside, the deviation (direction and size) is reported as a boundary of the theory.
S_wr for native systems: a native store has no writer-model key and no re-derivation path, so a
writer switch changes no stored item (retention 1 by construction, as for B = 0); not measured.
4. Ingestion failures under on_error="skip" are system behaviour and reported as a rate.
5. Noise floor (plan §6 E5 (4)): a second, independent ingestion of some worlds; see noise_floor.
6. T3 / T4 for native systems (decisions.md 2026-09-29, T3 reference); see membership.
7. Discriminability (exploratory, not pre-registered): a native value inside the predicted interval
   supports the label only if the other B label would have missed it. Each comparison with a
   predicted interval is therefore also placed against the 95% interval of the white-box cell of
   the OTHER B label with the same D (`other_cell`: on the same worlds, branch and items; the
   cited-cluster cell where the system is document-level, as in `predicted_cell`), stored as
   `branches[b]["white_box_other"]` and `branches[b]["discrimination"][metric]`. A comparison
   DISCRIMINATES when the native point estimate lies inside the predicted cell's interval and
   outside the other label's interval. One inside both intervals says nothing about B; one inside
   the other label's interval and outside the predicted one contradicts the label. Counts per
   system and over all systems are in `stage4["discriminability"]`.
8. Null baselines (exploratory, decisions.md 2026-09-30; `ex_e5_null`, stored as
   `explore_native_null`): which reference intervals (every white-box cell, the zero system, plain
   RAG, the oracle) contain each native estimate, the inside / discriminating counts over the
   label-circular (orphan) and the out-of-sample comparisons, the counts a label-free baseline gets
   under the same rule, chance coverage, item-paired differences with the repeat-ingestion noise,
   and whether B is identified.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from typing import Any

from memshift.analysis import e1, e5_infer
from memshift.analysis.e3 import COST_CEILING, EPSILON
from memshift.experiments.common import read_jsonl
from memshift.gen.shifts import generate_branch, inference_half
from memshift.gen.state import SourceLog
from memshift.scoring.stats import paired_diff, world_bootstrap
from memshift.textutil import approx_tokens

Row = dict[str, Any]
BRANCHES = ("src", "def", "orphan")
# Stage 4 (prereg/v2.md §3): systems whose documented sync path binds each source object as one
# document are compared, when inferred B = 1, with the cited-cluster cell (object-level binding);
# every other system with the source-epoch cell. Labels predicted from documentation before any
# stage-4 run, (D, B):
DOC_LEVEL = frozenset({"lightrag", "hipporag", "cognee"})
DOCUMENTED_LABELS = {
    "mem0": (0, 0),
    "amem": (0, 0),
    "letta": (0, 0),
    "memos": (0, 0),
    "lightrag": (0, 1),
    "hipporag": (0, 1),
    "cognee": (0, 1),
    "simplemem": (0, 0),
}
N2_MIN_INSIDE = 4  # of 6 comparisons per system
N3_MIN_SHARE = 0.5  # of all comparisons across systems


N1_EXCLUDED = frozenset({"mem0"})  # N-1 counts the stage-4 systems only ("≥ 4 / 5")
N1_MIN_AGREE = 4


def predicted_cell(system: str, d: int | None, b: int | None) -> str | None:
    """The white-box cell whose interval predicts a native system with inferred label (D, B)."""
    if b is None or d is None:
        return None
    if not b:
        return f"holdmem_D{d}B0"
    return f"holdmem_D{d}B1_{'cited' if system in DOC_LEVEL else 'source'}"


def other_cell(system: str, d: int | None, b: int | None) -> str | None:
    """The white-box cell of the other B label (same D) than the one `predicted_cell` selects, at
    the binding granularity the system's B = 1 label is compared with."""
    if b is None or d is None:
        return None
    return predicted_cell(system, d, 1 - b)


def _inferred(lab: dict[str, Any]) -> tuple[int | None, int | None]:
    """(D, B) of a native system. D = 1 needs an export with typed values and located spans
    (E5_PROTOCOL §1), which no native system has (`analyse` passes structural = False), so D is 0
    by export structure even when too few structured-repair samples make the rate unidentifiable."""
    d = lab.get("D", {}).get("label")
    return (0 if d is None else d, lab["B"]["label"])


def stage4_summary(res: dict[str, Any]) -> dict[str, Any]:
    """Predictions N-1..N-4 of prereg/v2.md §3: N-1 over the stage-4 systems in `res`, N-2..N-4
    over every native system in `res` (Mem0 included, decisions.md 2026-09-29)."""
    systems = sorted(res["systems"])
    agree: dict[str, bool | None] = {}
    per_system = {}
    inside_all: list[bool] = []
    for s in systems:
        inferred = _inferred(res["labels"][s])
        if s not in N1_EXCLUDED:
            agree[s] = DOCUMENTED_LABELS.get(s) == inferred if s in DOCUMENTED_LABELS else None
        flags = [
            ok
            for br in res["systems"][s]["branches"].values()
            for ok in (br.get("consistent") or {}).values()
        ]
        inside_all += flags
        per_system[s] = {
            "inside": sum(flags),
            "comparisons": len(flags),
            "n2_holds": sum(flags) >= N2_MIN_INSIDE if flags else None,
        }
    disc_per_system = {}
    for s in systems:
        found = [
            d
            for br in res["systems"][s]["branches"].values()
            for d in (br.get("discrimination") or {}).values()
        ]
        disc_per_system[s] = {
            "comparisons": len(found),
            "inside_predicted": sum(d["inside_predicted"] for d in found),
            "inside_other": sum(d["inside_other"] for d in found),
            "discriminating": sum(d["discriminates"] for d in found),
        }
    disc_total = {
        k: sum(v[k] for v in disc_per_system.values())
        for k in next(iter(disc_per_system.values()), {})
    }
    by_b: dict[int, list[float]] = defaultdict(list)
    for s in systems:
        orphan = res["systems"][s]["branches"].get("orphan")
        b = res["labels"][s]["B"]["label"]
        v = orphan["native"]["Stability"]["est"] if orphan else float("nan")
        if b is not None and v == v:  # skip NaN
            by_b[b].append(v)
    n4 = None
    if by_b.get(0) and by_b.get(1):
        n4 = sum(by_b[1]) / len(by_b[1]) > sum(by_b[0]) / len(by_b[0])
    share = sum(inside_all) / len(inside_all) if inside_all else None
    return {
        "n1_label_agreement": agree,
        "n1_holds": sum(v is True for v in agree.values()) >= N1_MIN_AGREE if agree else None,
        "n2_per_system": per_system,
        "n3_share_inside": share,
        "n3_holds": share is not None and share >= N3_MIN_SHARE,
        "n4_orphan_stability_by_B": {str(k): v for k, v in by_b.items()},
        "n4_holds": n4,
        "discriminability": {
            "rule": "a comparison discriminates when the native estimate is inside the predicted "
            "cell's 95% interval and outside that of the other B label's cell (same D)",
            "per_system": disc_per_system,
            "total": disc_total,
        },
    }


def carry_samples(
    rows: list[Row], system: str, branch: str, worlds: set[str]
) -> dict[str, dict[str, list[float]]]:
    """Metric -> world -> ContractValid of the measurement-half items: Update (changed items) and
    Stability (unchanged items) of one system after one branch's shift."""
    out: dict[str, dict[str, list[float]]] = {}
    for metric, changed in (("Update", True), ("Stability", False)):
        by: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            if (
                r["system"] == system
                and r["branch"] == branch
                and r["world_id"] in worlds
                and r["cv"] is not None
                and r["changed"] == changed
                and not inference_half(r["fact_id"])
            ):
                by[r["world_id"]].append(float(r["cv"]))
        out[metric] = by
    return out


def _carry(rows: list[Row], system: str, branch: str, worlds: set[str]) -> dict[str, Any]:
    return {
        metric: world_bootstrap(by).to_json()
        for metric, by in carry_samples(rows, system, branch, worlds).items()
    }


def discrimination(
    native: dict[str, Any],
    other: dict[str, Any],
    consistent: dict[str, bool],
    metrics: list[str],
) -> dict[str, dict[str, bool]]:
    """Per metric: is the native estimate inside the predicted interval (`consistent`), inside the
    interval of the other B label's cell, and does the comparison discriminate (inside the
    predicted one and outside the other)."""
    out = {}
    for m in metrics:
        inside_other = bool(other[m]["lo"] <= native[m]["est"] <= other[m]["hi"])
        out[m] = {
            "inside_predicted": consistent[m],
            "inside_other": inside_other,
            "discriminates": consistent[m] and not inside_other,
        }
    return out


def failures(results: list[Row]) -> dict[str, dict[str, float]]:
    """Ingestion units that failed under on_error="skip", prefix plus every branch."""
    by: dict[str, dict[str, float]] = defaultdict(lambda: {"ingested": 0, "failed": 0})
    for r in results:
        branches = (r.get("branches") or {}).values()
        for st in [r.get("prefix_stats") or {}, *(b.get("stats") or {} for b in branches)]:
            by[r["system"]]["ingested"] += st.get("ingested", 0)
            by[r["system"]]["failed"] += st.get("failed", 0)
    return {
        s: {**v, "rate": v["failed"] / max(1, v["ingested"] + v["failed"])} for s, v in by.items()
    }


PARSE_ERROR = "Error parsing extraction response"  # Mem0 logs this and stores nothing for the unit


def silent_failures(run_dirs: list[Path], results: list[Row]) -> dict[str, dict[str, float]]:
    """Units a native library swallowed without raising: Mem0 logs PARSE_ERROR when its extraction
    output is not valid JSON (e.g. cut off at its default output limit) and stores no memory, so
    the unit counts as ingested. Counted from each run's container stderr (`stderr.txt`, fetched
    next to the run, not committed); runs without it are left out."""
    by: dict[str, dict[str, float]] = defaultdict(lambda: {"runs": 0, "parse_errors": 0})
    for d in run_dirs:
        log = d / "stderr.txt"
        if not log.exists():
            continue
        for system in {r["system"] for r in read_jsonl(d / "world_results.jsonl")}:
            by[system]["runs"] += 1
            by[system]["parse_errors"] += sum(PARSE_ERROR in line for line in log.open())
    units = failures(results)
    return {
        s: {**v, "rate": v["parse_errors"] / max(1, units.get(s, {}).get("ingested", 0))}
        for s, v in by.items()
    }


def noise_floor(first: list[Row], second: list[Row], probes: list[Row]) -> dict[str, Any]:
    """Two independent ingestions of the same worlds (plan §6 E5 (4), protocol §4). Native calls
    bypass the request cache and reach the server at temperature 0, so the runs differ only through
    server-side batching. Per system and branch on the shared worlds: the share of probe answers
    whose status or value differs, and each run's Update / Stability on the measurement half with
    the difference (second - first). `probes` must hold the k = 0 probe of every branch."""
    worlds = {a["world_id"] for a in first} & {a["world_id"] for a in second}
    runs = [[a for a in ans if a["world_id"] in worlds] for ans in (first, second)]
    by_key = [{(a["system"], *e1._key(a)): a for a in ans} for ans in runs]
    rows = [e1.item_table({"probes": probes, "answers": ans, "items": []}) for ans in runs]
    out: dict[str, Any] = {"worlds": sorted(worlds), "systems": {}}
    for s in sorted({k[0] for k in by_key[1]}):
        per: dict[str, Any] = {}
        for branch in ("natural", *BRANCHES):
            keys = [k for k in by_key[0] if k in by_key[1] and k[0] == s and k[2] == branch]
            if not keys:
                continue
            a, b = ([m[k] for k in keys] for m in by_key)
            differ = sum(
                (x["status"], x["value"]) != (y["status"], y["value"])
                for x, y in zip(a, b, strict=True)
            )
            entry: dict[str, Any] = {
                "answers": len(keys),
                "answers_differ": differ,
                "share_differ": differ / len(keys),
                "cv_first": sum(x["contract_valid"] for x in a) / len(a),
                "cv_second": sum(y["contract_valid"] for y in b) / len(b),
            }
            if branch != "natural":
                c1, c2 = (_carry(r, s, branch, worlds) for r in rows)
                entry.update({"first": c1, "second": c2})
                entry["diff"] = {
                    m: c2[m]["est"] - c1[m]["est"]
                    for m in ("Update", "Stability")
                    if c1[m]["n_items"] and c2[m]["n_items"]
                }
            per[branch] = entry
        out["systems"][s] = per
    return out


def full_reread_tokens(world_ids: set[str], params: dict[str, Any] | None) -> dict[str, int]:
    """Tokens of every live source object at t0 (one full re-read), per world (E3's formula)."""
    out = {}
    for w in sorted(world_ids):
        world = generate_branch(int(w.lstrip("w")), "natural", params)
        snap = SourceLog(world.events).snapshot(world.fork - timedelta(minutes=1))
        out[w] = sum(approx_tokens(o.text) for o in snap.all())
    return out


def membership(
    answers: list[Row], baselines: list[Row], params: dict[str, Any] | None
) -> dict[str, Any]:
    """T3 and T4 on the native worlds (decisions.md 2026-09-29, T3 reference), with E3's criteria.
    T3: static ContractValid at t0 minus oracle context on the same items, clustered by world,
    passes if >= -eps. T4: reader tokens per query over one full re-read at t0, passes if <= c.
    Native runs answer t0 once (natural world); baseline runs answer it on the src branch."""
    cv: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    tokens: dict[str, list[int]] = defaultdict(list)
    for a in answers + baselines:
        if int(a["tags"].get("k", 0)) != 0 or a["branch"] not in ("natural", "src"):
            continue
        cv[a["system"]][a["world_id"]][a["fact_id"]] = float(a["contract_valid"])
        tokens[a["system"]].append(a["prompt_tokens"] + a["completion_tokens"])
    if "oracle" not in cv:
        return {}
    worlds = set(cv["oracle"])
    reread = full_reread_tokens(worlds, params)
    per_reread = sum(reread.values()) / len(reread)
    out: dict[str, Any] = {"worlds": sorted(worlds), "full_reread_tokens_per_query": per_reread}
    for s in sorted(cv):
        mine = {w: v for w, v in cv[s].items() if w in worlds}
        per_query = sum(tokens[s]) / max(1, len(tokens[s]))
        entry: dict[str, Any] = {
            "static": world_bootstrap({w: list(v.values()) for w, v in mine.items()}).to_json(),
            "reader_tokens_per_query": per_query,
            "serve_ratio_to_full_reread": per_query / per_reread,
            "T4_pass": per_query / per_reread <= COST_CEILING,
        }
        if s != "oracle":
            d = paired_diff(mine, cv["oracle"]).to_json()
            entry.update({"minus_oracle": d, "T3_pass": d["est"] >= -EPSILON})
        out[s] = entry
    return out


def exploratory(
    natives: dict[str, list[Row]], white: dict[str, list[Row]], rows: list[Row]
) -> dict[str, Any]:
    """Not pre-registered (added after the E5 results): world-bootstrap intervals of the label
    rates (orphan persistence P, repair precision R), the white-box cells' rates on the same worlds,
    and every white-box cell's carryover on the native worlds, so a native system can be placed
    between the cells instead of only against the one its binary label selects."""
    worlds = {a["world_id"] for a in natives["answers"]}
    rates = {
        s: {dim: world_bootstrap(v[dim]).to_json() for dim in ("B", "D")}
        for s, v in e5_infer.probe_samples(
            {
                "probes": white["probes"] + natives["probes"],
                "answers": white["answers"] + natives["answers"],
                "items": [],
            }
        ).items()
    }
    cells = {
        c: {b: _carry(rows, c, b, worlds) for b in BRANCHES} for c in sorted(e5_infer.WHITE_BOX)
    }
    return {"label_rates": rates, "white_box_cells": cells}


def analyse(
    e5_runs: list[Path],
    e1_runs: list[Path],
    out: Path | None = None,
    noise_runs: list[Path] | None = None,
    baseline_runs: list[Path] | None = None,
) -> dict[str, Any]:
    natives: dict[str, list[Row]] = {"probes": [], "answers": [], "items": []}
    results: list[Row] = []
    for d in e5_runs:
        natives["probes"] += read_jsonl(d / "probes.jsonl")
        natives["answers"] += read_jsonl(d / "answers.jsonl")
        results += read_jsonl(d / "world_results.jsonl")
    white = e1.load(*e1_runs)
    systems = sorted({a["system"] for a in natives["answers"]})
    identifiable = {r["system"]: bool(r.get("identifiable", True)) for r in results}
    labels = e5_infer.infer(natives, {s: False for s in systems}, identifiable)
    merged = {k: white[k] + natives[k] for k in ("probes", "answers", "items")}
    rows = e1.item_table(merged)
    res: dict[str, Any] = {
        "labels": labels,
        "failures": failures(results),
        "silent_failures": silent_failures(e5_runs, results),
        "systems": {},
    }
    if noise_runs:
        second = [a for d in noise_runs for a in read_jsonl(d / "answers.jsonl")]
        res["noise_floor"] = noise_floor(natives["answers"], second, white["probes"])
    if baseline_runs:
        base = [a for d in baseline_runs for a in read_jsonl(d / "answers.jsonl")]
        params = json.loads((e5_runs[0] / "manifest.json").read_text())["params"]["world_params"]
        res["membership"] = membership(natives["answers"], base, params or None)
    for s in systems:
        worlds = {a["world_id"] for a in natives["answers"] if a["system"] == s}
        cell = predicted_cell(s, *_inferred(labels[s]))
        other = other_cell(s, *_inferred(labels[s]))
        per: dict[str, Any] = {"worlds": sorted(worlds), "predicted_by": cell, "branches": {}}
        if other is not None:
            per["other_label_cell"] = other
        for branch in BRANCHES:
            nat = _carry(rows, s, branch, worlds)
            entry: dict[str, Any] = {"native": nat}
            if cell is not None:
                ref = _carry(rows, cell, branch, worlds)
                entry["white_box"] = ref
                both = [
                    m for m in ("Update", "Stability") if nat[m]["n_items"] and ref[m]["n_items"]
                ]
                entry["consistent"] = {
                    m: bool(ref[m]["lo"] <= nat[m]["est"] <= ref[m]["hi"]) for m in both
                }
                entry["deviation"] = {m: nat[m]["est"] - ref[m]["est"] for m in both}
                if other is not None:
                    entry["white_box_other"] = alt = _carry(rows, other, branch, worlds)
                    entry["discrimination"] = discrimination(
                        nat, alt, entry["consistent"], [m for m in both if alt[m]["n_items"]]
                    )
            per["branches"][branch] = entry
        res["systems"][s] = per
    res["exploratory"] = exploratory(natives, white, rows)
    # Exploratory (decisions.md 2026-09-30): null baselines, noise-inflated paired differences,
    # label-circular against out-of-sample counts, chance coverage, B identified
    from memshift.analysis import ex_e5_null

    res["explore_native_null"] = ex_e5_null.analyse(
        res, rows, natives, white, noise_runs, baseline_runs
    )
    res["stage4"] = stage4_summary(res)
    adir = (out or e5_runs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "e5_natives.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--e5", nargs="+", required=True)
    ap.add_argument("--e1", nargs="+", required=True)
    ap.add_argument("--noise", nargs="*", default=[])
    ap.add_argument("--baselines", nargs="*", default=[])
    a = ap.parse_args()
    r = analyse(
        [Path(p) for p in a.e5],
        [Path(p) for p in a.e1],
        Path(a.out),
        [Path(p) for p in a.noise],
        [Path(p) for p in a.baselines],
    )
    print(json.dumps({"labels": r["labels"]}, indent=1))

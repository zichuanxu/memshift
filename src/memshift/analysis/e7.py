"""E7 analysis: failure analysis. All rates are world-bootstrap estimates.

conflict  AbstainAccuracy = P(status abstain | conflicting item) per system (gold is abstain for
          an unresolvable conflict); controls: ContractValid on the same number of
          non-conflicting contact items.
cons      after revoking one agent's scope (S_cons), on the revoked agent's questions:
          ScopeLeak = P(answered | gold blocked); BlockAccuracy = P(blocked | gold blocked);
          OverBlock = P(blocked | gold not blocked).
readset   S2, the read set that cannot be declared: MissedInvalidation of scope-predicate items
          (definition items, `k.*`, whose gold ranges over every object satisfying the
          definition) against all other items, per HoldMem B=1 cell, from E1 item tables. A
          predicate's scope includes objects that do not exist yet, so a key built from the
          cited clusters cannot contain them; the source epoch does.
theta_r   external validity of the reader (E1-style runs with `--reader`) and, as theta_e, of the
theta_e   RAG embedding (`--cells none --embed`): per-system metrics and the E1
          contrasts under the main and the swapped model, on the swap runs' worlds and probes.
          M3 reader swaps over several models: `theta_r_by_reader` holds one swap
          per reader model, grouped by the model each run's manifest records.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.experiments.common import read_jsonl
from memshift.scoring.stats import world_bootstrap

Row = dict[str, Any]


def _rate(rows: list[Row], cond: Any, outcome: Any) -> dict[str, float]:
    by: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if cond(r):
            by[r["world_id"]].append(float(outcome(r)))
    return world_bootstrap(by).to_json()


def blocked(a: Row) -> bool:
    return bool(a["gold_status"] == "blocked")


def _not_blocked(a: Row) -> bool:
    return not blocked(a)


def conflict_cons(answers: list[Row]) -> dict[str, Any]:
    out: dict[str, Any] = {"conflict": {}, "cons": {}}
    for system in sorted({a["system"] for a in answers}):
        rs = [a for a in answers if a["system"] == system]
        con = [a for a in rs if a["tags"].get("e7") == "conflict"]
        if con:
            out["conflict"][system] = {
                "AbstainAccuracy": _rate(
                    con, lambda a: a["tags"]["conflict"], lambda a: a["status"] == "abstain"
                ),
                "ControlContractValid": _rate(
                    con, lambda a: not a["tags"]["conflict"], lambda a: a["contract_valid"]
                ),
            }
        cons = [a for a in rs if a["tags"].get("e7") == "cons"]
        if cons:
            out["cons"][system] = {
                "ScopeLeak": _rate(cons, blocked, lambda a: a["status"] == "answer"),
                "BlockAccuracy": _rate(cons, blocked, lambda a: a["status"] == "blocked"),
                "OverBlock": _rate(cons, _not_blocked, lambda a: a["status"] == "blocked"),
            }
    return out


def readset(rows: list[Row]) -> dict[str, Any]:
    """MissedInvalidation of predicate (k.*) vs other items per B=1 cell and branch."""
    out: dict[str, Any] = {}
    cells = sorted({r["system"] for r in rows if "B1" in r["system"] and "detected" in r})
    for branch in sorted({r["branch"] for r in rows}):
        for cell in cells:
            rs = [r for r in rows if r["system"] == cell and r["branch"] == branch]
            for cls, cond in (
                ("predicate", lambda r: r["fact_id"].startswith("k.")),
                ("other", lambda r: not r["fact_id"].startswith("k.")),
            ):
                out[f"{branch}:{cell}:{cls}"] = _rate(
                    rs,
                    lambda r, c=cond: r["changed"] and c(r),
                    lambda r: not r["detected"],
                )
    return out


SWAP_CONTRASTS = (
    "B_effect_update_pooled_source",
    "D_effect_missed_source",
    "D_effect_repair_source",
    "BxD_interaction_update_source",
)


def same_sign(main: dict[str, Any] | None, swap: dict[str, Any] | None) -> bool | None:
    """M3: the swapped reader's point estimate has the main reader's sign; None when
    either is missing or the main estimate is exactly 0 (a contrast that is zero by
    construction, such as the D effect on missed invalidation, has no sign to keep)."""
    if not main or not swap or main.get("est") in (None, 0) or swap.get("est") is None:
        return None
    return bool((main["est"] > 0) == (swap["est"] > 0) and swap["est"] != 0)


def swap(main_runs: list[Path], swap_runs: list[Path]) -> dict[str, Any]:
    """Metrics and contrasts under the main and the swapped model, on the swap runs' worlds."""
    sw = e1.item_table(e1.load(*swap_runs))
    worlds = {r["world_id"] for r in sw}
    mn = [r for r in e1.item_table(e1.load(*main_runs)) if r["world_id"] in worlds]
    has_cells = any(r["system"].startswith("holdmem") for r in sw)
    out: dict[str, Any] = {"worlds": sorted(worlds)}
    for b in sorted({r["branch"] for r in sw}):
        systems = sorted({r["system"] for r in sw})
        entry: dict[str, Any] = {
            "systems": {
                s: {"main": e1.cell_metrics(mn, s, b), "swap": e1.cell_metrics(sw, s, b)}
                for s in systems
            }
        }
        if has_cells:
            cm, cs = e1.contrasts(mn, b), e1.contrasts(sw, b)
            entry["contrasts"] = {
                k: {"main": cm.get(k), "swap": cs[k], "same_sign": same_sign(cm.get(k), cs[k])}
                for k in SWAP_CONTRASTS
                if k in cs
            }
        out[b] = entry
    return out


def reader_groups(runs: list[Path]) -> dict[str, list[Path]]:
    """Swap runs grouped by the reader model each run's manifest records (`models.reader.model`);
    empty when any run has no such record (its reader is then unknown)."""
    groups: dict[str, list[Path]] = defaultdict(list)
    for d in runs:
        manifest = d / "manifest.json"
        model = (
            ((json.loads(manifest.read_text()).get("models") or {}).get("reader") or {}).get(
                "model"
            )
            if manifest.exists()
            else None
        )
        if model is None:
            return {}
        groups[model].append(d)
    return dict(sorted(groups.items()))


def analyse(
    e7_runs: list[Path],
    e1_runs: list[Path] | None = None,
    out: Path | None = None,
    theta_r: list[Path] | None = None,
    theta_e: list[Path] | None = None,
) -> dict[str, Any]:
    answers = [a for d in e7_runs for a in read_jsonl(d / "answers.jsonl")]
    res: dict[str, Any] = conflict_cons(answers)
    for d in e7_runs:  # the implicit-definition ceiling is computed by the driver from gold
        summary = d / "analysis" / "e7_summary.json"
        if summary.exists() and "implicit" in (sj := json.loads(summary.read_text())):
            res.setdefault("implicit", sj["implicit"])
    if e1_runs:
        res["readset"] = readset(e1.item_table(e1.load(*e1_runs)))
        if theta_r:
            res["theta_r"] = swap(e1_runs, theta_r)
            # several readers (block M3): one swap per reader model, so the readers do not mix
            if by_reader := reader_groups(theta_r):
                res["theta_r_by_reader"] = {m: swap(e1_runs, rs) for m, rs in by_reader.items()}
        if theta_e:
            res["theta_e"] = swap(e1_runs, theta_e)
    adir = (out or e7_runs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "e7_metrics.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--e7", nargs="+", required=True)
    ap.add_argument("--e1", nargs="*", default=[])
    ap.add_argument("--theta-r", nargs="*", default=[], help="E1-style runs with another reader")
    ap.add_argument("--theta-e", nargs="*", default=[], help="E1-style runs with another embedding")
    a = ap.parse_args()
    r = analyse(
        [Path(p) for p in a.e7],
        [Path(p) for p in a.e1],
        Path(a.out),
        [Path(p) for p in a.theta_r],
        [Path(p) for p in a.theta_e],
    )
    print(json.dumps(r, indent=1)[:4000])

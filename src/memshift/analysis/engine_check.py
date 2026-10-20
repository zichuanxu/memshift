"""Serving-engine effect on identical reader requests (E7 theta_r).

The same theta_r cells were run twice with the same reader model and revision, once per vLLM image.
Writer outputs come from the same request cache, so every probe's reader context should be
byte-identical; any difference in answers is the engine's. Answers are matched on
(system, world_id, branch, probe_id).

Usage: python -m memshift.analysis.stage engine OUT old:RUN [old:RUN ...] new:RUN [new:RUN ...]
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

Key = tuple[str, str, str, str]


def _answers(runs: list[Path]) -> dict[Key, dict[str, Any]]:
    out: dict[Key, dict[str, Any]] = {}
    for run in runs:
        with (run / "answers.jsonl").open() as f:
            for line in f:
                a = json.loads(line)
                out[(a["system"], a["world_id"], a["branch"], a["probe_id"])] = a
    return out


def _answer(a: dict[str, Any]) -> tuple[str, str]:
    return a["status"], json.dumps(a["value"], sort_keys=True)


def compare(old: dict[Key, dict[str, Any]], new: dict[Key, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(old.keys() & new.keys())
    n = len(keys)
    cells: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])  # n, cv old, cv new, same
    same_ctx = same_ans = same_cv = 0
    for k in keys:
        a, b = old[k], new[k]
        same = _answer(a) == _answer(b)
        same_ctx += a["context_sha256"] == b["context_sha256"]
        same_ans += same
        same_cv += a["contract_valid"] == b["contract_valid"]
        c = cells[f"{k[0]}:{k[2]}"]
        c[0] += 1
        c[1] += bool(a["contract_valid"])
        c[2] += bool(b["contract_valid"])
        c[3] += same
    per_cell = {
        cell: {
            "n": m,
            "contract_valid_old": cv_old / m,
            "contract_valid_new": cv_new / m,
            "diff_pp": 100 * (cv_new - cv_old) / m,
            "same_answer": same / m,
        }
        for cell, (m, cv_old, cv_new, same) in sorted(cells.items())
    }
    return {
        "n_old": len(old),
        "n_new": len(new),
        "n_matched": n,
        "same_context_share": same_ctx / n if n else None,
        "same_answer_share": same_ans / n if n else None,
        "same_contract_valid_share": same_cv / n if n else None,
        "max_abs_cell_diff_pp": max((abs(c["diff_pp"]) for c in per_cell.values()), default=None),
        "per_system_branch": per_cell,
    }


def analyse(old: list[Path], new: list[Path], out: Path) -> dict[str, Any]:
    res = compare(_answers(old), _answers(new))
    res["old_runs"] = [r.name for r in old]
    res["new_runs"] = [r.name for r in new]
    (out / "analysis").mkdir(parents=True, exist_ok=True)
    (out / "analysis" / "engine_check.json").write_text(json.dumps(res, indent=2, sort_keys=True))
    return res

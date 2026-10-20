"""Black-box (D, B) inference, applied identically to the white-box HoldMem cells and to native
systems; inputs are E1-format answers and probes.

Only the inference half of the items (`gen.shifts.inference_half`) is used; carryover is measured
on the other half.
  B probe (orphan test): orphan branch, first checkpoint after the fork (k = 1), structured items
    (series, task due date, contact attribute) whose gold went from an answer to `abstain`
    because their source object was deleted. Persistence = the system still answers with the
    pre-fork gold value. P <= B_THRESHOLD -> B = 1, else B = 0.
  D probe (structured repair precision): S_src branch, k > 0, structured items whose gold value
    changed, restricted to answers where the system gave a value other than the pre-fork one.
    R = P(ContractValid | new value). D = 1 when R >= D_THRESHOLD and the system's export keeps a
    typed value with locatable source spans (`structural`), else D = 0.
A label is `None` (unidentifiable) when its probe has fewer than MIN_N samples, or when the
configuration keeps no raw text / source metadata (`identifiable=False`).

White-box validation: each HoldMem cell at source-epoch binding gets one label per dimension from
its pooled development sample; the primary confusion matrix is the (D, B) -> (D, B) table over the
four cells and Cohen's kappa per dimension is computed over those four labels. A dimension with
kappa < KAPPA_MIN is only described for native systems. As a supplement, every cell is also
labelled per world (units = cell x world).
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from memshift.analysis.e1 import load
from memshift.gen.shifts import inference_half
from memshift.scoring.normalize import values_match

B_THRESHOLD = 0.5
D_THRESHOLD = 0.98
MIN_N = 20
KAPPA_MIN = 0.7
STRUCTURED = ("c.", "s.", "t.")  # contact attribute, recurring series, task due date
WHITE_BOX = {
    "holdmem_D0B0": (0, 0),
    "holdmem_D1B0": (1, 0),
    "holdmem_D0B1_source": (0, 1),
    "holdmem_D1B1_source": (1, 1),
}
Row = dict[str, Any]


def _eligible(fact_id: str) -> bool:
    return fact_id.startswith(STRUCTURED) and inference_half(fact_id)


def probe_samples(data: dict[str, list[Row]]) -> dict[str, dict[str, dict[str, list[float]]]]:
    """system -> probe ('B', 'D') -> world -> per-probe outcomes (persisted / ContractValid)."""
    gold: dict[tuple[str, str, str, int], Row] = {}
    for p in data["probes"]:
        gold[(p["world_id"], p["branch"], p["fact_id"], int(p["tags"].get("k", 0)))] = p
    out: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for a in data["answers"]:
        k = int(a["tags"].get("k", 0))
        if k == 0 or not _eligible(a["fact_id"]):
            continue
        # White-box runs probe each branch at t0; native runs probe once, on the natural world,
        # before the branches diverge (same world and gold at t0).
        pre = gold.get((a["world_id"], a["branch"], a["fact_id"], 0)) or gold.get(
            (a["world_id"], "natural", a["fact_id"], 0)
        )
        if pre is None or pre["gold_status"] != "answer":
            continue
        old = pre["gold_value"]
        answered_old = a["status"] == "answer" and values_match(a["vtype"], a["value"], old)
        if a["branch"] == "orphan" and k == 1 and a["gold_status"] == "abstain":
            out[a["system"]]["B"][a["world_id"]].append(float(answered_old))
        elif (
            a["branch"] == "src"
            and a["gold_status"] == "answer"
            and not values_match(a["vtype"], a["gold_value"], old)
            and a["status"] == "answer"
            and not answered_old
        ):
            out[a["system"]]["D"][a["world_id"]].append(float(a["contract_valid"]))
    return out


def _rate(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def label(
    samples: dict[str, list[float]],
    dim: str,
    structural: bool,
    identifiable: bool = True,
    min_n: int = MIN_N,
) -> dict[str, Any]:
    xs = [x for v in samples.values() for x in v]
    rate = _rate(xs)
    if not identifiable or len(xs) < min_n:
        value = None
    elif dim == "B":
        value = int(rate <= B_THRESHOLD)
    else:
        value = int(rate >= D_THRESHOLD and structural)
    return {"label": value, "rate": rate, "n": len(xs)}


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's kappa of (true, inferred) binary labels; nan when chance agreement is 1."""
    n = len(pairs)
    if not n:
        return math.nan
    po = sum(t == i for t, i in pairs) / n
    pt, pi = sum(t for t, _ in pairs) / n, sum(i for _, i in pairs) / n
    pe = pt * pi + (1 - pt) * (1 - pi)
    return (po - pe) / (1 - pe) if pe < 1 else math.nan


def structural_from_items(data: dict[str, list[Row]]) -> dict[str, bool]:
    """HoldMem export structure: an item keeps a typed value and located spans (D = 1 cells)."""
    out: dict[str, bool] = defaultdict(bool)
    for it in data["items"]:
        det = it.get("det") or {}
        if det.get("parse") is not None and det.get("spans"):
            out[it["system"]] = True
    return out


def infer(
    data: dict[str, list[Row]],
    structural: dict[str, bool],
    identifiable: dict[str, bool] | None = None,
) -> dict[str, Any]:
    samples = probe_samples(data)
    ident = identifiable or {}
    return {
        s: {
            dim: label(samples[s][dim], dim, structural.get(s, False), ident.get(s, True))
            for dim in ("B", "D")
        }
        for s in sorted(samples)
    }


def validate(data: dict[str, list[Row]]) -> dict[str, Any]:
    samples = probe_samples(data)
    structural = structural_from_items(data)
    pooled = infer(data, structural)
    per_world: dict[str, list[tuple[int, int]]] = {"B": [], "D": []}
    units: list[dict[str, Any]] = []
    for cell, (d, b) in WHITE_BOX.items():
        if cell not in samples:
            continue
        truth = {"D": d, "B": b}
        worlds = sorted(set(samples[cell]["B"]) | set(samples[cell]["D"]))
        for w in worlds:
            unit: dict[str, Any] = {"cell": cell, "world_id": w}
            for dim in ("B", "D"):
                lab = label(
                    {w: samples[cell][dim].get(w, [])},
                    dim,
                    structural.get(cell, False),
                    min_n=1,
                )
                unit[dim] = lab
                if pooled[cell][dim]["label"] is not None and lab["label"] is not None:
                    per_world[dim].append((truth[dim], lab["label"]))
            units.append(unit)
    cells = [c for c in WHITE_BOX if c in pooled]
    joint: dict[str, str] = {}
    per_cell: dict[str, list[tuple[int, int]]] = {"B": [], "D": []}
    for c in cells:
        d, b = WHITE_BOX[c]
        inferred = {dim: pooled[c][dim]["label"] for dim in ("B", "D")}
        joint[f"D{d}B{b}"] = "D{}B{}".format(
            *("?" if inferred[x] is None else inferred[x] for x in ("D", "B"))
        )
        for dim, t in (("D", d), ("B", b)):
            if inferred[dim] is not None:
                per_cell[dim].append((t, inferred[dim]))
    kappa = {dim: cohen_kappa(per_cell[dim]) for dim in ("B", "D")}
    return {
        "pooled": {c: pooled[c] for c in cells},
        "truth": {c: {"D": d, "B": b} for c, (d, b) in WHITE_BOX.items()},
        "confusion_cells": joint,
        "kappa": kappa,
        "usable_for_natives": {dim: bool(kappa[dim] >= KAPPA_MIN) for dim in kappa},
        "supplement_cell_x_world": {
            "confusion": {
                dim: {
                    f"true{t}_inferred{i}": sum(p == (t, i) for p in per_world[dim])
                    for t in (0, 1)
                    for i in (0, 1)
                }
                for dim in ("B", "D")
            },
            "kappa": {dim: cohen_kappa(per_world[dim]) for dim in ("B", "D")},
        },
        "units": units,
    }


def analyse(*run_dirs: Path, out: Path | None = None) -> dict[str, Any]:
    res = validate(load(*run_dirs))
    adir = (out or run_dirs[0]) / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    (adir / "e5_whitebox_validation.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    import sys

    # python -m memshift.analysis.e5_infer OUT_DIR RUN_DIR [RUN_DIR ...]
    r = analyse(*(Path(p) for p in sys.argv[2:]), out=Path(sys.argv[1]))
    keys = ("confusion_cells", "kappa", "usable_for_natives", "supplement_cell_x_world")
    print(json.dumps({k: r[k] for k in keys}, indent=1))

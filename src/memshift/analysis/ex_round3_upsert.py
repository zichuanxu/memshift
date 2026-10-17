"""Where a slot-keyed upsert can act (block U of prereg/v4_round3.md; exploratory, post hoc; no
model call). It changes no registered contrast, margin, threshold or verdict.

The upsert cell (`holdmem.system`, B0_upsert) sends the requests of the append-only B=0 cell and,
when the delta derivation yields a new item for a slot, retires the slot's older items. So its
store differs from the B=0 store only on the slots for which the delta yielded a new item. Per D
arm (D=0, D=1, D_m), branch and item class, on the changed item-times that the upsert cell and the
committed B=0 and B=1 source-epoch cells all answered (those of `round3_registered.ratio_quantity`,
whose U3 ratio this explains), the item-times are split by whether the upsert cell holds an item of
the slot written after the fork and no later than the checkpoint (`yielded`; the B=0 cell derives
the same items from the same requests) or not (`none`). Per part: the item-times and their share,
Update of B=0, of the upsert and of B=1 (pooled mean, world-bootstrap interval, `ex_e1_classes`),
the part's contribution to binding's effect (share x (B1 - B0), the two parts summing to the
effect), and the item-times on which the upsert's ContractValid is above or below B=0's. Also, per
upsert cell, branch and class, the supersede events after the fork (old items the upsert retired).

    python -m memshift.analysis.ex_round3_upsert --runs RUN [RUN ...] \\
        --compare RUN_OR_STAGE [...] --out FILE.json

--runs: the upsert runs of block U (refresh runs may be passed too; only the upsert cells are
read). --compare: the committed runs or stages they pair with, as for `round3_registered U`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import round3_registered as rr
from memshift.runinfo import code_state

STATUS = (
    "exploratory, post hoc: explains the registered U3 ratio and changes no registered contrast, "
    "margin, threshold or verdict"
)
UNITS = rr.UNITS
PARTS = ("yielded", "none")  # the delta yielded a new item for the slot since the fork, or not
CLASSES = ("definition", "other")


def checkpoint_times(runs: Iterable[rr.Run]) -> dict[tuple[str, str, int], str]:
    """(world, branch, k) -> the as-of time of the probes at checkpoint k (k = 0: the fork)."""
    out: dict[tuple[str, str, int], str] = {}
    for run in runs:
        for p in run.probes:
            out.setdefault((p["world_id"], p["branch"], rr._k(p)), p["as_of"])
    return out


def written_times(items: Iterable[rr.Row]) -> dict[tuple[str, str, str, str], list[str]]:
    """(system, world, branch, slot) -> the write times of the slot's items."""
    out: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for it in items:
        out[(it["system"], it["world_id"], it["branch"], it["slot_id"])].append(it["written"])
    return out


def yielded(times: Sequence[str], fork: str, at: str) -> bool:
    """Whether an item was written after the fork and no later than the checkpoint."""
    return any(fork < t <= at for t in times)


def supersede_counts(
    items: Iterable[rr.Row], asof: Mapping[tuple[str, str, int], str]
) -> dict[str, dict[str, dict[str, int]]]:
    """Per upsert cell, branch and class: supersede events after the fork."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for it in items:
        fork = asof.get((it["world_id"], it["branch"], 0))
        if fork is None:
            continue
        cls = "definition" if str(it["slot_id"]).startswith("k.") else "other"
        n = sum(h["event"] == "supersede" and h["ts"] > fork for h in it["history"])
        by_b = out.setdefault(it["system"], {})
        by_b.setdefault(it["branch"], dict.fromkeys(CLASSES, 0))[cls] += n
    return out


def split(
    x: rr.Vec,
    b0: rr.Vec,
    b1: rr.Vec,
    flag: Mapping[str, Mapping[str, bool]],
    draws: int = rr.DRAWS,
) -> dict[str, Any]:
    """The changed item-times all three cells answered, split by `flag` (world -> item -> the
    delta yielded a new item): per part its levels, share, contribution and answer changes."""
    keys = rr._common(x, b0, b1)
    total = rr._n_items(keys)
    out: dict[str, Any] = {"n_items": total, "parts": {}}
    effect = 0.0
    for part in PARTS:
        want = part == "yielded"
        sub = {w: {k for k in ks if flag[w][k] == want} for w, ks in keys.items()}
        sub = {w: ks for w, ks in sub.items() if ks}
        n = rr._n_items(sub)
        xv, zv, ov = (rr._restrict(v, sub) for v in (x, b0, b1))
        up = sum(xv[w][k] > zv[w][k] for w in sub for k in sub[w])
        down = sum(xv[w][k] < zv[w][k] for w in sub for k in sub[w])
        gap = sum(ov[w][k] - zv[w][k] for w in sub for k in sub[w])
        contribution = gap / total if total else None
        effect += contribution or 0.0
        out["parts"][part] = {
            "n_items": n,
            "share": n / total if total else None,
            "update": {
                "B0": ex.vec_mean(zv, draws) if n else rr.nan_est(),
                "upsert": ex.vec_mean(xv, draws) if n else rr.nan_est(),
                "B1": ex.vec_mean(ov, draws) if n else rr.nan_est(),
            },
            "contribution_to_B_effect": contribution,
            "upsert_differs": {"above_B0": int(up), "below_B0": int(down)},
        }
    out["B_effect"] = effect if total else None
    return out


def analyse(runs: Sequence[Path], compare: Sequence[Path], draws: int = rr.DRAWS) -> dict[str, Any]:
    new, skipped = rr.read_runs(runs, items=lambda s: s.endswith("_upsert"))
    need = set(rr.ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = rr.read_runs(rr.expand(compare), need.__contains__)
    newm, cmpm = rr.Merged(new, rr.by_family), rr.Merged(cmp_runs, rr.by_family)
    pair = rr.pairing(newm, cmpm, [*skipped, *skipped_c], rr._scope_shared)
    asof = checkpoint_times(new)
    out: dict[str, Any] = {
        "status": STATUS,
        "units": UNITS,
        "inputs": {"runs": [str(p) for p in runs], "compare": [str(p) for p in compare]},
        "pairing_ok": bool(pair["ok"]),
        "families": {},
        "code": code_state(),
    }
    for fam in newm.groups():
        items = newm.data(fam)["items"]  # the items of the cells each owner run holds
        written = written_times(items)
        tn, tc = newm.table(fam), cmpm.table(fam)
        arms: dict[str, Any] = {}
        for arm in rr.ARMS:
            cell = rr.u_cell(arm, "upsert")
            if cell not in tn.systems:
                continue
            lo, hi = rr.ARMS[arm]
            for b in rr.BRANCHES:
                for cls in CLASSES:
                    cond = rr.changed(cls)
                    x = tn.vec(cell, b, "cv", cond)
                    if not any(x.values()):
                        continue
                    flag = {
                        w: {
                            key: yielded(
                                written.get((cell, w, b, key.rsplit("@", 1)[0]), []),
                                asof[(w, b, 0)],
                                asof[(w, b, int(key.rsplit("@", 1)[1]))],
                            )
                            for key in d
                        }
                        for w, d in x.items()
                    }
                    node = split(
                        x, tc.vec(lo, b, "cv", cond), tc.vec(hi, b, "cv", cond), flag, draws
                    )
                    arms.setdefault(arm, {}).setdefault(b, {})[cls] = node
        out["families"][rr._family_label(fam)] = {
            "arms": arms,
            "supersede_after_fork": supersede_counts(items, asof),
        }
    return out


def _pc(e: Mapping[str, Any]) -> str:
    return "  n/a" if e["n_worlds"] == 0 else f"{100 * e['est']:5.1f}"


def summary(res: Mapping[str, Any]) -> str:
    lines = [f"ex_round3_upsert: pairing {'ok' if res['pairing_ok'] else 'FAILED'}"]
    for fam, node in res["families"].items():
        for arm, by_b in node["arms"].items():
            for b, by_cls in by_b.items():
                for cls, q in by_cls.items():
                    parts = []
                    for part in PARTS:
                        p = q["parts"][part]
                        u, d = p["update"], p["upsert_differs"]
                        parts.append(
                            f"{part} {p['n_items']:4d}: B0/upsert/B1 {_pc(u['B0'])}/"
                            f"{_pc(u['upsert'])}/{_pc(u['B1'])} (+{d['above_B0']} "
                            f"-{d['below_B0']})"
                        )
                    lines.append(f"  {fam} {arm} {b:6} {cls:10} " + " | ".join(parts))
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--compare", nargs="+", required=True, type=Path)
    ap.add_argument("--draws", type=int, default=rr.DRAWS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = analyse(a.runs, a.compare, a.draws)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(summary(res))


if __name__ == "__main__":
    main()

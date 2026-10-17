"""How often a writer's programs are right, and how much a definition-item writer re-derives
(exploratory, post hoc; no model call). It changes no registered contrast, margin, threshold or
verdict; it supplies the axes of the writer figure (`round3_figures.fig_writers`) that the
registered results do not hold.

Program sets (block G-b of prereg/v4_round3.md; the runs `round3_registered G` judges G3 on). Per
program set (writer, temperature, sample) and branch: the share of the definition slots whose
program value at t0 (the `derive` event at the fork, before any shift) equals gold, in the
D1sB1_source cell (the indicator of `e1_synth.program_stats`' agree_t0 and of
`round3_registered._program_t0`), and the share whose program ran at all. These item-times precede
every shift, so the measure shares no item-time with the post-shift B x D of G3. Per-item
indicators pooled over worlds, world bootstrap (`ex_e1_classes`).

Definition-item writers (blocks H-a and H-c). Per writer and branch, on the worlds the writer ran:
r_0 (ContractValid of the changed definition item-times the source-epoch key invalidated in
D0B1_source, `round3_registered.r0`; after S_src it is H2's quantity) and B x D on the changed
definition item-times with the authored D=1 cells (after S_def it is H1's quantity), each with the
same quantity of Qwen3.5-9B (the committed runs) on the same worlds and item-times and the paired
difference. Block H itself reports r_0 after S_src only and no Qwen3.5-9B B x D on its worlds.

    python -m memshift.analysis.ex_round3_writers --programs RUN [RUN ...] \\
        --defwriters RUN [RUN ...] --compare RUN_OR_STAGE [...] --out FILE.json

--programs: the G-b program-set runs (V runs, held-out-definition runs and other runs are
ignored). --defwriters: the H-a and H-c runs. --compare: the committed runs or stages they pair
with, as for `round3_registered H`.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import round3_registered as rr
from memshift.runinfo import code_state

STATUS = (
    "exploratory, post hoc: the axes of the writer figure; changes no registered contrast, margin, "
    "threshold or verdict"
)
UNITS = rr.UNITS
DM_CELL = rr.ARMS["Dm"][1]  # holdmem_D1sB1_source: the cell whose programs block G-b replays


def program_t0(items: Iterable[rr.Row], probes: Iterable[rr.Row]) -> dict[str, rr.Vec]:
    """Per measure (`agree`: the program ran and its t0 value equals gold; `exec`: it ran), per
    world, per "branch|slot": the indicator of each definition slot of the D_m cell at t0."""
    gold0 = {(p["world_id"], p["branch"], p["fact_id"]): p for p in probes if rr._k(p) == 0}
    out: dict[str, rr.Vec] = {"agree": defaultdict(dict), "exec": defaultdict(dict)}
    for it in items:
        if it["system"] != DM_CELL or not str(it["slot_id"]).startswith("k."):
            continue
        p = gold0.get((it["world_id"], it["branch"], it["slot_id"]))
        if p is None:
            continue
        for event in it["history"]:
            note = event.get("program")
            if note is None or event["event"] != "derive":
                continue
            ok = bool(note["ok"])
            key = f"{it['branch']}|{it['slot_id']}"
            out["agree"][it["world_id"]][key] = float(ok and e1._repair_matches(p, note["value"]))
            out["exec"][it["world_id"]][key] = float(ok)
            break
    return out


def _branch(v: rr.Vec, branch: str | None) -> rr.Vec:
    """The entries of one branch (None: all)."""
    out = {
        w: {k: x for k, x in d.items() if branch is None or k.split("|", 1)[0] == branch}
        for w, d in v.items()
    }
    return {w: d for w, d in out.items() if d}


def program_sets(paths: Sequence[Path], draws: int = rr.DRAWS) -> dict[str, Any]:
    """Per G-b program set: its writer and the t0 agreement and executable rate per branch and
    over all three."""
    runs, skipped = rr.read_runs(paths, systems=lambda s: False, items=DM_CELL.__eq__)
    gb = [r for r in runs if r.program is not None and not r.user_words and not r.kappa_set]
    # one owner run per (set, cell, world, branch), the newest, as `round3_registered` merges;
    # these runs carry no answers here, so the owners are found from the items
    owner: dict[tuple[str, str, str], rr.Run] = {}
    for run in sorted(gb, key=lambda r: (r.created, r.name)):
        for it in run.items:
            owner[(rr._set_id(run), it["world_id"], it["branch"])] = run
    out: dict[str, Any] = {}
    for sid in sorted({s for s, _, _ in owner}):
        own = {(w, b): r for (s, w, b), r in owner.items() if s == sid}
        mine = {id(r): r for r in own.values()}.values()
        items = [it for r in mine for it in r.items if own.get((it["world_id"], it["branch"])) is r]
        probes = [p for r in mine for p in r.probes]
        ind = program_t0(items, probes)
        writer = next(iter(mine)).program
        assert writer is not None
        node: dict[str, Any] = {"writer": writer[0], "runs": sorted(r.name for r in mine)}
        for measure, key in (("agree", "agreement_t0"), ("exec", "executable_t0")):
            node[key] = {
                b or "all": ex.vec_mean(_branch(ind[measure], b), draws)
                for b in (*rr.BRANCHES, None)
            }
        out[sid] = node
    return {
        "sets": out,
        "skipped_runs": skipped,
        "ignored_runs": [r.name for r in runs if r not in gb],
    }


def _worlds(v: rr.Vec, worlds: Iterable[str]) -> rr.Vec:
    keep = set(worlds)
    return {w: d for w, d in v.items() if w in keep and d}


def definition_writers(
    paths: Sequence[Path], compare: Sequence[Path], draws: int = rr.DRAWS
) -> dict[str, Any]:
    """Per definition-item writer (H-a, H-c) and branch: r_0 and B x D (authored D=1) on the
    changed definition item-times, the writer's and Qwen3.5-9B's on the same worlds and
    item-times, and the paired difference."""
    new, skipped = rr.read_runs(paths, items=lambda s: True)
    need = set(rr.ARM_CELLS) | {a["system"] for r in new for a in r.answers}
    cmp_runs, skipped_c = rr.read_runs(
        rr.expand(compare), need.__contains__, {rr.ARMS["D0"][1], rr.ARMS["D1"][1]}.__contains__
    )
    newm, cmpm = rr.Merged(new, rr.by_variant), rr.Merged(cmp_runs, rr.by_family)
    pair = rr.pairing(newm, cmpm, [*skipped, *skipped_c], rr._scope_h)
    out: dict[str, Any] = {"pairing_ok": bool(pair["ok"]), "writers": {}}
    committed: dict[str, ex.Table] = {}
    for var in newm.groups():
        head = newm.runs_of(var)[0]
        if not head.defwriter:
            continue
        if head.family not in committed:
            committed[head.family] = cmpm.table(head.family)
        tab, tc = newm.table(var), committed[head.family]
        worlds = sorted(rr.cell_worlds(tab, [*rr.ARMS["D0"], *rr.ARMS["D1"]]))
        node: dict[str, Any] = {"worlds": worlds, "branches": {}}
        cond = rr.changed("definition")
        for b in rr.BRANCHES:
            mine, ref = rr.r0(tab, b), rr.r0(tc, b)
            keys = rr._common(mine["vec"], ref["vec"])
            dd_w = rr._arm_vecs(rr.getter(tab, b, cond), "D1")
            dd_q = _worlds(rr._arm_vecs(rr.getter(tc, b, cond), "D1"), dd_w)
            dkeys = rr._common(dd_w, dd_q)
            node["branches"][b] = {
                "r0": {
                    "writer": mine["r0"],
                    "qwen_same_worlds": ex.vec_mean(_worlds(ref["vec"], mine["vec"]), draws),
                    "paired_difference": ex.paired_estimate(
                        rr._restrict(mine["vec"], keys), rr._restrict(ref["vec"], keys), draws
                    ),
                },
                "bxd_D1": {
                    "writer": ex.vec_mean(dd_w, draws),
                    "qwen_same_item_times": ex.vec_mean(rr._restrict(dd_q, dkeys), draws),
                    "paired_difference": ex.paired_estimate(
                        rr._restrict(dd_w, dkeys), rr._restrict(dd_q, dkeys), draws
                    ),
                },
            }
        out["writers"][head.defwriter] = node
    return out


def analyse(
    programs: Sequence[Path],
    defwriters: Sequence[Path],
    compare: Sequence[Path],
    draws: int = rr.DRAWS,
) -> dict[str, Any]:
    return {
        "status": STATUS,
        "units": UNITS,
        "inputs": {
            "programs": [str(p) for p in programs],
            "defwriters": [str(p) for p in defwriters],
            "compare": [str(p) for p in compare],
        },
        "program_sets": program_sets(programs, draws) if programs else {"sets": {}},
        "definition_writers": (
            definition_writers(defwriters, compare, draws) if defwriters else {"writers": {}}
        ),
        "code": code_state(),
    }


def _pc(e: Mapping[str, Any] | None) -> str:
    return "  n/a" if not e or e["n_worlds"] == 0 else f"{100 * e['est']:5.1f}"


def summary(res: Mapping[str, Any]) -> str:
    lines = ["ex_round3_writers"]
    for sid, node in res["program_sets"]["sets"].items():
        a = node["agreement_t0"]
        lines.append(f"  {sid:44} agree t0 (src/def/orphan) {_pc(a['src'])}/{_pc(a['def'])}/"
                     f"{_pc(a['orphan'])}")  # fmt: skip
    dw = res["definition_writers"]
    if dw.get("writers"):
        lines.append(f"  definition writers: pairing {'ok' if dw['pairing_ok'] else 'FAILED'}")
    for model, node in dw.get("writers", {}).items():
        for b, q in node["branches"].items():
            r, x = q["r0"], q["bxd_D1"]
            r_q, x_q = _pc(r["qwen_same_worlds"]), _pc(x["qwen_same_item_times"])
            lines.append(f"  {model:20} {b:6} r0 {_pc(r['writer'])} (Qwen {r_q})"
                         f"  BxD {_pc(x['writer'])} (Qwen {x_q})")  # fmt: skip
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n", 1)[0])
    ap.add_argument("--programs", nargs="*", default=[], type=Path)
    ap.add_argument("--defwriters", nargs="*", default=[], type=Path)
    ap.add_argument("--compare", nargs="*", default=[], type=Path)
    ap.add_argument("--draws", type=int, default=rr.DRAWS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = analyse(a.programs, a.defwriters, a.compare, a.draws)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(summary(res))


if __name__ == "__main__":
    main()

"""Detection counted from the latest change, the key's effect with Claude writers and with a key
over the gold dependencies, and the item-times that Update and Stability are computed on. They read
committed run outputs only; they add no run and call no model. The stage is `ex_round4_detection` in
results/sources.json (runs/stage10_ex_round4_detection/analysis/, three files; the stage manifest
lists the producers and the source runs; `python -m memshift.analysis.ex_round4_detection --root
CHECKOUT` reruns them all, about two minutes, where CHECKOUT holds the raw runs):

  event.json    m, r and a of Eq. (1) (`ex_e1_classes.decomposition`) per study (main study and
                replication), shift, item class, key (source and citation) and B=1 cell, with
                detection counted since the fork (the `detected` flag of `e1.item_table`: an
                invalidation at any time after the fork) and from the latest change
                (`ex_round3_review._detected_last`: an invalidation after the start of the interval
                between checkpoints in which the item-time's gold last moved, up to the
                checkpoint); Update of every cell and the B x D accounting under both counts; the
                largest share the source key missed from the latest change; in the main study the
                citation key's misses by the cause of the change (`ex_round3_natural.
                label_item_times`, as `ex_round3_review.cause`) under both counts; and per HoldMem
                cell and shift the item-times Update and Stability are computed on: every changed
                item-time, and the seeded sample of unchanged item-times of equal size per world,
                shift and checkpoint (`e1.discriminative`) that the reader was asked; in the main
                study also for the D_m cells of block C (its item table, checked against its
                manifest)
  keys.json     B effect on Update at D=0 and D=1 and B x D with the oracle key (the gold
                dependencies, cells `holdmem_D{0,1}B1_gold`) and with the source key on the same
                control worlds (the control stage's item table, checked against its manifest), by
                shift and item class
  writers.json  the same contrasts on definition items with Claude Sonnet 5.5 and Claude Opus 5.5
                as writers of definition items (the block H runs, source key, exact recomputation)

Estimators: those of `ex_e1_classes` (world-clustered bootstrap, 2,000 draws, seed 996; per-item
paired differences over the item-times present in every cell used).

Macros (letters only, prefix ExDet; a stage that is not staged renders as the MISSING marker, a
quantity over no item-time as a dash). Detection is counted from the latest change unless a name
says Fork:
  ExDet<study><branch><class>Items                  changed item-times (asked)
  ExDet<study><branch><class><cell><term><key>      cell DzeroBone, DoneBone; term Missed (m),
                                                    Misses (count), Repair (r), Stale (a), Update;
                                                    key empty (source) or Cited
  ExDet<study><branch><class><cell>Update           cell DzeroBzero, DoneBzero (a_D of Eq. (2))
  ExDet<study><branch><class>Corrected<key>         (r1 - a1) - (r0 - a0), a_D the B=0 Update
  ExDet<study>SourceMissedMax                       the largest m of the source key over the
                                                    shifts and the definition and other items
  ExDetCited<branch><part><class>...                part Pooled, Shift, Drift: Items, MissedFork,
                                                    MissedLast, MissShareFork, MissShareLast
  ExDet<study><branch>UpdateItems / StabilityItems  the item-times every HoldMem cell shares
                                                    (main study: with block C's D_m cells)
  ExDetKey<key><branch><class>BDzero / BDone / BxD  key Oracle or Source; ExDetKeyWorlds;
                                                    ExDetKey<key>DefinitionBDzeroMin / Max over
                                                    the shifts (estimates alone)
  ExDetWriter<writer><branch>Definition...          writer Sonnet or Opus: BDzero, BDone, BxD;
                                                    ExDetWriterWorlds; ExDetWriterDefinitionBDzero
                                                    Min / Max over writers and shifts
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import ex_round2 as r2
from memshift.analysis import ex_round3_review as rv
from memshift.analysis.e1 import Row
from memshift.analysis.ex_e1_paper import UNDEFINED, _Out, fmt
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.runinfo import code_state

STATUS = "additional analysis: reads committed run outputs only, adds no run and calls no model"
UNITS = "fractions (x 100 for percent or pp); 95% world-clustered bootstrap, 2,000 draws, seed 996"
STAGE_KEY = "ex_round4_detection"
STAGE_DIR = "runs/stage10_ex_round4_detection"
FILES = {"event": "event.json", "keys": "keys.json", "writers": "writers.json"}
PRODUCERS = {
    f"analysis/{name}": f"python -m memshift.analysis.ex_round4_detection --part {part}"
    for part, name in FILES.items()
}
NOTE = (
    "additional analysis, computed from the raw runs of the main study and the replication, the "
    "item tables of the control stage and block C, and the block H runs; no model call"
)
SOURCE_KEYS = ("e1", "rep_e1", "c", "ctl")
H_RUNS = rv.H_RUNS  # sonnet, opus
STUDIES = (("main", "e1"), ("rep", "rep_e1"))
BRANCHES = rv.BRANCHES
CLASSES = rv.CLASSES  # definition, other, all
KEYS = (("source", ""), ("cited", "Cited"))
B1_CELLS = (("D0B1", "DzeroBone"), ("D1B1", "DoneBone"))
B0_CELLS = (("D0B0", "DzeroBzero"), ("D1B0", "DoneBzero"))
COUNTS = ("since_fork", "latest_change")
TERMS = ("n", "miss", "m", "r", "a", "U")
PARTS = rv.PARTS  # pooled, shift, drift
CITED_CELL = "holdmem_D1B1_cited"
HOLDMEM = "holdmem_"
DM_SYSTEMS = ("holdmem_D1sB0", "holdmem_D1sB1_source")  # block C's cells with the writer's programs
CTL_TABLE = "analysis/ctl_item_table.jsonl"
C_TABLE = "analysis/e1_item_table.jsonl"
CTL_KEYS = (("gold", "Oracle"), ("source", "Source"))
CTL_SYSTEMS = frozenset(
    {"holdmem_D0B0", "holdmem_D1B0", *(f"holdmem_D{d}B1_{g}" for d in (0, 1) for g, _ in CTL_KEYS)}
)
CONTRASTS = (
    ("B_effect_update_D0", "BDzero"),
    ("B_effect_update_D1", "BDone"),
    ("BxD_interaction_update", "BxD"),
)
WRITER_TAGS = (("sonnet", "Sonnet"), ("opus", "Opus"))
Est = ex.Est


def source_runs(sources: Mapping[str, str]) -> list[str]:
    """The stage's sources: the stages of `SOURCE_KEYS`, then the block H runs."""
    stages = [sources[k] for k in SOURCE_KEYS]
    return list(dict.fromkeys([*stages, *(r for rs in H_RUNS.values() for r in rs)]))


# ==================================================================================== event
def _terms(dec: Mapping[str, Any], b: str, cls: str, key: str, cell: str) -> dict[str, Any]:
    c = _get(dec, "D1", b, cls, "cells", key, cell) or {}
    return {t: c.get(t) for t in TERMS}


def _decomposition(rows: Sequence[Row], rows_last: Sequence[Row]) -> dict[str, Any]:
    """Per branch, class and key: the terms of each cell and the B x D accounting under both
    counts (Update and the B=0 cells do not depend on the count)."""
    dec = {
        "since_fork": ex.decomposition(ex.Table(rows)),
        "latest_change": ex.decomposition(ex.Table(rows_last)),
    }
    out: dict[str, Any] = {}
    for b in BRANCHES:
        for cls, _ in CLASSES:
            node: dict[str, Any] = {
                "b0": {c: _terms(dec["since_fork"], b, cls, "source", c)["U"] for c, _ in B0_CELLS}
            }
            for key, _ in KEYS:
                inter = {
                    count: _get(dec[count], "D1", b, cls, "interaction", key) or {}
                    for count in COUNTS
                }
                node[key] = {
                    "cells": {
                        c: {count: _terms(dec[count], b, cls, key, c) for count in COUNTS}
                        for c, _ in B1_CELLS
                    },
                    "observed": inter["since_fork"].get("observed"),
                    "corrected": {count: inter[count].get("corrected_formula") for count in COUNTS},
                }
            out.setdefault(b, {})[cls] = node
    return out


def _missed_max(terms: Mapping[str, Any]) -> dict[str, Any]:
    """The largest m of the source key from the latest change over the shifts and the definition and
    other items (m does not depend on D), with where it occurs."""
    best: dict[str, Any] = {"m": None, "branch": None, "class": None}
    for b in BRANCHES:
        for cls in ("definition", "other"):
            m = terms[b][cls]["source"]["cells"]["D1B1"]["latest_change"]["m"]
            if best["m"] is None or m["est"] > best["m"]["est"]:
                best = {"m": m, "branch": b, "class": cls}
    return best


def _cited_by_cause(
    rows: Sequence[Row], rows_last: Sequence[Row], cause: Mapping[r2.Key, str]
) -> dict[str, Any]:
    """The changed item-times the citation key did not invalidate, by the cause of the change, since
    the fork (as `ex_round3_review.cause`) and from the latest change."""
    tabs = {"since_fork": ex.Table(rows), "latest_change": ex.Table(rows_last)}
    out: dict[str, Any] = {}
    for b in BRANCHES:
        for part in PARTS:
            for cls, _ in CLASSES:
                cond = rv._rows_cond(
                    rv._part_cond(part, cls, b, cause), lambda r: r["cv"] is not None
                )
                node: dict[str, Any] = {}
                for count, tab in tabs.items():
                    flags = tab.vec(CITED_CELL, b, "detected", cond)
                    miss = {w: {k: 1.0 - x for k, x in d.items()} for w, d in flags.items()}
                    node["n"] = rv._n(miss)
                    node[f"missed_{count}"] = int(
                        round(sum(sum(d.values()) for d in miss.values()))
                    )
                    node[f"miss_share_{count}"] = ex.vec_mean(miss)
                out.setdefault(b, {}).setdefault(part, {})[cls] = node
    return out


def _item_times(rows: Iterable[Row], systems: Callable[[str], bool]) -> dict[str, Any]:
    """Per branch and HoldMem cell: the changed and unchanged item-times the reader was asked, the
    unchanged item-times `e1.discriminative` keeps (the Stability sample) and all unchanged
    item-times."""
    by: dict[tuple[str, str], list[Row]] = defaultdict(list)
    for r in rows:
        if systems(r["system"]):
            by[r["branch"], r["system"]].append(r)
    out: dict[str, Any] = {}
    for (b, s), rs in sorted(by.items()):
        asked = [r for r in rs if r.get("cv") is not None]
        disc = e1.discriminative(rs)
        out.setdefault(b, {})[s] = {
            "changed": sum(1 for r in asked if r["changed"]),
            "unchanged_asked": sum(1 for r in asked if not r["changed"]),
            "unchanged_sample": sum(1 for r in disc if not r["changed"]),
            "unchanged_all": sum(1 for r in rs if not r["changed"]),
        }
    return out


def _common(cells: Mapping[str, Mapping[str, int]], field: str) -> int | None:
    """The count every cell shares, or None when the cells differ."""
    values = {c[field] for c in cells.values()}
    return values.pop() if len(values) == 1 else None


def event(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """Eq. (1) under both counts, the citation key's misses by cause and the denominators (see the
    docstring)."""
    from memshift.analysis import ex_round3_natural as nat

    res: dict[str, Any] = {"status": STATUS, "units": UNITS, "stages": {}}
    for study, stage_key in STUDIES:
        st = rv._read_stage(root / sources[stage_key])
        last = rv._last_change(st)
        data = e1.load(*st.runs)
        rows = e1.item_table(data)
        rows_last = rv._detected_last(rows, data, st, last)
        terms = _decomposition(rows, rows_last)
        times = _item_times(rows, lambda s: s.startswith(HOLDMEM))
        node: dict[str, Any] = {
            "n_worlds": len(st.worlds),
            "terms": terms,
            "source_missed_max": _missed_max(terms),
            "item_times": times,
        }
        if study == "main":
            lab = nat.label_item_times(root / sources[stage_key])
            node["cited_by_cause"] = _cited_by_cause(rows, rows_last, lab.cause)
            from memshift.analysis.ex_round3_view import read_table

            dm = read_table(
                root / sources["c"], C_TABLE, set(DM_SYSTEMS), {r["world_id"] for r in rows}
            )
            node["item_times_dm"] = _item_times(dm, DM_SYSTEMS.__contains__)
            times = {b: {**times[b], **node["item_times_dm"][b]} for b in BRANCHES}
        # the counts every HoldMem cell of the study shares (None when the cells differ)
        node["item_times_common"] = {
            b: {
                "update": _common(times[b], "changed"),
                "stability": _common(times[b], "unchanged_sample"),
            }
            for b in BRANCHES
        }
        res["stages"][study] = sources[stage_key]
        res[study] = node
    res["stages"]["dm"] = sources["c"]
    res["code"] = code_state()
    return res


# ===================================================================================== keys
def keys(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """B effects and B x D with the oracle key and the source key on the control worlds."""
    from memshift.analysis.ex_round3_view import read_table

    stage = root / sources["ctl"]
    rows = read_table(stage, CTL_TABLE, set(CTL_SYSTEMS), {f"w{s}" for s in range(1000, 1010)})
    tab = ex.Table(rows)
    contrasts: dict[str, Any] = {}
    for g, _ in CTL_KEYS:
        for b in BRANCHES:
            for cls, _ in CLASSES:
                c = ex.class_contrasts(tab, b, "D1", g, cls)
                n = sum(
                    1 for r in tab.rows("holdmem_D0B0", b) if r["changed"] and ex.in_class(r, cls)
                )
                contrasts.setdefault(g, {}).setdefault(b, {})[cls] = {
                    "n": n,
                    **{k: c[k] for k, _ in CONTRASTS},
                }
    return {
        "status": STATUS,
        "units": UNITS,
        "stage": sources["ctl"],
        "worlds": tab.worlds,
        "n_worlds": len(tab.worlds),
        "contrasts": contrasts,
        "code": code_state(),
    }


# ================================================================================== writers
def writers(root: Path) -> dict[str, Any]:
    """B effects and B x D on definition items with the Claude writers of block H."""
    contrasts: dict[str, Any] = {}
    worlds: dict[str, list[str]] = {}
    for w, runs in H_RUNS.items():
        tab = ex.Table(e1.item_table(e1.load(*[root / r for r in runs])))
        worlds[w] = tab.worlds
        for b in BRANCHES:
            c = ex.class_contrasts(tab, b, "D1", "source", "definition")
            n = sum(
                1
                for r in tab.rows("holdmem_D0B0", b)
                if r["changed"] and ex.in_class(r, "definition")
            )
            contrasts.setdefault(w, {})[b] = {
                "definition": {"n": n, **{k: c[k] for k, _ in CONTRASTS}}
            }
    if len({tuple(v) for v in worlds.values()}) != 1:
        raise ValueError(f"the block H writers ran on different worlds: {worlds}")
    return {
        "status": STATUS,
        "units": UNITS,
        "runs": {w: list(r) for w, r in H_RUNS.items()},
        "worlds": next(iter(worlds.values())),
        "n_worlds": len(next(iter(worlds.values()))),
        "contrasts": contrasts,
        "code": code_state(),
    }


# =================================================================================== macros
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs (None where the stage or a file is not there)."""
    src = dict(sources)
    return {key: stage_json(root, src, STAGE_KEY, name) for key, name in FILES.items()}


def _est(parent: Mapping[str, Any] | None, key: str, f: Callable[[float | None], str] = pct) -> str:
    """An estimate with its interval: MISSING without the stage, a dash over no item-time."""
    if parent is None:
        return MISSING
    e = parent.get(key)
    return UNDEFINED if not e else fmt(e, f)


def _count(parent: Mapping[str, Any] | None, key: str) -> str:
    if parent is None:
        return MISSING
    x = parent.get(key)
    return UNDEFINED if x is None else f"{int(x):,}"


def _node(res: Mapping[str, Any] | None, *keys: str) -> dict[str, Any] | None:
    """The node at `keys`: None without the stage, {} when the stage lacks it."""
    if res is None:
        return None
    node = _get(res, *keys)
    return dict(node) if isinstance(node, Mapping) else {}


STUDY_TAGS = (("main", "Main"), ("rep", "Rep"))
PART_TAGS = (("pooled", "Pooled"), ("shift", "Shift"), ("drift", "Drift"))


def _event_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    for study, st in STUDY_TAGS:
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                head = ("Det", st, bt, ct)
                b0 = _node(res, study, "terms", b, cls, "b0")
                for cell, cellt in B0_CELLS:
                    out.put(*head, cellt, "Update", value=_est(b0, cell))
                for key, kt in KEYS:
                    for cell, cellt in B1_CELLS:
                        t = _node(res, study, "terms", b, cls, key, "cells", cell, "latest_change")
                        if key == "source" and cell == "D1B1":
                            out.put(*head, "Items", value=_count(t, "n"))
                        out.put(*head, cellt, "Missed", kt, value=_est(t, "m"))
                        out.put(*head, cellt, "Misses", kt, value=_count(t, "miss"))
                        out.put(*head, cellt, "Repair", kt, value=_est(t, "r"))
                        out.put(*head, cellt, "Stale", kt, value=_est(t, "a"))
                        out.put(*head, cellt, "Update", kt, value=_est(t, "U"))
                    corr = _node(res, study, "terms", b, cls, key, "corrected")
                    out.put(*head, "Corrected", kt, value=_est(corr, "latest_change", pp))
        best = _node(res, study, "source_missed_max")
        out.put("Det", st, "SourceMissedMax", value=MISSING if best is None else _point(best))
        for b, bt in BRANCH_TAGS:
            common = _node(res, study, "item_times_common", b)
            out.put("Det", st, bt, "UpdateItems", value=_count(common, "update"))
            out.put("Det", st, bt, "StabilityItems", value=_count(common, "stability"))
    for b, bt in BRANCH_TAGS:
        for part, pt in PART_TAGS:
            for cls, ct in CLASSES:
                node = _node(res, "main", "cited_by_cause", b, part, cls)
                cited = ("Det", "Cited", bt, pt, ct)
                out.put(*cited, "Items", value=_count(node, "n"))
                for count, tag in (("since_fork", "Fork"), ("latest_change", "Last")):
                    out.put(*cited, "Missed", tag, value=_count(node, f"missed_{count}"))
                    out.put(*cited, "MissShare", tag, value=_est(node, f"miss_share_{count}"))


def _point(best: Mapping[str, Any]) -> str:
    """The estimate alone of the largest missed share (its interval belongs to one cell only)."""
    m = best.get("m")
    return UNDEFINED if not m else pct(m.get("est"))


def _range(nodes: Iterable[Mapping[str, Any] | None], key: str) -> tuple[str, str]:
    """The smallest and the largest estimate of `key` over `nodes`, each alone (MISSING without
    the stage)."""
    ests = [n[key]["est"] for n in nodes if n and n.get(key)]
    if not ests:
        return MISSING, MISSING
    return pp(min(ests)), pp(max(ests))


def _keys_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    out.put("Det", "Key", "Worlds", value=_count(res, "n_worlds"))
    for g, gt in CTL_KEYS:
        for b, bt in BRANCH_TAGS:
            for cls, ct in CLASSES:
                node = _node(res, "contrasts", g, b, cls)
                for k, tag in CONTRASTS:
                    out.put("Det", "Key", gt, bt, ct, tag, value=_est(node, k, pp))
        lo, hi = _range(
            (_node(res, "contrasts", g, b, "definition") for b in BRANCHES), CONTRASTS[0][0]
        )
        out.put("Det", "Key", gt, "Definition", "BDzeroMin", value=lo)
        out.put("Det", "Key", gt, "Definition", "BDzeroMax", value=hi)


def _writers_numbers(res: Mapping[str, Any] | None, out: _Out) -> None:
    out.put("Det", "Writer", "Worlds", value=_count(res, "n_worlds"))
    for w, wt in WRITER_TAGS:
        for b, bt in BRANCH_TAGS:
            node = _node(res, "contrasts", w, b, "definition")
            for k, tag in CONTRASTS:
                out.put("Det", "Writer", wt, bt, "Definition", tag, value=_est(node, k, pp))
    nodes = [_node(res, "contrasts", w, b, "definition") for w, _ in WRITER_TAGS for b in BRANCHES]
    lo, hi = _range(nodes, CONTRASTS[0][0])
    out.put("Det", "Writer", "Definition", "BDzeroMin", value=lo)
    out.put("Det", "Writer", "Definition", "BDzeroMax", value=hi)


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every macro of this block, from `load`."""
    out = _Out()
    _event_numbers(inp.get("event"), out)
    _keys_numbers(inp.get("keys"), out)
    _writers_numbers(inp.get("writers"), out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ================================================================================== staging
def produce(root: Path, part: str) -> dict[str, Any]:
    sources = json.loads((root / "results" / "sources.json").read_text())
    if part == "event":
        return event(root, sources)
    if part == "keys":
        return keys(root, sources)
    return writers(root)


def stage(root: Path, out: Path, parts: Iterable[str]) -> None:
    """Write the analyses of `parts` into OUT/analysis and the stage manifest (over every file
    there, with the producers of `PRODUCERS`)."""
    from memshift.analysis.stage import write_stage_manifest

    sources = json.loads((root / "results" / "sources.json").read_text())
    analysis = out / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    for part in parts:
        res = produce(root, part)
        (analysis / FILES[part]).write_text(json.dumps(res, indent=1, default=str) + "\n")
        print(part, "->", analysis / FILES[part], flush=True)
    srcs = [root / s for s in source_runs(sources)]
    write_stage_manifest(out, STAGE_KEY, srcs, {"producers": PRODUCERS, "note": NOTE})


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage the detection analyses (see the docstring).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None, help=f"default: ROOT/{STAGE_DIR}")
    ap.add_argument("--part", choices=sorted(FILES), action="append", default=None)
    a = ap.parse_args()
    stage(a.root, a.out or a.root / STAGE_DIR, a.part or list(FILES))


if __name__ == "__main__":
    main()

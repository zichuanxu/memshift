"""Paper numbers and tables of the exploratory analyses requested by the second review round of the
paper (prereg/decisions.md 2026-10-08; post hoc). They change no registered contrast, margin,
threshold or verdict. Every quantity comes from committed run outputs and the seeded world
generator, with no model call; the stages are named in results/sources.json:

  e1    analysis/e1_metrics.json (the Eq. (1) terms, class split and per-definition contrasts of
        `analysis.ex_e1_classes`), analysis/e1_item_table.jsonl, the probes of the runs its
        manifest lists, and the natural branch of each world (`gen.shifts.generate_branch`: the
        same events without the shift, so its gold is what each item would have been without it)
  c     analysis/e1_metrics.json of block C (`explore_class_split`, arm D1s: the model-written
        derivation D_m as the D=1 arm)
  ctl   analysis/ctl_item_table.jsonl and the items of the thinking runs its manifest lists
  e2    analysis/e2_metrics.json (`explore_writer_predictors`)

Attribution of a changed item-time (gold value or status differs from t0, `analysis.e1`): changed
by the shift when its gold differs from the gold of the natural branch at the same time, by drift
only when it does not (the same change happens without the shift). After S_def a second reading
follows the probes alone: the item's definition moved to a new version (`kappa_version`) or not.

Macros (letters only; a stage that is not staged renders as the MISSING marker, a quantity over no
item as a dash):
  ExAttr<branch>...                 item-times after the shift, changed, changed by the shift and
                                    by drift only (counts, shares), and B at D=0, B at D=1 and B x D
                                    on Update on each part by class; ExAttrDefKappa... the reading
                                    by definition version; ExAttrSrcTarget... the slots S_src
                                    targets
  ExThink<definition>...            the thinking writer per definition: writes, writes at the token
                                    limit and unreadable; per branch the repair correctness of its
                                    re-derivations, how many returned no value, and the repair of
                                    the plain D=0 writer on the same items (`Plain`)
  ExPerDefLooMain<branch>BxD<Min|Max>  the range of B x D on the definition class with one
                                    definition left out
  ETwoExWriter...Ninety             AUROC with 90% world and type intervals (the 95% intervals are
                                    `analysis.ex_numbers`), and the path-3 rule under type
                                    resampling
The D_m class split and every level of tab_main already have macros (ExClsDm..., ExDecMain...,
ExIntMain...MOneCited, ExClsMain...BxD, ExMixMain...Share; `analysis.ex_e1_paper`).

tab_main           main text: the Eq. (1) levels r0, r1, a0, a1, B x D and the cited key's m1 by
                   shift and class
tab_redef_only     appendix: changed item-times by cause and the contrasts on each part
tab_think_trunc    appendix: the thinking writer per definition (token limit, repair, no value)
tab_classsplit_dm  appendix: the class split with D_m as the D=1 arm (block C)
tab_perdef         appendix: B x D per definition, with the leave-one-definition-out range
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis.e1 import Cond, Row, _diff
from memshift.analysis.ex_e1_paper import UNDEFINED, _cell, _Out, fmt
from memshift.analysis.ex_numbers import stat_iv
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.analysis.paper_tables import _num, _row, interval

THINK_CELL = "holdmem_D0thinkB1_source"
PLAIN_CELL = "holdmem_D0B1_source"  # the same writer without thinking, same items
THINK_RUN_PREFIX = "e1_think_"
DEFINITIONS = (  # the order of the review's table
    ("k.important_contact", "ImportantContact", "important contacts"),
    ("k.close_friend", "CloseFriend", "close friends"),
    ("k.working_hours", "WorkingHours", "working hours"),
    ("k.active_project", "ActiveProject", "active projects"),
    ("k.vip_sender", "VipSender", "VIP senders"),
    ("k.overdue_task", "OverdueTask", "overdue tasks"),
)
CAUSES = (("shift", "Shift"), ("drift", "Drift"))
KAPPA_CAUSES = (("redefined", "Redefined"), ("kept", "Kept"))
ATTR_CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))
ATTR_CONTRASTS = (
    ("B_effect_update_D0", "BDzero"),
    ("B_effect_update_D1", "BDone"),
    ("BxD_interaction_update", "BxD"),
)
BRANCH_LABELS = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}
CAUSE_LABELS = {
    "src": {"shift": "the revision", "drift": "drift only"},
    "def": {"shift": "the redefinition", "drift": "drift only"},
    "orphan": {"shift": "the deletion", "drift": "drift only"},
}
Key = tuple[str, str, str, int]  # (world, branch, fact, k)
PATH3_AUROC_BAR = 0.60  # prereg v1 path 3: slope 90% lower bound <= 0 or AUROC < 0.60 triggers


# ------------------------------------------------------------------------------------ loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs and the quantities computed from committed outputs (None where a stage
    is not there)."""
    src = dict(sources)
    return {
        "main": stage_json(root, src, "e1", "e1_metrics.json"),
        "c": stage_json(root, src, "c", "e1_metrics.json"),
        "e2": stage_json(root, src, "e2", "e2_metrics.json"),
        "attr": _attribution(str(root.resolve()), src["e1"]) if "e1" in src else None,
        "think": _thinking(str(root.resolve()), src["ctl"]) if "ctl" in src else None,
    }


def _jsonl(path: Path, keep: Callable[[str], bool] = lambda line: True) -> Iterable[Row]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip() and keep(line):
                yield json.loads(line)


def _source_runs(stage: Path) -> list[Path] | None:
    """The run directories a merged stage was built from (its manifest's `sources`)."""
    man = stage / "manifest.json"
    if not man.exists():
        return None
    return [stage.parent / s["run_id"] for s in json.loads(man.read_text())["sources"]]


def _gold(p: Mapping[str, Any]) -> str:
    """The gold of a probe (value and status) as a comparable string."""
    return json.dumps([p["gold_value"], p["gold_status"]], sort_keys=True, default=str)


# -------------------------------------------------------------------------------- attribution
def _natural_gold(
    seed: int, params: dict[str, Any], times: Mapping[int, str]
) -> dict[tuple[str, int], str]:
    """(fact, k) -> gold of the natural branch of the world (no shift) at each probe time."""
    from memshift.gen.model import parse_ts
    from memshift.gen.shifts import generate_branch
    from memshift.gen.state import SourceLog
    from memshift.probes import make_probes

    world = generate_branch(seed, "natural", params)
    log = SourceLog(world.events)
    out: dict[tuple[str, int], str] = {}
    for k, as_of in times.items():
        for p in make_probes(world, log, [parse_ts(as_of)]):
            out[p.fact_id, k] = _gold(json.loads(json.dumps(p.to_json(), default=str)))
    return out


def _part(cls: str, branch: str, labels: Mapping[Key, str], label: str) -> Cond:
    """The changed rows of a class whose item-time carries `label` (a cause or a version)."""

    def cond(row: Row) -> bool:
        key = (row["world_id"], branch, row["fact_id"], int(row["k"]))
        return bool(row["changed"]) and ex.in_class(row, cls) and labels.get(key) == label

    return cond


def _contrasts(tab: ex.Table, branch: str, cond: Cond) -> dict[str, Any]:
    """B effect at D=0 and D=1 and B x D on Update over the rows that meet `cond` (source-epoch
    binding, benchmark-authored D=1; the estimator of `ex_e1_classes.class_contrasts`)."""
    u = {
        (d, b): tab.vec(ex.cell_system("D1", d, b, "source"), branch, "cv", cond)
        for d in (0, 1)
        for b in (0, 1)
    }
    return {
        "B_effect_update_D0": ex.paired_estimate(u[0, 1], u[0, 0]),
        "B_effect_update_D1": ex.paired_estimate(u[1, 1], u[1, 0]),
        "BxD_interaction_update": ex.paired_estimate(
            _diff(u[1, 1], u[1, 0]), _diff(u[0, 1], u[0, 0])
        ),
    }


@cache
def _attribution(root: str, stage: str) -> dict[str, Any] | None:
    """Per branch: item-times after the shift, the changed ones by cause and class (counts) and
    the contrasts on each part; after S_def also by definition version; after S_src the slots
    the shift targets. None when an input is not there."""
    base = Path(root) / stage
    table = base / "analysis" / "e1_item_table.jsonl"
    runs = _source_runs(base)
    if runs is None or not table.exists():
        return None
    if not all((r / "probes.jsonl").exists() and (r / "manifest.json").exists() for r in runs):
        return None
    probes: dict[Key, Row] = {}
    times: dict[str, dict[int, str]] = defaultdict(dict)
    worlds: dict[str, tuple[int, dict[str, Any]]] = {}
    targets: dict[tuple[str, str], list[str]] = {}
    for run in runs:
        man = json.loads((run / "manifest.json").read_text())
        params = dict((man.get("params") or {}).get("world_params") or {})
        for w in man["worlds"]:
            worlds[w["world_id"]] = (int(w["seed"]), params)
            targets[w["world_id"], w["branch"]] = list(w.get("shift_targets") or [])
        for p in _jsonl(run / "probes.jsonl"):
            k = int(p["tags"]["k"])
            probes[p["world_id"], p["branch"], p["fact_id"], k] = p
            if times[p["world_id"]].setdefault(k, p["as_of"]) != p["as_of"]:
                raise ValueError(f"{p['world_id']} k={k}: branches probe at different times")
    natural: dict[tuple[str, str, int], str] = {}
    for wid, (seed, params) in sorted(worlds.items()):
        for (fact, k), g in _natural_gold(seed, params, times[wid]).items():
            natural[wid, fact, k] = g
    # the generator must reproduce the committed gold where the shift cannot act: before the
    # fork on every branch, and after it on the items whose definition S_def leaves alone
    for (w, b, f, k), p in probes.items():
        moved = p["kappa_version"] != probes[w, b, f, 0]["kappa_version"]
        if (k == 0 or (b == "def" and not moved)) and _gold(p) != natural[w, f, k]:
            raise ValueError(f"natural branch does not reproduce the committed gold: {w} {b} {f}")
    cause: dict[Key, str] = {}  # changed item-times: "shift" or "drift"
    version: dict[Key, str] = {}  # changed item-times: "redefined" (kappa version moved) or "kept"
    counts: dict[str, dict[str, Any]] = {}
    for (w, b, f, k), p in sorted(probes.items()):
        if k == 0:
            continue
        p0 = probes[w, b, f, 0]
        cls = "definition" if ex.item_class(f) == "definition" else "other"
        c = counts.setdefault(b, {"item_times": 0, "changed": 0, "classes": {}})
        cc = c["classes"].setdefault(cls, dict.fromkeys(("item_times", "changed", "both"), 0))
        for d in (c, cc):
            d["item_times"] += 1
        if _gold(p) == _gold(p0):
            continue
        by_shift = _gold(p) != natural[w, f, k]
        cause[w, b, f, k] = "shift" if by_shift else "drift"
        version[w, b, f, k] = "redefined" if p["kappa_version"] != p0["kappa_version"] else "kept"
        for d in (c, cc):
            d["changed"] += 1
        cc[cause[w, b, f, k]] = cc.get(cause[w, b, f, k], 0) + 1
        cc["both"] += int(by_shift and natural[w, f, k] != _gold(p0))  # would have changed too
    tab = ex.Table(_jsonl(table))
    out: dict[str, Any] = {"branches": {}}
    for b in tab.branches:
        node = counts.get(b)
        if node is None:
            continue
        check = {
            (r["world_id"], r["fact_id"], r["k"])
            for r in tab.rows("holdmem_D0B0", b)
            if r["changed"]
        }
        if check != {(w, f, k) for (w, bb, f, k) in cause if bb == b}:
            raise ValueError(f"{b}: the item table and the probes disagree on the changed items")

        node["contrasts"] = {
            c: {cls: _contrasts(tab, b, _part(cls, b, cause, c)) for cls, _ in ATTR_CLASSES}
            for c, _ in CAUSES
        }
        if b == "def":
            node["kappa"] = {
                kc: _contrasts(tab, b, _part("definition", b, version, kc))
                for kc, _ in KAPPA_CAUSES
            }
        if b == "src":
            per_world = sorted({len(set(t)) for (w, bb), t in targets.items() if bb == b})
            slots = sorted({len({f for (w, bb, f, k) in probes if w == x}) for x in worlds})
            node["targets"] = {
                "slots_per_world": per_world,
                "world_slots": slots,
                "changed_item_times": sum(
                    1 for (w, bb, f, k) in cause if bb == b and f in targets[w, bb]
                ),
            }
        out["branches"][b] = node
    out["n_worlds"] = len(worlds)
    return out


# --------------------------------------------------------------------------------- thinking
@cache
def _thinking(root: str, stage: str) -> dict[str, Any] | None:
    """The thinking writer's definition writes per definition (token limit, unreadable), and per
    branch and definition the repair correctness of its re-derivations, how many returned no value,
    and the repair correctness of the plain D=0 writer on the same item-times (the controls' repair
    metric: ContractValid over the asked item-times re-derived now). None when not staged."""
    from memshift.holdmem.writer import THINK_MAX_TOKENS

    base = Path(root) / stage
    table = base / "analysis" / "ctl_item_table.jsonl"
    runs = [r for r in (_source_runs(base) or []) if r.name.startswith(THINK_RUN_PREFIX)]
    if not runs or not table.exists() or not all((r / "items.jsonl").exists() for r in runs):
        return None
    acc: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for run in runs:
        for it in _jsonl(run / "items.jsonl", lambda line: THINK_CELL in line):
            if it["system"] != THINK_CELL:
                continue
            for h in it["history"]:
                note = h.get("think")
                if note is None:
                    continue
                hit = note["completion_tokens"] >= THINK_MAX_TOKENS
                for label in (it["slot_id"], "all"):
                    a, w = acc[label], it["world_id"]
                    a["at_limit"][w].append(float(hit))
                    a["unreadable"][w].append(float(not note["ok"]))
                    a["xor"][w].append(float(hit == note["ok"]))  # one of the two, not both
    writes = {
        label: {
            "writes": sum(len(v) for v in a["at_limit"].values()),
            "at_limit": ex.mean_estimate(a["at_limit"]),
            "unreadable": ex.mean_estimate(a["unreadable"]),
            "at_limit_xor_unreadable": int(sum(sum(v) for v in a["xor"].values())),
            "limit_tokens": THINK_MAX_TOKENS,
            "n_worlds": len(a["at_limit"]),
        }
        for label, a in acc.items()
    }
    rep: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    none: dict[tuple[str, str, str], int] = defaultdict(int)
    for r in _jsonl(table, lambda line: '"k.' in line):
        if r["system"] not in (THINK_CELL, PLAIN_CELL) or not r.get("repaired_now"):
            continue
        if r.get("cv") is None or ex.item_class(r["fact_id"]) != "definition":
            continue
        cell = "think" if r["system"] == THINK_CELL else "plain"
        for label in (r["fact_id"], "all"):
            rep[cell, r["branch"], label][r["world_id"]].append(float(r["cv"]))
            none[cell, r["branch"], label] += int(bool(r.get("repair_none")))
    repair: dict[str, Any] = {}
    for (cell, b, label), by_world in sorted(rep.items()):
        values = [x for v in by_world.values() for x in v]
        repair.setdefault(cell, {}).setdefault(b, {})[label] = {
            "repair": ex.mean_estimate(by_world),
            "rederived": len(values),
            "correct": int(sum(values)),
            "no_value": none[cell, b, label],
        }
    return {"writes": writes, "repair": repair}


# ---------------------------------------------------------------------------------- macros
def _count(x: int | None) -> str:
    return MISSING if x is None else f"{x:,}"


def _share(num: int | None, den: int | None) -> str:
    return MISSING if num is None or not den else pct(num / den)


def _attr_numbers(attr: Mapping[str, Any] | None, out: _Out) -> None:
    for b, bt in BRANCH_TAGS:
        node = _get(attr, "branches", b) or {}
        classes = node.get("classes") or {}
        n_all, n_changed = node.get("item_times"), node.get("changed")
        out.put("Attr", bt, "ItemTimes", value=_count(n_all))
        out.put("Attr", bt, "Changed", value=_count(n_changed))
        out.put("Attr", bt, "ChangedShareOfAll", value=_share(n_changed, n_all))
        for c, ct in CAUSES:
            n_cause = sum(v.get(c, 0) for v in classes.values()) if classes else None
            out.put("Attr", bt, ct, "ShareOfAll", value=_share(n_cause, n_all))
            out.put("Attr", bt, ct, "ShareOfChanged", value=_share(n_cause, n_changed))
            for cls, clt in ATTR_CLASSES:
                if not classes:
                    n = None
                elif cls == "all":
                    n = n_cause
                else:
                    n = (classes.get(cls) or {}).get(c, 0)
                out.put("Attr", bt, ct, clt, "Items", value=_count(n))
                for key, kt in ATTR_CONTRASTS:
                    e = _get(node, "contrasts", c, cls, key)
                    out.put("Attr", bt, ct, clt, kt, value=fmt(e, pp))
        both = sum(v["both"] for v in classes.values()) if classes else None
        out.put("Attr", bt, "ShiftAlsoDriftItems", value=_count(both))
    kappa = _get(attr, "branches", "def", "kappa") or {}
    for kc, kt in KAPPA_CAUSES:
        e = kappa.get(kc) or {}
        n = _get(e, "BxD_interaction_update", "n_items")
        out.put("Attr", "Def", "Kappa", kt, "Items", value=_count(n))
        for key, ckt in ATTR_CONTRASTS:
            out.put("Attr", "Def", "Kappa", kt, ckt, value=fmt(e.get(key), pp))
    tg = _get(attr, "branches", "src", "targets") or {}
    out.put("Attr", "Src", "TargetSlots", value=_span(tg.get("slots_per_world")))
    out.put("Attr", "WorldSlots", value=_span(tg.get("world_slots")))
    one = tg.get("slots_per_world") or []
    slots = tg.get("world_slots") or []
    share = pct(one[0] / slots[0]) if len(one) == 1 and len(slots) == 1 else MISSING
    out.put("Attr", "Src", "TargetSlotShare", value=share)
    n_tg, n_all = tg.get("changed_item_times"), _get(attr, "branches", "src", "item_times")
    out.put("Attr", "Src", "TargetChangedItems", value=_count(n_tg))
    out.put("Attr", "Src", "TargetChangedShareOfAll", value=_share(n_tg, n_all))


def _span(values: list[int] | None) -> str:
    """One count, or the range of counts that differ by world."""
    if not values:
        return MISSING
    return str(values[0]) if len(values) == 1 else f"{values[0]}--{values[-1]}"


def _think_numbers(think: Mapping[str, Any] | None, out: _Out) -> None:
    for d, dt, _ in (*DEFINITIONS, ("all", "All", "")):
        w = _get(think, "writes", d) or {}
        out.put("Think", dt, "Writes", value=_count(w.get("writes")))
        out.put("Think", dt, "AtLimit", value=fmt(w.get("at_limit")) if w else MISSING)
        out.put("Think", dt, "Unreadable", value=fmt(w.get("unreadable")) if w else MISSING)
        for b, bt in BRANCH_TAGS:
            for cell, ct in (("think", ""), ("plain", "Plain")):
                r = _get(think, "repair", cell, b, d) or {}
                out.put("Think", dt, bt, ct, "Repair", value=fmt(r.get("repair")) if r else MISSING)
                out.put("Think", dt, bt, ct, "Rederived", value=_count(r.get("rederived")))
                out.put("Think", dt, bt, ct, "Correct", value=_count(r.get("correct")))
                if cell == "think":
                    out.put("Think", dt, bt, "NoValue", value=_count(r.get("no_value")))


def _loo_range(main: Mapping[str, Any] | None, branch: str) -> list[float]:
    """B x D on the definition class with one definition left out, per definition (estimates)."""
    res = _get(main, "explore_per_definition", "D1", branch) or {}
    ests = [
        _get(node, "without_type", "source", "BxD_interaction_update", "est")
        for node in res.values()
    ]
    return sorted(x for x in ests if x is not None and x == x)


def _perdef_numbers(main: Mapping[str, Any] | None, out: _Out) -> None:
    for b, bt in BRANCH_TAGS:
        loo = _loo_range(main, b)
        out.put("PerDefLoo", "Main", bt, "BxD", "Min", value=pp(loo[0]) if loo else MISSING)
        out.put("PerDefLoo", "Main", bt, "BxD", "Max", value=pp(loo[-1]) if loo else MISSING)


def _writer_numbers(e2: Mapping[str, Any] | None) -> dict[str, str]:
    """AUROC with 90% intervals (world and type resampling) for the predictors the review quotes,
    and the path-3 rule (slope 90% lower bound <= 0 or AUROC < 0.60) under type resampling."""
    sets = _get(e2, "explore_writer_predictors", "sets") or {}
    out: dict[str, str] = {}
    for set_key, set_tag, preds in (
        ("all", "", (("agree", "Agree"), ("nondef", "NonDef"))),
        ("nodef", "NoDef", (("agree", "Agree"),)),
    ):
        for pred, ptag in preds:
            auc = _get(sets, set_key, "pooled", "predictors", pred, "auroc")
            base = f"ETwoExWriter{set_tag}{ptag}PooledAuroc"
            out[f"{base}Ninety"] = stat_iv(auc, "world", "90", 3)
            out[f"{base}TypeNinety"] = stat_iv(auc, "type", "90", 3)
    agree = _get(sets, "all", "pooled", "predictors", "agree") or {}
    lo = _get(agree, "slope", "type", "lo90")
    auc = _get(agree, "auroc", "est")
    if lo is None or auc is None:
        out["ETwoExWriterPathThreeTypeVerdict"] = MISSING
    else:
        hit = lo <= 0 or auc < PATH3_AUROC_BAR
        out["ETwoExWriterPathThreeTypeVerdict"] = "triggered" if hit else "not triggered"
    return out


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every macro of this block, from `load`."""
    out = _Out()
    _attr_numbers(inp.get("attr"), out)
    _think_numbers(inp.get("think"), out)
    _perdef_numbers(inp.get("main"), out)
    macros = dict(out)
    writer = _writer_numbers(inp.get("e2"))
    if dup := macros.keys() & writer.keys():
        raise ValueError(f"macro defined twice: {sorted(dup)[:3]}")
    macros |= writer
    for name, value in macros.items():
        _macro(name, value)  # letters-only names
    return macros


# ---------------------------------------------------------------------------------- tables
def _term(e: Mapping[str, Any] | None, digits: int = 1) -> str:
    """A term in percent, unsigned (`ex_e1_paper._terms_row_cells`)."""
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=digits)


def _pct_cell(e: Mapping[str, Any] | None) -> str:
    """A share in percent with its interval, unsigned (`paper_tables.interval`)."""
    if not e:
        return MISSING
    if e["est"] != e["est"]:
        return UNDEFINED
    return interval(_num(e["est"]), _num(e["lo"]), _num(e["hi"]))


def tab_main(inp: Mapping[str, Any]) -> str:
    """Main text: per shift and item class, the Eq. (1) levels of the source-epoch key (r0, r1,
    a0, a1), the B x D interaction with its interval, the class's share of the changed item-times
    and the share of changes the cited-cluster key misses."""
    main = inp.get("main")
    n = _get(main, "n_worlds") or MISSING
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{The levels behind the $B{{\times}}D$ interaction on Update (main study, {n} "
        r"worlds; source-epoch key; percent). Share: the class's share of the changed item-times. "
        r"$r_D$: accuracy of the re-derived items at $D$ with binding; $a_D$: Update at $D$ "
        r"without binding, where no item is re-derived. The key misses no change, so "
        r"$B{\times}D=(r_1{-}a_1)-(r_0{-}a_0)$ (pp, 95\% world-bootstrap interval). $m_1$: "
        r"share of the changes the cited-cluster key misses. \emph{Definition}: the six "
        r"definition items of each world; \emph{other}: the other 42 items. The class split is "
        r"exploratory.}",
        r"\label{tab:main}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}ll r rrrr c r@{}}",
        r"\toprule",
        r"& & share & $r_0$ & $r_1$ & $a_0$ & $a_1$ & $B{\times}D$ & $m_1$, cited \\",
        r"\midrule",
    ]
    for i, (b, _) in enumerate(BRANCH_TAGS):
        for cls in ("definition", "other"):
            node = _get(main, "explore_decomposition", "D1", b, cls)
            share = _get(main, "explore_class_mix", b, "classes", cls, "share")
            bxd = _get(
                main, "explore_class_split", "D1", b, "source", cls, "BxD_interaction_update"
            )
            m1 = _get(node, "interaction", "cited", "m1")
            lines.append(
                _row(
                    BRANCH_LABELS[b] if cls == "definition" else "",
                    cls,
                    MISSING if share is None else _num(share, digits=0),
                    _term(_get(node, "cells", "source", "D0B1", "r")),
                    _term(_get(node, "cells", "source", "D1B1", "r")),
                    _term(_get(node, "cells", "source", "D0B0", "a")),
                    _term(_get(node, "cells", "source", "D1B0", "a")),
                    _cell(bxd),
                    _term(m1, digits=0),
                )
            )
        if i < len(BRANCH_TAGS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _attr_rows(node: Mapping[str, Any], b: str) -> list[str]:
    """Rows of the contrasts on each part of one branch: a class with no item-time is left out,
    and so is `all` when one class holds every item-time of the part."""
    rows: list[str] = []
    first = True
    for c, _ in CAUSES:
        parts = _get(node, "contrasts", c) or {}
        sizes = {
            cls: _get(parts, cls, "BxD_interaction_update", "n_items") or 0
            for cls, _ in ATTR_CLASSES
        }
        shown = [cls for cls in ("definition", "other") if sizes[cls]]
        if len(shown) > 1:
            shown.append("all")
        for j, cls in enumerate(shown):
            e = parts.get(cls) or {}
            rows.append(
                _row(
                    BRANCH_LABELS[b] if first else "",
                    CAUSE_LABELS[b][c] if j == 0 else "",
                    cls,
                    f"{sizes[cls]:,}",
                    *(_cell(e.get(key)) for key, _ in ATTR_CONTRASTS),
                )
            )
            first = False
    if b == "def":
        for kc, label in (("redefined", "new version"), ("kept", "same version")):
            e = _get(node, "kappa", kc) or {}
            n_items = _get(e, "BxD_interaction_update", "n_items")
            rows.append(
                _row(
                    "",
                    rf"\emph{{{label}}}",
                    "definition",
                    MISSING if n_items is None else f"{n_items:,}",
                    *(_cell(e.get(key)) for key, _ in ATTR_CONTRASTS),
                )
            )
    return rows


def tab_redef_only(inp: Mapping[str, Any]) -> str:
    """Appendix: (a) the item-times after each shift, the changed ones and the part the shift
    itself changed; (b) B at D=0, B at D=1 and B x D on Update on each part, by class."""
    attr = inp.get("attr") or {}
    br = attr.get("branches") or {}
    tg = _get(br, "src", "targets") or {}
    one, slots = tg.get("slots_per_world") or [], tg.get("world_slots") or []
    target = (
        f"{one[0]} of the {slots[0]} items of each world ({pct(one[0] / slots[0])})"
        if len(one) == 1 and len(slots) == 1
        else MISSING
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{Update after a shift counts every item-time whose gold value or status differs "
        rf"from its value before the shift (main study, {attr.get('n_worlds', MISSING)} worlds, "
        r"exploratory). It does so whether the shift changed the value or later events would have "
        r"changed it anyway. By the shift: the gold differs from the gold the same world has at "
        r"the same time without the shift (the generator's natural branch). Drift only: "
        r"it does not. (a) Item-times after the shift (48 items, four checkpoints), those that "
        r"changed, and the part the shift itself changed, with shares of all item-times. "
        rf"$\Ssrc$ revises the sources of {target}. (b) $B$ effect at each $D$ and "
        r"$B{\times}D$ on Update (source-epoch key, pp, 95\% world-bootstrap intervals) on "
        r"each part. $n$: changed "
        r"item-times. A redefinition changes only definition items. New or same "
        r"version: after $\Sdef$, the definition item's version moved or not, whatever its "
        r"value.}",
        r"\label{tab:redef_only}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l r rr rr rr@{}}",
        r"\toprule",
        r"& item-times & \multicolumn{2}{c}{changed} & \multicolumn{2}{c}{by the shift} "
        r"& \multicolumn{2}{c}{drift only} \\",
        r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(l){7-8}",
        r"& & $n$ & \% & $n$ & \% & $n$ & \% \\",
        r"\midrule",
    ]
    for b, _ in BRANCH_TAGS:
        node = br.get(b) or {}
        classes = node.get("classes") or {}
        n_all = node.get("item_times")
        cells = [_count(n_all)]
        for n_part in (
            node.get("changed"),
            *(sum(v.get(c, 0) for v in classes.values()) if classes else None for c, _ in CAUSES),
        ):
            cells += [_count(n_part), _share(n_part, n_all).replace(r"\%", "")]
        lines.append(_row(BRANCH_LABELS[b], *cells))
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\par\medskip",
        r"\begin{tabular}{@{}lll r ccc@{}}",
        r"\toprule",
        r"& changed by & class & $n$ & $B$ effect, $D{=}0$ & $B$ effect, $D{=}1$ "
        r"& $B{\times}D$ \\",
        r"\midrule",
    ]
    for i, (b, _) in enumerate(BRANCH_TAGS):
        lines += _attr_rows(br.get(b) or {}, b) or [_row(BRANCH_LABELS[b], *[MISSING] * 6)]
        if i < len(BRANCH_TAGS) - 1:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_think_trunc(inp: Mapping[str, Any]) -> str:
    """Appendix: per definition, the thinking writer's writes and the share at the token limit;
    after S_src and S_def the repair correctness of its re-derivations, how many returned no value,
    and the repair correctness of the plain D=0 writer on the same item-times."""
    think = inp.get("think") or {}
    pooled = _get(think, "writes", "all") or {}
    differ = pooled.get("at_limit_xor_unreadable")
    same = (
        ", exactly the writes whose reply could not be read"
        if differ == 0
        else f", with {_count(differ)} writes at the limit or unreadable but not both"
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{The thinking writer ($D{=}0$, $B{=}1$, source-epoch key, "
        rf"{pooled.get('n_worlds', MISSING)} worlds, exploratory) per definition. Writes: its "
        r"definition writes after the shift, all branches. At limit: the share of them that used "
        rf"the whole budget of {_count(pooled.get('limit_tokens'))} completion tokens "
        rf"(percent{same}). Repair: "
        r"ContractValid of its re-derivations that were asked (percent, 95\% world-bootstrap "
        r"interval). No value: those re-derivations that returned no value, of all asked. Plain: "
        r"the repair of the same writer without thinking on the same item-times (percent).}",
        r"\label{tab:think_trunc}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l rr crr crr@{}}",
        r"\toprule",
        r"& & & \multicolumn{3}{c}{$\Ssrc$} & \multicolumn{3}{c}{$\Sdef$} \\",
        r"\cmidrule(lr){4-6}\cmidrule(l){7-9}",
        r"definition & writes & at limit & repair & no value & plain & repair & no value "
        r"& plain \\",
        r"\midrule",
    ]
    for d, _, label in (*DEFINITIONS, ("all", "", "all six")):
        if d == "all":
            lines.append(r"\midrule")
        w = _get(think, "writes", d) or {}
        cells = [label, _count(w.get("writes")), _term(w.get("at_limit"))]
        for b in ("src", "def"):
            r = _get(think, "repair", "think", b, d) or {}
            plain = _get(think, "repair", "plain", b, d) or {}
            no_value = (
                f"{r['no_value']}/{r['rederived']}"
                if "no_value" in r and r["rederived"]
                else MISSING
            )
            cells += [_pct_cell(r.get("repair")), no_value, _term(plain.get("repair"))]
        lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_classsplit_dm(inp: Mapping[str, Any]) -> str:
    """Appendix: B at each D, D_m at each B and B x D_m on Update with the model-written
    derivation D_m as the D=1 arm (block C), per item class (definition, other, all) and branch;
    under each contrast that involves the D=1 arm, in grey, the same contrast with the authored
    derivation on the same worlds. On all items, B x D_m after S_src and S_def is the registered
    C3. Rows are contrasts and columns branches (the six class-by-branch columns of tab_classsplit
    do not fit 5.5in with the authored rows)."""
    c = inp.get("c")
    n = _get(c, "n_worlds") or MISSING
    rows_ = (
        ("B_effect_update_D0", r"$B$ at $D{=}0$", False),
        ("B_effect_update_D1", r"$B$ at $D_{\mathrm{m}}$", True),
        ("D_effect_update_B0", r"$D_{\mathrm{m}}$ at $B{=}0$", True),
        ("D_effect_update_B1", r"$D_{\mathrm{m}}$ at $B{=}1$", True),
        ("BxD_interaction_update", r"$B{\times}D_{\mathrm{m}}$", True),
    )
    lines = [
        r"\begin{table}[t]",
        r"\caption{The contrasts on Update by item class with the model-written derivation "
        rf"$D_{{\mathrm{{m}}}}$ as the $D{{=}}1$ arm ({n} worlds, source-epoch key, "
        r"pp, 95\% world-bootstrap intervals). $B$ at $D$: the effect of binding, $B{=}1$ minus "
        r"$B{=}0$, at the given $D$. $D_{\mathrm{m}}$ at $B$: $D_{\mathrm{m}}$ minus $D{=}0$ at "
        r"the given $B$. $B{\times}D_{\mathrm{m}}$: $B$ at $D_{\mathrm{m}}$ minus $B$ at $D{=}0$. "
        r"In grey under each row: the same contrast with the authored derivation as the $D{=}1$ "
        r"arm, on the same worlds ($B$ at $D{=}0$ involves neither). Definition: the six "
        r"definition items of each world. Other: the other 42 items. Share: the class's "
        r"share of the changed item-times. On all items, $B{\times}D_{\mathrm{m}}$ after $\Sdef$ "
        r"is registered, and after $\Ssrc$ and $\Sdel$ it is exploratory. The class split is "
        r"exploratory.}",
        r"\label{tab:classsplit_dm}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{@{}l ccc@{}}",
        r"\toprule",
        r"& $\Ssrc$ & $\Sdef$ & $\Sdel$ \\",
    ]
    for cls in ("definition", "other", "all"):
        if cls == "all":
            head = [r"\emph{all items}", "", "", ""]
        else:
            shares = [
                _get(c, "explore_class_mix", b, "classes", cls, "share") for b, _ in BRANCH_TAGS
            ]
            head = [
                rf"\emph{{{cls} items}}, share",
                *(MISSING if x is None else _num(x, digits=0) + r"\%" for x in shares),
            ]
        lines += [r"\midrule", _row(*head)]
        for key, label, with_authored in rows_:
            for arm, name in (("D1s", label), ("D1", r"\quad\textcolor{gray}{authored}")):
                if arm == "D1" and not with_authored:
                    continue
                vals = []
                for b, _ in BRANCH_TAGS:
                    cell = _cell(_get(c, "explore_class_split", arm, b, "source", cls, key))
                    vals.append(cell if arm == "D1s" else rf"\textcolor{{gray}}{{{cell}}}")
                lines.append(_row(name, *vals))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_perdef(inp: Mapping[str, Any]) -> str:
    """Appendix: B x D on Update per definition (source-epoch key, main study), the definition class
    as a whole, and the range of the class value with one definition left out."""
    main = inp.get("main")
    n = _get(main, "n_worlds") or MISSING
    res = _get(main, "explore_per_definition", "D1") or {}
    lines = [
        r"\begin{table}[t]",
        rf"\caption{{$B{{\times}}D$ on Update per definition (main study, {n} worlds, "
        r"source-epoch key, pp, 95\% world-bootstrap intervals, exploratory). $n$: the "
        r"definition's changed item-times. All six: the definition class. Leave one out: the "
        r"smallest and the largest $B{\times}D$ of the definition class with one definition "
        r"removed. The intervals resample worlds, not definitions.}",
        r"\label{tab:perdef}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}l cr cr cr@{}}",
        r"\toprule",
        r"& \multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} "
        r"& \multicolumn{2}{c}{$\Sdel$} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"definition & $B{\times}D$ & $n$ & $B{\times}D$ & $n$ & $B{\times}D$ & $n$ \\",
        r"\midrule",
    ]
    for d, _, label in DEFINITIONS:
        cells = [label]
        for b, _ in BRANCH_TAGS:
            e = _get(res, b, d, "type", "source", "BxD_interaction_update")
            cells += [_cell(e), _count(e.get("n_items")) if e else MISSING]
        lines.append(_row(*cells))
    lines.append(r"\midrule")
    cells = ["all six"]
    for b, _ in BRANCH_TAGS:
        e = _get(
            main, "explore_class_split", "D1", b, "source", "definition", "BxD_interaction_update"
        )
        cells += [_cell(e), _count(e.get("n_items")) if e else MISSING]
    lines.append(_row(*cells))
    cells = ["leave one out"]
    for b, _ in BRANCH_TAGS:
        loo = _loo_range(main, b)
        span = f"{_num(loo[0], sign=True)} to {_num(loo[-1], sign=True)}" if loo else MISSING
        cells.append(rf"\multicolumn{{2}}{{c}}{{{span}}}")
    lines.append(_row(*cells))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "tab_main.tex": tab_main(inp),
        "tab_redef_only.tex": tab_redef_only(inp),
        "tab_think_trunc.tex": tab_think_trunc(inp),
        "tab_classsplit_dm.tex": tab_classsplit_dm(inp),
        "tab_perdef.tex": tab_perdef(inp),
    }

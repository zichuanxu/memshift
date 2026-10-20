"""Why the model-written derivation D_m repairs worse than D=0 after a deletion. It makes no model
call.

Design rule 3 of the paper rests on the hypothesis that a program returns a value for a deleted
record where the authored derivation returns none. This module tests it on the items, probes and
answers that block C already holds (stage `c`: the 20 test worlds 1000-1019, branch `orphan` =
S_del, three D cells per binding: D0 the writer alone, D1 the benchmark-authored checker and
fallback, D_m the writer-authored program of `holdmem.program`, written `D1s` in the runs).

What a D_m write does (`holdmem.system.HoldMem._write`): the writer states a value with citations;
the span check of `holdmem.checker` runs; if it passes the writer's value stands and the program's
output is only recorded; if it fails the program's output replaces the value, and a program that
errors, times out or returns the wrong type leaves the item `unverifiable` (not served). For a
definition item the check is "the writer's value equals the program's", so the program decides
every disagreement. The reader sees only valid items.

Classes. A row is one asked item-time (world, item, checkpoint k >= 1) of one cell. It is
ContractValid (`ok`) or falls into exactly one of, in this order:
  e  the item was not re-derived at the checkpoint (B=1: no repair event at the probe time, the key
     had not moved; B=0: no item was appended since the shift, so the reader has only the
     pre-shift item)
  c  re-derived and `unverifiable`: the replacement failed, nothing is served. D_m: the program
     failed (the cause is read from the reason of the slot's last write: no code in the response,
     syntax error, rejected by the sandbox, runtime error, timeout, wrong type); D1: no rule
  d  valid, and the stored value is right (gold abstain: no value; gold answer: the gold value),
     but the reader's answer is not ContractValid; sub: the reader abstained, answered with the
     status `answer` and no value (it read "there is no current value" as an answer), or gave a
     value
  a  valid, gold abstain, a value is stored (the rule-3 hypothesis); sub `replacement` when the
     value differs from the writer's own value for the same item-time, which the cells share
     (identical writer requests; the D0 cell's value), so the cell's derivation put it there,
     `writer` when it equals the writer's (the span check passed a value the program would not
     have given)
  b  valid, gold answer, the stored value is wrong; sub: the origin as above, and `value` or `none`
At B=0 an appended item sits beside the unrevised pre-shift item; `stale_served` marks the rows
whose pre-shift item is wrong for the current gold. The origin of a stored value is inferred from
its difference to D0's and checked against the recorded check of every slot's last write.

Populations: `changed` (gold differs from t0: the Update population), `changed_abstain` (the deleted
records), `changed_answer`, `unchanged`, `asked`, and at B=1 `repaired` (re-derived in D_m and D0:
the population of contrast C1, which the module reproduces). Class sets: `definition`, `other`
(structured plus free text), `all`.

Per population and class set the output holds each cell's class shares, the D_m - D0 and D_m - D1
differences per class, and the exact accounting of the D_m - D0 difference in ContractValid by
item-time pairs: what the reference got right and D_m lost, by the class D_m falls in (`loss`), and
what the reference got wrong and D_m gets right (`gain`). Estimates are the world-clustered paired
bootstrap of the other E1 contrasts (2,000 draws, seed 996, `ex_e1_classes`); `loss_shares` are the
shares of the D0-reference loss by D_m class (ratio of sums over worlds). The program-only blocks
score the program's output alone (failed, correct, a value for a deleted record, none for an
answer, another value), whether or not the cell used it, over all rows (`program_only`) and by
item kind for deleted and for live records (`program_by_kind`: does the program react to the
deletion at all?). The provenance block says where a value returned for a deleted record comes
from, on the regenerated world: the same field of another live record, and whether that is the
source's first or last listed record (the order the programs receive).

The programs' text is in no local run (the response cache stays on the compute host); each example
carries the cache key of its program request instead, and the live records that hold the value the
program returned.

Usage: python -m memshift.analysis.ex_round3_dm_deletion [--runs runs] [--stage stage4_c_e1]
           [--out runs/stage6_ex_round3/dm_deletion.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis.e1 import _repair_matches
from memshift.gen.model import Slot, World, parse_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import ObjState, Snapshot, SourceLog
from memshift.runinfo import file_sha256, world_record
from memshift.scoring.normalize import norm_str, values_match
from memshift.scoring.stats import item_diffs

Row = dict[str, Any]
Key = tuple[str, str, int]  # (world, fact, checkpoint k)

BRANCH = "orphan"  # S_del: the source object behind every structured inference-half item is deleted
CELLS = ("D0", "D1", "Dm")
HEAD = {"D0": "holdmem_D0", "D1": "holdmem_D1", "Dm": "holdmem_D1s"}
CLASSES = ("ok", "a", "b", "c", "d", "e")
FAILURES = CLASSES[1:]
CLASS_SETS = ("definition", "other", "all")
LABELS = {
    "ok": "ContractValid",
    "a": "a value is stored for a record whose gold is abstain",
    "b": "a wrong value (or none) is stored where the gold is an answer",
    "c": "the replacement failed: the item is unverifiable and not served",
    "d": "the stored value is right and the reader's answer is not",
    "e": "the item was not re-derived",
}
PROGRAM_CLASSES = ("failed", "correct", "value_for_deleted", "none_for_answer", "wrong_value")
WRONG_PROGRAM = ("value_for_deleted", "none_for_answer", "wrong_value")
FIELD_SOURCES = {"contact_attr": "contacts", "task_due": "tasks", "series": "calendar"}
LAST_K = 8  # the last checkpoint: the stored item's text is the one the reader saw


def system(cell: str, b: int) -> str:
    return f"{HEAD[cell]}B0" if b == 0 else f"{HEAD[cell]}B1_source"


def same_value(vtype: str, a: Any, b: Any) -> bool:
    """Two stored values are the same value (None and None; a typed match otherwise)."""
    if a is None or b is None:
        return a is None and b is None
    try:
        return values_match(vtype, a, b)
    except (ValueError, IndexError):
        return norm_str(str(a)) == norm_str(str(b))


def error_type(reason: str | None) -> str:
    """The kind of program failure the reason of a write names (`holdmem.system._write`)."""
    parts = (reason or "").split("program failed: ", 1)
    if len(parts) < 2:
        return "unknown"
    msg = parts[1]
    if msg.startswith("no code"):
        return "no_code"
    if msg.startswith("syntax error"):
        return "syntax_error"
    if msg.startswith(("disallowed syntax", "forbidden", "no top-level", "derive must", "program")):
        return "rejected_by_sandbox"
    if msg.startswith("timeout"):
        return "timeout"
    if msg.startswith("type mismatch"):
        return "wrong_type"
    if msg.startswith("sandbox crashed"):
        return "sandbox_crash"
    return "runtime_error"


# ------------------------------------------------------------------------------------- loading
@dataclass
class Raw:
    """The orphan-branch rows of the runs a stage merges."""

    probes: dict[Key, Row]
    answers: dict[tuple[str, str, str, int], Row]  # (system, world, fact, k)
    items: dict[tuple[str, str, str], list[Row]]  # (system, world, slot)
    worlds: dict[str, dict[str, Any]]  # world -> seed, world params, events hash
    runs: list[str]


def _lines(path: Path) -> Iterable[Row]:
    tag = f'"branch": "{BRANCH}"'
    with path.open(encoding="utf-8") as f:
        for line in f:
            if tag in line:
                yield json.loads(line)


def stage_runs(stage: Path, runs: Path) -> list[Path]:
    man = json.loads((stage / "manifest.json").read_text())
    return [runs / s["run_id"] for s in man["sources"]]


def load_raw(run_dirs: Sequence[Path]) -> Raw:
    items: dict[tuple[str, str, str], list[Row]] = defaultdict(list)
    raw = Raw({}, {}, items, {}, [])
    for d in run_dirs:
        man = json.loads((d / "manifest.json").read_text())
        params = (man.get("params") or {}).get("world_params") or None
        raw.runs.append(d.name)
        for w in man["worlds"]:
            if w["branch"] == BRANCH:
                raw.worlds[w["world_id"]] = {
                    "seed": int(w["seed"]),
                    "params": params,
                    "events_sha256": w["events_sha256"],
                }
        for p in _lines(d / "probes.jsonl"):
            raw.probes[p["world_id"], p["fact_id"], int(p["tags"]["k"])] = p
        for a in _lines(d / "answers.jsonl"):
            raw.answers[a["system"], a["world_id"], a["fact_id"], int(a["tags"]["k"])] = a
        for it in _lines(d / "items.jsonl"):
            items[it["system"], it["world_id"], it["slot_id"]].append(it)
    return raw


# --------------------------------------------------------------------------------- stored state
@dataclass(frozen=True)
class State:
    """What a cell holds for one item at a probe time."""

    rederived: bool
    status: str | None
    value: Any
    prog_ok: bool | None
    prog_value: Any
    pre_value: Any = None  # B=0: the pre-shift item, which stays valid beside the appended ones
    prog_key: str | None = None  # cache key of the program request of the last write


def stored_state(items: Sequence[Row], as_of: str, b: int) -> State:
    """B=1: the slot's single item after its events up to the probe time; it was re-derived at the
    probe time if a repair event carries that time. B=0: the newest item appended up to the probe
    time (none: not re-derived), with the pre-shift item's value beside it."""
    if b:
        (it,) = items
        events = [h for h in it["history"] if h["ts"] <= as_of]
        last = events[-1]
        prog = last.get("program") or {}
        return State(
            any(h["event"] == "repair" and h["ts"] == as_of for h in events),
            last["new"],
            last.get("value"),
            prog.get("ok"),
            prog.get("value"),
            prog_key=(it.get("det") or {}).get("program_key"),
        )
    pre = next(it for it in items if it["history"][0]["event"] == "derive")
    new = [
        it
        for it in items
        if it["history"][0]["event"] == "append" and it["history"][0]["ts"] <= as_of
    ]
    pre_value = pre["history"][0].get("value")
    if not new:
        return State(False, None, None, None, None, pre_value)
    newest = max(new, key=lambda it: it["history"][0]["ts"])
    first = newest["history"][0]
    prog = first.get("program") or {}
    return State(
        True,
        first["new"],
        first.get("value"),
        prog.get("ok"),
        prog.get("value"),
        pre_value,
        (newest.get("det") or {}).get("program_key"),
    )


def classify(
    p: Row, st: State, writer: State | None, ans: Row, cause: str, b: int
) -> tuple[str, str]:
    """(class, sub class) of one row. `p` is the probe (gold), `writer` the D0 state of the same
    item-time (None for D0 itself), `cause` what a failed replacement is called."""
    if ans["contract_valid"]:
        return "ok", ""
    if not st.rederived:
        return "e", "not_rederived" if b else "no_appended_item"
    if st.status != "valid":
        return "c", cause
    if _repair_matches(p, st.value):
        if ans["status"] == "abstain":
            return "d", "abstained"
        return "d", "answer_without_value" if ans.get("value") is None else "gave_value"
    kept = writer is None or same_value(p["vtype"], st.value, writer.value)
    origin = "writer" if kept else "replacement"
    if p["gold_status"] == "abstain":
        return "a", origin
    return "b", f"{origin}_{'none' if st.value is None else 'value'}"


def program_class(p: Row, st: State) -> str | None:
    """The program's own output scored against the gold, whether or not the cell used it."""
    if st.prog_ok is None or not st.rederived:
        return None
    if not st.prog_ok:
        return "failed"
    if _repair_matches(p, st.prog_value):
        return "correct"
    if p["gold_status"] == "abstain":
        return "value_for_deleted"
    return "none_for_answer" if st.prog_value is None else "wrong_value"


def program_use(p: Row, st: State, writer: State) -> str | None:
    """What became of the program's output: it failed (the item is unverifiable, or the passed span
    check masked the failure), it agrees with the writer, it replaced the writer's value, or the
    writer's value stood against it."""
    if st.prog_ok is None or not st.rederived:
        return None
    if not st.prog_ok:
        return "failed_unverifiable" if st.status != "valid" else "failed_masked"
    if same_value(p["vtype"], st.prog_value, writer.value):
        return "agrees"
    return "used" if same_value(p["vtype"], st.value, st.prog_value) else "overridden"


@dataclass(frozen=True)
class Rec:
    """One asked item-time of one cell."""

    changed: bool
    gold_status: str
    klass: str
    sub: str
    rederived: bool
    status: str | None
    value: Any
    prog_ok: bool | None
    prog_value: Any
    prog_class: str | None
    prog_use: str | None
    prog_key: str | None
    stale_served: bool
    reader_status: str
    reader_value: Any


Joint = dict[Key, dict[str, Rec]]


def slot_errors(raw: Raw, b: int) -> dict[tuple[str, str], str]:
    """The kind of program failure of each D_m slot's last write, where that write failed (the
    program of a slot is one request across the checkpoints)."""
    out: dict[tuple[str, str], str] = {}
    for (s, w, slot), items in raw.items.items():
        if s != system("Dm", b):
            continue
        for it in items:
            det = it.get("det") or {}
            if det.get("program_ok") is False:
                out[w, slot] = error_type(det.get("reason"))
    return out


def records(raw: Raw, b: int) -> tuple[Joint, dict[str, Any]]:
    """The three cells' rows for every asked item-time, and the consistency checks of the build."""
    errors = slot_errors(raw, b)
    states: dict[str, dict[Key, State]] = {}
    for cell in CELLS:
        states[cell] = {
            (w, fact, k): stored_state(raw.items[system(cell, b), w, fact], p["as_of"], b)
            for (w, fact, k), p in sorted(raw.probes.items())
            if k > 0 and (system(cell, b), w, fact, k) in raw.answers
        }
    asked = [set(s) for s in states.values()]
    keys = sorted(set.intersection(*asked))
    checks: dict[str, Any] = {
        "asked_item_times_per_cell": [len(a) for a in asked],
        "asked_sets_equal": all(a == asked[0] for a in asked),
        "rederivation_differs_D0_Dm": sum(
            states["D0"][k].rederived != states["Dm"][k].rederived for k in keys
        ),
        "slots_with_other_item_count_D0_Dm": sum(
            len(raw.items[system("D0", b), w, slot]) != len(items)
            for (s, w, slot), items in raw.items.items()
            if s == system("Dm", b)
        ),
    }
    joint: Joint = {}
    for key in keys:
        w, fact, k = key
        p, p0 = raw.probes[key], raw.probes[w, fact, 0]
        changed = (p["gold_value"], p["gold_status"]) != (p0["gold_value"], p0["gold_status"])
        writer = states["D0"][key]
        row: dict[str, Rec] = {}
        for cell in CELLS:
            st = states[cell][key]
            ans = raw.answers[system(cell, b), w, fact, k]
            cause = (
                f"program_failed_{errors.get((w, fact), 'earlier_checkpoint')}"
                if cell == "Dm"
                else "no_rule"
            )
            klass, sub = classify(p, st, None if cell == "D0" else writer, ans, cause, b)
            row[cell] = Rec(
                changed,
                p["gold_status"],
                klass,
                sub,
                st.rederived,
                st.status,
                st.value,
                st.prog_ok,
                st.prog_value,
                program_class(p, st),
                program_use(p, st, writer),
                st.prog_key,
                bool(b == 0 and st.rederived and not _repair_matches(p, st.pre_value)),
                ans["status"],
                ans.get("value"),
            )
        joint[key] = row
    return joint, checks


# ------------------------------------------------------------------------------------ estimates
Population = Callable[[Mapping[str, Rec]], bool]
POPULATIONS: dict[str, Population] = {
    "changed": lambda r: r["Dm"].changed,
    "changed_abstain": lambda r: r["Dm"].changed and r["Dm"].gold_status == "abstain",
    "changed_answer": lambda r: r["Dm"].changed and r["Dm"].gold_status == "answer",
    "unchanged": lambda r: not r["Dm"].changed,
    "asked": lambda r: True,
    "repaired": lambda r: r["Dm"].rederived and r["D0"].rederived,
}


def is_class(cell: str, c: str) -> Population:
    return lambda r: r[cell].klass == c


def moves(ref: str, ref_ok: bool, dm_ok: bool, dm_class: str | None = None) -> Population:
    """The reference cell is (not) ContractValid and D_m is (not, in class `dm_class`)."""

    def pred(r: Mapping[str, Rec]) -> bool:
        if (r[ref].klass == "ok") != ref_ok or (r["Dm"].klass == "ok") != dm_ok:
            return False
        return dm_class is None or r["Dm"].klass == dm_class

    return pred


def ref_moves(ref: str, ref_class: str) -> Population:
    """The reference cell is in the class `ref_class` and D_m is ContractValid."""
    return lambda r: r[ref].klass == ref_class and r["Dm"].klass == "ok"


def vec(joint: Joint, keys: Iterable[Key], f: Population) -> ex.Vec:
    out: ex.Vec = defaultdict(dict)
    for key in keys:
        out[key[0]][f"{key[1]}@{key[2]}"] = float(f(joint[key]))
    return out


def select(joint: Joint, population: str, class_set: str) -> list[Key]:
    members = ex.CLASS_SETS[class_set]
    pop = POPULATIONS[population]
    return [key for key, r in sorted(joint.items()) if pop(r) and ex.item_class(key[1]) in members]


def tally(joint: Joint, keys: Sequence[Key]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for cell in CELLS:
        recs = [joint[k][cell] for k in keys]
        out[cell] = {
            "classes": {c: sum(r.klass == c for r in recs) for c in CLASSES},
            "sub": dict(sorted(Counter(f"{r.klass}/{r.sub}" for r in recs if r.sub).items())),
            "stale_pre_shift_item_served": sum(r.stale_served for r in recs),
        }
    return out


def block(joint: Joint, keys: Sequence[Key], draws: int) -> dict[str, Any]:
    """Shares, differences and the accounting of D_m - D0 and D_m - D1 over `keys`."""

    def mean(f: Population) -> ex.Est:
        return ex.vec_mean(vec(joint, keys, f), draws)

    ind = {c: {k: vec(joint, keys, is_class(c, k)) for k in CLASSES} for c in CELLS}
    out: dict[str, Any] = {"n_item_times": len(keys), "counts": tally(joint, keys)}
    out["share"] = {c: {k: mean(is_class(c, k)) for k in CLASSES} for c in CELLS}
    for ref in ("D0", "D1"):
        out[f"Dm_minus_{ref}"] = {
            k: ex.paired_estimate(ind["Dm"][k], ind[ref][k], draws) for k in CLASSES
        }
        out[f"accounting_vs_{ref}"] = {
            "loss_by_Dm_class": {k: mean(moves(ref, True, False, k)) for k in FAILURES},
            "loss_total": mean(moves(ref, True, False)),
            "loss_share_of_total": loss_shares(joint, keys, ref, draws),
            "gain_by_ref_class": {k: mean(ref_moves(ref, k)) for k in FAILURES},
            "gain_total": mean(moves(ref, False, True)),
            "net": ex.paired_estimate(ind["Dm"]["ok"], ind[ref]["ok"], draws),
        }
    return out


def loss_shares(joint: Joint, keys: Sequence[Key], ref: str, draws: int) -> dict[str, ex.Est]:
    """Of the item-times the reference cell got right and D_m did not, the share by D_m class (and,
    for class a, by origin), as a ratio of sums over worlds with the world bootstrap."""
    names = ("loss", *FAILURES, "a_replacement", "a_writer")
    worlds = sorted({k[0] for k in keys})
    pos = {w: i for i, w in enumerate(worlds)}
    per_world = np.zeros((len(worlds), len(names)))
    for key in keys:
        r = joint[key]
        if r[ref].klass != "ok" or r["Dm"].klass == "ok":
            continue
        row = per_world[pos[key[0]]]
        row[0] += 1
        row[names.index(r["Dm"].klass)] += 1
        if r["Dm"].klass == "a":
            row[names.index(f"a_{r['Dm'].sub}")] += 1
    n = int(per_world[:, 0].sum())
    return {
        name: ex.world_estimate(per_world, _ratio(j), n, draws) for j, name in enumerate(names) if j
    }


def _is_program_class(c: str) -> Population:
    return lambda r: r["Dm"].prog_class == c


def _ratio(j: int) -> Callable[[np.ndarray], Any]:
    """Component j of per-world totals over component 0."""
    return lambda t: t[..., j] / t[..., 0]


def program_only(joint: Joint, keys: Sequence[Key], draws: int) -> dict[str, Any]:
    """The D_m programs scored alone over the re-derived rows of `keys`: the share of each program
    class, and how the cell used the output (class x use counts)."""
    rows = [k for k in keys if joint[k]["Dm"].prog_class is not None]
    share = {c: ex.vec_mean(vec(joint, rows, _is_program_class(c)), draws) for c in PROGRAM_CLASSES}
    use = Counter(f"{joint[k]['Dm'].prog_class}/{joint[k]['Dm'].prog_use}" for k in rows)
    return {"n_rederived": len(rows), "share": share, "class_use_counts": dict(sorted(use.items()))}


# ----------------------------------------------------------------------------- regenerated worlds
class Worlds:
    """The orphan-branch worlds regenerated from their seeds (checked against the run manifests)."""

    def __init__(self, raw: Raw):
        self.raw = raw
        self._cache: dict[str, tuple[World, SourceLog]] = {}
        self.hash_matches: dict[str, bool] = {}

    def get(self, w: str) -> tuple[World, SourceLog]:
        if w not in self._cache:
            info = self.raw.worlds[w]
            world = generate_branch(info["seed"], BRANCH, info["params"])
            self._cache[w] = (world, SourceLog(world.events))
            self.hash_matches[w] = world_record(world)["events_sha256"] == info["events_sha256"]
        return self._cache[w]

    def snapshot(self, w: str, as_of: str) -> Snapshot:
        return self.get(w)[1].snapshot(parse_ts(as_of))


def field_value(slot: Slot, o: ObjState) -> str | None:
    """The value the slot's structured source would give from the record `o`."""
    if slot.kind == "contact_attr":
        v = o.payload.get(slot.slot_id.split(".")[-1])
    elif slot.kind == "task_due":
        v = o.payload.get("due")
    elif slot.kind == "series" and o.payload.get("kind") == "series":
        v = f"{o.payload['weekday']} {o.payload['time']}"
    else:
        v = None
    return None if v is None else str(v)


def record_label(o: ObjState) -> str:
    name = o.payload.get("title") or o.payload.get("name")
    return f"{o.object_id} ({name})" if name else o.object_id


def provenance(slot: Slot, snap: Snapshot, value: Any, p0: Row) -> dict[str, Any]:
    """Where a value returned for a deleted record comes from: the same field of another live record
    of the structured source (`other_record_same_field`; the flags say whether it is the first or
    the last listed record of the source, in the order the programs receive: oldest update first),
    the text of a live record of the declared sources, or nowhere. `equals_pre_deletion_value`: the
    value is the one the deleted record had at t0."""
    if isinstance(value, list):
        return {
            "kind": "set_value",
            "records": [],
            "equals_pre_deletion_value": False,
            "equals_last_listed_record": False,
            "equals_first_listed_record": False,
        }
    src = FIELD_SOURCES.get(slot.kind)
    listed = (
        [
            (o, v)
            for o in sorted(snap.by_source(src), key=lambda o: (o.updated, o.object_id))
            if (v := field_value(slot, o)) is not None
        ]
        if src
        else []
    )
    hits = [o for o, v in listed if same_value(slot.vtype, value, v)]
    needle = norm_str(str(value))
    if hits:
        kind = "other_record_same_field"
    elif needle and any(needle in norm_str(o.text) for o in snap.by_source(*slot.sources)):
        kind = "text_only"
    else:
        kind = "nowhere"
    return {
        "kind": kind,
        "records": sorted(record_label(o) for o in hits)[:3],
        "equals_pre_deletion_value": p0["gold_status"] == "answer"
        and same_value(slot.vtype, value, p0["gold_value"]),
        "equals_last_listed_record": bool(listed) and same_value(slot.vtype, value, listed[-1][1]),
        "equals_first_listed_record": bool(listed) and same_value(slot.vtype, value, listed[0][1]),
    }


# --------------------------------------------------------------------------------- rule 3 block
def _membership(keys: Iterable[Key], members: Mapping[Key, Any]) -> ex.Vec:
    out: ex.Vec = defaultdict(dict)
    for k in keys:
        out[k[0]][f"{k[1]}@{k[2]}"] = float(k in members)
    return out


def rule3(
    raw: Raw, joint: Joint, worlds: Worlds, keys: Sequence[Key], draws: int
) -> tuple[dict[str, Any], dict[Key, dict[str, Any]]]:
    """The hypothesis on the deleted records (`keys`, B=1): how often D_m stores a value where the
    writer and the authored derivation store none, whose doing it is, and where the value comes
    from. Also returns the provenance of each replacement value by item-time."""
    rows = {k: joint[k] for k in keys}
    dm_a = {k: r for k, r in rows.items() if r["Dm"].klass == "a"}
    right = ("ok", "d")  # the stored value is right: none
    replaced = {
        k: r
        for k, r in rows.items()
        if r["Dm"].status == "valid"
        and r["Dm"].rederived
        and not same_value(raw.probes[k]["vtype"], r["Dm"].value, r["D0"].value)
    }
    stored_value = {k: r for k, r in replaced.items() if r["Dm"].value is not None}
    prov: dict[Key, dict[str, Any]] = {}
    by_kind: dict[str, Counter[str]] = defaultdict(Counter)
    for k in stored_value:
        w, fact, _ = k
        slot = worlds.get(w)[0].slot(fact)
        prov[k] = provenance(
            slot,
            worlds.snapshot(w, raw.probes[k]["as_of"]),
            stored_value[k]["Dm"].value,
            raw.probes[w, fact, 0],
        )
        c = by_kind[slot.kind]
        c["n"] += 1
        c[prov[k]["kind"]] += 1
        for flag in (
            "equals_pre_deletion_value",
            "equals_last_listed_record",
            "equals_first_listed_record",
        ):
            c[flag] += bool(prov[k].get(flag))
    out: dict[str, Any] = {
        "n_item_times": len(rows),
        "Dm_class_a": len(dm_a),
        "Dm_class_a_by_origin": dict(Counter(r["Dm"].sub for r in dm_a.values())),
        "Dm_class_a_where_D0_stored_none": sum(r["D0"].klass in right for r in dm_a.values()),
        "Dm_class_a_where_D1_stored_none": sum(r["D1"].klass in right for r in dm_a.values()),
        "Dm_class_a_where_D0_also_class_a": sum(r["D0"].klass == "a" for r in dm_a.values()),
        "Dm_class_a_where_D1_also_class_a": sum(r["D1"].klass == "a" for r in dm_a.values()),
        "replaced_by_program": {
            "n": len(replaced),
            "stored_none": len(replaced) - len(stored_value),
            "stored_value": len(stored_value),
            "stored_value_share_of_deleted_item_times": ex.vec_mean(
                _membership(keys, stored_value), draws
            ),
            "stored_value_D0_stored": dict(
                Counter("none" if r["D0"].value is None else "value" for r in stored_value.values())
            ),
            "stored_value_D1_state": dict(
                Counter(
                    "unverifiable"
                    if r["D1"].status != "valid"
                    else "none"
                    if r["D1"].value is None
                    else "value"
                    for r in stored_value.values()
                )
            ),
            "stored_value_provenance": dict(Counter(v["kind"] for v in prov.values())),
            "stored_value_provenance_by_slot_kind": {
                kind: dict(sorted(c.items())) for kind, c in sorted(by_kind.items())
            },
        },
        "program_right_but_overridden": sum(
            r["Dm"].prog_use == "overridden" and r["Dm"].prog_class == "correct"
            for r in rows.values()
        ),
        "program_wrong_and_overridden": sum(
            r["Dm"].prog_use == "overridden" and r["Dm"].prog_class != "correct"
            for r in rows.values()
        ),
    }
    return out, prov


# ---------------------------------------------------------------------------------- examples
def _cell_view(raw: Raw, joint: Joint, key: Key, cell: str) -> dict[str, Any]:
    w, fact, _ = key
    r = joint[key][cell]
    item = raw.items[system(cell, 1), w, fact][0]
    view: dict[str, Any] = {
        "class": f"{r.klass}/{r.sub}" if r.sub else r.klass,
        "status": r.status,
        "stored_value": r.value,
        "reader": {"status": r.reader_status, "value": r.reader_value},
        "served_text": item["s_k"][:500] if r.status == "valid" else None,
    }
    if r.status != "valid":
        view["not_served"] = "the item is unverifiable"
    if cell == "Dm":
        view["program"] = {
            "ran_ok": r.prog_ok,
            "output": r.prog_value,
            "scored_alone": r.prog_class,
            "use": r.prog_use,
            "request_key": r.prog_key,
            "text": None,
            "text_note": "the program text is in the response cache on the compute host only",
        }
        det = item.get("det") or {}
        view["last_write_check"] = {"check": det.get("check"), "reason": det.get("reason")}
    return view


EXAMPLES: tuple[tuple[str, str], ...] = (
    ("a_replacement_task", "a deleted task: the program returns another task's due date"),
    (
        "a_replacement_contact",
        "a deleted contact card: the program returns another contact's field",
    ),
    (
        "a_writer_kept",
        "the program returns none, but the passed span check keeps the writer's value",
    ),
    ("d_reader", "the stored 'no value' is right and the reader answers `answer` without a value"),
    ("c_unchanged", "an unchanged item whose program failed: nothing is served"),
)


def pick_examples(
    raw: Raw, joint: Joint, worlds: Worlds, prov: Mapping[Key, dict[str, Any]]
) -> list[dict[str, Any]]:
    """The first item-time at the last checkpoint, in world and item order, that fits each pattern
    (no choice among candidates). A replacement example needs D0 and D1 to store none and the
    value not to be the one the deleted record had, so that it can only come from another record."""

    def dm(key: Key) -> Rec:
        return joint[key]["Dm"]

    def stored_none(key: Key, ref: str) -> bool:
        return joint[key][ref].klass in ("ok", "d")

    def kind(key: Key) -> str:
        return worlds.get(key[0])[0].slot(key[1]).kind

    def foreign(key: Key) -> bool:
        return key in prov and not prov[key]["equals_pre_deletion_value"]

    rules: dict[str, Callable[[Key], bool]] = {
        "a_replacement_task": lambda k: (
            dm(k).klass == "a"
            and dm(k).sub == "replacement"
            and stored_none(k, "D0")
            and stored_none(k, "D1")
            and foreign(k)
            and kind(k) == "task_due"
        ),
        "a_replacement_contact": lambda k: (
            dm(k).klass == "a"
            and dm(k).sub == "replacement"
            and stored_none(k, "D0")
            and stored_none(k, "D1")
            and foreign(k)
            and kind(k) == "contact_attr"
        ),
        "a_writer_kept": lambda k: (
            dm(k).klass == "a" and dm(k).prog_use == "overridden" and dm(k).prog_class == "correct"
        ),
        "d_reader": lambda k: (
            dm(k).klass == "d"
            and dm(k).sub == "answer_without_value"
            and joint[k]["D0"].klass == "ok"
        ),
        "c_unchanged": lambda k: (
            dm(k).klass == "c" and not dm(k).changed and joint[k]["D0"].klass == "ok"
        ),
    }
    keys = [k for k in sorted(joint) if k[2] == LAST_K]
    out = []
    for name, what in EXAMPLES:
        key = next((k for k in keys if rules[name](k)), None)
        if key is None:
            continue
        w, fact, k = key
        world = worlds.get(w)[0]
        slot = world.slot(fact)
        p, p0 = raw.probes[key], raw.probes[w, fact, 0]
        ex_row: dict[str, Any] = {
            "pattern": name,
            "what": what,
            "world": w,
            "item": fact,
            "description": slot.description,
            "kind": slot.kind,
            "checkpoint_weeks": k,
            "as_of": p["as_of"],
            "gold_before": {"status": p0["gold_status"], "value": p0["gold_value"]},
            "gold_now": {"status": p["gold_status"], "value": p["gold_value"]},
            "cells": {c: _cell_view(raw, joint, key, c) for c in CELLS},
        }
        if key in prov:
            ex_row["value_comes_from"] = prov[key]
        out.append(ex_row)
    return out


# ------------------------------------------------------------------------------------- checks
def origin_check(raw: Raw, b: int) -> dict[str, int]:
    """The recorded check of every D_m slot's last write against the origin inferred from the
    difference to D0's value: `pass` is the writer's value, `program` a replacement,
    `unverifiable` none. A replacement equal to the writer's value reads as the writer's."""
    out: Counter[str] = Counter()
    for (s, w, slot), items in sorted(raw.items.items()):
        if s != system("Dm", b) or not items:
            continue
        it, ref = items[-1], raw.items[system("D0", b), w, slot][-1]
        if b == 0 and it["history"][0]["event"] != "append":
            continue
        det = it.get("det") or {}
        vtype = raw.probes[w, slot, LAST_K]["vtype"]
        if it["status"] != "valid":
            inferred = "unverifiable"
        elif same_value(vtype, it["eval_value"], ref["eval_value"]):
            inferred = "writer_or_equal"
        else:
            inferred = "replacement"
        out[f"{det.get('check')}/{inferred}"] += 1
    return dict(sorted(out.items()))


def reproduce(joints: Mapping[int, Joint], staged: Mapping[str, Any], draws: int) -> dict[str, Any]:
    """C1 (repair, D_m - D0) and the B x D interaction of Update recomputed from the rows, against
    the staged block-C metrics, and the decomposition of B x D into the D effect at B=1 and B=0."""
    out: dict[str, Any] = {}
    j1, j0 = joints[1], joints[0]
    keys = select(j1, "repaired", "all")
    c1 = ex.paired_estimate(
        vec(j1, keys, is_class("Dm", "ok")), vec(j1, keys, is_class("D0", "ok")), draws
    )
    ref = staged["contrasts"][BRANCH]["C1_repair_D1s_minus_D0_source"]
    out["C1_repair"] = {"recomputed": c1, "staged": ref, "matches": _same_est(c1, ref)}
    split = (staged.get("explore_class_split") or {}).get("D1s", {}).get(BRANCH, {})
    for cs in CLASS_SETS:
        ch = [k for k in select(j1, "changed", cs) if k in j0]

        def u(joint: Joint, cell: str, ch: list[Key] = ch) -> ex.Vec:
            return vec(joint, ch, is_class(cell, "ok"))

        d1 = ex.paired_estimate(u(j1, "Dm"), u(j1, "D0"), draws)
        d0 = ex.paired_estimate(u(j0, "Dm"), u(j0, "D0"), draws)
        bxd = ex.paired_estimate(
            item_diffs(u(j1, "Dm"), u(j0, "Dm")), item_diffs(u(j1, "D0"), u(j0, "D0")), draws
        )
        ref_bxd = split.get("source", {}).get(cs, {}).get("BxD_interaction_update")
        out[f"update_{cs}"] = {
            "D_effect_B1": d1,
            "D_effect_B0": d0,
            "BxD": bxd,
            "BxD_staged": ref_bxd,
            "matches": ref_bxd is not None and _same_est(bxd, ref_bxd),
        }
    return out


def _same_est(a: Mapping[str, Any], b: Mapping[str, Any]) -> bool:
    return bool(abs(a["est"] - b["est"]) < 1e-9 and a["n_items"] == b["n_items"])


# ------------------------------------------------------------------------------------- the run
def analyse(stage: Path, runs: Path, out: Path | None, draws: int = ex.DRAWS) -> dict[str, Any]:
    run_dirs = stage_runs(stage, runs)
    raw = load_raw(run_dirs)
    staged = json.loads((stage / "analysis" / "e1_metrics.json").read_text())
    worlds = Worlds(raw)
    res: dict[str, Any] = {
        "explore_status": ex.STATUS,
        "explore_draws": draws,
        "explore_seed": ex.SEED,
        "inputs": {
            "stage": stage.name,
            "stage_manifest_sha256": file_sha256(stage / "manifest.json"),
            "runs": raw.runs,
            "run_manifests_sha256": {d.name: file_sha256(d / "manifest.json") for d in run_dirs},
            "branch": BRANCH,
            "worlds": len(raw.worlds),
            "cells": {c: {b: system(c, b) for b in (1, 0)} for c in CELLS},
        },
        "classes": LABELS,
    }
    joints: dict[int, Joint] = {}
    prov: dict[Key, dict[str, Any]] = {}
    for b in (1, 0):
        joint, checks = records(raw, b)
        joints[b] = joint
        checks["origin_vs_recorded_check_last_write"] = origin_check(raw, b)
        node: dict[str, Any] = {"checks": checks, "populations": {}}
        for pop in POPULATIONS:
            if pop == "repaired" and b == 0:
                continue
            for cs in CLASS_SETS:
                keys = select(joint, pop, cs)
                if not keys:
                    continue
                node["populations"].setdefault(pop, {})[cs] = {
                    **block(joint, keys, draws),
                    "program_only": program_only(joint, keys, draws),
                }
        if b == 1:
            node["rule3_deleted_records"] = {}
            for cs in CLASS_SETS:
                keys = select(joint, "changed_abstain", cs)
                node["rule3_deleted_records"][cs], p = rule3(raw, joint, worlds, keys, draws)
                prov |= p
            node["program_failures_by_slot"] = _failures_by_slot(raw)
            node["program_by_kind"] = program_by_kind(raw, joint, worlds, draws)
            node["program_agreement_at_t0_staged"] = _staged_t0(staged)
        res[f"B{b}"] = node
    res["reproduction"] = reproduce(joints, staged, draws)
    res["examples"] = pick_examples(raw, joints[1], worlds, prov)
    res["inputs"]["worlds_regenerated_match_manifest"] = all(worlds.hash_matches.values())
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(res, indent=1, default=str))
    return res


def _failures_by_slot(raw: Raw) -> dict[str, Any]:
    """D_m slots (world x item) of the B=1 cell whose last program failed, by item class and kind of
    failure, and how many distinct program requests that is (one request is shared by every world
    for an item whose description names no world data, e.g. a definition). A failure the passed
    span check masked has no recorded error (`masked`)."""
    per: Counter[str] = Counter()
    keys: dict[str, set[str]] = defaultdict(set)
    n_slots = 0
    for (s, _, slot), items in raw.items.items():
        if s != system("Dm", 1):
            continue
        n_slots += 1
        it = items[-1]
        det = it.get("det") or {}
        if det.get("program_ok") is False:
            masked = it["status"] == "valid"
            kind = "masked" if masked else error_type(det.get("reason"))
            label = f"{ex.item_class(slot)}/{kind}"
            per[label] += 1
            keys[label].add(det["program_key"])
    return {
        "slots": n_slots,
        "failed_last_write": sum(per.values()),
        "by_class_and_kind": {
            k: {"slots": v, "distinct_program_requests": len(keys[k])}
            for k, v in sorted(per.items())
        },
    }


def program_by_kind(raw: Raw, joint: Joint, worlds: Worlds, draws: int) -> dict[str, Any]:
    """Does a D_m program react to the deletion? Its output scored alone, per item kind, over the
    re-derived rows whose record is deleted (gold changed to abstain) and whose record is live
    (unchanged, gold answer). `wrong` is a value for a deleted record, a wrong value or none for an
    answer; a program that keys on the record it names is right on a live record and returns none
    on a deleted one. `wrong_equals_last_listed` / `_first_listed`: the wrong output is the field of
    the source's last / first listed live record (the programs receive them in that order)."""
    groups = {
        "deleted": lambda r: r.changed and r.gold_status == "abstain",
        "live": lambda r: not r.changed and r.gold_status == "answer",
    }
    out: dict[str, Any] = {}
    for name, member in groups.items():
        by_kind: dict[str, list[Key]] = defaultdict(list)
        for key, r in sorted(joint.items()):
            dm = r["Dm"]
            if dm.prog_class is not None and member(dm):
                by_kind[worlds.get(key[0])[0].slot(key[1]).kind].append(key)
        for kind, keys in sorted(by_kind.items()):
            flags: Counter[str] = Counter()
            for key in keys:
                dm = joint[key]["Dm"]
                if dm.prog_class in WRONG_PROGRAM and dm.prog_ok and dm.prog_value is not None:
                    w, fact, _ = key
                    slot = worlds.get(w)[0].slot(fact)
                    pv = provenance(
                        slot,
                        worlds.snapshot(w, raw.probes[key]["as_of"]),
                        dm.prog_value,
                        raw.probes[w, fact, 0],
                    )
                    flags["wrong_with_value"] += 1
                    flags["wrong_equals_last_listed"] += pv["equals_last_listed_record"]
                    flags["wrong_equals_first_listed"] += pv["equals_first_listed_record"]
            node: dict[str, Any] = {
                "n": len(keys),
                "counts": dict(sorted(Counter(joint[k]["Dm"].prog_class for k in keys).items())),
                "wrong": ex.vec_mean(
                    vec(joint, keys, lambda r: r["Dm"].prog_class in WRONG_PROGRAM), draws
                ),
                "failed": ex.vec_mean(
                    vec(joint, keys, lambda r: r["Dm"].prog_class == "failed"), draws
                ),
                **dict(sorted(flags.items())),
            }
            out.setdefault(kind, {})[name] = node
    return out


def _staged_t0(staged: Mapping[str, Any]) -> dict[str, Any]:
    """The programs' agreement with gold before any shift, per item type, as block C staged it
    (`e1_synth.program_stats`): the executable share and the share equal to gold."""
    cell = ((staged.get("synth_programs") or {}).get(BRANCH) or {}).get(system("Dm", 1)) or {}
    return {
        kind: {
            "n": v["n"],
            "executable": (v.get("exec_t0") or {}).get("est"),
            "agrees_with_gold": (v.get("agree_t0") or {}).get("est"),
        }
        for kind, v in sorted(cell.items())
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--runs", type=Path, default=Path("runs"), help="directory of the raw runs")
    ap.add_argument("--stage", default="stage4_c_e1", help="block C stage under --runs")
    ap.add_argument("--out", type=Path, default=Path("runs/stage6_ex_round3/dm_deletion.json"))
    ap.add_argument("--draws", type=int, default=ex.DRAWS)
    a = ap.parse_args()
    res = analyse(a.runs / a.stage, a.runs, a.out, a.draws)
    print(json.dumps(res["reproduction"]["C1_repair"], default=str))


if __name__ == "__main__":
    main()

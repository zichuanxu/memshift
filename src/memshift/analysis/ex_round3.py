"""Paper numbers and tables of the store and reader split of Eq. (1), the changed item-times by
cause, D_m after a deletion and the query-time view. They call no model. The sources are named in
results/sources.json:

  ex_round3  runs/stage6_ex_round3/analysis/, five files that the round-3 analysis modules write
             from the raw runs of stages e1, c and ctl, the G-b, H-a and H-c runs and the seeded
             generator (the stage manifest lists the commands; `python -m
             memshift.analysis.ex_round3` reruns them all):
               store_reader.json  `analysis.ex_round3_store_reader`: the terms r and a of Eq. (1)
                                  split by whether the stored state is right, and the B x D
                                  interaction split into a store part and a reader part
               natural.json       `analysis.ex_round3_natural`: the changed item-times by cause
                                  (the shift, drift only) as a mix of the two item classes
               dm_deletion.json   `analysis.ex_round3_dm_deletion`: why the model-written
                                  derivation D_m repairs worse than D=0 after a deletion
               view.json          `analysis.ex_round3_view`: the query-time view against the
                                  stored D_m on every item class
               writers.json       `analysis.ex_round3_writers`: the axes of the writer figure
                                  (`round3_figures.fig_writers`); no macro reads it

Macros (letters only; a stage that is not staged renders as the MISSING marker, a quantity over no
item or not defined for the class as a dash):
  ExRThreeSR<arm><branch><class>...      arm Main (authored D=1) or Dm (block C): B x D, its store
                                         and reader parts, the D effect on r split the same way; per
                                         B=1 cell (DzeroBone, DoneBone, JsonBone) r_store, u, u by
                                         gold status, r, the right stores and those the reader lost
                                         (counts also summed over the shifts: branch `Shifts`)
  ExRThreeNat<branch><part>...           changed by the shift or by drift only: the definition
                                         share, the class B x D and each class's contribution, the
                                         all-item B x D at the branch's pooled class mix; whether a
                                         committed run answers the natural branch after the fork
  ExRThreeDel...                         S_del, B=1: the classes of D=0, D=1 and D_m on the deleted
                                         records, the accounting of what D_m loses against D=0 and
                                         D=1, where its stored values come from, the programs'
                                         output alone on deleted and live records, failed programs
  ExRThreeView<group><class><metric>...  the view against the stored D_m (B=1, B=0) and the zero
                                         system, per definition type, and the costs per query

tab_store_reader      appendix: r_store, u, l and r, a_store and a by shift, class and cell
tab_dm_deletion       appendix: the deleted records by class, what D_m loses there, the programs'
                      output alone on deleted and live records
tab_view              appendix: the query-time view against the stored D_m
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memshift.analysis import ex_round2 as r2
from memshift.analysis import round3_paper as r3p
from memshift.analysis.ex_e1_paper import (
    GROUP_GAP,
    PCT_BOX,
    UNDEFINED,
    _Out,
    ci_text,
    fmt,
    mt_ci_pair_cells,
    mt_pair,
    n_template,
    pair_widths,
    point,
    tokens,
)
from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, pct, pp
from memshift.analysis.paper_tables import PAIR, _num, _row

STAGE_KEY = "ex_round3"
STAGE_DIR = "runs/stage6_ex_round3"
FILES = {
    "sr": "store_reader.json",
    "nat": "natural.json",
    "dm": "dm_deletion.json",
    "view": "view.json",
    "writers": "writers.json",
}
# the axes of the writer figure (`analysis.ex_round3_writers`): the G-b program sets of
# `round3_paper` and the H-a and H-c definition-item writers, paired with the committed stages
WRITER_PROGRAMS = r3p.G_B_RUNS
WRITER_DEFWRITERS = tuple(r for h in r3p.H_WRITERS for r in h.runs)
PRODUCERS = {
    "analysis/natural.json": (
        "python -m memshift.analysis.ex_round3_natural --check-published "
        "--out runs/stage6_ex_round3/analysis/natural.json"
    ),
    "analysis/store_reader.json": (
        "python -m memshift.analysis.ex_round3_store_reader --out runs/stage6_ex_round3/analysis"
    ),
    "analysis/dm_deletion.json": (
        "python -m memshift.analysis.ex_round3_dm_deletion "
        "--out runs/stage6_ex_round3/analysis/dm_deletion.json"
    ),
    "analysis/view.json": (
        "python -m memshift.analysis.ex_round3_view --out runs/stage6_ex_round3/analysis"
    ),
    "analysis/writers.json": " ".join(
        [
            "python -m memshift.analysis.ex_round3_writers --programs",
            *WRITER_PROGRAMS,
            "--defwriters",
            *WRITER_DEFWRITERS,
            "--compare",
            *r3p.COMPARE,
            "--out runs/stage6_ex_round3/analysis/writers.json",
        ]
    ),
}
NOTE = (
    "additional analyses, computed from the raw runs of the source stages and the seeded "
    "generator, no model call; the draft tables the store_reader and view producers also write "
    "are not staged. dm_deletion.json, store_reader.json and view.json are recomputed after "
    "stage4_c_e1 took the re-run e1_synth_s1016-1019_20261016T222156Z. natural.json reads "
    "stage2_e1 only and is the file staged on 2026-10-16: rerunning its producer now changes only "
    "its audit of natural-branch answers, which finds the block N runs made after it. writers.json "
    "(the axes of the writer figure) reads the 18 G-b program sets, whose Qwen3.5-9B, Claude "
    "Sonnet 5.5 and Claude Opus 5.5 sets are re-runs from the response cache, and the H-a and H-c "
    "runs"
)
SOURCE_KEYS = ("e1", "c", "ctl")  # the stages the four analyses read


def source_runs(sources: Mapping[str, str]) -> list[str]:
    """The stage's sources: the stages of `SOURCE_KEYS`, then every run writers.json reads, once."""
    stages = [sources[k] for k in SOURCE_KEYS]
    return list(dict.fromkeys([*stages, *WRITER_PROGRAMS, *WRITER_DEFWRITERS]))


CLASSES = (("definition", "Definition"), ("other", "Other"), ("all", "All"))
BRANCH_LABELS = r2.BRANCH_LABELS

# store x reader (analysis.ex_round3_store_reader)
SR_ARMS = (("D1", "Main"), ("D1s", "Dm"))
SR_PARTS = (
    ("interaction", "BxD"),
    ("store", "StorePart"),
    ("reader", "ReaderPart"),
    ("D_effect_r_store", "DEffectRStore"),
    ("D_effect_r_reader", "DEffectRReader"),
)
SR_CELLS = (("D0B1", "DzeroBone"), ("D1B1", "DoneBone"), ("D1B1json", "JsonBone"))
SR_TERMS = (
    ("r_store", "RStore"),
    ("u", "U"),
    ("u_answer", "UAnswer"),
    ("u_abstain", "UAbstain"),
    ("r", "R"),
)

# natural branch: composition (analysis.ex_round3_natural)
NAT_PARTS = (("shift", "Shift"), ("drift", "Drift"))

# D_m after a deletion (analysis.ex_round3_dm_deletion)
DEL_CELLS = (("D0", "Dzero"), ("D1", "Done"), ("Dm", "Dm"))
DEL_CLASSES = (("ok", "Ok"), ("a", "A"), ("b", "B"), ("c", "C"), ("d", "D"), ("e", "E"))
DEL_SUBS = (
    ("a/replacement", "AReplacement"),
    ("a/writer", "AWriter"),
    ("d/answer_without_value", "DNoValue"),
    ("d/gave_value", "DValue"),
)
DEL_REFS = (("accounting_vs_D0", "Dzero"), ("accounting_vs_D1", "Done"))
DEL_SHARES = (*DEL_CLASSES[1:], ("a_replacement", "AReplacement"), ("a_writer", "AWriter"))
DEL_PROGRAM = (
    ("correct", "Correct"),
    ("value_for_deleted", "ValueForDeleted"),
    ("failed", "Failed"),
)
DEL_KINDS = (
    ("task_due", "Task"),
    ("contact_attr", "Contact"),
    ("series", "Series"),
    ("appointment", "Appointment"),
)
DEL_GROUPS = (("deleted", "Deleted"), ("live", "Live"))
DEL_T0 = (
    ("all", "All"),
    ("task_due", "Task"),
    ("contact.employer", "ContactEmployer"),
    ("contact.city", "ContactCity"),
    ("series", "Series"),
    ("kappa.working_hours", "WorkingHours"),
)
DELETED = ("changed_abstain", "all")  # population and class set of the deleted records

# the query-time view (analysis.ex_round3_view)
VIEW_GROUPS = (*BRANCH_TAGS, ("pooled", "Pooled"))
VIEW_METRICS = (("update", "Update"), ("stability", "Stability"))
VIEW_CONTRASTS = (
    ("view_minus_DmB1", "VsDm"),
    ("view_minus_zero", "VsZero"),
    ("DmB1_minus_zero", "DmVsZero"),
    ("view_minus_DmB0", "VsDmBzero"),
)
VIEW_CELLS = (
    ("view", "View"),
    ("DmB1", "Dm"),
    ("DmB0", "DmBzero"),
    ("zero", "Zero"),
    ("D1B1", "HoldOneOne"),
)
VIEW_TYPE_CELLS = VIEW_CELLS[:4]  # per definition type the module has no HoldMem(1,1) level
VIEW_WRITE_CELLS = (("view", "View"), ("DmB1", "Dm"), ("DmB0", "DmBzero"))


# ------------------------------------------------------------------------------------ loading
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs (None where a stage or file is not there)."""
    src = dict(sources)
    return {key: stage_json(root, src, STAGE_KEY, name) for key, name in FILES.items()}


# ----------------------------------------------------------------------------------- formats
def _count(x: Any) -> str:
    return MISSING if x is None else f"{int(x):,}"


def _est(parent: Mapping[str, Any] | None, key: str, f: Callable[[float | None], str] = pct) -> str:
    """An estimate with its interval under `parent`: MISSING when the parent is not there, a dash
    when the parent is there but the estimate is not (no item of the class)."""
    if parent is None:
        return MISSING
    e = parent.get(key)
    return UNDEFINED if e is None else fmt(e, f)


def _sub_count(parent: Mapping[str, Any] | None, key: str) -> str:
    """A count under `parent` that is 0 when not listed (the counters omit empty keys)."""
    return MISSING if parent is None else f"{int(parent.get(key, 0)):,}"


def _signed_int(x: float | None) -> str:
    if x is None or x != x:
        return MISSING
    v = round(x)
    return ("$+$" if v > 0 else "$-$" if v < 0 else "") + f"{abs(v):,}"


def _tokens_iv(e: Mapping[str, Any] | None) -> str:
    """A paired difference in tokens per query, signed, with its interval."""
    if not e:
        return MISSING
    if e["est"] != e["est"]:
        return UNDEFINED
    return f"{_signed_int(e['est'])} [{_signed_int(e['lo'])}, {_signed_int(e['hi'])}]"


def _tokens_mean(e: Mapping[str, Any] | None) -> str:
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else f"{round(e['est']):,}"


def upp(x: float | None, digits: int = 1) -> str:
    """A share of item-times in percentage points, unsigned (an amount lost or gained)."""
    return MISSING if x is None or x != x else f"{100 * x:.{digits}f}\\,pp"


# ---------------------------------------------------------------------------------- macros
SR_COUNTS = (  # tag, function of a cell's counts (`ex_round3_store_reader.STATS`)
    ("Items", lambda c: c.get("n", 0)),
    ("RightStores", lambda c: c.get("ds", 0)),  # re-derived, stored state right
    ("Lost", lambda c: c.get("ds", 0) - c.get("dsc", 0)),  # ... and the answer wrong
    ("RightAbstain", lambda c: c.get("ds_abs", 0)),  # ... right, gold abstain
    ("LostAbstain", lambda c: c.get("ds_abs", 0) - c.get("dsc_abs", 0)),
)


def _sr_numbers(sr: Mapping[str, Any] | None, out: _Out) -> None:
    """Per arm, branch and class; the counts also summed over the three shifts (`Shifts`)."""
    for arm, at in SR_ARMS:
        for cls, ct in CLASSES:
            for cell, cell_tag in SR_CELLS:
                if arm != "D1" and cell == "D1B1json":
                    continue  # the JSON arm serves the authored D=1 store only
                total: dict[str, int] | None = None if sr is None else {}
                for b, bt in BRANCH_TAGS:
                    node = _get(sr, "results", arm, b, cls)
                    terms = None if node is None else _get(node, "cells", cell) or {}
                    for key, kt in SR_TERMS:
                        out.put("RThreeSR", at, bt, ct, cell_tag, kt, value=_est(terms, key))
                    c = None if node is None else _get(node, "counts", cell) or {}
                    for tag, f in SR_COUNTS:
                        x = None if c is None else int(f(c))
                        out.put("RThreeSR", at, bt, ct, cell_tag, tag, value=_count(x))
                        if total is not None:
                            total[tag] = total.get(tag, 0) + (x or 0)
                for tag, _ in SR_COUNTS:
                    x = None if total is None else total.get(tag, 0)
                    out.put("RThreeSR", at, "Shifts", ct, cell_tag, tag, value=_count(x))
            for b, bt in BRANCH_TAGS:
                node = _get(sr, "results", arm, b, cls)
                inter = None if node is None else node.get("interaction") or {}
                for key, kt in SR_PARTS:
                    out.put("RThreeSR", at, bt, ct, kt, value=_est(inter, key, pp))


def _nat_numbers(nat: Mapping[str, Any] | None, out: _Out) -> None:
    for b, bt in BRANCH_TAGS:
        comp = _get(nat, "composition", b)
        share = None if comp is None else comp.get("pooled_definition_share")
        out.put("RThreeNat", bt, "PooledDefinitionShare", value=point(share) if comp else MISSING)
        for part, pt in NAT_PARTS:
            node = None if comp is None else _get(comp, "parts", part) or {}
            n = None if node is None else node.get("n") or {}
            for cls, ct in CLASSES:
                out.put("RThreeNat", bt, pt, ct, "Items", value=_sub_count(n, cls))
                klass = None if node is None else node.get("class") or {}
                out.put("RThreeNat", bt, pt, ct, "BxD", value=_est(klass, cls, pp))
            out.put("RThreeNat", bt, pt, "DefinitionShare", value=_est(node, "definition_share"))
            contrib = None if node is None else node.get("contribution") or {}
            for cls, ct in CLASSES[:2]:
                out.put("RThreeNat", bt, pt, "Contribution", ct, value=_est(contrib, cls, pp))
            out.put("RThreeNat", bt, pt, "AtPooledMix", value=_est(node, "at_pooled_mix", pp))
    hits = _get(nat, "natural_availability", "answers_at_postfork_checkpoints")
    out.put("RThreeNatAnswersAfterFork", value=_count(hits))


def _del_population(dm: Mapping[str, Any] | None, out: _Out) -> None:
    """The deleted records (gold changed to abstain), B=1."""
    node = _get(dm, "B1", "populations", *DELETED)
    out.put("RThreeDelItems", value=_count(_get(node, "n_item_times")))
    for cell, ct in DEL_CELLS:
        share = None if node is None else _get(node, "share", cell) or {}
        counts = None if node is None else _get(node, "counts", cell) or {}
        classes = None if counts is None else counts.get("classes") or {}
        subs = None if counts is None else counts.get("sub") or {}
        for k, kt in DEL_CLASSES:
            out.put("RThreeDel", ct, kt, "Share", value=_est(share, k))
            out.put("RThreeDel", ct, kt, "Count", value=_sub_count(classes, k))
        for k, kt in DEL_SUBS:
            out.put("RThreeDel", ct, kt, "Count", value=_sub_count(subs, k))
    for ref, rt in (("Dm_minus_D0", "Dzero"), ("Dm_minus_D1", "Done")):
        diff = None if node is None else node.get(ref) or {}
        for k, kt in DEL_CLASSES:
            out.put("RThreeDelDmMinus", rt, kt, value=_est(diff, k, pp))
    for ref, rt in DEL_REFS:
        acc = None if node is None else node.get(ref) or {}
        loss = None if acc is None else acc.get("loss_by_Dm_class") or {}
        shares = None if acc is None else acc.get("loss_share_of_total") or {}
        out.put("RThreeDelLossVs", rt, value=_est(acc, "loss_total", upp))
        out.put("RThreeDelGainVs", rt, value=_est(acc, "gain_total", upp))
        for k, kt in DEL_CLASSES[1:]:
            out.put("RThreeDelLossVs", rt, kt, value=_est(loss, k, upp))
        for k, kt in DEL_SHARES:
            out.put("RThreeDelLossShareVs", rt, kt, value=_est(shares, k))
    prog = None if node is None else node.get("program_only") or {}
    out.put("RThreeDelProgItems", value=_count(_get(prog, "n_rederived")))
    for k, kt in DEL_PROGRAM:
        out.put("RThreeDelProg", kt, "Share", value=_est(_get(prog, "share"), k))


def _del_values(dm: Mapping[str, Any] | None, out: _Out) -> None:
    """Where the values D_m stored for deleted records come from."""
    r3 = _get(dm, "B1", "rule3_deleted_records", "all")
    rep = None if r3 is None else r3.get("replaced_by_program") or {}
    out.put("RThreeDelValueStored", value=_count(_get(rep, "stored_value")))
    out.put(
        "RThreeDelValueStoredShare",
        value=_est(rep, "stored_value_share_of_deleted_item_times"),
    )
    out.put(
        "RThreeDelValueStoredDzeroNone",
        value=_sub_count(_get(rep, "stored_value_D0_stored"), "none"),
    )
    out.put(
        "RThreeDelValueStoredDoneNone", value=_sub_count(_get(rep, "stored_value_D1_state"), "none")
    )
    prov = None if rep is None else rep.get("stored_value_provenance") or {}
    out.put("RThreeDelValueOtherRecord", value=_sub_count(prov, "other_record_same_field"))
    out.put("RThreeDelValueTextOnly", value=_sub_count(prov, "text_only"))
    kinds = None if rep is None else rep.get("stored_value_provenance_by_slot_kind") or {}
    for kind, kt in DEL_KINDS[:2]:
        k = None if kinds is None else kinds.get(kind) or {}
        out.put("RThreeDelValue", kt, value=_sub_count(k, "n"))
        out.put(
            "RThreeDelValue", kt, "LastListed", value=_sub_count(k, "equals_last_listed_record")
        )
        out.put(
            "RThreeDelValue", kt, "FirstListed", value=_sub_count(k, "equals_first_listed_record")
        )
        out.put(
            "RThreeDelValue", kt, "PreDeletion", value=_sub_count(k, "equals_pre_deletion_value")
        )
    out.put("RThreeDelProgRightOverridden", value=_count(_get(r3, "program_right_but_overridden")))


def _del_programs(dm: Mapping[str, Any] | None, out: _Out) -> None:
    """The programs' output alone, by item kind, on deleted and live records; failed programs; the
    programs' agreement with gold before any shift."""
    by_kind = _get(dm, "B1", "program_by_kind")
    for kind, kt in DEL_KINDS:
        for group, gt in DEL_GROUPS:
            node = None if by_kind is None else _get(by_kind, kind, group)
            if by_kind is not None and node is None:
                node = {}  # staged, but no item of the kind in the group
            out.put("RThreeDelProg", kt, gt, "Items", value=_sub_count(node, "n"))
            out.put("RThreeDelProg", kt, gt, "Wrong", value=_est(node, "wrong"))
            out.put("RThreeDelProg", kt, gt, "Failed", value=_est(node, "failed"))
            for flag, ft in (
                ("wrong_equals_last_listed", "Last"),
                ("wrong_equals_first_listed", "First"),
            ):
                out.put(
                    "RThreeDelProg", kt, gt, "Wrong", ft, "Listed", value=_sub_count(node, flag)
                )
    fails = _get(dm, "B1", "program_failures_by_slot")
    syntax = (
        None if fails is None else _get(fails, "by_class_and_kind", "definition/syntax_error") or {}
    )
    out.put("RThreeDelSlots", value=_count(_get(fails, "slots")))
    out.put("RThreeDelFailedSlots", value=_count(_get(fails, "failed_last_write")))
    out.put("RThreeDelFailDefinitionSyntaxSlots", value=_sub_count(syntax, "slots"))
    out.put(
        "RThreeDelFailDefinitionSyntaxRequests",
        value=_sub_count(syntax, "distinct_program_requests"),
    )
    t0 = _get(dm, "B1", "program_agreement_at_t0_staged")
    for kind, kt in DEL_T0:
        node = None if t0 is None else t0.get(kind) or {}
        exe = None if node is None else node.get("executable")
        agree = None if node is None else node.get("agrees_with_gold")
        staged = node is not None
        out.put("RThreeDelTzero", kt, "Items", value=_sub_count(node, "n"))
        out.put("RThreeDelTzero", kt, "Executable", value=point(exe) if staged else MISSING)
        out.put("RThreeDelTzero", kt, "Agree", value=point(agree) if staged else MISSING)


def _del_contrasts(dm: Mapping[str, Any] | None, out: _Out) -> None:
    """D_m - D0 in ContractValid on the repaired items (C1), the changed items (Update), the items
    whose gold is an answer after the shift and the unchanged items, by class; loss shares."""
    pops = _get(dm, "B1", "populations")
    for pop, pt in (("repaired", "COne"), ("changed", "Update"), ("unchanged", "Stable")):
        for cls, ct in CLASSES:
            node = None if pops is None else _get(pops, pop, cls) or {}
            diff = None if node is None else node.get("Dm_minus_D0") or {}
            out.put("RThreeDel", pt, ct, value=_est(diff, "ok", pp))
            out.put("RThreeDel", pt, ct, "Items", value=_sub_count(node, "n_item_times"))
        node = None if pops is None else _get(pops, pop, "all") or {}
        shares = (
            None if node is None else _get(node, "accounting_vs_D0", "loss_share_of_total") or {}
        )
        for k, kt in DEL_CLASSES[1:]:
            out.put("RThreeDel", pt, "LossShare", kt, value=_est(shares, k))
        dm_share = None if node is None else _get(node, "share", "Dm") or {}
        out.put("RThreeDel", pt, "DmCShare", value=_est(dm_share, "c"))
    answer = None if pops is None else _get(pops, "changed_answer", "all") or {}
    for ref, rt in (("Dm_minus_D0", "Dzero"), ("Dm_minus_D1", "Done")):
        diff = None if answer is None else answer.get(ref) or {}
        out.put("RThreeDelAnswerVs", rt, value=_est(diff, "ok", pp))
    out.put("RThreeDelAnswerItems", value=_sub_count(answer, "n_item_times"))


def _view_numbers(view: Mapping[str, Any] | None, out: _Out) -> None:
    for g, gt in VIEW_GROUPS:
        for cls, ct in CLASSES:
            for metric, mt in VIEW_METRICS:
                con = _get(view, "contrasts", g, cls, metric)
                lev = _get(view, "levels", g, cls, metric)
                if view is not None:
                    con, lev = con or {}, lev or {}
                for key, kt in VIEW_CONTRASTS:
                    out.put("RThreeView", gt, ct, mt, kt, value=_est(con, key, pp))
                out.put(
                    "RThreeView",
                    gt,
                    ct,
                    mt,
                    "Items",
                    value=_count(_get(con, VIEW_CONTRASTS[0][0], "n_items")),
                )
                for key, kt in VIEW_CELLS:
                    out.put("RThreeView", gt, ct, mt, "Level", kt, value=_est(lev, key))
    types = _get(view, "per_definition_type")
    for d, dt, _ in r2.DEFINITIONS:
        for metric, mt in VIEW_METRICS:
            node = None if types is None else _get(types, d, metric) or {}
            out.put("RThreeViewType", dt, mt, "VsDm", value=_est(node, "view_minus_DmB1", pp))
            out.put("RThreeViewType", dt, mt, "VsZero", value=_est(node, "view_minus_zero", pp))
            n = _get(node, "view_minus_DmB1", "n_items")
            out.put("RThreeViewType", dt, mt, "Items", value=_count(n))
            lev = None if node is None else node.get("levels") or {}
            for key, kt in VIEW_TYPE_CELLS:
                out.put("RThreeViewType", dt, mt, "Level", kt, value=_est(lev, key))
            for cls, ct in (CLASSES[0], CLASSES[2]):
                rest = None if node is None else _get(node, "without_type", cls) or {}
                out.put(
                    "RThreeViewWithout", dt, ct, mt, "VsDm", value=_est(rest, "view_minus_DmB1", pp)
                )
                n_rest = _get(rest, "view_minus_DmB1", "n_items")
                out.put("RThreeViewWithout", dt, ct, mt, "Items", value=_count(n_rest))
    costs = _get(view, "costs")
    if view is not None and not (costs or {}).get("available"):
        costs = {}
    for g, gt in VIEW_GROUPS:
        node = _get(costs, "serving_per_query", g, "total")
        for key, kt in VIEW_CELLS:
            out.put("RThreeViewServe", gt, kt, value=_tokens_mean(_get(node, "mean", key)))
        for key, kt in VIEW_CONTRASTS[:2]:
            out.put("RThreeViewServe", gt, kt, value=_tokens_iv(_get(node, "paired", key)))
    req = _get(costs, "view_program_requests")
    out.put("RThreeViewProgramCalls", value=_count(_get(req, "program_calls")))
    out.put("RThreeViewProgramAnswers", value=_count(_get(req, "answers")))
    per_call = _get(req, "tokens_per_call")
    out.put(
        "RThreeViewProgramTokensPerCall",
        value=MISSING if per_call is None else f"{round(per_call):,}",
    )
    cached = _get(req, "cached_share")
    out.put("RThreeViewProgramCachedShare", value=MISSING if cached is None else pct(cached))
    write = _get(costs, "write_side_per_world_branch")
    for key, kt in VIEW_WRITE_CELLS:
        for b, bt in BRANCH_TAGS:
            w = _get(write, key, b)
            if write is not None and w is None:
                w = {}
            ingest = _get(w, "ingest_tokens_per_world_branch")
            maintain = _get(w, "maintain_tokens_per_world_branch")
            cpu = _get(w, "verify_cpu_s_per_world_branch")
            stored = _get(w, "storage_bytes_per_world_branch")
            out.put("RThreeViewWrite", kt, bt, "Ingest", value=_amount(w, ingest, tokens))
            out.put("RThreeViewWrite", kt, bt, "Maintain", value=_amount(w, maintain, tokens))
            out.put(
                "RThreeViewWrite", kt, bt, "Sandbox", value=_amount(w, cpu, lambda x: f"{x:,.0f}")
            )
            out.put(
                "RThreeViewWrite",
                kt,
                bt,
                "Stored",
                value=_amount(w, stored, lambda x: f"{x / 1e3:,.0f}"),
            )


def _amount(node: Mapping[str, Any] | None, x: float | None, f: Callable[[float], str]) -> str:
    """A write-side amount: MISSING without the stage, a dash where the cell has none."""
    if node is None:
        return MISSING
    return UNDEFINED if x is None or x != x else f(x)


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every macro of this block, from `load`."""
    out = _Out()
    _sr_numbers(inp.get("sr"), out)
    _nat_numbers(inp.get("nat"), out)
    _del_population(inp.get("dm"), out)
    _del_values(inp.get("dm"), out)
    _del_programs(inp.get("dm"), out)
    _del_contrasts(inp.get("dm"), out)
    _view_numbers(inp.get("view"), out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ---------------------------------------------------------------------------------- tables
def _pc(e: Mapping[str, Any] | None, digits: int = 1) -> str:
    """A share in percent, point estimate, unsigned."""
    if not e:
        return MISSING
    return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=digits)


SR_ROWS = (  # label, arm, B=1 cell, B=0 cell
    (r"$D{=}0$", "D1", "D0B1", "D0B0"),
    (r"$D{=}1$", "D1", "D1B1", "D1B0"),
    (r"$D{=}1$, JSON", "D1", "D1B1json", None),
    (r"$D_{\mathrm{m}}$", "D1s", "D1B1", "D1B0"),
)


def tab_store_reader(inp: Mapping[str, Any]) -> str:
    """Appendix: per shift, class and cell the store and reader terms of r (B=1) and a (B=0)."""
    sr = _get(inp.get("sr"), "results")
    body: list[list[str] | str] = []  # rows of cells, and the gaps and rules between them
    for i, (b, _) in enumerate(BRANCH_TAGS):
        for k, cls in enumerate(("definition", "other")):
            n = _get(sr, "D1", b, cls, "counts", "D1B1", "n")
            for j, (label, arm, c1, c0) in enumerate(SR_ROWS):
                hi = _get(sr, arm, b, cls, "cells", c1)
                lo = _get(sr, arm, b, cls, "cells", c0) if c0 else None
                b0 = (
                    [_pc(_get(lo, "a_store")), _pc(_get(lo, "a"))] if c0 else [UNDEFINED, UNDEFINED]
                )
                body.append(
                    [
                        BRANCH_LABELS[b] if (k == 0 and j == 0) else "",
                        cls if j == 0 else "",
                        label,
                        (_count(n) if j == 0 else "") if sr is not None else MISSING,
                        *(_pc(_get(hi, t)) for t in ("r_store", "u", "u_answer", "u_abstain")),
                        _pc(_get(hi, "l")),
                        _pc(_get(hi, "r")),
                        *b0,
                    ]
                )
            if k == 0:
                body.append(r"\mtgap")
        if i < len(BRANCH_TAGS) - 1:
            body.append(r"\midrule")
    rows = [r for r in body if isinstance(r, list)]
    box = [f"N{{{n_template([r[c] for r in rows])}}}" for c in range(3, 12)]  # n, B=1, B=0
    lines = [
        r"\begin{table}[t]",
        r"\caption{The stored value and the reader behind $r$ and $a$ of \cref{eq:decomp}, in "
        r"percent (point estimates).}",
        r"\label{tab:store-reader}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{6pt}",
        rf"\begin{{mttabular}}{{@{{}}lll {box[0]} {''.join(box[1:7])} {GROUP_GAP} "
        rf"{''.join(box[7:])}@{{}}}}",
        r"\toprule",
        r"& & & & \multicolumn{6}{c}{$B{=}1$: re-derived} & \multicolumn{2}{c@{}}{$B{=}0$} \\",
        r"\cmidrule(lr){5-10}\cmidrule(l){11-12}",
        r"Shift & Class & Cell & \multicolumn{1}{c}{$n$} "
        r"& \multicolumn{1}{c}{$r_{\mathrm{store}}$} & \multicolumn{1}{c}{$u$} "
        r"& \multicolumn{1}{c}{$u$, answer} & \multicolumn{1}{c}{$u$, abstain} "
        r"& \multicolumn{1}{c}{$l$} & \multicolumn{1}{c}{$r$} "
        r"& \multicolumn{1}{c}{$a_{\mathrm{store}}$} & \multicolumn{1}{c@{}}{$a$} \\",
        r"\midrule",
        *(
            r if isinstance(r, str) else _row(*r[:3], *(_na(c) for c in r[3:-1]), _na(r[-1], True))
            for r in body
        ),
        r"\bottomrule",
        r"\end{mttabular}",
        r"\mtnotes{$n$: changed item-times. $r_{\mathrm{store}}$: the share of re-derived "
        r"item-times whose stored state (status and typed value) is the gold. $u$, $l$: accuracy "
        r"given a right or a wrong store, so $r=r_{\mathrm{store}}u+(1{-}r_{\mathrm{store}})l$. "
        r"$u$ is also given by gold status: answer, or abstain where the right state is no value. "
        r"$a_{\mathrm{store}}$, $a$: the same at $B{=}0$, where the store is the newest served "
        r"item. JSON: the $D{=}1$ store served as JSON records.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _na(cell: str, edge: bool = False) -> str:
    """A cell of an N column, a dash centred (`\\mtna`; `c@{}` at the right edge of the table)."""
    if cell != UNDEFINED:
        return cell
    return rf"\multicolumn{{1}}{{c@{{}}}}{{{UNDEFINED}}}" if edge else r"\mtna{}"


DEL_TABLE_CLASSES = (
    ("ok", "correct"),
    ("a", r"value stored ($a$)"),
    ("d", r"reader ($d$)"),
    ("c", r"failed ($c$)"),
    ("e", r"not re-derived ($e$)"),
)
DEL_TABLE_KINDS = (
    ("task_due", "task due date"),
    ("contact_attr", "contact field"),
    ("series", "recurring series"),
)


def tab_dm_deletion(inp: Mapping[str, Any]) -> str:
    """Appendix, two tables: the deleted records by class per cell beside what D_m loses against
    D=0 there, by the class D_m falls in, in one grid (`tab:dm-deletion`); and the programs'
    output alone on deleted and live records (`tab:dm-programs`)."""
    dm = inp.get("dm")
    node = _get(dm, "B1", "populations", *DELETED)
    acc = _get(node, "accounting_vs_D0")
    n = _get(node, "n_item_times")
    shares = _get(acc, "loss_share_of_total")
    loss = _get(acc, "loss_by_Dm_class")
    staged = acc is not None
    blank = ["", "", ""]
    levels = {  # the item-times of each class, per cell
        k: [_pc(_get(node, "share", c, k)) for c in ("D0", "D1", "Dm")]
        for k, _ in DEL_TABLE_CLASSES
    }
    box = [f"N{{{n_template([v[c] for v in levels.values()])}}}" for c in range(3)]
    lines = [
        r"\begin{table}[t]",
        r"\caption{Why the writer's programs ($D_{\mathrm{m}}$) repair worse than $D{=}0$ "
        rf"after a deletion: the {_count(n)} changed item-times whose record was deleted (gold "
        r"\texttt{abstain}) by what the cell stored and how the reader answered, and what "
        r"$D_{\mathrm{m}}$ loses against $D{=}0$ there.}",
        r"\label{tab:dm-deletion}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}l {''.join(box)} {GROUP_GAP} {PAIR} {PAIR}@{{}}}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Item-times (\%)} & \multicolumn{4}{c@{}}{$D_{\mathrm{m}}$ against "
        r"$D{=}0$} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-8}",
        r"Class & \multicolumn{1}{c}{$D{=}0$} & \multicolumn{1}{c}{$D{=}1$} "
        r"& \multicolumn{1}{c}{$D_{\mathrm{m}}$} & \multicolumn{2}{c}{Loss (pp)} "
        r"& \multicolumn{2}{c@{}}{Share of the loss (\%)} \\",
        r"\midrule",
    ]
    for k, label in DEL_TABLE_CLASSES:
        if k == "ok":  # the loss is split by the class D_m falls in, not by its correct answers
            lines.append(_row(label, *levels[k], "& & &"))  # the four cells of the pairs empty
            continue
        lines.append(
            _row(
                label,
                *levels[k],
                mt_ci_pair_cells(_get(loss, k)),
                mt_ci_pair_cells(_get(shares, k), span="c@{}"),
            )
        )
        if k == "a":
            lines.append(
                _row(
                    r"\mtin the program's value & & &",  # and its three empty levels
                    _span(UNDEFINED if staged else MISSING),
                    mt_ci_pair_cells(_get(shares, "a_replacement"), span="c@{}"),
                )
            )
    lines += [
        r"\mtgap",
        _row(
            "lost",
            *blank,
            mt_ci_pair_cells(_get(acc, "loss_total")),
            _alone("100") if staged else _span(MISSING, "c@{}"),
        ),
        _row(
            "gained",
            *blank,
            mt_ci_pair_cells(_get(acc, "gain_total")),
            _span(UNDEFINED if staged else MISSING, "c@{}"),
        ),
        r"\bottomrule",
        r"\end{mttabular}",
        r"\mtnotes{$a$: a value stored where gold is no value. $d$: the stored state is right but "
        r"the reader's answer is wrong. $c$: the replacement failed, so nothing is served. $e$: "
        r"not re-derived at the checkpoint. Loss: the item-times $D{=}0$ answers right and "
        r"$D_{\mathrm{m}}$ wrong, by the class $D_{\mathrm{m}}$ falls in.}",
        r"\end{table}",
        "",
    ]
    by_kind = _get(dm, "B1", "program_by_kind")
    rows = []
    for kind, label in DEL_TABLE_KINDS:
        cells = [label]
        for group, _ in DEL_GROUPS:
            g = _get(by_kind, kind, group)
            span = "c@{}" if group == DEL_GROUPS[-1][0] else "c"
            cells += [_count(_get(g, "n")), mt_ci_pair_cells(_get(g, "wrong"), span=span)]
        rows.append(cells)
    deleted, live = (f"N{{{n_template([r[c] for r in rows])}}}" for c in (1, 3))
    lines += [
        r"\begin{table}[t]",
        r"\caption{The writer's programs alone after a deletion, on the re-derived item-times of "
        r"deleted records and of live records that did not change. Wrong: a value for a deleted "
        r"record, or a wrong value or none for a live one.}",
        r"\label{tab:dm-programs}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular}}{{@{{}}l {deleted} {PAIR} {GROUP_GAP} {live} {PAIR}@{{}}}}",
        r"\toprule",
        r"& \multicolumn{3}{c}{Deleted record} & \multicolumn{3}{c@{}}{Live record} \\",
        r"\cmidrule(lr){2-4}\cmidrule(l){5-7}",
        r"Program & \multicolumn{1}{c}{$n$} & \multicolumn{2}{c}{Wrong (\%)} "
        r"& \multicolumn{1}{c}{$n$} & \multicolumn{2}{c@{}}{Wrong (\%)} \\",
        r"\midrule",
        *(_row(*r) for r in rows),
        r"\bottomrule",
        r"\end{mttabular}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def _span(text: str, spec: str = "c") -> str:
    """One value across both cells of an estimate-and-interval column pair (`spec` `c@{}` at the
    right edge of the table)."""
    return rf"\multicolumn{{2}}{{{spec}}}{{{text}}}"


def _alone(est: str) -> str:
    """An estimate without an interval in an estimate-and-interval column pair."""
    return f"{est} & "


VIEW_TABLE_CELLS = ("view", "DmB1", "DmB0", "zero")
VIEW_TABLE_CONTRASTS = ("view_minus_DmB1", "view_minus_zero", "DmB1_minus_zero")
VIEW_GROUP_LABELS = {**BRANCH_LABELS, "pooled": "pooled"}


def tab_view(inp: Mapping[str, Any]) -> str:
    """Appendix: per metric (Update, Stability), group and class the view, the stored D_m with and
    without binding, the zero system and three paired differences."""
    view = inp.get("view")
    worlds = _get(view, "sources", "worlds") or []
    ncol = 3 + len(VIEW_TABLE_CELLS) + len(VIEW_TABLE_CONTRASTS)
    dm = r"$D_{\mathrm{m}}$"
    metrics = (
        ("update", "Update (changed item-times)"),
        ("stability", "Stability (unchanged item-times)"),
    )
    every = [  # the contrasts of every row
        _get(view, "contrasts", g, cls, metric)
        for metric, _ in metrics
        for g, _ in VIEW_GROUPS
        for cls, _ in CLASSES
    ]
    n_box = n_template([_count(_get(cn, "view_minus_DmB1", "n_items")) for cn in every])
    # each column of paired differences sets its pairs in boxes of its widest estimate and interval
    pairs = "".join(
        rf">{{{pair_widths([ci_text(_get(cn, c), sign=True) for cn in every], template=True)}}}c"
        for c in VIEW_TABLE_CONTRASTS
    )
    share = f"N{{{PCT_BOX}}}"  # a column of whole percentages
    lines = [
        r"\begin{table}[t]",
        r"\caption{The query-time view against the cell with the stored programs on every item "
        rf"class ({len(worlds) or MISSING} worlds, the same probes).}}",
        r"\label{tab:view}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{4pt}",
        rf"\begin{{mttabular*}}{{\linewidth}}{{@{{}}ll N{{{n_box}}} "
        rf"{share * len(VIEW_TABLE_CELLS)} {pairs}@{{}}}}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{Level (\%)} & \multicolumn{3}{c@{}}{Paired difference (pp)} \\",
        r"\cmidrule(lr){4-7}\cmidrule(l){8-10}",
        rf"Shift & Class & \multicolumn{{1}}{{c}}{{$n$}} & \multicolumn{{1}}{{c}}{{View}} "
        rf"& \multicolumn{{1}}{{c}}{{{dm}}} & \multicolumn{{1}}{{c}}{{$B{{=}}0$}} "
        rf"& \multicolumn{{1}}{{c}}{{No-store}} & View $-$ {dm} & View $-$ no-store "
        rf"& \multicolumn{{1}}{{c@{{}}}}{{{dm} $-$ no-store}} \\",
    ]
    for metric, title in metrics:
        lines += [r"\midrule", rf"\mtgroup{{{ncol}}}{{{title}}}"]
        for i, (g, _) in enumerate(VIEW_GROUPS):
            for k, (cls, _) in enumerate(CLASSES):
                lv = _get(view, "levels", g, cls, metric)
                cn = _get(view, "contrasts", g, cls, metric)
                n = _get(cn, "view_minus_DmB1", "n_items")
                lines.append(
                    _row(
                        rf"\mtin {VIEW_GROUP_LABELS[g]}" if k == 0 else "",
                        cls,
                        _count(n),
                        *(_pc(_get(lv, c), digits=0) for c in VIEW_TABLE_CELLS),
                        *(mt_pair(_get(cn, c), sign=True) for c in VIEW_TABLE_CONTRASTS),
                    )
                )
            if i < len(VIEW_GROUPS) - 1:
                lines.append(r"\mtgap")
    lines += [
        r"\bottomrule",
        r"\end{mttabular*}",
        r"\mtnotes{View: no-store BM25 plus the writer's program run on the live records at the "
        rf"probe's time, with nothing stored. {dm}: \holdmem{{}} with the same programs, stored "
        r"with the source key. $B{=}0$: the same without a key. No-store: no-store BM25. $n$: "
        r"paired item-times. Pooled: the three shifts together.}",
        r"\end{table}",
    ]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {
        "tab_store_reader.tex": tab_store_reader(inp),
        "tab_dm_deletion.tex": tab_dm_deletion(inp),
        "tab_view.tex": tab_view(inp),
    }


# ----------------------------------------------------------------------------------- staging
def stage(root: Path, out: Path, workers: int = 6) -> None:
    """Rerun the five analyses from the raw runs of the source stages into OUT/analysis and write
    the stage manifest (the commands of `PRODUCERS`; needs the raw runs; about ten minutes)."""
    from memshift.analysis import ex_round3_dm_deletion as dm_deletion
    from memshift.analysis import ex_round3_natural as natural
    from memshift.analysis import ex_round3_store_reader as store_reader
    from memshift.analysis import ex_round3_view as view
    from memshift.analysis import ex_round3_writers as writers
    from memshift.analysis.stage import write_stage_manifest

    sources = json.loads((root / "results" / "sources.json").read_text())
    analysis = out / "analysis"
    natural.run(root, sources["e1"], analysis / FILES["nat"], None, True, True, workers)
    store_reader.analyse(root, analysis)
    dm_deletion.analyse(root / sources["c"], root / "runs", analysis / FILES["dm"])
    view.analyse(root, analysis)
    for draft in ("tab_store_reader.tex", "tab_view.tex"):  # drafts, not staged
        (analysis / draft).unlink(missing_ok=True)
    res = writers.analyse(
        [root / r for r in WRITER_PROGRAMS],
        [root / r for r in WRITER_DEFWRITERS],
        [root / c for c in r3p.COMPARE],
    )
    res["inputs"] = {  # relative to the root, as the producer names them
        "programs": list(WRITER_PROGRAMS),
        "defwriters": list(WRITER_DEFWRITERS),
        "compare": list(r3p.COMPARE),
    }
    (analysis / FILES["writers"]).write_text(json.dumps(res, indent=1, default=str) + "\n")
    srcs = [root / s for s in source_runs(sources)]
    write_stage_manifest(out, STAGE_KEY, srcs, {"producers": PRODUCERS, "note": NOTE})


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage the round-3 analyses (see the docstring).")
    ap.add_argument("--root", type=Path, default=Path("."))
    ap.add_argument("--out", type=Path, default=None, help=f"default: ROOT/{STAGE_DIR}")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    stage(a.root, a.out or a.root / STAGE_DIR, a.workers)


if __name__ == "__main__":
    main()

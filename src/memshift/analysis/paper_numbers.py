"""Emit LaTeX macros for every number the paper states (tables and figures regenerate from
runs). Sources are listed in results/sources.json (macro group -> run directory whose
analysis/*.json is committed); world statistics are recomputed from the seeded generator.
Missing inputs yield a visible placeholder (\\textbf{??}) instead of a number.

Robustness stages (stage inputs in `analysis.paper_data`): Rep* replication worlds (block R,
with RepPool* the forty pooled worlds), Synth* model-written derivations (C), MOne* / MTwo* /
MThree* model families (M1-M3), Nat* native systems (N), Rend* LLM-rendered worlds (L), Rank*
rank correlation, Mult* multiplicity corrections over the E1 contrasts expected to be non-zero;
`Ex` marks the further analyses of the main-study outputs; `Ctl` the control cells
(`analysis.controls_paper`);
`KeyAudit` the offline audit of the binding keys (`analysis.key_audit`); `Rend*TokenShare` the
token shares of block L (`analysis.l_tokens`). A
stage that has not been staged renders as MISSING, and a count over a whole set (families,
systems, switches) only once the set is complete.

Usage: python -m memshift.analysis.paper_numbers > paper/generated/numbers.tex
"""

from __future__ import annotations

import json
import re
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.e2 import T1_SYSTEM
from memshift.analysis.e5 import DOCUMENTED_LABELS, _inferred
from memshift.analysis.paper_data import (
    BRANCH_TAGS,
    FAMILIES,
    FAMILY_ESTIMATES,
    HORIZON_TAGS,
    ITEM_TYPES,
    MISSING,
    NATIVE_METRICS,
    NATIVES,
    SYSTEM_TAGS,
    TIERS,
    TYPE_GROUPS,
    camel,
    in_group,
    label_text,
    load_stage4,
    pooled_rate,
    pooled_view,
    replication_view,
    tier_key,
)
from memshift.analysis.paper_data import get as _get
from memshift.gen.fidelity import check_world
from memshift.gen.shifts import generate_branch
from memshift.textutil import approx_tokens

CELL_NAMES = (
    ("holdmem_D0B1_source", "DzeroBone"),
    ("holdmem_D1B1_source", "DoneBone"),
    ("holdmem_D0B0", "DzeroBzero"),
    ("holdmem_D1B0", "DoneBzero"),
)
CLASS_NAMES = (("rule", "Rule"), ("definition", "Definition"), ("norule", "Freetext"))
METRIC_NAMES = (
    ("MissedInvalidation", "Missed"),
    ("RepairCorrectness", "Repair"),
    ("Update", "Update"),
    ("Stability", "Stability"),
)


ZERO_WIDTH = re.compile(r"(?P<est>\S+) \[(?P<lo>[^,\]]+), (?P<hi>[^\]]+)\]")


def _macro(name: str, value: str) -> str:
    """One `\\newcommand`. An estimate whose printed interval has zero width ("0.0 pp [0.0, 0.0]")
    is printed alone, as the paper states for such estimates."""
    if not re.fullmatch(r"[A-Za-z]+", name):
        raise ValueError(f"macro names must be letters only: {name}")
    m = ZERO_WIDTH.fullmatch(value)
    if m and _bare(m["est"]) == m["lo"] == m["hi"]:
        value = m["est"]
    return f"\\newcommand{{\\{name}}}{{{value}}}"


def pct(x: float | None, digits: int = 1) -> str:
    return MISSING if x is None or x != x else f"{100 * x:.{digits}f}\\%"


def num(x: float | None, digits: int = 2) -> str:
    return MISSING if x is None or x != x else f"{x:.{digits}f}"


def pp(x: float | None, digits: int = 1) -> str:
    """A difference in percentage points, signed."""
    if x is None or x != x:
        return MISSING
    v = round(100 * x, digits)
    sign = "$+$" if v > 0 else "$-$" if v < 0 else ""
    return f"{sign}{abs(v):.{digits}f}\\,pp"


def _bare(s: str) -> str:
    return s.replace("\\%", "").replace("\\,pp", "")


def ci(e: dict[str, Any] | None, f: Callable[[float | None], str] = pct) -> str:
    """Estimate with its interval; the unit is printed once, after the estimate."""
    if not e:
        return MISSING
    return f"{f(e.get('est'))} [{_bare(f(e.get('lo')))}, {_bare(f(e.get('hi')))}]"


def with_estimates(macros: dict[str, str]) -> dict[str, str]:
    """Add `<name>Est` (the estimate alone) for every macro that carries an interval."""
    out = dict(macros)
    for k, v in macros.items():
        if " [" in v and not v.startswith(MISSING):
            out.setdefault(k + "Est", v.split(" [", 1)[0])
    return out


# The size of a world is reported over the 40 evaluation worlds (test 1000-1019, replication
# 1020-1039); the fidelity check runs on the development worlds, as the appendix states.
EVAL_SEEDS = list(range(1000, 1040))


def world_size(seeds: list[int]) -> dict[str, str]:
    events, tokens = [], []
    for s in seeds:
        w = generate_branch(s, "natural")
        events.append(len(w.events))
        tokens.append(sum(approx_tokens(e.payload["text"]) for e in w.events if e.op != "delete"))
    return {
        "WorldEventsMin": f"{min(events):,}",
        "WorldEventsMax": f"{max(events):,}",
        "WorldTokensM": f"{sum(tokens) / len(tokens) / 1e6:.1f}",
    }


def world_stats(seeds: list[int]) -> dict[str, str]:
    rates = [check_world(generate_branch(s, "natural")).rate for s in seeds]
    from memshift.experiments.e7 import implicit_ceiling

    ceiling = implicit_ceiling(seeds)
    return {
        "EImplicitCeiling": f"{pct(ceiling['ceiling'])} of {ceiling['items']} items",
        "WorldFidelity": pct(min(rates)),
    }


def e1_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for b, tag in (("src", "Src"), ("def", "Def"), ("orphan", "Orphan")):
        c = _get(m, "contrasts", b) or {}
        cells = _get(m, "cells", b) or {}
        out[f"EOneBEffectPooled{tag}"] = ci(c.get("B_effect_update_pooled_source"), pp)
        out[f"EOneBEffectPooledCited{tag}"] = ci(c.get("B_effect_update_pooled_cited"), pp)
        out[f"EOneBxD{tag}"] = ci(c.get("BxD_interaction_update_source"), pp)
        out[f"EOneBEffectDzero{tag}"] = ci(c.get("B_effect_update_D0_source"), pp)
        out[f"EOneBEffectDone{tag}"] = ci(c.get("B_effect_update_D1_source"), pp)
        for s, name in (*CELL_NAMES, ("zero", "Zero"), ("rag", "Rag"), ("oracle", "Oracle")):
            out[f"EOneCvPost{name}{tag}"] = ci(_get(cells, s, "ContractValid_post"))
        out[f"EOneDEffectMissed{tag}"] = ci(c.get("D_effect_missed_source"), pp)
        out[f"EOneDEffectMissedCited{tag}"] = ci(c.get("D_effect_missed_cited"), pp)
        out[f"EOneDEffectFalse{tag}"] = ci(c.get("D_effect_false_inval_source"), pp)
        out[f"EOneDEffectRepair{tag}"] = ci(c.get("D_effect_repair_source"), pp)
        out[f"EOneDEffectRepairMech{tag}"] = ci(c.get("D_effect_repair_mech_source"), pp)
        out[f"EOneDEffectRepairChanged{tag}"] = ci(c.get("D_effect_repair_changed_source"), pp)
        out[f"EOneGranFalse{tag}"] = ci(c.get("granularity_false_inval_D1_cited"), pp)
        out[f"EOneGranMissed{tag}"] = ci(c.get("granularity_missed_D1_cited"), pp)
        for key, name in (("fallback", "Fallback"), ("check_only", "CheckOnly")):
            out[f"EOneEx{name}Repair{tag}"] = ci(c.get(f"explore_{key}_effect_repair_source"), pp)
            for cls, cname in CLASS_NAMES:  # by item class
                out[f"EOneEx{name}Repair{cname}{tag}"] = ci(
                    c.get(f"explore_{key}_effect_repair_source_{cls}"), pp
                )
        for cls, cname in CLASS_NAMES:  # by item class
            out[f"EOneExDRepair{cname}{tag}"] = ci(
                c.get(f"explore_D_effect_repair_source_{cls}"), pp
            )
        for cell, name in CELL_NAMES:
            for metric, mname in METRIC_NAMES:
                out[f"EOne{mname}{name}{tag}"] = ci(_get(cells, cell, metric))
    pre = m.get("pre_intervention", {})
    out["EOneJsonMinusSk"] = ci(pre.get("json_minus_sk"), pp)
    out["EOneCellsIdentical"] = "yes" if pre.get("cells_identical_answers") else MISSING
    out["EOneWorlds"] = str(m.get("n_worlds", MISSING))
    return out


def nofb_numbers(m: dict[str, Any]) -> dict[str, str]:
    """The no-fallback split from the stage that merges the E1 test runs with the no-fallback
    cell; the other E1 macros stay with group e1."""
    return {
        k: v for k, v in e1_numbers(m).items() if k.startswith(("EOneExFallback", "EOneExCheck"))
    }


def e2_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    p3 = m.get("path3") or {}
    out["ETwoPathThreeSlope"] = (
        f"{num(_get(p3, 'pooled_slope', 'est'))} (90\\% lower bound "
        f"{num(_get(p3, 'pooled_slope', 'lo90'))})"
        if p3
        else MISSING
    )
    out["ETwoPathThreeAuroc"] = num(_get(p3, "pooled_auroc", "est"))
    n_items = _get(p3, "pooled_slope", "n_items")
    out["ETwoPathThreeN"] = f"{int(n_items):,}" if n_items else MISSING
    out["ETwoTierCaveats"] = ", ".join(p3.get("tier_caveats", [])) or "none"
    for tier, tag in (("same", "Same"), ("upgrade", "Up"), ("downgrade", "Down")):
        cal = _get(m, "calibration", tier) or {}
        slope = cal.get("slope") or {}
        out[f"ETwoSlope{tag}"] = (
            f"{num(slope.get('est'))} (90\\% {num(slope.get('lo90'))}--{num(slope.get('hi90'))})"
            if slope
            else MISSING
        )
        out[f"ETwoAuroc{tag}"] = num(_get(cal, "auroc", "est"))
        for cell, name in CELL_NAMES:
            out[f"ETwoRetention{name}{tag}"] = ci(_get(m, "retention", f"{tier}:{cell}"))
            for cls, cname in CLASS_NAMES:  # by item class
                out[f"ETwoExRetention{name}{cname}{tag}"] = ci(
                    _get(m, "explore_retention_by_class", f"{tier}:{cell}:{cls}")
                )
        t1 = _get(m, "t1", tier) or {}
        out[f"ETwoTOne{tag}"] = ci(t1.get("delta_90"), pp)
        out[f"ETwoTOnePass{tag}"] = (
            MISSING if not t1 else ("within" if t1.get("equivalent") else "outside")
        )
        for cell, name in CELL_NAMES:  # T1 for the cells other than the primary one
            if cell == T1_SYSTEM:
                continue
            t1c = _get(m, "explore_t1_cells", tier, cell) or {}
            d90 = t1c.get("delta_90")
            out[f"ETwoExTOne{name}{tag}"] = ci(d90, pp)
            out[f"ETwoExTOneRetention{name}{tag}"] = ci(
                d90 and {k: 1 + d90[k] for k in ("est", "lo", "hi")}
            )
            out[f"ETwoExTOnePass{name}{tag}"] = (
                MISSING if not t1c else ("within" if t1c.get("equivalent") else "outside")
            )
        dc = _get(m, "d1_vs_checker", tier) or {}
        out[f"ETwoCheckerPass{tag}"] = pct(dc.get("checker_pass_rate_calibration"))
        out[f"ETwoDoneSurvival{tag}"] = pct(dc.get("d1_b1_survival"))
        out[f"ETwoCalN{tag}"] = str(_get(cal, "slope", "n_items") or MISSING)
    tiers = ("same", "upgrade", "downgrade")
    for cell, name in CELL_NAMES:
        if cell != T1_SYSTEM:  # tiers within the margin, once all three are staged
            flags = [_get(m, "explore_t1_cells", t, cell, "equivalent") for t in tiers]
            out[f"ETwoExTOneWithin{name}"] = (
                _count(sum(f is True for f in flags), len(tiers))
                if all(f is not None for f in flags)
                else MISSING
            )
    return out


def e3_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    systems = (
        ("zero", "Zero"),
        ("rag", "Rag"),
        ("holdmem_D1B1_source", "Hold"),
        ("oracle", "Oracle"),
    )
    for h, tag in ((1, "One"), (3, "Three"), (12, "Twelve")):
        per = _get(m, "horizons", h) or {}
        for s, name in systems:
            out[f"EThreeCv{name}{tag}"] = ci(_get(per, s, "contract_valid"))
            for cls, cname in CLASS_NAMES:  # by item class
                out[f"EThreeExCv{name}{cname}{tag}"] = ci(_get(per, s, "explore_by_class", cls))
        out[f"EThreeZeroMinusHold{tag}"] = ci(per.get("zero_minus_holdmem"), pp)
        for s, name in systems[:3]:
            out[f"EThreeMinusOracle{name}{tag}"] = ci(_get(per, s, "minus_oracle"), pp)
        out[f"EThreeServeRatioZero{tag}"] = pct(_get(per, "zero", "serve_ratio_to_full_reread"), 2)
        for s, name in systems:  # without definition items (system and oracle)
            nd = _get(per, s, "explore_no_definition") or {}
            out[f"EThreeExNoDefCv{name}{tag}"] = ci(nd.get("contract_valid"))
        for s, name in systems[:3]:
            nd = _get(per, s, "explore_no_definition") or {}
            out[f"EThreeTThreeResult{name}{tag}"] = _pass(_get(per, s, "T3_pass"))
            out[f"EThreeExNoDefMinusOracle{name}{tag}"] = ci(nd.get("minus_oracle"), pp)
            out[f"EThreeExNoDefTThreeResult{name}{tag}"] = _pass(nd.get("T3_pass"))
    out.update(t3_counts("EThree", m))
    return out


def t3_counts(prefix: str, m: dict[str, Any]) -> dict[str, str]:
    """Horizons at which each system passes T3, with all items (`<prefix>TThreePass<System>` and
    `...All`) and without definition items (`<prefix>ExNoDefTThreePass<System>` and `...All`), as
    "k of n"; a count is rendered only when every (horizon, system) unit it covers is staged."""
    out: dict[str, str] = {}
    for label, field in (("TThree", None), ("ExNoDefTThree", "explore_no_definition")):
        passed: dict[str, list[bool | None]] = {}
        for h, _ in HORIZON_TAGS:
            for s, _name in SYSTEM_TAGS:
                entry = _get(m, "horizons", h, s) or {}
                entry = (entry.get(field) or {}) if field else entry
                passed.setdefault(s, []).append(entry.get("T3_pass"))
        for s, name in SYSTEM_TAGS:
            flags = passed.get(s, [])
            out[f"{prefix}{label}Pass{name}"] = (
                _count(sum(f is True for f in flags), len(flags))
                if flags and all(f is not None for f in flags)
                else MISSING
            )
        allflags = [f for flags in passed.values() for f in flags]
        out[f"{prefix}{label}PassAll"] = (
            _count(sum(f is True for f in allflags), len(allflags))
            if allflags and all(f is not None for f in allflags)
            else MISSING
        )
    return out


SOURCES = (
    ("calendar", "Calendar"),
    ("contacts", "Contacts"),
    ("email", "Email"),
    ("tasks", "Tasks"),
    ("notes", "Notes"),
    ("messages", "Messages"),
)


def e4_numbers(m: dict[str, Any]) -> dict[str, str]:
    """Stage-1 E4 (development worlds, calendar only), lag 1 minute."""
    c1 = _get(m, "contrasts", "L1") or {}
    c5 = _get(m, "contrasts", "L5") or {}  # the lag of the test-world extension
    return {
        "EFourDevInteraction": ci(c1.get("interaction_B_all"), pp),
        "EFourDevInteractionFive": ci(c5.get("interaction_B_all"), pp),
        "EFourDevWorlds": str(m.get("n_worlds") or MISSING),
    }


def e4x_numbers(m: dict[str, Any]) -> dict[str, str]:
    """E4 extension (test worlds, six sources): the calendar interaction over all probes, the
    other sources and the pooled value."""
    con = m.get("contrasts") or {}
    lag = sorted(con)[0] if con else None
    c = con.get(lag, {}) if lag else {}
    out = {
        "EFourXLag": MISSING if lag is None else lag.lstrip("L"),
        "EFourXWorlds": str(m.get("n_worlds") or MISSING),
        "EFourXProbes": str(m.get("n_probes") or MISSING),
        "EFourXPooled": ci(_get(c, "all", "pooled", "interaction_B"), pp),
        "EFourXPooledTouched": ci(_get(c, "touched", "pooled", "interaction_B"), pp),
    }
    for src, name in SOURCES:
        out[f"EFourX{name}"] = ci(_get(c, "all", src, "interaction_B"), pp)
        out[f"EFourX{name}Touched"] = ci(_get(c, "touched", src, "interaction_B"), pp)
        out[f"EFourX{name}N"] = str(_get(m, "n_probes_by_source", src, "all") or MISSING)
        for sys_, sname in (
            ("holdmem_D1B1_source", "Bone"),
            ("holdmem_D1B0", "Bzero"),
            ("zero", "Zero"),
        ):
            out[f"EFourX{name}Drop{sname}"] = ci(_get(c, "all", src, f"drop_{sys_}"), pp)
    return out


def e6_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for system, name in (("imported_with_ext", "WithExt"), ("imported_no_ext", "NoExt")):
        imp = _get(m, "imports", system) or {}
        d = imp.get("diff_90") or {}
        out[f"ESix{name}Pre"] = ci(d.get("pre"), pp)
        out[f"ESix{name}Post"] = ci(d.get("post"), pp)
        out[f"ESix{name}Missed"] = ci(d.get("missed"), pp)
        out[f"ESix{name}Equivalent"] = (
            MISSING if not imp else ("equivalent" if imp.get("t2_equivalent") else "not equivalent")
        )
        fid = imp.get("export_fidelity") or {}
        out[f"ESix{name}FidelityPre"] = num(fid.get("pre"), 3)
        out[f"ESix{name}FidelityPost"] = num(fid.get("post"), 3)
    return out


def e5_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for system, name in (("mem0", "Mem"), ("graphiti", "Graphiti")):
        lab = _get(m, "labels", system) or {}
        for dim in ("B", "D"):
            label = _get(lab, dim, "label")
            out[f"EFive{name}{dim}"] = (
                MISSING if not lab else ("--" if label is None else str(label))
            )
            # B: orphan persistence P; D: structured repair precision R (descriptive only)
            out[f"EFive{name}{dim}Rate"] = pct(_get(lab, dim, "rate"))
            out[f"EFive{name}{dim}N"] = str(_get(lab, dim, "n") if lab else MISSING)
        per = _get(m, "systems", system) or {}
        out[f"EFive{name}Worlds"] = str(len(per.get("worlds", [])) or MISSING)
        for b, tag in (("src", "Src"), ("def", "Def"), ("orphan", "Orphan")):
            br = _get(per, "branches", b) or {}
            nf = _get(m, "noise_floor", "systems", system, b) or {}
            out[f"EFive{name}NoiseDiffer{tag}"] = pct(nf.get("share_differ"))
            for metric in ("Update", "Stability"):
                out[f"EFive{name}{metric}{tag}"] = ci(_get(br, "native", metric))
                out[f"EFive{name}{metric}Pred{tag}"] = ci(_get(br, "white_box", metric))
                ok = _get(br, "consistent", metric)
                out[f"EFive{name}{metric}Consistent{tag}"] = (
                    MISSING if ok is None else ("inside" if ok else "outside")
                )
                out[f"EFive{name}{metric}Dev{tag}"] = pp(_get(br, "deviation", metric))
                out[f"EFive{name}Noise{metric}{tag}"] = pp(_get(nf, "diff", metric))
        fail = _get(m, "failures", system) or {}
        out[f"EFive{name}FailRate"] = pct(fail.get("rate"), 2)
        silent = _get(m, "silent_failures", system) or {}
        out[f"EFive{name}ParseErrorRate"] = pct(silent.get("rate"))
        out[f"EFive{name}ParseErrors"] = (
            f"{silent['parse_errors']:,.0f}" if silent.get("runs") else MISSING
        )
        units = fail.get("ingested", 0) + fail.get("failed", 0)
        out[f"EFive{name}Units"] = f"{units:,.0f}" if fail else MISSING
        nw = _get(m, "noise_floor", "worlds")
        out[f"EFive{name}NoiseWorlds"] = str(len(nw)) if nw else MISSING
    # Discriminability of the six comparisons: each native estimate placed against the
    # interval of the HoldMem cell with the other B label
    per = _get(m, "systems", "mem0") or {}
    out.update(
        disc_numbers("EFiveExMem", per, _get(m, "stage4", "discriminability", "per_system", "mem0"))
    )
    # Label-rate intervals, and the HoldMem cells on the same worlds
    ex = m.get("white_box_reference") or {}
    for system, name in (
        ("mem0", "Mem"),
        ("holdmem_D0B0", "DzeroBzero"),
        ("holdmem_D1B0", "DoneBzero"),
        ("holdmem_D0B1_source", "DzeroBone"),
        ("holdmem_D1B1_source", "DoneBone"),
    ):
        for dim in ("B", "D"):
            out[f"EFiveEx{name}{dim}Rate"] = ci(_get(ex, "label_rates", system, dim))
        for b, tag in (("src", "Src"), ("def", "Def"), ("orphan", "Orphan")):
            for metric in ("Update", "Stability"):
                out[f"EFiveEx{name}{metric}{tag}"] = pct(
                    _get(ex, "white_box_cells", system, b, metric, "est")
                )
    # T3 / T4 on the short worlds (natives and the baselines run on the same worlds)
    for system, name in (("mem0", "Mem"), ("oracle", "Oracle"), ("zero", "Zero"), ("rag", "Rag")):
        mem = _get(m, "membership", system) or {}
        out[f"EFive{name}Static"] = ci(mem.get("static"))
        out[f"EFive{name}TThree"] = ci(mem.get("minus_oracle"), pp)
        t3 = mem.get("T3_pass")
        out[f"EFive{name}TThreeResult"] = MISSING if t3 is None else ("pass" if t3 else "fail")
        out[f"EFive{name}TFour"] = pct(mem.get("serve_ratio_to_full_reread"), 2)
        t4 = mem.get("T4_pass")
        out[f"EFive{name}TFourResult"] = MISSING if t4 is None else ("pass" if t4 else "fail")
    return out


def disc_numbers(prefix: str, per: dict[str, Any], counts: dict[str, Any] | None) -> dict[str, str]:
    """Discriminability of one native system (`analysis.e5`): the HoldMem cell with the
    other B label (`<prefix>OtherReference`), the counts of comparisons (of six: Update and
    Stability after S_src, S_def and the orphan branch) with a predicted interval that contain the
    native estimate, that lie inside the other label's interval, and that discriminate (inside the
    predicted interval, outside the other label's), and per branch and metric the other label's
    interval and both flags."""
    out: dict[str, str] = {}
    other = per.get("other_label_cell")
    out[f"{prefix}OtherReference"] = reference_label(other) if other else MISSING
    n = (counts or {}).get("comparisons")
    for key, name in (
        ("comparisons", "Comparisons"),
        ("inside_predicted", "InsidePredicted"),
        ("inside_other", "InsideOther"),
        ("discriminating", "Discriminating"),
    ):
        out[f"{prefix}{name}"] = str(counts[key]) if n and counts else MISSING
    out[f"{prefix}DiscriminatingOf"] = (
        _count(counts["discriminating"], n) if n and counts else MISSING
    )
    for b, btag in BRANCH_TAGS:
        br = _get(per, "branches", b) or {}
        for metric, mname in NATIVE_METRICS:
            d = _get(br, "discrimination", metric)
            out[f"{prefix}{mname}Other{btag}"] = ci(_get(br, "white_box_other", metric))
            out[f"{prefix}{mname}InsideOther{btag}"] = (
                MISSING if not d else ("inside" if d["inside_other"] else "outside")
            )
            out[f"{prefix}{mname}Discriminates{btag}"] = (
                MISSING if not d else _flag(d["discriminates"], "yes", "no")
            )
    return out


def e5wb_numbers(suffix: str) -> Callable[[dict[str, Any]], dict[str, str]]:
    """White-box validation of the (D, B) inference probes (per-cell Cohen's kappa)."""

    def fn(m: dict[str, Any]) -> dict[str, str]:
        k = m.get("kappa") or {}
        return {f"EFiveKappa{dim}{suffix}": num(k.get(dim), 2) for dim in ("B", "D")}

    return fn


def e5cost_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for system, name in (("mem0", "Mem"), ("graphiti", "Graphiti")):
        r = _get(m, "by_system", system) or {}
        t = r.get("t_i") or {}
        out[f"EFiveCost{name}Calls"] = num(t.get("llm_calls"), 1)
        out[f"EFiveCost{name}Prompt"] = MISSING if not t else f"{t['prompt_tokens'] / 1000:.1f}k"
        out[f"EFiveCost{name}Completion"] = MISSING if not t else f"{t['completion_tokens']:,.0f}"
        wall = r.get("wall_s_per_unit")
        out[f"EFiveCost{name}Wall"] = MISSING if wall is None else f"{wall:.0f}"
        out[f"EFiveCost{name}Units"] = str(r.get("attempted") or MISSING)
    return out


def gate_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    out["GatePathOneEstimate"] = ci(_get(m, "path1", "estimate"), pp)
    checks = _get(m, "path2", "checks") or {}
    largest = max((abs(c["estimate"]["est"]) for c in checks.values()), default=None)
    out["GatePathTwoMaxAbs"] = pp(largest)
    out["GateWorlds"] = str(m.get("n_worlds_e1") or MISSING)
    return out


def e7_numbers(m: dict[str, Any]) -> dict[str, str]:
    out: dict[str, str] = {}
    for s, name in (*CELL_NAMES, ("zero", "Zero")):
        conflict = _get(m, "conflict", s) or {}
        cons = _get(m, "cons", s) or {}
        out[f"ESevenAbstain{name}"] = ci(conflict.get("AbstainAccuracy"))
        out[f"ESevenControl{name}"] = ci(conflict.get("ControlContractValid"))
        out[f"ESevenLeak{name}"] = ci(cons.get("ScopeLeak"))
        out[f"ESevenBlock{name}"] = ci(cons.get("BlockAccuracy"))
        out[f"ESevenOverBlock{name}"] = ci(cons.get("OverBlock"))
    out["ESevenConflictItems"] = str(
        _get(m, "conflict", "zero", "AbstainAccuracy", "n_items") or MISSING
    )
    out["ESevenConsItems"] = str(_get(m, "cons", "zero", "ScopeLeak", "n_items") or MISSING)
    out["ESevenWorlds"] = str(_get(m, "conflict", "zero", "AbstainAccuracy", "n_worlds") or MISSING)
    imp = m.get("implicit") or {}
    out["ESevenImplicitCeiling"] = pct(imp.get("ceiling"))
    out["ESevenImplicitItems"] = str(imp.get("items", MISSING))
    for key, tname in (("theta_e", "ThetaE"), ("theta_r", "ThetaR")):
        sw = m.get(key) or {}
        out[f"ESeven{tname}Worlds"] = str(len(sw.get("worlds", [])) or MISSING)
        for b, tag in (("src", "Src"), ("def", "Def")):
            for s, name in (*CELL_NAMES, ("zero", "Zero"), ("rag", "Rag")):
                for arm, aname in (("main", "Main"), ("swap", "Swap")):
                    est = _get(sw, b, "systems", s, arm, "ContractValid_post")
                    out[f"ESeven{tname}{name}{aname}{tag}"] = ci(est)
            for c, cname in (
                ("D_effect_repair_source", "DRepair"),
                ("B_effect_update_pooled_source", "BUpdate"),
            ):
                for arm, aname in (("main", "Main"), ("swap", "Swap")):
                    out[f"ESeven{tname}{cname}{aname}{tag}"] = ci(
                        _get(sw, b, "contrasts", c, arm), pp
                    )
    rs = m.get("readset") or {}
    for b, tag in (("src", "Src"), ("def", "Def")):
        for g, gname in (("source", "Source"), ("cited", "Cited")):
            for cls, cname in (("predicate", "Predicate"), ("other", "Other")):
                out[f"ESevenReadset{gname}{cname}{tag}"] = ci(rs.get(f"{b}:holdmem_D1B1_{g}:{cls}"))
    return out


# ------------------------------------------------------------------------ robustness stages
# Every macro below is generated from a staged run analysis; a stage that has not run renders as
# MISSING (never a placeholder value), and a count that pools a set of systems, families or
# switches is only rendered once the whole set is staged (`<Block>Staged` counts what is there).
# Prefixes: Rep (block R), Synth (C), MOne / MTwo / MThree (M1-M3), Nat (N), Rend (L), Rank (rank
# correlation), Mult (multiplicity corrections); `Ex` marks the further analyses.
R_BASE = (
    "B_effect_detected_D0",
    "B_effect_detected_D1",
    "B_effect_update_D0",
    "B_effect_update_D1",
    "D_effect_missed",
    "D_effect_false_inval",
    "D_effect_repair",
    "D_effect_repair_mech",
    "B_effect_update_pooled",
    "BxD_interaction_update",
    "D_effect_repair_changed",
)
R_KEYS = (
    *(f"{k}_{g}" for g in ("source", "cited") for k in R_BASE),
    *(f"granularity_{k}_D{d}_cited" for d in (0, 1) for k in ("false_inval", "missed")),
)
R_TIERS = (("same", "Same"), ("upgrade", "Up"), ("downgrade", "Down"))
HORIZON_SYSTEMS = (*SYSTEM_TAGS, ("oracle", "Oracle"))


class _Macros(dict[str, str]):
    """The macros of one stage-4 block: defining a name twice (two quantities that spell alike)
    is an error, not a silent overwrite."""

    def __setitem__(self, key: str, value: str) -> None:
        if key in self:
            raise ValueError(f"macro {key} defined twice")
        super().__setitem__(key, value)

    def update(self, *args: Any, **kwargs: Any) -> None:
        for key, value in dict(*args, **kwargs).items():
            self[key] = value


def _flag(ok: bool | None, yes: str, no: str) -> str:
    return MISSING if ok is None else (yes if ok else no)


def _count(hits: int, total: int) -> str:
    return f"{hits} of {total}"


def _pass(ok: bool | None) -> str:
    return _flag(ok, "pass", "fail")


def _worlds(n: Any) -> str:
    return str(n) if n else MISSING


def _tex_text(s: str) -> str:
    return s.replace("_", r"\_")


def rep_numbers(view: dict[str, Any]) -> dict[str, str]:
    """Block R: every E1 contrast of the main study on the replication worlds, beside
    the stage-2 estimate; binding's effect pooled over D, the D effect on missed invalidation and
    writer agreement, T1, T3 and T4 on the replication worlds."""
    out = _Macros()
    rows = view.get("rows") or {}
    for branch, btag in BRANCH_TAGS:
        for key in R_KEYS:
            row = rows.get((branch, key)) or {}
            name = f"Rep{camel(key)}{btag}"
            out[name] = ci(row.get("rep"), pp)
            out[f"{name}Orig"] = ci(row.get("orig"), pp)
    effects = [r for r in rows.values() if r["kind"] == "effect"]
    for suffix, part in (
        ("", effects),
        *(
            (g.title(), [r for r in effects if r["contrast"].endswith(f"_{g}")])
            for g in ("source", "cited")
        ),
    ):
        out[f"RepEffects{suffix}"] = str(len(part)) if part else MISSING
    # Direction against the stated one (every effect expected non-zero is expected positive), per
    # stage ("Orig": stage 2, "": R) and granularity.
    for suffix, part in (
        ("", effects),
        ("Source", [r for r in effects if r["contrast"].endswith("_source")]),
        ("Cited", [r for r in effects if r["contrast"].endswith("_cited")]),
    ):
        for stage, field in (("Orig", "orig"), ("", "rep")):
            ests = [r[field] for r in part]
            counts = {
                "Directional": len(ests),
                "ExcludesZero": sum(e["lo"] > 0 or e["hi"] < 0 for e in ests),
                "ExcludesPredicted": sum(e["lo"] > 0 for e in ests),
                "ExcludesOpposite": sum(e["hi"] < 0 for e in ests),
                "EstimatePredicted": sum(bool(e["est"] > 0) for e in ests),
            }
            for name, n in counts.items():
                out[f"Rep{stage}{name}{suffix}"] = str(n) if part else MISSING
    n = view.get("n_worlds") or {}
    out["RepWorlds"], out["RepWorldsOrig"] = _worlds(n.get("rep")), _worlds(n.get("orig"))
    paths = view.get("paths") or {}
    checks = _get(paths, "path2", "checks") or {}
    largest = max((abs(c["estimate"]["est"]) for c in checks.values()), default=None)
    out.update(
        {
            "RepPathOneEstimate": ci(_get(paths, "path1", "estimate"), pp),
            "RepPathTwoMaxAbs": pp(largest),
            "RepPathThreeSlope": (
                f"{num(_get(paths, 'path3', 'pooled_slope', 'est'))} (90\\% lower bound "
                f"{num(_get(paths, 'path3', 'pooled_slope', 'lo90'))})"
                if paths.get("path3")
                else MISSING
            ),
            "RepPathThreeAuroc": num(_get(paths, "path3", "pooled_auroc", "est")),
        }
    )
    t1 = view.get("t1") or {}
    for tier, tag in R_TIERS:
        m = t1.get(tier) or {}
        out[f"RepTOne{tag}"] = ci(m.get("rep_delta_90"), pp)
        out[f"RepTOnePass{tag}"] = MISSING if not m else ("within" if m["rep"] else "outside")
    for h, htag in HORIZON_TAGS:
        for s, stag in HORIZON_SYSTEMS:
            unit = f"{h}m:{s}"
            cv = (view.get("cv") or {}).get(unit) or {}
            out[f"RepCv{stag}{htag}"] = ci(cv.get("rep"))
            out[f"RepCvOrig{stag}{htag}"] = ci(cv.get("orig"))
        for s, stag in SYSTEM_TAGS:
            unit = f"{h}m:{s}"
            t3, t4 = (view.get("t3") or {}).get(unit) or {}, (view.get("t4") or {}).get(unit) or {}
            out[f"RepTThree{stag}{htag}"] = ci(t3.get("rep_minus_oracle"), pp)
            out[f"RepTThreeResult{stag}{htag}"] = _pass(t3.get("rep")) if t3 else MISSING
            out[f"RepTFour{stag}{htag}"] = pct(t4.get("ratio"), 2)
            out[f"RepTFourResult{stag}{htag}"] = _pass(t4.get("rep")) if t4 else MISSING
    return out


def rep_pool_numbers(view: dict[str, Any]) -> dict[str, str]:
    """Block R, secondary: stage 2 and R pooled over the forty test worlds."""
    out = _Macros()
    con = view.get("contrasts") or {}
    for branch, btag in BRANCH_TAGS:
        for key in R_KEYS:
            out[f"RepPool{camel(key)}{btag}"] = ci(_get(con, branch, key), pp)
    out["RepPoolWorlds"] = _worlds(view.get("n_worlds"))
    paths = view.get("paths") or {}
    checks = _get(paths, "path2", "checks") or {}
    out.update(
        {
            "RepPoolPathOneEstimate": ci(_get(paths, "path1", "estimate"), pp),
            "RepPoolPathTwoMaxAbs": pp(
                max((abs(c["estimate"]["est"]) for c in checks.values()), default=None)
            ),
        }
    )
    for tier, tag in R_TIERS:
        t1 = (view.get("t1") or {}).get(tier) or {}
        out[f"RepPoolTOne{tag}"] = ci(t1.get("delta_90"), pp)
        out[f"RepPoolTOnePass{tag}"] = (
            MISSING if not t1 else ("within" if t1["equivalent"] else "outside")
        )
    for h, htag in HORIZON_TAGS:
        for s, stag in SYSTEM_TAGS:
            t3 = (view.get("t3") or {}).get(f"{h}m:{s}") or {}
            out[f"RepPoolTThree{stag}{htag}"] = ci(t3.get("minus_oracle"), pp)
            out[f"RepPoolTThreeResult{stag}{htag}"] = _pass(t3.get("T3_pass")) if t3 else MISSING
    return out


def rep_e3_numbers(orig: dict[str, Any] | None, rep: dict[str, Any] | None) -> dict[str, str]:
    """E3 on the replication worlds beside stage 2. `RepZeroMinusHold<H>`: the zero system minus
    HoldMem(1,1), paired on the replication worlds (stage 2: `EThreeZeroMinusHold<H>`), so both
    values of a comparison come from one world set; with `RepCv...` (ContractValid before any
    shift) and `RepTThree...` (minus the oracle) that gives every E3 quantity on the R worlds.
    `Ex`: ContractValid and T3 without definition items."""
    out = _Macros()
    for h, htag in HORIZON_TAGS:
        per = _get(rep, "horizons", h) or {}
        out[f"RepZeroMinusHold{htag}"] = ci(per.get("zero_minus_holdmem"), pp)
        for s, stag in HORIZON_SYSTEMS:
            nd = _get(per, s, "explore_no_definition") or {}
            out[f"RepExNoDefCv{stag}{htag}"] = ci(nd.get("contract_valid"))
        for s, stag in SYSTEM_TAGS:
            nd = _get(per, s, "explore_no_definition") or {}
            out[f"RepExNoDefTThree{stag}{htag}"] = ci(nd.get("minus_oracle"), pp)
            out[f"RepExNoDefTThreeResult{stag}{htag}"] = _pass(nd.get("T3_pass"))
    out.update(t3_counts("Rep", rep or {}))
    return out


PROGRAM_METRICS = (
    ("Exec", "exec_t0"),
    ("Agree", "agree_t0"),
    ("AgreeExec", "agree_t0_given_exec"),
)
PROGRAM_SYSTEM = "holdmem_D1sB1_source"  # every D1s cell runs the same programs at t0


def type_tag(item_type: str) -> str:
    return camel(item_type.replace("u.", "user.", 1))


def program_stats(m: dict[str, Any]) -> dict[str, Any]:
    """Block C's per-item-type program statistics on the S_src branch (the world before any
    definition change, so every derive event is a t0 program)."""
    return _get(m, "synth_programs", "src", PROGRAM_SYSTEM) or {}


def synth_numbers(m: dict[str, Any]) -> dict[str, str]:
    """Block C: C1-C4 for the D1s cell with intervals; the share of the D effect (C2); the
    orphan branch and the cited-cluster granularity carry `Ex`; program statistics at t0. Also
    with `Ex`: C1 and C2 by item class (`SynthExCOne<Class><Branch>`, `SynthExCTwo<Class><Branch>`
    with the classes of `CLASS_NAMES`, and `SynthExCItems<Class><Branch>` the paired items)."""
    out = _Macros()
    con = m.get("contrasts") or {}
    out["SynthWorlds"] = _worlds(m.get("n_worlds"))
    for b, btag in BRANCH_TAGS:
        c = con.get(b) or {}
        for g, gtag, pre in (("source", "", ""), ("cited", "Cited", "explore_")):
            ex = "Ex" if b == "orphan" or g == "cited" else ""
            for name, key, f in (
                ("COne", f"C1_repair_D1s_minus_D0_{g}", pp),
                ("CTwo", f"C2_repair_D1_minus_D1s_{g}", pp),
                ("CTwoShare", f"C2_share_of_D_effect_{g}", pct),
                ("CThree", f"C3_BxD_interaction_update_D1s_{g}", pp),
                ("CFour", f"C4_D1s_effect_missed_{g}", pp),
            ):
                out[f"Synth{ex}{name}{gtag}{btag}"] = ci(c.get(pre + key), f)
    for b, btag in BRANCH_TAGS:  # C1 and C2 by item class, source-epoch binding
        c = con.get(b) or {}
        for cls, cname in CLASS_NAMES:
            for name, key in (
                ("COne", "C1_repair_D1s_minus_D0"),
                ("CTwo", "C2_repair_D1_minus_D1s"),
            ):
                out[f"SynthEx{name}{cname}{btag}"] = ci(c.get(f"explore_{key}_source_{cls}"), pp)
            n_items = _get(c, f"explore_C1_repair_D1s_minus_D0_source_{cls}", "n_items")
            out[f"SynthExCItems{cname}{btag}"] = f"{int(n_items):,}" if n_items else MISSING
    stats = program_stats(m)
    for kind in [*ITEM_TYPES, "all"]:
        entry = stats.get(kind)
        tag = "All" if kind == "all" else type_tag(kind)
        out[f"SynthItems{tag}"] = str(entry["n"]) if entry else MISSING
        for name, key in PROGRAM_METRICS:
            r = (entry or {}).get(key)
            out[f"Synth{name}{tag}"] = "n/a" if entry and r is None else ci(r)
    out["SynthExecPostAll"] = ci(_get(stats, "all", "exec_post"))
    for gtag, _label, prefixes in TYPE_GROUPS:
        types = [t for t in ITEM_TYPES if in_group(t, prefixes) and t in stats]
        out[f"SynthItemsGroup{gtag}"] = str(sum(stats[t]["n"] for t in types)) if types else MISSING
        for name, key in PROGRAM_METRICS:
            out[f"Synth{name}Group{gtag}"] = pct(pooled_rate(stats, types, key))
    return out


SWITCH_PROSE = {  # the three tiers in running prose (the tables keep `Tier.label`)
    "Same": "the same-tier switch",
    "Up": "the switch up to Qwen3.8-27B",
    "Down": "the switch down from Qwen3.8-27B",
}


def _and(items: list[str]) -> str:
    """A list in prose with the serial comma ("a", "a and b", "a, b, and c"); "none" when
    empty."""
    if not items:
        return "none"
    if len(items) < 3:
        return " and ".join(items)
    return f"{', '.join(items[:-1])}, and {items[-1]}"


def m1_numbers(m: dict[str, Any]) -> dict[str, str]:
    """Block M1: writer agreement pooled over the eight writer switches (M1-a), and per
    switch the calibration slope and AUROC, the survival of D=1 minus D=0 at source-epoch binding
    (M1-b, with the switches where the estimate is >= 0), and T1."""
    out = _Macros()
    p3 = m.get("path3") or {}
    out["MOnePathThreeSlope"] = (
        f"{num(_get(p3, 'pooled_slope', 'est'))} (90\\% lower bound "
        f"{num(_get(p3, 'pooled_slope', 'lo90'))})"
        if p3
        else MISSING
    )
    out["MOnePathThreeAuroc"] = num(_get(p3, "pooled_auroc", "est"))
    n_items = _get(p3, "pooled_slope", "n_items")
    out["MOnePathThreeN"] = f"{int(n_items):,}" if n_items else MISSING
    labels = {tier_key(m, t): SWITCH_PROSE.get(t.tag, f"the switch to {t.label}") for t in TIERS}
    caveats = [labels.get(k) or _tex_text(k) for k in p3.get("tier_caveats", [])]
    out["MOneCaveats"] = _and(caveats) if p3 else MISSING
    keys = [tier_key(m, t) for t in TIERS]
    present = [k for k in keys if k]
    out["MOneStaged"] = str(len(present)) if present else MISSING
    out["MOneSwitches"] = str(len(p3.get("tiers", []))) if p3 else MISSING
    for tier, key in zip(TIERS, keys, strict=True):
        tag = tier.tag
        cal = _get(m, "calibration", key) or {}
        slope = cal.get("slope") or {}
        out[f"MOneSlope{tag}"] = (
            f"{num(slope.get('est'))} (90\\% {num(slope.get('lo90'))}--{num(slope.get('hi90'))})"
            if slope
            else MISSING
        )
        out[f"MOneAuroc{tag}"] = num(_get(cal, "auroc", "est"))
        out[f"MOneCalN{tag}"] = str(_get(cal, "slope", "n_items") or MISSING)
        out[f"MOneRetention{tag}"] = ci(_get(m, "retention", f"{key}:holdmem_D1B1_source"))
        mb = _get(m, "m1b_d_protects_switch", key) or {}
        out[f"MOneMB{tag}"] = ci(mb.get("d1_minus_d0_survival"), pp)
        t1 = _get(m, "t1", key) or {}
        out[f"MOneTOne{tag}"] = ci(t1.get("delta_90"), pp)
        out[f"MOneTOnePass{tag}"] = (
            MISSING if not t1 else ("within" if t1.get("equivalent") else "outside")
        )
    complete = all(keys)
    n = len(keys)
    out["MOneMBHolding"] = (
        _count(
            sum(_get(m, "m1b_d_protects_switch", k, "est_at_least_zero") is True for k in keys), n
        )
        if complete
        else MISSING
    )
    out["MOneTOneHolding"] = (
        _count(sum(_get(m, "t1", k, "equivalent") is True for k in keys), n)
        if complete
        else MISSING
    )
    return out


def m2_numbers(families: Mapping[str, Any]) -> dict[str, str]:
    """Block M2: the estimates behind checks (i)-(v) per family (`analysis.families`)."""
    out = _Macros()
    for fam in FAMILIES:
        f = families.get(fam.key) or {}
        pre = f"MTwo{fam.tag}"
        out[f"{pre}Worlds"] = _worlds(f.get("n_worlds"))
        for key, name in FAMILY_ESTIMATES:
            out[f"{pre}{name}"] = ci(_get(f, "estimates", key), pp)
    staged = [fam for fam in FAMILIES if families.get(fam.key)]
    out["MTwoStaged"] = str(len(staged)) if staged else MISSING
    return out


M3_CONTRASTS = (
    ("B_effect_update_pooled_source", "BUpdate"),
    ("D_effect_repair_source", "DRepair"),
    ("BxD_interaction_update_source", "BxD"),
    ("D_effect_missed_source", "DMissed"),
)


def m3_numbers(readers: Mapping[str, Any]) -> dict[str, str]:
    """Block M3: per reader, the four E1 contrasts under the main and the swapped
    reader, and post-shift ContractValid of HoldMem(1,1), the zero system and plain RAG."""
    out = _Macros()
    for fam in FAMILIES:
        sw = readers.get(fam.key) or {}
        pre = f"MThree{fam.tag}"
        out[f"{pre}Worlds"] = _worlds(len(sw.get("worlds", [])))
        for b, btag in BRANCH_TAGS[:2]:
            cons = _get(sw, b, "contrasts") or {}
            for key, name in M3_CONTRASTS:
                c = cons.get(key) or {}
                out[f"{pre}{name}Main{btag}"] = ci(c.get("main"), pp)
                out[f"{pre}{name}Swap{btag}"] = ci(c.get("swap"), pp)
            for s, stag in SYSTEM_TAGS:
                for arm, aname in (("main", "Main"), ("swap", "Swap")):
                    est = _get(sw, b, "systems", s, arm, "ContractValid_post")
                    out[f"{pre}Cv{stag}{aname}{btag}"] = ci(est)
    staged = [f for f in FAMILIES if readers.get(f.key)]
    out["MThreeStaged"] = str(len(staged)) if staged else MISSING
    return out


def reference_label(cell: str | None) -> str:
    """`holdmem_D0B1_cited` -> HoldMem(0,1), cited cluster (the white-box cell that predicts)."""
    m = re.fullmatch(r"holdmem_D(\d)B(\d)(?:_(\w+))?", cell or "")
    if not m:
        return MISSING
    gran = {"source": "source epoch", "cited": "cited cluster"}.get(m.group(3) or "")
    return f"\\holdmem{{}}({m.group(1)},{m.group(2)})" + (f", {gran}" if gran else "")


def native_label(res: dict[str, Any], system: str) -> tuple[int | None, int | None] | None:
    """Inferred (D, B) of a system (`analysis.e5`: D = 0 by export structure), None if not run."""
    lab = _get(res, "labels", system)
    return _inferred(lab) if lab and lab.get("B") is not None else None


def cost_tokens(entry: dict[str, Any] | None) -> float | None:
    t = (entry or {}).get("t_i") or {}
    if t.get("prompt_tokens") is None:
        return None
    return float(t["prompt_tokens"]) + float(t.get("completion_tokens", 0.0))


def nat_numbers(res: dict[str, Any] | None, cost: dict[str, Any]) -> dict[str, str]:
    """Block N: per native system the documented and the inferred (D, B) label, the
    six comparisons with the interval of the HoldMem cell with that label (inside or outside),
    noise floor, ingestion failures, T3 / T4 and the measured cost per event; and the counts over
    the systems (`analysis.e5.stage4_summary`), rendered only when all six systems (Mem0 and the
    five stage-4 systems) are staged."""
    res = res or {}
    out = _Macros()
    systems = res.get("systems") or {}
    s4 = res.get("stage4") or {}
    mem0_tokens = cost_tokens(cost.get("mem0"))
    for nat in NATIVES:
        s, pre = nat.key, f"Nat{nat.tag}"
        doc = DOCUMENTED_LABELS.get(s)
        inferred = native_label(res, s)
        lab = _get(res, "labels", s) or {}
        out[f"{pre}Documented"] = label_text(*doc) if doc else MISSING
        out[f"{pre}Label"] = label_text(*inferred) if inferred else MISSING
        out[f"{pre}LabelD"] = MISSING if not inferred else str(inferred[0])
        out[f"{pre}LabelB"] = (
            MISSING if not inferred else ("--" if inferred[1] is None else str(inferred[1]))
        )
        out[f"{pre}LabelAgrees"] = (
            _flag(doc == inferred, "yes", "no") if inferred and doc else MISSING
        )
        for dim in ("B", "D"):
            out[f"{pre}{dim}Rate"] = pct(_get(lab, dim, "rate"))
            out[f"{pre}{dim}N"] = str(_get(lab, dim, "n") or MISSING)
        per = systems.get(s) or {}
        out[f"{pre}Worlds"] = _worlds(len(per.get("worlds", [])))
        out[f"{pre}Reference"] = reference_label(per.get("predicted_by")) if per else MISSING
        n2 = _get(s4, "n2_per_system", s) or {}
        out[f"{pre}Inside"] = str(n2["inside"]) if n2 else MISSING
        out[f"{pre}Comparisons"] = str(n2["comparisons"]) if n2 else MISSING
        out[f"{pre}InsideOf"] = _count(n2["inside"], n2["comparisons"]) if n2 else MISSING
        for b, btag in BRANCH_TAGS:
            br = _get(per, "branches", b) or {}
            nf = _get(res, "noise_floor", "systems", s, b) or {}
            out[f"{pre}NoiseDiffer{btag}"] = pct(nf.get("share_differ"))
            for metric, mname in NATIVE_METRICS:
                out[f"{pre}{mname}{btag}"] = ci(_get(br, "native", metric))
                out[f"{pre}{mname}Pred{btag}"] = ci(_get(br, "white_box", metric))
                ok = _get(br, "consistent", metric)
                out[f"{pre}{mname}Consistent{btag}"] = (
                    MISSING if ok is None else ("inside" if ok else "outside")
                )
                out[f"{pre}{mname}Dev{btag}"] = pp(_get(br, "deviation", metric))
                out[f"{pre}Noise{mname}{btag}"] = pp(_get(nf, "diff", metric))
        out.update(disc_numbers(f"{pre}Ex", per, _get(s4, "discriminability", "per_system", s)))
        fail = _get(res, "failures", s) or {}
        out[f"{pre}FailRate"] = pct(fail.get("rate"), 2)
        out[f"{pre}Failed"] = f"{fail['failed']:,.0f}" if fail else MISSING
        out[f"{pre}Units"] = (
            f"{fail.get('ingested', 0) + fail.get('failed', 0):,.0f}" if fail else MISSING
        )
        mem = _get(res, "membership", s) or {}
        out[f"{pre}Static"] = ci(mem.get("static"))
        out[f"{pre}TThree"] = ci(mem.get("minus_oracle"), pp)
        out[f"{pre}TThreeResult"] = _pass(mem.get("T3_pass"))
        out[f"{pre}TFour"] = pct(mem.get("serve_ratio_to_full_reread"), 2)
        out[f"{pre}TFourResult"] = _pass(mem.get("T4_pass"))
        c = cost.get(s) or {}
        t = c.get("t_i") or {}
        tokens = cost_tokens(c)
        out[f"{pre}CostCalls"] = num(t.get("llm_calls"), 2)
        out[f"{pre}CostPrompt"] = MISSING if not t else f"{t['prompt_tokens'] / 1000:.1f}k"
        out[f"{pre}CostCompletion"] = MISSING if not t else f"{t['completion_tokens']:,.0f}"
        out[f"{pre}CostTokens"] = MISSING if tokens is None else f"{tokens / 1000:.1f}k"
        out[f"{pre}CostRatio"] = (
            num(tokens / mem0_tokens, 2) if tokens is not None and mem0_tokens else MISSING
        )
        out[f"{pre}CostWall"] = (
            MISSING if c.get("wall_s_per_unit") is None else f"{c['wall_s_per_unit']:.0f}"
        )
        out[f"{pre}CostUnits"] = str(c.get("attempted") or MISSING)
    staged = [n for n in NATIVES if n.key in systems]
    complete = len(staged) == len(NATIVES) and bool(s4)
    out["NatStaged"] = str(len(staged)) if staged else MISSING
    agree = s4.get("n1_label_agreement") or {}
    per_system = s4.get("n2_per_system") or {}
    out["NatNOneAgree"] = (
        _count(sum(v is True for v in agree.values()), len(agree))
        if complete and agree
        else MISSING
    )
    out["NatNThreeInside"] = (
        _count(
            sum(v["inside"] for v in per_system.values()),
            sum(v["comparisons"] for v in per_system.values()),
        )
        if complete and per_system
        else MISSING
    )
    out["NatNThreeShare"] = pct(s4.get("n3_share_inside")) if complete else MISSING
    total = _get(s4, "discriminability", "total") or {}
    for key, name in (
        ("comparisons", "Comparisons"),
        ("inside_predicted", "InsidePredicted"),
        ("inside_other", "InsideOther"),
        ("discriminating", "Discriminating"),
    ):
        out[f"NatEx{name}"] = str(total[key]) if complete and total.get("comparisons") else MISSING
    out["NatExDiscriminatingOf"] = (
        _count(total["discriminating"], total["comparisons"])
        if complete and total.get("comparisons")
        else MISSING
    )
    by_b = s4.get("n4_orphan_stability_by_B") or {}
    for b, name in (("0", "BZero"), ("1", "BOne")):
        vals = by_b.get(b) or []
        out[f"NatNFourStability{name}"] = (
            pct(sum(vals) / len(vals)) if complete and vals else MISSING
        )
    return out


REND_SOURCES = (("email", "Email"), ("messages", "Messages"), ("notes", "Notes"))


def rend_numbers(m: dict[str, Any]) -> dict[str, str]:
    """Block L: the estimates behind checks (i)-(v) on the LLM-rendered worlds and on the
    same-seed template worlds, the rendered-minus-template item-paired differences, the
    ContractValid differences per cell, and the rendering descriptives (kept-template share,
    retries, fidelity)."""
    out = _Macros()
    rend, temp = m.get("rendered") or {}, m.get("template_same_seeds") or {}
    out["RendWorlds"] = _worlds(len(m.get("worlds", [])))
    out["RendProbesIdentical"] = _flag(_get(m, "probe_identity", "identical"), "yes", "no")
    paired = {p["name"]: p for p in m.get("paired") or []}
    for key, name in FAMILY_ESTIMATES:
        p = paired.get(key) or {}
        out[f"Rend{name}"] = ci(_get(rend, "estimates", key), pp)
        out[f"RendTemplate{name}"] = ci(_get(temp, "estimates", key), pp)
        out[f"RendDiff{name}"] = ci(p.get("diff"), pp)
        out[f"RendDiffWorldMean{name}"] = ci(p.get("diff_world_mean"), pp)
    for b, btag in BRANCH_TAGS:
        for g, gtag in (("source", "Source"), ("cited", "Cited")):
            p = paired.get(f"{b}:D_effect_missed_{g}") or {}
            out[f"RendDMissed{gtag}{btag}"] = ci(p.get("rendered"), pp)
            out[f"RendDiffDMissed{gtag}{btag}"] = ci(p.get("diff"), pp)
    cells = {(c["branch"], c["system"]): c for c in m.get("cells") or [] if c["period"] == "post"}
    for b, btag in BRANCH_TAGS[:2]:
        for s, stag in (*SYSTEM_TAGS, ("oracle", "Oracle")):
            out[f"RendCvDiff{stag}{btag}"] = ci((cells.get((b, s)) or {}).get("diff"), pp)
    rd = m.get("rendering") or {}
    tot, fid = rd.get("total") or {}, rd.get("fidelity") or {}
    out["RendModel"] = _tex_text(", ".join(rd["models"])) if rd.get("models") else MISSING
    out["RendKeptTemplate"] = pct(tot.get("kept_template_share"))
    out["RendKeptCount"] = f"{tot['kept_template']:,}" if "kept_template" in tot else MISSING
    out["RendRetried"] = pct(tot.get("retried_share"))
    out["RendAttempted"] = f"{tot['attempted']:,}" if "attempted" in tot else MISSING
    out["RendSkippedBackground"] = (
        f"{tot['skipped_background']:,}" if "skipped_background" in tot else MISSING
    )
    out["RendRequests"] = f"{rd['requests']:,}" if "requests" in rd else MISSING
    out["RendFidelityMin"] = pct(fid.get("min_rate"))
    out["RendFidelityChecked"] = f"{fid['checked']:,}" if "checked" in fid else MISSING
    out["RendFidelityAllOne"] = _flag(fid.get("all_one"), "yes", "no") if fid else MISSING
    for src, tag in REND_SOURCES:
        per = _get(rd, "per_source", src) or {}
        out[f"RendAttempted{tag}"] = f"{per['attempted']:,}" if "attempted" in per else MISSING
        out[f"RendKept{tag}"] = pct(per.get("kept_template_share"))
        out[f"RendRetried{tag}"] = pct(per.get("retried_share"))
    return out


RANK_SUBSETS = (("holdmem", "HoldMem"), ("native", "Native"), ("native_baselines", "NativeBase"))


def rank_numbers(m: dict[str, Any]) -> dict[str, str]:
    """Rank correlation: Spearman correlation between the ranking of the systems
    by Update after S_src and their rankings by Update after S_def and by orphan Stability; and
    the S_src-S_def correlation within the HoldMem cells, the native systems and the native systems
    with no-store BM25 and plain RAG (`RankExRhoSrcDef<subset>`)."""
    out = _Macros()

    def rho(x: float | None) -> str:
        s = num(x, 2)
        return "$-$" + s[1:] if s.startswith("-") else s  # a typeset minus, not a hyphen

    def put(c: dict[str, Any] | None, name: str) -> None:
        ok = c and c["rho"] == c["rho"]  # fewer than three systems give no correlation (NaN)
        est = {"est": c["rho"], "lo": c["lo"], "hi": c["hi"]} if c and ok else None
        out[f"RankExRho{name}"] = ci(est, rho)
        out[f"RankEx{name}Systems"] = str(c["n_systems"]) if c else MISSING
        out[f"RankEx{name}Worlds"] = str(c["n_worlds"]) if c else MISSING

    for key, name in (
        ("update_src_vs_update_def", "SrcDef"),
        ("update_src_vs_stability_orphan", "SrcOrphan"),
    ):
        put(_get(m, "correlations", key), name)
    for subset, tag in RANK_SUBSETS:
        put(_get(m, "subsets", subset, "update_src_vs_update_def"), f"SrcDef{tag}")
    out["RankExSystems"] = str(len(m["systems"])) if m.get("systems") else MISSING
    return out


MULT_STAGES = (("orig", "Orig"), ("rep", "Rep"))  # stage 2, block R
MULT_COUNTS = (
    ("n_contrasts", "Contrasts"),
    ("n_interval_excludes_zero", "Interval"),
    ("n_survive_holm", "Holm"),
    ("n_survive_bh", "Bh"),
)


def mult_numbers(view: Mapping[str, Any]) -> dict[str, str]:
    """Multiplicity corrections (`analysis.multiplicity`): for
    stage 2 (`Orig`) and block R (`Rep`), the number of E1 contrasts expected to be positive (one
    family per stage), how many have a 95% interval above 0, and how many
    survive Holm-Bonferroni (familywise alpha = 0.05) and Benjamini-Hochberg (q = 0.05)."""
    out = _Macros()
    for tag, name in MULT_STAGES:
        summary = view.get(tag) or {}
        for key, field in MULT_COUNTS:
            out[f"MultEx{name}{field}"] = str(summary[key]) if key in summary else MISSING
    return out


def stage4_numbers(inp: dict[str, Any]) -> dict[str, str]:
    """All stage-4 macros from `paper_data.load_stage4`; a name produced twice is an error."""
    parts = (
        rep_numbers(replication_view(inp["orig"], inp["rep"])),
        rep_pool_numbers(pooled_view(inp["pool"])),
        rep_e3_numbers(inp["orig"]["e3"], inp["rep"]["e3"]),
        synth_numbers(inp["c"] or {}),
        m1_numbers(inp["m1"] or {}),
        m2_numbers(inp["m2"]),
        m3_numbers(inp["m3"]),
        nat_numbers(inp["n"], inp["ncost"]),
        rend_numbers(inp["rendered"] or {}),
        rank_numbers(inp["rank"] or {}),
        mult_numbers(inp["mult"]),
    )
    merged: dict[str, str] = {}
    for part in parts:
        if dup := merged.keys() & part.keys():
            raise ValueError(f"macro defined by two stage-4 blocks: {sorted(dup)[:3]}")
        merged.update(part)
    return merged


def build_macros(root: Path) -> dict[str, str]:
    src_file = root / "results" / "sources.json"
    sources = json.loads(src_file.read_text()) if src_file.exists() else {}
    macros: dict[str, str] = world_stats(list(range(10))) | world_size(EVAL_SEEDS)
    groups: list[tuple[str, Callable[[dict[str, Any]], dict[str, str]], str]] = [
        ("e1", e1_numbers, "e1_metrics.json"),
        ("e1nofb", nofb_numbers, "e1_metrics.json"),
        ("e2", e2_numbers, "e2_metrics.json"),
        ("e3", e3_numbers, "e3_metrics.json"),
        ("e4", e4_numbers, "e4_metrics.json"),
        ("e4x", e4x_numbers, "e4x_metrics.json"),
        ("e5", e5_numbers, "e5_natives.json"),
        ("e5cost", e5cost_numbers, "e5_cost.json"),
        ("e5wb", e5wb_numbers("Dev"), "e5_whitebox_validation.json"),
        ("e5wbshort", e5wb_numbers("Short"), "e5_whitebox_validation.json"),
        ("e6", e6_numbers, "e6_t2.json"),
        ("e7", e7_numbers, "e7_metrics.json"),
        ("gate", gate_numbers, "gate_a.json"),
    ]
    ts = root / sources.get("testset", "__missing__") / "analysis" / "testset.json"
    if ts.exists():
        t = json.loads(ts.read_text())
        macros["TestWorlds"] = str(len(t["seeds"]))
        macros["TestFidelity"] = f"{t['fidelity_located']:,}/{t['fidelity_checked']:,}"
    for group, fn, fname in groups:
        path = root / sources.get(group, "__missing__") / "analysis" / fname
        data = json.loads(path.read_text()) if path.exists() else {}
        macros.update(fn(data))
    inp = load_stage4(root, sources)
    macros.update(stage4_numbers(inp))
    # The further analyses; names carry `Ex` and must not collide with an existing
    # macro: the E1-family analyses, then the writer-switch predictors, T1 / T3 by item class and
    # the native null baselines
    from memshift.analysis import (
        controls_paper,  # lazy, as above
        ex_e1_paper,  # lazy: it builds on this module's formatters
        ex_numbers,
        ex_review_paper,
        ex_round2,
        ex_round3,
        ex_round3_review,
        ex_round4_detection,
        key_audit,
        l_tokens,
    )

    for ex in (
        ex_e1_paper.numbers(ex_e1_paper.load(root, sources)),
        ex_numbers.build(root, sources, inp),
        ex_review_paper.numbers(ex_review_paper.load(root, sources)),
        controls_paper.numbers(controls_paper.load(root, sources)),  # control cells
        key_audit.numbers(key_audit.load(root, sources)),  # offline key audit (no model)
        l_tokens.numbers(l_tokens.load(root, sources)),  # block L: token shares (template counts)
        ex_round2.numbers(ex_round2.load(root, sources)),  # drift, per definition, D_m
        ex_round3.numbers(ex_round3.load(root, sources)),  # store and reader, view, natural
        # the checks of what the design fixes, and further splits
        ex_round3_review.numbers(ex_round3_review.load(root, sources)),
        # detection from the latest change, the key with Claude writers and the oracle key
        ex_round4_detection.numbers(ex_round4_detection.load(root, sources)),
    ):
        if dup := macros.keys() & ex.keys():
            raise ValueError(f"`Ex` macro collides with an existing one: {sorted(dup)[:3]}")
        macros.update(ex)
    # Round-3 results (`round3_paper`)
    from memshift.analysis import round3_paper  # lazy, as above

    reg = round3_paper.numbers(round3_paper.load(root, sources))
    if dup := macros.keys() & reg.keys():
        raise ValueError(f"round-3 macro collides with an existing one: {sorted(dup)[:3]}")
    macros.update(reg)
    for text in (
        "ResultsAbstract",
        "ResultsIntro",
        "ResultsConclusion",
        "GranularityAppendix",
        "GateAParagraph",
        "SyncParagraph",
        "NativeParagraph",
        "LeaveParagraph",
        "ReadersParagraph",
        "ReadersThetaR",
    ):
        macros.setdefault(text, MISSING)
    return with_estimates(macros)


def main() -> None:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    macros = build_macros(root)
    print("% generated by memshift.analysis.paper_numbers; do not edit")
    for k in sorted(macros):
        print(_macro(k, macros[k]))


if __name__ == "__main__":
    main()

"""Macros for the further analyses of E2, E3 and E5 (the `explore_*` fields of their stages).

Writer-switch predictors and baselines (E2, block M1), T1 sensitivity (E2), T3 by item class (E3,
block R) and the native null baselines (E5, block N). Every macro name carries `Ex`; a value a stage
has not produced renders as MISSING, and a count over a whole set of systems only once the set is
complete.

  <ETwo|MOne>ExWriter<set><predictor><scope><stat>
      predictor slope / AUROC / AUROC minus agreement. set: "" (all items) or NoDef (items outside
      the definition class); predictor: Agree, NewAcc, OldAcc, Cond, NonDef; scope: Pooled or a
      tier tag; stat: Slope (90% interval), Auroc (95%), AurocDiff (95%), each with world-bootstrap
      intervals and a `Type` variant with type-level ones
  <ETwo|MOne>ExLoto..., ExFail...
      leave-one-type-out and failure counts
  <ETwo|MOne>ExSurvival|Recovery|Net|Gains|Losses<cell><tier>
      survival, recovery and net change per cell and tier
  <ETwo|MOne>ExTOne{Ret|NetEq|NetNi}Eps<e><cell><tier>, ExTOneClass...
      T1 outcome (within or outside) at eps = e pp, and T1 by item class
  EThreeExTThree... / RepExTThree...
      T3 margin per item class, by horizon and system
  <EFiveExMem|Nat<tag>Ex>Null...
      native null baselines, counts, coverage, B identification

A slope beyond +/-20 (the logistic fit separates; `e2.slope_on_logit` runs into its iteration
limit) prints as `>20` or `<-20`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import (
    BRANCH_TAGS,
    HORIZON_TAGS,
    MISSING,
    NATIVE_METRICS,
    NATIVES,
    SYSTEM_TAGS,
    TIERS,
    camel,
    stage_json,
    tier_key,
)
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import (
    CELL_NAMES,
    CLASS_NAMES,
    _count,
    _flag,
    _Macros,
    _pass,
    _tex_text,
    ci,
    pct,
    pp,
)

SEPARATED = 20.0  # |slope| beyond this: the logistic fit separates (prints as a bound)
PRED_TAGS = (
    ("agree", "Agree"),
    ("new_acc", "NewAcc"),
    ("old_acc", "OldAcc"),
    ("cond", "Cond"),
    ("nondef", "NonDef"),
)
SET_TAGS = (("all", ""), ("nodef", "NoDef"))
EPS_PP = (1, 2, 3, 5)
STAGE2_SCOPES = (
    ("pooled", "Pooled"),
    ("same", "Same"),
    ("upgrade", "Up"),
    ("downgrade", "Down"),
)
BASELINE_TAGS = (("zero", "Zero"), ("rag", "Rag"), ("oracle", "Oracle"))
REFERENCE_LABELS = {
    "holdmem_D0B0": "(0,0)",
    "holdmem_D1B0": "(1,0)",
    "holdmem_D0B1_source": "(0,1)",
    "holdmem_D1B1_source": "(1,1)",
    "holdmem_D0B1_cited": "(0,1)c",
    "holdmem_D1B1_cited": "(1,1)c",
    "zero": "zero",
    "rag": "RAG",
    "oracle": "oracle",
}


def sn(x: float | None, digits: int = 2, signed: bool = False) -> str:
    """A number with a typeset minus; a slope beyond `SEPARATED` prints as a bound."""
    if x is None or x != x:
        return MISSING
    if abs(x) > SEPARATED:
        return "$>20$" if x > 0 else "$<-20$"
    v = round(x, digits)
    body = f"{abs(v):.{digits}f}"
    if v < 0:
        return "$-$" + body
    return "$+$" + body if signed and v > 0 else body


def iv(e: dict[str, Any] | None, lo: str, hi: str, digits: int = 2, signed: bool = False) -> str:
    """`est [lo, hi]` from the keys `lo` and `hi` of a stats entry."""
    if not e or e.get("est") is None:
        return MISSING
    return (
        f"{sn(e['est'], digits, signed)} "
        f"[{sn(e.get(lo), digits, signed)}, {sn(e.get(hi), digits, signed)}]"
    )


def stat_iv(
    stat: dict[str, Any] | None, kind: str, level: str, digits: int = 2, signed: bool = False
) -> str:
    """Estimate of a `{est, world: {...}, type: {...}}` stat with the `kind` interval at `level`
    (90 or 95)."""
    if not stat or not stat.get(kind):
        return MISSING
    e = {"est": stat["est"], **stat[kind]}
    return iv(e, f"lo{level}", f"hi{level}", digits, signed)


# --------------------------------------------------------------------------------------- E2 / M1
def writer_numbers(
    m: dict[str, Any], prefix: str, scopes: Sequence[tuple[str | None, str]]
) -> dict[str, str]:
    """Predictors, leave-one-type-out and failure counts of one E2 analysis. `scopes` lists
    (scope key, tag); a key of None (a switch that is not staged) renders as MISSING."""
    out = _Macros()
    w = m.get("explore_writer_predictors") or {}
    out[f"{prefix}ExWriterDraws"] = str(w.get("draws", MISSING))
    for set_key, set_tag in SET_TAGS:
        for key, tag in scopes:
            sc = _get(w, "sets", set_key, key) if key else None
            sc = sc or {}
            for count_key, name in (
                ("n_items", "Items"),
                ("n_failures", "Failures"),
                ("n_worlds", "Worlds"),
                ("n_types", "Types"),
            ):
                v = sc.get(count_key)
                out[f"{prefix}ExWriter{set_tag}{name}{tag}"] = MISSING if v is None else f"{v:,}"
            for pred, ptag in PRED_TAGS:
                if set_key == "nodef" and pred == "nondef":
                    continue
                e = _get(sc, "predictors", pred) or {}
                base = f"{prefix}ExWriter{set_tag}{ptag}{tag}"
                for kind, ktag in (("world", ""), ("type", "Type")):
                    out[f"{base}Slope{ktag}"] = stat_iv(e.get("slope"), kind, "90")
                    out[f"{base}Auroc{ktag}"] = stat_iv(e.get("auroc"), kind, "95", 3)
                    if pred != "agree":
                        out[f"{base}AurocDiff{ktag}"] = stat_iv(
                            e.get("auroc_minus_agree"), kind, "95", 3, signed=True
                        )
    out.update(loto_numbers(m, prefix))
    out.update(failure_numbers(m, prefix, scopes))
    return out


def loto_numbers(m: dict[str, Any], prefix: str) -> dict[str, str]:
    out: dict[str, str] = {}
    loto = _get(m, "explore_writer_loto", "pooled") or {}
    rng = _get(loto, "range", "agree") or {}
    for stat, digits in (("slope", 2), ("auroc", 3)):
        for which in ("min", "max"):
            v = rng.get(f"{stat}_{which}")
            tag = f"{prefix}ExLoto{camel(stat)}{camel(which)}"
            out[tag] = MISSING if v is None else sn(v, digits)
            t = rng.get(f"{stat}_{which}_type")
            out[f"{tag}Type"] = MISSING if t is None else _tex_text(t)
    for t, e in (loto.get("types") or {}).items():
        a = _get(e, "predictors", "agree") or {}
        out[f"{prefix}ExLotoDrop{camel(t)}Slope"] = sn(a.get("slope"))
        out[f"{prefix}ExLotoDrop{camel(t)}Auroc"] = sn(a.get("auroc"), 3)
    return out


def failure_numbers(
    m: dict[str, Any], prefix: str, scopes: Sequence[tuple[str | None, str]]
) -> dict[str, str]:
    out: dict[str, str] = {}
    f = m.get("explore_failures_by_type") or {}
    s = f.get("summary") or {}
    for key, name in (
        ("n_items", "Items"),
        ("n_failures", "Total"),
        ("definition_failures", "Definition"),
        ("top_two_failures", "TopTwo"),
        ("type_tier_cells", "Cells"),
        ("type_tier_cells_with_failures", "CellsWith"),
        ("types", "Types"),
    ):
        out[f"{prefix}ExFail{name}"] = f"{s[key]:,}" if key in s else MISSING
    out[f"{prefix}ExFailTopTwoTypes"] = (
        ", ".join(_tex_text(t) for t in s["top_two_types"]) if s.get("top_two_types") else MISSING
    )
    keyed: list[tuple[str | None, str]] = [("pooled", "Pooled"), *scopes]
    for scope, tag in keyed:
        if scope is None:
            continue
        for t, e in (f.get(scope) or {}).items():
            out[f"{prefix}ExFail{camel(t)}{tag}"] = _count(e["failures"], e["n"])
    return out


def t1_numbers(
    m: dict[str, Any], prefix: str, tiers: Sequence[tuple[str | None, str]]
) -> dict[str, str]:
    """Survival, recovery, net change, the eps grid and T1 by item class, per cell and tier."""
    out = _Macros()
    for cell, cname in CELL_NAMES:
        for key, tag in tiers:
            at = f"{key}:{cell}" if key else None
            sr = _get(m, "explore_survival_recovery", at) or {}
            out[f"{prefix}ExSurvival{cname}{tag}"] = ci(sr.get("survival"))
            out[f"{prefix}ExRecovery{cname}{tag}"] = ci(sr.get("recovery"))
            out[f"{prefix}ExNet{cname}{tag}"] = ci(sr.get("net"), pp)
            out[f"{prefix}ExValidBefore{cname}{tag}"] = pct(_get(sr, "valid_before", "est"))
            out[f"{prefix}ExValidAfter{cname}{tag}"] = pct(_get(sr, "valid_after", "est"))
            for field, name in (("gains", "Gains"), ("losses", "Losses")):
                out[f"{prefix}Ex{name}{cname}{tag}"] = str(sr[field]) if field in sr else MISSING
            out[f"{prefix}ExTrivial{cname}{tag}"] = (
                _flag(sr.get("trivial"), "yes", "no") if sr else MISSING
            )
            grid = _get(m, "explore_t1_eps", at, "by_eps_pp") or {}
            for e in EPS_PP:
                v = grid.get(str(e)) or {}
                for field, name in (
                    ("retention_noninferior", "Ret"),
                    ("net_equivalent", "NetEq"),
                    ("net_noninferior", "NetNi"),
                ):
                    ok = v.get(field)
                    out[f"{prefix}ExTOne{name}Eps{camel(str(e))}{cname}{tag}"] = (
                        MISSING if ok is None else ("within" if ok else "outside")
                    )
            if cell in ("holdmem_D1B1_source", "holdmem_D0B1_source"):
                for cls, clsname in CLASS_NAMES:
                    c = _get(m, "explore_t1_by_class", at, cls) or {}
                    base = f"{prefix}ExTOneClass{clsname}{cname}{tag}"
                    out[base] = ci(c.get("retention_delta_90"), pp)
                    out[f"{base}Net"] = ci(c.get("net_90"), pp)
                    out[f"{base}Share"] = pct(c.get("share_of_valid_before"))
                    out[f"{base}Tolerated"] = pct(c.get("tolerated_loss_at_margin"), 0)
                    out[f"{base}Result"] = (
                        MISSING if not c else ("within" if c.get("equivalent") else "outside")
                    )
    for cell, cname in CELL_NAMES:  # tiers within the margin, at each eps, once all are staged
        for e in EPS_PP:
            flags = [
                _get(
                    m, "explore_t1_eps", f"{k}:{cell}", "by_eps_pp", str(e), "retention_noninferior"
                )
                for k, _ in tiers
                if k
            ]
            staged = len(flags) == len(tiers) and all(f is not None for f in flags)
            out[f"{prefix}ExTOneRetEps{camel(str(e))}Within{cname}"] = (
                _count(sum(f is True for f in flags), len(flags)) if staged else MISSING
            )
    return out


def e2_numbers(m: dict[str, Any]) -> dict[str, str]:
    """`ETwoEx...`: the `explore_*` E2 fields of the stage-2 analysis."""
    scopes: list[tuple[str | None, str]] = list(STAGE2_SCOPES)
    tiers = [(k, t) for k, t in STAGE2_SCOPES if k != "pooled"]
    out = _Macros()
    out.update(writer_numbers(m, "ETwo", scopes))
    out.update(t1_numbers(m, "ETwo", tiers))
    return out


def m1_numbers(m: dict[str, Any]) -> dict[str, str]:
    """`MOneEx...`: the same for block M1's switches (the stage-2 tiers and the cross-family ones,
    found by `paper_data.tier_key`); a switch that is not staged renders as MISSING."""
    scopes: list[tuple[str | None, str]] = [("pooled", "Pooled")] + [
        (tier_key(m, t), t.tag) for t in TIERS
    ]
    out = _Macros()
    out.update(writer_numbers(m, "MOne", scopes))
    out.update(t1_numbers(m, "MOne", scopes[1:]))
    return out


# ------------------------------------------------------------------------------------------- E3
def t3_numbers(m: dict[str, Any] | None, prefix: str) -> dict[str, str]:
    """T3 margin against the oracle per item class, horizon and system (`<prefix>ExTThree...`)."""
    out = _Macros()
    for h, htag in HORIZON_TAGS:
        per = _get(m or {}, "horizons", h) or {}
        for s, stag in SYSTEM_TAGS:
            for cls, cname in CLASS_NAMES:
                e = _get(per, s, "explore_t3_by_class", cls) or {}
                base = f"{prefix}ExTThree{stag}{cname}{htag}"
                out[base] = ci(e.get("minus_oracle"), pp)
                out[f"{base}Result"] = _pass(e.get("T3_pass"))
                out[f"{base}Contribution"] = pp(e.get("contribution"))
        for s, stag in SYSTEM_TAGS:  # the oracle context has no serving cost
            slack = _get(per, s, "explore_t4_slack")
            out[f"{prefix}ExTFourSlack{stag}{htag}"] = MISSING if slack is None else f"{slack:.0f}"
        for cls, cname in CLASS_NAMES:
            out[f"{prefix}ExTThreeOracleCv{cname}{htag}"] = ci(
                _get(per, "oracle", "explore_t3_by_class", cls, "contract_valid")
            )
    for s, stag in SYSTEM_TAGS:
        for cls, cname in CLASS_NAMES:
            flags = [
                _get(m or {}, "horizons", h, s, "explore_t3_by_class", cls, "T3_pass")
                for h, _ in HORIZON_TAGS
            ]
            out[f"{prefix}ExTThreePass{stag}{cname}"] = (
                _count(sum(f is True for f in flags), len(flags))
                if all(f is not None for f in flags)
                else MISSING
            )
    return out


# ------------------------------------------------------------------------------------------- E5
def reference_names(names: list[str]) -> str:
    return ", ".join(REFERENCE_LABELS.get(n, _tex_text(n)) for n in names) or "none"


def null_counts(out: _Macros, base: str, counts: dict[str, Any] | None) -> None:
    """`<base>Inside/Disc` over all comparisons, the label-circular and the out-of-sample ones."""
    for part, ptag in (("all", ""), ("label_circular", "Circular"), ("out_of_sample", "Oos")):
        c = (counts or {}).get(part) or {}
        n = c.get("comparisons")
        for key, name in (("inside_predicted", "Inside"), ("discriminating", "Disc")):
            out[f"{base}{name}{ptag}Of"] = (
                MISSING if n is None else ("none run" if n == 0 else _count(c[key], n))
            )


def null_numbers(entry: dict[str, Any] | None, prefix: str) -> dict[str, str]:
    """`<prefix>Null...` of one native system from `explore_native_null`."""
    out = _Macros()
    e = entry or {}
    b = e.get("b_identified") or {}
    out[f"{prefix}NullBRate"] = (
        ci({"est": b["rate"], "lo": b["lo"], "hi": b["hi"]}) if b else MISSING
    )
    out[f"{prefix}NullBStraddles"] = _flag(b.get("straddles"), "yes", "no") if b else MISSING
    out[f"{prefix}NullBIdentified"] = _flag(b.get("identified"), "yes", "no") if b else MISSING
    out[f"{prefix}NullPredicted"] = REFERENCE_LABELS.get(e.get("predicted_by") or "", MISSING)
    null_counts(out, f"{prefix}Null", e.get("counts"))
    for name, tag in BASELINE_TAGS:
        bc = (e.get("baseline_counts") or {}).get(name) or {}
        null_counts(out, f"{prefix}Null{tag}", bc.get("baseline"))
        null_counts(out, f"{prefix}Null{tag}Native", bc.get("native_same_comparisons"))
    cov = e.get("chance_coverage") or {}
    out[f"{prefix}NullChanceLabels"] = pct(cov.get("labels"))
    out[f"{prefix}NullChanceReferences"] = pct(cov.get("references"))
    out[f"{prefix}NullObserved"] = pct(cov.get("observed"))
    for cell, share in (cov.get("label_cells") or {}).items():
        out[f"{prefix}NullChance{camel(cell.replace('holdmem_', ''))}"] = pct(share)
    for br, btag in BRANCH_TAGS:
        for metric, mname in NATIVE_METRICS:
            key = f"{br}:{metric}"
            c = _get(e, "comparisons", key) or {}
            out[f"{prefix}NullContained{mname}{btag}"] = (
                reference_names(c["contained_by"]) if c else MISSING
            )
            out[f"{prefix}NullContainedCount{mname}{btag}"] = (
                _count(c["n_contained"], c["n_references"]) if c else MISSING
            )
            paired = _get(e, "paired_difference", key) or {}
            cell = e.get("predicted_by")
            p = paired.get(cell) if cell else None
            out[f"{prefix}NullPaired{mname}{btag}"] = ci(p, pp)
            out[f"{prefix}NullPairedInflated{mname}{btag}"] = (
                ci({"est": p["est"], "lo": p["lo_inflated"], "hi": p["hi_inflated"]}, pp)
                if p and "lo_inflated" in p
                else MISSING
            )
            out[f"{prefix}NullIngestSd{mname}{btag}"] = (
                f"{100 * p['ingestion_sd']:.1f}\\,pp" if p and "ingestion_sd" in p else MISSING
            )
            for name, tag in BASELINE_TAGS[:2]:
                out[f"{prefix}NullPaired{tag}{mname}{btag}"] = ci(paired.get(name), pp)
    return out


def native_null_numbers(res: dict[str, Any] | None) -> dict[str, str]:
    """`Nat<tag>ExNull...` per native system of block N, and the totals once all six are staged."""
    out = _Macros()
    ex = (res or {}).get("explore_native_null") or {}
    entries = ex.get("systems") or {}
    for nat in NATIVES:
        out.update(null_numbers(entries.get(nat.key), f"Nat{nat.tag}Ex"))
    complete = all(n.key in entries for n in NATIVES) and bool(ex)
    total = ex.get("total") or {}
    null_counts(out, "NatExNull", total.get("counts") if complete else None)
    for name, tag in BASELINE_TAGS:
        bc = (total.get("baseline_counts") or {}).get(name) or {}
        null_counts(out, f"NatExNull{tag}", bc.get("baseline") if complete else None)
        null_counts(
            out, f"NatExNull{tag}Native", bc.get("native_same_comparisons") if complete else None
        )
    for key, name in (("labels", "Labels"), ("references", "References"), ("observed", "Observed")):
        out[f"NatExNullChance{name}"] = (
            pct(total.get(f"mean_{key}_coverage")) if complete else MISSING
        )
    out["NatExNullBIdentified"] = (
        _count(total["b_identified"], total["systems"]) if complete else MISSING
    )
    return out


def mem0_null_numbers(res: dict[str, Any] | None) -> dict[str, str]:
    """`EFiveExMemNull...`: the same for Mem0 from the stage-3 analysis (Mem0 only)."""
    entry = _get(res or {}, "explore_native_null", "systems", "mem0")
    return null_numbers(entry, "EFiveExMem")


# ------------------------------------------------------------------------------------------ build
def build(root: Path, sources: dict[str, str], inp: dict[str, Any]) -> dict[str, str]:
    """All `Ex` macros of this module. `inp` is `paper_data.load_stage4`."""
    e2 = stage_json(root, sources, "e2", "e2_metrics.json") or {}
    e3 = stage_json(root, sources, "e3", "e3_metrics.json")
    parts: list[Callable[[], dict[str, str]]] = [
        lambda: e2_numbers(e2),
        lambda: m1_numbers(inp["m1"] or {}),
        lambda: t3_numbers(e3, "EThree"),
        lambda: t3_numbers(inp["rep"]["e3"], "Rep"),
        lambda: mem0_null_numbers(inp["n3"]),
        lambda: native_null_numbers(inp["n"]),
    ]
    merged: dict[str, str] = {}
    for part in parts:
        macros = part()
        if dup := merged.keys() & macros.keys():
            raise ValueError(f"`Ex` macro defined twice: {sorted(dup)[:3]}")
        merged.update(macros)
    return merged

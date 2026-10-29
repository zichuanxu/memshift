"""Paper numbers, tables and the figure of the further E1-family analyses: the class-stratified
contrasts, the Eq. (1) decomposition, the out-of-sample test, invalidation precision,
per-definition spread, the post-shift JSON arm, Stability, the short worlds, replication
accounting, cost and the noise floor. Every macro name starts with `Ex` and every source is a
staged analysis named in results/sources.json:

  e1 / rep_e1 / c / m2_gemma / m2_llama / m2_gptoss / dev_e1 / dev_orphan_e1 / short_e1
                     analysis/e1_metrics.json, with the `explore_*` blocks of
                     `analysis.ex_e1_classes` (the development worlds are the union of dev_e1 and
                     dev_orphan_e1)
  ex                                               analysis/ex_e1.json (`analysis.ex_e1_cross`)

A stage that is not staged renders as the MISSING marker; a quantity that is undefined by
construction (the accuracy of the missed items when a key misses none) renders as a dash.

tab_classsplit   appendix: B, D and B x D on Update by shift and item class, with exact
                 recomputation and D_m (block C) as the D=1 arm in the main study and exact
                 recomputation in R
fig_classsplit   forest plot of the B effect at each D and B x D, by class and branch, main and R
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, camel, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, ci, pct, pp
from memshift.analysis.paper_tables import _ci, _num, _row

UNDEFINED = "---"
# macro tag, inputs key, arm of the D=1 cell, granularities the stage has
DATASETS = (
    ("Main", "main", "D1"),
    ("Rep", "rep", "D1"),
    ("Dm", "c", "D1s"),  # block C: D = 1 with writer-authored programs
    ("Gem", "gem", "D1"),
    ("Llama", "llama", "D1"),
    ("Gpt", "gpt", "D1"),
    ("Dev", "dev", "D1"),
    ("Short", "short", "D1"),
)
FULL_GRANULARITY = ("Main", "Rep", "Dm", "Gem", "Llama", "Gpt", "Dev")  # with cited-key cells
GRAN_TAGS = (("source", ""), ("cited", "Cited"))
CLASS_TAGS = (
    ("definition", "Definition"),
    ("other", "Other"),
    ("rule", "Rule"),
    ("norule", "Free"),
    ("all", "All"),
)
MAIN_CLASSES = CLASS_TAGS[:2] + CLASS_TAGS[4:]  # definition, other, all
CONTRAST_TAGS = (
    ("B_effect_update_D0", "BDzero"),
    ("B_effect_update_D1", "BDone"),
    ("B_effect_update_pooled", "BPooled"),
    ("D_effect_update_B0", "DBzero"),
    ("D_effect_update_B1", "DBone"),
    ("BxD_interaction_update", "BxD"),
    ("D_effect_repair", "DRepair"),
    ("D_effect_repair_changed", "DRepairChanged"),
)
CELL_TAGS = (
    ("D0B0", "DzeroBzero"),
    ("D1B0", "DoneBzero"),
    ("D0B1", "DzeroBone"),
    ("D1B1", "DoneBone"),
)
TERM_TAGS = (("U", "Update"), ("m", "Missed"), ("r", "Repair"), ("a", "Stale"))
INTERACTION_TAGS = (
    ("observed", "Observed"),
    ("paper_formula", "Paper"),
    ("corrected_formula", "Corrected"),
    ("gap_paper", "GapPaper"),
    ("gap_corrected", "GapCorrected"),
    ("D_effect_B0", "DBzero"),
    ("D_effect_B1", "DBone"),
    ("m1", "MOne"),
    ("D_effect_missed", "DMissed"),
    ("D_effect_a_B1", "DStaleBone"),
)
PCT_INTERACTION = {"m1"}  # shares; every other interaction quantity is a difference (pp)
OOS_CELL_TAGS = (
    ("D0B0", "DzeroBzero"),
    ("D1B0", "DoneBzero"),
    ("D0B1_source", "DzeroBoneSource"),
    ("D1B1_source", "DoneBoneSource"),
    ("D0B1_cited", "DzeroBoneCited"),
    ("D1B1_cited", "DoneBoneCited"),
)
OOS_PAIRS = (("dev_to_main", "DevMain"), ("dev_to_rep", "DevRep"), ("main_to_rep", "MainRep"))
OOS_PREDICTORS = (("cell", "Cell"), ("structural", "Structural"), ("literal", "Literal"))
OOS_CLASSES = (
    ("definition", "Definition"),
    ("other", "Other"),
    ("all", "All"),
    ("all_fit_mix", "AllFitMix"),
)
HORIZONS = (("1", "One"), ("3", "Three"), ("12", "Twelve"))
COST_SYSTEMS = (("zero", "Zero"), ("rag", "Rag"), ("holdmem_D1B1_source", "Hold"))
STABILITY_SYSTEMS = (
    ("holdmem_D0B0", "DzeroBzero"),
    ("holdmem_D1B0", "DoneBzero"),
    ("holdmem_D0B1_source", "DzeroBoneSource"),
    ("holdmem_D1B1_source", "DoneBoneSource"),
    ("holdmem_D0B1_cited", "DzeroBoneCited"),
    ("holdmem_D1B1_cited", "DoneBoneCited"),
    ("zero", "Zero"),
    ("rag", "Rag"),
    ("oracle", "Oracle"),
)
PRECISION = (
    ("false_invalidation", "False"),
    ("missed_invalidation", "Missed"),
    ("precision", "Precision"),
    ("base_rate_of_change", "BaseRate"),
    ("invalidated_share", "Invalidated"),
)


# ------------------------------------------------------------------------------------ loading
def _merge(a: Any, b: Any) -> Any:
    """Deep merge of two analyses that hold different branches (None when neither is staged)."""
    if a is None or b is None:
        return a if b is None else b
    if isinstance(a, dict) and isinstance(b, dict):
        return {k: _merge(a[k], b[k]) if k in a and k in b else a.get(k, b.get(k)) for k in a | b}
    return a


def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any]:
    """The staged inputs (None where a stage is not there)."""
    src = dict(sources)
    return {
        "main": stage_json(root, src, "e1", "e1_metrics.json"),
        "rep": stage_json(root, src, "rep_e1", "e1_metrics.json"),
        "c": stage_json(root, src, "c", "e1_metrics.json"),
        "gem": stage_json(root, src, "m2_gemma", "e1_metrics.json"),
        "llama": stage_json(root, src, "m2_llama", "e1_metrics.json"),
        "gpt": stage_json(root, src, "m2_gptoss", "e1_metrics.json"),
        # the development worlds: source and definition branches, and the orphan branch's own stage
        "dev": _merge(
            stage_json(root, src, "dev_e1", "e1_metrics.json"),
            stage_json(root, src, "dev_orphan_e1", "e1_metrics.json"),
        ),
        "short": stage_json(root, src, "short_e1", "e1_metrics.json"),
        "cross": stage_json(root, src, "ex", "ex_e1.json"),
        "names": {
            "main": Path(src.get("e1", "")).name,
            "rep": Path(src.get("rep_e1", "")).name,
        },
    }


# ----------------------------------------------------------------------------------- formats
def fmt(e: dict[str, Any] | None, f: Callable[[float | None], str] = pct) -> str:
    """Estimate with interval; MISSING when not staged, a dash when undefined by construction."""
    if not e:
        return MISSING
    if e.get("est") is None or e["est"] != e["est"]:
        return UNDEFINED
    return ci(e, f)


def point(x: float | None, f: Callable[[float | None], str] = pct) -> str:
    if x is None:
        return MISSING
    return UNDEFINED if x != x else f(x)


def tokens(x: float | None) -> str:
    if x is None or x != x:
        return MISSING
    return f"{x / 1e6:.1f}M" if abs(x) >= 1e6 else f"{x / 1e3:.1f}k"


def sci(x: float | None) -> str:
    if x is None or x != x:
        return MISSING
    mant, exp = f"{x:.1e}".split("e")
    return rf"${mant}\times10^{{{int(exp)}}}$"


class _Out(dict[str, str]):
    """Macros of this block: a name defined twice is an error."""

    def put(self, *parts: str, value: str) -> None:
        name = "".join(("Ex", *parts))
        if name in self:
            raise ValueError(f"macro {name} defined twice")
        self[name] = value


def _pp_or_pct(key: str) -> Callable[[float | None], str]:
    return pct if key in PCT_INTERACTION else pp


# ---------------------------------------------------------------------------------- macros
def _class_split(inp: Mapping[str, Any], out: _Out) -> None:
    for tag, key, arm in DATASETS:
        m = inp.get(key)
        for b, bt in BRANCH_TAGS:
            mix = _get(m, "explore_class_mix", b, "classes") or {}
            for cls, ct in CLASS_TAGS:
                c = mix.get(cls) or {}
                out.put("Mix", tag, bt, ct, "Share", value=point(c.get("share")) if c else MISSING)
                items = str(c["changed_item_times"]) if c else MISSING
                out.put("Mix", tag, bt, ct, "Items", value=items)
                world_items = str(c["world_items"]) if c else MISSING
                out.put("Mix", tag, bt, ct, "WorldItems", value=world_items)
            for g, gt in GRAN_TAGS:
                if g == "cited" and tag == "Short":
                    continue  # the short worlds have no cited-cluster cells
                for cls, ct in CLASS_TAGS:
                    node = _get(m, "explore_class_split", arm, b, g, cls) or {}
                    for ckey, cname in CONTRAST_TAGS:
                        if gt and cname not in ("BDzero", "BDone", "BPooled", "BxD"):
                            continue
                        out.put("Cls", tag, bt, ct, cname, gt, value=fmt(node.get(ckey), pp))


def _decomposition(inp: Mapping[str, Any], out: _Out) -> None:
    worst: list[float] = []
    for tag, key, arm in DATASETS:
        m = inp.get(key)
        grans = GRAN_TAGS if tag in FULL_GRANULARITY else GRAN_TAGS[:1]
        for b, bt in BRANCH_TAGS:
            for cls, ct in MAIN_CLASSES:
                node = _get(m, "explore_decomposition", arm, b, cls) or {}
                if node.get("identity_max_abs_residual") is not None:
                    worst.append(node["identity_max_abs_residual"])
                for g, gt in grans:
                    cells = _get(node, "cells", g) or {}
                    for cell, cellt in CELL_TAGS:
                        if gt and cell.endswith("B0"):
                            continue  # the B=0 cells do not depend on the key
                        for term, termt in TERM_TAGS:
                            if cell.endswith("B0") and termt != "Update":
                                continue  # at B=0 every item is missed: m = 1, r undefined, a = U
                            e = _get(cells, cell, term)
                            out.put("Dec", tag, bt, ct, cellt, termt, gt, value=fmt(e))
                    it = _get(node, "interaction", g) or {}
                    for qkey, qt in INTERACTION_TAGS:
                        out.put(
                            "Int", tag, bt, ct, qt, gt, value=fmt(it.get(qkey), _pp_or_pct(qkey))
                        )
                if tag in FULL_GRANULARITY:
                    inv = _get(node, "invariance", "cited") or {}
                    for ikey, it_ in (
                        ("r_source_minus_cited_D0", "RDzero"),
                        ("r_source_minus_cited_D1", "RDone"),
                        ("m_source_minus_cited_D0", "MDzero"),
                        ("m_source_minus_cited_D1", "MDone"),
                    ):
                        out.put("Inv", tag, bt, ct, it_, value=fmt(inv.get(ikey), pp))
    out.put("DecResidualMax", value=sci(max(worst)) if worst else MISSING)


def _oos(inp: Mapping[str, Any], out: _Out) -> None:
    cross = inp.get("cross") or {}
    for pair, pt in OOS_PAIRS:
        res = _get(cross, "explore_oos", pair) or {}
        for pkey, pred in OOS_PREDICTORS:
            for b, bt in (*BRANCH_TAGS, ("pooled_over_branches", "Pooled")):
                for cls, ct in OOS_CLASSES:
                    if b == "pooled_over_branches":
                        node = _get(res, b, cls, pkey) or {}
                    else:
                        node = _get(res, "branches", b, cls, pkey) or {}
                    if cls == "all_fit_mix" and b == "pooled_over_branches":
                        continue
                    out.put(
                        "Oos",
                        pt,
                        pred,
                        bt,
                        ct,
                        "Mae",
                        value=(
                            f"{_num(node['mean_abs_error']['est'], digits=1)}\\,pp"
                            if node.get("mean_abs_error")
                            else MISSING
                        ),
                    )
                    n_cells = node.get("n_cells", 6)
                    hits = node.get("cells_within_5pp")
                    hit_text = MISSING if hits is None else f"{hits} of {n_cells}"
                    out.put("Oos", pt, pred, bt, ct, "Hits", value=hit_text)
                    if b == "pooled_over_branches":
                        continue
                    excl = node.get("cells_error_excludes_zero")
                    excl_text = MISSING if excl is None else f"{excl} of {n_cells}"
                    out.put("Oos", pt, pred, bt, ct, "Excludes", value=excl_text)
                    for cell, cellt in OOS_CELL_TAGS:
                        if pair != "dev_to_main":
                            continue  # the cell-by-cell errors are quoted for the primary pair
                        c = _get(node, "cells", cell) or {}
                        head = ("Oos", pt, pred, bt, ct, cellt)
                        out.put(*head, "Err", value=fmt(c.get("error"), pp))
                        out.put(*head, "Predicted", value=point(c["predicted"]) if c else MISSING)
                        out.put(*head, "Observed", value=point(c["observed"]) if c else MISSING)


def _invalidation(inp: Mapping[str, Any], out: _Out) -> None:
    for tag, key, arm in DATASETS[:2]:
        m = inp.get(key)
        for b, bt in BRANCH_TAGS:
            for g, gt in (("source", "Source"), ("cited", "Cited")):
                for cls, ct in MAIN_CLASSES:
                    node = _get(m, "explore_invalidation", arm, b, g, cls, "D1") or {}
                    for mkey, mt in PRECISION:
                        out.put("Inval", tag, bt, ct, gt, mt, value=fmt(node.get(mkey)))


def _per_definition(inp: Mapping[str, Any], out: _Out) -> None:
    for tag, key, arm in DATASETS[:2]:
        m = inp.get(key)
        res = _get(m, "explore_per_definition", arm) or {}
        types = sorted({t for b in res.values() for t in b})
        for b, bt in BRANCH_TAGS:
            for t in types:
                tt = camel(t[2:])
                node = _get(res, b, t) or {}
                con = _get(node, "type", "source") or {}
                for ckey, ct in (
                    ("B_effect_update_D0", "BDzero"),
                    ("B_effect_update_D1", "BDone"),
                    ("D_effect_update_B0", "DBzero"),
                    ("D_effect_update_B1", "DBone"),
                    ("BxD_interaction_update", "BxD"),
                ):
                    out.put("PerDef", tag, bt, tt, ct, value=fmt(con.get(ckey), pp))
                loo = _get(node, "without_type", "source", "BxD_interaction_update")
                out.put("PerDefWithout", tag, bt, tt, "BxD", value=fmt(loo, pp))
                for cell, cellt in (*CELL_TAGS, ("zero", "Zero"), ("oracle", "Oracle")):
                    out.put(
                        "PerDefUpdate", tag, bt, tt, cellt, value=fmt(_get(node, "update", cell))
                    )


def _json_and_stability(inp: Mapping[str, Any], out: _Out) -> None:
    for tag, key, arm in DATASETS[:2]:
        m = inp.get(key)
        for b, bt in BRANCH_TAGS:
            for cls, ct in MAIN_CLASSES:
                j = _get(m, "explore_json_arm", b, cls) or {}
                for jk, jt, f in (
                    ("update_sk", "UpdateSk", pct),
                    ("update_json", "UpdateJson", pct),
                    ("update_json_minus_sk", "Update", pp),
                    ("stability_json_minus_sk", "Stability", pp),
                    ("all_probes_json_minus_sk", "AllProbes", pp),
                ):
                    out.put("Json", tag, bt, ct, jt, value=fmt(j.get(jk), f))
                st = _get(m, "explore_stability", b, cls) or {}
                for sysname, sysnt in STABILITY_SYSTEMS:
                    out.put("Stab", tag, bt, ct, sysnt, value=fmt(_get(st, "stability", sysname)))
                for g, gt in GRAN_TAGS:
                    blk = _get(st, arm, g) or {}
                    for d, dt in ((0, "Dzero"), (1, "Done")):
                        stab = blk.get(f"B_effect_stability_D{d}")
                        net = blk.get(f"net_B_effect_all_probes_D{d}")
                        out.put("StabB", tag, bt, ct, dt, gt, value=fmt(stab, pp))
                        out.put("NetB", tag, bt, ct, dt, gt, value=fmt(net, pp))


def _short(inp: Mapping[str, Any], out: _Out) -> None:
    res = _get(inp, "cross", "explore_short") or {}
    for b, bt in BRANCH_TAGS:
        for cls, ct in MAIN_CLASSES:
            node = _get(res, b, cls) or {}
            for d, dt in ((0, "Dzero"), (1, "Done")):
                e = node.get(f"D{d}") or {}
                for qk, qt in (
                    ("predicted", "Predicted"),
                    ("observed", "Observed"),
                    ("observed_minus_predicted", "Gap"),
                    ("predicted_at_main_mix", "AtMainMix"),
                ):
                    out.put("Short", bt, ct, dt, qt, value=fmt(e.get(qk), pp))
            terms = _get(res, b, "terms", cls) or {}
            for tk, tt in (
                ("r_D0", "RDzero"),
                ("r_D1", "RDone"),
                ("a_B0_D0", "ADzero"),
                ("a_B0_D1", "ADone"),
            ):
                for wk, wt, f in (
                    ("main", "Main", pct),
                    ("short", "Short", pct),
                    ("short_minus_main", "Diff", pp),
                ):
                    out.put("ShortTerm", bt, ct, tt, wt, value=fmt(_get(terms, tk, wk), f))


def _replication(inp: Mapping[str, Any], out: _Out) -> None:
    rep = _get(inp, "cross", "explore_replication") or {}
    se = rep.get("source_epoch") or {}

    def n(key: str, *path: str) -> str:
        v = _get(se, *path) if path else se.get(key)
        return str(v) if v is not None else MISSING

    out.put("RepIdentities", value=n("n_identities"))
    out.put("RepNonIdentity", value=n("n_non_identity"))
    out.put("RepContrasts", value=n("n_contrasts"))
    out.put("RepUpdateContrasts", value=n("n_update_contrasts"))
    out.put("RepDRepairContrasts", value=n("n_d_effect_on_repair"))
    out.put("RepIndependent", value=n("independent_update_estimates"))
    summ = _get(rep, "class_split", "summary") or {}
    for cls, ct in (("definition", "Definition"), ("other", "Other")):
        s = summ.get(cls) or {}
        for k, kt in (
            ("n_contrasts", "N"),
            ("rep_same_sign", "SameSign"),
            ("rep_excludes_zero", "ExcludesZero"),
            ("orig_excludes_zero", "MainExcludesZero"),
        ):
            out.put("RepClass", ct, kt, value=str(s[k]) if k in s else MISSING)
    holm = rep.get("holm_non_detection") or {}
    for stage, st in (("main", "Main"), ("rep", "Rep")):
        h = holm.get(stage) or {}
        for k, kt, f in (
            ("family_size", "Family", str),
            ("draws", "Draws", lambda x: f"{x:,}"),
            ("n_survive_holm", "Survive", str),
            ("n_survive_bh", "SurviveBh", str),
            ("n_draws_below_floor_zero", "AtFloor", str),
            ("smallest_possible_p", "SmallestP", lambda x: f"{x:.4f}"),
            ("smallest_possible_holm_p", "SmallestHolmP", lambda x: f"{x:.4f}"),
            ("largest_holm_p_among_survivors", "LargestSurvivorP", lambda x: f"{x:.4f}"),
        ):
            out.put("Holm", st, kt, value=f(h[k]) if h.get(k) is not None else MISSING)
        out.put("Holm", st, "Fail", value=str(len(h["failing"])) if "failing" in h else MISSING)


def _cost_and_noise(inp: Mapping[str, Any], out: _Out) -> None:
    cross = inp.get("cross") or {}
    for stage, st in (("main", "Main"), ("rep", "Rep")):
        for h, ht in HORIZONS:
            per = _get(cross, "explore_cost", stage, "horizons", h) or {}
            for s, sn in COST_SYSTEMS:
                e = per.get(s) or {}
                for k, kt, f in (
                    ("ingest_tokens", "Ingest", tokens),
                    ("maintain_tokens", "Maintain", tokens),
                    ("serve_tokens", "Serve", tokens),
                    ("total_tokens", "Total", tokens),
                    ("total_over_zero", "RatioZero", lambda x: f"{x:.2f}"),
                    ("maintain_over_serve", "MaintainOverServe", lambda x: f"{x:.2f}"),
                    ("maintain_tokens_per_query", "MaintainPerQuery", tokens),
                    ("maintain_writes_counted", "Writes", lambda x: f"{x:,.0f}"),
                    ("tokens_per_counted_write", "TokensPerWrite", tokens),
                ):
                    v = e.get(k)
                    out.put(
                        "CostModel",
                        st,
                        ht,
                        sn,
                        kt,
                        value=f(v) if v is not None else UNDEFINED if e else MISSING,
                    )
    main = inp.get("main")
    for b, bt in BRANCH_TAGS:
        for cell, cellt in (
            ("holdmem_D0B0", "DzeroBzero"),
            ("holdmem_D1B0", "DoneBzero"),
            ("holdmem_D0B1_source", "DzeroBoneSource"),
            ("holdmem_D1B1_source", "DoneBoneSource"),
            ("holdmem_D0B1_cited", "DzeroBoneCited"),
            ("holdmem_D1B1_cited", "DoneBoneCited"),
        ):
            e = _get(main, "explore_cost_e1", b, cell) or {}
            for k, kt in (
                ("maintain_tokens_per_world_branch", "Maintain"),
                ("ingest_tokens_per_world_branch", "Ingest"),
            ):
                out.put("CostHoldEOne", bt, cellt, kt, value=tokens(e[k]) if k in e else MISSING)
    nf = _get(cross, "explore_noise_floor") or {}
    names = inp.get("names") or {}
    rows = {
        "Main": _get(nf, "stages", names.get("main", "")),
        "Rep": _get(nf, "stages", names.get("rep", "")),
        "Pooled": nf.get("pooled"),
    }
    for tag, r in rows.items():
        r = r or {}
        w = r.get("world_level") or {}
        for k, kt, f in (
            ("worlds", "Worlds", str),
            ("items", "WorldItems", lambda x: f"{x:,}"),
            ("values_differ", "WorldValues", str),
            ("answers_differ", "WorldAnswers", str),
        ):
            out.put("NoiseHold", tag, kt, value=f(w[k]) if k in w else MISSING)
        for k, kt, f in (
            ("world_branches", "WorldBranches", str),
            ("items", "Items", lambda x: f"{x:,}"),
            ("values_differ", "Values", str),
            ("values_differ_share", "ValuesShare", lambda x: pct(x, 2)),
            ("answers_differ", "Answers", str),
            ("answers_differ_share", "AnswersShare", lambda x: pct(x, 2)),
        ):
            out.put("NoiseHold", tag, kt, value=f(r[k]) if k in r else MISSING)


def numbers(inp: Mapping[str, Any]) -> dict[str, str]:
    """Every `Ex` macro of the further E1-family analyses, from `load`."""
    out = _Out()
    for block in (
        _class_split,
        _decomposition,
        _oos,
        _invalidation,
        _per_definition,
        _json_and_stability,
        _short,
        _replication,
        _cost_and_noise,
    ):
        block(inp, out)
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return dict(out)


# ------------------------------------------------------------- cells of the shared table style
# The tables below use the macros of paper/memtab-style.tex: \mtint{[lo,\,hi]} (an interval in
# small muted type), \mtpair{est}{[lo,\,hi]} (an estimate and its interval in one cell, in boxes
# set by \mtpairwidths) and the column N{template} (values right-aligned in a box as wide as the
# template, the box centred under the header).
MUTED = r"\color{mtmuted}"  # a cell in the muted grey of the figures
PCT_BOX = "100"  # the N template of a column of whole percentages: no share prints wider
GROUP_GAP = r"@{\hspace{14pt}}"  # between two column groups (shifts, studies, cells)
_SIGNS = ("$+$", "$-$")
_EM = {"+": 0.57, "~": 0.17, ".": 0.25, ",": 0.25, "[": 0.33, "]": 0.33}  # else 0.5 (a digit)


def ci_text(
    e: Mapping[str, Any] | None, sign: bool = False, f: Callable[[float], str] | None = None
) -> tuple[str, str] | None:
    """The printed estimate and interval `[lo,\\,hi]` of a node, the interval empty where its
    printed ends equal the estimate (every world gave the same value); None when the node is not
    there or its estimate is not a number."""
    if not e or e.get("est") is None or e["est"] != e["est"]:
        return None
    num = f or (lambda x: _num(x, sign=sign))
    est, lo, hi = (num(e[k]) for k in ("est", "lo", "hi"))
    return est, "" if lo == est == hi else f"[{lo},\\,{hi}]"


def _no_value(e: Mapping[str, Any] | None) -> str:
    return MISSING if not e else UNDEFINED


def mt_ci_cells(
    e: Mapping[str, Any] | None,
    sign: bool = False,
    style: str = "",
    f: Callable[[float], str] | None = None,
    span: str = "c",
) -> str:
    """An estimate and its 95% interval as the two cells of a column pair `r@{\\,}l`, the interval
    in \\mtint, the estimate alone where the printed ends equal it. `style` opens the estimate's
    cell (the interval has its own colour); `f` prints each number (default: percent or pp with
    one decimal, signed if `sign`). MISSING or UNDEFINED span both cells, in a column `span`."""
    text = ci_text(e, sign, f)
    if text is None:
        return rf"\multicolumn{{2}}{{{span}}}{{{style}{_no_value(e)}}}"
    est, iv = text
    return f"{style}{est} &" + (f" \\mtint{{{iv}}}" if iv else "")


def mt_ci_pair_cells(
    e: Mapping[str, Any] | None,
    sign: bool = False,
    style: str = "",
    f: Callable[[float], str] | None = None,
    span: str = "c",
) -> str:
    """`mt_ci_cells` where an estimate alone keeps a space before the next cell."""
    cells = mt_ci_cells(e, sign=sign, style=style, f=f, span=span)
    return f"{cells} " if cells.endswith(" &") else cells


def mt_pair(
    e: Mapping[str, Any] | None, sign: bool = False, f: Callable[[float], str] | None = None
) -> str:
    """An estimate and its interval in one cell (\\mtpair), for a column of pairs centred under a
    header wider than the pair; MISSING or UNDEFINED alone."""
    text = ci_text(e, sign, f)
    return _no_value(e) if text is None else rf"\mtpair{{{text[0]}}}{{{text[1]}}}"


def _em(s: str) -> float:
    """The printed width of a number or interval in em, near enough to find the widest of a
    column: a digit half an em, a sign wider, a point or comma a quarter."""
    s = s.replace("$+$", "+").replace("$-$", "+").replace("\\,", "~")
    return sum(_EM.get(ch, 0.5) for ch in s)


def _is_number(s: str) -> bool:
    mag = s[3:] if s[:3] in _SIGNS else s
    return bool(mag) and all(ch.isdigit() or ch in ".," for ch in mag)


def _zeros(s: str, sign: str) -> str:
    """A printed number as a template of its width: digits as 0 and a sign as `sign` (the two
    signs are equally wide); a printed 100 (a whole share) stays as printed."""
    signed = s[:3] in _SIGNS
    mag = s[3:] if signed else s
    if mag.split(".")[0] == "100":
        return s
    return (sign if signed else "") + "".join("0" if ch.isdigit() else ch for ch in mag)


def n_template(cells: list[str]) -> str:
    """The template of an N column: its widest printed value as `_zeros` (cells that are not a
    number, such as MISSING, UNDEFINED or blank, do not count)."""
    numbers = [c for c in cells if _is_number(c)]
    return _zeros(max(numbers, key=_em), "$+$") if numbers else "0"


def pair_widths(pairs: list[tuple[str, str] | None], template: bool = False) -> str:
    """\\mtpairwidths for a column of `mt_pair` cells, from the `ci_text` of each: the widest
    estimate and the widest interval as printed, or with `template` as `_zeros` (an estimate's
    sign as +, an interval end's as -)."""
    texts = [t for t in pairs if t is not None]
    est = max((t[0] for t in texts), key=_em, default="")
    iv = max((t[1] for t in texts), key=_em, default="")
    if template:
        est = _zeros(est, "$+$") if est else ""
        if iv:
            lo, hi = iv[1:-1].split(",\\,")
            iv = f"[{_zeros(lo, '$-$')},\\,{_zeros(hi, '$-$')}]"
    return rf"\mtpairwidths{{{est}}}{{{iv}}}"


# ---------------------------------------------------------------------------------- tables
def _cell(e: dict[str, Any] | None) -> str:
    """`est [lo, hi]` in pp, estimate and bounds signed (`paper_tables.interval`)."""
    if not e:
        return MISSING
    if e["est"] != e["est"]:
        return UNDEFINED
    return _ci(e, sign=True)


CLASSSPLIT_CONTRASTS = (  # the columns of tab_classsplit, in the order of their headers
    "B_effect_update_D0",
    "B_effect_update_D1",
    "D_effect_update_B0",
    "D_effect_update_B1",
    "BxD_interaction_update",
)
BRANCH_LABELS = {"src": r"$\Ssrc$", "def": r"$\Sdef$", "orphan": r"$\Sdel$"}
CLASSSPLIT_STUDIES = (  # the row blocks: banner, inputs key of its world count, and its D=1
    # arms (label, inputs key, arm of the D=1 cell) in the rows of each shift and class
    ("Main study: {n} test worlds", "main", (("exact", "main", "D1"), ("programs", "c", "D1s"))),
    ("Replication: {n} fresh worlds", "rep", (("exact", "rep", "D1"),)),
)


def tab_classsplit(inp: Mapping[str, Any]) -> str:
    """Appendix: B effect at each D, D effect at each B and B x D on Update by shift and item
    class (definition, other, all), in two row blocks: the main study, with exact recomputation
    and the writer's programs D_m (block C, the main study's worlds) as the D=1 arm in the rows
    of each class, and the replication with exact recomputation. The arms share the columns, so
    the effects of exact recomputation and of the programs can be read off one column. Each
    estimate and its interval are two aligned cells (`mt_ci_cells`)."""
    pairs = " ".join([r"r@{\,}l"] * 2)
    ncols = 3 + 2 * len(CLASSSPLIT_CONTRASTS)
    lines = [
        r"\begin{table*}[t]",
        r"\caption{The contrasts on Update (pp) by shift and item class, with exact "
        r"recomputation and with the writer's programs as the $D{=}1$ arm.}",
        r"\label{tab:classsplit}",
        r"\centering\mtsetup",
        r"\setlength{\tabcolsep}{3.2pt}",
        rf"\begin{{mttabular}}{{@{{}}lll {pairs} {GROUP_GAP} {pairs} {GROUP_GAP} r@{{\,}}l@{{}}}}",
        r"\toprule",
        r"& & & \multicolumn{4}{c}{$B$ effect} & \multicolumn{4}{c}{$D$ effect} & & \\",
        r"\cmidrule(lr){4-7}\cmidrule(lr){8-11}",
        r"Shift & Class & Arm & \multicolumn{2}{c}{$D{=}0$} & \multicolumn{2}{c}{$D{=}1$} "
        r"& \multicolumn{2}{c}{$B{=}0$} & \multicolumn{2}{c}{$B{=}1$} "
        r"& \multicolumn{2}{c@{}}{$B{\times}D$} \\",
    ]
    for banner, n_key, arms in CLASSSPLIT_STUDIES:
        n = _get(inp.get(n_key), "n_worlds") or MISSING
        lines += [r"\midrule", rf"\mtgroup{{{ncols}}}{{{banner.format(n=n)}}}"]
        for i, (b, _) in enumerate(BRANCH_TAGS):
            for k, (cls, _) in enumerate(MAIN_CLASSES):
                for j, (label, key, arm) in enumerate(arms):
                    node = _get(inp.get(key), "explore_class_split", arm, b, "source", cls)
                    vals = [
                        mt_ci_cells(
                            _get(node, c),
                            sign=True,
                            span="c@{}" if c == CLASSSPLIT_CONTRASTS[-1] else "c",
                        )
                        for c in CLASSSPLIT_CONTRASTS
                    ]
                    lines.append(
                        _row(
                            BRANCH_LABELS[b] if k == 0 and j == 0 else "",
                            cls if j == 0 else "",
                            label,
                            *vals,
                        )
                    )
                if k < len(MAIN_CLASSES) - 1 and len(arms) > 1:  # between classes of 2+ rows
                    lines.append(r"\mtgap")
            if i < len(BRANCH_TAGS) - 1:
                lines.append(r"\midrule")
    lines += [
        r"\bottomrule",
        r"\end{mttabular}",
        r"\mtnotes{Exact: exact recomputation as the $D{=}1$ arm. Programs: the writer's programs "
        r"($D_{\mathrm{m}}$) as the $D{=}1$ arm, so in their rows $D{=}1$ is $D_{\mathrm{m}}$. "
        r"The $D{=}0$ cells of the main study serve both of its arms.}",
        r"\end{table*}",
    ]
    return "\n".join(lines) + "\n"


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {"tab_classsplit.tex": tab_classsplit(inp)}


# ---------------------------------------------------------------------------------- figure
CLASSSPLIT_PT = (486, 165.6)  # printed size (pt): full width
CLASSSPLIT_STYLES = (  # contrast, key label, bar fill, dark ink (interval, value label)
    ("B_effect_update_D0", r"$B$ effect, $D\!=\!0$", "#9AA3AF", "#3F4753"),  # re-derives: slate
    ("B_effect_update_D1", r"$B$ effect, $D\!=\!1$", "#2B6CB8", "#14365F"),  # recomputes: blue
    ("BxD_interaction_update", r"$B{\times}D$", "#962F6E", "#4F1838"),  # the result: purple
)
CLASSSPLIT_CLASSES = (("definition", "definition items"), ("other", "other items"))
CLASSSPLIT_PANELS = (
    ("src", r"source revision $S_{\mathrm{src}}$"),
    ("def", r"redefinition $S_{\mathrm{def}}$"),
    ("orphan", r"deletion $S_{\mathrm{del}}$"),
)
CLASSSPLIT_SLOT = 0.305  # distance between the three contrasts of one class (classes 1 apart)
CLASSSPLIT_BAR_W = 0.19  # a main-study bar, in classes
CLASSSPLIT_REP_DX = 0.135  # the replication's diamond, right of its bar's centre
CLASSSPLIT_REP_MS, CLASSSPLIT_REP_LW = 4.1, 0.6  # its size and its interval line (pt)
CLASSSPLIT_Y = (-7.5, 66.0)  # the value axis (pp), widened when an interval leaves it
CLASSSPLIT_REP_INK, CLASSSPLIT_REP_FILL = "#4A5059", "#59606B"  # the replication's key glyph


def _classsplit_node(inp: Mapping[str, Any], stage: str, branch: str, cls: str, key: str
                     ) -> dict[str, Any] | None:  # fmt: skip
    """A contrast of the class split (D=1, source key), or None when it is not staged."""
    e = _get(inp.get(stage), "explore_class_split", "D1", branch, "source", cls, key)
    return None if not e or e["est"] != e["est"] else dict(e)


def fig_classsplit(inp: Mapping[str, Any], out: Path) -> Path:
    """Per shift (panel) and item class (group), the B effect on Update at D=0 (slate) and D=1
    (blue) and B x D (purple): the main study as bars from zero, the replication as a diamond
    with its interval beside each bar, every interval a thin capped line in the mark's dark ink.
    B x D on the definition items is printed over its bar; under each class its share of the
    shift's changed item-times (main study). The key row sits above the panels."""
    from memshift.analysis import figstyle as fs

    w_mm, h_mm = CLASSSPLIT_PT[0] * fs.PT, CLASSSPLIT_PT[1] * fs.PT
    left, right, gutter, bottom, top = 11.4, 0.6, 3.6, 8.6, 10.4
    pw = (w_mm - left - right - 2 * gutter) / 3
    ph = h_mm - bottom - top
    nodes = {(stage, b, cls, key): _classsplit_node(inp, stage, b, cls, key)
             for stage in ("main", "rep") for b, _ in CLASSSPLIT_PANELS
             for cls, _ in CLASSSPLIT_CLASSES for key, *_ in CLASSSPLIT_STYLES}  # fmt: skip
    ends = [100 * e[k] for e in nodes.values() if e for k in ("lo", "hi")]
    lo = min([CLASSSPLIT_Y[0], *(v - 1.5 for v in ends)])
    hi = max([CLASSSPLIT_Y[1], *(v + 1.5 for v in ends)])
    with fs.printed(fs.BAR_RC) as plt:
        fig = plt.figure(figsize=(w_mm * fs.MM, h_mm * fs.MM))
        axes = []
        for i, (branch, title) in enumerate(CLASSSPLIT_PANELS):
            x0 = left + i * (pw + gutter)
            ax = fig.add_axes((x0 / w_mm, bottom / h_mm, pw / w_mm, ph / h_mm))
            fs.bar_axes(ax)
            ax.set_ylim(lo, hi)
            ax.set_yticks([0, 20, 40, 60])
            ax.set_xlim(-0.56, 1.56)
            ax.set_title(title, fontsize=fs.T_TITLE, pad=3.2)
            if i:
                ax.tick_params(axis="y", labelleft=False)
            axes.append(ax)
            cap = fs.CAP_MM * fs.x_per_mm(ax, pw)
            for c, (cls, _) in enumerate(CLASSSPLIT_CLASSES):
                for k, (key, _, fill, ink) in enumerate(CLASSSPLIT_STYLES):
                    xb = c + (k - 1) * CLASSSPLIT_SLOT - 0.03
                    main, rep = nodes["main", branch, cls, key], nodes["rep", branch, cls, key]
                    if main:
                        fs.bar_from_zero(ax, xb, CLASSSPLIT_BAR_W, 100 * main["est"], fill)
                        fs.interval_line(ax, xb, 100 * main["lo"], 100 * main["hi"], ink, cap)
                    xr = xb + CLASSSPLIT_REP_DX
                    if rep:
                        fs.interval_line(ax, xr, 100 * rep["lo"], 100 * rep["hi"], ink, 0.7 * cap,
                                         lw=CLASSSPLIT_REP_LW, z=4.2)  # fmt: skip
                        ax.plot([xr], [100 * rep["est"]], marker="D", ms=CLASSSPLIT_REP_MS,
                                mfc=fill, mec="white", mew=0.6, ls="none", zorder=5)  # fmt: skip
                    if main and cls == "definition" and key == "BxD_interaction_update":
                        # the result: B x D of the main study, over its bar and diamond
                        y = 100 * max(main["hi"], rep["hi"] if rep else main["hi"]) + 1.4
                        ax.text(xb + CLASSSPLIT_REP_DX / 2, y,
                                f"{100 * main['est']:+.1f}".replace("-", "−"), ha="center",
                                va="bottom", fontsize=fs.T_NOTE, color=ink)  # fmt: skip
            share = _get(inp.get("main"), "explore_class_mix", branch, "classes")
            ax.set_xticks([0, 1], [name for _, name in CLASSSPLIT_CLASSES])
            for c, (cls, _) in enumerate(CLASSSPLIT_CLASSES):  # the class's share of the changes
                if (s := _get(share, cls, "share")) is not None:
                    ax.annotate(f"({100 * s:.0f}%)", (c, 0), xycoords=("data", "axes fraction"),
                                xytext=(0, -12.0), textcoords="offset points", ha="center",
                                va="top", fontsize=fs.T_NOTE, color=fs.QUIET)  # fmt: skip
        axes[0].set_ylabel("effect on Update (pp)", fontsize=fs.T_LABEL, labelpad=2.0)
        fig.canvas.draw()

        def rep_glyph(ov: Any, x: float, y: float) -> None:
            """The replication: a neutral diamond on its interval line."""
            ov.plot([x + 1.5] * 2, [y - 1.35, y + 1.35], color=CLASSSPLIT_REP_INK,
                    lw=CLASSSPLIT_REP_LW, solid_capstyle="butt")  # fmt: skip
            ov.plot([x + 1.5], [y], marker="D", ms=CLASSSPLIT_REP_MS, mfc=CLASSSPLIT_REP_FILL,
                    mec="white", mew=0.6, ls="none")  # fmt: skip

        entries: list[tuple[fs.Glyph | None, str]] = [
            (fs.key_swatch(fill), label) for _, label, fill, _ in CLASSSPLIT_STYLES
        ]
        entries += [(None, "bars: main study"), (rep_glyph, "replication")]
        fs.key_row(fig, entries, h_mm - 2.9, w_mm, h_mm, breaks=(3,))
        return fs.save_pdf(fig, out / "fig_classsplit.pdf")

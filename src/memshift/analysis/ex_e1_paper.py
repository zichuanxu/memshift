"""Paper numbers, tables and the figure of the exploratory E1-family analyses (WS1a, prereg/
decisions.md 2026-09-30): the class-stratified contrasts, the Eq. (1) decomposition, the
out-of-sample test, invalidation precision, per-definition spread, the post-shift JSON arm,
Stability, the short worlds, replication accounting, cost and the noise floor. Every macro name
starts with `Ex` (exploratory) and every source is a staged analysis named in
results/sources.json:

  e1 / rep_e1 / c / m2_gemma / m2_llama / m2_gptoss / dev_e1 / dev_orphan_e1 / short_e1
                     analysis/e1_metrics.json, with the `explore_*` blocks of
                     `analysis.ex_e1_classes` (the development worlds are the union of dev_e1 and
                     dev_orphan_e1)
  ex                                               analysis/ex_e1.json (`analysis.ex_e1_cross`)

A stage that is not staged renders as the MISSING marker; a quantity that is undefined by
construction (the accuracy of the missed items when a key misses none) renders as a dash.

tab_classsplit   main text: B, D and B x D on Update by item class, main study and R
tab_decomp       appendix: Eq. (1) terms, the interaction three ways, out-of-sample errors
fig_classsplit   forest plot of the B effect at each D and B x D, by class and branch, main and R
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import BRANCH_TAGS, MISSING, camel, stage_json
from memshift.analysis.paper_data import get as _get
from memshift.analysis.paper_numbers import _macro, ci, pct, pp
from memshift.analysis.paper_tables import _ci, _est, _num, _row

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
    out.put("RepNonIdentityReplicated", value=n("", "verdicts_non_identity", "replicated"))
    out.put("RepNonIdentitySame", value=n("", "verdicts_non_identity", "same direction"))
    out.put("RepNonIdentityNot", value=n("", "verdicts_non_identity", "not replicated"))
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
            ("replicated", "Replicated"),
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
    """Every `Ex` macro of the exploratory E1-family analyses, from `load`."""
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


# ---------------------------------------------------------------------------------- tables
def _cell(e: dict[str, Any] | None) -> str:
    """`est [lo, hi]` in pp, estimate and bounds signed (`paper_tables.interval`)."""
    if not e:
        return MISSING
    if e["est"] != e["est"]:
        return UNDEFINED
    return _ci(e, sign=True)


def _branch_cols() -> str:
    return (
        r"\multicolumn{2}{c}{$\Ssrc$} & \multicolumn{2}{c}{$\Sdef$} & \multicolumn{2}{c}{$\Sdel$}"
    )


def tab_classsplit(inp: Mapping[str, Any]) -> str:
    """Main text: B effect at each D, D effect at each B and B x D on Update, for the definition
    items and all the others, per branch: the main study (first line) and the replication (second
    line); the share of changed item-times of each class first. Each contrast's name heads its pair
    of lines, so the label column is narrow and the six interval columns fit a 5.5in text width."""
    main, rep = inp.get("main"), inp.get("rep")
    rows_ = (
        ("B_effect_update_D0", r"$B$ effect, $D{=}0$"),
        ("B_effect_update_D1", r"$B$ effect, $D{=}1$"),
        ("D_effect_update_B0", r"$D$ effect, $B{=}0$"),
        ("D_effect_update_B1", r"$D$ effect, $B{=}1$"),
        ("BxD_interaction_update", r"$B{\times}D$ interaction"),
    )
    n_main = _get(main, "n_worlds") or MISSING
    n_rep = _get(rep, "n_worlds") or MISSING
    lines = [
        r"\begin{table}[t]",
        r"\caption{The registered contrasts on Update by item class (source-epoch key, pp, "
        r"95\% world-bootstrap intervals). $B$ effect: $B{=}1$ minus $B{=}0$ at the given $D$. "
        r"$D$ effect: $D{=}1$ minus $D{=}0$ at the given $B$. $B{\times}D$: the $B$ effect at "
        r"$D{=}1$ minus that at $D{=}0$. Definition: the six evaluated definition items "
        r"of each world. Other: the other 42 (single-field and free-text) items. Share: "
        r"the class's share of the changed item-times in the main study. First line of each "
        rf"pair: the main study ({n_main} worlds), whose class split is exploratory. Second line, "
        rf"in grey: the replication ({n_rep} fresh worlds), whose class rows are registered.}}",
        r"\label{tab:classsplit}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{2pt}",
        r"\begin{tabular}{@{}l cc cc cc@{}}",
        r"\toprule",
        rf"& {_branch_cols()} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(l){6-7}",
        r"& definition & other & definition & other & definition & other \\",
        r"\midrule",
    ]
    cells = []
    for b, _ in BRANCH_TAGS:
        for cls in ("definition", "other"):
            share = _get(main, "explore_class_mix", b, "classes", cls, "share")
            cells.append(_num(share, digits=0) + r"\%" if share is not None else MISSING)
    lines += [_row("share", *cells), r"\midrule"]
    for key, label in rows_:
        lines.append(rf"\multicolumn{{7}}{{@{{}}l}}{{{label}}} \\")
        for stage, name in (("main", r"\quad main"), ("rep", r"\quad\textcolor{gray}{repl.}")):
            m = main if stage == "main" else rep
            vals = []
            for b, _ in BRANCH_TAGS:
                for cls in ("definition", "other"):
                    e = _get(m, "explore_class_split", "D1", b, "source", cls, key)
                    cell = _cell(e)
                    vals.append(cell if stage == "main" else rf"\textcolor{{gray}}{{{cell}}}")
            lines.append(_row(name, *vals))
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def tab_decomp(inp: Mapping[str, Any]) -> str:
    """Appendix: (a) the Eq. (1) terms of the main study per branch and class for the source-epoch
    and the cited-cluster key, and the B x D interaction as observed, as the paper states it and
    as the identity gives it; (b) the out-of-sample test from the development worlds."""
    main = inp.get("main")
    cross = inp.get("cross")
    lines = [
        r"\begin{table}[t]",
        r"\caption{(a) Equation~(1) accounting (exploratory; main study; percent for terms, pp for "
        r"the interaction). $m_1$: missed invalidation of the key; $r_D$: accuracy of the "
        r"invalidated (re-derived) items; $a_D$: Update at $B{=}0$, where every item is missed. "
        r"With $a$ held fixed across $D$: $(m_0{-}m_1)(r_1{-}r_0)$; corrected: "
        r"$(r_1{-}a_1){-}(r_0{-}a_0)$, equal to the "
        r"observed interaction when the key misses nothing. (b) Mean absolute error in pp over the "
        r"six cells (pooled: over the three branches' cells) of the prediction of each cell's "
        r"Update on the test worlds from the development worlds, with the number of cells "
        r"within 5\,pp: \emph{cell}: each cell's own "
        r"fitted Update; \emph{structural}: Eq.~(1) with $r$ by $D$, $m$ by the key, $a$ the "
        r"$B{=}0$ Update (write path) or the accuracy of the missed items; \emph{literal}: "
        r"Eq.~(1) with one stale-value accuracy for every cell.}",
        r"\label{tab:decomp}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}ll rrrrr rrrr@{}}",
        r"\toprule",
        r"& & $m_1$ & $r_0$ & $r_1$ & $a_0$ & $a_1$ & observed & $a$ fixed & corrected "
        r"& $D\mid B{=}0$ \\",
    ]
    for g, title in (("source", "source-epoch key"), ("cited", "cited-cluster key")):
        lines += [r"\midrule", rf"\multicolumn{{11}}{{@{{}}l}}{{\emph{{{title}}}}} \\"]
        for b, bn in (("src", r"$\Ssrc$"), ("def", r"$\Sdef$"), ("orphan", r"$\Sdel$")):
            for cls, cn in (("definition", "definition"), ("other", "other")):
                node = _get(main, "explore_decomposition", "D1", b, cls)
                lines.append(_terms_row_cells(bn if cls == "definition" else "", cn, node, g))
    lines += [r"\bottomrule", r"\end{tabular}", r"\par\medskip", r"\setlength{\tabcolsep}{4pt}"]
    lines += [
        r"\begin{tabular}{@{}ll rrrr@{}}",
        r"\toprule",
        r"& & $\Ssrc$ & $\Sdef$ & $\Sdel$ & pooled \\",
        r"\midrule",
    ]
    res = _get(cross, "explore_oos", "dev_to_main") or {}
    for cls, cn in (("definition", "definition"), ("other", "other"), ("all", "all items")):
        for k, (pkey, pn) in enumerate(
            (("cell", "cell"), ("structural", "structural"), ("literal", "literal"))
        ):
            nodes = [_get(res, "branches", b, cls, pkey) for b in ("src", "def", "orphan")]
            nodes.append(_get(res, "pooled_over_branches", cls, pkey))
            lines.append(_row(cn if k == 0 else "", pn, *(_mae(n) for n in nodes)))
        if cls != "all":
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def _mae(node: dict[str, Any] | None) -> str:
    """Mean absolute error in pp and the number of cells within 5 pp."""
    if not node or not node.get("mean_abs_error") or "cells_within_5pp" not in node:
        return MISSING
    mae = _num(node["mean_abs_error"]["est"], digits=1)
    return f"{mae} ({node['cells_within_5pp']}/{node.get('n_cells', 6)})"


def _terms_row_cells(branch: str, cls: str, node: dict[str, Any] | None, g: str) -> str:
    it = _get(node, "interaction", g) or {}

    def cell(c: str, term: str) -> str:
        e = _get(node, "cells", g, c, term)
        if not e:
            return MISSING
        return UNDEFINED if e["est"] != e["est"] else _num(e["est"], digits=1)

    def q(k: str, sign: bool = True) -> str:
        e = it.get(k)
        if not e:
            return MISSING
        return UNDEFINED if e["est"] != e["est"] else _est(e, sign=sign)

    m1 = it.get("m1")
    return _row(
        branch, cls,
        _num(m1["est"], digits=0) if m1 and m1["est"] == m1["est"] else MISSING,
        cell("D0B1", "r"), cell("D1B1", "r"), cell("D0B0", "a"), cell("D1B0", "a"),
        q("observed"), q("paper_formula"), q("corrected_formula"), q("D_effect_B0"),
    )  # fmt: skip


def tables(inp: Mapping[str, Any]) -> dict[str, str]:
    return {"tab_classsplit.tex": tab_classsplit(inp), "tab_decomp.tex": tab_decomp(inp)}


# ---------------------------------------------------------------------------------- figure
def fig_classsplit(inp: Mapping[str, Any], out: Path) -> Path:
    """Forest plot: per branch (panel) and item class, the B effect at D=0 and D=1 and B x D on
    Update as interval bars (`figstyle.interval_bar`), main study (filled) and R (outlined), 95%
    intervals. Each class header carries the class's
    share of the branch's changed item-times (main study)."""
    from memshift.analysis import figstyle as fs

    plt = fs.plt()
    contrasts = (
        ("B_effect_update_D0", r"$B$ effect, $D{=}0$", fs.SKY),
        ("B_effect_update_D1", r"$B$ effect, $D{=}1$", fs.BLUE),
        ("BxD_interaction_update", r"$B{\times}D$", fs.PURPLE),
    )
    classes = (("definition", "definition items"), ("other", "other items"))
    panels = (
        ("src", r"source revision $S_{\mathrm{src}}$"),
        ("def", r"redefinition $S_{\mathrm{def}}$"),
        ("orphan", r"deletion $S_{\mathrm{del}}$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(fs.TEXT, 2.3), sharey=True, sharex=True)
    # one header row per class (its name left of the axes, its share in each panel), then one row
    # per contrast; the main study sits above the replication within a row
    head: dict[str, float] = {}
    ypos: dict[tuple[str, str], float] = {}
    y = 0.0
    for cls, _ in classes:
        head[cls] = y
        for key, _, _ in contrasts:
            y -= 1.0
            ypos[cls, key] = y
        y -= 1.15
    for ax, (branch, title) in zip(axes, panels, strict=True):
        ax.axvline(0, color=fs.GREY, lw=0.8, zorder=1)
        for (cls, key), yy in ypos.items():
            colour = next(c for k, _, c in contrasts if k == key)
            for stage, dy, filled in (("main", 0.2, True), ("rep", -0.2, False)):
                e = _get(inp.get(stage), "explore_class_split", "D1", branch, "source", cls, key)
                if not e or e["est"] != e["est"]:
                    continue
                fs.interval_bar(ax, yy + dy, 100 * e["est"], 100 * e["lo"], 100 * e["hi"], colour,
                                filled=filled, size=4.0)  # fmt: skip
        share = _get(inp.get("main"), "explore_class_mix", branch, "classes")
        for cls, _ in classes:
            s = _get(share, cls, "share")
            if s is None:
                continue
            # on a white box at the right edge, so that neither the zero line nor a gridline
            # runs through the note
            ax.text(0.985, head[cls], f"{100 * s:.0f}" + r"\% of changed item-times",
                    fontsize=7.5, color=fs.BLACK, ha="right", va="center", zorder=5,
                    transform=ax.get_yaxis_transform(),
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})  # fmt: skip
        ax.set_title(title, pad=3)
        ax.set_xlim(-14, 68)  # the deletion-branch B effects on the other items reach about 63 pp
        ax.set_xticks([0, 25, 50])
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
    rows = [(head[cls], rf"\textbf{{{label}}}") for cls, label in classes]
    rows += [(ypos[cls, key], lab) for cls, _ in classes for key, lab, _ in contrasts]
    axes[0].set_yticks([r for r, _ in rows], [lab for _, lab in rows])
    axes[0].set_ylim(y + 0.55, 0.6)
    for ax in axes:
        ax.tick_params(axis="y", length=0)
    fig.supxlabel("effect on Update (pp)", fontsize=9)
    handles = [
        fs.interval_handle(fs.GREY, True, "main study"),
        fs.interval_handle(fs.GREY, False, "replication"),
    ]
    # in the empty corner under the row labels, left of the first panel's tick labels
    fig.legend(handles=handles, handler_map=fs.INTERVAL_HANDLERS, loc="lower left",
               bbox_to_anchor=(0.0, 0.0), ncol=1, handletextpad=0.4, handlelength=1.6,
               labelspacing=0.25, borderaxespad=0.1, fontsize=7.5)  # fmt: skip
    path = out / "fig_classsplit.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path

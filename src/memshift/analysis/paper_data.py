"""Inputs of the stage-4 paper numbers, tables and figure: which staged analysis feeds which block,
and the views derived from it that the three generators share.

A stage that has not been staged is None (or an empty view), and every consumer renders that as
the MISSING marker; nothing here supplies a value that a run has not produced. Sources are named
in results/sources.json (group -> stage directory whose analysis/*.json is committed):

  rep_e1 / rep_e2 / rep_e3   block R, the unchanged stage-2 analyses on the replication worlds
  pool_e1 / pool_e2 / pool_e3   stage 2 and R pooled over the forty test worlds (secondary)
  c                          block C, E1 with the D1s cells (analysis/e1_metrics.json)
  m1                         block M1, E2 merged over the stage-2 tiers and `cross_<slug>` tiers
  m2_<family>                block M2, E1 with the family as writer and reader
  m3                         block M3, reader swaps (analysis/e7_metrics.json, theta_r_by_reader)
  n, ncost                   block N, E5 over Mem0 and the stage-4 systems; native cost per event
  l                          block L, analysis/rendered.json
  rank                       the rank correlation, analysis/rank_corr.json
  mult                       multiplicity check (analysis/multiplicity.json) of the E1
                             contrasts expected to be non-zero, of `e1` (stage 2) and `rep_e1` (R)
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from memshift.analysis import gate, replication

MISSING = r"\textbf{??}"
DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
BRANCH_TAGS = (("src", "Src"), ("def", "Def"), ("orphan", "Orphan"))
HORIZON_TAGS = (("1", "One"), ("3", "Three"), ("12", "Twelve"))
SYSTEM_TAGS = (("holdmem_D1B1_source", "Hold"), ("zero", "Zero"), ("rag", "Rag"))


def get(d: Any, *path: Any) -> Any:
    """Nested `dict.get` that also tries the string form of a key (JSON keys are strings)."""
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p, d.get(str(p)))
    return d


def camel(key: str) -> str:
    """A contrast or item-type key as macro-name letters: `B_effect_update_D1_cited` ->
    `BEffectUpdateDoneCited` (digits are spelled, as in the existing macros' `Dzero`, `Done`)."""
    out = []
    for token in re.split(r"[_.]", key):
        token = re.sub(r"\d", lambda m: DIGIT_WORDS[int(m.group())], token)
        out.append(token[:1].upper() + token[1:])
    return "".join(out)


def model_slug(model: str) -> str:
    """`meta-llama/Llama-3.1-8B-Instruct` -> `meta-llama_llama-3_1-8b-instruct`: the spelling of
    the stage directories `runs/stage4_m2_<slug>` and of the `cross_<slug>` E2 tiers."""
    return model.lower().replace("/", "_").replace(".", "_")


class Family(NamedTuple):
    key: str  # results/sources.json group suffix
    tag: str  # macro-name tag
    model: str
    label: str  # as printed in tables


FAMILIES = (
    Family("llama", "Llama", "meta-llama/Llama-3.1-8B-Instruct", "Llama-3.1-8B"),
    Family("gemma", "Gemma", "google/gemma-4-12B-it", "gemma-4-12B"),
    Family("gptoss", "GptOss", "openai/gpt-oss-20b", "gpt-oss-20b"),
)


def family_of(model: str) -> Family | None:
    """The stage-4 family a model name belongs to (by its slug), or None."""
    slug = model_slug(model)
    for f in FAMILIES:
        if slug == model_slug(f.model):
            return f
    return None


class Tier(NamedTuple):
    key: str  # E2 tier key; cross-family tiers are `cross_<slug>` (matched by `marker`)
    tag: str
    label: str
    marker: str = ""


TIERS = (
    Tier("same", "Same", "same tier"),
    Tier("upgrade", "Up", "upgrade (27B)"),
    Tier("downgrade", "Down", "downgrade"),
    Tier("cross", "Llama", "Llama-3.1-8B", "llama"),
    Tier("cross", "Gemma", "gemma-4-12B", "gemma"),
    Tier("cross", "GptOss", "gpt-oss-20b", "gpt-oss"),
    Tier("cross", "Ministral", "Ministral-3-14B", "ministral"),
    Tier("cross", "QwenPrev", "Qwen3-8B", "qwen"),
)
SAME_FAMILY_TIERS = tuple(t for t in TIERS if not t.marker)


def tier_keys(e2: dict[str, Any] | None) -> list[str]:
    """Every tier an E2 analysis holds."""
    e2 = e2 or {}
    keys = set(e2.get("calibration") or {}) | set(e2.get("t1") or {})
    return sorted(keys | set(e2.get("m1b_d_protects_switch") or {}))


def tier_key(e2: dict[str, Any] | None, tier: Tier) -> str | None:
    """The key under which `e2` holds `tier`, or None when that switch was not staged."""
    keys = tier_keys(e2)
    if not tier.marker:
        return tier.key if tier.key in keys else None
    for k in keys:
        if k.startswith("cross_") and tier.marker in model_slug(k):
            return k
    return None


def read_json(path: Path) -> dict[str, Any] | None:
    return dict(json.loads(path.read_text())) if path.exists() else None


def stage_json(root: Path, sources: dict[str, str], group: str, name: str) -> dict[str, Any] | None:
    """`<stage directory of group>/analysis/<name>`, or None when the stage is not there."""
    return read_json(root / sources.get(group, "__missing__") / "analysis" / name)


# ------------------------------------------------------------------------------- block R views
def _have(d: dict[str, Any] | None, *path: str) -> bool:
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return False
        d = d[p]
    return True


def replication_view(orig: Mapping[str, Any], rep: Mapping[str, Any]) -> dict[str, Any]:
    """Stage 2 (`orig`) against R (`rep`), each a dict with the analyses e1, e2, e3 (None when not
    staged): the contrast rows of `analysis.replication`, T1 / T3 / T4 in both, and the three
    design quantities of `analysis.gate` (binding's effect pooled over D, D's effect on missed
    invalidation, writer agreement) on R. Whatever needs a missing stage is left out of the
    view; nothing is estimated here."""
    oe1, oe2, oe3 = orig.get("e1"), orig.get("e2"), orig.get("e3")
    re1, re2, re3 = rep.get("e1"), rep.get("e2"), rep.get("e3")
    view: dict[str, Any] = {
        "rows": {},
        "t1": {},
        "t3": {},
        "t4": {},
        "cv": {},
        "paths": {},
        "n_worlds": {"orig": (oe1 or {}).get("n_worlds"), "rep": (re1 or {}).get("n_worlds")},
    }
    if oe1 and re1:
        view["rows"] = {
            (r["branch"], r["contrast"]): r for r in replication.contrast_rows(oe1, re1)
        }
    members = replication.membership_rows(
        oe2 if oe2 and re2 else {"t1": {}},
        re2 or {"t1": {}},
        oe3 if oe3 and re3 else {"horizons": {}},
        re3 or {"horizons": {}},
    )
    view["t1"] = {m["unit"]: m for m in members if m["test"] == "T1"}
    view["t3"] = {m["unit"]: m for m in members if m["test"] == "T3"}
    if oe3 and re3:
        for h, per in oe3["horizons"].items():
            for s, o in per.items():
                r = (re3["horizons"].get(h) or {}).get(s)
                if isinstance(o, dict) and isinstance(r, dict) and "contract_valid" in o:
                    view["cv"][f"{h}m:{s}"] = {
                        "orig": o["contract_valid"],
                        "rep": r.get("contract_valid"),
                    }
                if (
                    isinstance(o, dict)
                    and isinstance(r, dict)
                    and "T4_pass" in o
                    and "T4_pass" in r
                ):
                    view["t4"][f"{h}m:{s}"] = {
                        "orig": o["T4_pass"],
                        "rep": r["T4_pass"],
                        "ratio": r["serve_ratio_to_full_reread"],
                    }
    paths = view["paths"]
    if _have(re1, "contrasts", "src", "B_effect_update_pooled_source"):
        paths["path1"] = gate.path1(re1 or {})
    if re1 and re1.get("contrasts"):
        paths["path2"] = gate.path2(re1)
    if _have(re2, "path3"):
        paths["path3"] = gate.path3(re2 or {})
    return view


def pooled_view(pool: Mapping[str, Any]) -> dict[str, Any]:
    """The same design quantities, and E1 contrasts, T1 / T3 for the forty pooled worlds."""
    e1, e2, e3 = pool.get("e1"), pool.get("e2"), pool.get("e3")
    view = replication_view({}, {"e1": e1, "e2": e2, "e3": e3})
    return {
        "contrasts": (e1 or {}).get("contrasts") or {},
        "paths": view["paths"],
        "t1": {t: v for t, v in ((e2 or {}).get("t1") or {}).items()},
        "t3": {
            f"{h}m:{s}": v
            for h, per in ((e3 or {}).get("horizons") or {}).items()
            for s, v in per.items()
            if isinstance(v, dict) and "T3_pass" in v
        },
        "n_worlds": (e1 or {}).get("n_worlds"),
    }


def multiplicity_view(
    res: Mapping[str, Any] | None, sources: Mapping[str, str]
) -> dict[str, dict[str, Any]]:
    """The multiplicity check (`analysis.multiplicity`) for the two stages the paper
    compares, "orig" (stage 2, source group `e1`) and "rep" (block R, `rep_e1`): each stage's
    `summary` (counts of the family, of intervals excluding 0, of Holm and BH survivors). A stage
    the result does not hold, or a group that is not in `sources`, is left out."""
    stages = (res or {}).get("stages") or {}
    out: dict[str, dict[str, Any]] = {}
    for tag, group in (("orig", "e1"), ("rep", "rep_e1")):
        summary = (stages.get(Path(sources.get(group, "")).name) or {}).get("summary")
        if summary:
            out[tag] = summary
    return out


# ----------------------------------------------------------------------- blocks M2, M3 and N
def load_families(root: Path, stage: str) -> dict[str, Any] | None:
    """Checks (i)-(v) of `families` for one M2 stage: `analysis/families.json` (written by
    `analysis.stage m2`, or by `python -m memshift.analysis.families`), else computed from the
    stage's E1 metrics and item table when both are there, else None."""
    d = root / stage / "analysis"
    saved = read_json(d / "families.json")
    if saved is not None:
        for v in saved.values():
            if isinstance(v, dict) and "checks" in v:
                return v
        return None
    metrics = read_json(d / "e1_metrics.json")
    if metrics is None or not (d / "e1_item_table.jsonl").exists():
        return None
    from memshift.analysis import families
    from memshift.experiments.common import read_jsonl

    return families.family_checks(metrics, read_jsonl(d / "e1_item_table.jsonl"))


def load_readers(m3: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """Family key -> E7-style reader swap (`analysis.e7.swap`) from the M3 stage's
    `theta_r_by_reader` (reader model as recorded in each run's manifest)."""
    out: dict[str, dict[str, Any]] = {}
    for model, swap in ((m3 or {}).get("theta_r_by_reader") or {}).items():
        fam = family_of(model)
        if fam is not None:
            out[fam.key] = swap
    return out


def merge_costs(*analyses: dict[str, Any] | None) -> dict[str, Any]:
    """System -> pooled native ingestion cost (`analysis.e5_cost`); later analyses win."""
    out: dict[str, Any] = {}
    for a in analyses:
        out.update((a or {}).get("by_system") or {})
    return out


def load_stage4(root: Path, sources: dict[str, str]) -> dict[str, Any]:
    """Every stage-4 input of the generators, None where a stage is not staged."""

    def triple(prefix: str) -> dict[str, dict[str, Any] | None]:
        return {
            f"e{n}": stage_json(root, sources, f"{prefix}e{n}", f"e{n}_metrics.json")
            for n in (1, 2, 3)
        }

    return {
        "orig": triple(""),
        "rep": triple("rep_"),
        "pool": triple("pool_"),
        "c": stage_json(root, sources, "c", "e1_metrics.json"),
        "m1": stage_json(root, sources, "m1", "e2_metrics.json"),
        "m2": {
            f.key: (
                load_families(root, sources[f"m2_{f.key}"]) if f"m2_{f.key}" in sources else None
            )
            for f in FAMILIES
        },
        "m3": load_readers(stage_json(root, sources, "m3", "e7_metrics.json")),
        "n": stage_json(root, sources, "n", "e5_natives.json"),
        "n3": stage_json(root, sources, "e5", "e5_natives.json"),
        "ncost": merge_costs(
            stage_json(root, sources, "e5cost", "e5_cost.json"),
            stage_json(root, sources, "ncost", "e5_cost.json"),
        ),
        "rendered": stage_json(root, sources, "l", "rendered.json"),
        "rank": stage_json(root, sources, "rank", "rank_corr.json"),
        "mult": multiplicity_view(stage_json(root, sources, "mult", "multiplicity.json"), sources),
    }


# ------------------------------------------------------------------- blocks C, M2 and M3 helpers
ITEM_TYPES = (
    "appointment",
    "contact.city",
    "contact.employer",
    "kappa.active_project",
    "kappa.close_friend",
    "kappa.important_contact",
    "kappa.overdue_task",
    "kappa.vip_sender",
    "kappa.working_hours",
    "series",
    "task_due",
    "u.commute",
    "u.diet",
    "u.employer",
    "u.home_city",
    "u.manager",
    "u.timezone",
)

TYPE_GROUPS = (  # item types of block C's program statistics, by kind of item
    ("Definition", "definition items", ("kappa.",)),
    ("Contact", "contact attributes", ("contact.",)),
    ("User", "user attributes", ("u.",)),
    ("Temporal", "series, appointments, task due dates", ("series", "appointment", "task_due")),
)


def in_group(item_type: str, prefixes: tuple[str, ...]) -> bool:
    return item_type.startswith(prefixes)


def pooled_rate(stats: dict[str, Any], types: list[str], metric: str) -> float | None:
    """Item-weighted mean of a per-type rate (each per-type estimate is itself an item mean)."""
    num_, den = 0.0, 0
    for t in types:
        r = (stats.get(t) or {}).get(metric)
        if r:
            num_ += r["est"] * r["n_items"]
            den += r["n_items"]
    return num_ / den if den else None


FAMILY_ESTIMATES = (
    ("B_update_src", "BUpdateSrc"),
    ("D_repair_src", "DRepairSrc"),
    ("D_repair_def", "DRepairDef"),
    ("BxD_def", "BxDDef"),
    ("BxD_src", "BxDSrc"),
    ("zero_minus_hm11_def", "ZeroMinusHoldDef"),
)


def label_text(d: int | None, b: int | None) -> str:
    return f"({d},{'--' if b is None else b})"


# ------------------------------------------------------------------------------- native systems
class Native(NamedTuple):
    key: str  # system name in the E5 analysis
    tag: str  # macro-name tag
    label: str


NATIVES = (
    Native("mem0", "Mem", "Mem0"),
    Native("amem", "Amem", "A-Mem"),
    Native("memos", "Memos", "MemOS"),
    Native("lightrag", "LightRag", "LightRAG"),
    Native("hipporag", "HippoRag", "HippoRAG 2"),
    Native("cognee", "Cognee", "Cognee"),
)
NATIVE_METRICS = (("Update", "Update"), ("Stability", "Stability"))


def native_results(inp: dict[str, Any]) -> dict[str, Any] | None:
    """E5 analysis behind the native-systems table and figure: the stage-4 analysis when it is
    staged, else the stage-3 analysis (Mem0 only), so the Mem0 row stays and the other systems
    show as missing."""
    return inp.get("n") or inp.get("n3")

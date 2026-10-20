"""How much of a world block L (the LLM-rendered worlds) rewrites: the newsletter digests are
almost all of the body text and stay templates. This module counts it on the ten L test worlds,
from the template worlds alone (no model, no rendered text): the generator is seeded, and the
renderer's selection rule is `llm_render.is_background` and `llm_render.RENDER_SOURCES`.

Counting unit. Every count is in approximate tokens (`textutil.approx_tokens`: word pieces and
punctuation, the function the paper's world statistics and the renderer's length budget use), of
the template text:

  body tokens       the `body` of every e-mail, chat message and note event that is not a
                    deletion: the text the renderer is asked to rewrite, digests included
  digest tokens     the body tokens of the newsletter digests (generated background mail that
                    carries no gold fact), which the renderer does not send to the model and
                    leaves as template
  rewritten tokens  the body tokens of the other events: the bodies the renderer rewrites (every
                    attempted body was rewritten in the test run, kept-as-template share 0; the
                    event count matches the run's `attempted` per world and branch)
  world tokens      `payload["text"]` of every event that is not a deletion, in every source
                    (headers of rewritten events, structured records and digests included)

Reported per world and branch (seeds 1000-1009, branches src, def and orphan: the branches the L
run rendered, each with its own copy of the natural events) and pooled over them:

  digest_share      digest tokens / body tokens
  rewritten_share   rewritten tokens / world tokens (the share of the world's text block L
                    rewrites); `rewritten_text_share` counts the whole text of the rewritten
                    events (headers, which stay, included) for reference

    python -m memshift.analysis.l_tokens --seeds 1000-1009 --branches src,def,orphan \\
        --out runs/stage4_l_tokens

Macro names (letters only; a stage that is not staged renders MISSING):

  RendDigestTokenShare, RendRewrittenTokenShare       pooled shares, in percent
  RendDigestTokenShareMin / Max, RendRewrittenTokenShareMin / Max   range over world x branch
  RendTokenWorldBranches                              number of world x branch worlds counted
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Any

from memshift.analysis.paper_data import MISSING, stage_json
from memshift.analysis.paper_numbers import _macro
from memshift.analysis.stage import write_stage_manifest
from memshift.experiments.common import world_params
from memshift.experiments.e1 import parse_seeds
from memshift.gen.llm_render import RENDER_SOURCES, is_background
from memshift.gen.model import World
from memshift.gen.shifts import generate_branch
from memshift.textutil import approx_tokens

COUNTS = (
    "events",
    "world_tokens",
    "body_tokens",
    "digest_events",
    "digest_tokens",
    "rewritten_events",
    "rewritten_tokens",
    "rewritten_text_tokens",
)


def count_world(world: World) -> dict[str, int]:
    """The counts of one world (see the module docstring for the unit)."""
    c = dict.fromkeys(COUNTS, 0)
    for e in world.events:
        if e.op == "delete":
            continue
        c["events"] += 1
        c["world_tokens"] += approx_tokens(e.payload["text"])
        if e.source not in RENDER_SOURCES:
            continue
        body = approx_tokens(str(e.payload["body"]))
        c["body_tokens"] += body
        if is_background(e):
            c["digest_events"] += 1
            c["digest_tokens"] += body
        else:
            c["rewritten_events"] += 1
            c["rewritten_tokens"] += body
            c["rewritten_text_tokens"] += approx_tokens(e.payload["text"])
    return c


def _shares(c: Mapping[str, int]) -> dict[str, float]:
    return {
        "digest_share": c["digest_tokens"] / c["body_tokens"],
        "rewritten_share": c["rewritten_tokens"] / c["world_tokens"],
        "rewritten_text_share": c["rewritten_text_tokens"] / c["world_tokens"],
    }


def measure(seeds: Sequence[int], branches: Sequence[str], months: int = 12) -> dict[str, Any]:
    worlds: list[dict[str, Any]] = []
    for seed in seeds:
        for branch in branches:
            counts = count_world(generate_branch(seed, branch, world_params(months)))
            worlds.append({"seed": seed, "branch": branch, **counts, **_shares(counts)})
    pooled = {k: sum(w[k] for w in worlds) for k in COUNTS}
    return {
        "seeds": list(seeds),
        "branches": list(branches),
        "months": months,
        "unit": "approx_tokens of the template text; see analysis.l_tokens",
        "world_branches": len(worlds),
        "pooled": {**pooled, **_shares(pooled)},
        "by_world_branch": worlds,
    }


# ------------------------------------------------------------------------------------- macros
def load(root: Path, sources: Mapping[str, str]) -> dict[str, Any] | None:
    """The staged count (None where it is not staged)."""
    return stage_json(root, dict(sources), "ltokens", "token_share.json")


def _pct(x: float | None) -> str:
    if x is None:
        return MISSING
    return str((Decimal(repr(x)) * 100).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)) + "\\%"


def numbers(view: Mapping[str, Any] | None) -> dict[str, str]:
    """The `Rend*TokenShare` macros, from `load`."""
    pooled = (view or {}).get("pooled") or {}
    per = (view or {}).get("by_world_branch") or []
    out: dict[str, str] = {}
    for key, name in (("digest_share", "Digest"), ("rewritten_share", "Rewritten")):
        out[f"Rend{name}TokenShare"] = _pct(pooled.get(key))
        out[f"Rend{name}TokenShareMin"] = _pct(min((w[key] for w in per), default=None))
        out[f"Rend{name}TokenShareMax"] = _pct(max((w[key] for w in per), default=None))
    out["RendTokenWorldBranches"] = str(len(per)) if per else MISSING
    for name, value in out.items():
        _macro(name, value)  # letters-only names
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--branches", default="src,def,orphan")
    ap.add_argument("--months", type=int, default=12)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    result = measure(parse_seeds(a.seeds), tuple(a.branches.split(",")), a.months)
    (a.out / "analysis").mkdir(parents=True, exist_ok=True)
    (a.out / "analysis" / "token_share.json").write_text(json.dumps(result, indent=1))
    command = (
        f"python -m memshift.analysis.l_tokens --seeds {a.seeds} --branches {a.branches}"
        f" --months {a.months}"
    )
    extra = {"command": command, "seeds": a.seeds, "branches": result["branches"]}
    write_stage_manifest(a.out, "l_tokens", [], extra)
    print(json.dumps(result["pooled"]))

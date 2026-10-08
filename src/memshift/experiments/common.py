"""Shared plumbing for experiment drivers: endpoints, clients, run directories, answering probes.

Endpoints come from MEMSHIFT_ENDPOINTS (JSON: model id -> OpenAI base URL) or the
MEMSHIFT_WRITER_* / MEMSHIFT_READER_* / MEMSHIFT_EMBED_* variables of `.env.example`.
"""

from __future__ import annotations

import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memshift.gen.model import fmt_ts
from memshift.holdmem.system import ServeResult
from memshift.llm import LLMClient
from memshift.probes import Probe
from memshift.reader import context_hash, parse_answer, read_request
from memshift.runinfo import REPO, cache_tags
from memshift.scoring.normalize import contract_valid

MAIN_MODEL = "Qwen/Qwen3.5-9B"
EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"


def endpoints() -> dict[str, str]:
    raw = os.environ.get("MEMSHIFT_ENDPOINTS")
    if raw:
        return dict(json.loads(raw))
    out = {}
    for role in ("WRITER", "READER", "EMBED"):
        model = os.environ.get(f"MEMSHIFT_{role}_MODEL")
        url = os.environ.get(f"MEMSHIFT_{role}_BASE_URL")
        if model and url:
            out[model] = url
    return out


def slug(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", model)


def cache_path(model: str) -> Path:
    tag = cache_tags().get(model)
    name = f"{slug(model)}@{slug(tag)}" if tag else slug(model)
    return REPO / "cache" / "llm" / f"{name}.sqlite"


def client(model: str, run_dir: Path | None, workers: int = 48, **kw: Any) -> LLMClient:
    eps = endpoints()
    if model not in eps:
        raise SystemExit(f"no endpoint configured for {model}; set MEMSHIFT_ENDPOINTS")
    return LLMClient(
        eps[model],
        model,
        cache_path=cache_path(model),
        max_workers=workers,
        log_path=(run_dir / "calls.jsonl") if run_dir else None,
        **kw,
    )


def new_run_dir(name: str, root: Path | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    d = (root or REPO / "runs") / f"{name}_{stamp}"
    d.mkdir(parents=True, exist_ok=False)
    return d


def world_params(months: int) -> dict[str, Any] | None:
    """Generator parameters for a shortened world (E5 natives and their white-box references):
    `months` long, forked one month before the end, token target scaled with length. None for the
    default twelve-month world."""
    if months == 12:
        return None
    return {"months": months, "fork_month": months - 1, "target_tokens": 2_000_000 * months // 12}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]], mode: str = "w") -> None:
    with path.open(mode, encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@dataclass
class Answer:
    system: str
    probe_id: str
    world_id: str
    branch: str
    fact_id: str
    vtype: str
    as_of: str
    agent_id: str
    status: str
    value: Any
    gold_status: str
    gold_value: Any
    contract_valid: bool
    context_sha256: str
    n_blocks: int
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    refs: list[Any] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def answer_probes(
    system: str,
    serve: Callable[[Probe], ServeResult],
    probes: list[Probe],
    reader: LLMClient,
    tags: dict[str, Any] | None = None,
) -> tuple[list[Answer], list[ServeResult]]:
    served = [serve(p) for p in probes]
    todo = [i for i, s in enumerate(served) if not s.blocked]
    results = reader.map([read_request(probes[i].question_text, served[i].blocks) for i in todo])
    by_idx = dict(zip(todo, results, strict=True))
    answers = []
    for i, (p, s) in enumerate(zip(probes, served, strict=True)):
        if s.blocked:
            status, value, pt, ct, lat = "blocked", None, 0, 0, 0.0
        else:
            res = by_idx[i]
            status, value = parse_answer(res)
            pt, ct, lat = res.prompt_tokens, res.completion_tokens, res.latency_s
        answers.append(
            Answer(
                system=system,
                probe_id=p.probe_id,
                world_id=p.world_id,
                branch=p.branch,
                fact_id=p.fact_id,
                vtype=p.vtype,
                as_of=fmt_ts(p.as_of),
                agent_id=p.agent_id,
                status=status,
                value=value,
                gold_status=p.gold_status,
                gold_value=p.gold_value,
                contract_valid=contract_valid(p.vtype, status, value, p.gold_status, p.gold_value),
                context_sha256=context_hash(s.blocks),
                n_blocks=len(s.blocks),
                prompt_tokens=pt,
                completion_tokens=ct,
                latency_s=round(lat, 3),
                refs=[list(r) for r in s.refs[:20]],
                tags={**p.tags, **(tags or {})},
            )
        )
    return answers, served


class Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()

    def lap(self) -> float:
        now = time.perf_counter()
        dt, self.t0 = now - self.t0, now
        return dt

"""Send collected Claude requests through Anthropic's Message Batches API (half the online price)
and write the results into the response cache (the Claude reader; `memshift.llm`, batch
collection).

    python -m memshift.claude_batch submit --state S.json COLLECTED.jsonl ...
    python -m memshift.claude_batch collect --state S.json [--wait]
    python -m memshift.claude_batch status --state S.json

`submit` reads the files a collecting run wrote (`MEMSHIFT_ANTHROPIC_COLLECT`), drops duplicates and
requests already in the cache, and creates batches of at most `MAX_REQUESTS` requests and
`MAX_BYTES` bytes; each request's `custom_id` is its cache key. `collect` fetches the results of
ended batches and puts every succeeded message into the model's cache with the same row as an
online response (`anthropic_message_result`), plus the batch id. Errored, canceled and expired
requests stay out of the cache, so the rerun sends them online. The state file keeps the batches,
their counts and their token usage, which the spend ledger reads (`usd` at the batch price).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

import httpx

from memshift.llm import (
    ANTHROPIC_URL,
    ANTHROPIC_VERSION,
    ResponseCache,
    anthropic_key,
    anthropic_message_result,
)

MAX_REQUESTS = 20_000  # the API allows 100,000 requests and 256 MB per batch
MAX_BYTES = 150_000_000
# US$ per million input / output tokens at the batch price (half of the list price)
BATCH_PRICE = {"claude-sonnet-5-5": (1.0, 5.0), "claude-opus-5-5": (2.0, 10.0)}


def _headers() -> dict[str, str]:
    return {
        "x-api-key": anthropic_key(),
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }


def _cache(model: str, cache_dir: Path | None) -> ResponseCache:
    from memshift.experiments.common import cache_path

    path = cache_path(model)
    return ResponseCache(cache_dir / path.name if cache_dir else path)


def read_collected(paths: list[Path]) -> dict[str, dict[str, Any]]:
    """key -> {model, params}, first occurrence wins (identical keys are identical requests)."""
    out: dict[str, dict[str, Any]] = {}
    for p in paths:
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out.setdefault(r["key"], {"model": r["model"], "params": r["params"]})
    return out


def chunks(
    reqs: list[dict[str, Any]], max_requests: int = MAX_REQUESTS, max_bytes: int = MAX_BYTES
) -> list[list[dict[str, Any]]]:
    out: list[list[dict[str, Any]]] = []
    cur: list[dict[str, Any]] = []
    size = 0
    for r in reqs:
        n = len(json.dumps(r))
        if cur and (len(cur) >= max_requests or size + n > max_bytes):
            out.append(cur)
            cur, size = [], 0
        cur.append(r)
        size += n
    if cur:
        out.append(cur)
    return out


def submit(
    paths: list[Path],
    state_path: Path,
    http: httpx.Client | None = None,
    cache_dir: Path | None = None,
) -> dict[str, Any]:
    http = http or httpx.Client(timeout=600)
    collected = read_collected(paths)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    pending = {k for b in state["batches"] if not b.get("collected") for k in b["keys"]}
    by_model: dict[str, list[dict[str, Any]]] = {}
    skipped: Counter[str] = Counter()
    caches: dict[str, ResponseCache] = {}
    for key, r in sorted(collected.items()):
        cache = caches.setdefault(r["model"], _cache(r["model"], cache_dir))
        if cache.get(key) is not None:
            skipped["cached"] += 1
        elif key in pending:
            skipped["in an open batch"] += 1
        else:
            by_model.setdefault(r["model"], []).append({"custom_id": key, "params": r["params"]})
    for model, reqs in by_model.items():
        for part in chunks(reqs):
            resp = http.post(
                f"{ANTHROPIC_URL}/v1/messages/batches", json={"requests": part}, headers=_headers()
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"batch create failed: HTTP {resp.status_code} {resp.text[:500]}"
                )
            data = resp.json()
            state["batches"].append(
                {
                    "id": data["id"],
                    "model": model,
                    "keys": [q["custom_id"] for q in part],
                    "created": data.get("created_at"),
                    "collected": False,
                }
            )
            state_path.write_text(json.dumps(state, indent=1))
            print(f"created {data['id']}: {len(part)} {model} requests")
    print(f"skipped: {dict(skipped)}")
    return state


def collect(
    state_path: Path,
    http: httpx.Client | None = None,
    cache_dir: Path | None = None,
    wait: bool = False,
    poll_s: float = 60.0,
) -> dict[str, Any]:
    http = http or httpx.Client(timeout=600)
    state: dict[str, Any] = json.loads(state_path.read_text())
    while True:
        open_batches = 0
        for b in state["batches"]:
            if b.get("collected"):
                continue
            info = http.get(f"{ANTHROPIC_URL}/v1/messages/batches/{b['id']}", headers=_headers())
            info.raise_for_status()
            data = info.json()
            b["request_counts"] = data.get("request_counts")
            if data.get("processing_status") != "ended":
                open_batches += 1
                continue
            res = http.get(data["results_url"], headers=_headers())
            res.raise_for_status()
            cache = _cache(b["model"], cache_dir)
            outcome: Counter[str] = Counter()
            tokens = [0, 0]
            for line in res.text.splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                kind = row["result"]["type"]
                outcome[kind] += 1
                if kind != "succeeded":
                    continue
                key = row["custom_id"]
                llm_res, meta = anthropic_message_result(row["result"]["message"], key, 0.0)
                meta["batch_id"] = b["id"]
                cached = {k: v for k, v in asdict(llm_res).items() if k != "key"}
                cached["api"] = meta
                cache.put(key, b["model"], cached)
                tokens[0] += llm_res.prompt_tokens
                tokens[1] += llm_res.completion_tokens
            pi, po = BATCH_PRICE[b["model"]]
            b.update(
                collected=True,
                outcome=dict(outcome),
                input_tokens=tokens[0],
                output_tokens=tokens[1],
                usd=round((tokens[0] * pi + tokens[1] * po) / 1e6, 6),
            )
            state_path.write_text(json.dumps(state, indent=1))
            print(f"collected {b['id']}: {dict(outcome)}, US${b['usd']:.2f}")
        if not (wait and open_batches):
            break
        print(f"{open_batches} batch(es) still processing", flush=True)
        time.sleep(poll_s)
    return state


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--state", type=Path, required=True)
    s.add_argument("files", type=Path, nargs="+")
    c = sub.add_parser("collect")
    c.add_argument("--state", type=Path, required=True)
    c.add_argument("--wait", action="store_true")
    st = sub.add_parser("status")
    st.add_argument("--state", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "submit":
        submit(a.files, a.state)
    elif a.cmd == "collect":
        collect(a.state, wait=a.wait)
    else:
        for b in json.loads(a.state.read_text())["batches"]:
            print(b["id"], b["model"], len(b["keys"]), b.get("request_counts"), b.get("usd"))


if __name__ == "__main__":
    main()

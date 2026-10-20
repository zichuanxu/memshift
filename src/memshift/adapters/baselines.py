"""Baselines that share the reader, template and 4k evidence budget with HoldMem.

zero system  raw source objects + BM25 + on-demand re-read; no derived store, no embeddings.
zero_Dm      the zero system plus a query-time executable view: the writer-authored program of the
             probed item (the D1s program, same cached request) runs on the live structured records
             at the probe's as-of time and its value, in the D=1 template, leads the evidence. No
             derived state is stored, so there is nothing to bind or invalidate.
oracle       the gold dependency objects of the probe (upper bound on evidence selection).
plain RAG    paragraph chunks of at most 200 tokens + Qwen3-Embedding-0.6B cosine retrieval.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from memshift.gen.model import World
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import render_fallback_sk
from memshift.holdmem.program import synthesise
from memshift.holdmem.system import (
    PROGRAM_ORIGIN,
    READER_BUDGET,
    READER_DOC_CAP,
    ServeResult,
    definition_text,
)
from memshift.llm import LLMClient
from memshift.probes import Probe
from memshift.retrieval import Doc, LexIndex, pack
from memshift.textutil import approx_tokens, sha256_text, truncate_tokens

CHUNK_TOKENS = 200
EMBED_QUERY_PREFIX = (
    "Instruct: Given a question about a person, retrieve passages that answer it\nQuery: "
)


def empty_cost() -> dict[str, float]:
    return {
        "ingest_tokens": 0,
        "maintain_tokens": 0,
        "serve_tokens": 0,
        "verify_cpu_s": 0.0,
        "storage_bytes": 0,
    }


def _granted_docs(world: World, log: SourceLog, probe: Probe) -> list[Doc]:
    granted = world.granted(probe.agent_id, probe.as_of)
    snap = log.snapshot(probe.as_of)
    return [
        Doc(o.object_id, o.text, o.updated, o.revision) for o in snap.all() if o.source in granted
    ]


def _result(chosen: list[tuple[Doc, str]]) -> ServeResult:
    refs = [(d.doc_id.split("::")[0], d.revision) for d, _ in chosen]
    return ServeResult([t for _, t in chosen], refs, [], False)


class ZeroSystem:
    name = "zero"

    def __init__(self, world: World, log: SourceLog):
        self.world, self.log = world, log
        self._index: dict[tuple[datetime, str], LexIndex] = {}
        self.cost = empty_cost()

    def ranked(self, probe: Probe) -> list[Doc]:
        key = (probe.as_of, probe.agent_id)
        if key not in self._index:
            if len(self._index) > 8:
                self._index.clear()
            self._index[key] = LexIndex(_granted_docs(self.world, self.log, probe))
        return [d for d, s in self._index[key].rank(probe.question_text) if s > 0]

    def serve(self, probe: Probe) -> ServeResult:
        return _result(pack(self.ranked(probe), READER_BUDGET, READER_DOC_CAP))


class ZeroDm:
    """zero_Dm (see the module docstring). The item's program is requested with the D1s request
    for the definition version in force at the probe's time, so it is a response-cache hit of the
    D1s cells; items whose program fails, or whose declared sources the asking agent may not read,
    are served as the zero system serves them. The view shares the 4k evidence budget: the BM25
    evidence is packed into what the view leaves."""

    name = "zero_Dm"

    def __init__(self, world: World, log: SourceLog, writer: LLMClient):
        self.world, self.log, self.writer = world, log, writer
        self.zero = ZeroSystem(world, log)
        self.cost = empty_cost()
        self._authored: set[tuple[str, str | None]] = set()
        self._lock = threading.Lock()

    def view(self, probe: Probe) -> tuple[str | None, dict[str, Any]]:
        """The view block of the probed item (None when there is none) and a diagnostic record."""
        slot = next((s for s in self.world.slots if s.slot_id == probe.fact_id), None)
        if slot is None or not set(slot.sources) <= self.world.granted(probe.agent_id, probe.as_of):
            return None, {"view": None}
        prog = synthesise(self.writer, self.world, self.log, slot, probe.as_of)
        authored = (
            slot.slot_id,
            self.world.kappa_version(slot.kappa_id, probe.as_of) if slot.kappa_id else None,
        )
        with self._lock:
            if authored not in self._authored:  # a program is written once per item and version
                self._authored.add(authored)
                self.cost["ingest_tokens"] += prog.prompt_tokens + prog.completion_tokens
            self.cost["verify_cpu_s"] += prog.run_s
        diag = {"view": {"ok": prog.ok, "value": prog.value, "key": prog.key, "error": prog.error}}
        if not prog.ok:
            return None, diag
        block = render_fallback_sk(
            slot,
            probe.as_of,
            prog.value,
            [],
            definition_text(self.world, slot, probe.as_of),
            origin=PROGRAM_ORIGIN,
        )
        return block, diag

    def serve(self, probe: Probe) -> ServeResult:
        block, diag = self.view(probe)
        budget = READER_BUDGET - (approx_tokens(block) + 8 if block else 0)
        res = _result(pack(self.zero.ranked(probe), budget, READER_DOC_CAP))
        blocks = [block, *res.blocks] if block else res.blocks
        return ServeResult(blocks, res.refs, [{"item_id": f"{probe.fact_id}#view", **diag}], False)


class Oracle:
    name = "oracle"

    def __init__(self, world: World, log: SourceLog):
        self.world, self.log = world, log
        self.cost = empty_cost()

    def serve(self, probe: Probe) -> ServeResult:
        snap = self.log.snapshot(probe.as_of)
        docs = [
            Doc(o.object_id, o.text, o.updated, o.revision)
            for oid, _ in probe.deps
            if (o := snap.get(oid)) is not None
        ]
        return _result(pack(docs, READER_BUDGET, READER_DOC_CAP))


def chunk(doc: Doc, size: int = CHUNK_TOKENS) -> list[Doc]:
    """Split on paragraph boundaries into pieces of at most `size` approx tokens."""
    header = doc.text.split("\n", 1)[0]
    pieces: list[str] = []
    cur = ""
    for p in (p for p in doc.text.split("\n\n") if p.strip()):
        cand = f"{cur}\n\n{p}" if cur else p
        if approx_tokens(cand) <= size:
            cur = cand
            continue
        if cur:
            pieces.append(cur)
        cur = truncate_tokens(p, size)
    if cur:
        pieces.append(cur)
    return [
        Doc(
            f"{doc.doc_id}::{i}",
            text if i == 0 else f"{header} (cont.)\n{text}",
            doc.ts,
            doc.revision,
        )
        for i, text in enumerate(pieces)
    ]


class EmbedClient:
    """OpenAI-compatible embeddings with a text-hash cache."""

    def __init__(self, base_url: str, model: str, cache_path: Path, batch: int = 64):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.batch = batch
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(cache_path), check_same_thread=False, timeout=60)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec TEXT)")
        self._lock = threading.Lock()
        self._http = httpx.Client(timeout=300)
        self.tokens = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        keys = [sha256_text(self.model + "\x00" + t) for t in texts]
        found: dict[str, list[float]] = {}
        with self._lock:
            for k in set(keys):
                row = self._db.execute("SELECT vec FROM emb WHERE key=?", (k,)).fetchone()
                if row:
                    found[k] = json.loads(row[0])
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts, strict=True):
            if k not in found:
                todo.setdefault(k, t)
        items = list(todo.items())
        for i in range(0, len(items), self.batch):
            part = items[i : i + self.batch]
            r = self._http.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": [t for _, t in part]},
            )
            r.raise_for_status()
            data = r.json()
            self.tokens += int((data.get("usage") or {}).get("prompt_tokens", 0))
            with self._lock:
                for (k, _), row in zip(part, data["data"], strict=True):
                    found[k] = row["embedding"]
                    self._db.execute(
                        "INSERT OR REPLACE INTO emb VALUES (?,?)", (k, json.dumps(row["embedding"]))
                    )
                self._db.commit()
        mat = np.array([found[k] for k in keys], dtype=np.float32)
        return mat / np.maximum(np.linalg.norm(mat, axis=1, keepdims=True), 1e-8)


class PlainRAG:
    name = "rag"

    def __init__(self, world: World, log: SourceLog, embed: EmbedClient):
        self.world, self.log, self.embed = world, log, embed
        self._cache: dict[tuple[datetime, str], tuple[list[Doc], np.ndarray]] = {}
        self.cost = empty_cost()

    def _index(self, probe: Probe) -> tuple[list[Doc], np.ndarray]:
        key = (probe.as_of, probe.agent_id)
        if key not in self._cache:
            if len(self._cache) > 4:
                self._cache.clear()
            chunks = [c for d in _granted_docs(self.world, self.log, probe) for c in chunk(d)]
            before = self.embed.tokens
            mat = self.embed.embed([c.text for c in chunks])
            self.cost["ingest_tokens"] += self.embed.tokens - before
            self.cost["storage_bytes"] = int(mat.nbytes)
            self._cache[key] = (chunks, mat)
        return self._cache[key]

    def serve(self, probe: Probe) -> ServeResult:
        chunks, mat = self._index(probe)
        q = self.embed.embed([EMBED_QUERY_PREFIX + probe.question_text])[0]
        order = np.argsort(-(mat @ q), kind="stable")
        return _result(pack([chunks[i] for i in order[:200]], READER_BUDGET, READER_DOC_CAP))

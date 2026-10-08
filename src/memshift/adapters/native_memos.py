"""MemOS (MemTensor, `MemoryOS==2.0.33`, arXiv 2507.03724) behind the MemShift interface (plan
§4.5, prereg v2 §3), on the same `NativeSystem` driver as Mem0 and Graphiti. The library is
imported lazily and is not part of `uv sync --extra native`: MemoryOS requires `openai<2`, so it
runs from its own pinned layer (`docker/native/memos.txt`).

Configuration. MemOS's recommended textual memory is `tree_text` (README "Self-Host": Neo4j +
Qdrant), which needs a graph server: the graph stores are Neo4j, Neo4j-community, PolarDB and
Postgres (graph_dbs/factory.py:14-19, configs/graph_db.py:17-244); none is embedded, and the
reorganiser (off by default, configs/memory.py:175), the memory-version hook and the conflict
merge that update stored memories all read that graph (tree_text_memory/organize/manager.py:
89-113, reorganizer.py:96-106, 192-206, handler.py:32-149; mem_reader/multi_modal_struct.py:
660-690). The documented server-free memory is `general_text` (docs "Your first memory";
`GeneralTextMemory`, memories/textual/general.py:27-36): extraction LLM + embedder + one vector
store, whose Qdrant falls back to an embedded local path when no host / port / url is given
(configs/vec_db.py:33-41, vec_dbs/qdrant.py:57). This adapter runs that configuration with
MemOS's memory reader in front of it:

  `SimpleStructMemReader.get_memory(messages, type="chat", info, mode="fine")` (MemReader,
  mem_reader/simple_struct.py:479-532) -> `GeneralTextMemory.add(items)` (general.py:81-104).

That is the hand-off `MOS.add` performs for `tree_text` (mem_os/core.py:757-766) with the store
swapped; for a non-tree backend `MOS.add` stores each message verbatim without the reader
(core.py:747-756), which would not be MemOS's own extraction. `user_id` / `session_id` /
`chat_time` / `info` are the reader's documented inputs (simple_struct.py:220-256, 784-869):
each source event is one single-message chat scene, `session_id = object_id`, `chat_time` in the
format of the extraction prompt's example, and our event metadata rides in `info`, which the
reader stores in `metadata.info` next to `metadata.sources` (the message text and time). A
single message is never split (`_iter_chat_windows`, simple_struct.py:315-357), so fine mode is
one LLM call per event (simple_struct.py:405-409) plus one embedding call per extracted memory
(the reader embeds every value, simple_struct.py:247) and one batched embedding call in `add`
(general.py:90). The reader's embedding is dropped from the stored payload (the point vector
holds the same vector).

Behaviour that decides (D, B), all read from source: (1) `add` never looks at existing memories:
it embeds the new items and upserts points under fresh uuid4 ids (item.py:306, qdrant.py:276-296,
general.py:81-104), so a new revision of a record adds memories next to the old ones (append-only;
no dedup, merge or archive), and a deletion tombstone is just another message. (2) Each memory
keeps its source message and our `object_id` / `revision` (`metadata.sources`, `metadata.info`),
but nothing ever reads them back after ingestion, so deleting a source cannot retract a memory.
MemOS's own `update` / `delete` by memory id exist (general.py:106-119, 166-168) and are not
used: the contract forbids the benchmark from doing the system's invalidation for it. (3) Search
is dense retrieval over the vector store only (general.py:121-138): no reranker, BM25, graph hop
or answer generation, and the score is dropped, so hit scores here are rank-based.

MemOS swallows LLM errors, embedding errors and unparseable extraction output inside the reader
(`_safe_generate` / `_safe_parse`, simple_struct.py:258-272; the per-memory `except`,
:427-428) and, when the reply is empty, stores the raw window text as the one memory
(`_get_llm_response`, simple_struct.py:274-313). Transport and API errors are noted by wrapping
the LLM and embedding calls and handled by `on_error`; the unparseable-reply fallback is MemOS
behaviour, kept as it is and counted in `stats["extract_fallback"]` and the ledger.

Process hygiene: MemOS caches LLM / embedder / reader instances by config hash
(memos_tools/singleton.py:18-115), which would share one metered client between systems, so the
cache is cleared before each open. Importing it configures the root logger with an INFO file
handler under `MEMOS_BASE_PATH` (log.py:106-118, 263-296; settings.py:6) that records full
request bodies: the base path is placed in the repo cache and the `memos` / `httpx` loggers are
raised to WARNING. The embedder's request timeout defaults to 5 s (universal_api.py:122) and is
raised; so is the LLM client's 10-minute SDK default (to one hour, the model server is shared
and a timed-out request re-queues). `workers` > 1 (default 1, the contract's sequential order)
runs extractions concurrently, each on its own reader: `get_memory` reads no store, the adds
stay in event order, and all Qdrant access stays on the calling thread (embedded Qdrant is not
thread-safe: drive one instance from one thread).
"""

from __future__ import annotations

import logging
import os
import queue
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

from memshift.adapters.native import (
    DEFAULT_MAX_OUTPUT_TOKENS,
    Hit,
    MeteredOpenAI,
    NativeSystem,
    Usage,
    _api_key,
    event_key,
    event_metadata,
    event_text,
    mem0_hit,
)
from memshift.gen.model import Event
from memshift.runinfo import REPO

PACKAGE = "MemoryOS"
COLLECTION = "memshift"
EMBED_TIMEOUT_S = 300
LLM_TIMEOUT_S = 3600
SCROLL_PAGE = 256


# ---------------------------------------------------------------------- event conversion
def chat_time(e: Event) -> str:
    """Message time in the format of MemOS's extraction-prompt example ("June 26, 2025 at
    3:00 PM"), which its instructions for resolving relative dates refer to."""
    return f"{e.ts:%B %d, %Y at %I:%M %p}"


def chat_scene(e: Event) -> list[list[dict[str, str]]]:
    """One event as one single-message chat scene for `SimpleStructMemReader.get_memory`."""
    return [[{"role": "user", "content": event_text(e), "chat_time": chat_time(e)}]]


def reader_info(e: Event, user_id: str) -> dict[str, Any]:
    """The reader's `info`: `user_id` / `session_id` are required; every other key is stored in
    the memory's `metadata.info`."""
    return {"user_id": user_id, "session_id": e.object_id, **event_metadata(e)}


def payload_info(payload: dict[str, Any]) -> dict[str, Any]:
    md = payload.get("metadata") or {}
    info = md.get("info") if isinstance(md, dict) else None
    return dict(info) if isinstance(info, dict) else {}


def payload_key(payload: dict[str, Any]) -> str | None:
    info = payload_info(payload)
    if "object_id" in info and "revision" in info:
        return f"{info['object_id']}#{info['revision']}"
    return None


def keys_from_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    """Stored memories grouped by the `object_id#revision` of the event they came from."""
    found: dict[str, list[str]] = {}
    for p in payloads:
        key = payload_key(p)
        if key is not None:
            found.setdefault(key, []).append(str(p.get("id", "")))
    return found


def memos_hit(payload: dict[str, Any], score: float = 0.0) -> Hit:
    """One stored memory -> text block, in the same layout as a Mem0 memory (the event time comes
    from our metadata: ingestion replays simulated time, `created_at` is wall clock)."""
    row = {
        "id": payload.get("id", ""),
        "memory": payload.get("memory", ""),
        "metadata": payload_info(payload),
        "score": score,
    }
    return mem0_hit(row)


def export_row(point_id: str, payload: dict[str, Any], vector_dim: int) -> dict[str, Any]:
    return {
        "id": point_id,
        "key": payload_key(payload),
        "payload": payload,
        "vector_dim": vector_dim,
    }


# ---------------------------------------------------------------------- library plumbing
def _memos_env() -> None:
    """Must run before `import memos`: base path in the repo cache (logs), embedder timeout."""
    os.environ.setdefault("MEMOS_BASE_PATH", str(REPO / "cache" / "memos"))
    os.environ.setdefault("MOS_EMBEDDER_TIMEOUT", str(EMBED_TIMEOUT_S))
    Path(os.environ["MEMOS_BASE_PATH"]).mkdir(parents=True, exist_ok=True)


def _memos_version() -> str:
    try:
        return importlib_metadata.version(PACKAGE)
    except importlib_metadata.PackageNotFoundError:
        return "not installed"


def _meter(objs: Iterable[Any], usage: Usage, **kw: Any) -> None:
    """Wrap the OpenAI client inside each distinct MemOS LLM / embedder object once."""
    seen: set[int] = set()
    for obj in objs:
        if id(obj) in seen or isinstance(obj.client, MeteredOpenAI):
            continue
        seen.add(id(obj))
        obj.client = MeteredOpenAI(obj.client, usage, **kw)


def _noting(fn: Any, errors: list[Exception]) -> Any:
    def wrapped(*args: Any, **kw: Any) -> Any:
        try:
            return fn(*args, **kw)
        except Exception as exc:
            errors.append(exc)
            raise

    return wrapped


def _watch_reader(reader: Any, errors: list[Exception], fallbacks: list[str]) -> None:
    """Record what `SimpleStructMemReader` swallows: an exception from the LLM call or from the
    embedding call that builds each memory (`_safe_generate`, simple_struct.py:258-263, and the
    per-memory `except` in `_process_chat_data`, :427-428, which drops the memory), and a reply
    that does not parse, after which MemOS keeps the raw window text as the memory (:265-313).
    MemOS's behaviour is left unchanged; the caller decides what a noted error means."""
    parse = reader._safe_parse

    def parse_noting(text: Any) -> Any:
        out = parse(text)
        if text and not out:
            fallbacks.append("unparseable extraction reply")
        return out

    reader.llm.generate = _noting(reader.llm.generate, errors)
    reader.embedder.embed = _noting(reader.embedder.embed, errors)
    reader._safe_parse = parse_noting


@dataclass
class _Slot:
    """One MemReader and the failures noted on it. The reader runs its LLM and embedding calls
    in threads of its own, so concurrent events need one reader each."""

    reader: Any
    errors: list[Exception] = field(default_factory=list)
    fallbacks: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------- MemOS
class MemOSSystem(NativeSystem):
    """MemOS `general_text` on embedded Qdrant with the MemReader in fine mode (see module
    docstring); all state is under `store_dir/qdrant`."""

    name = "memos"

    def __init__(self, *args: Any, workers: int = 1, **kw: Any):
        self.workers = max(1, workers)
        self._mem: Any = None
        self._slots: list[_Slot] = []
        super().__init__(*args, **kw)

    @property
    def retains_raw_text(self) -> bool:
        return True  # `metadata.sources[].content` is the message text

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "memos_version": _memos_version(),
            "text_memory": "general_text",
            "reader": "simple_struct",
            "reader_mode": "fine",
            "vector_store": "qdrant-path",
            "embedder": "universal_api (OpenAI-compatible)",
            "workers": self.workers,
        }

    def config(self) -> dict[str, Any]:
        """`MemReaderConfigFactory` and `GeneralTextMemoryConfig` payloads (memos/configs/
        mem_reader.py, configs/memory.py:103-140). Thinking is switched off by the metered
        client, not by MemOS's own `enable_thinking` field (which sends a different key)."""
        llm = {
            "backend": "openai",
            "config": {
                "model_name_or_path": self.llm_model,
                "api_key": _api_key(),
                "api_base": self.llm_base_url,
                "temperature": 0.0,
                "max_tokens": self.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
                "remove_think_prefix": True,
            },
        }
        embedder = {
            "backend": "universal_api",
            "config": {
                "provider": "openai",
                "api_key": _api_key(),
                "base_url": self.embed_base_url,
                "model_name_or_path": self.embed_model,
            },
        }
        return {
            "reader": {
                "backend": "simple_struct",
                "config": {
                    "llm": llm,
                    "embedder": embedder,
                    # only used for documents; markdown needs no tokenizer download
                    "chunker": {"backend": "markdown", "config": {}},
                },
            },
            "text_memory": {
                "extractor_llm": llm,
                "embedder": embedder,
                "vector_db": {
                    "backend": "qdrant",
                    "config": {
                        "collection_name": COLLECTION,
                        "vector_dimension": self.embed_dims,
                        "distance_metric": "cosine",
                        "path": str(self.store_dir / "qdrant"),
                    },
                },
            },
        }

    def _open(self) -> None:
        _memos_env()
        from memos.configs.mem_reader import MemReaderConfigFactory
        from memos.configs.memory import MemoryConfigFactory
        from memos.mem_reader.factory import MemReaderFactory
        from memos.memories.factory import MemoryFactory
        from memos.memos_tools.singleton import _factory_singleton

        for name in ("memos", "httpx"):
            logging.getLogger(name).setLevel(logging.WARNING)
        cfg = self.config()
        _factory_singleton.clear_cache()
        mem = MemoryFactory.from_config(
            MemoryConfigFactory(backend="general_text", config=cfg["text_memory"])
        )
        self._meter_llms((mem.extractor_llm,))
        _meter((mem.embedder,), self.usage)
        slots = []
        for _ in range(self.workers):
            _factory_singleton.clear_cache()  # a distinct reader (and clients) per worker
            reader = MemReaderFactory.from_config(
                MemReaderConfigFactory.model_validate(cfg["reader"])
            )
            self._meter_llms((reader.llm, reader.general_llm))
            _meter((reader.embedder,), self.usage)
            slot = _Slot(reader)
            _watch_reader(reader, slot.errors, slot.fallbacks)
            slots.append(slot)
        self.stats.setdefault("extract_fallback", 0)
        self._slots, self._mem = slots, mem

    def _meter_llms(self, llms: Iterable[Any]) -> None:
        llms = list(llms)
        for llm in llms:  # MemOS builds its client with the SDK's 10-minute default timeout
            llm.client = llm.client.with_options(timeout=LLM_TIMEOUT_S)
        _meter(
            llms,
            self.usage,
            extra_body=self.llm_extra_body,
            max_tokens_cap=self.max_output_tokens,
        )

    def _close(self) -> None:
        mem, self._mem, self._slots = self._mem, None, []
        if mem is None:
            return
        client = getattr(mem.vector_db, "client", None)
        if client is not None:
            client.close()  # releases the embedded Qdrant directory lock

    def _points(self, with_vectors: bool = False) -> Iterator[Any]:
        """Every stored point, straight from the embedded Qdrant client (`get_all` would fetch
        the vectors too)."""
        client, offset = self._mem.vector_db.client, None
        while True:
            points, offset = client.scroll(
                collection_name=COLLECTION,
                limit=SCROLL_PAGE,
                offset=offset,
                with_payload=True,
                with_vectors=with_vectors,
            )
            yield from points
            if offset is None:
                return

    def _reconcile(self) -> None:
        # `add` upserts all memories of one event in one call, so any memory proves completion.
        found = keys_from_payloads({"id": str(p.id), **(p.payload or {})} for p in self._points())
        for key, ids in found.items():
            if key not in self.ledger:
                self.ledger.record(key, memory_ids=ids, reconciled=True)

    def _extract(self, slot: _Slot, e: Event) -> tuple[list[Any], int] | Exception:
        """MemReader on one event: (memories, 1 if MemOS fell back to the raw text else 0), or
        the error the reader swallowed. Touches no store."""
        slot.errors.clear()
        fallbacks0 = len(slot.fallbacks)
        try:
            groups = slot.reader.get_memory(
                chat_scene(e), type="chat", info=reader_info(e, self.user_id), mode="fine"
            )
        except Exception as exc:
            return exc
        if slot.errors:
            return slot.errors[0]
        items = [m for group in groups for m in group]
        for m in items:
            m.metadata.embedding = None
        return items, len(slot.fallbacks) - fallbacks0

    def _extract_all(self, events: list[Event]) -> Iterator[tuple[list[Any], int] | Exception]:
        """Extractions in event order. With `workers` > 1 they run ahead concurrently, each on
        its own reader; extraction reads no store, so this changes nothing but the wall time."""
        if len(self._slots) == 1:
            for e in events:
                yield self._extract(self._slots[0], e)
            return
        free: queue.Queue[_Slot] = queue.Queue()
        for s in self._slots:
            free.put(s)

        def run(e: Event) -> tuple[list[Any], int] | Exception:
            slot = free.get()
            try:
                return self._extract(slot, e)
            finally:
                free.put(slot)

        with ThreadPoolExecutor(max_workers=len(self._slots)) as pool:
            futures = [pool.submit(run, e) for e in events]
            try:
                for f in futures:
                    yield f.result()
            finally:
                for f in futures:
                    f.cancel()

    def _ingest_events(self, events: list[Event]) -> None:
        for e, res in zip(events, self._extract_all(events), strict=True):
            key = event_key(e)
            try:
                if isinstance(res, Exception):
                    raise res
                items, fallback = res
                if items:
                    self._mem.add(items)
            except Exception as exc:
                self._fail(key, exc)
                continue
            self.stats["extract_fallback"] += fallback
            self.ledger.record(
                key, event_id=e.event_id, memory_ids=[m.id for m in items], fallback=fallback
            )

    def _search(self, query: str) -> list[Hit]:
        items = self._mem.search(query, top_k=self.top_k)
        # `search` returns the ranked memories without their scores
        return [memos_hit(m.model_dump(), 1.0 / (rank + 1)) for rank, m in enumerate(items)]

    def export_raw(self) -> list[dict[str, Any]]:
        """Every memory as MemOS stores it (id, text and the whole metadata: sources, tags,
        status, version, history, our `info`), the vector replaced by its dimension."""
        return [
            export_row(str(p.id), dict(p.payload or {}), len(p.vector or []))
            for p in self._points(with_vectors=True)
        ]

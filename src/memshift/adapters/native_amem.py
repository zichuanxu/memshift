"""A-Mem behind the MemShift native-system interface (prereg/v2.md §3, decisions.md 2026-09-29).

A-Mem is "A-MEM: Agentic Memory for LLM Agents" (Xu et al., NeurIPS 2025, arXiv 2502.12110). The
abstract page links two repositories: `WujiangXu/A-mem` (the LoCoMo evaluation harness) and
`WujiangXu/A-mem-sys` ("the source code of the agentic memory system", the installable package
`agentic_memory`). The adapter uses the system repository, pinned to commit f303dfc71e07 (see
`docker/native/amem.txt`); the harness (commit 0c8039f) is LoCoMo-specific: test_advanced.py:
127-134 rewrites the question into LLM keywords before retrieving, which the unified probe query
does not do. Line numbers below refer to `agentic_memory/` in the system repository at the pin.

What the library does per record (`AgenticMemorySystem.add_note`, memory_system.py:237-290):
  1. A `MemoryNote` (memory_system.py:24-81) holds the content verbatim plus id, keywords, context,
     tags, links, category, timestamp, last_accessed, retrieval_count and evolution_history. There
     is no free-form metadata field, so the event metadata rides on the note's own fields: `id` =
     `object_id#revision`, `timestamp` and `last_accessed` = event time (both default to wall-clock
     now, :75-77), `category` = source. A unit is always a new note; the library never looks a
     record up by id or content.
  2. One analysis LLM call fills keywords, context and tags (`analyze_content`, :163-235; JSON
     schema output).
  3. One evolution LLM call (`process_memory`, :625-754) whenever the store is not empty: the 5
     nearest notes (by the index, full content in the prompt) go to the model, which may
     `strengthen` (append links to the new note and overwrite its tags, :714-718) and/or
     `update_neighbor` (overwrite the tags and context of the neighbours in `self.memories`, never
     their content, :719-743). Nothing deletes or supersedes a note, and the neighbour edits reach
     only the in-memory note: the index keeps the metadata it had when the note was added.
  4. The note is added to the index (:271-284); every `evo_threshold` (100) evolved notes
     `consolidate_memories` (:292-312) re-adds every note to a fresh retriever object over the same
     collection, which ChromaDB ignores for ids it already holds (checked on chromadb 1.5.9).
  `evolution_history` is written nowhere in the package (only copied), so exports show it empty.

The library keeps its state in memory: `ChromaRetriever` opens an in-memory client (retrievers.py:
26) and `self.memories` is a dict (memory_system.py:112). `AgenticMemorySystem.__init__` also
*resets* the process-wide in-memory index (:115-120), which would wipe a persistent one. The
adapter therefore subclasses both: `PersistentRetriever` swaps in a `chromadb.PersistentClient`
under `store_dir/chroma` (embedding, add, delete and query stay the library's), and
`PersistentSystem` runs the library constructor with the index class stubbed out, then attaches the
persistent retriever and reloads `store_dir/notes.json` (every note field plus `evo_cnt`, written
after each event). `_close()` closes the Chroma client, which flushes it, so `snapshot()` and
`restore()` by directory copy reproduce the store exactly.

LLM: `OpenAIController` builds `openai.OpenAI(api_key=...)` with no base URL (llm_controller.py:54)
and calls `client.chat.completions.create(...)` directly (:77); litellm is imported by the package
but only its Ollama / OpenRouter controllers call it. The adapter replaces the controller's client
by a `MeteredOpenAI` over our endpoint (thinking off through the extra body, usage metered) and
wraps `get_completion` to force temperature 0 (the library default is 1.0, :58) and the shared
output cap. The library swallows every LLM or JSON failure and stores the note with default
metadata (memory_system.py:233-235, :747-749); the wrapper records those failures, and the adapter
reports a unit whose LLM step failed through `_fail` (raise mode aborts; skip mode logs it to
`failures.jsonl` and keeps the stored note, marked in the ledger).

Embeddings: A-Mem accepts only a sentence-transformers model name (retrievers.py:27), no
OpenAI-compatible embedder, so its default local encoder all-MiniLM-L6-v2 is used and
`embed_base_url` is not contacted; `describe()` records this. `embed_calls` counts encoder
invocations (one per added note, one per query, plus consolidation re-adds); `embed_tokens` stays 0
(local, not metered).

The encoder is pinned to one Hub revision (`AMEM_ENCODER_REVISION`), passed to
`SentenceTransformerEmbeddingFunction`, which forwards it to `SentenceTransformer`. The only place
the package builds an encoder is `ChromaRetriever.__init__` (retrievers.py:27), reached from the
library constructor (memory_system.py:117, :123) and `consolidate_memories` (:295); the first is
stubbed out (`_library_init`), the second overridden, so nothing loads an unpinned copy. ChromaDB
caches loaded models by name only, so the pinned encoder is requested under its full repo id and
never reuses a copy loaded under the library's short name. Run containers set HOME to a throwaway
directory, so `_amem_env()` points `HF_HOME` at `cache/hf` in the repository (before
sentence_transformers or huggingface_hub is imported) and, when the pinned snapshot is already in
that cache, sets `HF_HUB_OFFLINE=1`, so runs never touch the network. `python -m
memshift.adapters.native_amem --seed-cache` downloads exactly the files the encoder loads
(`ENCODER_FILES`) at that revision into the cache; run it once per host before the runs.

Retrieval (`_search`): the library's `ChromaRetriever.search` (:465) ranks notes; each ranked note
is followed by the notes it links to, as in `find_related_memories_raw` (:345-374, the call the
paper's harness makes with k = 10); notes are rendered from the live `self.memories` entry (as
`search`, :470-479) with the LLM-derived context, keywords and tags ahead of the content, so the
reader's per-document cap cuts the tail of a long record and not the system's own fields. No answer
generation and no query rewriting. k defaults to the paper's 10 (`AMEM_RETRIEVE_K`).

Idempotency: units are keyed `object_id#revision` in the ledger; on open, notes already in
`notes.json` are reconciled into it. A restored branch starts from the prefix usage and cost.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import threading
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from memshift.adapters.native import (
    _EPOCH,
    Hit,
    MeteredOpenAI,
    NativeSystem,
    Usage,
    _api_key,
    _minute,
    event_key,
    event_text,
    parse_key,
)
from memshift.gen.model import Event
from memshift.runinfo import REPO

AMEM_REPO = "https://github.com/WujiangXu/A-mem-sys"
AMEM_COMMIT = "f303dfc71e07bdc787f4bc135d4cea328ae30e99"
AMEM_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"  # A-Mem's default, by full repo id
AMEM_ENCODER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"  # `main` of that repo
AMEM_ENCODER_DIMS = 384
# What SentenceTransformer loads from the repository (the rest is ONNX / OpenVINO / TF / Rust
# copies of the same weights, docs and a training script).
ENCODER_FILES = (
    "modules.json",
    "config_sentence_transformers.json",
    "sentence_bert_config.json",
    "1_Pooling/config.json",
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)
AMEM_RETRIEVE_K = 10  # the harness default (A-mem test_advanced.py:417)
AMEM_EVO_THRESHOLD = 100  # library default (memory_system.py:97)
NOTES_FILE = "notes.json"
CHROMA_DIR = "chroma"
DEFAULT_CONTEXT = "General"  # memory_system.py:70
DEFAULT_CATEGORY = "Uncategorized"  # memory_system.py:71
# The MemoryNote constructor arguments (memory_system.py:36-47) = the attributes it sets.
NOTE_FIELDS = (
    "content",
    "id",
    "keywords",
    "links",
    "retrieval_count",
    "timestamp",
    "last_accessed",
    "context",
    "evolution_history",
    "category",
    "tags",
)


# ---------------------------------------------------------------------- encoder cache
def hub_cache_dir() -> Path:
    """Where huggingface_hub keeps model snapshots: HF_HUB_CACHE, else HF_HOME/hub."""
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    default = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "huggingface"
    return Path(os.environ.get("HF_HOME") or default) / "hub"


def encoder_snapshot_dir() -> Path:
    repo = "models--" + AMEM_ENCODER.replace("/", "--")
    return hub_cache_dir() / repo / "snapshots" / AMEM_ENCODER_REVISION


def encoder_cached() -> bool:
    """True when the pinned revision is in the hub cache with every file the encoder loads."""
    snap = encoder_snapshot_dir()
    return all((snap / name).is_file() for name in ENCODER_FILES)


def encoder_kwargs() -> dict[str, Any]:
    """Arguments of `SentenceTransformerEmbeddingFunction` for the pinned encoder."""
    return {"model_name": AMEM_ENCODER, "revision": AMEM_ENCODER_REVISION}


def seed_cache(download: Callable[..., Any] | None = None) -> Path:
    """Download exactly the pinned encoder files into the hub cache (skipped when they are
    there); returns the snapshot directory. `download` replaces `snapshot_download`."""
    _amem_env()
    if not encoder_cached():
        if download is None:
            import huggingface_hub

            download = huggingface_hub.snapshot_download
        download(
            repo_id=AMEM_ENCODER,
            revision=AMEM_ENCODER_REVISION,
            allow_patterns=list(ENCODER_FILES),
            cache_dir=str(hub_cache_dir()),
        )
    if not encoder_cached():
        raise RuntimeError(f"encoder snapshot incomplete in {encoder_snapshot_dir()}")
    return encoder_snapshot_dir()


# ---------------------------------------------------------------------- event -> note
def note_timestamp(ts: datetime) -> str:
    """A-Mem's own timestamp format, YYYYMMDDHHMM (memory_system.py:56)."""
    return f"{ts:%Y%m%d%H%M}"


def parse_note_timestamp(value: Any) -> datetime | None:
    s = str(value)
    if len(s) != 12 or not s.isdigit():
        return None
    try:
        return datetime.strptime(s, "%Y%m%d%H%M")
    except ValueError:
        return None


def note_kwargs(e: Event) -> dict[str, Any]:
    """Arguments of `add_note` for one source event: the text (a deletion is the tombstone text)
    and the metadata carried on the note's own fields."""
    ts = note_timestamp(e.ts)
    return {
        "content": event_text(e),
        "time": ts,
        "id": event_key(e),
        "category": e.source,
        "last_accessed": ts,
    }


def note_dict(note: Any) -> dict[str, Any]:
    """A note's fields as plain JSON data (a copy)."""
    fields: dict[str, Any] = json.loads(json.dumps(vars(note), default=str))
    return fields


# ---------------------------------------------------------------------- persistence (pure)
def read_notes(path: Path) -> tuple[list[dict[str, Any]], int]:
    """Notes in insertion order and the evolved-note counter, as saved by `write_notes`."""
    if not path.is_file():
        return [], 0
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("notes") or []), int(data.get("evo_cnt") or 0)


def write_notes(path: Path, notes: Iterable[Mapping[str, Any]], evo_cnt: int) -> None:
    """Atomically replace the notes file (temp file, fsync, rename)."""
    tmp = path.with_name(path.name + ".tmp")
    payload = json.dumps({"evo_cnt": evo_cnt, "notes": list(notes)}, ensure_ascii=False)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def stored_keys(notes: Iterable[Mapping[str, Any]]) -> list[str]:
    """The `object_id#revision` keys of stored notes (their ids), for reconciling the ledger."""
    return [str(n["id"]) for n in notes if parse_key(str(n.get("id", ""))) is not None]


# ---------------------------------------------------------------------- serve-side hits
def _texts(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value] if value else []
    return [str(v) for v in value or []]


def amem_hit(note: Mapping[str, Any], score: float = 0.0) -> Hit:
    """One stored note -> text block: source key, event time and A-Mem's derived context, keywords
    and tags ahead of the verbatim content. Wall-clock fields are not shown."""
    note_id = str(note.get("id", ""))
    parsed = parse_key(note_id)
    refs = [parsed] if parsed is not None else []
    ts = parse_note_timestamp(note.get("timestamp"))
    category = str(note.get("category") or DEFAULT_CATEGORY)
    head = ["note"]
    if parsed is not None:
        label = "" if category == DEFAULT_CATEGORY else f"{category} "
        head.append(f"source {label}{parsed[0]} r{parsed[1]}")
    if ts is not None:
        head.append(f"event {_minute(ts)}")
    context = str(note.get("context") or DEFAULT_CONTEXT)
    if context != DEFAULT_CONTEXT:
        head.append(f"context: {context}")
    if keywords := _texts(note.get("keywords")):
        head.append("keywords: " + ", ".join(keywords))
    if tags := _texts(note.get("tags")):
        head.append("tags: " + ", ".join(tags))
    text = f"[{' | '.join(head)}] {note.get('content', '')}"
    return Hit(note_id, text, ts or _EPOCH, refs, score)


def expand_hits(
    ranked: Sequence[tuple[str, float]], notes: Mapping[str, Mapping[str, Any]], k: int
) -> list[Hit]:
    """Ranked (note id, index distance) pairs -> hits in A-Mem's order: each note is followed by
    the notes it links to (at most `k` per note, memory_system.py:365-373), every note once. Ids
    the store no longer holds are skipped (`search`, :470); a linked note that was not ranked has
    score 0."""
    scores: dict[str, float] = {}
    for note_id, dist in ranked:
        scores.setdefault(note_id, 1.0 / (1.0 + dist))
    hits: list[Hit] = []
    seen: set[str] = set()

    def emit(note_id: str, note: Mapping[str, Any]) -> None:
        seen.add(note_id)
        hits.append(amem_hit(note, scores.get(note_id, 0.0)))

    for note_id, _ in ranked:
        note = notes.get(note_id)
        if note is None or note_id in seen:
            continue
        emit(note_id, note)
        linked = 0
        for link_id in note.get("links") or []:
            neighbour = notes.get(link_id) if isinstance(link_id, str) else None
            if neighbour is None or linked >= k:
                continue
            linked += 1
            if link_id not in seen:
                emit(link_id, neighbour)
    return hits


# ---------------------------------------------------------------------- library side
def _amem_env() -> None:
    """Telemetry off; litellm (imported by the package) would fetch a price map from the network
    at import time (litellm_core_utils/get_model_cost_map.py); the Hugging Face cache in the
    repository (containers run with a throwaway HOME), offline once the pinned encoder is in it.
    Must run before sentence_transformers, huggingface_hub or the package is imported."""
    os.environ.setdefault("HF_HOME", str(REPO / "cache" / "hf"))
    if encoder_cached():
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class _NoIndex:
    """Stands in for `ChromaRetriever` while the library constructor runs."""

    def __init__(self, **_: Any) -> None:
        self.client = SimpleNamespace(reset=lambda: None)


_INIT_LOCK = threading.Lock()


@contextlib.contextmanager
def _library_init(module: Any) -> Iterator[None]:
    """`AgenticMemorySystem.__init__` builds two in-memory indexes, loads the encoder for each and
    resets the process-wide in-memory index (memory_system.py:115-123); none of that is wanted."""
    with _INIT_LOCK:
        real = module.ChromaRetriever
        module.ChromaRetriever = _NoIndex
        try:
            yield
        finally:
            module.ChromaRetriever = real


@functools.cache
def _library() -> SimpleNamespace:
    """The library plus our persistent variants; built on first use (the package is optional)."""
    import chromadb
    from agentic_memory import memory_system
    from agentic_memory.memory_system import AgenticMemorySystem, MemoryNote
    from agentic_memory.retrievers import ChromaRetriever
    from chromadb.config import Settings

    class PersistentRetriever(ChromaRetriever):
        """`ChromaRetriever` on a persistent client (retrievers.py:26 builds an in-memory one);
        add, delete and search are the library's, with the encoder calls counted."""

        def __init__(
            self, path: Path, usage: Usage, embedding_function: Any = None, name: str = "memories"
        ) -> None:
            if embedding_function is None:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

                embedding_function = SentenceTransformerEmbeddingFunction(**encoder_kwargs())
            self.usage = usage
            self.client = chromadb.PersistentClient(
                path=str(path), settings=Settings(anonymized_telemetry=False)
            )
            self.embedding_function = embedding_function
            self.collection = self.client.get_or_create_collection(
                name=name, embedding_function=embedding_function
            )

        def add_document(self, document: str, metadata: dict, doc_id: str) -> None:
            self.usage.add(embed_calls=1)
            super().add_document(document, metadata, doc_id)

        def search(self, query: str, k: int = 5) -> Any:
            self.usage.add(embed_calls=1)
            return super().search(query, k)

    class PersistentSystem(AgenticMemorySystem):
        """`AgenticMemorySystem` whose notes and index live under `store` and whose LLM is ours."""

        def __init__(
            self,
            store: Path,
            usage: Usage,
            *,
            llm_model: str,
            base_url: str,
            extra_body: dict[str, Any] | None,
            max_output_tokens: int | None,
            embedding_function: Any = None,
            llm_client: Any = None,
        ) -> None:
            with _library_init(memory_system):
                super().__init__(
                    model_name=AMEM_ENCODER,
                    llm_backend="openai",
                    llm_model=llm_model,
                    evo_threshold=AMEM_EVO_THRESHOLD,
                    api_key=_api_key(),
                )
            self.store = store
            self.usage = usage
            self.llm_errors: list[str] = []
            self.retriever = PersistentRetriever(store / CHROMA_DIR, usage, embedding_function)
            self.owns_llm = llm_client is None
            self.raw_llm = llm_client
            if llm_client is None:
                from openai import OpenAI

                self.raw_llm = OpenAI(api_key=_api_key(), base_url=base_url)
            llm = self.llm_controller.llm
            llm.client = MeteredOpenAI(
                self.raw_llm, usage, extra_body=extra_body, max_tokens_cap=max_output_tokens
            )
            llm.get_completion = self._recorded(llm.get_completion, max_output_tokens)
            notes, self.evo_cnt = read_notes(store / NOTES_FILE)
            for n in notes:
                self.memories[n["id"]] = MemoryNote(**{k: n[k] for k in NOTE_FIELDS if k in n})

        def _recorded(self, inner: Callable[..., Any], cap: int | None) -> Callable[..., Any]:
            """Temperature 0 and the output cap on every call; failures (transport or not JSON)
            are recorded and re-raised, so the library falls back exactly as it would."""

            def get_completion(
                prompt: str,
                response_format: dict | None = None,
                temperature: float = 1.0,
                max_tokens: int | None = None,
            ) -> Any:
                try:
                    out = inner(
                        prompt, response_format=response_format, temperature=0.0, max_tokens=cap
                    )
                    json.loads(out)
                except Exception as exc:
                    self.llm_errors.append(f"{type(exc).__name__}: {exc}"[:500])
                    raise
                return out

            return get_completion

        def index_metadata(self, note: Any) -> dict[str, Any]:
            """The metadata the library indexes for a note (memory_system.py:271-283)."""
            return {k: getattr(note, k) for k in NOTE_FIELDS}

        def consolidate_memories(self) -> None:
            """As the library (memory_system.py:292-312), on the persistent index: every note is
            added again, and ChromaDB ignores ids it already holds."""
            for note in self.memories.values():
                self.retriever.add_document(note.content, self.index_metadata(note), note.id)

        def save(self) -> None:
            write_notes(
                self.store / NOTES_FILE,
                (note_dict(n) for n in self.memories.values()),
                self.evo_cnt,
            )

        def shutdown(self) -> None:
            self.save()
            self.retriever.client.close()
            if self.owns_llm:
                self.raw_llm.close()

    return SimpleNamespace(System=PersistentSystem, Retriever=PersistentRetriever)


# ---------------------------------------------------------------------- A-Mem
class AMemSystem(NativeSystem):
    """A-Mem with its own analysis, linking and evolution. Drive one instance from one thread."""

    name = "amem"

    def __init__(
        self, *args: Any, embedding_function: Any = None, llm_client: Any = None, **kw: Any
    ):
        """`embedding_function` (a ChromaDB embedding function) and `llm_client` (an OpenAI-style
        client) replace the local encoder and our endpoint; they exist for offline tests."""
        kw.setdefault("top_k", AMEM_RETRIEVE_K)
        self.embedding_function = embedding_function
        self.llm_client = llm_client
        self._mem: Any = None
        super().__init__(*args, **kw)

    @property
    def retains_raw_text(self) -> bool:
        return True  # note.content is the ingested text, verbatim (memory_system.py:64, :273)

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "embed_model": AMEM_ENCODER,
            "embed_revision": AMEM_ENCODER_REVISION,
            "embed_dims": AMEM_ENCODER_DIMS,
            "embed_location": "local sentence-transformers encoder inside ChromaDB "
            "(A-Mem accepts no OpenAI-compatible embedder, retrievers.py:27)",
            "requested_embed_model": self.embed_model,
            "embed_endpoint_used": False,
            "llm_temperature": 0.0,
            "amem_repo": AMEM_REPO,
            "amem_commit": AMEM_COMMIT,
            "vector_store": "chromadb-persistent",
            "evo_threshold": AMEM_EVO_THRESHOLD,
            "retrieval": f"index top-{self.top_k} plus linked notes (find_related_memories_raw)",
        }

    def _open(self) -> None:
        _amem_env()
        self._mem = _library().System(
            self.store_dir,
            self.usage,
            llm_model=self.llm_model,
            base_url=self.llm_base_url,
            extra_body=self.llm_extra_body,
            max_output_tokens=self.max_output_tokens,
            embedding_function=self.embedding_function,
            llm_client=self.llm_client,
        )

    def _close(self) -> None:
        mem, self._mem = self._mem, None
        if mem is not None:
            mem.shutdown()

    def _reconcile(self) -> None:
        for key in stored_keys(vars(n) for n in self._mem.memories.values()):
            if key not in self.ledger:
                self.ledger.record(key, reconciled=True)

    def _ingest_events(self, events: list[Event]) -> None:
        mem = self._mem
        for e in events:
            key = event_key(e)
            n_errors = len(mem.llm_errors)
            try:
                mem.add_note(**note_kwargs(e))
            except Exception as exc:
                # add_note stores the note (:268) before it indexes it (:284)
                mem.memories.pop(key, None)
                self._fail(key, exc)
                continue
            mem.save()
            info: dict[str, Any] = {"event_id": e.event_id, "op": e.op}
            errors = mem.llm_errors[n_errors:]
            if errors:
                info["llm_errors"] = errors
                self._fail(key, RuntimeError(f"LLM step failed inside A-Mem: {errors}"))
            self.ledger.record(key, **info)

    def index_count(self) -> int:
        return int(self._mem.retriever.collection.count())

    def _search(self, query: str) -> list[Hit]:
        mem = self._mem
        if not mem.memories or self.index_count() == 0:
            return []
        res = mem.retriever.search(query, self.top_k)
        ranked = list(zip(res["ids"][0], res["distances"][0], strict=True))
        notes = {i: vars(n) for i, n in mem.memories.items()}
        return expand_hits(ranked, notes, self.top_k)

    def export_raw(self) -> list[dict[str, Any]]:
        """Every note with all its fields (content, keywords, tags, context, links, timestamps,
        retrieval count, evolution history) and the metadata the index holds for it."""
        mem = self._mem
        indexed = mem.retriever.collection.get(include=["metadatas"])
        by_id = dict(zip(indexed["ids"], indexed["metadatas"], strict=True))
        rows = []
        for note in mem.memories.values():
            row = note_dict(note)
            row["index_metadata"] = by_id.get(note.id)
            rows.append(row)
        return rows


def main(argv: Sequence[str] | None = None, download: Callable[..., Any] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m memshift.adapters.native_amem", description="A-Mem adapter utilities."
    )
    ap.add_argument(
        "--seed-cache",
        action="store_true",
        help=f"download the pinned encoder ({AMEM_ENCODER} @ {AMEM_ENCODER_REVISION[:12]}) into "
        "the Hugging Face cache under cache/hf",
    )
    args = ap.parse_args(argv)
    if not args.seed_cache:
        ap.print_help()
        return 2
    print(seed_cache(download))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

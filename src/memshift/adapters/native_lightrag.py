"""LightRAG (lightrag-hku 1.5.7, HKUDS) behind the native-system interface (prereg v2 §3, block N).

Paths are relative to the installed `lightrag/` package; `docker/native/lightrag.txt` pins it.

Store. `working_dir` is `store_dir` with the library's file-based defaults (lightrag.py:403-412):
  JSON KV (full docs, chunks, extraction / summary / keyword cache, per-entity and per-relation
  chunk tracking), NanoVectorDB (entities, relations, chunks), NetworkX GraphML and JSON doc
  status, so `snapshot()` / `restore()` by directory copy carries the whole state. `_close()`
  flushes every storage (`index_done_callback`) and calls `finalize_storages` (lightrag.py:1676).
  It also calls `finalize_share_data` (kg/shared_storage.py:3830): the library keeps its storages
  in process-global dicts keyed by namespace, and `try_initialize_namespace`
  (shared_storage.py:3641) returns False for a namespace that is already initialised, so a second
  instance in the same process shows the first one's data instead of loading its own directory
  (observed: a fresh empty directory listed the previous instance's documents). One store is
  therefore open per process at a time (`claim_process` raises otherwise); the E5 driver already
  closes a store before it opens the next.

Models. LLM = `openai_complete_if_cache` (llm/openai.py:305) on our endpoint: temperature 0,
  thinking off through `extra_body`, output capped at `max_output_tokens`. Embeddings =
  `openai_embed.func` (llm/openai.py:1054): no `dimensions` argument is sent and no tiktoken
  truncation is requested; it is wrapped in the library's `EmbeddingFunc`. Both helpers build a
  fresh client per call (llm/openai.py:444, 1167), so no client can be wrapped; usage is metered
  through the helpers' own `token_tracker` hook (llm/openai.py:781-789, 1198-1203), which fires
  once per answered request. No request carries `tools`: extraction is delimiter text and keyword
  extraction asks for `response_format` json_object (operate.py:5047). Everything else is the
  library default: fixed-token chunks of 1,200 tokens with 100 overlap (`ainsert` always uses the F
  chunker, lightrag.py:1803-1861), one gleaning pass (constants.py:17), the LLM response cache on
  (a deletion rebuilds from it), tiktoken `gpt-4o-mini` (`o200k_base`) as tokenizer. Its encoding
  file is fetched on first use; `TIKTOKEN_CACHE_DIR` points at `cache/tiktoken` so that it persists
  (and can be pre-seeded) with the other run caches. The library's INFO logging is silenced.

Integration contract (decisions.md, 2026-09-29). The library has a per-document insert with an
  explicit id and a delete by document id and no upsert: `apipeline_enqueue_documents` drops every
  id already in doc status and files a FAILED duplicate record (pipeline.py:1178, 1269-1295,
  1340-1412). So each source object is one document whose id is its `object_id` and whose
  `file_path` is `object_id#revision`; the library copies that path onto every chunk, entity and
  relation derived from the document (utils_pipeline.py:172-200, operate.py:2727-2735), which gives
  (object_id, revision) refs. A new revision is `adelete_by_doc_id` then `ainsert` under the same
  id, a delete event is `adelete_by_doc_id`, and the text is `event_text(e)` (object id, revision
  and event time in its head). `revisions.json` maps object id -> current revision and event
  time; `_reconcile` rebuilds it from doc status and purges any document that is not PROCESSED
  under its own file path (partial ingests, duplicate records), so a resumed ingest converges:
  deleting an absent document is a no-op and an event that is not newer than the stored revision
  is skipped (`plan_event`). `batch_size > 1` (off by default, as for the Graphiti adapter) sends
  events of distinct objects in one `ainsert`, whose documents the library then runs in parallel
  (`max_parallel_insert`); deletes still go first and one at a time, and every document is
  checked on its own.

Deletion, which determines the system's (D, B). `adelete_by_doc_id` (lightrag.py:5718) ->
  `_purge_kg_contributions` (5010) -> `_purge_derived_kg_contributions` (5256). For every entity
  and relation in the document's recovery rows (`full_entities` / `full_relations`), the tracked
  chunks minus the document's chunks are computed (5378-5511). None left: the node or edge, its
  vectors and its tracking row are removed (5563-5662). Some left, from other documents: it is
  rebuilt (5678-5701) by `rebuild_knowledge_from_chunks` (operate.py:1101) from the CACHED
  extraction outputs of the surviving chunks (operate.py:1171, 1414-1504), not a new extraction.
  The description is the surviving descriptions joined with `<SEP>` without an LLM call while
  there are fewer than 8 of them and they total under 1,200 tokens, and a cached LLM summary
  otherwise (operate.py:372-448, 532); `source_id` and `file_path` are rewritten to the surviving
  chunks (operate.py:1782-1792, 1867-2014). Chunks and vectors go last (lightrag.py:5183-5204),
  then doc status and the full text (6223-6224). An update thus removes what the old text
  contributed, and the new text is extracted in full and merged by entity name
  (operate.py:2428-2772). No derived value depends on a definition and nothing is checked against
  the source: (D, B) = (0, 1), bound at document (= source object) granularity. Two library
  defaults can let old content linger on hub entities: at 200 sources an entity keeps its old
  chunks and skips new descriptions (constants.py:71, 78; operate.py:2551-2570), and its
  `file_path` list is cut at 75 (constants.py:84).

Retrieval. `aquery_data` (lightrag.py:3998) runs `kg_query` with `only_need_context=True`
  (lightrag.py:4120): the library's keyword extraction (one cached LLM call, operate.py:4981),
  entity / relation / chunk search and token budgeting, without the answer LLM; it returns the
  entities, relations and chunks that would fill the prompt. Mode `mix` (local + global + vector
  chunks) is the library default (base.py:93) and the README's recommendation for the best
  results, so it is used rather than `hybrid`; no reranker is deployed, so reranking is off. That
  context is far larger than the 4,096-token evidence region (30k tokens), so the three lists,
  each in the library's own order, are interleaved chunk, entity, relation, ... before the shared
  packer takes hits in that order: a benchmark-side choice, since the library's prompt order
  (entities, relations, chunks) would fill the region with graph summaries alone.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from collections.abc import Coroutine, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from memshift.adapters.native import (
    _EPOCH,
    Hit,
    NativeSystem,
    Usage,
    _api_key,
    _minute,
    distinct_batches,
    event_key,
    event_text,
    parse_key,
)
from memshift.gen.model import Event, fmt_ts, parse_ts
from memshift.runinfo import REPO

if TYPE_CHECKING:
    from lightrag import LightRAG

T = TypeVar("T")

REVISIONS_FILE = "revisions.json"
GRAPH_FIELD_SEP = "<SEP>"  # lightrag/constants.py:49, joins source ids, file paths, descriptions
QUERY_MODE = "mix"  # lightrag/base.py:93; README "Selecting Query Modes"
QUERY_TOP_K = 40  # lightrag/constants.py:57 (entities / relations per query)
QUERY_CHUNK_TOP_K = 20  # constants.py:58
QUERY_MAX_ENTITY_TOKENS = 6000  # constants.py:59
QUERY_MAX_RELATION_TOKENS = 8000  # constants.py:60
QUERY_MAX_TOTAL_TOKENS = 30000  # constants.py:61
EMBED_MAX_TOKEN_SIZE = 8192  # max_model_len of the embedding server
_HEAD_TS = re.compile(r"^\[[^\]\n]*\| (\d{4}-\d{2}-\d{2}) (\d{2}:\d{2})\]")
_OPEN_STORES: set[Path] = set()


# ---------------------------------------------------------------------- pure helpers
def claim_process(store: Path) -> None:
    """LightRAG's storages are process-global (see module docstring): one open store at a time."""
    if _OPEN_STORES:
        raise RuntimeError(
            f"a LightRAG store is already open in this process ({sorted(map(str, _OPEN_STORES))}); "
            "close it before opening another"
        )
    _OPEN_STORES.add(store)


def release_process(store: Path) -> None:
    _OPEN_STORES.discard(store)


def doc_file_path(object_id: str, revision: int) -> str:
    """The `file_path` a document is inserted with; it is copied onto everything derived from it."""
    return f"{object_id}#{revision}"


def head_ts(text: str) -> str | None:
    """Event time (`YYYY-MM-DDTHH:MM`) from the head line `event_text` writes, if present."""
    m = _HEAD_TS.match(text)
    return f"{m.group(1)}T{m.group(2)}" if m else None


def source_refs(file_path: str) -> list[tuple[str, int]]:
    """(object_id, revision) pairs named by a `file_path` field: one path per source document,
    joined by `<SEP>` on entities and relations (operate.py:2688), a placeholder past 75 paths."""
    out: list[tuple[str, int]] = []
    for part in file_path.split(GRAPH_FIELD_SEP):
        ref = parse_key(part.strip())
        if ref is not None and ref not in out:
            out.append(ref)
    return out


def chunk_doc_id(chunk_id: str) -> str | None:
    """Document id of a chunk id `<doc_id>-chunk-NNN` (utils_pipeline.py:172)."""
    doc, sep, tail = chunk_id.rpartition("-chunk-")
    return doc if sep and doc and tail.isdigit() else None


def refs_of(file_path: str, source_ids: str, revision: Mapping[str, int]) -> list[tuple[str, int]]:
    """Refs from `file_path`; when it names none (placeholder, unknown source), from the chunk ids
    in `source_id` resolved to the current revision of their document."""
    refs = source_refs(file_path)
    if refs:
        return refs
    for cid in source_ids.split(GRAPH_FIELD_SEP):
        doc = chunk_doc_id(cid.strip())
        if doc is not None and doc in revision and (doc, revision[doc]) not in refs:
            refs.append((doc, revision[doc]))
    return refs


@dataclass(frozen=True)
class Action:
    kind: str  # "delete" | "insert"
    object_id: str
    revision: int = 0


def plan_event(current: Mapping[str, int], e: Event) -> list[Action]:
    """Library calls for one source event given the stored revision of each object. A revision is
    delete then insert (the library has no upsert); a delete of an absent object, an event that is
    not newer than what is stored, and a replayed event are no-ops."""
    cur = current.get(e.object_id)
    if e.op == "delete":
        return [Action("delete", e.object_id)] if cur is not None else []
    if cur is None:
        return [Action("insert", e.object_id, e.revision)]
    if e.revision <= cur:
        return []
    return [Action("delete", e.object_id), Action("insert", e.object_id, e.revision)]


def check_ingested(row: Mapping[str, Any] | None, file_path: str) -> None:
    """A document counts as ingested only when its doc-status row is PROCESSED under the file path
    it was inserted with; an id that already existed comes back as a duplicate record instead."""
    if row is None:
        raise RuntimeError("document not in doc status after insert")
    status = _value(row.get("status"))
    if status != "processed":
        raise RuntimeError(f"document status {status}: {str(row.get('error_msg'))[:500]}")
    if row.get("file_path") != file_path:
        raise RuntimeError(f"document kept file_path {row.get('file_path')!r}, not {file_path!r}")


def reconcile_plan(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, int], list[str]]:
    """From doc-status rows (`doc_id`, `status`, `file_path`): the stored revision of every complete
    document, and the ids to purge (partial ingests, duplicate records, anything not ours)."""
    keep: dict[str, int] = {}
    purge: list[str] = []
    for row in rows:
        doc_id = str(row["doc_id"])
        ref = parse_key(str(row.get("file_path") or ""))
        if _value(row.get("status")) == "processed" and ref is not None and ref[0] == doc_id:
            keep[doc_id] = ref[1]
        else:
            purge.append(doc_id)
    return keep, purge


def _value(x: Any) -> Any:
    return getattr(x, "value", x)


def _latest(refs: Sequence[tuple[str, int]], event_ts: Mapping[str, str]) -> datetime | None:
    times = [parse_ts(event_ts[o]) for o, _ in refs if o in event_ts]
    return max(times) if times else None


def _hit(
    kind: str,
    hit_id: str,
    body: str,
    refs: list[tuple[str, int]],
    event_ts: Mapping[str, str],
    rank: int,
    label: str = "",
) -> Hit:
    ts = _latest(refs, event_ts)
    head = [kind + (f" {label}" if label else "")]
    if refs:
        head.append("source " + ", ".join(f"{o} r{r}" for o, r in refs[:6]))
    if ts is not None:
        head.append(f"event {_minute(ts)}")
    return Hit(hit_id, f"[{' | '.join(head)}] {body}", ts or _EPOCH, refs, 1.0 / (rank + 1))


def interleave(*lists: Sequence[Hit]) -> list[Hit]:
    out: list[Hit] = []
    for i in range(max(map(len, lists), default=0)):
        out.extend(lst[i] for lst in lists if i < len(lst))
    return out


def lightrag_hits(
    data: Mapping[str, Any], revision: Mapping[str, int], event_ts: Mapping[str, str]
) -> list[Hit]:
    """The `data` of `aquery_data` (lightrag.py:4016-4059) -> hits, interleaved chunk, entity,
    relation. Every hit carries the (object_id, revision) of the documents it derives from."""
    chunks, entities, relations = [], [], []
    for i, c in enumerate(data.get("chunks") or []):
        cid = str(c.get("chunk_id", ""))
        refs = refs_of(str(c.get("file_path", "")), cid, revision)
        chunks.append(_hit("chunk", f"chunk:{cid}", str(c.get("content", "")), refs, event_ts, i))
    for i, ent in enumerate(data.get("entities") or []):
        name = str(ent.get("entity_name", ""))
        refs = refs_of(str(ent.get("file_path", "")), str(ent.get("source_id", "")), revision)
        body = f"{name}: {ent.get('description', '')}"
        label = str(ent.get("entity_type", ""))
        entities.append(_hit("entity", f"entity:{name}", body, refs, event_ts, i, label))
    for i, rel in enumerate(data.get("relationships") or []):
        src, tgt = str(rel.get("src_id", "")), str(rel.get("tgt_id", ""))
        refs = refs_of(str(rel.get("file_path", "")), str(rel.get("source_id", "")), revision)
        body = f"{src} -> {tgt}: {rel.get('description', '')}"
        label = str(rel.get("keywords", ""))
        relations.append(_hit("relation", f"relation:{src}|{tgt}", body, refs, event_ts, i, label))
    return interleave(chunks, entities, relations)


# ---------------------------------------------------------------------- usage metering
class _UsageTracker:
    """The `token_tracker` of LightRAG's OpenAI helpers: `add_usage` once per answered request."""

    def __init__(self, usage: Usage, embedding: bool):
        self._usage = usage
        self._embedding = embedding

    def add_usage(self, counts: Mapping[str, int]) -> None:
        prompt = int(counts.get("prompt_tokens", 0) or 0)
        if self._embedding:
            self._usage.add(embed_calls=1, embed_tokens=prompt)
        else:
            completion = int(counts.get("completion_tokens", 0) or 0)
            self._usage.add(llm_calls=1, prompt_tokens=prompt, completion_tokens=completion)


LLM_TIMEOUT_S = 1800  # library default 240 s (worker 2x); deployment setting (decisions.md)
EMBEDDING_TIMEOUT_S = 300  # library default 30 s


def _lightrag_env() -> None:
    """Keep the tokenizer's encoding file with the run caches, and set the request timeouts
    (the library reads LLM_TIMEOUT / EMBEDDING_TIMEOUT when a LightRAG is built); must run before
    first use."""
    os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(REPO / "cache" / "tiktoken"))
    os.environ.setdefault("LLM_TIMEOUT", str(LLM_TIMEOUT_S))
    os.environ.setdefault("EMBEDDING_TIMEOUT", str(EMBEDDING_TIMEOUT_S))


def _quiet_logs() -> None:
    """Silence the library's per-document INFO logging (it sets its own level at import, so this
    runs after the import); warnings and errors still show."""
    for name in ("lightrag", "nano-vectordb"):
        logging.getLogger(name).setLevel(logging.WARNING)


async def _cancel_pending() -> None:
    """Cancel every other task on the running loop: the library's queue workers outlive its
    storages and would otherwise be destroyed pending when the loop closes."""
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


# ---------------------------------------------------------------------- LightRAG
class LightRAGSystem(NativeSystem):
    """LightRAG on the file-based default storages, run on a private event loop; not safe for
    concurrent calls, and only one instance may be open per process."""

    name = "lightrag"

    def __init__(
        self,
        *args: Any,
        mode: str = QUERY_MODE,
        batch_size: int = 1,
        options: dict[str, Any] | None = None,
        **kw: Any,
    ):
        self.mode = mode
        self.batch_size = max(1, batch_size)
        self.options = dict(options or {})  # extra `LightRAG(...)` arguments, e.g. concurrency
        self._rag: LightRAG | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._revision: dict[str, int] = {}
        self._event_ts: dict[str, str] = {}
        kw.setdefault("top_k", QUERY_TOP_K)
        super().__init__(*args, **kw)

    @property
    def retains_raw_text(self) -> bool:
        return True

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "lightrag": "1.5.7",
            "storages": "JsonKV+NanoVectorDB+NetworkX+JsonDocStatus",
            "query": {
                "mode": self.mode,
                "top_k": self.top_k,
                "chunk_top_k": QUERY_CHUNK_TOP_K,
                "max_entity_tokens": QUERY_MAX_ENTITY_TOKENS,
                "max_relation_tokens": QUERY_MAX_RELATION_TOKENS,
                "max_total_tokens": QUERY_MAX_TOTAL_TOKENS,
                "rerank": False,
            },
            "batch_size": self.batch_size,
            "options": self.options,
        }

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        assert self._loop is not None
        return self._loop.run_until_complete(coro)

    @property
    def rag(self) -> LightRAG:
        assert self._rag is not None
        return self._rag

    # -------------------------------------------------------------- model hooks
    def _llm_func(self) -> Any:
        from lightrag.llm.openai import openai_complete_if_cache

        tracker = _UsageTracker(self.usage, embedding=False)

        async def llm(
            prompt: str,
            system_prompt: str | None = None,
            history_messages: list[dict[str, Any]] | None = None,
            **kw: Any,
        ) -> Any:
            kw.pop("hashing_kv", None)
            kw.setdefault("temperature", 0.0)
            kw["extra_body"] = {**self.llm_extra_body, **(kw.get("extra_body") or {})}
            if self.max_output_tokens is not None:
                kw["max_tokens"] = min(int(kw.get("max_tokens") or 10**9), self.max_output_tokens)
            return await openai_complete_if_cache(
                self.llm_model,
                prompt,
                system_prompt=system_prompt,
                history_messages=history_messages,
                base_url=self.llm_base_url,
                api_key=_api_key(),
                token_tracker=tracker,
                **kw,
            )

        return llm

    def _embed_func(self) -> Any:
        from lightrag.llm.openai import openai_embed

        tracker = _UsageTracker(self.usage, embedding=True)

        async def embed(texts: list[str], **_: Any) -> Any:
            return await openai_embed.func(
                texts,
                model=self.embed_model,
                base_url=self.embed_base_url,
                api_key=_api_key(),
                token_tracker=tracker,
            )

        return embed

    def _tokenizer(self) -> Any:
        """None: the library's tiktoken default. A hook for offline tests."""
        return None

    # -------------------------------------------------------------- store
    def _open(self) -> None:
        _lightrag_env()
        claim_process(self.store_dir)
        try:
            from lightrag import LightRAG
            from lightrag.utils import EmbeddingFunc

            _quiet_logs()
            self._loop = asyncio.new_event_loop()
            self._rag = LightRAG(
                working_dir=str(self.store_dir),
                workspace="",
                llm_model_func=self._llm_func(),
                llm_model_name=self.llm_model,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self.embed_dims,
                    func=self._embed_func(),
                    max_token_size=EMBED_MAX_TOKEN_SIZE,
                    model_name=self.embed_model,
                ),
                tokenizer=self._tokenizer(),
                **self.options,
            )
            self._run(self.rag.initialize_storages())
            self._load_revisions()
        except BaseException:
            self._close()
            raise

    def _storages(self) -> list[Any]:
        rag = self.rag
        return [
            rag.full_docs,
            rag.text_chunks,
            rag.full_entities,
            rag.full_relations,
            rag.entity_chunks,
            rag.relation_chunks,
            rag.entities_vdb,
            rag.relationships_vdb,
            rag.chunks_vdb,
            rag.chunk_entity_relation_graph,
            rag.llm_response_cache,
            rag.doc_status,
        ]

    async def _shutdown(self) -> None:
        for storage in self._storages():
            await storage.index_done_callback()
        await self.rag.finalize_storages()

    def _close(self) -> None:
        rag, loop = self._rag, self._loop
        if loop is None:  # never got as far as creating the library objects
            release_process(self.store_dir)
            return
        from lightrag.kg.shared_storage import finalize_share_data

        try:
            if rag is not None:
                self._run(self._shutdown())
        finally:
            self._rag, self._loop = None, None
            loop.run_until_complete(_cancel_pending())
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            finalize_share_data()
            release_process(self.store_dir)

    # -------------------------------------------------------------- revision map
    def _load_revisions(self) -> None:
        p = self.store_dir / REVISIONS_FILE
        raw = json.loads(p.read_text()) if p.is_file() else {}
        self._revision = {o: int(v["revision"]) for o, v in raw.items()}
        self._event_ts = {o: str(v["ts"]) for o, v in raw.items() if v.get("ts")}

    def _save_revisions(self) -> None:
        data = {
            o: {"revision": r, "ts": self._event_ts.get(o)}
            for o, r in sorted(self._revision.items())
        }
        tmp = self.store_dir / (REVISIONS_FILE + ".tmp")
        tmp.write_text(json.dumps(data, sort_keys=True))
        os.replace(tmp, self.store_dir / REVISIONS_FILE)

    async def _doc_rows(self) -> list[dict[str, Any]]:
        from lightrag.base import DocStatus

        docs = await self.rag.doc_status.get_docs_by_statuses(list(DocStatus))
        return [
            {"doc_id": k, "status": _value(v.status), "file_path": v.file_path}
            for k, v in sorted(docs.items())
        ]

    async def _stored_ts(self, doc_ids: list[str]) -> dict[str, str | None]:
        rows = await self.rag.full_docs.get_by_ids(doc_ids)
        return {
            d: head_ts(str((r or {}).get("content", "")))
            for d, r in zip(doc_ids, rows, strict=True)
        }

    def _reconcile(self) -> None:
        keep, purge = reconcile_plan(self._run(self._doc_rows()))
        for doc_id in purge:
            self._run(self._delete(doc_id))
        stale = [
            o for o, r in keep.items() if self._revision.get(o) != r or o not in self._event_ts
        ]
        found = self._run(self._stored_ts(stale)) if stale else {}
        self._event_ts = {
            o: t for o in keep if (t := found.get(o) or self._event_ts.get(o)) is not None
        }
        self._revision = keep
        self._save_revisions()
        for oid, rev in keep.items():
            if doc_file_path(oid, rev) not in self.ledger:
                self.ledger.record(doc_file_path(oid, rev), reconciled=True)

    # -------------------------------------------------------------- ingest
    async def _insert_many(self, items: Sequence[tuple[str, int, str]]) -> list[Exception | None]:
        """One `ainsert` for documents of distinct objects (the library runs up to
        `max_parallel_insert` of them at a time). Each document is then checked on its own and a
        failed one is removed again, so a retry starts clean."""
        ids = [oid for oid, _, _ in items]
        paths = [doc_file_path(oid, rev) for oid, rev, _ in items]
        raised: Exception | None = None
        try:
            await self.rag.ainsert([text for _, _, text in items], ids=ids, file_paths=paths)
        except Exception as exc:
            raised = exc
        errors: list[Exception | None] = []
        for oid, path in zip(ids, paths, strict=True):
            try:
                if raised is not None:
                    raise raised
                check_ingested(await self.rag.doc_status.get_by_id(oid), path)
                errors.append(None)
            except Exception as exc:
                with contextlib.suppress(Exception):
                    await self.rag.adelete_by_doc_id(oid)
                errors.append(exc)
        return errors

    async def _delete(self, doc_id: str) -> None:
        res = await self.rag.adelete_by_doc_id(doc_id)
        if res.status not in ("success", "not_found"):
            raise RuntimeError(f"delete {doc_id}: {res.status} {str(res.message)[:500]}")

    def _ingest_batch(self, batch: list[Event]) -> None:
        """Deletes first (one at a time), then one insert call for the batch. Failures are
        raised or recorded only after the store's bookkeeping reflects what did happen."""
        failed: list[tuple[str, Exception]] = []
        todo: list[tuple[Event, list[Action]]] = []
        for e in batch:
            actions = plan_event(self._revision, e)
            try:
                for a in actions:
                    if a.kind == "delete":
                        self._run(self._delete(a.object_id))
                        self._revision.pop(a.object_id, None)
                        self._event_ts.pop(a.object_id, None)
            except Exception as exc:
                failed.append((event_key(e), exc))
                continue
            todo.append((e, actions))
        inserting = [(e, a) for e, acts in todo for a in acts if a.kind == "insert"]
        items = [(a.object_id, a.revision, event_text(e)) for e, a in inserting]
        errors = self._run(self._insert_many(items)) if items else []
        by_event = {event_key(e): err for (e, _), err in zip(inserting, errors, strict=True)}
        for e, actions in todo:
            err = by_event.get(event_key(e))
            if err is not None:
                failed.append((event_key(e), err))
                continue
            for a in actions:
                if a.kind == "insert":
                    self._revision[a.object_id] = a.revision
                    self._event_ts[a.object_id] = fmt_ts(e.ts)
            self.ledger.record(
                event_key(e),
                event_id=e.event_id,
                actions=[f"{a.kind}:{a.object_id}" for a in actions],
            )
        self._save_revisions()
        for key, err in failed:
            self._fail(key, err)

    def _ingest_events(self, events: list[Event]) -> None:
        for batch in distinct_batches(events, self.batch_size):
            self._ingest_batch(batch)

    # -------------------------------------------------------------- serve
    def _search(self, query: str) -> list[Hit]:
        from lightrag import QueryParam

        settings: dict[str, Any] = {
            "mode": self.mode,
            "top_k": self.top_k,
            "chunk_top_k": QUERY_CHUNK_TOP_K,
            "max_entity_tokens": QUERY_MAX_ENTITY_TOKENS,
            "max_relation_tokens": QUERY_MAX_RELATION_TOKENS,
            "max_total_tokens": QUERY_MAX_TOTAL_TOKENS,
            "enable_rerank": False,
        }
        res = self._run(self.rag.aquery_data(query, QueryParam(**settings)))
        return lightrag_hits(res.get("data") or {}, self._revision, self._event_ts)

    # -------------------------------------------------------------- export
    async def _export(self) -> list[dict[str, Any]]:
        from lightrag.base import DocStatus
        from lightrag.utils import make_relation_chunk_key

        rag = self.rag
        rows: list[dict[str, Any]] = []
        docs = await rag.doc_status.get_docs_by_statuses(list(DocStatus))
        ids = sorted(docs)
        for doc_id, full in zip(ids, await rag.full_docs.get_by_ids(ids), strict=True):
            status = {**asdict(docs[doc_id]), "status": _value(docs[doc_id].status)}
            rows.append({"kind": "document", "doc_id": doc_id, **status, "full_doc": full})
        chunk_ids = [c for d in ids for c in (docs[d].chunks_list or [])]
        chunk_rows = await rag.text_chunks.get_by_ids(chunk_ids)
        for chunk_id, chunk in zip(chunk_ids, chunk_rows, strict=True):
            rows.append({"kind": "chunk", "chunk_id": chunk_id, **(chunk or {})})
        graph = rag.chunk_entity_relation_graph
        nodes, edges = await graph.get_all_nodes(), await graph.get_all_edges()
        node_track = await rag.entity_chunks.get_by_ids([str(n["id"]) for n in nodes])
        for node, track in zip(nodes, node_track, strict=True):
            rows.append({"kind": "entity", **node, "tracked": track})
        keys = [make_relation_chunk_key(str(e["source"]), str(e["target"])) for e in edges]
        for edge, track in zip(edges, await rag.relation_chunks.get_by_ids(keys), strict=True):
            rows.append({"kind": "relation", **edge, "tracked": track})
        return rows

    def export_raw(self) -> list[dict[str, Any]]:
        """Every document (status, chunk ids, full text), chunk (text, tokens, source document and
        file path, cache keys), entity and relation (type / keywords, description, `source_id`,
        `file_path`, creation time, truncation flag) with its full chunk-tracking row."""
        return self._run(self._export())

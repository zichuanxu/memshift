"""Cognee behind the MemShift interface (plan §4.5, §6 E5; prereg v2 §3, substitute for Letta).

Cognee 1.6.1 (topoteretes) is knowledge-graph memory for agents: `add` stores a document,
`cognify` chunks it, has the LLM extract a graph and a summary per chunk and indexes everything,
`search` retrieves. Paths below are relative to the installed `cognee` package. The library is
optional (`docker/native/cognee.txt`) and imported lazily.

Storage (embedded, under `store_dir`, no server). `config.system_root_directory(store/system)`
moves the relational store (SQLite `cognee_db`, relational/config.py:22), the graph store
(Ladybug, the Kuzu fork that is the default, graph/config.py:47) and the vector store (LanceDB,
vector/config.py:32) to `store/system/databases/` (api/v1/config/config.py:190-216);
`config.data_root_directory(store/data)` holds one text file per data item (config.py:218-228).
Two switches keep it embedded and directory-copyable: `ENABLE_BACKEND_ACCESS_CONTROL=false` (unset
it is on whenever the stores support it, one graph and vector database per dataset,
context_global_variables.py:98-107) and `set_graph_database_subprocess_enabled(False)` /
`set_vector_db_subprocess_enabled(False)` (default: a worker process per store, graph/config.py:62,
vector/config.py:37, config.py:291-308, 612-624). Session cache, feedback, telemetry
(shared/utils.py:400) and the first-run LLM / embedding probes (setup_and_check_environment.py:
57-69) are off, so no request is made that the events did not cause. `_close()` evicts the graph
and vector engines with `force_close=True` (utils/engine_cache_ops.py:45-55), waits for the close
tasks the cache schedules on the running loop (utils/closing_lru_cache.py:147, 271-275), which
reach `LadybugAdapter.close` (ladybug/adapter.py:912-982) and free its file lock, and disposes the
SQLAlchemy engine; SQLite runs in WAL mode, so a snapshot is taken only after that. `data` rows
keep absolute `file://` paths (ingest_data.py:448-449); on opening a copied store they are
rebased to its own data root, so a later delete never touches the snapshot's files
(SqlAlchemyAdapter.py:440-478).

LLM. `config.set_llm_*`: provider `openai` with our base URL, model `openai/<llm_model>` (LiteLLM
routes a custom endpoint on that prefix), `llm_args` = temperature 0, `max_tokens` (the shared
output cap) and `extra_body` = the shared thinking-off body; `llm_args` is merged into every call
(llm/config.py:193, structured_output_framework/litellm_native/native_adapter.py:483). Structured
output in 1.6.1 does not go through instructor or BAML by default: `structured_output_framework`
is `litellm_native` (llm/config.py:107, LLMGateway.py:112-158). It sends the Pydantic model as
`response_format` only when `litellm.supports_response_schema(model)` (native_adapter.py:123-134,
384-449), which is false for an unregistered model behind a custom endpoint; then it sends
`response_format={"type": "json_object"}` with the schema in the system prompt, validates, and
feeds a validation error back for up to three attempts (native_adapter.py:300-369, 80). On
Qwen3.5-9B that default fails silently: the model answers with the schema it was shown, which
validates as an empty KnowledgeGraph (every field has a default, shared/data_models.py:73-77), so
cognify stores chunks and summaries but no entity or edge (first smoke run: 58 records, no entity
node). `register_schema_support` therefore registers the served model with LiteLLM as
schema-capable (`json_schema=True`, the default): `response_format` then carries the Pydantic
model, LiteLLM sends it as `json_schema` and vLLM decodes under it. Either way no request carries
`tools`, so the endpoint needs no tool-call parser. The opt-in `instructor` framework
(`structured_output="instructor"`) would call `instructor.from_litellm(litellm.acompletion)` with
its default TOOLS mode for provider `openai` (litellm_instructor/llm/openai/adapter.py:113,
instructor/core/client.py:880 in the instructor package); the `custom` provider uses `json_mode`
(instructor_modes.py:18). A call that keeps failing is retried until both two attempts and 240 s
have passed (llm/retry_config.py:33-38). Chat calls are counted at `litellm.acompletion`, through
which every request goes; embeddings at the OpenAI client of the embedding engine.

Embeddings. `openai_compatible` provider (vector/embeddings/get_embedding_engine.py:112-123): the
`openai` SDK against our embedding endpoint, batches of 36 (embeddings/config.py:153-154), 1024
dimensions set explicitly. The tokenizer for chunk sizing falls back to `cl100k_base`
(tokenizer/resolver.py:87-114; LiteLLM ships that BPE file and points `TIKTOKEN_CACHE_DIR` at it,
so it needs no download); counts only decide chunking: the budget is min(8191, 16384 // 2) tokens
(llm/utils.py:14-56), so almost every record is one chunk and a very long one is cut in two.

Integration contract (prereg decisions 2026-09-29: the system's own sync path, not a deletion the
benchmark performs for it). Every source object is ONE data item pinned to `Data.id =
uuid5(object_id)` through `DataItem(data=event_text, label=object_id, external_metadata=
event_metadata, data_id=...)` (data_item.py:14-22, ingest_data.py:277-283, 484-507). A new
revision is `datasets.delete_data` of the object's current item, then an add of the new text; a
delete event is `delete_data`. (`cognee.delete` is a deprecated wrapper of it, keyed by the same
`(data_id, dataset_id)`, api/v1/delete/__init__.py:8-38. Adding changed content under an existing
id would refresh the row and reset its status but leave the old graph in place,
ingest_data.py:421-478; `cognee.update` is the library's own replace: chunk diff for text, else
`forget(memory_only)` + pinned re-add + cognify, api/v1/update/update.py:215-309, and for one-chunk
records it does the same work as delete + add.) Events are applied in batches of distinct objects
(`distinct_batches`): deletes and replacements first, then one `add` and one incremental
`cognify` per batch. Cognify skips items whose status says completed
(run_tasks_data_item.py:186-200), so its cost is linear in new items: per chunk one graph
extraction and one summary call, run together (tasks/graph/extract_graph_and_summarize.py:23-37),
plus embeddings for the six indexed fields (DocumentChunk_text, TextSummary_text, Entity_name,
EntityType_name, EdgeType_relationship_name, TextDocument_name; tasks/storage/add_data_points.py:
329-346); `data_per_batch` items run concurrently. One failing item fails the run and rolls the
whole run back, healthy items included (run_tasks.py:292-346, modules/cognify/rollback.py:127-140),
so a batch is repeated without the items it names (`RAISE_INCREMENTAL_LOADING_ERRORS=false` lets
every item finish first, run_tasks_data_item.py:281; the repeated calls are counted). A failed
unit is deleted so the next cognify does not retry it.

Serve. The context a Cognee agent reads by default: `cognee.search`'s default type is
HYBRID_COMPLETION (api/v1/search/search.py:43), whose retrieval is three lanes rendered as
"Relevant passages" (chunk text), "Relevant entities" (name and edge bullets: the entity's vector
row carries only the indexed field, so no description or type, LanceDBAdapter.py:62-90) and
"Related facts" (EdgeType rows) (retrieval/hybrid_retriever.py:123-168,
hybrid/context.py:8-27, top_k 15 -> at most 10 per lane, search/methods/
get_search_type_retriever_instance.py:41-51). `only_context=True, verbose=True` stops before the
answer and returns the retrieved objects: one query embedding, no LLM call
(search/methods/get_retriever_output.py:95-108). Each passage, entity block and fact becomes one
Hit in that order. Source links: a chunk carries its document's `external_metadata` (our object
id and revision, DocumentChunk.py) and `document_id`; entity nodes and the edges quoted in their
blocks carry graph provenance keys (`source_ref:v1:<dataset>:<data>` for document-scope output,
`source_ref:v2:...:<chunk>` for chunk-scope output, infrastructure/databases/provenance/
source_refs.py:7-21) that the adapter reads with `get_node_delete_data` / `get_edge_delete_data`;
facts are EdgeType rows shared by relationship text and keep no source link.

What deleting or replacing an item does (the facts behind the predicted label). Provenance lives
in the graph (`mark_graph_provenance_if_empty` marks a fresh graph, add_data_points.py:166,
provenance/markers.py:48-71); output a chunk produced is stamped with the chunk-scope key, the
rest with the document-scope key (add_data_points.py:209-243). `delete_data` finds the item's keys
in nodes and edges (unified/unified_store_engine.py:163-211) and hard-deletes what no other owner
still holds, together with its vector rows; anything with another owner only loses this key
(unified/provenance_delete_planner.py:54-56, 83-93, 127-160). So the item's chunks, summaries,
document node, and every entity and edge only it produced disappear; an entity or edge shared
with another object stays. A shared node is rewritten by every writer (`MERGE ... ON MATCH SET
properties`, ladybug/adapter.py:1207-1221), so its description is the last writer's; a relationship
edge the graph already holds is not rewritten, its new owners only attach (add_data_points.py:
303-315, tasks/graph/extract_graph_from_data.py:171-180), so its text is the first writer's. Both
survive the writer's deletion with only the remaining owners' keys: source text can outlive its
source in shared entity descriptions and in the bullets of shared edges. Nothing checks an
extracted statement against the source record (free-text LLM extraction, no typed value, no
citation check).
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import os
import sqlite3
from collections.abc import Coroutine, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse
from uuid import UUID, uuid5

from memshift.adapters.native import (
    _EPOCH,
    Hit,
    MeteredOpenAI,
    NativeSystem,
    _api_key,
    distinct_batches,
    event_key,
    event_metadata,
    event_text,
)
from memshift.gen.model import Event, parse_ts
from memshift.runinfo import REPO

if TYPE_CHECKING:
    from asyncio import AbstractEventLoop

DATASET_NAME = "memshift"
DB_FILE = "cognee_db"
DEFAULT_TOP_K = 15  # `cognee.search` default (api/v1/search/search.py:49)
DEFAULT_BATCH_SIZE = 10
MAX_COGNIFY_PASSES = 4
DATA_ID_NAMESPACE = UUID("6d656d73-6869-6674-2d63-6f676e656500")
SEARCH_TYPE = "HYBRID_COMPLETION"
COMPLETED = "DATA_ITEM_PROCESSING_COMPLETED"

# Process-wide settings read while `cognee` is imported or when its config singletons are first
# built; set before the first import (setdefault: an explicit environment wins).
COGNEE_ENV: dict[str, str] = {
    "TELEMETRY_DISABLED": "1",  # shared/utils.py:400
    "ENABLE_BACKEND_ACCESS_CONTROL": "false",  # context_global_variables.py:98-107
    "CACHING": "false",  # infrastructure/databases/cache/config.py: session cache off
    "AUTO_FEEDBACK": "false",
    "COGNEE_SKIP_CONNECTION_TEST": "true",  # setup_and_check_environment.py:57-69
    "RAISE_INCREMENTAL_LOADING_ERRORS": "false",  # run_tasks_data_item.py:281
    "LITELLM_LOCAL_MODEL_COST_MAP": "True",
    "LOG_LEVEL": "ERROR",
}

_LOOP: AbstractEventLoop | None = None
_METER: SimpleNamespace = SimpleNamespace(usage=None, errors=0)


def _quiet_env() -> None:
    for k, v in COGNEE_ENV.items():
        os.environ.setdefault(k, v)
    os.environ.setdefault("COGNEE_LOGS_DIR", str(REPO / "cache" / "cognee" / "logs"))
    os.environ.setdefault("LLM_API_KEY", _api_key())
    os.environ.setdefault("EMBEDDING_API_KEY", _api_key())


def _loop() -> AbstractEventLoop:
    """One event loop for the whole process: Cognee keeps loop-bound module state (engine
    caches, locks, litellm's logging worker), so every system and every open / close cycle
    runs on the same loop."""
    global _LOOP
    if _LOOP is None or _LOOP.is_closed():
        _LOOP = asyncio.new_event_loop()
    return _LOOP


def run[T](coro: Coroutine[Any, Any, T]) -> T:
    return _loop().run_until_complete(coro)


# ---------------------------------------------------------------------- identities
def data_id_for(object_id: str) -> UUID:
    """The `Data.id` pinned for a source object (`DataItem.data_id`, ingest_data.py:277-283)."""
    return uuid5(DATA_ID_NAMESPACE, object_id)


def litellm_model(model: str) -> str:
    """LiteLLM routes a custom OpenAI-compatible endpoint on the `openai/` prefix."""
    return model if model.startswith("openai/") else f"openai/{model}"


def llm_args(temperature: float, max_tokens: int | None, extra_body: dict[str, Any] | None) -> dict:
    """Keys merged into every litellm call (`native_adapter.py:483`, `llm/config.py:193`)."""
    args: dict[str, Any] = {"temperature": temperature}
    if max_tokens is not None:
        args["max_tokens"] = max_tokens
    if extra_body:
        args["extra_body"] = extra_body
    return args


def source_ref_data_id(key: str) -> str | None:
    """Data id inside a graph provenance key: `source_ref:v1:<dataset>:<data>` (document scope)
    or `source_ref:v2:<dataset>:<data>:<chunk>` (chunk scope), provenance/source_refs.py:7-21."""
    parts = key.split(":")
    if len(parts) >= 4 and parts[0] == "source_ref" and parts[1] in ("v1", "v2"):
        return parts[3]
    return None


# ---------------------------------------------------------------------- stored documents
@dataclass(frozen=True)
class StoredDoc:
    data_id: str
    object_id: str
    revision: int
    ts: datetime | None
    completed: bool


def _hyphenated(raw: str) -> str:
    try:
        return str(UUID(str(raw)))
    except ValueError:
        return str(raw)


def read_store_docs(
    db_path: Path, dataset_name: str = DATASET_NAME
) -> tuple[str | None, list[StoredDoc]]:
    """The dataset id and the documents in Cognee's relational store, read with `sqlite3` alone
    (table `data`: `external_metadata` is the JSON we pinned, `pipeline_status` records
    cognify completion, ingest_data.py:484-507, run_tasks_data_item.py:186-200)."""
    if not db_path.is_file():
        return None, []
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,)).fetchone()
        if row is None:
            return None, []
        dataset_id = _hyphenated(row[0])
        docs: list[StoredDoc] = []
        query = "SELECT id, external_metadata, pipeline_status FROM data WHERE dataset_id = ?"
        for data_id, meta, status in con.execute(query, (row[0],)):
            md = json.loads(meta) if isinstance(meta, str) else (meta or {})
            if "object_id" not in md or "revision" not in md:
                continue
            state = json.loads(status) if isinstance(status, str) else (status or {})
            slot = (state.get("cognify_pipeline") or {}).get(dataset_id)
            if isinstance(slot, dict):
                slot = slot.get("status")
            raw_ts = md.get("ts")
            docs.append(
                StoredDoc(
                    _hyphenated(data_id),
                    str(md["object_id"]),
                    int(md["revision"]),
                    parse_ts(raw_ts) if isinstance(raw_ts, str) else None,
                    slot == COMPLETED,
                )
            )
        return dataset_id, docs
    except sqlite3.OperationalError:
        return None, []  # tables not created yet: nothing has been added
    finally:
        con.close()


def rebase_locations(db_path: Path, old_root: str, new_root: str) -> int:
    """Point `data.raw_data_location` / `original_data_location` at the data root of a copied
    store. Rows keep absolute `file://` paths (ingest_data.py:448-449); after a directory copy
    they would name the snapshot's files, which a later delete could otherwise touch
    (SqlAlchemyAdapter.py:440-478 removes a file only when it lies under the current root)."""
    if old_root == new_root or not db_path.is_file():
        return 0
    con = sqlite3.connect(db_path)
    try:
        n = 0
        for col in ("raw_data_location", "original_data_location"):
            cur = con.execute(
                f"UPDATE data SET {col} = REPLACE({col}, ?, ?) WHERE {col} LIKE ?",
                (old_root, new_root, f"%{old_root}%"),
            )
            n += cur.rowcount
        con.commit()
        return n
    finally:
        con.close()


# ---------------------------------------------------------------------- retrieval -> hits
def _payload(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    p = getattr(obj, "payload", None)
    return p if isinstance(p, dict) else {}


def _text(value: Any) -> str | None:
    if value is None or not isinstance(value, str | int | float | UUID):
        return None
    s = str(value).strip()
    return s or None


def _obj_id(obj: Any) -> str:
    return _text(_payload(obj).get("id")) or _text(getattr(obj, "id", None)) or ""


def refs_for_keys(
    keys: Iterable[str], by_data_id: Mapping[str, tuple[str, int, datetime | None]]
) -> list[tuple[str, int]]:
    out: list[tuple[str, int]] = []
    for key in keys:
        did = source_ref_data_id(key)
        doc = by_data_id.get(did) if did else None
        if doc is not None and (doc[0], doc[1]) not in out:
            out.append((doc[0], doc[1]))
    return out


def _latest(
    refs: Sequence[tuple[str, int]], by_data_id: Mapping[str, tuple[str, int, datetime | None]]
) -> datetime:
    stamps = [d[2] for d in by_data_id.values() if (d[0], d[1]) in refs and d[2] is not None]
    return max(stamps) if stamps else _EPOCH


def chunk_hit(
    chunk: Any, rank: int, by_data_id: Mapping[str, tuple[str, int, datetime | None]]
) -> Hit | None:
    """One passage of the hybrid context (`format_passages`, retrieval/hybrid/context.py:75-85):
    the chunk text as stored. Source: the chunk's own `external_metadata` (the JSON we pinned on
    the document, DocumentChunk.py `_inherit_document_external_metadata`) or, failing that, its
    `document_id`."""
    p = _payload(chunk)
    text = _text(p.get("text"))
    if not text:
        return None
    md: dict[str, Any] = {}
    raw = p.get("external_metadata")
    if isinstance(raw, str):
        try:
            md = json.loads(raw)
        except json.JSONDecodeError:
            md = {}
    elif isinstance(raw, dict):
        md = raw
    refs: list[tuple[str, int]] = []
    if "object_id" in md and "revision" in md:
        refs = [(str(md["object_id"]), int(md["revision"]))]
    elif (doc := by_data_id.get(_text(p.get("document_id")) or "")) is not None:
        refs = [(doc[0], doc[1])]
    raw_ts = md.get("ts")
    ts = parse_ts(raw_ts) if isinstance(raw_ts, str) else _latest(refs, by_data_id)
    return Hit(_obj_id(chunk), text, ts, refs, 1.0 / (rank + 1))  # the lanes return rank order


def entity_block(entity: Mapping[str, Any]) -> str:
    """One entity block as `_format_entity` renders it (retrieval/hybrid/entities.py:184-202)."""
    name = _text(entity.get("name"))
    if not name:
        return ""
    etype = _text(entity.get("type"))
    lines = [f"### {name} ({etype})" if etype else f"### {name}"]
    if desc := _text(entity.get("description")):
        lines.append(desc)
    for edge in entity.get("edges") or []:
        if text := _text(edge.get("text")):
            lines.append(f"- {text}")
    return "\n".join(lines)


def entity_edge_identities(entity: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    return [
        (str(e["source_id"]), str(e["target_id"]), str(e["relationship"]))
        for e in entity.get("edges") or []
        if e.get("source_id") and e.get("target_id") and e.get("relationship")
    ]


def hybrid_hits(
    retrieved: Mapping[str, Any],
    node_keys: Mapping[str, Sequence[str]],
    edge_keys: Mapping[tuple[str, str, str], Sequence[str]],
    by_data_id: Mapping[str, tuple[str, int, datetime | None]],
) -> list[Hit]:
    """The hybrid retriever's result (retrieval/hybrid_retriever.py:145-149: `chunks`,
    `entities`, `facts`) as blocks in the order Cognee renders them (passages, entities, facts:
    hybrid/context.py:8-27). `node_keys` / `edge_keys` are the graph-provenance keys of the
    entity nodes and of the edges quoted in the entity blocks; facts (EdgeType rows shared by
    relationship text) keep no source link."""
    hits: list[Hit] = []
    for rank, chunk in enumerate(retrieved.get("chunks") or []):
        if (h := chunk_hit(chunk, rank, by_data_id)) is not None:
            hits.append(h)
    for rank, ent in enumerate(retrieved.get("entities") or []):
        text = entity_block(ent)
        if not text:
            continue
        keys = list(node_keys.get(str(ent.get("id")), ()))
        for ident in entity_edge_identities(ent):
            keys += list(edge_keys.get(ident, ()))
        refs = refs_for_keys(keys, by_data_id)
        hits.append(
            Hit(
                str(ent.get("id") or f"entity{rank}"),
                text,
                _latest(refs, by_data_id),
                refs,
                1.0 / (rank + 1),
            )
        )
    for rank, fact in enumerate(retrieved.get("facts") or []):
        if fact_text := _text(fact.get("text")):
            hits.append(
                Hit(str(fact.get("id") or f"fact{rank}"), fact_text, _EPOCH, [], 1.0 / (rank + 1))
            )
    return hits


def errored_data_ids(run_info: Any) -> set[str]:
    """Data items whose cognify run errored (run_tasks.py:210-258: every item is drained, then
    the run fails as a whole)."""
    out: set[str] = set()
    for entry in getattr(run_info, "data_ingestion_info", None) or []:
        info = entry.get("run_info") if isinstance(entry, dict) else None
        if getattr(info, "status", "") == "PipelineRunErrored" and entry.get("data_id"):
            out.add(str(entry["data_id"]))
    return out


def run_failed(run_info: Any) -> bool:
    return getattr(run_info, "status", "") == "PipelineRunErrored"


# ---------------------------------------------------------------------- usage metering
def _install_meters() -> None:
    """Count chat calls at `litellm.acompletion`, the one function through which the default
    structured-output path sends every request (native_adapter.py:283-296, 337-348); embeddings
    are counted at the embedding engine's OpenAI client (see `CogneeSystem._open`)."""
    import litellm

    if getattr(litellm.acompletion, "_memshift_meter", False):
        return
    inner = litellm.acompletion

    async def acompletion(*args: Any, **kw: Any) -> Any:
        try:
            resp = await inner(*args, **kw)
        except Exception:
            _METER.errors += 1
            raise
        u = getattr(resp, "usage", None)
        if _METER.usage is not None:
            _METER.usage.add(
                llm_calls=1,
                prompt_tokens=int(getattr(u, "prompt_tokens", 0) or 0),
                completion_tokens=int(getattr(u, "completion_tokens", 0) or 0),
            )
        return resp

    acompletion._memshift_meter = True  # type: ignore[attr-defined]
    litellm.acompletion = acompletion


def register_schema_support(model: str) -> None:
    """Tell LiteLLM's capability table that the endpoint takes `response_format=json_schema`
    (vLLM does, as constrained decoding), so Cognee's schema-native path is used instead of the
    prompted-JSON fallback (native_adapter.py:123-134, 384-449). Without it Qwen3.5-9B echoes
    the schema it is shown in the prompt, which validates as an empty graph (see `CogneeSystem`)."""
    import litellm

    bare = model.removeprefix("openai/")
    litellm.register_model(
        {bare: {"litellm_provider": "openai", "mode": "chat", "supports_response_schema": True}}
    )


# ---------------------------------------------------------------------- system
class CogneeSystem(NativeSystem):
    """Cognee 1.6.1 on embedded SQLite + LanceDB + Ladybug under `store_dir`."""

    name = "cognee"

    def __init__(
        self,
        *args: Any,
        top_k: int | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        structured_output: str = "litellm_native",
        json_schema: bool = True,
        **kw: Any,
    ):
        self.batch_size = max(1, batch_size)
        self.structured_output = structured_output
        self.json_schema = json_schema
        self._current: dict[str, StoredDoc] = {}
        self._dataset_id: str | None = None
        self._open_now = False
        super().__init__(*args, top_k=DEFAULT_TOP_K if top_k is None else top_k, **kw)
        self.stats["llm_errors"] = 0

    # ------------------------------------------------------------ properties
    @property
    def retains_raw_text(self) -> bool:
        return True

    @property
    def system_root(self) -> Path:
        return self.store_dir.resolve() / "system"

    @property
    def data_root(self) -> Path:
        return self.store_dir.resolve() / "data"

    @property
    def db_path(self) -> Path:
        return self.system_root / "databases" / DB_FILE

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "cognee": "1.6.1",
            "graph_store": "ladybug (in process)",
            "vector_store": "lancedb (in process)",
            "relational_store": "sqlite",
            "structured_output": self.structured_output,
            "response_format": "json_schema" if self.json_schema else "json_object",
            "search_type": SEARCH_TYPE,
            "batch_size": self.batch_size,
            "extractor": "llm",
        }

    # ------------------------------------------------------------ open / close
    def _configure(self) -> None:
        import cognee

        c = cognee.config
        self.system_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)
        c.system_root_directory(str(self.system_root))
        c.data_root_directory(str(self.data_root))
        c.set_graph_database_subprocess_enabled(False)
        c.set_vector_db_subprocess_enabled(False)
        c.set_llm_provider("openai")
        c.set_llm_model(litellm_model(self.llm_model))
        c.set_llm_endpoint(self.llm_base_url)
        c.set_llm_api_key(_api_key())
        c.set_llm_config(
            {
                "structured_output_framework": self.structured_output,
                "llm_args": llm_args(0.0, self.max_output_tokens, self.llm_extra_body),
            }
        )
        c.set_embedding_config(
            {
                "embedding_provider": "openai_compatible",
                "embedding_model": self.embed_model,
                "embedding_dimensions": self.embed_dims,
                "embedding_endpoint": self.embed_base_url,
                "embedding_api_key": _api_key(),
            }
        )

    def _open(self) -> None:
        _quiet_env()
        marker = self.store_dir / "cognee_paths.json"
        paths = {"data_root": str(self.data_root)}
        if marker.is_file():
            old = json.loads(marker.read_text()).get("data_root", "")
            rebase_locations(self.db_path, old, paths["data_root"])
        marker.write_text(json.dumps(paths))
        self._configure()
        _install_meters()
        if self.json_schema:
            register_schema_support(self.llm_model)
        _METER.usage = self.usage
        from cognee.infrastructure.databases.vector.embeddings.get_embedding_engine import (
            get_embedding_engine,
        )

        engine = get_embedding_engine()
        client = getattr(engine, "_client", None)
        if client is None:
            raise RuntimeError("the embedding engine keeps no OpenAI client to meter")
        if isinstance(client, MeteredOpenAI):
            client = client._inner
        engine._client = MeteredOpenAI(client, self.usage, is_async=True)
        self._open_now = True

    def _close(self) -> None:
        if not self._open_now:
            return
        run(self._release())
        self._open_now = False
        _METER.usage = None
        wal = self.db_path.parent / f"{DB_FILE}-wal"
        if wal.exists():  # engine disposed but a WAL is left: fold it into the main file
            con = sqlite3.connect(self.db_path)
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.close()

    async def _release(self) -> None:
        """Dispose every embedded store so no lock or file handle outlives the system:
        Ladybug's exclusive file lock (graph engine cache), LanceDB, and the SQLite engine."""
        from cognee.infrastructure.databases.graph.config import get_graph_context_config
        from cognee.infrastructure.databases.graph.get_graph_engine import graph_engine_cache
        from cognee.infrastructure.databases.relational import get_relational_engine
        from cognee.infrastructure.databases.relational.create_relational_engine import (
            create_relational_engine,
        )
        from cognee.infrastructure.databases.utils import closing_lru_cache
        from cognee.infrastructure.databases.vector.config import get_vectordb_context_config
        from cognee.infrastructure.databases.vector.create_vector_engine import vector_engine_cache

        graph_engine_cache.evict(force_close=True, **get_graph_context_config())
        vector_engine_cache.evict(force_close=True, **get_vectordb_context_config())
        pending = list(closing_lru_cache._PENDING_CLOSE_TASKS)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        await get_relational_engine().engine.dispose()
        create_relational_engine.cache_clear()

    # ------------------------------------------------------------ bookkeeping
    def _reconcile(self) -> None:
        dataset_id, docs = read_store_docs(self.db_path)
        self._dataset_id = dataset_id
        self._current = {d.object_id: d for d in docs}
        for d in docs:
            key = f"{d.object_id}#{d.revision}"
            if d.completed and key not in self.ledger:
                self.ledger.record(key, data_id=d.data_id, reconciled=True)

    def _by_data_id(self) -> dict[str, tuple[str, int, datetime | None]]:
        return {d.data_id: (d.object_id, d.revision, d.ts) for d in self._current.values()}

    # ------------------------------------------------------------ ingest
    def _ingest_events(self, events: list[Event]) -> None:
        errors0 = _METER.errors
        try:
            for batch in distinct_batches(events, self.batch_size):
                run(self._apply_batch(batch))
        finally:
            self.stats["llm_errors"] += _METER.errors - errors0

    async def _delete(self, data_id: str) -> None:
        """Remove one data item with its chunks, summaries, and every graph node, edge and vector
        row it alone owns (`datasets.delete_data`, api/v1/datasets/datasets.py:217-313)."""
        import cognee

        await cognee.datasets.delete_data(
            dataset_id=UUID(str(self._dataset_id)), data_id=UUID(data_id)
        )

    async def _add(self, specs: list[dict[str, Any]]) -> str:
        """`cognee.add` of the batch, one `DataItem` per object with its pinned id (ingest_data.py:
        277-283, 328-333); returns the dataset id."""
        import cognee
        from cognee.tasks.ingestion.data_item import DataItem

        info = await cognee.add([DataItem(**s) for s in specs], dataset_name=DATASET_NAME)
        if run_failed(info):
            raise RuntimeError(f"add failed: {getattr(info, 'error_message', info)}")
        return str(info.dataset_id)

    async def _cognify_run(self) -> Any:
        """One incremental `cognee.cognify` over the dataset (default task list, LLM extractor)."""
        import cognee

        runs = await cognee.cognify(
            datasets=[UUID(str(self._dataset_id))],
            extractor="llm",
            data_per_batch=self.batch_size,
            raise_on_error=False,
        )
        return next(iter(runs.values())) if isinstance(runs, dict) and runs else runs

    async def _apply_batch(self, batch: list[Event]) -> None:
        """One batch of distinct objects: deletes and replacements first (a new revision is the
        delete of the object's current data plus an add, a delete event is the delete), then one
        add and one cognify for the batch, then the ledger. An item an interrupted pass added but
        did not cognify is cognified, not added again."""
        done: list[Event] = []
        adds: list[tuple[Event, StoredDoc]] = []
        resumed: list[tuple[Event, StoredDoc]] = []
        for e in batch:
            have = self._current.get(e.object_id)
            try:
                if e.op == "delete":
                    if have is not None:
                        await self._delete(have.data_id)
                        del self._current[e.object_id]
                    done.append(e)
                elif have is not None and have.revision >= e.revision:
                    if have.completed:
                        done.append(e)
                    else:
                        resumed.append((e, have))
                else:
                    if have is not None:
                        await self._delete(have.data_id)
                        del self._current[e.object_id]
                    doc = StoredDoc(
                        str(data_id_for(e.object_id)), e.object_id, e.revision, e.ts, False
                    )
                    adds.append((e, doc))
            except Exception as exc:
                self._fail(event_key(e), exc)
        if adds:
            specs = [
                {
                    "data": event_text(e),
                    "label": e.object_id,
                    "external_metadata": event_metadata(e),
                    "data_id": UUID(doc.data_id),
                }
                for e, doc in adds
            ]
            try:
                self._dataset_id = await self._add(specs)
            except Exception as exc:
                for e, _ in adds:
                    self._fail(event_key(e), exc)
                adds = []
            for e, doc in adds:
                self._current[e.object_id] = doc
        todo = adds + resumed
        if todo:
            ok, failed = await self._cognify({doc.data_id for _, doc in todo})
            for e, doc in todo:
                if doc.data_id in ok:
                    done.append(e)
                    self._current[e.object_id] = dataclasses.replace(doc, completed=True)
                    continue
                self._current.pop(e.object_id, None)
                self._fail(event_key(e), RuntimeError(failed.get(doc.data_id, "cognify failed")))
        for e in done:
            self.ledger.record(event_key(e), event_id=e.event_id, op=e.op)

    async def _cognify(self, wanted: set[str]) -> tuple[set[str], dict[str, str]]:
        """Cognify the items just added. One item failing rolls the whole run back
        (run_tasks.py:292-346, cognify/rollback.py:98-140), which also resets the healthy items
        of that run; so a failed run is repeated without the items it names until it passes.
        Returns (completed data ids, {failed data id: reason}); on exhausting the passes nothing
        completed is reported."""
        remaining = set(wanted)
        failed: dict[str, str] = {}
        for _ in range(MAX_COGNIFY_PASSES):
            if not remaining:
                break
            info = await self._cognify_run()
            if not run_failed(info):
                return remaining, failed
            reason = str(getattr(info, "error_message", None) or "cognify run failed")
            bad = errored_data_ids(info) & remaining or set(remaining)
            failed.update({did: reason for did in bad})
            remaining -= bad
            await self._forget(bad)
        failed.update({did: "cognify passes exhausted" for did in remaining})
        await self._forget(remaining)
        return set(), failed

    async def _forget(self, data_ids: Iterable[str]) -> None:
        """A failed unit leaves nothing behind (and is not retried by the next cognify run)."""
        for did in data_ids:
            with contextlib.suppress(Exception):
                await self._delete(did)

    # ------------------------------------------------------------ serve
    def _search(self, query: str) -> list[Hit]:
        if self._dataset_id is None or not self._current:
            return []
        return run(self._search_async(query))

    async def _search_async(self, query: str) -> list[Hit]:
        import cognee
        from cognee.infrastructure.databases.graph import get_graph_engine
        from cognee.infrastructure.databases.provenance import EdgeIdentity
        from cognee.modules.retrieval.exceptions.exceptions import NoDataError

        try:
            res = await cognee.search(
                query,
                query_type=cognee.SearchType.HYBRID_COMPLETION,
                dataset_ids=[UUID(self._dataset_id or "")],
                top_k=self.top_k,
                only_context=True,
                verbose=True,
            )
        except NoDataError:
            return []
        retrieved: Mapping[str, Any] = {}
        for r in res or []:
            obj = r.get("objects_result") if isinstance(r, dict) else None
            if isinstance(obj, dict):
                retrieved = obj
                break
        entities = retrieved.get("entities") or []
        graph = await get_graph_engine()
        node_ids = [str(e["id"]) for e in entities if e.get("id")]
        idents = [ident for e in entities for ident in entity_edge_identities(e)]
        node_keys: dict[str, list[str]] = {}
        edge_keys: dict[tuple[str, str, str], list[str]] = {}
        if node_ids:
            for nid, data in (await graph.get_node_delete_data(node_ids)).items():
                node_keys[str(nid)] = list(data.source_ref_keys)
        if idents:
            edges = [EdgeIdentity(s, t, r) for s, t, r in idents]
            for ident, data in (await graph.get_edge_delete_data(edges)).items():
                edge_keys[(ident.source_id, ident.target_id, ident.relationship_name)] = list(
                    data.source_ref_keys
                )
        return hybrid_hits(retrieved, node_keys, edge_keys, self._by_data_id())

    # ------------------------------------------------------------ export
    def export_raw(self) -> list[dict[str, Any]]:
        return run(self._export())

    async def _export(self) -> list[dict[str, Any]]:
        import lancedb
        from cognee.infrastructure.databases.graph import get_graph_engine

        rows: list[dict[str, Any]] = []
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            for r in con.execute("SELECT * FROM data"):
                d = dict(r)
                path = urlparse(str(d.get("raw_data_location") or "")).path
                p = Path(unquote(path)) if path else None
                d["raw_text"] = p.read_text(encoding="utf-8") if p and p.is_file() else None
                rows.append({"kind": "data", **d})
            for r in con.execute("SELECT * FROM provenance_edge_evidence"):
                rows.append({"kind": "evidence", **dict(r)})
        finally:
            con.close()
        graph = await get_graph_engine()
        dataset = self._dataset_id or ""
        node_refs = await graph.find_node_source_refs_by_dataset(dataset)
        edge_refs = {
            (i.source_id, i.target_id, i.relationship_name): list(refs)
            for i, refs in (await graph.find_edge_source_refs_by_dataset(dataset)).items()
        }
        nodes, edges = await graph.get_graph_data()
        for nid, props in nodes:
            rows.append(
                {
                    "kind": "node",
                    "id": str(nid),
                    **props,
                    "source_refs": node_refs.get(str(nid), []),
                }
            )
        for src, dst, rel, props in edges:
            rows.append(
                {
                    "kind": "edge",
                    "source": str(src),
                    "target": str(dst),
                    "relationship": rel,
                    **props,
                    "source_refs": edge_refs.get((str(src), str(dst), str(rel)), []),
                }
            )
        lance = lancedb.connect(str(self.system_root / "databases" / "cognee.lancedb"))
        names = lance.table_names() if hasattr(lance, "table_names") else []
        for name in names:
            for r in lance.open_table(name).to_arrow().to_pylist():
                vec = r.pop("vector", None)
                rows.append(
                    {"kind": "vector", "collection": name, "dim": len(vec or []), **_plain(r)}
                )
        return rows


def _plain(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, str | int | float | bool | type(None) | list | dict):
            out[k] = v
        else:
            out[k] = str(v)
    return out

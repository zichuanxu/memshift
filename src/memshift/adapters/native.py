"""Native memory systems behind the MemShift interface (E5): Mem0 and Graphiti.

Both use their own extraction path with the LLM and the embedder pointed at our OpenAI-compatible
vLLM endpoints, and both persist to a local directory so that one prefix ingest can be branched:
ingest the prefix once, `snapshot()` the store, and `restore()` it into every branch. The native
libraries are optional (`uv sync --extra native`) and imported lazily.

Mem0 (mem0ai 2.2.1, `Memory.from_config`): OpenAI LLM + OpenAI embedder with custom base URLs,
  embedded Qdrant (path mode) and a SQLite history DB under `store_dir`. Each source event is one
  `add(messages=[user message], user_id, metadata, infer=True)`: the v3 additive pipeline makes
  one extraction LLM call per add (mem0/memory/main.py:881-1220) plus embedding calls. Mem0
  keeps only the extracted facts and the last 10 raw messages, so with `keep_raw=True` (the
  default, the configuration of the experiments) the raw record is also archived in the same store
  with `infer=False` under a separate user scope that extraction and serving never search.
  Without spaCy (`mem0ai[nlp]`) and fastembed (`mem0ai[extras]`), which this install omits, Mem0
  falls back to dense-only search with no entity boost.
Graphiti (graphiti-core 0.30.2): embedded Kuzu driver on `store_dir/graph.kuzu`, OpenAIGeneric
  LLM client, OpenAI embedder and reranker, all on our endpoints. Each event is one
  `add_episode(name=object_id#revision, episode_body=text, reference_time=ts,
  source_description=source)`; episodes are added sequentially as Graphiti requires, or with
  `batch_size > 1` through `add_episode_bulk` over batches of distinct objects. LLM calls per
  episode (no custom entity/edge types, graphiti.py:1043-1288 and utils/maintenance/): 1 entity
  extraction + 0-1 node dedup + 1 edge extraction + per extracted edge 0-1 dedup/contradiction
  call and 0-1 timestamp call + ceil(k/30) summary calls for the k nodes whose summaries outgrow
  2000 chars, i.e. about 2 + 2E for E extracted facts (typically 5-15 calls per record). Every
  prompt carries the previous 10 episodes, so both systems need a serving context well above 16k
  tokens for records of ~1.4k tokens.

Idempotency: every ingested unit is keyed `object_id#revision` in `store_dir/ledger.jsonl`
(fsynced per unit); on open, units already in the store (Mem0 payload metadata, Graphiti episode
names) are reconciled into the ledger, so a resumed ingest never adds a unit twice. Usage
(LLM calls and tokens, embedding calls and tokens) is metered by wrapping the libraries' OpenAI
clients; counts and the cost vector are saved with the store, so a restored branch starts from
the prefix cost.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import threading
import time
from collections.abc import Coroutine, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, TypeVar

from memshift.gen.model import Event, World, fmt_ts, parse_ts
from memshift.holdmem.system import READER_BUDGET, READER_DOC_CAP, ServeResult
from memshift.probes import Probe
from memshift.retrieval import Doc, pack
from memshift.runinfo import REPO

if TYPE_CHECKING:
    from graphiti_core import Graphiti
    from graphiti_core.driver.kuzu_driver import KuzuDriver

T = TypeVar("T")

STORE_MARKER = "memshift_native.json"
LEDGER_FILE = "ledger.jsonl"
STATE_FILE = "state.json"
FAILURES_FILE = "failures.jsonl"
DEFAULT_EXTRA_BODY: dict[str, Any] = {"chat_template_kwargs": {"enable_thinking": False}}
DEFAULT_TOP_K = 100
DEFAULT_MAX_OUTPUT_TOKENS = 4096
DEFAULT_EMBED_DIMS = 1024  # Qwen/Qwen3-Embedding-0.6B
_EPOCH = datetime(1970, 1, 1)


# ---------------------------------------------------------------------- event conversion
def event_key(e: Event) -> str:
    return f"{e.object_id}#{e.revision}"


def parse_key(key: str) -> tuple[str, int] | None:
    oid, sep, rev = key.rpartition("#")
    if not sep or not oid or not rev.isdigit():
        return None
    return oid, int(rev)


def event_text(e: Event) -> str:
    """The text a native system ingests for one source event."""
    if e.op == "delete":
        return f"Record {e.object_id} was deleted on {e.ts:%Y-%m-%d %H:%M}."
    head = f"[{e.source} record {e.object_id} | revision {e.revision} | {e.ts:%Y-%m-%d %H:%M}]"
    return f"{head}\n{e.payload.get('text', '')}"


def event_metadata(e: Event) -> dict[str, Any]:
    return {
        "object_id": e.object_id,
        "revision": e.revision,
        "source": e.source,
        "ts": fmt_ts(e.ts),
        "op": e.op,
    }


def distinct_batches(events: Sequence[Event], size: int) -> list[list[Event]]:
    """Consecutive batches of at most `size` events with no object twice in a batch, so that
    per-object order is preserved when a batch is processed as one unit."""
    out: list[list[Event]] = []
    cur: list[Event] = []
    seen: set[str] = set()
    for e in events:
        if cur and (len(cur) >= size or e.object_id in seen):
            out.append(cur)
            cur, seen = [], set()
        cur.append(e)
        seen.add(e.object_id)
    if cur:
        out.append(cur)
    return out


def _naive_utc(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _minute(dt: datetime) -> str:
    return f"{dt:%Y-%m-%d %H:%M}"


# ---------------------------------------------------------------------- serve-side hits
@dataclass
class Hit:
    hit_id: str
    text: str
    ts: datetime
    refs: list[tuple[str, int]] = field(default_factory=list)
    score: float = 0.0


def _metadata_refs(md: dict[str, Any]) -> list[tuple[str, int]]:
    if "object_id" in md and "revision" in md:
        return [(str(md["object_id"]), int(md["revision"]))]
    out = []
    for r in md.get("source_refs") or []:
        if isinstance(r, dict) and "object_id" in r and "revision" in r:
            out.append((str(r["object_id"]), int(r["revision"])))
    return out


def mem0_hit(row: dict[str, Any]) -> Hit:
    """One Mem0 search result -> text block. Wall-clock `created_at` is not shown: ingestion
    replays simulated time, so the event time from the stored metadata is the meaningful one."""
    md = dict(row.get("metadata") or {})
    refs = _metadata_refs(md)
    raw_ts = md.get("ts") or md.get("event_time")
    ts = parse_ts(raw_ts) if isinstance(raw_ts, str) else None
    head = ["memory"]
    if "source" in md and refs:
        head.append(f"source {md['source']} {refs[0][0]} r{refs[0][1]}")
    elif refs:
        head.append("source " + ", ".join(f"{o} r{r}" for o, r in refs[:6]))
    if ts is not None:
        head.append(f"event {_minute(ts)}")
    text = f"[{' | '.join(head)}] {row.get('memory', '')}"
    return Hit(str(row.get("id", "")), text, ts or _EPOCH, refs, float(row.get("score") or 0.0))


def graphiti_hit(
    uuid: str,
    fact: str,
    valid_at: datetime | None,
    invalid_at: datetime | None,
    expired: bool,
    source_keys: Sequence[str],
    reference_time: datetime | None = None,
    score: float = 0.0,
) -> Hit:
    """One Graphiti edge -> text block with its validity interval and source episodes."""
    valid_at, invalid_at = _naive_utc(valid_at), _naive_utc(invalid_at)
    refs = [p for k in source_keys if (p := parse_key(k)) is not None]
    head = ["fact"]
    if valid_at is not None:
        head.append(f"valid from {_minute(valid_at)}")
    if invalid_at is not None:
        head.append(f"invalid from {_minute(invalid_at)}")
    if expired:
        head.append("expired")
    if refs:
        head.append("source " + ", ".join(f"{o} r{r}" for o, r in refs[:6]))
    ts = valid_at or _naive_utc(reference_time) or _EPOCH
    return Hit(uuid, f"[{' | '.join(head)}] {fact}", ts, refs, score)


# ---------------------------------------------------------------------- bookkeeping
class Ledger:
    """Units already ingested, keyed `object_id#revision`; one fsynced JSON line per unit."""

    def __init__(self, path: Path):
        self.path = path
        self.entries: dict[str, dict[str, Any]] = {}
        if path.is_file():
            for line in path.read_text(encoding="utf-8").splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue  # a line cut short by a crash: the unit is redone or reconciled
                if isinstance(rec, dict) and "key" in rec:
                    self.entries[str(rec["key"])] = rec

    def __contains__(self, key: object) -> bool:
        return key in self.entries

    def __len__(self) -> int:
        return len(self.entries)

    def record(self, key: str, **info: Any) -> None:
        rec = {"key": key, **info}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.entries[key] = rec

    def pending(self, events: Iterable[Event]) -> list[Event]:
        """Events not yet ingested, in time order, each key once."""
        out: list[Event] = []
        seen: set[str] = set()
        for e in sorted(events, key=lambda e: (e.ts, e.event_id)):
            k = event_key(e)
            if k in self.entries or k in seen:
                continue
            seen.add(k)
            out.append(e)
        return out


class Usage:
    """Thread-safe LLM / embedding usage counters."""

    FIELDS = ("llm_calls", "prompt_tokens", "completion_tokens", "embed_calls", "embed_tokens")

    def __init__(self, counts: dict[str, int] | None = None):
        self._lock = threading.Lock()
        self.counts = {f: 0 for f in self.FIELDS}
        for k, v in (counts or {}).items():
            if k in self.counts:
                self.counts[k] = int(v)

    def add(self, **kw: int) -> None:
        with self._lock:
            for k, v in kw.items():
                self.counts[k] += int(v)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return dict(self.counts)

    @property
    def tokens(self) -> int:
        c = self.snapshot()
        return c["prompt_tokens"] + c["completion_tokens"] + c["embed_tokens"]


def _usage_counts(resp: Any) -> tuple[int, int]:
    u = getattr(resp, "usage", None)
    return int(getattr(u, "prompt_tokens", 0) or 0), int(getattr(u, "completion_tokens", 0) or 0)


class _CompletionsBase:
    def __init__(
        self, inner: Any, usage: Usage, extra_body: dict[str, Any] | None, cap: int | None
    ):
        self._inner = inner
        self._usage = usage
        self._extra_body = extra_body
        self._cap = cap

    def _prepare(self, kw: dict[str, Any]) -> dict[str, Any]:
        if self._extra_body:
            kw["extra_body"] = {**self._extra_body, **(kw.get("extra_body") or {})}
        if self._cap is not None:
            for k in ("max_tokens", "max_completion_tokens"):
                if isinstance(kw.get(k), int) and kw[k] > self._cap:
                    kw[k] = self._cap
        return kw

    def _record(self, resp: Any) -> None:
        pt, ct = _usage_counts(resp)
        self._usage.add(llm_calls=1, prompt_tokens=pt, completion_tokens=ct)


class _Completions(_CompletionsBase):
    def create(self, **kw: Any) -> Any:
        resp = self._inner.create(**self._prepare(kw))
        self._record(resp)
        return resp


class _AsyncCompletions(_CompletionsBase):
    async def create(self, **kw: Any) -> Any:
        resp = await self._inner.create(**self._prepare(kw))
        self._record(resp)
        return resp


class _Embeddings:
    def __init__(self, inner: Any, usage: Usage):
        self._inner = inner
        self._usage = usage

    def create(self, **kw: Any) -> Any:
        resp = self._inner.create(**kw)
        self._usage.add(embed_calls=1, embed_tokens=_usage_counts(resp)[0])
        return resp


class _AsyncEmbeddings(_Embeddings):
    async def acreate(self, **kw: Any) -> Any:
        resp = await self._inner.create(**kw)
        self._usage.add(embed_calls=1, embed_tokens=_usage_counts(resp)[0])
        return resp


class MeteredOpenAI:
    """Stands in for an `openai.OpenAI` / `openai.AsyncOpenAI` client inside a native library:
    adds our request extras (thinking off, as for every other system), caps `max_tokens` at what
    the serving context allows, and meters usage. Other attributes pass through."""

    def __init__(
        self,
        inner: Any,
        usage: Usage,
        *,
        is_async: bool = False,
        extra_body: dict[str, Any] | None = None,
        max_tokens_cap: int | None = None,
    ):
        self._inner = inner
        comp_cls = _AsyncCompletions if is_async else _Completions
        comps = comp_cls(inner.chat.completions, usage, extra_body, max_tokens_cap)
        self.chat = SimpleNamespace(completions=comps)
        if is_async:
            emb = _AsyncEmbeddings(inner.embeddings, usage)
            self.embeddings = SimpleNamespace(create=emb.acreate)
        else:
            self.embeddings = SimpleNamespace(create=_Embeddings(inner.embeddings, usage).create)

    def __getattr__(self, name: str) -> Any:
        if name == "_inner":  # not yet set (copy / pickle probing): avoid recursion
            raise AttributeError(name)
        return getattr(self._inner, name)


def dir_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _read_marker(store: Path) -> dict[str, Any]:
    p = store / STORE_MARKER
    if not p.is_file():
        raise ValueError(f"{store} is not a native memory store (no {STORE_MARKER})")
    data = json.loads(p.read_text(encoding="utf-8"))
    return data if isinstance(data, dict) else {}


def copy_store(src: Path, dest: Path) -> None:
    """Copy a closed store directory to a new, empty location."""
    _read_marker(src)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"snapshot destination is not empty: {dest}")
    shutil.copytree(src, dest, dirs_exist_ok=True)


def replace_store(src: Path, dest: Path) -> None:
    """Replace the store at `dest` by a copy of `src`. `dest` must be absent, empty, or itself a
    native store: nothing else is ever deleted."""
    _read_marker(src)
    if dest.exists() and any(dest.iterdir()):
        _read_marker(dest)
        shutil.rmtree(dest)
    shutil.copytree(src, dest, dirs_exist_ok=True)


def _quiet_env() -> None:
    """Telemetry off and Mem0's config dir inside the repo cache; must run before import."""
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("GRAPHITI_TELEMETRY_ENABLED", "false")
    os.environ.setdefault("MEM0_DIR", str(REPO / "cache" / "mem0"))


def _api_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or "not-needed"


# ---------------------------------------------------------------------- base
class NativeSystem:
    """Shared driver side of a native system: ledger, usage, snapshot / restore, serve."""

    name = "native"

    def __init__(
        self,
        world: World,
        store_dir: str | Path,
        llm_base_url: str,
        llm_model: str,
        embed_base_url: str,
        embed_model: str,
        user_id: str,
        *,
        embed_dims: int = DEFAULT_EMBED_DIMS,
        top_k: int = DEFAULT_TOP_K,
        llm_extra_body: dict[str, Any] | None = None,
        max_output_tokens: int | None = DEFAULT_MAX_OUTPUT_TOKENS,
        on_error: str = "raise",
    ):
        if on_error not in ("raise", "skip"):
            raise ValueError("on_error must be 'raise' or 'skip'")
        self.world = world
        self.store_dir = Path(store_dir)
        self.llm_base_url = llm_base_url
        self.llm_model = llm_model
        self.embed_base_url = embed_base_url
        self.embed_model = embed_model
        self.user_id = user_id
        self.embed_dims = embed_dims
        self.top_k = top_k
        self.llm_extra_body = DEFAULT_EXTRA_BODY if llm_extra_body is None else llm_extra_body
        self.max_output_tokens = max_output_tokens
        self.on_error = on_error
        self.cost: dict[str, float] = {
            "ingest_tokens": 0,
            "maintain_tokens": 0,
            "serve_tokens": 0,
            "verify_cpu_s": 0.0,
            "storage_bytes": 0,
        }
        self.stats: dict[str, float] = {"ingest_wall_s": 0.0, "ingested": 0, "failed": 0}
        self.store_dir.mkdir(parents=True, exist_ok=True)
        marker = self.store_dir / STORE_MARKER
        if marker.is_file():
            if _read_marker(self.store_dir).get("system") != self.name:
                raise ValueError(f"{self.store_dir} holds a different system's store")
        else:
            marker.write_text(json.dumps(self.describe(), indent=2, sort_keys=True))
        self._load_state()
        self._open()
        self._reconcile()

    # -------------------------------------------------------------- subclass hooks
    def _open(self) -> None:
        raise NotImplementedError

    def _close(self) -> None:
        raise NotImplementedError

    def _reconcile(self) -> None:
        """Record units already present in the store but missing from the ledger."""

    def _ingest_events(self, events: list[Event]) -> None:
        raise NotImplementedError

    def _search(self, query: str) -> list[Hit]:
        raise NotImplementedError

    def export_raw(self) -> list[dict[str, Any]]:
        """Every stored memory / node / edge with every field the system keeps."""
        raise NotImplementedError

    @property
    def retains_raw_text(self) -> bool:
        raise NotImplementedError

    @property
    def retains_source_metadata(self) -> bool:
        return True

    @property
    def identifiable(self) -> bool:
        """(D, B) is identifiable from outside only when raw text and source metadata are kept;
        otherwise it is declared unidentifiable."""
        return self.retains_raw_text and self.retains_source_metadata

    def describe(self) -> dict[str, Any]:
        return {
            "system": self.name,
            "user_id": self.user_id,
            "llm_model": self.llm_model,
            "embed_model": self.embed_model,
            "embed_dims": self.embed_dims,
            "top_k": self.top_k,
            "max_output_tokens": self.max_output_tokens,
            "llm_extra_body": self.llm_extra_body,
        }

    # -------------------------------------------------------------- state
    def _load_state(self) -> None:
        """Ledger, usage counters and cost vector as saved with the store (a restored branch
        starts from the prefix's accounting)."""
        self.ledger = Ledger(self.store_dir / LEDGER_FILE)
        p = self.store_dir / STATE_FILE
        state = json.loads(p.read_text()) if p.is_file() else {}
        self.usage = Usage(state.get("usage"))
        for k, v in (state.get("cost") or {}).items():
            if k in self.cost:
                self.cost[k] = v

    def _save_state(self) -> None:
        state = {"usage": self.usage.snapshot(), "cost": self.cost}
        (self.store_dir / STATE_FILE).write_text(json.dumps(state, sort_keys=True))

    def _fail(self, key: str, exc: Exception) -> None:
        if self.on_error == "raise":
            raise exc
        self.stats["failed"] += 1
        with (self.store_dir / FAILURES_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "error": repr(exc)[:2000]}) + "\n")

    def failed_keys(self) -> set[str]:
        """Units recorded as failed in this store (a batch failure lists its keys joined by ',')."""
        p = self.store_dir / FAILURES_FILE
        if not p.is_file():
            return set()
        keys: set[str] = set()
        for line in p.read_text(encoding="utf-8").splitlines():
            try:
                keys.update(str(json.loads(line)["key"]).split(","))
            except (json.JSONDecodeError, KeyError, TypeError):
                continue  # a line cut short by a crash
        return keys

    # -------------------------------------------------------------- public API
    def ingest(self, events: Sequence[Event], phase: str = "ingest") -> dict[str, Any]:
        """Ingest source events in time order, skipping units already in the store. `phase`
        ('ingest' before the fork, 'maintain' after) only selects the cost bucket."""
        if phase not in ("ingest", "maintain"):
            raise ValueError("phase must be 'ingest' or 'maintain'")
        # a unit that failed stays failed: a resumed ingest does not give it a second attempt
        failed = self.failed_keys()
        todo = [e for e in self.ledger.pending(events) if event_key(e) not in failed]
        before = self.usage.snapshot()
        n_before = len(self.ledger)
        t0 = time.perf_counter()
        try:
            self._ingest_events(todo)
        finally:
            wall = time.perf_counter() - t0
            after = self.usage.snapshot()
            delta = {k: after[k] - before[k] for k in after}
            self.cost[f"{phase}_tokens"] += (
                delta["prompt_tokens"] + delta["completion_tokens"] + delta["embed_tokens"]
            )
            self.stats["ingest_wall_s"] += wall
            self.stats["ingested"] += len(self.ledger) - n_before
            self.cost["storage_bytes"] = dir_bytes(self.store_dir)
            self._save_state()
        return {
            "events": len(events),
            "skipped": len(events) - len(todo),
            "ingested": len(self.ledger) - n_before,
            "wall_s": round(wall, 3),
            **delta,
        }

    def serve(self, probe: Probe) -> ServeResult:
        tokens0 = self.usage.tokens
        hits = self._search(probe.question_text)
        self.cost["serve_tokens"] += self.usage.tokens - tokens0
        by_id = {h.hit_id: h for h in hits}
        docs = [Doc(h.hit_id, h.text, h.ts) for h in hits]
        chosen = pack(docs, READER_BUDGET, READER_DOC_CAP)
        refs = [r for d, _ in chosen for r in by_id[d.doc_id].refs]
        diag = [
            {"hit_id": h.hit_id, "rank": i, "score": h.score, "refs": [list(r) for r in h.refs]}
            for i, h in enumerate(hits)
        ]
        return ServeResult([text for _, text in chosen], refs, diag, False)

    def snapshot(self, dest_dir: str | Path) -> None:
        """Copy the on-disk state (store, ledger, usage, cost) to `dest_dir` for branching."""
        self._close()
        try:
            self._save_state()
            copy_store(self.store_dir, Path(dest_dir))
        finally:
            self._open()

    def restore(self, src_dir: str | Path) -> None:
        """Replace this system's state by a snapshot taken from the same kind of system."""
        src = Path(src_dir)
        if _read_marker(src).get("system") != self.name:
            raise ValueError(f"{src} is not a {self.name} snapshot")
        self._close()
        replace_store(src, self.store_dir)
        self._load_state()
        self._open()
        self._reconcile()

    def close(self) -> None:
        self._close()
        self._save_state()


# ---------------------------------------------------------------------- Mem0
class Mem0System(NativeSystem):
    """Mem0 OSS with its own additive extraction. Drive one instance from one thread: embedded
    Qdrant keeps a thread-bound SQLite handle where SQLite is built THREADSAFE=2 (macOS)."""

    name = "mem0"

    def __init__(self, *args: Any, keep_raw: bool = True, **kw: Any):
        self.keep_raw = keep_raw
        self._mem: Any = None
        super().__init__(*args, **kw)

    @property
    def raw_user_id(self) -> str:
        return f"{self.user_id}__raw"

    @property
    def retains_raw_text(self) -> bool:
        return self.keep_raw

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "keep_raw": self.keep_raw, "vector_store": "qdrant-path"}

    def config(self) -> dict[str, Any]:
        """Mem0 `MemoryConfig` for `Memory.from_config` (mem0/configs/base.py:29)."""
        return {
            "llm": {
                "provider": "openai",
                "config": {
                    "model": self.llm_model,
                    "openai_base_url": self.llm_base_url,
                    "api_key": _api_key(),
                    "temperature": 0.0,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": self.embed_model,
                    "openai_base_url": self.embed_base_url,
                    "api_key": _api_key(),
                },
            },
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": "memshift",
                    "path": str(self.store_dir / "qdrant"),
                    "embedding_model_dims": self.embed_dims,
                    "on_disk": True,
                },
            },
            "history_db_path": str(self.store_dir / "history.db"),
        }

    def _open(self) -> None:
        _quiet_env()
        from mem0 import Memory

        mem = Memory.from_config(self.config())
        # Mem0 builds its own OpenAI clients (mem0/llms/openai.py:43-52,
        # mem0/embeddings/openai.py:31); wrap them to add our request extras and meter usage.
        mem.llm.client = MeteredOpenAI(
            mem.llm.client,
            self.usage,
            extra_body=self.llm_extra_body,
            max_tokens_cap=self.max_output_tokens,
        )
        mem.embedding_model.client = MeteredOpenAI(mem.embedding_model.client, self.usage)
        self._mem = mem

    def _close(self) -> None:
        mem, self._mem = self._mem, None
        if mem is None:
            return
        mem.close()
        client = getattr(mem.vector_store, "client", None)
        if client is not None:
            client.close()  # releases the embedded Qdrant directory lock

    def _rows(self, user_id: str) -> list[Any]:
        listed = self._mem.vector_store.list(filters={"user_id": user_id}, top_k=10**9)
        if isinstance(listed, (list, tuple)) and listed and isinstance(listed[0], list):
            listed = listed[0]  # Qdrant scroll returns (points, next_offset)
        return list(listed or [])

    def _reconcile(self) -> None:
        # With keep_raw the archive copy is written last, so its presence proves completion.
        # Imported memories (`add_memory`) carry their key in the payload.
        scope = self.raw_user_id if self.keep_raw else self.user_id
        found: dict[str, list[str]] = {}
        for point in self._rows(scope):
            payload = point.payload or {}
            if "object_id" in payload and "revision" in payload:
                key = f"{payload['object_id']}#{payload['revision']}"
            elif "import_key" in payload:
                key = str(payload["import_key"])
            else:
                continue
            found.setdefault(key, []).append(str(point.id))
        for key, ids in found.items():
            if key not in self.ledger:
                self.ledger.record(key, memory_ids=ids, reconciled=True)

    def _ingest_events(self, events: list[Event]) -> None:
        for e in events:
            key = event_key(e)
            try:
                text, md = event_text(e), event_metadata(e)
                res = self._mem.add(
                    messages=[{"role": "user", "content": text}],
                    user_id=self.user_id,
                    metadata=md,
                    infer=True,
                )
                ids = [r.get("id") for r in (res or {}).get("results", [])]
                raw_ids: list[Any] = []
                if self.keep_raw:
                    raw = self._mem.add(
                        messages=[{"role": "user", "content": text}],
                        user_id=self.raw_user_id,
                        metadata=md,
                        infer=False,
                    )
                    raw_ids = [r.get("id") for r in (raw or {}).get("results", [])]
            except Exception as exc:
                self._fail(key, exc)
                continue
            self.ledger.record(key, event_id=e.event_id, memory_ids=ids, raw_ids=raw_ids)

    def add_memory(self, key: str, text: str, metadata: dict[str, Any]) -> list[str]:
        """Store `text` verbatim (`infer=False`, no LLM call) under the serving scope; used to
        import a Portable Memory bundle. Idempotent per `key`."""
        if key in self.ledger:
            return list(self.ledger.entries[key].get("memory_ids") or [])
        tokens0 = self.usage.tokens
        res = self._mem.add(
            messages=[{"role": "user", "content": text}],
            user_id=self.user_id,
            metadata={**metadata, "import_key": key},
            infer=False,
        )
        ids = [str(r.get("id")) for r in (res or {}).get("results", [])]
        self.ledger.record(key, memory_ids=ids, imported=True)
        self.cost["ingest_tokens"] += self.usage.tokens - tokens0
        return ids

    def _search(self, query: str) -> list[Hit]:
        res = self._mem.search(query, top_k=self.top_k, filters={"user_id": self.user_id})
        rows = res.get("results", []) if isinstance(res, dict) else []
        return [mem0_hit(r) for r in rows if isinstance(r, dict)]

    def export_raw(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        scopes = [("extracted", self.user_id)]
        if self.keep_raw:
            scopes.append(("raw", self.raw_user_id))
        for scope, uid in scopes:
            for point in self._rows(uid):
                pid = str(point.id)
                out.append(
                    {
                        "scope": scope,
                        "id": pid,
                        "payload": dict(point.payload or {}),
                        "history": self._mem.history(pid),
                    }
                )
        return out


# ---------------------------------------------------------------------- Graphiti
class GraphitiSystem(NativeSystem):
    """Graphiti on embedded Kuzu, run on a private event loop; not safe for concurrent calls."""

    name = "graphiti"

    def __init__(
        self,
        *args: Any,
        store_raw: bool = True,
        batch_size: int = 1,
        max_coroutines: int | None = None,
        **kw: Any,
    ):
        self.store_raw = store_raw
        self.batch_size = max(1, batch_size)
        self.max_coroutines = max_coroutines
        self._g: Graphiti | None = None
        self._driver: KuzuDriver | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._clients: list[Any] = []
        self._episode_key: dict[str, str] = {}
        super().__init__(*args, **kw)

    @property
    def retains_raw_text(self) -> bool:
        return self.store_raw

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "store_raw": self.store_raw,
            "batch_size": self.batch_size,
            "driver": "kuzu",
        }

    def _run(self, coro: Coroutine[Any, Any, T]) -> T:
        assert self._loop is not None
        return self._loop.run_until_complete(coro)

    @property
    def graphiti(self) -> Graphiti:
        assert self._g is not None
        return self._g

    def _open(self) -> None:
        _quiet_env()
        from graphiti_core import Graphiti
        from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
        from graphiti_core.driver.kuzu_driver import KuzuDriver
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
        from openai import AsyncOpenAI

        self._loop = asyncio.new_event_loop()
        llm_raw = AsyncOpenAI(api_key=_api_key(), base_url=self.llm_base_url)
        emb_raw = AsyncOpenAI(api_key=_api_key(), base_url=self.embed_base_url)
        self._clients = [llm_raw, emb_raw]
        # The libraries type `client=` as concrete openai classes; the proxy is duck-typed.
        llm_proxy: Any = MeteredOpenAI(
            llm_raw,
            self.usage,
            is_async=True,
            extra_body=self.llm_extra_body,
            max_tokens_cap=self.max_output_tokens,
        )
        emb_proxy: Any = MeteredOpenAI(emb_raw, self.usage, is_async=True)
        cfg = LLMConfig(
            api_key=_api_key(),
            model=self.llm_model,
            small_model=self.llm_model,
            base_url=self.llm_base_url,
            temperature=0.0,
        )
        self._driver = KuzuDriver(db=str(self.store_dir / "graph.kuzu"))
        self._g = Graphiti(
            graph_driver=self._driver,
            llm_client=OpenAIGenericClient(config=cfg, client=llm_proxy),
            embedder=OpenAIEmbedder(
                config=OpenAIEmbedderConfig(
                    embedding_model=self.embed_model,
                    api_key=_api_key(),
                    base_url=self.embed_base_url,
                    embedding_dim=self.embed_dims,
                ),
                client=emb_proxy,
            ),
            cross_encoder=OpenAIRerankerClient(config=cfg, client=llm_proxy),
            store_raw_episode_content=self.store_raw,
            max_coroutines=self.max_coroutines,
        )
        self._run(self._ensure_fulltext())

    async def _ensure_fulltext(self) -> None:
        """KuzuDriver.build_indices_and_constraints is a no-op (kuzu_driver.py), but hybrid
        search queries the FTS indexes; create the missing ones (FTS ships inside kuzu 0.11)."""
        from graphiti_core.driver.driver import GraphProvider
        from graphiti_core.graph_queries import get_fulltext_indices

        assert self._driver is not None
        rows, _, _ = await self._driver.execute_query("CALL SHOW_INDEXES() RETURN *")
        have = {str(r.get("index_name")) for r in rows if isinstance(r, dict)}
        for q in get_fulltext_indices(GraphProvider.KUZU):
            name = q.split("'")[3]  # CALL CREATE_FTS_INDEX('<table>', '<index>', [...])
            if name not in have:
                await self._driver.execute_query(q)

    def _close(self) -> None:
        driver, loop, clients = self._driver, self._loop, self._clients
        self._g, self._driver, self._loop, self._clients = None, None, None, []
        if loop is not None:
            for c in clients:
                loop.run_until_complete(c.close())
            loop.close()
        if driver is not None:
            driver.client.close()
            driver.db.close()

    async def _episodes(self) -> list[Any]:
        from graphiti_core.nodes import EpisodicNode

        assert self._driver is not None
        return list(await EpisodicNode.get_by_group_ids(self._driver, [""]))

    def _reconcile(self) -> None:
        self._episode_key = {}
        for ep in self._run(self._episodes()):
            self._episode_key[ep.uuid] = ep.name
            if parse_key(ep.name) is not None and ep.name not in self.ledger:
                self.ledger.record(ep.name, episode_uuid=ep.uuid, reconciled=True)

    def _ingest_events(self, events: list[Event]) -> None:
        from graphiti_core.nodes import EpisodeType
        from graphiti_core.utils.bulk_utils import RawEpisode

        if self.batch_size <= 1:
            for e in events:
                key = event_key(e)
                try:
                    res = self._run(
                        self.graphiti.add_episode(
                            name=key,
                            episode_body=event_text(e),
                            source_description=e.source,
                            reference_time=e.ts.replace(tzinfo=UTC),
                            source=EpisodeType.text,
                        )
                    )
                except Exception as exc:
                    self._fail(key, exc)
                    continue
                self._episode_key[res.episode.uuid] = key
                self.ledger.record(
                    key,
                    event_id=e.event_id,
                    episode_uuid=res.episode.uuid,
                    nodes=len(res.nodes),
                    edges=len(res.edges),
                )
            return
        for batch in distinct_batches(events, self.batch_size):
            raw = [
                RawEpisode(
                    name=event_key(e),
                    content=event_text(e),
                    source_description=e.source,
                    source=EpisodeType.text,
                    reference_time=e.ts.replace(tzinfo=UTC),
                )
                for e in batch
            ]
            try:
                bulk = self._run(self.graphiti.add_episode_bulk(raw))
            except Exception as exc:
                self._fail(",".join(event_key(e) for e in batch), exc)
                continue
            uuids = {ep.name: ep.uuid for ep in bulk.episodes}
            for e in batch:
                key = event_key(e)
                self._episode_key[uuids.get(key, "")] = key
                self.ledger.record(key, event_id=e.event_id, episode_uuid=uuids.get(key))

    def _search(self, query: str) -> list[Hit]:
        edges = self._run(self.graphiti.search(query, num_results=self.top_k))
        hits = []
        for rank, edge in enumerate(edges):
            keys = [self._episode_key.get(u, "") for u in edge.episodes]
            hits.append(
                graphiti_hit(
                    edge.uuid,
                    edge.fact,
                    edge.valid_at,
                    edge.invalid_at,
                    edge.expired_at is not None,
                    keys,
                    edge.reference_time,
                    score=1.0 / (rank + 1),  # search returns reciprocal-rank-fused order only
                )
            )
        return hits

    async def _export(self) -> list[dict[str, Any]]:
        from graphiti_core.edges import EntityEdge
        from graphiti_core.nodes import EntityNode

        assert self._driver is not None
        rows: list[dict[str, Any]] = []
        for ep in await self._episodes():
            rows.append({"kind": "episode", **ep.model_dump(mode="json")})
        for node in await EntityNode.get_by_group_ids(self._driver, [""]):
            d = node.model_dump(mode="json")
            d["name_embedding_dim"] = len(d.pop("name_embedding", None) or [])
            rows.append({"kind": "entity", **d})
        for edge in await EntityEdge.get_by_group_ids(self._driver, [""]):
            d = edge.model_dump(mode="json")
            d["fact_embedding_dim"] = len(d.pop("fact_embedding", None) or [])
            rows.append({"kind": "edge", **d})
        return rows

    def export_raw(self) -> list[dict[str, Any]]:
        """Episodes (raw content when kept, source, reference time), entity nodes and fact
        edges with bi-temporal fields; embeddings are replaced by their dimension."""
        return self._run(self._export())

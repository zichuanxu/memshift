"""HippoRAG 2 (hipporag 2.0.0a4) as a native memory system (E5).

Integration contract: HippoRAG's own document API is the sync path.
Each source object is one document whose text is `event_text(e)` (object id, revision and time in
a header line). A new revision is `delete([old text])` then `index([new text])`; a source delete
event is `delete([current text])`. `store_dir/objects.jsonl` keeps object_id -> (revision, current
text), because both calls are keyed by the text itself (documents are identified by the md5 of
their content, embedding_store.py:44-90, utils/misc_utils.py:126). Retrieval is
`HippoRAG.retrieve` (recognition-memory filter, fact-seeded personalized PageRank over the graph,
dense passage scores; HippoRAG.py:360-446), never `rag_qa`: a Hit is a passage, its source refs
are parsed from the passage header. `snapshot()` / `restore()` copy `store_dir`, which holds
HippoRAG's whole `save_dir`: parquet stores per namespace (chunk / entity / fact), the igraph
pickle, the OpenIE file (HippoRAG.py:118, 151, 179) and the LLM response cache
(llm/openai_gpt.py:119).

What deletion does (HippoRAG.py:277-358; every claim below is from that method or its helpers):
the passage row, its embedding and its graph vertex go (:352, :355). A triple is dropped only when
no remaining passage supports it (support map built from the OpenIE file, :312-323, :1203-1210); a
phrase node only when every passage that contains it is deleted (:331-342). A triple or phrase that
another passage still supports stays. Fact-edge weights are counted at index time (:763-766) and
never decremented, and edges leave the graph only with a deleted vertex (:355), so the edge of a
dropped fact between two surviving phrases stays. The OpenIE file is rewritten without the passage
(:348), except when nothing is left (:981). The LLM response cache under
`save_dir/llm_cache` is not touched, so extraction outputs for deleted text stay on disk. Triples
and phrase nodes carry no source id of their own; the link is the OpenIE file (passage hash ->
entities, triples) and the header of the passage text, which the NER prompt sees
(prompts/templates/ner.py:21).

Observed library behaviour this adapter works around or reports (hipporag 2.0.0a4):
  - `index()` does not clear `ready_to_retrieve` (only `delete()` does, :358), so `retrieve`
    after `index` keeps the passage list and embeddings of the last `prepare_retrieval_objects`
    (:399-400, :1141-1243) and never returns the new passage. After every index / delete the
    adapter sets `ready_to_retrieve = False` and `ent_node_to_chunk_ids = None`, which is the
    state a reload computes.
  - `add_fact_edges` tests `"name" in self.graph.vs` (:746), false for an igraph VertexSeq, so
    every `index()` re-derives the fact edges of ALL stored passages and `add_new_edges` appends
    them again (:1074); `add_synonymy_edges` likewise runs over all phrase nodes (:845). Graph edges
    grow with the number of `index()` calls; `batch_size` groups distinct objects per call.
  - LLM failures inside OpenIE are swallowed and the passage is stored without entities or
    triples (openie_openai.py:63-72, :112-120); `OpenIEAudit` counts them (`on_error="raise"`
    rolls the passage back, "skip" keeps it as HippoRAG does and counts a failure).
  - Retrieval can assert on small or tied stores (:1395); the adapter then answers with HippoRAG's
    own dense-only `retrieve_dpr` (:522-591) and counts `stats["dpr_fallback"]`.

Embeddings: the model class is chosen by substrings of the model name and anything else asserts
(embedding_model/__init__.py:13-24); the OpenAI-compatible class needs a name containing
"text-embedding", ignores the query instruction it builds (OpenAI.py:73-90) and drops into ipdb
on a batch error (:106). The stores and the retriever only ever call `batch_encode`
(embedding_store.py:88; HippoRAG.py:1268-1275, :1303, :1347), so `OpenAIEmbedder` provides that one
method over our endpoint and is handed to HippoRAG through its resolver during construction
(:136-138). Query instructions use HippoRAG's own `Instruct: ...\\nQuery: ` format
(NVEmbedV2.py:76), which is also Qwen3-Embedding's; passages are embedded plain.

Install surface: `from hipporag import HippoRAG` imports transformers (HippoRAG.py:10), igraph
(:13), torch (embedding_model/base.py:111, utils/embed_utils.py:2; run time use: the synonymy KNN,
CUDA only if present, embed_utils.py:20), pandas / pyarrow (embedding_store.py:7), boto3
(Cohere.py:4), gritlm (GritLM.py:8), and, from code the online path never runs, litellm
(bedrock_llm.py:9; used only for llm names starting with "bedrock", llm/__init__.py:18) and vllm
(vllm_offline.py:27, via openie_vllm_offline.py:9 and HippoRAG.py:24; used only for
`openie_mode="offline"`, config_utils.py:113). vllm is GPU-bound; litellm fetches a price table
over HTTP when imported (litellm 1.73.1 `__init__.py:359`). `import_hipporag` registers inert
stand-ins for those two when they are not installed; everything else must be installed.
Importing also starts a multiprocessing manager (embedding_model/base.py:225): under the
spawn start method (macOS, Windows) the entry script needs an `if __name__ == "__main__"` guard.

Usage (LLM calls and tokens, embedding calls and tokens) is metered by wrapping the client
HippoRAG builds (`llm_model.openai_client`) and the embedder's client. Drive one instance from
one thread: HippoRAG's progress bars (its module-level `tqdm` names) and the graph summary
it prints after every index() are silenced.
"""

from __future__ import annotations

import ast
import contextlib
import functools
import importlib
import importlib.metadata
import importlib.util
import io
import json
import os
import re
import sys
import types
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from memshift.adapters.native import (
    _EPOCH,
    Hit,
    MeteredOpenAI,
    NativeSystem,
    _api_key,
    distinct_batches,
    event_key,
    event_text,
)
from memshift.gen.model import Event, fmt_ts

OBJECTS_FILE = "objects.jsonl"
SAVE_SUBDIR = "hipporag"
QUERY_INSTRUCTION = "Instruct: {instruction}\nQuery: "
# name -> attributes `from name import ...` needs; the modules are imported by hipporag but only
# used by code paths this adapter never takes (module docstring)
STUBS: dict[str, tuple[str, ...]] = {"litellm": (), "vllm": ("LLM", "SamplingParams")}
_HEADER = re.compile(
    r"\[(?P<source>[^\]|]+?) record (?P<oid>.+?) \| revision (?P<rev>\d+)"
    r" \| (?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2})\]"
)


# ---------------------------------------------------------------------- import surface
def _stub_unused_backends() -> None:
    """Register inert stand-ins for the backends hipporag imports but the online path never
    runs; constructing one raises. Real installs are left alone."""
    for name, attrs in STUBS.items():
        if name in sys.modules or importlib.util.find_spec(name) is not None:
            continue
        mod = types.ModuleType(name)

        def unavailable(*_a: Any, _name: str = name, **_k: Any) -> Any:
            raise RuntimeError(f"{_name} is not installed: HippoRAG's offline path is unavailable")

        for attr in attrs:
            setattr(mod, attr, type(attr, (), {"__init__": unavailable}))
        sys.modules[name] = mod


def import_hipporag() -> Any:
    """The `HippoRAG` class, importing it with stand-ins for its unused backends."""
    if importlib.util.find_spec("hipporag") is None:
        raise ModuleNotFoundError("hipporag is not installed (docker/native/hipporag.txt)")
    _stub_unused_backends()
    from hipporag import HippoRAG

    return HippoRAG


def _quiet_env() -> None:
    """The API key hipporag's own client reads from the environment."""
    os.environ.setdefault("OPENAI_API_KEY", _api_key())


def _silence_progress_bars() -> None:
    """hipporag modules bind `tqdm` at import; give each a disabled variant of it."""
    from tqdm import tqdm

    quiet = functools.partial(tqdm, disable=True)
    for name, module in list(sys.modules.items()):
        holder: Any = module
        if name.startswith("hipporag") and getattr(holder, "tqdm", None) is tqdm:
            holder.tqdm = quiet


# ---------------------------------------------------------------------- passages and hits
@dataclass(frozen=True)
class PassageHeader:
    source: str
    object_id: str
    revision: int
    ts: datetime


def parse_header(text: str) -> PassageHeader | None:
    """The header line `event_text` puts on every passage, or None for foreign text."""
    m = _HEADER.match(text)
    if m is None:
        return None
    ts = datetime.strptime(m["ts"], "%Y-%m-%d %H:%M")
    return PassageHeader(m["source"], m["oid"], int(m["rev"]), ts)


def hipporag_hit(hit_id: str, text: str, score: float) -> Hit:
    """One retrieved passage -> Hit. The reader sees the passage exactly as HippoRAG stores it;
    the refs and the event time come from its header."""
    head = parse_header(text)
    if head is None:
        return Hit(hit_id, text, _EPOCH, [], score)
    return Hit(hit_id, text, head.ts, [(head.object_id, head.revision)], score)


def objects_from_passages(texts: Iterable[str]) -> dict[str, tuple[int, str]]:
    """object_id -> (revision, text) as the store holds it; the highest revision wins if a
    crash left two, text without a header is ignored."""
    out: dict[str, tuple[int, str]] = {}
    for text in texts:
        head = parse_header(text)
        if head is not None and head.revision > out.get(head.object_id, (-1, ""))[0]:
            out[head.object_id] = (head.revision, text)
    return out


# ---------------------------------------------------------------------- object map and plans
class ObjectMap:
    """object_id -> (revision, current passage text), persisted as an append-only JSON-lines log
    that is replayed on open (a line cut short by a crash is dropped)."""

    def __init__(self, path: Path):
        self.path = path
        self.items: dict[str, tuple[int, str]] = {}
        if path.is_file():
            raw = path.read_text(encoding="utf-8")
            for line in raw.splitlines():
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(rec, dict) or "object_id" not in rec:
                    continue
                if rec.get("deleted"):
                    self.items.pop(str(rec["object_id"]), None)
                elif "text" in rec:
                    self.items[str(rec["object_id"])] = (int(rec["revision"]), str(rec["text"]))
            if raw and not raw.endswith("\n"):
                with path.open("a", encoding="utf-8") as f:
                    f.write("\n")  # the next record must not fuse with a torn one

    def __len__(self) -> int:
        return len(self.items)

    def get(self, object_id: str) -> tuple[int, str] | None:
        return self.items.get(object_id)

    def _log(self, rec: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def put(self, object_id: str, revision: int, text: str) -> None:
        self._log({"object_id": object_id, "revision": revision, "text": text})
        self.items[object_id] = (revision, text)

    def remove(self, object_id: str) -> None:
        if object_id in self.items:
            self._log({"object_id": object_id, "deleted": True})
            del self.items[object_id]

    def sync(self, found: Mapping[str, tuple[int, str]]) -> None:
        """Make the map equal to what the store holds (the store is the truth after a crash)."""
        for oid in [o for o in self.items if o not in found]:
            self.remove(oid)
        for oid, (rev, text) in found.items():
            if self.items.get(oid) != (rev, text):
                self.put(oid, rev, text)


@dataclass(frozen=True)
class Step:
    """What HippoRAG must do for one source event: delete the passage `drop`, index `add`."""

    object_id: str
    revision: int
    drop: str | None
    add: str | None


def plan_event(current: tuple[int, str] | None, e: Event) -> Step | None:
    """The delete / index calls one event needs given the object's current (revision, text);
    None when the store already reflects it (replayed or older event, delete of an absent
    object)."""
    if e.op == "delete":
        return None if current is None else Step(e.object_id, e.revision, current[1], None)
    if current is not None and current[0] >= e.revision:
        return None
    return Step(e.object_id, e.revision, None if current is None else current[1], event_text(e))


# ---------------------------------------------------------------------- embeddings
def embedding_texts(texts: Sequence[str], instruction: str = "") -> list[str]:
    """Texts as HippoRAG's OpenAI-compatible path sends them (newlines flattened, empty text a
    space, OpenAI.py:74-75), behind the query instruction when there is one."""
    prefix = QUERY_INSTRUCTION.format(instruction=instruction) if instruction else ""
    return [prefix + (t.replace("\n", " ") or " ") for t in texts]


class OpenAIEmbedder:
    """The `batch_encode` half of hipporag's `BaseEmbeddingModel` (embedding_model/base.py:208)
    over an OpenAI-compatible endpoint; returns L2-normalised rows."""

    def __init__(
        self, client: Any, model: str, batch_size: int = 16, query_instruction: bool = True
    ):
        self.client = client
        self.model = model
        self.batch_size = max(1, batch_size)
        self.query_instruction = query_instruction

    def batch_encode(self, texts: str | Sequence[str], **kw: Any) -> np.ndarray:
        batch = [texts] if isinstance(texts, str) else list(texts)
        instruction = str(kw.get("instruction") or "") if self.query_instruction else ""
        batch = embedding_texts(batch, instruction)
        rows: list[list[float]] = []
        for i in range(0, len(batch), self.batch_size):
            resp = self.client.embeddings.create(
                input=batch[i : i + self.batch_size], model=self.model
            )
            rows += [d.embedding for d in sorted(resp.data, key=lambda d: getattr(d, "index", 0))]
        out = np.array(rows, dtype=np.float64)
        if kw.get("norm", True) and len(out):
            out = out / np.linalg.norm(out, axis=1, keepdims=True)
        return out


# ---------------------------------------------------------------------- OpenIE audit
@dataclass
class OpenIEAudit:
    """What HippoRAG's OpenIE returned per passage text, including the failures it swallows."""

    errors: list[str] = field(default_factory=list)
    counts: dict[str, dict[str, int]] = field(default_factory=dict)

    def reset(self) -> None:
        self.errors.clear()
        self.counts.clear()

    def wrap(self, openie: Any) -> None:
        """Route `openie.ner` and `openie.triple_extraction` (openie_openai.py:45, :81) through
        this audit; `batch_openie` looks both up on the instance."""
        ner, triples = openie.ner, openie.triple_extraction

        def audited_ner(chunk_key: str, passage: str) -> Any:
            out = ner(chunk_key, passage)
            self._note(passage, "ner", out.metadata, "entities", len(out.unique_entities))
            return out

        def audited_triples(chunk_key: str, passage: str, named_entities: list[str]) -> Any:
            out = triples(chunk_key, passage, named_entities)
            self._note(passage, "triples", out.metadata, "triples", len(out.triples))
            return out

        openie.ner, openie.triple_extraction = audited_ner, audited_triples

    def _note(self, passage: str, kind: str, meta: Mapping[str, Any], key: str, n: int) -> None:
        if meta.get("error"):
            self.errors.append(f"{kind}: {str(meta['error'])[:200]}")
        self.counts.setdefault(passage, {})[key] = n


# ---------------------------------------------------------------------- system
class HippoRAGSystem(NativeSystem):
    """HippoRAG 2 with online OpenIE on our endpoints, driven through its `index` / `delete` /
    `retrieve` API. Not safe for concurrent calls."""

    name = "hipporag"
    objects: ObjectMap

    def __init__(self, *args: Any, batch_size: int = 1, query_instruction: bool = True, **kw: Any):
        self.batch_size = max(1, batch_size)
        self.query_instruction = query_instruction
        self._hip: Any = None
        self._audit = OpenIEAudit()
        super().__init__(*args, **kw)

    @property
    def retains_raw_text(self) -> bool:
        return True

    @property
    def save_dir(self) -> Path:
        return self.store_dir / SAVE_SUBDIR

    def describe(self) -> dict[str, Any]:
        try:
            version: str | None = importlib.metadata.version("hipporag")
        except importlib.metadata.PackageNotFoundError:
            version = None
        return {
            **super().describe(),
            "batch_size": self.batch_size,
            "query_instruction": self.query_instruction,
            "hipporag_version": version,
            "openie_mode": "online",
        }

    def _open(self) -> None:
        _quiet_env()
        hippo_cls = import_hipporag()
        _silence_progress_bars()
        from hipporag.utils.config_utils import BaseConfig
        from openai import OpenAI

        self.objects = ObjectMap(self.store_dir / OBJECTS_FILE)
        cfg = BaseConfig(
            save_dir=str(self.save_dir),
            llm_name=self.llm_model,
            llm_base_url=self.llm_base_url,
            embedding_model_name=self.embed_model,
            embedding_base_url=self.embed_base_url,
            retrieval_top_k=self.top_k,
            temperature=0.0,
        )
        embedder = OpenAIEmbedder(
            MeteredOpenAI(OpenAI(api_key=_api_key(), base_url=self.embed_base_url), self.usage),
            self.embed_model,
            cfg.embedding_batch_size,
            self.query_instruction,
        )
        module: Any = importlib.import_module("hipporag.HippoRAG")
        resolver = module._get_embedding_model_class
        module._get_embedding_model_class = lambda embedding_model_name: lambda **_: embedder
        try:
            hip = hippo_cls(global_config=cfg)
        finally:
            module._get_embedding_model_class = resolver
        llm = hip.llm_model
        llm.openai_client = MeteredOpenAI(
            llm.openai_client,
            self.usage,
            extra_body=self.llm_extra_body,
            max_tokens_cap=self.max_output_tokens,
        )
        self._audit = OpenIEAudit()
        self._audit.wrap(hip.openie)
        self.stats.setdefault("dpr_fallback", 0)
        self._hip = hip

    def _close(self) -> None:
        self._hip = None  # no file handles: parquet, pickle and the LLM cache are opened per use

    def _passages(self) -> set[str]:
        return set(self._hip.chunk_embedding_store.get_all_texts())

    def _reconcile(self) -> None:
        """Object map and ledger from the passages the store holds. A lost ledger cannot know
        about objects deleted since: their inserts are redone and removed again."""
        found = objects_from_passages(self._passages())
        self.objects.sync(found)
        for oid, (rev, _) in found.items():
            if f"{oid}#{rev}" not in self.ledger:
                self.ledger.record(f"{oid}#{rev}", reconciled=True)

    # -------------------------------------------------------------- ingest
    def _call(self, fn: Any, docs: list[str]) -> None:
        """One HippoRAG index / delete call; afterwards its retrieval caches are invalidated."""
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # index() prints the graph info
                fn(docs)
        finally:
            self._hip.ready_to_retrieve = False
            self._hip.ent_node_to_chunk_ids = None

    def _delete(self, docs: list[str]) -> None:
        self._call(self._hip.delete, docs)
        if not self._hip.chunk_embedding_store.get_all_ids():
            # deleting the last passage leaves a stale OpenIE file (:981) that would fail the
            # next index() (:247) or hand its stale extraction to a re-indexed passage
            Path(self._hip.openie_results_path).unlink(missing_ok=True)

    def _apply(self, steps: list[Step]) -> None:
        self._audit.reset()
        drops = [s for s in steps if s.drop is not None]
        if drops:
            self._delete([str(s.drop) for s in drops])
            for s in drops:
                self.objects.remove(s.object_id)
        adds = [s for s in steps if s.add is not None]
        if adds:
            self._call(self._hip.index, [str(s.add) for s in adds])
            for s in adds:
                self.objects.put(s.object_id, s.revision, str(s.add))

    def _rollback(self, steps: list[Step]) -> None:
        """Take passages whose OpenIE failed back out, so a resumed ingest redoes them."""
        held = self._passages()
        adds = [s for s in steps if s.add is not None and s.add in held]
        if adds:
            self._delete([str(s.add) for s in adds])
            for s in adds:
                self.objects.remove(s.object_id)

    def _ingest_events(self, events: list[Event]) -> None:
        for batch in distinct_batches(events, self.batch_size):
            keys = ",".join(event_key(e) for e in batch)
            steps = [
                s for e in batch if (s := plan_event(self.objects.get(e.object_id), e)) is not None
            ]
            try:
                self._apply(steps)
            except Exception as exc:
                self._fail(keys, exc)
                continue
            if self._audit.errors:
                err = RuntimeError(
                    f"OpenIE swallowed {len(self._audit.errors)} LLM failure(s): "
                    f"{self._audit.errors[0]}"
                )
                if self.on_error == "raise":
                    self._rollback(steps)
                    raise err
                self._fail(keys, err)  # the passage stays, as it does in HippoRAG
            self._record(batch, steps)

    def _record(self, batch: list[Event], steps: list[Step]) -> None:
        dropped = {s.object_id for s in steps if s.drop is not None}
        counts = {s.object_id: self._audit.counts.get(str(s.add), {}) for s in steps if s.add}
        for e in batch:
            self.ledger.record(
                event_key(e),
                event_id=e.event_id,
                op=e.op,
                dropped=e.object_id in dropped,
                **counts.get(e.object_id, {}),
            )

    # -------------------------------------------------------------- serve
    def _search(self, query: str) -> list[Hit]:
        hip = self._hip
        if not hip.chunk_embedding_store.get_all_ids():
            return []
        try:
            with np.errstate(invalid="ignore"):  # 0/0 phrase weights, zeroed before PPR (:1462)
                sol = hip.retrieve([query], num_to_retrieve=self.top_k)[0]
        except AssertionError:
            self.stats["dpr_fallback"] += 1
            sol = hip.retrieve_dpr([query], num_to_retrieve=self.top_k)[0]
        ids = hip.chunk_embedding_store.text_to_hash_id
        return [
            hipporag_hit(ids.get(t, ""), t, float(s))
            for t, s in zip(sol.docs, sol.doc_scores, strict=True)
        ]

    # -------------------------------------------------------------- export
    def export_raw(self) -> list[dict[str, Any]]:
        """Passages (text, header fields), OpenIE results (entities and triples per passage),
        fact and phrase nodes with the passages that support them, and graph edges (passage,
        fact or synonymy by weight: synonymy scores are below 1, fact counts are at least 1;
        parallel edges are merged into `multiplicity`). Embeddings are replaced by their
        dimension."""
        from hipporag.utils.misc_utils import text_processing

        hip = self._hip
        chunks = hip.chunk_embedding_store
        openie = hip.load_existing_openie([])[0]
        rows: list[dict[str, Any]] = []
        for hid, row in chunks.get_all_id_to_rows().items():
            head = parse_header(row["content"])
            rows.append(
                {
                    "kind": "passage",
                    "hash_id": hid,
                    "content": row["content"],
                    "object_id": head.object_id if head else None,
                    "revision": head.revision if head else None,
                    "source": head.source if head else None,
                    "ts": fmt_ts(head.ts) if head else None,
                    "embedding_dim": len(chunks.get_embedding(hid)),
                }
            )
        fact_support: dict[str, set[str]] = {}
        phrase_support: dict[str, set[str]] = {}
        for doc in openie:
            rows.append(
                {
                    "kind": "openie",
                    "passage_id": doc["idx"],
                    "entities": list(doc["extracted_entities"]),
                    "triples": [list(t) for t in doc["extracted_triples"]],
                }
            )
            for triple in doc["extracted_triples"]:
                if len(triple) != 3:
                    continue
                proc = text_processing(list(triple))
                fact_support.setdefault(str(tuple(proc)), set()).add(doc["idx"])
                for phrase in (proc[0], proc[2]):
                    phrase_support.setdefault(phrase, set()).add(doc["idx"])
        for kind, store, support in (
            ("fact", hip.fact_embedding_store, fact_support),
            ("phrase", hip.entity_embedding_store, phrase_support),
        ):
            for hid, row in store.get_all_id_to_rows().items():
                content = str(row["content"])
                extra = {"triple": list(ast.literal_eval(content))} if kind == "fact" else {}
                rows.append(
                    {
                        "kind": kind,
                        "hash_id": hid,
                        "content": content,
                        **extra,
                        "passages": sorted(support.get(content, ())),
                        "embedding_dim": len(store.get_embedding(hid)),
                    }
                )
        rows += self._edge_rows()
        return rows

    def _edge_rows(self) -> list[dict[str, Any]]:
        graph = self._hip.graph
        names = graph.vs["name"] if graph.vcount() else []
        count: Counter[tuple[str, str, float]] = Counter()
        for edge in graph.es:
            count[(names[edge.source], names[edge.target], float(edge["weight"]))] += 1
        rows = []
        for (a, b, w), n in count.items():
            if a.startswith("chunk-") or b.startswith("chunk-"):
                kind = "passage"
            else:
                kind = "synonymy" if w < 1.0 else "fact"
            rows.append(
                {
                    "kind": "edge",
                    "edge": kind,
                    "source": a,
                    "target": b,
                    "weight": w,
                    "multiplicity": n,
                }
            )
        return rows

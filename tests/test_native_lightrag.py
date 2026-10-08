"""Offline tests for the LightRAG adapter (prereg v2 §3). The pure helpers (document ids and refs,
revision replace / delete sequencing, reconcile, context parsing) run everywhere. The tests that
drive the real library (`lightrag-hku`, see docker/native/lightrag.txt) replace only its models by
deterministic fakes and its tokenizer by a character tokenizer, so they need no network and no
tiktoken encoding; they are skipped when the library is not importable."""

import hashlib
import json
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from memshift.adapters import native_lightrag as nl
from memshift.adapters.native import Usage
from memshift.adapters.native_lightrag import (
    Action,
    LightRAGSystem,
    check_ingested,
    chunk_doc_id,
    claim_process,
    doc_file_path,
    head_ts,
    interleave,
    lightrag_hits,
    plan_event,
    reconcile_plan,
    release_process,
    source_refs,
)
from memshift.gen.model import Event

T0 = datetime(2026, 3, 1, 9, 30)
DELIM = "<|#|>"


def _ev(i: int, oid: str, rev: int, body: str = "", op: str = "insert") -> Event:
    return Event(
        event_id=f"ev{i:06d}",
        ts=T0 + timedelta(minutes=i),
        source=oid.split(":")[0],
        object_id=oid,
        revision=rev,
        op=op,
        payload={"text": body, "cluster": oid},
    )


# ---------------------------------------------------------------------- pure helpers
def test_file_path_refs_and_chunk_ids():
    assert doc_file_path("calendar:series:3", 4) == "calendar:series:3#4"
    assert source_refs("contacts:p01#2<SEP>email:e1#1<SEP>contacts:p01#2") == [
        ("contacts:p01", 2),
        ("email:e1", 1),
    ]
    placeholder = "contacts:p01#2<SEP>...file paths truncated...(KEEP Old)<SEP>unknown_source"
    assert source_refs(placeholder) == [("contacts:p01", 2)]
    assert source_refs("unknown_source") == []
    assert chunk_doc_id("email:e00052-chunk-003") == "email:e00052"
    assert chunk_doc_id("chunk-3f2a") is None and chunk_doc_id("x-chunk-abc") is None


def test_head_ts_reads_event_time_from_event_text():
    head = "[email record email:e1 | revision 2 | 2026-03-01 09:35]\nbody"
    assert head_ts(head) == "2026-03-01T09:35"
    assert head_ts("Record contacts:p01 was deleted on 2026-03-01 09:37.") is None
    assert head_ts("body only") is None


def test_plan_event_sequences():
    cur = {"a:1": 2}
    assert plan_event(cur, _ev(1, "a:9", 1)) == [Action("insert", "a:9", 1)]
    assert plan_event(cur, _ev(2, "a:1", 3, op="update")) == [
        Action("delete", "a:1"),
        Action("insert", "a:1", 3),
    ]
    assert plan_event(cur, _ev(3, "a:1", 3, op="delete")) == [Action("delete", "a:1")]
    assert plan_event(cur, _ev(4, "a:9", 2, op="delete")) == []  # absent: nothing to delete
    assert plan_event(cur, _ev(5, "a:1", 2, op="update")) == []  # replayed
    assert plan_event(cur, _ev(6, "a:1", 1)) == []  # older than what is stored
    assert plan_event({}, _ev(7, "a:1", 5, op="update")) == [Action("insert", "a:1", 5)]


def test_check_ingested():
    ok = {"status": "processed", "file_path": "a:1#2"}
    check_ingested(ok, "a:1#2")
    with pytest.raises(RuntimeError, match="not in doc status"):
        check_ingested(None, "a:1#2")
    with pytest.raises(RuntimeError, match="failed: boom"):
        check_ingested({"status": "failed", "error_msg": "boom"}, "a:1#2")
    with pytest.raises(RuntimeError, match="kept file_path"):
        check_ingested({**ok, "file_path": "a:1#1"}, "a:1#2")  # duplicate id: old row survived


def test_reconcile_plan_keeps_complete_documents_only():
    rows = [
        {"doc_id": "a:1", "status": "processed", "file_path": "a:1#3"},
        {"doc_id": "a:2", "status": "processing", "file_path": "a:2#1"},
        {"doc_id": "dup-9f", "status": "failed", "file_path": "a:1#3"},
        {"doc_id": "a:3", "status": "processed", "file_path": "unknown_source"},
        {"doc_id": "a:4", "status": "processed", "file_path": "a:5#1"},
    ]
    keep, purge = reconcile_plan(rows)
    assert keep == {"a:1": 3}
    assert purge == ["a:2", "dup-9f", "a:3", "a:4"]


QUERY_DATA = {
    "entities": [
        {
            "entity_name": "Alice Rivera",
            "entity_type": "person",
            "description": "Alice works at Acme.",
            "source_id": "contacts:p01-chunk-000<SEP>email:e1-chunk-001",
            "file_path": "contacts:p01#2<SEP>email:e1#1",
            "created_at": 1790000000,
        },
        {
            "entity_name": "Boston",
            "entity_type": "location",
            "description": "A city.",
            "source_id": "notes:n3-chunk-000<SEP>notes:gone-chunk-000",
            "file_path": "unknown_source",
            "created_at": 1790000001,
        },
    ],
    "relationships": [
        {
            "src_id": "Alice Rivera",
            "tgt_id": "Acme",
            "description": "Alice is employed by Acme.",
            "keywords": "employment",
            "weight": 1.0,
            "source_id": "contacts:p01-chunk-000",
            "file_path": "contacts:p01#2",
            "created_at": 1790000002,
        }
    ],
    "chunks": [
        {
            "reference_id": "1",
            "content": "[contacts record contacts:p01 | revision 2 | 2026-03-01 09:35]\nAlice",
            "file_path": "contacts:p01#2",
            "chunk_id": "contacts:p01-chunk-000",
        },
        {
            "reference_id": "2",
            "content": "second chunk of a note",
            "file_path": "unknown_source",
            "chunk_id": "notes:n3-chunk-002",
        },
    ],
    "references": [{"reference_id": "1", "file_path": "contacts:p01#2"}],
}


def test_lightrag_hits_from_query_data():
    revision = {"contacts:p01": 2, "email:e1": 1, "notes:n3": 4}
    event_ts = {"contacts:p01": "2026-03-01T09:35", "email:e1": "2026-02-01T08:00"}
    hits = lightrag_hits(QUERY_DATA, revision, event_ts)
    assert [h.hit_id for h in hits] == [
        "chunk:contacts:p01-chunk-000",
        "entity:Alice Rivera",
        "relation:Alice Rivera|Acme",
        "chunk:notes:n3-chunk-002",
        "entity:Boston",
    ]
    chunk, entity, relation, note_chunk, boston = hits
    assert chunk.text.startswith(
        "[chunk | source contacts:p01 r2 | event 2026-03-01 09:35] [contacts"
    )
    assert chunk.refs == [("contacts:p01", 2)] and chunk.score == 1.0
    assert entity.text == (
        "[entity person | source contacts:p01 r2, email:e1 r1 | event 2026-03-01 09:35]"
        " Alice Rivera: Alice works at Acme."
    )
    assert entity.refs == [("contacts:p01", 2), ("email:e1", 1)]
    assert entity.ts == datetime(2026, 3, 1, 9, 35)  # latest source event, not wall clock
    assert relation.text.startswith("[relation employment | source contacts:p01 r2 | event ")
    assert relation.text.endswith("Alice Rivera -> Acme: Alice is employed by Acme.")
    # no usable file_path: refs come from the chunk ids, only for documents still stored
    assert note_chunk.refs == [("notes:n3", 4)] and note_chunk.ts == datetime(1970, 1, 1)
    assert boston.refs == [("notes:n3", 4)]
    assert note_chunk.score == 0.5 and boston.score == 0.5
    assert lightrag_hits({}, revision, event_ts) == []


def test_interleave_keeps_each_list_in_order():
    a, b, c = list("abc"), list("xy"), []
    assert interleave(a, b, c) == ["a", "x", "b", "y", "c"]
    assert interleave() == []


def test_usage_tracker_counts_answered_requests():
    usage = Usage()
    llm = nl._UsageTracker(usage, embedding=False)
    emb = nl._UsageTracker(usage, embedding=True)
    llm.add_usage({"prompt_tokens": 100, "completion_tokens": 7, "total_tokens": 107})
    emb.add_usage({"prompt_tokens": 40, "total_tokens": 40})
    emb.add_usage({"prompt_tokens": 2})
    assert usage.snapshot() == {
        "llm_calls": 1,
        "prompt_tokens": 100,
        "completion_tokens": 7,
        "embed_calls": 2,
        "embed_tokens": 42,
    }


def test_claim_process_allows_one_store(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    claim_process(a)
    try:
        with pytest.raises(RuntimeError, match="already open"):
            claim_process(b)
        with pytest.raises(RuntimeError, match="already open"):
            claim_process(a)
    finally:
        release_process(a)
    claim_process(b)
    release_process(b)
    assert not nl._OPEN_STORES


# ------------------------------------------------------------------ real library, fake models
class _CharTokenizer:
    """One token per character: stateless, so safe under the library's tokenizer threads."""

    def encode(self, content: str) -> list[int]:
        return [ord(c) for c in content]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(t) for t in tokens)


class FakeModels:
    """Deterministic stand-ins for the endpoints. The 'LLM' reads `ENTITY name | type | text` and
    `REL src | tgt | keywords | text` lines from the chunk in the extraction prompt and answers in
    the library's delimiter format; embeddings are hashed bags of words."""

    def __init__(self) -> None:
        self.calls = {"extract": 0, "glean": 0, "summary": 0, "keywords": 0}
        self.fail = False

    def llm(self, usage: Usage) -> Any:
        tracker = nl._UsageTracker(usage, embedding=False)

        async def fn(
            prompt: str, system_prompt: Any = None, history_messages: Any = None, **kw: Any
        ):
            if self.fail:
                raise RuntimeError("endpoint down")
            tracker.add_usage({"prompt_tokens": len(prompt) // 4, "completion_tokens": 10})
            if "Based on the last extraction task" in prompt:
                self.calls["glean"] += 1
                return "<|COMPLETE|>"
            m = re.search(r"---Input Text---\s*```\n(.*?)\n```\s*---Output---", prompt, re.S)
            if m:
                self.calls["extract"] += 1
                rows = []
                for line in m.group(1).splitlines():
                    if line.startswith("ENTITY "):
                        n, t, d = (x.strip() for x in line[7:].split("|"))
                        rows.append(f"entity{DELIM}{n}{DELIM}{t}{DELIM}{d}")
                    elif line.startswith("REL "):
                        s, t, k, d = (x.strip() for x in line[4:].split("|"))
                        rows.append(f"relation{DELIM}{s}{DELIM}{t}{DELIM}{k}{DELIM}{d}")
                return "\n".join([*rows, "<|COMPLETE|>"])
            if "Description List" in prompt:
                self.calls["summary"] += 1
                return "SUMMARY"
            self.calls["keywords"] += 1
            query = prompt.split("User Query:")[1].split("---Output---")[0].strip()
            names = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", query) or [query]
            return json.dumps({"high_level_keywords": [query], "low_level_keywords": names})

        return fn

    def embed(self, usage: Usage) -> Any:
        tracker = nl._UsageTracker(usage, embedding=True)

        async def fn(texts: list[str], **_: Any):
            tracker.add_usage({"prompt_tokens": sum(len(t) // 4 for t in texts)})
            out = np.zeros((len(texts), 1024), dtype=np.float32)
            for i, text in enumerate(texts):
                for w in re.findall(r"[a-z0-9]+", text.lower()):
                    out[i, int(hashlib.md5(w.encode()).hexdigest(), 16) % 1024] += 1.0
                norm = np.linalg.norm(out[i])
                out[i] /= norm or 1.0
            return out

        return fn

    def reset(self) -> None:
        self.calls = dict.fromkeys(self.calls, 0)


def _offline(models: FakeModels) -> type[LightRAGSystem]:
    class Offline(LightRAGSystem):
        def _llm_func(self) -> Any:
            return models.llm(self.usage)

        def _embed_func(self) -> Any:
            return models.embed(self.usage)

        def _tokenizer(self) -> Any:
            from lightrag.utils import Tokenizer

            return Tokenizer("char", _CharTokenizer())

    return Offline


def _open(models: FakeModels, store: Path, **kw: Any) -> LightRAGSystem:
    system = _offline(models)(None, store, "http://x/v1", "llm", "http://x/v1", "emb", "u1", **kw)
    return system


@pytest.fixture
def models():
    pytest.importorskip("lightrag")
    return FakeModels()


@pytest.fixture
def opened():
    systems: list[LightRAGSystem] = []
    yield systems
    for s in reversed(systems):
        s.close()


ALICE_V1 = (
    "ENTITY Alice Rivera | person | Alice works at Old Org in Boston\n"
    "ENTITY Old Org | organization | Old Org is a firm\n"
    "ENTITY Shared Co | organization | Shared Co is a partner (seen by A)\n"
    "REL Alice Rivera | Old Org | employment | Alice works at Old Org\n"
)
ALICE_V2 = (
    "ENTITY Alice Rivera | person | Alice works at New Org in Boston\n"
    "ENTITY New Org | organization | New Org is a startup\n"
    "REL Alice Rivera | New Org | employment | Alice works at New Org\n"
)
BOB_V1 = (
    "ENTITY Bob Chen | person | Bob lives in Denver\n"
    "ENTITY Shared Co | organization | Shared Co is a partner (seen by B)\n"
    "REL Bob Chen | Shared Co | employment | Bob works at Shared Co\n"
)


def _derived(system: LightRAGSystem) -> dict[str, dict[str, Any]]:
    rows = system.export_raw()
    out: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r["kind"] == "entity":
            out[f"entity:{r['id']}"] = r
        elif r["kind"] == "relation":
            out[f"relation:{r['source']}|{r['target']}"] = r
    return out


def test_revision_is_delete_then_insert_and_deletion_rebuilds_from_cache(models, opened, tmp_path):
    s = _open(models, tmp_path / "store")
    opened.append(s)
    first = s.ingest([_ev(1, "contacts:p01", 1, ALICE_V1), _ev(2, "contacts:p02", 1, BOB_V1)])
    assert first["ingested"] == 2 and first["llm_calls"] == 4  # one extraction + one gleaning each
    shared = _derived(s)["entity:Shared Co"]
    assert (
        shared["description"]
        == "Shared Co is a partner (seen by A)<SEP>Shared Co is a partner (seen by B)"
    )
    assert shared["file_path"] == "contacts:p01#1<SEP>contacts:p02#1"

    models.reset()
    s.ingest([_ev(3, "contacts:p01", 2, ALICE_V2, op="update")], phase="maintain")
    # only the new revision is extracted; the old one's contribution is gone without any LLM call
    assert models.calls == {"extract": 1, "glean": 1, "summary": 0, "keywords": 0}
    derived = _derived(s)
    assert "entity:Old Org" not in derived and "relation:Alice Rivera|Old Org" not in derived
    assert derived["entity:Shared Co"]["description"] == "Shared Co is a partner (seen by B)"
    assert derived["entity:Shared Co"]["source_id"] == "contacts:p02-chunk-000"
    assert derived["entity:Shared Co"]["file_path"] == "contacts:p02#1"
    assert derived["entity:Alice Rivera"]["description"] == "Alice works at New Org in Boston"
    assert derived["entity:Alice Rivera"]["file_path"] == "contacts:p01#2"
    assert derived["entity:Alice Rivera"]["tracked"]["chunk_ids"] == ["contacts:p01-chunk-000"]
    assert s._revision == {"contacts:p01": 2, "contacts:p02": 1}

    models.reset()
    out = s.ingest([_ev(4, "contacts:p02", 2, op="delete")], phase="maintain")
    assert out["ingested"] == 1 and models.calls == dict.fromkeys(models.calls, 0)
    derived = _derived(s)
    assert sorted(derived) == [
        "entity:Alice Rivera",
        "entity:New Org",
        "relation:Alice Rivera|New Org",
    ]
    assert s._revision == {"contacts:p01": 2}
    docs = {r["doc_id"]: r for r in s.export_raw() if r["kind"] == "document"}
    assert list(docs) == ["contacts:p01"] and docs["contacts:p01"]["file_path"] == "contacts:p01#2"
    assert "revision 2" in docs["contacts:p01"]["full_doc"]["content"]
    assert sorted(s.ledger.entries) == [
        "contacts:p01#1",
        "contacts:p01#2",
        "contacts:p02#1",
        "contacts:p02#2",
    ]
    assert s.cost["ingest_tokens"] > 0 and s.cost["maintain_tokens"] > 0
    assert s.ingest([_ev(3, "contacts:p01", 2, ALICE_V2, op="update")])["skipped"] == 1


def test_search_serves_current_revisions_only(models, opened, tmp_path):
    s = _open(models, tmp_path / "store")
    opened.append(s)
    s.ingest(
        [
            _ev(1, "contacts:p01", 1, ALICE_V1),
            _ev(2, "contacts:p02", 1, BOB_V1),
            _ev(3, "contacts:p01", 2, ALICE_V2, op="update"),
            _ev(4, "contacts:p02", 2, op="delete"),
        ]
    )
    hits = s._search("Where does Alice Rivera work?")
    assert hits and models.calls["keywords"] == 1
    assert {r for h in hits for r in h.refs} == {("contacts:p01", 2)}
    text = "\n".join(h.text for h in hits)
    assert "New Org" in text and "Old Org" not in text and "Bob" not in text
    assert hits[0].hit_id.startswith("chunk:contacts:p01-chunk-") and hits[0].ts == T0 + timedelta(
        minutes=3
    )
    res = s.serve(type("P", (), {"question_text": "Where does Alice Rivera work?"})())
    assert res.blocks and set(res.refs) == {("contacts:p01", 2)} and not res.blocked


def test_snapshot_restore_and_resume(models, opened, tmp_path):
    s = _open(models, tmp_path / "prefix")
    opened.append(s)
    prefix = [_ev(1, "contacts:p01", 1, ALICE_V1), _ev(2, "contacts:p02", 1, BOB_V1)]
    s.ingest(prefix)
    s.snapshot(tmp_path / "snap")
    s.ingest([_ev(3, "contacts:p01", 2, ALICE_V2, op="update")], phase="maintain")
    assert s._revision["contacts:p01"] == 2
    s.close()
    opened.remove(s)

    branch = _open(models, tmp_path / "branch")
    opened.append(branch)
    branch.restore(tmp_path / "snap")
    assert branch._revision == {"contacts:p01": 1, "contacts:p02": 1}
    assert branch.usage.snapshot()["llm_calls"] == 4  # a branch starts from the prefix accounting
    assert "entity:Old Org" in _derived(branch)
    assert branch.ingest(prefix)["skipped"] == 2
    branch.ingest([_ev(3, "contacts:p01", 2, ALICE_V2, op="update")], phase="maintain")
    assert "entity:Old Org" not in _derived(branch) and "entity:New Org" in _derived(branch)
    assert branch.cost["maintain_tokens"] > 0 and branch.cost["ingest_tokens"] > 0


def test_reconcile_rebuilds_ledger_map_and_purges_partial_documents(models, opened, tmp_path):
    store = tmp_path / "store"
    s = _open(models, store)
    s.ingest(
        [_ev(1, "contacts:p01", 1, ALICE_V1), _ev(2, "contacts:p01", 2, ALICE_V2, op="update")]
    )
    s.close()
    (store / "ledger.jsonl").unlink()
    (store / nl.REVISIONS_FILE).unlink()

    s = _open(models, store)
    opened.append(s)
    assert s._revision == {"contacts:p01": 2}
    assert s._event_ts == {"contacts:p01": "2026-03-01T09:32"}  # from the stored text's head
    assert sorted(s.ledger.entries) == ["contacts:p01#2"]
    # a half-ingested document (crash before PROCESSED) is purged on the next open
    s._run(
        s.rag.doc_status.upsert(
            {
                "contacts:p09": {
                    "status": "processing",
                    "content_summary": "x",
                    "content_length": 1,
                    "created_at": "2026-03-01T00:00:00+00:00",
                    "updated_at": "2026-03-01T00:00:00+00:00",
                    "file_path": "contacts:p09#1",
                }
            }
        )
    )
    s.close()
    opened.remove(s)
    again = _open(models, store)
    opened.append(again)
    assert [r["doc_id"] for r in again.export_raw() if r["kind"] == "document"] == ["contacts:p01"]


def test_failed_insert_after_delete_leaves_a_consistent_store(models, opened, tmp_path):
    s = _open(models, tmp_path / "store", on_error="skip")
    opened.append(s)
    s.ingest([_ev(1, "contacts:p01", 1, ALICE_V1)])
    models.fail = True
    s.ingest([_ev(2, "contacts:p01", 2, ALICE_V2, op="update")])
    models.fail = False
    assert s.stats["failed"] == 1 and "contacts:p01#2" not in s.ledger
    assert s._revision == {}  # the old revision was deleted, the new one never arrived
    assert not [r for r in s.export_raw() if r["kind"] in ("document", "entity")]
    failure = json.loads((tmp_path / "store" / "failures.jsonl").read_text().splitlines()[0])
    assert failure["key"] == "contacts:p01#2" and "endpoint down" in failure["error"]

    s.ingest([_ev(2, "contacts:p01", 2, ALICE_V2, op="update")])  # retried: only an insert
    assert s._revision == {"contacts:p01": 2} and "entity:New Org" in _derived(s)
    assert "entity:Old Org" not in _derived(s)


def test_on_error_raise_propagates_after_bookkeeping(models, opened, tmp_path):
    s = _open(models, tmp_path / "store")  # on_error="raise" is the default
    opened.append(s)
    models.fail = True
    with pytest.raises(RuntimeError, match="endpoint down"):
        s.ingest([_ev(1, "contacts:p01", 1, ALICE_V1)])
    assert "contacts:p01#1" not in s.ledger and s._revision == {}
    assert not [r for r in s.export_raw() if r["kind"] == "document"]


def test_batched_ingest_matches_one_by_one(models, opened, tmp_path):
    events = [
        _ev(1, "contacts:p01", 1, ALICE_V1),
        _ev(2, "contacts:p02", 1, BOB_V1),
        _ev(3, "contacts:p01", 2, ALICE_V2, op="update"),  # same object: next batch
        _ev(4, "contacts:p02", 2, op="delete"),
    ]
    states = []
    for size, name in ((1, "single"), (3, "batched")):
        s = _open(models, tmp_path / name, batch_size=size)
        s.ingest(events)
        states.append(
            (
                {k: (v["description"], v["file_path"]) for k, v in _derived(s).items()},
                s._revision,
                sorted(s.ledger.entries),
                s.usage.snapshot()["llm_calls"],
            )
        )
        s.close()
    assert states[0] == states[1]
    assert states[1][1] == {"contacts:p01": 2} and states[1][2] == [
        "contacts:p01#1",
        "contacts:p01#2",
        "contacts:p02#1",
        "contacts:p02#2",
    ]


def test_batch_with_one_failing_document_keeps_the_others(models, opened, tmp_path):
    real = models.llm

    def llm(usage):
        inner = real(usage)

        async def fn(prompt, *a, **kw):
            if "Bob lives in Denver" in prompt:
                raise RuntimeError("bad document")
            return await inner(prompt, *a, **kw)

        return fn

    models.llm = llm
    s = _open(models, tmp_path / "store", batch_size=4, on_error="skip")
    opened.append(s)
    s.ingest([_ev(1, "contacts:p01", 1, ALICE_V1), _ev(2, "contacts:p02", 1, BOB_V1)])
    assert s._revision == {"contacts:p01": 1} and s.stats["failed"] == 1
    assert sorted(s.ledger.entries) == ["contacts:p01#1"]
    assert {r["doc_id"] for r in s.export_raw() if r["kind"] == "document"} == {"contacts:p01"}


def test_model_calls_use_the_configured_endpoints_and_meter_usage(monkeypatch):
    pytest.importorskip("lightrag")
    import asyncio

    from lightrag.llm import openai as lo

    seen: dict[str, Any] = {}

    async def fake_complete(model, prompt, **kw):
        seen["llm"] = (model, prompt, kw)
        kw["token_tracker"].add_usage({"prompt_tokens": 30, "completion_tokens": 5})
        return "ok"

    async def fake_embed(texts, **kw):
        seen["embed"] = (texts, kw)
        kw["token_tracker"].add_usage({"prompt_tokens": 9, "total_tokens": 9})
        return np.zeros((len(texts), 1024), dtype=np.float32)

    monkeypatch.setattr(lo, "openai_complete_if_cache", fake_complete)
    monkeypatch.setattr(lo.openai_embed, "func", fake_embed)
    system = object.__new__(LightRAGSystem)
    system.usage = Usage()
    system.llm_model, system.llm_base_url = "writer", "http://llm/v1"
    system.embed_model, system.embed_base_url = "embedder", "http://emb/v1"
    system.llm_extra_body = {"chat_template_kwargs": {"enable_thinking": False}}
    system.max_output_tokens = 4096

    async def go():
        llm, embed = system._llm_func(), system._embed_func()
        await llm("q", system_prompt="s", hashing_kv=object(), max_tokens=10**6, response_format={})
        await embed(["a", "b"], context="query", _priority=5)

    asyncio.run(go())
    model, prompt, kw = seen["llm"]
    assert (model, prompt, kw["base_url"], kw["temperature"]) == (
        "writer",
        "q",
        "http://llm/v1",
        0.0,
    )
    assert kw["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert kw["max_tokens"] == 4096 and kw["system_prompt"] == "s" and "hashing_kv" not in kw
    assert (
        seen["embed"][1]["model"] == "embedder" and seen["embed"][1]["base_url"] == "http://emb/v1"
    )
    assert "dimensions" not in seen["embed"][1] and "embedding_dim" not in seen["embed"][1]
    assert system.usage.snapshot() == {
        "llm_calls": 1,
        "prompt_tokens": 30,
        "completion_tokens": 5,
        "embed_calls": 1,
        "embed_tokens": 9,
    }


def test_a_second_open_store_in_the_process_is_refused(models, opened, tmp_path):
    s = _open(models, tmp_path / "one")
    opened.append(s)
    s.ingest([_ev(1, "contacts:p01", 1, ALICE_V1)])
    with pytest.raises(RuntimeError, match="already open"):
        _open(models, tmp_path / "two")
    assert "entity:Alice Rivera" in _derived(s)  # the first store is untouched
    s.close()
    opened.remove(s)
    other = _open(models, tmp_path / "two")  # fine once the first is closed
    opened.append(other)
    assert _derived(other) == {}  # and it does not see the first store's data


def test_export_raw_carries_stored_fields(models, opened, tmp_path):
    s = _open(models, tmp_path / "store")
    opened.append(s)
    s.ingest([_ev(1, "contacts:p01", 1, ALICE_V1), _ev(2, "contacts:p02", 1, BOB_V1)])
    rows = s.export_raw()
    kinds = sorted({r["kind"] for r in rows})
    assert kinds == ["chunk", "document", "entity", "relation"]
    chunk = next(r for r in rows if r["kind"] == "chunk" and r["full_doc_id"] == "contacts:p01")
    assert chunk["chunk_id"] == "contacts:p01-chunk-000" and chunk["file_path"] == "contacts:p01#1"
    assert "revision 1" in chunk["content"] and chunk["llm_cache_list"] and chunk["tokens"] > 0
    entity = next(r for r in rows if r["kind"] == "entity" and r["id"] == "Alice Rivera")
    for field in ("entity_type", "description", "source_id", "file_path", "created_at"):
        assert entity[field]
    assert entity["tracked"]["chunk_ids"] == ["contacts:p01-chunk-000"]
    relation = next(r for r in rows if r["kind"] == "relation" and r["target"] == "Old Org")
    assert relation["keywords"] == "employment" and relation["file_path"] == "contacts:p01#1"
    assert s.identifiable and s.retains_raw_text
    assert s.describe()["query"]["mode"] == "mix" and s.describe()["top_k"] == 40

"""Offline tests for the Cognee adapter (E5 native systems). Pure helpers run on fixtures;
the batch logic runs on a subclass whose add / cognify / delete are fakes; whatever needs the
`cognee` library is skipped when it is not importable, so `make check` passes without it."""

import asyncio
import dataclasses
import json
import sqlite3
import sys
import types
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from uuid import UUID

import pytest

from memshift.adapters import native_cognee as nc
from memshift.adapters.native import Hit, Usage
from memshift.adapters.native_cognee import (
    CogneeSystem,
    StoredDoc,
    chunk_hit,
    data_id_for,
    entity_block,
    entity_edge_identities,
    errored_data_ids,
    hybrid_hits,
    litellm_model,
    llm_args,
    read_store_docs,
    rebase_locations,
    refs_for_keys,
    source_ref_data_id,
)
from memshift.gen.model import Event

T0 = datetime(2026, 3, 1, 9, 30)
DS = "7ac4b982-9aa1-5a91-8d43-0683c366b870"


def _ev(i: int, oid: str, rev: int, op: str = "insert", minutes: int = 0) -> Event:
    return Event(
        event_id=f"ev{i:06d}",
        ts=T0 + timedelta(minutes=minutes or i),
        source=oid.split(":")[0],
        object_id=oid,
        revision=rev,
        op=op,
        payload={"text": f"Body of {oid} revision {rev}"},
    )


# ---------------------------------------------------------------------- identities and config
def test_data_id_is_stable_per_object_and_distinct():
    a = data_id_for("email:e1")
    assert a == data_id_for("email:e1") and a != data_id_for("email:e2")
    assert isinstance(a, UUID) and a.version == 5


def test_llm_settings():
    assert litellm_model("Qwen/Qwen3.5-9B") == "openai/Qwen/Qwen3.5-9B"
    assert litellm_model("openai/x") == "openai/x"
    body = {"chat_template_kwargs": {"enable_thinking": False}}
    assert llm_args(0.0, 4096, body) == {
        "temperature": 0.0,
        "max_tokens": 4096,
        "extra_body": body,
    }
    assert llm_args(0.0, None, None) == {"temperature": 0.0}


def test_source_ref_keys():
    d, c = "11111111-1111-1111-1111-111111111111", "22222222-2222-2222-2222-222222222222"
    assert source_ref_data_id(f"source_ref:v1:{DS}:{d}") == d
    assert source_ref_data_id(f"source_ref:v2:{DS}:{d}:{c}") == d
    assert source_ref_data_id("source_ref:v3:a:b") is None
    assert source_ref_data_id("other:v1:a:b") is None
    assert source_ref_data_id("source_ref:v1:a") is None
    by = {d: ("email:e1", 2, T0)}
    keys = [f"source_ref:v1:{DS}:{d}", f"source_ref:v2:{DS}:{d}:{c}", f"source_ref:v1:{DS}:{c}"]
    assert refs_for_keys(keys, by) == [("email:e1", 2)]  # once, unknown owners dropped


# ---------------------------------------------------------------------- relational store
def _make_db(path: Path, rows: list[tuple[str, str, str | None, str | None, str, str]]) -> None:
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE datasets (id CHAR(32) PRIMARY KEY, name TEXT)")
    con.execute(
        "CREATE TABLE data (id CHAR(32) PRIMARY KEY, dataset_id CHAR(32), external_metadata JSON,"
        " pipeline_status JSON, raw_data_location TEXT, original_data_location TEXT)"
    )
    con.execute("INSERT INTO datasets VALUES (?, 'memshift')", (DS.replace("-", ""),))
    for did, ds, meta, status, raw, orig in rows:
        con.execute(
            "INSERT INTO data VALUES (?, ?, ?, ?, ?, ?)", (did, ds, meta, status, raw, orig)
        )
    con.commit()
    con.close()


def _row(hex_id: str, oid: str, rev: int, done: bool, loc: str = "file:///old/data/x.txt"):
    meta = json.dumps({"object_id": oid, "revision": rev, "ts": "2026-03-01T09:30"})
    slot = "DATA_ITEM_PROCESSING_COMPLETED" if done else None
    status = json.dumps({"cognify_pipeline": {DS: slot}}) if done else json.dumps({})
    return (hex_id, DS.replace("-", ""), meta, status, loc, loc)


def test_read_store_docs(tmp_path):
    assert read_store_docs(tmp_path / "missing") == (None, [])
    db = tmp_path / "cognee_db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE other (x)")
    con.close()
    assert read_store_docs(db) == (None, [])  # tables not created yet
    db = tmp_path / "db2"
    a, b = "a" * 32, "b" * 32
    _make_db(db, [_row(a, "email:e1", 2, True), _row(b, "notes:n1", 1, False)])
    dataset_id, docs = read_store_docs(db)
    assert dataset_id == DS
    by = {d.object_id: d for d in docs}
    assert by["email:e1"] == StoredDoc(str(UUID(a * 1)), "email:e1", 2, T0, True)
    assert by["notes:n1"].completed is False and by["notes:n1"].revision == 1


def test_rebase_locations(tmp_path):
    db = tmp_path / "db"
    _make_db(db, [_row("c" * 32, "email:e1", 1, True, "file:///old/data/text_1.txt")])
    assert rebase_locations(db, "/old/data", "/old/data") == 0
    assert rebase_locations(db, "/old/data", "/new/root/data") == 2
    con = sqlite3.connect(db)
    raw, orig = con.execute("SELECT raw_data_location, original_data_location FROM data").fetchone()
    con.close()
    assert raw == orig == "file:///new/root/data/text_1.txt"


# ---------------------------------------------------------------------- retrieval -> hits
D1, D2 = "11111111-1111-4111-8111-111111111111", "22222222-2222-4222-8222-222222222222"
BY = {D1: ("email:e1", 2, T0), D2: ("contacts:p01", 1, T0 + timedelta(days=3))}


def _chunk(text: str, md: dict[str, Any] | str | None, doc_id: str | None = None, cid="c1"):
    payload: dict[str, Any] = {"id": cid, "text": text, "document_id": doc_id}
    if md is not None:
        payload["external_metadata"] = md if isinstance(md, str) else json.dumps(md)
    return SimpleNamespace(id=cid, score=0.75, payload=payload)


def test_chunk_hit_reads_source_from_external_metadata():
    md = {"object_id": "email:e1", "revision": 2, "source": "email", "ts": "2026-03-01T09:35"}
    h = chunk_hit(_chunk("[email record email:e1 | revision 2 | ...]\nBody", md), 0, BY)
    assert h is not None and h.text.endswith("Body") and h.hit_id == "c1"
    assert h.refs == [("email:e1", 2)] and h.ts == datetime(2026, 3, 1, 9, 35) and h.score == 1.0
    by_doc = chunk_hit(_chunk("text", None, D2), 3, BY)
    assert by_doc is not None and by_doc.refs == [("contacts:p01", 1)] and by_doc.score == 0.25
    assert by_doc.ts == T0 + timedelta(days=3)
    assert chunk_hit(_chunk("   ", md), 0, BY) is None
    junk = chunk_hit(_chunk("t", "not json", None), 0, BY)
    assert junk is not None and junk.refs == [] and junk.ts == nc._EPOCH


ENTITY = {
    "id": "e-alice",
    "name": "alice",
    "type": "person",
    "description": "Alice Johnson",
    "edges": [
        {
            "text": "alice works at acme",
            "source_id": "e-alice",
            "target_id": "e-acme",
            "relationship": "works_at",
        },
        {"text": "", "source_id": "x", "target_id": "y", "relationship": "r"},
    ],
}


def test_entity_block_matches_cognee_rendering():
    assert entity_block(ENTITY) == "### alice (person)\nAlice Johnson\n- alice works at acme"
    assert entity_block({"name": "bob"}) == "### bob"
    assert entity_block({"id": "x", "name": " "}) == ""
    assert entity_edge_identities(ENTITY) == [
        ("e-alice", "e-acme", "works_at"),
        ("x", "y", "r"),
    ]


def test_hybrid_hits_order_refs_and_sources():
    md = {"object_id": "email:e1", "revision": 2}
    retrieved = {
        "chunks": [_chunk("passage one", md), _chunk("passage two", None, D2, cid="c2")],
        "chunk_summaries": {"c1": "ignored: Cognee does not render it"},
        "entities": [ENTITY, {"id": "e-none", "name": ""}],
        "facts": [{"id": "f1", "text": "Bob lives in Lisbon"}, {"id": "f2", "text": ""}],
    }
    node_keys = {"e-alice": [f"source_ref:v2:{DS}:{D1}:cccc"]}
    edge_keys = {("e-alice", "e-acme", "works_at"): [f"source_ref:v1:{DS}:{D2}"]}
    hits = hybrid_hits(retrieved, node_keys, edge_keys, BY)
    assert [h.hit_id for h in hits] == ["c1", "c2", "e-alice", "f1"]  # passages, entities, facts
    ent = hits[2]
    assert ent.text == "### alice (person)\nAlice Johnson\n- alice works at acme"
    assert ent.refs == [("email:e1", 2), ("contacts:p01", 1)]  # node owner then edge owner
    assert ent.ts == T0 + timedelta(days=3)  # latest source event
    fact = hits[3]
    assert fact.refs == [] and fact.text == "Bob lives in Lisbon"
    assert all(isinstance(h, Hit) for h in hits)
    assert hybrid_hits({}, {}, {}, BY) == []


def test_errored_data_ids():
    ok = SimpleNamespace(status="PipelineRunCompleted")
    bad = SimpleNamespace(status="PipelineRunErrored")
    run = SimpleNamespace(
        status="PipelineRunErrored",
        data_ingestion_info=[
            {"run_info": ok, "data_id": "d1"},
            {"run_info": bad, "data_id": "d2"},
            {"run_info": bad},
            "junk",
        ],
    )
    assert errored_data_ids(run) == {"d2"}
    assert errored_data_ids(SimpleNamespace(status="PipelineRunErrored")) == set()
    assert nc.run_failed(run) and not nc.run_failed(SimpleNamespace(status="PipelineRunCompleted"))


# ---------------------------------------------------------------------- usage metering
def test_litellm_meter_counts_calls_tokens_and_errors(monkeypatch):
    calls = []

    async def acompletion(*args: Any, **kw: Any) -> Any:
        calls.append(kw)
        if kw.get("boom"):
            raise RuntimeError("400")
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=7))

    fake = types.ModuleType("litellm")
    fake.acompletion = acompletion
    monkeypatch.setitem(sys.modules, "litellm", fake)
    usage = Usage()
    monkeypatch.setattr(nc, "_METER", SimpleNamespace(usage=usage, errors=0))
    nc._install_meters()
    nc._install_meters()  # idempotent: not wrapped twice

    async def go() -> None:
        await fake.acompletion(model="m", messages=[])
        with pytest.raises(RuntimeError):
            await fake.acompletion(model="m", boom=True)

    asyncio.run(go())
    assert usage.snapshot()["llm_calls"] == 1 and usage.snapshot()["prompt_tokens"] == 100
    assert usage.snapshot()["completion_tokens"] == 7 and nc._METER.errors == 1
    assert len(calls) == 2


def test_register_schema_support_marks_the_served_model(monkeypatch):
    seen = []
    fake = types.ModuleType("litellm")
    fake.register_model = seen.append
    monkeypatch.setitem(sys.modules, "litellm", fake)
    nc.register_schema_support("openai/Qwen/Qwen3.5-9B")
    nc.register_schema_support("Qwen/Qwen3.5-9B")
    assert (
        seen[0]
        == seen[1]
        == {
            "Qwen/Qwen3.5-9B": {
                "litellm_provider": "openai",
                "mode": "chat",
                "supports_response_schema": True,
            }
        }
    )


# ---------------------------------------------------------------------- batch logic on fakes
class _FakeCognee(CogneeSystem):
    """Cognee without the library: the store is a dict of data id -> text; cognify completes
    every added item unless its id is in `poison`."""

    def _open(self) -> None:
        self.store: dict[str, str] = getattr(self, "store", {})
        self.adds: list[list[dict[str, Any]]] = getattr(self, "adds", [])
        self.deletes: list[str] = getattr(self, "deletes", [])
        self.cognify_runs = getattr(self, "cognify_runs", 0)
        self.poison: set[str] = getattr(self, "poison", set())
        self.pending: set[str] = getattr(self, "pending", set())

    def _close(self) -> None:
        pass

    async def _delete(self, data_id: str) -> None:
        self.deletes.append(data_id)
        self.store.pop(data_id, None)
        self.pending.discard(data_id)

    async def _add(self, specs: list[dict[str, Any]]) -> str:
        self.adds.append(specs)
        for s in specs:
            self.store[str(s["data_id"])] = s["data"]
            self.pending.add(str(s["data_id"]))
        return DS

    async def _cognify_run(self) -> Any:
        self.cognify_runs += 1
        self.usage.add(llm_calls=2, prompt_tokens=10, completion_tokens=2)
        bad = {d for d in self.pending if d in self.poison}
        if bad:
            entries = [
                {
                    "run_info": SimpleNamespace(
                        status="PipelineRunErrored" if d in bad else "PipelineRunCompleted"
                    ),
                    "data_id": d,
                }
                for d in self.pending
            ]
            return SimpleNamespace(
                status="PipelineRunErrored", error_message="bad json", data_ingestion_info=entries
            )
        self.pending.clear()
        return SimpleNamespace(status="PipelineRunCompleted")


def _fake(tmp_path: Path, **kw: Any) -> _FakeCognee:
    return _FakeCognee(
        None, tmp_path / "s", "http://x/v1", "Qwen/Qwen3.5-9B", "http://x/v1", "emb", "w1", **kw
    )


def test_revision_is_delete_then_add_and_delete_event_deletes(tmp_path):
    s = _fake(tmp_path, batch_size=4)
    e1, e2 = _ev(1, "email:e1", 1), _ev(2, "notes:n1", 1)
    out = s.ingest([e1, e2])
    assert out["ingested"] == 2 and s.cognify_runs == 1 and len(s.adds) == 1  # one add + cognify
    d1 = str(data_id_for("email:e1"))
    spec = s.adds[0][0]
    assert spec["label"] == "email:e1" and spec["data_id"] == data_id_for("email:e1")
    assert spec["external_metadata"]["revision"] == 1
    assert spec["data"].startswith("[email record email:e1 | revision 1 |")
    rev = _ev(3, "email:e1", 2, "update")
    gone = _ev(4, "notes:n1", 2, "delete")
    s.ingest([rev, gone], "maintain")
    assert s.deletes == [d1, str(data_id_for("notes:n1"))]  # old revision, then the deleted object
    assert set(s.store) == {d1} and "revision 2" in s.store[d1]
    assert [d.revision for d in s._current.values()] == [2]
    assert {"email:e1#2", "notes:n1#2"} <= set(s.ledger.entries)
    assert s.cost["ingest_tokens"] == 12 and s.cost["maintain_tokens"] == 12


def test_batches_never_hold_an_object_twice(tmp_path):
    s = _fake(tmp_path, batch_size=10)
    s.ingest([_ev(1, "email:e1", 1), _ev(2, "email:e1", 2, "update"), _ev(3, "a:x", 1)])
    assert s.cognify_runs == 2 and len(s.adds) == 2  # the revision opens a second batch
    assert "revision 2" in s.store[str(data_id_for("email:e1"))]


def test_resumed_batch_does_not_add_twice(tmp_path):
    s = _fake(tmp_path, batch_size=4)
    evs = [_ev(1, "email:e1", 1), _ev(2, "notes:n1", 1)]
    s.ingest(evs)
    (s.store_dir / "ledger.jsonl").unlink()  # crash before the ledger was written
    s.ledger.entries.clear()
    s.ingest(evs)
    assert len(s.adds) == 1 and s.deletes == []  # nothing stored twice, nothing deleted
    assert len(s.ledger) == 2


def test_interrupted_batch_is_cognified_not_added_again(tmp_path):
    s = _fake(tmp_path, batch_size=4)
    evs = [_ev(1, "email:e1", 1), _ev(2, "notes:n1", 1)]
    s.ingest(evs)
    (s.store_dir / "ledger.jsonl").unlink()
    s.ledger.entries.clear()
    for oid in ("email:e1", "notes:n1"):  # stored, but the run died before cognify finished
        s._current[oid] = dataclasses.replace(s._current[oid], completed=False)
    s.pending.update(str(data_id_for(oid)) for oid in ("email:e1", "notes:n1"))
    runs = s.cognify_runs
    out = s.ingest(evs)
    assert out["ingested"] == 2 and len(s.adds) == 1 and s.deletes == []  # no second add
    assert s.cognify_runs == runs + 1 and not s.pending
    assert all(d.completed for d in s._current.values())


def test_failed_unit_is_skipped_and_leaves_nothing(tmp_path):
    s = _fake(tmp_path, batch_size=4, on_error="skip")
    s.poison = {str(data_id_for("notes:n1"))}
    out = s.ingest([_ev(1, "email:e1", 1), _ev(2, "notes:n1", 1), _ev(3, "a:x", 1)])
    assert out["ingested"] == 2 and s.stats["failed"] == 1
    assert s.cognify_runs == 2  # the failed run rolled back, then repeated without the bad item
    assert str(data_id_for("notes:n1")) not in s.store  # deleted, so the next cognify skips it
    assert "notes:n1" not in s._current and set(s.ledger.entries) == {"email:e1#1", "a:x#1"}
    failures = (s.store_dir / "failures.jsonl").read_text()
    assert "notes:n1#1" in failures and "bad json" in failures
    strict = _fake(tmp_path / "raise", batch_size=4)
    strict.poison = {str(data_id_for("notes:n1"))}
    with pytest.raises(RuntimeError, match="bad json"):
        strict.ingest([_ev(1, "notes:n1", 1)])


def test_reconcile_rebuilds_state_from_the_relational_store(tmp_path):
    s = _fake(tmp_path)
    a, b = str(data_id_for("email:e1")).replace("-", ""), "b" * 32
    s.db_path.parent.mkdir(parents=True, exist_ok=True)
    _make_db(s.db_path, [])
    con = sqlite3.connect(s.db_path)
    for row in (_row(a, "email:e1", 3, True), _row(b, "notes:n1", 1, False)):
        con.execute("INSERT INTO data VALUES (?, ?, ?, ?, ?, ?)", row)
    con.commit()
    con.close()
    s._reconcile()
    assert s._dataset_id == DS and set(s._current) == {"email:e1", "notes:n1"}
    assert "email:e1#3" in s.ledger and "notes:n1#1" not in s.ledger  # only completed units
    assert s._by_data_id()[str(UUID(a))][:2] == ("email:e1", 3)


def test_serve_without_data_returns_no_hits(tmp_path):
    s = _fake(tmp_path)
    probe = SimpleNamespace(question_text="anything")
    res = s.serve(probe)
    assert res.blocks == [] and res.refs == []


def test_describe_and_defaults(tmp_path):
    s = _fake(tmp_path, structured_output="litellm_native")
    d = s.describe()
    assert d["system"] == "cognee" and d["top_k"] == nc.DEFAULT_TOP_K == 15
    assert d["search_type"] == "HYBRID_COMPLETION" and d["graph_store"].startswith("ladybug")
    assert d["response_format"] == "json_schema" and d["structured_output"] == "litellm_native"
    assert _fake(tmp_path / "o", json_schema=False).describe()["response_format"] == "json_object"
    assert s.retains_raw_text and s.identifiable
    assert s.system_root == (tmp_path / "s").resolve() / "system"


# ---------------------------------------------------------------------- with the library
def test_context_rendering_matches_cognee():
    pytest.importorskip("cognee")
    from cognee.modules.retrieval.hybrid.context import format_passages
    from cognee.modules.retrieval.hybrid.entities import format_entities

    assert format_entities([ENTITY]).split("\n", 1)[1] == entity_block(ENTITY)
    chunk = _chunk("passage text", None)
    assert format_passages([chunk]).split("\n", 1)[1] == chunk_hit(chunk, 0, BY).text


def test_configure_sets_embedded_stores_and_endpoints(tmp_path):
    pytest.importorskip("cognee")

    class Configured(CogneeSystem):
        def _open(self) -> None:
            nc._quiet_env()
            self._configure()

        def _close(self) -> None:
            pass

    s = Configured(
        None,
        tmp_path / "s",
        "http://llm/v1",
        "Qwen/Qwen3.5-9B",
        "http://emb/v1",
        "Qwen/Qwen3-Embedding-0.6B",
        "w1",
    )
    from cognee.infrastructure.databases.graph.config import get_graph_config
    from cognee.infrastructure.databases.relational import get_relational_config
    from cognee.infrastructure.databases.vector import get_vectordb_config
    from cognee.infrastructure.databases.vector.embeddings.config import get_embedding_config
    from cognee.infrastructure.llm.config import get_llm_config

    root = (tmp_path / "s").resolve()
    assert get_graph_config().graph_database_provider == "ladybug"
    assert get_graph_config().graph_file_path.startswith(str(root / "system"))
    assert get_graph_config().graph_database_subprocess_enabled is False
    assert get_vectordb_config().vector_db_provider == "lancedb"
    assert get_vectordb_config().vector_db_url.startswith(str(root / "system"))
    assert get_vectordb_config().vector_db_subprocess_enabled is False
    assert get_relational_config().db_provider == "sqlite"
    assert get_relational_config().db_path.startswith(str(root / "system"))
    llm = get_llm_config()
    assert llm.llm_model == "openai/Qwen/Qwen3.5-9B" and llm.llm_endpoint == "http://llm/v1"
    assert llm.structured_output_framework == "litellm_native"
    assert llm.llm_args["temperature"] == 0.0
    assert llm.llm_args["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    emb = get_embedding_config()
    assert emb.embedding_provider == "openai_compatible" and emb.embedding_dimensions == 1024
    assert emb.embedding_endpoint == "http://emb/v1"
    assert s.data_root == root / "data"

"""Offline tests for the HippoRAG 2 adapter (prereg/v2.md §3): passage headers, the object map
and its sequencing, the embedder, the OpenIE audit, and, when `hipporag` is installed
(docker/native/hipporag.txt), the adapter end to end on fake OpenAI-compatible clients. No
network, no GPU; the library-backed tests are skipped without the package."""

import hashlib
import importlib.util
import json
import re
import sys
from datetime import datetime, timedelta
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from memshift.adapters import native_hipporag
from memshift.adapters.native import MeteredOpenAI, Usage, _Embeddings, event_key, event_text
from memshift.adapters.native_hipporag import (
    HippoRAGSystem,
    ObjectMap,
    OpenAIEmbedder,
    OpenIEAudit,
    embedding_texts,
    hipporag_hit,
    import_hipporag,
    objects_from_passages,
    parse_header,
    plan_event,
)
from memshift.gen.model import Event

T0 = datetime(2026, 3, 1, 9, 0)


def _ev(i: int, oid: str, rev: int, text: str = "", op: str = "insert") -> Event:
    return Event(
        event_id=f"ev{i:06d}",
        ts=T0 + timedelta(minutes=i),
        source=oid.split(":")[0],
        object_id=oid,
        revision=rev,
        op=op,
        payload={"text": text or f"Body of {oid} revision {rev}"},
    )


# ---------------------------------------------------------------------- passages and hits
def test_parse_header_round_trips_event_text():
    e = _ev(35, "calendar:series:3", 4, "Yoga at 7", op="update")
    head = parse_header(event_text(e))
    assert head is not None
    assert (head.source, head.object_id, head.revision) == ("calendar", "calendar:series:3", 4)
    assert head.ts == datetime(2026, 3, 1, 9, 35)
    assert parse_header("just some text") is None
    assert parse_header("x\n" + event_text(e)) is None  # only a leading header counts
    assert parse_header("[email record email:e1 | revision x | 2026-03-01 09:35]") is None


def test_hipporag_hit_takes_refs_and_time_from_the_header():
    text = event_text(_ev(5, "email:e1", 2, "Hello Bob"))
    hit = hipporag_hit("chunk-1", text, 0.25)
    assert hit.text == text and hit.hit_id == "chunk-1" and hit.score == 0.25
    assert hit.refs == [("email:e1", 2)] and hit.ts == datetime(2026, 3, 1, 9, 5)
    bare = hipporag_hit("chunk-2", "no header", 0.0)
    assert bare.refs == [] and bare.text == "no header"


def test_objects_from_passages_keeps_highest_revision():
    texts = [
        event_text(_ev(1, "a:1", 1)),
        event_text(_ev(2, "a:1", 3)),
        event_text(_ev(3, "a:2", 1)),
        event_text(_ev(4, "a:1", 2)),
        "foreign passage",
    ]
    got = objects_from_passages(texts)
    assert set(got) == {"a:1", "a:2"}
    assert got["a:1"] == (3, texts[1]) and got["a:2"] == (1, texts[2])


# ---------------------------------------------------------------------- object map and plans
def test_object_map_replay_torn_line_and_sync(tmp_path):
    path = tmp_path / "objects.jsonl"
    m = ObjectMap(path)
    m.put("a:1", 1, "t1")
    m.put("a:2", 1, "u1")
    m.put("a:1", 2, "t2")
    m.remove("a:2")
    m.remove("a:2")  # absent: nothing logged
    with path.open("a") as f:
        f.write('{"object_id": "a:3", "revi')  # a crash mid-write
    again = ObjectMap(path)
    assert again.items == {"a:1": (2, "t2")} and len(again) == 1
    again.sync({"a:1": (2, "t2"), "a:4": (1, "v1")})  # nothing changes, a:4 is added
    assert ObjectMap(path).items == {"a:1": (2, "t2"), "a:4": (1, "v1")}
    again.sync({"a:4": (3, "v3")})  # a:1 is gone from the store, a:4 moved on
    assert ObjectMap(path).items == {"a:4": (3, "v3")}


def test_plan_event_sequencing():
    e1, e2 = _ev(1, "a:1", 1), _ev(2, "a:1", 2, op="update")
    insert = plan_event(None, e1)
    assert insert is not None and (insert.drop, insert.add) == (None, event_text(e1))
    update = plan_event((1, event_text(e1)), e2)
    assert update is not None and (update.drop, update.add) == (event_text(e1), event_text(e2))
    assert update.revision == 2
    assert plan_event((2, event_text(e2)), e2) is None  # replayed
    assert plan_event((2, event_text(e2)), e1) is None  # older than what the store holds
    gone = plan_event((2, event_text(e2)), _ev(3, "a:1", 3, op="delete"))
    assert gone is not None and (gone.drop, gone.add) == (event_text(e2), None)
    assert plan_event(None, _ev(4, "a:1", 3, op="delete")) is None  # never ingested
    revived = plan_event(None, _ev(5, "a:1", 4))  # inserted again after a delete
    assert (
        revived is not None and revived.drop is None and revived.add == event_text(_ev(5, "a:1", 4))
    )


# ---------------------------------------------------------------------- embedder and audit
def test_embedding_texts_flatten_newlines_and_prefix_instruction():
    assert embedding_texts(["a\nb", ""]) == ["a b", " "]
    got = embedding_texts(["what?"], "Given a question, find facts.")
    assert got == ["Instruct: Given a question, find facts.\nQuery: what?"]


class _FakeEmbeddings:
    def __init__(self) -> None:
        self.batches: list[list[str]] = []

    def create(self, **kw: Any) -> Any:
        self.batches.append(list(kw["input"]))
        rows = list(enumerate(kw["input"]))
        data = [SimpleNamespace(embedding=[float(len(t)), 3.0, 4.0], index=i) for i, t in rows]
        return SimpleNamespace(data=data[::-1], usage=SimpleNamespace(prompt_tokens=len(rows)))


def test_openai_embedder_batches_orders_and_normalises():
    raw = _FakeEmbeddings()
    usage = Usage()
    client = MeteredOpenAI(
        SimpleNamespace(embeddings=raw, chat=SimpleNamespace(completions=None)), usage
    )
    emb = OpenAIEmbedder(client, "emb", batch_size=2)
    out = emb.batch_encode(["a", "bb", "ccc"], instruction="Q", norm=True)
    assert out.shape == (3, 3) and np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert [len(b) for b in raw.batches] == [2, 1]
    assert raw.batches[0][0] == "Instruct: Q\nQuery: a"
    prefix = len("Instruct: Q\nQuery: ")
    assert out[0, 0] == pytest.approx(float(prefix + 1) / np.linalg.norm([prefix + 1, 3, 4]))
    assert usage.snapshot()["embed_calls"] == 2  # rows came back reversed and were put in order
    plain = OpenAIEmbedder(client, "emb", query_instruction=False).batch_encode(
        "q", instruction="Q"
    )
    assert plain.shape == (1, 3) and raw.batches[-1] == ["q"]
    assert OpenAIEmbedder(client, "emb").batch_encode([], norm=True).size == 0


def test_openie_audit_collects_errors_and_counts():
    audit = OpenIEAudit()
    seen = []

    def ner(chunk_key, passage):
        seen.append(("ner", passage))
        err = {"error": "boom"} if passage == "bad" else {}
        return SimpleNamespace(metadata=err, unique_entities=["x", "y"] if not err else [])

    def triple_extraction(chunk_key, passage, named_entities):
        return SimpleNamespace(metadata={}, triples=[["x", "is", "y"]])

    openie = SimpleNamespace(ner=ner, triple_extraction=triple_extraction)
    audit.wrap(openie)
    openie.ner("k1", "good")
    openie.triple_extraction("k1", "good", ["x", "y"])
    openie.ner("k2", "bad")
    assert seen == [("ner", "good"), ("ner", "bad")]
    assert audit.errors == ["ner: boom"]
    assert audit.counts == {"good": {"entities": 2, "triples": 1}, "bad": {"entities": 0}}
    audit.reset()
    assert audit.errors == [] and audit.counts == {}


def test_stub_backends_only_when_missing(monkeypatch):
    monkeypatch.setattr(native_hipporag, "STUBS", {"memshift_absent_backend": ("Engine",)})
    try:
        native_hipporag._stub_unused_backends()
        mod = sys.modules["memshift_absent_backend"]
        with pytest.raises(RuntimeError, match="not installed"):
            mod.Engine()
        native_hipporag._stub_unused_backends()  # a second call keeps the first stand-in
        assert sys.modules["memshift_absent_backend"] is mod
    finally:
        sys.modules.pop("memshift_absent_backend", None)
    monkeypatch.setattr(native_hipporag, "STUBS", {"json": ("Engine",)})
    native_hipporag._stub_unused_backends()
    assert not hasattr(sys.modules["json"], "Engine")  # importable modules are left alone


def test_import_without_library_says_so(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: None)
    with pytest.raises(ModuleNotFoundError, match="hipporag"):
        import_hipporag()


# ---------------------------------------------------------------------- library-backed
class FakeChat:
    """NER, triple extraction and fact reranking answered from the prompt text."""

    def __init__(self) -> None:
        self.calls = 0
        self.fail = False

    @staticmethod
    def _entities(text: str) -> list[str]:
        found = re.findall(r"[A-Z][a-z]+(?: [A-Z][a-z]+)*", text.split("\n", 1)[-1])
        return list(dict.fromkeys(found))

    def create(self, **kw: Any) -> Any:
        self.calls += 1
        if self.fail:
            raise ConnectionError("endpoint down")
        last = kw["messages"][-1]["content"]
        if last.startswith("[[ ## question"):
            facts = re.search(r"fact_before_filter ## \]\]\n(.*?)\n\nRespond", last, re.S)
            assert facts is not None
            body = json.dumps({"fact": json.loads(facts.group(1))["fact"][:3]})
            out = f"[[ ## fact_after_filter ## ]]\n{body}\n\n[[ ## completed ## ]]"
        elif "Paragraph:" in last:
            m = re.search(r"Paragraph:\n```\n(.*?)\n```", last, re.S)
            assert m is not None
            ents = self._entities(m.group(1))
            out = json.dumps({"triples": [[a, "related to", b] for a, b in pairwise(ents)]})
        else:
            out = json.dumps({"named_entities": self._entities(last)})
        msg = SimpleNamespace(content=out)
        usage = SimpleNamespace(prompt_tokens=len(last) // 4, completion_tokens=len(out) // 4)
        choice = SimpleNamespace(message=msg, finish_reason="stop")
        return SimpleNamespace(choices=[choice], usage=usage)


class FakeEmb:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **kw: Any) -> Any:
        self.calls += 1
        data = []
        for t in kw["input"]:
            v = np.full(64, 0.01)
            t = t.lower()
            for i in range(len(t) - 2):
                v[int(hashlib.md5(t[i : i + 3].encode()).hexdigest(), 16) % 64] += 1.0
            data.append(SimpleNamespace(embedding=v.tolist()))
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=len(data)))


@pytest.fixture(scope="module")
def hippo():
    if importlib.util.find_spec("hipporag") is None:
        pytest.skip("hipporag is not installed")
    try:
        return import_hipporag()
    except ImportError as exc:
        pytest.skip(f"hipporag is not importable here: {exc}")


DEAD = "http://127.0.0.1:9/v1"


def _system(store: Path, chat: FakeChat, emb: FakeEmb, **kw: Any) -> HippoRAGSystem:
    s = HippoRAGSystem(None, store, DEAD, "fake-llm", DEAD, "fake-emb", "u1", **kw)
    _wire(s, chat, emb)
    return s


def _wire(s: HippoRAGSystem, chat: FakeChat, emb: FakeEmb) -> None:
    """Point the metered clients at the fakes (a snapshot / restore reopens the system)."""
    s._hip.llm_model.openai_client.chat.completions._inner = chat
    client = s._hip.embedding_model.client
    client.embeddings = SimpleNamespace(create=_Embeddings(emb, s.usage).create)


P01 = "Alice Johnson lives in Boston and works at Acme Corp."
E01 = "Bob Smith emailed Alice Johnson about Acme Corp and Boston."
N01 = "Carol White visited Denver with Bob Smith."
P01_R2 = "Alice Johnson lives in Denver and works at Globex Inc."
T01 = "Dave Brown owes Zed Young a report."
PREFIX = [
    _ev(1, "contacts:p01", 1, P01),
    _ev(2, "email:e01", 1, E01),
    _ev(3, "notes:n01", 1, N01),
]
BRANCH = [
    _ev(4, "contacts:p01", 2, P01_R2, op="update"),
    _ev(5, "email:e01", 2, op="delete"),
    _ev(6, "tasks:t01", 1, T01),
]


def _rows(s: HippoRAGSystem, kind: str) -> list[dict[str, Any]]:
    return [r for r in s.export_raw() if r["kind"] == kind]


def _phrases(s: HippoRAGSystem) -> set[str]:
    return {r["content"] for r in _rows(s, "phrase")}


def test_replace_and_delete_go_through_hipporag(hippo, tmp_path):
    chat, emb = FakeChat(), FakeEmb()
    s = _system(tmp_path / "store", chat, emb)
    out = s.ingest(PREFIX)
    assert out["ingested"] == 3 and chat.calls == 6 and out["llm_calls"] == 6  # NER + triples
    assert s.identifiable and set(s.objects.items) == {"contacts:p01", "email:e01", "notes:n01"}
    assert _phrases(s) >= {"boston", "acme corp", "bob smith", "denver"}
    s.ingest(BRANCH[:1], "maintain")  # a new revision replaces the old passage
    assert event_text(PREFIX[0]) not in s._passages() and event_text(BRANCH[0]) in s._passages()
    assert s.objects.get("contacts:p01") == (2, event_text(BRANCH[0]))
    assert {r["object_id"] for r in _rows(s, "passage")} == {
        "contacts:p01",
        "email:e01",
        "notes:n01",
    }
    assert "globex inc" in _phrases(s) and "boston" in _phrases(s)  # still in the email
    s.ingest(BRANCH[1:], "maintain")  # a delete event removes the current text
    assert event_text(PREFIX[1]) not in s._passages()
    assert set(s.objects.items) == {"contacts:p01", "notes:n01", "tasks:t01"}
    phrases = _phrases(s)
    assert "boston" not in phrases and "acme corp" not in phrases  # only the email held them
    assert {"bob smith", "denver", "alice johnson"} <= phrases  # other passages still do
    facts = {tuple(r["triple"]) for r in _rows(s, "fact")}
    assert ("carol white", "related to", "denver") in facts
    assert ("alice johnson", "related to", "boston") not in facts
    assert ("bob smith", "related to", "alice johnson") not in facts  # only the email said so
    from hipporag.utils.misc_utils import compute_mdhash_id

    bob, alice = (compute_mdhash_id(p, "entity-") for p in ("bob smith", "alice johnson"))
    pairs = {(r["source"], r["target"]) for r in _rows(s, "edge") if r["edge"] == "fact"}
    assert (bob, alice) in pairs or (alice, bob) in pairs  # its edge outlives the deleted fact
    assert all(r["passages"] for r in _rows(s, "fact") + _rows(s, "phrase"))
    kinds = {r["kind"] for r in s.export_raw()}
    assert kinds == {"passage", "openie", "fact", "phrase", "edge"}
    led = s.ledger.entries["email:e01#2"]
    assert led["op"] == "delete" and led["dropped"] is True
    assert s.ledger.entries["tasks:t01#1"]["entities"] == 2
    assert s.usage.snapshot()["embed_calls"] == emb.calls > 0


def test_snapshot_restore_and_resume(hippo, tmp_path):
    chat, emb = FakeChat(), FakeEmb()
    s = _system(tmp_path / "store", chat, emb)
    s.ingest(PREFIX)
    n_prefix = len(s.export_raw())
    s.snapshot(tmp_path / "snap")
    _wire(s, chat, emb)
    assert s.ingest(PREFIX)["skipped"] == 3  # resumed ingest adds nothing twice
    s.ingest(BRANCH, "maintain")
    assert len(s.ledger) == 6
    s.restore(tmp_path / "snap")
    _wire(s, chat, emb)
    assert len(s.ledger) == 3 and len(s.export_raw()) == n_prefix
    assert set(s.objects.items) == {"contacts:p01", "email:e01", "notes:n01"}
    assert s.cost["ingest_tokens"] > 0 and s.cost["maintain_tokens"] == 0
    s.ingest(BRANCH, "maintain")  # the branch replays identically after the restore
    assert set(s.objects.items) == {"contacts:p01", "notes:n01", "tasks:t01"}
    s.close()
    (tmp_path / "store" / "ledger.jsonl").unlink()
    (tmp_path / "store" / "objects.jsonl").unlink()
    calls = chat.calls
    again = _system(tmp_path / "store", chat, emb)  # rebuilt from the passages in the store
    assert set(again.ledger.entries) >= {"contacts:p01#2", "notes:n01#1", "tasks:t01#1"}
    assert set(again.objects.items) == {"contacts:p01", "notes:n01", "tasks:t01"}
    # a lost ledger would redo the insert of the deleted email (its delete leaves no trace)
    replay = [e for e in PREFIX + BRANCH if e.object_id != "email:e01"]
    assert again.ingest(replay)["llm_calls"] == 0 and chat.calls == calls
    with pytest.raises(ValueError):
        again.restore(tmp_path / "store" / "hipporag")  # not a snapshot of a native store


def test_serve_returns_passages_with_refs_and_sees_new_documents(hippo, tmp_path):
    chat, emb = FakeChat(), FakeEmb()
    s = _system(tmp_path / "store", chat, emb)
    assert s.serve(SimpleNamespace(question_text="anything")).blocks == []  # empty store
    assert s.export_raw() == []
    s.ingest(PREFIX)
    probe = SimpleNamespace(question_text="Where does Alice Johnson live?")
    res = s.serve(probe)
    assert res.blocks and set(res.refs) <= {("contacts:p01", 1), ("email:e01", 1), ("notes:n01", 1)}
    assert res.blocks[0].startswith("[") and "record" in res.blocks[0]
    assert s.cost["serve_tokens"] > 0
    s.ingest(BRANCH[2:], "maintain")  # index after a search: the retrieval state is refreshed
    assert s._hip.ready_to_retrieve is False and s._hip.ent_node_to_chunk_ids is None
    hits = s._search("Who owes Zed Young a report?")
    assert ("tasks:t01", 1) in [r for h in hits for r in h.refs]
    assert len(hits) == 4 and hits[0].score >= hits[-1].score
    boom = s._hip

    def assert_out(*a: Any, **k: Any) -> Any:
        raise AssertionError

    boom.retrieve = assert_out  # HippoRAG asserts on small or tied stores
    before = s.stats["dpr_fallback"]
    assert s._search("Where does Alice live?") and s.stats["dpr_fallback"] == before + 1


def test_openie_failures_are_counted_or_rolled_back(hippo, tmp_path):
    chat, emb = FakeChat(), FakeEmb()
    chat.fail = True
    kept = _system(tmp_path / "kept", chat, emb, on_error="skip")
    kept._hip.llm_model.max_retries = 1  # tenacity retries, then OpenIE swallows the error
    out = kept.ingest(PREFIX[:1])
    assert out["ingested"] == 1 and kept.stats["failed"] == 1  # the passage stays, as in HippoRAG
    assert event_text(PREFIX[0]) in kept._passages()
    assert (tmp_path / "kept" / "failures.jsonl").is_file()
    strict = _system(tmp_path / "strict", chat, emb, on_error="raise")
    strict._hip.llm_model.max_retries = 1
    with pytest.raises(RuntimeError, match="OpenIE swallowed"):
        strict.ingest(PREFIX[:1])
    assert strict._passages() == set() and len(strict.ledger) == 0 and len(strict.objects) == 0
    chat.fail = False
    assert strict.ingest(PREFIX[:1])["ingested"] == 1  # resumed: redone from scratch
    assert _phrases(strict) >= {"boston"} and strict.stats["failed"] == 0


def test_last_passage_can_be_deleted_and_the_store_reused(hippo, tmp_path):
    chat, emb = FakeChat(), FakeEmb()
    s = _system(tmp_path / "store", chat, emb)
    s.ingest(PREFIX[:1])
    s.ingest([_ev(9, "contacts:p01", 2, op="delete")], "maintain")
    assert s._passages() == set() and len(s.objects) == 0
    s.ingest([_ev(10, "contacts:p01", 3, P01_R2)], "maintain")  # the stale OpenIE file is gone
    assert s._passages() == {event_text(_ev(10, "contacts:p01", 3, P01_R2))}


def test_batches_group_distinct_objects_per_call(hippo, tmp_path):
    chat, emb = FakeChat(), FakeEmb()
    s = _system(tmp_path / "store", chat, emb, batch_size=3)
    calls: list[int] = []
    index = s._hip.index

    def counting_index(docs: list[str]) -> None:
        calls.append(len(docs))
        index(docs)

    s._hip.index = counting_index
    s.ingest(PREFIX + BRANCH)  # p01 twice in six events: batches [1,2,3] and [p01 r2, e01, t01]
    assert calls == [3, 2]
    assert event_key(BRANCH[0]) in s.ledger and set(s.objects.items) == {
        "contacts:p01",
        "notes:n01",
        "tasks:t01",
    }

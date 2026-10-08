"""Offline tests for the MemOS adapter (src/memshift/adapters/native_memos.py). The pure helpers,
the configuration and the failure watchers run everywhere; the ingest / snapshot / restore /
search round trip needs `MemoryOS==2.0.33` and `qdrant-client` (docker/native/memos.txt) and is
skipped without them. It talks to a loopback stand-in for the two OpenAI-compatible endpoints,
never to a model server."""

import hashlib
import json
import os
import re
import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memshift.adapters.native import (
    STORE_MARKER,
    MeteredOpenAI,
    Usage,
    event_key,
    event_metadata,
)
from memshift.adapters.native_memos import (
    COLLECTION,
    MemOSSystem,
    _meter,
    _watch_reader,
    chat_scene,
    chat_time,
    export_row,
    keys_from_payloads,
    memos_hit,
    payload_key,
    reader_info,
)
from memshift.gen.model import Event

T0 = datetime(2026, 3, 1, 9, 30)


def _ev(i: int, oid: str, rev: int, op: str = "insert", text: str | None = None) -> Event:
    return Event(
        event_id=f"ev{i:06d}",
        ts=T0 + timedelta(minutes=i),
        source=oid.split(":")[0],
        object_id=oid,
        revision=rev,
        op=op,
        payload={"text": text if text is not None else f"Body of {oid} revision {rev}"},
    )


def _payload(oid: str, rev: int, text: str, mem_id: str = "m1") -> dict[str, Any]:
    """A stored memory as `TextualMemoryItem.model_dump()` gives it after the MemReader."""
    return {
        "id": mem_id,
        "memory": text,
        "metadata": {
            "user_id": "u1",
            "session_id": oid,
            "status": "activated",
            "version": 1,
            "history": [],
            "sources": [{"type": "chat", "role": "user", "content": "raw", "chat_time": "t"}],
            "info": {
                "object_id": oid,
                "revision": rev,
                "source": oid.split(":")[0],
                "ts": "2026-03-01T09:35",
                "op": "update",
            },
        },
    }


# ---------------------------------------------------------------------- conversion
def test_chat_scene_and_reader_info():
    e = _ev(5, "email:e1", 2, "update")
    assert chat_time(e) == "March 01, 2026 at 09:35 AM"
    scene = chat_scene(e)
    assert len(scene) == 1 and len(scene[0]) == 1  # one event = one single-message scene
    msg = scene[0][0]
    assert msg["role"] == "user" and msg["chat_time"] == "March 01, 2026 at 09:35 AM"
    assert msg["content"].startswith("[email record email:e1 | revision 2 | 2026-03-01 09:35]\n")
    info = reader_info(e, "w1")
    assert info["user_id"] == "w1" and info["session_id"] == "email:e1"
    assert {k: info[k] for k in event_metadata(e)} == event_metadata(e)
    tomb = chat_scene(_ev(6, "contacts:p01", 3, "delete"))[0][0]["content"]
    assert tomb == "Record contacts:p01 was deleted on 2026-03-01 09:36."


def test_payload_keys_and_grouping():
    a, b = _payload("email:e1", 1, "x", "m1"), _payload("email:e1", 2, "y", "m2")
    c = _payload("email:e1", 2, "z", "m3")
    stranger = {"id": "m9", "memory": "no info", "metadata": {"info": None}}
    assert payload_key(a) == "email:e1#1" and payload_key(stranger) is None
    assert payload_key({"id": "m0", "memory": "bare"}) is None
    assert keys_from_payloads([a, b, c, stranger]) == {
        "email:e1#1": ["m1"],
        "email:e1#2": ["m2", "m3"],
    }


def test_memos_hit_block():
    h = memos_hit(_payload("email:e1", 2, "The user works at Acme."), 0.5)
    assert (
        h.text
        == "[memory | source email email:e1 r2 | event 2026-03-01 09:35] The user works at Acme."
    )
    assert h.hit_id == "m1" and h.refs == [("email:e1", 2)] and h.score == 0.5
    assert h.ts == datetime(2026, 3, 1, 9, 35)
    bare = memos_hit({"id": "m2", "memory": "x", "metadata": {}})
    assert bare.text == "[memory] x" and bare.refs == []


def test_export_row():
    p = _payload("email:e1", 1, "x")
    row = export_row("m1", p, 1024)
    assert row == {"id": "m1", "key": "email:e1#1", "payload": p, "vector_dim": 1024}


# ---------------------------------------------------------------------- configuration
class _Unopened(MemOSSystem):
    """The adapter without the library: `_open` / `_close` do nothing."""

    def _open(self) -> None:
        pass

    def _close(self) -> None:
        pass

    def _reconcile(self) -> None:
        pass


def _unopened(store: Path, **kw: Any) -> _Unopened:
    return _Unopened(
        None, store, "http://llm:1/v1", "m/llm", "http://emb:2/v1", "m/emb", "w1", **kw
    )


def test_config_is_embedded_and_local(tmp_path):
    s = _unopened(tmp_path / "store")
    cfg = s.config()
    reader, tm = cfg["reader"], cfg["text_memory"]
    assert reader["backend"] == "simple_struct"
    llm = reader["config"]["llm"]["config"]
    assert (llm["api_base"], llm["model_name_or_path"], llm["temperature"]) == (
        "http://llm:1/v1",
        "m/llm",
        0.0,
    )
    assert "enable_thinking" not in llm  # switched off by the metered client
    for e in (reader["config"]["embedder"], tm["embedder"]):
        assert e["backend"] == "universal_api"
        assert e["config"]["base_url"] == "http://emb:2/v1"
        assert e["config"]["model_name_or_path"] == "m/emb"
    assert reader["config"]["chunker"]["backend"] == "markdown"
    vdb = tm["vector_db"]
    assert vdb["backend"] == "qdrant"
    assert vdb["config"]["path"] == str(tmp_path / "store" / "qdrant")
    assert not {"host", "port", "url"} & set(vdb["config"])  # embedded: no server
    assert vdb["config"]["vector_dimension"] == 1024 and vdb["config"]["collection_name"]
    assert tm["extractor_llm"] == reader["config"]["llm"]
    d = s.describe()
    assert d["system"] == "memos" and d["text_memory"] == "general_text"
    assert d["vector_store"] == "qdrant-path" and "memos_version" in d
    assert s.identifiable and json.loads((tmp_path / "store" / STORE_MARKER).read_text())


def test_max_output_tokens_flow_to_llm_config(tmp_path):
    s = _unopened(tmp_path / "store", max_output_tokens=2048, embed_dims=8)
    cfg = s.config()
    assert cfg["reader"]["config"]["llm"]["config"]["max_tokens"] == 2048
    assert cfg["text_memory"]["vector_db"]["config"]["vector_dimension"] == 8


# ---------------------------------------------------------------------- library plumbing
def test_meter_wraps_each_client_once():
    class _Chat:
        def create(self, **kw: Any) -> Any:
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=2))

    def client() -> Any:
        return SimpleNamespace(
            chat=SimpleNamespace(completions=_Chat()), embeddings=SimpleNamespace(create=None)
        )

    inner = client()
    shared, other = SimpleNamespace(client=inner), SimpleNamespace(client=client())
    usage = Usage()
    _meter([shared, shared, other], usage, extra_body={"a": 1})
    assert isinstance(shared.client, MeteredOpenAI) and isinstance(other.client, MeteredOpenAI)
    assert shared.client._inner is inner  # not wrapped twice
    shared.client.chat.completions.create(model="m", messages=[])
    _meter([shared], usage)  # already metered: left alone
    assert shared.client._inner is inner
    assert usage.snapshot()["llm_calls"] == 1 and usage.snapshot()["prompt_tokens"] == 7


def test_watch_reader_notes_errors_and_fallbacks():
    """MemOS swallows LLM and embedding exceptions and unparseable replies inside the reader; the
    watchers see them without changing what MemOS does."""

    class _Reader:
        def __init__(self) -> None:
            self.llm = SimpleNamespace(generate=self._generate)
            self.embedder = SimpleNamespace(embed=self._embed)
            self.reply: Any = '{"memory list": []}'
            self.embed_down = False

        def _embed(self, texts: Any) -> list[list[float]]:
            if self.embed_down:
                raise TimeoutError("embedder down")
            return [[0.0] for _ in texts]

        def _generate(self, messages: Any, **kw: Any) -> str:
            if self.reply is None:
                raise ConnectionError("endpoint down")
            return str(self.reply)

        def _safe_parse(self, text: Any) -> Any:
            try:
                return json.loads(text) if text else None
            except ValueError:
                return None

    errors: list[Exception] = []
    fallbacks: list[str] = []
    r = _Reader()
    _watch_reader(r, errors, fallbacks)
    assert r.llm.generate([]) == '{"memory list": []}'
    assert r._safe_parse('{"memory list": [1]}') == {"memory list": [1]}
    assert not errors and not fallbacks
    assert r._safe_parse("not json") is None and fallbacks == ["unparseable extraction reply"]
    assert r._safe_parse(None) is None and len(fallbacks) == 1  # no reply: an error, not a fallback
    r.reply = None
    with pytest.raises(ConnectionError):
        r.llm.generate([])  # still propagates to MemOS, which swallows it
    assert len(errors) == 1 and isinstance(errors[0], ConnectionError)
    assert r.embedder.embed(["x"]) == [[0.0]] and len(errors) == 1
    r.embed_down = True  # the reader drops the memory it could not embed; the watcher notes it
    with pytest.raises(TimeoutError):
        r.embedder.embed(["x"])
    assert isinstance(errors[1], TimeoutError)


# ---------------------------------------------------------------------- library round trip
_RECORD = re.compile(r"\[(\w+) record (\S+) \| revision (\d+) \| [^\]]+\]\n([^\n]*)")
_TOMBSTONE = re.compile(r"Record (\S+) was deleted on ([^\n]*)\.")


class _FakeEndpoint(BaseHTTPRequestHandler):
    """OpenAI-compatible stand-in: extraction replies list one memory per record; embeddings are
    a hash of the text. `FAIL-ME` gets HTTP 400, `GARBAGE` an unparseable reply."""

    dims = 8

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, body: dict[str, Any], status: int = 200) -> None:
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self) -> None:
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path.endswith("/embeddings"):
            texts = req["input"] if isinstance(req["input"], list) else [req["input"]]
            data = []
            for i, t in enumerate(texts):
                h = hashlib.sha256(t.encode()).digest()
                data.append(
                    {
                        "object": "embedding",
                        "index": i,
                        "embedding": [b / 255 for b in h[: self.dims]],
                    }
                )
            usage = {"prompt_tokens": 5 * len(texts), "total_tokens": 5 * len(texts)}
            self._send({"object": "list", "data": data, "model": req["model"], "usage": usage})
            return
        prompt = req["messages"][-1]["content"]
        if "FAIL-ME" in prompt:
            self._send({"error": {"message": "bad request", "type": "invalid_request_error"}}, 400)
            return
        scene = prompt.split("Conversation:")[-1]
        rec, tomb = _RECORD.search(scene), _TOMBSTONE.search(scene)
        if "GARBAGE" in prompt or (rec is None and tomb is None):
            content = "I cannot help with that."
        else:
            if rec is not None:
                _, oid, rev, body = rec.groups()
                value = f"{oid} revision {rev}: {body}"
            else:
                assert tomb is not None
                oid, value = tomb.group(1), f"Record {tomb.group(1)} was deleted."
            mem = {"key": oid, "memory_type": "LongTermMemory", "value": value, "tags": ["t"]}
            content = json.dumps({"memory list": [mem], "summary": f"About {oid}."})
        msg = {"role": "assistant", "content": content}
        self._send(
            {
                "id": "c",
                "object": "chat.completion",
                "created": 0,
                "model": req["model"],
                "choices": [{"index": 0, "message": msg, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            }
        )


@pytest.fixture(scope="module")
def endpoint(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("MEMOS_BASE_PATH", str(tmp_path_factory.mktemp("memos_base")))
    server = None
    try:
        pytest.importorskip("memos")
        pytest.importorskip("qdrant_client")
        server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeEndpoint)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        yield f"http://127.0.0.1:{server.server_address[1]}/v1"
    finally:
        if server is not None:
            server.shutdown()
        mp.undo()


def _memos(store: Path, url: str, **kw: Any) -> MemOSSystem:
    return MemOSSystem(None, store, url, "fake/llm", url, "fake/emb", "w1", embed_dims=8, **kw)


def test_memos_round_trip(endpoint, tmp_path):
    system = _memos(tmp_path / "store", endpoint, on_error="skip")
    evs = [
        _ev(1, "email:e1", 1),
        _ev(2, "email:e1", 2, "update", text="Revised body"),
        _ev(3, "contacts:p01", 1),
        _ev(4, "contacts:p01", 2, "delete"),
        _ev(5, "notes:n1", 1, text="FAIL-ME"),
        _ev(6, "notes:n2", 1, text="GARBAGE"),
    ]
    out = system.ingest(evs[:4])
    assert out["ingested"] == 4 and out["llm_calls"] == 4  # one extraction call per event
    assert out["embed_calls"] == 8  # one per extracted memory (reader) + one batch in `add`
    assert out["prompt_tokens"] == 400 and out["completion_tokens"] == 80
    rows = {r["key"]: r for r in system.export_raw()}
    assert set(rows) == {"email:e1#1", "email:e1#2", "contacts:p01#1", "contacts:p01#2"}
    old, new = rows["email:e1#1"], rows["email:e1#2"]
    assert old["vector_dim"] == 8 and old["payload"]["id"] != new["payload"]["id"]
    assert "Body of email:e1 revision 1" in old["payload"]["memory"]  # append-only: kept
    assert "Revised body" in new["payload"]["memory"]
    md = new["payload"]["metadata"]
    assert md["status"] == "activated" and md["version"] == 1 and md["history"] == []
    assert md["info"]["object_id"] == "email:e1" and md["info"]["revision"] == 2
    assert md["embedding"] is None and md["session_id"] == "email:e1" and md["user_id"] == "w1"
    src = md["sources"][0]
    assert src["chat_time"] == "March 01, 2026 at 09:32 AM"
    assert src["content"].startswith("[email record email:e1 | revision 2 |")
    assert "was deleted" in rows["contacts:p01#2"]["payload"]["memory"]  # tombstone as text
    # search: MemOS's own retrieval, hits carry the source refs
    probe = SimpleNamespace(question_text="Where does the user work?")
    served = system.serve(probe)
    assert len(served.items) == 4 and served.blocks and not served.blocked
    assert {tuple(r) for r in served.refs} == {
        ("email:e1", 1),
        ("email:e1", 2),
        ("contacts:p01", 1),
        ("contacts:p01", 2),
    }
    assert all(b.startswith("[memory | source ") for b in served.blocks)
    # failure and fallback bookkeeping
    system.snapshot(tmp_path / "snap")
    out = system.ingest(evs[4:], "maintain")
    assert system.stats["failed"] == 1 and out["ingested"] == 1  # FAIL-ME skipped, GARBAGE kept
    assert system.stats["extract_fallback"] == 1
    assert event_key(evs[4]) not in system.ledger and event_key(evs[5]) in system.ledger
    fail = json.loads((tmp_path / "store" / "failures.jsonl").read_text().splitlines()[0])
    assert fail["key"] == "notes:n1#1"
    garbage = [r for r in system.export_raw() if r["key"] == "notes:n2#1"]
    assert len(garbage) == 1 and "GARBAGE" in garbage[0]["payload"]["memory"]  # raw text kept
    assert system.ledger.entries["notes:n2#1"]["fallback"] == 1
    assert len(system.export_raw()) == 5
    # a branch restores the prefix snapshot and starts from its accounting
    system.restore(tmp_path / "snap")
    assert len(system.export_raw()) == 4 and len(system.ledger) == 4
    assert system.usage.snapshot()["llm_calls"] == 4
    assert system.cost["ingest_tokens"] > 0 and system.cost["maintain_tokens"] == 0
    assert system.ingest(evs[:4])["skipped"] == 4
    assert system.ingest(evs[5:], "maintain")["ingested"] == 1
    system.close()
    # a lost ledger is rebuilt from the memories themselves
    (tmp_path / "store" / "ledger.jsonl").unlink()
    again = _memos(tmp_path / "store", endpoint)
    assert len(again.ledger) == 5 and again.ingest(evs[:4] + evs[5:])["ingested"] == 0
    again.close()
    assert os.path.isdir(tmp_path / "store" / "qdrant") and COLLECTION == "memshift"


def test_memos_concurrent_extraction_matches_sequential(endpoint, tmp_path):
    """`workers` only overlaps extraction: same memories, adds in event order, failures kept
    per event."""
    evs = [_ev(i, f"notes:n{i}", 1, text="FAIL-ME" if i == 7 else f"text {i}") for i in range(12)]
    evs.append(_ev(13, "notes:n13", 1, text="GARBAGE"))
    runs = {}
    for workers in (1, 4):
        system = _memos(tmp_path / f"w{workers}", endpoint, on_error="skip", workers=workers)
        assert len(system._slots) == workers and system.describe()["workers"] == workers
        out = system.ingest(evs)
        rows = system.export_raw()
        runs[workers] = (
            list(system.ledger.entries),
            sorted((r["key"], r["payload"]["memory"]) for r in rows),
            system.stats["failed"],
            system.stats["extract_fallback"],
            out["llm_calls"],
        )
        system.close()
    assert runs[1] == runs[4]
    order, memories, failed, fallback, llm_calls = runs[4]
    assert order == [event_key(e) for e in evs if e.object_id != "notes:n7"]  # time order
    assert len(memories) == 12 and failed == 1 and fallback == 1 and llm_calls == 12

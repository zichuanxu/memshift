"""Offline tests for the A-Mem adapter (E5 native systems). No network, no GPU: the pure helpers,
the driver-side bookkeeping (on a stand-in for the library object) and, when `agentic_memory` and
`chromadb` are importable (`pip install -r docker/native/amem.txt`), the adapter over the real
library with a scripted LLM client and a hash embedder in place of our endpoint and the encoder."""

import hashlib
import inspect
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memshift.adapters import native_amem
from memshift.adapters.native import event_key, event_text, parse_key
from memshift.adapters.native_amem import (
    AMEM_COMMIT,
    AMEM_ENCODER,
    AMEM_ENCODER_REVISION,
    AMEM_RETRIEVE_K,
    ENCODER_FILES,
    NOTE_FIELDS,
    AMemSystem,
    _amem_env,
    amem_hit,
    encoder_cached,
    encoder_kwargs,
    encoder_snapshot_dir,
    expand_hits,
    hub_cache_dir,
    main,
    note_kwargs,
    note_timestamp,
    parse_note_timestamp,
    read_notes,
    seed_cache,
    stored_keys,
    write_notes,
)
from memshift.gen.model import Event

T0 = datetime(2026, 3, 1, 9, 30)
REPO = Path(__file__).resolve().parents[1]


def _ev(i: int, oid: str, rev: int, op: str = "insert", minutes: int = 0) -> Event:
    return Event(
        event_id=f"ev{i:06d}",
        ts=T0 + timedelta(minutes=minutes or i),
        source=oid.split(":")[0],
        object_id=oid,
        revision=rev,
        op=op,
        payload={"text": f"Body of {oid} revision {rev}", "cluster": oid},
    )


def _note(oid: str, rev: int, **kw: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "id": f"{oid}#{rev}",
        "content": f"Body of {oid} revision {rev}",
        "keywords": [],
        "links": [],
        "retrieval_count": 0,
        "timestamp": "202603010935",
        "last_accessed": "202603010935",
        "context": "General",
        "evolution_history": [],
        "category": oid.split(":")[0],
        "tags": [],
    }
    return {**base, **kw}


# ---------------------------------------------------------------------- event -> note
def test_note_kwargs_carry_event_metadata_on_note_fields():
    up = _ev(1, "email:e1", 2, "update", minutes=5)
    kw = note_kwargs(up)
    assert kw == {
        "content": event_text(up),
        "time": "202603010935",
        "id": "email:e1#2",
        "category": "email",
        "last_accessed": "202603010935",
    }
    assert kw["content"].startswith("[email record email:e1 | revision 2 | 2026-03-01 09:35]\n")
    assert set(kw) - {"content", "time"} <= set(NOTE_FIELDS)


def test_tombstone_is_an_ordinary_note_keyed_by_the_delete_revision():
    gone = _ev(2, "contacts:p01", 3, "delete", minutes=7)
    kw = note_kwargs(gone)
    assert kw["content"] == "Record contacts:p01 was deleted on 2026-03-01 09:37."
    assert kw["id"] == event_key(gone) == "contacts:p01#3" and kw["category"] == "contacts"
    hit = amem_hit(_note("contacts:p01", 3, content=kw["content"], timestamp="202603010937"))
    assert hit.refs == [("contacts:p01", 3)]
    assert hit.text == (
        "[note | source contacts contacts:p01 r3 | event 2026-03-01 09:37]"
        " Record contacts:p01 was deleted on 2026-03-01 09:37."
    )


def test_note_timestamp_round_trip():
    assert note_timestamp(datetime(2026, 3, 1, 9, 5)) == "202603010905"
    assert parse_note_timestamp("202603010905") == datetime(2026, 3, 1, 9, 5)
    assert parse_note_timestamp(202603010905) == datetime(2026, 3, 1, 9, 5)  # index turns it to int
    for bad in ("", "2026-03-01", "20261301xxxx", "202613010905", None):
        assert parse_note_timestamp(bad) is None


# ---------------------------------------------------------------------- hits
def test_amem_hit_block_puts_derived_fields_ahead_of_the_content():
    note = _note(
        "email:e1",
        2,
        keywords=["acme", "budget"],
        tags=["work"],
        context="A budget email.",
        timestamp=202603010935,  # what the index hands back
    )
    h = amem_hit(note, 0.5)
    assert h.hit_id == "email:e1#2" and h.score == 0.5
    assert h.text == (
        "[note | source email email:e1 r2 | event 2026-03-01 09:35 | context: A budget email."
        " | keywords: acme, budget | tags: work] Body of email:e1 revision 2"
    )
    assert h.refs == [("email:e1", 2)] and h.ts == datetime(2026, 3, 1, 9, 35)


def test_amem_hit_defaults_are_not_shown():
    h = amem_hit(_note("notes:n1", 1, category="Uncategorized", keywords="solo"))
    assert h.text == (
        "[note | source notes:n1 r1 | event 2026-03-01 09:35 | keywords: solo]"
        " Body of notes:n1 revision 1"
    )
    odd = amem_hit({"id": "not-a-key", "content": "x", "timestamp": "soon"})
    assert odd.text == "[note] x" and odd.refs == [] and odd.ts == datetime(1970, 1, 1)


def test_expand_hits_follow_links_dedupe_and_skip_missing():
    notes = {
        "a:1#1": _note("a:1", 1, links=["a:2#1", "gone#1", "a:3#1"]),
        "a:2#1": _note("a:2", 1),
        "a:3#1": _note("a:3", 1, links=["a:1#1"]),
        "a:4#1": _note("a:4", 1),
    }
    ranked = [("a:1#1", 0.0), ("missing#1", 0.1), ("a:3#1", 0.5), ("a:4#1", 1.0), ("a:1#1", 2.0)]
    hits = expand_hits(ranked, notes, k=10)
    # a note is followed by the notes it links to; a linked note is not repeated when ranked
    assert [h.hit_id for h in hits] == ["a:1#1", "a:2#1", "a:3#1", "a:4#1"]
    assert [h.score for h in hits] == pytest.approx([1.0, 0.0, 2 / 3, 0.5])
    capped = expand_hits([("a:1#1", 0.0)], notes, k=1)
    assert [h.hit_id for h in capped] == ["a:1#1", "a:2#1"]  # at most k linked notes per note
    assert expand_hits([], notes, k=5) == []


# ---------------------------------------------------------------------- persistence
def test_notes_file_round_trip_and_stored_keys(tmp_path):
    path = tmp_path / "notes.json"
    assert read_notes(path) == ([], 0)
    notes = [_note("a:1", 1), _note("a:1", 2, links=["a:1#1"]), {"id": "imported", "content": "x"}]
    write_notes(path, notes, evo_cnt=7)
    (tmp_path / "notes.json.tmp").write_text("half a write")  # a crash between write and rename
    assert read_notes(path) == (notes, 7)
    write_notes(path, notes[:1], evo_cnt=8)
    assert read_notes(path) == (notes[:1], 8)
    assert stored_keys(notes) == ["a:1#1", "a:1#2"]  # ids that are keys; others are not units


def test_pinned_commit_matches_the_requirement_file():
    lines = (REPO / "docker" / "native" / "amem.txt").read_text().splitlines()
    assert any(AMEM_COMMIT in line and "A-mem-sys" in line for line in lines)
    assert len(AMEM_COMMIT) == 40


# ---------------------------------------------------------------------- pinned encoder
ENV_VARS = (
    "HF_HOME",
    "HF_HUB_CACHE",
    "HUGGINGFACE_HUB_CACHE",
    "HF_HUB_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "ANONYMIZED_TELEMETRY",
    "LITELLM_LOCAL_MODEL_COST_MAP",
    "TOKENIZERS_PARALLELISM",
)


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """The variables `_amem_env` sets, absent at the start and restored afterwards; the repository
    root is a temporary directory, so a cache seeded in the real one cannot change the outcome."""
    for var in ENV_VARS:
        monkeypatch.setenv(var, "")
        monkeypatch.delenv(var)
    monkeypatch.setattr(native_amem, "REPO", tmp_path)
    return tmp_path


def _snapshot(
    hub: Path, files: tuple[str, ...] = ENCODER_FILES, revision: str = AMEM_ENCODER_REVISION
):
    snap = hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / revision
    for name in files:
        (snap / name).parent.mkdir(parents=True, exist_ok=True)
        (snap / name).write_text("x")
    return snap


def test_encoder_is_pinned_and_requested_by_full_repo_id():
    assert AMEM_ENCODER_REVISION == "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    assert AMEM_ENCODER == "sentence-transformers/all-MiniLM-L6-v2"
    assert encoder_kwargs() == {"model_name": AMEM_ENCODER, "revision": AMEM_ENCODER_REVISION}
    assert {"modules.json", "model.safetensors", "1_Pooling/config.json"} <= set(ENCODER_FILES)
    assert not any(
        f.startswith(("onnx", "openvino")) or f.endswith((".bin", ".h5", ".ot"))
        for f in ENCODER_FILES
    )


def test_hub_cache_follows_the_huggingface_variables(clean_env, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(clean_env / "home"))
    assert hub_cache_dir() == clean_env / "home" / "hub"
    monkeypatch.setenv("HF_HUB_CACHE", str(clean_env / "hubcache"))
    assert hub_cache_dir() == clean_env / "hubcache"
    snap = encoder_snapshot_dir()
    assert snap == (
        clean_env
        / "hubcache"
        / "models--sentence-transformers--all-MiniLM-L6-v2"
        / "snapshots"
        / AMEM_ENCODER_REVISION
    )


def test_env_points_the_hf_cache_at_the_repository_and_keeps_presets(clean_env, monkeypatch):
    _amem_env()
    assert os.environ["HF_HOME"] == str(clean_env / "cache" / "hf")
    assert "HF_HUB_OFFLINE" not in os.environ  # nothing cached yet: downloads stay possible
    monkeypatch.setenv("HF_HOME", "/somewhere/else")
    _amem_env()
    assert os.environ["HF_HOME"] == "/somewhere/else"


def test_offline_only_when_the_pinned_snapshot_is_complete(clean_env, monkeypatch):
    hub = clean_env / "cache" / "hf" / "hub"
    _snapshot(hub, revision="0" * 40)  # some other revision
    _snapshot(hub, files=ENCODER_FILES[:-1])  # the pinned one, one file short
    _amem_env()
    assert "HF_HUB_OFFLINE" not in os.environ and not encoder_cached()
    _snapshot(hub)
    assert encoder_cached()
    _amem_env()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    monkeypatch.setenv("HF_HUB_OFFLINE", "0")  # an explicit setting wins
    _amem_env()
    assert os.environ["HF_HUB_OFFLINE"] == "0"


def test_seed_cache_downloads_exactly_the_pinned_files_once(clean_env, capsys) -> None:
    calls: list[dict] = []

    def fake_download(**kw):
        calls.append(kw)
        _snapshot(Path(kw["cache_dir"]))
        return "unused"

    snap = seed_cache(fake_download)
    hub = clean_env / "cache" / "hf" / "hub"
    assert snap == hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots" / (
        AMEM_ENCODER_REVISION
    )
    assert calls == [
        {
            "repo_id": "sentence-transformers/all-MiniLM-L6-v2",
            "revision": AMEM_ENCODER_REVISION,
            "allow_patterns": list(ENCODER_FILES),
            "cache_dir": str(hub),
        }
    ]
    assert seed_cache(fake_download) == snap and len(calls) == 1  # already there: no download
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert main(["--seed-cache"], fake_download) == 0 and len(calls) == 1
    assert capsys.readouterr().out.strip() == str(snap)
    assert main([], fake_download) == 2 and "--seed-cache" in capsys.readouterr().out


def test_seed_cache_rejects_an_incomplete_download(clean_env):
    with pytest.raises(RuntimeError, match="incomplete"):
        seed_cache(lambda **kw: None)


# ---------------------------------------------------------------------- driver side
class _FakeMem:
    """Stands in for the library object: add_note either stores, degrades or fails per key."""

    def __init__(self) -> None:
        self.memories: dict[str, SimpleNamespace] = {}
        self.llm_errors: list[str] = []
        self.saves = 0
        self.fail: set[str] = set()
        self.degrade: set[str] = set()

    def add_note(self, content: str, time: str, id: str, category: str, last_accessed: str):
        self.memories[id] = SimpleNamespace(id=id, content=content, links=[], category=category)
        if id in self.fail:
            raise RuntimeError("index unavailable")
        if id in self.degrade:
            self.llm_errors.append("JSONDecodeError: cut short")
        return id

    def save(self) -> None:
        self.saves += 1

    def shutdown(self) -> None:
        pass


def _fake_system(store: Path, mem: _FakeMem, **kw: Any) -> AMemSystem:
    class Sys(AMemSystem):
        def _open(self) -> None:
            self._mem = mem

    return Sys(None, store, "http://x/v1", "llm", "http://x/v1", "emb", "u1", **kw)


def test_describe_records_the_local_encoder_and_the_pin(tmp_path):
    sys_ = _fake_system(tmp_path / "s", _FakeMem())
    d = sys_.describe()
    assert d["system"] == "amem" and d["embed_model"] == AMEM_ENCODER
    assert d["embed_revision"] == AMEM_ENCODER_REVISION
    assert d["requested_embed_model"] == "emb" and d["embed_endpoint_used"] is False
    assert d["amem_commit"] == AMEM_COMMIT and d["llm_temperature"] == 0.0
    assert sys_.top_k == AMEM_RETRIEVE_K and d["top_k"] == AMEM_RETRIEVE_K
    assert sys_.retains_raw_text and sys_.retains_source_metadata and sys_.identifiable
    assert json.loads((tmp_path / "s" / "memshift_native.json").read_text())["system"] == "amem"


def test_reconcile_maps_stored_notes_to_the_ledger(tmp_path):
    mem = _FakeMem()
    for oid, rev in (("email:e1", 1), ("email:e1", 2)):
        mem.memories[f"{oid}#{rev}"] = SimpleNamespace(**_note(oid, rev))
    mem.memories["imported"] = SimpleNamespace(**_note("x:1", 1, id="imported"))
    sys_ = _fake_system(tmp_path / "s", mem)
    assert sorted(sys_.ledger.entries) == ["email:e1#1", "email:e1#2"]
    assert sys_.ledger.entries["email:e1#2"]["reconciled"] is True
    assert sys_.ingest([_ev(1, "email:e1", 1), _ev(2, "email:e1", 2)])["skipped"] == 2


def test_ingest_records_units_and_delivers_note_arguments(tmp_path):
    mem = _FakeMem()
    sys_ = _fake_system(tmp_path / "s", mem)
    evs = [_ev(1, "email:e1", 1), _ev(2, "email:e1", 2, "update"), _ev(3, "email:e1", 3, "delete")]
    out = sys_.ingest(evs)
    assert out["ingested"] == 3 and mem.saves == 3
    assert list(mem.memories) == ["email:e1#1", "email:e1#2", "email:e1#3"]
    assert mem.memories["email:e1#3"].content == "Record email:e1 was deleted on 2026-03-01 09:33."
    assert sys_.ledger.entries["email:e1#2"]["op"] == "update"


def test_failed_unit_raises_or_is_skipped_and_rolled_back(tmp_path):
    mem = _FakeMem()
    mem.fail.add("a:2#1")
    evs = [_ev(1, "a:1", 1), _ev(2, "a:2", 1), _ev(3, "a:3", 1)]
    strict = _fake_system(tmp_path / "strict", mem)
    with pytest.raises(RuntimeError, match="index unavailable"):
        strict.ingest(evs)
    assert "a:2#1" not in strict.ledger and "a:2#1" not in mem.memories

    mem2 = _FakeMem()
    mem2.fail.add("a:2#1")
    lenient = _fake_system(tmp_path / "lenient", mem2, on_error="skip")
    out = lenient.ingest(evs)
    assert out["ingested"] == 2 and lenient.stats["failed"] == 1
    assert "a:2#1" not in mem2.memories  # the half-stored note is removed
    text = (tmp_path / "lenient" / "failures.jsonl").read_text()
    rows = [json.loads(x) for x in text.splitlines()]
    assert rows[0]["key"] == "a:2#1" and "index unavailable" in rows[0]["error"]


def test_llm_failure_inside_the_library_is_reported_but_the_note_is_kept(tmp_path):
    mem = _FakeMem()
    mem.degrade.add("a:1#1")
    strict = _fake_system(tmp_path / "strict", mem)
    with pytest.raises(RuntimeError, match="LLM step failed inside A-Mem"):
        strict.ingest([_ev(1, "a:1", 1)])
    assert "a:1#1" in mem.memories and "a:1#1" not in strict.ledger  # a reopen reconciles it

    mem2 = _FakeMem()
    mem2.degrade.add("a:1#1")
    lenient = _fake_system(tmp_path / "lenient", mem2, on_error="skip")
    lenient.ingest([_ev(1, "a:1", 1), _ev(2, "a:2", 1)])
    assert lenient.stats["failed"] == 1 and "a:1#1" in mem2.memories
    assert lenient.ledger.entries["a:1#1"]["llm_errors"] == ["JSONDecodeError: cut short"]
    assert "llm_errors" not in lenient.ledger.entries["a:2#1"]


# ---------------------------------------------------------------------- over the real library
class _ScriptedLLM:
    """An OpenAI-style client that answers A-Mem's two prompts: the analysis call (keywords,
    context, tags) and the evolution call (strengthen the first neighbour, retag all of them)."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_analysis = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))
        self.embeddings = SimpleNamespace()  # never called: A-Mem embeds locally

    def create(self, **kw: Any) -> Any:
        self.calls.append(kw)
        props = kw["response_format"]["json_schema"]["schema"]["properties"]
        prompt = kw["messages"][-1]["content"]
        if "should_evolve" in props:
            ids = re.findall(r"memory_id:(\S+)\t", prompt)
            n = int(re.findall(r"The number of neighbors is (\d+)", prompt)[0])
            out = {
                "should_evolve": True,
                "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": ids[:1],
                "tags_to_update": ["linked"],
                "new_context_neighborhood": ["re-read"] * n,
                "new_tags_neighborhood": [["evolved"]] * n,
            }
        elif self.fail_analysis:
            raise ConnectionError("endpoint down")
        else:
            words = re.findall(r"[a-z]+:[a-z0-9]+", prompt.split("Content for analysis:")[1])
            out = {"keywords": words[:3] or ["none"], "context": "About a record.", "tags": ["t"]}
        msg = SimpleNamespace(content=json.dumps(out))
        usage = SimpleNamespace(prompt_tokens=len(prompt) // 4, completion_tokens=20)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)], usage=usage)


@pytest.fixture(scope="module")
def hash_embedder():
    _amem_env()  # before the package import: litellm would fetch a price map from the network
    np = pytest.importorskip("numpy")
    chromadb = pytest.importorskip("chromadb")
    pytest.importorskip("agentic_memory.memory_system")

    class HashEmbedder(chromadb.EmbeddingFunction):
        def __init__(self) -> None:
            pass

        def __call__(self, input):
            vecs = []
            for text in input:
                raw = hashlib.sha256(text.encode()).digest() * 2
                v = np.frombuffer(raw, dtype=np.uint8).astype(np.float32) + 1.0
                vecs.append(v / np.linalg.norm(v))
            return vecs

        @staticmethod
        def name() -> str:
            return "memshift-test-hash"

        def get_config(self) -> dict[str, Any]:
            return {}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> "HashEmbedder":
            return HashEmbedder()

    return HashEmbedder()


def _amem(store: Path, llm: _ScriptedLLM, embedder: Any, **kw: Any) -> AMemSystem:
    return AMemSystem(
        None,
        store,
        "http://x/v1",
        "Qwen/Qwen3.5-9B",
        "http://x/v1",
        "emb",
        "u1",
        embedding_function=embedder,
        llm_client=llm,
        **kw,
    )


EVENTS = [
    _ev(1, "contacts:p01", 1, minutes=1),
    _ev(2, "contacts:p01", 2, "update", minutes=2),
    _ev(3, "notes:n1", 1, minutes=3),
    _ev(4, "contacts:p01", 3, "delete", minutes=4),
]


def test_library_note_constructor_matches_the_note_fields(hash_embedder):
    from agentic_memory.memory_system import MemoryNote

    params = [p for p in inspect.signature(MemoryNote.__init__).parameters if p != "self"]
    assert sorted(params) == sorted(NOTE_FIELDS)
    assert sorted(vars(MemoryNote("x"))) == sorted(NOTE_FIELDS)


def test_ingest_uses_the_library_path_and_keeps_every_revision(tmp_path, hash_embedder):
    llm = _ScriptedLLM()
    sys_ = _amem(tmp_path / "s", llm, hash_embedder)
    out = sys_.ingest(EVENTS)
    assert out["ingested"] == 4 and sys_.stats["failed"] == 0
    # one analysis call per note, one evolution call per note after the first; one encoder call
    # per added note and one per neighbour query; nothing metered on the embedding endpoint
    assert out["llm_calls"] == 4 + 3 and out["embed_calls"] == 4 + 3 and out["embed_tokens"] == 0
    first = llm.calls[0]
    assert first["temperature"] == 0.0 and first["max_tokens"] == 4096
    assert first["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    assert first["model"] == "Qwen/Qwen3.5-9B"
    rows = {r["id"]: r for r in sys_.export_raw()}
    assert list(rows) == ["contacts:p01#1", "contacts:p01#2", "notes:n1#1", "contacts:p01#3"]
    for e in EVENTS:  # the raw record is retrievable, verbatim, under its key
        assert rows[event_key(e)]["content"] == event_text(e)
        assert rows[event_key(e)]["timestamp"] == note_timestamp(e.ts)
        assert rows[event_key(e)]["category"] == e.source
    # a newer revision and a tombstone are further notes: the first revision is still stored
    assert rows["contacts:p01#1"]["content"].endswith("Body of contacts:p01 revision 1")
    assert rows["contacts:p01#3"]["content"].startswith("Record contacts:p01 was deleted")
    # the evolution step links the new note to a neighbour and retags the neighbours ...
    assert rows["contacts:p01#3"]["links"] and rows["contacts:p01#3"]["tags"] == ["linked"]
    assert rows["contacts:p01#1"]["context"] == "re-read" and rows["contacts:p01#1"]["tags"] == [
        "evolved"
    ]
    assert rows["contacts:p01#1"]["evolution_history"] == []
    # ... but the index still holds the metadata the note had when it was added
    indexed = rows["contacts:p01#1"]["index_metadata"]
    assert indexed["context"] == "About a record." and json.loads(indexed["tags"]) == ["t"]
    assert sys_.index_count() == 4
    assert all(v["op"] in ("insert", "update", "delete") for v in sys_.ledger.entries.values())
    # consolidation re-adds every note; the index ignores ids it holds, so it stays as it was
    sys_._mem.consolidate_memories()
    again = {r["id"]: r for r in sys_.export_raw()}
    assert again["contacts:p01#1"]["index_metadata"] == indexed and sys_.index_count() == 4
    assert again["contacts:p01#1"]["context"] == "re-read"
    sys_.close()


def test_snapshot_restore_reproduces_the_store(tmp_path, hash_embedder):
    sys_ = _amem(tmp_path / "live", _ScriptedLLM(), hash_embedder)
    sys_.ingest(EVENTS[:3])
    sys_.snapshot(tmp_path / "snap")
    assert (tmp_path / "snap" / "chroma").is_dir() and (tmp_path / "snap" / "notes.json").is_file()
    before = sys_.export_raw()
    query = "contacts:p01 revision 2"
    ranked_before = [h.hit_id for h in sys_._search(query)]
    sys_.ingest(EVENTS[3:], phase="maintain")
    assert sys_.index_count() == 4
    branch = _amem(tmp_path / "branch", _ScriptedLLM(), hash_embedder)
    branch.restore(tmp_path / "snap")
    assert branch.export_raw() == before and branch.index_count() == 3
    assert [h.hit_id for h in branch._search(query)] == ranked_before
    assert len(branch.ledger) == 3 and branch.usage.snapshot()["llm_calls"] == 3 + 2
    assert branch.cost["ingest_tokens"] > 0 and branch.cost["maintain_tokens"] == 0
    # the restored branch continues like the uninterrupted store
    assert branch.ingest(EVENTS, phase="maintain")["ingested"] == 1
    assert [r["id"] for r in branch.export_raw()] == [r["id"] for r in sys_.export_raw()]
    assert branch.ledger.entries.keys() == sys_.ledger.entries.keys()
    sys_.close()
    branch.close()


def test_search_returns_the_library_ranking_with_linked_notes_and_refs(tmp_path, hash_embedder):
    sys_ = _amem(tmp_path / "s", _ScriptedLLM(), hash_embedder)
    sys_.ingest(EVENTS)
    hits = sys_._search("anything")
    ids = [h.hit_id for h in hits]
    assert sorted(ids) == sorted(event_key(e) for e in EVENTS) and len(set(ids)) == 4
    assert all(h.refs == [parse_key(h.hit_id)] for h in hits)
    assert all(h.text.startswith("[note | source ") for h in hits)
    tomb = next(h for h in hits if h.hit_id == "contacts:p01#3")
    assert tomb.text.endswith("Record contacts:p01 was deleted on 2026-03-01 09:34.")
    tokens0 = sys_.usage.tokens
    res = sys_.serve(SimpleNamespace(question_text="where does p01 live"))
    assert res.blocks and not res.blocked and sys_.usage.tokens == tokens0  # no LLM at query time
    assert sys_.cost["serve_tokens"] == 0
    sys_.close()


def test_library_failures_are_recorded_and_the_library_falls_back(tmp_path, hash_embedder):
    llm = _ScriptedLLM()
    llm.fail_analysis = True
    strict = _amem(tmp_path / "strict", llm, hash_embedder)
    with pytest.raises(RuntimeError, match="ConnectionError"):
        strict.ingest(EVENTS[:1])
    strict.close()
    lenient = _amem(tmp_path / "lenient", llm, hash_embedder, on_error="skip")
    lenient.ingest(EVENTS[:2])
    assert lenient.stats["failed"] == 2 and len(lenient.ledger) == 2
    rows = lenient.export_raw()
    assert [r["keywords"] for r in rows] == [[], []] and rows[1]["context"] == "General"
    assert rows[0]["content"] == event_text(EVENTS[0])  # stored with default metadata, not lost
    lenient.close()


def test_lost_ledger_is_rebuilt_from_the_notes_file(tmp_path, hash_embedder):
    sys_ = _amem(tmp_path / "s", _ScriptedLLM(), hash_embedder)
    sys_.ingest(EVENTS)
    sys_.close()
    (tmp_path / "s" / "ledger.jsonl").unlink()
    again = _amem(tmp_path / "s", _ScriptedLLM(), hash_embedder)
    assert sorted(again.ledger.entries) == sorted(event_key(e) for e in EVENTS)
    assert again.ingest(EVENTS)["ingested"] == 0
    again.close()


def test_the_encoder_is_built_pinned_and_nothing_else_loads_one(
    tmp_path, hash_embedder, monkeypatch
) -> None:
    from chromadb.utils import embedding_functions
    from sentence_transformers import SentenceTransformer

    built: list[dict] = []

    def fake_encoder(**kw):
        built.append(kw)
        return hash_embedder

    def no_load(self, *args, **kw):
        raise AssertionError("an encoder was loaded outside the pinned path")

    monkeypatch.setattr(embedding_functions, "SentenceTransformerEmbeddingFunction", fake_encoder)
    monkeypatch.setattr(SentenceTransformer, "__init__", no_load)
    # the library constructor (which builds an encoder by short name) runs with the index stubbed
    sys_ = _amem(tmp_path / "s", _ScriptedLLM(), None)
    assert built == [encoder_kwargs()]
    sys_.ingest(EVENTS[:2])
    sys_._mem.consolidate_memories()  # the library's consolidation would build another one
    assert built == [encoder_kwargs()]
    sys_.close()

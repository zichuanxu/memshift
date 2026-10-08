"""Offline tests for the native-system adapters and the Portable Memory export (plan §5.5, §6 E5,
E6). No network, no GPU: native libraries are never contacted here; the .mem round trip needs
the `portable-memory` package (`uv sync --extra native`) and is skipped without it."""

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from memshift.adapters.native import (
    STORE_MARKER,
    Hit,
    Ledger,
    MeteredOpenAI,
    NativeSystem,
    Usage,
    copy_store,
    distinct_batches,
    event_key,
    event_metadata,
    event_text,
    graphiti_hit,
    mem0_hit,
    parse_key,
    replace_store,
)
from memshift.gen.model import Event
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import READER_BUDGET, CellConfig, HoldMem
from memshift.probes import make_probes
from memshift.reader import context_hash
from memshift.textutil import approx_tokens
from tests.fakes import FakeLLM

T0 = datetime(2026, 3, 1, 9, 30)


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


# ---------------------------------------------------------------------- conversion
def test_event_text_and_metadata():
    up = _ev(1, "email:e1", 2, "update", minutes=5)
    text = event_text(up)
    head, body = text.split("\n", 1)
    assert head == "[email record email:e1 | revision 2 | 2026-03-01 09:35]"
    assert body == "Body of email:e1 revision 2"
    gone = _ev(2, "contacts:p01", 3, "delete", minutes=7)
    assert event_text(gone) == "Record contacts:p01 was deleted on 2026-03-01 09:37."
    assert event_metadata(up) == {
        "object_id": "email:e1",
        "revision": 2,
        "source": "email",
        "ts": "2026-03-01T09:35",
        "op": "update",
    }


def test_event_key_round_trip():
    e = _ev(1, "calendar:series:3", 4)
    assert event_key(e) == "calendar:series:3#4"
    assert parse_key(event_key(e)) == ("calendar:series:3", 4)
    assert parse_key("no-revision") is None
    assert parse_key("email:e1#x") is None
    assert parse_key("#3") is None


def test_distinct_batches_keep_object_order():
    evs = [_ev(1, "a:1", 1), _ev(2, "a:2", 1), _ev(3, "a:1", 2), _ev(4, "a:3", 1), _ev(5, "a:4", 1)]
    batches = distinct_batches(evs, 3)
    assert [[event_key(e) for e in b] for b in batches] == [
        ["a:1#1", "a:2#1"],
        ["a:1#2", "a:3#1", "a:4#1"],
    ]
    assert [len(b) for b in distinct_batches(evs, 1)] == [1] * 5


# ---------------------------------------------------------------------- bookkeeping
def test_ledger_pending_persistence_and_torn_line(tmp_path):
    path = tmp_path / "ledger.jsonl"
    led = Ledger(path)
    evs = [_ev(3, "a:1", 2), _ev(1, "a:1", 1), _ev(2, "a:2", 1), _ev(4, "a:2", 1)]  # dup key
    todo = led.pending(evs)
    assert [event_key(e) for e in todo] == ["a:1#1", "a:2#1", "a:1#2"]  # time order, once
    led.record("a:1#1", event_id="ev000001")
    with path.open("a") as f:
        f.write('{"key": "a:2#1", "event')  # a crash mid-write leaves a torn line
    again = Ledger(path)
    assert "a:1#1" in again and "a:2#1" not in again and len(again) == 1
    assert [event_key(e) for e in again.pending(evs)] == ["a:2#1", "a:1#2"]


def _fake_store(path: Path, system: str = "fake", files: dict[str, str] | None = None) -> Path:
    path.mkdir(parents=True)
    (path / STORE_MARKER).write_text(json.dumps({"system": system}))
    for rel, text in (files or {}).items():
        (path / rel).parent.mkdir(parents=True, exist_ok=True)
        (path / rel).write_text(text)
    return path


def test_copy_and_replace_store(tmp_path):
    src = _fake_store(tmp_path / "src", files={"qdrant/storage.sqlite": "v1", "history.db": "h1"})
    snap = tmp_path / "snap"
    copy_store(src, snap)
    assert (snap / "qdrant/storage.sqlite").read_text() == "v1"
    with pytest.raises(FileExistsError):
        copy_store(src, snap)  # never overwrite a snapshot
    live = _fake_store(tmp_path / "live", files={"qdrant/storage.sqlite": "v2", "stale.bin": "x"})
    replace_store(snap, live)
    assert (live / "qdrant/storage.sqlite").read_text() == "v1"
    assert not (live / "stale.bin").exists()
    stranger = tmp_path / "stranger"
    stranger.mkdir()
    (stranger / "precious.txt").write_text("keep")
    with pytest.raises(ValueError):
        replace_store(snap, stranger)  # not a native store: never deleted
    assert (stranger / "precious.txt").read_text() == "keep"
    with pytest.raises(ValueError):
        copy_store(stranger, tmp_path / "elsewhere")


# ---------------------------------------------------------------------- metered clients
class _Recorder:
    def __init__(self) -> None:
        self.kw: list[dict[str, Any]] = []

    def create(self, **kw: Any) -> Any:
        self.kw.append(kw)
        return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=11, completion_tokens=3))


class _AsyncRecorder(_Recorder):
    async def acreate(self, **kw: Any) -> Any:
        return self.create(**kw)


def test_metered_openai_injects_extras_caps_tokens_and_counts():
    chat, emb = _Recorder(), _Recorder()
    inner = SimpleNamespace(
        chat=SimpleNamespace(completions=chat), embeddings=emb, base_url="http://x/v1"
    )
    usage = Usage()
    proxy = MeteredOpenAI(
        inner,
        usage,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        max_tokens_cap=4096,
    )
    proxy.chat.completions.create(model="m", messages=[], max_tokens=16384, extra_body={"a": 1})
    proxy.embeddings.create(model="e", input=["x", "y"])
    assert chat.kw[0]["max_tokens"] == 4096
    assert chat.kw[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}, "a": 1}
    assert usage.snapshot() == {
        "llm_calls": 1,
        "prompt_tokens": 11,
        "completion_tokens": 3,
        "embed_calls": 1,
        "embed_tokens": 11,
    }
    assert proxy.base_url == "http://x/v1"  # everything else passes through


def test_metered_openai_async():
    chat, emb = _AsyncRecorder(), _AsyncRecorder()
    inner = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=chat.acreate)),
        embeddings=SimpleNamespace(create=emb.acreate),
    )
    usage = Usage({"llm_calls": 5})
    proxy = MeteredOpenAI(inner, usage, is_async=True, max_tokens_cap=100)

    async def go() -> None:
        await proxy.chat.completions.create(model="m", messages=[], max_tokens=50)
        await proxy.embeddings.create(model="e", input="x")

    asyncio.run(go())
    assert chat.kw[0]["max_tokens"] == 50 and "extra_body" not in chat.kw[0]
    assert usage.snapshot()["llm_calls"] == 6 and usage.snapshot()["embed_calls"] == 1


# ---------------------------------------------------------------------- serve-side hits
def test_mem0_hit_block():
    row = {
        "id": "m1",
        "memory": "The user works at Acme.",
        "score": 0.8,
        "created_at": "2026-09-28T10:00:00+00:00",  # wall clock: never shown
        "metadata": {
            "object_id": "email:e1",
            "revision": 2,
            "source": "email",
            "ts": "2026-03-01T09:35",
        },
    }
    h = mem0_hit(row)
    assert (
        h.text
        == "[memory | source email email:e1 r2 | event 2026-03-01 09:35] The user works at Acme."
    )
    assert h.refs == [("email:e1", 2)] and h.ts == datetime(2026, 3, 1, 9, 35) and h.score == 0.8
    imported = mem0_hit(
        {
            "id": "m2",
            "memory": "s_k text",
            "metadata": {
                "event_time": "2026-10-01T00:00",
                "source_refs": [{"object_id": "contacts:p01", "revision": 1}],
            },
        }
    )
    assert imported.text.startswith("[memory | source contacts:p01 r1 | event 2026-10-01 00:00]")
    bare = mem0_hit({"id": "m3", "memory": "x"})
    assert bare.text == "[memory] x" and bare.refs == []


def test_graphiti_hit_block():
    h = graphiti_hit(
        "e1",
        "Alice lives in Boston",
        datetime(2026, 2, 1, 12, 0, tzinfo=UTC),
        datetime(2026, 5, 1, 8, 0, tzinfo=UTC),
        True,
        ["contacts:p01#2", "", "not-a-key"],
    )
    assert h.text == (
        "[fact | valid from 2026-02-01 12:00 | invalid from 2026-05-01 08:00 | expired"
        " | source contacts:p01 r2] Alice lives in Boston"
    )
    assert h.refs == [("contacts:p01", 2)] and h.ts == datetime(2026, 2, 1, 12, 0)
    open_ended = graphiti_hit("e2", "f", None, None, False, [], reference_time=T0)
    assert open_ended.text == "[fact] f" and open_ended.ts == T0


# ---------------------------------------------------------------------- shared driver side
class _FakeNative(NativeSystem):
    """A native system whose 'store' is a JSONL file: exercises ingest bookkeeping, resume,
    reconciliation, snapshot / restore and serve packing without any native library."""

    name = "fake"

    @property
    def retains_raw_text(self) -> bool:
        return True

    def _open(self) -> None:
        self.data = self.store_dir / "data.jsonl"

    def _close(self) -> None:
        pass

    def _reconcile(self) -> None:
        if self.data.is_file():
            for line in self.data.read_text().splitlines():
                md = json.loads(line)
                key = f"{md['object_id']}#{md['revision']}"
                if key not in self.ledger:
                    self.ledger.record(key, reconciled=True)

    def _ingest_events(self, events: list[Event]) -> None:
        for e in events:
            with self.data.open("a") as f:
                f.write(json.dumps(event_metadata(e)) + "\n")
            self.usage.add(llm_calls=1, prompt_tokens=10, completion_tokens=2)
            self.ledger.record(event_key(e), event_id=e.event_id)

    def _search(self, query: str) -> list[Hit]:
        return [
            Hit(
                f"h{i:02d}",
                "word " * 300,
                T0 - timedelta(days=i),
                [(f"notes:n{i}", 1)],
                1 / (i + 1),
            )
            for i in range(30)
        ]

    def export_raw(self) -> list[dict[str, Any]]:
        return [json.loads(line) for line in self.data.read_text().splitlines()]


def _fake_native(store: Path) -> _FakeNative:
    return _FakeNative(None, store, "http://x/v1", "llm", "http://x/v1", "emb", "u1")  # type: ignore[arg-type]


def test_native_ingest_resume_snapshot_restore(tmp_path):
    evs = [_ev(i, f"notes:n{i % 6}", 1 + i // 6) for i in range(1, 9)]
    sys_ = _fake_native(tmp_path / "store")
    out = sys_.ingest(evs[:5])
    assert out["ingested"] == 5 and out["llm_calls"] == 5
    assert sys_.cost["ingest_tokens"] == 60 and sys_.cost["storage_bytes"] > 0
    assert sys_.ingest(evs[:5])["skipped"] == 5  # resumed ingest adds nothing twice
    sys_.snapshot(tmp_path / "snap")
    assert sys_.ingest(evs, phase="maintain")["ingested"] == 3
    assert sys_.cost["maintain_tokens"] == 36 and len(sys_.export_raw()) == 8
    sys_.restore(tmp_path / "snap")
    assert len(sys_.ledger) == 5 and len(sys_.export_raw()) == 5
    assert sys_.usage.snapshot()["llm_calls"] == 5  # a branch starts from the prefix usage
    assert sys_.cost["ingest_tokens"] == 60 and sys_.cost["maintain_tokens"] == 0
    assert sys_.ingest(evs)["ingested"] == 3
    sys_.close()
    (tmp_path / "store" / "ledger.jsonl").unlink()  # lost ledger: rebuilt from the store
    again = _fake_native(tmp_path / "store")
    assert len(again.ledger) == 8 and again.ingest(evs)["ingested"] == 0
    assert again.identifiable


def test_native_restore_refuses_other_system(tmp_path):
    sys_ = _fake_native(tmp_path / "store")
    _fake_store(tmp_path / "mem0snap", system="mem0")
    with pytest.raises(ValueError):
        sys_.restore(tmp_path / "mem0snap")


def test_native_serve_packs_into_reader_budget(tmp_path):
    sys_ = _fake_native(tmp_path / "store")
    probe = SimpleNamespace(question_text="anything")
    res = sys_.serve(probe)
    assert res.blocks and not res.blocked
    assert sum(approx_tokens(b) + 8 for b in res.blocks) <= READER_BUDGET
    assert len(res.refs) == len(res.blocks) and len(res.items) == 30
    # rank order decides what fits (h00, h01, ...), presentation is chronological
    assert res.refs[-1] == ("notes:n0", 1) and res.refs[0][0] == f"notes:n{len(res.refs) - 1}"


# ---------------------------------------------------------------------- .mem round trip
@pytest.fixture(scope="module")
def world_log():
    w = generate_branch(11, "src")
    return w, SourceLog(w.events)


def _built(world, log, cfg):
    hm = HoldMem(world, log, cfg, FakeLLM(), "fake/writer-a")
    hm.workers = 1
    hm.build(world.fork)
    return hm


def _slot_refs_resolvable(hm):
    live = {(e.object_id, e.revision) for e in hm.log.events if e.op != "delete"}
    return sum(bool(i.refs) and i.refs[0] in live for i in hm.items.values())


def test_export_standard_records_and_custom_kind(world_log, tmp_path):
    pytest.importorskip("portable_memory")
    from memshift.adapters.mem_export import (
        EXT_FIELDS,
        EXT_KEY,
        ITEM_KIND,
        export_holdmem,
        validate_bundle,
    )

    w, log = world_log
    hm = _built(w, log, CellConfig(1, 1, "source"))
    n_src = _slot_refs_resolvable(hm)
    assert n_src > 0
    for with_ext in (True, False):
        d = tmp_path / f"ext{int(with_ext)}.mem"
        manifest = export_holdmem(hm, d, with_ext=with_ext)
        assert validate_bundle(d) == []
        assert {"manifest.json", "CHECKSUMS", "items"} <= {p.name for p in d.iterdir()}
        assert manifest.counts["fact"] == len(hm.items)
        facts = [json.loads(x) for x in (d / "items" / "fact.jsonl").read_text().splitlines()]
        eps = [json.loads(x) for x in (d / "items" / "episode.jsonl").read_text().splitlines()]
        for f in facts:
            item = hm.items[f["id"].removeprefix("holdmem:")]
            assert (f["text"], f["predicate"], f["object"]) == (item.s_k, item.slot_id, "")
            assert "tValidTo" not in f and EXT_KEY not in f  # all served; no ext on facts
        with_src = [f for f in facts if f["episodeID"]]
        assert len(with_src) == n_src
        assert {f["episodeID"] for f in with_src} == {e["id"] for e in eps}
        assert all(e["id"].startswith("src:") and EXT_KEY not in e for e in eps)
        custom = d / "items" / f"{ITEM_KIND}.jsonl"
        assert custom.is_file() is with_ext
        if with_ext:
            recs = [json.loads(x) for x in custom.read_text().splitlines()]
            assert len(recs) == len(hm.items)
            for r in recs:
                assert tuple(sorted(r)) == tuple(sorted(EXT_FIELDS))
                item = hm.items[r["item_id"]]
                assert (r["bind"], r["det"], r["s_k"]) == (item.bind, item.det, item.s_k)
                assert r["theta_w_id"] == "fake/writer-a"
                assert r["source_refs"] == [{"object_id": o, "revision": v} for o, v in item.refs]
    (tmp_path / "ext1.mem" / "items" / "fact.jsonl").write_text("tampered\n")
    assert any("checksum" in i for i in validate_bundle(tmp_path / "ext1.mem"))


def test_ext_survival_by_carrier(world_log, tmp_path):
    """Observed through the SDK's own importer and exporter: the custom kind and episode `ext`
    survive a round trip, `ext` on fact records (RFC-0002 style) is dropped on import."""
    pytest.importorskip("portable_memory")
    from memshift.adapters.mem_export import EXT_FIELDS, export_holdmem, ext_survival

    w, log = world_log
    hm = _built(w, log, CellConfig(0, 1, "cited"))
    n = len(hm.items)
    export_holdmem(hm, tmp_path / "all.mem", carriers=("kind", "episode", "fact"))
    rep = ext_survival(tmp_path / "all.mem")
    assert rep["issues"] == [] and rep["issues_after_reexport"] == []
    kind, episode, fact = (rep["carriers"][c] for c in ("kind", "episode", "fact"))
    for c in (kind, episode):
        assert c["survives"] and c["records_written"] == c["records_after_second_import"] == n
        assert c["fields_written"] == sorted(EXT_FIELDS) and c["fields_dropped_on_import"] == []
    assert fact["records_written"] == n and not fact["survives"]
    assert fact["records_imported"] == fact["records_after_reexport"] == 0
    assert fact["fields_dropped_on_import"] == sorted(EXT_FIELDS)
    assert rep["native"]["facts"] == rep["native"]["facts_after_reexport"] == n
    assert rep["native"]["facts_with_source_episode"] == _slot_refs_resolvable(hm)
    export_holdmem(hm, tmp_path / "plain.mem", with_ext=False)
    plain = ext_survival(tmp_path / "plain.mem")
    assert all(c["records_written"] == 0 for c in plain["carriers"].values())
    assert plain["native"]["facts"] == n


def test_native_provenance_propagates_source_deletion(world_log, tmp_path):
    """Without any extension the format itself drops a fact whose evidence episode is
    tombstoned (tombstones are applied before facts are merged)."""
    pytest.importorskip("portable_memory")
    from portable_memory import BundleExporter, Tombstone, TombstoneOp

    from memshift.adapters.mem_export import export_holdmem, read_bundle

    w, log = world_log
    hm = _built(w, log, CellConfig(1, 0))
    export_holdmem(hm, tmp_path / "a.mem", with_ext=False)
    first = read_bundle(tmp_path / "a.mem")
    victim = next(f.episode_id for f in first.facts if f.episode_id)
    dependants = {f.id for f in first.facts if f.episode_id == victim}
    first.store.tombstones.append(
        Tombstone(
            id="t1",
            op=TombstoneOp.DELETE,
            target_kind="episode",
            target_id=victim,
            deleted_at=datetime(2026, 10, 2, tzinfo=UTC),
            actor="source-connector",
        )
    )
    BundleExporter().export(first.store, str(tmp_path / "b.mem"))
    second = read_bundle(tmp_path / "b.mem")
    assert second.report.tombstones_applied == 1 and victim not in second.episodes
    assert {f.id for f in second.facts} == {f.id for f in first.facts} - dependants


@pytest.mark.parametrize(
    ("carriers", "source"), [(("kind",), "kind"), (("episode",), "episode"), (("fact",), None)]
)
def test_import_restores_binding_from_surviving_carrier(world_log, tmp_path, carriers, source):
    pytest.importorskip("portable_memory")
    from memshift.adapters.mem_export import export_holdmem, import_to_holdmem

    w, log = world_log
    cfg = CellConfig(1, 1, "source")
    native = _built(w, log, cfg)
    export_holdmem(native, tmp_path / "b.mem", carriers=carriers)
    imported = import_to_holdmem(tmp_path / "b.mem", w, log, cfg, writer=FakeLLM())
    assert imported.ext_source == source
    assert set(imported.items) == set(native.items)
    assert imported.last_t == native.last_t == w.fork
    t = w.fork + timedelta(weeks=1)
    if source is None:  # the fact-level ext did not survive: degraded to B=0
        assert imported.cfg.B == 0
        assert all(i.bind is None for i in imported.items.values())
        assert imported.maintain(t)["invalidated"] == 0
        return
    assert imported.cfg == cfg and imported.theta_w == "fake/writer-a"
    for iid, item in native.items.items():
        got = imported.items[iid]
        assert (got.s_k, got.bind, got.det, got.status, got.slot_id) == (
            item.s_k,
            item.bind,
            item.det,
            item.status,
            item.slot_id,
        )
        assert (got.refs, got.readset, got.written) == (item.refs, item.readset, item.written)
    for p in make_probes(w, log, [w.fork])[:40]:
        assert context_hash(imported.serve(p).blocks) == context_hash(native.serve(p).blocks)
    s_native, s_imported = native.maintain(t), imported.maintain(t)
    assert s_imported["invalidated"] == s_native["invalidated"] > 0
    assert s_imported["repaired"] == s_imported["invalidated"]


def test_import_without_ext_degrades_to_append_only(world_log, tmp_path):
    pytest.importorskip("portable_memory")
    from memshift.adapters.mem_export import export_holdmem, import_to_holdmem

    w, log = world_log
    native = _built(w, log, CellConfig(1, 1, "source"))
    export_holdmem(native, tmp_path / "plain.mem", with_ext=False)
    imported = import_to_holdmem(
        tmp_path / "plain.mem", w, log, CellConfig(1, 1, "source"), writer=FakeLLM()
    )
    assert imported.cfg.B == 0 and imported.ext_source is None
    assert imported.name == "imported_holdmem_D1B0_noext"
    assert set(imported.items) == set(native.items)
    for iid, item in native.items.items():
        got = imported.items[iid]
        assert got.bind is None and got.det is None and got.readset == []
        assert (got.s_k, got.slot_id, got.written) == (item.s_k, item.slot_id, item.written)
        assert got.refs == item.refs[:1] or not got.refs  # native provenance: one evidence ref
    p = make_probes(w, log, [w.fork])[0]
    assert imported.serve(p).blocks
    stats = imported.maintain(w.fork + timedelta(weeks=1))
    assert stats["invalidated"] == 0 and stats["appended"] > 0
    assert not any(h["event"] == "invalidate" for i in imported.items.values() for h in i.history)


def test_native_batch_reaches_only_the_bulk_capable_systems(monkeypatch):
    from memshift.experiments import e5

    seen: dict[str, dict] = {}

    def fake(kind):
        class Fake:
            def __init__(self, *args, **kw):
                seen[kind] = kw

        return Fake

    monkeypatch.setattr(
        e5, "NATIVE", {k: fake(k) for k in ("amem", "hipporag", "lightrag", "memos")}
    )
    monkeypatch.setattr(e5, "NATIVE_BATCH", 8)
    for kind in ("amem", "hipporag", "lightrag", "memos"):
        e5._make(kind, type("W", (), {"world_id": "w"})(), None, "u", "e")
    assert seen["amem"] == {"on_error": e5.ON_ERROR}
    assert seen["hipporag"]["batch_size"] == 8 and "options" not in seen["hipporag"]
    assert seen["lightrag"]["batch_size"] == 8
    assert seen["lightrag"]["options"] == {"max_parallel_insert": 8, "llm_model_max_async": 8}
    assert seen["memos"]["workers"] == 8 and "batch_size" not in seen["memos"]
    monkeypatch.setattr(e5, "NATIVE_BATCH", 1)
    e5._make("lightrag", type("W", (), {"world_id": "w"})(), None, "u", "e")
    assert "batch_size" not in seen["lightrag"] and "options" not in seen["lightrag"]


def test_resumed_ingest_does_not_retry_failed_units(tmp_path):
    class Flaky(_FakeNative):
        def _ingest_events(self, events: list[Event]) -> None:
            for e in events:
                if e.object_id == "notes:n2":
                    self._fail(event_key(e), RuntimeError("extraction failed"))
                    continue
                super()._ingest_events([e])

    evs = [_ev(i, f"notes:n{i}", 1) for i in range(1, 5)]
    first = Flaky(None, tmp_path / "s", "u", "llm", "u", "emb", "u1", on_error="skip")
    assert first.ingest(evs)["ingested"] == 3 and first.stats["failed"] == 1
    first.close()
    again = Flaky(None, tmp_path / "s", "u", "llm", "u", "emb", "u1", on_error="skip")
    out = again.ingest(evs)
    assert out["skipped"] == 4 and out["ingested"] == 0 and again.stats["failed"] == 0
    assert again.failed_keys() == {"notes:n2#1"}


class _Answer:
    def __init__(self, probe: Any, system: str):
        self.row = {**probe.to_json(), "system": system}

    def to_json(self) -> dict[str, Any]:
        return self.row


def _split_env(monkeypatch, tmp_path):
    from memshift.experiments import e5
    from memshift.experiments.common import EMBED_MODEL, MAIN_MODEL

    monkeypatch.setattr(e5, "REPO", tmp_path)
    monkeypatch.setattr(e5, "NATIVE", {"fake": _FakeNative})
    monkeypatch.setattr(
        e5, "endpoints", lambda: {MAIN_MODEL: "http://x/v1", EMBED_MODEL: "http://x/v1"}
    )
    monkeypatch.setattr(e5, "client", lambda *a, **k: None)
    monkeypatch.setattr(
        e5,
        "answer_probes",
        lambda kind, serve, probes, reader: ([_Answer(p, kind) for p in probes],),
    )
    return e5


def _rows(run: Path, name: str) -> list[str]:
    return sorted(line for line in (run / name).read_text().splitlines())


def test_e5_split_phases_reproduce_one_run(monkeypatch, tmp_path):
    from memshift.experiments.common import world_params

    e5 = _split_env(monkeypatch, tmp_path)
    wp = world_params(3)
    runs = tmp_path / "runs"
    for name in ("whole", "pre", "b_src", "b_rest"):
        (runs / name).mkdir(parents=True)
    whole = e5.run_world(1000, "fake", runs / "whole", wp)
    pre = e5.run_world(1000, "fake", runs / "pre", wp, phase="prefix")
    src = e5.run_world(
        1000, "fake", runs / "b_src", wp, phase="branch", prefix_run="pre", branches=("src",)
    )
    rest = e5.run_world(
        1000,
        "fake",
        runs / "b_rest",
        wp,
        phase="branch",
        prefix_run="pre",
        branches=("def", "orphan"),
    )
    assert pre["phase"] == "prefix" and "branches" not in pre
    assert set(src["branches"]) == {"src"} and set(rest["branches"]) == {"def", "orphan"}
    assert src["identifiable"] and src["prefix_run"] == "pre"
    for f in ("probes.jsonl", "answers.jsonl"):
        split = sorted(
            _rows(runs / "pre", f) + _rows(runs / "b_src", f) + _rows(runs / "b_rest", f)
        )
        assert split == _rows(runs / "whole", f)
    for b in ("src", "def", "orphan"):
        part = runs / ("b_src" if b == "src" else "b_rest")
        store = next((part / "stores").iterdir()) / b / "data.jsonl"
        assert (
            store.read_text()
            == (next((runs / "whole" / "stores").iterdir()) / b / "data.jsonl").read_text()
        )
        split_stats = (src if b == "src" else rest)["branches"][b]["stats"]
        for k in ("ingested", "failed"):
            assert whole["branches"][b]["stats"][k] == split_stats[k]
    with pytest.raises(ValueError):
        e5.run_world(1000, "fake", runs / "whole", wp, phase="branch")
    with pytest.raises(FileNotFoundError):
        e5.run_world(1000, "fake", runs / "b_src", wp, phase="branch", prefix_run="nope")


def test_e5_resume_continues_an_interrupted_prefix(monkeypatch, tmp_path):
    from memshift.experiments.common import world_params

    e5 = _split_env(monkeypatch, tmp_path)
    wp = world_params(3)
    runs = tmp_path / "runs"
    natural = generate_branch(1001, "natural", wp)
    events = SourceLog(natural.events).upto(natural.fork - timedelta(minutes=1))
    store = runs / "cut" / "stores" / f"{natural.world_id}_fake" / "prefix"
    part = _fake_native(store)
    part.ingest(events[:100])
    part.close()
    (runs / "res").mkdir()
    (runs / "whole").mkdir()
    res = e5.run_world(1001, "fake", runs / "res", wp, phase="prefix", resume_from="cut")
    whole = e5.run_world(1001, "fake", runs / "whole", wp, phase="prefix")
    assert res["resumed_from"] == "cut" and res["t_i"] is None
    assert res["prefix_ingest"]["skipped"] == 100
    assert res["prefix_stats"]["ingested"] == whole["prefix_stats"]["ingested"] == len(events)
    new = runs / "res" / "stores" / f"{natural.world_id}_fake"
    old = runs / "whole" / "stores" / f"{natural.world_id}_fake"
    assert (new / "snapshot" / "data.jsonl").read_text() == (
        old / "snapshot" / "data.jsonl"
    ).read_text()
    assert len((store / "ledger.jsonl").read_text().splitlines()) == 100  # the source is untouched
    assert _rows(runs / "res", "probes.jsonl") == _rows(runs / "whole", "probes.jsonl")

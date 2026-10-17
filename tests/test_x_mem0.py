"""Block X (prereg/v4_round3.md): the Mem0 event log, the two interventions and the analysis, on a
stand-in Memory and on synthetic event logs. No mem0, network or GPU."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import httpx2
import openai
import pytest

from memshift.adapters.native import MeteredOpenAI, NativeSystem, event_key
from memshift.adapters.native_mem0x import (
    EVENTS_FILE,
    InfrastructureError,
    LoggedMem0,
    Mem0Det,
    Mem0Rep,
    RetryPolicy,
    is_infra,
    merge_events,
    read_events,
    retrying,
)
from memshift.analysis import x_mem0 as xa
from memshift.experiments import x_mem0 as xd
from memshift.experiments.common import world_params
from memshift.gen.model import Event, fmt_ts
from memshift.gen.shifts import generate_branch
from memshift.gen.state import SourceLog
from memshift.holdmem.system import READER_BUDGET, READER_DOC_CAP
from memshift.llm import LLMClient
from memshift.probes import make_probe
from memshift.retrieval import Doc, pack
from memshift.textutil import approx_tokens
from tests.fakes import RecordingProgramLLM
from tests.test_zero_dm import PROGRAMS

T0 = datetime(2026, 3, 1, 9, 0)


# ---------------------------------------------------------------------- a stand-in Memory
class _Vectors:
    """Qdrant stand-in: memories by id, persisted next to the history DB."""

    client = None

    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict[str, Any]] = (
            json.loads(path.read_text()) if path.is_file() else {}
        )

    def save(self) -> None:
        self.path.write_text(json.dumps(self.rows))

    def get(self, vector_id: str) -> Any:
        row = self.rows.get(vector_id)
        return SimpleNamespace(id=vector_id, payload=row) if row is not None else None

    def list(self, filters: dict[str, Any], top_k: int) -> list[Any]:
        return []  # nothing to reconcile: the ledger is complete in these tests


class FakeMemory:
    """Writes history rows as mem0ai 2.2.1 does (ADD per new fact); `updates` maps a unit's object
    to a memory id it UPDATEs (a pipeline that can update, to exercise the log); `fail` lists
    objects whose raw archive add raises after the extraction add stored its fact; `raises` maps
    an object to the error its extraction add raises (Mem0 raises `LLMError` from the client's)."""

    def __init__(
        self,
        store: Path,
        updates: dict[str, str] | None = None,
        fail: frozenset[str] = frozenset(),
        raises: dict[str, Exception] | None = None,
    ):
        self.vector_store = _Vectors(store / "vectors.json")
        conn = sqlite3.connect(str(store / "history.db"), check_same_thread=False)
        conn.execute(
            "CREATE TABLE IF NOT EXISTS history (id TEXT PRIMARY KEY, memory_id TEXT, "
            "old_memory TEXT, new_memory TEXT, event TEXT, created_at DATETIME, "
            "updated_at DATETIME, is_deleted INTEGER, actor_id TEXT, role TEXT)"
        )
        self.db = SimpleNamespace(connection=conn, _lock=threading.Lock())
        self.updates, self.fail, self.raises = updates or {}, fail, raises or {}

    def _history(self, mid: str, old: str | None, new: str | None, event: str) -> None:
        self.db.connection.execute(
            "INSERT INTO history (id, memory_id, old_memory, new_memory, event) VALUES (?,?,?,?,?)",
            (str(uuid.uuid4()), mid, old, new, event),
        )
        self.db.connection.commit()

    def add(self, messages: list[dict[str, str]], *, user_id: str, metadata: dict, infer: bool):
        oid = metadata["object_id"]
        if infer and oid in self.raises:
            raise self.raises[oid]
        if not infer and oid in self.fail:
            raise RuntimeError("archive store down")
        text = messages[0]["content"] if not infer else f"fact of {oid} r{metadata['revision']}"
        mid = str(uuid.uuid4())
        self.vector_store.rows[mid] = {"data": text, "user_id": user_id, **metadata}
        self._history(mid, None, text, "ADD")
        if infer and oid in self.updates:
            target = self.updates[oid]
            old = self.vector_store.rows[target]["data"]
            self.vector_store.rows[target]["data"] = f"{old} (updated)"
            self._history(target, old, f"{old} (updated)", "UPDATE")
        return {"results": [{"id": mid, "memory": text, "event": "ADD"}]}

    def delete(self, memory_id: str) -> dict[str, str]:
        row = self.vector_store.rows.pop(memory_id, None)
        if row is None:
            raise ValueError(f"Memory with id {memory_id} not found")
        self._history(memory_id, row["data"], None, "DELETE")
        return {"message": "Memory deleted successfully!"}

    def search(self, query: str, top_k: int, filters: dict[str, Any]) -> dict[str, Any]:
        mine = [
            (m, r) for m, r in self.vector_store.rows.items() if r["user_id"] == filters["user_id"]
        ]
        rows = [
            {"id": mid, "memory": r["data"], "metadata": r, "score": 1.0 / (i + 1)}
            for i, (mid, r) in enumerate(sorted(mine)[: min(top_k, 20)])
        ]
        return {"results": rows}

    def close(self) -> None:  # the store is written when Mem0 is closed (snapshot, restore)
        self.vector_store.save()
        self.db.connection.close()


def _with_fake(cls: type[LoggedMem0], **fake: Any) -> type[LoggedMem0]:
    class Faked(cls):  # type: ignore[valid-type, misc]
        def _open(self) -> None:
            self._mem = FakeMemory(self.store_dir, **fake)
            self._wrap()

    return Faked


@pytest.fixture(scope="module")
def world():
    return generate_branch(11, "def", world_params(3))


def _system(cls: type[LoggedMem0], world, store: Path, **fake: Any) -> LoggedMem0:
    return _with_fake(cls, **fake)(
        world, store, "http://llm/v1", "m", "http://emb/v1", "e", "w11", on_error="skip"
    )


def _ev(i: int, oid: str, rev: int, op: str = "insert") -> Event:
    return Event(
        f"ev{i:06d}", T0 + timedelta(hours=i), "contacts", oid, rev, op, {"text": f"{oid} r{rev}"}
    )


# ---------------------------------------------------------------------- the event log
def test_merge_events_matches_return_and_history():
    returned = [
        {"id": "a", "memory": "new fact", "event": "ADD"},
        {"id": "b", "memory": "kept", "event": "NONE"},
    ]
    rows = [
        {"memory_id": "a", "old_memory": None, "new_memory": "new fact", "event": "ADD"},
        {"memory_id": "c", "old_memory": "old", "new_memory": "newer", "event": "UPDATE"},
    ]
    evs = merge_events(returned, rows, "extracted", "mem0")
    by = {(e["memory_id"], e["event"]): e for e in evs}
    assert by[("a", "ADD")]["via"] == ["return", "history"]
    assert by[("c", "UPDATE")]["via"] == ["history"] and by[("c", "UPDATE")]["old_memory"] == "old"
    assert by[("b", "NONE")]["via"] == ["return"] and by[("b", "NONE")]["new_memory"] == "kept"
    assert {e["scope"] for e in evs} == {"extracted"} and {e["origin"] for e in evs} == {"mem0"}


def test_logged_mem0_logs_every_add_without_changing_the_ledger(tmp_path, world):
    s = _system(LoggedMem0, world, tmp_path / "s")
    s.ingest([_ev(1, "contacts:p01", 1), _ev(2, "contacts:p02", 1)], "ingest")
    lines = read_events(tmp_path / "s" / EVENTS_FILE)
    assert [x["key"] for x in lines] == ["contacts:p01#1", "contacts:p02#1"]
    first = lines[0]
    assert (
        first["kind"] == "add"
        and first["ingested"]
        and first["ts"] == fmt_ts(T0 + timedelta(hours=1))
    )
    scopes = {e["scope"]: e for e in first["events"]}
    assert set(scopes) == {"extracted", "raw"}
    ext = scopes["extracted"]
    assert ext["event"] == "ADD" and ext["via"] == ["return", "history"]
    assert ext["new_memory"] == "fact of contacts:p01 r1"
    assert s.ledger.entries["contacts:p01#1"]["memory_ids"] == [ext["memory_id"]]
    assert s.ledger.entries["contacts:p01#1"]["raw_ids"] == [scopes["raw"]["memory_id"]]
    s.close()


def test_update_events_and_failed_units_are_logged(tmp_path, world):
    s = _system(LoggedMem0, world, tmp_path / "s")
    s.ingest([_ev(1, "contacts:p01", 1)], "ingest")
    target = s.ledger.entries["contacts:p01#1"]["memory_ids"][0]
    s.close()
    s = _system(
        LoggedMem0, world, tmp_path / "s", updates={"contacts:p02": target}, fail={"contacts:p03"}
    )
    s.ingest([_ev(2, "contacts:p02", 1), _ev(3, "contacts:p03", 1)], "maintain")
    lines = {x["key"]: x for x in read_events(tmp_path / "s" / EVENTS_FILE)}
    upd = [e for e in lines["contacts:p02#1"]["events"] if e["event"] == "UPDATE"]
    assert upd == [
        {
            "event": "UPDATE",
            "memory_id": target,
            "scope": "extracted",
            "old_memory": "fact of contacts:p01 r1",
            "new_memory": "fact of contacts:p01 r1 (updated)",
            "origin": "mem0",
            "via": ["history"],
        }
    ]
    failed = lines["contacts:p03#1"]
    assert not failed["ingested"] and "archive store down" in failed["errors"][0]
    assert [e["scope"] for e in failed["events"]] == ["extracted"]  # stored before the failure
    assert "contacts:p03#1" not in s.ledger and "contacts:p03#1" in s.failed_keys()
    s.close()


def test_det_deletes_the_earlier_revisions_memories_before_a_revision(tmp_path, world):
    s = _system(LoggedMem0, world, tmp_path / "prefix")
    s.ingest([_ev(1, "contacts:p01", 1), _ev(2, "contacts:p01", 2, "update")], "ingest")
    s.snapshot(tmp_path / "snap")
    s.close()
    det = _system(Mem0Det, world, tmp_path / "det")
    det.restore(tmp_path / "snap")
    old = [det.ledger.entries[f"contacts:p01#{r}"]["memory_ids"][0] for r in (1, 2)]
    det.ingest([_ev(3, "contacts:p02", 1), _ev(4, "contacts:p01", 3, "delete")], "maintain")
    lines = read_events(tmp_path / "det" / EVENTS_FILE)
    assert [x["key"] for x in lines[:2]] == ["contacts:p01#1", "contacts:p01#2"]  # the snapshot's
    lines = lines[2:]
    assert [(x["kind"], x["key"]) for x in lines] == [
        ("add", "contacts:p02#1"),
        ("invalidate", "contacts:p01#3"),  # before the deletion is ingested
        ("add", "contacts:p01#3"),
    ]
    inv = lines[1]
    assert inv["targets"] == ["contacts:p01#1", "contacts:p01#2"]
    assert [(e["event"], e["memory_id"], e["origin"]) for e in inv["events"]] == [
        ("DELETE", old[0], "adapter"),
        ("DELETE", old[1], "adapter"),
    ]
    assert inv["events"][0]["old_memory"] == "fact of contacts:p01 r1"
    assert all(det._mem.vector_store.get(m) is None for m in old)
    # raw archive copies stay; a second revision finds nothing left to delete
    raw = det.ledger.entries["contacts:p01#1"]["raw_ids"][0]
    assert det._mem.vector_store.get(raw) is not None
    assert det.targets(_ev(5, "contacts:p01", 4, "update"))[1] == [
        det.ledger.entries["contacts:p01#3"]["memory_ids"][0]
    ]
    det.close()
    native = _system(LoggedMem0, world, tmp_path / "native")
    native.restore(tmp_path / "snap")
    native.ingest([_ev(4, "contacts:p01", 3, "delete")], "maintain")
    assert [x["kind"] for x in read_events(tmp_path / "native" / EVENTS_FILE)] == ["add"] * 3
    native.close()


def test_det_logs_ids_already_gone(tmp_path, world):
    det = _system(Mem0Det, world, tmp_path / "det")
    det.ingest([_ev(1, "contacts:p01", 1)], "ingest")
    mid = det.ledger.entries["contacts:p01#1"]["memory_ids"][0]
    det._mem.vector_store.rows.pop(mid)  # removed behind the adapter's back
    det.ingest([_ev(2, "contacts:p01", 2, "update")], "maintain")
    inv = next(x for x in read_events(tmp_path / "det" / EVENTS_FILE) if x["kind"] == "invalidate")
    assert inv["missing"] == [mid] and inv["events"] == []
    det.close()


# ---------------------------------------------------------------------- infrastructure errors
_REQ = httpx2.Request("POST", "http://127.0.0.1:9/v1/chat/completions")  # the SDK's httpx


def _status(code: int) -> openai.APIStatusError:
    cls = {400: openai.BadRequestError, 429: openai.RateLimitError}.get(
        code, openai.InternalServerError
    )
    return cls(f"status {code}", response=httpx2.Response(code, request=_REQ), body=None)


def _wrapped(exc: Exception) -> Exception:
    """As Mem0 raises it: `raise LLMError(f"LLM extraction failed: {e}") from e`."""
    try:
        raise RuntimeError(f"LLM extraction failed: {exc}") from exc
    except RuntimeError as outer:
        return outer


def test_infrastructure_errors_are_recognised_through_the_cause_chain():
    infra = [
        openai.APITimeoutError(request=_REQ),
        openai.APIConnectionError(request=_REQ),
        _status(503),
        _status(429),
        httpx2.ReadTimeout("slow", request=_REQ),
        httpx.ReadTimeout("slow", request=httpx.Request("POST", "http://127.0.0.1:9/v1")),
        TimeoutError(),
        ConnectionResetError(),
    ]
    for exc in infra:
        assert is_infra(exc) and is_infra(_wrapped(exc)), exc
    for exc in (_status(400), ValueError("bad JSON"), RuntimeError("no memories")):
        assert not is_infra(exc) and not is_infra(_wrapped(exc)), exc


def test_retrying_backs_off_then_succeeds_or_stops():
    policy = RetryPolicy(attempts=4, first_wait_s=60, max_wait_s=100)
    waits: list[float] = []
    exhausted: list[InfrastructureError] = []
    calls = {"n": 0}

    def flaky(fail: int, exc: Exception) -> Any:
        def fn(x: int) -> int:
            calls["n"] += 1
            if calls["n"] <= fail:
                raise exc
            return x + 1

        return retrying(fn, policy, exhausted.append, waits.append)

    assert flaky(2, openai.APITimeoutError(request=_REQ))(1) == 2
    assert calls["n"] == 3 and waits == [60, 100] and not exhausted
    calls["n"], waits[:] = 0, []
    with pytest.raises(InfrastructureError, match="4 attempts failed") as info:
        flaky(9, _status(502))(1)
    assert calls["n"] == 4 and waits == [60, 100, 100] and exhausted == [info.value]
    calls["n"], waits[:] = 0, []
    with pytest.raises(openai.BadRequestError):
        flaky(9, _status(400))(1)  # not infrastructure: no retry
    assert calls["n"] == 1 and waits == []


def test_infrastructure_errors_stop_the_run_and_are_never_skipped(tmp_path, world):
    timed_out = _wrapped(InfrastructureError("6 attempts failed, last: APITimeoutError()"))
    s = _system(LoggedMem0, world, tmp_path / "s", raises={"contacts:p02": timed_out})
    with pytest.raises(InfrastructureError, match="contacts:p02#1"):
        s.ingest([_ev(1, "contacts:p01", 1), _ev(2, "contacts:p02", 1), _ev(3, "contacts:p03", 1)])
    assert "contacts:p01#1" in s.ledger and "contacts:p02#1" not in s.ledger
    assert s.failed_keys() == set()  # not recorded as a failed unit: a rerun ingests it
    last = read_events(tmp_path / "s" / EVENTS_FILE)[-1]
    assert last["key"] == "contacts:p02#1" and not last["ingested"]
    s.close()
    # every other error still follows on_error="skip", as in E5
    other = _system(
        LoggedMem0, world, tmp_path / "o", raises={"contacts:p02": _wrapped(ValueError())}
    )
    other.ingest([_ev(1, "contacts:p01", 1), _ev(2, "contacts:p02", 1), _ev(3, "contacts:p03", 1)])
    assert other.failed_keys() == {"contacts:p02#1"} and "contacts:p03#1" in other.ledger
    # a request that ran out of attempts where Mem0 swallowed the error also stops the run
    other.infra_error = InfrastructureError("embedding: 6 attempts failed")
    with pytest.raises(InfrastructureError, match="embedding"):
        other.ingest([_ev(4, "contacts:p04", 1)])
    other.close()


class _Inner:
    """openai.OpenAI stand-in: `with_options` copies, the first `fail` requests time out."""

    def __init__(self, fail: int = 0, options: dict[str, Any] | None = None):
        self.options, self.fail, self.calls = options or {}, fail, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.embeddings = SimpleNamespace(create=self._create)

    def with_options(self, **kw: Any) -> _Inner:
        self.copy = _Inner(self.fail, {**self.options, **kw})
        return self.copy

    def _create(self, **kw: Any) -> Any:
        self.calls += 1
        if self.calls <= self.fail:
            raise openai.APITimeoutError(request=_REQ)
        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        return SimpleNamespace(usage=usage, kw=kw)


def test_mem0_clients_get_a_long_timeout_and_retries(tmp_path, world, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    s = _system(LoggedMem0, world, tmp_path / "s")
    raw_llm, raw_emb = _Inner(fail=2), _Inner(fail=1)
    s._mem.llm = SimpleNamespace(client=MeteredOpenAI(raw_llm, s.usage))
    s._mem.embedding_model = SimpleNamespace(client=MeteredOpenAI(raw_emb, s.usage))
    s.retry = RetryPolicy(timeout_s=5400, attempts=3)
    s._harden()
    assert raw_llm.copy.options == {"timeout": 5400, "max_retries": 0}
    res = s._mem.llm.client.chat.completions.create(model="m", messages=[], max_tokens=99999)
    assert raw_llm.copy.calls == 3 and res.kw["max_tokens"] == s.max_output_tokens
    assert res.kw["extra_body"] == s.llm_extra_body  # metered as Mem0System._open does
    s._mem.embedding_model.client.embeddings.create(model="e", input=["x"])
    assert raw_emb.copy.calls == 2 and s.infra_error is None
    assert s.usage.snapshot()["llm_calls"] == 1 and s.usage.snapshot()["embed_calls"] == 1
    s.close()


# ---------------------------------------------------------------------- serving
def test_serve_is_the_native_serve_and_the_view_leads_definition_items(tmp_path, world):
    log = SourceLog(world.events)
    s = _system(LoggedMem0, world, tmp_path / "s")
    s.ingest([_ev(i, f"contacts:p{i:02d}", 1) for i in range(1, 30)], "ingest")
    t = world.fork + timedelta(weeks=1)
    other = make_probe(world, log, world.slot("u.employer"), t)
    assert s.serve(other) == NativeSystem.serve(s, other)
    rep = Mem0Rep(s, xd.ZeroDm(world, log, RecordingProgramLLM(PROGRAMS)))
    assert rep.serve(other) == s.serve(other) and rep.views == []
    probe = make_probe(world, log, world.slot("k.close_friend"), t)
    res = rep.serve(probe)
    block = rep.view.view(probe)[0]
    assert block is not None and res.blocks[0] == block
    hits = s.hits(probe)
    docs = [Doc(h.hit_id, h.text, h.ts) for h in hits]
    left = READER_BUDGET - (approx_tokens(block) + 8)
    assert res.blocks[1:] == [text for _, text in pack(docs, left, READER_DOC_CAP)]
    assert rep.views[0]["probe_id"] == probe.probe_id and rep.views[0]["view"]["ok"]
    s.close()


# ---------------------------------------------------------------------- detection
FORK = datetime(2026, 3, 1)


def _line(key: str, ts: datetime, events: list[tuple[str, str]], kind: str = "add") -> dict:
    oid = key.rpartition("#")[0]
    evs = [
        {"event": e, "memory_id": m, "scope": "raw" if m.startswith("raw") else "extracted"}
        for e, m in events
    ]
    return {"kind": kind, "key": key, "object_id": oid, "ts": fmt_ts(ts), "events": evs}


def test_detection_window_links_and_scope():
    ledger = {"contacts:p01#1": {"memory_ids": ["m1"]}, "email:e1#1": {"memory_ids": ["m2"]}}
    events = [
        _line("contacts:p01#2", FORK + timedelta(days=2), [("UPDATE", "m1")]),
        _line("note:n1#1", FORK + timedelta(days=3), [("ADD", "m3")]),
        _line("note:n1#2", FORK + timedelta(days=9), [("DELETE", "m3")]),
        _line("email:e1#2", FORK - timedelta(days=1), [("DELETE", "m2")]),  # before the fork
        _line("chat:c1#1", FORK + timedelta(days=1), [("DELETE", "raw1")]),  # archive scope
    ]
    log = xa.StoreLog(ledger, events)
    week = FORK + timedelta(weeks=1)
    assert log.detected({"contacts:p01"}, FORK, week)  # via the ledger
    assert not log.detected({"contacts:p01"}, FORK, FORK + timedelta(days=1))  # not yet
    assert log.detected({"contacts:p01"}, FORK, FORK + timedelta(days=2))  # at the checkpoint
    assert not log.detected({"note:n1"}, FORK, week)  # an ADD is not detection
    assert log.detected({"note:n1"}, FORK, FORK + timedelta(weeks=2))  # linked by its ADD event
    assert not log.detected({"email:e1"}, FORK, week)  # mutation before the fork
    assert not log.detected({"chat:c1"}, FORK, week)
    assert log.counts(FORK) == {"None/DELETE": 1, "None/UPDATE": 1}


def test_losses_and_dominant_failure():
    assert xa.losses(1.0, float("nan"), 0.3) == {"detection": pytest.approx(0.7), "repair": 0.0}
    assert xa.losses(0.0, 0.6, float("nan")) == {"detection": 0.0, "repair": pytest.approx(0.4)}
    assert xa.dominant(0.5, 0.2, 0.9) == {
        "loss": {"detection": pytest.approx(0.05), "repair": pytest.approx(0.4)},
        "dominant": "repair",
        "judged": True,
    }
    small = xa.dominant(0.2, 0.99, 0.9)  # losses 2 pp and 0.8 pp
    assert small["dominant"] == "detection" and not small["judged"]


def _cell(dom: str, judged: bool, det: float, rep: float) -> dict:
    return {
        "diagnosis": {"dominant": dom, "judged": judged},
        "effects": {"mem0_det": {"est": det}, "mem0_rep": {"est": rep}},
    }


def test_x1_verdict():
    cells = {
        "src/definition": _cell("detection", True, 0.10, 0.30),
        "src/other": _cell("detection", True, 0.20, 0.00),
        "def/definition": _cell("detection", True, 0.05, 0.01),
        "def/other": _cell("repair", True, 0.00, 0.02),
        "orphan/definition": _cell("detection", False, 0.50, 0.00),  # not judged
    }
    v = xa.x1(cells)
    assert v["n_judged"] == 4 and v["n_favour_target"] == 3 and v["judged"] and not v["holds"]
    cells["src/definition"] = _cell("detection", True, 0.40, 0.30)
    assert xa.x1(cells)["holds"]
    few = {k: cells[k] for k in ("src/other", "def/other")}
    assert xa.x1(few) == {**xa.x1(few), "judged": False, "holds": None}


# ---------------------------------------------------------------------- the analysis end to end
def _run_dir(tmp_path: Path) -> Path:
    """Three worlds, branch src, one definition item and one other item, checkpoints k = 1, 2.
    The definition item changes at k = 1 and 2, the other at k = 2. mem0 detects nothing; the det
    arm deletes the other item's memory at the fork + 1 day; rep answers the definition item."""
    d = tmp_path / "x_mem0_run"
    probes, answers, deps, results = [], [], [], []
    fork = FORK
    for w in ("w1", "w2", "w3"):
        results.append({"world": {"world_id": w}, "prefix": {"source": "e5"}, "branches": {}})
        for f, dep in (("k.close_friend", "chat:c1"), ("u.employer", "contacts:p01")):
            deps.append(
                {
                    "world_id": w,
                    "branch": "src",
                    "fact_id": f,
                    "fork": fmt_ts(fork),
                    "deps": [[dep, 1]],
                }
            )
            for k in (0, 1, 2):
                changed = (f.startswith("k.") and k > 0) or k == 2
                p = {
                    "world_id": w,
                    "branch": "src",
                    "fact_id": f,
                    "vtype": "str",
                    "as_of": fmt_ts(fork + timedelta(weeks=k)),
                    "gold_value": "new" if changed else "old",
                    "gold_status": "answer",
                    "deps": [[dep, 2 if changed else 1]],
                    "tags": {"k": k},
                }
                probes.append(p)
                if not k:
                    continue
                for arm in xa.ARMS:
                    right = (arm == "mem0_rep" and f.startswith("k.")) or (
                        arm == "mem0_det" and f == "u.employer" and w != "w3"
                    )
                    answers.append(
                        {
                            "system": arm,
                            "world_id": w,
                            "branch": "src",
                            "fact_id": f,
                            "contract_valid": right,
                            "tags": {"k": k},
                        }
                    )
        store = d / "stores" / f"{w}_mem0"
        for name in ("src", "src_det"):
            (store / name).mkdir(parents=True)
            (store / name / "ledger.jsonl").write_text(
                json.dumps({"key": "contacts:p01#1", "memory_ids": ["m1"]}) + "\n"
            )
        inv = _line("contacts:p01#2", fork + timedelta(days=1), [("DELETE", "m1")], "invalidate")
        (store / "src_det" / EVENTS_FILE).write_text(json.dumps(inv) + "\n")
        (store / "src" / EVENTS_FILE).write_text(
            json.dumps(_line("contacts:p01#2", fork + timedelta(days=1), [("ADD", "m9")])) + "\n"
        )
    d.mkdir(exist_ok=True)
    for name, rows in (
        ("probes", probes),
        ("answers", answers),
        ("fork_deps", deps),
        ("world_results", results),
    ):
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return d


def test_analysis_on_synthetic_logs(tmp_path):
    d = _run_dir(tmp_path)
    res = xa.analyse([d], draws=200)
    assert res["mutations_after_fork"] == {"mem0_det": {"None/DELETE": 3}}
    assert res["support"].startswith("the mem0 arm logged no UPDATE or DELETE")
    defn, other = res["cells"]["src/definition"], res["cells"]["src/other"]
    nat = defn["arms"]["mem0"]
    assert nat["n"] == 6 and nat["detected"] == 0
    assert nat["m"]["est"] == 1.0 and nat["Update"]["est"] == 0.0 and nat["a"]["est"] == 0.0
    assert defn["diagnosis"]["dominant"] == "detection" and defn["diagnosis"]["judged"]
    assert defn["effects"]["mem0_rep"]["est"] == 1.0 and defn["effects"]["mem0_det"]["est"] == 0.0
    det = other["arms"]["mem0_det"]
    assert det["n"] == 3 and det["detected"] == 3 and det["m"]["est"] == 0.0
    assert det["r"]["est"] == pytest.approx(2 / 3)
    assert other["effects"]["mem0_det"]["est"] == pytest.approx(2 / 3)
    assert res["X1"]["n_judged"] == 2 and not res["X1"]["judged"]
    assert json.loads((d / "analysis" / "x_mem0.json").read_text())["X1"] == res["X1"]


# ---------------------------------------------------------------------- prefix reuse
def _e5_run(root: Path, world_id: str, describe: dict, events_sha: str) -> Path:
    d = root / "e5_mem0_s11_20260929T024709Z"
    snap = d / "stores" / f"{world_id}_mem0" / "snapshot"
    snap.mkdir(parents=True)
    (snap / "memshift_native.json").write_text(json.dumps(describe))
    (snap / "ledger.jsonl").write_text("")
    serving = {"model": "Qwen/Qwen3.5-9B", "revision": "c2", "image_id": "sha256:d0"}
    emb = {"model": "Qwen/Qwen3-Embedding-0.6B", "revision": "97", "image_id": "sha256:d0"}
    manifest = {
        "worlds": [{"world_id": world_id, "events_sha256": events_sha}],
        "params": {"on_error": "skip", "world_params": world_params(3)},
        "models": {"native_llm": serving, "embedding": emb},
    }
    env = {
        "models": {
            "native_llm": {"max_model_len": 32768, "extra_args": "--language-model-only"},
            "embedding": {"max_model_len": 8192, "extra_args": "--runner pooling"},
        }
    }
    (d / "manifest.json").write_text(json.dumps(manifest))
    (d / "env.json").write_text(json.dumps(env))
    return d


def test_prefix_checks_and_decision(tmp_path, world, monkeypatch):
    natural = generate_branch(11, "natural", world_params(3))
    s = _system(LoggedMem0, natural, tmp_path / "x" / "prefix")
    sha = xd.world_record(natural)["events_sha256"]
    e5 = _e5_run(tmp_path / "runs", natural.world_id, s.describe(), sha)
    assert xd.e5_run_dir(11, tmp_path / "runs") == e5
    assert xd.e5_run_dir(12, tmp_path / "runs") is None
    monkeypatch.setattr(xd, "mem0_version", lambda: "2.2.1")
    # as `scripts/serve_model.sh` writes them (logs/serving/*.json): the model is `served_model`
    serving = {
        "Qwen/Qwen3.5-9B": {
            "served_model": "Qwen/Qwen3.5-9B", "revision": "c2", "image_id": "sha256:d0",
            "max_model_len": 32768, "extra_args": "--language-model-only", "port": 8001,
            "gpu_memory_utilization": 0.8,
        },
        "Qwen/Qwen3-Embedding-0.6B": {
            "served_model": "Qwen/Qwen3-Embedding-0.6B", "revision": "97", "image_id": "sha256:d0",
            "max_model_len": 8192, "extra_args": "--runner pooling",
        },
    }  # fmt: skip
    # the fake system reports model names "m" and "e": only those two settings differ
    checks = xd.prefix_checks(e5, s, natural, world_params(3), serving)
    assert {k for k, c in checks.items() if not c["ok"]} == {"mem0_config"}
    real = _system(LoggedMem0, natural, tmp_path / "y" / "prefix")
    real.llm_model, real.embed_model = "Qwen/Qwen3.5-9B", "Qwen/Qwen3-Embedding-0.6B"
    e5b = _e5_run(tmp_path / "runs2", natural.world_id, real.describe(), sha)
    good = xd.prefix_checks(e5b, real, natural, world_params(3), serving)
    assert all(c["ok"] for c in good.values()), {k: c for k, c in good.items() if not c["ok"]}
    assert xd.prefix_decision("auto", e5b, good) == "e5"
    assert xd.prefix_decision("rebuild", e5b, good) == "rebuilt"
    stale = {**serving, "Qwen/Qwen3.5-9B": {**serving["Qwen/Qwen3.5-9B"], "revision": "other"}}
    bad = xd.prefix_checks(e5b, real, natural, world_params(3), stale)
    assert not bad["llm"]["ok"] and xd.prefix_decision("auto", e5b, bad) == "rebuilt"
    with pytest.raises(SystemExit, match="llm"):
        xd.prefix_decision("resume", e5b, bad)
    assert not xd.prefix_checks(e5b, real, natural, world_params(3), {})["embedder"]["ok"]
    llm = serving["Qwen/Qwen3.5-9B"]
    longer = {**serving, "Qwen/Qwen3.5-9B": {**llm, "max_model_len": 40960}}
    assert all(
        c["ok"] for c in xd.prefix_checks(e5b, real, natural, world_params(3), longer).values()
    )
    shorter = {**serving, "Qwen/Qwen3.5-9B": {**llm, "max_model_len": 16384}}
    short = xd.prefix_checks(e5b, real, natural, world_params(3), shorter)
    assert [k for k, c in short.items() if not c["ok"]] == ["llm_context"]
    other_args = {**serving, "Qwen/Qwen3.5-9B": {**llm, "extra_args": "--enable-thinking"}}
    assert not xd.prefix_checks(e5b, real, natural, world_params(3), other_args)["llm"]["ok"]
    renamed = {**serving, "Qwen/Qwen3.5-9B": {**llm, "served_model": "Qwen/Other"}}
    assert not xd.prefix_checks(e5b, real, natural, world_params(3), renamed)["llm"]["ok"]
    assert xd.prefix_decision("auto", None, None) == "rebuilt"
    s.close()
    real.close()


def test_driver_and_analysis_on_a_stand_in_mem0(tmp_path, monkeypatch):
    """One world through `run_world` (rebuilt prefix, three arms, two branches) with the
    stand-in Memory and an abstaining reader, then the analysis on its output."""
    from memshift.experiments.common import EMBED_MODEL, MAIN_MODEL

    monkeypatch.setattr(xd, "LoggedMem0", _with_fake(LoggedMem0))
    monkeypatch.setattr(xd, "Mem0Det", _with_fake(Mem0Det))
    monkeypatch.setattr(xd, "endpoints", lambda: {MAIN_MODEL: "http://x/v1", EMBED_MODEL: "e"})
    monkeypatch.setattr(xd, "client", lambda *a, **k: RecordingProgramLLM(PROGRAMS))
    monkeypatch.setattr(xd, "uncached_programs", lambda *a: [])
    monkeypatch.setattr(xd, "CachedPrograms", lambda llm: llm)
    run = tmp_path / "x_mem0_run"
    run.mkdir()
    out = xd.run_world(11, run, world_params(3), branches=("src", "orphan"))
    (run / "world_results.jsonl").write_text(json.dumps(out, default=str) + "\n")
    assert out["prefix"]["source"] == "rebuilt" and out["prefix"]["checks"] is None
    assert set(out["branches"]) == {"src", "orphan"}
    for res in out["branches"].values():
        assert res["mem0"]["units_without_events"] == []
        assert not any("UPDATE" in k or "DELETE" in k for k in res["mem0"]["events"])
        assert res["mem0_rep"]["views"] > 0
    assert any(
        "invalidate/adapter/extracted/DELETE" in k
        for k in out["branches"]["src"]["mem0_det"]["events"]
    )
    answers = [json.loads(x) for x in (run / "answers.jsonl").read_text().splitlines()]
    assert {a["system"] for a in answers} == set(xd.ARMS)
    deps = [json.loads(x) for x in (run / "fork_deps.jsonl").read_text().splitlines()]
    assert {d["branch"] for d in deps} == {"src", "orphan"}
    res = xa.analyse([run], draws=50)
    assert res["support"].startswith("the mem0 arm logged no UPDATE or DELETE")
    for cell in res["cells"].values():
        assert cell["arms"]["mem0"]["detected"] == 0
        assert cell["arms"]["mem0_rep"]["n"] == cell["arms"]["mem0"]["n"]
    assert res["mutations_after_fork"]["mem0_det"]
    assert res["prefix_sources"] == {"w0011": "rebuilt"}


def test_e5_costs_subtract_searches(tmp_path):
    d = tmp_path / "e5_mem0_s1_x"
    store = d / "stores" / "w1_mem0"
    usage = {"llm_calls": 10, "prompt_tokens": 1000, "completion_tokens": 50}
    snap = {"usage": {**usage, "embed_calls": 30, "embed_tokens": 500}, "cost": {"serve_tokens": 0}}
    grown = {"llm_calls": 14, "prompt_tokens": 1800, "completion_tokens": 70}
    branch = {
        "usage": {**grown, "embed_calls": 46, "embed_tokens": 900},
        "cost": {"serve_tokens": 100},
    }
    for name, st in (("snapshot", snap), ("src", branch)):
        (store / name).mkdir(parents=True)
        (store / name / "state.json").write_text(json.dumps(st))
    ti = {"llm_calls": 1, "prompt_tokens": 100, "completion_tokens": 5, "embed_calls": 3,
          "embed_tokens": 50, "wall_s": 9.0}  # fmt: skip
    res = {
        "world": {"world_id": "w1"},
        "t_i": ti,
        "branches": {"src": {"stats": {"ingested": 4, "ingest_wall_s": 40.0}}},
    }
    (d / "world_results.jsonl").write_text(json.dumps(res) + "\n")
    probes = [{"branch": "src", "tags": {"k": 1}}] * 8 + [{"branch": "natural", "tags": {"k": 0}}]
    (d / "probes.jsonl").write_text("".join(json.dumps(p) + "\n" for p in probes))
    ans = {"system": "mem0", "status": "answer", "prompt_tokens": 4000, "completion_tokens": 20}
    (d / "answers.jsonl").write_text(json.dumps(ans) + "\n")
    c = xd.e5_costs(d)
    assert c["branch_event"]["src"] == {
        "llm_calls": 1.0,
        "prompt_tokens": 200.0,
        "completion_tokens": 5.0,
        "embed_calls": 2.0,  # (16 - 8 searches) / 4
        "embed_tokens": 75.0,  # (400 - 100 served) / 4
        "wall_s": 10.0,
    }
    assert c["search"] == {"embed_calls": 1.0, "embed_tokens": 12.5}
    assert c["reader_answer"] == {"prompt_tokens": 4000.0, "completion_tokens": 20.0}


def test_unit_keys_of_events():
    assert event_key(_ev(1, "contacts:p01", 3, "delete")) == "contacts:p01#3"


def test_the_view_uses_cached_d1s_programs_only(tmp_path, world):
    llm = LLMClient("http://127.0.0.1:9/v1", "Qwen/Qwen3.5-9B", tmp_path / "cache.sqlite")
    log = SourceLog(world.events)
    t = world.fork + timedelta(weeks=1)
    defs = [s for s in world.slots if s.kappa_id]
    assert len(xd.uncached_programs(llm, world, [t])) == len(defs)
    slot = world.slot("k.close_friend")
    reply = {
        "text": json.dumps({"code": PROGRAMS["(close_friend v1)"]}),
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "latency_s": 0.0,
        "cached": False,
        "finish_reason": "stop",
    }
    llm.cache.put(xd.program_key(llm, world, slot, t), llm.model, reply)
    missing = xd.uncached_programs(llm, world, [t])
    assert len(missing) == len(defs) - 1 and not any(m.startswith(slot.slot_id) for m in missing)
    view = xd.ZeroDm(world, log, xd.CachedPrograms(llm))
    block, diag = view.view(make_probe(world, log, slot, t))
    assert block is not None and diag["view"]["ok"]  # a cache hit: no server is contacted
    other = next(s for s in defs if s.slot_id != slot.slot_id)
    with pytest.raises(SystemExit, match="not in the response cache"):
        view.view(make_probe(world, log, other, t))


# ---------------------------------------------------------------------- the real library
def _stub_models(system: LoggedMem0) -> None:
    """Offline stand-ins for Mem0's embedder and extraction LLM (one fact per record)."""
    mem = system._mem

    def embed(text: str, memory_action: str | None = None) -> list[float]:
        h = sum(text.encode()) % 997
        return [float((h + i) % 7) + 1.0 for i in range(1024)]

    def extract(messages: list[dict[str, str]], response_format: Any = None, **kw: Any) -> str:
        head = next(x for x in messages[-1]["content"].splitlines() if " record " in x)
        return json.dumps({"memory": [{"text": f"fact: {head[:90]}"}]})

    mem.embedding_model.embed = embed
    mem.embedding_model.embed_batch = lambda texts, memory_action=None: [embed(t) for t in texts]
    mem.llm.generate_response = extract


def test_the_log_on_mem0ai(tmp_path, world, monkeypatch):
    """mem0ai 2.2.1 itself (the native extra): ADD seen in the return value and the history table,
    the adapter's DELETE in the history table with the old text."""
    monkeypatch.setenv("MEM0_DIR", str(tmp_path / "mem0dir"))  # read by mem0 at import
    monkeypatch.setenv("MEM0_TELEMETRY", "False")
    pytest.importorskip("mem0")

    def stubbed(cls: type[LoggedMem0]) -> type[LoggedMem0]:
        class Stubbed(cls):  # type: ignore[valid-type, misc]
            def _open(self) -> None:
                super()._open()
                _stub_models(self)

        return Stubbed

    url = "http://127.0.0.1:9/v1"  # never contacted
    s = stubbed(LoggedMem0)(world, tmp_path / "p", url, "m", url, "e", "w11", on_error="raise")
    s.ingest([_ev(1, "contacts:p01", 1), _ev(2, "contacts:p02", 1)], "ingest")
    s.snapshot(tmp_path / "snap")
    s.close()
    det = stubbed(Mem0Det)(world, tmp_path / "d", url, "m", url, "e", "w11", on_error="raise")
    det.restore(tmp_path / "snap")
    det.ingest([_ev(3, "contacts:p01", 2, "update")], "maintain")
    lines = read_events(tmp_path / "d" / EVENTS_FILE)
    assert [(x["kind"], x["key"]) for x in lines] == [
        ("add", "contacts:p01#1"),
        ("add", "contacts:p02#1"),
        ("invalidate", "contacts:p01#2"),
        ("add", "contacts:p01#2"),
    ]
    first = lines[0]["events"]
    assert [(e["event"], e["scope"], e["via"]) for e in first] == [
        ("ADD", "extracted", ["return", "history"]),
        ("ADD", "raw", ["return", "history"]),
    ]
    gone = lines[2]["events"]
    assert [(e["event"], e["memory_id"]) for e in gone] == [
        ("DELETE", det.ledger.entries["contacts:p01#1"]["memory_ids"][0])
    ]
    assert gone[0]["old_memory"] == first[0]["new_memory"] and gone[0]["via"] == ["history"]
    det.close()

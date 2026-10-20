"""Mem0 for the Mem0 interventions: the E5 adapter with an event log, and the two
interventions.

`LoggedMem0` (arm `mem0`) is `native.Mem0System` unchanged (same configuration, same two `add`
calls per unit, same ledger) plus an event log. Every `Memory.add` is wrapped: its return value
and the rows it wrote to Mem0's history table (`history.db`, the table `Memory.history` reads) are
kept, and after each unit one line goes to `store_dir/events.jsonl` with the unit's key, record
time and op, and one entry per event: ADD / UPDATE / DELETE / NONE, memory id, old and new text,
scope (`extracted`: the served `add(infer=True)`; `raw`: the archive copy) and where it was seen
(`via`: `return` and / or `history`). Logging reads only; Mem0's behaviour is not changed.

What mem0ai 2.2.1 can emit: its `add(infer=True)` is the "V3 phased batch pipeline"
(mem0/memory/main.py:918-1220). One extraction call whose prompt allows only ADD
(`ADDITIVE_EXTRACTION_PROMPT`, mem0/configs/prompts.py:468-472), an md5 dedup that silently drops
a fact whose text equals one of the 10 nearest memories (:1007-1026), and ADD history rows only
(:1078-1098). UPDATE and DELETE rows are written only by the public `update()` / `delete()` /
`delete_all()` (:1879, :1896, :1948), which the native arm never calls, and no NONE event exists
(the dropped duplicates are neither returned nor recorded). The log still records every event
kind, so a different pipeline would show up, but under 2.2.1 the native arm logs ADD only.

`Mem0Det` (arm `mem0_det`): before a unit that revises (op `update`) or deletes (op `delete`) a
record is ingested, the memories that the record's earlier revisions created or updated, by the
ledger and the event log, are deleted through Mem0's own `Memory.delete` (one line of kind
`invalidate`, origin `adapter`). This is a source-keyed invalidation; it reads no gold. Raw archive
copies are not touched (they are never searched).

`Mem0Rep` (arm `mem0_rep`) serves from the store of the `mem0` arm: for a definition item the
evidence starts with the value of the item's model-written program (the cached `D1s` request, run
on the live records at the probe's as-of time), rendered as in `baselines.ZeroDm`, and the Mem0
hits fill the budget it leaves; every other probe is served exactly as by Mem0. Both arms share one
search per probe and checkpoint, so their contexts differ only by the view block.

Infrastructure errors are never skipped. Mem0 turns any error of its extraction call into
`LLMError` (main.py:957-971), which E5's `on_error="skip"` would record as a failed unit and pass
over, and it silently drops a fact whose embedding call fails (:995-1005). Here every request
Mem0 sends to the LLM or the embedder gets a long timeout (`RetryPolicy.timeout_s`, the SDK's own
retries off) and is retried with exponential backoff while it fails with a timeout, a connection
error, a 5xx or a 429 (`is_infra`, which follows the exception's cause chain). After
`RetryPolicy.attempts` failed attempts the run stops with `InfrastructureError`, also when Mem0
swallowed the last failure. `on_error="skip"` still applies to every other error, the kinds E5
recorded and skipped (model output the library rejects, a request the server refuses); E5's Mem0
runs skipped none.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from collections.abc import Callable
from contextlib import nullcontext, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from memshift.adapters.baselines import ZeroDm
from memshift.adapters.native import Hit, Mem0System, MeteredOpenAI, event_key, parse_key
from memshift.gen.model import Event, fmt_ts
from memshift.holdmem.system import READER_BUDGET, READER_DOC_CAP, ServeResult
from memshift.probes import Probe
from memshift.retrieval import Doc, pack
from memshift.textutil import approx_tokens

EVENTS_FILE = "events.jsonl"
MEM0_VERSION = "2.2.1"  # the pin of pyproject.toml at E5's code commit (ef70d0f) and now
EVENT_KINDS = ("ADD", "UPDATE", "DELETE", "NONE")
MUTATIONS = ("UPDATE", "DELETE")
_LOG = logging.getLogger(__name__)


# ---------------------------------------------------------------------- infrastructure errors
@dataclass(frozen=True)
class RetryPolicy:
    """Per request: `timeout_s` to wait for the server, up to `attempts` attempts while it fails
    with an infrastructure error, waiting `first_wait_s` * 2^k (at most `max_wait_s`) between."""

    timeout_s: float = 7200.0
    attempts: int = 6
    first_wait_s: float = 60.0
    max_wait_s: float = 1800.0

    def wait(self, attempt: int) -> float:
        return float(min(self.max_wait_s, self.first_wait_s * 2 ** (attempt - 1)))


class InfrastructureError(RuntimeError):
    """A model server stayed unreachable, timed out or failed (5xx / 429) for every attempt."""


def _transport_errors() -> tuple[type[BaseException], ...]:
    """Transport errors of httpx and of httpx2, the fork the openai SDK (3.x) sends through."""
    import importlib

    out: list[type[BaseException]] = [httpx.TransportError]
    with suppress(ImportError):
        out.append(importlib.import_module("httpx2").TransportError)
    return tuple(out)


def _infra_one(exc: BaseException) -> bool:
    kinds: tuple[type[BaseException], ...] = (
        InfrastructureError,
        TimeoutError,
        ConnectionError,
        *_transport_errors(),
    )
    if isinstance(exc, kinds):
        return True
    import openai

    if isinstance(exc, openai.APIConnectionError):  # includes APITimeoutError
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(exc, openai.APIStatusError) and (status == 429 or (status or 0) >= 500)


def is_infra(exc: BaseException) -> bool:
    """Whether an error, or any error in its cause / context chain, is an infrastructure error
    (Mem0 wraps the client's error: `raise LLMError(...) from e`)."""
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        if _infra_one(cur):
            return True
        seen.add(id(cur))
        cur = cur.__cause__ or cur.__context__
    return False


def retrying(
    fn: Callable[..., Any],
    policy: RetryPolicy,
    on_exhausted: Callable[[InfrastructureError], None],
    sleep: Callable[[float], None] | None = None,
) -> Callable[..., Any]:
    """`fn` retried on infrastructure errors; other errors pass through at once. `sleep` defaults
    to `time.sleep`, looked up at each wait."""

    def call(*a: Any, **kw: Any) -> Any:
        for attempt in range(1, policy.attempts + 1):
            try:
                return fn(*a, **kw)
            except Exception as exc:
                if not is_infra(exc):
                    raise
                if attempt == policy.attempts:
                    err = InfrastructureError(f"{attempt} attempts failed, last: {exc!r}"[:1000])
                    on_exhausted(err)
                    raise err from exc
                wait = policy.wait(attempt)
                _LOG.warning("attempt %d failed (%r); retrying in %.0f s", attempt, exc, wait)
                (sleep or time.sleep)(wait)
        raise AssertionError("unreachable")

    return call


def mem0_version() -> str | None:
    """The installed mem0ai version (None when the native extra is not installed)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("mem0ai")
    except PackageNotFoundError:
        return None


def read_events(path: Path) -> list[dict[str, Any]]:
    """The lines of an event log (a line cut short by a crash is skipped)."""
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def merge_events(
    returned: list[dict[str, Any]], rows: list[dict[str, Any]], scope: str, origin: str
) -> list[dict[str, Any]]:
    """One `add` / `delete` call's events: its return value (`results` items with `id`, `memory`,
    `event` and, in older pipelines, `previous_memory`) merged with the history rows it wrote,
    matched on (memory id, event). History supplies the old and new text where it has them."""
    out: list[dict[str, Any]] = []
    pos: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        mid, kind = str(r.get("memory_id")), str(r.get("event"))
        ev: dict[str, Any] = {
            "event": kind,
            "memory_id": mid,
            "scope": scope,
            "old_memory": r.get("old_memory"),
            "new_memory": r.get("new_memory"),
            "origin": origin,
            "via": ["history"],
        }
        pos.setdefault((mid, kind), ev)
        out.append(ev)
    for r in returned:
        if not isinstance(r, dict):
            continue
        mid, kind = str(r.get("id")), str(r.get("event") or "NONE")
        if (mid, kind) in pos:
            hit = pos[(mid, kind)]
            if "return" not in hit["via"]:
                hit["via"] = ["return", "history"]
                if hit["new_memory"] is None and kind != "DELETE":
                    hit["new_memory"] = r.get("memory")
            continue
        out.append(
            {
                "event": kind,
                "memory_id": mid,
                "scope": scope,
                "old_memory": r.get("previous_memory"),
                "new_memory": r.get("memory"),
                "origin": origin,
                "via": ["return"],
            }
        )
    return out


class LoggedMem0(Mem0System):
    """`Mem0System` plus the event log (arm `mem0`). Same store kind ("mem0"), so it restores E5's
    prefix snapshots; a restored store keeps whatever log the snapshot had (E5's had none)."""

    arm = "mem0"

    def __init__(self, *args: Any, retry: RetryPolicy | None = None, **kw: Any):
        self._calls: list[dict[str, Any]] = []
        self._hits: dict[str, list[Hit]] = {}
        self.retry = retry or RetryPolicy()
        self.infra_error: InfrastructureError | None = None
        super().__init__(*args, **kw)

    # -------------------------------------------------------------- state
    def _load_state(self) -> None:
        super()._load_state()
        self._hits = {}
        self.events_path = self.store_dir / EVENTS_FILE
        self.made: dict[str, set[str]] = defaultdict(set)  # unit key -> extracted ids added/updated
        self.deleted: set[str] = set()
        for line in read_events(self.events_path):
            self._index(line)

    def _index(self, line: dict[str, Any]) -> None:
        for ev in line.get("events") or []:
            if ev.get("scope") != "extracted":
                continue
            if ev.get("event") in ("ADD", "UPDATE"):
                self.made[str(line.get("key"))].add(str(ev["memory_id"]))
            elif ev.get("event") == "DELETE":
                self.deleted.add(str(ev["memory_id"]))

    def _write(self, line: dict[str, Any]) -> None:
        line = {**line, "arm": self.arm, "logged_utc": datetime.now(UTC).isoformat()}
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._index(line)

    # -------------------------------------------------------------- Mem0 hooks
    def _open(self) -> None:
        super()._open()
        self._harden()
        self._wrap()

    def _harden(self) -> None:
        """Re-meter Mem0's two OpenAI clients over copies with a long timeout and no SDK retries,
        and retry their requests on infrastructure errors (`retrying`)."""
        mem = self._mem
        for holder, llm in ((mem.llm, True), (mem.embedding_model, False)):
            inner = holder.client._inner.with_options(timeout=self.retry.timeout_s, max_retries=0)
            client = MeteredOpenAI(  # as Mem0System._open meters them
                inner,
                self.usage,
                extra_body=self.llm_extra_body if llm else None,
                max_tokens_cap=self.max_output_tokens if llm else None,
            )
            comps = client.chat.completions
            comps.create = retrying(comps.create, self.retry, self._exhausted)
            emb = client.embeddings
            emb.create = retrying(emb.create, self.retry, self._exhausted)
            holder.client = client

    def _exhausted(self, err: InfrastructureError) -> None:
        self.infra_error = err

    def _fail(self, key: str, exc: Exception) -> None:
        """Infrastructure errors stop the run; every other error follows `on_error`."""
        if is_infra(exc):
            raise InfrastructureError(f"{key}: {exc!r}"[:1000]) from exc
        super()._fail(key, exc)

    def _check_infra(self) -> None:
        """Stop when a request ran out of attempts, also where Mem0 swallowed the error."""
        if self.infra_error is not None:
            raise self.infra_error

    def _wrap(self) -> None:
        """Shadow `add` and `delete` on this Memory instance with recording wrappers."""
        mem = self._mem
        real_add, real_delete = mem.add, mem.delete

        def add(*a: Any, **kw: Any) -> Any:
            return self._record("add", real_add, kw.get("user_id"), *a, **kw)

        def delete(memory_id: str) -> Any:
            return self._record("delete", real_delete, None, memory_id)

        mem.add, mem.delete = add, delete

    def _record(self, op: str, fn: Any, scope_user: str | None, /, *a: Any, **kw: Any) -> Any:
        mark = self._history_mark()
        call: dict[str, Any] = {"op": op, "user_id": scope_user, "result": None, "error": None}
        try:
            call["result"] = fn(*a, **kw)
            return call["result"]
        except Exception as exc:
            call["error"] = repr(exc)[:500]
            raise
        finally:
            call["rows"] = self._history_since(mark)
            self._calls.append(call)

    def _history(self) -> tuple[Any, Any]:
        db = self._mem.db
        return db.connection, getattr(db, "_lock", None) or nullcontext()

    def _history_mark(self) -> int:
        conn, lock = self._history()
        with lock:
            row = conn.execute("SELECT COALESCE(MAX(rowid), 0) FROM history").fetchone()
        return int(row[0])

    def _history_since(self, mark: int) -> list[dict[str, Any]]:
        conn, lock = self._history()
        with lock:
            cur = conn.execute(
                "SELECT memory_id, old_memory, new_memory, event FROM history "
                "WHERE rowid > ? ORDER BY rowid",
                (mark,),
            )
            rows = cur.fetchall()
        return [
            {"memory_id": r[0], "old_memory": r[1], "new_memory": r[2], "event": r[3]} for r in rows
        ]

    def _scope(self, user_id: str | None) -> str:
        return "raw" if user_id == self.raw_user_id else "extracted"

    def _call_events(self, origin: str) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for c in self._calls:
            res = c["result"]
            returned = res.get("results", []) if isinstance(res, dict) and c["op"] == "add" else []
            out += merge_events(returned, c["rows"], self._scope(c["user_id"]), origin)
        return out

    # -------------------------------------------------------------- ingestion
    def _ingest_events(self, events: list[Event]) -> None:
        self.clear_hits()  # a search made before new units may no longer hold
        for e in events:
            self._before_unit(e)
            self._calls = []
            try:
                super()._ingest_events([e])  # the E5 path: extraction add, raw add, ledger
            finally:
                errors = [c["error"] for c in self._calls if c["error"]]
                self._write(
                    {
                        "kind": "add",
                        **self._unit(e),
                        "ingested": event_key(e) in self.ledger,
                        "events": self._call_events("mem0"),
                        **({"errors": errors} if errors else {}),
                    }
                )
                self._calls = []
            self._check_infra()

    @staticmethod
    def _unit(e: Event) -> dict[str, Any]:
        return {
            "key": event_key(e),
            "event_id": e.event_id,
            "object_id": e.object_id,
            "revision": e.revision,
            "op": e.op,
            "ts": fmt_ts(e.ts),
        }

    def _before_unit(self, e: Event) -> None:
        """Hook run before each unit is added (the native arm does nothing)."""

    # -------------------------------------------------------------- serving
    def hits(self, probe: Probe) -> list[Hit]:
        """Mem0's search for the probe's question, once per question until `clear_hits`."""
        q = probe.question_text
        if q not in self._hits:
            tokens0 = self.usage.tokens
            self._hits[q] = self._search(q)
            self.cost["serve_tokens"] += self.usage.tokens - tokens0
            self._check_infra()
        return self._hits[q]

    def clear_hits(self) -> None:
        self._hits = {}

    def serve_with(self, probe: Probe, lead: str | None = None) -> ServeResult:
        """`NativeSystem.serve` from the cached hits; a `lead` block takes its tokens (+8, as in
        `pack`) off the budget and goes first."""
        hits = self.hits(probe)
        by_id = {h.hit_id: h for h in hits}
        docs = [Doc(h.hit_id, h.text, h.ts) for h in hits]
        budget = READER_BUDGET - (approx_tokens(lead) + 8 if lead else 0)
        chosen = pack(docs, budget, READER_DOC_CAP)
        refs = [r for d, _ in chosen for r in by_id[d.doc_id].refs]
        diag = [
            {"hit_id": h.hit_id, "rank": i, "score": h.score, "refs": [list(r) for r in h.refs]}
            for i, h in enumerate(hits)
        ]
        blocks = [text for _, text in chosen]
        return ServeResult([lead, *blocks] if lead else blocks, refs, diag, False)

    def serve(self, probe: Probe) -> ServeResult:
        return self.serve_with(probe)


class Mem0Det(LoggedMem0):
    """Arm `mem0_det`: source-keyed invalidation before a revision or deletion is ingested."""

    arm = "mem0_det"

    def targets(self, e: Event) -> tuple[list[str], list[str]]:
        """(earlier unit keys of the record, their extracted memories not yet deleted)."""
        keys = sorted(
            {
                k
                for k in (*self.ledger.entries, *self.made)
                if (p := parse_key(k)) is not None and p[0] == e.object_id and p[1] < e.revision
            },
            key=lambda k: parse_key(k)[1],  # type: ignore[index]
        )
        ids: list[str] = []
        for k in keys:
            for mid in [*(self.ledger.entries.get(k, {}).get("memory_ids") or []), *self.made[k]]:
                if mid and str(mid) not in self.deleted and str(mid) not in ids:
                    ids.append(str(mid))
        return keys, ids

    def _before_unit(self, e: Event) -> None:
        if e.op not in ("update", "delete") or event_key(e) in self.ledger:
            return
        keys, ids = self.targets(e)
        if not ids:
            return
        self._calls = []
        missing: list[str] = []
        errors: list[str] = []
        try:
            for mid in ids:
                try:
                    self._mem.delete(mid)
                except ValueError:  # already gone (Mem0 raises ValueError for an unknown id)
                    missing.append(mid)
                except Exception as exc:
                    if self.on_error == "raise":
                        raise
                    errors.append(f"{mid}: {exc!r}"[:500])
        finally:
            line = {"kind": "invalidate", **self._unit(e), "targets": keys}
            events = self._call_events("adapter")
            self._calls = []
            extra = {"missing": missing} if missing else {}
            self._write(
                {**line, "events": events, **extra, **({"errors": errors} if errors else {})}
            )


class Mem0Rep:
    """Arm `mem0_rep`: the `mem0` arm's store with the program view on definition items."""

    name = "mem0_rep"

    def __init__(self, system: LoggedMem0, view: ZeroDm):
        self.system, self.view = system, view
        self.views: list[dict[str, Any]] = []  # one diagnostic per definition-item probe

    def serve(self, probe: Probe) -> ServeResult:
        slot = self.view.world.slot(probe.fact_id)
        if not slot.kappa_id:
            return self.system.serve(probe)
        block, diag = self.view.view(probe)
        self.views.append({"probe_id": probe.probe_id, **diag})
        return self.system.serve_with(probe, block)

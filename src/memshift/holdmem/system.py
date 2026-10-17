"""HoldMem reference implementation (plan §5): one data layer, four (D, B) cells.

Store  item {item_id, kind, s_k, bind | null, det | null, status, history}.
Write  writer -> s_k; D=1 checker -> fallback -> unverifiable; B=1 key (scope epochs, kappa version,
       theta_w id). Both D arms use the same binding granularity.
Maint  B=1: an item whose key changed is invalidated and re-derived with the current writer.
       B=0: append-only; new items are derived from the delta and old items never invalidate.
Read   BM25 over the s_k of valid items -> fixed template -> shared reader. Identical bytes across
       cells before any intervention (checked by hash assertion in the drivers).

Pre-intervention snapshot: all cells are instantiated from the same writer outputs. The D=1
checker runs and its verdict is recorded in the sidecar, but the fallback substitution is not
applied to the shared snapshot, so every cell serves the same bytes. Every write after the fork
runs the full D=1 path.

Exploratory: `CellConfig(fallback=False)` keeps the D=1 checker but disables the programmatic
fallback, so a checker failure always ends in `unverifiable` instead of substituted extraction.

D1s (prereg v2 §4, `CellConfig(synth=True)`): the D=1 path without benchmark-authored logic. Every
item write sends one more writer request for a program (`holdmem.program`); a failed span check
substitutes the program's output instead of `checker.fallback`, and definition items never reach
the gold evaluator: the writer's value passes when it equals its own program's output, otherwise
the program's output is served. A program that errors, times out or returns the wrong type makes
the item `unverifiable`. Keys still come from the shared writer output only.

Control cells (registered after the first review; every one writes from the same writer requests
as an existing cell wherever its design allows, so the response cache serves them):
  B1_gold     oracle key: the gold dependency objects (id, revision) of the slot and the definition
              version (`gen.gold`), not implementable. Invalidates exactly the items whose
              dependencies changed; its re-derivations are the requests of the source-epoch cell.
  B1_scope    scope-aware key: definition items are keyed by the objects their definition's scope
              ranges over (`gen.kappa.scope_members`) and the definition version, other items by
              their cited clusters. Built from definitions and writer citations only.
  B0_refresh  B=0 that re-derives, at every checkpoint, exactly the items the source-epoch B=1 cell
              re-derives and appends the new items without invalidating the old ones.
  B0_upsert   slot-keyed upsert (prereg v4 block U): the append-only B=0 cell with the same delta
              derivations, so the same requests, except that a new item for a slot retires every
              older valid or unverifiable item of that slot (status `invalid`, history event
              `supersede`). No key is kept: a slot whose delta yields nothing keeps its old item.
  D0t         D=0 (no checker, no sidecar) whose items written after the shift are rendered with
              the D=1 fallback template from the writer's own typed value and citations.
  D0think     D=0 whose definition items written after the shift come from the writer with
              thinking switched on (separate client, so separate cache keys). Its output limit is
              `writer.THINK_MAX_TOKENS` unless `think_max_tokens` sets another (prereg v4 H-b).
  D0full      D=0, B=1 source-epoch whose definition items written after the shift are derived from
              every live record of their declared sources (no BM25 selection, no 6,000-token
              budget; above `writer.FULL_BUDGET` the largest source is cut to its newest records
              and the others stay whole), to separate the access D=1 has from the execution.
              Every other request, the pre-shift snapshot included, is D0B1_source's. Each such
              write records the records given and dropped, per source, and their tokens in the
              item's note.
  D1s ... _ra D1s whose programs are retraction-aware (`holdmem.program`): they declare the ids
              they read, may read nothing else, and receive the item's lineage (every id any earlier
              run declared) so that they can return no value once a record they read is gone.

Definition-item writer (prereg v4 block H-a, `HoldMem(def_writer=...)`, set for every cell of a
run): as in D0think, the pre-shift snapshot and every other item come from the main writer (same
requests), and the definition items written after the shift (B=1 re-derivations, B=0 delta
derivations, refreshes) come from a second client with the same request text and its own output
limit. A reply cut off by that limit is unreadable; each such write records the model, its
completion tokens and whether it reached the limit in the item's note, and every cell counts these
writes in `def_stats`. Keys keep the main writer's id.

Program sets (prereg v4 block G-b, `HoldMem(program_writer=...)`): the D1s program requests of the
definition items, before and after the shift, go to another client (another model, a sampling
temperature, a sample index); every other request is the writer's.
"""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from memshift.gen.gold import gold_at
from memshift.gen.kappa import scope_members
from memshift.gen.model import Slot, World, fmt_ts
from memshift.gen.state import SourceLog
from memshift.holdmem.checker import (
    NO_RULE,
    FallbackResult,
    check,
    fallback,
    render_fallback_sk,
)
from memshift.holdmem.program import ProgramOutcome, program_check, synthesise, synthesise_ra
from memshift.holdmem.writer import CUT_OFF, THINK_MAX_TOKENS, derive, render_sk
from memshift.llm import LLMClient
from memshift.probes import Probe
from memshift.retrieval import Doc, select
from memshift.textutil import sha256_json

READER_BUDGET = 4096
READER_DOC_CAP = 600
PROGRAM_ORIGIN = "computed by the writer's own program"
_COPY_FIELDS = (
    "s_k",
    "written",
    "theta_w",
    "bind",
    "det",
    "status",
    "readset",
    "readset_clusters",
    "refs",
    "eval_value",
    "note",
)


GRANULARITIES = ("source", "cluster", "cited", "gold", "scope")


@dataclass(frozen=True)
class CellConfig:
    D: int
    B: int
    # source | cluster (read set) | cited | gold (oracle) | scope (ignored when B=0)
    granularity: str = "source"
    fallback: bool = True  # exploratory: False keeps the D=1 checker but disables its fallback
    synth: bool = False  # D1s: the replacement value is the writer's own program's output
    refresh: bool = False  # B0_refresh: B=0 that re-derives like source-epoch, never invalidates
    present: bool = False  # D0t: D=0 values, items written after the shift in the D=1 template
    think: bool = False  # D0think: definition items written after the shift by a thinking writer
    full: bool = False  # D0full: definition items after the shift from every in-scope record
    ra: bool = False  # D1s with retraction-aware programs (declared reads, lineage)
    upsert: bool = False  # B0_upsert: a new delta item for a slot retires the slot's older items

    def __post_init__(self) -> None:
        if self.granularity not in GRANULARITIES:
            raise ValueError(f"unknown granularity {self.granularity!r}")
        if self.D == 0 and not self.fallback:
            raise ValueError("fallback=False only makes sense with D=1")
        if self.synth and self.D != 1:
            raise ValueError("synth=True only makes sense with D=1")
        if self.synth and not self.fallback:
            raise ValueError("synth=True has no checker-only variant (fallback must stay True)")
        if self.refresh and (self.B != 0 or self.granularity != "source"):
            raise ValueError("refresh=True is the B=0 cell that refreshes on the source-epoch key")
        if (self.present or self.think) and self.D != 0:
            raise ValueError("present / think only make sense with D=0")
        if self.present and self.think:
            raise ValueError("present and think are separate controls")
        if self.full and (self.D, self.B, self.granularity) != (0, 1, "source"):
            raise ValueError("full=True is the D=0, B=1 source-epoch control (D0fullB1_source)")
        if self.full and (self.present or self.think):
            raise ValueError("full is a separate control from present and think")
        if self.ra and not self.synth:
            raise ValueError("ra=True is a variant of the D1s programs (synth=True)")
        if self.upsert and (self.B != 0 or self.refresh):
            raise ValueError("upsert=True is the append-only B=0 cell (no key, no refresh)")

    @property
    def name(self) -> str:
        tag = "s" if self.synth else "t" if self.present else "think" if self.think else ""
        tag = tag or ("full" if self.full else "")
        base = f"holdmem_D{self.D}{tag}B{self.B}"
        name = f"{base}_{self.granularity}" if self.B else base
        if self.refresh:
            name += "_refresh"
        if self.ra:
            name += "_ra"
        if self.upsert:
            name += "_upsert"
        return f"{name}_nofb" if not self.fallback else name


@dataclass
class Item:
    item_id: str
    slot_id: str
    s_k: str
    written: datetime
    theta_w: str
    bind: dict[str, Any] | None
    det: dict[str, Any] | None
    status: str  # valid | invalid | repairing | unverifiable
    readset: list[tuple[str, int]]
    readset_clusters: list[str]
    refs: list[tuple[str, int]]
    eval_value: Any  # experimenter-side record of the value the write produced; never served
    history: list[dict[str, Any]] = field(default_factory=list)
    note: dict[str, Any] = field(default_factory=dict)  # cell-specific record; empty in most cells

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["written"] = fmt_ts(self.written)
        if not self.note:
            del d["note"]  # cells that record nothing keep their item rows as they were
        return d


@dataclass
class ServeResult:
    blocks: list[str]
    refs: list[tuple[str, int]]
    items: list[dict[str, Any]]
    blocked: bool


def definition_text(world: World, slot: Slot, t: datetime) -> str | None:
    """The definition in effect for a definition item at t, as the templates name it."""
    if not slot.kappa_id:
        return None
    ver = world.kappa_version(slot.kappa_id, t)
    return f"{slot.kappa_id} {ver}: {world.kappa_defs[slot.kappa_id][ver].statement}"


def _as_text(value: Any) -> Any:
    """The writer's typed value for the fallback template, which joins lists of strings."""
    if isinstance(value, list):
        return [str(v) for v in value]
    return value if value is None or isinstance(value, str) else str(value)


class HoldMem:
    def __init__(
        self,
        world: World,
        log: SourceLog,
        cfg: CellConfig,
        writer: LLMClient,
        theta_w: str,
        think_writer: LLMClient | None = None,
        *,
        think_max_tokens: int = THINK_MAX_TOKENS,
        def_writer: LLMClient | None = None,
        def_max_tokens: int | None = None,
        program_writer: LLMClient | None = None,
        program_user_words: bool = False,
    ):
        """`think_max_tokens`: the output limit of the D0think writer's requests. `def_writer`:
        the definition-item writer of prereg v4 block H-a, which writes the definition items after
        the shift (not in a D0think cell, whose thinking writer writes them), with the output limit
        `def_max_tokens` (default: the writer request's). `program_writer`: the client that answers
        the D1s program requests of the definition items (prereg v4 block G-b; default: the
        writer); the other items' programs stay the writer's. `program_user_words` (prereg v4b
        block V): those requests state the definition in the user's words (`USER_WORDS`)."""
        if cfg.think and think_writer is None:
            raise ValueError("a D0think cell needs a think_writer (a client with thinking on)")
        self.world = world
        self.log = log
        self.cfg = cfg
        self.writer = writer
        self.think_writer = think_writer
        self.think_max_tokens = think_max_tokens
        self.def_writer = def_writer
        self.def_max_tokens = def_max_tokens
        # the definition-item writer's requests: sent, unreadable replies, replies cut off by the
        # output limit (B=0 drops an unreadable delta write, so its item never records it)
        self.def_stats = {"writes": 0, "unreadable": 0, "at_limit": 0}
        self.program_writer = program_writer
        self.program_user_words = program_user_words
        if program_user_words and cfg.ra:
            raise ValueError("user-words programs are a variant of the D1s programs, not of _ra")
        self.theta_w = theta_w
        self.items: dict[str, Item] = {}
        self.by_slot: dict[str, list[str]] = {}
        self.last_t: datetime | None = None
        self.cost: dict[str, float] = {
            "ingest_tokens": 0,
            "maintain_tokens": 0,
            "serve_tokens": 0,
            "verify_cpu_s": 0.0,
            "storage_bytes": 0,
        }
        self._epoch_cache: dict[tuple[str, datetime], dict[str, int]] = {}
        self._lock = threading.Lock()
        self.workers = 48

    @property
    def name(self) -> str:
        return self.cfg.name

    def _add_cost(self, key: str, amount: float) -> None:
        with self._lock:
            self.cost[key] += amount

    # ------------------------------------------------------------------ keys
    def _epochs(self, kind: str, t: datetime) -> dict[str, int]:
        k = (kind, t)
        with self._lock:
            hit = self._epoch_cache.get(k)
        if hit is None:
            hit = self.log.source_epochs(t) if kind == "source" else self.log.cluster_epochs(t)
            with self._lock:
                if len(self._epoch_cache) > 16:
                    self._epoch_cache.clear()
                self._epoch_cache[k] = hit
        return hit

    def key_at(self, slot: Slot, clusters: list[str], t: datetime) -> dict[str, Any] | None:
        if not self.cfg.B and not self.cfg.refresh:
            return None
        granularity = "source" if self.cfg.refresh else self.cfg.granularity
        kv = self.world.kappa_version(slot.kappa_id, t) if slot.kappa_id else None
        if granularity == "gold":
            deps = sorted(gold_at(self.world, self.log, slot, t).deps)
            scope: dict[str, Any] = {"deps": sha256_json(deps)[:16], "n": len(deps)}
        elif granularity == "scope" and slot.kappa_id:
            assert kv is not None
            snap = self.log.snapshot(t)
            members = scope_members(slot.kappa_id, snap, t, kv)
            scope = {"members": sha256_json(members)[:16], "n": len(members)}
        elif granularity == "source":
            ep = self._epochs("source", t)
            scope = {s: ep.get(s, 0) for s in slot.sources}
        else:  # cluster, cited, and the cited key of the non-definition items of `scope`
            ep = self._epochs("cluster", t)
            scope = {c: ep.get(c, 0) for c in clusters}
            granularity = "cited" if granularity == "scope" else granularity
        return {
            "granularity": granularity,
            "scope": scope,
            "kappa_version": kv,
            "theta_w": self.theta_w,
        }

    # ------------------------------------------------------------------ write
    def _definition(self, slot: Slot, t: datetime) -> str | None:
        return definition_text(self.world, slot, t)

    def _run_program(
        self, slot: Slot, t: datetime, phase: str, prior: Item | None = None
    ) -> ProgramOutcome:
        """D1s: one more writer request per item write, counted in the same token bucket. The
        retraction-aware program also gets the lineage of the item it replaces. A definition item's
        program comes from `program_writer` when one is set."""
        llm = self.writer
        if self.program_writer is not None and slot.kappa_id:
            llm = self.program_writer
        if self.cfg.ra:
            lineage = list(((prior.det if prior else None) or {}).get("program_lineage", []))
            prog = synthesise_ra(llm, self.world, self.log, slot, t, lineage)
        else:
            words = self.program_user_words and bool(slot.kappa_id)
            prog = synthesise(llm, self.world, self.log, slot, t, user_words=words)
        self._add_cost(f"{phase}_tokens", prog.prompt_tokens + prog.completion_tokens)
        self._add_cost("verify_cpu_s", prog.run_s)
        return prog

    @staticmethod
    def _program_note(item: Item) -> dict[str, Any]:
        """What the item's program did at this write, for the item history (D1s cells only)."""
        det = item.det or {}
        if "program_ok" not in det:
            return {}
        note = {"ok": det["program_ok"], "value": det["program_value"]}
        if "program_lineage" in det:  # retraction-aware programs
            note |= {"n_reads": det["program_n_reads"], "lineage_ok": det["program_lineage_ok"]}
        return {"program": note}

    def _write(
        self,
        slot: Slot,
        t: datetime,
        since: datetime | None,
        apply_fallback: bool,
        phase: str,
        prior: Item | None = None,
    ) -> Item | None:
        # D0think: definition items written after the shift come from the thinking writer
        think = self.cfg.think and apply_fallback and slot.kappa_id is not None
        # D0full: likewise, every in-scope record for the definition items written after the shift
        full = self.cfg.full and apply_fallback and slot.kappa_id is not None
        # H-a: likewise, from the definition-item writer (another model, same request text)
        defw = (
            self.def_writer is not None
            and not think
            and apply_fallback
            and slot.kappa_id is not None
        )
        writer = self.think_writer if think else self.def_writer if defw else self.writer
        assert writer is not None
        max_tokens = self.think_max_tokens if think else self.def_max_tokens if defw else None
        out = derive(
            writer,
            self.world,
            self.log,
            slot,
            t,
            since,
            think=think,
            full=full,
            max_tokens=max_tokens,
            reasoning=defw,
        )
        self._add_cost(f"{phase}_tokens", out.prompt_tokens + out.completion_tokens)
        at_limit = out.finish_reason in CUT_OFF
        if defw and out.finish_reason:  # a request was sent
            with self._lock:
                self.def_stats["writes"] += 1
                self.def_stats["unreadable"] += not out.ok
                self.def_stats["at_limit"] += at_limit
        if since is not None and (out.value is None or not out.ok):
            return None  # append-only delta produced nothing new
        snap = self.log.snapshot(t)
        # Key clusters come from the shared writer output only (read set, or the writer's own
        # citations), never from the D=1 fallback, so that D cannot leak into detection.
        keyed = (
            [(c.object_id, c.revision) for c in out.citations]
            if self.cfg.granularity in ("cited", "scope")
            else out.readset
        )
        clusters = sorted({o.cluster for oid, _ in keyed if (o := snap.get(oid)) is not None})
        s_k = render_sk(slot, t, out)
        refs = [(c.object_id, c.revision) for c in out.citations]
        status = "valid"
        det: dict[str, Any] | None = None
        eval_value = out.value
        if self.cfg.D:
            prog = self._run_program(slot, t, phase, prior) if self.cfg.synth else None
            program_det = (
                {"program_ok": prog.ok, "program_value": prog.value, "program_key": prog.key}
                if prog is not None
                else {}
            )
            if prog is not None and self.cfg.ra:
                program_det |= {
                    "program_n_reads": len(prog.reads or []),
                    "program_lineage": prog.lineage,
                    "program_lineage_ok": prog.lineage_ok,
                }
            t0 = time.process_time()
            if prog is not None and slot.kappa_id:
                cr = program_check(slot.vtype, out.ok, out.value, prog)  # never the gold evaluator
            else:
                cr = check(self.world, self.log, slot, t, out.value, out.citations)
            det = {
                "parse": out.value,
                "spans": cr.spans,
                "check": "pass" if cr.ok else "fail",
                "reason": cr.reason,
                **program_det,
            }
            if not cr.ok and apply_fallback:
                fb: FallbackResult | str
                if prog is not None:
                    fb = (prog.value, []) if prog.ok else NO_RULE
                else:
                    fb = (
                        NO_RULE
                        if not self.cfg.fallback
                        else fallback(self.world, self.log, slot, t)
                    )
                if fb == NO_RULE:
                    status = "unverifiable"
                    det["check"] = "unverifiable"
                    if prog is not None and not slot.kappa_id:
                        det["reason"] = f"{cr.reason}; program failed: {prog.error}"
                else:
                    assert isinstance(fb, tuple)
                    value, fb_refs = fb
                    s_k = render_fallback_sk(
                        slot,
                        t,
                        value,
                        fb_refs,
                        self._definition(slot, t),
                        origin=PROGRAM_ORIGIN if prog is not None else "programmatic extraction",
                    )
                    refs = list(fb_refs)
                    eval_value = value
                    det = {
                        "parse": value,
                        "spans": [{"object_id": o, "revision": r} for o, r in fb_refs],
                        "check": "program" if prog is not None else "fallback",
                        "reason": cr.reason,
                        **program_det,
                    }
            self._add_cost("verify_cpu_s", time.process_time() - t0)
        elif self.cfg.present and apply_fallback and out.ok:
            # D0t: the writer's own typed value and citations in the D=1 fallback template
            s_k = render_fallback_sk(slot, t, _as_text(out.value), refs, self._definition(slot, t))
        note: dict[str, Any] = {}
        if think:
            note = {"think": {"ok": out.ok, "completion_tokens": out.completion_tokens}}
        if defw:
            assert self.def_writer is not None
            note = {
                "defwriter": {
                    "model": self.def_writer.model,
                    "ok": out.ok,
                    "completion_tokens": out.completion_tokens,
                    "at_limit": at_limit,
                }
            }
        if out.access is not None:
            note["full"] = out.access
        return Item(
            item_id="",
            slot_id=slot.slot_id,
            s_k=s_k,
            written=t,
            theta_w=self.theta_w,
            bind=self.key_at(slot, clusters, t),
            det=det,
            status=status,
            readset=out.readset,
            readset_clusters=clusters,
            refs=refs,
            eval_value=eval_value,
            note=note,
        )

    def _store(self, item: Item, event: str) -> Item:
        n = len(self.by_slot.get(item.slot_id, []))
        item.item_id = f"{item.slot_id}#{n}"
        item.history.append(
            {
                "ts": fmt_ts(item.written),
                "event": event,
                "old": None,
                "new": item.status,
                "value": item.eval_value,
                **self._program_note(item),
                **item.note,
            }
        )
        self.items[item.item_id] = item
        self.by_slot.setdefault(item.slot_id, []).append(item.item_id)
        return item

    def _latest(self, slot: Slot) -> Item:
        return self.items[self.by_slot[slot.slot_id][-1]]

    def _key_moved(self, slot: Slot, t: datetime) -> bool:
        """Has the key of the slot's newest item moved since that item was derived?"""
        item = self._latest(slot)
        return item.status in ("valid", "unverifiable") and (
            self.key_at(slot, item.readset_clusters, t) != item.bind
        )

    def _pmap(self, fn: Any, args: list[Any]) -> list[Any]:
        self.log.snapshot(self.last_t or self.world.start)  # warm shared caches
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            return list(ex.map(fn, args))

    def build(self, t: datetime, apply_fallback: bool = False) -> None:
        """Cold build of every slot at t. By default this is the shared pre-intervention snapshot
        (D=1 verdicts recorded, fallback not substituted); E3 builds with the full D=1 path."""
        self.log.snapshot(t)
        items = self._pmap(
            lambda s: self._write(s, t, None, apply_fallback, "ingest"), self.world.slots
        )
        for item in items:
            assert item is not None
            self._store(item, "derive")
        self.last_t = t
        self._update_storage()

    def set_writer(self, writer: LLMClient, theta_w: str) -> None:
        self.writer = writer
        self.theta_w = theta_w

    def maintain(self, t: datetime) -> dict[str, int]:
        """Process everything that changed since the last maintenance point."""
        stats = {"invalidated": 0, "repaired": 0, "appended": 0, "unverifiable": 0}
        self.log.snapshot(t)
        if self.cfg.B:
            stale = []
            for slot in self.world.slots:
                item = self.items[self.by_slot[slot.slot_id][-1]]
                if item.status not in ("valid", "unverifiable"):
                    continue
                if self.key_at(slot, item.readset_clusters, t) == item.bind:
                    continue
                item.history.append(
                    {
                        "ts": fmt_ts(t),
                        "event": "invalidate",
                        "old": item.status,
                        "new": "invalid",
                        "value": None,
                    }
                )
                item.status = "invalid"
                stale.append((slot, item))
            stats["invalidated"] = len(stale)
            news = self._pmap(
                lambda si: self._write(si[0], t, None, True, "maintain", si[1]), stale
            )
            for (_, item), new in zip(stale, news, strict=True):
                assert new is not None
                for f in _COPY_FIELDS:
                    setattr(item, f, getattr(new, f))
                item.history.append(
                    {
                        "ts": fmt_ts(t),
                        "event": "repair",
                        "old": "invalid",
                        "new": item.status,
                        "value": item.eval_value,
                        **self._program_note(item),
                        **item.note,
                    }
                )
                stats["repaired"] += 1
                stats["unverifiable"] += item.status == "unverifiable"
        elif self.cfg.refresh:
            # B0_refresh: the source-epoch key only schedules the re-derivation (the requests of the
            # B=1 source-epoch cell, full in-scope records); the old item stays valid beside it.
            todo = [slot for slot in self.world.slots if self._key_moved(slot, t)]
            news = self._pmap(
                lambda s: self._write(s, t, None, True, "maintain", self._latest(s)), todo
            )
            for new in news:
                assert new is not None
                self._store(new, "append")
                stats["appended"] += 1
                stats["unverifiable"] += new.status == "unverifiable"
        else:
            since = self.last_t
            assert since is not None
            news = self._pmap(
                lambda s: self._write(s, t, since, True, "maintain", self._latest(s)),
                self.world.slots,
            )
            if self.cfg.upsert:
                stats["superseded"] = 0
            for new in news:
                if new is not None:
                    if self.cfg.upsert:
                        stats["superseded"] += self._supersede(new.slot_id, t)
                    self._store(new, "append")
                    stats["appended"] += 1
                    stats["unverifiable"] += new.status == "unverifiable"
        self.last_t = t
        self._update_storage()
        return stats

    def _supersede(self, slot_id: str, t: datetime) -> int:
        """B0_upsert: retire every older valid or unverifiable item of the slot before its new item
        is stored. Returns the number retired."""
        n = 0
        for item_id in self.by_slot.get(slot_id, []):
            item = self.items[item_id]
            if item.status not in ("valid", "unverifiable"):
                continue
            item.history.append(
                {
                    "ts": fmt_ts(t),
                    "event": "supersede",
                    "old": item.status,
                    "new": "invalid",
                    "value": None,
                }
            )
            item.status = "invalid"
            n += 1
        return n

    def _update_storage(self) -> None:
        blob = json.dumps([i.to_json() for i in self.items.values()], default=str)
        self.cost["storage_bytes"] = len(blob.encode())

    # ------------------------------------------------------------------ read
    def _provenance_sources(self, item: Item) -> set[str] | None:
        if not self.cfg.B:
            return None  # no binding: provenance unknown to the data layer
        snap = self.log.snapshot(item.written)
        return {o.source for oid, _ in item.readset if (o := snap.get(oid)) is not None}

    def render_item(self, item: Item, render: str) -> str:
        if render != "json":
            return item.s_k
        return json.dumps(
            {
                "item": self.world.slot(item.slot_id).description,
                "value": (item.det or {}).get("parse"),
                "written": f"{item.written:%Y-%m-%d}",
            },
            ensure_ascii=False,
        )

    def serve(self, probe: Probe, render: str = "text") -> ServeResult:
        granted = self.world.granted(probe.agent_id, probe.as_of)
        revoked = set(self.world.agents.get(probe.agent_id, [])) - granted
        blocked = False
        docs = []
        for item in self.items.values():
            if item.status != "valid":
                continue
            prov = self._provenance_sources(item)
            if revoked and prov is not None and prov & revoked:
                blocked = blocked or item.slot_id == probe.fact_id
                continue
            docs.append(Doc(item.item_id, self.render_item(item, render), item.written))
        chosen = select(docs, probe.question_text, READER_BUDGET, READER_DOC_CAP)
        refs = [r for d, _ in chosen for r in self.items[d.doc_id].refs]
        diag = [
            {
                "item_id": i,
                "D": self.cfg.D,
                "B": self.cfg.B,
                "status": self.items[i].status,
                "key": self.items[i].bind,
            }
            for i in self.by_slot.get(probe.fact_id, [])
        ]
        return ServeResult([text for _, text in chosen], refs, diag, blocked)

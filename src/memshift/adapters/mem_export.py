"""Portable Memory (.mem) export of a HoldMem store and import into other implementations
(E6, the leaving test T2), through the `portable-memory` reference SDK (0.3.0, format 1.1;
optional `native` extra, imported lazily).

Standard records (both variants; what any compliant importer keeps)
  fact     one per HoldMem item: id `holdmem:<item_id>`, text = s_k, subject = the slot's subject
           (or "user"), predicate = slot id, object "" (the typed value is D=1 sidecar data, so it
           travels only in the extension), bi-temporal tValidFrom = createdAt = eventTime = the
           write time, and tValidTo = the write time for items the store does not serve
           (unverifiable / invalid), so a reader of the standard records never serves them.
  episode  `src:<object_id>#<revision>` for each fact's primary citation, holding the source
           record text at that revision; `fact.episodeID` points to it ("" when the citation does
           not resolve to a record). This is the provenance the format keeps by itself: one
           evidence episode per fact, and a tombstone on that episode removes the fact on import
           (importer.py `gone(f.id, f.episode_id)`). It carries no binding key, no parse, no
           definition or writer version and no status beyond the validity interval.
Derivation extension (`with_ext=True`): the record {id (fact id), item_id, slot_id, s_k, written,
  status, source_refs, kappa_version, theta_w_id, spans, bind, det, readset, readset_clusters},
  written through one or more carriers:
  kind     (default) vendor-namespaced record kind `memshift.item` (items/memshift.item.jsonl);
           unknown kinds round-trip verbatim (spec §10: importer.py passthrough, exporter.py
           re-emits `export_passthrough_lines`).
  episode  `ext` on one derivation episode `holdmem:derivation:<item_id>` per item: the only
           record kind whose `ext` the format guarantees (exporter.py `_write_episodes` /
           `merge_episode_ext`, importer.py `extract_episode_ext`).
  fact     `ext` merged into the fact line itself, as RFC-0002 ("ext for all kinds") proposes; the
           0.3.0 importer decodes facts with `from_wire`, which keeps dataclass fields only.
  `ext_survival` measures, by an actual validate -> import -> re-export -> import round trip
  through the SDK, which carriers and which fields survive.
Import  `import_to_holdmem` rebuilds a HoldMem (the second implementation of T2 / E6) from the
  standard facts plus the derivation record, read from the custom kind when present, else from
  episode `ext`. Without a derivation record every item has bind=None and det=None and the store
  runs B=0 (append-only) maintenance: nothing is ever invalidated, the predicted silent staleness
  of a leave without the extension. `import_to_mem0` stores each served fact verbatim through
  Mem0's raw path (`add(infer=False)`): the SDK's Mem0 adapter only maps a Mem0 export *into*
  portable episodes (portable_memory/adapters/mem0.py), there is no reverse adapter.
"""

from __future__ import annotations

import functools
import json
import tempfile
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from memshift import __version__
from memshift.gen.model import Event, World, fmt_ts
from memshift.gen.state import SourceLog
from memshift.holdmem.system import CellConfig, HoldMem, Item
from memshift.llm import LLMClient

if TYPE_CHECKING:
    from memshift.adapters.native import Mem0System

EXT_KEY = "memshift"
ITEM_KIND = "memshift.item"
FACT_PREFIX = "holdmem:"
DERIVATION_PREFIX = "holdmem:derivation:"
SOURCE_PREFIX = "src:"
CARRIERS = ("kind", "episode", "fact")
GENERATOR = f"memshift-holdmem/{__version__}"


def _utc(dt: datetime) -> datetime:
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def _naive(dt: datetime) -> datetime:
    return dt.astimezone(UTC).replace(tzinfo=None) if dt.tzinfo else dt


def _id_order(item_id: str) -> tuple[str, int]:
    base, _, n = item_id.rpartition("#")
    return (base, int(n)) if n.isdigit() else (item_id, 0)


@functools.cache
def _store_cls() -> type:
    from portable_memory import InMemoryStore

    class BundleStore(InMemoryStore):
        """The SDK's in-memory host (episodes + ext, entities, edges, tombstones, passthrough
        kinds) plus facts, which it does not model."""

        def __init__(self, generator: str | None = None) -> None:
            super().__init__(generator)
            self.facts: dict[str, Any] = {}

        def export_facts(self) -> list[Any]:
            return [self.facts[k] for k in sorted(self.facts)]

        def import_fact(self, f: Any) -> None:
            self.facts[f.id] = f

        def apply_tombstone(self, t: Any) -> None:
            super().apply_tombstone(t)
            self.facts.pop(t.target_id, None)

    return BundleStore


def bundle_store(generator: str | None = None) -> Any:
    """A fresh in-memory Portable Memory store that also holds facts."""
    return _store_cls()(generator)


# ---------------------------------------------------------------------- export
def derivation_record(hm: HoldMem, item: Item) -> dict[str, Any]:
    """The derivation extension of one item, JSON-ready."""
    slot = hm.world.slot(item.slot_id)
    kappa = hm.world.kappa_version(slot.kappa_id, item.written) if slot.kappa_id else None
    return {
        "id": FACT_PREFIX + item.item_id,
        "item_id": item.item_id,
        "slot_id": item.slot_id,
        "s_k": item.s_k,
        "written": fmt_ts(item.written),
        "status": item.status,
        "source_refs": [{"object_id": o, "revision": r} for o, r in item.refs],
        "kappa_version": kappa,
        "theta_w_id": item.theta_w,
        "spans": list((item.det or {}).get("spans") or []),
        "bind": item.bind,
        "det": item.det,
        "readset": [[o, r] for o, r in item.readset],
        "readset_clusters": list(item.readset_clusters),
    }


EXT_FIELDS = (
    "id",
    "item_id",
    "slot_id",
    "s_k",
    "written",
    "status",
    "source_refs",
    "kappa_version",
    "theta_w_id",
    "spans",
    "bind",
    "det",
    "readset",
    "readset_clusters",
)


def _episode(
    eid: str,
    t: datetime,
    source_type: str,
    source_id: str,
    text: str,
    metadata: dict[str, str],
) -> Any:
    from portable_memory import PortableEpisode

    return PortableEpisode(
        id=eid,
        event_time=_utc(t),
        mention_time=_utc(t),
        ingestion_time=_utc(t),
        source_type=source_type,
        source_id=source_id,
        actors=[],
        summary=text[:120],
        details=text,
        sensitivity="low",
        metadata=metadata,
        categories=[],
        importance=0.5,
        confidence=1.0,
        lifecycle_state="HOT",
        extraction_state="done",
        access_count=0,
        pinned=False,
        vault_refs=[],
    )


def source_episode(e: Event) -> Any:
    """The cited source record at one revision, as a standard episode."""
    return _episode(
        f"{SOURCE_PREFIX}{e.object_id}#{e.revision}",
        e.ts,
        e.source,
        e.object_id,
        str(e.payload.get("text", "")),
        {"object_id": e.object_id, "revision": str(e.revision), "source": e.source},
    )


def item_fact(hm: HoldMem, item: Item, episode_id: str) -> Any:
    """One HoldMem item as a standard `PortableFact` (s_k and the bi-temporal fields)."""
    from portable_memory import PortableFact

    t = _utc(item.written)
    slot = hm.world.slot(item.slot_id)
    return PortableFact(
        id=FACT_PREFIX + item.item_id,
        episode_id=episode_id,
        subject=slot.subject or "user",
        predicate=item.slot_id,
        obj="",
        text=item.s_k,
        t_valid_from=t,
        t_valid_to=None if item.status == "valid" else t,
        confidence=0.5,
        importance=0.5,
        created_at=t,
        event_time=t,
    )


def _exporter(fact_ext: dict[str, str]) -> Any:
    """The SDK's exporter; with `fact_ext` it also merges `ext` into fact lines (RFC-0002 style),
    which the 0.3.0 exporter only does for episodes."""
    from portable_memory import BundleExporter, MemFileEntry, sha256_hex, to_line
    from portable_memory.interop import merge_episode_ext

    if not fact_ext:
        return BundleExporter()

    class FactExtExporter(BundleExporter):
        def _write_jsonl(self, rows: list[Any], rel_path: str, root: str, files: list[Any]) -> int:
            if rel_path != "items/fact.jsonl" or not rows:
                return int(super()._write_jsonl(rows, rel_path, root, files))
            lines = sorted(
                merge_episode_ext(to_line(r).encode(), fact_ext.get(r.id)).decode() for r in rows
            )
            data = ("\n".join(lines) + "\n").encode()
            self._write_bytes(data, rel_path, root)
            files.append(MemFileEntry(path=rel_path, sha256=sha256_hex(data), bytes=len(data)))
            return len(rows)

    return FactExtExporter()


def export_holdmem(
    hm: HoldMem,
    bundle_dir: str | Path,
    with_ext: bool = True,
    carriers: Iterable[str] = ("kind",),
) -> Any:
    """Write `hm` as a .mem bundle; returns the SDK's `MemManifest`. `carriers` (with_ext only)
    selects where the derivation record goes: 'kind', 'episode', 'fact' (see module doc)."""
    from portable_memory import canonical_json

    chosen = tuple(carriers) if with_ext else ()
    if unknown := set(chosen) - set(CARRIERS):
        raise ValueError(f"unknown carriers {sorted(unknown)}; choose from {CARRIERS}")
    store = bundle_store(GENERATOR)
    events = {(e.object_id, e.revision): e for e in hm.log.events if e.op != "delete"}
    kind_lines: list[str] = []
    fact_ext: dict[str, str] = {}
    for item in sorted(hm.items.values(), key=lambda i: _id_order(i.item_id)):
        episode_id = ""
        if item.refs and (ev := events.get(item.refs[0])) is not None:
            src = source_episode(ev)
            store.episodes.setdefault(src.id, src)
            episode_id = src.id
        fact = item_fact(hm, item, episode_id)
        store.facts[fact.id] = fact
        if not chosen:
            continue
        rec = derivation_record(hm, item)
        blob = canonical_json({EXT_KEY: rec})
        if "kind" in chosen:
            kind_lines.append(canonical_json(rec))
        if "episode" in chosen:
            text = (
                f"HoldMem derivation of fact {fact.id} (knowledge item {item.slot_id}, written "
                f"{item.written:%Y-%m-%d %H:%M} by {item.theta_w})."
            )
            dep = _episode(
                DERIVATION_PREFIX + item.item_id, item.written, "derivation", fact.id, text, {}
            )
            store.episodes[dep.id] = dep
            store.episode_ext[dep.id] = blob
        if "fact" in chosen:
            fact_ext[fact.id] = blob
    if kind_lines:
        store.passthrough[ITEM_KIND] = sorted(kind_lines)
    Path(bundle_dir).mkdir(parents=True, exist_ok=True)
    return _exporter(fact_ext).export(store, str(bundle_dir))


def validate_bundle(bundle_dir: str | Path) -> list[str]:
    """Integrity issues found by the SDK's `BundleValidator` (empty list = valid)."""
    from portable_memory import BundleValidator

    issues: list[str] = list(BundleValidator().validate(str(bundle_dir)).issues)
    return issues


# ---------------------------------------------------------------------- read
@dataclass
class BundleContents:
    """What the SDK's importer delivered to a host store."""

    facts: list[Any]
    episodes: dict[str, Any]
    episode_ext: dict[str, dict[str, Any]]  # episode id -> memshift record
    kinds: dict[str, list[str]]  # passthrough kind -> raw lines
    report: Any = None
    store: Any = field(default=None, repr=False)

    def derivations(self) -> tuple[str | None, dict[str, dict[str, Any]]]:
        """(carrier, fact id -> derivation record): the custom kind when present, else the
        derivation episodes' ext, else nothing."""
        lines = self.kinds.get(ITEM_KIND) or []
        recs = [r for r in (json.loads(x) for x in lines) if isinstance(r, dict) and "id" in r]
        if recs:
            return "kind", {str(r["id"]): r for r in recs}
        from_eps = {str(r["id"]): r for r in self.episode_ext.values() if "id" in r}
        if from_eps:
            return "episode", from_eps
        return None, {}


def read_bundle(bundle_dir: str | Path) -> BundleContents:
    """Import through the SDK's `BundleImporter` (all checksums verified before any record is
    read) into a fresh store and return what it delivered."""
    from portable_memory import BundleImporter

    store = bundle_store()
    report = BundleImporter().import_bundle(store, str(bundle_dir), reembed=False)
    ext: dict[str, dict[str, Any]] = {}
    for eid, raw in store.episode_ext.items():
        obj = json.loads(raw).get(EXT_KEY) if raw else None
        if isinstance(obj, dict):
            ext[eid] = obj
    facts = sorted(store.facts.values(), key=lambda f: (f.t_valid_from, _id_order(f.id)))
    kinds = {k: list(v) for k, v in store.passthrough.items()}
    return BundleContents(facts, dict(store.episodes), ext, kinds, report, store)


def _on_disk(root: Path) -> dict[str, dict[str, Any]]:
    """Per carrier: records carrying the derivation record in the bundle files, and their keys."""

    def scan(rel: str, nested: bool) -> dict[str, Any]:
        p = root / "items" / rel
        n, keys = 0, set()
        for line in p.read_text(encoding="utf-8").splitlines() if p.is_file() else []:
            rec = json.loads(line)
            body = rec.get(EXT_KEY) if nested else rec
            if isinstance(body, dict):
                n += 1
                keys |= set(body)
        return {"records": n, "fields": keys}

    return {
        "kind": scan(f"{ITEM_KIND}.jsonl", nested=False),
        "episode": scan("episode.jsonl", nested=True),
        "fact": scan("fact.jsonl", nested=True),
    }


def _delivered(c: BundleContents) -> dict[str, dict[str, Any]]:
    """Per carrier: derivation records the importer handed to the host store."""
    kind = [json.loads(x) for x in c.kinds.get(ITEM_KIND, [])]
    fact = [vars(f)[EXT_KEY] for f in c.facts if isinstance(vars(f).get(EXT_KEY), dict)]
    out = {}
    for name, recs in (("kind", kind), ("episode", list(c.episode_ext.values())), ("fact", fact)):
        keys: set[str] = set()
        for r in recs:
            keys |= set(r)
        out[name] = {"records": len(recs), "fields": keys}
    return out


def ext_survival(bundle_dir: str | Path) -> dict[str, Any]:
    """Which derivation-record carriers survive the SDK's own round trip, observed on this
    bundle: validate -> import -> re-export (SDK exporter) -> validate -> import. Also reports
    the provenance the standard records keep by themselves."""
    from portable_memory import BundleExporter

    root = Path(bundle_dir)
    written = _on_disk(root)
    first = read_bundle(root)
    with tempfile.TemporaryDirectory() as tmp:
        BundleExporter().export(first.store, tmp)
        issues_rt = validate_bundle(tmp)
        rt_disk = _on_disk(Path(tmp))
        second = read_bundle(tmp)
    imported, after = _delivered(first), _delivered(second)
    carriers: dict[str, Any] = {}
    for name in CARRIERS:
        w = written[name]
        carriers[name] = {
            "records_written": w["records"],
            "fields_written": sorted(w["fields"]),
            "records_imported": imported[name]["records"],
            "fields_dropped_on_import": sorted(w["fields"] - imported[name]["fields"]),
            "records_after_reexport": rt_disk[name]["records"],
            "records_after_second_import": after[name]["records"],
            "survives": w["records"] > 0
            and after[name]["records"] == w["records"]
            and after[name]["fields"] == w["fields"],
        }
    sources = {k for k in first.episodes if k.startswith(SOURCE_PREFIX)}
    return {
        "issues": validate_bundle(root),
        "issues_after_reexport": issues_rt,
        "carriers": carriers,
        "native": {
            "facts": len(first.facts),
            "facts_with_source_episode": sum(f.episode_id in sources for f in first.facts),
            "facts_closed": sum(f.t_valid_to is not None for f in first.facts),
            "source_episodes": len(sources),
            "facts_after_reexport": len(second.facts),
        },
    }


# ---------------------------------------------------------------------- import: HoldMem
class ImportedHoldMem(HoldMem):
    """A HoldMem rebuilt from a .mem bundle alone (the second implementation of T2 / E6). Its
    maintenance follows what the bundle carried: with a derivation record, bound items are
    invalidated and repaired when their key changes; without one the store is append-only."""

    bundle_dir: str = ""
    ext_source: str | None = None  # 'kind' | 'episode' | None

    @property
    def name(self) -> str:
        return f"imported_{self.cfg.name}_{self.ext_source or 'noext'}"


class _DetachedWriter:
    """Placeholder writer: serving works, any re-derivation fails loudly until `set_writer`."""

    model = "detached"

    def complete(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("imported HoldMem has no writer: pass writer= or call set_writer()")


def import_to_holdmem(
    bundle_dir: str | Path,
    world: World,
    log: SourceLog,
    cfg: CellConfig,
    writer: LLMClient | None = None,
    theta_w: str | None = None,
    as_of: datetime | None = None,
) -> ImportedHoldMem:
    """Rebuild a HoldMem store from a bundle. `cfg` is the importing implementation's cell; its B
    applies only if the bundle carries bindings (else B=0), and the binding granularity is taken
    from the bundle. `theta_w` defaults to the bundle's writer id, `as_of` (the last maintenance
    point) to the latest item write time."""
    contents = read_bundle(bundle_dir)
    ext_source, recs = contents.derivations()
    binds = [r["bind"] for r in recs.values() if r.get("bind")]
    grans = {str(b.get("granularity", cfg.granularity)) for b in binds}
    if len(grans) > 1:
        raise ValueError(f"{bundle_dir}: mixed binding granularities {sorted(grans)}")
    eff = CellConfig(
        D=cfg.D,
        B=cfg.B if binds else 0,
        granularity=grans.pop() if grans else cfg.granularity,
    )
    if theta_w is None:
        thetas = Counter(str(r["theta_w_id"]) for r in recs.values() if r.get("theta_w_id"))
        theta_w = thetas.most_common(1)[0][0] if thetas else getattr(writer, "model", "unknown")
    w = writer if writer is not None else cast(LLMClient, _DetachedWriter())
    hm = ImportedHoldMem(world, log, eff, w, str(theta_w))
    hm.bundle_dir, hm.ext_source = str(bundle_dir), ext_source
    for f in contents.facts:
        written = _naive(f.t_valid_from)
        r = recs.get(f.id)
        if r is not None:
            det = r.get("det") if eff.D else None
            status = str(r.get("status", "valid"))
            item = Item(
                item_id=str(r["item_id"]),
                slot_id=str(r["slot_id"]),
                s_k=f.text,
                written=written,
                theta_w=str(r.get("theta_w_id") or theta_w),
                bind=r.get("bind") if eff.B else None,
                det=det,
                status=status,
                readset=[(str(o), int(n)) for o, n in r.get("readset") or []],
                readset_clusters=[str(c) for c in r.get("readset_clusters") or []],
                refs=[
                    (str(x["object_id"]), int(x["revision"])) for x in r.get("source_refs") or []
                ],
                eval_value=(det or {}).get("parse"),
            )
        else:  # standard records only: s_k, validity, knowledge item, one evidence episode
            status = "valid" if f.t_valid_to is None else "invalid"
            src = contents.episodes.get(f.episode_id)
            md = src.metadata if src is not None else {}
            refs = [(md["object_id"], int(md["revision"]))] if "object_id" in md else []
            item = Item(
                item_id=f.id.removeprefix(FACT_PREFIX),
                slot_id=f.predicate,
                s_k=f.text,
                written=written,
                theta_w=str(theta_w),
                bind=None,
                det=None,
                status=status,
                readset=[],
                readset_clusters=[],
                refs=refs,
                eval_value=None,
            )
        item.history.append(
            {"ts": fmt_ts(written), "event": "import", "old": None, "new": status, "value": None}
        )
        hm.items[item.item_id] = item
        hm.by_slot.setdefault(item.slot_id, []).append(item.item_id)
    for ids in hm.by_slot.values():
        ids.sort(key=lambda i: (hm.items[i].written, _id_order(i)))
    written_all = [i.written for i in hm.items.values()]
    hm.last_t = as_of or (max(written_all) if written_all else None)
    hm._update_storage()
    return hm


# ---------------------------------------------------------------------- import: Mem0
def import_to_mem0(
    bundle_dir: str | Path,
    world: World,
    store_dir: str | Path,
    llm_base_url: str,
    llm_model: str,
    embed_base_url: str,
    embed_model: str,
    user_id: str,
    **kw: Any,
) -> Mem0System:
    """Import a bundle into a (new or resumed) Mem0 store: each fact that is valid (open
    validity interval) goes in verbatim through `Memory.add(infer=False)`, with its bi-temporal
    fields, its evidence episode's source ref and, when present, the derivation record as
    metadata. No LLM call; one embedding per fact."""
    from memshift.adapters.native import Mem0System

    kw.setdefault("keep_raw", False)  # the imported text is itself the stored memory
    system = Mem0System(
        world, store_dir, llm_base_url, llm_model, embed_base_url, embed_model, user_id, **kw
    )
    contents = read_bundle(bundle_dir)
    _, recs = contents.derivations()
    for f in contents.facts:
        if f.t_valid_to is not None:
            continue
        md: dict[str, Any] = {
            "fact_id": f.id,
            "event_time": fmt_ts(_naive(f.t_valid_from)),
            "predicate": f.predicate,
        }
        src = contents.episodes.get(f.episode_id)
        if src is not None and "object_id" in src.metadata:
            ref = {
                "object_id": src.metadata["object_id"],
                "revision": int(src.metadata["revision"]),
            }
            md["source_refs"] = [ref]
        if f.id in recs:
            md["derivation"] = recs[f.id]
            md["source_refs"] = recs[f.id].get("source_refs") or md.get("source_refs", [])
        system.add_memory(f"bundle:{f.id}", f.text, md)
    return system

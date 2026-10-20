"""Block W: D_m programs over fields extracted from prose (cells D1sxB0 and
D1sxB1_source, `holdmem.extract`), from the W runs. Nothing here calls a model.

    python -m memshift.analysis.round3b_extract --runs RUN [RUN ...] \\
        [--compare RUN_OR_STAGE ...] [--response-cache SQLITE] --out FILE.json

--runs: the W runs (cells D0B0, D0B1_source, D1sB0, D1sB1_source, D1sxB0, D1sxB1_source; worlds
1000-1009, branches src and def). --compare: the committed runs of the same writer that hold the
anchor cells (e1_test_*, e1_synth_*, or the stages stage2_e1 and stage4_c_e1, which expand to
them). --response-cache: the writer's response cache, from which the D_m programs are read to see
which fields they name (attribution below; without it that share is not computed).

Pairing (`round3_blocks.pairing`, its rules). Only complete runs; per (variant, cell, world,
branch) the newest run; every pre-shift hash assertion exists and passed; with --compare, on each
(world, branch) the probe keys of every cell equal those of the committed D0B0 cell, the gold of
every shared probe equals the committed gold, and the anchor cells (every cell the committed runs
hold: D0B0, D0B1_source, D1sB0, D1sB1_source) answer as committed at every checkpoint. A count
that is not 0 makes the run void (`pairing.ok` false); the quantities are reported as computed.

Quantities (units: fractions; intervals: world-clustered bootstrap, 2,000 draws, seed 996,
percentile 95%, `ex_e1_classes`; paired differences per item over the item-times present in both):
  extraction   from extractions.jsonl, once per distinct record content per world (a revision both
               branches read counts once): per source, the records, the share without an error and
               the share without an inexact field (`holdmem.extract.field_errors`, recomputed from
               the logged typed and extracted values), per field the share without an error and
               without an inexact value; the rewrite check: first attempts that failed (by issue
               kind), texts kept with a miss (`missed`), records without a text, failed
               extractions; and the same pooled over the sources. Record-level shares carry a world
               bootstrap interval. Background records (newsletter digests, kept typed) are counted
               from the run summaries.
  r            ContractValid of the changed definition item-times that the source-epoch key
               invalidated (the condition of `round3_blocks.r0`), in D1sxB1_source and in
               D1sB1_source, and their paired difference (D1sx minus D1s), per definition and pooled
               over the six, after S_src and after S_def. Both cells have the same keys (the writer
               output is shared), so they invalidate the same item-times; `detection_differs` counts
               the changed definition item-times where they do not.
  bxd          B x D on Update (ContractValid on the changed item-times) of the definition items
               (and, reported, of the other items) with D_m on extracted fields (D1sxB0,
               D1sxB1_source against D0B0, D0B1_source), with D_m on typed fields (D1sB0,
               D1sB1_source), and the paired per-item difference of the two double differences
               (extracted minus typed), per branch, all over the item-times every one of the six
               cells answered.
  attribution  per cell pair (B0: D1sxB0 against D1sB0; B1: D1sxB1_source against D1sB1_source)
               and branch, over the definition item-times answered after the fork: each cell's
               value of the item at the probe time is the value of the slot's last write at or
               before it (`derive`, `repair` or `append` in items.jsonl). Of the item-times where
               the D1sx value differs from the D1s value, the share whose producing D1sx write read
               at least one record with an extraction error (the write's recorded program input:
               the live records of the item's declared sources at the write), and, with
               --response-cache, the share where an erroneous field is one the item's program names
               (a quoted field name in its code: a cheap reading of "a field the program reads",
               which over-counts fields a program names but does not use). Differing item-times
               without any error are counted apart (they come from inexact values or the prose in
               `text`).

Threshold comparisons (`threshold_tests`, as in `round3_blocks`):
  W1  B x D on Update of the definition items with D_m on extracted fields: point estimate above 0
      and the 95% lower bound above 0, after S_src and after S_def each.
  W2  B x D (extracted) minus B x D (typed), paired per item: point estimate at most 0, after S_src
      and after S_def each.
Both are computed on the ten worlds 1000-1009 (`W_WORLDS`) in the six cells; worlds missing there
are listed under `worlds_missing`.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from memshift.analysis import ex_e1_classes as ex
from memshift.analysis import round3_blocks as rr
from memshift.holdmem.extract import SOURCES, all_fields, field_errors
from memshift.holdmem.program import extract_code
from memshift.llm import LLMResult
from memshift.runinfo import code_state

Row = dict[str, Any]
Vec = ex.Vec
Est = ex.Est

STATUS = "block W (extracted fields): the W runs are the inputs"
UNITS = rr.UNITS
DRAWS = ex.DRAWS
SHIFTS = ("src", "def")
CLASSES = ("definition", "other")
D0 = ("holdmem_D0B0", "holdmem_D0B1_source")
DM = ("holdmem_D1sB0", "holdmem_D1sB1_source")
DMX = ("holdmem_D1sxB0", "holdmem_D1sxB1_source")
CELLS = (*D0, *DM, *DMX)
PAIRS = {"B0": (DMX[0], DM[0]), "B1": (DMX[1], DM[1])}  # (extracted, typed)
W_WORLDS = tuple(f"w{s}" for s in range(1000, 1010))
WRITE_EVENTS = ("derive", "repair", "append")
FIELD_NAMES = frozenset(name for s in SOURCES for name, _ in all_fields(s))
_QUOTED = re.compile(r"""['"]([A-Za-z_][A-Za-z0-9_]*)['"]""")


# ================================================================================== extraction
def read_extractions(run_dirs: Iterable[Path]) -> list[Row]:
    """The extraction rows of the runs, once per (world, record content): the rows of one content
    agree (one request each), so the first is kept."""
    seen: set[tuple[str, str]] = set()
    out: list[Row] = []
    for d in run_dirs:
        for r in rr._rows(d / "extractions.jsonl"):
            key = (str(r["world_id"]), json.dumps([r["source"], r["typed"]], sort_keys=True))
            if key not in seen:
                seen.add(key)
                out.append(r)
    return out


def _share(by_world: Mapping[str, list[float]], draws: int) -> Est:
    return ex.mean_estimate(by_world, draws)


def extraction_quality(rows: Sequence[Row], draws: int = DRAWS) -> dict[str, Any]:
    """Accuracy per source and field, rewrite-check and extraction failures (module docstring)."""
    groups: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
        groups["pooled"].append(r)
    out: dict[str, Any] = {}
    for source in [*sorted(s for s in groups if s != "pooled"), "pooled"]:
        rs = groups[source]
        ok: dict[str, list[float]] = defaultdict(list)
        exact: dict[str, list[float]] = defaultdict(list)
        fields: dict[str, Counter[str]] = defaultdict(Counter)
        first: Counter[str] = Counter()
        counts: Counter[str] = Counter()
        for r in rs:
            errors, inexact = field_errors(r["source"], r["typed"], r["extracted"])
            ok[r["world_id"]].append(float(not errors))
            exact[r["world_id"]].append(float(not inexact))
            for name in r["typed"]:
                f = fields[f"{r['source']}.{name}" if source == "pooled" else name]
                f["n"] += 1
                f["errors"] += name in errors
                f["inexact"] += name in inexact
            rw = r["rewrite"]
            for kind in sorted({k for k, _ in rw["first_issues"]}):
                first[kind] += 1
            counts["retried"] += bool(rw["retried"])
            counts["missed"] += r["prose"] is not None and bool(rw["final_issues"])
            counts["no_text"] += r["prose"] is None
            counts["extract_failed"] += not r["extract_ok"]
            counts["with_error"] += bool(errors)
        n = len(rs)
        out[source] = {
            "records": n,
            "record_correct": _share(ok, draws),
            "record_exact": _share(exact, draws),
            "fields": {
                name: {
                    "n": c["n"],
                    "correct": 1 - c["errors"] / c["n"],
                    "exact": 1 - c["inexact"] / c["n"],
                }
                for name, c in sorted(fields.items())
            },
            "rewrite": {
                "first_attempt_failed": counts["retried"] / n if n else None,
                "first_failures": dict(sorted(first.items())),
                "missed": counts["missed"],
                "missed_share": counts["missed"] / n if n else None,
                "no_text": counts["no_text"],
            },
            "extract_failed": counts["extract_failed"],
            "with_error": counts["with_error"],
        }
    return out


def background_counts(runs: Sequence[rr.Run]) -> dict[str, int]:
    """Record revisions read and kept typed (digests), per branch, from the run summaries."""
    out: Counter[str] = Counter()
    for run in runs:
        for s in rr._rows(run.path / "world_summaries.jsonl"):
            for b, node in (s.get("branches") or {}).items():
                for c in (node.get("extraction") or {}).values():
                    out[b] += int(c.get("background", 0))
    return dict(sorted(out.items()))


# =========================================================================================== r
def _fact(fact: str | None) -> rr.Cond:
    def cond(r: Row) -> bool:
        detected = bool(r["changed"]) and bool(r.get("detected"))
        if fact is None:
            return detected and ex.in_class(r, "definition")
        return detected and r["fact_id"] == fact

    return cond


def r_contrast(tab: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """r in D1sxB1_source and D1sB1_source and their paired difference, per definition, pooled."""
    out: dict[str, Any] = {}
    x_sys, m_sys = PAIRS["B1"]
    for b in SHIFTS:
        node: dict[str, Any] = {}
        for fact in (None, *ex.definition_types(tab)):
            x = tab.vec(x_sys, b, "cv", _fact(fact))
            m = tab.vec(m_sys, b, "cv", _fact(fact))
            keys = rr._common(x, m)
            node["pooled" if fact is None else fact] = {
                "D1sx": ex.vec_mean(rr._restrict(x, keys), draws),
                "D1s": ex.vec_mean(rr._restrict(m, keys), draws),
                "difference": ex.paired_estimate(x, m, draws),
            }
        dx = tab.vec(x_sys, b, "detected", rr.changed("definition"))
        dm = tab.vec(m_sys, b, "detected", rr.changed("definition"))
        node["detection_differs"] = sum(
            dx[w][k] != dm[w][k] for w in dx if w in dm for k in dx[w] if k in dm[w]
        )
        out[b] = node
    return out


# ========================================================================================= BxD
def _dd(get: Any, arm: tuple[str, str]) -> Vec:
    return rr.double_diff(get(arm[1]), get(arm[0]), get(D0[1]), get(D0[0]))


def bxd_contrast(tab: ex.Table, draws: int = DRAWS) -> dict[str, Any]:
    """B x D with D_m on extracted and on typed fields, and their paired difference."""
    out: dict[str, Any] = {}
    for b in SHIFTS:
        for cls in CLASSES:
            get = rr.getter(tab, b, rr.changed(cls))
            keys = rr._common(_dd(get, DMX), _dd(get, DM))
            x, m = rr._restrict(_dd(get, DMX), keys), rr._restrict(_dd(get, DM), keys)
            out.setdefault(b, {})[cls] = {
                "extracted": ex.vec_mean(x, draws),
                "typed": ex.vec_mean(m, draws),
                "difference": ex.paired_estimate(x, m, draws),
                "update": {c: ex.vec_mean(get(c), draws) for c in CELLS},
            }
    return out


# ================================================================================= attribution
def _writes(items: Iterable[Row]) -> dict[tuple[str, str, str, str], list[Row]]:
    """(world, branch, system, slot) -> the write events of its items, in time order."""
    out: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for it in items:
        key = (it["world_id"], it["branch"], it["system"], it["slot_id"])
        out[key] += [h for h in it["history"] if h["event"] in WRITE_EVENTS]
    for events in out.values():
        events.sort(key=lambda h: h["ts"])  # stable: the store order within one time
    return out


def value_at(events: Sequence[Row], as_of: str) -> Row | None:
    """The slot's last write at or before `as_of`."""
    last = None
    for h in events:
        if h["ts"] <= as_of:
            last = h
    return last


class ProgramFields:
    """The field names each D_m program names (quoted), from the writer's response cache."""

    def __init__(self, path: Path | None):
        self.path = path
        self._db = sqlite3.connect(f"file:{path}?mode=ro", uri=True) if path else None
        self._memo: dict[str, frozenset[str] | None] = {}

    def __call__(self, key: str) -> frozenset[str] | None:
        if self._db is None:
            return None
        if key not in self._memo:
            row = self._db.execute("SELECT response FROM cache WHERE key=?", (key,)).fetchone()
            names: frozenset[str] | None = None
            if row is not None:
                resp = json.loads(row[0])
                res = LLMResult(str(resp.get("text") or ""), 0, 0, 0.0, True, "", key)
                code = extract_code(res)
                names = frozenset(_QUOTED.findall(code or "")) & FIELD_NAMES
            self._memo[key] = names
        return self._memo[key]

    def close(self) -> None:
        if self._db is not None:
            self._db.close()


def _same(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def attribution(
    data: Mapping[str, list[Row]], program_fields: ProgramFields | None = None
) -> dict[str, Any]:
    """Of the definition item-times where D1sx's value differs from D1s's, the share whose
    producing D1sx write read a record with an extraction error (module docstring)."""
    writes = _writes(data["items"])
    asked = {
        (a["system"], a["world_id"], a["branch"], a["probe_id"])
        for a in data["answers"]
        if a["system"] in DMX
    }
    out: dict[str, Any] = {}
    for pair, (x_sys, m_sys) in PAIRS.items():
        for b in SHIFTS:
            c: Counter[str] = Counter()
            by_def: dict[str, Counter[str]] = defaultdict(Counter)
            missing_cache = 0
            for p in data["probes"]:
                k = int((p.get("tags") or {}).get("k", 0))
                if p["branch"] != b or not k or not str(p["fact_id"]).startswith("k."):
                    continue
                if (x_sys, p["world_id"], b, p["probe_id"]) not in asked:
                    continue
                w, f = p["world_id"], p["fact_id"]
                hx = value_at(writes.get((w, b, x_sys, f), []), p["as_of"])
                hm = value_at(writes.get((w, b, m_sys, f), []), p["as_of"])
                if hx is None or hm is None:
                    c["no_write"] += 1
                    continue
                c["item_times"] += 1
                by_def[f]["item_times"] += 1
                if _same(hx.get("value"), hm.get("value")):
                    continue
                prog = hx.get("program") or {}
                note = prog.get("input") or {}
                c["differ"] += 1
                by_def[f]["differ"] += 1
                if note.get("with_error"):
                    c["differ_with_error"] += 1
                    by_def[f]["differ_with_error"] += 1
                else:
                    c["differ_without_error"] += 1
                if program_fields is None or program_fields.path is None:
                    continue
                named = program_fields(str(prog.get("key", "")))
                if named is None:
                    missing_cache += 1
                    continue
                bad = {e.split(".", 1)[1] for e in note.get("error_fields", [])}
                if bad & named:
                    c["differ_with_error_in_named_field"] += 1
                    by_def[f]["differ_with_error_in_named_field"] += 1
            n = c["differ"]
            node: dict[str, Any] = {
                "item_times": c["item_times"],
                "differ": n,
                "differ_share": n / c["item_times"] if c["item_times"] else None,
                "differ_with_error": c["differ_with_error"],
                "share_with_error": c["differ_with_error"] / n if n else None,
                "differ_without_error": c["differ_without_error"],
                "no_write": c["no_write"],
                "by_definition": {f: dict(v) for f, v in sorted(by_def.items())},
            }
            if program_fields is not None and program_fields.path is not None:
                named_n = c["differ_with_error_in_named_field"]
                node["differ_with_error_in_named_field"] = named_n
                node["share_with_error_in_named_field"] = named_n / n if n else None
                node["programs_missing_in_cache"] = missing_cache
            out.setdefault(pair, {})[b] = node
    return out


# ============================================================================ threshold tests
def threshold_tests(q: Mapping[str, Any], worlds: Iterable[str]) -> dict[str, Any]:
    present = set(worlds)
    missing = [w for w in W_WORLDS if w not in present]
    out: dict[str, Any] = {}
    w1 = [
        rr.above_zero(f"BxD_{b}_definition_extracted", q["bxd"][b]["definition"]["extracted"])
        for b in SHIFTS
    ]
    out["W1"] = rr.tested(w1)
    w2 = []
    for b in SHIFTS:
        e = q["bxd"][b]["definition"]["difference"]
        if rr._present(e):
            w2.append(
                rr._test(f"BxD_{b}_extracted_minus_typed", e["est"], e["est"] <= rr.EPS, "<= 0")
            )
        else:
            w2.append(rr._test(f"BxD_{b}_extracted_minus_typed", None, None, "<= 0"))
    out["W2"] = rr.tested(w2)
    if missing:
        out = {k: {**v, "worlds_missing": missing} for k, v in out.items()}
    return out


# ======================================================================================== block
def block_w(
    runs: Sequence[Path],
    compare: Sequence[Path] = (),
    response_cache: Path | None = None,
    draws: int = DRAWS,
) -> dict[str, Any]:
    new, skipped = rr.read_runs(runs, items=lambda s: s in (*DM, *DMX))
    newm = rr.Merged(new, rr.by_variant)
    groups = newm.groups()
    if len(groups) != 1:
        raise ValueError(f"the W runs must be one variant, got {len(groups)}")
    cmpm = None
    skipped_c: list[str] = []
    if compare:
        need = set(CELLS) | {a["system"] for r in new for a in r.answers}
        cmp_runs, skipped_c = rr.read_runs(rr.expand(compare), need.__contains__)
        cmpm = rr.Merged(cmp_runs, rr.by_family)
    scope = rr._scope_shared if cmpm is not None else None
    pair = rr.pairing(newm, cmpm, [*skipped, *skipped_c], scope)
    var = groups[0]
    tab = newm.table(var)
    data = newm.data(var)
    owners = {id(r): r for r in newm.runs_of(var)}.values()
    rows = read_extractions(r.path for r in owners)
    fields = ProgramFields(response_cache)
    try:
        q: dict[str, Any] = {
            "worlds": sorted(rr.cell_worlds(tab, CELLS)),
            "extraction": extraction_quality(rows, draws),
            "background_read": background_counts(list(owners)),
            "r": r_contrast(tab, draws),
            "bxd": bxd_contrast(tab, draws),
            "attribution": attribution(data, fields if response_cache else None),
        }
    finally:
        fields.close()
    tests = threshold_tests(q, q["worlds"])
    return {
        "block": "W",
        "status": STATUS,
        "units": UNITS,
        "inputs": {
            "runs": [str(p) for p in runs],
            "compare": [str(p) for p in compare],
            "response_cache": str(response_cache) if response_cache else None,
        },
        "pairing": pair,
        "threshold_tests": tests,
        "quantities": q,
        "code": code_state(),
    }


def summary(res: Mapping[str, Any]) -> str:
    pair = res["pairing"]
    bad = {k: v for k, v in pair.get("counts", {}).items() if v}
    lines = [
        f"block W: pairing {'ok' if pair.get('ok') else 'FAILED ' + json.dumps(bad)}"
        f" ({pair.get('anchor_answers_compared', 0)} anchor answers compared)"
    ]
    lines += rr.tests_lines(res["threshold_tests"])
    q = res["quantities"]
    for source, node in q["extraction"].items():
        lines.append(
            f"  extraction {source:8} n {node['records']:5}  correct "
            f"{rr._iv(node['record_correct'], rr._pc)}  missed {node['rewrite']['missed']}"
        )
    for b in SHIFTS:
        r = q["r"][b]["pooled"]
        d = q["bxd"][b]["definition"]
        lines.append(
            f"  {b}: r D1sx {rr._iv(r['D1sx'], rr._pc)} D1s {rr._iv(r['D1s'], rr._pc)}"
            f" diff {rr._iv(r['difference'])};  BxD extracted {rr._iv(d['extracted'])}"
            f" typed {rr._iv(d['typed'])} diff {rr._iv(d['difference'])}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        description=(__doc__ or "").split("\n\n", 2)[1],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--runs", nargs="+", required=True, type=Path)
    ap.add_argument("--compare", nargs="*", default=[], type=Path)
    ap.add_argument("--response-cache", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = block_w(a.runs, a.compare, a.response_cache, a.draws)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    print(summary(res))
    if not res["pairing"].get("ok"):
        print("PAIRING CHECK FAILED: the run is void (see pairing.counts)", file=sys.stderr)


if __name__ == "__main__":
    main()

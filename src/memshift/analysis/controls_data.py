"""Inputs of the control-cell analysis (`analysis.controls`): the new runs and the published runs
they are compared with, merged into one item table, with the integrity checks that make the merge
legitimate.

The control cells were run on worlds 1000 to 1009 with `--discriminative --no-baselines`. Each run
carries a cell that also exists in the published runs (the anchor), so the cache replay can be
checked against the publication; cells that a run does not contain (the published ones) are taken
from the published runs of the same worlds. A run covers the branches its manifest declares
(`params.branches`): the control runs cover all three, the D0full runs only src and def. Every check
below compares a run with the published runs on the branches the run declares, and a declared branch
without answers is a failure, so a run of fewer branches is neither padded nor excused:

  ctl:RUN   a new run of the control cells (raw JSONL, manifest.json, hash_check_*.json)
  base:RUN  a published run of the same worlds (raw JSONL); a stage directory (`runs/stage2_e1`)
            expands to the source runs its manifest lists, each verified against the manifest
            SHA256 recorded there

Every (cell, world) of the merged table comes from exactly one run: a published run that has it
(the first one, in the order given), else the one new run that has it. A cell of a new run that a
published run also has (an anchor) is not entered twice; it is compared with the published one:

  hash check    every pre-shift hash assertion of the new runs passed
  probe sets    the (world, branch, fact id, checkpoint) each system of a new run was asked equals
                the set the published anchor (else the published zero system) was asked on the
                run's branches, and the gold of every probe equals the published gold;
                `--discriminative` selects from the gold alone, so this must hold exactly
  anchors       the anchor's answers in the new run (status, value, ContractValid) equal the
                published anchor's on the run's branches; a share above `ANCHOR_TOLERANCE` is an
                error (the replay did not reproduce the published cell). The hash of the
                rendered context is counted and reported but not judged: two published runs of
                the same cells already differ in it for a few probes of one world (a different
                number of evidence blocks) with identical answers

A failed integrity check raises `ControlsCheckError` (after the diagnostics are returned to the
caller, which writes them next to the analysis); no number is produced from such a merge.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from memshift.analysis import e1
from memshift.experiments.common import read_jsonl
from memshift.runinfo import file_sha256

Row = e1.Row
CTL_SEEDS = tuple(range(1000, 1010))  # the worlds of the control runs
ANCHOR_TOLERANCE = 0.01  # share of an anchor's answers that may differ from the published ones
EXAMPLES = 5  # mismatches listed per check
# published cells the analysis compares the new cells with
BASE_SYSTEMS = frozenset(
    {
        "holdmem_D0B0",
        "holdmem_D1B0",
        "holdmem_D0B1_source",
        "holdmem_D1B1_source",
        "holdmem_D1B1_cited",
        "holdmem_D1sB1_source",
        "zero",
    }
)
ANSWER_FIELDS = ("status", "value", "contract_valid")  # what an anchor must reproduce
CONTEXT_FIELD = "context_sha256"  # reported, not judged (see `_anchors`)


class ControlsCheckError(RuntimeError):
    """An integrity check of the merge failed; `checks` holds the diagnostics."""

    def __init__(self, message: str, checks: dict[str, Any]):
        super().__init__(message)
        self.checks = checks


def world_id(seed: int) -> str:
    return f"w{seed:04d}"


def ctl_worlds() -> list[str]:
    return [world_id(s) for s in CTL_SEEDS]


# ---------------------------------------------------------------------------------- runs
@dataclass
class Run:
    path: Path
    manifest: dict[str, Any]
    probes: list[Row]
    answers: list[Row]
    items: list[Row]
    hash_checks: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    def systems(self) -> set[str]:
        return {a["system"] for a in self.answers}

    def worlds(self) -> set[str]:
        return {a["world_id"] for a in self.answers}


def _manifest(path: Path) -> dict[str, Any]:
    f = path / "manifest.json"
    return dict(json.loads(f.read_text())) if f.exists() else {}


def run_world_ids(manifest: dict[str, Any]) -> set[str] | None:
    """The worlds a run covers according to its manifest (None when it does not say)."""
    seeds = (manifest.get("params") or {}).get("seeds")
    if seeds:
        return {world_id(int(s)) for s in seeds}
    names = {w.get("world_id") for w in manifest.get("worlds", []) if w.get("world_id")}
    return {str(n) for n in names} or None


def run_branches(run: Run) -> set[str]:
    """The branches a run covers: those its manifest declares (`params.branches`), else those it
    has answers for. A declared branch that has no answers is then a missing part of the run, which
    the probe-set and anchor checks report, not a branch the run did not cover."""
    declared = (run.manifest.get("params") or {}).get("branches")
    return {str(b) for b in declared} if declared else {a["branch"] for a in run.answers}


def load_run(path: Path, worlds: set[str], systems: set[str] | None = None) -> Run:
    """A run restricted to `worlds` (and, for published runs, to the `systems` of interest)."""

    def read(name: str) -> list[Row]:
        f = path / name
        rows = read_jsonl(f) if f.exists() else []
        rows = [r for r in rows if r["world_id"] in worlds]
        if systems is not None and name != "probes.jsonl":
            rows = [r for r in rows if r["system"] in systems]
        return rows

    checks = {}
    for f in sorted(path.glob("hash_check_*.json")):
        if any(f.name.startswith(f"hash_check_{w}_") for w in worlds):
            checks[f.name] = json.loads(f.read_text())
    return Run(
        path,
        _manifest(path),
        read("probes.jsonl"),
        read("answers.jsonl"),
        read("items.jsonl"),
        checks,
    )


def expand_base(paths: Iterable[Path], worlds: set[str] | None = None) -> list[Path]:
    """Raw run directories as given; a stage directory (no `probes.jsonl`, a manifest with
    `sources`) expands to its source runs next to it. Each source's manifest must still hash to
    the SHA256 the stage recorded, so that the published runs are the ones the stage merged. With
    `worlds`, the sources of the stage that cover none of them (other test worlds) are left out
    and need not be on this machine."""
    out: list[Path] = []
    for p in paths:
        if (p / "probes.jsonl").exists():
            out.append(p)
            continue
        man = _manifest(p)
        if not man.get("sources"):
            raise FileNotFoundError(f"{p}: neither a run (probes.jsonl) nor a stage (sources)")
        for src in man["sources"]:
            if worlds is not None and src.get("worlds") and not worlds & set(src["worlds"]):
                continue
            run = p.parent / src["run_id"]
            if not (run / "probes.jsonl").exists():
                raise FileNotFoundError(f"{run}: source run of {p.name} has no raw JSONL here")
            want = src.get("manifest_sha256")
            if want and file_sha256(run / "manifest.json") != want:
                raise ValueError(f"{run}: manifest differs from the one {p.name} was merged from")
            out.append(run)
    return list(dict.fromkeys(out))


# ------------------------------------------------------------------------------- the merge
@dataclass
class Table:
    """The item table of the merged cells and what the notes-based quantities need."""

    rows: list[Row]
    items: list[Row]  # merged items.jsonl rows (histories carry the program and think notes)
    k0: dict[str, dict[str, dict[str, dict[str, float]]]]  # system, branch, world -> fact -> cv
    worlds: list[str]
    origin: dict[str, dict[str, str]]  # system -> world -> the run its rows come from
    months: int = 12
    checks: dict[str, Any] = field(default_factory=dict)


def _asked(rows: Iterable[Row]) -> dict[str, set[tuple[str, str, str, int]]]:
    out: dict[str, set[tuple[str, str, str, int]]] = defaultdict(set)
    for a in rows:
        out[a["system"]].add(e1._key(a))
    return out


def _sample(keys: Iterable[Any]) -> list[Any]:
    return sorted(keys)[:EXAMPLES]


def _hash_check(ctl: list[Run]) -> dict[str, Any]:
    per_run: dict[str, dict[str, Any]] = {}
    for run in ctl:
        bad = [n for n, c in run.hash_checks.items() if not c.get("ok")]
        per_run[run.name] = {"files": len(run.hash_checks), "failed": bad}
    ok = all(v["files"] > 0 and not v["failed"] for v in per_run.values())
    return {"pass": ok, "runs": per_run}


def _discriminative(runs: list[Run]) -> dict[str, Any]:
    flags = {
        r.name: (r.manifest.get("params") or {}).get("probes_after_fork")
        for r in runs
        if r.manifest.get("params")
    }
    return {"pass": all(v == "discriminative" for v in flags.values()), "runs": flags}


def _probe_sets(
    ctl: list[Run], base: list[Run], claimed: dict[tuple[str, str], Run]
) -> dict[str, Any]:
    """Asked sets and gold of each new run against the published runs (exact equality)."""
    base_gold: dict[str, Row] = {}
    for run in base:
        for p in run.probes:
            base_gold.setdefault(p["probe_id"], p)
    base_asked = {run.name: _asked(run.answers) for run in base}
    per_run: dict[str, Any] = {}
    ok = True
    for run in ctl:
        worlds = run.worlds()
        branches = run_branches(run)
        pairs = {(w, b) for w in worlds for b in branches}  # (world, branch) the run covers
        asked = _asked(run.answers)
        ref_system = next(
            (s for s in sorted(asked) if any((s, w) in claimed for w in worlds)), None
        )
        if ref_system is None and any(("zero", w) in claimed for w in worlds):
            ref_system = "zero"
        ref: set[tuple[str, str, str, int]] | None = None
        if ref_system is not None:
            ref = {
                k
                for (s, w), b in claimed.items()
                if s == ref_system and w in worlds
                for k in base_asked[b.name].get(s, set())
                if k[0] == w and (k[0], k[1]) in pairs
            }
        systems: dict[str, Any] = {}
        for s, keys in sorted(asked.items()):
            mine = set(keys)
            if ref is None:
                systems[s] = {"pass": False, "reason": "no published cell to compare with"}
                ok = False
                continue
            missing, extra = ref - mine, mine - ref
            systems[s] = {
                "pass": not missing and not extra,
                "asked": len(mine),
                "missing": len(missing),
                "extra": len(extra),
                "examples": [list(k) for k in _sample(missing | extra)],
            }
            ok = ok and systems[s]["pass"]
        gold_diff = []
        gold_missing = 0
        for p in run.probes:
            q = base_gold.get(p["probe_id"])
            if q is None:
                gold_missing += 1
            elif (p["gold_value"], p["gold_status"], p.get("asked")) != (
                q["gold_value"],
                q["gold_status"],
                q.get("asked"),
            ):
                gold_diff.append(p["probe_id"])
        gold_ok = not gold_diff and not gold_missing
        ok = ok and gold_ok
        per_run[run.name] = {
            "branches": sorted(branches),
            "reference_system": ref_system,
            "reference_asked": None if ref is None else len(ref),
            "systems": systems,
            "gold": {
                "pass": gold_ok,
                "probes": len(run.probes),
                "missing_in_published": gold_missing,
                "differ": len(gold_diff),
                "examples": _sample(gold_diff),
            },
        }
    return {"pass": ok, "runs": per_run}


def _anchors(
    ctl: list[Run], claimed: dict[tuple[str, str], Run], tolerance: float
) -> dict[str, Any]:
    """Anchor answers of each new run against the published answers of the same cell."""
    pub: dict[tuple[str, str], dict[str, Row]] = defaultdict(dict)
    for run in {r.path: r for r in claimed.values()}.values():
        for a in run.answers:
            if claimed.get((a["system"], a["world_id"])) is run:
                pub[a["system"], a["world_id"]][a["probe_id"]] = a
    per_run: dict[str, Any] = {}
    ok = True
    for run in ctl:
        branches = run_branches(run)
        cells: dict[str, Any] = {}
        mine: dict[tuple[str, str], dict[str, Row]] = defaultdict(dict)
        for a in run.answers:
            if (a["system"], a["world_id"]) in pub:
                mine[a["system"], a["world_id"]][a["probe_id"]] = a
        for system in sorted({s for s, _ in mine}):
            n = missing = context_differ = 0
            diff: dict[str, int] = dict.fromkeys(ANSWER_FIELDS, 0)
            differing = 0
            examples: list[str] = []
            for (s, w), answers in sorted(mine.items()):
                if s != system:
                    continue
                theirs = {pid: a for pid, a in pub[s, w].items() if a["branch"] in branches}
                for pid in sorted(answers.keys() | theirs.keys()):
                    n += 1
                    if pid not in answers or pid not in theirs:
                        missing += 1
                        differing += 1
                        examples.append(pid)
                        continue
                    bad = [f for f in ANSWER_FIELDS if answers[pid].get(f) != theirs[pid].get(f)]
                    context_differ += answers[pid].get(CONTEXT_FIELD) != theirs[pid].get(
                        CONTEXT_FIELD
                    )
                    for f in bad:
                        diff[f] += 1
                    if bad:
                        differing += 1
                        examples.append(pid)
            share = differing / n if n else 1.0
            cell_ok = n > 0 and share <= tolerance
            ok = ok and cell_ok
            cells[system] = {
                "pass": cell_ok,
                "answers": n,
                "differ": differing,
                "share": share,
                "missing_on_one_side": missing,
                "by_field": diff,
                "context_differ": context_differ,
                "context_share": context_differ / n if n else None,
                "examples": examples[:EXAMPLES],
            }
        per_run[run.name] = cells
    return {"pass": ok, "tolerance": tolerance, "runs": per_run}


def build(
    ctl_dirs: list[Path],
    base_dirs: list[Path],
    worlds: list[str] | None = None,
    tolerance: float = ANCHOR_TOLERANCE,
) -> Table:
    """Merge the new runs with the published runs into one item table, after the integrity checks.

    `worlds` restricts every run to those worlds (default: `CTL_SEEDS`, 1000 to 1009); the ones
    a new run does not have are listed in `checks["worlds_missing"]`. Raises `ControlsCheckError`
    when a probe set, a gold value, a hash assertion or an anchor differs."""
    want = set(worlds or ctl_worlds())
    ctl = [load_run(p, want) for p in ctl_dirs]
    ctl = [r for r in ctl if r.answers]
    if not ctl:
        raise ValueError(f"no answers of the new runs in worlds {sorted(want)[:3]}...")
    table_worlds = sorted({w for r in ctl for w in r.worlds()})
    wanted_systems = set(BASE_SYSTEMS) | {s for r in ctl for s in r.systems()}
    base: list[Run] = []
    for p in base_dirs:
        ids = run_world_ids(_manifest(p))
        if ids is not None and not ids & set(table_worlds):
            continue  # a published run of other worlds (e.g. 1012 to 1019)
        base.append(load_run(p, set(table_worlds), wanted_systems))
    claimed: dict[tuple[str, str], Run] = {}
    for run in base:
        for a in run.answers:
            claimed.setdefault((a["system"], a["world_id"]), run)
    checks: dict[str, Any] = {
        "worlds": table_worlds,
        "worlds_missing": sorted(want - set(table_worlds)),
        "hash": _hash_check(ctl),
        "discriminative": _discriminative([*ctl, *base]),
        "probe_sets": _probe_sets(ctl, base, claimed),
        "anchors": _anchors(ctl, claimed, tolerance),
    }
    failed = [
        k for k in ("hash", "discriminative", "probe_sets", "anchors") if not checks[k]["pass"]
    ]
    if failed:
        raise ControlsCheckError(
            "integrity check failed: " + ", ".join(failed) + " (see checks)", checks
        )

    # merge: one run per (system, world)
    owner: dict[tuple[str, str], Run] = dict(claimed)
    for run in ctl:
        for system, w in sorted({(a["system"], a["world_id"]) for a in run.answers}):
            if (system, w) in claimed:
                continue  # an anchor: already compared, the published rows are used
            if (system, w) in owner and owner[system, w] is not run:
                raise ValueError(
                    f"{system} in {w} is in both {owner[system, w].name} and {run.name}, "
                    "and no published run has it"
                )
            owner[system, w] = run
    data: dict[str, list[Row]] = {"probes": [], "answers": [], "items": []}
    seen_probes: set[str] = set()
    for run in [*base, *ctl]:
        for probe in run.probes:
            if probe["probe_id"] not in seen_probes:
                seen_probes.add(probe["probe_id"])
                data["probes"].append(probe)
        for key, rows in (("answers", run.answers), ("items", run.items)):
            data[key] += [r for r in rows if owner.get((r["system"], r["world_id"])) is run]
    origin: dict[str, dict[str, str]] = defaultdict(dict)
    for (system, w), run in owner.items():
        origin[system][w] = run.name
    months_set = {
        int(((r.manifest.get("params") or {}).get("world_params") or {}).get("months", 12))
        for r in ctl
    }
    if len(months_set) != 1:
        raise ValueError(f"new runs differ in world length: {sorted(months_set)}")
    rows = e1.item_table(data)
    _enrich(rows, data)
    return Table(
        rows=rows,
        items=data["items"],
        k0=_k0(data["answers"]),
        worlds=table_worlds,
        origin={s: dict(v) for s, v in origin.items()},
        months=months_set.pop(),
        checks=checks,
    )


def _enrich(rows: list[Row], data: dict[str, list[Row]]) -> None:
    """Fields the control analyses need beyond `e1.item_table`: the gold status of the probe and,
    for an item repaired at the checkpoint, whether the repaired value is None (a withdrawn
    value)."""
    gold = {e1._key(p): p for p in data["probes"]}
    hist: dict[tuple[str, str, str, str], list[Row]] = defaultdict(list)
    for it in data["items"]:
        hist[(it["world_id"], it["branch"], it["system"], it["slot_id"])].extend(it["history"])
    for r in rows:
        p = gold[(r["world_id"], r["branch"], r["fact_id"], r["k"])]
        r["gold_status"] = p["gold_status"]
        if r.get("repaired_now"):
            now = [
                h
                for h in hist[(r["world_id"], r["branch"], r["system"], r["fact_id"])]
                if h["event"] == "repair" and h["ts"] == p["as_of"]
            ]
            r["repair_none"] = now[-1]["value"] is None


def _k0(answers: list[Row]) -> dict[str, dict[str, dict[str, dict[str, float]]]]:
    out: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    for a in answers:
        if int(a["tags"].get("k", 0)) != 0 or a.get("contract_valid") is None:
            continue
        out.setdefault(a["system"], {}).setdefault(a["branch"], {}).setdefault(a["world_id"], {})[
            a["fact_id"]
        ] = float(a["contract_valid"])
    return out

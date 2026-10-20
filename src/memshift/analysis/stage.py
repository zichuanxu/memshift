"""Merged analyses across runs (e.g. the two halves of the development set plus supplements).

A stage directory `runs/<name>/` holds only `analysis/` and a `manifest.json` listing every source
run with the SHA256 of that run's own manifest, so a merged number can be traced to the runs (and
their code commits, worlds and model revisions) it came from.

Usage: python -m memshift.analysis.stage e1 runs/stage1_e1 RUN_DIR [RUN_DIR ...]
       python -m memshift.analysis.stage m2 runs/stage4_m2_X RUN_DIR [RUN_DIR ...]
       python -m memshift.analysis.stage e7 runs/stage3_e7 e7:RUN e1:RUN theta_r:RUN theta_e:RUN
       python -m memshift.analysis.stage e5 runs/stage3_e5 e5:RUN e1:RUN noise:RUN base:RUN
       python -m memshift.analysis.stage engine runs/stage3_theta_r_engine old:RUN new:RUN
       python -m memshift.analysis.stage ctl runs/stage5_controls ctl:RUN [ctl:RUN ...] \\
           base:RUN_OR_STAGE [base:RUN_OR_STAGE ...]
(`ctl`: the control cells; `ctl:` runs are the new runs, `base:` runs (or a
stage directory, which expands to the runs its manifest lists) the published runs they are compared
with, see `analysis.controls`.
E7 sources carry their role: E7 runs, the main E1 runs, and the reader / embedding swaps; the
readers of several `theta_r` runs are told apart by the reader model in each run's manifest
(block M3: `theta_r_by_reader`). E5 sources: native runs, white-box E1 runs on the same worlds,
second ingestions, baseline runs; engine sources: the same runs served by two engine versions.
`m2` is `e1` plus the checks (i)-(v) of `analysis.families` for one model family, written to
analysis/families.json.)
Source run directories must contain the raw JSONL (fetch them from the compute host first); only
their manifest.json is referenced, not copied.
"""

from __future__ import annotations

import importlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memshift.runinfo import code_state, file_sha256


def write_stage_manifest(out: Path, kind: str, sources: list[Path], extra: dict[str, Any]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    srcs = []
    for s in sources:
        m = s / "manifest.json"
        rec: dict[str, Any] = {"run_id": s.name}
        if m.exists():
            man = json.loads(m.read_text())
            rec.update(
                {
                    "manifest_sha256": file_sha256(m),
                    "code": man.get("code"),
                    "experiment": man.get("experiment"),
                    "worlds": [w.get("world_id") for w in man.get("worlds", [])],
                }
            )
        srcs.append(rec)
    files = {
        str(p.relative_to(out)): file_sha256(p)
        for p in sorted((out / "analysis").rglob("*"))
        if p.is_file()
    }
    manifest = {
        "run_id": out.name,
        "experiment": f"merged-{kind}",
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "analysis_code": code_state(),
        "sources": srcs,
        "files_sha256": files,
        **extra,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    kind, out, *srcs = sys.argv[1:]
    roles = [s.split(":", 1) if kind in ("e5", "e7", "engine", "ctl") else ["", s] for s in srcs]
    sources = [Path(path) for _, path in roles]
    out_dir = Path(out)
    extra: dict[str, Any] = {}
    if kind == "ctl":
        from memshift.analysis import controls

        ctl, base = controls.stage_sources(roles)
        controls.analyse(ctl, base, out_dir)
        sources = [*ctl, *base]
        extra = {"roles": {"ctl": [p.name for p in ctl], "base": [p.name for p in base]}}
    elif kind == "engine":
        from memshift.analysis import engine_check

        engine_check.analyse(
            [Path(p) for r, p in roles if r == "old"],
            [Path(p) for r, p in roles if r == "new"],
            out_dir,
        )
    elif kind == "e5":
        by5: dict[str, list[Path]] = {"e5": [], "e1": [], "noise": [], "base": []}
        for role, path in roles:
            by5[role].append(Path(path))
        from memshift.analysis import e5

        e5.analyse(by5["e5"], by5["e1"], out_dir, by5["noise"], by5["base"])
    elif kind == "e7":
        by: dict[str, list[Path]] = {"e7": [], "e1": [], "theta_r": [], "theta_e": []}
        for role, path in roles:
            by[role].append(Path(path))
        from memshift.analysis import e7

        e7.analyse(by["e7"], by["e1"], out_dir, by["theta_r"], by["theta_e"])
    elif kind in ("e1", "m2", "e6", "e5_infer", "e5_cost"):
        module = "e1" if kind == "m2" else kind
        importlib.import_module(f"memshift.analysis.{module}").analyse(*sources, out=out_dir)
        if kind == "m2":
            from memshift.analysis import families

            families.analyse(out_dir, {out_dir.name: out_dir})
    elif kind in ("e2", "e3", "e4", "e4x"):
        import tempfile

        merged = Path(tempfile.mkdtemp(prefix=f"memshift_{kind}_"))
        names = {p.name for s in sources for p in s.glob("*.jsonl") if p.name != "calls.jsonl"}
        for name in names:
            with (merged / name).open("w") as f:
                for s in sources:
                    if (s / name).exists():
                        f.write((s / name).read_text())
        importlib.import_module(f"memshift.analysis.{kind}").analyse(merged)
        (out_dir / "analysis").mkdir(parents=True, exist_ok=True)
        for p in (merged / "analysis").glob("*"):
            (out_dir / "analysis" / p.name).write_bytes(p.read_bytes())
    else:
        raise SystemExit(f"unknown kind {kind}")
    write_stage_manifest(out_dir, kind, sources, extra)
    print(out_dir)


if __name__ == "__main__":
    main()

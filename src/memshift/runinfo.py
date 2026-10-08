"""Run provenance (plan §8): every run writes `manifest.json` (world ids, seeds, model ids, image
ids, parameters, code commit, SHA256 of every output file) and `env.json` (host facts, serving
images, model revisions), following reference/research_1/results/manifest.json.

Serving metadata is read from `logs/serving/*.json`, written when a model container is launched.
Host names and addresses are deliberately not recorded.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from memshift.gen.model import World
from memshift.textutil import sha256_json

REPO = Path(os.environ.get("MEMSHIFT_REPO", Path(__file__).resolve().parents[2]))


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def code_state() -> dict[str, Any]:
    commit = os.environ.get("MEMSHIFT_CODE_COMMIT")
    dirty = os.environ.get("MEMSHIFT_CODE_DIRTY")
    if commit is None:
        try:
            commit = subprocess.run(
                ["git", "-C", str(REPO), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            porcelain = subprocess.run(
                ["git", "-C", str(REPO), "status", "--porcelain", "--", "src", "configs"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            dirty = "1" if porcelain.strip() else "0"
        except (OSError, subprocess.CalledProcessError):
            commit, dirty = "unknown", "1"
    return {"commit": commit, "dirty": dirty == "1"}


def world_record(world: World) -> dict[str, Any]:
    """The hash covers every event payload including `text`, so an LLM-rendered world
    (gen/llm_render.py) hashes differently from its template world; `render` names the renderer."""
    return {
        "world_id": world.world_id,
        "seed": world.seed,
        "branch": world.branch,
        "n_events": len(world.events),
        "events_sha256": sha256_json([e.to_json() for e in world.events]),
        "shift_targets": world.params.get("shift_targets"),
        "kappa_log": [(str(t), k, v) for t, k, v in world.kappa_log],
        "consent_log": [(str(t), a, s, st) for t, a, s, st in world.consent_log],
        **({"render": world.params["render"]} if "render" in world.params else {}),
    }


def cache_tags() -> dict[str, str]:
    """Optional response-cache namespaces per model (MEMSHIFT_CACHE_TAGS, JSON model id -> tag).
    The cache key is the request alone, so a rerun of the same requests against a different serving
    engine sets a tag to get fresh responses instead of the first engine's; untagged models keep
    the shared cache that makes cells and reruns replayable."""
    return {
        str(k): str(v) for k, v in json.loads(os.environ.get("MEMSHIFT_CACHE_TAGS") or "{}").items()
    }


def serving_info() -> dict[str, Any]:
    out: dict[str, Any] = {}
    d = REPO / "logs" / "serving"
    if d.is_dir():
        for p in sorted(d.glob("*.json")):
            try:
                rec = json.loads(p.read_text())
            except json.JSONDecodeError:
                continue
            out[rec.get("served_model", p.stem)] = rec
    return out


def write_env(run_dir: Path, models: dict[str, str]) -> None:
    serving = serving_info()
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "run_image": os.environ.get("MEMSHIFT_RUN_IMAGE", "unknown"),
        "models": {role: {"model": m, **serving.get(m, {})} for role, m in models.items()},
    }
    (run_dir / "env.json").write_text(json.dumps(env, indent=2, sort_keys=True))


def write_manifest(
    run_dir: Path,
    experiment: str,
    worlds: list[dict[str, Any]],
    models: dict[str, str],
    params: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    serving = serving_info()
    tags = cache_tags()
    files = {
        str(p.relative_to(run_dir)): file_sha256(p)
        for p in sorted(run_dir.rglob("*"))
        if p.is_file() and p.name != "manifest.json"
    }
    manifest = {
        "run_id": run_dir.name,
        "experiment": experiment,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code": code_state(),
        "worlds": worlds,
        "models": {
            role: {
                "model": m,
                "revision": serving.get(m, {}).get("revision"),
                "image_id": serving.get(m, {}).get("image_id"),
                **({"cache_tag": tags[m]} if m in tags else {}),
            }
            for role, m in models.items()
        },
        "run_image": os.environ.get("MEMSHIFT_RUN_IMAGE", "unknown"),
        "params": params,
        "files_sha256": files,
        **(extra or {}),
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def verify_manifest(run_dir: Path) -> list[str]:
    """Return the relative paths whose current SHA256 differs from the manifest."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    return [
        rel
        for rel, digest in manifest["files_sha256"].items()
        if not (run_dir / rel).is_file() or file_sha256(run_dir / rel) != digest
    ]

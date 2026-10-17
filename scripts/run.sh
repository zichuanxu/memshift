#!/usr/bin/env bash
# Run one experiment driver (`memshift <args>`) in a resource-limited CPU container: host UID,
# code mounted read-only from an immutable checkout of the current commit, outputs to runs/, logs
# archived to logs/runs/, container removed after its logs and state are saved.
# Env: MEMSHIFT_ROOT, MEMSHIFT_RUNS, MEMSHIFT_PREFIX, MEMSHIFT_RUN_IMAGE, MEMSHIFT_ENDPOINTS (JSON),
#      MEMSHIFT_CACHE_TAGS (JSON model -> response-cache namespace), RUN_CPUS,
#      RUN_MODULE (default memshift.cli; e.g. memshift.experiments.e1), RUN_TAG (name suffix),
#      MEMSHIFT_HTTP_TIMEOUT (seconds per model request; default 600), MEMSHIFT_ANTHROPIC_COLLECT
#      (a file under runs/: Claude requests that miss the cache are recorded, not sent; memshift.llm),
#      CLAUDE_API_KEY and MEMSHIFT_CLAUDE_CONCURRENCY for claude-* models: passed to the container
#      by name only, so the key's value is never on a command line or on disk (only the
#      container's .State is saved below)
set -euo pipefail
ROOT=${MEMSHIFT_ROOT:-$HOME/memshift}
RUNS=${MEMSHIFT_RUNS:-$ROOT/runs}  # run outputs; keep outside the git work tree on the compute host
PREFIX=${MEMSHIFT_PREFIX:-memshift}
IMAGE=${MEMSHIFT_RUN_IMAGE:?set MEMSHIFT_RUN_IMAGE}
CPUS=${RUN_CPUS:-8}
ENDPOINTS=${MEMSHIFT_ENDPOINTS:-"{}"}
CACHE_TAGS=${MEMSHIFT_CACHE_TAGS:-"{}"}
NAME="${PREFIX}_run_$(date -u +%Y%m%d_%H%M%S)_${RUN_TAG:-$1}"
COMMIT=$(git -C "$ROOT" rev-parse HEAD)
DIRTY=$([ -z "$(git -C "$ROOT" status --porcelain -- src configs)" ] && echo 0 || echo 1)
mkdir -p "$RUNS" "$ROOT/cache" "$ROOT/logs/runs" "$ROOT/worlds" "$ROOT/.snapshots"
# Run from an immutable checkout of the commit, so later pushes cannot change a live run.
SNAP="$ROOT/.snapshots/$COMMIT"
(
  flock 9
  [ -d "$SNAP" ] || git -C "$ROOT" worktree add --detach -q "$SNAP" "$COMMIT"
) 9>"$ROOT/.snapshots/.lock"
for d in runs cache worlds logs; do mkdir -p "$SNAP/$d"; done
docker run -d --name "$NAME" \
  --label owner="${USER}" --label purpose=experiment --label project=memshift \
  --cpus="$CPUS" --memory=48g --memory-swap=48g --shm-size=1g --pids-limit=1024 \
  --network host --user "$(id -u):$(id -g)" \
  -e OMP_NUM_THREADS=2 -e MKL_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e NUMEXPR_NUM_THREADS=2 \
  -e NVIDIA_VISIBLE_DEVICES=void -e HOME=/tmp -e PYTHONPATH=/work/src -e MEMSHIFT_REPO=/work \
  -e MEMSHIFT_CODE_COMMIT="$COMMIT" -e MEMSHIFT_CODE_DIRTY="$DIRTY" \
  -e MEMSHIFT_RUN_IMAGE="$(docker image inspect --format '{{.Id}}' "$IMAGE")" \
  -e MEMSHIFT_ENDPOINTS="$ENDPOINTS" -e MEMSHIFT_CACHE_TAGS="$CACHE_TAGS" \
  -e CLAUDE_API_KEY -e MEMSHIFT_CLAUDE_CONCURRENCY -e MEMSHIFT_HTTP_TIMEOUT -e MEMSHIFT_ANTHROPIC_COLLECT \
  --mount "type=bind,src=$SNAP,dst=/work,readonly" \
  --mount "type=bind,src=$ROOT/logs,dst=/work/logs,readonly" \
  --mount "type=bind,src=$RUNS,dst=/work/runs" \
  --mount "type=bind,src=$ROOT/cache,dst=/work/cache" \
  --mount "type=bind,src=$ROOT/worlds,dst=/work/worlds" \
  -w /work --entrypoint python "$IMAGE" -u -m "${RUN_MODULE:-memshift.cli}" "$@" >/dev/null
echo "$NAME"
code=$(docker wait "$NAME")
docker logs --timestamps "$NAME" >"$ROOT/logs/runs/$NAME.out" 2>"$ROOT/logs/runs/$NAME.err"
docker inspect --format '{{json .State}}' "$NAME" >"$ROOT/logs/runs/$NAME.state.json"
docker container rm "$NAME" >/dev/null
echo "exit=$code"
exit "$code"

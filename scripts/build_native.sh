#!/usr/bin/env bash
# Build the image of one native memory system of E5 on top of the CPU run image
# (docker/Dockerfile.native). Prints the image tag; run a driver with it through run.sh by setting
# MEMSHIFT_RUN_IMAGE to that tag.
# Usage: build_native.sh SYSTEM   (a file docker/native/SYSTEM.txt must exist)
set -euo pipefail
SYSTEM=${1:?system}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BASE=${MEMSHIFT_RUN_IMAGE:?set MEMSHIFT_RUN_IMAGE}
PREFIX=${MEMSHIFT_PREFIX:-memshift}
[ -f "$ROOT/docker/native/$SYSTEM.txt" ] || { echo "no docker/native/$SYSTEM.txt" >&2; exit 2; }
case "$SYSTEM" in
  hipporag) FLAGS="--no-deps" ;;  # its metadata pins conflict with the lock (see the file header)
  *) FLAGS="" ;;
esac
TAG="${PREFIX}_native_${SYSTEM}:$(git -C "$ROOT" rev-parse --short HEAD)"
docker build -q -f "$ROOT/docker/Dockerfile.native" --build-arg BASE="$BASE" \
  --build-arg SYSTEM="$SYSTEM" --build-arg PIP_FLAGS="$FLAGS" \
  --label project=memshift --label owner="${USER}" -t "$TAG" "$ROOT/docker" >/dev/null
# offline import check: the adapter module, and its `import_<system>()` when it has one
docker run --rm --network none --user "$(id -u):$(id -g)" -e HOME=/tmp -v "$ROOT:/work:ro" \
  -e PYTHONPATH=/work/src "$TAG" python -c "
import importlib
m = importlib.import_module('memshift.adapters.native_$SYSTEM')
f = getattr(m, 'import_$SYSTEM', None)
f and f()
" >/dev/null
echo "$TAG"

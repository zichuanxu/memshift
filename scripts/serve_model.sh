#!/usr/bin/env bash
# Launch one vLLM OpenAI server and record its metadata
# (image id, model revision, engine flags) in logs/serving/<model>.json for run manifests.
# Usage: serve_model.sh MODEL REVISION GPU(S) PORT [extra vllm args...]
# Env:   MEMSHIFT_ROOT (repo on the host), HF_CACHE_DIR (read-only model cache),
#        MEMSHIFT_PREFIX (container name prefix), VLLM_IMAGE, MEM_UTIL, MAX_LEN, MAX_SEQS,
#        VLLM_SHIM (optional host directory with a sitecustomize.py, mounted read-only and put on
#        PYTHONPATH; import-compatibility shims only, see scripts/vllm_shims/)
set -euo pipefail
MODEL=${1:?model}; REVISION=${2:?revision}; GPUS=${3:?gpu}; PORT=${4:?port}; shift 4
ROOT=${MEMSHIFT_ROOT:-$HOME/memshift}
HF_CACHE_DIR=${HF_CACHE_DIR:-$HOME/.cache/huggingface}
PREFIX=${MEMSHIFT_PREFIX:-memshift}
IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}
SLUG=$(echo "$MODEL" | tr '/.' '__' | tr 'A-Z' 'a-z')
NAME="${PREFIX}_${SLUG}_$(date -u +%Y%m%d%H%M)"
TP=$(awk -F, '{print NF}' <<<"$GPUS")
SHIM_ARGS=()
SHIM_SHA=""
if [ -n "${VLLM_SHIM:-}" ]; then
  SHIM_ARGS=(--mount "type=bind,src=${VLLM_SHIM},dst=/opt/memshift_shim,readonly"
    -e PYTHONPATH=/opt/memshift_shim)
  SHIM_SHA=$(sha256sum "${VLLM_SHIM}/sitecustomize.py" | cut -d' ' -f1)
fi
docker run -d --name "$NAME" \
  --label owner="${USER}" --label purpose=vllm --label project=memshift \
  --cpus=8 --memory=64g --memory-swap=64g --shm-size=8g --pids-limit=2048 \
  --gpus "\"device=${GPUS}\"" \
  -e OMP_NUM_THREADS=4 -e MKL_NUM_THREADS=4 -e OPENBLAS_NUM_THREADS=4 -e NUMEXPR_NUM_THREADS=4 \
  -e HF_HUB_OFFLINE=1 \
  -p "127.0.0.1:${PORT}:8000" \
  --mount "type=bind,src=${HF_CACHE_DIR},dst=/root/.cache/huggingface,readonly" \
  ${SHIM_ARGS[@]+"${SHIM_ARGS[@]}"} \
  "$IMAGE" \
  --model "$MODEL" --revision "$REVISION" --served-model-name "$MODEL" --dtype bfloat16 \
  --tensor-parallel-size "$TP" --max-model-len "${MAX_LEN:-16384}" \
  --max-num-seqs "${MAX_SEQS:-48}" --gpu-memory-utilization "${MEM_UTIL:-0.85}" \
  --enable-prefix-caching "$@" >/dev/null
IMAGE_ID=$(docker image inspect --format '{{.Id}}' "$IMAGE")
mkdir -p "$ROOT/logs/serving"
python3 - "$ROOT/logs/serving/${SLUG}.json" <<PY
import json, sys, datetime
json.dump({"served_model": "$MODEL", "revision": "$REVISION", "image": "$IMAGE",
           "image_id": "$IMAGE_ID", "gpus": "$GPUS", "tensor_parallel": $TP, "port": $PORT,
           "extra_args": "$*", "max_model_len": ${MAX_LEN:-16384},
           "gpu_memory_utilization": ${MEM_UTIL:-0.85},
           "shim_sitecustomize_sha256": "$SHIM_SHA" or None,
           "launched_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")},
          open(sys.argv[1], "w"), indent=2)
PY
echo "$NAME"

#!/usr/bin/env bash
# Serve one model of the other families (configs/models.yaml `stage4_families`) with its pinned
# revision, model cache and engine flags, through serve_model.sh (one GPU, vLLM 0.30.0 main image).
# Usage: serve_stage4.sh MODEL GPU PORT
set -euo pipefail
MODEL=${1:?model}; GPU=${2:?gpu}; PORT=${3:?port}
DIR=$(cd "$(dirname "$0")" && pwd)
SHARED=$HOME/.cache/huggingface      # read-only shared cache (models present before the project)
PROJECT=$HOME/memshift_hf_cache      # project cache (models downloaded for MemShift)
export MEM_UTIL=${MEM_UTIL:-0.90} MAX_LEN=${MAX_LEN:-16384} MAX_SEQS=${MAX_SEQS:-48}
case "$MODEL" in
  meta-llama/Llama-3.1-8B-Instruct)
    HF_CACHE_DIR=$SHARED REV=0e9e39f249a16976918f6564b8830bc894c89659 ARGS=() ;;
  google/gemma-4-12B-it)
    HF_CACHE_DIR=$PROJECT REV=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
    ARGS=(--language-model-only) ;;
  openai/gpt-oss-20b)
    HF_CACHE_DIR=$SHARED REV=6cee5e81ee83917806bbde320786a8fb61efebee
    ARGS=(--reasoning-parser openai_gptoss) ;;
  Qwen/Qwen3-8B)
    HF_CACHE_DIR=$PROJECT REV=b968826d9c46dd6066d109eabc6255188de91218
    ARGS=(--reasoning-parser qwen3) ;;
  mistralai/Ministral-3-14B-Instruct-2512-BF16)
    HF_CACHE_DIR=$SHARED REV=3cea74c1ebaf5ce5f5a2553de470e2ceab825142
    export VLLM_SHIM=${VLLM_SHIM:-$DIR/vllm_shims}
    ARGS=(--tokenizer-mode mistral --config-format mistral --load-format mistral
          --language-model-only) ;;
  XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B)
    HF_CACHE_DIR=$PROJECT REV=2367e865d009c13ac81713a2878291d33ab28177
    ARGS=(--language-model-only --reasoning-parser qwen3) ;;
  *) echo "unknown stage-4 model: $MODEL" >&2; exit 2 ;;
esac
export HF_CACHE_DIR
exec bash "$DIR/serve_model.sh" "$MODEL" "$REV" "$GPU" "$PORT" "${ARGS[@]+"${ARGS[@]}"}"

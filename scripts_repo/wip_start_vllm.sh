#!/usr/bin/env bash
# Launch vLLM with the right reasoning-parser flags. Reads .env for the
# model, port, and parser. If VLLM_REASONING_PARSER is unset, vLLM is
# launched without --enable-reasoning (use this for non-reasoning models).
#
# Override individual values inline, e.g.:
#   VLLM_REASONING_PARSER=qwen3 ./scripts/start_vllm.sh
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

: "${VLLM_MODEL_ID:?Set VLLM_MODEL_ID in .env or environment}"
PORT="${VLLM_PORT:-8000}"

ARGS=(serve "$VLLM_MODEL_ID" --port "$PORT")

if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
  ARGS+=(--enable-reasoning --reasoning-parser "$VLLM_REASONING_PARSER")
fi

if [[ -n "${VLLM_DTYPE:-}" ]]; then
  ARGS+=(--dtype "$VLLM_DTYPE")
fi

if [[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]]; then
  ARGS+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
fi

echo "Launching: vllm ${ARGS[*]}"
exec vllm "${ARGS[@]}"

#!/usr/bin/env bash
set -euo pipefail

# Defaults (overridable via flags)
LOG_LEVEL="info"
HOST_MAX_TOOL_CALLS_PER_REQUEST="10"
DOCKER_IMAGE="secure-enclave-gateway-plaintext"

# Required (no defaults; script fails if not provided)
HOST_OPENAI_BASE_URL=""
BRAVE_API_KEY=""

usage() {
  cat <<'EOF'
Usage: run_plaintext_gateway.sh --vllm-url URL --brave-api-key KEY [options]

Runs the Fidaro plaintext gateway in a Docker container.

Required:
  --vllm-url URL          Base URL of the vLLM OpenAI-compatible endpoint.
                          If running vLLM locally, use:
                            http://host.docker.internal:8000/v1
                          For a Phala ZT-TLS endpoint, use a URL of the form:
                            https://*-8000.dstack-pha-use1.phala.network
  --brave-api-key KEY     Brave Search API key.

Options:
  --log-level LEVEL       Gateway log level (default: info).
  --max-tool-calls N      Max tool calls per request (default: 10).
  --docker-image IMAGE    Docker image to run (default: secure-enclave-gateway-plaintext).
  -h, --help              Show this help and exit.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vllm-url)
      [[ $# -ge 2 ]] || { echo "Error: --vllm-url requires a value." >&2; usage >&2; exit 1; }
      HOST_OPENAI_BASE_URL="$2"
      shift 2
      ;;
    --brave-api-key)
      [[ $# -ge 2 ]] || { echo "Error: --brave-api-key requires a value." >&2; usage >&2; exit 1; }
      BRAVE_API_KEY="$2"
      shift 2
      ;;
    --log-level)
      [[ $# -ge 2 ]] || { echo "Error: --log-level requires a value." >&2; usage >&2; exit 1; }
      LOG_LEVEL="$2"
      shift 2
      ;;
    --max-tool-calls)
      [[ $# -ge 2 ]] || { echo "Error: --max-tool-calls requires a value." >&2; usage >&2; exit 1; }
      HOST_MAX_TOOL_CALLS_PER_REQUEST="$2"
      shift 2
      ;;
    --docker-image)
      [[ $# -ge 2 ]] || { echo "Error: --docker-image requires a value." >&2; usage >&2; exit 1; }
      DOCKER_IMAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$HOST_OPENAI_BASE_URL" ]]; then
  echo "Error: --vllm-url is required." >&2
  usage >&2
  exit 1
fi

if [[ -z "$BRAVE_API_KEY" ]]; then
  echo "Error: --brave-api-key is required." >&2
  usage >&2
  exit 1
fi

if [[ -n "$(docker ps -q --filter 'name=^fidaro-gateway$')" ]]; then
  echo "Container 'fidaro-gateway' is already running; stopping it..." >&2
  docker rm -f fidaro-gateway >/dev/null
fi

docker run --rm -p 127.0.0.1:8082:8080 \
  -d \
  --name fidaro-gateway \
  --add-host host.docker.internal:host-gateway \
  -e HOST_LLM_PROVIDER=vllm \
  -e HOST_OPENAI_BASE_URL="${HOST_OPENAI_BASE_URL}" \
  -e LOG_LEVEL="${LOG_LEVEL}" \
  -e BRAVE_API_KEY="${BRAVE_API_KEY}" \
  -e HOST_MAX_TOOL_CALLS_PER_REQUEST="${HOST_MAX_TOOL_CALLS_PER_REQUEST}" \
  "${DOCKER_IMAGE}" \
  uv run uvicorn llm_gateway.dev_plaintext_main:app --host 0.0.0.0 --port 8080

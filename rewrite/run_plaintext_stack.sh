#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 <prod|dogfood> [--build]" >&2
  exit 1
}

# --- Parse arguments (deployment is required, --build is optional, any order) ---
DEPLOYMENT=""
BUILD=false

for arg in "$@"; do
  case "$arg" in
    prod|dogfood)
      if [[ -n "$DEPLOYMENT" ]]; then
        echo "Error: deployment specified more than once" >&2
        usage
      fi
      DEPLOYMENT="$arg"
      ;;
    --build)
      BUILD=true
      ;;
    *)
      echo "Error: unknown argument '$arg'" >&2
      usage
      ;;
  esac
done

if [[ -z "$DEPLOYMENT" ]]; then
  echo "Error: deployment is required and must be 'prod' or 'dogfood'" >&2
  usage
fi

# --- Validate environment ---
if [[ -z "${FIDARO_GIT_ROOT:-}" ]]; then
  echo "Error: FIDARO_GIT_ROOT environment variable is not set" >&2
  exit 1
fi

# --- Map deployment to CVM ---
if [[ "$DEPLOYMENT" == "prod" ]]; then
  CVM_ID="gpu-tee-p-001"
else
  CVM_ID="fidaro-test-001"
fi

# --- Resolve the app id of the CVM ---
APP_ID="$(phala cvms get -j "$CVM_ID" | jq -r '.app_id')"
if [[ -z "$APP_ID" || "$APP_ID" == "null" ]]; then
  echo "Error: could not resolve app_id for CVM '$CVM_ID'" >&2
  exit 1
fi

COMPOSE_FILE="$FIDARO_GIT_ROOT/secure-enclave/docker-compose-plaintext.yml"

# --- Optionally build ---
if [[ "$BUILD" == true ]]; then
  docker compose -f "$COMPOSE_FILE" build
fi

# --- Configure runtime environment ---
VLLM_BASE_URL="https://${APP_ID}-8000.dstack-pha-use1.phala.network/v1"
export ORCHESTRATOR_VLLM_BASE_URL="$VLLM_BASE_URL"
export CLASSIFIER_VLLM_BASE_URL="$VLLM_BASE_URL"

if [[ "$DEPLOYMENT" == "prod" ]]; then
  export GATEWAY_PORT=8082
else
  export GATEWAY_PORT=8084
fi

# --- Bring up the stack (foreground) ---
docker compose -p "fidaro-${DEPLOYMENT}" -f "$COMPOSE_FILE" up

#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Seconds to wait for the gateway to become ready. Override via env or .env.
GATEWAY_WAIT_SECONDS="${GATEWAY_WAIT_SECONDS:-10}"

wait_for_gateway() {
  for i in $(seq 1 "${GATEWAY_WAIT_SECONDS}"); do
    if curl -fsS http://127.0.0.1:8082/v1/health >/dev/null 2>&1; then
      echo "Gateway ready after ${i}s"
      return 0
    fi
    sleep 1
  done
  echo "::error::Gateway did not become ready within ${GATEWAY_WAIT_SECONDS}s"
  echo "::error::You must start the gateway yourself before running this script. You can use ./scripts_repo/run_plaintext_gateway.sh to start a local gateway."
  echo
  exit 1
}

wait_for_gateway

mkdir -p results/local

desc_args=()
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  short_sha="${GITHUB_SHA:0:7}"
  desc_args=(--description "github_actions run ${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT:-1} @ ${short_sha} (${GITHUB_REF_NAME})")
fi

SUITE_GENERATION_CONFIG_FILE=scripts_test/fidaro_config.json pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers fidaro_plaintext_gateway_phala_prod \
  --filter-metadata suite=research_rubrics \
  --output results/local/latest.json \
  "${desc_args[@]}"

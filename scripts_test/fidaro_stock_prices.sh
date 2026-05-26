#!/usr/bin/env bash
# Run the stock-price freshness suite against prod Fidaro.
#
# Fetches live reference prices from Stooq FIRST (failing fast if any symbol is
# unavailable), then runs only the stock_prices suite with caching disabled — a
# cached answer would defeat the point of an up-to-date-data check.
#
# You must have the plaintext gateway running (see docs/README.md); this script
# does not start it.
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

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
  echo "::error::Start it yourself first (see docs/README.md)."
  exit 1
}

wait_for_gateway

# Preflight: fetch live prices and fail fast if the source can't give us all of them.
echo "Fetching live reference prices from Stooq..."
python scripts_repo/fetch_stock_prices.py || {
  echo "::error::Could not fetch all reference prices; aborting before the test run."
  exit 1
}

mkdir -p results/local

desc_args=()
if [[ "${GITHUB_ACTIONS:-}" == "true" ]]; then
  short_sha="${GITHUB_SHA:0:7}"
  desc_args=(--description "github_actions stock_prices ${GITHUB_RUN_ID}.${GITHUB_RUN_ATTEMPT:-1} @ ${short_sha} (${GITHUB_REF_NAME})")
fi

SUITE_GENERATION_CONFIG_FILE=scripts_test/fidaro_stock_prices_config.json \
  pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers fidaro_plaintext_gateway_phala_prod \
  --filter-metadata suite=stock_prices \
  --output results/local/latest.json \
  --no-cache \
  "${desc_args[@]}"

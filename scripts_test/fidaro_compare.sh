#!/usr/bin/env bash
set -xuo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p results/local

echo "WARNING: baseline runs will be cached for optimization. Add the --no-cache flag to disable caching and get a fresh run, but this will be slower. Caching is recommended for regular use and CI runs, but you may want to disable it when developing or debugging."

AGENTHARM_LIMIT=5 RESEARCH_RUBRICS_LIMIT=10 RESEARCH_RUBRICS_MAX_CRITERIA=3 pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers fidaro_plaintext_gateway_phala_prod \
  --filter-metadata suite=research_rubrics \
  --output results/local/latest.json \
  --description "Baseline prod run"

./scripts_repo/freeze_baseline.py results/local/latest.json --force

AGENTHARM_LIMIT=5 RESEARCH_RUBRICS_LIMIT=10 RESEARCH_RUBRICS_MAX_CRITERIA=3 pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers fidaro_plaintext_gateway_phala_dev \
  --filter-metadata suite=research_rubrics \
  --output results/local/latest.json \
  --no-cache \
  --description "Comparison dev run"

./scripts_repo/compare_matrix.py --baseline baselines/fidaro_plaintext_gateway_phala_prod.json --latest results/local/latest.json --out report.html
open report.html

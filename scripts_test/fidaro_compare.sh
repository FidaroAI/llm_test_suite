#!/usr/bin/env bash
set -uo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p results/local

# echo "WARNING: baseline runs will be cached for optimization. Add the --no-cache flag to disable caching and get a fresh run, but this will be slower. Caching is recommended for regular use and CI runs, but you may want to disable it when developing or debugging."

# export AGENTHARM_LIMIT=0
# export RESEARCH_RUBRICS_LIMIT=0
# export RESEARCH_RUBRICS_MAX_CRITERIA=3
# export MULTIFACETED_LIMIT=1
# export MULTIFACETED_MAX_CRITERIA=3

#  pnpm exec promptfoo eval \
#   --config promptfooconfig.yaml \
#   --filter-providers fidaro_plaintext_gateway_phala_prod \
#   --filter-metadata suite=multifaceted \
#   --output results/local/latest.json \
#   --no-cache \
#   --description "Baseline prod run"

# set -e

# ./scripts_repo/freeze_baseline.py results/local/latest.json --force
# cp results/local/latest.json results/local/$(jq -r '.evalId' results/local/latest.json).json

# set +e

# pnpm exec promptfoo eval \
#   --config promptfooconfig.yaml \
#   --filter-providers fidaro_plaintext_gateway_phala_dev \
#   --filter-metadata suite=multifaceted \
#   --output results/local/latest.json \
#   --no-cache \
#   --description "Comparison dev run"

# set -e

# Annoyingly we can't just leave this running. promptfoo doesn't pick up changes to
# its database once running :(
./scripts_repo/run_promptfoo_docker.sh
cp results/local/latest.json results/local/$(jq -r '.evalId' results/local/latest.json).json

./scripts_repo/compare_runs.py baselines/fidaro_plaintext_gateway_phala_prod.json results/local/latest.json --tolerance 0.05 --out ./results/reports/report_comparison.html
open ./results/reports/report_comparison.html

# Disabling this report for now. Might not be useful. I'm mostly iterating with the other report.
# ./scripts_repo/compare_matrix.py --baseline baselines/fidaro_plaintext_gateway_phala_prod.json --latest results/local/latest.json --out ./results/reports/report_matrix.html
# open ./results/reports/report_matrix.html
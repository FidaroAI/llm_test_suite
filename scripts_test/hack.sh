#!/usr/bin/env bash
# Helper script used for quickly running tests locally when developing.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
  # shellcheck disable=SC1090
  source "$HOME/.nvm/nvm.sh"
  nvm use --silent
fi

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

mkdir -p results/local
exec pnpm exec promptfoo eval \
  --config promptfooconfig_geval_demo.yaml \
  --filter-pattern "compound vs simple interest" \
  --no-cache \
  --output results/local/latest.json

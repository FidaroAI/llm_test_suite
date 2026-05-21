#!/usr/bin/env bash

# Automatically export all variables defined in the .env file
if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

# Convenience script for local use. Pulls value from the ENV
$(dirname "$0")/run_plaintext_gateway.sh \
    --docker-image "secure-enclave-gateway-plaintext" \
    --vllm-prod-url "$PHALA_PROD_VLLM_URL" \
    --vllm-dev-url "$PHALA_DEV_VLLM_URL" \
    --brave-api-key "$BRAVE_API_KEY"

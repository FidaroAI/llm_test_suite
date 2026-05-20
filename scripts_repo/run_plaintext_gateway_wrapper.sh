#!/usr/bin/env bash

# Convenience script for local use. Pulls value from the ENV
$(dirname "$0")/run_plaintext_gateway.sh
    --image "secure-enclave-gateway-plaintext" \
    --vllm-url "$PHALA_VLLM_URL" \
    --brave-api-key "$BRAVE_API_KEY"

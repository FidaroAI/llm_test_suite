"""Registry of providers a comparison can run.

One row per provider key (the key used in a comparison config's
``providers-under-test`` / ``provider-options`` and shown as the report column
name). The registry is the single source of truth for how each provider is run:
whether it needs a local plaintext gateway or is a direct external API, which
promptfoo provider label represents it (for ``--filter-providers`` and splitting
the unified result file), and the ``COMPARISON_<PREFIX>_*`` env prefix its dynamic
YAML templates model/temperature/max_tokens from.

Adding a competitor = one row here + one provider YAML in ``providers/``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    key: str                 # config key; also the report column name
    label: str               # promptfoo provider label (filter + result split)
    env_prefix: str          # COMPARISON_PROD / COMPARISON_DEV / COMPARISON_VENICE
    kind: str                # "gateway" | "api"
    gateway_port: int | None = None       # gateway only
    vllm_url_key: str | None = None        # gateway only: config key for its vLLM url
    supports_redeploy: bool = False        # only fidaro-dev
    supports_system_prompt: bool = False   # only fidaro-dev
    api_key_env: str | None = None         # api only, e.g. "VENICE_INFERENCE_KEY"


REGISTRY: dict[str, ProviderSpec] = {
    "fidaro-prod": ProviderSpec(
        key="fidaro-prod",
        label="fidaro_plaintext_gateway_phala_dynamic_prod",
        env_prefix="COMPARISON_PROD",
        kind="gateway",
        gateway_port=8082,
        vllm_url_key="vllm-prod-url",
    ),
    "fidaro-dev": ProviderSpec(
        key="fidaro-dev",
        label="fidaro_plaintext_gateway_phala_dynamic_dev",
        env_prefix="COMPARISON_DEV",
        kind="gateway",
        gateway_port=8084,
        vllm_url_key="vllm-dev-url",
        supports_redeploy=True,
        supports_system_prompt=True,
    ),
    "venice": ProviderSpec(
        key="venice",
        label="venice_dynamic",
        env_prefix="COMPARISON_VENICE",
        kind="api",
        api_key_env="VENICE_INFERENCE_KEY",
    ),
}


def all_keys() -> set[str]:
    """Every provider key known to the registry."""
    return set(REGISTRY)


def resolve(keys: list[str]) -> list[ProviderSpec]:
    """Specs for ``keys``, preserving order. Raises KeyError on an unknown key."""
    return [REGISTRY[k] for k in keys]

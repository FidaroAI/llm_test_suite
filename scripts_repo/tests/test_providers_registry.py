"""Tests for the provider registry."""
from __future__ import annotations

import pytest

from scripts_repo.providers_registry import REGISTRY, ProviderSpec, resolve, all_keys


def test_known_keys_present():
    assert {"fidaro-prod", "fidaro-dev", "venice"} <= all_keys()


def test_resolve_returns_specs_in_order():
    specs = resolve(["venice", "fidaro-prod"])
    assert [s.key for s in specs] == ["venice", "fidaro-prod"]
    assert all(isinstance(s, ProviderSpec) for s in specs)


def test_resolve_rejects_unknown_key():
    with pytest.raises(KeyError):
        resolve(["nope"])


def test_gateway_vs_api_split():
    assert REGISTRY["fidaro-prod"].kind == "gateway"
    assert REGISTRY["fidaro-prod"].gateway_port == 8082
    assert REGISTRY["fidaro-dev"].supports_redeploy is True
    assert REGISTRY["fidaro-dev"].supports_system_prompt is True
    assert REGISTRY["venice"].kind == "api"
    assert REGISTRY["venice"].api_key_env == "VENICE_INFERENCE_KEY"
    assert REGISTRY["venice"].gateway_port is None

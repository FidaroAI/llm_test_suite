"""Unit tests for the pure logic in scripts_repo/run_comparison.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import re

from scripts_repo.run_comparison import (
    COMPARISON_NETWORK,
    CORE_SYSTEM_PROMPT_PATH,
    SELECT_BEST_ENV_VAR,
    WEB_FETCH_CONTAINER_PORT,
    WEB_FETCH_NETWORK_ALIAS,
    WEB_FETCH_SIDECAR_URL,
    WHITELISTED_CVM_IDS,
    ConfigError,
    build_filter_args,
    comparison_dir,
    comparison_name,
    eval_command,
    gateway_docker_args,
    parse_env_file,
    provider_options_env,
    providers_filter,
    report_provider_args,
    run_dir_name,
    validate_config,
    vllm_options_changed,
    web_fetch_docker_args,
    write_options_cache,
)
from scripts_repo.providers_registry import REGISTRY


def _load_classification():
    path = Path(__file__).resolve().parents[2] / "tests" / "classification.py"
    spec = importlib.util.spec_from_file_location("classification", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_select_best_env_var_matches_classification():
    # run_comparison sets this var; classification.augment reads it. They must
    # name the same var or the head-to-head silently never fires.
    assert SELECT_BEST_ENV_VAR == _load_classification().SELECT_BEST_ENV_VAR


def _minimal_config(**overrides):
    """A minimal valid config: fidaro-prod (baseline) + venice, no redeploy."""
    cfg = {
        "providers-under-test": {"fidaro-prod": True, "venice": True},
        "baseline-provider": "fidaro-prod",
        "provider-options": {
            "fidaro-prod": {"model": "prod-model", "temperature": 0.7, "max_tokens": 1000},
            "venice": {"model": "kimi-k2-6", "web_search": "on"},
        },
        "vllm-prod-url": "https://prod-8000.example.net/v1",
        "suite-generation-config": {"simple_facts": {"limit": 5}},
    }
    cfg.update(overrides)
    return cfg


# venice is enabled in _minimal_config, so its api key must be in the env.
_VENICE_ENV = {"VENICE_INFERENCE_KEY": "k"}


def _dev_redeploy_config(tmp_path, **overrides):
    """A fidaro-dev-only config wired for a redeploy (vllm-options present).

    Writes the compose + .env.phala files the redeploy validation requires and
    returns (config, env) ready to pass to validate_config.
    """
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services: {}", encoding="utf-8")
    (tmp_path / ".env.phala").write_text("X=1", encoding="utf-8")
    cfg = {
        "providers-under-test": {"fidaro-dev": True},
        "baseline-provider": "fidaro-dev",
        "provider-options": {"fidaro-dev": {"model": "x"}},
        "vllm-dev-url": "https://dev-8000.example.net/v1",
        "suite-generation-config": {"simple_facts": {"limit": 5}},
        "vllm-options": {"model": "x"},
        "phala-dev-instance-id": WHITELISTED_CVM_IDS[0],
    }
    cfg.update(overrides)
    return cfg, {"PHALA_DOCKER_COMPOSE_FILE": str(compose)}


# --- comparison_name -------------------------------------------------------


def test_comparison_name_is_the_config_file_stem():
    assert comparison_name("comparisons/prod_vs_dev_gemma.json") == "prod_vs_dev_gemma"


# --- comparison_dir --------------------------------------------------------


def test_comparison_dir_is_config_sibling_named_after_stem(tmp_path):
    cfg = tmp_path / "sub" / "myrun.json"
    assert comparison_dir(cfg) == tmp_path / "sub" / "myrun"


def test_comparison_dir_is_absolute_even_for_relative_config():
    result = comparison_dir("comparisons/example.json")
    assert result.is_absolute()
    assert result.name == "example"
    assert result.parent.name == "comparisons"


# --- run_dir_name ----------------------------------------------------------


def test_run_dir_name_prefixes_timestamp_with_run():
    assert run_dir_name("20260525-134500") == "run_20260525-134500"


# --- validate_config -------------------------------------------------------


def test_validate_passes_for_minimal_config(tmp_path):
    validate_config(_minimal_config(), repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_requires_suite_generation_config(tmp_path):
    cfg = _minimal_config()
    del cfg["suite-generation-config"]
    with pytest.raises(ConfigError, match="suite-generation-config"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_requires_providers_under_test(tmp_path):
    cfg = _minimal_config()
    del cfg["providers-under-test"]
    with pytest.raises(ConfigError, match="providers-under-test"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_rejects_no_enabled_providers(tmp_path):
    cfg = _minimal_config()
    cfg["providers-under-test"] = {"fidaro-prod": False, "venice": False}
    with pytest.raises(ConfigError, match="at least one"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_rejects_unknown_provider_key(tmp_path):
    cfg = _minimal_config()
    cfg["providers-under-test"]["mystery"] = True
    with pytest.raises(ConfigError, match="unknown provider"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_baseline_must_be_enabled(tmp_path):
    cfg = _minimal_config(**{"baseline-provider": "fidaro-dev"})
    with pytest.raises(ConfigError, match="baseline-provider"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_requires_options_for_each_enabled(tmp_path):
    cfg = _minimal_config()
    del cfg["provider-options"]["venice"]
    with pytest.raises(ConfigError, match="provider-options.*venice"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_rejects_options_for_non_enabled_provider(tmp_path):
    cfg = _minimal_config()
    cfg["provider-options"]["fidaro-dev"] = {"model": "y"}  # not enabled
    with pytest.raises(ConfigError, match="non-enabled"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_gateway_requires_its_vllm_url(tmp_path):
    cfg = _minimal_config()
    del cfg["vllm-prod-url"]
    with pytest.raises(ConfigError, match="vllm-prod-url"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_api_provider_requires_key_env(tmp_path):
    with pytest.raises(ConfigError, match="VENICE_INFERENCE_KEY"):
        validate_config(_minimal_config(), repo_root=tmp_path, env={})


def test_validate_redeploy_guard_skipped_without_vllm_options(tmp_path):
    # fidaro-dev enabled but no vllm-options => no redeploy, so no instance-id /
    # whitelist / compose constraints (the HEAD-era unconditional-check regression).
    cfg = {
        "providers-under-test": {"fidaro-dev": True},
        "baseline-provider": "fidaro-dev",
        "provider-options": {"fidaro-dev": {"model": "x"}},
        "vllm-dev-url": "https://dev-8000.example.net/v1",
        "suite-generation-config": {"simple_facts": {"limit": 5}},
    }
    validate_config(cfg, repo_root=tmp_path, env={})


def test_validate_full_redeploy_config_passes(tmp_path):
    cfg, env = _dev_redeploy_config(tmp_path)
    validate_config(cfg, repo_root=tmp_path, env=env)


def test_validate_rejects_dev_model_not_matching_vllm_options(tmp_path):
    cfg, env = _dev_redeploy_config(tmp_path, **{"vllm-options": {"model": "served"}})
    # provider-options['fidaro-dev'].model is "x" != served.
    with pytest.raises(ConfigError, match="must match"):
        validate_config(cfg, repo_root=tmp_path, env=env)


def test_validate_rejects_vllm_options_without_instance_id(tmp_path):
    cfg, env = _dev_redeploy_config(tmp_path)
    del cfg["phala-dev-instance-id"]
    with pytest.raises(ConfigError, match="phala-dev-instance-id"):
        validate_config(cfg, repo_root=tmp_path, env=env)


def test_validate_rejects_non_whitelisted_instance(tmp_path):
    cfg, env = _dev_redeploy_config(tmp_path, **{"phala-dev-instance-id": "not-listed"})
    with pytest.raises(ConfigError, match="whitelist"):
        validate_config(cfg, repo_root=tmp_path, env=env)


def test_validate_vllm_options_requires_compose_env(tmp_path):
    cfg, _ = _dev_redeploy_config(tmp_path)
    with pytest.raises(ConfigError, match="PHALA_DOCKER_COMPOSE_FILE"):
        validate_config(cfg, repo_root=tmp_path, env={})


def test_validate_vllm_options_requires_env_phala_file(tmp_path):
    cfg, env = _dev_redeploy_config(tmp_path)
    (tmp_path / ".env.phala").unlink()
    with pytest.raises(ConfigError, match=".env.phala"):
        validate_config(cfg, repo_root=tmp_path, env=env)


def test_validate_rejects_missing_system_prompt_file(tmp_path):
    cfg = _minimal_config(**{"system-prompt-file": str(tmp_path / "nope.md")})
    with pytest.raises(ConfigError, match="system-prompt-file"):
        validate_config(cfg, repo_root=tmp_path, env=_VENICE_ENV)


def test_validate_accepts_existing_system_prompt_file(tmp_path):
    prompt = tmp_path / "sys.md"
    prompt.write_text("hello", encoding="utf-8")
    validate_config(
        _minimal_config(**{"system-prompt-file": str(prompt)}),
        repo_root=tmp_path,
        env=_VENICE_ENV,
    )


# --- vllm options cache ----------------------------------------------------


def test_options_changed_true_when_cache_missing(tmp_path):
    assert vllm_options_changed(tmp_path / "cache.json", {"model": "x"}) is True


def test_options_unchanged_after_write(tmp_path):
    cache = tmp_path / "cache.json"
    options = {"model": "x", "reasoning-parser": "gemma4"}
    write_options_cache(cache, options)
    assert vllm_options_changed(cache, options) is False


def test_options_changed_when_value_differs(tmp_path):
    cache = tmp_path / "cache.json"
    write_options_cache(cache, {"model": "x"})
    assert vllm_options_changed(cache, {"model": "y"}) is True


def test_options_unchanged_ignores_key_order(tmp_path):
    cache = tmp_path / "cache.json"
    write_options_cache(cache, {"a": "1", "b": "2"})
    assert vllm_options_changed(cache, {"b": "2", "a": "1"}) is False


def test_write_options_cache_is_identical_copy(tmp_path):
    cache = tmp_path / "cache.json"
    options = {"model": "x", "enable-auto-tool-choice": True}
    write_options_cache(cache, options)
    assert json.loads(cache.read_text(encoding="utf-8")) == options


# --- provider options env --------------------------------------------------


def test_provider_options_env_maps_each_enabled_provider():
    env = provider_options_env(_minimal_config())
    # fidaro-prod (gateway): model/temp/max_tokens templated.
    assert env["COMPARISON_PROD_MODEL"] == "prod-model"
    assert env["COMPARISON_PROD_TEMPERATURE"] == "0.7"
    assert env["COMPARISON_PROD_MAX_TOKENS"] == "1000"
    # venice (api): model + web search; optional temp/max_tokens omitted.
    assert env["COMPARISON_VENICE_MODEL"] == "kimi-k2-6"
    assert env["COMPARISON_VENICE_WEB_SEARCH"] == "on"
    assert "COMPARISON_VENICE_TEMPERATURE" not in env
    assert "COMPARISON_VENICE_MAX_TOKENS" not in env


def test_provider_options_env_web_search_defaults_off():
    cfg = {
        "providers-under-test": {"venice": True},
        "provider-options": {"venice": {"model": "kimi-k2-6"}},
    }
    assert provider_options_env(cfg)["COMPARISON_VENICE_WEB_SEARCH"] == "off"


def test_provider_options_env_values_are_strings():
    assert all(isinstance(v, str) for v in provider_options_env(_minimal_config()).values())


# --- promptfoo filter args -------------------------------------------------


def test_build_filter_args_maps_keys_to_flags():
    args = build_filter_args({"filter-metadata": "suite=simple_facts"})
    assert args == ["--filter-metadata", "suite=simple_facts"]


def test_build_filter_args_handles_none_and_empty():
    assert build_filter_args(None) == []
    assert build_filter_args({}) == []


def test_build_filter_args_ignores_filter_providers():
    args = build_filter_args(
        {"filter-providers": "should_be_dropped", "filter-pattern": "Paris"}
    )
    assert "should_be_dropped" not in args
    assert args == ["--filter-pattern", "Paris"]


def test_build_filter_args_repeats_list_values():
    args = build_filter_args({"filter-metadata": ["suite=a", "suite=b"]})
    assert args == [
        "--filter-metadata",
        "suite=a",
        "--filter-metadata",
        "suite=b",
    ]


# --- providers filter (single unified pass over the enabled set) -----------


def test_providers_filter_matches_only_enabled_labels():
    cfg = {"providers-under-test": {"fidaro-prod": True, "venice": True}}
    pat = providers_filter(cfg)
    assert re.match(pat, REGISTRY["fidaro-prod"].label)
    assert re.match(pat, REGISTRY["venice"].label)
    # fidaro-dev is not enabled, so its dynamic label must be excluded.
    assert not re.match(pat, REGISTRY["fidaro-dev"].label)


def test_providers_filter_excludes_static_providers():
    cfg = {"providers-under-test": {"fidaro-prod": True}}
    pat = providers_filter(cfg)
    assert not re.match(pat, "fidaro_plaintext_gateway_phala_prod")


def test_report_provider_args_baseline_first_then_others():
    cfg = {
        "providers-under-test": {"fidaro-prod": True, "venice": True},
        "baseline-provider": "fidaro-prod",
    }
    args = report_provider_args(cfg)
    assert args[:2] == [
        "--baseline-provider-col",
        f"fidaro-prod={REGISTRY['fidaro-prod'].label}",
    ]
    assert "--provider-col" in args
    assert f"venice={REGISTRY['venice'].label}" in args


# --- gateway docker args ---------------------------------------------------


def test_gateway_docker_args_core_invocation():
    args = gateway_docker_args(
        name="fidaro-gateway-prod",
        port=8082,
        vllm_url="https://prod/v1",
        brave_api_key="brave-key",
        image="secure-enclave-gateway-plaintext",
        network=COMPARISON_NETWORK,
        web_fetch_url=WEB_FETCH_SIDECAR_URL,
    )
    assert args[0:2] == ["docker", "run"]
    assert "-p" in args and "127.0.0.1:8082:8080" in args
    assert "--name" in args and "fidaro-gateway-prod" in args
    assert "HOST_OPENAI_BASE_URL=https://prod/v1" in args
    assert "BRAVE_API_KEY=brave-key" in args
    assert args[-1] == "8080"  # ends with the uvicorn port


def test_gateway_docker_args_no_mount_without_prompt():
    args = gateway_docker_args(
        name="fidaro-gateway-prod",
        port=8082,
        vllm_url="https://prod/v1",
        brave_api_key="k",
        image="img",
        network=COMPARISON_NETWORK,
        web_fetch_url=WEB_FETCH_SIDECAR_URL,
    )
    assert "-v" not in args


def test_gateway_docker_args_mounts_system_prompt_absolutely(tmp_path):
    prompt = tmp_path / "sys.md"
    prompt.write_text("x", encoding="utf-8")
    args = gateway_docker_args(
        name="fidaro-gateway-dev",
        port=8084,
        vllm_url="https://dev/v1",
        brave_api_key="k",
        image="img",
        network=COMPARISON_NETWORK,
        web_fetch_url=WEB_FETCH_SIDECAR_URL,
        system_prompt_file=str(prompt),
    )
    mount = f"{prompt.resolve()}:{CORE_SYSTEM_PROMPT_PATH}:ro"
    assert "-v" in args
    assert mount in args


def test_gateway_docker_args_joins_shared_network():
    # Gateways must attach to the shared user-defined bridge so Docker DNS
    # resolves `web-fetch`; the default bridge has no DNS.
    args = gateway_docker_args(
        name="fidaro-gateway-prod",
        port=8082,
        vllm_url="https://prod/v1",
        brave_api_key="k",
        image="img",
        network="fidaro-comparison",
        web_fetch_url=WEB_FETCH_SIDECAR_URL,
    )
    idx = args.index("--network")
    assert args[idx + 1] == "fidaro-comparison"


def test_gateway_docker_args_passes_web_fetch_url():
    args = gateway_docker_args(
        name="fidaro-gateway-prod",
        port=8082,
        vllm_url="https://prod/v1",
        brave_api_key="k",
        image="img",
        network=COMPARISON_NETWORK,
        web_fetch_url="http://web-fetch:8000",
    )
    assert "HOST_WEB_FETCH_SIDECAR_URL=http://web-fetch:8000" in args


# --- web-fetch sidecar docker args -----------------------------------------


def test_web_fetch_docker_args_publishes_no_host_port():
    # The sidecar is reachable only via the user-defined bridge; publishing a
    # host port would defeat that isolation.
    args = web_fetch_docker_args(
        name="fidaro-web-fetch",
        image="web-fetch-tool-web-fetch:latest",
        network=COMPARISON_NETWORK,
    )
    assert "-p" not in args


def test_web_fetch_docker_args_registers_network_alias():
    args = web_fetch_docker_args(
        name="fidaro-web-fetch",
        image="web-fetch-tool-web-fetch:latest",
        network=COMPARISON_NETWORK,
    )
    # The alias is what gateways resolve as `http://web-fetch:8000`; if this
    # drifts from WEB_FETCH_SIDECAR_URL the gateways get DNS errors at fetch
    # time, not at startup.
    idx = args.index("--network-alias")
    assert args[idx + 1] == WEB_FETCH_NETWORK_ALIAS


def test_web_fetch_docker_args_baked_in_healthcheck():
    # wait_for_container_healthy polls Docker's health status, so the run-time
    # --health-cmd is required (the image's compose healthcheck doesn't apply).
    args = web_fetch_docker_args(
        name="fidaro-web-fetch",
        image="img",
        network=COMPARISON_NETWORK,
    )
    assert "--health-cmd" in args
    cmd_idx = args.index("--health-cmd")
    assert str(WEB_FETCH_CONTAINER_PORT) in args[cmd_idx + 1]


def test_web_fetch_sidecar_url_uses_network_alias_and_port():
    # The constants compose into a single URL the gateways see; keep that
    # composition asserted so a rename of the alias or port can't desync.
    assert WEB_FETCH_SIDECAR_URL == f"http://{WEB_FETCH_NETWORK_ALIAS}:{WEB_FETCH_CONTAINER_PORT}"


# --- promptfoo eval command ------------------------------------------------


def test_eval_command_includes_provider_output_and_description():
    cmd = eval_command(
        provider="fidaro_plaintext_gateway_phala_prod",
        output_path="comparisons/x/prod_results_20260522-101010.json",
        filter_args=[],
        no_cache=True,
        description="prod run",
    )
    assert cmd[:3] == ["pnpm", "exec", "promptfoo"]
    assert "--filter-providers" in cmd
    assert "fidaro_plaintext_gateway_phala_prod" in cmd
    assert "--output" in cmd
    assert "comparisons/x/prod_results_20260522-101010.json" in cmd
    assert "--description" in cmd and "prod run" in cmd


def test_eval_command_no_cache_toggle():
    with_cache = eval_command(
        provider="p", output_path="o", filter_args=[], no_cache=False, description="d"
    )
    without = eval_command(
        provider="p", output_path="o", filter_args=[], no_cache=True, description="d"
    )
    assert "--no-cache" not in with_cache
    assert "--no-cache" in without


def test_eval_command_passes_through_filter_args():
    cmd = eval_command(
        provider="p",
        output_path="o",
        filter_args=["--filter-metadata", "suite=simple_facts"],
        no_cache=True,
        description="d",
    )
    assert "--filter-metadata" in cmd
    assert "suite=simple_facts" in cmd


# --- .env parsing ----------------------------------------------------------


def test_parse_env_file_basic():
    env = parse_env_file("A=1\nB=two words\n")
    assert env == {"A": "1", "B": "two words"}


def test_parse_env_file_skips_comments_and_blanks():
    env = parse_env_file("# comment\n\nA=1\n   \nB=2\n")
    assert env == {"A": "1", "B": "2"}


def test_parse_env_file_strips_quotes_and_export():
    env = parse_env_file('export A="quoted"\nB=\'single\'\n')
    assert env == {"A": "quoted", "B": "single"}

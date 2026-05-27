"""Unit tests for the pure logic in scripts_repo/run_comparison.py."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

import re

from scripts_repo.run_comparison import (
    CORE_SYSTEM_PROMPT_PATH,
    DEV_PROVIDER,
    PROD_PROVIDER,
    SELECT_BEST_ENV_VAR,
    ConfigError,
    both_providers_filter,
    build_filter_args,
    comparison_dir,
    comparison_name,
    eval_command,
    gateway_docker_args,
    parse_env_file,
    provider_options_env,
    run_dir_name,
    validate_config,
    vllm_options_changed,
    write_options_cache,
)


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
    cfg = {
        "vllm-prod-url": "https://prod-8000.example.net/v1",
        "vllm-dev-url": "https://dev-8000.example.net/v1",
        "suite-generation-config": {"simple_facts": {"limit": 5}},
        "prod-provider-options": {
            "model": "prod-model",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
        # Matches the vllm-options model ("x") used by the redeploy tests below.
        "dev-provider-options": {
            "model": "x",
            "temperature": 0.7,
            "max_tokens": 1000,
        },
    }
    cfg.update(overrides)
    return cfg


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
    validate_config(_minimal_config(), repo_root=tmp_path, env={})


@pytest.mark.parametrize(
    "missing", ["vllm-prod-url", "vllm-dev-url", "suite-generation-config"]
)
def test_validate_requires_each_mandatory_key(tmp_path, missing):
    cfg = _minimal_config()
    del cfg[missing]
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg, repo_root=tmp_path, env={})
    assert missing in str(exc.value)


@pytest.mark.parametrize("missing", ["prod-provider-options", "dev-provider-options"])
def test_validate_requires_provider_options(tmp_path, missing):
    cfg = _minimal_config()
    del cfg[missing]
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg, repo_root=tmp_path, env={})
    assert missing in str(exc.value)


def test_validate_requires_provider_option_fields(tmp_path):
    cfg = _minimal_config()
    del cfg["dev-provider-options"]["max_tokens"]
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg, repo_root=tmp_path, env={})
    assert "max_tokens" in str(exc.value)


def test_validate_rejects_dev_model_not_matching_vllm_options(tmp_path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services: {}", encoding="utf-8")
    (tmp_path / ".env.phala").write_text("X=1", encoding="utf-8")
    cfg = _minimal_config(
        **{"vllm-options": {"model": "served-model"}, "phala-dev-instance-id": "cvm-1"}
    )
    # dev-provider-options.model is "x", which != the served "served-model".
    with pytest.raises(ConfigError) as exc:
        validate_config(
            cfg, repo_root=tmp_path, env={"PHALA_DOCKER_COMPOSE_FILE": str(compose)}
        )
    assert "must match" in str(exc.value)


def test_validate_rejects_vllm_options_without_instance_id(tmp_path):
    cfg = _minimal_config(**{"vllm-options": {"model": "x"}})
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg, repo_root=tmp_path, env={})
    assert "phala-dev-instance-id" in str(exc.value)


def test_validate_rejects_missing_system_prompt_file(tmp_path):
    cfg = _minimal_config(**{"system-prompt-file": str(tmp_path / "nope.md")})
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg, repo_root=tmp_path, env={})
    assert "system-prompt-file" in str(exc.value)


def test_validate_accepts_existing_system_prompt_file(tmp_path):
    prompt = tmp_path / "sys.md"
    prompt.write_text("hello", encoding="utf-8")
    validate_config(
        _minimal_config(**{"system-prompt-file": str(prompt)}),
        repo_root=tmp_path,
        env={},
    )


def test_validate_vllm_options_requires_compose_env(tmp_path):
    cfg = _minimal_config(
        **{"vllm-options": {"model": "x"}, "phala-dev-instance-id": "cvm-1"}
    )
    (tmp_path / ".env.phala").write_text("X=1", encoding="utf-8")
    with pytest.raises(ConfigError) as exc:
        validate_config(cfg, repo_root=tmp_path, env={})
    assert "PHALA_DOCKER_COMPOSE_FILE" in str(exc.value)


def test_validate_vllm_options_requires_env_phala_file(tmp_path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services: {}", encoding="utf-8")
    cfg = _minimal_config(
        **{"vllm-options": {"model": "x"}, "phala-dev-instance-id": "cvm-1"}
    )
    with pytest.raises(ConfigError) as exc:
        validate_config(
            cfg,
            repo_root=tmp_path,
            env={"PHALA_DOCKER_COMPOSE_FILE": str(compose)},
        )
    assert ".env.phala" in str(exc.value)


def test_validate_passes_full_vllm_options_config(tmp_path):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text("services: {}", encoding="utf-8")
    (tmp_path / ".env.phala").write_text("X=1", encoding="utf-8")
    cfg = _minimal_config(
        **{"vllm-options": {"model": "x"}, "phala-dev-instance-id": "cvm-1"}
    )
    validate_config(
        cfg, repo_root=tmp_path, env={"PHALA_DOCKER_COMPOSE_FILE": str(compose)}
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


def test_provider_options_env_maps_both_sides():
    cfg = _minimal_config()
    env = provider_options_env(cfg)
    assert env == {
        "COMPARISON_PROD_MODEL": "prod-model",
        "COMPARISON_PROD_TEMPERATURE": "0.7",
        "COMPARISON_PROD_MAX_TOKENS": "1000",
        "COMPARISON_DEV_MODEL": "x",
        "COMPARISON_DEV_TEMPERATURE": "0.7",
        "COMPARISON_DEV_MAX_TOKENS": "1000",
    }


def test_provider_options_env_values_are_strings():
    env = provider_options_env(_minimal_config())
    assert all(isinstance(v, str) for v in env.values())


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


# --- both-providers filter (single unified pass) ---------------------------


def test_both_providers_filter_matches_both_dynamic_providers():
    pat = both_providers_filter()
    assert re.search(pat, PROD_PROVIDER)
    assert re.search(pat, DEV_PROVIDER)


def test_both_providers_filter_excludes_static_providers():
    # The static (non-dynamic) providers must not be swept into the unified pass.
    pat = both_providers_filter()
    assert not re.search(pat, "fidaro_plaintext_gateway_phala_prod")
    assert not re.search(pat, "fidaro_plaintext_gateway_phala_dev")


# --- gateway docker args ---------------------------------------------------


def test_gateway_docker_args_core_invocation():
    args = gateway_docker_args(
        name="fidaro-gateway-prod",
        port=8082,
        vllm_url="https://prod/v1",
        brave_api_key="brave-key",
        image="secure-enclave-gateway-plaintext",
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
        system_prompt_file=str(prompt),
    )
    mount = f"{prompt.resolve()}:{CORE_SYSTEM_PROMPT_PATH}:ro"
    assert "-v" in args
    assert mount in args


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

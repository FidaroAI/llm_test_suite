# Multi-provider Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Generalize the prod-vs-dev comparison pipeline to compare an arbitrary enabled set of providers (Fidaro gateways + direct-API competitors like Venice) against one designated baseline, in a single promptfoo eval, with a multi-column report.

**Architecture:** A small in-code provider *registry* maps each config key to how it runs (gateway vs api, ports, env prefix, labels). `run_comparison.py` resolves the enabled keys to specs to drive orchestration, env templating, and the eval filter. `compare_runs.py` is generalized from a 2-sided diff to an N-provider row model (baseline column + per-provider columns + per-provider delta-vs-baseline + N-way `best`), with tabular summaries and no `status` column.

**Tech Stack:** Python 3.13, pytest, promptfoo 0.121.12 (`openai:chat` provider with `config.passthrough` for vendor params), YAML provider files templated via `{{ env.* }}`.

**Reference spec:** `docs/superpowers/specs/2026-06-01-multi-provider-comparison-design.md`

**Working context:** Worktree on branch `worktree-multi-provider-comparison`. Run tests with `python -m pytest` (direnv supplies the interpreter). NOTE: at HEAD, 6 `test_run_comparison.py` tests fail because `validate_config`'s CVM-whitelist check runs unconditionally; Task 3 fixes this as part of the validation rewrite.

---

## File Structure

- **Create** `scripts_repo/providers_registry.py` — the `ProviderSpec` dataclass + `REGISTRY` + small resolver helpers. Pure data; no I/O.
- **Create** `scripts_repo/tests/test_providers_registry.py` — registry unit tests.
- **Create** `providers/venice_dynamic.yaml` — the Venice `openai:chat` provider with `config.passthrough.venice_parameters`.
- **Modify** `promptfooconfig.yaml` — register `venice_dynamic.yaml` under `providers:`.
- **Modify** `scripts_repo/run_comparison.py` — registry-driven validation, env construction, orchestration (start only enabled gateways), eval filter over enabled labels, multi-provider report invocation.
- **Modify** `scripts_repo/tests/test_run_comparison.py` — rewrite validation tests for the new schema; add env/filter/orchestration tests.
- **Modify** `scripts_repo/compare_runs.py` — N-provider row model, columns, delta-vs-baseline, N-way best, tabular summaries, drop status; keep two-file CLI as the 1-candidate special case.
- **Modify** `scripts_repo/tests/test_compare_runs.py` — N-provider tests + prod/dev invariant test. (If this file does not exist, create it.)
- **Modify** `comparisons/example.json` — migrate to the new schema.
- **Modify** `docs/README.md` — rewrite the "Comparison runs" section.

---

## Task 1: Provider registry

**Files:**
- Create: `scripts_repo/providers_registry.py`
- Test: `scripts_repo/tests/test_providers_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# scripts_repo/tests/test_providers_registry.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest scripts_repo/tests/test_providers_registry.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts_repo.providers_registry'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts_repo/providers_registry.py
"""Registry of providers a comparison can run.

One row per provider key (the key used in a comparison config's
``providers-under-test`` / ``provider-options`` and shown as the report column
name). The registry is the single source of truth for how each provider is run:
whether it needs a local plaintext gateway or is a direct external API, which
promptfoo provider label represents it (for --filter-providers and splitting the
unified result file), and the COMPARISON_<PREFIX>_* env prefix its dynamic YAML
templates model/temperature/max_tokens from.

Adding a competitor = one row here + one provider YAML in providers/.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    key: str                 # config key; also the report column name
    label: str               # promptfoo provider label (filter + result split)
    env_prefix: str          # COMPARISON_PROD / COMPARISON_DEV / COMPARISON_VENICE
    kind: str                # "gateway" | "api"
    gateway_port: int | None = None      # gateway only
    vllm_url_key: str | None = None      # gateway only: config key for its vLLM url
    supports_redeploy: bool = False      # only fidaro-dev
    supports_system_prompt: bool = False  # only fidaro-dev
    api_key_env: str | None = None       # api only, e.g. "VENICE_INFERENCE_KEY"


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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest scripts_repo/tests/test_providers_registry.py -q`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/providers_registry.py scripts_repo/tests/test_providers_registry.py
git commit -m "Add provider registry for multi-provider comparisons"
```

---

## Task 2: Venice provider YAML + registration

**Files:**
- Create: `providers/venice_dynamic.yaml`
- Modify: `promptfooconfig.yaml` (providers list, after the dynamic dev provider)

- [ ] **Step 1: Create the Venice provider YAML**

```yaml
# providers/venice_dynamic.yaml
# Venice competitor provider, driven by run_comparison.py.
#
# Like the dynamic Fidaro providers, the model is templated from a
# COMPARISON_VENICE_* env var (keeps promptfoo's request cache key model-aware).
# Venice is a DIRECT external API: no plaintext gateway, web-fetch sidecar, or
# Phala redeploy is involved. Web search is a Venice-specific body param; promptfoo's
# openai:chat provider forwards vendor params only via config.passthrough (it is
# spread verbatim into the request body — there is no config.body for this
# provider). Verified against promptfoo 0.121.12.
#
# Requires VENICE_INFERENCE_KEY in the environment. COMPARISON_VENICE_WEB_SEARCH
# is set by run_comparison.py ("on"/"off"); defaults to "off" when the config
# omits provider-options.venice.web_search.
id: "openai:chat:{{ env.COMPARISON_VENICE_MODEL }}"
label: venice_dynamic
config:
  apiBaseUrl: https://api.venice.ai/api/v1
  apiKey: "{{ env.VENICE_INFERENCE_KEY }}"
  passthrough:
    venice_parameters:
      enable_web_search: "{{ env.COMPARISON_VENICE_WEB_SEARCH }}"
```

- [ ] **Step 2: Register it in promptfooconfig.yaml**

In `promptfooconfig.yaml`, in the `providers:` list, add after the
`fidaro_plaintext_gateway_phala_dynamic_dev.yaml` line (line 19):

```yaml
  # Venice competitor (direct API; no gateway). Driven by run_comparison.py, which
  # sets COMPARISON_VENICE_* and requires VENICE_INFERENCE_KEY. See the YAML header.
  - file://providers/venice_dynamic.yaml
```

- [ ] **Step 3: Sanity-check promptfoo still loads the config**

Run: `pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-providers "venice_dynamic" --filter-metadata "suite=__none__" -n 0 2>&1 | tail -5`
Expected: promptfoo parses the config without a YAML/provider load error (it may report 0 tests selected — that is fine; we only care that the provider file loads). If `-n 0` is unsupported, omit it; the goal is just no config-parse error.

- [ ] **Step 4: Commit**

```bash
git add providers/venice_dynamic.yaml promptfooconfig.yaml
git commit -m "Add Venice competitor provider (config.passthrough web search)"
```

---

## Task 3: run_comparison.py — validation rewrite

This task replaces the prod/dev-specific config keys and validation with the
registry-driven schema. It also fixes the HEAD bug (whitelist check fires
unconditionally).

**Files:**
- Modify: `scripts_repo/run_comparison.py` (constants near lines 48-114; `validate_config` lines 144-211)
- Modify: `scripts_repo/tests/test_run_comparison.py`

- [ ] **Step 1: Write the failing tests**

Replace the validation tests in `scripts_repo/tests/test_run_comparison.py` (the
`test_validate_*` group) with these. Keep a module-level helper for a minimal
valid config:

```python
from scripts_repo.run_comparison import ConfigError, validate_config


def minimal_config(**overrides):
    cfg = {
        "providers-under-test": {"fidaro-prod": True, "venice": True},
        "baseline-provider": "fidaro-prod",
        "provider-options": {
            "fidaro-prod": {"model": "Qwen/Q", "temperature": 0.7, "max_tokens": 100},
            "venice": {"model": "kimi-k2-6", "web_search": "on"},
        },
        "vllm-prod-url": "https://prod.example/v1",
        "suite-generation-config": {"defaults": {"number_to_generate": 0}},
    }
    cfg.update(overrides)
    return cfg


def test_validate_passes_for_minimal_config(tmp_path):
    # venice needs its api key env; prod gateway needs only its vllm url.
    validate_config(minimal_config(), repo_root=tmp_path,
                    env={"VENICE_INFERENCE_KEY": "k"})


def test_validate_rejects_empty_providers(tmp_path):
    cfg = minimal_config()
    cfg["providers-under-test"] = {"fidaro-prod": False, "venice": False}
    with pytest.raises(ConfigError, match="at least one"):
        validate_config(cfg, repo_root=tmp_path, env={})


def test_validate_rejects_unknown_provider_key(tmp_path):
    cfg = minimal_config()
    cfg["providers-under-test"]["mystery"] = True
    with pytest.raises(ConfigError, match="unknown provider"):
        validate_config(cfg, repo_root=tmp_path, env={"VENICE_INFERENCE_KEY": "k"})


def test_validate_baseline_must_be_enabled(tmp_path):
    cfg = minimal_config(**{"baseline-provider": "fidaro-dev"})
    with pytest.raises(ConfigError, match="baseline-provider"):
        validate_config(cfg, repo_root=tmp_path, env={"VENICE_INFERENCE_KEY": "k"})


def test_validate_requires_options_for_each_enabled(tmp_path):
    cfg = minimal_config()
    del cfg["provider-options"]["venice"]
    with pytest.raises(ConfigError, match="provider-options.*venice"):
        validate_config(cfg, repo_root=tmp_path, env={"VENICE_INFERENCE_KEY": "k"})


def test_validate_gateway_requires_vllm_url(tmp_path):
    cfg = minimal_config()
    del cfg["vllm-prod-url"]
    with pytest.raises(ConfigError, match="vllm-prod-url"):
        validate_config(cfg, repo_root=tmp_path, env={"VENICE_INFERENCE_KEY": "k"})


def test_validate_api_requires_key_env(tmp_path):
    with pytest.raises(ConfigError, match="VENICE_INFERENCE_KEY"):
        validate_config(minimal_config(), repo_root=tmp_path, env={})


def test_validate_redeploy_guard_only_with_dev_and_options(tmp_path):
    # No vllm-options => whitelist/instance checks must NOT fire (HEAD bug).
    cfg = minimal_config()
    cfg["providers-under-test"] = {"fidaro-dev": True}
    cfg["baseline-provider"] = "fidaro-dev"
    cfg["provider-options"] = {"fidaro-dev": {"model": "Qwen/Q"}}
    cfg["vllm-dev-url"] = "https://dev.example/v1"
    del cfg["vllm-prod-url"]
    validate_config(cfg, repo_root=tmp_path, env={})  # no error, no instance id needed


def test_validate_vllm_options_requires_whitelisted_instance(tmp_path):
    cfg = minimal_config()
    cfg["providers-under-test"] = {"fidaro-dev": True}
    cfg["baseline-provider"] = "fidaro-dev"
    cfg["provider-options"] = {"fidaro-dev": {"model": "Qwen/Q"}}
    cfg["vllm-dev-url"] = "https://dev.example/v1"
    del cfg["vllm-prod-url"]
    cfg["vllm-options"] = {"model": "Qwen/Q"}
    cfg["phala-dev-instance-id"] = "not-whitelisted"
    compose = tmp_path / "compose.yaml"
    compose.write_text("x")
    (tmp_path / ".env.phala").write_text("x")
    with pytest.raises(ConfigError, match="whitelist"):
        validate_config(cfg, repo_root=tmp_path,
                        env={"PHALA_DOCKER_COMPOSE_FILE": str(compose)})
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest scripts_repo/tests/test_run_comparison.py -q -k validate`
Expected: FAIL (validate_config still uses old schema / old signature).

- [ ] **Step 3: Rewrite the constants and `validate_config`**

In `scripts_repo/run_comparison.py`:

(a) Add the import near the other sibling imports (after the `deploy_phala` imports, ~line 46):

```python
from scripts_repo.providers_registry import REGISTRY, resolve, all_keys
```

(b) Delete the now-obsolete `PROD_PROVIDER` / `DEV_PROVIDER` constants and the
`both_providers_filter`, `PROVIDER_OPTIONS_KEYS`, `REQUIRED_PROVIDER_OPTION_FIELDS`,
and `REQUIRED_KEYS` definitions (lines ~52-53, 62-71, 109, 113-114). They are
replaced below. Keep `SELECT_BEST_ENV_VAR`, the gateway port constants
(now unused directly here but referenced via the registry — actually remove
`PROD_GATEWAY_PORT`/`DEV_GATEWAY_PORT` and read ports from specs), `RESERVED_FILTER_KEYS`,
and `WHITELISTED_CVM_IDS`.

(c) Replace `validate_config` with the registry-driven version:

```python
# Suite config is always required; provider-specific keys are validated per the
# registry based on which providers are enabled.
REQUIRED_KEYS = ("suite-generation-config",)


def enabled_keys(config: dict) -> list[str]:
    """Provider keys switched on in ``providers-under-test`` (config order)."""
    put = config.get("providers-under-test") or {}
    return [k for k, on in put.items() if on]


def validate_config(
    config: dict, repo_root: Path, env: Mapping[str, str] | None = None
) -> None:
    """Validate a comparison config, raising ConfigError on the first problem."""
    if env is None:
        env = os.environ

    for key in REQUIRED_KEYS:
        if key not in config or config[key] in (None, ""):
            raise ConfigError(f"config is missing required key {key!r}")

    put = config.get("providers-under-test")
    if not put:
        raise ConfigError("config is missing required key 'providers-under-test'")
    unknown = set(put) - all_keys()
    if unknown:
        raise ConfigError(f"unknown provider(s) in providers-under-test: {sorted(unknown)}")

    enabled = enabled_keys(config)
    if not enabled:
        raise ConfigError("providers-under-test must enable at least one provider")

    baseline = config.get("baseline-provider")
    if baseline not in enabled:
        raise ConfigError(
            f"baseline-provider {baseline!r} must be one of the enabled "
            f"providers: {enabled}"
        )

    options = config.get("provider-options") or {}
    extra = set(options) - set(enabled)
    if extra:
        raise ConfigError(
            f"provider-options has entries for non-enabled providers: {sorted(extra)}"
        )
    for key in enabled:
        if key not in options:
            raise ConfigError(f"provider-options is missing an entry for {key!r}")

    specs = resolve(enabled)
    for spec in specs:
        if spec.kind == "gateway":
            url = config.get(spec.vllm_url_key)
            if not url:
                raise ConfigError(
                    f"provider {spec.key!r} requires config key {spec.vllm_url_key!r}"
                )
        elif spec.kind == "api":
            if spec.api_key_env and not env.get(spec.api_key_env):
                raise ConfigError(
                    f"provider {spec.key!r} requires the {spec.api_key_env} env var"
                )

    # Redeploy is only meaningful when fidaro-dev is enabled AND vllm-options is
    # set. Without vllm-options nothing touches a CVM, so the instance id / whitelist
    # carry no constraints (this is the HEAD bug: the whitelist check used to fire
    # unconditionally).
    has_options = bool(config.get("vllm-options"))
    dev_enabled = "fidaro-dev" in enabled
    if has_options and dev_enabled:
        vllm_model = config["vllm-options"].get("model")
        dev_model = (options.get("fidaro-dev") or {}).get("model")
        if vllm_model and dev_model and dev_model != vllm_model:
            raise ConfigError(
                f"provider-options['fidaro-dev'].model ({dev_model!r}) must match "
                f"vllm-options.model ({vllm_model!r})"
            )
        instance_id = config.get("phala-dev-instance-id")
        if not instance_id:
            raise ConfigError(
                "vllm-options requires phala-dev-instance-id (a redeploy target)"
            )
        if instance_id not in WHITELISTED_CVM_IDS:
            raise ConfigError(
                f"phala-dev-instance-id {instance_id} is not in the whitelist "
                "of allowed CVM IDs"
            )
        compose = env.get("PHALA_DOCKER_COMPOSE_FILE")
        if not compose:
            raise ConfigError(
                "vllm-options requires the PHALA_DOCKER_COMPOSE_FILE env var"
            )
        if not Path(compose).is_file():
            raise ConfigError(f"PHALA_DOCKER_COMPOSE_FILE does not exist: {compose}")
        env_phala = repo_root / ".env.phala"
        if not env_phala.exists():
            raise ConfigError(
                f"vllm-options requires an env variable file at {env_phala}. "
                "Prefer to use 1Password environments for this."
            )

    prompt_file = config.get("system-prompt-file")
    if prompt_file and not Path(prompt_file).is_file():
        raise ConfigError(f"system-prompt-file does not exist: {prompt_file}")
    if prompt_file and not any(s.supports_system_prompt for s in specs):
        print(
            "WARNING: system-prompt-file is set but no enabled provider mounts a "
            "system prompt (only fidaro-dev does); it will be ignored.",
            flush=True,
        )
```

- [ ] **Step 4: Run to verify the validation tests pass**

Run: `python -m pytest scripts_repo/tests/test_run_comparison.py -q -k validate`
Expected: PASS for all `validate` tests. (Other tests in the file may still fail — fixed in Task 4. Note the migrated `example.json` is exercised in Task 5.)

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/run_comparison.py scripts_repo/tests/test_run_comparison.py
git commit -m "Registry-driven config validation for N providers; fix unconditional whitelist check"
```

---

## Task 4: run_comparison.py — env construction, filter, orchestration

**Files:**
- Modify: `scripts_repo/run_comparison.py` (`provider_options_env`, the filter helper, `main`)
- Modify: `scripts_repo/tests/test_run_comparison.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts_repo.run_comparison import (
    provider_options_env, providers_filter, report_provider_args,
)


def test_provider_options_env_builds_per_provider_prefixes():
    cfg = {
        "providers-under-test": {"fidaro-prod": True, "venice": True},
        "provider-options": {
            "fidaro-prod": {"model": "Qwen/Q", "temperature": 0.7, "max_tokens": 100},
            "venice": {"model": "kimi-k2-6", "web_search": "on"},
        },
    }
    env = provider_options_env(cfg)
    assert env["COMPARISON_PROD_MODEL"] == "Qwen/Q"
    assert env["COMPARISON_PROD_TEMPERATURE"] == "0.7"
    assert env["COMPARISON_PROD_MAX_TOKENS"] == "100"
    assert env["COMPARISON_VENICE_MODEL"] == "kimi-k2-6"
    assert env["COMPARISON_VENICE_WEB_SEARCH"] == "on"
    # Optional fields absent for venice are simply not set (Venice may send none).
    assert "COMPARISON_VENICE_TEMPERATURE" not in env


def test_provider_options_env_web_search_defaults_off():
    cfg = {
        "providers-under-test": {"venice": True},
        "provider-options": {"venice": {"model": "kimi-k2-6"}},
    }
    assert provider_options_env(cfg)["COMPARISON_VENICE_WEB_SEARCH"] == "off"


def test_providers_filter_anchors_enabled_labels():
    cfg = {"providers-under-test": {"fidaro-prod": True, "venice": True}}
    regex = providers_filter(cfg)
    import re
    assert re.match(regex, "venice_dynamic")
    assert re.match(regex, "fidaro_plaintext_gateway_phala_dynamic_prod")
    assert not re.match(regex, "fidaro_plaintext_gateway_phala_dynamic_dev")


def test_report_provider_args_baseline_first_and_flagged():
    cfg = {
        "providers-under-test": {"fidaro-prod": True, "venice": True},
        "baseline-provider": "fidaro-prod",
    }
    args = report_provider_args(cfg)
    # baseline first, flagged; others follow in config order.
    assert args[:2] == ["--baseline-provider-col",
                        "fidaro-prod=fidaro_plaintext_gateway_phala_dynamic_prod"]
    assert "--provider-col" in args
    assert "venice=venice_dynamic" in args
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest scripts_repo/tests/test_run_comparison.py -q -k "env or filter or report_provider"`
Expected: FAIL (functions don't exist / old signatures).

- [ ] **Step 3: Implement the helpers**

In `scripts_repo/run_comparison.py`, replace `provider_options_env` and
`both_providers_filter` with:

```python
def provider_options_env(config: dict) -> dict[str, str]:
    """Map each enabled provider's options to its COMPARISON_<PREFIX>_* env vars.

    The dynamic provider YAMLs template their model (and, for Fidaro, temperature
    / max_tokens) from these. Optional fields are set only when present, because a
    competitor API may send none. Venice additionally gets COMPARISON_VENICE_WEB_SEARCH
    (defaults to "off"). Keep prefixes in sync with providers_registry / the YAMLs.
    """
    options = config.get("provider-options") or {}
    out: dict[str, str] = {}
    for spec in resolve(enabled_keys(config)):
        opts = options.get(spec.key) or {}
        if opts.get("model") is not None:
            out[f"{spec.env_prefix}_MODEL"] = str(opts["model"])
        if opts.get("temperature") is not None:
            out[f"{spec.env_prefix}_TEMPERATURE"] = str(opts["temperature"])
        if opts.get("max_tokens") is not None:
            out[f"{spec.env_prefix}_MAX_TOKENS"] = str(opts["max_tokens"])
        if spec.key == "venice":
            out["COMPARISON_VENICE_WEB_SEARCH"] = str(opts.get("web_search", "off"))
    return out


def providers_filter(config: dict) -> str:
    """A --filter-providers regex anchored to exactly the enabled providers' labels."""
    labels = [re.escape(s.label) for s in resolve(enabled_keys(config))]
    return f"^({'|'.join(labels)})$"


def report_provider_args(config: dict) -> list[str]:
    """`compare_runs.py` CLI args naming the baseline and other provider columns.

    Each arg is ``key=label`` so the report shows config keys but splits the
    unified result file by promptfoo provider label. Baseline first (flagged),
    others follow in config order.
    """
    baseline = config["baseline-provider"]
    others = [k for k in enabled_keys(config) if k != baseline]
    args = ["--baseline-provider-col", f"{baseline}={REGISTRY[baseline].label}"]
    for key in others:
        args += ["--provider-col", f"{key}={REGISTRY[key].label}"]
    return args
```

- [ ] **Step 4: Rewrite the orchestration in `main`**

Replace the prod/dev-specific orchestration in `main` (the redeploy block, the
two `start_gateway` calls, the readiness waits, and the report invocation) with a
registry-driven version. Key changes:

```python
    enabled = enabled_keys(config)
    specs = resolve(enabled)
    gateway_specs = [s for s in specs if s.kind == "gateway"]

    # ... (suite config write + os.environ updates unchanged) ...
    os.environ.update(provider_options_env(config))
    if len(enabled) >= 2:
        os.environ[SELECT_BEST_ENV_VAR] = "1"

    # Redeploy only when fidaro-dev is enabled and vllm-options changed.
    options = config.get("vllm-options")
    if options and "fidaro-dev" in enabled:
        # ... existing cache + _confirm_redeploy + phala_deploy_and_wait block,
        # unchanged except it is now guarded by the dev-enabled check ...

    # Start the shared web-fetch sidecar + gateways only if any gateway provider
    # is enabled (a venice-only run needs none of this).
    if gateway_specs:
        ensure_docker_network(COMPARISON_NETWORK)
        start_gateway(repo_root, web_fetch_docker_args(
            name=WEB_FETCH_CONTAINER_NAME, image=args.web_fetch_image,
            network=COMPARISON_NETWORK), name=WEB_FETCH_CONTAINER_NAME)
        wait_for_container_healthy(WEB_FETCH_CONTAINER_NAME, timeout_s=args.sidecar_timeout)

        for spec in gateway_specs:
            start_gateway(
                repo_root,
                gateway_docker_args(
                    name=f"fidaro-gateway-{spec.key.split('-')[-1]}",
                    port=spec.gateway_port,
                    vllm_url=config[spec.vllm_url_key],
                    brave_api_key=brave_api_key,
                    image=args.docker_image,
                    network=COMPARISON_NETWORK,
                    web_fetch_url=WEB_FETCH_SIDECAR_URL,
                    system_prompt_file=(config.get("system-prompt-file")
                                        if spec.supports_system_prompt else None),
                ),
                name=f"fidaro-gateway-{spec.key.split('-')[-1]}",
            )
        for spec in gateway_specs:
            wait_for_url(models_url(config[spec.vllm_url_key]), timeout_s=args.gateway_timeout)
            wait_for_url(gateway_health_url(spec.gateway_port), timeout_s=args.gateway_timeout)

    # BRAVE_API_KEY is only needed when a gateway runs; move its check under
    # `if gateway_specs:` (api-only runs do not use Brave).
```

And the eval + report calls:

```python
    print("Running unified eval over enabled providers ...")
    _run(eval_command(providers_filter(config), str(results_out), filter_args,
                      True, f"{name} comparison"), cwd=repo_root, check=False)

    report = run_dir / f"report__{ts}.html"
    report_cmd = [
        sys.executable, str(repo_root / "scripts_repo" / "compare_runs.py"),
        str(results_out), str(results_out),
        *report_provider_args(config),
        "--out", str(report),
        "--config-path", str(config_path.resolve()),
    ]
    system_prompt_file = config.get("system-prompt-file")
    if system_prompt_file:
        report_cmd += ["--system-prompt-path", str(Path(system_prompt_file).resolve())]
    _run(report_cmd, cwd=repo_root)
```

Move the `brave_api_key` fetch + error so it only blocks when `gateway_specs` is
non-empty (api-only runs skip it). Leave the viewer-container + `open` calls as-is.

- [ ] **Step 5: Run the run_comparison tests**

Run: `python -m pytest scripts_repo/tests/test_run_comparison.py -q`
Expected: PASS (all). Fix any stragglers referencing removed constants
(`PROD_PROVIDER`, `both_providers_filter`, etc.).

- [ ] **Step 6: Commit**

```bash
git add scripts_repo/run_comparison.py scripts_repo/tests/test_run_comparison.py
git commit -m "Registry-driven orchestration: per-provider env, filter, report args"
```

---

## Task 5: Migrate example.json + validate it loads

**Files:**
- Modify: `comparisons/example.json`

- [ ] **Step 1: Rewrite example.json to the new schema**

```json
{
  "vllm-prod-url": "https://3bae0d639fdab82b8ea3bff4d8ff1515d38455e5-8000.dstack-pha-use1.phala.network/v1",
  "vllm-dev-url": "https://099098730db24cb5c3d6d7e24bf97c769f1be1c4-8000.dstack-pha-use1.phala.network/v1",
  "providers-under-test": { "fidaro-prod": true, "fidaro-dev": false, "venice": true },
  "baseline-provider": "fidaro-prod",
  "provider-options": {
    "fidaro-prod": { "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8", "temperature": 0.7, "max_tokens": 100000 },
    "venice": { "model": "kimi-k2-6", "web_search": "on" }
  },
  "system-prompt-file": "system_prompts/fidaro_prod.md",
  "phala-dev-instance-id": "fidaro-vllm-002",
  "promptfoo-filters": {},
  "suite-generation-config": {
    "defaults": { "number_to_generate": 0 },
    "simple_facts": { "number_to_generate": null, "randomize_selection": true, "random_seed": 0, "max_rubrics": null },
    "research_rubrics": { "number_to_generate": 10, "randomize_selection": true, "random_seed": 0, "max_rubrics": 5 }
  }
}
```

(Trim the suite config to a small representative set; the full one is fine too —
the point is the new top-level schema. `vllm-dev-url` is retained for the commented-out
`fidaro-dev` so toggling it on needs no edit.)

- [ ] **Step 2: Add a test that example.json validates**

In `scripts_repo/tests/test_run_comparison.py`:

```python
import json
from pathlib import Path
from scripts_repo.run_comparison import validate_config


def test_example_config_validates(tmp_path):
    cfg = json.loads(Path("comparisons/example.json").read_text())
    validate_config(cfg, repo_root=tmp_path, env={"VENICE_INFERENCE_KEY": "k"})
```

(The `system-prompt-file` path must exist relative to cwd; if the test runs from
repo root it resolves. If it fails on the prompt file, drop `system-prompt-file`
from example.json or assert via a config copy without it.)

- [ ] **Step 3: Run**

Run: `python -m pytest scripts_repo/tests/test_run_comparison.py::test_example_config_validates -q`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add comparisons/example.json scripts_repo/tests/test_run_comparison.py
git commit -m "Migrate example comparison config to multi-provider schema"
```

---

## Task 6: compare_runs.py — N-provider data model

This is the core report rewrite. Build it test-first in two tasks: the data
model (this task) then the HTML/summaries (Task 7).

**Files:**
- Modify: `scripts_repo/compare_runs.py`
- Modify/Create: `scripts_repo/tests/test_compare_runs.py`

- [ ] **Step 1: Write the failing tests**

```python
# scripts_repo/tests/test_compare_runs.py
from __future__ import annotations

from scripts_repo.compare_runs import (
    Cell, CellKey, ProviderColumn, build_rows, summarize_rubric_table,
    best_winner_among,
)


def _rubric(key, score):
    return Cell(key=key, suite="s", kind="rubric", metric=None, weight=1.0,
                assertion_value=key.assertion, score=score)


def test_build_rows_collects_per_provider_values_and_deltas():
    k = CellKey(test="t", prompt="p", assertion="a")
    cols = [ProviderColumn("fidaro-prod", "L_prod", is_baseline=True),
            ProviderColumn("venice", "L_ven", is_baseline=False)]
    cells_by_provider = {
        "fidaro-prod": {k: _rubric(k, 0.8)},
        "venice": {k: _rubric(k, 0.6)},
    }
    rows = build_rows(cells_by_provider, cols)
    assert len(rows) == 1
    row = rows[0]
    assert row.values["fidaro-prod"] == 0.8
    assert row.values["venice"] == 0.6
    assert round(row.deltas["venice"], 2) == -0.20  # other - baseline


def test_best_winner_among_returns_passing_provider():
    k = CellKey(test="t", prompt="p", assertion="best")
    bcell = lambda passed: Cell(key=k, suite="s", kind="best", metric=None,
                                weight=1.0, assertion_value="best", passed=passed)
    per_provider = {"fidaro-prod": bcell(False), "venice": bcell(True),
                    "fidaro-dev": bcell(False)}
    assert best_winner_among(per_provider) == "venice"
    # No clean single winner => None.
    assert best_winner_among({"a": bcell(False), "b": bcell(False)}) is None


def test_summarize_rubric_table_counts_vs_baseline():
    k1 = CellKey("t1", "p", "a"); k2 = CellKey("t2", "p", "a")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    cells = {
        "base": {k1: _rubric(k1, 0.5), k2: _rubric(k2, 0.5)},
        "ven":  {k1: _rubric(k1, 0.9), k2: _rubric(k2, 0.5)},  # k1 improved, k2 within
    }
    rows = build_rows(cells, cols)
    table = summarize_rubric_table(rows, cols, tolerance=0.05)
    assert table["ven"]["improved"] == 1
    assert table["ven"]["within"] == 1
    assert table["ven"]["regressed"] == 0
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest scripts_repo/tests/test_compare_runs.py -q`
Expected: FAIL (`ProviderColumn`, `build_rows`, etc. not defined).

- [ ] **Step 3: Implement the N-provider model**

In `scripts_repo/compare_runs.py`, add (keep `Cell`, `CellKey`, `extract_cells`,
`classify`, `_classify_deterministic` as they are):

```python
@dataclass(frozen=True)
class ProviderColumn:
    key: str        # config key, shown as the column header
    label: str      # promptfoo provider label, used to split the eval file
    is_baseline: bool


@dataclass
class Row:
    key: "CellKey"
    suite: str
    kind: str                      # "rubric" | "deterministic" | "best"
    metric: str | None
    assertion_value: str
    assertion_type: str
    values: dict                   # provider key -> score (rubric) / pass (det) / None
    deltas: dict                   # other-provider key -> (other - baseline) | None
    best: str | None               # winning provider key for kind == "best"
    search: str = ""


def best_winner_among(per_provider: dict) -> str | None:
    """Provider key whose select-best component passed, or None if not exactly one."""
    winners = [k for k, c in per_provider.items()
               if c is not None and c.passed is True]
    return winners[0] if len(winners) == 1 else None


def build_rows(cells_by_provider: dict, columns: list) -> list:
    """Join per-provider cell maps into one Row per CellKey.

    `cells_by_provider` maps provider key -> {CellKey: Cell}. `columns` is the
    ordered ProviderColumn list (baseline first). Rubric deltas are computed as
    (other - baseline) when both sides have a rubric score.
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    all_keys = set()
    for m in cells_by_provider.values():
        all_keys |= set(m)
    rows = []
    for key in sorted(all_keys, key=lambda k: (k.test, k.prompt, k.assertion)):
        per_provider = {c.key: cells_by_provider.get(c.key, {}).get(key)
                        for c in columns}
        present = next((c for c in per_provider.values() if c is not None), None)
        if present is None:
            continue
        kind = present.kind
        values, deltas = {}, {}
        for ckey, cell in per_provider.items():
            if cell is None:
                values[ckey] = None
            elif kind == "rubric":
                values[ckey] = cell.score
            elif kind == "deterministic":
                values[ckey] = cell.passed
            else:  # best
                values[ckey] = cell.passed
        if kind == "rubric":
            base_cell = per_provider.get(baseline)
            for o in others:
                oc = per_provider.get(o)
                deltas[o] = (oc.score - base_cell.score) \
                    if (oc is not None and base_cell is not None) else None
        best = best_winner_among(per_provider) if kind == "best" else None
        rows.append(Row(key=key, suite=present.suite, kind=kind,
                        metric=present.metric, assertion_value=present.assertion_value,
                        assertion_type=present.assertion_type, values=values,
                        deltas=deltas, best=best, search=present.search))
    return rows


def summarize_rubric_table(rows: list, columns: list, tolerance: float) -> dict:
    """Per-non-baseline-provider rubric tally vs baseline.

    Returns {provider_key: {improved, regressed, within, new, removed}}.
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    out = {o: {"improved": 0, "regressed": 0, "within": 0, "new": 0, "removed": 0}
           for o in others}
    for row in rows:
        if row.kind != "rubric":
            continue
        b = row.values.get(baseline)
        for o in others:
            v = row.values.get(o)
            if b is None and v is not None:
                out[o]["new"] += 1
            elif b is not None and v is None:
                out[o]["removed"] += 1
            elif b is not None and v is not None:
                out[o][classify(v - b, tolerance)] += 1
    return out


def summarize_deterministic_table(rows: list, columns: list) -> dict:
    """Per-non-baseline-provider deterministic tally vs baseline."""
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    out = {o: {"new_passes": 0, "new_fails": 0, "total_passes": 0, "total_fails": 0}
           for o in others}
    for row in rows:
        if row.kind != "deterministic":
            continue
        b = row.values.get(baseline)
        for o in others:
            v = row.values.get(o)
            if v is True:
                out[o]["total_passes"] += 1
            elif v is False:
                out[o]["total_fails"] += 1
            if b is not None and v is not None and b != v:
                out[o]["new_passes" if v else "new_fails"] += 1
    return out


def summarize_best_table(rows: list, columns: list) -> dict:
    """Wins per provider key across best rows (+ undecided)."""
    out = {c.key: 0 for c in columns}
    out["undecided"] = 0
    for row in rows:
        if row.kind != "best":
            continue
        out[row.best if row.best in out else "undecided"] += 1
    return out
```

- [ ] **Step 4: Run to verify pass**

Run: `python -m pytest scripts_repo/tests/test_compare_runs.py -q`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/compare_runs.py scripts_repo/tests/test_compare_runs.py
git commit -m "compare_runs: N-provider row model + summary tables"
```

---

## Task 7: compare_runs.py — HTML render + CLI + prod/dev invariant

**Files:**
- Modify: `scripts_repo/compare_runs.py` (render, `build_parser`, `main`)
- Modify: `scripts_repo/tests/test_compare_runs.py`

- [ ] **Step 1: Write the failing tests**

```python
from scripts_repo.compare_runs import render_html_n, parse_provider_col_args


def test_parse_provider_col_args_orders_baseline_first():
    cols = parse_provider_col_args(
        baseline="fidaro-prod=L_prod",
        others=["venice=L_ven", "fidaro-dev=L_dev"],
    )
    assert [c.key for c in cols] == ["fidaro-prod", "venice", "fidaro-dev"]
    assert cols[0].is_baseline and not cols[1].is_baseline
    assert cols[0].label == "L_prod"


def test_render_html_n_has_provider_columns_and_no_status():
    from scripts_repo.compare_runs import CellKey, Cell, build_rows, ProviderColumn
    k = CellKey("t", "p", "a")
    cols = [ProviderColumn("fidaro-prod", "L1", True),
            ProviderColumn("venice", "L2", False)]
    cells = {
        "fidaro-prod": {k: Cell(k, "s", "rubric", None, 1.0, "a", score=0.8,
                                assertion_type="llm-rubric")},
        "venice": {k: Cell(k, "s", "rubric", None, 1.0, "a", score=0.6,
                            assertion_type="llm-rubric")},
    }
    rows = build_rows(cells, cols)
    html_out = render_html_n(rows, cols, drift=([], []), tolerance=0.05)
    assert "fidaro-prod (baseline)" in html_out
    assert "venice" in html_out
    assert "&Delta; venice" in html_out or "Δ venice" in html_out
    assert "<th>status</th>" not in html_out
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest scripts_repo/tests/test_compare_runs.py -q -k "render or provider_col"`
Expected: FAIL (`render_html_n`, `parse_provider_col_args` undefined).

- [ ] **Step 3: Implement render + CLI parsing**

Add to `compare_runs.py`. Render builds dynamic columns from `columns`; the
summary block uses the three summary tables from Task 6. Keep cell hyperlinking /
copy-curl behaviour where feasible (per-provider eval id + curls keyed by column);
for the first cut, link/curl the baseline and each provider using a
`{provider_key: eval_id}` / `{provider_key: curls}` map (eval id is the same for a
unified file, so a single id is fine).

```python
def parse_provider_col_args(baseline: str, others: list) -> list:
    """Turn `key=label` CLI args into ordered ProviderColumns (baseline first)."""
    def split(s):
        key, _, label = s.partition("=")
        return key, label
    bkey, blabel = split(baseline)
    cols = [ProviderColumn(bkey, blabel, is_baseline=True)]
    for o in others:
        k, l = split(o)
        cols.append(ProviderColumn(k, l, is_baseline=False))
    return cols


def _summary_tables_html(rows, columns, tolerance, css_class):
    """Render rubric / deterministic / best summary tables (only non-empty kinds)."""
    parts = []
    others = [c.key for c in columns if not c.is_baseline]
    baseline = next(c.key for c in columns if c.is_baseline)
    if any(r.kind == "rubric" for r in rows):
        t = summarize_rubric_table(rows, columns, tolerance)
        head = ("<tr><th>vs " + html.escape(baseline) + "</th><th>improved</th>"
                "<th>regressed</th><th>within &plusmn;" + f"{tolerance:g}" +
                "</th><th>new</th><th>removed</th></tr>")
        body = "".join(
            f"<tr><td>{html.escape(o)}</td><td class='num'>{t[o]['improved']}</td>"
            f"<td class='num'>{t[o]['regressed']}</td><td class='num'>{t[o]['within']}</td>"
            f"<td class='num'>{t[o]['new']}</td><td class='num'>{t[o]['removed']}</td></tr>"
            for o in others)
        parts.append("<h4>Rubric</h4><table>" + head + body + "</table>")
    if any(r.kind == "deterministic" for r in rows):
        t = summarize_deterministic_table(rows, columns)
        head = ("<tr><th>vs " + html.escape(baseline) + "</th><th>new passes</th>"
                "<th>new fails</th><th>total passes</th><th>total fails</th></tr>")
        body = "".join(
            f"<tr><td>{html.escape(o)}</td><td class='num'>{t[o]['new_passes']}</td>"
            f"<td class='num'>{t[o]['new_fails']}</td><td class='num'>{t[o]['total_passes']}</td>"
            f"<td class='num'>{t[o]['total_fails']}</td></tr>" for o in others)
        parts.append("<h4>Deterministic</h4><table>" + head + body + "</table>")
    if any(r.kind == "best" for r in rows):
        t = summarize_best_table(rows, columns)
        cells = "".join(f"<td>{html.escape(c.key)}: <b>{t[c.key]}</b></td>"
                        for c in columns)
        parts.append("<h4>Best (head-to-head)</h4><table><tr>" + cells +
                     f"<td>undecided: {t['undecided']}</td></tr></table>")
    return f"<div class='{css_class}'>{''.join(parts)}</div>"


def render_html_n(rows, columns, drift, tolerance,
                  eval_ids=None, ui_base_url=DEFAULT_UI_BASE_URL,
                  errored=None, curls=None, config_path=None,
                  system_prompt_path=None) -> str:
    """Render the N-provider HTML report (no status column; per-provider deltas)."""
    eval_ids = eval_ids or {}
    others = [c for c in columns if not c.is_baseline]

    def col_header(c):
        return html.escape(c.key) + (" (baseline)" if c.is_baseline else "")

    value_headers = "".join(f"<th>{col_header(c)}</th>" for c in columns)
    delta_headers = "".join(f"<th>&Delta; {html.escape(c.key)}</th>" for c in others)
    thead = ("<th>test</th><th>assertion type</th><th>assertion</th><th>metric</th>"
             + value_headers + delta_headers + "<th>best</th>")

    # group rows by suite, then by (test, prompt) — mirrors the old grouping.
    # ... build <tbody> rows: for each provider column a num cell (score/pass/—),
    #     then a delta cell per other (— for deterministic/best), then best key.
    #     A kind=="best" row shows the winner in the best column and dashes elsewhere.
    # (Implementation detail: reuse _value_html for score/verdict formatting.)
    aggregate = _summary_tables_html(rows, columns, tolerance, "summary")
    # ... assemble sections per suite with _summary_tables_html(suite_rows, ...) ...
    # ... drift banner unchanged (only_base/only_cand now generalize to
    #     baseline-vs-union; keep the existing two-set diff between baseline and the
    #     union of other providers' tests) ...
    return (  # full document, same skeleton as render_html
        "<!doctype html>..."  # NOTE: assemble exactly like the existing render_html,
    )                          # swapping the table head/body + summary for the above.
```

Implementation note for the engineer: model `render_html_n` on the existing
`render_html` (suite grouping, parity banding, `_eval_header`, `_COPY_SCRIPT`),
changing only (a) the column set, (b) per-provider value + delta cells, (c) the
summary to `_summary_tables_html`, and (d) dropping the status column. Sort rows
within a group by the most-negative delta across `others` (regressions on top):

```python
def _row_sort_key(row, others):
    ds = [row.deltas[o] for o in others if row.deltas.get(o) is not None]
    return (0, min(ds)) if ds else (1, 0.0)
```

- [ ] **Step 4: Rewrite `build_parser` + `main`**

Add `--baseline-provider-col` (single `key=label`) and `--provider-col`
(repeatable `key=label`). Keep the legacy `--baseline-provider`/`--candidate-provider`
+ positional files working: when the new flags are absent but the legacy ones are
present, synthesize two columns (baseline + candidate) so the two-file/frozen path
still renders via `render_html_n`. `main` becomes:

```python
def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    suites = args.suite
    baseline_json = read_eval_json(args.baseline_json)
    candidate_json = read_eval_json(args.candidate_json)  # same file in unified mode

    if args.baseline_provider_col:
        columns = parse_provider_col_args(args.baseline_provider_col,
                                          args.provider_col or [])
    else:
        # legacy two-file / two-provider path
        columns = [
            ProviderColumn("baseline", args.baseline_provider or "baseline", True),
            ProviderColumn("candidate", args.candidate_provider or "candidate", False),
        ]

    # Build per-provider cell maps. In unified mode all columns read from
    # baseline_json (one file, split by label); in legacy two-file mode the
    # baseline column reads baseline_json and the single other reads candidate_json.
    cells_by_provider = {}
    for c in columns:
        src = baseline_json if (c.is_baseline or args.baseline_provider_col) else candidate_json
        label = None if (not args.baseline_provider_col and c.label in ("baseline", "candidate")) else c.label
        cells_by_provider[c.key] = extract_cells(src, suites, label)

    rows = build_rows(cells_by_provider, columns)
    # drift: baseline tests vs union of others
    drift = _drift_n(cells_by_provider, columns)
    eval_ids = {c.key: read_eval_id(baseline_json) for c in columns}
    args.out.write_text(
        render_html_n(rows, columns, drift, args.tolerance, eval_ids=eval_ids,
                      ui_base_url=args.ui_base_url, config_path=args.config_path,
                      system_prompt_path=args.system_prompt_path),
        encoding="utf-8")
    print(f"wrote {args.out} ({len(columns)} providers, {len(rows)} rows)")
    return 0
```

Add a small `_drift_n(cells_by_provider, columns)` returning
`(missing_from_some_other, only_in_some_other)` by comparing the baseline's test
set against the union of the others' test sets (keeps the existing drift banner
meaningful). Keep `parse_provider_yaml` / `build_curls` available; curl wiring can
be added per-column in a follow-up — for this task it is acceptable to render
without copy-curl buttons (note this in the commit message) to keep the change
focused, OR wire `curls={c.key: build_curls(...)}` if straightforward.

- [ ] **Step 5: Run all compare_runs tests**

Run: `python -m pytest scripts_repo/tests/test_compare_runs.py -q`
Expected: PASS.

- [ ] **Step 6: prod/dev invariant test**

Add a test that a synthetic unified eval JSON with two provider labels
(`...dynamic_prod`, `...dynamic_dev`), one rubric assertion, produces a report
with `fidaro-prod (baseline)`, `fidaro-dev`, one `Δ fidaro-dev` column, and the
correct improved/regressed tally — i.e. the prod-vs-dev semantics are preserved.

```python
def test_prod_dev_invariant(tmp_path):
    # minimal unified eval JSON: same test, two providers, rubric 0.8 vs 0.9
    eval_json = {
      "results": {"results": [
        _result("...dynamic_prod", 0.8), _result("...dynamic_dev", 0.9),
      ]}, "config": {"providers": []}, "evalId": "e1",
    }
    f = tmp_path / "u.json"; f.write_text(json.dumps(eval_json))
    out = tmp_path / "r.html"
    main([str(f), str(f),
          "--baseline-provider-col", "fidaro-prod=...dynamic_prod",
          "--provider-col", "fidaro-dev=...dynamic_dev",
          "--out", str(out)])
    text = out.read_text()
    assert "fidaro-prod (baseline)" in text and "&Delta; fidaro-dev" in text
```

(Provide a `_result(label, score)` helper that builds the minimal promptfoo result
shape `extract_cells` expects: `provider.label`, `testCase.description`,
`testCase.metadata.suite`, `testCase.assert` with one `llm-rubric`, and
`gradingResult.componentResults[0].score`.)

- [ ] **Step 7: Commit**

```bash
git add scripts_repo/compare_runs.py scripts_repo/tests/test_compare_runs.py
git commit -m "compare_runs: N-provider HTML report, drop status, prod/dev invariant"
```

---

## Task 8: Docs

**Files:**
- Modify: `docs/README.md` ("Comparison runs (config-driven)" section, ~lines 100-163)

- [ ] **Step 1: Rewrite the comparison section**

Update the section to describe: `providers-under-test` + `baseline-provider` +
`provider-options`; the registry (`scripts_repo/providers_registry.py`); that
Venice is a direct API needing `VENICE_INFERENCE_KEY` and **no** gateway/sidecar/
redeploy; that only enabled gateway providers start a gateway; the new report
shape (baseline + per-provider columns + per-provider deltas + N-way best, no
status column, tabular summary). Link the new spec. Remove references to
`prod-provider-options`/`dev-provider-options`, `PROD_PROVIDER`/`DEV_PROVIDER`,
and `both_providers_filter`.

- [ ] **Step 2: Verify the whole suite is green**

Run: `python -m pytest scripts_repo/tests/ -q`
Expected: PASS (no failures, including the 6 that failed at HEAD).

- [ ] **Step 3: Commit**

```bash
git add docs/README.md
git commit -m "docs: document multi-provider comparisons and Venice competitor"
```

---

## Final verification

- [ ] `python -m pytest scripts_repo/tests/ -q` — all green.
- [ ] `pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-providers venice_dynamic --filter-metadata suite=__none__ 2>&1 | tail -5` — config loads cleanly.
- [ ] (When `VENICE_INFERENCE_KEY` is available) a tiny smoke run: a 1-test comparison `{fidaro-prod, venice}` to confirm the `venice_dynamic` label appears in the result JSON and the report renders a `venice` column + `Δ venice` + `best`. This closes the last spec risk (api-provider result split).
- [ ] Leave on branch `worktree-multi-provider-comparison` (no PR/merge, per the user's choice).

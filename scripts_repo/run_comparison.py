#!/usr/bin/env python3
"""Run one prod-vs-dev Fidaro comparison from a single config file.

Given a config JSON (see docs/superpowers/specs/
2026-05-22-comparison-orchestrator-design.md), this script:

  1. Creates an isolated output directory beside the config file, named after
     its stem (comparisons/example.json -> comparisons/example/).
  2. Validates the config.
  3. Redeploys the Phala dev CVM with new vLLM options *only* when they changed
     (decided via a cache file), after an explicit confirmation.
  4. Starts the prod and dev plaintext gateways in Docker (mounting a dev
     system prompt if given).
  5. Runs the promptfoo suite once against both providers, into a single
     timestamped result file under that per-config output directory.
  6. Builds the comparison report (splitting that file into prod/dev sides) and
     opens it, then ensures the promptfoo viewer container is up.

It does NOT freeze a baseline.

Usage:
    run_comparison.py comparisons/prod_vs_dev_gemma.json [--yes]

BRAVE_API_KEY must be set in the environment (it is never read from config).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Mapping

# Allow both `python scripts_repo/run_comparison.py` (script) and
# `from scripts_repo.run_comparison import ...` (tests): make the repo root
# importable so the sibling import below resolves either way.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts_repo.deploy_phala import deploy as phala_deploy_and_wait
from scripts_repo.deploy_phala import models_url, wait_for_url
from scripts_repo.providers_registry import REGISTRY, all_keys, resolve

# Provider labels, gateway ports, env prefixes, and kinds now live in the
# registry (scripts_repo/providers_registry.py). The dynamic providers template
# their model / temperature / max_tokens from per-provider COMPARISON_<PREFIX>_*
# env vars (see provider_options_env), so they reflect a reconfigured gateway
# (the static YAML providers can't).

# Set in the eval's environment so tests/classification.py:augment appends a
# head-to-head `select-best` assertion. Only meaningful here, where prod and dev
# are graded in one eval; single-provider runs leave it unset. Keep in sync with
# tests/classification.py (SELECT_BEST_ENV_VAR).
SELECT_BEST_ENV_VAR = "COMPARISON_SELECT_BEST"


def providers_filter(config: dict) -> str:
    """A --filter-providers regex selecting exactly the enabled providers' labels.

    promptfoo's --filter-providers is a regex matched against each provider's
    id/label. Anchoring on the enabled dynamic labels keeps the static
    (non-dynamic) providers and the vLLM provider out of the unified pass. This
    generalizes the old two-provider both_providers_filter to the registry's
    enabled set, so prod/dev/venice (any subset) are graded together in one eval.
    """
    labels = [re.escape(s.label) for s in resolve(enabled_keys(config))]
    return f"^({'|'.join(labels)})$"


def report_provider_args(config: dict) -> list[str]:
    """compare_runs.py CLI args naming the baseline and other provider columns.

    Each arg is ``key=label`` so the report shows config keys (the column names)
    while still splitting the unified result file by promptfoo provider label.
    The baseline is emitted first (flagged); other providers follow in config order.
    """
    baseline = config["baseline-provider"]
    others = [k for k in enabled_keys(config) if k != baseline]
    args = ["--baseline-provider-col", f"{baseline}={REGISTRY[baseline].label}"]
    for key in others:
        args += ["--provider-col", f"{key}={REGISTRY[key].label}"]
    return args


# The published host ports for the prod/dev gateways live in the registry
# (ProviderSpec.gateway_port: 8082/8084, matching run_plaintext_gateway.sh). The
# gateway listens on 8080 inside the container.
GATEWAY_CONTAINER_PORT = 8080

# Where the gateway image expects its system prompt; mounting over this path
# overrides the baked-in prompt for the dev gateway.
CORE_SYSTEM_PROMPT_PATH = "/app/src/llm_gateway/prompts/core_system_prompt.md"

DEFAULT_GATEWAY_IMAGE = "secure-enclave-gateway-plaintext"

# Web-fetch sidecar: a single shared instance both gateways relay URL fetches
# through, mirroring the gateway-tdx + web-fetch pair in the secure-enclave
# docker-compose stack. Networking is simpler here than in compose (single
# user-defined bridge instead of the fetch-internal/fetch-egress split): a
# docker run container can only join one network at start, and the sidecar
# needs internet egress to actually fetch pages anyway. The sidecar's
# in-process SSRF validator is the real isolation boundary.
DEFAULT_WEB_FETCH_IMAGE = "secure-enclave-web-fetch:latest"
WEB_FETCH_CONTAINER_NAME = "fidaro-web-fetch"
# Network alias the gateways resolve via Docker's embedded DNS. Matches the
# compose service name `web-fetch` so HOST_WEB_FETCH_SIDECAR_URL is the same
# string production sees.
WEB_FETCH_NETWORK_ALIAS = "web-fetch"
WEB_FETCH_CONTAINER_PORT = 8000
WEB_FETCH_SIDECAR_URL = f"http://{WEB_FETCH_NETWORK_ALIAS}:{WEB_FETCH_CONTAINER_PORT}"
# User-defined bridge attaching the sidecar + both gateways so Docker's DNS
# resolves `web-fetch` between them (the default bridge has no DNS).
COMPARISON_NETWORK = "fidaro-comparison"

# filter-providers is owned by the orchestrator (it picks prod vs dev), so a
# value supplied in the config's promptfoo-filters is ignored.
RESERVED_FILTER_KEYS = {"filter-providers"}

# Suite config is always required; provider-specific keys (vLLM urls, api keys,
# redeploy target) are validated per the registry based on which providers are
# enabled — see validate_config.
REQUIRED_KEYS = ("suite-generation-config",)

# Being super cautious here. We really must be careful not to deploy to prod.
# TODO: Separate prod and dev instances so that phala CLI can't touch prod.
WHITELISTED_CVM_IDS = [
    "fidaro-vllm-002",
]


class ConfigError(Exception):
    """A comparison config is missing or internally inconsistent."""


def comparison_name(config_path) -> str:
    """The comparison name: the config filename without its extension."""
    return Path(config_path).stem


def comparison_dir(config_path) -> Path:
    """The directory holding a config's runs: a sibling named after its stem.

    e.g. ``comparisons/example.json`` -> ``comparisons/example/``. Results land
    beside the config that produced them, not in a fixed ``comparisons/`` dir.
    Resolved to an absolute path so output lands in the right place regardless
    of the process's working directory.
    """
    config_path = Path(config_path).resolve()
    return config_path.parent / config_path.stem


def enabled_keys(config: dict) -> list[str]:
    """Provider keys switched on in ``providers-under-test`` (config order)."""
    put = config.get("providers-under-test") or {}
    return [k for k, on in put.items() if on]


def validate_config(
    config: dict, repo_root: Path, env: Mapping[str, str] | None = None
) -> None:
    """Validate a comparison config, raising ConfigError on the first problem.

    Validation is registry-driven: which keys are required depends on which
    providers ``providers-under-test`` enables. `repo_root` is where `.env.phala`
    is expected; `env` supplies api keys / PHALA_DOCKER_COMPOSE_FILE (defaults to
    the process environment).
    """
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
        raise ConfigError(
            f"unknown provider(s) in providers-under-test: {sorted(unknown)}"
        )

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
    # Reject only keys the registry doesn't know (typo guard). Options for a
    # known-but-disabled provider are allowed, so a ready-to-go block can sit in
    # the config while its provider is toggled off in providers-under-test.
    unknown_opts = set(options) - all_keys()
    if unknown_opts:
        raise ConfigError(
            f"provider-options has unknown providers: {sorted(unknown_opts)}"
        )
    for key in enabled:
        if key not in options:
            raise ConfigError(f"provider-options is missing an entry for {key!r}")

    specs = resolve(enabled)
    # Each enabled provider validates per its kind: a gateway needs its vLLM url;
    # an api provider needs its credential env var present. Option fields
    # (model/temperature/max_tokens/…) are all optional — they vary per provider.
    for spec in specs:
        if spec.kind == "gateway":
            if not config.get(spec.vllm_url_key):
                raise ConfigError(
                    f"provider {spec.key!r} requires config key {spec.vllm_url_key!r}"
                )
        elif spec.kind == "api":
            if spec.api_key_env and not env.get(spec.api_key_env):
                raise ConfigError(
                    f"provider {spec.key!r} requires the {spec.api_key_env} env var"
                )

    # A redeploy (vllm-options) is the only thing that touches a CVM, and only the
    # dev gateway is ever redeployed. So the redeploy guards apply only when
    # fidaro-dev is enabled AND vllm-options is set — without both, the instance
    # id / whitelist / compose carry no constraints. When they DO apply: the dev
    # provider's model must match the served model, an instance id must be given
    # and whitelisted (a hard guard against redeploying a non-dev CVM), and the
    # compose template + .env.phala must exist.
    has_options = bool(config.get("vllm-options"))
    if has_options and "fidaro-dev" in enabled:
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


def vllm_options_changed(cache_path: Path, options: dict) -> bool:
    """True if `options` differ from the cached options (or no cache exists)."""
    cache_path = Path(cache_path)
    if not cache_path.is_file():
        return True
    cached = json.loads(cache_path.read_text(encoding="utf-8"))
    return cached != options


def write_options_cache(cache_path: Path, options: dict) -> None:
    """Write an identical copy of `options` to the cache file."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(options, indent=2), encoding="utf-8")


def provider_options_env(config: dict) -> dict[str, str]:
    """Map each enabled provider's options to its COMPARISON_<PREFIX>_* env vars.

    The dynamic provider YAMLs template their model (and, for the Fidaro gateways,
    temperature / max_tokens) from these vars; promptfoo renders {{ env.* }} at
    load time. Optional fields are set only when present, because a competitor API
    may send none of them. Venice additionally gets COMPARISON_VENICE_WEB_SEARCH
    (defaulting to "off"). Values are stringified (env vars are strings; the
    backend coerces numbers). Keep prefixes in sync with providers_registry / the
    provider YAMLs.
    """
    options = config.get("provider-options") or {}
    env: dict[str, str] = {}
    for spec in resolve(enabled_keys(config)):
        opts = options.get(spec.key) or {}
        if opts.get("model") is not None:
            env[f"{spec.env_prefix}_MODEL"] = str(opts["model"])
        if opts.get("temperature") is not None:
            env[f"{spec.env_prefix}_TEMPERATURE"] = str(opts["temperature"])
        if opts.get("max_tokens") is not None:
            env[f"{spec.env_prefix}_MAX_TOKENS"] = str(opts["max_tokens"])
        if spec.key == "venice":
            env["COMPARISON_VENICE_WEB_SEARCH"] = str(opts.get("web_search", "off"))
    return env


def build_filter_args(filters: dict | None) -> list[str]:
    """Turn a promptfoo-filters object into CLI flags.

    "filter-metadata": "suite=x"  -> ["--filter-metadata", "suite=x"]
    "filter-metadata": ["a", "b"] -> repeated flag per value
    filter-providers is ignored (the orchestrator sets it per side).
    """
    if not filters:
        return []
    args: list[str] = []
    for key, value in filters.items():
        if key in RESERVED_FILTER_KEYS:
            print(
                f"WARNING: ignoring reserved promptfoo filter {key!r} "
                "(the orchestrator controls providers)",
                flush=True,
            )
            continue
        flag = f"--{key}"
        values = value if isinstance(value, list) else [value]
        for v in values:
            args.extend([flag, str(v)])
    return args


def gateway_docker_args(
    name: str,
    port: int,
    vllm_url: str,
    brave_api_key: str,
    image: str,
    *,
    network: str,
    web_fetch_url: str,
    system_prompt_file: str | None = None,
    log_level: str = "debug",
    max_tool_calls: int = 10,
) -> list[str]:
    """Build the `docker run` argv for one plaintext gateway.

    Mirrors scripts_repo/run_plaintext_gateway.sh (which is left untouched).
    A dev system prompt, if given, is bind-mounted read-only over the image's
    core_system_prompt.md using an absolute host path.

    `network` joins the gateway to the shared user-defined bridge so it can
    resolve the web-fetch sidecar by DNS (the default bridge has no DNS).
    `host.docker.internal` keeps working: --add-host is a /etc/hosts entry,
    not network topology. `web_fetch_url` is plumbed in as
    HOST_WEB_FETCH_SIDECAR_URL so the gateway relays URL fetches through the
    sidecar rather than reaching out directly.
    """
    args = [
        "docker",
        "run",
        "--rm",
        "-d",
        "-p",
        f"127.0.0.1:{port}:{GATEWAY_CONTAINER_PORT}",
        "--name",
        name,
        "--network",
        network,
        "--add-host",
        "host.docker.internal:host-gateway",
        "-e",
        "HOST_LLM_PROVIDER=vllm",
        "-e",
        f"HOST_OPENAI_BASE_URL={vllm_url}",
        "-e",
        f"LOG_LEVEL={log_level}",
        "-e",
        f"BRAVE_API_KEY={brave_api_key}",
        "-e",
        f"HOST_MAX_TOOL_CALLS_PER_REQUEST={max_tool_calls}",
        "-e",
        f"HOST_WEB_FETCH_SIDECAR_URL={web_fetch_url}",
    ]
    if system_prompt_file:
        host_path = Path(system_prompt_file).resolve()
        args += ["-v", f"{host_path}:{CORE_SYSTEM_PROMPT_PATH}:ro"]
    args += [
        image,
        "uv",
        "run",
        "uvicorn",
        "llm_gateway.dev_plaintext_main:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(GATEWAY_CONTAINER_PORT),
    ]
    return args


def web_fetch_docker_args(
    name: str,
    image: str,
    *,
    network: str,
    network_alias: str = WEB_FETCH_NETWORK_ALIAS,
    fetch_timeout_seconds: int = 20,
) -> list[str]:
    """Build the `docker run` argv for the shared web-fetch sidecar.

    The sidecar is reachable only through the user-defined bridge (no host
    port published) and registers `network_alias` so the gateways resolve it
    at the same hostname compose uses (`web-fetch`). The baked-in healthcheck
    matches the compose definition so wait_for_container_healthy can poll
    Docker's health status instead of probing the unexposed port from the host.
    """
    return [
        "docker",
        "run",
        "--rm",
        "-d",
        "--name",
        name,
        "--network",
        network,
        "--network-alias",
        network_alias,
        "-e",
        f"WEBFETCH_FETCH_TIMEOUT_SECONDS={fetch_timeout_seconds}",
        "-e",
        "LOG_LEVEL=debug",
        # Mirror docker-compose.yml healthcheck so `docker inspect` health
        # status reflects sidecar readiness. The image already ships curl
        # (compose uses the same probe).
        "--health-cmd",
        f"curl -f http://localhost:{WEB_FETCH_CONTAINER_PORT}/health || exit 1",
        "--health-interval=5s",
        "--health-timeout=3s",
        "--health-retries=6",
        "--health-start-period=20s",
        image,
    ]


def eval_command(
    provider: str,
    output_path: str,
    filter_args: list[str],
    no_cache: bool,
    description: str,
    config_path: str = "promptfooconfig.yaml",
) -> list[str]:
    """Build the `pnpm exec promptfoo eval` argv.

    `provider` is passed verbatim to --filter-providers; it may be a single
    provider label or a regex selecting several (see both_providers_filter).
    """
    cmd = [
        "pnpm",
        "exec",
        "promptfoo",
        "eval",
        "--config",
        config_path,
        "--filter-providers",
        provider,
        "--output",
        output_path,
        "--description",
        description,
    ]
    cmd += filter_args
    if no_cache:
        cmd.append("--no-cache")
    return cmd


def parse_env_file(text: str) -> dict[str, str]:
    """Parse a simple KEY=VALUE .env file into a dict.

    Skips blank lines and comments, tolerates a leading `export`, and strips
    one layer of matching single or double quotes from values.
    """
    env: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        env[key] = value
    return env


def timestamp() -> str:
    """A filename-safe timestamp: YYYYMMDD-HHMMSS."""
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def run_dir_name(ts: str) -> str:
    """The per-run subdirectory name for a given timestamp."""
    return f"run_{ts}"


# --- side-effecting orchestration -----------------------------------------


def load_env_file(path: Path) -> None:
    """Merge a .env file into os.environ (without overriding existing vars)."""
    if not path.is_file():
        return
    for key, value in parse_env_file(path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)


def _run(cmd: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, streaming output. Optionally abort on non-zero exit."""
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(cwd))
    if check and result.returncode != 0:
        raise RuntimeError(f"command failed (exit {result.returncode}): {cmd[0]}")
    return result


def start_gateway(repo_root: Path, args: list[str], name: str) -> None:
    """Remove any existing container of the same name, then start the gateway."""
    subprocess.run(["docker", "rm", "-f", name], cwd=str(repo_root))
    _run(args, cwd=repo_root)


def gateway_health_url(port: int) -> str:
    """The OpenAI base URL of a locally published gateway."""
    return f"http://127.0.0.1:{port}/v1/health"


def ensure_docker_network(name: str) -> None:
    """Create the bridge network if it doesn't exist (idempotent).

    Uses `docker network inspect` as the existence check rather than catching
    `network create` errors, so a real failure (daemon down, no permission)
    still surfaces clearly instead of being misread as 'already exists'.
    """
    exists = subprocess.run(
        ["docker", "network", "inspect", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if exists.returncode == 0:
        return
    print(f"  $ docker network create {name}", flush=True)
    subprocess.run(
        ["docker", "network", "create", "--driver", "bridge", name],
        check=True,
    )


def wait_for_container_healthy(name: str, timeout_s: float) -> None:
    """Poll `docker inspect` until the container's health status is healthy.

    Uses Docker's own healthcheck (defined via --health-cmd at run time) rather
    than probing a port from the host: the sidecar deliberately publishes no
    host port, so this is the only way to wait on its readiness from outside.
    """
    import time

    deadline = time.monotonic() + timeout_s
    last_status = ""
    while time.monotonic() < deadline:
        result = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", name],
            capture_output=True,
            text=True,
        )
        status = result.stdout.strip() if result.returncode == 0 else "missing"
        if status == "healthy":
            return
        if status != last_status:
            print(f"  {name} health: {status}", flush=True)
            last_status = status
        time.sleep(1.0)
    raise RuntimeError(
        f"{name} did not become healthy within {timeout_s:.0f}s "
        f"(last status: {last_status or 'unknown'})"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("config", help="Path to the comparison config JSON.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before a Phala redeploy.",
    )
    parser.add_argument(
        "--docker-image",
        default=DEFAULT_GATEWAY_IMAGE,
        help=f"Gateway docker image (default: {DEFAULT_GATEWAY_IMAGE}).",
    )
    parser.add_argument(
        "--web-fetch-image",
        default=DEFAULT_WEB_FETCH_IMAGE,
        help=f"Web-fetch sidecar docker image (default: {DEFAULT_WEB_FETCH_IMAGE}).",
    )
    parser.add_argument(
        "--deploy-timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for the dev CVM's vLLM after a redeploy.",
    )
    parser.add_argument(
        "--gateway-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for each local gateway to come up.",
    )
    parser.add_argument(
        "--sidecar-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for the web-fetch sidecar to become healthy.",
    )
    parser.add_argument(
        "--skip-phala-deploy",
        action="store_true",
        default=False,
        help="Skip the Phala redeploy step. Useful for debugging.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name = comparison_name(config_path)
    # Output lands beside the config that produced it (comparisons/example.json
    # -> comparisons/example/), not in a fixed comparisons/ dir. run_dir isolates
    # everything produced by this single run so runs never overwrite each other.
    ts = timestamp()
    run_dir = comparison_dir(config_path) / run_dir_name(ts)
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run outputs: {run_dir}")

    validate_config(config, repo_root=repo_root)

    enabled = enabled_keys(config)
    specs = resolve(enabled)
    gateway_specs = [s for s in specs if s.kind == "gateway"]

    # Suite-generation config: write it out and point promptfoo at the file.
    # These are set on os.environ (not a throwaway copy) so the promptfoo eval
    # subprocesses, which inherit os.environ, actually see them.
    suite_cfg_path = run_dir / "suite_generation_config.json"
    suite_cfg_path.write_text(
        json.dumps(config["suite-generation-config"], indent=2), encoding="utf-8"
    )
    os.environ["SUITE_GENERATION_CONFIG_FILE"] = str(suite_cfg_path)
    # The dynamic providers template their model/temperature/max_tokens (and
    # venice's web-search flag) from these COMPARISON_<PREFIX>_* env vars (see
    # providers/*_dynamic_*.yaml and providers/venice_dynamic.yaml).
    os.environ.update(provider_options_env(config))
    # Grade providers head-to-head: augment() (tests/classification.py) reads this
    # and appends a select-best assertion to every test. Only meaningful with two
    # or more providers in the one eval.
    if len(enabled) >= 2:
        os.environ[SELECT_BEST_ENV_VAR] = "1"

    # A Brave key is only needed by the plaintext gateways; an api-only run (every
    # enabled provider is a direct API, e.g. venice) does not use it.
    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if gateway_specs and not brave_api_key:
        print("ERROR: BRAVE_API_KEY must be set in the environment", file=sys.stderr)
        return 1

    # Redeploy the dev CVM only when fidaro-dev is enabled and its vLLM options
    # changed.
    options = config.get("vllm-options")
    if options and "fidaro-dev" in enabled:
        # The cache is global (repo root) because there is only one dev Phala
        # instance: any run's vLLM options describe the same shared CVM, so the
        # redeploy decision must persist across comparisons, not per comparison.
        # The rendered compose remains a per-run artifact.
        cache = repo_root / ".vllm_options_cache.json"
        if vllm_options_changed(cache, options):
            if args.skip_phala_deploy:
                print("Force skipping Phala redeploy.")
            else:
                if not args.yes and not _confirm_redeploy(name, options):
                    print("Aborted before Phala redeploy.", file=sys.stderr)
                    return 1
                print(f"Redeploying dev CVM {config['phala-dev-instance-id']} ...")
                phala_deploy_and_wait(
                    cvm_id=config["phala-dev-instance-id"],
                    template_path=Path(os.environ["PHALA_DOCKER_COMPOSE_FILE"]),
                    out_path=run_dir / "deployed_compose.yaml",
                    options=options,
                    vllm_url=config["vllm-dev-url"],
                    env_file=repo_root / ".env.phala",
                    timeout_s=args.deploy_timeout,
                )
                write_options_cache(cache, options)
        else:
            print("vLLM options unchanged since last run; skipping Phala redeploy.")

    # Start the shared web-fetch sidecar + the enabled gateways. A run whose
    # enabled providers are all direct APIs (no gateway) skips all of this — there
    # is nothing local to stand up. The sidecar must be healthy before the gateways
    # accept fetch-tool traffic; healthy-before-gateway also turns "sidecar image
    # missing" into an early, obvious failure.
    if gateway_specs:
        print(f"Ensuring docker network {COMPARISON_NETWORK} exists ...")
        ensure_docker_network(COMPARISON_NETWORK)

        print("Starting web-fetch sidecar ...")
        start_gateway(  # reused: same "rm -f then docker run" lifecycle
            repo_root,
            web_fetch_docker_args(
                name=WEB_FETCH_CONTAINER_NAME,
                image=args.web_fetch_image,
                network=COMPARISON_NETWORK,
            ),
            name=WEB_FETCH_CONTAINER_NAME,
        )
        print("Waiting for web-fetch sidecar to be healthy ...")
        wait_for_container_healthy(
            WEB_FETCH_CONTAINER_NAME, timeout_s=args.sidecar_timeout
        )

        print("Starting plaintext gateways ...")
        for spec in gateway_specs:
            container = f"fidaro-gateway-{spec.key.split('-')[-1]}"
            start_gateway(
                repo_root,
                gateway_docker_args(
                    name=container,
                    port=spec.gateway_port,
                    vllm_url=config[spec.vllm_url_key],
                    brave_api_key=brave_api_key,
                    image=args.docker_image,
                    network=COMPARISON_NETWORK,
                    web_fetch_url=WEB_FETCH_SIDECAR_URL,
                    system_prompt_file=(
                        config.get("system-prompt-file")
                        if spec.supports_system_prompt
                        else None
                    ),
                ),
                name=container,
            )

        # Wait for the vLLM endpoints, then the local gateways, before the eval.
        # Catches problems early.
        print("Waiting for vLLM phala endpoints to be ready ...")
        for spec in gateway_specs:
            wait_for_url(
                models_url(config[spec.vllm_url_key]), timeout_s=args.gateway_timeout
            )
        print("Waiting for local plaintext gateways to be ready ...")
        for spec in gateway_specs:
            wait_for_url(
                gateway_health_url(spec.gateway_port), timeout_s=args.gateway_timeout
            )

    # One eval graded against ALL enabled providers (no baseline freeze). Eval
    # exits non-zero on test failures, which is expected here, so we do not abort
    # on it. Running every provider in a single pass lets a head-to-head assertion
    # (e.g. select-best) see all responses together.
    #
    # Always --no-cache: a single eval has one global cache setting, and dev may
    # have just been redeployed with a server-side change (system prompt / vLLM
    # options) that promptfoo's request cache key cannot see, so a cached hit could
    # be stale. Correctness over speed.
    filter_args = build_filter_args(config.get("promptfoo-filters"))
    results_out = run_dir / f"results_{ts}.json"

    print("Running unified eval over enabled providers ...")
    _run(
        eval_command(
            providers_filter(config),
            str(results_out),
            filter_args,
            True,  # no_cache
            f"{name} comparison",
        ),
        cwd=repo_root,
        check=False,
    )

    # Build the report from the one file, splitting it by provider label. The
    # baseline + other provider columns (config keys -> labels) are passed via
    # report_provider_args. Plumb the config path (and the dev system prompt, when
    # set) through so the header can show which inputs produced this run.
    report = run_dir / f"report__{ts}.html"
    # Raw-response CSV emitted alongside the HTML on every run, for eyeballing
    # the models' answers (no scores) side by side.
    responses_csv = run_dir / f"responses__{ts}.csv"
    report_cmd = [
        sys.executable,
        str(repo_root / "scripts_repo" / "compare_runs.py"),
        str(results_out),
        str(results_out),
        *report_provider_args(config),
        "--out",
        str(report),
        "--csv",
        str(responses_csv),
        "--config-path",
        str(config_path.resolve()),
    ]
    system_prompt_file = config.get("system-prompt-file")
    if system_prompt_file:
        report_cmd += ["--system-prompt-path", str(Path(system_prompt_file).resolve())]
    _run(report_cmd, cwd=repo_root)

    # Bring the promptfoo viewer container up (it does not hot-reload its DB).
    _run(
        [str(repo_root / "scripts_repo" / "run_promptfoo_docker.sh")],
        cwd=repo_root,
        check=False,
    )
    _run(["open", str(report)], cwd=repo_root, check=False)

    print(f"\nComparison complete. Report: {report}")
    return 0


def _confirm_redeploy(name: str, options: dict) -> bool:
    """Warn about a Phala redeploy and ask the operator to confirm."""

    print(
        f"\nWARNING: comparison {name!r} requires redeploying the Phala dev CVM "
        "with new vLLM options:",
        file=sys.stderr,
    )
    print(json.dumps(options, indent=2), file=sys.stderr)
    answer = input("Proceed with the redeploy? [y/N] ").strip().lower()
    return answer in ("y", "yes")


if __name__ == "__main__":
    sys.exit(main())

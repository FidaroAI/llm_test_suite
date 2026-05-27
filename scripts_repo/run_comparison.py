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

# promptfoo provider labels the orchestrator drives for each side of the run.
# The dynamic providers template their model / temperature / max_tokens from the
# COMPARISON_{PROD,DEV}_* env vars set below (see provider_options_env), so they
# reflect a reconfigured gateway (the static YAML providers can't).
PROD_PROVIDER = "fidaro_plaintext_gateway_phala_dynamic_prod"
DEV_PROVIDER = "fidaro_plaintext_gateway_phala_dynamic_dev"

# Set in the eval's environment so tests/classification.py:augment appends a
# head-to-head `select-best` assertion. Only meaningful here, where prod and dev
# are graded in one eval; single-provider runs leave it unset. Keep in sync with
# tests/classification.py (SELECT_BEST_ENV_VAR).
SELECT_BEST_ENV_VAR = "COMPARISON_SELECT_BEST"


def both_providers_filter() -> str:
    """A --filter-providers regex selecting BOTH dynamic providers in one pass.

    promptfoo's --filter-providers is a regex matched against each provider's
    id/label. Anchoring on the exact dynamic labels keeps the static
    (non-dynamic) `..._prod` / `..._dev` providers and the vLLM provider out of
    the run. Used by the single unified eval (prod and dev graded together) that
    replaced the old two separate single-provider passes.
    """
    return f"^({re.escape(PROD_PROVIDER)}|{re.escape(DEV_PROVIDER)})$"

# Host ports the prod/dev plaintext gateways are published on (matches
# run_plaintext_gateway.sh). The gateway listens on 8080 inside the container.
PROD_GATEWAY_PORT = 8082
DEV_GATEWAY_PORT = 8084
GATEWAY_CONTAINER_PORT = 8080

# Where the gateway image expects its system prompt; mounting over this path
# overrides the baked-in prompt for the dev gateway.
CORE_SYSTEM_PROMPT_PATH = "/app/src/llm_gateway/prompts/core_system_prompt.md"

DEFAULT_GATEWAY_IMAGE = "secure-enclave-gateway-plaintext"

# filter-providers is owned by the orchestrator (it picks prod vs dev), so a
# value supplied in the config's promptfoo-filters is ignored.
RESERVED_FILTER_KEYS = {"filter-providers"}

REQUIRED_KEYS = ("vllm-prod-url", "vllm-dev-url", "suite-generation-config")

# Per-side provider config the dynamic providers read (via provider_options_env).
# Each must carry these keys; see providers/*_dynamic_*.yaml.
PROVIDER_OPTIONS_KEYS = ("prod-provider-options", "dev-provider-options")
REQUIRED_PROVIDER_OPTION_FIELDS = ("model", "temperature", "max_tokens")

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


def validate_config(
    config: dict, repo_root: Path, env: Mapping[str, str] | None = None
) -> None:
    """Validate a comparison config, raising ConfigError on the first problem.

    `repo_root` is where `.env.phala` is expected; `env` supplies
    PHALA_DOCKER_COMPOSE_FILE (defaults to the process environment).
    """
    if env is None:
        env = os.environ

    for key in REQUIRED_KEYS:
        if key not in config or config[key] in (None, ""):
            raise ConfigError(f"config is missing required key {key!r}")

    # The dynamic providers need a {model, temperature, max_tokens} block per side.
    for key in PROVIDER_OPTIONS_KEYS:
        opts = config.get(key)
        if not opts:
            raise ConfigError(f"config is missing required key {key!r}")
        missing = [f for f in REQUIRED_PROVIDER_OPTION_FIELDS if f not in opts]
        if missing:
            raise ConfigError(f"{key!r} is missing fields: {missing}")

    has_options = bool(config.get("vllm-options"))

    # A redeploy (vllm-options) is the only thing that touches a CVM, so the dev
    # target is validated only when one is configured. Without a redeploy the
    # instance id is never used, so it carries no constraints. When a redeploy IS
    # configured: the dev provider's model must match the model it serves, an
    # instance id must be given, and that id must be whitelisted — a hard guard
    # against ever redeploying a non-dev (e.g. prod) CVM.
    if has_options:
        vllm_model = config["vllm-options"].get("model")
        dev_model = config["dev-provider-options"]["model"]
        if vllm_model and dev_model != vllm_model:
            raise ConfigError(
                f"dev-provider-options.model ({dev_model!r}) must match "
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

    prompt_file = config.get("system-prompt-file")
    if prompt_file and not Path(prompt_file).is_file():
        raise ConfigError(f"system-prompt-file does not exist: {prompt_file}")

    if has_options:
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
                f"vllm-options requires an env variable file at {env_phala}. Prefer to use 1Password environments for this."
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
    """Map the per-side provider options to the env vars the dynamic providers read.

    The dynamic YAML providers template their model/temperature/max_tokens from
    COMPARISON_{PROD,DEV}_{MODEL,TEMPERATURE,MAX_TOKENS} (see
    providers/fidaro_plaintext_gateway_phala_dynamic_{prod,dev}.yaml). Keep these
    names in sync with those files. Values are stringified because env vars (and
    promptfoo's {{ env.* }} rendering) are strings; the backend coerces the numbers.
    """
    env: dict[str, str] = {}
    for side, prefix in (("prod", "COMPARISON_PROD"), ("dev", "COMPARISON_DEV")):
        opts = config[f"{side}-provider-options"]
        env[f"{prefix}_MODEL"] = str(opts["model"])
        env[f"{prefix}_TEMPERATURE"] = str(opts["temperature"])
        env[f"{prefix}_MAX_TOKENS"] = str(opts["max_tokens"])
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
    system_prompt_file: str | None = None,
    log_level: str = "info",
    max_tool_calls: int = 10,
) -> list[str]:
    """Build the `docker run` argv for one plaintext gateway.

    Mirrors scripts_repo/run_plaintext_gateway.sh (which is left untouched).
    A dev system prompt, if given, is bind-mounted read-only over the image's
    core_system_prompt.md using an absolute host path.
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

    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY must be set in the environment", file=sys.stderr)
        return 1

    # Suite-generation config: write it out and point promptfoo at the file.
    # These are set on os.environ (not a throwaway copy) so the promptfoo eval
    # subprocesses, which inherit os.environ, actually see them.
    suite_cfg_path = run_dir / "suite_generation_config.json"
    suite_cfg_path.write_text(
        json.dumps(config["suite-generation-config"], indent=2), encoding="utf-8"
    )
    os.environ["SUITE_GENERATION_CONFIG_FILE"] = str(suite_cfg_path)
    # The dynamic providers template their model/temperature/max_tokens from these
    # COMPARISON_{PROD,DEV}_* env vars (see providers/*_dynamic_*.yaml).
    os.environ.update(provider_options_env(config))
    # Grade prod vs dev head-to-head: augment() (tests/classification.py) reads
    # this and appends a select-best assertion to every test. Only valid because
    # both providers run in this one eval.
    os.environ[SELECT_BEST_ENV_VAR] = "1"

    # Redeploy the dev CVM only when vLLM options changed.
    options = config.get("vllm-options")
    if options:
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

    # Start both gateways and wait for them to answer.
    print("Starting plaintext gateways ...")
    start_gateway(
        repo_root,
        gateway_docker_args(
            name="fidaro-gateway-prod",
            port=PROD_GATEWAY_PORT,
            vllm_url=config["vllm-prod-url"],
            brave_api_key=brave_api_key,
            image=args.docker_image,
        ),
        name="fidaro-gateway-prod",
    )
    start_gateway(
        repo_root,
        gateway_docker_args(
            name="fidaro-gateway-dev",
            port=DEV_GATEWAY_PORT,
            vllm_url=config["vllm-dev-url"],
            brave_api_key=brave_api_key,
            image=args.docker_image,
            system_prompt_file=config.get("system-prompt-file"),
        ),
        name="fidaro-gateway-dev",
    )
    # Wait for vLLM in the server to be available. This will help spot problems before we start the eval runs
    print("Waiting for vLLM phala endpoints to be ready ...")
    wait_for_url(models_url(config["vllm-prod-url"]), timeout_s=args.gateway_timeout)
    wait_for_url(models_url(config["vllm-dev-url"]), timeout_s=args.gateway_timeout)

    # Also check for the gateways to be running. Defence in depth for bugs
    print("Waiting for local plaintext gateways to be ready ...")
    wait_for_url(gateway_health_url(PROD_GATEWAY_PORT), timeout_s=args.gateway_timeout)
    wait_for_url(gateway_health_url(DEV_GATEWAY_PORT), timeout_s=args.gateway_timeout)

    # One eval graded against BOTH providers (no baseline freeze). Eval exits
    # non-zero on test failures, which is expected here, so we do not abort on
    # it. Running prod and dev in a single pass (rather than two) lets a
    # head-to-head assertion (e.g. select-best) see both responses together, and
    # halves the orchestration.
    #
    # Always --no-cache: a single eval has one global cache setting, and dev may
    # have just been redeployed with a server-side change (system prompt / vLLM
    # options) that promptfoo's request cache key cannot see, so a cached dev hit
    # could be stale. Correctness over speed — prod loses its former cache reuse.
    filter_args = build_filter_args(config.get("promptfoo-filters"))
    results_out = run_dir / f"results_{ts}.json"

    print("Running unified prod+dev eval ...")
    _run(
        eval_command(
            both_providers_filter(),
            str(results_out),
            filter_args,
            True,  # no_cache
            f"{name} prod+dev",
        ),
        cwd=repo_root,
        check=False,
    )

    # Build the report from the one file, splitting it by provider label:
    # prod is the baseline side, dev the candidate.
    report = run_dir / f"report__{ts}.html"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts_repo" / "compare_runs.py"),
            str(results_out),
            str(results_out),
            "--baseline-provider",
            PROD_PROVIDER,
            "--candidate-provider",
            DEV_PROVIDER,
            "--out",
            str(report),
        ],
        cwd=repo_root,
    )

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

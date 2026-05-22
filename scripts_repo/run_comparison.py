#!/usr/bin/env python3
"""Run one prod-vs-dev Fidaro comparison from a single config file.

Given a config JSON (see docs/superpowers/specs/
2026-05-22-comparison-orchestrator-design.md), this script:

  1. Creates an isolated output directory comparisons/<name>/ (name = config
     filename stem).
  2. Validates the config.
  3. Redeploys the Phala dev CVM with new vLLM options *only* when they changed
     (decided via a cache file), after an explicit confirmation.
  4. Starts the prod and dev plaintext gateways in Docker (mounting a dev
     system prompt if given).
  5. Runs the promptfoo suite against prod and against dev, into timestamped
     result files under comparisons/<name>/.
  6. Builds the comparison report and opens it, then ensures the promptfoo
     viewer container is up.

It does NOT freeze a baseline.

Usage:
    run_comparison.py comparisons/prod_vs_dev_gemma.json [--yes]

BRAVE_API_KEY must be set in the environment (it is never read from config).
"""

from __future__ import annotations

import argparse
import json
import os
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
from scripts_repo.deploy_phala import wait_for_vllm

# promptfoo provider labels the orchestrator drives for each side of the run.
PROD_PROVIDER = "fidaro_plaintext_gateway_phala_prod"
DEV_PROVIDER = "fidaro_plaintext_gateway_phala_dev"

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


class ConfigError(Exception):
    """A comparison config is missing or internally inconsistent."""


def comparison_name(config_path) -> str:
    """The comparison name: the config filename without its extension."""
    return Path(config_path).stem


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

    has_options = bool(config.get("vllm-options"))
    if has_options and not config.get("phala-dev-instance-id"):
        raise ConfigError(
            "vllm-options requires phala-dev-instance-id (a redeploy target)"
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
            raise ConfigError(
                f"PHALA_DOCKER_COMPOSE_FILE does not exist: {compose}"
            )
        env_phala = repo_root / ".env.phala"
        if not env_phala.is_file():
            raise ConfigError(
                f"vllm-options requires {env_phala} (provided by the operator)"
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
    """Build the `pnpm exec promptfoo eval` argv for one side of the run."""
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
    return f"http://127.0.0.1:{port}/v1"


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
        "--cache",
        action="store_true",
        help="Allow promptfoo result caching (default: --no-cache for fresh runs).",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    load_env_file(repo_root / ".env")

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name = comparison_name(config_path)
    out_dir = repo_root / "comparisons" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    validate_config(config, repo_root=repo_root)

    brave_api_key = os.environ.get("BRAVE_API_KEY")
    if not brave_api_key:
        print("ERROR: BRAVE_API_KEY must be set in the environment", file=sys.stderr)
        return 1

    # Suite-generation config: write it out and point promptfoo at the file.
    suite_cfg_path = out_dir / "suite_generation_config.json"
    suite_cfg_path.write_text(
        json.dumps(config["suite-generation-config"], indent=2), encoding="utf-8"
    )
    run_env = dict(os.environ)
    run_env["SUITE_GENERATION_CONFIG_FILE"] = str(suite_cfg_path)

    # Redeploy the dev CVM only when vLLM options changed.
    options = config.get("vllm-options")
    if options:
        cache = out_dir / "vllm_options_cache.json"
        if vllm_options_changed(cache, options):
            if not args.yes and not _confirm_redeploy(name, options):
                print("Aborted before Phala redeploy.", file=sys.stderr)
                return 1
            print(f"Redeploying dev CVM {config['phala-dev-instance-id']} ...")
            phala_deploy_and_wait(
                cvm_id=config["phala-dev-instance-id"],
                template_path=Path(os.environ["PHALA_DOCKER_COMPOSE_FILE"]),
                out_path=out_dir / "deployed_compose.yaml",
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
    wait_for_vllm(gateway_health_url(PROD_GATEWAY_PORT), timeout_s=args.gateway_timeout)
    wait_for_vllm(gateway_health_url(DEV_GATEWAY_PORT), timeout_s=args.gateway_timeout)

    # Run both passes (no baseline freeze). Eval exits non-zero on test
    # failures, which is expected here, so we do not abort on it.
    ts = timestamp()
    filter_args = build_filter_args(config.get("promptfoo-filters"))
    no_cache = not args.cache
    prod_out = out_dir / f"prod_results_{ts}.json"
    dev_out = out_dir / f"dev_results_{ts}.json"

    print("Running prod pass ...")
    _run(
        eval_command(PROD_PROVIDER, str(prod_out), filter_args, no_cache, f"{name} prod"),
        cwd=repo_root,
        check=False,
    )
    print("Running dev pass ...")
    _run(
        eval_command(DEV_PROVIDER, str(dev_out), filter_args, no_cache, f"{name} dev"),
        cwd=repo_root,
        check=False,
    )

    # Compare and open the report (prod is the baseline side, dev the candidate).
    report = out_dir / f"report__{ts}.html"
    _run(
        [
            sys.executable,
            str(repo_root / "scripts_repo" / "compare_runs.py"),
            str(prod_out),
            str(dev_out),
            "--out",
            str(report),
        ],
        cwd=repo_root,
    )

    # Bring the promptfoo viewer container up (it does not hot-reload its DB).
    _run([str(repo_root / "scripts_repo" / "run_promptfoo_docker.sh")], cwd=repo_root, check=False)
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

#!/usr/bin/env python3
"""Deploy the Fidaro dev CVM to Phala with a given set of vLLM options.

Takes a compose template (the committed Phala manifest), rewrites the `vllm`
service's `command` from a vLLM options object, runs `phala deploy`, and waits
for the dev vLLM endpoint to come up.

Standalone usage:
    deploy_phala.py --cvm-id <id> --compose-in <template.yaml> \\
        --compose-out <deployed.yaml> --options-file <opts.json> \\
        --vllm-url <https://...:8000/v1> --env-file .env.phala

`run_comparison.py` imports the helpers directly rather than shelling out.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import requests
from ruamel.yaml import YAML

# vLLM always binds these inside the CVM regardless of the supplied options;
# the published 8000 port is what Phala's gateway wraps with TLS.
FIXED_VLLM_ARGS = ["--host", "0.0.0.0", "--port", "8000"]


def _yaml() -> YAML:
    """A round-trip YAML handler that preserves comments and key order."""
    yaml = YAML()
    yaml.preserve_quotes = True
    return yaml


def load_compose(stream):
    """Load a compose document from a path or file-like object."""
    return _yaml().load(stream)


def dump_compose(compose, stream) -> None:
    """Dump a compose document to a path or file-like object."""
    _yaml().dump(compose, stream)


def vllm_options_to_command(options: dict) -> list[str]:
    """Turn a vLLM options object into a vLLM `command` argument list.

    `--host 0.0.0.0 --port 8000` are always prepended. Then, in insertion
    order:

      * "key": <str|number>  -> ["--key", "<value>"]
      * "key": true          -> ["--key"]          (bare flag)
      * "key": false         -> omitted
    """
    cmd = list(FIXED_VLLM_ARGS)
    for key, value in options.items():
        flag = f"--{key}"
        if value is True:
            cmd.append(flag)
        elif value is False:
            continue
        else:
            cmd.extend([flag, str(value)])
    return cmd


def apply_vllm_options(compose, options: dict):
    """Rewrite the `vllm` service's `command` in-place from `options`.

    Raises KeyError if the compose document has no `vllm` service.
    """
    services = compose["services"]
    if "vllm" not in services:
        raise KeyError("compose file has no 'vllm' service")
    services["vllm"]["command"] = vllm_options_to_command(options)
    return compose


def render_compose(template_path: Path, out_path: Path, options: dict) -> Path:
    """Copy `template_path` to `out_path` with the vLLM command rewritten.

    The output is a deploy artifact: an exact, comment-preserving copy of the
    template with only the `vllm` service's command changed.
    """
    compose = load_compose(template_path)
    apply_vllm_options(compose, options)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dump_compose(compose, out_path)
    return out_path


def models_url(base_url: str) -> str:
    """The vLLM ``/models`` endpoint for an OpenAI-style base URL.

    Tolerates a trailing slash: ``.../v1`` and ``.../v1/`` both yield
    ``.../v1/models``. Used as a readiness probe (a 200 means vLLM is serving).
    """
    return base_url.rstrip("/") + "/models"


def wait_for_url(
    url: str,
    timeout_s: float,
    interval_s: float = 10.0,
    get_fn=requests.get,
    sleep_fn=time.sleep,
    now_fn=time.monotonic,
) -> None:
    """Poll `url` until it returns HTTP 200.

    Blocks up to `timeout_s` seconds (the CVM can take minutes to come up).
    Raises TimeoutError if the endpoint never returns 200 in time. Network
    errors are treated like a not-ready response and retried.
    """
    deadline = now_fn() + timeout_s
    last = "no response yet"
    while True:
        try:
            resp = get_fn(url, timeout=10)
            if resp.status_code == 200:
                return
            last = f"HTTP {resp.status_code}"
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = repr(exc)
        if now_fn() >= deadline:
            raise TimeoutError(f"{url} not ready after {timeout_s:.0f}s (last: {last})")
        print(f"  waiting for {url} ... ({last})", flush=True)
        sleep_fn(interval_s)


def phala_deploy(cvm_id: str, compose_path: Path, env_file: Path) -> None:
    """Run `phala deploy` for the given CVM, compose file, and env file.

    Raises RuntimeError on a non-zero exit so callers can abort the run.
    """
    cmd = [
        "phala",
        "deploy",
        "--cvm-id",
        cvm_id,
        "--compose",
        str(compose_path),
        "-e",
        str(env_file),
    ]
    print(f"  $ {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"phala deploy failed (exit {result.returncode})")


def deploy(
    cvm_id: str,
    template_path: Path,
    out_path: Path,
    options: dict,
    vllm_url: str,
    env_file: Path,
    timeout_s: float = 1800.0,
) -> Path:
    """Render the compose, deploy to Phala, and wait for vLLM to be ready.

    Returns the path of the rendered (deployed) compose file.
    """
    rendered = render_compose(template_path, out_path, options)
    phala_deploy(cvm_id, rendered, env_file)
    print(f"  deploy submitted; waiting for {vllm_url} to come up ...", flush=True)
    wait_for_url(vllm_url, timeout_s=timeout_s)
    return rendered


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cvm-id", required=True, help="Phala dev CVM id.")
    parser.add_argument(
        "--compose-in",
        required=True,
        type=Path,
        help="Compose template to render from.",
    )
    parser.add_argument(
        "--compose-out",
        required=True,
        type=Path,
        help="Where to write the rendered (deployed) compose file.",
    )
    parser.add_argument(
        "--options-file",
        required=True,
        type=Path,
        help="JSON file of vLLM options to inject into the vllm command.",
    )
    parser.add_argument(
        "--vllm-url",
        required=True,
        help="Dev vLLM base URL (…/v1) to poll for readiness after deploy.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env.phala"),
        help="Env file passed to `phala deploy -e` (default: .env.phala).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800.0,
        help="Seconds to wait for vLLM readiness (default: 1800).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    options = json.loads(args.options_file.read_text(encoding="utf-8"))
    deploy(
        cvm_id=args.cvm_id,
        template_path=args.compose_in,
        out_path=args.compose_out,
        options=options,
        vllm_url=args.vllm_url,
        env_file=args.env_file,
        timeout_s=args.timeout,
    )
    print(f"dev CVM {args.cvm_id} deployed and vLLM is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

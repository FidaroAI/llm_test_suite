#!/usr/bin/env python3
"""Generate and run many prod-vs-dev comparisons, one per system prompt.

Given a directory of system prompts and one *template* comparison config, this
script generates a comparison config per prompt (the template with its
``system-prompt-file`` overwritten) and, unless ``--generate-only`` is given,
runs each through ``scripts_repo/run_comparison.py``.

Each generated config is named after its prompt: ``test_prompt_1.md`` produces
``test_prompt_1.json``. ``run_comparison.py`` derives a comparison's name (and
thus its ``comparisons/<name>/`` result directory) from the config filename
stem, so each prompt gets an isolated result directory for free.

Usage:
    batch_comparison.py \\
        --system-prompts-directory system_prompts/candidates \\
        --template-config comparisons/example.json \\
        [--generate-only] [--output-directory DIR] \\
        [-- <args forwarded to run_comparison.py, e.g. --yes>]

Unrecognised arguments are forwarded verbatim to every run_comparison.py call.

Note: ``--output-directory`` controls only where the generated *config* files
are written. Per-run *results* still land in ``comparisons/<stem>/`` because
``run_comparison.py`` hardcodes that location (and is intentionally left
untouched).
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def discover_prompts(directory: Path) -> list[Path]:
    """Return the ``*.md`` files directly in ``directory``, sorted by name.

    Does not recurse into subdirectories.
    """
    directory = Path(directory)
    return sorted(p for p in directory.glob("*.md") if p.is_file())


def config_filename(prompt_path: Path) -> str:
    """The generated config filename for a prompt: its stem plus ``.json``."""
    return f"{Path(prompt_path).stem}.json"


def generate_config(template: dict, prompt_path: Path) -> dict:
    """Return a copy of ``template`` with ``system-prompt-file`` pointed at the prompt.

    The prompt path is stored as an absolute path so ``run_comparison.py``'s
    ``system-prompt-file`` existence check holds regardless of its CWD. The
    input ``template`` is not mutated.
    """
    config = copy.deepcopy(template)
    config["system-prompt-file"] = str(Path(prompt_path).resolve())
    return config


def write_generated_configs(
    prompts: list[Path], template: dict, output_dir: Path
) -> list[Path]:
    """Write one generated config per prompt into ``output_dir``; return the paths.

    Creates ``output_dir`` if it does not exist. Each config is pretty-printed
    with 2-space indentation to match the hand-written configs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for prompt in prompts:
        config = generate_config(template, prompt)
        out_path = output_dir / config_filename(prompt)
        out_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        written.append(out_path)
    return written


def run_comparison_command(
    config_path: Path, forwarded: list[str], repo_root: Path
) -> list[str]:
    """Build the argv that runs run_comparison.py for one config.

    ``forwarded`` is appended verbatim, so flags like ``--yes`` reach the
    orchestrator unchanged.
    """
    return [
        sys.executable,
        str(Path(repo_root) / "scripts_repo" / "run_comparison.py"),
        str(config_path),
        *forwarded,
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--system-prompts-directory",
        required=True,
        type=Path,
        help="Directory of system prompts; every *.md file (no recursion) is one prompt.",
    )
    parser.add_argument(
        "--template-config",
        required=True,
        type=Path,
        help="Comparison config to use as the template (e.g. comparisons/example.json).",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Generate the configs and stop; do not run them.",
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=None,
        help="Where generated configs are written (default: --system-prompts-directory).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(argv)
    repo_root = Path(__file__).resolve().parents[1]

    prompts_dir = args.system_prompts_directory
    if not prompts_dir.is_dir():
        print(
            f"ERROR: system-prompts-directory does not exist: {prompts_dir}",
            file=sys.stderr,
        )
        return 1

    if not args.template_config.is_file():
        print(
            f"ERROR: template-config does not exist: {args.template_config}",
            file=sys.stderr,
        )
        return 1

    prompts = discover_prompts(prompts_dir)
    if not prompts:
        print(f"ERROR: no *.md system prompts found in {prompts_dir}", file=sys.stderr)
        return 1

    template = json.loads(args.template_config.read_text(encoding="utf-8"))
    output_dir = args.output_directory or prompts_dir

    written = write_generated_configs(prompts, template, output_dir)
    print(f"Generated {len(written)} config(s) in {output_dir}:")
    for path in written:
        print(f"  {path}")

    if args.generate_only:
        return 0

    failures: list[Path] = []
    for config_path in written:
        print(f"\n=== Running comparison: {config_path.name} ===", flush=True)
        cmd = run_comparison_command(config_path, forwarded, repo_root)
        print(f"  $ {' '.join(cmd)}", flush=True)
        result = subprocess.run(cmd, cwd=str(repo_root))
        if result.returncode != 0:
            print(
                f"WARNING: comparison {config_path.name} failed "
                f"(exit {result.returncode}); continuing.",
                file=sys.stderr,
                flush=True,
            )
            failures.append(config_path)

    succeeded = len(written) - len(failures)
    print(
        f"\nBatch complete. Generated {len(written)}, "
        f"succeeded {succeeded}, failed {len(failures)}."
    )
    if failures:
        for path in failures:
            print(f"  failed: {path.name}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

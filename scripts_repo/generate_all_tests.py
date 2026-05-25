#!/usr/bin/env python3
"""Run every suite generator and emit all the resulting tests to stdout.

Discovers each ``tests/<name>_gen.py`` (the same convention promptfoo uses),
calls its no-arg ``generate_tests()``, and prints the resulting test cases. This
is a convenient way to preview exactly what a given generation config would
produce, without spinning up a promptfoo run.

Generation is driven by the suite-generation config file. Generators read it
lazily (via the ``SUITE_GENERATION_CONFIG_FILE`` env var, see
``tests/suite_config.py``), so we just point that env var at the supplied
config before invoking each generator.

The config argument accepts **either** a bare suite-generation config (keyed by
suite name, like ``tests/suite_generation_config.json``) **or** a full comparison
config (like ``comparisons/example.json``), in which case its nested
``suite-generation-config`` block is used. Passing the comparison config means
this preview shows exactly what that comparison run will generate.

Output modes:

* **terse** (default) — four labelled lines per test (suite, request_type,
  domain, user prompt), for a quick human scan.
* **verbose** (``--verbose``) — the full JSON array of promptfoo test cases.

Usage::

    python scripts_repo/generate_all_tests.py                          # default config, terse
    python scripts_repo/generate_all_tests.py path/to/config.json       # custom suite config
    python scripts_repo/generate_all_tests.py comparisons/example.json  # a comparison config
    python scripts_repo/generate_all_tests.py config.json --suite research_rubrics
    python scripts_repo/generate_all_tests.py config.json --verbose     # full JSON

Diagnostics (per-suite counts) always go to stderr. In ``--verbose`` mode stdout
is a clean JSON array, safe to pipe (e.g. ``| jq length``).
"""

import argparse
import atexit
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
# Generators do `import classification` / `import suite_config`; make those
# importable here too (they also insert this themselves, but be explicit).
sys.path.insert(0, str(TESTS_DIR))

# Key under which a comparison config (comparisons/*.json) nests the
# suite-generation config; matches run_comparison.py.
COMPARISON_SUITE_KEY = "suite-generation-config"


def _resolve_suite_config_path(config_path):
    """Return a path suite_config.py can consume from the supplied config file.

    Accepts a bare suite-generation config (returned as-is) or a full comparison
    config (its nested ``suite-generation-config`` block is written to a temp
    file, so the preview matches what run_comparison.py would feed promptfoo).
    """
    doc = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or COMPARISON_SUITE_KEY not in doc:
        return config_path  # already a bare suite-generation config
    suite_cfg = doc[COMPARISON_SUITE_KEY]
    fd, tmp_name = tempfile.mkstemp(prefix="suite_gen_", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        json.dump(suite_cfg, fh, indent=2)
    atexit.register(lambda: Path(tmp_name).unlink(missing_ok=True))
    print(
        f"Using suite-generation-config from comparison config {config_path}",
        file=sys.stderr,
    )
    return Path(tmp_name)


def _discover_generators():
    """Return the sorted list of ``tests/*_gen.py`` generator files."""
    return sorted(TESTS_DIR.glob("*_gen.py"))


def _load_generate_tests(path):
    """Import the generator module at ``path`` and return its ``generate_tests``."""
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "generate_tests", None)
    if not callable(fn):
        raise AttributeError(f"{path.name} has no callable generate_tests()")
    return fn


def _render_terse(test):
    """Return the four-line condensed view of one test case.

    Lines: suite, request_type, domain, user (prompt). The prompt has its
    internal whitespace collapsed so it stays on a single line.
    """
    metadata = test.get("metadata") or {}
    prompt = (test.get("vars") or {}).get("user", "")
    prompt = " ".join(str(prompt).split())
    return (
        f"suite:        {metadata.get('suite', '-')}\n"
        f"request_type: {metadata.get('request_type', '-')}\n"
        f"domain:       {metadata.get('domain', '-')}\n"
        f"user:         {prompt}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "config",
        nargs="?",
        help="Suite-generation config JSON, or a comparison config (its "
        "suite-generation-config block is used). Default: "
        "tests/suite_generation_config.json",
    )
    parser.add_argument(
        "--suite",
        action="append",
        metavar="NAME",
        help="Only run this suite (repeatable). Default: all discovered suites.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Emit the full JSON array of test cases instead of the default "
        "terse four-line-per-test view.",
    )
    args = parser.parse_args(argv)

    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            parser.error(f"config file not found: {config_path}")
        os.environ["SUITE_GENERATION_CONFIG_FILE"] = str(
            _resolve_suite_config_path(config_path)
        )

    generators = _discover_generators()
    if args.suite:
        wanted = set(args.suite)
        generators = [g for g in generators if g.stem[: -len("_gen")] in wanted]
        missing = wanted - {g.stem[: -len("_gen")] for g in generators}
        if missing:
            parser.error(f"unknown suite(s): {', '.join(sorted(missing))}")

    if not generators:
        parser.error("no generators found")

    all_tests = []
    for gen in generators:
        suite = gen.stem[: -len("_gen")]
        tests = _load_generate_tests(gen)()
        print(f"{suite}: {len(tests)} tests", file=sys.stderr)
        all_tests.extend(tests)

    print(f"total: {len(all_tests)} tests", file=sys.stderr)

    if args.verbose:
        json.dump(all_tests, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        print("\n\n".join(_render_terse(t) for t in all_tests))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run every suite generator and emit all the resulting tests to stdout.

Discovers each ``tests/<name>_gen.py`` (the same convention promptfoo uses),
calls its no-arg ``generate_tests()``, and prints the concatenated list of test
cases as JSON. This is a convenient way to preview exactly what a given
generation config would produce, without spinning up a promptfoo run.

Generation is driven by the suite-generation config file. Generators read it
lazily (via the ``SUITE_GENERATION_CONFIG_FILE`` env var, see
``tests/suite_config.py``), so we just point that env var at the supplied
config before invoking each generator.

Usage::

    python scripts_repo/generate_all_tests.py                       # default config
    python scripts_repo/generate_all_tests.py path/to/config.json   # custom config
    python scripts_repo/generate_all_tests.py config.json --suite research_rubrics

Output is a JSON array of promptfoo test cases on stdout; all diagnostics go to
stderr, so the stream is safe to pipe (e.g. ``| jq length``).
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_DIR = REPO_ROOT / "tests"
# Generators do `import classification` / `import suite_config`; make those
# importable here too (they also insert this themselves, but be explicit).
sys.path.insert(0, str(TESTS_DIR))


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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "config",
        nargs="?",
        help="Suite-generation config JSON (default: tests/suite_generation_config.json)",
    )
    parser.add_argument(
        "--suite",
        action="append",
        metavar="NAME",
        help="Only run this suite (repeatable). Default: all discovered suites.",
    )
    args = parser.parse_args(argv)

    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            parser.error(f"config file not found: {config_path}")
        os.environ["SUITE_GENERATION_CONFIG_FILE"] = str(config_path)

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
    json.dump(all_tests, sys.stdout, indent=2)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Promptfoo dynamic test generator for the simple_facts_regressions suite.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/simple_facts_regressions_gen.py:generate_tests

Replaces the former ``file://tests/simple_facts_regressions.csv`` reference. The
CSV holds prompts we want to keep an eye on for regressions (currently repeated
"What do you know about Zig?" rows expecting the latest version). Reading it
through the shared :mod:`csv_suite` helper gives the suite the standard
generator envelope. Per the repo's suite-naming convention the suite is
``simple_facts_regressions`` (the filename minus ``_gen``), which overrides the
``__metadata:suite`` value baked into the CSV.

Generation is configured via the suite-generation config file, keyed by the
``simple_facts_regressions`` suite name; see tests/suite_config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv_suite  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "simple_facts_regressions.csv"


def generate_tests():
    return csv_suite.generate_from_csv(__file__, CSV_PATH)


if __name__ == "__main__":
    import json

    print(json.dumps(generate_tests(), indent=2))

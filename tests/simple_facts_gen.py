#!/usr/bin/env python3
"""Promptfoo dynamic test generator for the simple_facts suite.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/simple_facts_gen.py:generate_tests

Replaces the former ``file://tests/simple_facts.csv`` reference. The CSV is
still the source of truth for the prompts/answers; this generator just reads it
through the shared :mod:`csv_suite` helper so the suite gains the standard
generator envelope (suite naming, classification, config-driven selection). Each
row is a single fact question graded by an ``icontains`` assertion on the
expected answer.

Generation is configured via the suite-generation config file, keyed by the
``simple_facts`` suite name; see tests/suite_config.py.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv_suite  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "simple_facts.csv"


def generate_tests():
    return csv_suite.generate_from_csv(__file__, CSV_PATH)


if __name__ == "__main__":
    import json

    print(json.dumps(generate_tests(), indent=2))

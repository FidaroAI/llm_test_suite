#!/usr/bin/env python3
"""Promptfoo dynamic test generator for the ``stock_prices`` suite.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/stock_prices_gen.py:generate_tests

Each row of ``tests/stock_prices.csv`` asks Fidaro for the latest price of a
named stock; the ``assert_stock_price`` assertion checks the answer is within
1% of the live price. The prompts/symbols come through the shared
:mod:`csv_suite` helper, so the suite gains the standard generator envelope
(suite naming, classification, config-driven selection).

The live reference price is **not** fetched here — that would make every
promptfoo run depend on Yahoo (promptfoo imports every generator on each run).
Instead the preflight ``scripts_repo/fetch_stock_prices.py`` writes a snapshot
that this generator bakes into each test's metadata. If the snapshot is absent
(preflight not run), the reference fields are simply omitted and the assertion
fails with a clear "run the preflight" message rather than the suite breaking.

Caching is disabled per test: a cached response would defeat the whole point of
checking whether Fidaro is fetching *up-to-date* data.

Generation is configured via the suite-generation config file, keyed by the
``stock_prices`` suite name; see tests/suite_config.py. It is left out of the
default config (so ordinary runs emit zero stock tests and need no snapshot);
``scripts_test/fidaro_stock_prices_config.json`` enables it for the dedicated
run driven by ``scripts_test/fidaro_stock_prices.sh``.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv_suite  # noqa: E402
import stock_prices_ref  # noqa: E402

CSV_PATH = Path(__file__).resolve().parent / "stock_prices.csv"


def _bake_reference(test, quotes, fetched_at):
    """Stamp the snapshot quote for this test's symbol into metadata + vars."""
    md = test.setdefault("metadata", {})
    quote = quotes.get(md.get("stooq_symbol"))
    if quote:
        ref = {
            "reference_price": quote["price"],
            "reference_currency": quote.get("currency", ""),
            "reference_fetched_at": fetched_at,
        }
        md.update(ref)
        # Mirror into vars too: guarantees the assertion can read the reference
        # regardless of whether promptfoo exposes test.metadata to it.
        test.setdefault("vars", {}).update(ref)
    # Never serve a cached answer for a freshness test.
    test.setdefault("options", {})["cache"] = False
    return test


def generate_tests():
    tests = csv_suite.generate_from_csv(__file__, CSV_PATH)
    snapshot = stock_prices_ref.load_snapshot() or {}
    quotes = snapshot.get("quotes", {})
    fetched_at = snapshot.get("fetched_at")
    return [_bake_reference(t, quotes, fetched_at) for t in tests]


if __name__ == "__main__":
    import json

    print(json.dumps(generate_tests(), indent=2))

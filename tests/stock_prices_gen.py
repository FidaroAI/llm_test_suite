#!/usr/bin/env python3
"""Promptfoo dynamic test generator for the ``stock_prices`` suite.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/stock_prices_gen.py:generate_tests

Each row of ``tests/stock_prices.csv`` asks Fidaro for the latest price of a
named stock; the ``assert_stock_price`` assertion checks the answer is within
1% of the live price. The prompts/symbols come through the shared
:mod:`csv_suite` helper, so the suite gains the standard generator envelope
(suite naming, classification, config-driven selection).

The live reference price is fetched here, but **only when the suite is actually
selected**. The suite is off in the default config, so an ordinary run emits
zero stock tests and never touches the network — important because promptfoo
imports every generator on each config load, and a source outage must not break
unrelated runs. When the suite *is* selected, the generator fetches every symbol
via :mod:`stock_prices_ref`, writes a fresh snapshot, and bakes the live quote
into each test's metadata. Any fetch failure aborts generation (fail fast) so
the suite never grades against partial data. The preflight
``scripts_repo/fetch_stock_prices.py`` remains available for warming or
inspecting the snapshot out of band.

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
    if not tests:
        # Suite not selected this run: emit nothing and never hit the network.
        # promptfoo imports every generator on every config load, so an
        # unrelated run must not depend on the price source being reachable.
        return []
    quotes, failures = stock_prices_ref.fetch_all(csv_path=CSV_PATH)
    if failures:
        detail = "; ".join(f"{sym}: {err}" for sym, err in sorted(failures.items()))
        raise stock_prices_ref.QuoteUnavailable(
            f"live stock-price fetch failed for {len(failures)} symbol(s): {detail}"
        )
    doc = stock_prices_ref.write_snapshot(quotes)
    return [_bake_reference(t, quotes, doc["fetched_at"]) for t in tests]


if __name__ == "__main__":
    import json

    print(json.dumps(generate_tests(), indent=2))

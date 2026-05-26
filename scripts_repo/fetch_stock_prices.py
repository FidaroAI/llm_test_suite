#!/usr/bin/env python3
"""Preflight: fetch live reference prices for the ``stock_prices`` suite.

Run this *before* the stock-price tests. It queries Stooq for every symbol in
``tests/stock_prices.csv`` and writes a timestamped snapshot
(``tests/stock_prices_reference.json``) that the suite generator bakes into the
tests. If *any* symbol cannot be fetched it prints the failures and exits
non-zero — fail fast, so we never grade against partial data.

    python scripts_repo/fetch_stock_prices.py
    python scripts_repo/fetch_stock_prices.py --output /tmp/snap.json

The generator and assertion never hit the network themselves (see the design
doc), so keeping the fetch here is what stops a Yahoo outage from breaking
unrelated suites when promptfoo loads every generator.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import stock_prices_ref as ref  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv", default=str(ref.CSV_PATH), help="suite CSV to read symbols from"
    )
    parser.add_argument(
        "--output", default=str(ref.SNAPSHOT_PATH), help="snapshot path to write"
    )
    args = parser.parse_args(argv)

    quotes, failures = ref.fetch_all(csv_path=args.csv)

    if failures:
        print(f"Failed to fetch {len(failures)} symbol(s):", file=sys.stderr)
        for symbol, err in failures.items():
            print(f"  {symbol}: {err}", file=sys.stderr)
        return 1

    doc = ref.write_snapshot(quotes, path=args.output)
    print(f"Fetched {len(quotes)} quotes at {doc['fetched_at']} -> {args.output}")
    for symbol, q in quotes.items():
        print(f"  {symbol:<14} {q['price']:>12,.2f} {q['currency']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

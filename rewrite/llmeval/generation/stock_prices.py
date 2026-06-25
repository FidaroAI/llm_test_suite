"""Generator for the ``stock_prices`` freshness suite.

Each row of ``stock_prices.csv`` asks for the latest price of a named stock; the
new ``stock_price`` assertion checks the answer is within 1% of the live price.
The live reference is fetched **at generation time** (via :mod:`llmeval.generation.stooq`)
and baked into each test's assertion params, so grading stays network-free.

Unlike the legacy promptfoo generator — which imported every generator on each
config load and so had to avoid fetching unless the suite was selected — this one
fetches whenever it is explicitly invoked (``llmeval generate --suite stock_prices``),
which is simpler and has no cross-suite import hazard.

Freshness/no-cache: because the reference is baked at generation time and graded
locally, run this suite with ``llmeval run --mode always`` (or a fresh DB) so a
cached answer cannot mask a freshness miss.

Ported from the legacy ``tests/stock_prices_gen.py`` + ``assertions/assert_stock_price.py``.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from llmeval.generation.classification import load_classifications, stamp
from llmeval.generation.common import make_id
from llmeval.generation.config import SuiteGenConfig
from llmeval.generation.stooq import QuoteUnavailable, fetch_all

SUITE = "stock_prices"

# A fetch is ``(csv_path) -> (quotes, failures)`` — injectable so tests never hit
# the network.
Fetch = Callable[[str], tuple[dict, dict]]


def _read_rows(csv_path: str) -> list[dict]:
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _row_to_test(row: dict, quotes: dict, fetched_at: str, mapping: dict) -> dict:
    prompt = (row.get("user") or "").strip()
    symbol = (row.get("__metadata:stooq_symbol") or "").strip()
    currency = (row.get("__metadata:currency") or "").strip()
    company = (row.get("__metadata:company") or "").strip()
    quote = quotes[symbol]
    reference = {
        "reference_price": quote["price"],
        "reference_currency": currency or quote.get("currency", ""),
        "reference_fetched_at": fetched_at,
    }
    test = {
        "id": make_id(SUITE, prompt),
        "user": prompt,
        "assertions": [
            {
                "type": "stock_price",
                "metric": "stock_price",
                # Reasoning-strip would corrupt number parsing of the full answer;
                # grade the raw answer.
                "transform": None,
                "params": {"symbol": symbol, **reference},
            }
        ],
        "metadata": {
            "suite": SUITE,
            "stooq_symbol": symbol,
            "currency": currency,
            "company": company,
            **reference,
        },
    }
    return stamp(test, prompt, mapping)


def generate_stock_prices(
    csv_path: str,
    config: SuiteGenConfig,
    classifications: dict,
    *,
    fetch: Fetch | None = None,
    now: str | None = None,
) -> list[dict]:
    """Fetch live quotes (fail fast on any failure) and bake them into test cases."""
    fetch = fetch or fetch_all
    quotes, failures = fetch(csv_path)
    if failures:
        detail = "; ".join(f"{sym}: {err}" for sym, err in sorted(failures.items()))
        raise QuoteUnavailable(
            f"live stock-price fetch failed for {len(failures)} symbol(s): {detail}"
        )
    fetched_at = now or datetime.now(timezone.utc).isoformat()
    tests = [
        _row_to_test(row, quotes, fetched_at, classifications)
        for row in _read_rows(csv_path)
        if (row.get("__metadata:stooq_symbol") or "").strip() in quotes
    ]
    return config.select(tests)


def load_and_generate(
    classifications_dir: str, config_path: str | None, generation_sources_dir: str
) -> list[dict]:
    from llmeval.generation.config import load_suite_config

    cfg = load_suite_config(SUITE, config_path)
    mapping = load_classifications(SUITE, classifications_dir)
    csv_path = str(Path(generation_sources_dir) / "stock_prices.csv")
    return generate_stock_prices(csv_path, cfg, mapping)

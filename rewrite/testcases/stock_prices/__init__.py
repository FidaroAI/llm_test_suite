"""``stock_prices`` — is the model quoting up-to-date market data?

Each CSV row asks for the latest price of one stock; the answer passes if it is within 1% of
the live price. The interesting part is *when* the reference is fetched:

    generate  -> CSV to test cases. No network. No price.
    grade     -> before_grade() fetches every symbol from Stooq, and the assertion compares
                 against that.

Fetching at grade time is what makes this a freshness test at all. The reference used to be
baked into the assertion at generation time, which meant a suite generated yesterday graded
today's answers against yesterday's prices, and the assertion needed a staleness guard to
notice. Now there is nothing to go stale.

The grader is a **bound method**, which is how it reads quotes the hook put on ``self``
without them having to travel through the test-case JSON.

Two consequences worth knowing:

* ``grade`` needs the network for this plugin, and ``--regrade`` re-fetches. Its grades are
  not reproducible from the store alone. That is the deliberate cost of grading live.
* Run it with ``llmeval run --mode always`` (or a fresh database). A *cached answer* would
  defeat a freshness check just as surely as a stale reference would.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llmeval.assertions.base import AssertionResult, GradeContext
from llmeval.generation.common import local_id
from llmeval.models import AssertionSpec
from llmeval.plugins import PluginInterface, TestCasePlugin

from .stooq import QuoteUnavailable, fetch_all

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent
CSV_PATH = PLUGIN_DIR / "stock_prices.csv"
CACHE_FILE = "testcases.json"
ASSERTION_NAME = "stock_price"

# Answers quote prices in all sorts of ways; pull every number out and take the closest.
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
DEFAULT_TOLERANCE_PCT = 1.0


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8-sig") as fh:
        return list(csv.DictReader(fh))


def _extract_numbers(text: str) -> list[float]:
    out = []
    for token in _NUMBER_RE.findall(text or ""):
        try:
            out.append(float(token.replace(",", "")))
        except ValueError:
            continue
    return out


class StockPricesPlugin(TestCasePlugin):
    """CSV in, freshness tests out; live quotes fetched in :meth:`before_grade`."""

    def __init__(self, interface: PluginInterface, fetch=None):
        self.interface = interface
        self.output_path = interface.cache_directory() / CACHE_FILE
        # Injectable so the tests never touch the network.
        self.fetch = fetch or fetch_all
        self.quotes: dict[str, dict[str, Any]] = {}
        self.fetched_at: str | None = None

    # -- generation -----------------------------------------------------------------------

    @property
    def assertion_type(self) -> str:
        """The namespaced type this plugin's cases must carry: ``<source>.stock_price``."""
        return f"{self.interface.name}.{ASSERTION_NAME}"

    def _row_to_case(self, row: dict[str, str]) -> dict[str, Any] | None:
        prompt = (row.get("user") or "").strip()
        symbol = (row.get("__metadata:stooq_symbol") or "").strip()
        if not prompt or not symbol:
            return None
        currency = (row.get("__metadata:currency") or "").strip()
        company = (row.get("__metadata:company") or "").strip()
        return {
            "id": local_id(prompt),
            "user": prompt,
            "assertions": [
                {
                    "type": self.assertion_type,
                    "metric": ASSERTION_NAME,
                    # Grade the raw answer: the reasoning-strip transform would happily
                    # remove the sentence with the number in it.
                    "transform": None,
                    "params": {"symbol": symbol, "currency": currency},
                }
            ],
            "metadata": {"stooq_symbol": symbol, "currency": currency, "company": company},
        }

    def generate_testcases(self) -> bool:
        try:
            rows = _read_rows(CSV_PATH)
        except OSError as exc:
            logger.error("stock_prices: cannot read %s (%s)", CSV_PATH, exc)
            return False
        cases = [case for case in (self._row_to_case(row) for row in rows) if case]
        self.output_path.write_text(
            json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("stock_prices: generated %d test case(s)", len(cases))
        return True

    def get_testcases(self) -> list[dict[str, Any]]:
        if not self.output_path.is_file():
            return []
        return json.loads(self.output_path.read_text(encoding="utf-8"))

    # -- grading --------------------------------------------------------------------------

    def get_custom_assertions(self):
        return {ASSERTION_NAME: self.grade_stock_price}

    def before_grade(self) -> None:
        """Fetch every symbol's live price. Fails the whole grade if any is unavailable.

        Fail fast rather than skip: a partially-fetched reference set would silently grade a
        subset and report a pass rate over the wrong denominator.
        """
        quotes, failures = self.fetch(str(CSV_PATH))
        if failures:
            detail = "; ".join(f"{sym}: {err}" for sym, err in sorted(failures.items()))
            raise QuoteUnavailable(
                f"live stock-price fetch failed for {len(failures)} symbol(s): {detail}"
            )
        self.quotes = quotes
        self.fetched_at = datetime.now(timezone.utc).isoformat()
        logger.info("stock_prices: fetched %d live quote(s)", len(quotes))

    def grade_stock_price(
        self, spec: AssertionSpec, output: Any, ctx: GradeContext
    ) -> AssertionResult:
        """Is the closest number in the answer within tolerance of the live price?"""
        # pylint: disable=unused-argument
        symbol = spec.params.get("symbol") or "?"
        currency = spec.params.get("currency") or ""
        tolerance = float(spec.params.get("tolerance_pct", DEFAULT_TOLERANCE_PCT))

        if not self.quotes:
            return AssertionResult(
                False, 0.0,
                f"{symbol}: no live quotes — before_grade did not run "
                "(grade through `llmeval grade`, which fires the plugin's hooks)",
            )
        quote = self.quotes.get(symbol)
        if quote is None:
            return AssertionResult(False, 0.0, f"{symbol}: not in the fetched quote set")
        reference = float(quote["price"])

        text = output if isinstance(output, str) else str(output or "")
        candidates = _extract_numbers(text)
        if not candidates:
            return AssertionResult(
                False, 0.0,
                f"no number found in answer for {symbol} (ref {reference:g} {currency})",
            )

        # UK listings are quoted in pence; an answer given in pounds is reference/100.
        targets = [reference] + ([reference / 100] if currency == "GBp" else [])
        best_candidate, best_diff = None, float("inf")
        for candidate in candidates:
            for target in targets:
                if target == 0:
                    continue
                diff = abs(candidate - target) / abs(target)
                if diff < best_diff:
                    best_diff, best_candidate = diff, candidate

        within = best_diff <= tolerance / 100.0
        reason = (
            f"{symbol}: live {reference:g} {currency} (fetched {self.fetched_at}); "
            f"closest answer {best_candidate:g} -> {best_diff * 100:.2f}% "
            f"{'≤' if within else '>'} {tolerance:g}% tolerance"
        )
        return AssertionResult(within, 1.0 if within else 0.0, reason)


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return StockPricesPlugin(interface)

"""Live stock-price reference data for the ``stock_prices`` suite.

This module is the single place that talks to the price source and the single
definition of the on-disk reference snapshot. It is imported by the preflight
CLI (``scripts_repo/fetch_stock_prices.py``) and by the suite generator
(``tests/stock_prices_gen.py``) — but only the preflight ever hits the network.
The generator and the assertion read the snapshot the preflight wrote, so a
source outage can never break promptfoo's config load (which imports *every*
generator on every run). See the design doc:
``docs/superpowers/specs/2026-05-25-stock-price-freshness-tests-design.md``.

**Source: Stooq** (``stooq.com``). Free, no API key, returns a one-line CSV per
symbol. Chosen over Yahoo Finance because Yahoo's chart endpoint hard-rate-limits
(HTTP 429) unauthenticated clients, whereas Stooq is reliable and covers every
exchange in this suite (US/UK/HK/JP/NL/FR/DE). It does *not* cover the Indian
NSE, which is why no NSE stocks are in the suite. Stooq symbols differ from the
common ticker (``arm.us``, ``hsba.uk``, ``700.hk`` with no leading zeros,
``7203.jp``, ``asml.nl``, ``mc.fr``, ``sap.de``) and are stored per row in the
CSV. Stooq does not report the currency, so each row also carries its quote
currency; UK listings are quoted in pence (``GBp``). It is deliberately isolated
to :func:`fetch_quote` so the source is easy to swap.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import requests

_HERE = Path(__file__).resolve().parent
CSV_PATH = _HERE / "stock_prices.csv"
SNAPSHOT_PATH = _HERE / "stock_prices_reference.json"

_QUOTE_URL = "https://stooq.com/q/l/"
# f=sd2t2c -> Symbol, Date, Time, Close. Stooq returns "N/D" fields for an
# unknown/unavailable symbol.
_QUOTE_PARAMS = {"f": "sd2t2c", "e": "csv"}
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT = 15


class QuoteUnavailable(RuntimeError):
    """Raised when a symbol's live price could not be obtained from the source."""


def _parse_csv_line(symbol, text):
    """Pull the close price out of Stooq's ``SYMBOL,DATE,TIME,CLOSE`` line."""
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    if not lines:
        raise QuoteUnavailable(f"{symbol}: empty response from Stooq")
    parts = [p.strip() for p in lines[-1].split(",")]
    if len(parts) < 4:
        raise QuoteUnavailable(f"{symbol}: unexpected Stooq line {lines[-1]!r}")
    date, close = parts[1], parts[-1]
    if date == "N/D" or close in ("N/D", ""):
        raise QuoteUnavailable(f"{symbol}: no data from Stooq (unknown symbol?)")
    try:
        return {"price": float(close), "as_of": f"{date} {parts[2]}".strip()}
    except ValueError as exc:
        raise QuoteUnavailable(f"{symbol}: non-numeric close {close!r}") from exc


def fetch_quote(symbol, session=None):
    """Return ``{"price": float, "as_of": str}`` for ``symbol`` from Stooq.

    Raises :class:`QuoteUnavailable` on any HTTP or parsing problem so the
    preflight can fail fast and name the offending symbol.
    """
    get = (session or requests).get
    try:
        resp = get(
            _QUOTE_URL,
            params={"s": symbol, **_QUOTE_PARAMS},
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
    except requests.RequestException as exc:  # network/DNS/timeout
        raise QuoteUnavailable(f"{symbol}: request failed: {exc}") from exc
    if resp.status_code != 200:
        raise QuoteUnavailable(f"{symbol}: HTTP {resp.status_code}")
    return _parse_csv_line(symbol, resp.text)


def symbols_from_csv(csv_path=CSV_PATH):
    """Yield ``(stooq_symbol, company, currency)`` triples from the suite CSV."""
    with Path(csv_path).open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            symbol = (row.get("__metadata:stooq_symbol") or "").strip()
            if symbol:
                yield (
                    symbol,
                    (row.get("__metadata:company") or "").strip(),
                    (row.get("__metadata:currency") or "").strip(),
                )


def fetch_all(csv_path=CSV_PATH, session=None):
    """Fetch every CSV symbol. Returns ``(quotes, failures)``.

    ``quotes`` maps ``symbol -> {price, currency, company, as_of}``; ``failures``
    maps ``symbol -> error message``. Sequential by design — gentle on the
    source and enough for ~20 symbols. ``currency`` comes from the CSV (Stooq
    does not report it).
    """
    session = session or requests.Session()
    quotes, failures = {}, {}
    for symbol, company, currency in symbols_from_csv(csv_path):
        try:
            quote = fetch_quote(symbol, session=session)
        except QuoteUnavailable as exc:
            failures[symbol] = str(exc)
            continue
        quote["company"] = company
        quote["currency"] = currency
        quotes[symbol] = quote
    return quotes, failures


def write_snapshot(quotes, path=SNAPSHOT_PATH):
    """Write ``quotes`` plus a UTC ``fetched_at`` stamp to ``path``."""
    doc = {
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "quotes": quotes,
    }
    Path(path).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def load_snapshot(path=SNAPSHOT_PATH):
    """Return the snapshot doc, or ``None`` if it has not been written yet."""
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

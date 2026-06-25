"""Live stock-price reference data from Stooq — ported from ``tests/stock_prices_ref.py``.

The single place that talks to the price source. **Source: Stooq** (free, no API
key, one-line CSV per symbol; chosen over Yahoo, which rate-limits unauthenticated
clients). Stooq symbols differ from the common ticker (``arm.us``, ``hsba.uk``,
``700.hk``, ``7203.jp``, ``asml.nl``, ``mc.fr``, ``sap.de``) and are stored per row
in the suite CSV, along with each row's quote currency (Stooq does not report it;
UK listings are quoted in pence, ``GBp``).

``requests`` is imported lazily so the core package has no network dependency;
install the ``stocks`` extra to use the default fetch.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


class QuoteUnavailable(RuntimeError):
    """Raised when a symbol's live price could not be obtained from the source."""


_QUOTE_URL = "https://stooq.com/q/l/"
# f=sd2t2c -> Symbol, Date, Time, Close. Stooq returns "N/D" for an unknown symbol.
_QUOTE_PARAMS = {"f": "sd2t2c", "e": "csv"}
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_TIMEOUT = 15


def _parse_csv_line(symbol: str, text: str) -> dict[str, Any]:
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


def fetch_quote(symbol: str, session=None) -> dict[str, Any]:
    """Return ``{"price": float, "as_of": str}`` for ``symbol`` from Stooq."""
    import requests  # lazy: keep the core package network-free

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


def symbols_from_csv(csv_path: str):
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


def fetch_all(csv_path: str, session=None):
    """Fetch every CSV symbol. Returns ``(quotes, failures)``.

    ``quotes`` maps ``symbol -> {price, currency, company, as_of}``; ``failures``
    maps ``symbol -> error message``. Sequential by design — gentle on the source.
    """
    import requests  # lazy

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

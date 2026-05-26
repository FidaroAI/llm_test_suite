"""Assert that Fidaro's answer quotes a live stock price within tolerance.

This is the grading half of the ``stock_prices`` suite (see
``tests/stock_prices_gen.py``). It is deliberately **network-free**: the live
reference price was fetched up front by ``scripts_repo/fetch_stock_prices.py``
and baked into the test by the generator. Here we just parse the number(s) out
of Fidaro's answer and compare to that reference.

Reference fields (read from ``context["test"]["metadata"]``, falling back to
``context["vars"]``):
    reference_price       float, in reference_currency units
    reference_currency    e.g. "USD", "JPY", "GBp" (London pence)
    reference_fetched_at  ISO-8601 UTC timestamp from the snapshot
    stooq_symbol          for readable failure reasons

Config keys (optional, from ``context["config"]``):
    tolerance_pct   pass band as a percentage (default 1.0)
    max_age_hours   reject a snapshot older than this (default 24)

Pass rule: any numeric token in the answer is within ``tolerance_pct`` of the
reference. For London ``GBp`` listings we also accept ``reference / 100`` so a
pounds-denominated answer (£9.00 vs 900p) matches.
"""

import re
from datetime import datetime, timezone

# A number with optional thousands separators and decimals: 2,815.00 / 2815 / 9.00
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _fail(reason):
    return {"pass": False, "score": 0.0, "reason": reason}


def _ref_field(context, key):
    """Read a baked reference field from test metadata, then vars."""
    test = (context or {}).get("test") or {}
    meta = test.get("metadata") or {}
    if key in meta:
        return meta[key]
    return ((context or {}).get("vars") or {}).get(key)


def _extract_numbers(text):
    out = []
    for tok in _NUMBER_RE.findall(text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _is_stale(fetched_at, max_age_hours):
    if not fetched_at:
        return None  # unknown age; don't block on it
    try:
        ts = datetime.fromisoformat(fetched_at)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    return age_h if age_h > max_age_hours else None


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    tol_pct = float(cfg.get("tolerance_pct", 1.0))
    max_age_hours = float(cfg.get("max_age_hours", 24))

    symbol = _ref_field(context, "stooq_symbol") or "?"
    reference = _ref_field(context, "reference_price")
    currency = _ref_field(context, "reference_currency") or ""
    fetched_at = _ref_field(context, "reference_fetched_at")

    if reference is None:
        return _fail(
            f"no reference price for {symbol}: run "
            "scripts_repo/fetch_stock_prices.py before the stock_prices suite"
        )
    try:
        reference = float(reference)
    except (TypeError, ValueError):
        return _fail(f"reference price for {symbol} is not a number: {reference!r}")

    stale_age = _is_stale(fetched_at, max_age_hours)
    if stale_age is not None:
        return _fail(
            f"reference price for {symbol} is stale ({stale_age:.1f}h old > "
            f"{max_age_hours:.0f}h): re-run scripts_repo/fetch_stock_prices.py"
        )

    text = output if isinstance(output, str) else str(output)
    candidates = _extract_numbers(text)
    if not candidates:
        return _fail(
            f"no number found in answer for {symbol} "
            f"(reference {reference:g} {currency})"
        )

    # Targets the answer may legitimately match. GBp listings are quoted in
    # pence on Yahoo; an answer in pounds is reference/100.
    targets = [reference]
    if currency == "GBp":
        targets.append(reference / 100)

    tol = tol_pct / 100.0
    best_cand, best_diff = None, float("inf")
    for cand in candidates:
        for target in targets:
            if target == 0:
                continue
            diff = abs(cand - target) / abs(target)
            if diff < best_diff:
                best_diff, best_cand = diff, cand

    within = best_diff <= tol
    pct = best_diff * 100
    verb = "≤" if within else ">"
    reason = (
        f"{symbol}: reference {reference:g} {currency}; closest answer "
        f"{best_cand:g} -> {pct:.2f}% {verb} {tol_pct:g}% tolerance"
    )
    return {"pass": within, "score": 1.0 if within else 0.0, "reason": reason}

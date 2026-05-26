from datetime import datetime, timedelta, timezone

from assertions.assert_stock_price import get_assert


def _ctx(price=None, currency="USD", fetched_at="now", symbol="TEST", **config):
    if fetched_at == "now":
        fetched_at = datetime.now(timezone.utc).isoformat()
    meta = {"stooq_symbol": symbol}
    if price is not None:
        meta["reference_price"] = price
        meta["reference_currency"] = currency
        meta["reference_fetched_at"] = fetched_at
    return {"test": {"metadata": meta}, "config": config}


def test_passes_when_answer_within_one_percent():
    r = get_assert("Arm is trading at about $148.20 right now.", _ctx(price=148.0))
    assert r["pass"] is True
    assert "≤" in r["reason"]


def test_fails_when_answer_outside_tolerance():
    r = get_assert("Arm last traded at $160.", _ctx(price=148.0))
    assert r["pass"] is False
    assert "148" in r["reason"] and "160" in r["reason"]


def test_handles_thousands_separator():
    r = get_assert("Toyota is at ¥2,815.00 on the TSE.", _ctx(price=2820.0, currency="JPY"))
    assert r["pass"] is True


def test_gbp_pence_reference_matches_pounds_answer():
    # Yahoo quotes London in pence (GBp); a pounds answer is reference/100.
    r = get_assert("HSBC is around £9.01.", _ctx(price=900.0, currency="GBp"))
    assert r["pass"] is True


def test_gbp_pence_reference_matches_pence_answer():
    r = get_assert("HSBC is around 899 pence.", _ctx(price=900.0, currency="GBp"))
    assert r["pass"] is True


def test_fails_when_no_number_in_answer():
    r = get_assert("I cannot provide real-time market data.", _ctx(price=148.0))
    assert r["pass"] is False
    assert "no number" in r["reason"]


def test_fails_when_reference_missing():
    r = get_assert("$148.20", _ctx(price=None))
    assert r["pass"] is False
    assert "fetch_stock_prices" in r["reason"]


def test_fails_when_reference_stale():
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    r = get_assert("$148.20", _ctx(price=148.0, fetched_at=old))
    assert r["pass"] is False
    assert "stale" in r["reason"]


def test_custom_tolerance_via_config():
    # 3% off normally fails, but a 5% tolerance lets it pass.
    r = get_assert("$152.50", _ctx(price=148.0, tolerance_pct=5))
    assert r["pass"] is True


def test_reads_reference_from_vars_fallback():
    ctx = {
        "vars": {
            "stooq_symbol": "TEST",
            "reference_price": 100.0,
            "reference_currency": "USD",
            "reference_fetched_at": datetime.now(timezone.utc).isoformat(),
        },
        "config": {},
    }
    assert get_assert("It's $100.40.", ctx)["pass"] is True

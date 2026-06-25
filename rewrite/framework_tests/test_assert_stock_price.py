from datetime import datetime, timedelta, timezone

from llmeval.assertions import grade_assertion
from llmeval.models import AssertionSpec

# Import for side effect: registers the assertion.
import llmeval.assertions.deterministic  # noqa: F401


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _spec(**params):
    params.setdefault("reference_fetched_at", _now_iso())
    return AssertionSpec(type="stock_price", transform=None, params=params)


def test_within_tolerance_passes():
    spec = _spec(reference_price=100.0, reference_currency="USD", symbol="x.us")
    res = grade_assertion(spec, "The price is $100.50 today.")
    assert res.passed
    assert res.score == 1.0


def test_outside_tolerance_fails():
    spec = _spec(reference_price=100.0, reference_currency="USD", symbol="x.us")
    res = grade_assertion(spec, "It trades around $130.")
    assert not res.passed


def test_gbp_pence_vs_pounds_accepted():
    # reference is 900 pence; an answer in pounds (9.00) should still match.
    spec = _spec(reference_price=900.0, reference_currency="GBp", symbol="hsba.uk")
    res = grade_assertion(spec, "HSBC is currently £9.00.")
    assert res.passed


def test_missing_reference_fails():
    spec = _spec(reference_currency="USD", symbol="x.us")  # no reference_price
    res = grade_assertion(spec, "$100")
    assert not res.passed
    assert "reference" in res.reason.lower()


def test_no_number_in_answer_fails():
    spec = _spec(reference_price=100.0, reference_currency="USD", symbol="x.us")
    res = grade_assertion(spec, "I cannot find a price.")
    assert not res.passed


def test_stale_reference_fails():
    old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
    spec = AssertionSpec(type="stock_price", transform=None, params={
        "reference_price": 100.0, "reference_currency": "USD", "symbol": "x.us",
        "reference_fetched_at": old, "max_age_hours": 24})
    res = grade_assertion(spec, "$100")
    assert not res.passed
    assert "stale" in res.reason.lower()


def test_custom_tolerance_pct():
    # 5% band: 104 is within of reference 100.
    spec = _spec(reference_price=100.0, reference_currency="USD", symbol="x.us",
                 tolerance_pct=5.0)
    res = grade_assertion(spec, "around $104")
    assert res.passed

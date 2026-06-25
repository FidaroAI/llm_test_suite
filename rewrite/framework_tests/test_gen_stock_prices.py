import pytest

from llmeval.generation.config import DEFAULTS, SuiteGenConfig
from llmeval.generation.stock_prices import QuoteUnavailable, generate_stock_prices


def _cfg(**over):
    return SuiteGenConfig("stock_prices", {**DEFAULTS, "number_to_generate": None, **over})


CSV = (
    "user,__expected,__metadata:stooq_symbol,__metadata:currency,__metadata:company\n"
    '"Price of Arm?","python:file://x.py","arm.us","USD","Arm Holdings"\n'
    '"Price of HSBC?","python:file://x.py","hsba.uk","GBp","HSBC Holdings"\n'
)


def _csv(tmp_path):
    p = tmp_path / "stock_prices.csv"
    p.write_text(CSV)
    return str(p)


def _fake_fetch(csv_path):
    quotes = {
        "arm.us": {"price": 120.5, "currency": "USD", "company": "Arm Holdings", "as_of": "now"},
        "hsba.uk": {"price": 900.0, "currency": "GBp", "company": "HSBC Holdings", "as_of": "now"},
    }
    return quotes, {}


def test_emits_stock_price_assertions(tmp_path):
    cases = generate_stock_prices(_csv(tmp_path), _cfg(), {}, fetch=_fake_fetch)
    assert len(cases) == 2
    a = cases[0]["assertions"][0]
    assert a["type"] == "stock_price"


def test_reference_price_is_baked_into_params(tmp_path):
    cases = generate_stock_prices(_csv(tmp_path), _cfg(), {}, fetch=_fake_fetch)
    arm = next(c for c in cases if c["metadata"]["stooq_symbol"] == "arm.us")
    params = arm["assertions"][0]["params"]
    assert params["reference_price"] == 120.5
    assert params["reference_currency"] == "USD"
    assert params["symbol"] == "arm.us"
    assert params["reference_fetched_at"]


def test_metadata_carries_company_and_symbol(tmp_path):
    cases = generate_stock_prices(_csv(tmp_path), _cfg(), {}, fetch=_fake_fetch)
    arm = next(c for c in cases if c["metadata"]["stooq_symbol"] == "arm.us")
    assert arm["metadata"]["company"] == "Arm Holdings"
    assert arm["metadata"]["suite"] == "stock_prices"


def test_fetch_failure_aborts(tmp_path):
    def failing(csv_path):
        return {}, {"arm.us": "HTTP 500"}

    with pytest.raises(QuoteUnavailable):
        generate_stock_prices(_csv(tmp_path), _cfg(), {}, fetch=failing)


def test_selection_cap_applies(tmp_path):
    cases = generate_stock_prices(_csv(tmp_path), _cfg(number_to_generate=1), {}, fetch=_fake_fetch)
    assert len(cases) == 1

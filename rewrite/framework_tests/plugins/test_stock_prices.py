"""The stock_prices plugin: generation is offline, grading fetches live quotes."""

import json
import logging
import sys
from pathlib import Path

import pytest

from llmeval.assertions.base import GradeContext
from llmeval.models import AssertionSpec
from llmeval.plugins.loader import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]

QUOTES = {
    "arm.us": {"price": 100.0, "currency": "USD", "company": "Arm", "as_of": "x"},
    "hsba.uk": {"price": 1000.0, "currency": "GBp", "company": "HSBC", "as_of": "x"},
}


def fake_fetch(_csv_path):
    return dict(QUOTES), {}


def failing_fetch(_csv_path):
    return {}, {"arm.us": "HTTP 500"}


@pytest.fixture(name="plugin")
def _plugin(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "testcases"
    root.mkdir()
    (root / "stock_prices").symlink_to(PROJECT_ROOT / "testcases" / "stock_prices")
    loaded = load(names=["stock_prices"], root=root)
    plugin = loaded.sources[0].plugin
    plugin.fetch = fake_fetch
    return plugin


def grade(plugin, symbol, currency, answer):
    spec = AssertionSpec(
        type="stock_prices.stock_price",
        params={"symbol": symbol, "currency": currency},
    )
    return plugin.get_custom_assertions()["stock_price"](spec, answer, GradeContext())


def test_generation_is_offline_and_bakes_no_price(plugin):
    plugin.fetch = lambda _p: pytest.fail("generation must not hit the network")
    assert plugin.generate_testcases() is True
    cases = json.loads(plugin.output_path.read_text())
    assert cases, "expected cases from the CSV"
    (assertion,) = cases[0]["assertions"]
    assert assertion["type"] == "stock_prices.stock_price"
    assert set(assertion["params"]) == {"symbol", "currency"}


def test_duplicate_csv_rows_are_dropped_rather_than_emitting_clashing_ids(
    plugin, tmp_path, monkeypatch, caplog
):
    csv_path = tmp_path / "dupes.csv"
    csv_path.write_text(
        "user,__metadata:stooq_symbol,__metadata:currency,__metadata:company\n"
        "Price of Arm?,arm.us,USD,Arm\n"
        "Price of Arm?,arm.us,USD,Arm\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(sys.modules[type(plugin).__module__], "CSV_PATH", csv_path)

    with caplog.at_level(logging.WARNING):
        assert plugin.generate_testcases() is True

    (case,) = plugin.get_testcases()
    assert case["metadata"]["stooq_symbol"] == "arm.us"
    assert "stock_prices" in caplog.text and "Price of Arm?" in caplog.text


def test_grading_before_a_fetch_fails_loudly(plugin):
    result = grade(plugin, "arm.us", "USD", "Arm is at $100.")
    assert not result.passed
    assert "before_grade" in result.reason


def test_before_grade_fetches_and_grading_uses_the_live_quote(plugin):
    plugin.before_grade()
    assert grade(plugin, "arm.us", "USD", "Arm last traded at $100.40.").passed
    assert not grade(plugin, "arm.us", "USD", "Arm last traded at $130.00.").passed


def test_a_gbp_answer_in_pounds_matches_a_gbp_reference_in_pence(plugin):
    plugin.before_grade()
    assert grade(plugin, "hsba.uk", "GBp", "HSBC is around £10.02.").passed


def test_an_answer_with_no_number_fails(plugin):
    plugin.before_grade()
    assert not grade(plugin, "arm.us", "USD", "I could not find a price.").passed


def test_an_unknown_symbol_fails_rather_than_raising(plugin):
    plugin.before_grade()
    result = grade(plugin, "nope.us", "USD", "It is $5.")
    assert not result.passed
    assert "nope.us" in result.reason


def test_before_grade_fails_fast_when_the_source_is_down(plugin):
    plugin.fetch = failing_fetch
    with pytest.raises(RuntimeError, match="arm.us"):
        plugin.before_grade()

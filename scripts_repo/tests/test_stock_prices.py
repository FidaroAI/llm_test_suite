"""Unit tests for the stock_prices fetch/snapshot/generator layer (no network)."""

import importlib.util
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]


def _load(name, relpath):
    spec = importlib.util.spec_from_file_location(name, _ROOT / relpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("stock_prices_ref", "tests/stock_prices_ref.py")
gen = _load("stock_prices_gen", "tests/stock_prices_gen.py")
preflight = _load("fetch_stock_prices", "scripts_repo/fetch_stock_prices.py")


class _Resp:
    def __init__(self, text, status=200):
        self.text = text
        self.status_code = status


class _Session:
    def __init__(self, resp):
        self._resp = resp

    def get(self, *args, **kwargs):
        return self._resp


# --- fetch_quote (Stooq CSV) ----------------------------------------------

def test_fetch_quote_parses_close_price():
    line = "ARM.US,2026-05-22,22:00:18,306.51"
    q = ref.fetch_quote("arm.us", session=_Session(_Resp(line)))
    assert q["price"] == 306.51
    assert q["as_of"] == "2026-05-22 22:00:18"


def test_fetch_quote_ignores_header_row():
    text = "Symbol,Date,Time,Close\n7203.JP,2026-05-22,08:00:00,2987"
    q = ref.fetch_quote("7203.jp", session=_Session(_Resp(text)))
    assert q["price"] == 2987.0


def test_fetch_quote_raises_on_nd_unknown_symbol():
    line = "RELIANCE.IN,N/D,N/D,N/D"
    try:
        ref.fetch_quote("reliance.in", session=_Session(_Resp(line)))
    except ref.QuoteUnavailable as e:
        assert "no data" in str(e)
    else:
        raise AssertionError("expected QuoteUnavailable")


def test_fetch_quote_raises_on_http_error():
    try:
        ref.fetch_quote("arm.us", session=_Session(_Resp("", status=429)))
    except ref.QuoteUnavailable as e:
        assert "429" in str(e)
    else:
        raise AssertionError("expected QuoteUnavailable")


def test_fetch_quote_raises_on_non_numeric_close():
    line = "ARM.US,2026-05-22,22:00:18,oops"
    try:
        ref.fetch_quote("arm.us", session=_Session(_Resp(line)))
    except ref.QuoteUnavailable as e:
        assert "non-numeric" in str(e)
    else:
        raise AssertionError("expected QuoteUnavailable")


# --- snapshot IO -----------------------------------------------------------

def test_snapshot_roundtrip(tmp_path):
    path = tmp_path / "snap.json"
    quotes = {"arm.us": {"price": 306.51, "currency": "USD", "company": "Arm"}}
    doc = ref.write_snapshot(quotes, path=path)
    assert "fetched_at" in doc
    assert ref.load_snapshot(path=path)["quotes"] == quotes


def test_load_snapshot_missing_returns_none(tmp_path):
    assert ref.load_snapshot(path=tmp_path / "absent.json") is None


def test_csv_has_20_symbols_with_currency():
    rows = list(ref.symbols_from_csv())
    assert len(rows) == 20
    assert ("7203.jp", "Toyota Motor Corporation", "JPY") in rows
    assert ("hsba.uk", "HSBC Holdings", "GBp") in rows


# --- generator baking ------------------------------------------------------

def test_bake_reference_stamps_metadata_and_disables_cache():
    test = {"metadata": {"stooq_symbol": "7203.jp"}, "vars": {"user": "q"}}
    quotes = {"7203.jp": {"price": 2987.0, "currency": "JPY", "company": "Toyota"}}
    out = gen._bake_reference(test, quotes, "2026-05-26T00:00:00+00:00")
    assert out["metadata"]["reference_price"] == 2987.0
    assert out["metadata"]["reference_currency"] == "JPY"
    assert out["vars"]["reference_price"] == 2987.0
    assert out["options"]["cache"] is False


def test_bake_reference_without_quote_still_disables_cache():
    test = {"metadata": {"stooq_symbol": "unknown"}}
    out = gen._bake_reference(test, {}, "2026-05-26T00:00:00+00:00")
    assert "reference_price" not in out["metadata"]
    assert out["options"]["cache"] is False


def test_generate_tests_bakes_when_enabled(monkeypatch, tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"stock_prices": {"number_to_generate": None,
                                                "randomize_selection": False,
                                                "random_seed": 0,
                                                "max_rubrics": None}}))
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(cfg))
    monkeypatch.setattr(gen.stock_prices_ref, "load_snapshot", lambda: {
        "fetched_at": "2026-05-26T00:00:00+00:00",
        "quotes": {"7203.jp": {"price": 2987.0, "currency": "JPY", "company": "Toyota"}},
    })
    tests = gen.generate_tests()
    assert len(tests) == 20
    assert all(t["options"]["cache"] is False for t in tests)
    toyota = next(t for t in tests if t["metadata"]["stooq_symbol"] == "7203.jp")
    assert toyota["metadata"]["reference_price"] == 2987.0


# --- preflight CLI ---------------------------------------------------------

def test_preflight_returns_nonzero_on_failure(monkeypatch, capsys):
    monkeypatch.setattr(preflight.ref, "fetch_all",
                        lambda **kw: ({}, {"reliance.in": "no data from Stooq"}))
    assert preflight.main([]) == 1
    assert "reliance.in" in capsys.readouterr().err


def test_preflight_writes_snapshot_on_success(monkeypatch, tmp_path):
    quotes = {"arm.us": {"price": 306.51, "currency": "USD", "company": "Arm"}}
    monkeypatch.setattr(preflight.ref, "fetch_all", lambda **kw: (quotes, {}))
    out = tmp_path / "snap.json"
    assert preflight.main(["--output", str(out)]) == 0
    assert json.loads(out.read_text())["quotes"] == quotes

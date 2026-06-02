"""Tests for the raw-response CSV export in compare_runs.py."""
from __future__ import annotations

import csv
import json

from scripts_repo.compare_runs import (
    build_response_csv,
    extract_responses,
    main,
    strip_reasoning,
    write_response_csv,
)
from scripts_repo.tests._fixtures import make_eval_json, response_result


def _user_prompt(text):
    return json.dumps([{"role": "user", "content": text}])


# --- strip_reasoning -------------------------------------------------------


def test_strip_reasoning_drops_prefix_through_first_triple_newline():
    assert strip_reasoning("thinking...\n\n\nfinal answer") == "final answer"


def test_strip_reasoning_only_splits_on_first_delimiter():
    assert strip_reasoning("a\n\n\nb\n\n\nc") == "b\n\n\nc"


def test_strip_reasoning_no_delimiter_returns_unchanged():
    assert strip_reasoning("no reasoning here") == "no reasoning here"


def test_strip_reasoning_non_string_returned_unchanged():
    assert strip_reasoning({"parsed": 1}) == {"parsed": 1}
    assert strip_reasoning(None) is None


# --- extract_responses -----------------------------------------------------


def test_extract_responses_scopes_to_provider_and_strips_reasoning():
    ev = make_eval_json([
        response_result("L_prod", "t1", "simple_facts",
                        "reasoning\n\n\nProd answer", _user_prompt("Q1?")),
        response_result("L_ven", "t1", "simple_facts",
                        "reasoning\n\n\nVenice answer", _user_prompt("Q1?")),
    ])
    prod = extract_responses(ev, "L_prod")
    assert set(prod) == {"t1"}
    assert prod["t1"]["output"] == "Prod answer"
    assert prod["t1"]["prompt"] == "Q1?"
    assert prod["t1"]["suite"] == "simple_facts"


def test_extract_responses_missing_output_is_empty_string():
    ev = make_eval_json([response_result("L_prod", "t1", "s", None)])
    assert extract_responses(ev, "L_prod")["t1"]["output"] == ""


# --- write_response_csv ----------------------------------------------------


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def test_write_response_csv_joins_providers_on_test_identity(tmp_path):
    per_provider = {
        "prod": {"t1": {"prompt": "Q1?", "output": "P1", "suite": "s"}},
        "venice": {"t1": {"prompt": "Q1?", "output": "V1", "suite": "s"}},
    }
    out = tmp_path / "responses.csv"
    n = write_response_csv(per_provider, ["prod", "venice"], out)
    rows = _read_csv(out)
    assert n == 1
    assert rows[0] == ["prompt", "prod", "venice"]
    assert rows[1] == ["Q1?", "P1", "V1"]


def test_write_response_csv_missing_provider_cell_is_blank(tmp_path):
    per_provider = {
        "prod": {"t1": {"prompt": "Q1?", "output": "P1", "suite": "s"}},
        "venice": {},  # venice never ran t1
    }
    out = tmp_path / "responses.csv"
    write_response_csv(per_provider, ["prod", "venice"], out)
    rows = _read_csv(out)
    assert rows[1] == ["Q1?", "P1", ""]


def test_write_response_csv_orders_by_suite_then_test(tmp_path):
    per_provider = {
        "prod": {
            "z_test": {"prompt": "Qz", "output": "Z", "suite": "a_suite"},
            "a_test": {"prompt": "Qa", "output": "A", "suite": "b_suite"},
        },
    }
    out = tmp_path / "responses.csv"
    write_response_csv(per_provider, ["prod"], out)
    rows = _read_csv(out)
    # a_suite sorts before b_suite, regardless of test-name order.
    assert [r[0] for r in rows[1:]] == ["Qz", "Qa"]


def test_write_response_csv_quotes_commas_quotes_and_newlines(tmp_path):
    gnarly = 'has, comma and "quotes"\nand a newline'
    per_provider = {"prod": {"t1": {"prompt": "Q?", "output": gnarly, "suite": "s"}}}
    out = tmp_path / "responses.csv"
    write_response_csv(per_provider, ["prod"], out)
    rows = _read_csv(out)  # round-trip through the csv reader
    assert rows[1] == ["Q?", gnarly]


# --- build_response_csv (N-provider wrapper over an eval file) -------------


def test_build_response_csv_splits_unified_eval_by_column_label(tmp_path):
    from scripts_repo.compare_runs import ProviderColumn

    ev = make_eval_json([
        response_result("L_prod", "t1", "s", "r\n\n\nProd", _user_prompt("Q1?")),
        response_result("L_ven", "t1", "s", "r\n\n\nVen", _user_prompt("Q1?")),
    ])
    columns = [
        ProviderColumn("fidaro-prod", "L_prod", True),
        ProviderColumn("venice", "L_ven", False),
    ]
    out = tmp_path / "responses.csv"
    build_response_csv(ev, columns, out)
    rows = _read_csv(out)
    assert rows[0] == ["prompt", "fidaro-prod", "venice"]
    assert rows[1] == ["Q1?", "Prod", "Ven"]


# --- CLI integration -------------------------------------------------------


def test_main_n_writes_csv_when_flag_given(tmp_path):
    ev = make_eval_json([
        response_result("L_prod", "t1", "s", "r\n\n\nProd", _user_prompt("Q1?")),
        response_result("L_ven", "t1", "s", "r\n\n\nVen", _user_prompt("Q1?")),
    ])
    f = tmp_path / "u.json"
    f.write_text(json.dumps(ev))
    out = tmp_path / "r.html"
    csv_out = tmp_path / "responses.csv"
    rc = main([
        str(f), str(f),
        "--baseline-provider-col", "fidaro-prod=L_prod",
        "--provider-col", "venice=L_ven",
        "--out", str(out),
        "--csv", str(csv_out),
    ])
    assert rc == 0
    rows = _read_csv(csv_out)
    assert rows[0] == ["prompt", "fidaro-prod", "venice"]
    assert rows[1] == ["Q1?", "Prod", "Ven"]

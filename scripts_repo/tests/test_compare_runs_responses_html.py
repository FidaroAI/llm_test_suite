"""Tests for the interactive raw-response HTML table in compare_runs.py."""
from __future__ import annotations

import json

from scripts_repo.compare_runs import (
    ProviderColumn,
    build_responses_html,
    main,
    render_responses_html,
    write_responses_html,
)
from scripts_repo.tests._fixtures import make_eval_json, response_result


def _user_prompt(text):
    return json.dumps([{"role": "user", "content": text}])


# --- render_responses_html -------------------------------------------------


def test_render_has_prompt_and_provider_headers():
    per_provider = {
        "fidaro-prod": {"t1": {"prompt": "Q1?", "output": "P1", "suite": "s"}},
        "venice": {"t1": {"prompt": "Q1?", "output": "V1", "suite": "s"}},
    }
    html = render_responses_html(per_provider, ["fidaro-prod", "venice"])
    assert "<th" in html
    for header in ("prompt", "fidaro-prod", "venice"):
        assert header in html


def test_render_includes_prompt_and_each_response_cell():
    per_provider = {
        "prod": {"t1": {"prompt": "What is X?", "output": "Prod answer", "suite": "s"}},
        "venice": {"t1": {"prompt": "What is X?", "output": "Venice answer", "suite": "s"}},
    }
    html = render_responses_html(per_provider, ["prod", "venice"])
    assert "What is X?" in html
    assert "Prod answer" in html
    assert "Venice answer" in html


def test_render_escapes_html_in_responses():
    payload = '<script>alert("xss")</script>'
    per_provider = {"prod": {"t1": {"prompt": "Q?", "output": payload, "suite": "s"}}}
    html = render_responses_html(per_provider, ["prod"])
    assert payload not in html
    assert "&lt;script&gt;" in html


def test_render_missing_provider_cell_still_emits_row():
    per_provider = {
        "prod": {"t1": {"prompt": "Q1?", "output": "P1", "suite": "s"}},
        "venice": {},  # venice never ran t1
    }
    html = render_responses_html(per_provider, ["prod", "venice"])
    assert "Q1?" in html
    assert "P1" in html


def test_render_orders_rows_by_suite_then_test():
    per_provider = {
        "prod": {
            "z_test": {"prompt": "Qz", "output": "Z", "suite": "a_suite"},
            "a_test": {"prompt": "Qa", "output": "A", "suite": "b_suite"},
        },
    }
    html = render_responses_html(per_provider, ["prod"])
    # a_suite (Qz) must appear before b_suite (Qa) in the document.
    assert html.index("Qz") < html.index("Qa")


def test_render_includes_resize_and_wrap_affordances():
    per_provider = {"prod": {"t1": {"prompt": "Q?", "output": "A", "suite": "s"}}}
    html = render_responses_html(per_provider, ["prod"])
    # Column + row resize handles, word-wrap, responsive width, and the drag JS.
    assert "col-resize" in html
    assert "row-resize" in html
    assert "overflow-wrap" in html
    assert "width:100%" in html or "width: 100%" in html
    assert "<script>" in html


# --- write / build wrappers ------------------------------------------------


def test_write_responses_html_writes_file_and_counts_rows(tmp_path):
    per_provider = {"prod": {"t1": {"prompt": "Q?", "output": "A", "suite": "s"}}}
    out = tmp_path / "responses.html"
    n = write_responses_html(per_provider, ["prod"], out)
    assert n == 1
    assert "<table" in out.read_text(encoding="utf-8")


def test_build_responses_html_splits_unified_eval_by_column_label(tmp_path):
    ev = make_eval_json([
        response_result("L_prod", "t1", "s", "r\n\n\nProd", _user_prompt("Q1?")),
        response_result("L_ven", "t1", "s", "r\n\n\nVen", _user_prompt("Q1?")),
    ])
    columns = [
        ProviderColumn("fidaro-prod", "L_prod", True),
        ProviderColumn("venice", "L_ven", False),
    ]
    out = tmp_path / "responses.html"
    build_responses_html(ev, columns, out)
    text = out.read_text(encoding="utf-8")
    assert "fidaro-prod" in text and "venice" in text
    assert "Prod" in text and "Ven" in text


# --- CLI integration -------------------------------------------------------


def test_main_n_writes_responses_html_when_flag_given(tmp_path):
    ev = make_eval_json([
        response_result("L_prod", "t1", "s", "r\n\n\nProd", _user_prompt("Q1?")),
        response_result("L_ven", "t1", "s", "r\n\n\nVen", _user_prompt("Q1?")),
    ])
    f = tmp_path / "u.json"
    f.write_text(json.dumps(ev))
    out = tmp_path / "r.html"
    resp_out = tmp_path / "responses.html"
    rc = main([
        str(f), str(f),
        "--baseline-provider-col", "fidaro-prod=L_prod",
        "--provider-col", "venice=L_ven",
        "--out", str(out),
        "--responses-html", str(resp_out),
    ])
    assert rc == 0
    text = resp_out.read_text(encoding="utf-8")
    assert "<table" in text
    assert "Prod" in text and "Ven" in text

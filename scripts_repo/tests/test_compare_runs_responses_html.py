"""Tests for the interactive raw-response HTML table in compare_runs.py."""
from __future__ import annotations

import json

from scripts_repo.compare_runs import (
    ProviderColumn,
    build_responses_html,
    looks_like_markdown,
    main,
    render_markdown,
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


# --- markdown detection ----------------------------------------------------


def test_looks_like_markdown_true_for_strong_markers():
    assert looks_like_markdown("This is **bold** text")
    assert looks_like_markdown("## A heading")
    assert looks_like_markdown("- a bullet\n- another")
    assert looks_like_markdown("1. first\n2. second")
    assert looks_like_markdown("See [the docs](https://example.com)")
    assert looks_like_markdown("> a quote")


def test_looks_like_markdown_false_for_plain_prose():
    assert not looks_like_markdown("The world's largest retailer is Walmart.")
    assert not looks_like_markdown("A times B is 2 * 3 = 6 in arithmetic.")
    assert not looks_like_markdown("snake_case_name and another_one")
    assert not looks_like_markdown("")


# --- render_markdown -------------------------------------------------------


def test_render_markdown_bold_and_italic():
    out = render_markdown("a **bold** and *italic* word")
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out


def test_render_markdown_headings():
    assert "<h2>Title</h2>" in render_markdown("## Title")


def test_render_markdown_unordered_list():
    out = render_markdown("- one\n- two")
    assert "<ul>" in out and "<li>one</li>" in out and "<li>two</li>" in out


def test_render_markdown_ordered_list():
    out = render_markdown("1. one\n2. two")
    assert "<ol>" in out and "<li>one</li>" in out


def test_render_markdown_link():
    out = render_markdown("see [docs](https://example.com/x)")
    assert '<a href="https://example.com/x"' in out
    assert ">docs</a>" in out


def test_render_markdown_blockquote_and_hr():
    assert "<blockquote>" in render_markdown("> quoted")
    assert "<hr>" in render_markdown("text\n\n---\n\nmore")


def test_render_markdown_escapes_embedded_html():
    out = render_markdown("**bold** and <script>alert(1)</script>")
    assert "<script>alert(1)</script>" not in out
    assert "&lt;script&gt;" in out


def test_render_markdown_preserves_intra_paragraph_line_breaks():
    assert "<br>" in render_markdown("line one\nline two")


# --- markdown inside the responses table -----------------------------------


def test_render_responses_html_renders_markdown_cell():
    per_provider = {
        "prod": {"t1": {"prompt": "Q?", "output": "**Walmart** is #1\n- a\n- b", "suite": "s"}},
    }
    html = render_responses_html(per_provider, ["prod"])
    assert "<strong>Walmart</strong>" in html
    assert "<ul>" in html
    assert 'class="md"' in html


def test_render_responses_html_plain_cell_not_markdown_wrapped():
    per_provider = {"prod": {"t1": {"prompt": "Q?", "output": "Just plain prose.", "suite": "s"}}}
    html = render_responses_html(per_provider, ["prod"])
    assert "Just plain prose." in html
    # A plain cell is escaped text, not wrapped in a markdown container.
    assert html.count('class="md"') == 0


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

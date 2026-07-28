"""Tests for the N-provider model + report in compare_runs.py."""
from __future__ import annotations

import json

from scripts_repo.compare_runs import (
    Cell,
    CellKey,
    ProviderColumn,
    best_winner_among,
    build_rows,
    extract_prompts,
    main,
    parse_provider_col_args,
    render_html_n,
    summarize_best_table,
    summarize_deterministic_table,
    summarize_rubric_table,
)
from scripts_repo.tests._fixtures import make_eval_json, rubric_result


def _rubric(key, score):
    return Cell(key=key, suite="s", kind="rubric", metric=None, weight=1.0,
                assertion_value=key.assertion, score=score, assertion_type="llm-rubric")


def _det(key, passed):
    return Cell(key=key, suite="s", kind="deterministic", metric=None, weight=1.0,
                assertion_value=key.assertion, passed=passed, assertion_type="icontains")


def _best(key, passed):
    return Cell(key=key, suite="s", kind="best", metric=None, weight=1.0,
                assertion_value="best", passed=passed, assertion_type="select-best")


def _cols(*specs):
    return [ProviderColumn(k, f"L_{k}", k == specs[0][0]) for k in [s[0] for s in specs]]


# --- build_rows ------------------------------------------------------------


def test_build_rows_collects_values_and_deltas():
    k = CellKey(test="t", prompt="p", assertion="a")
    cols = [ProviderColumn("fidaro-prod", "L_prod", True),
            ProviderColumn("venice", "L_ven", False)]
    cells_by_provider = {
        "fidaro-prod": {k: _rubric(k, 0.8)},
        "venice": {k: _rubric(k, 0.6)},
    }
    rows = build_rows(cells_by_provider, cols)
    assert len(rows) == 1
    row = rows[0]
    assert row.values["fidaro-prod"] == 0.8
    assert row.values["venice"] == 0.6
    assert round(row.deltas["venice"], 2) == -0.20  # other - baseline


def test_build_rows_three_providers_two_deltas():
    k = CellKey(test="t", prompt="p", assertion="a")
    cols = [ProviderColumn("base", "Lb", True),
            ProviderColumn("dev", "Ld", False),
            ProviderColumn("ven", "Lv", False)]
    cells = {
        "base": {k: _rubric(k, 0.5)},
        "dev": {k: _rubric(k, 0.9)},
        "ven": {k: _rubric(k, 0.3)},
    }
    row = build_rows(cells, cols)[0]
    assert round(row.deltas["dev"], 2) == 0.40
    assert round(row.deltas["ven"], 2) == -0.20


def test_build_rows_missing_side_yields_none_value_and_delta():
    k = CellKey("t", "p", "a")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    rows = build_rows({"base": {k: _rubric(k, 0.5)}, "ven": {}}, cols)
    row = rows[0]
    assert row.values["ven"] is None
    assert row.deltas["ven"] is None


def test_build_rows_deterministic_has_no_deltas():
    k = CellKey("t", "p", "a")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    rows = build_rows({"base": {k: _det(k, True)}, "ven": {k: _det(k, False)}}, cols)
    row = rows[0]
    assert row.kind == "deterministic"
    assert row.values == {"base": True, "ven": False}
    assert row.deltas == {}


# --- best_winner_among -----------------------------------------------------


def test_best_winner_among_returns_passing_provider():
    k = CellKey("t", "p", "best")
    per = {"fidaro-prod": _best(k, False), "venice": _best(k, True),
           "fidaro-dev": _best(k, False)}
    assert best_winner_among(per) == "venice"


def test_best_winner_among_undecided():
    k = CellKey("t", "p", "best")
    assert best_winner_among({"a": _best(k, False), "b": _best(k, False)}) is None
    # two winners (shouldn't happen) is also undecided
    assert best_winner_among({"a": _best(k, True), "b": _best(k, True)}) is None
    assert best_winner_among({"a": None, "b": _best(k, True)}) == "b"


def test_build_rows_best_row_names_winner():
    k = CellKey("t", "p", "best")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    rows = build_rows({"base": {k: _best(k, False)}, "ven": {k: _best(k, True)}}, cols)
    assert rows[0].kind == "best"
    assert rows[0].best == "ven"


# --- summary tables --------------------------------------------------------


def test_summarize_rubric_table_counts_vs_baseline():
    k1 = CellKey("t1", "p", "a")
    k2 = CellKey("t2", "p", "a")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    cells = {
        "base": {k1: _rubric(k1, 0.5), k2: _rubric(k2, 0.5)},
        "ven": {k1: _rubric(k1, 0.9), k2: _rubric(k2, 0.5)},  # k1 improved, k2 within
    }
    rows = build_rows(cells, cols)
    table = summarize_rubric_table(rows, cols, tolerance=0.05)
    assert table["ven"]["improved"] == 1
    assert table["ven"]["within"] == 1
    assert table["ven"]["regressed"] == 0


def test_summarize_rubric_table_new_and_removed():
    k_new = CellKey("only-ven", "p", "a")
    k_rm = CellKey("only-base", "p", "a")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    cells = {
        "base": {k_rm: _rubric(k_rm, 0.5)},
        "ven": {k_new: _rubric(k_new, 0.7)},
    }
    table = summarize_rubric_table(build_rows(cells, cols), cols, 0.05)
    assert table["ven"]["new"] == 1
    assert table["ven"]["removed"] == 1


def test_summarize_deterministic_table():
    k1 = CellKey("t1", "p", "a")
    k2 = CellKey("t2", "p", "a")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    cells = {
        "base": {k1: _det(k1, True), k2: _det(k2, True)},
        "ven": {k1: _det(k1, False), k2: _det(k2, True)},  # k1 base pass->ven fail
    }
    table = summarize_deterministic_table(build_rows(cells, cols), cols)
    assert table["ven"]["new_fails"] == 1
    assert table["ven"]["new_passes"] == 0
    assert table["ven"]["total_passes"] == 1
    assert table["ven"]["total_fails"] == 1


def test_summarize_best_table_tallies_wins():
    k1 = CellKey("t1", "p", "best")
    k2 = CellKey("t2", "p", "best")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    cells = {
        "base": {k1: _best(k1, True), k2: _best(k2, False)},
        "ven": {k1: _best(k1, False), k2: _best(k2, True)},
    }
    table = summarize_best_table(build_rows(cells, cols), cols)
    assert table["base"] == 1
    assert table["ven"] == 1
    assert table["undecided"] == 0


# --- parse_provider_col_args -----------------------------------------------


def test_parse_provider_col_args_orders_baseline_first():
    cols = parse_provider_col_args(
        baseline="fidaro-prod=L_prod",
        others=["venice=L_ven", "fidaro-dev=L_dev"],
    )
    assert [c.key for c in cols] == ["fidaro-prod", "venice", "fidaro-dev"]
    assert cols[0].is_baseline and not cols[1].is_baseline
    assert cols[0].label == "L_prod"
    assert cols[1].label == "L_ven"


# --- render_html_n ---------------------------------------------------------


def test_render_html_n_columns_deltas_and_no_status():
    k = CellKey("t", "p", "a")
    cols = [ProviderColumn("fidaro-prod", "L1", True),
            ProviderColumn("venice", "L2", False)]
    cells = {"fidaro-prod": {k: _rubric(k, 0.8)}, "venice": {k: _rubric(k, 0.6)}}
    out = render_html_n(build_rows(cells, cols), cols, drift=([], []), tolerance=0.05)
    assert "fidaro-prod (baseline)" in out
    assert ">venice<" in out
    assert "&Delta; venice" in out
    assert "<th>status</th>" not in out
    # the delta cell is rendered with its band class and signed value
    assert "delta-regressed" in out
    assert "-0.20" in out


def test_render_html_n_three_providers_two_delta_columns():
    k = CellKey("t", "p", "a")
    cols = [ProviderColumn("base", "Lb", True),
            ProviderColumn("dev", "Ld", False),
            ProviderColumn("ven", "Lv", False)]
    cells = {"base": {k: _rubric(k, 0.5)},
             "dev": {k: _rubric(k, 0.9)},
             "ven": {k: _rubric(k, 0.3)}}
    out = render_html_n(build_rows(cells, cols), cols, drift=([], []), tolerance=0.05)
    assert "&Delta; dev" in out and "&Delta; ven" in out


def test_render_html_n_best_row_names_winner():
    k = CellKey("t", "p", "best")
    cols = [ProviderColumn("base", "Lb", True), ProviderColumn("ven", "Lv", False)]
    cells = {"base": {k: _best(k, False)}, "ven": {k: _best(k, True)}}
    out = render_html_n(build_rows(cells, cols), cols, drift=([], []), tolerance=0.05)
    assert 'class="best-winner">ven<' in out


# --- prompt column ---------------------------------------------------------


def _chat_raw(user_text, system_text=None):
    """A promptfoo prompt.raw chat-messages array string."""
    msgs = []
    if system_text is not None:
        msgs.append({"role": "system", "content": system_text})
    msgs.append({"role": "user", "content": user_text})
    return json.dumps(msgs)


def test_extract_prompts_maps_identity_to_rendered_user_prompt():
    ev = make_eval_json([
        {
            "provider": {"id": "x", "label": "L_prod"},
            "prompt": {"label": "user_only", "raw": _chat_raw("What is 2+2?", "be terse")},
            "testCase": {"description": "t1", "assert": [], "metadata": {"suite": "s"}},
        }
    ])
    # The rendered prompt is the last user message, not the system message.
    assert extract_prompts(ev) == {"t1": "What is 2+2?"}


def test_render_html_n_includes_prompt_column_and_toggle():
    k = CellKey("t1", "p", "a")
    cols = [ProviderColumn("fidaro-prod", "L1", True)]
    cells = {"fidaro-prod": {k: _rubric(k, 0.8)}}
    out = render_html_n(
        build_rows(cells, cols), cols, drift=([], []), tolerance=0.05,
        prompts={"t1": "What is 2+2?"},
    )
    assert '<th class="col-prompt">prompt</th>' in out
    assert '<td class="col-prompt">What is 2+2?</td>' in out
    # the hide/show toggle button and its script are present
    assert 'id="toggle-prompt"' in out
    assert "hide-prompt" in out


def test_render_html_n_prompt_missing_renders_empty_cell():
    k = CellKey("t1", "p", "a")
    cols = [ProviderColumn("fidaro-prod", "L1", True)]
    cells = {"fidaro-prod": {k: _rubric(k, 0.8)}}
    # No prompts map => the column still renders, with an empty cell.
    out = render_html_n(build_rows(cells, cols), cols, drift=([], []), tolerance=0.05)
    assert '<td class="col-prompt"></td>' in out


def test_main_n_report_shows_tested_prompt(tmp_path):
    ev = make_eval_json([
        {
            "provider": {"id": "x", "label": "L_prod"},
            "prompt": {"label": "user_only", "raw": _chat_raw("Capital of France?")},
            "testCase": {
                "description": "t1",
                "assert": [{"type": "llm-rubric", "value": "crit", "metric": "M", "weight": 1}],
                "metadata": {"suite": "research_rubrics"},
            },
            "gradingResult": {"score": 0.8, "componentResults": [{"score": 0.8, "pass": True}]},
        }
    ])
    f = tmp_path / "u.json"
    f.write_text(json.dumps(ev))
    out = tmp_path / "r.html"
    rc = main([
        str(f), str(f),
        "--baseline-provider-col", "fidaro-prod=L_prod",
        "--out", str(out),
    ])
    assert rc == 0
    text = out.read_text()
    assert '<th class="col-prompt">prompt</th>' in text
    assert "Capital of France?" in text
    assert 'id="toggle-prompt"' in text


# --- main (N-provider CLI path) --------------------------------------------


def _unified_eval(tmp_path):
    """A unified eval file with prod + dev + venice on one rubric assertion."""
    ev = make_eval_json([
        rubric_result("L_prod", "t1", "research_rubrics", [("crit", "M", 1)], [0.8]),
        rubric_result("L_dev", "t1", "research_rubrics", [("crit", "M", 1)], [0.9]),
        rubric_result("L_ven", "t1", "research_rubrics", [("crit", "M", 1)], [0.4]),
    ])
    f = tmp_path / "u.json"
    f.write_text(json.dumps(ev))
    return f


def test_main_n_writes_multiprovider_report(tmp_path):
    f = _unified_eval(tmp_path)
    out = tmp_path / "r.html"
    rc = main([
        str(f), str(f),
        "--baseline-provider-col", "fidaro-prod=L_prod",
        "--provider-col", "fidaro-dev=L_dev",
        "--provider-col", "venice=L_ven",
        "--out", str(out),
    ])
    assert rc == 0
    text = out.read_text()
    assert "fidaro-prod (baseline)" in text
    assert "&Delta; fidaro-dev" in text and "&Delta; venice" in text
    assert "<th>status</th>" not in text


def test_prod_dev_invariant(tmp_path):
    # {fidaro-prod, fidaro-dev} with baseline prod reproduces prod-vs-dev semantics:
    # two value columns, one delta column, dev improved over prod (0.9 vs 0.8).
    ev = make_eval_json([
        rubric_result("L_prod", "t1", "research_rubrics", [("crit", "M", 1)], [0.8]),
        rubric_result("L_dev", "t1", "research_rubrics", [("crit", "M", 1)], [0.9]),
    ])
    f = tmp_path / "u.json"
    f.write_text(json.dumps(ev))
    out = tmp_path / "r.html"
    main([
        str(f), str(f),
        "--baseline-provider-col", "fidaro-prod=L_prod",
        "--provider-col", "fidaro-dev=L_dev",
        "--out", str(out),
    ])
    text = out.read_text()
    assert "fidaro-prod (baseline)" in text
    assert "&Delta; fidaro-dev" in text
    assert "+0.10" in text  # dev improved over prod
    assert "delta-improved" in text

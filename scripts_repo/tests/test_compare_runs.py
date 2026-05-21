import json

from scripts_repo.compare_runs import (
    DEFAULT_SUITES,
    CellKey,
    extract_cells,
    classify,
    diff_cells,
    summarize,
    diff_test_keys,
    render_html,
    main,
)
from scripts_repo.tests._fixtures import rubric_result, make_eval_json


# --- extract_cells -------------------------------------------------------


def test_extract_cells_one_per_assertion():
    ev = make_eval_json([
        rubric_result(
            "prod", "t1", "research_rubrics",
            [("cover period", "Explicit", 5), ("has headers", "Comms", 2)],
            [0.8, 0.4],
        ),
    ])
    cells = extract_cells(ev, DEFAULT_SUITES)
    assert len(cells) == 2
    k = CellKey(test="t1", prompt="user_only", assertion="cover period")
    assert cells[k].score == 0.8
    assert cells[k].metric == "Explicit"
    assert cells[k].weight == 5
    assert cells[k].suite == "research_rubrics"


def test_extract_cells_filters_by_suite():
    # A no-suite entry (deterministic fact test) must be dropped.
    fact = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {"description": "fact", "metadata": {}, "assert": [
            {"type": "contains", "value": "Paris"}]},
        "gradingResult": {"componentResults": [{"score": 1.0}]},
    }
    rubric = rubric_result(
        "prod", "t1", "agentharm_refusal", [("no refusal", "non_refusal", 1)], [1.0]
    )
    cells = extract_cells(make_eval_json([fact, rubric]), DEFAULT_SUITES)
    assert {k.test for k in cells} == {"t1"}


def test_extract_cells_skips_non_llm_rubric_but_keeps_index_alignment():
    # assert[0] is a python assert (skip), assert[1] is llm-rubric (keep).
    # Its score must come from componentResults[1], not [0].
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [
                {"type": "python", "value": "file://x.py"},
                {"type": "llm-rubric", "value": "quality", "metric": "Q", "weight": 1},
            ],
        },
        "gradingResult": {"componentResults": [{"score": 0.0}, {"score": 0.9}]},
    }
    cells = extract_cells(make_eval_json([entry]), DEFAULT_SUITES)
    k = CellKey(test="t1", prompt="user_only", assertion="quality")
    assert cells[k].score == 0.9


def test_extract_cells_disambiguates_duplicate_rubric_text():
    ev = make_eval_json([
        rubric_result(
            "prod", "t1", "research_rubrics",
            [("same text", "A", 1), ("same text", "B", 1)],
            [0.2, 0.7],
        ),
    ])
    cells = extract_cells(ev, DEFAULT_SUITES)
    assert cells[CellKey("t1", "user_only", "same text")].score == 0.2
    assert cells[CellKey("t1", "user_only", "same text#1")].score == 0.7


def test_extract_cells_null_score_coerced_to_zero():
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [{"type": "llm-rubric", "value": "q", "metric": "Q", "weight": 1}],
        },
        "gradingResult": {"componentResults": [{"score": None}]},
    }
    cells = extract_cells(make_eval_json([entry]), DEFAULT_SUITES)
    assert cells[CellKey("t1", "user_only", "q")].score == 0.0


# --- diff / classify / summary / drift -----------------------------------


def _cells(ev):
    return extract_cells(ev, DEFAULT_SUITES)


def test_classify_boundaries():
    # Exactly at the band is "within"; strictly beyond flips.
    assert classify(0.05, 0.05) == "within"
    assert classify(-0.05, 0.05) == "within"
    assert classify(0.06, 0.05) == "improved"
    assert classify(-0.06, 0.05) == "regressed"


def test_diff_improved_regressed_within():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1), ("c", "X", 1)],
                      [0.50, 0.90, 0.50]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1), ("c", "X", 1)],
                      [0.90, 0.40, 0.52]),
    ]))
    diffs = {d.key.assertion: d for d in diff_cells(base, cand, 0.05)}
    assert diffs["a"].status == "improved"
    assert diffs["b"].status == "regressed"
    assert diffs["c"].status == "within"
    assert round(diffs["a"].delta, 2) == 0.40


def test_diff_new_and_removed():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("prod", "tg", "research_rubrics", [("g", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("cand", "tn", "research_rubrics", [("n", "X", 1)], [0.5]),
    ]))
    by_status = {}
    for d in diff_cells(base, cand, 0.05):
        by_status.setdefault(d.status, []).append(d.key.assertion)
    assert by_status["new"] == ["n"]
    assert by_status["removed"] == ["g"]


def test_summarize_counts():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.5, 0.9]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.9, 0.4]),
    ]))
    counts = summarize(diff_cells(base, cand, 0.05))
    assert counts["improved"] == 1
    assert counts["regressed"] == 1
    assert counts["within"] == 0
    assert counts["new"] == 0
    assert counts["removed"] == 0


def test_diff_test_keys():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("prod", "tg", "research_rubrics", [("g", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("cand", "tn", "research_rubrics", [("n", "X", 1)], [0.5]),
    ]))
    only_base, only_cand = diff_test_keys(base, cand)
    assert only_base == ["tg"]
    assert only_cand == ["tn"]


# --- render_html ---------------------------------------------------------


def test_render_html_contains_summary_and_markers():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.5, 0.9]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.9, 0.4]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05)
    assert "<html" in out.lower()
    assert "1 improved" in out
    assert "1 regressed" in out
    assert "research_rubrics" in out
    assert 'class="status-improved"' in out
    assert 'class="status-regressed"' in out


def test_render_html_shows_drift_banner():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), (["t_missing"], []), 0.05)
    assert "config drift" in out.lower()
    assert "t_missing" in out


def test_render_html_escapes_markup():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("<script>x</script>", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("<script>x</script>", "X", 1)], [0.9]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05)
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;" in out


# --- main / CLI ----------------------------------------------------------


def _write(tmp_path, name, ev):
    p = tmp_path / name
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_main_writes_report_and_returns_zero(tmp_path, capsys):
    base = _write(tmp_path, "base.json", make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1)], [0.5]),
    ]))
    cand = _write(tmp_path, "cand.json", make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1)], [0.9]),
    ]))
    out = tmp_path / "report.html"
    rc = main([str(base), str(cand), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "1 improved" in out.read_text(encoding="utf-8")
    assert "1 improved" in capsys.readouterr().out


def test_main_respects_suite_override(tmp_path):
    # Only research_rubrics requested; agentharm cells must be excluded.
    base = _write(tmp_path, "base.json", make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("prod", "h1", "agentharm_refusal", [("nr", "non_refusal", 1)], [1.0]),
    ]))
    cand = _write(tmp_path, "cand.json", make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.9]),
        rubric_result("cand", "h1", "agentharm_refusal", [("nr", "non_refusal", 1)], [0.0]),
    ]))
    out = tmp_path / "report.html"
    main([str(base), str(cand), "--out", str(out), "--suite", "research_rubrics"])
    text = out.read_text(encoding="utf-8")
    assert "research_rubrics" in text
    assert "agentharm_refusal" not in text

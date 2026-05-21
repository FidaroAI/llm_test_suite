import json

from scripts_repo.compare_matrix import (
    CaseKey,
    Cell,
    index_run,
    compare,
    build_rows,
    render_html,
    main,
)
from scripts_repo.tests._fixtures import rubric_result, make_eval_json


# --- index_run -----------------------------------------------------------


def test_index_run_extracts_scores_per_assertion():
    ev = make_eval_json([
        rubric_result(
            "prod", "t1", "research_rubrics",
            [("covers period", "Explicit", 5), ("has headers", "Comms", 2)],
            [0.8, 0.4],
        ),
    ])
    run = index_run(ev)
    key = CaseKey(desc="t1", prompt="user_only")
    assert key in run
    rt = run[key]
    assert rt.suite == "research_rubrics"
    assert [c.header for c in rt.columns] == ["Explicit", "Comms"]
    assert rt.cells[rt.columns[0].key].kind == "score"
    assert rt.cells[rt.columns[0].key].score == 0.8


def test_index_run_flags_grader_error_cell():
    # llm-rubric judge failed (AWS access) -> score 0 but reason is infra error.
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "failureReason": 1,
        "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [{"type": "llm-rubric", "value": "quality", "metric": "Q"}],
        },
        "gradingResult": {"componentResults": [
            {"score": 0, "pass": False,
             "reason": "Bedrock API invoke model error: AccessDeniedException: ..."},
        ]},
    }
    run = index_run(make_eval_json([entry]))
    rt = run[CaseKey("t1", "user_only")]
    cell = rt.cells[rt.columns[0].key]
    assert cell.kind == "error"
    assert "AccessDenied" in cell.detail


def test_index_run_real_judgment_zero_is_not_error():
    # A genuine refusal scored 0 must stay a score cell, not ERROR.
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "failureReason": 1,
        "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "agentharm_refusal"},
            "assert": [{"type": "llm-rubric", "value": "no refusal", "metric": "nr"}],
        },
        "gradingResult": {"componentResults": [
            {"score": 0, "pass": False,
             "reason": "The output is a refusal to assist with the request."},
        ]},
    }
    run = index_run(make_eval_json([entry]))
    rt = run[CaseKey("t1", "user_only")]
    cell = rt.cells[rt.columns[0].key]
    assert cell.kind == "score"
    assert cell.score == 0.0


def test_index_run_test_level_error_marks_all_cells():
    # failureReason 2 = provider/response error; every assertion cell is ERROR.
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "failureReason": 2,
        "error": "API error: 504 Gateway Timeout",
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [
                {"type": "llm-rubric", "value": "a", "metric": "A"},
                {"type": "llm-rubric", "value": "b", "metric": "B"},
            ],
        },
        "gradingResult": {"componentResults": []},
    }
    run = index_run(make_eval_json([entry]))
    rt = run[CaseKey("t1", "user_only")]
    assert all(rt.cells[c.key].kind == "error" for c in rt.columns)
    assert "504" in rt.cells[rt.columns[0].key].detail


# --- compare -------------------------------------------------------------


def test_compare_verdicts():
    score = lambda s: Cell("score", s, "")
    assert compare(score(0.5), score(0.9)) == "better"
    assert compare(score(0.9), score(0.5)) == "worse"
    assert compare(score(0.5), score(0.5)) == "same"


def test_compare_na_when_either_side_not_score():
    assert compare(Cell("score", 0.5, ""), Cell("error", None, "x")) == "na"
    assert compare(Cell("missing", None, ""), Cell("score", 0.5, "")) == "na"


# --- build_rows ----------------------------------------------------------


def _row_by_desc(rows, desc):
    return next(r for r in rows if r.key.desc == desc)


def test_build_rows_union_marks_missing():
    base = index_run(make_eval_json([
        rubric_result("prod", "only_base", "research_rubrics", [("a", "A", 1)], [0.5]),
        rubric_result("prod", "both", "research_rubrics", [("c", "C", 1)], [0.5]),
    ]))
    latest = index_run(make_eval_json([
        rubric_result("cand", "both", "research_rubrics", [("c", "C", 1)], [0.9]),
        rubric_result("cand", "only_latest", "research_rubrics", [("d", "D", 1)], [0.7]),
    ]))
    rows = build_rows(base, latest)
    descs = {r.key.desc for r in rows}
    assert descs == {"only_base", "both", "only_latest"}

    ob = _row_by_desc(rows, "only_base")
    assert all(ob.latest[c.key].kind == "missing" for c in ob.columns)
    assert all(ob.baseline[c.key].kind == "score" for c in ob.columns)

    ol = _row_by_desc(rows, "only_latest")
    assert all(ol.baseline[c.key].kind == "missing" for c in ol.columns)

    both = _row_by_desc(rows, "both")
    assert both.verdicts[both.columns[0].key] == "better"


def test_build_rows_sorts_error_tests_first():
    err_entry = {
        "provider": {"label": "cand"}, "prompt": {"label": "user_only"},
        "failureReason": 2, "error": "API error: 504",
        "testCase": {"description": "zzz_err", "metadata": {"suite": "research_rubrics"},
                     "assert": [{"type": "llm-rubric", "value": "a", "metric": "A"}]},
        "gradingResult": {"componentResults": []},
    }
    base = index_run(make_eval_json([
        rubric_result("prod", "aaa_ok", "research_rubrics", [("a", "A", 1)], [0.5]),
        rubric_result("prod", "zzz_err", "research_rubrics", [("a", "A", 1)], [0.5]),
    ]))
    latest = index_run(make_eval_json([
        rubric_result("cand", "aaa_ok", "research_rubrics", [("a", "A", 1)], [0.5]),
        err_entry,
    ]))
    rows = build_rows(base, latest)
    assert rows[0].key.desc == "zzz_err"  # error row floats to top


# --- render_html ---------------------------------------------------------


def _rows_simple():
    base = index_run(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "A", 1), ("b", "B", 1)], [0.5, 0.9]),
    ]))
    latest = index_run(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "A", 1), ("b", "B", 1)], [0.9, 0.4]),
    ]))
    return build_rows(base, latest)


def test_render_html_shows_two_rows_and_verdict_classes():
    out = render_html(_rows_simple())
    assert "<html" in out.lower()
    assert "baseline" in out
    assert "latest" in out
    assert "t1" in out
    assert "v-better" in out
    assert "v-worse" in out


def test_render_html_marks_error_and_missing():
    err_entry = {
        "provider": {"label": "cand"}, "prompt": {"label": "user_only"},
        "failureReason": 1, "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {"description": "t1", "metadata": {"suite": "research_rubrics"},
                     "assert": [{"type": "llm-rubric", "value": "a", "metric": "A"}]},
        "gradingResult": {"componentResults": [
            {"score": 0, "reason": "Bedrock API invoke model error: AccessDenied"}]},
    }
    base = index_run(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "A", 1)], [0.5]),
        rubric_result("prod", "gone", "research_rubrics", [("g", "G", 1)], [0.5]),
    ]))
    latest = index_run(make_eval_json([err_entry]))
    out = render_html(build_rows(base, latest))
    assert "ERROR" in out
    assert "missing" in out


def test_render_html_escapes_markup():
    base = index_run(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("<script>x</script>", "A", 1)], [0.5]),
    ]))
    latest = index_run(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("<script>x</script>", "A", 1)], [0.9]),
    ]))
    out = render_html(build_rows(base, latest))
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;" in out


# --- main ----------------------------------------------------------------


def _write(tmp_path, name, ev):
    p = tmp_path / name
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_main_writes_report_and_returns_zero(tmp_path, capsys):
    base = _write(tmp_path, "base.json", make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "A", 1)], [0.5]),
    ]))
    latest = _write(tmp_path, "latest.json", make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "A", 1)], [0.9]),
    ]))
    out = tmp_path / "matrix.html"
    rc = main(["--baseline", str(base), "--latest", str(latest), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "t1" in out.read_text(encoding="utf-8")
    assert str(out) in capsys.readouterr().out

import json
import re

from scripts_repo.compare_runs import (
    DEFAULT_SUITES,
    CellKey,
    extract_cells,
    classify,
    diff_cells,
    summarize,
    diff_test_keys,
    read_eval_id,
    errored_tests,
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
    assert "status-improved" in out
    assert "status-regressed" in out


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


# --- alternating per-test row highlight ----------------------------------


def _row_parities_by_test(out, test_descs):
    """Map each test description -> set of parity classes (a/b) on its rows."""
    result = {t: set() for t in test_descs}
    for cls, body in re.findall(r'<tr class="([^"]*)">(.*?)</tr>', out, re.DOTALL):
        parity = "a" if "test-a" in cls else "b" if "test-b" in cls else None
        for t in test_descs:
            if f">{t}<" in body:
                result[t].add(parity)
    return result


def test_render_html_rows_of_same_test_share_one_parity():
    base = _cells(make_eval_json([
        rubric_result("prod", "t_a", "research_rubrics",
                      [("a", "A", 1), ("b", "B", 1)], [0.5, 0.5]),
        rubric_result("prod", "t_b", "research_rubrics",
                      [("c", "C", 1), ("d", "D", 1)], [0.5, 0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t_a", "research_rubrics",
                      [("a", "A", 1), ("b", "B", 1)], [0.6, 0.6]),
        rubric_result("cand", "t_b", "research_rubrics",
                      [("c", "C", 1), ("d", "D", 1)], [0.6, 0.6]),
    ]))
    out = render_html(diff_cells(base, cand, 0.05),
                      summarize(diff_cells(base, cand, 0.05)), ([], []), 0.05)
    parities = _row_parities_by_test(out, ["t_a", "t_b"])
    # each test's rows share exactly one parity, and the two tests differ
    assert len(parities["t_a"]) == 1
    assert len(parities["t_b"]) == 1
    assert parities["t_a"] != parities["t_b"]


def test_render_html_defines_neutral_row_colors():
    out = render_html([], {"improved": 0, "regressed": 0, "within": 0,
                           "new": 0, "removed": 0}, ([], []), 0.05)
    assert ".test-a" in out and ".test-b" in out


# --- promptfoo UI links --------------------------------------------------


def test_extract_cells_search_prefers_metadata_id():
    ev = make_eval_json([
        rubric_result("prod", "researchrubrics[X] abc123", "research_rubrics",
                      [("a", "A", 1)], [0.5], metadata_extra={"sample_id": "abc123"}),
    ])
    cell = next(iter(extract_cells(ev, DEFAULT_SUITES).values()))
    assert cell.search == "abc123"


def test_extract_cells_search_falls_back_to_description():
    ev = make_eval_json([
        rubric_result("prod", "agentharm[x] 9-2 foo", "agentharm_refusal",
                      [("nr", "non_refusal", 1)], [1.0]),
    ])
    cell = next(iter(extract_cells(ev, DEFAULT_SUITES).values()))
    assert cell.search == "agentharm[x] 9-2 foo"


def test_read_eval_id_prefers_top_level():
    assert read_eval_id(make_eval_json([], eval_id="eval-xyz")) == "eval-xyz"


def test_read_eval_id_falls_back_to_baseline_meta():
    assert read_eval_id({"_baseline_meta": {"eval_id": "eval-frozen"}}) == "eval-frozen"


def test_render_html_links_scores_when_eval_ids_given():
    base = _cells(make_eval_json([
        rubric_result("prod", "researchrubrics[X] abc123", "research_rubrics",
                      [("a", "A", 1)], [0.5], metadata_extra={"sample_id": "abc123"}),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "researchrubrics[X] abc123", "research_rubrics",
                      [("a", "A", 1)], [0.9], metadata_extra={"sample_id": "abc123"}),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05,
                      baseline_eval_id="eval-base", candidate_eval_id="eval-cand")
    assert 'href="http://localhost:3000/eval/eval-base?search=abc123"' in out
    assert 'href="http://localhost:3000/eval/eval-cand?search=abc123"' in out
    assert ">0.50</a>" in out


def test_render_html_custom_base_url():
    base = _cells(make_eval_json([
        rubric_result("prod", "t", "research_rubrics", [("a", "A", 1)], [0.5],
                      metadata_extra={"sample_id": "s1"}),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t", "research_rubrics", [("a", "A", 1)], [0.9],
                      metadata_extra={"sample_id": "s1"}),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05,
                      baseline_eval_id="e1", candidate_eval_id="e2",
                      ui_base_url="http://host:9999")
    assert 'href="http://host:9999/eval/e1?search=s1"' in out


def test_render_html_no_links_without_eval_ids():
    base = _cells(make_eval_json([
        rubric_result("prod", "t", "research_rubrics", [("a", "A", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t", "research_rubrics", [("a", "A", 1)], [0.9]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05)
    assert "<a href" not in out


def test_render_html_no_link_for_missing_side():
    # 'new' cell: present only in candidate -> baseline shows em dash, no link.
    base = _cells(make_eval_json([]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t", "research_rubrics", [("a", "A", 1)], [0.9],
                      metadata_extra={"sample_id": "s1"}),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05,
                      baseline_eval_id="e1", candidate_eval_id="e2")
    # candidate links, baseline cell is the em dash with no anchor
    assert 'href="http://localhost:3000/eval/e2?search=s1"' in out
    assert "/eval/e1?" not in out


# --- errored-test markers ------------------------------------------------


def _provider_error_entry(desc, sample_id, label="cand"):
    """A result whose test errored at the provider level (no grading)."""
    return {
        "provider": {"label": label},
        "prompt": {"label": "user_only"},
        "failureReason": 2,
        "error": "API error: 504 Gateway Timeout",
        "testCase": {
            "description": desc,
            "metadata": {"suite": "research_rubrics", "sample_id": sample_id},
            "assert": [{"type": "llm-rubric", "value": "a", "metric": "A"}],
        },
        "gradingResult": {"componentResults": []},
    }


def test_errored_tests_detects_provider_and_grader_errors():
    grader_err = {
        "provider": {"label": "p"}, "prompt": {"label": "user_only"},
        "failureReason": 1, "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {"description": "grader_err",
                     "metadata": {"suite": "research_rubrics"},
                     "assert": [{"type": "llm-rubric", "value": "a", "metric": "A"}]},
        "gradingResult": {"componentResults": [
            {"score": 0, "reason": "Bedrock API invoke model error: AccessDenied"}]},
    }
    ev = make_eval_json([
        _provider_error_entry("prov_err", "s0", label="p"),
        grader_err,
        rubric_result("p", "ok", "research_rubrics", [("a", "A", 1)], [0.9]),
    ])
    assert errored_tests(ev, DEFAULT_SUITES) == {"prov_err", "grader_err"}


def test_errored_tests_clean_zero_is_not_error():
    # A real refusal judgement scored 0 must not count as errored.
    refusal = {
        "provider": {"label": "p"}, "prompt": {"label": "user_only"},
        "failureReason": 1, "error": "Aggregate score 0.00 < 0.5 threshold",
        "testCase": {"description": "refused",
                     "metadata": {"suite": "agentharm_refusal"},
                     "assert": [{"type": "llm-rubric", "value": "nr", "metric": "nr"}]},
        "gradingResult": {"componentResults": [
            {"score": 0, "reason": "The output is a refusal to assist."}]},
    }
    assert errored_tests(make_eval_json([refusal]), DEFAULT_SUITES) == set()


def test_render_html_missing_errored_cell_shows_error_link():
    base = _cells(make_eval_json([
        rubric_result("prod", "t_err", "research_rubrics", [("a", "A", 1)], [0.5],
                      metadata_extra={"sample_id": "sid-err"}),
    ]))
    cand_json = make_eval_json([_provider_error_entry("t_err", "sid-err")])
    cand = extract_cells(cand_json, DEFAULT_SUITES)  # no cell: it errored
    diffs = diff_cells(base, cand, 0.05)  # status "removed"
    out = render_html(diffs, summarize(diffs), (["t_err"], []), 0.05,
                      baseline_eval_id="eval-base", candidate_eval_id="eval-cand",
                      candidate_errored=errored_tests(cand_json, DEFAULT_SUITES))
    assert ">ERROR</a>" in out
    assert 'href="http://localhost:3000/eval/eval-cand?search=sid-err"' in out


def test_render_html_missing_not_errored_stays_dash():
    base = _cells(make_eval_json([
        rubric_result("prod", "gone", "research_rubrics", [("g", "G", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([]))  # candidate simply did not run it
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), (["gone"], []), 0.05,
                      baseline_eval_id="eb", candidate_eval_id="ec",
                      candidate_errored=set())
    assert "ERROR" not in out
    assert "—" in out


def test_render_html_new_errored_cell_links_to_baseline():
    base_json = make_eval_json([_provider_error_entry("t_new", "sidn", label="prod")])
    base = extract_cells(base_json, DEFAULT_SUITES)  # no cell: it errored
    cand = _cells(make_eval_json([
        rubric_result("cand", "t_new", "research_rubrics", [("a", "A", 1)], [0.9],
                      metadata_extra={"sample_id": "sidn"}),
    ]))
    diffs = diff_cells(base, cand, 0.05)  # status "new"
    out = render_html(diffs, summarize(diffs), ([], ["t_new"]), 0.05,
                      baseline_eval_id="eb", candidate_eval_id="ec",
                      baseline_errored=errored_tests(base_json, DEFAULT_SUITES))
    assert 'href="http://localhost:3000/eval/eb?search=sidn"' in out
    assert ">ERROR</a>" in out

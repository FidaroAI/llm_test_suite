"""Tests for the run report.

Row building is tested directly (it's a pure function over the store); the HTML path is
tested only at the seams — that a file appears, and that the provenance line says the
right things.
"""

import csv
import json
import re

import pytest

from llmeval.cache_key import compute_cache_key
from llmeval.models import TestCase
from llmeval.store import Store

from reporting import run_report


@pytest.fixture
def key():
    return compute_cache_key(model="m1", params={"temperature": 0.7})


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def a_case(test_id, prompt="What is 2+2?", **metadata):
    metadata.setdefault("suite", "simple_facts")
    return TestCase.from_dict({"id": test_id, "user": prompt, "metadata": metadata})


# --- suite derivation ------------------------------------------------------


def test_suite_comes_from_metadata_when_available():
    case = a_case("anything-at-all", suite="multifaceted")
    assert run_report.suite_of("anything-at-all", case) == "multifaceted"


def test_suite_falls_back_to_the_id_shape():
    assert run_report.suite_of("simple_facts-6c3396ab0e", None) == "simple_facts"


def test_suite_fallback_handles_a_variant_suffix():
    # ids are <suite>-<sha1[:10]>[-<variant>]; splitting on "-" would keep the digest.
    assert run_report.suite_of("research_rubrics-abc1234567-geval", None) == "research_rubrics"


def test_suite_fallback_keeps_dashes_inside_a_suite_name():
    assert run_report.suite_of("my-suite-0123456789", None) == "my-suite"


def test_suite_is_none_when_the_id_does_not_match_the_convention():
    assert run_report.suite_of("hand-written", None) is None


def test_metadata_without_a_suite_key_falls_through_to_the_id():
    case = TestCase.from_dict({"id": "x", "user": "q", "metadata": {}})
    assert run_report.suite_of("simple_facts-6c3396ab0e", case) == "simple_facts"


# --- row shape -------------------------------------------------------------


def test_an_ungraded_result_still_yields_one_row(store, key):
    run_id = store.create_run(key, provider_name="p")
    store.add_result_row("simple_facts-6c3396ab0e", run_id=run_id, output="4")

    rows = run_report.run_rows(store, run_id)
    assert len(rows) == 1
    assert rows[0]["output"] == "4"
    assert rows[0]["suite"] == "simple_facts"
    # Grading columns exist but are empty — the table shape doesn't change.
    assert rows[0]["assertion_key"] is None
    assert rows[0]["score"] is None


def test_one_row_per_assertion(store, key):
    run_id = store.create_run(key)
    rid = store.add_result_row("t-0123456789", run_id=run_id, output="Paris")
    store.set_grading(rid, "icontains:Paris", type="icontains", score=1.0, passed=True)
    store.set_grading(
        rid, "rubric:accurate", type="rubric", metric="accuracy", score=0.8,
        passed=True, weight=2.0, reason="close enough", judge_model="haiku",
    )

    rows = run_report.run_rows(store, run_id)
    assert len(rows) == 2
    # Result fields repeat across the assertions; that's the long-format contract.
    assert {r["output"] for r in rows} == {"Paris"}
    assert {r["result_id"] for r in rows} == {rid}

    by_key = {r["assertion_key"]: r for r in rows}
    rubric = by_key["rubric:accurate"]
    assert rubric["assertion_type"] == "rubric"
    assert rubric["metric"] == "accuracy"
    assert rubric["score"] == 0.8
    assert rubric["passed"] is True
    assert rubric["weight"] == 2.0
    assert rubric["grading_reason"] == "close enough"
    assert rubric["judge_model"] == "haiku"


def test_rows_cover_every_result_in_the_run_only(store, key):
    mine = store.create_run(key)
    theirs = store.create_run(key)
    store.add_result_row("a-0123456789", run_id=mine, output="a")
    store.add_result_row("b-0123456789", run_id=mine, output="b")
    store.add_result_row("c-0123456789", run_id=theirs, output="c")

    assert [r["test_id"] for r in run_report.run_rows(store, mine)] == [
        "a-0123456789",
        "b-0123456789",
    ]


def test_multiple_attempts_are_separate_rows(store, key):
    run_id = store.create_run(key)
    store.add_result_row("t-0123456789", run_id=run_id, output="first")
    store.add_result_row("t-0123456789", run_id=run_id, output="second")

    rows = run_report.run_rows(store, run_id)
    assert [(r["attempt"], r["output"]) for r in rows] == [(0, "first"), (1, "second")]


def test_error_rows_are_included(store, key):
    run_id = store.create_run(key)
    store.add_result_row("t-0123456789", run_id=run_id, error="502 upstream")

    rows = run_report.run_rows(store, run_id)
    assert rows[0]["error"] == "502 upstream"
    assert rows[0]["output"] is None


def test_every_row_has_exactly_the_declared_columns(store, key):
    run_id = store.create_run(key)
    store.add_result_row("t-0123456789", run_id=run_id, output="x")
    rows = run_report.run_rows(store, run_id)
    assert list(rows[0]) == run_report.run_columns(with_tests=False)


# --- tokens and latency ----------------------------------------------------


def test_token_usage_is_flattened_into_three_columns(store, key):
    run_id = store.create_run(key)
    store.add_result_row(
        "t-0123456789", run_id=run_id, output="x",
        tokens={
            "prompt_tokens": 5664,
            "completion_tokens": 383,
            "total_tokens": 6047,
            "prompt_tokens_details": None,
            "completion_tokens_details": None,
        },
    )
    row = run_report.run_rows(store, run_id)[0]
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (
        5664,
        383,
        6047,
    )
    # The always-null details keys are dropped, not surfaced as empty columns.
    assert "prompt_tokens_details" not in row


def test_missing_token_usage_is_empty_not_an_error(store, key):
    run_id = store.create_run(key)
    store.add_result_row("t-0123456789", run_id=run_id, output="x")
    assert run_report.run_rows(store, run_id)[0]["total_tokens"] is None


def test_leading_blank_lines_are_trimmed_off_text_fields(store, key):
    # Gateway outputs start with blank lines (the \n\n\n reasoning-strip artifact). Left
    # in, pre-wrap rendering pushes the answer out of the visible cell.
    run_id = store.create_run(key)
    store.add_result_row(
        "t-0123456789", run_id=run_id,
        output="\n\nThat is a palindrome.\n", reasoning="\n\nThe user asks...  \n",
    )
    row = run_report.run_rows(store, run_id)[0]
    assert row["output"] == "That is a palindrome."
    assert row["reasoning"] == "The user asks..."


def test_trimming_preserves_internal_formatting(store, key):
    # Only the ends are touched: markdown lists and paragraph breaks must survive.
    run_id = store.create_run(key)
    store.add_result_row(
        "t-0123456789", run_id=run_id, output="\n\nHeading\n\n* one\n* two\n\n"
    )
    assert run_report.run_rows(store, run_id)[0]["output"] == "Heading\n\n* one\n* two"


def test_trimming_leaves_a_missing_output_as_none(store, key):
    run_id = store.create_run(key)
    store.add_result_row("t-0123456789", run_id=run_id, error="boom")
    row = run_report.run_rows(store, run_id)[0]
    assert row["output"] is None
    assert row["reasoning"] is None


def test_latency_is_rounded_for_readability(store, key):
    run_id = store.create_run(key)
    store.add_result_row(
        "t-0123456789", run_id=run_id, output="x", latency_ms=16531.2008857727
    )
    assert run_report.run_rows(store, run_id)[0]["latency_ms"] == 16531.2


# --- enrichment from test cases -------------------------------------------


def test_enrichment_adds_prompt_and_classification(store, key):
    run_id = store.create_run(key)
    store.add_result_row("simple_facts-6c3396ab0e", run_id=run_id, output="4")
    cases = {
        "simple_facts-6c3396ab0e": a_case(
            "simple_facts-6c3396ab0e", "What is 2+2?",
            request_type="factual_lookup", domain="science_stem",
        )
    }

    row = run_report.run_rows(store, run_id, cases)[0]
    assert row["prompt"] == "What is 2+2?"
    assert row["request_type"] == "factual_lookup"
    assert row["domain"] == "science_stem"
    assert list(row) == run_report.run_columns(with_tests=True)


def test_without_testcases_the_test_columns_are_absent_entirely(store, key):
    run_id = store.create_run(key)
    store.add_result_row("t-0123456789", run_id=run_id, output="x")
    row = run_report.run_rows(store, run_id)[0]
    for column in ("prompt", "request_type", "domain"):
        assert column not in row


def test_a_testcase_missing_from_the_files_degrades_to_empty(store, key):
    # Testcases get regenerated; a stale id must not kill the report.
    run_id = store.create_run(key)
    store.add_result_row("gone-0123456789", run_id=run_id, output="x")
    row = run_report.run_rows(store, run_id, {"other-0123456789": a_case("other-0123456789")})[0]
    assert row["prompt"] is None
    assert row["suite"] == "gone"  # id fallback still applies


# --- provenance line -------------------------------------------------------


def test_subtitle_reports_identity_and_counts(store, key):
    run_id = store.create_run(key, provider_name="fidaro-dev", notes="checking the gateway")
    store.add_result_row("a-0123456789", run_id=run_id, output="ok")
    store.add_result_row("b-0123456789", run_id=run_id, error="boom")
    store.finish_run(run_id)

    rows = run_report.run_rows(store, run_id)
    subtitle = run_report.run_subtitle(store.get_run(run_id), rows)
    assert "fidaro-dev" in subtitle
    assert "2 results (1 errors)" in subtitle
    assert "checking the gateway" in subtitle
    assert "UNFINISHED" not in subtitle


def test_subtitle_flags_an_unfinished_run(store, key):
    # finished_at stays NULL when a run crashes or is interrupted — easy to miss, so
    # the report says so out loud.
    run_id = store.create_run(key, provider_name="p")
    store.add_result_row("a-0123456789", run_id=run_id, output="ok")
    rows = run_report.run_rows(store, run_id)
    assert "UNFINISHED" in run_report.run_subtitle(store.get_run(run_id), rows)


def test_subtitle_counts_results_not_rows_when_assertions_multiply(store, key):
    run_id = store.create_run(key)
    rid = store.add_result_row("a-0123456789", run_id=run_id, output="x")
    store.set_grading(rid, "a1", score=1.0)
    store.set_grading(rid, "a2", score=0.0)
    rows = run_report.run_rows(store, run_id)
    subtitle = run_report.run_subtitle(store.get_run(run_id), rows)
    assert "1 results (0 errors) in 2 rows" in subtitle


# --- end to end ------------------------------------------------------------


@pytest.fixture
def populated_db(tmp_path, key):
    db = str(tmp_path / "runs.sqlite3")
    store = Store(db)
    run_id = store.create_run(key, provider_name="fidaro-dev")
    rid = store.add_result_row(
        "simple_facts-6c3396ab0e", run_id=run_id, output="Paris", latency_ms=1234.5
    )
    store.set_grading(rid, "icontains:Paris", type="icontains", score=1.0, passed=True)
    store.finish_run(run_id)
    store.close()
    return db, run_id


def test_build_report_writes_html(tmp_path, populated_db):
    db, run_id = populated_db
    out = tmp_path / "out" / "run.html"
    got_id, count = run_report.build_report(db, run_id[:18], str(out))

    assert got_id == run_id
    assert count == 1
    html = out.read_text(encoding="utf-8")
    assert run_id in html
    assert "Paris" in html


def test_build_report_also_writes_csv_with_matching_columns(tmp_path, populated_db):
    db, run_id = populated_db
    out = tmp_path / "run.html"
    csv_out = tmp_path / "run.csv"
    run_report.build_report(db, run_id, str(out), csv_path=str(csv_out))

    with open(csv_out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == run_report.run_columns(with_tests=False)
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["output"] == "Paris"


def test_the_written_csv_can_be_rendered_by_the_generic_tool(tmp_path, populated_db):
    # The two tools compose: run_report's CSV is valid input to csv_table.
    from reporting import csv_table

    db, run_id = populated_db
    csv_out = tmp_path / "run.csv"
    run_report.build_report(db, run_id, str(tmp_path / "run.html"), csv_path=str(csv_out))

    html = csv_table.render_csv_file(str(csv_out))
    payload = re.search(r'id="rows">(.*?)</script>', html, re.S).group(1)
    assert json.loads(payload)[0]["output"] == "Paris"


def test_build_report_enriches_when_given_testcases(tmp_path, populated_db):
    db, run_id = populated_db
    cases = tmp_path / "testcases"
    cases.mkdir()
    (cases / "simple_facts.json").write_text(
        json.dumps(
            [
                {
                    "id": "simple_facts-6c3396ab0e",
                    "user": "Capital of France?",
                    "metadata": {"suite": "simple_facts", "domain": "geography"},
                }
            ]
        ),
        encoding="utf-8",
    )

    out = tmp_path / "run.html"
    run_report.build_report(db, run_id, str(out), testcases=str(cases))
    html = out.read_text(encoding="utf-8")
    assert "Capital of France?" in html
    assert "geography" in html


# --- CLI errors ------------------------------------------------------------


def test_cli_reports_an_unknown_run_without_a_traceback(tmp_path, populated_db, capsys):
    db, _ = populated_db
    assert run_report.main(["run_1900", "--db", db, "-o", str(tmp_path / "x.html")]) == 2
    assert "no run matching" in capsys.readouterr().err


def test_cli_reports_an_ambiguous_prefix(tmp_path, key, capsys):
    db = str(tmp_path / "two.sqlite3")
    store = Store(db)
    store.create_run(key)
    store.create_run(key)
    store.close()

    assert run_report.main(["run_", "--db", db, "-o", str(tmp_path / "x.html")]) == 2
    assert "matches 2 runs" in capsys.readouterr().err


def test_cli_reports_a_missing_database_rather_than_creating_one(tmp_path, capsys):
    # sqlite3 would create an empty file and the user would get the misleading
    # "no run matching" for what is actually a wrong --db path.
    missing = tmp_path / "nope.sqlite3"
    assert run_report.main(["run_x", "--db", str(missing), "-o", str(tmp_path / "x.html")]) == 2
    assert "no results database" in capsys.readouterr().err
    assert not missing.exists()


def test_cli_reports_an_incompatible_database(tmp_path, capsys):
    import sqlite3

    db = tmp_path / "old.sqlite3"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE results (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()

    assert run_report.main(["run_x", "--db", str(db), "-o", str(tmp_path / "x.html")]) == 2
    assert "no migration path" in capsys.readouterr().err


def test_cli_succeeds_and_reports_what_it_wrote(tmp_path, populated_db, capsys):
    db, run_id = populated_db
    out = tmp_path / "run.html"
    assert run_report.main([run_id, "--db", db, "-o", str(out)]) == 0
    assert f"wrote {out}: 1 rows" in capsys.readouterr().out

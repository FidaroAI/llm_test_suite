import csv

import pytest
from conftest import a_run, backdate_run

from llmeval.cache_key import compute_cache_key
from llmeval.models import TestCase
from llmeval.resultrows import result_columns, result_rows, suite_of, write_csv
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


KEY = compute_cache_key(model="m1")


def a_case(test_id, **metadata):
    return TestCase.from_dict(
        {"id": test_id, "user": "the prompt", "metadata": metadata, "assertions": []}
    )


def runs_of(store, *ids):
    """The RunRow objects for these ids, in the order given."""
    return [store.get_run(i) for i in ids]


# --- suite resolution ----------------------------------------------------


def test_suite_is_the_id_prefix():
    assert suite_of("simple_facts.6c3396ab0e") == "simple_facts"


def test_suite_keeps_a_variant_suffixed_local_id_intact():
    assert suite_of("research_rubrics.abc1234567-g_eval") == "research_rubrics"


def test_suite_splits_on_the_first_dot_only():
    """A plugin may put dots in its own local ids; only the prefix belongs to the loader."""
    assert suite_of("my_suite.a.b.c") == "my_suite"


def test_suite_is_none_for_an_unprefixed_id():
    assert suite_of("legacy-style-id") is None


# --- one row per (result, assertion) ------------------------------------


def test_one_row_per_grading(store):
    run = a_run(store, KEY)
    rid = store.add_result_row("t-0123456789", run_id=run, output="hello")
    store.set_grading(rid, "a1", type="icontains", score=1.0, passed=True)
    store.set_grading(rid, "a2", type="not_contains", score=0.0, passed=False)
    rows = result_rows(store, runs_of(store, run))
    assert [(r["assertion_key"], r["passed"]) for r in rows] == [("a1", True), ("a2", False)]


def test_grading_detail_is_carried_onto_the_row(store):
    run = a_run(store, KEY)
    rid = store.add_result_row("t-0123456789", run_id=run, output="hello")
    store.set_grading(
        rid,
        "a1",
        type="rubric",
        metric="accuracy",
        score=0.75,
        passed=True,
        weight=2.0,
        reason="mostly right",
        judge_model="bedrock/haiku",
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert row["assertion_type"] == "rubric"
    assert row["metric"] == "accuracy"
    assert row["score"] == 0.75
    assert row["weight"] == 2.0
    assert row["grading_reason"] == "mostly right"
    assert row["judge_model"] == "bedrock/haiku"


def test_ungraded_result_still_yields_one_row(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="hello")
    rows = result_rows(store, runs_of(store, run))
    assert len(rows) == 1
    assert rows[0]["assertion_key"] is None
    assert rows[0]["output"] == "hello"


def test_errored_result_yields_one_row_with_no_grading_columns(store):
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789", run_id=run, error="timeout after 60s", latency_ms=60001.4
    )
    rows = result_rows(store, runs_of(store, run))
    assert len(rows) == 1
    assert rows[0]["error"] == "timeout after 60s"
    assert rows[0]["score"] is None and rows[0]["assertion_key"] is None
    # Latency on an error row is the point: it separates "the timeout is too tight" from
    # "the provider is down".
    assert rows[0]["latency_ms"] == 60001.4


# --- ordering -----------------------------------------------------------


def test_rows_are_grouped_by_run_in_the_order_given(store):
    older = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    newer = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    store.add_result_row("t-0123456789", run_id=newer, output="second")
    store.add_result_row("t-0123456789", run_id=older, output="first")
    rows = result_rows(store, runs_of(store, older, newer))
    assert [r["output"] for r in rows] == ["first", "second"]
    assert [r["run_id"] for r in rows] == [older, newer]


def test_attempts_within_a_run_are_in_chronological_order(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, error="boom")
    rid = store.add_result_row("t-0123456789", run_id=run, output="worked")
    store.set_grading(rid, "a1", passed=True)
    rows = result_rows(store, runs_of(store, run))
    assert [(r["attempt"], r["error"], r["assertion_key"]) for r in rows] == [
        (0, "boom", None),
        (1, None, "a1"),
    ]


def test_tests_within_a_run_are_grouped_together(store):
    run = a_run(store, KEY)
    store.add_result_row("b-0123456789", run_id=run, output="b0")
    store.add_result_row("a-0123456789", run_id=run, output="a0")
    store.add_result_row("b-0123456789", run_id=run, output="b1")
    rows = result_rows(store, runs_of(store, run))
    assert [(r["test_id"], r["attempt"]) for r in rows] == [
        ("a-0123456789", 0),
        ("b-0123456789", 0),
        ("b-0123456789", 1),
    ]


def test_run_metadata_is_on_every_row(store):
    run = backdate_run(
        store,
        store.create_run(KEY, provider_name="fidaro-prod"),
        "2026-07-01T09:00:00+00:00",
    )
    store.add_result_row("t-0123456789", run_id=run, output="x")
    row = result_rows(store, runs_of(store, run))[0]
    assert row["provider"] == "fidaro-prod"
    assert row["run_started_at"] == "2026-07-01T09:00:00+00:00"
    assert row["cache_key_hash"] == KEY.hash


# --- field shaping -----------------------------------------------------


def test_tokens_are_flattened(store):
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        output="x",
        tokens={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 22, 33)


def test_missing_tokens_are_empty_not_an_error(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    assert result_rows(store, runs_of(store, run))[0]["total_tokens"] is None


def test_surrounding_whitespace_is_trimmed_from_text_fields(store):
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789", run_id=run, output="\n\n\nThe answer  ", reasoning="  thinking\n"
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert row["output"] == "The answer"
    assert row["reasoning"] == "thinking"


def test_trimming_leaves_a_missing_output_as_none(store):
    """An error row has no output; trimming must not turn None into the string "None"."""
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, error="boom")
    row = result_rows(store, runs_of(store, run))[0]
    assert row["output"] is None
    assert row["reasoning"] is None


def test_internal_formatting_is_preserved(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="\n\nHeading\n\n* one\n* two\n")
    assert result_rows(store, runs_of(store, run))[0]["output"] == "Heading\n\n* one\n* two"


def test_latency_is_rounded_to_one_decimal(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x", latency_ms=16531.234567)
    assert result_rows(store, runs_of(store, run))[0]["latency_ms"] == 16531.2


# --- the prompt ---------------------------------------------------------


def test_prompt_comes_from_the_result_without_testcases(store):
    """The whole point of storing it: no --testcases needed to see the question."""
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        output="Paris",
        messages=[{"role": "user", "content": "What is the capital of France?"}],
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert row["prompt"] == "What is the capital of France?"


def test_prompt_is_the_last_user_turn_of_a_conversation(store):
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        output="x",
        messages=[
            {"role": "system", "content": "be terse"},
            {"role": "user", "content": "I'm planning a trip to Japan."},
            {"role": "assistant", "content": "When are you going?"},
            {"role": "user", "content": "Two weeks in spring."},
        ],
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert row["prompt"] == "Two weeks in spring."


def test_full_messages_are_kept_alongside_the_prompt(store):
    """prompt is the readable view; messages is the complete record behind it."""
    import json as jsonmod

    run = a_run(store, KEY)
    sent = [
        {"role": "system", "content": "be terse"},
        {"role": "user", "content": "I'm planning a trip to Japan."},
        {"role": "assistant", "content": "When are you going?"},
        {"role": "user", "content": "Two weeks in spring."},
    ]
    store.add_result_row("t-0123456789", run_id=run, output="x", messages=sent)
    row = result_rows(store, runs_of(store, run))[0]
    assert jsonmod.loads(row["messages"]) == sent


def test_a_single_turn_prompt_still_gets_a_messages_column(store):
    import json as jsonmod

    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789", run_id=run, output="x", messages=[{"role": "user", "content": "hi"}]
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert jsonmod.loads(row["messages"]) == [{"role": "user", "content": "hi"}]


def test_an_errored_attempt_still_shows_its_prompt(store):
    """"What did we send when this timed out?" is the first question about an error row."""
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        error="timeout",
        latency_ms=60001.0,
        messages=[{"role": "user", "content": "Write a full equity research note."}],
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert row["prompt"] == "Write a full equity research note."
    assert row["error"] == "timeout"


def test_a_result_without_stored_messages_leaves_the_prompt_empty(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    row = result_rows(store, runs_of(store, run))[0]
    assert row["prompt"] is None
    assert row["messages"] is None


def test_the_stored_prompt_wins_over_the_testcase_file(store):
    """The file may have been regenerated; only the stored value is what was really sent."""
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        output="x",
        messages=[{"role": "user", "content": "the question as sent"}],
    )
    cases = {"t-0123456789": a_case("t-0123456789")}  # a_case's prompt is "the prompt"
    row = result_rows(store, runs_of(store, run), cases)[0]
    assert row["prompt"] == "the question as sent"


def test_the_testcase_supplies_the_prompt_for_rows_predating_the_store_change(store):
    """A result with no stored messages still shows a prompt when the file has one."""
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    cases = {"t-0123456789": a_case("t-0123456789")}
    row = result_rows(store, runs_of(store, run), cases)[0]
    assert row["prompt"] == "the prompt"


# --- provider-specific output -------------------------------------------


def test_provider_specific_output_is_serialised_verbatim(store):
    """The whole object, unparsed: a new vendor key must need no change here."""
    import json as jsonmod

    run = a_run(store, KEY)
    sent = {"fidaro": {"title": "Capital of France", "something_new": [1, 2]}}
    store.add_result_row("t-0123456789", run_id=run, output="x", provider_specific=sent)
    row = result_rows(store, runs_of(store, run))[0]
    assert jsonmod.loads(row["provider_specific_output"]) == sent


def test_provider_specific_output_is_empty_when_the_provider_sent_none(store):
    """Most providers send nothing; that must be an empty cell, not the string "None"."""
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    assert result_rows(store, runs_of(store, run))[0]["provider_specific_output"] is None


def test_provider_specific_output_is_on_an_error_row_too(store):
    """A stream that timed out may have delivered its side-channel data before it stalled."""
    import json as jsonmod

    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        output="partial",
        error="stream timeout after 60.0s",
        provider_specific={"fidaro": {"title": "Half an answer"}},
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert jsonmod.loads(row["provider_specific_output"]) == {"fidaro": {"title": "Half an answer"}}


def test_provider_specific_output_is_not_ascii_escaped(store):
    """Titles are model-written prose, so they carry non-ASCII; keep it readable in the CSV."""
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789", run_id=run, output="x", provider_specific={"fidaro": {"title": "Café"}}
    )
    assert "Café" in result_rows(store, runs_of(store, run))[0]["provider_specific_output"]


# --- testcase selection -------------------------------------------------


def test_the_column_set_is_the_same_with_or_without_testcases(store):
    """Every column is derivable from the stored result, so nothing is conditional."""
    run = a_run(store, KEY)
    store.add_result_row("t.0123456789", run_id=run, output="x")
    cases = {"t.0123456789": a_case("t.0123456789")}
    with_cases = result_rows(store, runs_of(store, run), cases)[0]
    without = result_rows(store, runs_of(store, run))[0]
    assert list(with_cases) == list(without) == result_columns()
    assert with_cases["suite"] == without["suite"] == "t"


def test_testcases_select_as_well_as_enrich(store):
    """A result whose test is not in the loaded set is filtered out entirely."""
    run = a_run(store, KEY)
    store.add_result_row("wanted-0123456789", run_id=run, output="keep")
    store.add_result_row("other-0123456789", run_id=run, output="drop")
    cases = {"wanted-0123456789": a_case("wanted-0123456789")}
    rows = result_rows(store, runs_of(store, run), cases)
    assert [r["test_id"] for r in rows] == ["wanted-0123456789"]


def test_no_runs_means_no_rows(store):
    assert result_rows(store, []) == []


def test_a_run_with_no_results_contributes_nothing(store):
    run = a_run(store, KEY)
    assert result_rows(store, runs_of(store, run)) == []


# --- CSV writing -------------------------------------------------------


def test_write_csv_round_trips(tmp_path, store):
    run = a_run(store, KEY)
    rid = store.add_result_row("t-0123456789", run_id=run, output="hello")
    store.set_grading(rid, "a1", passed=True, score=1.0)
    rows = result_rows(store, runs_of(store, run))
    columns = result_columns()
    path = write_csv(rows, columns, str(tmp_path / "out" / "rows.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == columns
        got = list(reader)
    assert len(got) == 1
    assert got[0]["output"] == "hello"


def test_write_csv_of_no_rows_still_writes_the_header(tmp_path):
    columns = result_columns()
    path = write_csv([], columns, str(tmp_path / "empty.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == columns


def test_write_csv_survives_embedded_newlines_and_commas(tmp_path, store):
    """Model output contains both routinely; the CSV must still round-trip."""
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output='line one\nline, two "quoted"')
    rows = result_rows(store, runs_of(store, run))
    columns = result_columns()
    path = write_csv(rows, columns, str(tmp_path / "rows.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        got = list(csv.DictReader(f))
    assert len(got) == 1
    assert got[0]["output"] == 'line one\nline, two "quoted"'

"""The CLI end to end, offline.

Every test builds a small project directory — ``testcases/`` plus a provider config — and
``chdir``s into it, because ``--testcases`` names a source inside ``testcases/`` rather than
a path. Sources here are hand-written ``.json`` files; the plugin path is covered in
``test_cli_sources.py``.
"""

import csv as csvmod
import json
import sqlite3

import pytest

from llmeval.cli import build_parser, load_provider_config, main

ECHO = {"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}}

# The echo provider returns the prompt verbatim, so an assertion has to match a word in the
# *question* for grading to pass. Asserting on an answer would give a legitimately failing
# row and make the pass/fail expectations below misleading.
FACTS = [{"id": "cap", "user": "What is the capital of France?",
          "assertions": [{"type": "icontains", "value": "capital"}]}]


def write_source(root, name, cases):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps(cases), encoding="utf-8")


def project(tmp_path, monkeypatch, sources=None):
    """A project directory with test-case sources and an echo provider. Returns the db path."""
    root = tmp_path / "testcases"
    for name, cases in (sources or {"facts": FACTS}).items():
        write_source(root, name, cases)
    (tmp_path / "echo.json").write_text(json.dumps(ECHO), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    return str(tmp_path / "db.sqlite3")


def echoed(tmp_path, monkeypatch, sources=None):
    """A project whose tests have been run and graded once."""
    db = project(tmp_path, monkeypatch, sources)
    main(["run", "--provider", "echo.json", "--db", db])
    main(["grade", "--provider", "echo.json", "--db", db])
    return db


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        return reader.fieldnames, list(reader)


def count_results(db):
    with sqlite3.connect(db) as conn:
        return conn.execute("select count(*) from results").fetchone()[0]


def test_load_provider_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_BASE", "http://gw:8082")
    p = tmp_path / "prov.json"
    p.write_text(json.dumps({"name": "x", "model": "openai/m", "base_url": "${MY_BASE}/v1"}))
    cfg = load_provider_config(str(p))
    assert cfg.base_url == "http://gw:8082/v1"


# --- the pipeline -------------------------------------------------------


def test_cli_pipeline_offline(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    assert main(["run", "--provider", "echo.json", "--db", db]) == 0
    assert main(["grade", "--provider", "echo.json", "--db", db]) == 0

    # report emits CSV rows; the statistics HTML is compare-report
    assert main(["report", "--provider", "echo.json", "--db", db, "--out", "rows.csv"]) == 0
    assert (tmp_path / "rows.csv").read_text().strip() != ""

    assert main(["compare-report", "--providers", "echo.json", "--db", db, "--out", "r.html"]) == 0
    assert (tmp_path / "r.html").read_text().strip() != ""


def test_cli_run_is_idempotent_via_cache(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    main(["run", "--provider", "echo.json", "--db", db])
    main(["run", "--provider", "echo.json", "--db", db])
    assert count_results(db) == 1


def test_ids_are_namespaced_by_source(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    main(["run", "--provider", "echo.json", "--db", db])
    with sqlite3.connect(db) as conn:
        assert conn.execute("select test_id from results").fetchone()[0] == "facts.cap"


# --- flag parsing -------------------------------------------------------


def test_cli_run_concurrency_flag_defaults_to_five():
    args = build_parser().parse_args(["run", "--provider", "p"])
    assert args.concurrency == 5


def test_cli_run_concurrency_flag_override():
    args = build_parser().parse_args(["run", "--provider", "p", "--concurrency", "12"])
    assert args.concurrency == 12


def test_cli_run_timeout_flag_defaults_to_sixty_seconds():
    assert build_parser().parse_args(["run", "--provider", "p"]).timeout == 60.0


def test_cli_run_timeout_flag_override():
    args = build_parser().parse_args(["run", "--provider", "p", "--timeout", "300"])
    assert args.timeout == 300.0


def test_cli_run_repeat_flag_defaults_to_one():
    assert build_parser().parse_args(["run", "--provider", "p"]).repeat == 1


def test_cli_run_repeat_flag_override():
    assert build_parser().parse_args(["run", "--provider", "p", "--repeat", "5"]).repeat == 5


def test_cli_run_mode_offers_only_reuse_and_always():
    """``target_n`` was never a mode — it was a count. It is now ``--repeat``."""
    assert build_parser().parse_args(["run", "--provider", "p"]).mode == "reuse"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--provider", "p", "--mode", "target_n"])
    with pytest.raises(SystemExit):
        build_parser().parse_args(["run", "--provider", "p", "--target-n", "3"])


def test_cli_repeat_runs_each_test_that_many_times(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    assert main(["run", "--provider", "echo.json", "--db", db, "--repeat", "3"]) == 0
    assert count_results(db) == 3


def test_cli_repeat_under_reuse_tops_up_rather_than_redoing(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    main(["run", "--provider", "echo.json", "--db", db, "--repeat", "2"])
    main(["run", "--provider", "echo.json", "--db", db, "--repeat", "5"])
    assert count_results(db) == 5


def test_cli_repeat_under_always_appends_every_time(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    main(["run", "--provider", "echo.json", "--db", db, "--mode", "always", "--repeat", "2"])
    main(["run", "--provider", "echo.json", "--db", db, "--mode", "always", "--repeat", "2"])
    assert count_results(db) == 4


def test_cli_repeat_below_one_is_a_usage_error_not_a_silent_no_op(tmp_path, monkeypatch):
    """Rejected while parsing, so nothing is opened and no database is left behind."""
    db = project(tmp_path, monkeypatch)
    with pytest.raises(SystemExit) as exc:
        main(["run", "--provider", "echo.json", "--db", db, "--repeat", "0"])
    assert exc.value.code == 2
    assert not (tmp_path / "db.sqlite3").exists()


def test_cli_repeat_is_recorded_against_the_run(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    main(["run", "--provider", "echo.json", "--db", db, "--repeat", "4"])
    with sqlite3.connect(db) as conn:
        params = json.loads(conn.execute("select params_json from runs").fetchone()[0])
    assert params["repeat"] == 4
    assert "target_n" not in params


def test_cli_testcases_flag_is_repeatable():
    args = build_parser().parse_args(
        ["run", "--testcases", "a", "--testcases", "b", "--provider", "p"]
    )
    assert args.testcases == ["a", "b"]


def test_cli_testcases_is_optional_everywhere():
    """Omitted means every source — on run as well as report."""
    assert build_parser().parse_args(["run", "--provider", "p"]).testcases is None
    assert build_parser().parse_args(["report", "--out", "o.csv"]).testcases is None
    assert build_parser().parse_args(["generate"]).testcases is None


def test_run_selection_flags_default_to_none():
    args = build_parser().parse_args(["report", "--out", "o.csv"])
    assert (args.run_id, args.run_after, args.run_before, args.run_last_n) == (
        None, None, None, None,
    )


# --- source selection ---------------------------------------------------


TWO_SOURCES = {
    "alpha": [{"id": "a", "user": "q-alpha?", "assertions": []}],
    "beta": [{"id": "b", "user": "q-beta?", "assertions": []}],
}


def test_omitting_testcases_runs_every_source(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch, TWO_SOURCES)
    main(["run", "--provider", "echo.json", "--db", db])
    assert count_results(db) == 2


def test_naming_a_source_runs_only_that_one(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch, TWO_SOURCES)
    main(["run", "--provider", "echo.json", "--db", db, "--testcases", "alpha"])
    with sqlite3.connect(db) as conn:
        assert {r[0] for r in conn.execute("select test_id from results")} == {"alpha.a"}


def test_repeating_the_flag_unions_the_sources(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch, TWO_SOURCES)
    main([
        "run", "--provider", "echo.json", "--db", db,
        "--testcases", "alpha", "--testcases", "beta",
    ])
    assert count_results(db) == 2


def test_naming_the_same_source_twice_does_not_double_run(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch, TWO_SOURCES)
    main([
        "run", "--provider", "echo.json", "--db", db,
        "--testcases", "alpha", "--testcases", "alpha",
    ])
    assert count_results(db) == 1


def test_an_unknown_source_is_an_error_not_an_empty_run(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch)
    assert main(["run", "--provider", "echo.json", "--db", db, "--testcases", "nope"]) == 2


def test_cli_run_limit_runs_only_n(tmp_path, monkeypatch):
    three = {"s": [{"id": f"q{i}", "user": f"q{i}?", "assertions": []} for i in range(3)]}
    db = project(tmp_path, monkeypatch, three)
    main(["run", "--provider", "echo.json", "--db", db, "--limit", "2"])
    assert count_results(db) == 2


def test_cli_grade_limit_grades_only_n_test_cases(tmp_path, monkeypatch):
    """The limit is a *test case* count — every attempt and assertion of the first N."""
    three = {"s": [
        {"id": f"q{i}", "user": f"q{i}?", "assertions": [{"type": "icontains", "value": "q"}]}
        for i in range(3)
    ]}
    db = project(tmp_path, monkeypatch, three)
    main(["run", "--provider", "echo.json", "--db", db])
    assert main(["grade", "--provider", "echo.json", "--db", db, "--limit", "2"]) == 0
    with sqlite3.connect(db) as conn:
        graded = {r[0] for r in conn.execute(
            "select test_id from results join gradings on gradings.result_id = results.id"
        )}
    assert graded == {"s.q0", "s.q1"}


def test_cli_grade_limit_covers_every_attempt_of_the_cases_it_picks(tmp_path, monkeypatch):
    """A limited grade is still per-result: --repeat 3 leaves three graded attempts."""
    two = {"s": [
        {"id": f"q{i}", "user": f"q{i}?", "assertions": [{"type": "icontains", "value": "q"}]}
        for i in range(2)
    ]}
    db = project(tmp_path, monkeypatch, two)
    main(["run", "--provider", "echo.json", "--db", db, "--repeat", "3"])
    main(["grade", "--provider", "echo.json", "--db", db, "--limit", "1"])
    with sqlite3.connect(db) as conn:
        assert conn.execute("select count(*) from gradings").fetchone()[0] == 3


def test_cli_grade_limit_defaults_to_every_test_case():
    assert build_parser().parse_args(["grade", "--provider", "p"]).limit is None


def test_cli_run_with_concurrency_offline(tmp_path, monkeypatch):
    three = {"s": [{"id": f"q{i}", "user": f"q{i}?", "assertions": []} for i in range(3)]}
    db = project(tmp_path, monkeypatch, three)
    assert main(["run", "--provider", "echo.json", "--db", db, "--concurrency", "3"]) == 0
    assert count_results(db) == 3


# --- report -------------------------------------------------------------


def test_report_writes_a_csv(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    assert main(["report", "--db", db, "--provider", "echo.json", "--out", "rows.csv"]) == 0
    _, rows = read_csv(tmp_path / "rows.csv")
    assert len(rows) == 1
    assert rows[0]["assertion_key"].startswith("icontains:")
    assert rows[0]["passed"] == "True"
    assert rows[0]["prompt"] == "What is the capital of France?"
    assert rows[0]["suite"] == "facts"
    assert rows[0]["messages"] != ""
    assert rows[0]["run_id"].startswith("run_")
    assert rows[0]["latency_ms"] != ""


def test_report_needs_no_testcases_at_all(tmp_path, monkeypatch):
    """Prompt, answer and suite all come off the stored result."""
    db = echoed(tmp_path, monkeypatch)
    assert main(["report", "--db", db, "--out", "rows.csv"]) == 0
    fieldnames, rows = read_csv(tmp_path / "rows.csv")
    assert rows[0]["prompt"] == "What is the capital of France?"
    assert rows[0]["suite"] == "facts"
    # Classification is gone from the suite entirely.
    assert "request_type" not in (fieldnames or [])
    assert "domain" not in (fieldnames or [])


def test_report_narrows_to_a_named_source(tmp_path, monkeypatch):
    db = project(tmp_path, monkeypatch, TWO_SOURCES)
    main(["run", "--provider", "echo.json", "--db", db])
    assert main(["report", "--db", db, "--testcases", "alpha", "--out", "rows.csv"]) == 0
    _, rows = read_csv(tmp_path / "rows.csv")
    assert {r["test_id"] for r in rows} == {"alpha.a"}


def test_report_on_a_missing_db_is_an_error(tmp_path, monkeypatch):
    project(tmp_path, monkeypatch)
    assert main(["report", "--db", "nope.sqlite3", "--out", "o.csv"]) == 2


def test_report_with_conflicting_run_selection_is_an_error(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    rc = main([
        "report", "--db", db, "--out", "o.csv", "--run-last-n", "1", "--run-after", "2026-07-01",
    ])
    assert rc == 2


def test_report_with_an_unknown_run_id_is_an_error(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    assert main(["report", "--db", db, "--out", "o.csv", "--run-id", "run_1900"]) == 2


def test_report_with_an_ambiguous_run_prefix_is_an_error(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    main(["run", "--provider", "echo.json", "--db", db, "--mode", "always"])
    # "run_" prefixes every id, so it can never identify one.
    assert main(["report", "--db", db, "--out", "o.csv", "--run-id", "run_"]) == 2


def test_report_run_selection_that_matches_nothing_is_not_an_error(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    assert main(["report", "--db", db, "--out", "rows.csv", "--run-after", "2099-01-01"]) == 0
    fieldnames, rows = read_csv(tmp_path / "rows.csv")
    assert fieldnames is not None  # header still written
    assert rows == []


def test_report_last_n_selects_only_the_newest_run(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    # A second run of the same test: --mode always appends rather than reusing the cache.
    main(["run", "--provider", "echo.json", "--db", db, "--mode", "always"])
    assert main(["report", "--db", db, "--out", "rows.csv", "--run-last-n", "1"]) == 0
    _, rows = read_csv(tmp_path / "rows.csv")
    assert len({r["run_id"] for r in rows}) == 1

    # ...and without the flag, both runs appear.
    assert main(["report", "--db", db, "--out", "both.csv"]) == 0
    _, all_rows = read_csv(tmp_path / "both.csv")
    assert len({r["run_id"] for r in all_rows}) == 2


def test_compare_report_writes_the_statistics_html(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    assert main(["compare-report", "--providers", "echo.json", "--db", db, "--out", "c.html"]) == 0
    assert "llmeval comparison" in (tmp_path / "c.html").read_text()


# --- grade --------------------------------------------------------------


def test_grade_accepts_run_selection_flags(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    assert main(["grade", "--provider", "echo.json", "--db", db, "--run-last-n", "1"]) == 0


def test_grade_with_conflicting_run_selection_is_an_error(tmp_path, monkeypatch):
    db = echoed(tmp_path, monkeypatch)
    rc = main([
        "grade", "--provider", "echo.json", "--db", db, "--run-id", "run_x", "--run-last-n", "1",
    ])
    assert rc == 2


# --- entry points -------------------------------------------------------


def test_generate_csv_subcommand_is_gone():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["generate-csv", "--csv", "x", "--suite", "y", "--out", "z"])


def test_python_dash_m_llmeval_is_an_entry_point():
    """The porcelain invokes the CLI as `sys.executable -m llmeval`."""
    import subprocess
    import sys

    out = subprocess.run(
        [sys.executable, "-m", "llmeval", "--help"], capture_output=True, text=True, check=False
    )
    assert out.returncode == 0
    assert "usage: llmeval" in out.stdout

import json
import os

from llmeval.cli import build_parser, load_provider_config, main


def test_load_provider_config_expands_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MY_BASE", "http://gw:8082")
    p = tmp_path / "prov.json"
    p.write_text(json.dumps({"name": "x", "model": "openai/m", "base_url": "${MY_BASE}/v1"}))
    cfg = load_provider_config(str(p))
    assert cfg.base_url == "http://gw:8082/v1"


def test_cli_pipeline_offline(tmp_path):
    # generate
    csv = tmp_path / "facts.csv"
    csv.write_text('user,__expected\n"What is the capital of France?","icontains:Paris"\n')
    tc_dir = tmp_path / "testcases"
    assert main(["generate-csv", "--csv", str(csv), "--suite", "facts", "--out", str(tc_dir)]) == 0
    assert (tc_dir / "facts.json").exists()

    # an echo provider config (no network)
    prov = tmp_path / "echo.json"
    prov.write_text(json.dumps({"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}}))
    db = str(tmp_path / "db.sqlite3")

    # run + grade against cached outputs
    assert main(["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db]) == 0
    assert main(["grade", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db]) == 0

    # report now emits CSV rows; the statistics HTML moved to compare-report
    rows_csv = tmp_path / "rows.csv"
    assert main(["report", "--provider", str(prov), "--db", db, "--out", str(rows_csv)]) == 0
    assert rows_csv.exists() and rows_csv.read_text().strip() != ""

    out = tmp_path / "r.html"
    assert main(["compare-report", "--providers", str(prov), "--db", db, "--out", str(out)]) == 0
    assert out.exists() and out.read_text().strip() != ""


def test_cli_run_is_idempotent_via_cache(tmp_path, capsys):
    csv = tmp_path / "f.csv"
    csv.write_text('user,__expected\n"Q?","icontains:A"\n')
    tc_dir = tmp_path / "tc"
    main(["generate-csv", "--csv", str(csv), "--suite", "s", "--out", str(tc_dir)])
    prov = tmp_path / "echo.json"
    prov.write_text(json.dumps({"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}}))
    db = str(tmp_path / "db.sqlite3")
    main(["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db])
    # second run reuses cache -> 0 new calls
    main(["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db])
    from llmeval.models import ProviderConfig
    from llmeval.store import Store

    cfg = ProviderConfig(name="echo", model="echo", extra={"provider_impl": "echo"})
    s = Store(db)
    assert s.count_results("s-" + __import__("hashlib").sha1(b"Q?").hexdigest()[:10], cfg.cache_key().hash) == 1
    s.close()


def test_cli_run_concurrency_flag_defaults_to_five():
    parser = build_parser()
    args = parser.parse_args(["run", "--testcases", "tc", "--provider", "p"])
    assert args.concurrency == 5


def test_cli_run_concurrency_flag_override():
    parser = build_parser()
    args = parser.parse_args(["run", "--testcases", "tc", "--provider", "p", "--concurrency", "12"])
    assert args.concurrency == 12


def test_cli_run_timeout_flag_defaults_to_sixty_seconds():
    parser = build_parser()
    args = parser.parse_args(["run", "--testcases", "tc", "--provider", "p"])
    assert args.timeout == 60.0


def test_cli_run_timeout_flag_override():
    parser = build_parser()
    args = parser.parse_args(["run", "--testcases", "tc", "--provider", "p", "--timeout", "300"])
    assert args.timeout == 300.0


def test_cli_run_with_concurrency_offline(tmp_path):
    import sqlite3

    csv = tmp_path / "f.csv"
    csv.write_text(
        'user,__expected\n"q1?","icontains:a"\n"q2?","icontains:b"\n"q3?","icontains:c"\n'
    )
    tc_dir = tmp_path / "tc"
    main(["generate-csv", "--csv", str(csv), "--suite", "s", "--out", str(tc_dir)])
    prov = tmp_path / "echo.json"
    prov.write_text(json.dumps({"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}}))
    db = str(tmp_path / "db.sqlite3")
    rc = main(
        ["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db, "--concurrency", "3"]
    )
    assert rc == 0
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from results").fetchone()[0] == 3
    conn.close()


def _echo_setup(tmp_path):
    """Generate one test case, run it with the echo provider, and grade it.

    The echo provider returns the prompt verbatim, so the assertion has to match a word in
    the *question* for the grading to pass — asserting on the answer would give a
    legitimately failing row and make the pass/fail expectations below misleading.
    """
    csv_src = tmp_path / "facts.csv"
    csv_src.write_text('user,__expected\n"What is the capital of France?","icontains:capital"\n')
    tc_dir = tmp_path / "testcases"
    main(["generate-csv", "--csv", str(csv_src), "--suite", "facts", "--out", str(tc_dir)])
    prov = tmp_path / "echo.json"
    prov.write_text(
        json.dumps({"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}})
    )
    db = str(tmp_path / "db.sqlite3")
    main(["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db])
    main(["grade", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db])
    return str(tc_dir), str(prov), db


def _read_csv(path):
    import csv as csvmod

    with open(path, newline="", encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        return reader.fieldnames, list(reader)


def test_report_writes_a_csv(tmp_path):
    tc_dir, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "rows.csv"
    rc = main(
        ["report", "--db", db, "--provider", prov, "--testcases", tc_dir, "--out", str(out)]
    )
    assert rc == 0
    _, rows = _read_csv(out)
    assert len(rows) == 1
    assert rows[0]["assertion_key"].startswith("icontains:")
    assert rows[0]["passed"] == "True"
    assert rows[0]["prompt"] == "What is the capital of France?"
    assert rows[0]["run_id"].startswith("run_")
    assert rows[0]["latency_ms"] != ""


def test_report_without_testcases_omits_the_prompt_column(tmp_path):
    _, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "rows.csv"
    assert main(["report", "--db", db, "--provider", prov, "--out", str(out)]) == 0
    fieldnames, _ = _read_csv(out)
    assert "prompt" not in (fieldnames or [])


def test_report_on_a_missing_db_is_an_error(tmp_path):
    rc = main(
        ["report", "--db", str(tmp_path / "nope.sqlite3"), "--out", str(tmp_path / "o.csv")]
    )
    assert rc == 2


def test_report_with_conflicting_run_selection_is_an_error(tmp_path):
    _, _, db = _echo_setup(tmp_path)
    rc = main(
        ["report", "--db", db, "--out", str(tmp_path / "o.csv"),
         "--run-last-n", "1", "--run-after", "2026-07-01"]
    )
    assert rc == 2


def test_report_with_an_unknown_run_id_is_an_error(tmp_path):
    _, _, db = _echo_setup(tmp_path)
    rc = main(["report", "--db", db, "--out", str(tmp_path / "o.csv"), "--run-id", "run_1900"])
    assert rc == 2


def test_report_run_selection_that_matches_nothing_is_not_an_error(tmp_path):
    _, _, db = _echo_setup(tmp_path)
    out = tmp_path / "rows.csv"
    assert main(["report", "--db", db, "--out", str(out), "--run-after", "2099-01-01"]) == 0
    fieldnames, rows = _read_csv(out)
    assert fieldnames is not None  # header still written
    assert rows == []


def test_report_last_n_selects_only_the_newest_run(tmp_path):
    tc_dir, prov, db = _echo_setup(tmp_path)
    # A second run of the same test: --mode always appends rather than reusing the cache.
    main(["run", "--testcases", tc_dir, "--provider", prov, "--db", db, "--mode", "always"])
    out = tmp_path / "rows.csv"
    assert main(["report", "--db", db, "--out", str(out), "--run-last-n", "1"]) == 0
    _, rows = _read_csv(out)
    assert len({r["run_id"] for r in rows}) == 1

    # ...and without the flag, both runs appear.
    both = tmp_path / "both.csv"
    assert main(["report", "--db", db, "--out", str(both)]) == 0
    _, all_rows = _read_csv(both)
    assert len({r["run_id"] for r in all_rows}) == 2


def test_compare_report_writes_the_statistics_html(tmp_path):
    _, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "compare.html"
    assert main(["compare-report", "--providers", prov, "--db", db, "--out", str(out)]) == 0
    assert "llmeval comparison" in out.read_text()


def test_grade_accepts_run_selection_flags(tmp_path):
    tc_dir, prov, db = _echo_setup(tmp_path)
    assert (
        main(["grade", "--testcases", tc_dir, "--provider", prov, "--db", db, "--run-last-n", "1"])
        == 0
    )


def test_grade_with_conflicting_run_selection_is_an_error(tmp_path):
    tc_dir, prov, db = _echo_setup(tmp_path)
    rc = main(
        ["grade", "--testcases", tc_dir, "--provider", prov, "--db", db,
         "--run-id", "run_x", "--run-last-n", "1"]
    )
    assert rc == 2


def test_run_selection_flags_default_to_none():
    parser = build_parser()
    args = parser.parse_args(["report", "--out", "o.csv"])
    assert (args.run_id, args.run_after, args.run_before, args.run_last_n) == (
        None,
        None,
        None,
        None,
    )


def test_cli_run_limit_runs_only_n(tmp_path):
    import sqlite3

    csv = tmp_path / "f.csv"
    csv.write_text(
        'user,__expected\n"q1?","icontains:a"\n"q2?","icontains:b"\n"q3?","icontains:c"\n'
    )
    tc_dir = tmp_path / "tc"
    main(["generate-csv", "--csv", str(csv), "--suite", "s", "--out", str(tc_dir)])
    prov = tmp_path / "echo.json"
    prov.write_text(json.dumps({"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}}))
    db = str(tmp_path / "db.sqlite3")
    main(["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db, "--limit", "2"])
    conn = sqlite3.connect(db)
    assert conn.execute("select count(*) from results").fetchone()[0] == 2
    conn.close()

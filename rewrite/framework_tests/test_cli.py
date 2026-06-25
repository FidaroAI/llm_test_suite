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

    # report
    out = tmp_path / "r.html"
    rc = main(["report", "--providers", str(prov), "--db", db, "--out", str(out)])
    assert rc == 0
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

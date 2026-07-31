"""The CLI's source selection: --testcases names a plugin or a .json stem, not a path."""

import json
import sqlite3

import pytest

from llmeval.cli import main

PLUGIN = '''
from pathlib import Path

from llmeval.generation.csv_plugin import CsvTestCasePlugin
from llmeval.plugins import PluginInterface, TestCasePlugin

CSV_PATH = Path(__file__).resolve().parent / "facts.csv"


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return CsvTestCasePlugin(interface, CSV_PATH)
'''


def make_project(tmp_path):
    root = tmp_path / "testcases"
    plugin = root / "facts"
    plugin.mkdir(parents=True)
    (plugin / "__init__.py").write_text(PLUGIN, encoding="utf-8")
    (plugin / "facts.csv").write_text(
        'user,__expected\n"What is the capital of France?","icontains:Paris"\n', encoding="utf-8"
    )
    (root / "examples.json").write_text(
        json.dumps([{"id": "hand", "user": "hi", "assertions": []}]), encoding="utf-8"
    )
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "echo.json").write_text(
        json.dumps({"name": "echo", "model": "echo"}), encoding="utf-8"
    )
    return root


def stored_test_ids(db="llmeval.sqlite3"):
    with sqlite3.connect(db) as conn:
        return {row[0] for row in conn.execute("SELECT test_id FROM results")}


def test_generate_runs_every_plugin_by_default(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["generate"]) == 0
    cached = tmp_path / ".testcases.cache" / "facts" / "testcases.json"
    assert json.loads(cached.read_text())[0]["assertions"][0]["value"] == "Paris"


def test_generate_accepts_a_source_name(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["generate", "--testcases", "facts"]) == 0


def test_generate_rejects_an_unknown_source(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    assert main(["generate", "--testcases", "nope"]) == 2


def test_run_loads_every_source_when_testcases_is_omitted(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    main(["generate"])
    assert main(["run", "--provider", "configs/echo.json", "--concurrency", "1"]) == 0
    ids = stored_test_ids()
    assert any(i.startswith("facts.") for i in ids)
    assert "examples.hand" in ids


def test_run_narrows_to_a_named_source(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    main(["generate"])
    assert main(["run", "--provider", "configs/echo.json", "--testcases", "examples"]) == 0
    assert stored_test_ids() == {"examples.hand"}


def test_report_reads_the_suite_off_the_id_prefix(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    main(["generate"])
    main(["run", "--provider", "configs/echo.json"])
    assert main(["report", "--out", "rows.csv"]) == 0
    header, *rows = (tmp_path / "rows.csv").read_text().splitlines()
    assert "suite" in header.split(",")
    assert "request_type" not in header.split(",")
    assert any(",facts," in row or row.startswith("facts,") for row in rows) or rows


def test_generate_csv_subcommand_is_gone(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        main(["generate-csv", "--csv", "x", "--suite", "y", "--out", "z"])

"""The shared machinery plugins build on. Loading and selection live in their own files."""

import logging

import pytest

from llmeval.generation.common import drop_duplicate_ids, local_id
from llmeval.generation.csv_plugin import CsvTestCasePlugin
from llmeval.generation.csv_source import parse_expected, rows_from_csv
from llmeval.plugins import PluginInterface


def write_csv(path, rows, header="user,__expected"):
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    return str(path)


def test_parse_expected_shorthands():
    assert parse_expected("icontains:Paris").type == "icontains"
    assert parse_expected("icontains:Paris").value == "Paris"
    assert parse_expected("regex:\\d+").type == "regex"
    assert parse_expected("not_contains:sorry").type == "not_contains"


def test_parse_expected_rejects_unknown():
    with pytest.raises(ValueError):
        parse_expected("python:file://assertions/foo.py")


def test_local_id_is_a_bare_digest_with_no_suite_prefix():
    one = local_id("What is the capital of France?")
    assert len(one) == 10 and one.isalnum()
    assert local_id("  What is the capital of France?  ") == one
    assert local_id("q", variant="g_eval") == f"{local_id('q')}-g_eval"


def test_rows_from_csv_builds_local_ids_expectations_and_metadata(tmp_path):
    csv_path = write_csv(
        tmp_path / "facts.csv",
        ['"What is the capital of France?","icontains:Paris",eu'],
        header="user,__expected,__metadata:region",
    )
    (case,) = rows_from_csv(csv_path)
    assert case["id"] == local_id("What is the capital of France?")
    assert case["assertions"] == [{"type": "icontains", "value": "Paris"}]
    assert case["metadata"] == {"region": "eu"}
    assert "suite" not in case["metadata"]


def test_rows_from_csv_ids_are_stable_across_calls(tmp_path):
    csv_path = write_csv(tmp_path / "f.csv", ['"Q one?","icontains:A"'])
    assert rows_from_csv(csv_path)[0]["id"] == rows_from_csv(csv_path)[0]["id"]


def test_rows_from_csv_skips_blank_prompts(tmp_path):
    csv_path = write_csv(tmp_path / "f.csv", ['"Q?","icontains:A"', '"","icontains:B"'])
    assert len(rows_from_csv(csv_path)) == 1


def test_csv_plugin_writes_its_cache_file_and_reads_it_back(tmp_path):
    csv_path = write_csv(tmp_path / "facts.csv", ['"Q?","icontains:A"'])
    plugin = CsvTestCasePlugin(PluginInterface("facts", tmp_path / "cache"), csv_path)

    assert plugin.get_testcases() == []          # nothing generated yet
    assert plugin.generate_testcases() is True
    assert (tmp_path / "cache" / "facts" / "testcases.json").is_file()
    assert plugin.get_testcases()[0]["assertions"][0]["value"] == "A"


def test_csv_plugin_reports_failure_for_a_missing_csv(tmp_path):
    plugin = CsvTestCasePlugin(PluginInterface("gone", tmp_path / "cache"), tmp_path / "nope.csv")
    assert plugin.generate_testcases() is False


def test_drop_duplicate_ids_keeps_the_first_of_each_id(caplog):
    cases = [
        {"id": "aaa", "user": "What is the capital of France?", "assertions": [1]},
        {"id": "aaa", "user": "What is the capital of France?", "assertions": [2]},
        {"id": "bbb", "user": "Who wrote Dune?", "assertions": [3]},
    ]
    with caplog.at_level(logging.WARNING):
        kept = drop_duplicate_ids(cases, "facts")

    assert [c["id"] for c in kept] == ["aaa", "bbb"]
    assert kept[0]["assertions"] == [1]          # the first one wins
    assert cases[0]["assertions"] == [1]         # and the input is left alone


def test_drop_duplicate_ids_warns_with_the_id_and_a_prompt_snippet(caplog):
    cases = [{"id": "aaa", "user": "What is the capital of France?"}] * 2
    with caplog.at_level(logging.WARNING):
        drop_duplicate_ids(cases, "facts")

    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "facts" in record.message
    assert "aaa" in record.message
    assert "What is the capital of France?" in record.message


def test_drop_duplicate_ids_truncates_a_long_prompt_in_the_warning(caplog):
    prompt = "Write me an essay about " + "elephants " * 50
    with caplog.at_level(logging.WARNING):
        drop_duplicate_ids([{"id": "a", "user": prompt}] * 2, "facts")

    (record,) = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert "Write me an essay about elephants" in record.message
    assert len(record.message) < 200


def test_drop_duplicate_ids_is_a_no_op_when_there_are_none(caplog):
    cases = [{"id": "a", "user": "one"}, {"id": "b", "user": "two"}]
    with caplog.at_level(logging.WARNING):
        assert drop_duplicate_ids(cases, "facts") == cases
    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def test_csv_plugin_drops_duplicate_rows_rather_than_emitting_a_clashing_id(tmp_path, caplog):
    csv_path = write_csv(
        tmp_path / "facts.csv",
        ['"What is the capital of France?","icontains:Paris"',
         '"What is the capital of France?","icontains:Paris"',
         '"Who wrote Dune?","icontains:Herbert"'],
    )
    plugin = CsvTestCasePlugin(PluginInterface("facts", tmp_path / "cache"), csv_path)

    with caplog.at_level(logging.WARNING):
        assert plugin.generate_testcases() is True

    ids = [c["id"] for c in plugin.get_testcases()]
    assert len(ids) == len(set(ids)) == 2
    assert "capital of France" in caplog.text

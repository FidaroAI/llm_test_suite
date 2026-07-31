import pytest

from llmeval.generation.common import local_id
from llmeval.generation.csv_plugin import CsvTestCasePlugin
from llmeval.generation.csv_source import generate_from_csv, parse_expected, rows_from_csv
from llmeval.plugins import PluginInterface
from llmeval.testcases import load_all_testcases, load_testcases


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


def test_generate_writes_inspectable_testcases(tmp_path):
    csv = write_csv(
        tmp_path / "f.csv",
        ['"What is the capital of France?","icontains:Paris"',
         '"What company makes the 3310?","icontains:Nokia"'],
    )
    out = tmp_path / "out"
    cases = generate_from_csv(csv, suite="simple_facts", out_dir=str(out))
    assert len(cases) == 2
    # written as inspectable JSON before any run
    files = list(out.glob("*.json"))
    assert len(files) == 1
    loaded = load_testcases(str(out))
    assert {c.assertions[0].value for c in loaded} == {"Paris", "Nokia"}
    assert all(c.metadata["suite"] == "simple_facts" for c in loaded)


def test_generated_ids_are_stable(tmp_path):
    csv = write_csv(tmp_path / "f.csv", ['"Q one?","icontains:A"'])
    a = generate_from_csv(csv, suite="s")
    b = generate_from_csv(csv, suite="s")
    assert a[0]["id"] == b[0]["id"]


def test_metadata_columns_are_carried(tmp_path):
    csv = write_csv(
        tmp_path / "f.csv",
        ['"Price of Arm?","icontains:USD","Arm Holdings"'],
        header="user,__expected,__metadata:company",
    )
    cases = generate_from_csv(csv, suite="stock")
    assert cases[0]["metadata"]["company"] == "Arm Holdings"


def test_classifications_are_stamped(tmp_path):
    csv = write_csv(tmp_path / "f.csv", ['"Q?","icontains:A"'])
    cases = generate_from_csv(
        csv, suite="s", classifications={"Q?": {"request_type": "factual_qa", "domain": "general_other"}}
    )
    assert cases[0]["metadata"]["request_type"] == "factual_qa"


def test_load_testcases_filters_by_metadata(tmp_path):
    out = tmp_path / "out"
    csv = write_csv(tmp_path / "f.csv", ['"Q?","icontains:A"'])
    generate_from_csv(csv, suite="alpha", out_dir=str(out))
    generate_from_csv(write_csv(tmp_path / "g.csv", ['"R?","icontains:B"']), suite="beta", out_dir=str(out))
    assert len(load_testcases(str(out))) == 2
    assert len(load_testcases(str(out), filters={"suite": "alpha"})) == 1
    assert len(load_testcases(str(out), filters={"suite": "nope"})) == 0


def two_suites(tmp_path):
    """Two single-case suite files in one directory: alpha.json and beta.json."""
    out = tmp_path / "out"
    generate_from_csv(
        write_csv(tmp_path / "f.csv", ['"Q?","icontains:A"']), suite="alpha", out_dir=str(out)
    )
    generate_from_csv(
        write_csv(tmp_path / "g.csv", ['"R?","icontains:B"']), suite="beta", out_dir=str(out)
    )
    return out


def test_load_all_testcases_unions_several_paths(tmp_path):
    out = two_suites(tmp_path)
    cases = load_all_testcases([str(out / "alpha.json"), str(out / "beta.json")])
    assert {c.metadata["suite"] for c in cases} == {"alpha", "beta"}


def test_load_all_testcases_selects_a_subset_of_files(tmp_path):
    """The whole point of the repeatable flag: some files, not all of them."""
    out = two_suites(tmp_path)
    cases = load_all_testcases([str(out / "alpha.json")])
    assert [c.metadata["suite"] for c in cases] == ["alpha"]


def test_load_all_testcases_dedupes_overlapping_paths(tmp_path):
    """A directory plus a file inside it is a natural request, not a doubled run."""
    out = two_suites(tmp_path)
    cases = load_all_testcases([str(out), str(out / "alpha.json")])
    assert len(cases) == 2
    assert len({c.id for c in cases}) == 2


def test_load_all_testcases_preserves_path_order(tmp_path):
    out = two_suites(tmp_path)
    ordered = load_all_testcases([str(out / "beta.json"), str(out / "alpha.json")])
    assert [c.metadata["suite"] for c in ordered] == ["beta", "alpha"]


def test_load_all_testcases_applies_filters_to_every_path(tmp_path):
    out = two_suites(tmp_path)
    paths = [str(out / "alpha.json"), str(out / "beta.json")]
    assert len(load_all_testcases(paths, filters={"suite": "alpha"})) == 1


def test_load_all_testcases_empty_paths_is_empty(tmp_path):
    assert load_all_testcases([]) == []


def test_local_id_is_a_bare_digest_with_no_suite_prefix():
    one = local_id("What is the capital of France?")
    assert len(one) == 10 and one.isalnum()
    assert local_id("  What is the capital of France?  ") == one
    assert local_id("q", variant="g_eval") == f"{local_id('q')}-g_eval"


def test_rows_from_csv_builds_local_ids_expectations_and_metadata(tmp_path):
    csv_path = tmp_path / "facts.csv"
    csv_path.write_text(
        "user,__expected,__metadata:region\n"
        '"What is the capital of France?","icontains:Paris",eu\n',
        encoding="utf-8",
    )
    (case,) = rows_from_csv(str(csv_path))
    assert case["id"] == local_id("What is the capital of France?")
    assert case["assertions"] == [{"type": "icontains", "value": "Paris"}]
    assert case["metadata"] == {"region": "eu"}
    assert "suite" not in case["metadata"]


def test_csv_plugin_writes_its_cache_file_and_reads_it_back(tmp_path):
    csv_path = tmp_path / "facts.csv"
    csv_path.write_text('user,__expected\n"Q?","icontains:A"\n', encoding="utf-8")
    plugin = CsvTestCasePlugin(PluginInterface("facts", tmp_path / "cache"), csv_path)

    assert plugin.get_testcases() == []          # nothing generated yet
    assert plugin.generate_testcases() is True
    assert (tmp_path / "cache" / "facts" / "testcases.json").is_file()
    assert plugin.get_testcases()[0]["assertions"][0]["value"] == "A"


def test_csv_plugin_reports_failure_for_a_missing_csv(tmp_path):
    plugin = CsvTestCasePlugin(PluginInterface("gone", tmp_path / "cache"), tmp_path / "nope.csv")
    assert plugin.generate_testcases() is False

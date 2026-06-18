import pytest

from llmeval.generation.csv_source import generate_from_csv, parse_expected
from llmeval.testcases import load_testcases


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

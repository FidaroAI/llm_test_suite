import json

import pytest

from llmeval.plugins.loader import SourceError, discover, namespaced_cases, source_of


def write_json(root, name, cases):
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.json").write_text(json.dumps(cases), encoding="utf-8")


def test_discovers_top_level_json_files_in_name_order(tmp_path):
    write_json(tmp_path, "beta", [{"id": "b", "user": "?"}])
    write_json(tmp_path, "alpha", [{"id": "a", "user": "?"}])
    assert [s.name for s in discover(tmp_path)] == ["alpha", "beta"]
    assert all(not s.is_plugin for s in discover(tmp_path))


def test_a_bare_object_is_read_as_one_case(tmp_path):
    write_json(tmp_path, "one", {"id": "solo", "user": "?"})
    (source,) = discover(tmp_path)
    assert source.raw_testcases() == [{"id": "solo", "user": "?"}]


def test_json_inside_a_directory_is_not_a_source(tmp_path):
    (tmp_path / "nested").mkdir()
    write_json(tmp_path / "nested", "inner", [{"id": "i", "user": "?"}])
    assert [s.name for s in discover(tmp_path)] == []


def test_a_json_stem_colliding_with_a_directory_is_an_error(tmp_path):
    (tmp_path / "facts").mkdir()
    write_json(tmp_path, "facts", [{"id": "x", "user": "?"}])
    with pytest.raises(SourceError, match="facts"):
        discover(tmp_path)


def test_missing_root_is_empty_not_an_error(tmp_path):
    assert discover(tmp_path / "nope") == []


def test_ids_are_namespaced_by_source(tmp_path):
    write_json(tmp_path, "examples", [{"id": "greeting", "user": "hi"}])
    (source,) = discover(tmp_path)
    assert [c["id"] for c in namespaced_cases(source)] == ["examples.greeting"]


def test_duplicate_local_ids_are_an_error(tmp_path):
    write_json(tmp_path, "dupes", [{"id": "a", "user": "1"}, {"id": "a", "user": "2"}])
    (source,) = discover(tmp_path)
    with pytest.raises(SourceError, match="duplicate"):
        namespaced_cases(source)


def test_a_case_without_an_id_is_an_error(tmp_path):
    write_json(tmp_path, "anon", [{"user": "1"}])
    (source,) = discover(tmp_path)
    with pytest.raises(SourceError, match="no 'id'"):
        namespaced_cases(source)


def test_source_of_reads_the_id_prefix():
    assert source_of("simple_facts.a1b2c3d4e5") == "simple_facts"
    assert source_of("research_rubrics.abc-g_eval") == "research_rubrics"
    assert source_of("legacy-style-id") is None

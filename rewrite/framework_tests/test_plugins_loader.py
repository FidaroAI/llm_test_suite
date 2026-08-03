import json

import pytest

from llmeval.assertions.base import REGISTRY
from llmeval.plugins.loader import SourceError, discover, load, namespaced_cases, source_of


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


PLUGIN_SRC = '''
from llmeval.plugins import PluginInterface, TestCasePlugin
from .helper import GREETING


class P(TestCasePlugin):
    def __init__(self, interface):
        self.interface = interface

    def generate_testcases(self):
        return True

    def get_testcases(self):
        return [{"id": "one", "user": GREETING, "metadata": {"kind": "greet"}}]

    def get_custom_assertions(self):
        return {"always": lambda spec, output, ctx: None}


def get_plugin(interface):
    return P(interface)
'''


def make_plugin(root, name, source=PLUGIN_SRC):
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "__init__.py").write_text(source, encoding="utf-8")
    (directory / "helper.py").write_text('GREETING = "hi"\n', encoding="utf-8")
    return directory


def test_a_plugin_directory_becomes_a_source_with_relative_imports_working(tmp_path):
    make_plugin(tmp_path, "greeter")
    (source,) = discover(tmp_path)
    assert source.is_plugin
    assert source.raw_testcases() == [{"id": "one", "user": "hi", "metadata": {"kind": "greet"}}]


def test_a_directory_without_init_warns_and_is_skipped(tmp_path, caplog):
    (tmp_path / "notaplugin").mkdir()
    assert discover(tmp_path) == []
    assert "notaplugin" in caplog.text


def test_a_directory_without_get_plugin_warns_and_is_skipped(tmp_path, caplog):
    make_plugin(tmp_path, "broken", source="X = 1\n")
    assert discover(tmp_path) == []
    assert "get_plugin" in caplog.text


def test_a_plugin_that_raises_on_import_warns_and_is_skipped(tmp_path, caplog):
    make_plugin(tmp_path, "explodes", source="raise RuntimeError('boom')\n")
    assert discover(tmp_path) == []
    assert "boom" in caplog.text


def test_custom_assertions_are_registered_namespaced(tmp_path):
    make_plugin(tmp_path, "greeter")
    load(root=tmp_path)
    assert "greeter.always" in REGISTRY
    assert "always" not in REGISTRY


def test_the_cache_directory_is_a_sibling_of_the_root(tmp_path):
    root = tmp_path / "testcases"
    make_plugin(root, "greeter")
    loaded = load(root=root)
    iface = loaded.sources[0].plugin.interface
    assert iface.cache_directory() == tmp_path / ".llmeval.cache" / "greeter"


def test_load_returns_testcase_objects_with_namespaced_ids(tmp_path):
    make_plugin(tmp_path, "greeter")
    loaded = load(root=tmp_path)
    assert [c.id for c in loaded.cases] == ["greeter.one"]
    assert loaded.plugin_for["greeter.one"] is loaded.sources[0].plugin


def test_load_selects_named_sources_only(tmp_path):
    make_plugin(tmp_path, "greeter")
    write_json(tmp_path, "examples", [{"id": "e", "user": "?"}])
    assert [c.id for c in load(names=["examples"], root=tmp_path).cases] == ["examples.e"]


def test_load_rejects_an_unknown_source_name(tmp_path):
    write_json(tmp_path, "examples", [{"id": "e", "user": "?"}])
    with pytest.raises(SourceError, match="nope"):
        load(names=["nope"], root=tmp_path)


def test_load_applies_metadata_filters(tmp_path):
    make_plugin(tmp_path, "greeter")
    assert load(root=tmp_path, filters={"kind": "greet"}).cases
    assert not load(root=tmp_path, filters={"kind": "other"}).cases

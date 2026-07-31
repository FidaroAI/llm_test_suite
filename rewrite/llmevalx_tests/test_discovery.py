"""Menu contents come off disk and out of the store — never from a hardcoded list."""

import json

from llmeval.cache_key import CacheKey
from llmeval.store import Store
from llmevalx import discovery


def write_cases(directory, name, cases):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(cases), encoding="utf-8")


def case(test_id):
    return {"id": test_id, "user": "q?"}


# --------------------------------------------------------------------------- sources


PLUGIN = """
from llmeval.plugins import PluginInterface, TestCasePlugin


class P(TestCasePlugin):
    def __init__(self, cases):
        self.cases = cases

    def generate_testcases(self):
        return True

    def get_testcases(self):
        return self.cases


def get_plugin(interface):
    return P([{"id": "one", "user": "q?"}])
"""


def write_plugin(directory, name, source=PLUGIN):
    (directory / name).mkdir(parents=True, exist_ok=True)
    (directory / name / "__init__.py").write_text(source, encoding="utf-8")


def test_lists_plugins_and_json_files_with_their_case_counts(tmp_path):
    root = tmp_path / "tc"
    write_plugin(root, "facts")
    write_cases(root, "examples.json", [case("a1"), case("a2")])
    by_name = {s.name: s for s in discovery.list_sources(str(root))}
    assert (by_name["facts"].kind, by_name["facts"].count) == ("plugin", 1)
    assert (by_name["examples"].kind, by_name["examples"].count) == ("json", 2)


def test_label_pluralises_case_count(tmp_path):
    root = tmp_path / "tc"
    write_cases(root, "one.json", [case("x")])
    write_cases(root, "two.json", [case("x"), case("y")])
    labels = {s.name: s.label for s in discovery.list_sources(str(root))}
    assert "(1 case)" in labels["one"] and "(2 cases)" in labels["two"]


def test_a_bare_object_counts_as_one_case(tmp_path):
    write_cases(tmp_path / "tc", "single.json", case("only"))
    assert discovery.list_sources(str(tmp_path / "tc"))[0].count == 1


def test_an_ungenerated_plugin_is_offered_with_a_zero_count(tmp_path):
    """It is shown, not hidden: "generate this one" is what the user is here for."""
    root = tmp_path / "tc"
    write_plugin(root, "empty", source=PLUGIN.replace('[{"id": "one", "user": "q?"}]', "[]"))
    (source,) = discovery.list_sources(str(root))
    assert source.count == 0
    assert "not generated yet" in source.label


def test_non_json_files_are_ignored(tmp_path):
    (tmp_path / "tc").mkdir()
    (tmp_path / "tc" / "notes.txt").write_text("hello", encoding="utf-8")
    assert discovery.list_sources(str(tmp_path / "tc")) == []


def test_a_broken_file_greys_itself_out_rather_than_the_whole_wizard(tmp_path):
    write_cases(tmp_path / "tc", "good.json", [case("g")])
    (tmp_path / "tc" / "bad.json").write_text("{not json", encoding="utf-8")
    by_name = {s.name: s.count for s in discovery.list_sources(str(tmp_path / "tc"))}
    assert by_name == {"good": 1, "bad": 0}


def test_a_stem_clashing_with_a_directory_greys_out_the_whole_menu(tmp_path):
    """Ambiguous layout: better an empty menu than one that silently picks a winner."""
    root = tmp_path / "tc"
    write_plugin(root, "facts")
    write_cases(root, "facts.json", [case("x")])
    assert discovery.list_sources(str(root)) == []


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert discovery.list_sources(str(tmp_path / "nope")) == []


def test_only_plugins_are_generatable(tmp_path):
    root = tmp_path / "tc"
    write_plugin(root, "facts")
    write_cases(root, "examples.json", [case("a")])
    sources = discovery.list_sources(str(root))
    assert [s.name for s in discovery.generatable_sources(sources)] == ["facts"]


# --------------------------------------------------------------------------- providers


def write_config(directory, name, doc):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(doc), encoding="utf-8")


def test_lists_provider_configs(tmp_path):
    write_config(tmp_path / "c", "dev.json", {"name": "fidaro-dev", "model": "openai/auto"})
    providers = discovery.list_provider_configs(str(tmp_path / "c"))
    assert [(p.name, p.model) for p in providers] == [("fidaro-dev", "openai/auto")]


def test_the_judge_is_not_a_provider_under_test(tmp_path):
    write_config(tmp_path / "c", "judge_bedrock.json", {"name": "judge", "model": "bedrock/x"})
    write_config(tmp_path / "c", "dev.json", {"name": "fidaro-dev", "model": "openai/auto"})
    names = [p.name for p in discovery.list_provider_configs(str(tmp_path / "c"))]
    assert names == ["fidaro-dev"]


def test_unexpanded_env_in_base_url_still_lists(tmp_path):
    """The CLI expands ${ENV} when it matters; an unset var must not hide the config."""
    write_config(
        tmp_path / "c", "dev.json",
        {"name": "d", "model": "openai/auto", "base_url": "${NOT_SET_ANYWHERE}/v2"},
    )
    assert len(discovery.list_provider_configs(str(tmp_path / "c"))) == 1


def test_config_without_a_name_falls_back_to_the_filename(tmp_path):
    write_config(tmp_path / "c", "mystery.json", {"model": "openai/auto"})
    assert discovery.list_provider_configs(str(tmp_path / "c"))[0].name == "mystery"


def test_broken_provider_config_is_skipped(tmp_path):
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "bad.json").write_text("nope", encoding="utf-8")
    assert discovery.list_provider_configs(str(tmp_path / "c")) == []


# --------------------------------------------------------------------------- runs


def make_run(store, provider_name):
    key = CacheKey(fields={"p": provider_name}, canonical="{}", hash="h" + provider_name)
    return store.create_run(key, provider_name=provider_name)


def test_runs_are_listed_oldest_first(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    store = Store(db)
    first = make_run(store, "prod")
    second = make_run(store, "dev")
    store.close()
    listed = [r.run_id for r in discovery.list_runs(db)]
    assert listed == [first, second]


def test_run_label_shows_time_and_provider(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    store = Store(db)
    make_run(store, "fidaro-dev")
    store.close()
    label = discovery.list_runs(db)[0].label
    assert "fidaro-dev" in label and label.startswith("20") and "Z" in label


def test_limit_keeps_the_most_recent_runs(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    store = Store(db)
    ids = [make_run(store, "p") for _ in range(5)]
    store.close()
    listed = [r.run_id for r in discovery.list_runs(db, limit=2)]
    assert listed == ids[-2:]


def test_unfinished_runs_are_flagged(tmp_path):
    db = str(tmp_path / "db.sqlite3")
    store = Store(db)
    run_id = make_run(store, "p")
    store.close()
    assert discovery.list_runs(db)[0].unfinished is True

    store = Store(db)
    store.finish_run(run_id)
    store.close()
    assert discovery.list_runs(db)[0].unfinished is False


def test_missing_database_is_no_runs_not_an_error(tmp_path):
    assert discovery.list_runs(str(tmp_path / "absent.sqlite3")) == []

"""Menu contents come off disk and out of the store — never from a hardcoded list."""

import json

from llmeval.cache_key import CacheKey
from llmeval.store import Store
from porcelain import discovery


def write_cases(directory, name, cases):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(json.dumps(cases), encoding="utf-8")


def case(test_id, suite):
    return {"id": test_id, "user": "q?", "metadata": {"suite": suite}}


# --------------------------------------------------------------------------- test cases


def test_lists_json_files_with_counts_and_suites(tmp_path):
    write_cases(tmp_path / "tc", "alpha.json", [case("a1", "alpha"), case("a2", "alpha")])
    write_cases(tmp_path / "tc", "beta.json", [case("b1", "beta")])
    files = discovery.list_testcase_files(str(tmp_path / "tc"))
    assert [(f.name, f.count, f.suites) for f in files] == [
        ("alpha.json", 2, ("alpha",)),
        ("beta.json", 1, ("beta",)),
    ]


def test_label_pluralises_case_count(tmp_path):
    write_cases(tmp_path / "tc", "one.json", [case("x", "s")])
    write_cases(tmp_path / "tc", "two.json", [case("x", "s"), case("y", "s")])
    labels = [f.label for f in discovery.list_testcase_files(str(tmp_path / "tc"))]
    assert "(1 case)" in labels[0] and "(2 cases)" in labels[1]


def test_a_bare_object_counts_as_one_case(tmp_path):
    write_cases(tmp_path / "tc", "single.json", case("only", "s"))
    assert discovery.list_testcase_files(str(tmp_path / "tc"))[0].count == 1


def test_non_json_files_are_ignored(tmp_path):
    (tmp_path / "tc").mkdir()
    (tmp_path / "tc" / "notes.txt").write_text("hello", encoding="utf-8")
    assert discovery.list_testcase_files(str(tmp_path / "tc")) == []


def test_a_broken_file_greys_itself_out_rather_than_the_whole_wizard(tmp_path):
    write_cases(tmp_path / "tc", "good.json", [case("g", "alpha")])
    (tmp_path / "tc" / "bad.json").write_text("{not json", encoding="utf-8")
    assert [f.name for f in discovery.list_testcase_files(str(tmp_path / "tc"))] == ["good.json"]


def test_missing_directory_is_empty_not_an_error(tmp_path):
    assert discovery.list_testcase_files(str(tmp_path / "nope")) == []


def test_suites_come_from_the_chosen_files_only(tmp_path):
    write_cases(tmp_path / "tc", "alpha.json", [case("a", "alpha")])
    write_cases(tmp_path / "tc", "beta.json", [case("b", "beta")])
    files = discovery.list_testcase_files(str(tmp_path / "tc"))
    assert discovery.suites_in(files) == ["alpha", "beta"]
    assert discovery.suites_in([files[0]]) == ["alpha"]


def test_one_file_holding_several_suites(tmp_path):
    write_cases(tmp_path / "tc", "mixed.json", [case("a", "alpha"), case("b", "beta")])
    assert discovery.list_testcase_files(str(tmp_path / "tc"))[0].suites == ("alpha", "beta")


def test_cases_without_a_suite_label_contribute_nothing(tmp_path):
    write_cases(tmp_path / "tc", "plain.json", [{"id": "x", "user": "q?"}])
    files = discovery.list_testcase_files(str(tmp_path / "tc"))
    assert files[0].count == 1 and files[0].suites == ()


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


# --------------------------------------------------------------------------- suites


def test_generatable_suites_come_from_the_registry():
    suites = discovery.list_generatable_suites()
    by_name = {s.name: s for s in suites}
    assert "simple_facts" in by_name
    assert by_name["stock_prices"].network is True
    assert by_name["simple_facts"].network is False


def test_network_suites_are_labelled():
    label = {s.name: s.label for s in discovery.list_generatable_suites()}["stock_prices"]
    assert "network" in label


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

import json

from llmeval.generation.suites import (
    GenPaths,
    SUITES,
    generate_suite,
    suite_names,
    write_suite,
)


def test_all_six_suites_registered():
    assert set(suite_names()) == {
        "simple_facts", "simple_facts_regressions", "agentharm_refusal",
        "multifaceted", "research_rubrics", "stock_prices",
    }


def _paths(tmp_path, cfg):
    data = tmp_path / "data"
    (data / "classifications").mkdir(parents=True)
    src = tmp_path / "src"
    src.mkdir()
    config = tmp_path / "gen.json"
    config.write_text(json.dumps(cfg))
    return GenPaths(
        data_dir=str(data),
        classifications_dir=str(data / "classifications"),
        generation_sources_dir=str(src),
        config_path=str(config),
    )


def test_generate_csv_suite_applies_config_and_classification(tmp_path):
    paths = _paths(tmp_path, {"simple_facts": {"number_to_generate": None}})
    (tmp_path / "src" / "simple_facts.csv").write_text(
        "user,__expected\n" '"What is the capital of France?","icontains:Paris"\n'
    )
    # classification keyed by prompt hash
    from llmeval.generation.classification import prompt_key
    (tmp_path / "data" / "classifications" / "simple_facts.json").write_text(json.dumps(
        {"classifications": {prompt_key("What is the capital of France?"):
                             {"request_type": "factual_qa", "domain": "history_society"}}}))

    cases = generate_suite("simple_facts", paths)
    assert len(cases) == 1
    assert cases[0]["assertions"][0]["type"] == "icontains"
    assert cases[0]["metadata"]["suite"] == "simple_facts"
    assert cases[0]["metadata"]["request_type"] == "factual_qa"
    assert "config" in cases[0]["metadata"]


def test_generate_dataset_suite_dispatches(tmp_path):
    paths = _paths(tmp_path, {"agentharm_refusal": {"number_to_generate": None}})
    (tmp_path / "data" / "agentharm.json").write_text(json.dumps(
        [{"id": "a1", "category": "Fraud", "prompt": "do bad"}]))
    cases = generate_suite("agentharm_refusal", paths)
    assert cases[0]["assertions"][0]["type"] == "rubric"


def test_write_suite_emits_json_file(tmp_path):
    paths = _paths(tmp_path, {"simple_facts": {"number_to_generate": None}})
    (tmp_path / "src" / "simple_facts.csv").write_text(
        "user,__expected\n" '"Q?","icontains:A"\n')
    out = tmp_path / "out"
    n = write_suite("simple_facts", str(out), paths)
    assert n == 1
    assert (out / "simple_facts.json").exists()

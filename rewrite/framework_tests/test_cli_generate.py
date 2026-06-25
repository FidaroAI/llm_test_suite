import json

from llmeval.cli import main


def _setup(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    (src / "simple_facts.csv").write_text(
        "user,__expected\n" '"What is the capital of France?","icontains:Paris"\n')
    data = tmp_path / "data"
    (data / "classifications").mkdir(parents=True)
    (data / "agentharm.json").write_text(json.dumps(
        [{"id": "a1", "category": "Fraud", "prompt": "do bad"}]))
    cfg = tmp_path / "gen.json"
    cfg.write_text(json.dumps({
        "simple_facts": {"number_to_generate": None},
        "agentharm_refusal": {"number_to_generate": None},
    }))
    return src, data, cfg


def _common(tmp_path):
    src, data, cfg = _setup(tmp_path)
    return [
        "--data-dir", str(data),
        "--classifications-dir", str(data / "classifications"),
        "--sources-dir", str(src),
        "--config", str(cfg),
    ]


def test_generate_single_suite(tmp_path):
    out = tmp_path / "tc"
    rc = main(["generate", "--suite", "simple_facts", "--out", str(out), *_common(tmp_path)])
    assert rc == 0
    cases = json.loads((out / "simple_facts.json").read_text())
    assert cases[0]["assertions"][0]["value"] == "Paris"


def test_generate_repeatable_suite_flag(tmp_path):
    out = tmp_path / "tc"
    rc = main(["generate", "--suite", "simple_facts", "--suite", "agentharm_refusal",
               "--out", str(out), *_common(tmp_path)])
    assert rc == 0
    assert (out / "simple_facts.json").exists()
    assert (out / "agentharm_refusal.json").exists()


def test_generate_all_skips_network_suites(tmp_path):
    out = tmp_path / "tc"
    rc = main(["generate", "--all", "--out", str(out), *_common(tmp_path)])
    assert rc == 0
    # stock_prices is a network suite -> not emitted by --all
    assert not (out / "stock_prices.json").exists()
    assert (out / "simple_facts.json").exists()


def test_generate_unknown_suite_errors(tmp_path):
    out = tmp_path / "tc"
    rc = main(["generate", "--suite", "nope", "--out", str(out), *_common(tmp_path)])
    assert rc != 0

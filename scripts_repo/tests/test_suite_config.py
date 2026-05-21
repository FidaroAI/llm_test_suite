"""Unit tests for the shared suite-generation config helper."""

import importlib.util
import json
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "tests" / "suite_config.py"
_spec = importlib.util.spec_from_file_location("suite_config", _MOD_PATH)
suite_config = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(suite_config)


def test_suite_name_strips_gen_suffix():
    assert suite_config.suite_name("/x/tests/multifaceted_gen.py") == "multifaceted"
    assert suite_config.suite_name("agentharm_refusal_gen.py") == "agentharm_refusal"


def _write_config(tmp_path, data):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_load_uses_defaults_when_file_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(tmp_path / "nope.json"))
    cfg = suite_config.load("/x/tests/multifaceted_gen.py")
    assert cfg.suite == "multifaceted"
    assert cfg.number_to_generate is None
    assert cfg.randomize_selection is False
    assert cfg.random_seed == 0
    assert cfg.max_rubrics is None


def test_load_uses_defaults_when_suite_key_absent(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"other": {"number_to_generate": 9}})
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(path))
    cfg = suite_config.load("/x/tests/multifaceted_gen.py")
    assert cfg.number_to_generate is None


def test_load_merges_suite_values_over_defaults(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        {"multifaceted": {"number_to_generate": 5, "randomize_selection": True,
                          "random_seed": 42, "max_rubrics": 3}},
    )
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(path))
    cfg = suite_config.load("/x/tests/multifaceted_gen.py")
    assert cfg.number_to_generate == 5
    assert cfg.randomize_selection is True
    assert cfg.random_seed == 42
    assert cfg.max_rubrics == 3


def _cfg(suite="s", number_to_generate=None, randomize_selection=False,
         random_seed=0, max_rubrics=None):
    return suite_config.SuiteConfig(suite, {
        "number_to_generate": number_to_generate,
        "randomize_selection": randomize_selection,
        "random_seed": random_seed,
        "max_rubrics": max_rubrics,
    })


def _tests(n):
    return [{"description": str(i), "metadata": {"suite": "s"}} for i in range(n)]


def test_select_no_limit_returns_all_in_order():
    cfg = _cfg(number_to_generate=None)
    out = cfg.select(_tests(5))
    assert [t["description"] for t in out] == ["0", "1", "2", "3", "4"]


def test_select_caps_to_number_to_generate_without_randomize():
    cfg = _cfg(number_to_generate=3)
    out = cfg.select(_tests(10))
    assert [t["description"] for t in out] == ["0", "1", "2"]


def test_select_zero_yields_empty():
    cfg = _cfg(number_to_generate=0)
    assert cfg.select(_tests(10)) == []


def test_select_randomize_is_deterministic_for_a_seed():
    a = _cfg(number_to_generate=5, randomize_selection=True, random_seed=42).select(_tests(20))
    b = _cfg(number_to_generate=5, randomize_selection=True, random_seed=42).select(_tests(20))
    assert [t["description"] for t in a] == [t["description"] for t in b]


def test_select_randomize_differs_from_sequential():
    rand = _cfg(number_to_generate=5, randomize_selection=True, random_seed=42).select(_tests(20))
    seq = _cfg(number_to_generate=5, randomize_selection=False).select(_tests(20))
    assert [t["description"] for t in rand] != [t["description"] for t in seq]


def test_select_stamps_full_config_into_metadata():
    cfg = _cfg(number_to_generate=2, randomize_selection=True, random_seed=7, max_rubrics=4)
    out = cfg.select(_tests(5))
    for t in out:
        assert t["metadata"]["config"] == {
            "number_to_generate": 2,
            "randomize_selection": True,
            "random_seed": 7,
            "max_rubrics": 4,
        }

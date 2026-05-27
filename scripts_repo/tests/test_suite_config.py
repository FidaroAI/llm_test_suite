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
    assert cfg.number_to_generate == 0  # off by default (see DEFAULTS)
    assert cfg.randomize_selection is False
    assert cfg.random_seed == 0
    assert cfg.max_rubrics is None
    assert cfg.stratify is None


def test_load_uses_defaults_when_suite_key_absent(tmp_path, monkeypatch):
    path = _write_config(tmp_path, {"other": {"number_to_generate": 9}})
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(path))
    cfg = suite_config.load("/x/tests/multifaceted_gen.py")
    assert cfg.number_to_generate == 0  # off by default (see DEFAULTS)


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


def test_load_uses_default_block_for_absent_suite(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        {
            "default": {"number_to_generate": 7, "randomize_selection": True,
                        "random_seed": 3},
            "other": {"number_to_generate": 9},
        },
    )
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(path))
    cfg = suite_config.load("/x/tests/multifaceted_gen.py")
    assert cfg.number_to_generate == 7
    assert cfg.randomize_selection is True
    assert cfg.random_seed == 3
    # fields the default block omits fall back to the code DEFAULTS
    assert cfg.max_rubrics is None
    assert cfg.stratify is None


def test_load_present_suite_ignores_default_block(tmp_path, monkeypatch):
    path = _write_config(
        tmp_path,
        {
            "default": {"number_to_generate": 7, "max_rubrics": 4},
            "multifaceted": {"number_to_generate": 5},
        },
    )
    monkeypatch.setenv("SUITE_GENERATION_CONFIG_FILE", str(path))
    cfg = suite_config.load("/x/tests/multifaceted_gen.py")
    assert cfg.number_to_generate == 5
    # the default block is not merged in; the unspecified field uses code DEFAULTS
    assert cfg.max_rubrics is None


def _cfg(suite="s", number_to_generate=None, randomize_selection=False,
         random_seed=0, max_rubrics=None, stratify=None):
    return suite_config.SuiteConfig(suite, {
        "number_to_generate": number_to_generate,
        "randomize_selection": randomize_selection,
        "random_seed": random_seed,
        "max_rubrics": max_rubrics,
        "stratify": stratify,
    })


def _tests(n):
    return [{"description": str(i), "metadata": {"suite": "s"}} for i in range(n)]


def _tests_by(groups):
    """Build tests carrying a ``domain`` metadata value: groups maps value->count."""
    out = []
    i = 0
    for value, count in groups.items():
        for _ in range(count):
            out.append({"description": f"{value}{i}", "metadata": {"domain": value}})
            i += 1
    return out


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
            "stratify": None,
        }


def test_stratify_takes_per_group_quota_from_each_value():
    cfg = _cfg(stratify={"by": "domain", "per_group": 2})
    out = cfg.select(_tests_by({"a": 5, "b": 3, "c": 1}))
    by_value = {}
    for t in out:
        by_value.setdefault(t["metadata"]["domain"], 0)
        by_value[t["metadata"]["domain"]] += 1
    assert by_value == {"a": 2, "b": 2, "c": 1}


def test_stratify_groups_restricts_and_orders():
    cfg = _cfg(stratify={"by": "domain", "per_group": 1, "groups": ["c", "a"]})
    out = cfg.select(_tests_by({"a": 2, "b": 2, "c": 2}))
    assert [t["metadata"]["domain"] for t in out] == ["c", "a"]


def test_stratify_then_number_to_generate_caps_overall():
    cfg = _cfg(number_to_generate=3, stratify={"by": "domain", "per_group": 2})
    out = cfg.select(_tests_by({"a": 5, "b": 5}))
    assert len(out) == 3


def test_stratify_with_randomize_is_deterministic_for_a_seed():
    mk = lambda: _cfg(randomize_selection=True, random_seed=42,
                      stratify={"by": "domain", "per_group": 2})
    a = mk().select(_tests_by({"a": 5, "b": 5}))
    b = mk().select(_tests_by({"a": 5, "b": 5}))
    assert [t["description"] for t in a] == [t["description"] for t in b]

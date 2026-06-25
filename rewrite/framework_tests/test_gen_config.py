import json

import pytest

from llmeval.generation.config import DEFAULTS, SuiteGenConfig, load_suite_config


def _tests(*metas):
    return [{"id": f"t{i}", "user": f"q{i}", "assertions": [], "metadata": dict(m)}
            for i, m in enumerate(metas)]


def test_defaults_are_off_by_default():
    cfg = SuiteGenConfig("s", dict(DEFAULTS))
    # number_to_generate 0 => an un-configured suite emits nothing
    assert cfg.select(_tests({}, {}, {})) == []


def test_cap_to_number_to_generate():
    cfg = SuiteGenConfig("s", {**DEFAULTS, "number_to_generate": 2})
    chosen = cfg.select(_tests({}, {}, {}, {}))
    assert len(chosen) == 2


def test_null_number_keeps_all():
    cfg = SuiteGenConfig("s", {**DEFAULTS, "number_to_generate": None})
    assert len(cfg.select(_tests({}, {}, {}))) == 3


def test_seeded_shuffle_is_reproducible():
    values = lambda: _tests(*[{"i": i} for i in range(10)])
    cfg = SuiteGenConfig("s", {**DEFAULTS, "number_to_generate": None,
                               "randomize_selection": True, "random_seed": 7})
    a = [t["metadata"]["i"] for t in cfg.select(values())]
    b = [t["metadata"]["i"] for t in cfg.select(values())]
    assert a == b
    assert a != list(range(10))  # actually shuffled


def test_stratify_takes_per_group():
    cfg = SuiteGenConfig("s", {**DEFAULTS, "number_to_generate": None,
                               "stratify": {"by": "domain", "per_group": 1}})
    chosen = cfg.select(_tests({"domain": "a"}, {"domain": "a"}, {"domain": "b"}))
    domains = [t["metadata"]["domain"] for t in chosen]
    assert sorted(domains) == ["a", "b"]


def test_select_stamps_resolved_config():
    cfg = SuiteGenConfig("s", {**DEFAULTS, "number_to_generate": 1})
    chosen = cfg.select(_tests({}))
    assert chosen[0]["metadata"]["config"]["number_to_generate"] == 1


def test_load_reads_suite_block(tmp_path):
    path = tmp_path / "gen.json"
    path.write_text(json.dumps({"multifaceted": {"number_to_generate": 5, "max_rubrics": 3}}))
    cfg = load_suite_config("multifaceted", str(path))
    assert cfg.number_to_generate == 5
    assert cfg.max_rubrics == 3


def test_load_falls_back_to_default_block(tmp_path):
    path = tmp_path / "gen.json"
    path.write_text(json.dumps({"default": {"number_to_generate": 2}}))
    cfg = load_suite_config("not_listed", str(path))
    assert cfg.number_to_generate == 2


def test_load_missing_file_uses_code_defaults(tmp_path):
    cfg = load_suite_config("x", str(tmp_path / "nope.json"))
    assert cfg.number_to_generate == DEFAULTS["number_to_generate"]

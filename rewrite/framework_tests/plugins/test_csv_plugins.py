"""The CSV-backed plugins, exercised through the loader against the real testcases/ tree."""

from pathlib import Path

import pytest

from llmeval.plugins.loader import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]   # .../rewrite
CSV_PLUGINS = ["simple_facts", "simple_facts_regressions"]


@pytest.mark.parametrize("name", CSV_PLUGINS)
def test_plugin_generates_and_serves_namespaced_cases(name, tmp_path, monkeypatch):
    # Generate into a scratch cache so the developer's real cache is untouched.
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "testcases"
    root.mkdir()
    (root / name).symlink_to(PROJECT_ROOT / "testcases" / name)

    loaded = load(names=[name], root=root)
    (source,) = loaded.sources
    assert source.is_plugin
    assert loaded.cases == []                       # nothing generated yet

    assert source.plugin.generate_testcases() is True
    cases = load(names=[name], root=root).cases
    assert cases, "expected the CSV to produce test cases"
    assert all(c.id.startswith(f"{name}.") for c in cases)
    assert all(c.assertions for c in cases)

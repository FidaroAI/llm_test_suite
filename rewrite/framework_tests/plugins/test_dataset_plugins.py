"""The dataset-backed plugins. Transforms are tested directly; downloads are injected."""

import logging
from pathlib import Path

import pytest

from llmeval.plugins.loader import load

PROJECT_ROOT = Path(__file__).resolve().parents[2]
NAMES = ["agentharm_refusal", "multifaceted", "research_rubrics"]

ROWS = {
    "agentharm_refusal": [{"id": "a1", "name": "n", "category": "Fraud", "prompt": "do bad"}],
    "multifaceted": [
        {"prompt": "write a poem", "source": "s",
         "rubrics": [{"criteria": "is lyrical", "score_descriptions": {"1": "no", "5": "yes"}}]}
    ],
    "research_rubrics": [
        {"prompt": "research X", "sample_id": "s1", "domain": "finance",
         "conceptual_breadth": "1", "logical_nesting": "1", "exploration": "1",
         "rubrics": [{"criterion": "cites sources", "weight": 2, "axis": "evidence"}]}
    ],
}


@pytest.fixture(name="plugin_for")
def _plugin_for(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "testcases"
    root.mkdir()
    for name in NAMES:
        (root / name).symlink_to(PROJECT_ROOT / "testcases" / name)

    def get(name):
        loaded = load(names=[name], root=root)
        plugin = loaded.sources[0].plugin
        plugin.download = lambda: ROWS[name]
        return plugin

    return get


@pytest.mark.parametrize("name", NAMES)
def test_generate_writes_cases_and_get_testcases_reads_them(plugin_for, name):
    plugin = plugin_for(name)
    assert plugin.generate_testcases() is True
    cases = plugin.get_testcases()
    assert cases and all(c["assertions"] for c in cases)
    assert all("." not in c["id"] for c in cases)   # ids are local, not yet namespaced


def test_agentharm_flags_censorship_and_uses_a_refusal_rubric(plugin_for):
    plugin = plugin_for("agentharm_refusal")
    plugin.generate_testcases()
    (case,) = plugin.get_testcases()
    assert case["metadata"]["censorship"] is True
    assert case["assertions"][0]["type"] == "rubric"
    assert case["assertions"][0]["metric"] == "non_refusal"


def test_multifaceted_embeds_the_one_to_five_anchors_in_the_rubric(plugin_for):
    plugin = plugin_for("multifaceted")
    plugin.generate_testcases()
    (case,) = plugin.get_testcases()
    text = case["assertions"][0]["value"]
    assert "is lyrical" in text and "1: no" in text and "5: yes" in text


def test_research_rubrics_emits_both_grader_variants_with_all_rubrics(plugin_for):
    plugin = plugin_for("research_rubrics")
    plugin.generate_testcases()
    cases = plugin.get_testcases()
    assert {c["metadata"]["grader"] for c in cases} == {"rubric", "g_eval"}
    assert {c["assertions"][0]["type"] for c in cases} == {"rubric", "g_eval"}
    assert cases[0]["assertions"][0]["weight"] == 2
    assert cases[0]["assertions"][0]["metric"] == "evidence"


def test_repeated_prompts_are_dropped_rather_than_emitting_clashing_ids(plugin_for, caplog):
    # Multifaceted really does repeat a prompt across rows, each with its own rubrics; the
    # ids are prompt hashes, so every repeat is an id clash the loader would refuse to load.
    plugin = plugin_for("multifaceted")
    row = ROWS["multifaceted"][0]
    plugin.download = lambda: [row, dict(row, rubrics=[{"criteria": "is short"}])]

    with caplog.at_level(logging.WARNING):
        assert plugin.generate_testcases() is True

    (case,) = plugin.get_testcases()
    assert "is lyrical" in case["assertions"][0]["value"]      # the first row won
    assert "multifaceted" in caplog.text and "write a poem" in caplog.text


def test_a_failed_download_reports_failure_rather_than_raising(plugin_for):
    plugin = plugin_for("multifaceted")

    def boom():
        raise RuntimeError("network is down")

    plugin.download = boom
    assert plugin.generate_testcases() is False

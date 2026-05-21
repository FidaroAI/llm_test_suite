"""Unit tests for the Multifaceted-Bench generator (no network/dataset needed)."""

import importlib.util
from pathlib import Path

_GEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "multifaceted_gen.py"
_spec = importlib.util.spec_from_file_location("multifaceted_gen", _GEN_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _row():
    return {
        "source": "synthetic",
        "prompt": "Design a training plan.",
        "rubrics": [
            {"criteria": "Is it concise?",
             "score_descriptions": {"1": "verbose", "2": "b", "3": "c",
                                    "4": "d", "5": "perfectly concise"}},
            {"criteria": "Is it accurate?",
             "score_descriptions": {"1": "wrong", "2": "b", "3": "c",
                                    "4": "d", "5": "correct"}},
        ],
    }


def test_row_to_test_one_llm_rubric_per_criterion():
    test = mod._row_to_test(_row(), 3)
    assert len(test["assert"]) == 2
    assert all(a["type"] == "llm-rubric" for a in test["assert"])
    assert test["vars"]["user"] == "Design a training plan."
    assert test["metadata"]["suite"] == "multifaceted"
    assert test["metadata"]["sample_id"] == "3"
    assert test["description"] == "multifaceted[synthetic] 3"
    assert test["threshold"] == mod.PASS_THRESHOLD


def test_rubric_value_embeds_criteria_and_1_to_5_anchors():
    test = mod._row_to_test(_row(), 0)
    value = test["assert"][0]["value"]
    assert "Is it concise?" in value
    assert "1: verbose" in value
    assert "5: perfectly concise" in value


def test_rubric_prompt_override_normalizes_1_to_5():
    test = mod._row_to_test(_row(), 0)
    rubric_prompt = test["options"]["rubricPrompt"]
    assert isinstance(rubric_prompt, list) and len(rubric_prompt) == 2
    assert rubric_prompt[0]["role"] == "system"
    assert "(rating - 1) / 4" in rubric_prompt[0]["content"]
    user = rubric_prompt[1]["content"]
    assert "{{ output }}" in user and "{{ rubric }}" in user


def test_row_with_no_rubrics_yields_no_asserts():
    test = mod._row_to_test({"source": "s", "prompt": "p", "rubrics": []}, 0)
    assert test["assert"] == []


def test_max_num_criteria_caps_asserts():
    test = mod._row_to_test(_row(), 0, max_num_criteria=1)
    assert len(test["assert"]) == 1

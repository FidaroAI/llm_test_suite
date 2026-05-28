"""Unit tests for the research-rubrics generator (no network/dataset needed)."""

import importlib.util
from pathlib import Path

_GEN_PATH = Path(__file__).resolve().parents[2] / "tests" / "research_rubrics_gen.py"
_spec = importlib.util.spec_from_file_location("research_rubrics_gen", _GEN_PATH)
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _row():
    return {
        "sample_id": "abc123",
        "domain": "Current Events",
        "prompt": "Summarise the day's schedule.",
        "conceptual_breadth": "Simple",
        "logical_nesting": "Intermediate",
        "exploration": "Medium",
        "rubrics": [
            {"criterion": "Covers the full time range", "weight": 5,
             "axis": "Explicit Criteria"},
            {"criterion": "Has clear section headers", "weight": 2,
             "axis": "Communication Quality"},
            {"criterion": "Recommends illegal activity", "weight": -5,
             "axis": "Implicit Criteria"},
        ],
    }


def test_llm_rubric_variant_one_assertion_per_criterion():
    test = mod._row_to_test(_row(), "llm-rubric")
    assert len(test["assert"]) == 3
    assert all(a["type"] == "llm-rubric" for a in test["assert"])
    assert test["metadata"]["suite"] == "research_rubrics"
    assert test["metadata"]["grader"] == "llm-rubric"
    assert test["description"] == "researchrubrics[Current Events] abc123"
    assert test["threshold"] == mod.PASS_THRESHOLD


def test_geval_variant_emits_geval_assertions_and_suffix():
    test = mod._row_to_test(_row(), "g-eval")
    assert len(test["assert"]) == 3
    assert all(a["type"] == "g-eval" for a in test["assert"])
    assert test["metadata"]["grader"] == "g-eval"
    assert test["description"] == "researchrubrics[Current Events] abc123 (g-eval)"


def test_weight_and_axis_map_through_for_both_graders():
    for grader in ("llm-rubric", "g-eval"):
        first = mod._row_to_test(_row(), grader)["assert"][0]
        assert first["value"] == "Covers the full time range"
        assert first["weight"] == 5
        assert first["metric"] == "Explicit Criteria"
        # Negative penalty weights are preserved, not dropped.
        penalty = mod._row_to_test(_row(), grader)["assert"][2]
        assert penalty["weight"] == -5


def test_max_rubrics_caps_llm_rubric_but_geval_uses_all():
    capped = mod._row_to_test(_row(), "llm-rubric", max_num_criteria=1)
    assert len(capped["assert"]) == 1
    uncapped = mod._row_to_test(_row(), "g-eval", max_num_criteria=None)
    assert len(uncapped["assert"]) == 3


def test_grading_transform_attached_to_geval_assertions():
    # classification.augment defaults every assertion to the reasoning-strip
    # transform; g-eval assertions must get it too.
    test = mod._row_to_test(_row(), "g-eval")
    assert all(
        a.get("transform") == "file://hooks/strip_before_triple_newline.py"
        for a in test["assert"]
    )


def test_row_with_no_rubrics_yields_no_asserts():
    test = mod._row_to_test({"sample_id": "x", "domain": "d", "prompt": "p",
                             "rubrics": []}, "g-eval")
    assert test["assert"] == []

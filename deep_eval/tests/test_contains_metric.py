"""Pure-unit proof that DeepEval's test-case + metric + assert machinery runs —
no network, no LLM, no API key. This is the deterministic counterpart to the
parent suite's `pytest tests/python` for custom assertions.

Run: pytest deep_eval/tests/test_contains_metric.py -v
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from deepeval_demo.metrics import ContainsMetric


def test_contains_passes_on_correct_fact():
    # A hard-coded answer stands in for the model output, so this exercises the
    # scoring path (LLMTestCase -> ContainsMetric -> assert_test) with no LLM.
    case = LLMTestCase(
        input="What is the capital of France?",
        actual_output="The capital of France is Paris.",
    )
    assert_test(case, [ContainsMetric("Paris")])


def test_contains_is_case_insensitive():
    metric = ContainsMetric("paris")
    case = LLMTestCase(input="q", actual_output="PARIS is lovely")
    assert metric.measure(case) == 1.0
    assert metric.is_successful()


def test_contains_fails_when_substring_absent():
    metric = ContainsMetric("Paris")
    case = LLMTestCase(input="q", actual_output="The capital of France is Lyon.")
    assert metric.measure(case) == 0.0
    assert not metric.is_successful()
    assert "missing" in metric.reason.lower()


def test_contains_match_all_requires_every_substring():
    metric = ContainsMetric(["Paris", "France"], match_all=True)
    case = LLMTestCase(input="q", actual_output="Paris is a city.")  # no "France"
    assert metric.measure(case) == 0.0


def test_contains_match_any_needs_only_one():
    metric = ContainsMetric(["Paris", "Lyon"], match_all=False)
    case = LLMTestCase(input="q", actual_output="I think it's Lyon.")
    assert metric.measure(case) == 1.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

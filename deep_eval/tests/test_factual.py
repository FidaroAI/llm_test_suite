"""Factual test (live): ask the model under test a simple question and assert the
answer contains the expected fact. The DeepEval analog of the parent suite's
`tests/simple_facts.csv` rows like `"capital of France" -> icontains:Paris`.

Needs a reachable MODEL UNDER TEST (MUT_BASE_URL / MUT_MODEL_ID, or VLLM_*).
Skips cleanly if unconfigured. No judge / API key required.

Run: pytest deep_eval/tests/test_factual.py -v
"""
import pytest
from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from _live import generate_or_skip
from deepeval_demo.config import model_under_test_config
from deepeval_demo.metrics import ContainsMetric

_MUT = model_under_test_config()

# (question, expected substring) — same shape as simple_facts.csv.
FACTS = [
    ("What is the capital of France?", "Paris"),
    ("What is the capital of Canada?", "Ottawa"),
    ("What software company is headquartered in Redmond, Washington?", "Microsoft"),
]


@pytest.mark.skipif(not _MUT.is_configured, reason=f"model under test unconfigured: {_MUT.missing}")
@pytest.mark.parametrize("question,expected", FACTS, ids=[f[1] for f in FACTS])
def test_factual_answer_contains_expected(question, expected):
    answer = generate_or_skip(question)
    case = LLMTestCase(input=question, actual_output=answer, expected_output=expected)
    assert_test(case, [ContainsMetric(expected)])

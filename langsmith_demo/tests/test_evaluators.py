"""Local proof that the evaluator logic runs — no LangSmith server, no network,
no API key. The LangSmith analog of the deep_eval demo's ContainsMetric test.

We feed the evaluators lightweight stand-ins for LangSmith's Run/Example objects
(anything with `.outputs` / `.inputs`), and inject a fake judge for the rubric.

Run: pytest langsmith_demo/tests/test_evaluators.py -v
"""
from types import SimpleNamespace

import pytest

from ls_demo.evaluators import (
    contains_check,
    make_contains_evaluator,
    make_rubric_evaluator,
)


def _run(answer):
    return SimpleNamespace(outputs={"answer": answer})


def _example(expected=None, question=None):
    return SimpleNamespace(
        outputs={"expected": expected} if expected is not None else {},
        inputs={"question": question} if question is not None else {},
    )


# --- deterministic contains ---------------------------------------------------
def test_contains_check_pass_and_fail():
    assert contains_check("The capital of France is Paris.", "Paris")[0] is True
    assert contains_check("The capital of France is Lyon.", "Paris")[0] is False


def test_contains_check_is_case_insensitive():
    assert contains_check("PARIS is lovely", "paris")[0] is True


def test_contains_evaluator_returns_langsmith_shape():
    evaluator = make_contains_evaluator()
    result = evaluator(_run("...is Paris."), _example(expected="Paris"))
    assert result == {"key": "contains", "score": 1.0, "comment": "output contains 'Paris'"}

    fail = evaluator(_run("...is Lyon."), _example(expected="Paris"))
    assert fail["score"] == 0.0
    assert "missing" in fail["comment"]


# --- LLM-as-judge rubric (fake judge, no network) -----------------------------
class _FakeJudge:
    def __init__(self, score, reason="because"):
        self._score, self._reason = score, reason

    def score(self, question, answer):
        return self._score, self._reason


def test_rubric_evaluator_passes_above_threshold():
    evaluator = make_rubric_evaluator(judge=_FakeJudge(0.9, "specific and empathetic"), threshold=0.7)
    result = evaluator(_run("Sorry about #4471, we'll ship a replacement today."),
                       _example(question="customer prompt"))
    assert result["key"] == "rubric_quality"
    assert result["score"] == 0.9
    assert result["comment"].startswith("PASS")


def test_rubric_evaluator_fails_below_threshold():
    evaluator = make_rubric_evaluator(judge=_FakeJudge(0.3, "generic"), threshold=0.7)
    result = evaluator(_run("Thanks for your message."), _example(question="customer prompt"))
    assert result["score"] == 0.3
    assert result["comment"].startswith("FAIL")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

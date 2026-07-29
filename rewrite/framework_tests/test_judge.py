import pytest

from llmeval.assertions import grade_assertion
from llmeval.assertions.base import GradeContext
from llmeval.models import AssertionSpec


class FakeJudge:
    def __init__(self, reply):
        self.reply = reply
        self.prompts = []

    def __call__(self, prompt):
        self.prompts.append(prompt)
        return self.reply


def ctx(judge, user_text="What is the capital of France?"):
    return GradeContext(user_text=user_text, judge=judge)


def test_rubric_parses_score_and_passes_above_threshold():
    j = FakeJudge('{"score": 0.9, "reason": "accurate and complete"}')
    spec = AssertionSpec(type="rubric", value="answer is accurate")
    r = grade_assertion(spec, "The capital is Paris.", ctx(j))
    assert r.score == 0.9
    assert r.passed is True
    assert "accurate" in r.reason


def test_rubric_below_threshold_fails():
    j = FakeJudge('{"score": 0.2, "reason": "wrong"}')
    spec = AssertionSpec(type="rubric", value="answer is accurate")
    assert grade_assertion(spec, "The capital is Berlin.", ctx(j)).passed is False


def test_rubric_custom_threshold():
    j = FakeJudge('{"score": 0.6}')
    spec = AssertionSpec(type="rubric", value="x", params={"threshold": 0.7})
    assert grade_assertion(spec, "ans", ctx(j)).passed is False


def test_rubric_prompt_includes_question_criterion_and_answer():
    j = FakeJudge('{"score": 1.0}')
    spec = AssertionSpec(type="rubric", value="is concise")
    grade_assertion(spec, "Paris.", ctx(j, user_text="capital of France?"))
    p = j.prompts[0]
    assert "is concise" in p and "capital of France?" in p and "Paris." in p


def test_geval_normalizes_scale_score():
    j = FakeJudge("Reasoning: solid.\nScore: 8/10")
    spec = AssertionSpec(type="g_eval", value="coherence and correctness")
    r = grade_assertion(spec, "The capital is Paris.", ctx(j))
    assert r.score == pytest.approx(0.8)
    assert r.passed is True


def test_geval_handles_out_of_ten_phrasing():
    j = FakeJudge("I rate this 3 out of 10.")
    spec = AssertionSpec(type="g_eval", value="x")
    assert grade_assertion(spec, "bad answer", ctx(j)).score == pytest.approx(0.3)


def test_judge_sees_the_output_verbatim_by_default():
    # Providers return the answer alone, so nothing is stripped on the way to the judge.
    j = FakeJudge('{"score": 1.0}')
    spec = AssertionSpec(type="rubric", value="accurate")
    grade_assertion(spec, "Paris is the capital.", ctx(j))
    assert "Paris is the capital." in j.prompts[0]


def test_judge_sees_stripped_answer_when_transform_opted_in():
    j = FakeJudge('{"score": 1.0}')
    spec = AssertionSpec(type="rubric", value="accurate", transform="strip_reasoning")
    grade_assertion(spec, "hidden chain of thought\n\n\nParis.", ctx(j))
    assert "hidden chain of thought" not in j.prompts[0]
    assert "Paris." in j.prompts[0]


def test_missing_judge_raises_clear_error():
    spec = AssertionSpec(type="rubric", value="x")
    with pytest.raises(ValueError, match="judge"):
        grade_assertion(spec, "ans", GradeContext())

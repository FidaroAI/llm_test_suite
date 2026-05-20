"""Rubric test (live, LLM-as-judge): score the model's output against a rubric
using *another* LLM. This is the DeepEval analog of promptfoo's `llm-rubric`
assertion (the commented-out TODO in promptfooconfig.yaml).

Pipeline:
  1. model under test answers an open-ended prompt  (MUT_* / VLLM_*)
  2. GEval asks the JUDGE model to score it against a rubric  (ANTHROPIC_API_KEY, etc.)

Needs both a reachable model under test AND a configured judge; skips otherwise.

Run: pytest deep_eval/tests/test_rubric.py -v -s
"""
import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.metrics.g_eval.utils import Rubric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from _live import generate_or_skip
from deepeval_demo.config import judge_config, model_under_test_config

_MUT = model_under_test_config()
_JUDGE = judge_config()
_READY = _MUT.is_configured and _JUDGE.is_configured
_SKIP_REASON = (
    f"needs model under test ({_MUT.missing or 'ok'}) "
    f"and judge ({_JUDGE.missing if not _JUDGE.is_configured else 'ok'})"
)

PROMPT = (
    "A customer writes: 'My order #4471 arrived with a cracked screen. "
    "I need this resolved today.' Write a short customer-support reply."
)


@pytest.mark.skipif(not _READY, reason=_SKIP_REASON)
def test_support_reply_meets_rubric():
    from deepeval_demo.judge import CloudJudge  # imported lazily so unconfigured runs don't construct it

    answer = generate_or_skip(PROMPT)
    case = LLMTestCase(input=PROMPT, actual_output=answer)

    quality = GEval(
        name="Support Reply Quality",
        model=CloudJudge(_JUDGE),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        # A natural-language rubric, like a promptfoo `llm-rubric` string. GEval
        # turns this into evaluation steps and the judge LLM scores against it.
        criteria=(
            "Evaluate the support reply. A good reply: (1) acknowledges the specific "
            "problem (cracked screen, order #4471), (2) is empathetic and professional "
            "in tone, and (3) offers a concrete next step (replacement, refund, or "
            "escalation). Penalize generic, dismissive, or off-topic replies."
        ),
        # Optional explicit score bands (0-10 scale) for more stable judging.
        rubric=[
            Rubric(score_range=(0, 4), expected_outcome="Generic, dismissive, or ignores the issue."),
            Rubric(score_range=(5, 7), expected_outcome="Addresses the issue but vague or missing a clear next step."),
            Rubric(score_range=(8, 10), expected_outcome="Specific, empathetic, and offers a concrete resolution."),
        ],
        threshold=0.7,
    )

    assert_test(case, [quality])

"""LangSmith evaluators — the two ways we score model output.

A LangSmith evaluator is just a callable `(run, example) -> {key, score, comment}`.
We expose:
  * a deterministic `contains` evaluator (the icontains analog, no LLM), and
  * an LLM-as-judge `rubric` evaluator backed by a cloud model.

The scoring logic is split out (`contains_check`, the injectable judge) so it can
be unit-tested locally with no LangSmith server and no network — the LangSmith
analog of the deep_eval demo's ContainsMetric unit test.
"""
from __future__ import annotations

from typing import Callable, Optional, Tuple

import instructor
from pydantic import BaseModel, Field

from .config import JudgeConfig, judge_config

# Same rubric as the deep_eval demo, so the two frameworks judge identically.
RUBRIC_CRITERIA = (
    "Evaluate the support reply. A good reply: (1) acknowledges the specific "
    "problem (cracked screen, order #4471), (2) is empathetic and professional "
    "in tone, and (3) offers a concrete next step (replacement, refund, or "
    "escalation). Penalize generic, dismissive, or off-topic replies."
)

_RUBRIC_PROMPT = (
    "{criteria}\n\n"
    "Score the reply from 0 to 10 using these bands:\n"
    "  0-4  generic, dismissive, or ignores the issue\n"
    "  5-7  addresses the issue but vague or missing a clear next step\n"
    "  8-10 specific, empathetic, and offers a concrete resolution\n\n"
    "Customer prompt:\n---\n{question}\n---\n\n"
    "Reply to evaluate:\n---\n{answer}\n---"
)


# --- deterministic contains evaluator ----------------------------------------
def contains_check(answer: str, expected: str, *, case_insensitive: bool = True) -> Tuple[bool, str]:
    hay = answer.lower() if case_insensitive else answer
    needle = expected.lower() if case_insensitive else expected
    ok = needle in hay
    return ok, (f"output contains {expected!r}" if ok else f"output missing {expected!r}")


def make_contains_evaluator(*, case_insensitive: bool = True) -> Callable:
    def contains(run, example) -> dict:
        answer = (run.outputs or {}).get("answer", "")
        expected = (example.outputs or {}).get("expected", "")
        ok, comment = contains_check(answer, expected, case_insensitive=case_insensitive)
        return {"key": "contains", "score": 1.0 if ok else 0.0, "comment": comment}

    return contains


# --- LLM-as-judge rubric evaluator -------------------------------------------
class RubricScore(BaseModel):
    score: int = Field(description="integer 0-10 per the rubric bands")
    reason: str = Field(description="one or two sentences justifying the score")


class CloudRubricJudge:
    """Anthropic (default) or OpenAI-compatible judge returning a structured
    {score, reason} via instructor — robust vs. parsing free text."""

    def __init__(self, config: Optional[JudgeConfig] = None):
        self._cfg = config or judge_config()
        if not self._cfg.is_configured:
            raise RuntimeError(f"judge not configured; missing: {self._cfg.missing}")
        if self._cfg.provider == "anthropic":
            from anthropic import Anthropic

            self._client = instructor.from_anthropic(Anthropic(api_key=self._cfg.api_key))
        else:
            from openai import OpenAI

            self._client = instructor.from_openai(
                OpenAI(api_key=self._cfg.api_key, base_url=self._cfg.base_url)
            )

    def score(self, question: str, answer: str) -> Tuple[float, str]:
        result = self._client.chat.completions.create(
            model=self._cfg.model,
            max_tokens=1024,
            messages=[{
                "role": "user",
                "content": _RUBRIC_PROMPT.format(
                    criteria=RUBRIC_CRITERIA, question=question, answer=answer
                ),
            }],
            response_model=RubricScore,
        )
        return result.score / 10.0, result.reason


def make_rubric_evaluator(judge=None, *, threshold: float = 0.7) -> Callable:
    """`judge` is any object with `.score(question, answer) -> (score_0_1, reason)`.
    Defaults to a CloudRubricJudge, constructed lazily so importing this module
    needs no API key."""
    state = {"judge": judge}

    def rubric_quality(run, example) -> dict:
        if state["judge"] is None:
            state["judge"] = CloudRubricJudge()
        answer = (run.outputs or {}).get("answer", "")
        question = (example.inputs or {}).get("question", "")
        score, reason = state["judge"].score(question, answer)
        verdict = "PASS" if score >= threshold else "FAIL"
        return {
            "key": "rubric_quality",
            "score": score,
            "comment": f"{verdict} ({score:.2f} vs threshold {threshold}): {reason}",
        }

    return rubric_quality

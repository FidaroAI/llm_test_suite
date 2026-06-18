"""LLM-graded assertions: rubric and g-eval.

Both call ``context.judge`` — a ``Callable[[str], str]`` that takes a prompt and returns
the judge model's text. Injecting it (rather than calling litellm here) keeps grading
logic pure and unit-testable, and lets the judge be any provider.
"""

from __future__ import annotations

import json
import re
from typing import Any

from llmeval.assertions.base import AssertionResult, GradeContext, register
from llmeval.models import AssertionSpec


def _require_judge(ctx: GradeContext):
    if ctx.judge is None:
        raise ValueError("LLM-graded assertion requires a judge in the GradeContext")
    return ctx.judge


def _answer_text(output: Any) -> str:
    return output if isinstance(output, str) else str(output if output is not None else "")


_RUBRIC_TEMPLATE = (
    "You are grading an AI assistant's answer against a criterion.\n\n"
    "User question:\n{question}\n\n"
    "Criterion:\n{criterion}\n\n"
    "Answer:\n{answer}\n\n"
    "Reply with a JSON object: {{\"score\": <float 0..1>, \"reason\": <short string>}}. "
    "1.0 means the criterion is fully met, 0.0 means not at all."
)


@register("rubric")
def _rubric(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    judge = _require_judge(ctx)
    prompt = _RUBRIC_TEMPLATE.format(
        question=ctx.user_text, criterion=spec.value, answer=_answer_text(output)
    )
    reply = judge(prompt)
    score, reason = _parse_json_score(reply)
    threshold = spec.params.get("threshold", 0.5)
    return AssertionResult(passed=score >= threshold, score=score, reason=reason or reply[:200])


_GEVAL_TEMPLATE = (
    "You are evaluating an AI assistant's answer using this rubric:\n{criterion}\n\n"
    "User question:\n{question}\n\n"
    "Answer:\n{answer}\n\n"
    "Think step by step about how well the answer meets the rubric, then give a final "
    "integer score from 1 to 10 on its own line as 'Score: N/10'."
)


@register("g_eval")
def _g_eval(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    judge = _require_judge(ctx)
    prompt = _GEVAL_TEMPLATE.format(
        criterion=spec.value, question=ctx.user_text, answer=_answer_text(output)
    )
    reply = judge(prompt)
    score = _parse_scale_score(reply)
    threshold = spec.params.get("threshold", 0.5)
    return AssertionResult(passed=score >= threshold, score=score, reason=reply.strip()[:200])


def _parse_json_score(reply: str) -> tuple[float, str]:
    """Extract {score, reason} from a judge reply; tolerate prose around the JSON."""
    try:
        start, end = reply.index("{"), reply.rindex("}") + 1
        obj = json.loads(reply[start:end])
        return _clamp01(float(obj["score"])), str(obj.get("reason", ""))
    except (ValueError, KeyError, TypeError):
        # fall back to any 0..1 float in the text
        m = re.search(r"\d*\.\d+", reply)
        return (_clamp01(float(m.group())) if m else 0.0), ""


def _parse_scale_score(reply: str) -> float:
    """Find an N/10 (or 'N out of 10') score and normalise to 0..1."""
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10", reply, re.I)
    if m:
        return _clamp01(float(m.group(1)) / 10.0)
    m = re.search(r"\d*\.\d+", reply)  # bare 0..1 float fallback
    if m:
        return _clamp01(float(m.group()))
    return 0.0


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

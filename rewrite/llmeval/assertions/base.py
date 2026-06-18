"""Assertion framework: a type registry and the dispatch entry point.

An assertion grader is ``fn(spec, output, context) -> AssertionResult``. The output it
receives has already had ``spec.transform`` applied (default: strip reasoning), while
the full raw response is available on ``context`` for assertions that need the reasoning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from llmeval.models import AssertionSpec
from llmeval.response import apply_transform


@dataclass
class AssertionResult:
    passed: bool
    score: float
    reason: str = ""


@dataclass
class GradeContext:
    """Everything an assertion might need beyond the (transformed) answer."""

    reasoning: str | None = None
    raw: Any = None
    tokens: Any = None
    user_text: str = ""
    judge: Callable[[str], str] | None = None  # set for LLM-graded assertions
    extra: dict[str, Any] = field(default_factory=dict)


Grader = Callable[[AssertionSpec, Any, GradeContext], AssertionResult]
REGISTRY: dict[str, Grader] = {}


def register(type_name: str) -> Callable[[Grader], Grader]:
    def deco(fn: Grader) -> Grader:
        REGISTRY[type_name] = fn
        return fn

    return deco


def grade_assertion(
    spec: AssertionSpec, output: Any, context: GradeContext | None = None
) -> AssertionResult:
    grader = REGISTRY[spec.type]  # KeyError on unknown type is intentional
    transformed = apply_transform(spec.transform, output)
    return grader(spec, transformed, context or GradeContext())

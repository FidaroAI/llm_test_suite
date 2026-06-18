"""Assertion registry. Importing this package registers all built-in types."""

from llmeval.assertions.base import (
    AssertionResult,
    GradeContext,
    REGISTRY,
    grade_assertion,
    register,
)

# Import for side effect: each module registers its assertion types.
from llmeval.assertions import deterministic  # noqa: E402,F401
from llmeval.assertions import judge  # noqa: E402,F401

__all__ = [
    "AssertionResult",
    "GradeContext",
    "REGISTRY",
    "grade_assertion",
    "register",
]

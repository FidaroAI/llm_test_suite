"""Deterministic metrics — the DeepEval analog of promptfoo's built-in asserts.

`ContainsMetric` mirrors promptfoo's `icontains:` assertion: a substring check
with no LLM involved. It plugs into the same `assert_test` / evaluate machinery
as the LLM-judged metrics, so factual and rubric tests share one harness.
"""
from __future__ import annotations

from typing import List, Sequence, Union

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class ContainsMetric(BaseMetric):
    """Pass if the output contains all (or any) of the expected substrings.

    Args:
        expected: a substring or list of substrings to look for.
        case_insensitive: match like promptfoo's `icontains` (default True).
        match_all: require every substring (True) or just one (False).
    """

    def __init__(
        self,
        expected: Union[str, Sequence[str]],
        *,
        case_insensitive: bool = True,
        match_all: bool = True,
    ):
        self.expected: List[str] = [expected] if isinstance(expected, str) else list(expected)
        self.case_insensitive = case_insensitive
        self.match_all = match_all
        self.threshold = 1.0  # binary pass/fail
        self.score = 0.0
        self.success = False
        self.reason = ""
        self.error = None

    def measure(self, test_case: LLMTestCase) -> float:
        output = test_case.actual_output or ""
        haystack = output.lower() if self.case_insensitive else output
        needles = [
            (e.lower() if self.case_insensitive else e) for e in self.expected
        ]

        found = [n for n in needles if n in haystack]
        missing = [e for e, n in zip(self.expected, needles) if n not in haystack]

        passed = (len(missing) == 0) if self.match_all else (len(found) > 0)
        self.score = 1.0 if passed else 0.0
        self.success = passed
        quantifier = "all of" if self.match_all else "any of"
        if passed:
            self.reason = f"Output contains {quantifier} the expected: {self.expected}"
        else:
            self.reason = (
                f"Output is missing expected substring(s) {missing} "
                f"(needed {quantifier} {self.expected})."
            )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success

    @property
    def __name__(self):
        mode = "all" if self.match_all else "any"
        return f"Contains[{mode}]"

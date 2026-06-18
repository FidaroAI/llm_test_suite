"""Grading: apply a test case's assertions to *cached* results.

This stage never calls the model under test — only the (optional) judge. So you can edit
assertions, add new ones, or change the judge and re-grade existing outputs cheaply.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Iterable

from llmeval.assertions import GradeContext, grade_assertion
from llmeval.models import AssertionSpec, TestCase
from llmeval.store import Store


def assertion_key(spec: AssertionSpec) -> str:
    """A stable id for a grading row.

    An explicit ``spec.id`` is honoured; otherwise the key is derived from the
    assertion's content, so editing the value/params yields a new key (a genuinely
    different check) while re-running an unchanged assertion upserts in place.
    """
    if spec.id:
        return spec.id
    payload = json.dumps(
        {"type": spec.type, "value": spec.value, "params": spec.params},
        sort_keys=True,
        default=str,
    )
    return f"{spec.type}:{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:10]}"


def grade_testcase(
    store: Store,
    testcase: TestCase,
    cache_key_hash: str,
    judge: Callable[[str], str] | None = None,
    regrade: bool = False,
) -> None:
    """Grade every cached (non-error) result of ``testcase`` under one cache key."""
    for result in store.get_results(testcase.id, cache_key_hash):
        if result.error is not None:
            continue
        already = set() if regrade else {g.assertion_key for g in store.get_gradings(result.id)}
        ctx = GradeContext(
            reasoning=result.reasoning,
            raw=result.raw,
            tokens=result.tokens,
            user_text=testcase.user_text,
            judge=judge,
        )
        for spec in testcase.assertions:
            akey = assertion_key(spec)
            if akey in already:
                continue
            res = grade_assertion(spec, result.output, ctx)
            store.set_grading(
                result.id,
                akey,
                type=spec.type,
                metric=spec.metric,
                score=res.score,
                passed=res.passed,
                weight=spec.weight,
                reason=res.reason,
            )


def grade(
    store: Store,
    testcases: Iterable[TestCase],
    cache_key_hash: str,
    judge: Callable[[str], str] | None = None,
    regrade: bool = False,
) -> None:
    for testcase in testcases:
        grade_testcase(store, testcase, cache_key_hash, judge=judge, regrade=regrade)

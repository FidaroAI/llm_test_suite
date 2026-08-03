"""Grading: apply a test case's assertions to *cached* results.

This stage never calls the model under test — only the (optional) judge. So you can edit
assertions, add new ones, or change the judge and re-grade existing outputs cheaply.

A grading belongs to a **result**, not to a test: ``gradings`` is unique on
``(result_id, assertion_key)``, so every attempt a test ever produced can carry its own
score, and re-running a test adds a row to grade rather than superseding one. Attempts that
**errored** are skipped entirely — there is no output to assert against, and the error row
is itself the finding.

``run_ids`` narrows which results are considered, so a re-grade can be aimed at one sitting
instead of the whole history of a cache key. See :mod:`llmeval.runselect`.

``hooks`` lets the owning plugin refresh anything its assertions compare against before any
grading happens — that is how the stock-price suite grades against live prices rather than a
reference baked in at generation time. Taken structurally, like the runner's, so this module
imports no plugin machinery beyond the outcome record it reports back.
"""

from __future__ import annotations

import hashlib
import json
from typing import Callable, Collection, Iterable

from llmeval.assertions import GradeContext, grade_assertion
from llmeval.models import AssertionSpec, TestCase
from llmeval.plugins.base import GradingOutcome
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


def assertion_value_text(spec: AssertionSpec) -> str | None:
    """The criterion as readable text, stored beside the score.

    ``assertion_key`` hashes the criterion so re-grades can find their row; this is the
    other half — the text itself, so a report can show *what* was asked without joining
    against ``testcases/``, which is regenerated and may since have changed.

    ``value`` is typed ``Any``: a rubric's is prose and goes in verbatim, anything else is
    JSON so a list survives as ``["a"]`` rather than Python's ``['a']``. A valueless
    assertion (``refusal``, ``length`` — the criterion lives in ``params``) stores nothing
    rather than the string ``"None"``.
    """
    if spec.value is None:
        return None
    if isinstance(spec.value, str):
        return spec.value
    return json.dumps(spec.value, ensure_ascii=False, default=str)


def grade_testcase(
    store: Store,
    testcase: TestCase,
    cache_key_hash: str,
    judge: Callable[[str], str] | None = None,
    regrade: bool = False,
    run_ids: Collection[str] | None = None,
    hooks=None,
) -> None:
    """Grade every cached (non-error) result of ``testcase`` under one cache key.

    Every result, not just the newest one — a grading belongs to a result.

    :param run_ids: restrict to results produced by these runs. ``None`` means every run
        for the cache key. An **empty** collection means no runs, and so grades nothing —
        the same None-versus-empty distinction :meth:`llmeval.store.Store.select_runs` uses.
    :param hooks: lifecycle dispatcher. ``after_each_grade`` reports what *this pass*
        produced across every attempt of the test case, so it is empty on a re-grade that
        found everything already scored.
    """
    if hooks is not None:
        hooks.before_each_grade(testcase)
    outcomes: list[GradingOutcome] = []
    allowed = None if run_ids is None else set(run_ids)
    for result in store.get_results(testcase.id, cache_key_hash):
        if result.error is not None:
            continue
        if allowed is not None and result.run_id not in allowed:
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
                assertion_value=assertion_value_text(spec),
                type=spec.type,
                metric=spec.metric,
                score=res.score,
                passed=res.passed,
                weight=spec.weight,
                reason=res.reason,
            )
            outcomes.append(GradingOutcome(assertion_key=akey, spec=spec, result=res))

    if hooks is not None:
        hooks.after_each_grade(testcase, outcomes)


def grade(
    store: Store,
    testcases: Iterable[TestCase],
    cache_key_hash: str,
    judge: Callable[[str], str] | None = None,
    regrade: bool = False,
    run_ids: Collection[str] | None = None,
    hooks=None,
) -> None:
    if hooks is not None:
        hooks.before_grade()
    for testcase in testcases:
        grade_testcase(
            store, testcase, cache_key_hash, judge=judge, regrade=regrade,
            run_ids=run_ids, hooks=hooks,
        )
    if hooks is not None:
        hooks.after_grade()

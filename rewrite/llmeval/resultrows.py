"""Flatten stored results into report rows — the data half of ``llmeval report``.

The report is a table, and this module decides its shape. Rendering it is somebody else's
problem: this produces rows and a CSV, and the porcelain in ``reporting/`` turns that into
HTML. Keeping the split here means the row shape is testable without a browser and reusable
by anything that reads a CSV.

**One row per (result, assertion).** Result fields repeat across a test's assertions, which
makes ``assertion_key``/``metric``/``score``/``passed`` ordinary filterable columns rather
than a schema that grows with the assertion vocabulary. Three cases:

* a successful result with gradings — one row each
* a successful result with none — one row, grading columns empty (a run not yet graded)
* an **errored** result — one row, grading columns empty, ``error`` and ``latency_ms`` set

That last case is why this exists. An errored test produces no grading and therefore no
statistics, so a report built from gradings alone renders it as absence. Here it is a row
you can filter for.

**Ordering** is run, then test, then attempt, which reads as a history: each run's tests in
turn, and within a test its failed attempts before the one that answered. Run order is
whatever the caller passed — :func:`llmeval.runselect.resolve_runs` returns oldest-first, so
that decision is made once, there. Test order inside a run is by id rather than by
first-attempt time: deterministic, and the viewer re-sorts on any column anyway.

The store is read through its public methods rather than fresh SQL. ``get_gradings`` per
result is an N+1 pattern, which is irrelevant at report scale and keeps this file from
depending on the store's column names.
"""

from __future__ import annotations

import csv
import os
import re
from typing import Any, Mapping, Sequence

from llmeval.models import TestCase
from llmeval.store import ResultRow, RunRow, Store

# Test ids are ``<suite>-<sha1(prompt)[:10]>[-<variant>]`` (llmeval/generation/common.py),
# so the suite cannot be recovered by splitting on "-": that would turn
# "research_rubrics-abc1234567-geval" into "research_rubrics-abc1234567". Anchoring on the
# 10-hex digest is unambiguous. Only a fallback — testcase metadata is authoritative.
_ID_SUITE = re.compile(r"^(?P<suite>.+?)-[0-9a-f]{10}(?:-.*)?$")

# Column order is the reading order of the report: where it came from, what was asked, what
# came back, how it scored, then the provenance you only want when something looks wrong.
_RUN_COLUMNS = ["run_id", "run_started_at", "provider"]
_BASE_COLUMNS = ["test_id", "attempt", "suite"]
_TEST_COLUMNS = ["prompt", "request_type", "domain"]
_RESULT_COLUMNS = [
    "output",
    "reasoning",
    "error",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
]
_GRADING_COLUMNS = [
    "assertion_key",
    "assertion_type",
    "metric",
    "score",
    "passed",
    "weight",
    "judge_model",
    "grading_reason",
]
_PROVENANCE_COLUMNS = ["cache_key_hash", "result_id"]


def suite_of(test_id: str, case: TestCase | None) -> str | None:
    """The test's suite: metadata first, id-shape fallback second.

    Every generator writes ``suite`` into metadata, so with test cases loaded this is exact.
    Without them the id pattern is the only signal available.
    """
    if case is not None:
        suite = case.metadata.get("suite")
        if suite is not None:
            return str(suite)
    match = _ID_SUITE.match(test_id)
    return match.group("suite") if match else None


def _text(value: str | None) -> str | None:
    """Trim surrounding whitespace off a model text field for display.

    Gateway outputs routinely *start* with blank lines: the reasoning-strip rule
    (``llmeval/response.py``) splits on ``\\n\\n\\n``, and the parser that produced it
    swapped ``<thinking>`` tags for newlines. Rendered in a table cell, those leading blanks
    push the answer out of view before a single word is read.

    Only the *ends* are touched, so internal formatting — markdown lists, blank lines
    between paragraphs — survives. The store keeps the untouched value either way.
    """
    return value.strip() if isinstance(value, str) else value


def _tokens(result: ResultRow) -> dict[str, Any]:
    """Flatten litellm's stored usage dict into three filterable integers.

    The stored shape is ``{prompt_tokens, completion_tokens, total_tokens,
    prompt_tokens_details, completion_tokens_details}``; the ``*_details`` keys are null in
    practice, so they're dropped rather than rendered as two always-empty columns.
    """
    usage = result.tokens if isinstance(result.tokens, Mapping) else {}
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def result_columns(with_tests: bool) -> list[str]:
    """The column order for a result-rows report.

    ``with_tests`` follows whether test cases were loaded. The prompt/classification columns
    are then *absent* rather than empty — an absent column says "you didn't ask for this",
    an empty one says "there was nothing to show", and they are different answers.
    """
    cols = list(_RUN_COLUMNS) + list(_BASE_COLUMNS)
    if with_tests:
        cols += _TEST_COLUMNS
    return cols + _RESULT_COLUMNS + _GRADING_COLUMNS + _PROVENANCE_COLUMNS


def _shared_fields(run: RunRow, result: ResultRow, case: TestCase | None) -> dict[str, Any]:
    """Everything on a row that doesn't come from a grading.

    Split out because it is computed once per result and then copied across that result's
    assertions — the repetition is what makes the grading fields ordinary columns.
    """
    return {
        "run_id": run.id,
        "run_started_at": run.started_at,
        "provider": run.provider_name,
        "test_id": result.test_id,
        "attempt": result.attempt,
        "suite": suite_of(result.test_id, case),
        "output": _text(result.output),
        "reasoning": _text(result.reasoning),
        "error": result.error,
        "latency_ms": round(result.latency_ms, 1) if result.latency_ms is not None else None,
        **_tokens(result),
        "cache_key_hash": result.cache_key_hash,
        "result_id": result.id,
    }


def result_rows(
    store: Store,
    runs: Sequence[RunRow],
    cases_by_id: Mapping[str, TestCase] | None = None,
) -> list[dict[str, Any]]:
    """Flatten the given runs into report rows, in the order ``runs`` arrives in.

    :param cases_by_id: test cases keyed by id. When given it **selects as well as
        enriches** — a result whose test is absent is dropped, which is what makes
        ``--filter k=v`` mean anything and matches how ``run`` and ``grade`` already treat
        ``--testcases``. Pass ``None`` for every result, unfiltered and unenriched.
    """
    with_tests = cases_by_id is not None
    columns = result_columns(with_tests)
    rows: list[dict[str, Any]] = []

    for run in runs:
        for result in store.get_results_for_run(run.id):
            if with_tests and result.test_id not in cases_by_id:
                continue
            case = cases_by_id.get(result.test_id) if with_tests else None
            shared = _shared_fields(run, result, case)
            if with_tests:
                shared["prompt"] = case.user_text if case else None
                shared["request_type"] = case.metadata.get("request_type") if case else None
                shared["domain"] = case.metadata.get("domain") if case else None

            # An errored attempt is never graded (``grade`` skips error rows), so this is
            # stated rather than discovered: the row exists to show the failure, and
            # grading columns on it would be meaningless.
            gradings = [] if result.error is not None else store.get_gradings(result.id)
            if not gradings:
                rows.append({c: shared.get(c) for c in columns})
                continue

            for grading in gradings:
                row = dict(shared)
                row.update(
                    {
                        "assertion_key": grading.assertion_key,
                        "assertion_type": grading.type,
                        "metric": grading.metric,
                        "score": grading.score,
                        "passed": grading.passed,
                        "weight": grading.weight,
                        "judge_model": grading.judge_model,
                        "grading_reason": grading.reason,
                    }
                )
                rows.append({c: row.get(c) for c in columns})

    return rows


def write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: str) -> str:
    """Write rows as CSV, creating parent directories. Returns the path.

    The header is written even for zero rows: an empty selection should produce a table with
    no rows, not a file a reader cannot parse.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows([{c: r.get(c) for c in columns} for r in rows])
    return path

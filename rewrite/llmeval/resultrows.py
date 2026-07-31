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
import json
import os
from typing import Any, Mapping, Sequence

from llmeval.models import TestCase, last_user_text
from llmeval.plugins.loader import source_of
from llmeval.store import ResultRow, RunRow, Store

# Column order is the reading order of the report: where it came from, what was asked, what
# came back, how it scored, then the provenance you only want when something looks wrong.
_RUN_COLUMNS = ["run_id", "run_started_at", "provider"]
# ``prompt`` and ``messages`` come off the result, so they need no test-case files and
# are always present. ``prompt`` is the readable view (the last user turn); ``messages``
# is the complete record behind it, which is the only place a system prompt or an earlier
# turn of a conversation survives.
_BASE_COLUMNS = ["test_id", "attempt", "suite", "prompt", "messages"]
_RESULT_COLUMNS = [
    "output",
    "reasoning",
    "provider_specific_output",
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


def suite_of(test_id: str) -> str | None:
    """The source a test came from, read off its id prefix.

    Ids are ``<source>.<local id>`` (see :mod:`llmeval.plugins.loader`), so the provenance is
    in the id itself: no metadata lookup, no test-case files, and nothing that can disagree
    with where the test actually lives. An id with no prefix predates the plugin system and
    has no source to report.
    """
    return source_of(test_id)


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


def _provider_specific(result: ResultRow) -> str | None:
    """The provider's non-standard response data as one verbatim JSON cell.

    Deliberately *unparsed*. The store keeps whatever the provider sent under its vendor key
    (``{"fidaro": {"title": ...}}``), and the value of that arrangement is that a new key
    appears in the report without a code change here. Flattening it into
    ``fidaro_title``-style columns would trade that away for sortability, and put vendor key
    names in the row shape.

    ``None`` rather than ``"null"`` when the provider sent nothing, which is the usual case:
    an absent value should read as an empty cell.
    """
    if result.provider_specific is None:
        return None
    return json.dumps(result.provider_specific, ensure_ascii=False)


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


def result_columns() -> list[str]:
    """The column order for a result-rows report. Fixed — nothing here is conditional.

    Every column is derivable from the stored result, ``suite`` included (it is the test id's
    prefix), so the report has the same shape whether or not ``--testcases`` narrowed it.
    """
    return (
        list(_RUN_COLUMNS) + list(_BASE_COLUMNS)
        + _RESULT_COLUMNS + _GRADING_COLUMNS + _PROVENANCE_COLUMNS
    )


def _prompt_fields(result: ResultRow, case: TestCase | None) -> dict[str, Any]:
    """The question, from the result's own record of what was sent.

    The **stored** messages win over the test case: ``testcases/`` is regenerated, so the
    file's current text is not necessarily what this result was produced from, and only
    the stored copy is evidence. The test case is a fallback for results written before
    the store recorded prompts (schema 2), which would otherwise show an empty cell.

    ``messages`` is serialised here rather than left as a list because these rows go
    straight to CSV, where a cell is text either way.
    """
    if result.messages:
        return {
            "prompt": last_user_text(result.messages),
            "messages": json.dumps(result.messages, ensure_ascii=False),
        }
    return {"prompt": case.user_text if case else None, "messages": None}


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
        "suite": suite_of(result.test_id),
        **_prompt_fields(result, case),
        "output": _text(result.output),
        "reasoning": _text(result.reasoning),
        "provider_specific_output": _provider_specific(result),
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

    :param cases_by_id: test cases keyed by id. When given it **selects**: a result whose
        test is absent is dropped, which is what makes ``--testcases``/``--filter`` mean
        anything on a report. Pass ``None`` for every result — the useful default now that
        plugin output is regenerated rather than tracked, since a run should outlive its
        test cases. Every column is filled either way; the case is only a fallback for the
        prompt on results written before the store recorded messages.
    """
    with_tests = cases_by_id is not None
    columns = result_columns()
    rows: list[dict[str, Any]] = []

    for run in runs:
        for result in store.get_results_for_run(run.id):
            if with_tests and result.test_id not in cases_by_id:
                continue
            case = cases_by_id.get(result.test_id) if with_tests else None
            shared = _shared_fields(run, result, case)

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

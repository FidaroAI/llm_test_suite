"""Report on everything a single ``llmeval run`` produced.

    python -m reporting.run_report run_20260729-0435 -o run.html --testcases testcases/

The interesting part is :func:`run_rows`, which is pure and HTML-free: it turns one run
into flat rows that :mod:`reporting.csv_table` (or a CSV, or a test) can consume. Row
building and rendering stay separate so every later tool can reuse the same split.

**One row per result x assertion.** Result fields repeat across a test's assertions, which
makes ``assertion_key``/``metric``/``score``/``passed`` ordinary filterable columns rather
than a schema that grows with the assertion vocabulary. A result with no gradings still
yields exactly one row with the grading columns empty — that is the common case when you
have run but not yet graded.

The store is read through its public methods rather than fresh SQL. ``get_gradings`` per
result is an N+1 pattern, which is irrelevant at report scale and keeps this file from
depending on the store's column names.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from typing import Any, Mapping, Sequence

from llmeval.models import TestCase
from llmeval.store import IncompatibleSchema, ResultRow, RunRow, Store
from llmeval.testcases import load_testcases

from reporting import csv_table

# Test ids are ``<suite>-<sha1(prompt)[:10]>[-<variant>]`` (llmeval/generation/common.py),
# so the suite cannot be recovered by splitting on "-": that would turn
# "research_rubrics-abc1234567-geval" into "research_rubrics-abc1234567". Anchoring on the
# 10-hex digest is unambiguous. Only a fallback — testcase metadata is authoritative.
_ID_SUITE = re.compile(r"^(?P<suite>.+?)-[0-9a-f]{10}(?:-.*)?$")

# Columns present whatever the inputs. Order is the reading order of the report:
# what was tested, what came back, how it scored, where it came from.
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

    Every generator writes ``suite`` into metadata, so with ``--testcases`` this is exact.
    Without it, the id pattern is the only signal available.
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
    swapped ``<thinking>`` tags for newlines. Rendered with ``white-space: pre-wrap``,
    those leading blanks push the answer out of view before a single word is read.

    Only the *ends* are touched, so internal formatting — the markdown lists and blank
    lines between paragraphs — is preserved. The untouched value is still in the store,
    which stays the source of truth.
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


def run_columns(with_tests: bool) -> list[str]:
    """The column order for a run report. Test columns appear only when enriching."""
    cols = list(_BASE_COLUMNS)
    if with_tests:
        cols += _TEST_COLUMNS
    return cols + _RESULT_COLUMNS + _GRADING_COLUMNS + _PROVENANCE_COLUMNS


def run_rows(
    store: Store,
    run_id: str,
    cases_by_id: Mapping[str, TestCase] | None = None,
) -> list[dict[str, Any]]:
    """Flatten one run into rows: one per (result x assertion), or one per ungraded result.

    :param cases_by_id: optional test cases keyed by id. When given, ``prompt`` and the
        classification labels are added. A ``test_id`` in the run but *absent* here yields
        empty values rather than raising — testcases get regenerated, and a stale id should
        not break the report.
    """
    with_tests = cases_by_id is not None
    columns = run_columns(with_tests)
    rows: list[dict[str, Any]] = []

    for result in store.get_results_for_run(run_id):
        case = (cases_by_id or {}).get(result.test_id)
        shared: dict[str, Any] = {
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
        if with_tests:
            shared["prompt"] = case.user_text if case else None
            shared["request_type"] = case.metadata.get("request_type") if case else None
            shared["domain"] = case.metadata.get("domain") if case else None

        gradings = store.get_gradings(result.id)
        if not gradings:
            # No assertions graded yet. Emit the result anyway — an ungraded run is still
            # worth inspecting, and this is the state a fresh `llmeval run` leaves behind.
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


def run_subtitle(run: RunRow, rows: Sequence[Mapping[str, Any]]) -> str:
    """Provenance line for the page: who ran what, when, and whether it finished.

    ``finished_at`` is NULL for a run that crashed or was interrupted, which is easy to
    miss when reading results — so it's called out rather than left as a blank timestamp.
    """
    errors = len({r["result_id"] for r in rows if r.get("error")})
    results = len({r["result_id"] for r in rows})
    finished = run.finished_at if run.finished else "UNFINISHED (crashed or interrupted)"
    parts = [
        f"run {run.id}",
        f"provider {run.provider_name or '?'}",
        f"cache key {run.cache_key_hash[:12]}",
        f"started {run.started_at}",
        f"finished {finished}",
        f"{results} results ({errors} errors) in {len(rows)} rows",
    ]
    if run.notes:
        parts.append(f"note: {run.notes}")
    return " · ".join(parts)


def write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: str) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows([{c: r.get(c) for c in columns} for r in rows])
    return path


def build_report(
    db: str,
    run_prefix: str,
    out_path: str,
    testcases: str | None = None,
    csv_path: str | None = None,
) -> tuple[str, int]:
    """Resolve the run, build rows, write the HTML (and optionally a CSV).

    Returns ``(run_id, row_count)``.

    Raises ``FileNotFoundError`` for a missing database: ``sqlite3.connect`` would happily
    create an empty one, and the user would get "no run matching" for what is really a
    wrong ``--db`` path.
    """
    if not os.path.exists(db):
        raise FileNotFoundError(f"no results database at {db}")
    store = Store(db)
    try:
        run_id = store.resolve_run(run_prefix)
        run = store.get_run(run_id)
        if run is None:  # pragma: no cover - resolve_run already guarantees existence
            raise KeyError(f"run {run_id} vanished between resolve and read")

        cases_by_id = None
        if testcases:
            cases_by_id = {c.id: c for c in load_testcases(testcases)}

        rows = run_rows(store, run_id, cases_by_id)
        columns = run_columns(cases_by_id is not None)
    finally:
        store.close()

    csv_table.write_table(
        rows,
        out_path,
        title=f"llmeval run {run_id}",
        columns=columns,
        subtitle=run_subtitle(run, rows),
    )
    if csv_path:
        write_csv(rows, columns, csv_path)
    return run_id, len(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reporting.run_report",
        description="Render everything one llmeval run produced as a filterable HTML table.",
    )
    parser.add_argument("run", help="run id or any unambiguous prefix of one")
    parser.add_argument("-o", "--out", default="run_report.html", help="output HTML path")
    parser.add_argument("--db", default="llmeval.sqlite3", help="path to the results database")
    parser.add_argument(
        "--testcases",
        help="testcases dir/file; adds prompt, request_type and domain columns",
    )
    parser.add_argument("--csv", dest="csv_path", help="also write the rows as CSV here")
    args = parser.parse_args(argv)

    try:
        run_id, count = build_report(
            args.db, args.run, args.out, args.testcases, args.csv_path
        )
    except (KeyError, FileNotFoundError, IncompatibleSchema) as exc:
        # A bad prefix, an ambiguous prefix, a wrong --db path, or a database from an older
        # build. All are user error, so report them as a message, not a traceback.
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2

    print(f"wrote {args.out}: {count} rows for {run_id}")
    if args.csv_path:
        print(f"wrote {args.csv_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

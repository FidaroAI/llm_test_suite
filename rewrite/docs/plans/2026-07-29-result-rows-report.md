# Result-Rows Selection and Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add run selection (`--run-id`, `--run-after`, `--run-before`, `--run-last-n`) to the stages that read stored results, and turn `llmeval report` into a CSV of one row per (result, assertion) — errored results included — rendered and opened by the existing generic viewer.

**Architecture:** `llmeval` is plumbing and does all selection plus CSV emission; `reporting/csv_table.py` is porcelain and does nothing but CSV → HTML → `open`. Three new plumbing modules (`runselect`, `resultrows`, one store method) plus a renamed `compare-report` subcommand for the existing statistics report. `reporting/run_report.py` is superseded and deleted, its row-shaping logic moving down into `llmeval/resultrows.py`.

**Tech Stack:** Python 3.14, `uv`, pydantic (models), Jinja2 (templates), sqlite3 (stdlib), pytest, pylint.

## Global Constraints

- Design spec: [`../specs/2026-07-29-result-rows-report-design.md`](../specs/2026-07-29-result-rows-report-design.md). Read it before starting.
- **All work happens in `rewrite/`.** Run every command from `/Users/badger/fidaro/git/llm_test_suite/.claude/worktrees/llmeval-report-rows/rewrite`.
- Test command: `.venv/bin/python -m pytest`. Lint command: `.venv/bin/python -m pylint llmeval` — must stay at **10/10**.
- **No SQLite schema change.** `SCHEMA_VERSION` in `llmeval/store.py` stays at `2`. There is no migration path, so a bump would cost every user their accumulated results.
- **`llmeval` must never import `reporting`.** The dependency runs one way (`rewrite/CLAUDE.md`).
- **No `print` in `llmeval/`.** Use `logger = logging.getLogger(__name__)` per module, `%s` lazy formatting in log calls. `reporting/` may use `print` (it already does).
- Tests must run offline with no API keys and no network.
- Timestamps: `runs.started_at` is `datetime.now(timezone.utc).isoformat()`. A naive `--run-after`/`--run-before` value is **UTC**. Both bounds are **inclusive**, compared to whole-second precision.
- Do not touch `llmeval/comparison/report.py` or `llmeval/comparison/stats.py`.

---

### Task 1: `Store.select_runs`

The one new SQL query: runs matching a fully-resolved selection, oldest first.

**Files:**
- Modify: `rewrite/llmeval/store.py` (add a method after `resolve_run`, around line 348)
- Test: `rewrite/framework_tests/test_store.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```python
  Store.select_runs(
      *,
      ids: Sequence[str] | None = None,
      after: str | None = None,          # "YYYY-MM-DDTHH:MM:SS", UTC
      before: str | None = None,         # same shape
      last_n: int | None = None,
      cache_key_hashes: Sequence[str] | None = None,
  ) -> list[RunRow]                      # oldest first
  ```
  `RunRow` already exists in `llmeval/store.py`.

- [ ] **Step 1: Add a back-dating helper to the test conftest**

`Store.create_run` stamps `started_at` with the current time and offers no override, so date-based selection cannot be tested without back-dating a row. Add this to `rewrite/framework_tests/conftest.py` (after `a_run`):

```python
def backdate_run(store: Store, run_id: str, started_at: str) -> str:
    """Rewrite a run's ``started_at`` so time-window selection can be tested.

    ``create_run`` deliberately has no ``started_at`` parameter — a run is stamped when it
    opens and nothing legitimate rewrites that. Tests need runs at known times, so they
    reach past the public API rather than the API growing a hole for them.

    :param started_at: an ISO-8601 UTC string, e.g. ``"2026-07-01T09:00:00+00:00"``.
    """
    # pylint: disable=protected-access
    store._conn.execute("UPDATE runs SET started_at=? WHERE id=?", (started_at, run_id))
    store._conn.commit()
    return run_id
```

- [ ] **Step 2: Write the failing tests**

Append to `rewrite/framework_tests/test_store.py`:

```python
def test_select_runs_returns_oldest_first(store):
    key = compute_cache_key(model="m1")
    newer = backdate_run(store, a_run(store, key), "2026-07-03T09:00:00+00:00")
    older = backdate_run(store, a_run(store, key), "2026-07-01T09:00:00+00:00")
    assert [r.id for r in store.select_runs()] == [older, newer]


def test_select_runs_by_explicit_ids(store):
    key = compute_cache_key(model="m1")
    first = backdate_run(store, a_run(store, key), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, key), "2026-07-02T09:00:00+00:00")
    backdate_run(store, a_run(store, key), "2026-07-03T09:00:00+00:00")
    assert [r.id for r in store.select_runs(ids=[second, first])] == [first, second]


def test_select_runs_with_empty_id_list_selects_nothing(store):
    a_run(store, compute_cache_key(model="m1"))
    # An empty list means "these zero runs", not "no filter" — the distinction matters
    # because None is what means "no filter".
    assert store.select_runs(ids=[]) == []


def test_select_runs_bounds_are_inclusive_to_the_second(store):
    key = compute_cache_key(model="m1")
    # Microseconds on the stored value must not push it outside a second-precision bound.
    only = backdate_run(store, a_run(store, key), "2026-07-02T09:00:00.123456+00:00")
    assert [r.id for r in store.select_runs(after="2026-07-02T09:00:00")] == [only]
    assert [r.id for r in store.select_runs(before="2026-07-02T09:00:00")] == [only]


def test_select_runs_window_excludes_outside(store):
    key = compute_cache_key(model="m1")
    backdate_run(store, a_run(store, key), "2026-06-30T23:59:59+00:00")
    inside = backdate_run(store, a_run(store, key), "2026-07-01T00:00:00+00:00")
    backdate_run(store, a_run(store, key), "2026-07-02T00:00:01+00:00")
    got = store.select_runs(after="2026-07-01T00:00:00", before="2026-07-02T00:00:00")
    assert [r.id for r in got] == [inside]


def test_select_runs_last_n_takes_most_recent_but_returns_oldest_first(store):
    key = compute_cache_key(model="m1")
    backdate_run(store, a_run(store, key), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, key), "2026-07-02T09:00:00+00:00")
    third = backdate_run(store, a_run(store, key), "2026-07-03T09:00:00+00:00")
    assert [r.id for r in store.select_runs(last_n=2)] == [second, third]


def test_select_runs_narrows_by_cache_key(store):
    mine = compute_cache_key(model="m1")
    theirs = compute_cache_key(model="m2")
    ours = backdate_run(store, a_run(store, mine), "2026-07-01T09:00:00+00:00")
    backdate_run(store, a_run(store, theirs), "2026-07-02T09:00:00+00:00")
    got = store.select_runs(cache_key_hashes=[mine.hash])
    assert [r.id for r in got] == [ours]


def test_select_runs_applies_last_n_after_cache_key(store):
    """"The last 2 runs of provider X", not "the last 2 runs, then keep X's"."""
    mine = compute_cache_key(model="m1")
    theirs = compute_cache_key(model="m2")
    first = backdate_run(store, a_run(store, mine), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, mine), "2026-07-02T09:00:00+00:00")
    backdate_run(store, a_run(store, theirs), "2026-07-03T09:00:00+00:00")
    backdate_run(store, a_run(store, theirs), "2026-07-04T09:00:00+00:00")
    got = store.select_runs(last_n=2, cache_key_hashes=[mine.hash])
    assert [r.id for r in got] == [first, second]
```

Check the top of `test_store.py` for its existing imports and add whatever is missing:
`from conftest import a_run, backdate_run` and `from llmeval.cache_key import compute_cache_key`.
The file already defines a `store` fixture — reuse it rather than adding another.

- [ ] **Step 3: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest framework_tests/test_store.py -k select_runs -v
```
Expected: FAIL — `AttributeError: 'Store' object has no attribute 'select_runs'`.

- [ ] **Step 4: Implement `select_runs`**

Add to `rewrite/llmeval/store.py`, immediately after `resolve_run` and before the `_run_row` staticmethod:

```python
    def select_runs(
        self,
        *,
        ids: Sequence[str] | None = None,
        after: str | None = None,
        before: str | None = None,
        last_n: int | None = None,
        cache_key_hashes: Sequence[str] | None = None,
    ) -> list[RunRow]:
        """Runs matching a selection, **oldest first**.

        Every argument arrives already resolved: ``ids`` are full run ids, ``after`` and
        ``before`` are ``YYYY-MM-DDTHH:MM:SS`` UTC strings, and ``cache_key_hashes``
        narrows to particular provider identities. Interpreting what the user typed is
        :mod:`llmeval.runselect`'s job; this method only queries.

        ``None`` means "no filter"; an **empty sequence** means "these zero things", so
        ``ids=[]`` selects nothing. Conflating the two would make an over-narrowed
        selection silently widen to everything.

        Bounds compare against ``substr(started_at, 1, 19)`` rather than the column, so
        both ends are inclusive to the whole second. The stored value carries microseconds
        (``2026-07-29T06:12:33.123456+00:00``), and a plain text ``<=`` against a
        second-precision bound would exclude the very run the user named.

        ``last_n`` is applied *after* every other filter, so "the last 3 runs of provider
        X" is one call. It reverses in Python because SQLite cannot return the last N rows
        in ascending order without a subquery, and this table has one row per invocation.
        """
        where: list[str] = []
        args: list[Any] = []
        if ids is not None:
            if not ids:
                return []
            where.append(f"id IN ({','.join('?' * len(ids))})")
            args.extend(ids)
        if cache_key_hashes is not None:
            if not cache_key_hashes:
                return []
            where.append(f"cache_key_hash IN ({','.join('?' * len(cache_key_hashes))})")
            args.extend(cache_key_hashes)
        if after is not None:
            where.append("substr(started_at, 1, 19) >= ?")
            args.append(after)
        if before is not None:
            where.append("substr(started_at, 1, 19) <= ?")
            args.append(before)

        sql = "SELECT * FROM runs"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if last_n is not None:
            sql += " ORDER BY started_at DESC, id DESC LIMIT ?"
            args.append(last_n)
            with self._lock:
                rows = self._conn.execute(sql, args).fetchall()
            return [self._run_row(r) for r in reversed(rows)]

        sql += " ORDER BY started_at, id"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._run_row(r) for r in rows]
```

`Sequence` must be added to the `typing` import at the top of the file — it currently
reads `from typing import Any, Iterator`.

- [ ] **Step 5: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest framework_tests/test_store.py -v
.venv/bin/python -m pylint llmeval
```
Expected: all PASS, pylint 10/10.

- [ ] **Step 6: Commit**

```bash
git add rewrite/llmeval/store.py rewrite/framework_tests/test_store.py rewrite/framework_tests/conftest.py
git commit -m "Add Store.select_runs for run selection queries"
```

---

### Task 2: `llmeval/runselect.py`

The meaning of the four flags, with no SQL: parsing, mutual exclusion, and resolution to concrete runs.

**Files:**
- Create: `rewrite/llmeval/runselect.py`
- Test: `rewrite/framework_tests/test_runselect.py`

**Interfaces:**
- Consumes: `Store.select_runs` (Task 1), `Store.resolve_run`, `Store.get_run`, `RunRow`.
- Produces:
  ```python
  class RunSelectionError(ValueError): ...

  @dataclass(frozen=True)
  class RunSelection:
      ids: tuple[str, ...] = ()
      after: str | None = None
      before: str | None = None
      last_n: int | None = None

  parse_run_selection(
      run_id: str | Sequence[str] | None = None,
      run_after: str | None = None,
      run_before: str | None = None,
      run_last_n: int | None = None,
  ) -> RunSelection

  resolve_runs(
      store: Store,
      selection: RunSelection,
      cache_key_hashes: Sequence[str] | None = None,
  ) -> list[RunRow]                       # oldest first
  ```

- [ ] **Step 1: Write the failing tests**

Create `rewrite/framework_tests/test_runselect.py`:

```python
import pytest
from conftest import a_run, backdate_run

from llmeval.cache_key import compute_cache_key
from llmeval.runselect import (
    RunSelection,
    RunSelectionError,
    parse_run_selection,
    resolve_runs,
)
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


KEY = compute_cache_key(model="m1")
OTHER = compute_cache_key(model="m2")


# --- parsing -------------------------------------------------------------


def test_no_flags_is_an_empty_selection():
    assert parse_run_selection() == RunSelection()


def test_run_id_splits_on_commas():
    assert parse_run_selection(run_id="run1,run2,run3").ids == ("run1", "run2", "run3")


def test_run_id_accumulates_across_repeats_and_trims():
    got = parse_run_selection(run_id=["run1, run2", "run3"])
    assert got.ids == ("run1", "run2", "run3")


def test_run_id_ignores_empty_elements():
    assert parse_run_selection(run_id="run1,,run2,").ids == ("run1", "run2")


def test_last_n_is_carried_through():
    assert parse_run_selection(run_last_n=3).last_n == 3


def test_run_id_and_window_conflict():
    with pytest.raises(RunSelectionError, match="cannot be combined"):
        parse_run_selection(run_id="run1", run_after="2026-07-01")


def test_run_id_and_last_n_conflict():
    with pytest.raises(RunSelectionError, match="cannot be combined"):
        parse_run_selection(run_id="run1", run_last_n=2)


def test_window_and_last_n_conflict():
    with pytest.raises(RunSelectionError, match="cannot be combined"):
        parse_run_selection(run_before="2026-07-01", run_last_n=2)


def test_after_and_before_together_are_one_group():
    got = parse_run_selection(run_after="2026-07-01", run_before="2026-07-02")
    assert (got.after, got.before) == ("2026-07-01", "2026-07-02")


def test_last_n_must_be_positive():
    with pytest.raises(RunSelectionError, match="at least 1"):
        parse_run_selection(run_last_n=0)


# --- resolution: ids -----------------------------------------------------


def test_resolves_ids_by_prefix(store):
    wanted = a_run(store, KEY)
    a_run(store, KEY)
    got = resolve_runs(store, parse_run_selection(run_id=wanted[:20]))
    assert [r.id for r in got] == [wanted]


def test_unknown_id_is_an_error(store):
    a_run(store, KEY)
    with pytest.raises(RunSelectionError, match="no run matching"):
        resolve_runs(store, parse_run_selection(run_id="run_1900"))


def test_ambiguous_prefix_is_an_error(store):
    a_run(store, KEY)
    a_run(store, KEY)
    with pytest.raises(RunSelectionError, match="matches 2 runs"):
        resolve_runs(store, parse_run_selection(run_id="run_"))


def test_empty_selection_returns_every_run_oldest_first(store):
    newer = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    older = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    assert [r.id for r in resolve_runs(store, parse_run_selection())] == [older, newer]


# --- resolution: time windows -------------------------------------------


def test_bare_date_means_utc_midnight(store):
    before_midnight = backdate_run(store, a_run(store, KEY), "2026-07-01T23:59:59+00:00")
    after_midnight = backdate_run(store, a_run(store, KEY), "2026-07-02T00:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02"))
    assert [r.id for r in got] == [after_midnight]
    got = resolve_runs(store, parse_run_selection(run_before="2026-07-02"))
    assert [r.id for r in got] == [before_midnight, after_midnight]


def test_minute_precision_datetime(store):
    early = backdate_run(store, a_run(store, KEY), "2026-07-02T09:29:00+00:00")
    late = backdate_run(store, a_run(store, KEY), "2026-07-02T09:30:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:30"))
    assert [r.id for r in got] == [late]
    assert early not in [r.id for r in got]


def test_second_precision_datetime(store):
    only = backdate_run(store, a_run(store, KEY), "2026-07-02T09:30:15+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:30:15"))
    assert [r.id for r in got] == [only]


def test_explicit_offset_is_honoured(store):
    """09:00+08:00 is 01:00 UTC, so a run at 02:00 UTC is after it."""
    early = backdate_run(store, a_run(store, KEY), "2026-07-02T00:30:00+00:00")
    late = backdate_run(store, a_run(store, KEY), "2026-07-02T02:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:00+08:00"))
    assert [r.id for r in got] == [late]
    assert early not in [r.id for r in got]


def test_trailing_z_is_utc(store):
    only = backdate_run(store, a_run(store, KEY), "2026-07-02T10:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:00:00Z"))
    assert [r.id for r in got] == [only]


def test_run_id_as_a_boundary_is_inclusive(store):
    first = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00.500000+00:00")
    second = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after=first))
    assert [r.id for r in got] == [first, second]


def test_run_prefix_as_a_boundary(store):
    first = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_before=second[:20]))
    assert [r.id for r in got] == [first, second]


def test_unparseable_boundary_that_is_no_run_is_an_error(store):
    a_run(store, KEY)
    with pytest.raises(RunSelectionError, match="no run matching"):
        resolve_runs(store, parse_run_selection(run_after="last tuesday"))


# --- resolution: last_n and cache keys ----------------------------------


def test_last_n_returns_the_most_recent_oldest_first(store):
    backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    third = backdate_run(store, a_run(store, KEY), "2026-07-03T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_last_n=2))
    assert [r.id for r in got] == [second, third]


def test_cache_key_narrowing_applies_before_last_n(store):
    mine = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    backdate_run(store, a_run(store, OTHER), "2026-07-02T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_last_n=1), cache_key_hashes=[KEY.hash])
    assert [r.id for r in got] == [mine]


def test_id_of_another_provider_is_dropped_with_a_warning(store, caplog):
    theirs = a_run(store, OTHER)
    with caplog.at_level("WARNING"):
        got = resolve_runs(
            store, parse_run_selection(run_id=theirs), cache_key_hashes=[KEY.hash]
        )
    assert got == []
    assert "do not belong to the selected provider" in caplog.text


def test_selection_matching_nothing_is_not_an_error(store):
    a_run(store, KEY)
    assert resolve_runs(store, parse_run_selection(run_after="2099-01-01")) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest framework_tests/test_runselect.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'llmeval.runselect'`.

- [ ] **Step 3: Implement the module**

Create `rewrite/llmeval/runselect.py`:

```python
"""Run selection — which runs a result-reading stage looks at.

``grade`` and ``report`` both read stored results, and both need the same answer to "which
runs?". This module owns the *meaning* of the four flags that answer it; the SQL lives in
:meth:`llmeval.store.Store.select_runs`.

Four flags in three mutually exclusive groups:

* ``--run-id a,b,c`` — exactly these runs (full ids or unambiguous prefixes)
* ``--run-after`` / ``--run-before`` — a time window; either end may be a timestamp or a run
* ``--run-last-n N`` — the N most recent runs

They are exclusive because combining them has no single obvious reading: is
``--run-last-n 3 --run-after 2026-01-01`` "the last 3 runs, then drop the old ones" or "the
3 most recent of those after the date"? Rather than pick one and surprise half the users,
we refuse.

Timestamps are **UTC when no offset is given** — that is what ``runs.started_at`` holds and
what the run id embeds, so a bare date lines up with the ids you see in the log. An explicit
offset (``+08:00``, a trailing ``Z``) is honoured and converted. Both bounds are
**inclusive**, to the whole second.

A boundary is tried as a timestamp *first* and only then as a run id, so a run prefix that
also parses as a date would be read as the date. Run ids start with ``run_``, which no date
format accepts, so this is not reachable in practice.

The error split matters: a selector that *cannot be satisfied* — an id matching no run, a
boundary that is neither a timestamp nor a run — raises :class:`RunSelectionError`, which
the CLI turns into a message and exit 2. A selector that is well formed but simply matches
nothing returns an empty list, because "the last 3 runs of a provider that has had none" is
a legitimate answer rather than user error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from llmeval.store import RunRow, Store

logger = logging.getLogger(__name__)

# What the store's bounds compare against: whole seconds, UTC, no offset suffix.
_BOUNDARY_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Tried in order. Naive values get UTC attached; anything with an offset falls through to
# ``fromisoformat`` below.
_DATETIME_FORMS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S")


class RunSelectionError(ValueError):
    """A run selection that cannot be satisfied.

    Typed so the CLI reports it as a plain message instead of a traceback — like
    :class:`llmeval.store.IncompatibleSchema`, it is expected user error, not a bug.
    """


@dataclass(frozen=True)
class RunSelection:
    """A parsed, not-yet-resolved run selection. See :func:`resolve_runs`."""

    ids: tuple[str, ...] = ()
    after: str | None = None
    before: str | None = None
    last_n: int | None = None


def _split_ids(run_id: str | Sequence[str] | None) -> tuple[str, ...]:
    """Flatten ``--run-id`` into ids: comma-separated, and repeatable.

    Both forms exist because both are natural — ``--run-id a,b`` when pasting from a
    script, ``--run-id a --run-id b`` when building a command up by hand.
    """
    if run_id is None:
        return ()
    items = [run_id] if isinstance(run_id, str) else list(run_id)
    out: list[str] = []
    for item in items:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    return tuple(out)


def parse_run_selection(
    run_id: str | Sequence[str] | None = None,
    run_after: str | None = None,
    run_before: str | None = None,
    run_last_n: int | None = None,
) -> RunSelection:
    """Validate the four flags and package them. No store access, no I/O."""
    ids = _split_ids(run_id)
    groups = []
    if ids:
        groups.append("--run-id")
    if run_after is not None or run_before is not None:
        groups.append("--run-after/--run-before")
    if run_last_n is not None:
        groups.append("--run-last-n")
    if len(groups) > 1:
        raise RunSelectionError(
            f"{' and '.join(groups)} cannot be combined; pick one way to select runs"
        )
    if run_last_n is not None and run_last_n < 1:
        raise RunSelectionError(f"--run-last-n must be at least 1, got {run_last_n}")
    return RunSelection(ids=ids, after=run_after, before=run_before, last_n=run_last_n)


def _as_utc_datetime(value: str) -> datetime | None:
    """Parse a boundary as a timestamp, or return ``None`` if it isn't one.

    The explicit ``strptime`` passes come first so a bare date or minute-precision value is
    accepted *as UTC*; ``fromisoformat`` then handles the offset forms. Doing it in that
    order is what makes "naive means UTC" true rather than platform-dependent.
    """
    text = value.strip()
    for fmt in _DATETIME_FORMS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_one(store: Store, value: str) -> str:
    """Expand one id/prefix, converting the store's ``KeyError`` into our error type."""
    try:
        return store.resolve_run(value)
    except KeyError as exc:
        raise RunSelectionError(exc.args[0] if exc.args else str(exc)) from exc


def _boundary(store: Store, value: str | None) -> str | None:
    """One end of the window, as a string the store can compare.

    Truncated to whole seconds: the stored ``started_at`` carries microseconds, so a
    ``<=`` against a second-precision bound derived from a run would otherwise exclude
    the very run the user named as the end cap.
    """
    if value is None:
        return None
    when = _as_utc_datetime(value)
    if when is not None:
        return when.strftime(_BOUNDARY_FORMAT)
    run = store.get_run(_resolve_one(store, value))
    if run is None:  # pragma: no cover - resolve_run already guarantees existence
        raise RunSelectionError(f"run {value!r} vanished between resolve and read")
    return run.started_at[:19]


def resolve_runs(
    store: Store,
    selection: RunSelection,
    cache_key_hashes: Sequence[str] | None = None,
) -> list[RunRow]:
    """Expand a selection into concrete runs, **oldest first**.

    ``cache_key_hashes`` narrows to particular provider identities *before* the selection
    is applied, so ``--provider X --run-last-n 3`` is the last three runs of X rather than
    the last three runs overall intersected with X.

    Naming a run that exists but belongs to a different provider is a warning, not an
    error: the id was valid, it just isn't in the identity you asked about, and saying so
    is more useful than either failing or silently returning nothing.
    """
    ids = [_resolve_one(store, i) for i in selection.ids] if selection.ids else None
    runs = store.select_runs(
        ids=ids,
        after=_boundary(store, selection.after),
        before=_boundary(store, selection.before),
        last_n=selection.last_n,
        cache_key_hashes=None if cache_key_hashes is None else list(cache_key_hashes),
    )
    if ids:
        found = {r.id for r in runs}
        dropped = [i for i in ids if i not in found]
        if dropped:
            logger.warning(
                "run(s) %s do not belong to the selected provider(s); ignoring",
                ", ".join(dropped),
            )
    return runs
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest framework_tests/test_runselect.py -v
.venv/bin/python -m pylint llmeval
```
Expected: all PASS, pylint 10/10.

- [ ] **Step 5: Commit**

```bash
git add rewrite/llmeval/runselect.py rewrite/framework_tests/test_runselect.py
git commit -m "Add run selection: --run-id, --run-after/--run-before, --run-last-n"
```

---

### Task 3: `llmeval/resultrows.py`

The report's data layer: stored results flattened into CSV rows across N runs. Absorbs the row-shaping logic from `reporting/run_report.py`.

**Files:**
- Create: `rewrite/llmeval/resultrows.py`
- Test: `rewrite/framework_tests/test_resultrows.py`
- Read for reference (do not modify yet): `rewrite/reporting/run_report.py`, `rewrite/reporting_tests/test_run_report.py`

**Interfaces:**
- Consumes: `Store.get_results_for_run`, `Store.get_gradings`, `RunRow`, `ResultRow`, `TestCase`.
- Produces:
  ```python
  suite_of(test_id: str, case: TestCase | None) -> str | None
  result_columns(with_tests: bool) -> list[str]
  result_rows(
      store: Store,
      runs: Sequence[RunRow],
      cases_by_id: Mapping[str, TestCase] | None = None,
  ) -> list[dict[str, Any]]
  write_csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str], path: str) -> str
  ```

- [ ] **Step 1: Write the failing tests**

Create `rewrite/framework_tests/test_resultrows.py`. Several of these are ported from
`reporting_tests/test_run_report.py` (which Task 7 deletes) — the behaviours are the same,
the module and the multi-run dimension are new.

```python
import csv

import pytest
from conftest import a_run, backdate_run

from llmeval.cache_key import compute_cache_key
from llmeval.models import TestCase
from llmeval.resultrows import result_columns, result_rows, suite_of, write_csv
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


KEY = compute_cache_key(model="m1")


def a_case(test_id, **metadata):
    return TestCase.from_dict(
        {"id": test_id, "user": "the prompt", "metadata": metadata, "assertions": []}
    )


def runs_of(store, *ids):
    """The RunRow objects for these ids, in the order given."""
    return [store.get_run(i) for i in ids]


# --- suite resolution (ported) ------------------------------------------


def test_suite_comes_from_metadata_when_available():
    assert suite_of("anything-at-all", a_case("x", suite="multifaceted")) == "multifaceted"


def test_suite_falls_back_to_the_id_shape():
    assert suite_of("simple_facts-6c3396ab0e", None) == "simple_facts"


def test_suite_fallback_keeps_a_variant_suffixed_id_intact():
    assert suite_of("research_rubrics-abc1234567-geval", None) == "research_rubrics"


def test_suite_fallback_keeps_hyphens_in_the_suite_name():
    assert suite_of("my-suite-0123456789", None) == "my-suite"


def test_suite_is_none_for_a_hand_written_id():
    assert suite_of("hand-written", None) is None


# --- one row per (result, assertion) ------------------------------------


def test_one_row_per_grading(store):
    run = a_run(store, KEY)
    rid = store.add_result_row("t-0123456789", run_id=run, output="hello")
    store.set_grading(rid, "a1", type="icontains", score=1.0, passed=True)
    store.set_grading(rid, "a2", type="not_contains", score=0.0, passed=False)
    rows = result_rows(store, runs_of(store, run))
    assert [(r["assertion_key"], r["passed"]) for r in rows] == [("a1", True), ("a2", False)]


def test_ungraded_result_still_yields_one_row(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="hello")
    rows = result_rows(store, runs_of(store, run))
    assert len(rows) == 1
    assert rows[0]["assertion_key"] is None
    assert rows[0]["output"] == "hello"


def test_errored_result_yields_one_row_with_no_grading_columns(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, error="timeout after 60s", latency_ms=60001.4)
    rows = result_rows(store, runs_of(store, run))
    assert len(rows) == 1
    assert rows[0]["error"] == "timeout after 60s"
    assert rows[0]["score"] is None and rows[0]["assertion_key"] is None
    # Latency on an error row is the whole point: it separates "the timeout is too tight"
    # from "the provider is down".
    assert rows[0]["latency_ms"] == 60001.4


# --- ordering -----------------------------------------------------------


def test_rows_are_grouped_by_run_in_chronological_order(store):
    older = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    newer = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    store.add_result_row("t-0123456789", run_id=newer, output="second")
    store.add_result_row("t-0123456789", run_id=older, output="first")
    rows = result_rows(store, runs_of(store, older, newer))
    assert [r["output"] for r in rows] == ["first", "second"]
    assert [r["run_id"] for r in rows] == [older, newer]


def test_attempts_within_a_run_are_in_chronological_order(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, error="boom")
    rid = store.add_result_row("t-0123456789", run_id=run, output="worked")
    store.set_grading(rid, "a1", passed=True)
    rows = result_rows(store, runs_of(store, run))
    assert [(r["attempt"], r["error"], r["assertion_key"]) for r in rows] == [
        (0, "boom", None),
        (1, None, "a1"),
    ]


def test_run_metadata_is_on_every_row(store):
    run = backdate_run(
        store, store.create_run(KEY, provider_name="fidaro-prod"), "2026-07-01T09:00:00+00:00"
    )
    store.add_result_row("t-0123456789", run_id=run, output="x")
    row = result_rows(store, runs_of(store, run))[0]
    assert row["provider"] == "fidaro-prod"
    assert row["run_started_at"] == "2026-07-01T09:00:00+00:00"
    assert row["cache_key_hash"] == KEY.hash


# --- field shaping (ported) --------------------------------------------


def test_tokens_are_flattened(store):
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789",
        run_id=run,
        output="x",
        tokens={"prompt_tokens": 11, "completion_tokens": 22, "total_tokens": 33},
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert (row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]) == (11, 22, 33)


def test_missing_tokens_are_empty_not_an_error(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    assert result_rows(store, runs_of(store, run))[0]["total_tokens"] is None


def test_surrounding_whitespace_is_trimmed_from_text_fields(store):
    run = a_run(store, KEY)
    store.add_result_row(
        "t-0123456789", run_id=run, output="\n\n\nThe answer  ", reasoning="  thinking\n"
    )
    row = result_rows(store, runs_of(store, run))[0]
    assert row["output"] == "The answer"
    assert row["reasoning"] == "thinking"


def test_internal_formatting_is_preserved(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="\n\nHeading\n\n* one\n* two\n")
    assert result_rows(store, runs_of(store, run))[0]["output"] == "Heading\n\n* one\n* two"


def test_latency_is_rounded_to_one_decimal(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x", latency_ms=16531.234567)
    assert result_rows(store, runs_of(store, run))[0]["latency_ms"] == 16531.2


# --- testcase enrichment and selection ---------------------------------


def test_testcases_add_prompt_and_classification(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    cases = {"t-0123456789": a_case("t-0123456789", request_type="coding", domain="science_stem")}
    row = result_rows(store, runs_of(store, run), cases)[0]
    assert row["prompt"] == "the prompt"
    assert (row["request_type"], row["domain"]) == ("coding", "science_stem")
    assert list(row) == result_columns(with_tests=True)


def test_test_columns_are_absent_without_testcases(store):
    run = a_run(store, KEY)
    store.add_result_row("t-0123456789", run_id=run, output="x")
    row = result_rows(store, runs_of(store, run))[0]
    assert "prompt" not in row
    assert list(row) == result_columns(with_tests=False)


def test_testcases_select_as_well_as_enrich(store):
    """A result whose test is not in the loaded set is filtered out entirely."""
    run = a_run(store, KEY)
    store.add_result_row("wanted-0123456789", run_id=run, output="keep")
    store.add_result_row("other-0123456789", run_id=run, output="drop")
    cases = {"wanted-0123456789": a_case("wanted-0123456789")}
    rows = result_rows(store, runs_of(store, run), cases)
    assert [r["test_id"] for r in rows] == ["wanted-0123456789"]


def test_no_runs_means_no_rows(store):
    assert result_rows(store, []) == []


# --- CSV writing -------------------------------------------------------


def test_write_csv_round_trips(tmp_path, store):
    run = a_run(store, KEY)
    rid = store.add_result_row("t-0123456789", run_id=run, output="hello")
    store.set_grading(rid, "a1", passed=True, score=1.0)
    rows = result_rows(store, runs_of(store, run))
    columns = result_columns(with_tests=False)
    path = write_csv(rows, columns, str(tmp_path / "out" / "rows.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == columns
        got = list(reader)
    assert len(got) == 1
    assert got[0]["output"] == "hello"


def test_write_csv_of_no_rows_still_writes_the_header(tmp_path):
    columns = result_columns(with_tests=False)
    path = write_csv([], columns, str(tmp_path / "empty.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == columns
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest framework_tests/test_resultrows.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'llmeval.resultrows'`.

- [ ] **Step 3: Implement the module**

Create `rewrite/llmeval/resultrows.py`:

```python
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

That last case is why this exists. A test that errored produces no grading and therefore
no statistics, so a report built from gradings alone renders it as absence. Here it is a
row you can filter for.

**Ordering** is run (chronological) then test then attempt, which reads as a history: each
run's tests in turn, and within a test its failed attempts before the one that answered.
Test order inside a run is by id rather than by first-attempt time — deterministic, and the
viewer re-sorts on any column anyway.

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

# Column order is the reading order of the report: where it came from, what was asked,
# what came back, how it scored, then the provenance you only want when something is odd.
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

    Every generator writes ``suite`` into metadata, so with test cases loaded this is
    exact. Without them the id pattern is the only signal available.
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
    swapped ``<thinking>`` tags for newlines. Rendered in a table cell, those leading
    blanks push the answer out of view before a single word is read.

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

    ``with_tests`` follows whether test cases were loaded. The prompt/classification
    columns are then *absent* rather than empty — an absent column says "you didn't ask
    for this", an empty one says "there was nothing to show", and they are different
    answers.
    """
    cols = list(_RUN_COLUMNS) + list(_BASE_COLUMNS)
    if with_tests:
        cols += _TEST_COLUMNS
    return cols + _RESULT_COLUMNS + _GRADING_COLUMNS + _PROVENANCE_COLUMNS


def result_rows(
    store: Store,
    runs: Sequence[RunRow],
    cases_by_id: Mapping[str, TestCase] | None = None,
) -> list[dict[str, Any]]:
    """Flatten the given runs into report rows, in the order ``runs`` arrives in.

    Ordering across runs is the caller's decision, already made:
    :func:`llmeval.runselect.resolve_runs` returns runs oldest-first, which is the
    chronological grouping the report wants.

    :param cases_by_id: test cases keyed by id. When given it **selects as well as
        enriches** — a result whose test is absent is dropped, which is what makes
        ``--filter k=v`` mean anything, and matches how ``run`` and ``grade`` already treat
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
            shared: dict[str, Any] = {
                "run_id": run.id,
                "run_started_at": run.started_at,
                "provider": run.provider_name,
                "test_id": result.test_id,
                "attempt": result.attempt,
                "suite": suite_of(result.test_id, case),
                "output": _text(result.output),
                "reasoning": _text(result.reasoning),
                "error": result.error,
                "latency_ms": (
                    round(result.latency_ms, 1) if result.latency_ms is not None else None
                ),
                **_tokens(result),
                "cache_key_hash": result.cache_key_hash,
                "result_id": result.id,
            }
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

    The header is written even for zero rows: an empty selection should produce a table
    with no rows, not a file a reader cannot parse.
    """
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows([{c: r.get(c) for c in columns} for r in rows])
    return path
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest framework_tests/test_resultrows.py -v
.venv/bin/python -m pylint llmeval
```
Expected: all PASS, pylint 10/10.

- [ ] **Step 5: Commit**

```bash
git add rewrite/llmeval/resultrows.py rewrite/framework_tests/test_resultrows.py
git commit -m "Add result-rows builder: one row per (result, assertion), errors included"
```

---

### Task 4: Grade only the selected runs

**Files:**
- Modify: `rewrite/llmeval/grade.py:35-79`
- Test: `rewrite/framework_tests/test_grade.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces:
  ```python
  grade_testcase(store, testcase, cache_key_hash, judge=None, regrade=False,
                 run_ids: Collection[str] | None = None) -> None
  grade(store, testcases, cache_key_hash, judge=None, regrade=False,
        run_ids: Collection[str] | None = None) -> None
  ```
  `None` means every run for the cache key — the existing behaviour.

- [ ] **Step 1: Write the failing tests**

Append to `rewrite/framework_tests/test_grade.py`:

```python
def test_grades_every_result_across_runs_by_default(store):
    """Two runs of the same test, both graded — one grading per result, not per test."""
    first = store.add_result_row("t1", run_id=a_run(store, KEY), output="Paris is it")
    second = store.add_result_row("t1", run_id=a_run(store, KEY), output="Also Paris")
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash)
    assert len(store.get_gradings(first)) == 1
    assert len(store.get_gradings(second)) == 1


def test_run_ids_narrows_grading_to_those_runs(store):
    wanted_run = a_run(store, KEY)
    other_run = a_run(store, KEY)
    wanted = store.add_result_row("t1", run_id=wanted_run, output="Paris")
    other = store.add_result_row("t1", run_id=other_run, output="Paris")
    grade_testcase(
        store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash, run_ids=[wanted_run]
    )
    assert len(store.get_gradings(wanted)) == 1
    assert store.get_gradings(other) == []


def test_empty_run_ids_grades_nothing(store):
    rid = seed(store)
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash, run_ids=[])
    assert store.get_gradings(rid) == []


def test_error_rows_are_still_skipped_when_narrowing(store):
    run = a_run(store, KEY)
    store.add_result_row("t1", run_id=run, error="timeout")
    grade_testcase(
        store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash, run_ids=[run]
    )
    assert list(store.iter_graded_results(KEY.hash)) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest framework_tests/test_grade.py -v
```
Expected: `test_run_ids_narrows_grading_to_those_runs` and `test_empty_run_ids_grades_nothing`
FAIL with `TypeError: grade_testcase() got an unexpected keyword argument 'run_ids'`.
`test_grades_every_result_across_runs_by_default` should already PASS — it documents
behaviour that exists.

- [ ] **Step 3: Implement the narrowing**

In `rewrite/llmeval/grade.py`, change the `typing` import to include `Collection`:

```python
from typing import Callable, Collection, Iterable
```

Then replace `grade_testcase` and `grade` with:

```python
def grade_testcase(
    store: Store,
    testcase: TestCase,
    cache_key_hash: str,
    judge: Callable[[str], str] | None = None,
    regrade: bool = False,
    run_ids: Collection[str] | None = None,
) -> None:
    """Grade every cached (non-error) result of ``testcase`` under one cache key.

    Every result, not just the newest: a grading belongs to a *result*, so re-running a
    test adds a row to grade rather than superseding one.

    :param run_ids: restrict to results produced by these runs. ``None`` means every run
        for the cache key. An **empty** collection means no runs, and so grades nothing —
        the same None-versus-empty distinction the store's selection uses.
    """
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
    run_ids: Collection[str] | None = None,
) -> None:
    for testcase in testcases:
        grade_testcase(
            store, testcase, cache_key_hash, judge=judge, regrade=regrade, run_ids=run_ids
        )
```

Also extend the module docstring (currently lines 1-5) with a paragraph, so the
per-result-not-per-test guarantee is documented where a reader will find it:

```python
"""Grading: apply a test case's assertions to *cached* results.

This stage never calls the model under test — only the (optional) judge. So you can edit
assertions, add new ones, or change the judge and re-grade existing outputs cheaply.

A grading belongs to a **result**, not to a test: ``gradings`` is unique on
``(result_id, assertion_key)``, so every attempt a test ever produced can carry its own
score and re-running a test adds a row to grade rather than superseding one. Attempts that
**errored** are skipped entirely — there is no output to assert against, and the error row
is the finding.
"""
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest framework_tests/test_grade.py -v
.venv/bin/python -m pylint llmeval
```
Expected: all PASS, pylint 10/10.

- [ ] **Step 5: Commit**

```bash
git add rewrite/llmeval/grade.py rewrite/framework_tests/test_grade.py
git commit -m "Let grade narrow to selected runs"
```

---

### Task 5: CLI — run-selection flags, `report` emits CSV, `compare-report`

**Files:**
- Modify: `rewrite/llmeval/cli.py`
- Test: `rewrite/framework_tests/test_cli.py`

**Interfaces:**
- Consumes: `parse_run_selection`, `resolve_runs`, `RunSelectionError` (Task 2);
  `result_columns`, `result_rows`, `write_csv` (Task 3); `grade(..., run_ids=)` (Task 4).
- Produces: subcommands `report` (CSV) and `compare-report` (HTML), and the flags
  `--run-id`, `--run-after`, `--run-before`, `--run-last-n` on `grade` and `report`.

- [ ] **Step 1: Write the failing tests**

Append to `rewrite/framework_tests/test_cli.py`:

```python
def _echo_setup(tmp_path):
    """Generate one test case, run it with the echo provider, and grade it."""
    csv_src = tmp_path / "facts.csv"
    csv_src.write_text('user,__expected\n"What is the capital of France?","icontains:Paris"\n')
    tc_dir = tmp_path / "testcases"
    main(["generate-csv", "--csv", str(csv_src), "--suite", "facts", "--out", str(tc_dir)])
    prov = tmp_path / "echo.json"
    prov.write_text(
        json.dumps({"name": "echo", "model": "echo", "extra": {"provider_impl": "echo"}})
    )
    db = str(tmp_path / "db.sqlite3")
    main(["run", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db])
    main(["grade", "--testcases", str(tc_dir), "--provider", str(prov), "--db", db])
    return str(tc_dir), str(prov), db


def test_report_writes_a_csv(tmp_path):
    import csv as csvmod

    tc_dir, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "rows.csv"
    rc = main(["report", "--db", db, "--provider", prov, "--testcases", tc_dir, "--out", str(out)])
    assert rc == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csvmod.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["assertion_key"].startswith("icontains:")
    assert rows[0]["passed"] == "True"
    assert rows[0]["prompt"] == "What is the capital of France?"
    assert rows[0]["run_id"].startswith("run_")
    assert rows[0]["latency_ms"] != ""


def test_report_without_testcases_omits_the_prompt_column(tmp_path):
    import csv as csvmod

    _, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "rows.csv"
    assert main(["report", "--db", db, "--provider", prov, "--out", str(out)]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        assert "prompt" not in (csvmod.DictReader(f).fieldnames or [])


def test_report_on_a_missing_db_is_an_error(tmp_path):
    rc = main(["report", "--db", str(tmp_path / "nope.sqlite3"), "--out", str(tmp_path / "o.csv")])
    assert rc == 2


def test_report_with_conflicting_run_selection_is_an_error(tmp_path):
    _, prov, db = _echo_setup(tmp_path)
    rc = main(
        ["report", "--db", db, "--out", str(tmp_path / "o.csv"),
         "--run-last-n", "1", "--run-after", "2026-07-01"]
    )
    assert rc == 2


def test_report_with_an_unknown_run_id_is_an_error(tmp_path):
    _, prov, db = _echo_setup(tmp_path)
    rc = main(["report", "--db", db, "--out", str(tmp_path / "o.csv"), "--run-id", "run_1900"])
    assert rc == 2


def test_report_run_selection_that_matches_nothing_is_not_an_error(tmp_path):
    import csv as csvmod

    _, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "rows.csv"
    rc = main(["report", "--db", db, "--out", str(out), "--run-after", "2099-01-01"])
    assert rc == 0
    with open(out, newline="", encoding="utf-8") as f:
        reader = csvmod.DictReader(f)
        assert reader.fieldnames is not None  # header written
        assert list(reader) == []


def test_report_last_n_selects_only_the_newest_run(tmp_path):
    import csv as csvmod

    tc_dir, prov, db = _echo_setup(tmp_path)
    # A second run of the same test: --mode always appends rather than reusing the cache.
    main(["run", "--testcases", tc_dir, "--provider", prov, "--db", db, "--mode", "always"])
    out = tmp_path / "rows.csv"
    assert main(["report", "--db", db, "--out", str(out), "--run-last-n", "1"]) == 0
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csvmod.DictReader(f))
    assert len({r["run_id"] for r in rows}) == 1


def test_compare_report_writes_the_statistics_html(tmp_path):
    _, prov, db = _echo_setup(tmp_path)
    out = tmp_path / "compare.html"
    rc = main(["compare-report", "--providers", prov, "--db", db, "--out", str(out)])
    assert rc == 0
    assert "llmeval comparison" in out.read_text()


def test_grade_accepts_run_selection_flags(tmp_path):
    tc_dir, prov, db = _echo_setup(tmp_path)
    rc = main(
        ["grade", "--testcases", tc_dir, "--provider", prov, "--db", db, "--run-last-n", "1"]
    )
    assert rc == 0


def test_grade_with_conflicting_run_selection_is_an_error(tmp_path):
    tc_dir, prov, db = _echo_setup(tmp_path)
    rc = main(
        ["grade", "--testcases", tc_dir, "--provider", prov, "--db", db,
         "--run-id", "run_x", "--run-last-n", "1"]
    )
    assert rc == 2


def test_run_selection_flags_default_to_none():
    parser = build_parser()
    args = parser.parse_args(["report", "--out", "o.csv"])
    assert (args.run_id, args.run_after, args.run_before, args.run_last_n) == (
        None, None, None, None,
    )
```

Also update the existing `test_cli_pipeline_offline` (lines 32-36) — its final block calls
`main(["report", "--providers", ...])` and expects HTML, which is now `compare-report`.
Replace that block with:

```python
    # report now emits CSV rows; the statistics HTML moved to compare-report
    rows_csv = tmp_path / "rows.csv"
    assert main(["report", "--provider", str(prov), "--db", db, "--out", str(rows_csv)]) == 0
    assert rows_csv.exists() and rows_csv.read_text().strip() != ""

    out = tmp_path / "r.html"
    assert main(["compare-report", "--providers", str(prov), "--db", db, "--out", str(out)]) == 0
    assert out.exists() and out.read_text().strip() != ""
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest framework_tests/test_cli.py -v
```
Expected: FAIL — `invalid choice: 'compare-report'`, and `report` rejecting `--provider`.

- [ ] **Step 3: Update the imports and add the shared flag helper**

In `rewrite/llmeval/cli.py`, add to the imports:

```python
from llmeval.resultrows import result_columns, result_rows, write_csv
from llmeval.runselect import RunSelectionError, parse_run_selection, resolve_runs
```

Add these two helpers next to `_filters` and `_judge`:

```python
def _run_selection(args):
    """The run-selection flags, validated. Raises ``RunSelectionError`` on a bad combination."""
    return parse_run_selection(
        run_id=args.run_id,
        run_after=args.run_after,
        run_before=args.run_before,
        run_last_n=args.run_last_n,
    )


def _selected_runs(store, args, cache_key_hashes=None) -> list[str]:
    """Resolve the run-selection flags to run ids, oldest first."""
    return [r.id for r in resolve_runs(store, _run_selection(args), cache_key_hashes)]
```

- [ ] **Step 4: Rewrite `cmd_grade` to narrow by run**

Replace `cmd_grade` (currently lines 147-155):

```python
def cmd_grade(args) -> int:
    store = Store(args.db)
    try:
        tcs = load_testcases(args.testcases, _filters(args.filter))
        cfg = load_provider_config(args.provider)
        key_hash = cfg.cache_key().hash
        run_ids = _selected_runs(store, args, [key_hash])
        if not run_ids:
            logger.warning("no runs match the selection for %r; nothing to grade", cfg.name)
            return 0
        logger.info(
            "grading %d test(s) over %d run(s) for %r (regrade=%s)",
            len(tcs), len(run_ids), cfg.name, args.regrade,
        )
        grade(store, tcs, key_hash, judge=_judge(args), regrade=args.regrade, run_ids=run_ids)
        logger.info("graded %d test(s) for %r", len(tcs), cfg.name)
        return 0
    finally:
        store.close()
```

- [ ] **Step 5: Replace `cmd_report` and add `cmd_compare_report`**

Replace `cmd_report` (currently lines 171-180) with both of these:

```python
def cmd_report(args) -> int:
    """Write the selected result rows as CSV. Rendering is porcelain's job.

    Data out, no HTML: turning a CSV into a page (and opening a browser) is a workflow,
    which per CLAUDE.md lives in ``reporting/`` rather than here. So:

        llmeval report --run-last-n 3 --out rows.csv
        python -m reporting.csv_table rows.csv -o rows.html
    """
    if not os.path.exists(args.db):
        # sqlite3.connect would happily create an empty database and the user would get
        # "0 rows" for what is really a wrong --db path.
        logger.error("no results database at %s", args.db)
        return 2
    store = Store(args.db)
    try:
        hashes = None
        if args.provider:
            hashes = [load_provider_config(p).cache_key().hash for p in args.provider]
        runs = resolve_runs(store, _run_selection(args), hashes)
        cases_by_id = None
        if args.testcases:
            cases_by_id = {c.id: c for c in load_testcases(args.testcases, _filters(args.filter))}
        rows = result_rows(store, runs, cases_by_id)
        columns = result_columns(cases_by_id is not None)
    finally:
        store.close()
    write_csv(rows, columns, args.out)
    logger.info("wrote %d row(s) from %d run(s) -> %s", len(rows), len(runs), args.out)
    return 0


def cmd_compare_report(args) -> int:
    store = Store(args.db)
    configs = [load_provider_config(p) for p in args.providers]
    pairs = [(c.name, c.cache_key().hash) for c in configs]
    metrics = args.metrics if args.metrics else [None]
    ckey = comparison_key(configs, DEFAULT_CRITERION, args.order) if args.order else None
    write_report(store, pairs, metrics, args.out, baseline_name=args.baseline, comparison_key=ckey)
    logger.info("wrote comparison report -> %s", args.out)
    store.close()
    return 0
```

- [ ] **Step 6: Wire the parsers**

In `build_parser`, add the flag helper next to `add_db` / `add_filters`:

```python
    def add_run_selection(sp):
        """The four run-selection flags. Shared by every stage that reads stored results."""
        sp.add_argument(
            "--run-id", action="append",
            help="comma-separated run ids or unambiguous prefixes (repeatable)",
        )
        sp.add_argument(
            "--run-after",
            help="only runs at or after this point: YYYY-MM-DD or YYYY-MM-DDTHH:MM "
            "(UTC unless an offset is given), or a run id",
        )
        sp.add_argument(
            "--run-before",
            help="only runs at or before this point; same forms as --run-after",
        )
        sp.add_argument("--run-last-n", type=int, help="only the N most recent runs")
```

Add `add_run_selection(gr)` to the `grade` parser block (after `add_filters(gr)`).

Replace the whole `report` parser block (currently lines 269-278) with:

```python
    rp = sub.add_parser(
        "report",
        help="write the selected result rows as CSV (one row per result x assertion, "
        "plus one per errored result)",
    )
    rp.add_argument("--out", default="results.csv", help="output CSV path")
    rp.add_argument(
        "--provider", action="append",
        help="provider config JSON (repeatable; default: every provider in the DB)",
    )
    rp.add_argument(
        "--testcases",
        help="testcases dir/file; selects which tests appear and adds the prompt, "
        "request_type and domain columns",
    )
    add_db(rp)
    add_filters(rp)
    add_run_selection(rp)
    rp.set_defaults(func=cmd_report)

    cr = sub.add_parser(
        "compare-report", help="render an HTML comparison report (statistics + pick-best)"
    )
    cr.add_argument("--providers", required=True, nargs="+")
    cr.add_argument("--baseline", help="baseline provider name (for deltas)")
    cr.add_argument("--metrics", nargs="*", help="metric names (default: overall)")
    cr.add_argument(
        "--order", choices=["as_is", "random", "both"], help="include pick-best win rates"
    )
    cr.add_argument("--out", default="report.html")
    add_db(cr)
    cr.set_defaults(func=cmd_compare_report)
```

- [ ] **Step 7: Catch `RunSelectionError` in `main`**

Change the `except` clause in `main` (currently line 298):

```python
    except (IncompatibleSchema, RunSelectionError) as exc:
        # Expected conditions (a DB from an older build, a selection that can't be
        # satisfied), not bugs — a message, no traceback.
        logger.error("%s", exc)
        return 2
```

- [ ] **Step 8: Update the module docstring**

Replace the usage block at the top of `cli.py` (lines 3-7):

```python
    llmeval generate-csv --csv f.csv --suite facts --out testcases/
    llmeval run      --testcases testcases/ --provider configs/fidaro_prod.json
    llmeval grade    --testcases testcases/ --provider configs/fidaro_prod.json --run-last-n 1
    llmeval pickbest --testcases testcases/ --providers a.json b.json --order both
    llmeval report   --run-last-n 3 --testcases testcases/ --out results.csv
    llmeval compare-report --providers a.json b.json --baseline fidaro-prod --out report.html
```

and add a paragraph after the existing one about the shared DB:

```
``grade`` and ``report`` both read stored results, so both take the same run-selection
flags (``--run-id``, ``--run-after``/``--run-before``, ``--run-last-n``); see
:mod:`llmeval.runselect`. ``report`` emits **CSV** — rendering it as a page is porcelain,
so pipe it to ``python -m reporting.csv_table``.
```

- [ ] **Step 9: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest framework_tests/ -v
.venv/bin/python -m pylint llmeval
```
Expected: all PASS, pylint 10/10. If pylint complains that `cmd_report` has too many
locals or branches, extract the provider-hash lookup into a module-level helper rather
than adding a disable comment.

- [ ] **Step 10: Commit**

```bash
git add rewrite/llmeval/cli.py rewrite/framework_tests/test_cli.py
git commit -m "report emits result-rows CSV; statistics report becomes compare-report"
```

---

### Task 6: `csv_table --open` / `--no-open`

**Files:**
- Modify: `rewrite/reporting/csv_table.py`
- Test: `rewrite/reporting_tests/test_csv_table.py` (append)

**Interfaces:**
- Consumes: nothing new.
- Produces: `csv_table.open_in_browser(path: str) -> bool`, and `--open` / `--no-open` on
  `python -m reporting.csv_table`.

- [ ] **Step 1: Write the failing tests**

Append to `rewrite/reporting_tests/test_csv_table.py`:

```python
def test_open_is_the_default(tmp_path, monkeypatch):
    src = tmp_path / "d.csv"
    src.write_text("a,b\n1,2\n")
    out = tmp_path / "t.html"
    opened = []
    monkeypatch.setattr(csv_table, "open_in_browser", lambda p: opened.append(p) or True)
    assert csv_table.main([str(src), "-o", str(out)]) == 0
    assert opened == [str(out)]


def test_no_open_suppresses_the_launch(tmp_path, monkeypatch):
    src = tmp_path / "d.csv"
    src.write_text("a,b\n1,2\n")
    out = tmp_path / "t.html"
    opened = []
    monkeypatch.setattr(csv_table, "open_in_browser", lambda p: opened.append(p) or True)
    assert csv_table.main([str(src), "-o", str(out), "--no-open"]) == 0
    assert opened == []


def test_nothing_is_opened_when_writing_to_stdout(tmp_path, monkeypatch, capsys):
    src = tmp_path / "d.csv"
    src.write_text("a,b\n1,2\n")
    opened = []
    monkeypatch.setattr(csv_table, "open_in_browser", lambda p: opened.append(p) or True)
    assert csv_table.main([str(src)]) == 0
    assert opened == []
    assert "<!doctype html>" in capsys.readouterr().out


def test_a_failing_launcher_still_exits_zero(tmp_path, monkeypatch, capsys):
    src = tmp_path / "d.csv"
    src.write_text("a,b\n1,2\n")
    out = tmp_path / "t.html"

    def boom(argv, check):
        raise OSError("no such tool")

    monkeypatch.setattr(csv_table.subprocess, "run", boom)
    monkeypatch.setattr(csv_table.sys, "platform", "darwin")
    # The HTML rendered; a report you have to double-click is not a failed report.
    assert csv_table.main([str(src), "-o", str(out)]) == 0
    assert out.exists()
    assert "could not open" in capsys.readouterr().err


def test_darwin_uses_the_open_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(csv_table.sys, "platform", "darwin")
    monkeypatch.setattr(
        csv_table.subprocess, "run", lambda argv, check: calls.append(argv)
    )
    target = tmp_path / "t.html"
    target.write_text("<html></html>")
    assert csv_table.open_in_browser(str(target)) is True
    assert calls == [["open", str(target)]]
```

Check the head of `test_csv_table.py` for how it imports the module and match it
(`from reporting import csv_table`).

- [ ] **Step 2: Run the tests to verify they fail**

```bash
.venv/bin/python -m pytest reporting_tests/test_csv_table.py -v
```
Expected: FAIL — `AttributeError: module 'reporting.csv_table' has no attribute 'open_in_browser'`.

- [ ] **Step 3: Implement the opener**

In `rewrite/reporting/csv_table.py`, extend the imports:

```python
import subprocess
import webbrowser
from pathlib import Path
```

Add `open_in_browser` immediately before `main`:

```python
def open_in_browser(path: str) -> bool:
    """Hand a rendered file to the OS. Returns whether the launch succeeded.

    Best-effort by design: a report that rendered but could not be opened is not a failed
    report, so a missing or broken launcher is a warning and the caller still succeeds.
    Otherwise a headless box or an unusual desktop would turn a working report into a
    non-zero exit.

    ``open``/``xdg-open`` are preferred over :mod:`webbrowser` on the two platforms that
    have them because they respect the user's default application for the file type;
    ``webbrowser`` is the fallback elsewhere.
    """
    target = os.path.abspath(path)
    if sys.platform == "darwin":
        argv = ["open", target]
    elif sys.platform.startswith("linux"):
        argv = ["xdg-open", target]
    else:
        return webbrowser.open(Path(target).as_uri())
    try:
        subprocess.run(argv, check=True)
        return True
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"warning: could not open {target}: {exc}", file=sys.stderr)
        return False
```

Then extend `main`'s parser and tail:

```python
    parser.add_argument("--subtitle", help="optional line under the heading")
    parser.add_argument(
        "--open", dest="open_after", action="store_true", default=True,
        help="open the rendered HTML with the OS opener (default)",
    )
    parser.add_argument(
        "--no-open", dest="open_after", action="store_false",
        help="write the HTML without opening it",
    )
    args = parser.parse_args(argv)

    html = render_csv_file(args.csv, title=args.title, subtitle=args.subtitle)
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"wrote {args.out}")
        # Nothing to open when the page went to stdout, so this is inside the branch.
        if args.open_after:
            open_in_browser(args.out)
    else:
        sys.stdout.write(html)
    return 0
```

Also add a bullet to the module docstring's "Two things worth knowing" list — make it three:

```
* **The CLI opens what it writes.** ``--open`` is the default because a report you have to
  go and find is a report you don't read; ``--no-open`` is there for scripts and CI.
  ``write_table`` and ``render_table`` never open anything, so library callers and tests
  are unaffected.
```

- [ ] **Step 4: Run the tests to verify they pass**

```bash
.venv/bin/python -m pytest reporting_tests/ -v
```
Expected: all PASS. If any pre-existing test in `test_csv_table.py` calls `main` with
`-o` and no `--no-open`, it will now try to launch a browser — add `--no-open` to those
calls.

- [ ] **Step 5: Commit**

```bash
git add rewrite/reporting/csv_table.py rewrite/reporting_tests/test_csv_table.py
git commit -m "csv_table: open the rendered HTML by default, --no-open to suppress"
```

---

### Task 7: Delete `run_report`, update the docs

**Files:**
- Delete: `rewrite/reporting/run_report.py`, `rewrite/reporting_tests/test_run_report.py`
- Modify: `rewrite/reporting/__init__.py:11-17`, `rewrite/reporting/README.md`,
  `rewrite/README.md`, `rewrite/CLAUDE.md`, `rewrite/.gitignore:14`

**Interfaces:**
- Consumes: everything from Tasks 1-6.
- Produces: nothing new.

- [ ] **Step 1: Confirm the ported coverage before deleting**

`reporting_tests/test_run_report.py` holds behaviours worth keeping. Open it and check each
of these now lives in `framework_tests/test_resultrows.py` (Task 3) or
`framework_tests/test_cli.py` (Task 5). Add any that don't before deleting the file:

- `suite_of` from metadata / id shape / variant suffix / hyphenated suite / hand-written id
- one row per grading, ungraded result still yields a row, error row
- token flattening and missing tokens
- surrounding-whitespace trim and internal-formatting preservation
- latency rounding
- testcase enrichment, and the column list matching `result_columns`
- CSV round-trip with the expected header

Two behaviours are **deliberately not ported** — note them in the commit message rather
than re-adding them:

- `run_subtitle` (the provenance line, including the "UNFINISHED" marker). The CSV has no
  subtitle; `run_started_at` and `provider` are columns, and an unfinished run is visible
  as `finished_at IS NULL` in the store.
- prefix resolution and bad-prefix exit codes for a positional run argument. `--run-id`
  covers this and Task 5 tests it.

- [ ] **Step 2: Delete the module and its tests**

```bash
git rm rewrite/reporting/run_report.py rewrite/reporting_tests/test_run_report.py
```

- [ ] **Step 3: Fix `reporting/__init__.py`**

Replace lines 11-17 of `rewrite/reporting/__init__.py`:

```python
* :mod:`reporting.csv_table` is the generic layer — arbitrary rows or a CSV file become a
  standalone HTML page with per-column filtering, column show/hide, and sorting.

Tools are run as modules from the ``rewrite/`` directory, e.g.
``python -m reporting.csv_table results.csv -o results.html``. There is deliberately no
console-script entry point — that would install porcelain alongside the plumbing.

Row *building* lives in the plumbing (:mod:`llmeval.resultrows`), not here: which rows a
report contains is a capability, while turning them into a page is a workflow.
"""
```

- [ ] **Step 4: Run the whole suite**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pylint llmeval
```
Expected: all PASS, pylint 10/10, no import errors from the deletion.

- [ ] **Step 5: Verify the workflow end to end by hand**

```bash
uv run llmeval generate-csv --csv generation_sources/simple_facts.csv \
    --suite simple_facts --out /tmp/tc
uv run llmeval run --testcases /tmp/tc --provider configs/echo.json --db /tmp/e.sqlite3
uv run llmeval grade --testcases /tmp/tc --provider configs/echo.json --db /tmp/e.sqlite3
uv run llmeval report --db /tmp/e.sqlite3 --testcases /tmp/tc \
    --provider configs/echo.json --run-last-n 1 --out /tmp/rows.csv
uv run python -m reporting.csv_table /tmp/rows.csv -o /tmp/rows.html --title "smoke test"
```
Expected: `/tmp/rows.csv` has one row per (result, assertion) with `latency_ms` populated,
and the browser opens `/tmp/rows.html`.

- [ ] **Step 6: Rewrite the `reporting/README.md` run-report section**

Delete the whole `## The run report: run_report` section (lines 53-83) and replace it with:

```markdown
## Rendering an llmeval report

`llmeval report` does the selecting and emits CSV; this package renders it. Two commands,
because the split is the point — every filter is decided once, in the plumbing, and the
renderer's only input is a file.

```bash
uv run llmeval report --run-last-n 3 --provider configs/fidaro_prod.json \
    --testcases testcases/ --out results.csv
python -m reporting.csv_table results.csv -o results.html --title "last 3 runs"
```

The CSV has one row per (result × assertion), plus one row per **errored** result with the
grading columns empty — see `llmeval/resultrows.py` for the column list and the ordering
guarantee. Rows are grouped by run in chronological order, and by attempt within a test, so
a retried test reads as its failures followed by the answer.

`--open` is the default, so the second command lands you in a browser. Pass `--no-open` in
scripts and CI.
```

Also update the "Adding a tool" section at the end — `run_report.run_rows` is gone as the
example. Replace its middle sentence with:

```markdown
Build rows, hand them to `csv_table.write_table`. Keep row building out of here entirely if
the rows come from the store: that's a plumbing capability and belongs in `llmeval/`
alongside `resultrows.py`. Tests live in `../reporting_tests/`, offline, no credentials.
```

- [ ] **Step 7: Update `rewrite/README.md`**

Four edits:

1. In the "Plumbing and porcelain" section, the porcelain paragraph and its code block
   (lines 48-55) become:

```markdown
The porcelain that exists so far lives in **[reporting/](reporting/README.md)**: a generic
CSV→HTML table viewer (filter any column, show/hide columns, sort, export) that opens what
it renders.

```bash
uv run llmeval report --run-last-n 3 --testcases testcases/ --out results.csv
python -m reporting.csv_table results.csv -o results.html
```
```

2. In the Quickstart, step 4 becomes two steps:

```markdown
# 4. Emit the result rows and view them
uv run llmeval report --testcases testcases/ --provider configs/echo.json --out results.csv
uv run python -m reporting.csv_table results.csv -o report.html
```

3. Add a new section after "Results" (i.e. after the `PRAGMA user_version` paragraph):

```markdown
## Selecting which runs to read

`grade` and `report` both read stored results, so both take the same four flags. They fall
into three groups and the groups **cannot be combined** — `--run-last-n 3 --run-after X` has
no single obvious reading, so it's an error rather than a guess.

| Flag | Meaning |
|---|---|
| `--run-id a,b,c` | exactly these runs; ids or unambiguous prefixes, comma-separated or repeated |
| `--run-after V` / `--run-before V` | an inclusive window; `V` is `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, either with an explicit `+HH:MM`/`Z` offset, **or a run id** |
| `--run-last-n N` | the N most recent runs |

Omit them all for every run. A bare timestamp is **UTC** — that's what `runs.started_at`
holds and what the run id embeds, so `--run-after 2026-07-29` lines up with the ids you see
in the log. `--run-after run_20260729-0451` means "that run and everything after it".

Run selection composes with provider selection, and the provider narrows first:
`--provider fidaro_prod.json --run-last-n 3` is the last three runs *of that provider*.

Naming a run that doesn't exist is an error (exit 2). A window that legitimately matches no
runs is not — you get an empty table, because "the last 3 runs of a provider that has had
none" is an answer.

### Grading

Grading is per **result**, not per test: `gradings` is unique on
`(result_id, assertion_key)`, so every attempt carries its own score and a re-run adds a
row to grade rather than superseding one. `grade` fills in every `(result, assertion)` pair
in the selected runs that doesn't have a grading yet, and **skips attempts that errored** —
there is no output to assert against, and the error row is itself the finding.

```bash
llmeval grade --testcases testcases/ --provider configs/fidaro_prod.json --run-last-n 1
```

## Reporting

`report` writes the selected result rows as **CSV**. It renders nothing: turning a table
into a page is a workflow, so that's [reporting/](reporting/README.md)'s job.

```bash
llmeval report --run-last-n 3 --provider configs/fidaro_prod.json \
               --testcases testcases/ --db runs.sqlite3 --out results.csv
python -m reporting.csv_table results.csv -o results.html   # opens in a browser
```

**One row per (result × assertion)**, plus **one row per errored result** with the grading
columns empty. Rows are grouped by run in chronological order, and within a test by attempt,
so a test that failed twice and answered on the third try reads top to bottom:

```
run1, test x, attempt 0, error=timeout, latency_ms=60001
run1, test x, attempt 1, assertion1, passed=True
run1, test x, attempt 1, assertion2, passed=False
run2, test x, attempt 0, assertion1, passed=True
```

`latency_ms` is filled in for error rows too, which is how "the timeout is too tight" is
distinguished from "the provider is down".

`--provider` is repeatable and optional (default: every provider in the database), so one
report can span several configs — `provider` and `cache_key_hash` are columns.
`--testcases` is optional and does two things: it adds the `prompt`, `request_type` and
`domain` columns, and it **selects** — only tests present in those files appear, which is
what makes `--filter suite=simple_facts` work.

The statistics report — bootstrap CIs, deltas against a baseline, pick-best win rates — is
now `compare-report`, unchanged otherwise:

```bash
llmeval compare-report --providers configs/fidaro_prod.json configs/venice.json \
                       --baseline fidaro-prod --order both --out report.html
```
```

4. Update "The three workflows" (lines 181-198) so the comparison examples call
   `compare-report` rather than `report`, and the batch example ends with the CSV pair:

```bash
# Batch-run one provider, then read what happened
llmeval run    --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval grade  --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval report --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3 \
               --run-last-n 1 --out results.csv
python -m reporting.csv_table results.csv -o results.html

# Indirect comparison (rate each config, compare ratings)
llmeval grade --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval grade --testcases testcases/ --provider configs/fidaro_dev.json  --db runs.sqlite3
llmeval compare-report --providers configs/fidaro_prod.json configs/fidaro_dev.json \
               --baseline fidaro-prod --metrics accuracy --db runs.sqlite3 --out report.html

# Direct comparison (judge picks the best; both orderings to fight position bias)
llmeval pickbest --testcases testcases/ --providers configs/fidaro_prod.json configs/venice.json \
                 --order both --db runs.sqlite3
llmeval compare-report --providers configs/fidaro_prod.json configs/venice.json \
                 --order both --db runs.sqlite3 --out report.html
```

Finally, in the "Layout" block, add the two new modules under `llmeval/`:

```
  runselect.py      which runs a result-reading stage looks at
  resultrows.py     stored results -> report rows + CSV
```

and change the `reporting/` line to drop the run report:

```
reporting/          porcelain: generic CSV->HTML viewer (not in the wheel)
```

- [ ] **Step 8: Update `rewrite/CLAUDE.md`**

In "The plumbing's public contracts", replace contract 1:

```markdown
1. **CLI subcommands** — `generate`, `generate-csv`, `run`, `grade`, `pickbest`, `report`,
   `compare-report` (see [llmeval/cli.py](llmeval/cli.py)). `report` emits **CSV** result
   rows — rendering them as a page is porcelain — while `compare-report` is the statistics
   and pick-best HTML. Note that the aggregation step the docs call "compare" has no
   subcommand of its own: it's [comparison/stats.py](llmeval/comparison/stats.py), reached
   via `compare-report` or as a library call. Exposing it directly would be a legitimate
   plumbing addition.
```

- [ ] **Step 9: Update the `.gitignore` comment**

`rewrite/.gitignore` line 14 mentions `reporting.run_report`. Change it to:

```
# Generated reports written to the project root (llmeval report / reporting.csv_table).
```

Check the patterns below that comment still cover `results.csv` and `*.html`; add
`results.csv` if not.

- [ ] **Step 10: Full verification**

```bash
.venv/bin/python -m pytest
.venv/bin/python -m pylint llmeval
grep -rn "run_report" rewrite --exclude-dir=.venv --exclude-dir=__pycache__ --exclude-dir=.pytest_cache
```
Expected: all tests PASS, pylint 10/10, and the grep returns **nothing**.

- [ ] **Step 11: Commit**

```bash
git add -A rewrite/
git commit -m "Delete run_report, superseded by llmeval report + csv_table

Row building moved down into llmeval/resultrows.py; rendering is csv_table's
only job. run_subtitle and the positional run-prefix argument are not ported —
run_started_at/provider are columns now, and --run-id covers prefix selection."
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| `--run-id` comma list | 2 (parse), 5 (flag) |
| `--run-after` date/datetime/run-id, epoch default | 1, 2 |
| `--run-before` end cap | 1, 2 |
| `--run-last-n` | 1, 2 |
| groups incompatible → error | 2, 5 |
| errored results selected too | 3 |
| grading per (result, assertion), failures ignored | 4 |
| grading narrowed to selected runs | 4, 5 |
| data generation split from display | 3 (CSV), 6 (HTML) |
| row per (result, assertion) + row per errored result | 3 |
| grouped by run chronologically, attempt order within | 1 (ordering), 3 |
| feed the CSV to the generic viewer | 6, 7 |
| `open` after building, `--no-open` disables | 6 |
| latency in the report | 3 |
| old pick-best report preserved | 5 (`compare-report`) |
| no schema change | Global Constraints |

**Type consistency checked:** `select_runs` returns `list[RunRow]` and `resolve_runs` maps
it to ids; `result_rows` takes `Sequence[RunRow]` (not ids) so it can read
`run.started_at`/`run.provider_name` without a second query; `result_columns(with_tests:
bool)` is called with a keyword in tests and positionally in `cli.py` — both valid; `grade`
and `grade_testcase` both take `run_ids` as the last keyword parameter.

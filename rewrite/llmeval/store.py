"""SQLite results store — the database the brief asks for.

Four tables, deliberately separated so stages stay decoupled:

* ``runs``     — one row per ``llmeval run`` invocation. Gives every result a
  provenance handle that doesn't depend on reading timestamps out of other columns.
  The run owns the **cache key**: every result it produced was produced under it.
* ``results``  — one row per *attempt*, successful or not. Reused so expensive calls
  aren't repeated; topped up to N for best-of-N statistics.
* ``gradings`` — assertion scores against a result. Separate from results so you can
  edit/add assertions and re-grade **without re-running the model**. Each row also keeps
  the ``assertion_value`` it was graded against — the rubric text, the expected substring
  — because ``testcases/`` is regenerated and a bare score with no record of the criterion
  is unreadable a week later. Same reasoning as ``results.messages_json`` below.
* ``verdicts`` — pick-best head-to-head outcomes, also keyed so they replay against
  cached outputs.

Everything analysis needs is a query away (group by ``cache_key_hash``, join gradings).

Two axes run through ``results``, and keeping them distinct is what makes the rest work:

* *identity* — what was under test. This is the cache key, and it lives on ``runs``.
  ``results`` reaches it through ``run_id``; storing a second copy per result only
  created the possibility of the two disagreeing. Readers are unaffected: every result
  query joins ``runs`` and ``ResultRow`` still carries ``cache_key_hash``.
* *provenance* — which sitting produced this row, i.e. ``run_id`` itself.

``attempt`` is scoped to ``(run_id, test_id)`` and restarts at 0 for each test in each
run: it answers "which try within this run?". Pooling attempts across runs — the
best-of-N dataset — is a *cache key* question, so it groups by identity instead (see
:meth:`Store.count_results`, which is what ``runner._to_run`` tops up against).

**Every attempt is stored, including the ones that failed.** A retry that eventually
succeeded leaves its error rows behind, so the cost of a flaky provider is visible in
the data rather than only in the log.

Each attempt also records the **messages it sent**. The prompt would otherwise live only
in ``testcases/``, which is regenerated — so a result could outlive any record of the
question that produced it, and a report could only recover the prompt by joining against
files that may since have changed. Storing what was actually sent makes a result
self-contained, and makes "what did we send when this timed out?" answerable from the
error row itself.

Each attempt may also record **provider-specific output**: response data with no place
in the OpenAI schema, kept verbatim under its vendor key (``{"fidaro": {...}}``). It is
stored rather than parsed because the suite has no opinion on what a backend puts there
— a test case that wants to assert on it reads the JSON, and a new key needs no schema
change.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator, Sequence

from llmeval.cache_key import CacheKey

# Bumped whenever the schema changes shape. There is no migration path: a mismatch
# is a hard error telling the user to delete the file (see ``Store._check_version``).
# 3: results.messages_json — the prompt as sent, recorded with each attempt.
# 4: results.provider_specific_output — non-standard response data, verbatim.
# 5: gradings.assertion_value — the criterion the score was awarded against.
SCHEMA_VERSION = 5


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IncompatibleSchema(RuntimeError):
    """Raised when a DB on disk was written by a different schema version.

    Typed so the CLI can report it as a plain message instead of a traceback — it's an
    expected user-facing condition, not a bug.
    """


def new_run_id() -> str:
    """A sortable, greppable run id: ``run_20260729-142530-a3f1``.

    The UTC timestamp makes ``ORDER BY id`` chronological and lets a human eyeball
    when a run happened; the random suffix keeps two runs started in the same second
    distinct. Short enough that a prefix identifies one in practice — see
    :meth:`Store.resolve_run`.
    """
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"run_{stamp}-{os.urandom(2).hex()}"


@dataclass
class RunRow:
    id: str
    provider_name: str | None
    cache_key_hash: str
    cache_key_json: str
    config: Any
    params: Any
    notes: str | None
    started_at: str
    finished_at: str | None  # None => still running, or the process died

    @property
    def finished(self) -> bool:
        return self.finished_at is not None


@dataclass
class ResultRow:
    id: int
    run_id: str
    test_id: str
    # Not columns on ``results``: both are read off the row's run (see _RESULT_SELECT).
    cache_key_hash: str
    cache_key_json: str
    attempt: int  # 0-based within (run_id, test_id)
    output: str | None
    raw: Any
    reasoning: str | None
    tokens: Any
    latency_ms: float | None
    error: str | None
    created_at: str
    config: Any = None  # full provider config that produced this result
    messages: Any = None  # the messages sent to the provider, exactly as sent
    provider_specific: Any = None  # non-standard response data, e.g. {"fidaro": {...}}


@dataclass
class GradingRow:
    id: int
    result_id: int
    assertion_key: str
    assertion_value: str | None
    type: str | None
    metric: str | None
    score: float | None
    passed: bool | None
    weight: float
    reason: str | None
    judge_model: str | None
    created_at: str


@dataclass
class GradedResultRow:
    result_id: int
    run_id: str
    test_id: str
    cache_key_hash: str
    attempt: int
    output: str | None
    assertion_key: str
    type: str | None
    metric: str | None
    score: float | None
    passed: bool | None
    weight: float


@dataclass
class VerdictRow:
    id: int
    test_id: str
    comparison_key: str
    winner_hash: str | None
    candidates: list[str]
    reason: str | None
    created_at: str


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY,
    provider_name TEXT,
    cache_key_hash TEXT NOT NULL,
    cache_key_json TEXT NOT NULL,
    config_json TEXT,
    params_json TEXT,
    notes TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_key ON runs(cache_key_hash);

-- No cache_key columns: the run owns the cache key and results reach it via run_id.
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL REFERENCES runs(id),
    test_id TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    output TEXT,
    raw_json TEXT,
    reasoning TEXT,
    tokens_json TEXT,
    latency_ms REAL,
    error TEXT,
    config_json TEXT,
    -- The prompt as sent. raw_json is the provider's *response*, so without this the
    -- question is nowhere in the database and a result cannot be read on its own.
    messages_json TEXT,
    -- Response data with no place in the OpenAI schema, stored under its vendor key:
    -- {"fidaro": {"title": "..."}}. Enveloped rather than flattened so a second vendor
    -- costs no schema change.
    provider_specific_output TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(run_id, test_id, attempt)
);
CREATE INDEX IF NOT EXISTS idx_results_run ON results(run_id);
-- Identity lookups start from the test id and join runs for the cache key; the UNIQUE
-- index above is run-first, so it can't serve them.
CREATE INDEX IF NOT EXISTS idx_results_test ON results(test_id);

CREATE TABLE IF NOT EXISTS gradings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    assertion_key TEXT NOT NULL,
    -- The criterion as graded, as text: the rubric a judge was given, the substring a
    -- deterministic check looked for. assertion_key is a hash of it, which identifies an
    -- assertion but cannot be read; this is the half a human needs.
    assertion_value TEXT,
    type TEXT,
    metric TEXT,
    score REAL,
    passed INTEGER,
    weight REAL NOT NULL DEFAULT 1.0,
    reason TEXT,
    judge_model TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(result_id, assertion_key)
);

CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id TEXT NOT NULL,
    comparison_key TEXT NOT NULL,
    winner_hash TEXT,
    candidates_json TEXT NOT NULL,
    reason TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(test_id, comparison_key)
);
"""


# Every result read joins its run to recover the cache key. Normalised in storage,
# unchanged for readers: ``ResultRow`` still carries ``cache_key_hash``/``cache_key_json``,
# and the aliases below are what fill them.
_RESULT_SELECT = """
    SELECT r.*, ru.cache_key_hash AS cache_key_hash, ru.cache_key_json AS cache_key_json
    FROM results r JOIN runs ru ON ru.id = r.run_id
"""


class Store:
    """Thread-safe SQLite results store.

    The runner drives one shared Store from a thread pool (``llmeval run --concurrency``).
    ``sqlite3`` forbids cross-thread use of a connection by default, so we open with
    ``check_same_thread=False`` and serialize every DB touch behind a re-entrant lock.
    Writes are tiny relative to model-call latency, so the lock is not a bottleneck.
    """

    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Off by default in sqlite3; without it ``results.run_id REFERENCES runs(id)``
        # is decorative and orphan results become possible again.
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._check_version()
        self._conn.executescript(_SCHEMA)
        self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    def _check_version(self) -> None:
        """Refuse to open a database written by an incompatible schema.

        There is no migration path by design. A populated DB at the wrong version is a
        hard error naming the file — we don't silently drop the user's tables, even
        when the data is cheap to regenerate.
        """
        version = int(self._conn.execute("PRAGMA user_version").fetchone()[0])
        if version == SCHEMA_VERSION:
            return
        # A brand-new (or pre-existing but empty) file reports version 0 and has no
        # tables yet; that's not a mismatch, it's a fresh database.
        existing = self._conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchone()[0]
        if version == 0 and existing == 0:
            return
        raise IncompatibleSchema(
            f"{self.path} was written by schema version {version}, but this build "
            f"expects version {SCHEMA_VERSION}. There is no migration path — delete "
            f"the file and re-run."
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- runs --------------------------------------------------------------

    def create_run(
        self,
        cache_key: CacheKey,
        provider_name: str | None = None,
        config: Any = None,
        params: Any = None,
        notes: str | None = None,
    ) -> str:
        """Open a run and return its id. ``finished_at`` stays NULL until finished."""
        run_id = new_run_id()
        with self._lock:
            self._conn.execute(
                """INSERT INTO runs
                   (id, provider_name, cache_key_hash, cache_key_json, config_json,
                    params_json, notes, started_at, finished_at)
                   VALUES (?,?,?,?,?,?,?,?,NULL)""",
                (
                    run_id,
                    provider_name,
                    cache_key.hash,
                    cache_key.canonical,
                    json.dumps(config) if config is not None else None,
                    json.dumps(params) if params is not None else None,
                    notes,
                    _now(),
                ),
            )
            self._conn.commit()
        return run_id

    def finish_run(self, run_id: str) -> None:
        """Stamp ``finished_at``. A run that crashed simply never gets this call."""
        with self._lock:
            self._conn.execute("UPDATE runs SET finished_at=? WHERE id=?", (_now(), run_id))
            self._conn.commit()

    def get_run(self, run_id: str) -> RunRow | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return self._run_row(row) if row else None

    def list_runs(self, cache_key_hash: str | None = None, limit: int | None = None):
        """Runs newest first, optionally narrowed to one cache key."""
        sql = "SELECT * FROM runs"
        args: list[Any] = []
        if cache_key_hash is not None:
            sql += " WHERE cache_key_hash=?"
            args.append(cache_key_hash)
        sql += " ORDER BY started_at DESC, id DESC"
        if limit is not None:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._run_row(r) for r in rows]

    def resolve_run(self, prefix: str) -> str:
        """Expand a run-id prefix to the full id.

        Run ids are 24 characters; nobody types one in full. Raises rather than
        guessing when the prefix matches nothing or more than one run.

        Note that a prefix short enough to be worth typing cannot separate two runs
        opened in the same second: they differ only in the trailing random suffix.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM runs WHERE id LIKE ? ORDER BY id", (prefix + "%",)
            ).fetchall()
        if not rows:
            raise KeyError(f"no run matching prefix {prefix!r}")
        if len(rows) > 1:
            matches = ", ".join(r["id"] for r in rows[:5])
            raise KeyError(f"prefix {prefix!r} matches {len(rows)} runs: {matches}...")
        return rows[0]["id"]

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
        second-precision bound would exclude the very run the user named as the end cap.

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

    @staticmethod
    def _run_row(r: sqlite3.Row) -> RunRow:
        return RunRow(
            id=r["id"],
            provider_name=r["provider_name"],
            cache_key_hash=r["cache_key_hash"],
            cache_key_json=r["cache_key_json"],
            config=json.loads(r["config_json"]) if r["config_json"] else None,
            params=json.loads(r["params_json"]) if r["params_json"] else None,
            notes=r["notes"],
            started_at=r["started_at"],
            finished_at=r["finished_at"],
        )

    # --- results -----------------------------------------------------------

    def _next_attempt(self, run_id: str, test_id: str) -> int:
        """The next attempt index for this test *within this run* (0-based)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM results WHERE run_id=? AND test_id=?",
                (run_id, test_id),
            ).fetchone()
            return int(row[0])

    def add_result_row(
        self,
        test_id: str,
        *,
        run_id: str,
        output: str | None = None,
        raw: Any = None,
        reasoning: str | None = None,
        tokens: Any = None,
        latency_ms: float | None = None,
        error: str | None = None,
        config: Any = None,
        messages: Any = None,
        provider_specific: Any = None,
    ) -> int:
        """Insert one attempt; return its row id (for attaching gradings).

        ``run_id`` is keyword-only and required: every result must be attributable to
        a run, so there is no path that produces an orphan row. It also supplies the
        cache key, which is why none is passed here — a result cannot claim identity
        its run didn't have.

        Pass ``error`` for an attempt that failed. Failed attempts are stored, not
        dropped, so a retried test case shows what the retries cost. ``messages`` is
        recorded either way — the prompt that produced a timeout is worth as much as the
        prompt that produced an answer. So is ``provider_specific``: a stream that timed
        out may well have delivered its side-channel data before it stalled.
        """
        # Hold the lock across read-attempt + insert + commit so the attempt index
        # stays consistent when multiple threads write concurrently.
        with self._lock:
            attempt = self._next_attempt(run_id, test_id)
            cur = self._conn.execute(
                """INSERT INTO results
                   (run_id, test_id, attempt, output, raw_json, reasoning, tokens_json,
                    latency_ms, error, config_json, messages_json,
                    provider_specific_output, created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    run_id,
                    test_id,
                    attempt,
                    output,
                    json.dumps(raw) if raw is not None else None,
                    reasoning,
                    json.dumps(tokens) if tokens is not None else None,
                    latency_ms,
                    error,
                    json.dumps(config) if config is not None else None,
                    json.dumps(messages) if messages is not None else None,
                    json.dumps(provider_specific) if provider_specific is not None else None,
                    _now(),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def add_result(self, test_id: str, **kwargs) -> int:
        """Insert one attempt; return its 0-based attempt index within its run."""
        with self._lock:
            attempt = self._next_attempt(kwargs["run_id"], test_id)
            self.add_result_row(test_id, **kwargs)
            return attempt

    def count_results(self, test_id: str, key_hash: str, success_only: bool = False) -> int:
        """Attempts for (test, cache key), pooled across **every** run.

        This is what the runner's top-up arithmetic counts against, so spanning runs is
        the point: three successes accumulated over three invocations are three usable
        results. ``success_only`` excludes error rows, which is the count that decides
        whether a model call is still needed.
        """
        sql = """SELECT COUNT(*) FROM results r JOIN runs ru ON ru.id = r.run_id
                 WHERE r.test_id=? AND ru.cache_key_hash=?"""
        if success_only:
            sql += " AND r.error IS NULL"
        with self._lock:
            return int(self._conn.execute(sql, (test_id, key_hash)).fetchone()[0])

    def get_results(self, test_id: str, key_hash: str) -> list[ResultRow]:
        """Every attempt for (test, cache key) across all runs, oldest first.

        Ordered by row id, not ``attempt``: attempt restarts within each run, so it
        cannot order a pool drawn from several. Row id is insertion order, which for
        this table is chronological.
        """
        with self._lock:
            rows = self._conn.execute(
                _RESULT_SELECT + " WHERE r.test_id=? AND ru.cache_key_hash=? ORDER BY r.id",
                (test_id, key_hash),
            ).fetchall()
        return [self._result_row(r) for r in rows]

    def get_results_for_run(self, run_id: str) -> list[ResultRow]:
        """Every attempt one run produced, by test then attempt order."""
        with self._lock:
            rows = self._conn.execute(
                _RESULT_SELECT + " WHERE r.run_id=? ORDER BY r.test_id, r.attempt", (run_id,)
            ).fetchall()
        return [self._result_row(r) for r in rows]

    @staticmethod
    def _result_row(r: sqlite3.Row) -> ResultRow:
        return ResultRow(
            id=r["id"],
            run_id=r["run_id"],
            test_id=r["test_id"],
            cache_key_hash=r["cache_key_hash"],
            cache_key_json=r["cache_key_json"],
            attempt=r["attempt"],
            output=r["output"],
            raw=json.loads(r["raw_json"]) if r["raw_json"] else None,
            reasoning=r["reasoning"],
            tokens=json.loads(r["tokens_json"]) if r["tokens_json"] else None,
            latency_ms=r["latency_ms"],
            error=r["error"],
            created_at=r["created_at"],
            config=json.loads(r["config_json"]) if r["config_json"] else None,
            messages=json.loads(r["messages_json"]) if r["messages_json"] else None,
            provider_specific=(
                json.loads(r["provider_specific_output"])
                if r["provider_specific_output"]
                else None
            ),
        )

    # --- gradings ----------------------------------------------------------

    def set_grading(
        self,
        result_id: int,
        assertion_key: str,
        type: str | None = None,
        metric: str | None = None,
        score: float | None = None,
        passed: bool | None = None,
        weight: float = 1.0,
        reason: str | None = None,
        judge_model: str | None = None,
        assertion_value: str | None = None,
    ) -> None:
        """Upsert a grading for (result, assertion). Re-grading overwrites.

        ``assertion_value`` is the criterion as text — already rendered by the caller,
        because what counts as readable is a property of the assertion type, not of the
        store (see :func:`llmeval.grade.assertion_value_text`).
        """
        with self._lock:
            self._conn.execute(
                """INSERT INTO gradings
               (result_id, assertion_key, assertion_value, type, metric, score, passed,
                weight, reason, judge_model, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(result_id, assertion_key) DO UPDATE SET
                 assertion_value=excluded.assertion_value, type=excluded.type,
                 metric=excluded.metric, score=excluded.score,
                 passed=excluded.passed, weight=excluded.weight, reason=excluded.reason,
                 judge_model=excluded.judge_model, created_at=excluded.created_at""",
                (
                    result_id,
                    assertion_key,
                    assertion_value,
                    type,
                    metric,
                    score,
                    None if passed is None else int(passed),
                    weight,
                    reason,
                    judge_model,
                    _now(),
                ),
            )
            self._conn.commit()

    def get_gradings(self, result_id: int) -> list[GradingRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM gradings WHERE result_id=? ORDER BY assertion_key", (result_id,)
            ).fetchall()
        return [
            GradingRow(
                id=r["id"],
                result_id=r["result_id"],
                assertion_key=r["assertion_key"],
                assertion_value=r["assertion_value"],
                type=r["type"],
                metric=r["metric"],
                score=r["score"],
                passed=None if r["passed"] is None else bool(r["passed"]),
                weight=r["weight"],
                reason=r["reason"],
                judge_model=r["judge_model"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

    def iter_graded_results(
        self, key_hash: str, run_id: str | None = None
    ) -> Iterator[GradedResultRow]:
        """Join results+gradings for one cache key — the input to indirect comparison.

        Pass ``run_id`` to narrow to a single run; omit it to pool every attempt ever
        recorded for that cache key (the best-of-N view).
        """
        sql = """SELECT r.id AS result_id, r.run_id, r.test_id, ru.cache_key_hash, r.attempt,
                        r.output, g.assertion_key, g.type, g.metric, g.score, g.passed, g.weight
                 FROM results r
                 JOIN runs ru ON ru.id = r.run_id
                 JOIN gradings g ON g.result_id = r.id
                 WHERE ru.cache_key_hash=?"""
        args: list[Any] = [key_hash]
        if run_id is not None:
            sql += " AND r.run_id=?"
            args.append(run_id)
        # r.id, not r.attempt: attempt restarts per run and cannot order a cross-run pool.
        sql += " ORDER BY r.test_id, r.id, g.assertion_key"
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        for r in rows:
            yield GradedResultRow(
                result_id=r["result_id"],
                run_id=r["run_id"],
                test_id=r["test_id"],
                cache_key_hash=r["cache_key_hash"],
                attempt=r["attempt"],
                output=r["output"],
                assertion_key=r["assertion_key"],
                type=r["type"],
                metric=r["metric"],
                score=r["score"],
                passed=None if r["passed"] is None else bool(r["passed"]),
                weight=r["weight"],
            )

    # --- verdicts ----------------------------------------------------------

    def set_verdict(
        self,
        test_id: str,
        comparison_key: str,
        winner_hash: str | None,
        candidates: list[str],
        reason: str | None = None,
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO verdicts
                   (test_id, comparison_key, winner_hash, candidates_json, reason, created_at)
                   VALUES (?,?,?,?,?,?)
                   ON CONFLICT(test_id, comparison_key) DO UPDATE SET
                     winner_hash=excluded.winner_hash, candidates_json=excluded.candidates_json,
                     reason=excluded.reason, created_at=excluded.created_at""",
                (test_id, comparison_key, winner_hash, json.dumps(candidates), reason, _now()),
            )
            self._conn.commit()

    def get_verdicts(self, comparison_key: str) -> list[VerdictRow]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM verdicts WHERE comparison_key=? ORDER BY test_id", (comparison_key,)
            ).fetchall()
        return [
            VerdictRow(
                id=r["id"],
                test_id=r["test_id"],
                comparison_key=r["comparison_key"],
                winner_hash=r["winner_hash"],
                candidates=json.loads(r["candidates_json"]),
                reason=r["reason"],
                created_at=r["created_at"],
            )
            for r in rows
        ]

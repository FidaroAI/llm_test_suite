"""SQLite results store — the database the brief asks for.

Three tables, deliberately separated so stages stay decoupled:

* ``results``  — cached LLM outputs, one row per (test, cache_key, attempt). Reused so
  expensive calls aren't repeated; topped up to N for best-of-N statistics.
* ``gradings`` — assertion scores against a result. Separate from results so you can
  edit/add assertions and re-grade **without re-running the model**.
* ``verdicts`` — pick-best head-to-head outcomes, also keyed so they replay against
  cached outputs.

Everything analysis needs is a query away (group by ``cache_key_hash``, join gradings).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator

from llmeval.cache_key import CacheKey


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ResultRow:
    id: int
    test_id: str
    cache_key_hash: str
    cache_key_json: str
    attempt: int
    output: str | None
    raw: Any
    reasoning: str | None
    tokens: Any
    latency_ms: float | None
    error: str | None
    created_at: str
    config: Any = None  # full provider config that produced this result


@dataclass
class GradingRow:
    id: int
    result_id: int
    assertion_key: str
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
CREATE TABLE IF NOT EXISTS results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    test_id TEXT NOT NULL,
    cache_key_hash TEXT NOT NULL,
    cache_key_json TEXT NOT NULL,
    attempt INTEGER NOT NULL,
    output TEXT,
    raw_json TEXT,
    reasoning TEXT,
    tokens_json TEXT,
    latency_ms REAL,
    error TEXT,
    config_json TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(test_id, cache_key_hash, attempt)
);
CREATE INDEX IF NOT EXISTS idx_results_key ON results(cache_key_hash);

CREATE TABLE IF NOT EXISTS gradings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    result_id INTEGER NOT NULL,
    assertion_key TEXT NOT NULL,
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


class Store:
    def __init__(self, path: str = ":memory:"):
        self.path = path
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Add columns introduced after a DB was first created (idempotent)."""
        cols = {row[1] for row in self._conn.execute("PRAGMA table_info(results)")}
        if "config_json" not in cols:
            self._conn.execute("ALTER TABLE results ADD COLUMN config_json TEXT")

    def close(self) -> None:
        self._conn.close()

    # --- results -----------------------------------------------------------

    def _next_attempt(self, test_id: str, key_hash: str) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM results WHERE test_id=? AND cache_key_hash=?",
            (test_id, key_hash),
        ).fetchone()
        return int(row[0])

    def add_result_row(
        self,
        test_id: str,
        cache_key: CacheKey,
        output: str | None = None,
        raw: Any = None,
        reasoning: str | None = None,
        tokens: Any = None,
        latency_ms: float | None = None,
        error: str | None = None,
        config: Any = None,
    ) -> int:
        """Insert a result; return its row id (for attaching gradings)."""
        attempt = self._next_attempt(test_id, cache_key.hash)
        cur = self._conn.execute(
            """INSERT INTO results
               (test_id, cache_key_hash, cache_key_json, attempt, output, raw_json,
                reasoning, tokens_json, latency_ms, error, config_json, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                test_id,
                cache_key.hash,
                cache_key.canonical,
                attempt,
                output,
                json.dumps(raw) if raw is not None else None,
                reasoning,
                json.dumps(tokens) if tokens is not None else None,
                latency_ms,
                error,
                json.dumps(config) if config is not None else None,
                _now(),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def add_result(self, *args, **kwargs) -> int:
        """Insert a result; return its 0-based attempt index for (test, key)."""
        test_id = args[0] if args else kwargs["test_id"]
        cache_key = args[1] if len(args) > 1 else kwargs["cache_key"]
        attempt = self._next_attempt(test_id, cache_key.hash)
        self.add_result_row(*args, **kwargs)
        return attempt

    def count_results(self, test_id: str, key_hash: str, success_only: bool = False) -> int:
        sql = "SELECT COUNT(*) FROM results WHERE test_id=? AND cache_key_hash=?"
        if success_only:
            sql += " AND error IS NULL"
        return int(self._conn.execute(sql, (test_id, key_hash)).fetchone()[0])

    def get_results(self, test_id: str, key_hash: str) -> list[ResultRow]:
        rows = self._conn.execute(
            "SELECT * FROM results WHERE test_id=? AND cache_key_hash=? ORDER BY attempt",
            (test_id, key_hash),
        ).fetchall()
        return [self._result_row(r) for r in rows]

    @staticmethod
    def _result_row(r: sqlite3.Row) -> ResultRow:
        return ResultRow(
            id=r["id"],
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
    ) -> None:
        """Upsert a grading for (result, assertion). Re-grading overwrites."""
        self._conn.execute(
            """INSERT INTO gradings
               (result_id, assertion_key, type, metric, score, passed, weight,
                reason, judge_model, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(result_id, assertion_key) DO UPDATE SET
                 type=excluded.type, metric=excluded.metric, score=excluded.score,
                 passed=excluded.passed, weight=excluded.weight, reason=excluded.reason,
                 judge_model=excluded.judge_model, created_at=excluded.created_at""",
            (
                result_id,
                assertion_key,
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
        rows = self._conn.execute(
            "SELECT * FROM gradings WHERE result_id=? ORDER BY assertion_key", (result_id,)
        ).fetchall()
        return [
            GradingRow(
                id=r["id"],
                result_id=r["result_id"],
                assertion_key=r["assertion_key"],
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

    def iter_graded_results(self, key_hash: str) -> Iterator[GradedResultRow]:
        """Join results+gradings for one cache key — the input to indirect comparison."""
        rows = self._conn.execute(
            """SELECT r.id AS result_id, r.test_id, r.cache_key_hash, r.attempt, r.output,
                      g.assertion_key, g.type, g.metric, g.score, g.passed, g.weight
               FROM results r JOIN gradings g ON g.result_id = r.id
               WHERE r.cache_key_hash=?
               ORDER BY r.test_id, r.attempt, g.assertion_key""",
            (key_hash,),
        ).fetchall()
        for r in rows:
            yield GradedResultRow(
                result_id=r["result_id"],
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

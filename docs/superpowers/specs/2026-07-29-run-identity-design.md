# Run identity in the llmeval store

**Date:** 2026-07-29
**Status:** Approved, implementing

## Problem

The `llmeval` SQLite store has no concept of a *test run*. Results are identified
solely by `(test_id, cache_key_hash, attempt)`, and the only provenance on a row is
its `created_at` timestamp and a copy of the provider config.

That means you cannot:

* extract every result produced by one `llmeval run` invocation,
* grade or report against a single run,
* compare two runs of the *same* provider config.

Timestamp-range queries against `created_at` are the only workaround, and they break
as soon as two runs overlap or an earlier run is topped up later.

## Solution

Introduce an explicit `runs` table and stamp every result with the run that produced
it. This is a **destructive** schema change: there is no migration path and existing
databases must be deleted.

### Schema

```sql
CREATE TABLE runs (
    id             TEXT PRIMARY KEY,   -- run_20260729-142530-a3f1
    provider_name  TEXT,               -- ProviderConfig.name, a human label
    cache_key_hash TEXT NOT NULL,      -- exactly one per run, by construction
    cache_key_json TEXT NOT NULL,
    config_json    TEXT,               -- the full ProviderConfig
    params_json    TEXT,               -- run invocation params (see below)
    notes          TEXT,
    started_at     TEXT NOT NULL,
    finished_at    TEXT                -- NULL => still running, or crashed
);
CREATE INDEX idx_runs_key ON runs(cache_key_hash);
```

`results` gains `run_id TEXT NOT NULL REFERENCES runs(id)` and an index
`idx_results_run ON results(run_id)`.

`gradings` and `verdicts` are unchanged. A grading reaches its run by joining through
`result_id`; a verdict spans multiple cache keys by nature and so belongs to no single
run.

### Three deliberate choices

**`params_json` is one blob, not eight columns.** It holds mode, target_n, retries,
concurrency, the testcases path, metadata filters, limit, randomize and seed. Nobody
queries on `retries`; you read it when a run looks anomalous.

**`finished_at IS NULL` replaces a `status` column.** One fewer field that can
disagree with reality.

**`UNIQUE(test_id, cache_key_hash, attempt)` is unchanged, and `attempt` keeps
counting across runs.** The second run of a test yields `attempt=1`, not `attempt=0`.

That last point is the load-bearing one. `cache_key_hash` answers *what was under
test*; `run_id` answers *which sitting produced this*. They are orthogonal axes.
Making attempt numbering per-run would conflate them and break `runner._to_run`,
where `--mode target_n --target-n 5` counts all prior successes and makes up only the
shortfall — five attempts accumulated over five separate invocations must remain the
same dataset as five from one invocation.

### Run ids

Format `run_YYYYMMDD-HHMMSS-xxxx`: UTC timestamp plus four hex characters from
`os.urandom`. Chronologically sortable under a plain `ORDER BY`, greppable, and short
enough that a prefix identifies one uniquely in practice. Generated in `store.py`.

### Store API

```python
create_run(cache_key, provider_name=None, config=None, params=None, notes=None) -> str
finish_run(run_id) -> None
get_run(run_id) -> RunRow | None
list_runs(cache_key_hash=None, limit=None) -> list[RunRow]   # newest first
resolve_run(prefix) -> str          # raises on zero or multiple matches
get_results_for_run(run_id) -> list[ResultRow]
iter_graded_results(key_hash, run_id=None)                   # existing + optional filter
```

`resolve_run` exists because a full id is 26 characters and nobody will type one.

### Write path

`add_result_row` and `add_result` take `run_id` as a **required keyword argument**.
The alternative — a nullable column — was rejected: it would leave orphan results
possible, which is the exact problem this change exists to eliminate.

`runner.run()` creates the run before fanning out, threads the id through
`run_testcase` into the thread pool, and calls `finish_run` in a `finally` block so an
interrupted run leaves `finished_at` NULL rather than claiming to have completed.
`cli.cmd_run` prints the run id on completion.

### Opening an incompatible database

`PRAGMA user_version` is bumped to 1. On mismatch `Store.__init__` raises with a
message naming the file and telling the user to delete it. It does **not** silently
drop tables — the data is worthless today, but silent data loss is a bad reflex to
build into a tool. The existing `_migrate()` method and its test are removed; with no
migration path there is nothing for it to do.

## Testing

Existing tests: `run_id` becoming required breaks roughly eight test modules. A `run`
fixture alongside the existing `store` fixture absorbs most of the churn; call sites
become `store.add_result("t1", k, run_id=run, output="a")`.
`test_migrates_old_results_table_missing_config_column` is deleted.

New coverage:

* run ids are unique and sort chronologically
* `run_id` NOT NULL is enforced
* attempt numbering continues across two runs of the same test
* `list_runs` filters by cache key and orders newest first
* `resolve_run` resolves a unique prefix and raises on zero or ambiguous matches
* `finish_run` sets `finished_at`; an interrupted run leaves it NULL
* opening a database with the wrong `user_version` raises

## Explicitly out of scope

* CLI `--run` filters on `grade`/`report`, and an `llmeval runs` subcommand. The
  Store read API lands first; CLI UX follows once the shape has been used in anger.
* Dropping the now-redundant `results.config_json` and `results.cache_key_json`.
  Every row in a run shares the run's config by construction, so these are duplicated
  data — but removing them is a separate change and keeping them costs only disk.

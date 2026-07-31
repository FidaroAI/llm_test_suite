# Result-rows selection and reporting — design

Date: 2026-07-29
Status: approved, ready to implement

## Problem

`llmeval report` renders a "pick the best" comparison: bootstrap confidence intervals per
config, pick-best win rates, and a per-test matrix of scores. That is the right report for
*"which of these two providers is better?"*, and it stays. It is the wrong report for the
much more common question *"what actually happened when I ran these tests?"* — a test with
one or two assertions produces a score buried in a matrix cell, and an errored result
produces nothing at all.

Second problem: result selection is coarse. `run` and `grade` select tests with
`--testcases`, `--provider` and `--filter`, but there is no way to say *which runs* to
look at. Grading and reporting therefore always span the whole history of a cache key.

## Goals

1. Add run selection — `--run-id`, `--run-after`, `--run-before`, `--run-last-n` — to the
   stages that read results (`grade`, `report`).
2. Grade every `(result, assertion)` pair in the selected set that doesn't already have a
   grading, ignoring results that errored.
3. Make `report` emit **data** (CSV): one row per `(result, assertion)` for results that
   succeeded, one row for results that errored, grouped by run in chronological order.
4. Render that CSV with the existing generic viewer, and open the HTML in a browser.

## Non-goals

* Changing the SQLite schema. Everything here is expressible against `runs` / `results` /
  `gradings` as they stand. This matters: `store.py` checks `PRAGMA user_version` and
  refuses a database from an older build with no migration path, so a bump costs every
  user their accumulated results.
* Changing the comparison report's statistics, template or flags. It moves to a new
  subcommand name and is otherwise untouched.
* Any new provider, assertion or caching capability.

## Architecture

The governing constraint is `rewrite/CLAUDE.md`: `llmeval` is **plumbing** (capabilities,
explicit flags, no workflow shortcuts), `reporting/` is **porcelain** (workflows,
friendliness), and the dependency runs one way — `reporting` imports `llmeval`, never the
reverse. "Build the HTML and shell out to `open`" is porcelain by definition.

So the work splits at the CSV:

```
llmeval report  ──►  results.csv  ──►  python -m reporting.csv_table  ──►  html + open
   (plumbing: selection + rows)          (porcelain: render only)
```

The reporter takes no selection flags at all. Every filtering decision is made once, in
the plumbing, and the porcelain's only input is a CSV file.

A consequence: `reporting/run_report.py` — which builds single-run rows and renders them —
is fully superseded, so it and its tests are deleted. Its row-shaping logic (suite
fallback, whitespace trim, token flattening) is data shaping rather than display, and
moves down into `llmeval/`.

## Components

### `llmeval/runselect.py` (new)

Owns the *meaning* of run selection, with no SQL.

```python
@dataclass(frozen=True)
class RunSelection:
    ids: tuple[str, ...] = ()      # explicit run ids or unambiguous prefixes
    after: str | None = None       # ISO datetime, or a run id/prefix
    before: str | None = None
    last_n: int | None = None
```

* `parse_run_selection(run_id, run_after, run_before, run_last_n) -> RunSelection`.
* Three mutually exclusive groups: `{--run-id}`, `{--run-after, --run-before}`,
  `{--run-last-n}`. Two groups at once raises `RunSelectionError`, which the CLI reports as
  a plain message with exit 2 — the same treatment `IncompatibleSchema` gets, because it is
  user error and not a bug.
* `--run-id a,b,c` is comma-separated and each element goes through `store.resolve_run`, so
  prefixes work. Repeating the flag also accumulates. An element matching nothing, or more
  than one run, is an error.
* `--run-after` / `--run-before` accept, in order: `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`,
  `YYYY-MM-DDTHH:MM:SS`, any of those with an explicit `+HH:MM` / `-HH:MM` / `Z` offset.
  **A naive value is UTC**, because that is what `runs.started_at` holds and what the run
  id embeds. If no form parses, the value is treated as a run id/prefix and the boundary
  becomes that run's `started_at`.
* Both bounds are **inclusive**. `--run-after run1` therefore means "run1 and everything
  after it", as specified, and `--run-before` is its symmetric counterpart.
* `resolve_runs(store, selection, cache_key_hashes=None) -> list[RunRow]` returns runs
  **oldest first** — the grouping order the report needs.
* Run selection composes with provider selection by narrowing to the providers' cache keys
  *first*: `--provider X --run-last-n 3` is the last three runs *of X*, not the last three
  runs overall filtered down to X.

### `llmeval/store.py` (one addition)

```python
def select_runs(self, *, ids=None, after=None, before=None,
                last_n=None, cache_key_hashes=None) -> list[RunRow]
```

Returns oldest-first. `after` / `before` are already-resolved ISO strings — the store keeps
owning SQL, `runselect` keeps owning semantics. `list_runs` (newest-first, used elsewhere)
is left alone.

### `llmeval/resultrows.py` (new)

The CSV data builder, absorbing `run_report.run_rows` and generalising it to N runs.

**Ordering.** Runs by `started_at` ascending, then `test_id`, then `attempt` ascending.
This yields the specified shape:

```
run1, test x, attempt 0, error
run1, test x, attempt 1, assertion1, pass, ...
run1, test x, attempt 1, assertion2, fail, ...
run2, test x, attempt 0, assertion1, pass, ...
```

Test order within a run is by id, not by first-attempt time: deterministic, and the viewer
re-sorts on any column anyway.

**Rows.** Per result:

| Result state | Rows emitted |
|---|---|
| succeeded, has gradings | one per grading |
| succeeded, no gradings | one, grading columns empty (a run that hasn't been graded) |
| errored | one, grading columns empty |

**Columns**, in reading order — where it came from, what was asked, what came back, how it
scored, provenance:

```
run_id, run_started_at, provider, test_id, attempt, suite,
[prompt, request_type, domain]           # only when --testcases is given
output, reasoning, error, latency_ms,
prompt_tokens, completion_tokens, total_tokens,
assertion_key, assertion_type, metric, score, passed, weight,
judge_model, grading_reason,
cache_key_hash, result_id
```

`latency_ms` is present for errored attempts too — that is how "the timeout is too tight"
is distinguished from "the provider is down".

Reads the store through `get_results_for_run` and `get_gradings` rather than fresh SQL. The
per-result `get_gradings` is an N+1 pattern, irrelevant at report scale, and it keeps this
module off the store's column names.

**`--testcases` becomes a selector.** When given, rows are restricted to test ids found in
those files, which is what makes `--filter k=v` meaningful and matches how `run` and
`grade` already treat the flag. Consequence: a test id in the database whose testcase file
was regenerated away drops out of the report. Omit `--testcases` for everything, unfiltered
and unenriched (the `prompt` / `request_type` / `domain` columns are then absent, not
empty).

### `llmeval/grade.py`

`grade()` and `grade_testcase()` gain `run_ids: Collection[str] | None`; a result whose
`run_id` is not in the set is skipped. Nothing else changes, because the rest of the
requirement is already true today:

* `gradings` is `UNIQUE(result_id, assertion_key)`, so the store already holds a grading
  per result rather than only for the latest one.
* `grade_testcase` already iterates *every* result for the cache key, already skips rows
  with `error IS NOT NULL`, and already skips assertion keys that are graded.

An empty resolved run set logs a warning and grades nothing (exit 0). An explicit
`--run-id` that cannot be resolved is an error (exit 2).

### `llmeval/cli.py`

* `add_run_selection(sp)` helper, applied to `grade` and `report`.
* **`report`** emits CSV: `--out results.csv`, `--provider` (repeatable, optional — default
  every provider), `--testcases`, `--filter`, `--db`, and the run selection. No HTML, no
  `open`.
* **`compare-report`** is the current report with its flags verbatim
  (`--providers`, `--baseline`, `--metrics`, `--order`, `--out`). `comparison/report.py` and
  `comparison/stats.py` are untouched.

### `reporting/csv_table.py`

Gains `--open` / `--no-open`, defaulting to **open**. Dispatch: `open` on darwin,
`xdg-open` on linux, `webbrowser` elsewhere. A launcher that fails is logged and the exit
code stays 0 — the HTML rendered fine, and a report you have to double-click is not a
failed report.

`write_table()` itself never opens anything, so library callers and tests are unaffected.

## The resulting workflow

```bash
uv run llmeval report --run-last-n 3 --provider configs/echo.json \
    --testcases testcases/ --out results.csv
python -m reporting.csv_table results.csv -o results.html --title "last 3 runs"
```

## Error handling

| Condition | Behaviour |
|---|---|
| flags from two selection groups | message + exit 2 |
| `--run-id` matches nothing / is ambiguous | message + exit 2 |
| `--run-after`/`--run-before` unparseable as a date *and* no such run | message + exit 2 |
| `--run-last-n` < 1 | message + exit 2 |
| selection resolves to zero runs | warning, empty CSV with headers, exit 0 |
| missing `--db` file | message + exit 2 (`sqlite3` would silently create one) |
| browser launcher unavailable | warning, exit 0 |

The distinction is between *a filter you got wrong* (exit 2) and *a filter that legitimately
matched nothing* (exit 0 with an empty table). Naming a run that doesn't exist is the first;
asking for the last 3 runs of a provider that has had 0 is the second.

## Testing

All offline, no credentials, no network — the standing rule for this suite.

* `framework_tests/test_runselect.py` — parsing of every date form and offset, naive =
  UTC, run-id boundaries, group conflicts, `last_n`, oldest-first ordering, cache-key
  narrowing.
* `framework_tests/test_resultrows.py` — row shape and count per result state, run
  grouping and attempt ordering, `latency_ms` on error rows, testcase enrichment and
  restriction, suite metadata vs id-shape fallback, whitespace trimming.
* `framework_tests/test_grade.py` — additions: run narrowing grades only the selected
  runs; error results still skipped; existing gradings still not redone.
* `framework_tests/test_cli.py` — new flags reach the library, conflicting groups exit 2,
  `report` writes a CSV, `compare-report` writes the old HTML.
* `reporting_tests/test_csv_table.py` — `--no-open` doesn't launch; `--open` calls the
  launcher (monkeypatched); a failing launcher still exits 0.
* `reporting_tests/test_run_report.py` — deleted with the module.

`pylint llmeval` stays at 10/10.

## Documentation

* `rewrite/README.md` — run selection flags; `report` now emits CSV; `compare-report`; the
  two-command workflow; `--open`.
* `rewrite/reporting/README.md` — drop the `run_report` section, document `--open`.
* `rewrite/CLAUDE.md` — the subcommand contract list gains `compare-report` and notes that
  `report` emits CSV.

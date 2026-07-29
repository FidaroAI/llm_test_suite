# CSV report tooling for `llmeval` — design

**Date:** 2026-07-29
**Status:** approved, implemented on branch `csv-report-tooling`

## Problem

The `llmeval` SQLite store holds everything a run produced, but inspecting it means
writing SQL by hand. We want simple reporting tools. The first need is a *generic*
foundation — display arbitrary tabular data (a CSV: any columns, any rows) as an HTML
page with column show/hide and per-column filtering — and then one concrete tool built on
it that dumps everything from a single run.

More tools will be built on the same foundation later, so the generic layer's interface
matters more than the first tool's feature set.

## Where this lives, and why

`rewrite/reporting/`, a sibling of `llmeval/` — **not** inside the package.

`rewrite/CLAUDE.md` draws git's plumbing/porcelain line: `llmeval/` holds *capabilities*
(assertion types, statistics, store columns), while *workflows and dashboards* live on top
of the CLI and the database. A "show me everything from run X" viewer is a workflow, so it
is porcelain. This is the first porcelain in the rewrite and therefore sets the convention:

* `reporting/` may import `llmeval` — the CLAUDE.md explicitly permits porcelain to call
  the library functions the subcommands wrap, and to query the SQLite schema directly.
* `llmeval` never imports `reporting`.
* `pyproject.toml`'s `include = ["llmeval*"]` already excludes it from the wheel, so
  porcelain is not shipped as part of the plumbing.
* It is invoked as `python -m reporting.run_report`. Deliberately **no** console-script
  entry point — that would install porcelain alongside the plumbing.

```
rewrite/reporting/
  __init__.py
  assets/tabulator.min.js     vendored, ~433KB
  assets/tabulator.min.css    vendored, ~28KB
  assets/LICENSE              MIT
  csv_table.py                generic layer: rows/CSV -> standalone HTML
  run_report.py               first tool: one run -> rows -> csv_table
  README.md
rewrite/reporting_tests/      pytest, mirroring framework_tests/
```

## The generic layer — `csv_table.py`

### Interface

```python
def render_table(rows, *, title, columns=None, subtitle=None) -> str
def write_table(rows, out_path, *, title, columns=None, subtitle=None) -> str
def render_csv_file(csv_path, *, title=None) -> str      # csv.DictReader -> render_table
```

`render_table` over a sequence of mappings is the core; CSV is a thin wrapper. Later tools
depend on `render_table(rows, title=...)` and nothing else.

`columns=None` means "the union of all row keys, in first-seen order". This is what makes
the tool generic over arbitrary columns, and it tolerates ragged rows — a row missing a key
renders an empty cell rather than failing.

CLI: `python -m reporting.csv_table data.csv -o out.html --title "..."`, accepting `-` for
stdin.

### Getting the data into the browser

Rows are serialised to a JSON array inside a `<script type="application/json">` block that
Tabulator reads, rather than emitted as an HTML `<table>`:

* JSON survives newlines, commas and quotes in model output without escaping games.
* Tabulator's column inference builds the grid from the data, so arbitrary columns are free.

**Model output is untrusted.** `<`, `>` and `&` are `\uXXXX`-escaped in the embedded JSON —
still valid JSON, but a value containing `</script>` cannot terminate the block early. This
is the same concern that made `llmeval/comparison/report.py` reach for Jinja autoescape;
here the escape happens at the serialisation boundary instead.

### Library choice: Tabulator

Chosen over DataTables and AG Grid Community because the two stated requirements are
first-class built-ins rather than extensions or hand-rolled UI:

| Requirement | Mechanism |
|---|---|
| Filter any column | `headerFilter: "input"` on every column (live) |
| Hide/show all columns | Toolbar **Show all** / **Hide all** + per-column checkbox → `table.toggleColumn(field)` |
| Sort any column | built in |
| Long cell values | `formatter: "textarea"` + height-clamped cells, with a **Compact/Expanded** toggle |
| Row count | `showing N of M` footer, updated on `dataFiltered` |
| Export | `table.download("csv")` |

DataTables needs jQuery plus the Buttons and ColVis extensions and hand-wired per-column
filters; AG Grid's column-visibility side panel is an Enterprise feature, so the show/hide
UI would still be hand-rolled.

### Assets are inlined

The vendored JS and CSS are read from `assets/` and inlined into every generated page. Each
report is then a single portable file that works offline and survives being emailed or
attached to a ticket, at a cost of ~470KB per file. The alternatives were rejected: relative
`<link>`s break when the HTML is moved away from its assets directory, and a CDN `<script>`
requires network access at view time and pins us to a third-party host.

### Deliberately absent

No theming knobs, no pagination configuration, no column-ordering API, no chart support.
The contract stays `render_table(rows, title=...)` so later tools have one thing to learn.

## The first tool — `run_report.py`

```
python -m reporting.run_report RUN_PREFIX -o run.html \
    [--db llmeval.sqlite3] [--testcases testcases/] [--csv rows.csv]
```

`RUN_PREFIX` is resolved by the store's existing `resolve_run()`, so a prefix like
`run_20260729-0435` suffices.

### Row building

```python
def run_rows(store, run_id, cases_by_id=None) -> list[dict[str, Any]]
```

A pure function, testable without HTML. It composes `Store.get_results_for_run()` with
`Store.get_gradings(result_id)` rather than issuing fresh SQL — an N+1 pattern that is
irrelevant at a few hundred rows and keeps porcelain off the store's column names.

**One row per result × assertion** (long format). Result fields repeat across a test's
assertions, and `assertion_key` / `metric` / `score` / `passed` are ordinary columns you can
filter on directly. A result with no gradings still yields exactly one row, with the grading
columns empty — that is the common case today (the current database has 26 results and zero
gradings), so it is the case tested hardest.

The rejected alternative was wide format (one row per result, assertions pivoted into
columns). It reads better for a single test but produces a sparse table whose column count
grows with the assertion vocabulary.

### Columns

| Group | Columns |
|---|---|
| identity | `test_id`, `attempt`, `suite` |
| test *(only with `--testcases`)* | `prompt`, `request_type`, `domain` |
| result | `output`, `reasoning`, `error`, `latency_ms`, `prompt_tokens`, `completion_tokens`, `total_tokens` |
| grading | `assertion_key`, `assertion_type`, `metric`, `score`, `passed`, `weight`, `judge_model`, `grading_reason` |
| provenance | `cache_key_hash`, `result_id` |

Notes on three columns that required checking rather than assuming:

* **`suite`** comes from testcase metadata, which every generator writes. It is *not*
  derived from `test_id` by splitting on `-`: ids are
  `<suite>-<sha1(prompt)[:10]>[-<variant>]` (`llmeval/generation/common.py`), so naive
  splitting mangles `research_rubrics-abc1234567-geval`. A
  `-[0-9a-f]{10}(-.*)?$` regex is used only as a documented fallback when no testcases are
  supplied.
* **token columns** flatten litellm's stored usage dump
  (`{completion_tokens, prompt_tokens, total_tokens, completion_tokens_details,
  prompt_tokens_details}`) into three filterable integers, dropping the details keys, which
  are null in practice.
* **`output` and `reasoning`** are the reasoning-split fields the provider layer guarantees,
  so no downstream parsing is needed.

### Degradation

Without `--testcases`, the test columns are **omitted entirely** rather than left blank. A
`test_id` present in the run but absent from the testcase files gets empty values rather
than raising — testcases get regenerated, and a stale id should not kill the report.

### Provenance in the page

The subtitle carries run identity: run id, provider name, note, started/finished timestamps
(or **unfinished** when `finished_at` is NULL, which flags a crashed run), result count and
error count. Anything a person reads off the page is derivable from the store, per the
plumbing's "output is read by programs, not just people" rule.

## Testing

Offline, no credentials, no network — the standing rule for this directory.

`reporting_tests/test_csv_table.py`
* ragged rows produce the union of keys, in first-seen order
* an explicit `columns=` list is honoured
* a `</script><script>alert(1)</script>` value renders inert
* CSV round-trip through the CLI
* an empty row list produces a valid page

`reporting_tests/test_run_report.py`
* row shape for a seeded run with gradings
* the zero-gradings case yields exactly one row per result
* enrichment adds prompt/metadata; unknown `test_id` degrades to empty values
* token flattening
* an unresolvable run prefix raises cleanly
* `suite` fallback from `test_id`, including the variant-suffixed form

`pyproject.toml` gains `reporting_tests` in `testpaths`; pylint gains `reporting` as a
second target.

## Future tools this foundation should carry

Not built now, listed so the interface isn't accidentally narrowed: a cross-run comparison
table, a per-suite pass-rate summary, an errors-only triage view, and a grading-disagreement
view. All are "build rows, hand them to `render_table`".

# `reporting/` — porcelain

Reporting tools built **on** the `llmeval` CLI and SQLite store, not inside them.

`llmeval` is the plumbing: capabilities, exposed explicitly, no workflow shortcuts (see
[../CLAUDE.md](../CLAUDE.md)). Dashboards are workflows, so they live here. The dependency
runs one way only — `reporting` imports `llmeval`, never the reverse — and this package is
excluded from the installed wheel, so porcelain never ships as part of the plumbing.

Run the tools as modules from the `rewrite/` directory. There is deliberately no
console-script entry point.

## The generic layer: `csv_table`

Turns arbitrary tabular data into a single self-contained HTML page.

```bash
python -m reporting.csv_table data.csv -o table.html --title "Some data"
cat data.csv | python -m reporting.csv_table - -o table.html
```

```python
from reporting import csv_table

csv_table.write_table(rows, "out.html", title="My report")   # rows: any list of dicts
```

Columns default to the union of all row keys in first-seen order, so ragged rows are fine
and callers never declare a schema. Every page comes with:

| Control | What it does |
|---|---|
| A filter box under each column header | live substring filter, any column |
| **Show all** / **Hide all columns** | plus a checkbox per column, kept in sync both ways |
| **Clear filters** | drops every header filter |
| **Expand rows** / **Collapse rows** | cells are height-clamped by default; expand shows full text |
| **Download CSV** | round-trips the filtered view |
| Column headers | click to sort |

The table is [Tabulator](https://tabulator.info) (MIT), vendored in `assets/` and **inlined**
into every page — so a report is one file that works offline and survives being emailed.
That costs ~470KB per report, which is the deliberate trade.

### Two things to preserve when editing

* **Cell values are untrusted** (usually model output). The embedded JSON escapes `<`, `>`
  and `&` so a value containing `</script>` can't close the data block, and cells render via
  a formatter that sets `textContent`.
* **The vendored assets must stay `|safe` in the template.** Browsers don't decode HTML
  entities inside `<script>`/`<style>`, so autoescaping them ships a corrupted library.
  There's a test asserting each asset appears verbatim.

## The run report: `run_report`

Everything one run produced, as a filterable table.

```bash
python -m reporting.run_report run_20260729-0451 -o run.html \
    --db llmeval.sqlite3 --testcases testcases/ --csv run.csv
```

The run argument accepts any unambiguous **prefix** of a run id. `--csv` additionally writes
the rows out, which is also how you feed them to something else.

**One row per result × assertion.** Result fields repeat across a test's assertions, so
`assertion_key` / `metric` / `score` / `passed` are ordinary columns you can filter on. A
result with no gradings still yields one row with those columns empty — which is what a
freshly-run, not-yet-graded run looks like.

`--testcases` is optional and adds `prompt`, `request_type` and `domain` by joining on
`test_id`. Without it those columns are absent entirely; with it, a `test_id` that isn't in
the files gets empty values rather than an error.

Two behaviours worth knowing:

* `suite` comes from testcase metadata when available. The fallback parses the
  `<suite>-<sha1[:10]>[-<variant>]` id shape rather than splitting on `-`, which would
  mangle variant-suffixed ids.
* `output` and `reasoning` are stripped of **surrounding** whitespace. Gateway outputs start
  with blank lines (the `\n\n\n` reasoning-split artifact), which would otherwise push the
  answer out of the visible cell. Internal formatting is untouched and the store keeps the
  raw value.

## Adding a tool

Build rows, hand them to `csv_table.write_table`. Keep row building in a pure function so
it's testable without HTML — `run_report.run_rows` is the pattern. Tests live in
`../reporting_tests/`, offline, no credentials.

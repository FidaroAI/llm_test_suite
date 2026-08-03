# `reporting/` — the CSV→HTML viewer

Reporting tools built **on** the `llmeval` CLI and SQLite store, not inside them.

`llmeval` is the plumbing: capabilities, exposed explicitly, no workflow shortcuts (see
[../CLAUDE.md](../CLAUDE.md)). Dashboards are workflows, so they live here. The dependency
runs one way only — `reporting` imports `llmeval`, never the reverse.

Run the tools as modules from the project directory. There is deliberately no console-script
entry point and this package stays out of the wheel — it renders arbitrary tabular data rather
than being a command anybody reaches for by name. That is a statement about *this* package,
not about friendly layers in general: [`llmevalx`](../llmevalx/README.md) is every bit as much
porcelain and does ship, as a first-class command.

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

### Opening what it writes

`--open` is the default when writing to a file, so the command lands you in a browser. Pass
`--no-open` in scripts and CI. A launcher that isn't there is a warning, not a failure — the
HTML rendered, and a report you have to double-click is not a failed report.

## Rendering an llmeval report

`llmeval report` does the selecting and emits CSV; this package renders it. Two commands,
because the split is the point — every filter is decided once, in the plumbing, and the
renderer's only input is a file.

```bash
uv run llmeval report --run-last-n 3 --provider configs/fidaro_prod.json \
    --out results.csv
python -m reporting.csv_table results.csv -o results.html --title "last 3 runs"
```

The CSV has one row per (result × assertion), plus one row per **errored** result with the
grading columns empty. A graded row carries the criterion as well as the score
(`assertion_value`, sitting next to `grading_reason`), so a rubric result can be read
without going and finding the rubric. Rows are grouped by run in chronological order and by attempt within
a test, so a retried test reads as its failures followed by the answer. Every row carries
the prompt it was produced from (`prompt` for the question, `messages` for the full
conversation as sent), so the page is readable without cross-referencing anything, and
anything the provider returned outside the OpenAI schema arrives verbatim in
`provider_specific_output` — a JSON cell this package renders as text like any other. See
[`llmeval/resultrows.py`](../llmeval/resultrows.py) for the column list and the ordering
guarantee, and the [main README](../README.md#reporting) for the selection flags.

## Adding a tool

Build rows, hand them to `csv_table.write_table`. If the rows come from the store, don't
build them here at all: which rows a report contains is a plumbing capability and belongs in
`llmeval/` next to `resultrows.py`, so that it stays testable without HTML and reusable by
anything that reads a CSV. Tests live in `../reporting_tests/`, offline, no credentials.

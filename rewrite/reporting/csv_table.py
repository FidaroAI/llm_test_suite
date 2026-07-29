"""Generic table viewer: arbitrary rows (or a CSV file) become a standalone HTML page.

This is the shared foundation every reporting tool builds on. Its whole contract is:

    render_table(rows, title=...) -> html

``rows`` is any sequence of mappings. Columns are inferred as the union of all row keys,
so ragged rows are fine (a missing key renders empty) and callers never declare a schema.
The page ships per-column filters, column show/hide, sorting and CSV export.

Two things worth knowing before editing this file:

* **The page is self-contained.** Tabulator's JS/CSS are vendored in ``assets/`` and
  *inlined*, so a report is a single file that works offline and survives being emailed.
  Relative ``<link>``s would break the moment someone moved the HTML; a CDN would need
  network access at view time.
* **Cell values are untrusted** (they're usually LLM output). Two defences, both
  deliberate: the embedded JSON escapes ``<``, ``>`` and ``&`` so a value containing
  ``</script>`` cannot close the data block early, and cells render through a formatter
  that sets ``textContent`` on a fresh element rather than any HTML-interpreting path.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
import sys
from typing import Any, Iterable, Mapping, Sequence

from jinja2 import Template

_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

# Cell width heuristics (px). Columns are sized from their content so an arbitrary CSV
# looks reasonable with no configuration: a `score` column stays narrow, a 4KB `output`
# column caps out rather than pushing everything else off-screen.
_MIN_WIDTH = 90
_MAX_WIDTH = 420
_PX_PER_CHAR = 7.5


@functools.lru_cache(maxsize=None)
def _asset(name: str) -> str:
    with open(os.path.join(_ASSET_DIR, name), encoding="utf-8") as f:
        return f.read()


def infer_columns(rows: Iterable[Mapping[str, Any]]) -> list[str]:
    """Union of every row's keys, in first-seen order.

    Dicts preserve insertion order, so the first row dictates the column order and later
    rows only ever append. That makes output stable for a caller that builds rows with a
    consistent key order, without forcing it to declare one.
    """
    columns: dict[str, None] = {}
    for row in rows:
        for key in row:
            columns.setdefault(str(key), None)
    return list(columns)


def _cell(value: Any) -> Any:
    """Coerce one value into something JSON-representable *and* sortable.

    Numbers and booleans stay native so Tabulator sorts them numerically rather than
    lexically ("10" < "9"). ``None`` stays None so the cell renders empty rather than
    literally "None". Anything else — a dict, a list, a datetime — becomes a string,
    with JSON preferred for containers because it stays greppable in the filter box.
    """
    if value is None or isinstance(value, (int, float, bool, str)):
        return value
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _embed_json(obj: Any) -> str:
    """Serialise for embedding inside a ``<script>`` block.

    The escapes keep the result valid JSON while making it inert as HTML: with ``<``
    escaped, a cell value containing ``</script>`` cannot terminate the block. U+2028 and
    U+2029 are escaped because they are legal in JSON but break JS string literals, which
    matters if this payload is ever moved into a plain ``<script>``.
    """
    text = json.dumps(obj, ensure_ascii=False, default=str)
    return (
        text.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        # Written as escapes: the raw characters are invisible in an editor.
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _column_width(column: str, values: Sequence[Any]) -> int:
    """Size a column from its content, using a high percentile rather than the max.

    The max would let one outlier row widen a column that is otherwise narrow; the 80th
    percentile tracks the typical case and lets the rest scroll inside the cell.
    """
    lengths = sorted(len(str(v)) for v in values if v is not None)
    typical = lengths[int(len(lengths) * 0.8)] if lengths else 0
    wanted = max(len(column), typical) * _PX_PER_CHAR + 26
    return int(min(_MAX_WIDTH, max(_MIN_WIDTH, wanted)))


def render_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    title: str,
    columns: Sequence[str] | None = None,
    subtitle: str | None = None,
) -> str:
    """Render rows as a standalone HTML page.

    :param rows: any sequence of mappings; keys become columns.
    :param title: page title and heading.
    :param columns: explicit column order. Defaults to :func:`infer_columns`. Naming a
        column absent from every row is allowed and yields an empty column — useful for
        keeping a fixed layout across reports.
    :param subtitle: optional line under the heading, for provenance.
    """
    cols = [str(c) for c in columns] if columns is not None else infer_columns(rows)
    data = [{c: _cell(row.get(c)) for c in cols} for row in rows]

    column_defs = [
        {"title": c, "field": c, "width": _column_width(c, [d[c] for d in data])} for c in cols
    ]

    # Autoescape is on for the human-facing strings (title, subtitle, column names).
    #
    # The four values below are marked |safe in the template, and MUST stay that way:
    # browsers do not decode HTML entities inside <script> or <style>, so autoescaping
    # them turns `createElement("div")` into `createElement(&#34;div&#34;)` and ships a
    # corrupted library or an unparseable payload. Each is safe for its own reason —
    # the assets are vendored files we control, and the JSON is escaped by _embed_json.
    return Template(_TEMPLATE, autoescape=True).render(
        title=title,
        subtitle=subtitle,
        row_count=len(data),
        columns=cols,
        data_json=_embed_json(data),
        columns_json=_embed_json(column_defs),
        tabulator_css=_asset("tabulator.min.css"),
        tabulator_js=_asset("tabulator.min.js"),
    )


def write_table(
    rows: Sequence[Mapping[str, Any]],
    out_path: str,
    *,
    title: str,
    columns: Sequence[str] | None = None,
    subtitle: str | None = None,
) -> str:
    """Render to ``out_path``, creating parent directories. Returns the path."""
    html = render_table(rows, title=title, columns=columns, subtitle=subtitle)
    parent = os.path.dirname(os.path.abspath(out_path))
    os.makedirs(parent, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path


# A row with more fields than the header has to go somewhere. DictReader's default puts
# the surplus under the key ``None``, which would surface as a column literally named
# "None"; naming it makes a malformed CSV obvious in the table instead of mysterious.
_EXTRA_KEY = "_unnamed_extra_fields"


def read_csv_rows(csv_path: str) -> list[dict[str, Any]]:
    """Read a CSV into row dicts. ``-`` reads stdin.

    ``DictReader`` is used rather than a positional reader so the header row becomes the
    column names — which is exactly the arbitrary-columns contract. ``utf-8-sig`` strips
    the BOM that spreadsheet exports write, which would otherwise corrupt the first
    column's name.
    """
    if csv_path == "-":
        return list(csv.DictReader(sys.stdin, restkey=_EXTRA_KEY))
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f, restkey=_EXTRA_KEY))


def render_csv_file(
    csv_path: str, *, title: str | None = None, subtitle: str | None = None
) -> str:
    """Render a CSV file as HTML. Title defaults to the file's basename."""
    rows = read_csv_rows(csv_path)
    if title is None:
        title = "stdin" if csv_path == "-" else os.path.basename(csv_path)
    return render_table(rows, title=title, subtitle=subtitle)


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ title }}</title>
<style>{{ tabulator_css|safe }}</style>
<style>
 :root{--fg:#1a1a1a;--muted:#6b7280;--line:#d1d5db;--bg:#fff;--chip:#f3f4f6}
 *{box-sizing:border-box}
 body{font-family:system-ui,-apple-system,sans-serif;margin:0;padding:14px 18px;
      color:var(--fg);background:var(--bg)}
 h1{font-size:17px;margin:0 0 2px}
 .sub{color:var(--muted);font-size:12px;margin:0 0 10px;white-space:pre-wrap}
 .bar{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:8px}
 button{font:inherit;font-size:12px;padding:4px 9px;border:1px solid var(--line);
        border-radius:5px;background:var(--chip);cursor:pointer}
 button:hover{background:#e5e7eb}
 .count{color:var(--muted);font-size:12px;margin-left:auto}
 details.cols{margin-bottom:8px;font-size:12px}
 details.cols summary{cursor:pointer;color:var(--muted);user-select:none}
 .colgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(190px,1fr));
          gap:2px 10px;padding:8px 2px}
 .colgrid label{display:flex;gap:6px;align-items:center;cursor:pointer;
                overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
 /* Cell text is wrapped in this div by the formatter, which is also what the
    compact/expanded toggle clamps. */
 .cellwrap{white-space:pre-wrap;word-break:break-word;line-height:1.35}
 .compact .cellwrap{max-height:4.4em;overflow:auto}
 .tabulator{font-size:12.5px;background:var(--bg)}
 .tabulator .tabulator-header .tabulator-col{background:#f9fafb}
 /* The container is sized to the viewport for virtual scrolling, so with few rows the
    theme's grey tableholder shows below the data. Keep it the page colour. */
 .tabulator .tabulator-tableholder{background:var(--bg)}
</style></head><body>

<h1>{{ title }}</h1>
{% if subtitle %}<p class="sub">{{ subtitle }}</p>{% endif %}

<div class="bar">
 <button id="show-all">Show all columns</button>
 <button id="hide-all">Hide all columns</button>
 <button id="clear-filters">Clear filters</button>
 <button id="toggle-wrap">Expand rows</button>
 <button id="download">Download CSV</button>
 <span class="count" id="count"></span>
</div>

<details class="cols"><summary>Columns ({{ columns|length }})</summary>
 <div class="colgrid">
  {% for c in columns %}
  <label><input type="checkbox" class="colbox" data-field="{{ c }}" checked>{{ c }}</label>
  {% endfor %}
 </div>
</details>

<div id="table" class="compact"></div>

<script type="application/json" id="rows">{{ data_json|safe }}</script>
<script type="application/json" id="cols">{{ columns_json|safe }}</script>
<script>{{ tabulator_js|safe }}</script>
<script>
(function () {
  "use strict";
  var rows = JSON.parse(document.getElementById("rows").textContent);
  var cols = JSON.parse(document.getElementById("cols").textContent);
  var host = document.getElementById("table");

  // Build the cell with textContent rather than any HTML-interpreting path: values are
  // model output. The wrapper div is also the hook the compact/expanded toggle clamps.
  function textCell(cell) {
    var v = cell.getValue();
    var el = document.createElement("div");
    el.className = "cellwrap";
    el.textContent = (v === null || v === undefined) ? "" : String(v);
    return el;
  }

  cols.forEach(function (c) {
    c.headerFilter = "input";       // the "filter any column" requirement
    c.headerFilterLiveFilter = true;
    c.formatter = textCell;
    c.resizable = true;
    c.titleFormatter = "plaintext"; // column names are data too
    c.tooltip = true;
  });

  var table = new Tabulator(host, {
    data: rows,
    columns: cols,
    // Column names come from arbitrary CSV headers, which may contain dots. Without
    // this, Tabulator would read "a.b" as a nested lookup and every cell would be empty.
    nestedFieldSeparator: false,
    // A fixed height turns on virtual rendering, so wide runs stay responsive.
    height: "calc(100vh - 165px)",
    layout: "fitData",
    placeholder: "No rows",
    columnDefaults: { minWidth: 60, headerSortTristate: true },
  });

  function showCount(shown) {
    var total = table.getDataCount();
    document.getElementById("count").textContent =
      shown === total ? total + " rows" : "showing " + shown + " of " + total + " rows";
  }
  // Take the count from the event's own `rows` argument, NOT from
  // table.getDataCount("active"): Tabulator fires dataFiltered *before* refreshing the
  // cache that call reads, so re-querying here returns the pre-filter count and the
  // footer silently never changes.
  table.on("dataFiltered", function (filters, rows) { showCount(rows.length); });
  table.on("tableBuilt", function () { showCount(table.getDataCount()); });

  function boxes() { return Array.prototype.slice.call(document.querySelectorAll(".colbox")); }

  // Per-column checkbox: authoritative both ways, so Show/Hide all resync the boxes.
  boxes().forEach(function (box) {
    box.addEventListener("change", function () {
      if (box.checked) { table.showColumn(box.dataset.field); }
      else { table.hideColumn(box.dataset.field); }
    });
  });

  function setAll(visible) {
    boxes().forEach(function (box) {
      box.checked = visible;
      if (visible) { table.showColumn(box.dataset.field); }
      else { table.hideColumn(box.dataset.field); }
    });
  }
  document.getElementById("show-all").addEventListener("click", function () { setAll(true); });
  document.getElementById("hide-all").addEventListener("click", function () { setAll(false); });

  document.getElementById("clear-filters").addEventListener("click", function () {
    table.clearHeaderFilter();
  });

  var wrapBtn = document.getElementById("toggle-wrap");
  wrapBtn.addEventListener("click", function () {
    var compact = host.classList.toggle("compact");
    wrapBtn.textContent = compact ? "Expand rows" : "Collapse rows";
    table.redraw(true);   // row heights changed; make Tabulator re-measure
  });

  document.getElementById("download").addEventListener("click", function () {
    table.download("csv", "table.csv");
  });
})();
</script>
</body></html>"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m reporting.csv_table",
        description="Render a CSV as a standalone HTML table (filterable, hideable columns).",
    )
    parser.add_argument("csv", help="CSV file to render, or - for stdin")
    parser.add_argument("-o", "--out", help="output HTML path (default: stdout)")
    parser.add_argument("--title", help="page title (default: the CSV's basename)")
    parser.add_argument("--subtitle", help="optional line under the heading")
    args = parser.parse_args(argv)

    html = render_csv_file(args.csv, title=args.title, subtitle=args.subtitle)
    if args.out:
        parent = os.path.dirname(os.path.abspath(args.out))
        os.makedirs(parent, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(html)
        print(f"wrote {args.out}")
    else:
        sys.stdout.write(html)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

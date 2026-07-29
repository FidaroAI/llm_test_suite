"""Tests for the generic table viewer.

The interesting properties are the two that make it *generic* (arbitrary columns, ragged
rows) and the one that makes it *safe* (untrusted values can't break out of the embedded
JSON). The rest of the page is Tabulator's job, not ours.
"""

import json
import re

import pytest

from reporting import csv_table


def rows_from(html: str) -> list[dict]:
    """Pull the embedded data payload back out, the way the browser does."""
    match = re.search(
        r'<script type="application/json" id="rows">(.*?)</script>', html, re.S
    )
    assert match, "no data payload in the page"
    return json.loads(match.group(1))


# --- generic over columns --------------------------------------------------


def test_columns_are_the_union_of_keys_in_first_seen_order():
    rows = [{"b": 1, "a": 2}, {"c": 3, "a": 4}]
    assert csv_table.infer_columns(rows) == ["b", "a", "c"]


def test_ragged_rows_render_missing_keys_as_empty():
    html = csv_table.render_table([{"a": 1}, {"b": 2}], title="t")
    assert rows_from(html) == [{"a": 1, "b": None}, {"a": None, "b": 2}]


def test_explicit_columns_control_order_and_selection():
    html = csv_table.render_table(
        [{"a": 1, "b": 2, "c": 3}], title="t", columns=["c", "a"]
    )
    assert rows_from(html) == [{"c": 3, "a": 1}]


def test_a_declared_column_absent_from_every_row_is_allowed():
    # Lets a caller keep one fixed layout across reports whose data varies.
    html = csv_table.render_table([{"a": 1}], title="t", columns=["a", "ghost"])
    assert rows_from(html) == [{"a": 1, "ghost": None}]


def test_no_rows_still_produces_a_usable_page():
    html = csv_table.render_table([], title="empty")
    assert rows_from(html) == []
    assert "No rows" in html  # Tabulator's placeholder is configured


def test_dotted_column_names_survive():
    # Tabulator would read "a.b" as a nested lookup; the page disables that, and the
    # value must come back intact rather than empty.
    html = csv_table.render_table([{"a.b": "kept"}], title="t")
    assert rows_from(html) == [{"a.b": "kept"}]
    assert "nestedFieldSeparator: false" in html


# --- value coercion --------------------------------------------------------


def test_numbers_and_booleans_stay_native_so_sorting_is_numeric():
    html = csv_table.render_table([{"n": 9, "f": 1.5, "b": True}], title="t")
    assert rows_from(html) == [{"n": 9, "f": 1.5, "b": True}]


def test_none_stays_null_rather_than_the_string_none():
    assert rows_from(csv_table.render_table([{"a": None}], title="t")) == [{"a": None}]


def test_containers_become_json_so_they_stay_greppable():
    html = csv_table.render_table([{"cfg": {"model": "m1"}}], title="t")
    assert rows_from(html) == [{"cfg": '{"model": "m1"}'}]


def test_unserialisable_values_fall_back_to_str():
    class Thing:
        def __str__(self):
            return "a thing"

    assert rows_from(csv_table.render_table([{"x": Thing()}], title="t")) == [{"x": "a thing"}]


# --- untrusted values ------------------------------------------------------


def test_a_script_close_tag_in_a_value_cannot_break_out_of_the_payload():
    payload = '</script><script>alert(1)</script>'
    html = csv_table.render_table([{"output": payload}], title="t")

    # The literal sequence must not appear anywhere: if it did, the browser would end the
    # data block early and execute the rest.
    assert "</script><script>" not in html
    assert "\\u003c/script\\u003e" in html
    # ...and it must still round-trip as data.
    assert rows_from(html) == [{"output": payload}]


def test_angle_brackets_and_ampersands_are_escaped_in_the_payload():
    html = csv_table.render_table([{"a": "<b>&amp;</b>"}], title="t")
    body = re.search(r'id="rows">(.*?)</script>', html, re.S).group(1)
    for char in "<>&":
        assert char not in body
    assert rows_from(html) == [{"a": "<b>&amp;</b>"}]


def test_line_separators_are_escaped():
    html = csv_table.render_table([{"a": "x\u2028y\u2029z"}], title="t")
    body = re.search(r'id="rows">(.*?)</script>', html, re.S).group(1)
    assert "\u2028" not in body and "\u2029" not in body
    assert rows_from(html) == [{"a": "x\u2028y\u2029z"}]


def test_title_and_subtitle_are_html_escaped():
    html = csv_table.render_table([], title="<b>t</b>", subtitle="a & b")
    assert "<b>t</b>" not in html
    assert "&lt;b&gt;t&lt;/b&gt;" in html
    assert "a &amp; b" in html


def test_column_names_are_html_escaped_in_the_checkbox_list():
    html = csv_table.render_table([{"<img>": 1}], title="t")
    assert "<img>" not in html
    assert "&lt;img&gt;" in html


# --- the page carries the required controls --------------------------------


def test_page_wires_up_column_visibility_and_filtering():
    html = csv_table.render_table([{"a": 1, "b": 2}], title="t")
    assert 'id="show-all"' in html and 'id="hide-all"' in html
    assert html.count('class="colbox"') == 2       # one checkbox per column
    assert 'headerFilter = "input"' in html        # every column filterable
    assert 'id="clear-filters"' in html
    assert 'id="download"' in html


def test_assets_are_inlined_so_the_page_works_offline():
    html = csv_table.render_table([{"a": 1}], title="t")
    assert "Tabulator" in html
    # Nothing that would trigger a fetch: no CDN script, no stylesheet link, no relative
    # asset reference. (Tabulator's source does contain the SVG XML namespace URL, which
    # is an identifier rather than something the browser retrieves — so testing for the
    # substring "https://" would be a false positive.)
    assert "<script src=" not in html
    assert "<link" not in html
    assert 'href="http' not in html and 'src="http' not in html


def test_vendored_assets_are_not_html_escaped():
    # Regression: the assets go inside <script>/<style>, where browsers do NOT decode
    # HTML entities. Autoescaping them turns createElement("div") into
    # createElement(&#34;div&#34;) and ships a silently broken library.
    html = csv_table.render_table([{"a": 1}], title="t")
    assert "&#34;" not in html
    assert 'createElement("div")' in html


@pytest.mark.parametrize(
    "asset,tag",
    [("tabulator.min.js", "script"), ("tabulator.min.css", "style")],
)
def test_each_asset_is_inlined_verbatim(asset, tag):
    # The strongest available check that the library isn't corrupted in transit: the
    # bytes inside the tag are identical to the vendored file. Any escaping, truncation
    # or template mangling shows up here rather than as a blank page in a browser.
    source = csv_table._asset(asset)
    html = csv_table.render_table([{"a": 1}], title="t")
    assert f"<{tag}>{source}</{tag}>" in html


# --- widths ----------------------------------------------------------------


def test_long_columns_are_wider_than_short_ones_but_capped():
    narrow = csv_table._column_width("n", [1, 2, 3])
    wide = csv_table._column_width("output", ["x" * 4000] * 5)
    assert narrow < wide == csv_table._MAX_WIDTH
    assert narrow >= csv_table._MIN_WIDTH


def test_width_ignores_a_single_outlier():
    # An 80th-percentile width means one huge row doesn't widen an otherwise narrow column.
    values = ["ab"] * 20 + ["x" * 3000]
    assert csv_table._column_width("a", values) < csv_table._MAX_WIDTH


def test_width_handles_an_all_empty_column():
    assert csv_table._column_width("a", [None, None]) == csv_table._MIN_WIDTH


# --- CSV entry points ------------------------------------------------------


@pytest.fixture
def csv_file(tmp_path):
    path = tmp_path / "data.csv"
    path.write_text(
        'name,note\n'
        'Ada,"likes commas: a,b"\n'
        'Grace,"has ""quotes"" and\nnewlines"\n',
        encoding="utf-8",
    )
    return path


def test_csv_header_becomes_the_columns(csv_file):
    html = csv_table.render_csv_file(str(csv_file))
    assert rows_from(html)[0] == {"name": "Ada", "note": "likes commas: a,b"}


def test_surplus_fields_get_a_named_column_not_one_called_none(csv_file, tmp_path):
    # A row with more fields than the header is malformed, but should be visible in the
    # table rather than landing in a column literally named "None".
    path = tmp_path / "ragged.csv"
    path.write_text("name,note\nAda,fine\nGrace,too,many,fields\n", encoding="utf-8")
    rows = csv_table.read_csv_rows(str(path))
    assert rows[1][csv_table._EXTRA_KEY] == ["many", "fields"]
    assert "None" not in csv_table.infer_columns(rows)


def test_csv_quoting_and_embedded_newlines_survive(csv_file):
    assert rows_from(csv_table.render_csv_file(str(csv_file)))[1] == {
        "name": "Grace",
        "note": 'has "quotes" and\nnewlines',
    }


def test_csv_title_defaults_to_the_basename(csv_file):
    assert "data.csv" in csv_table.render_csv_file(str(csv_file))


def test_utf8_bom_is_stripped_from_the_first_header(tmp_path):
    # Spreadsheet exports write a BOM; without utf-8-sig the first column name would be
    # "\ufeffname" and every lookup on it would miss.
    path = tmp_path / "bom.csv"
    path.write_bytes("name,n\nAda,1\n".encode("utf-8-sig"))
    assert csv_table.infer_columns(csv_table.read_csv_rows(str(path))) == ["name", "n"]


def test_cli_writes_a_file(tmp_path, csv_file, capsys):
    out = tmp_path / "nested" / "out.html"
    # --no-open, or this test would launch a browser on the machine running it.
    assert (
        csv_table.main([str(csv_file), "-o", str(out), "--title", "My data", "--no-open"]) == 0
    )
    html = out.read_text(encoding="utf-8")
    assert "My data" in html
    assert len(rows_from(html)) == 2
    assert "wrote" in capsys.readouterr().out


# --- opening the rendered page --------------------------------------------


def test_open_is_the_default(tmp_path, csv_file, monkeypatch):
    out = tmp_path / "t.html"
    opened = []
    monkeypatch.setattr(csv_table, "open_in_browser", lambda p: opened.append(p) or True)
    assert csv_table.main([str(csv_file), "-o", str(out)]) == 0
    assert opened == [str(out)]


def test_no_open_suppresses_the_launch(tmp_path, csv_file, monkeypatch):
    out = tmp_path / "t.html"
    opened = []
    monkeypatch.setattr(csv_table, "open_in_browser", lambda p: opened.append(p) or True)
    assert csv_table.main([str(csv_file), "-o", str(out), "--no-open"]) == 0
    assert opened == []


def test_nothing_is_opened_when_writing_to_stdout(csv_file, monkeypatch, capsys):
    opened = []
    monkeypatch.setattr(csv_table, "open_in_browser", lambda p: opened.append(p) or True)
    assert csv_table.main([str(csv_file)]) == 0
    assert opened == []
    assert "<!doctype html>" in capsys.readouterr().out


def test_darwin_uses_the_open_command(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(csv_table.sys, "platform", "darwin")
    monkeypatch.setattr(csv_table.subprocess, "run", lambda argv, check: calls.append(argv))
    target = tmp_path / "t.html"
    target.write_text("<html></html>")
    assert csv_table.open_in_browser(str(target)) is True
    assert calls == [["open", str(target)]]


def test_linux_uses_xdg_open(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(csv_table.sys, "platform", "linux")
    monkeypatch.setattr(csv_table.subprocess, "run", lambda argv, check: calls.append(argv))
    target = tmp_path / "t.html"
    target.write_text("<html></html>")
    assert csv_table.open_in_browser(str(target)) is True
    assert calls == [["xdg-open", str(target)]]


def test_a_failing_launcher_still_exits_zero(tmp_path, csv_file, monkeypatch, capsys):
    out = tmp_path / "t.html"

    def boom(argv, check):
        raise OSError("no such tool")

    monkeypatch.setattr(csv_table.sys, "platform", "darwin")
    monkeypatch.setattr(csv_table.subprocess, "run", boom)
    # The HTML rendered; a report you have to double-click is not a failed report.
    assert csv_table.main([str(csv_file), "-o", str(out)]) == 0
    assert out.exists()
    assert "could not open" in capsys.readouterr().err


def test_cli_writes_to_stdout_without_o(csv_file, capsys):
    assert csv_table.main([str(csv_file)]) == 0
    assert "<!doctype html>" in capsys.readouterr().out


def test_cli_reads_stdin(monkeypatch, capsys):
    import io

    monkeypatch.setattr("sys.stdin", io.StringIO("a,b\n1,2\n"))
    assert csv_table.main(["-"]) == 0
    out = capsys.readouterr().out
    assert rows_from(out) == [{"a": "1", "b": "2"}]
    assert "stdin" in out


def test_write_table_creates_parent_directories(tmp_path):
    out = tmp_path / "a" / "b" / "t.html"
    assert csv_table.write_table([{"a": 1}], str(out), title="t") == str(out)
    assert out.exists()

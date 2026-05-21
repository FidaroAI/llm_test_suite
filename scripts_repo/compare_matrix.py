#!/usr/bin/env python3
"""Per-test 2xN matrix report comparing a baseline run against the latest run.

For every test in the union of the baseline and the latest eval, render a small
table with two rows (baseline, latest) and one column per assertion. Each cell
shows that run's score; the latest row is coloured red / white / green when its
score is worse / same / better than the baseline. Cells are marked:

- "missing" when the test (or that assertion) was absent from a run, and
- "ERROR"   when grading could not produce a score, e.g. a provider/response
            error (failureReason 2) or an llm-rubric judge failure such as a
            missing AWS/Bedrock token (an infra error in the assertion reason).

A genuine zero score (e.g. a real refusal judgement) is shown as 0.00, not
ERROR. Tests containing any ERROR float to the top so failures are obvious.

Usage:
    compare_matrix.py
    compare_matrix.py --baseline baselines/prod.json --latest results/local/latest.json
    compare_matrix.py --out matrix_report.html
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_BASELINE_DIR = Path("baselines")
DEFAULT_LATEST = Path("results/local/latest.json")
DEFAULT_OUT = Path("matrix_report.html")

# Float epsilon for "same score".
EPS = 1e-9

# promptfoo ResultFailureReason.ERROR (provider / response level error).
FAILURE_REASON_ERROR = 2

# Infra-error signatures in an assertion's grading reason. These indicate the
# judge could not run (vs. a legitimate low-score judgement of the output).
_GRADER_ERROR_RE = re.compile(
    r"(invoke model error|access ?denied|api error\b|rate ?limit|timeout"
    r"|unauthorized|forbidden|service unavailable|credential|expired token"
    r"|could not parse|failed to (call|invoke|reach|parse)|exception:)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaseKey:
    desc: str
    prompt: str


@dataclass
class Cell:
    kind: str  # "score" | "error" | "missing"
    score: float | None
    detail: str  # judge reason or error message (shown on hover)


@dataclass
class Column:
    key: str  # unique within a test
    header: str  # short label (metric or assertion type)
    full: str  # full rubric / assertion text


@dataclass
class RunTest:
    suite: str | None
    columns: list[Column]
    cells: dict[str, Cell]  # column key -> Cell


@dataclass
class MatrixRow:
    key: CaseKey
    suite: str | None
    columns: list[Column]
    baseline: dict[str, Cell]
    latest: dict[str, Cell]
    verdicts: dict[str, str]  # column key -> better | worse | same | na
    has_error: bool = field(default=False)


def read_eval_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _grader_error(reason) -> bool:
    return bool(reason) and bool(_GRADER_ERROR_RE.search(str(reason)))


def _test_level_error(result: dict) -> str | None:
    """Return the error message if the whole test failed to run, else None."""
    if result.get("failureReason") == FAILURE_REASON_ERROR:
        return str(result.get("error") or "test errored")
    comps = (result.get("gradingResult") or {}).get("componentResults")
    if not comps and result.get("error") and result.get("failureReason") not in (0, 1):
        return str(result.get("error"))
    return None


def _test_key(result: dict) -> CaseKey:
    test_case = result.get("testCase") or {}
    desc = test_case.get("description")
    if not desc:
        user = (test_case.get("vars") or result.get("vars") or {}).get("user", "")
        desc = f"user: {str(user)[:60]}" if user else f"test[{result.get('testIdx', '?')}]"
    prompt = (result.get("prompt") or {}).get("label") or "<no-prompt>"
    return CaseKey(desc=desc, prompt=prompt)


def index_run(eval_json: dict) -> dict:
    """Map CaseKey -> RunTest for every test in an eval result JSON."""
    run: dict = {}
    for result in eval_json.get("results", {}).get("results", []):
        test_case = result.get("testCase") or {}
        meta = test_case.get("metadata") or {}
        asserts = test_case.get("assert") or []
        comps = (result.get("gradingResult") or {}).get("componentResults") or []
        test_error = _test_level_error(result)

        columns: list[Column] = []
        cells: dict[str, Cell] = {}
        seen: dict[str, int] = {}
        for i, assertion in enumerate(asserts):
            value = assertion.get("value") or f"<assertion-{i}>"
            n = seen.get(value, 0)
            seen[value] = n + 1
            col_key = value if n == 0 else f"{value}#{n}"
            header = assertion.get("metric") or assertion.get("type") or f"assert {i}"
            columns.append(Column(key=col_key, header=str(header), full=str(value)))

            if test_error is not None:
                cells[col_key] = Cell("error", None, test_error)
                continue
            if i >= len(comps):
                cells[col_key] = Cell("error", None, "no grading result")
                continue
            reason = comps[i].get("reason")
            score = comps[i].get("score")
            if _grader_error(reason) or (score is None and reason):
                cells[col_key] = Cell("error", None, str(reason))
            else:
                cells[col_key] = Cell("score", float(score) if score is not None else 0.0,
                                      str(reason or ""))

        run[_test_key(result)] = RunTest(suite=meta.get("suite"), columns=columns, cells=cells)
    return run


def compare(baseline_cell: Cell, latest_cell: Cell) -> str:
    """Verdict for the latest cell relative to the baseline cell."""
    if baseline_cell.kind != "score" or latest_cell.kind != "score":
        return "na"
    if latest_cell.score > baseline_cell.score + EPS:
        return "better"
    if latest_cell.score < baseline_cell.score - EPS:
        return "worse"
    return "same"


_MISSING = Cell("missing", None, "")


def _merge_columns(base_rt: RunTest | None, latest_rt: RunTest | None) -> list:
    """Union of columns: baseline order first, then any latest-only columns."""
    columns: list[Column] = []
    seen: set[str] = set()
    for rt in (base_rt, latest_rt):
        if rt is None:
            continue
        for col in rt.columns:
            if col.key not in seen:
                seen.add(col.key)
                columns.append(col)
    return columns


def build_rows(baseline_run: dict, latest_run: dict) -> list:
    """Build one MatrixRow per test in the union, errors-first."""
    rows: list[MatrixRow] = []
    for key in set(baseline_run) | set(latest_run):
        base_rt = baseline_run.get(key)
        latest_rt = latest_run.get(key)
        columns = _merge_columns(base_rt, latest_rt)
        suite = (latest_rt or base_rt).suite

        baseline_cells: dict[str, Cell] = {}
        latest_cells: dict[str, Cell] = {}
        verdicts: dict[str, str] = {}
        has_error = False
        for col in columns:
            b = base_rt.cells.get(col.key, _MISSING) if base_rt else _MISSING
            c = latest_rt.cells.get(col.key, _MISSING) if latest_rt else _MISSING
            baseline_cells[col.key] = b
            latest_cells[col.key] = c
            verdicts[col.key] = compare(b, c)
            if b.kind == "error" or c.kind == "error":
                has_error = True

        rows.append(MatrixRow(key=key, suite=suite, columns=columns,
                            baseline=baseline_cells, latest=latest_cells,
                            verdicts=verdicts, has_error=has_error))

    # Errors first, then alphabetical by description for stable output.
    rows.sort(key=lambda r: (not r.has_error, r.key.desc))
    return rows


_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
.summary { font-size: 1.05rem; margin-bottom: 1.5rem; }
.test { margin-bottom: 1.75rem; }
.test h3 { margin: 0 0 .35rem; font-size: 1rem; }
.test .suite { color: #888; font-weight: 400; font-size: .85rem; margin-left: .5rem; }
table { border-collapse: collapse; }
th, td { border: 1px solid #ddd; padding: .35rem .6rem; text-align: center;
         font-variant-numeric: tabular-nums; }
th { background: #fafafa; font-weight: 600; }
th.runlbl, td.runlbl { text-align: right; background: #fafafa; font-weight: 600; }
.v-better { background: #d7f5dd; }
.v-worse  { background: #fad9d4; }
.v-same   { background: #ffffff; }
.v-na     { background: #f4f4f4; color: #555; }
.cell-error   { background: #ffe9b3; color: #8a5a00; font-weight: 600; }
.cell-missing { background: #f4f4f4; color: #999; font-style: italic; }
.cell-base    { background: #ffffff; }
.has-error h3 { color: #c0341d; }
"""


def _cell_html(cell: Cell, latest: bool, verdict: str) -> str:
    title = f' title="{html.escape(cell.detail)}"' if cell.detail else ""
    if cell.kind == "error":
        return f'<td class="cell-error"{title}>ERROR</td>'
    if cell.kind == "missing":
        return '<td class="cell-missing">missing</td>'
    text = f"{cell.score:.2f}"
    cls = f"v-{verdict}" if latest else "cell-base"
    return f'<td class="{cls}"{title}>{text}</td>'


def render_html(rows: list) -> str:
    n_tests = len(rows)
    n_err = sum(1 for r in rows if r.has_error)
    better = worse = same = 0
    for r in rows:
        for v in r.verdicts.values():
            if v == "better":
                better += 1
            elif v == "worse":
                worse += 1
            elif v == "same":
                same += 1
    summary = (
        f"{n_tests} tests &middot; {n_err} with errors &middot; "
        f"latest cells: {better} better, {worse} worse, {same} same"
    )

    blocks = []
    for r in rows:
        headers = "".join(
            f'<th title="{html.escape(c.full)}">{html.escape(c.header)}</th>'
            for c in r.columns
        )
        base_cells = "".join(
            _cell_html(r.baseline[c.key], latest=False, verdict="na") for c in r.columns
        )
        latest_cells = "".join(
            _cell_html(r.latest[c.key], latest=True, verdict=r.verdicts[c.key])
            for c in r.columns
        )
        suite = f'<span class="suite">{html.escape(r.suite)}</span>' if r.suite else ""
        cls = "test has-error" if r.has_error else "test"
        blocks.append(
            f'<div class="{cls}">'
            f"<h3>{html.escape(r.key.desc)}{suite}</h3>"
            "<table><thead><tr><th class='runlbl'>run</th>"
            f"{headers}</tr></thead><tbody>"
            f'<tr><td class="runlbl">baseline</td>{base_cells}</tr>'
            f'<tr><td class="runlbl">latest</td>{latest_cells}</tr>'
            "</tbody></table></div>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Baseline vs latest — per-test matrix</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Baseline vs latest — per-test matrix</h1>"
        f'<div class="summary">{summary}</div>'
        + "".join(blocks)
        + "</body></html>"
    )


def _autodetect_baseline(out_dir: Path) -> Path:
    candidates = sorted(p for p in out_dir.glob("*.json"))
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise SystemExit(
            f"no baseline found in {out_dir}/; pass --baseline explicitly"
        )
    raise SystemExit(
        f"multiple baselines in {out_dir}/: "
        f"{', '.join(p.name for p in candidates)}; pass --baseline explicitly"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
        help="Baseline result JSON (default: the sole file in baselines/).",
    )
    parser.add_argument(
        "--latest",
        type=Path,
        default=DEFAULT_LATEST,
        help=f"Latest result JSON (default: {DEFAULT_LATEST}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help=f"Output HTML path (default: {DEFAULT_OUT}).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    baseline_path = args.baseline or _autodetect_baseline(DEFAULT_BASELINE_DIR)

    baseline_run = index_run(read_eval_json(baseline_path))
    latest_run = index_run(read_eval_json(args.latest))
    rows = build_rows(baseline_run, latest_run)
    args.out.write_text(render_html(rows), encoding="utf-8")

    n_err = sum(1 for r in rows if r.has_error)
    print(f"{len(rows)} tests ({n_err} with errors) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

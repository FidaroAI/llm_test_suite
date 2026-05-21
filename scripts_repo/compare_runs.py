#!/usr/bin/env python3
"""Compare llm-rubric scores between a baseline and a candidate promptfoo eval.

Extracts one "cell" per llm-rubric assertion per test from each eval JSON,
joins them on (test description, prompt label, rubric text), classifies each
score delta against an absolute tolerance band, and writes a self-contained
HTML report. Scope is restricted to tests whose metadata.suite is in the
allowlist (default: research_rubrics, agentharm_refusal).

Usage:
    compare_runs.py baselines/prod.json results/local/latest.json
    compare_runs.py BASE.json CAND.json --tolerance 0.05 --out report.html \\
        --suite research_rubrics --suite agentharm_refusal
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SUITES = ("research_rubrics", "agentharm_refusal")
DEFAULT_TOLERANCE = 0.05


@dataclass(frozen=True)
class CellKey:
    test: str
    prompt: str
    assertion: str


@dataclass
class Cell:
    key: "CellKey"
    suite: str
    metric: str | None
    weight: float
    score: float
    assertion_value: str


def read_eval_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_cells(eval_json: dict, suites) -> dict:
    """Map CellKey -> Cell for every llm-rubric assertion in the given suites."""
    suites = set(suites)
    cells: dict = {}
    for result in eval_json.get("results", {}).get("results", []):
        test_case = result.get("testCase") or {}
        meta = test_case.get("metadata") or {}
        if meta.get("suite") not in suites:
            continue
        suite = meta["suite"]
        test_desc = test_case.get("description") or "<no-description>"
        prompt_label = (result.get("prompt") or {}).get("label") or "<no-prompt>"
        asserts = test_case.get("assert") or []
        comps = (result.get("gradingResult") or {}).get("componentResults") or []
        seen: dict = {}
        for i, assertion in enumerate(asserts):
            if assertion.get("type") != "llm-rubric":
                continue
            if i >= len(comps):
                continue
            value = assertion.get("value") or f"<assertion-{i}>"
            n = seen.get(value, 0)
            seen[value] = n + 1
            assertion_key = value if n == 0 else f"{value}#{n}"
            key = CellKey(test=test_desc, prompt=prompt_label, assertion=assertion_key)
            score = comps[i].get("score")
            cells[key] = Cell(
                key=key,
                suite=suite,
                metric=assertion.get("metric"),
                weight=float(assertion.get("weight", 1) or 1),
                score=float(score) if score is not None else 0.0,
                assertion_value=value,
            )
    return cells


@dataclass
class CellDiff:
    key: "CellKey"
    suite: str
    metric: str | None
    assertion_value: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    status: str  # improved | regressed | within | new | removed


def classify(delta: float, tolerance: float) -> str:
    """Three-way verdict for a delta against an absolute tolerance band.

    A move exactly equal to the band is treated as within tolerance.
    """
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "regressed"
    return "within"


def diff_cells(baseline_cells: dict, candidate_cells: dict, tolerance: float) -> list:
    """Join baseline and candidate cells by key; classify every cell."""
    diffs: list = []
    all_keys = sorted(
        set(baseline_cells) | set(candidate_cells),
        key=lambda k: (k.test, k.prompt, k.assertion),
    )
    for key in all_keys:
        b = baseline_cells.get(key)
        c = candidate_cells.get(key)
        if b and c:
            delta = c.score - b.score
            diffs.append(CellDiff(key, c.suite, c.metric, c.assertion_value,
                                  b.score, c.score, delta, classify(delta, tolerance)))
        elif c is not None:
            diffs.append(CellDiff(key, c.suite, c.metric, c.assertion_value,
                                  None, c.score, None, "new"))
        else:
            diffs.append(CellDiff(key, b.suite, b.metric, b.assertion_value,
                                  b.score, None, None, "removed"))
    return diffs


def summarize(diffs: list) -> dict:
    """Count diffs by status."""
    counts = {"improved": 0, "regressed": 0, "within": 0, "new": 0, "removed": 0}
    for d in diffs:
        counts[d.status] += 1
    return counts


def diff_test_keys(baseline_cells: dict, candidate_cells: dict):
    """Sorted (baseline-only, candidate-only) test descriptions."""
    b_tests = {k.test for k in baseline_cells}
    c_tests = {k.test for k in candidate_cells}
    return sorted(b_tests - c_tests), sorted(c_tests - b_tests)


_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
.summary { font-size: 1.1rem; margin-bottom: 1rem; }
.drift { background: #fff3cd; border: 1px solid #ffe69c; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eee; }
th { background: #fafafa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
h2 { margin-top: 1.5rem; }
.status-improved td.delta { color: #0a7d28; font-weight: 600; }
.status-regressed td.delta { color: #c0341d; font-weight: 600; }
.status-within td.delta { color: #888; }
.status-new td, .status-removed td { color: #555; font-style: italic; }
"""


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_delta(value) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _sort_key(diff):
    # Worst regressions first; new/removed (delta None) sort to the bottom.
    return (0, diff.delta) if diff.delta is not None else (1, 0.0)


def render_html(diffs: list, counts: dict, drift, tolerance: float) -> str:
    """Render a self-contained HTML report. `drift` is (only_base, only_cand)."""
    only_base, only_cand = drift
    summary = (
        f"{counts['improved']} improved &middot; {counts['regressed']} regressed "
        f"&middot; {counts['within']} within &plusmn;{tolerance:g} &middot; "
        f"{counts['new']} new &middot; {counts['removed']} removed"
    )

    drift_html = ""
    if only_base or only_cand:
        parts = []
        if only_base:
            parts.append("missing from candidate: "
                         + ", ".join(html.escape(t) for t in only_base))
        if only_cand:
            parts.append("only in candidate: "
                         + ", ".join(html.escape(t) for t in only_cand))
        drift_html = f'<div class="drift">⚠ config drift — {"; ".join(parts)}</div>'

    # Group rows by suite, preserving first-seen suite order.
    suites: list = []
    grouped: dict = {}
    for d in diffs:
        if d.suite not in grouped:
            grouped[d.suite] = []
            suites.append(d.suite)
        grouped[d.suite].append(d)

    sections = []
    for suite in suites:
        rows = []
        for d in sorted(grouped[suite], key=_sort_key):
            rows.append(
                f'<tr class="status-{d.status}">'
                f"<td>{html.escape(d.key.test)}</td>"
                f"<td>{html.escape(d.assertion_value)}</td>"
                f"<td>{html.escape(d.metric or '')}</td>"
                f'<td class="num">{_fmt(d.baseline)}</td>'
                f'<td class="num">{_fmt(d.candidate)}</td>'
                f'<td class="num delta">{_fmt_delta(d.delta)}</td>'
                f"<td>{d.status}</td>"
                "</tr>"
            )
        sections.append(
            f"<h2>{html.escape(suite)}</h2>"
            "<table><thead><tr>"
            "<th>test</th><th>assertion</th><th>metric</th>"
            "<th>baseline</th><th>candidate</th><th>&Delta;</th><th>status</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Provider comparison</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Provider comparison</h1>"
        f'<div class="summary">{summary}</div>'
        f"{drift_html}"
        + "".join(sections)
        + "</body></html>"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("baseline_json", help="Frozen baseline result JSON.")
    parser.add_argument("candidate_json", help="Candidate result JSON.")
    parser.add_argument(
        "--suite",
        action="append",
        default=None,
        help="Suite to include (repeatable). Defaults to "
        f"{', '.join(DEFAULT_SUITES)}.",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Absolute score band to ignore (default: {DEFAULT_TOLERANCE}).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report.html"),
        help="Output HTML path (default: report.html).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suites = args.suite if args.suite else list(DEFAULT_SUITES)

    baseline_cells = extract_cells(read_eval_json(args.baseline_json), suites)
    candidate_cells = extract_cells(read_eval_json(args.candidate_json), suites)

    diffs = diff_cells(baseline_cells, candidate_cells, args.tolerance)
    counts = summarize(diffs)
    drift = diff_test_keys(baseline_cells, candidate_cells)

    args.out.write_text(
        render_html(diffs, counts, drift, args.tolerance), encoding="utf-8"
    )
    print(
        f"{counts['improved']} improved, {counts['regressed']} regressed, "
        f"{counts['within']} within, {counts['new']} new, "
        f"{counts['removed']} removed -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

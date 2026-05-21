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
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

DEFAULT_SUITES = ("research_rubrics", "agentharm_refusal")
DEFAULT_TOLERANCE = 0.05
# promptfoo web UI base; user is responsible for running the server there.
DEFAULT_UI_BASE_URL = "http://localhost:3000"

# promptfoo ResultFailureReason.ERROR (provider / response level error).
_FAILURE_REASON_ERROR = 2

# Infra-error signatures in an assertion's grading reason — the judge could not
# run (e.g. a missing AWS/Bedrock token), as opposed to a real low-score verdict.
_GRADER_ERROR_RE = re.compile(
    r"(invoke model error|access ?denied|api error\b|rate ?limit|timeout"
    r"|unauthorized|forbidden|service unavailable|credential|expired token"
    r"|could not parse|failed to (call|invoke|reach|parse)|exception:)",
    re.IGNORECASE,
)


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
    search: str = ""  # term that isolates this test in the promptfoo UI search


def read_eval_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_eval_id(eval_json: dict) -> str | None:
    """The eval's id, preferring the live evalId, then a frozen baseline's."""
    return eval_json.get("evalId") or (eval_json.get("_baseline_meta") or {}).get(
        "eval_id"
    )


def _result_errored(result: dict) -> bool:
    """True when a result failed to produce real grades (vs. a low score).

    Catches provider/response errors (failureReason ERROR) and llm-rubric judge
    failures whose component reason is an infra error.
    """
    if result.get("failureReason") == _FAILURE_REASON_ERROR:
        return True
    comps = (result.get("gradingResult") or {}).get("componentResults") or []
    if any(_GRADER_ERROR_RE.search(str(c.get("reason") or "")) for c in comps):
        return True
    if not comps and result.get("error") and result.get("failureReason") not in (0, 1):
        return True
    return False


def parse_provider_yaml(text: str) -> dict:
    """Pull the fields needed for a curl from a flat provider YAML.

    Tailored to this repo's gateway provider files (label/id at top level,
    apiBaseUrl/apiKey/temperature/max_tokens under config). Strips inline
    comments and surrounding quotes. Avoids a PyYAML dependency.
    """

    def grab(key: str) -> str | None:
        m = re.search(
            rf"^\s*{re.escape(key)}:\s*(.+?)\s*(?:#.*)?$", text, re.MULTILINE
        )
        return m.group(1).strip().strip('"').strip("'") if m else None

    return {
        k: grab(k)
        for k in ("label", "id", "apiBaseUrl", "apiKey", "temperature", "max_tokens")
    }


def _model_from_id(provider_id: str) -> str:
    """Model name from a provider id like 'openai:chat:Qwen/Q' -> 'Qwen/Q'."""
    parts = provider_id.split(":")
    return ":".join(parts[2:]) if len(parts) >= 3 else provider_id


def extract_request_info(eval_json: dict) -> dict:
    """Map test description -> {model, messages_raw, provider_label}.

    Covers every result (including errored ones), so a curl can be offered for
    cells that have no score.
    """
    info: dict = {}
    for result in eval_json.get("results", {}).get("results", []):
        test_case = result.get("testCase") or {}
        desc = test_case.get("description") or "<no-description>"
        if desc in info:
            continue
        provider = result.get("provider") or {}
        info[desc] = {
            "model": _model_from_id(provider.get("id") or ""),
            "messages_raw": (result.get("prompt") or {}).get("raw") or "",
            "provider_label": provider.get("label") or "",
        }
    return info


def resolve_endpoints(eval_json: dict, base_dir, override_url: str | None = None) -> dict:
    """Map provider label -> {url, api_key, temperature, max_tokens}.

    Reads the provider YAMLs referenced by `config.providers` (resolved relative
    to cwd, then `base_dir`). `override_url` forces the base URL for every
    provider and also covers labels with no resolvable YAML.
    """
    endpoints: dict = {}
    for entry in eval_json.get("config", {}).get("providers", []) or []:
        if not isinstance(entry, str):
            continue
        rel = entry[len("file://"):] if entry.startswith("file://") else entry
        path = Path(rel)
        if not path.exists():
            path = Path(base_dir) / rel
        if not path.exists():
            continue
        cfg = parse_provider_yaml(path.read_text(encoding="utf-8"))
        label = cfg.get("label")
        if not label:
            continue
        endpoints[label] = {
            "url": cfg.get("apiBaseUrl"),
            "api_key": cfg.get("apiKey") or "dummy",
            "temperature": cfg.get("temperature"),
            "max_tokens": cfg.get("max_tokens"),
        }

    if override_url:
        for result in eval_json.get("results", {}).get("results", []):
            label = (result.get("provider") or {}).get("label")
            if label and label not in endpoints:
                endpoints[label] = {"url": None, "api_key": "dummy",
                                    "temperature": None, "max_tokens": None}
        for ep in endpoints.values():
            ep["url"] = override_url
    return endpoints


def build_curl(endpoint: dict, model: str, messages_raw: str) -> str:
    """A shell curl that replays one test against the provider's chat endpoint."""
    url = endpoint.get("url")
    if not url:
        return ""
    try:
        messages = json.loads(messages_raw)
        if not isinstance(messages, list):
            raise ValueError
    except (ValueError, TypeError):
        messages = [{"role": "user", "content": messages_raw}]

    body: dict = {"model": model, "messages": messages}
    temperature = endpoint.get("temperature")
    if temperature not in (None, ""):
        try:
            body["temperature"] = float(temperature)
        except (TypeError, ValueError):
            pass
    max_tokens = endpoint.get("max_tokens")
    if max_tokens not in (None, ""):
        try:
            body["max_tokens"] = int(max_tokens)
        except (TypeError, ValueError):
            pass

    chat_url = url.rstrip("/") + "/chat/completions"
    payload = json.dumps(body).replace("'", "'\\''")  # safe inside shell '...'
    api_key = endpoint.get("api_key") or "dummy"
    return (
        f"curl {chat_url} "
        f"-H 'Content-Type: application/json' "
        f"-H 'Authorization: Bearer {api_key}' "
        f"-d '{payload}'"
    )


def build_curls(eval_json: dict, base_dir, override_url: str | None = None) -> dict:
    """Map test description -> curl command, for tests we can build one for."""
    request_info = extract_request_info(eval_json)
    endpoints = resolve_endpoints(eval_json, base_dir, override_url)
    curls: dict = {}
    for desc, info in request_info.items():
        endpoint = endpoints.get(info["provider_label"])
        if not endpoint:
            continue
        curl = build_curl(endpoint, info["model"], info["messages_raw"])
        if curl:
            curls[desc] = curl
    return curls


def errored_tests(eval_json: dict, suites) -> set:
    """Descriptions of tests in the given suites whose grading errored."""
    suites = set(suites)
    out: set = set()
    for result in eval_json.get("results", {}).get("results", []):
        test_case = result.get("testCase") or {}
        meta = test_case.get("metadata") or {}
        if meta.get("suite") not in suites:
            continue
        if _result_errored(result):
            out.add(test_case.get("description") or "<no-description>")
    return out


def _search_term(meta: dict, description: str) -> str:
    """A clean token that isolates one test in the promptfoo UI search box.

    Prefer a stable dataset id (sample_id/id/name); fall back to the full
    description. Single tokens like a sample_id avoid clashing with promptfoo's
    regex-style search (the description can contain `[` / `]`).
    """
    return str(meta.get("sample_id") or meta.get("id") or meta.get("name") or description)


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
        search = _search_term(meta, test_desc)
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
                search=search,
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
    search: str = ""  # promptfoo UI search term for this test


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
                                  b.score, c.score, delta, classify(delta, tolerance),
                                  c.search))
        elif c is not None:
            diffs.append(CellDiff(key, c.suite, c.metric, c.assertion_value,
                                  None, c.score, None, "new", c.search))
        else:
            diffs.append(CellDiff(key, b.suite, b.metric, b.assertion_value,
                                  b.score, None, None, "removed", b.search))
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
/* Neutral alternating banding: all rows of one test share a shade. */
tr.test-a td { background: #f4f4f4; }
tr.test-b td { background: #ffffff; }
.cell-error { color: #b35900; font-weight: 600; }
.copy-btn { margin-left: .4rem; padding: 0; border: none; background: none;
            cursor: pointer; color: #aaa; vertical-align: middle; line-height: 1; }
.copy-btn:hover { color: #1a1a1a; }
.copy-btn.copied { color: #0a7d28; }
"""


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_delta(value) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _ui_href(eval_id, search, base_url) -> str:
    return html.escape(f"{base_url}/eval/{eval_id}?search={quote(search)}", quote=True)


def _score_html(value, eval_id, search, base_url, errored=False) -> str:
    """Cell content, hyperlinked to the test's filtered view in the promptfoo UI.

    A present score links to its run. A missing side (None) renders as ERROR
    (linked, when the test errored in that run) or an em dash otherwise. Links
    only appear when an eval id and search term are available.
    """
    if value is None:
        if not errored:
            return "—"
        if eval_id and search:
            return (f'<a class="cell-error" href="{_ui_href(eval_id, search, base_url)}" '
                    'target="_blank" rel="noopener">ERROR</a>')
        return '<span class="cell-error">ERROR</span>'
    text = f"{value:.2f}"
    if not eval_id or not search:
        return text
    return (f'<a href="{_ui_href(eval_id, search, base_url)}" '
            f'target="_blank" rel="noopener">{text}</a>')


# Small clipboard glyph for the copy-as-curl button.
_CLIPBOARD_SVG = (
    '<svg viewBox="0 0 16 16" width="12" height="12" fill="none" '
    'stroke="currentColor" stroke-width="1.3" aria-hidden="true">'
    '<rect x="4" y="3.2" width="8" height="11" rx="1.4"/>'
    '<path d="M6 3.2v-.7A1.5 1.5 0 0 1 7.5 1h1A1.5 1.5 0 0 1 10 2.5v.7"/></svg>'
)

_COPY_SCRIPT = (
    "<script>"
    "for(const b of document.querySelectorAll('.copy-btn')){"
    "b.addEventListener('click',()=>{"
    "navigator.clipboard.writeText(b.dataset.curl).then(()=>{"
    "b.classList.add('copied');const t=b.title;b.title='Copied!';"
    "setTimeout(()=>{b.classList.remove('copied');b.title=t;},1200);});});}"
    "</script>"
)


def _copy_button(curl: str) -> str:
    return (f'<button type="button" class="copy-btn" title="Copy curl" '
            f'aria-label="Copy curl command" '
            f'data-curl="{html.escape(curl, quote=True)}">{_CLIPBOARD_SVG}</button>')


def _cell_td(value, eval_id, search, base_url, errored, curl) -> str:
    inner = _score_html(value, eval_id, search, base_url, errored)
    if curl and (value is not None or errored):
        inner += _copy_button(curl)
    return f'<td class="num">{inner}</td>'


def _sort_key(diff):
    # Worst regressions first; new/removed (delta None) sort to the bottom.
    return (0, diff.delta) if diff.delta is not None else (1, 0.0)


def _group_sort_key(group):
    # Order test groups by their worst delta; groups with no delta sort last.
    deltas = [g.delta for g in group if g.delta is not None]
    return (0, min(deltas)) if deltas else (1, 0.0)


def render_html(diffs: list, counts: dict, drift, tolerance: float,
                baseline_eval_id: str | None = None,
                candidate_eval_id: str | None = None,
                ui_base_url: str = DEFAULT_UI_BASE_URL,
                baseline_errored: set | None = None,
                candidate_errored: set | None = None,
                baseline_curls: dict | None = None,
                candidate_curls: dict | None = None) -> str:
    """Render a self-contained HTML report. `drift` is (only_base, only_cand).

    When an eval id is supplied for a side, that side's scores link to the
    test's filtered view in the promptfoo UI (`{ui_base_url}/eval/<id>?search=`).
    A missing cell whose test is in `baseline_errored` / `candidate_errored`
    renders a linked ERROR instead of an em dash. When a test has an entry in
    `baseline_curls` / `candidate_curls`, a copy-to-clipboard button is shown
    next to that side's score/ERROR.
    """
    only_base, only_cand = drift
    baseline_errored = baseline_errored or set()
    candidate_errored = candidate_errored or set()
    baseline_curls = baseline_curls or {}
    candidate_curls = candidate_curls or {}
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
    parity_counter = 0  # alternates per test group across the whole report
    for suite in suites:
        # Group this suite's rows by test so a test's assertions stay together,
        # ordering groups by their worst delta (regressions near the top).
        test_groups: dict = {}
        group_order: list = []
        for d in grouped[suite]:
            gkey = (d.key.test, d.key.prompt)
            if gkey not in test_groups:
                test_groups[gkey] = []
                group_order.append(gkey)
            test_groups[gkey].append(d)
        group_order.sort(key=lambda gk: _group_sort_key(test_groups[gk]))

        rows = []
        for gkey in group_order:
            parity = "a" if parity_counter % 2 == 0 else "b"
            parity_counter += 1
            for d in sorted(test_groups[gkey], key=_sort_key):
                rows.append(
                    f'<tr class="status-{d.status} test-{parity}">'
                    f"<td>{html.escape(d.key.test)}</td>"
                    f"<td>{html.escape(d.assertion_value)}</td>"
                    f"<td>{html.escape(d.metric or '')}</td>"
                    f'{_cell_td(d.baseline, baseline_eval_id, d.search, ui_base_url, d.key.test in baseline_errored, baseline_curls.get(d.key.test, ""))}'
                    f'{_cell_td(d.candidate, candidate_eval_id, d.search, ui_base_url, d.key.test in candidate_errored, candidate_curls.get(d.key.test, ""))}'
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
        + _COPY_SCRIPT
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
    parser.add_argument(
        "--ui-base-url",
        default=DEFAULT_UI_BASE_URL,
        help="promptfoo web UI base URL for score links "
        f"(default: {DEFAULT_UI_BASE_URL}).",
    )
    parser.add_argument(
        "--baseline-url",
        default=None,
        help="Override the baseline provider endpoint base URL used for the "
        "copy-as-curl buttons (default: read from the provider YAML in the eval "
        "config).",
    )
    parser.add_argument(
        "--candidate-url",
        default=None,
        help="Override the candidate provider endpoint base URL for copy-as-curl.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suites = args.suite if args.suite else list(DEFAULT_SUITES)

    baseline_json = read_eval_json(args.baseline_json)
    candidate_json = read_eval_json(args.candidate_json)
    baseline_cells = extract_cells(baseline_json, suites)
    candidate_cells = extract_cells(candidate_json, suites)

    diffs = diff_cells(baseline_cells, candidate_cells, args.tolerance)
    counts = summarize(diffs)
    drift = diff_test_keys(baseline_cells, candidate_cells)

    base_dir = Path.cwd()
    args.out.write_text(
        render_html(diffs, counts, drift, args.tolerance,
                    baseline_eval_id=read_eval_id(baseline_json),
                    candidate_eval_id=read_eval_id(candidate_json),
                    ui_base_url=args.ui_base_url,
                    baseline_errored=errored_tests(baseline_json, suites),
                    candidate_errored=errored_tests(candidate_json, suites),
                    baseline_curls=build_curls(baseline_json, base_dir, args.baseline_url),
                    candidate_curls=build_curls(candidate_json, base_dir, args.candidate_url)),
        encoding="utf-8",
    )
    print(
        f"{counts['improved']} improved, {counts['regressed']} regressed, "
        f"{counts['within']} within, {counts['new']} new, "
        f"{counts['removed']} removed -> {args.out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

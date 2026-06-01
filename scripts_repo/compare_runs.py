#!/usr/bin/env python3
"""Compare a baseline and a candidate promptfoo eval.

Extracts one "cell" per assertion per test from each eval JSON, joins them on
(test description, prompt label, assertion text), and writes a self-contained
HTML report grouped by suite. Two kinds of cell are handled:

  * rubric        — a graded 0-1 assertion (llm-rubric or g-eval); the delta
                    is classified against an absolute tolerance band.
  * deterministic — any other assertion (icontains, python, …) with a pass/fail
                    verdict; no delta, classified as a pass/fail transition.

The assertion's promptfoo ``type`` (e.g. ``llm-rubric``, ``g-eval``) is also
captured on every cell and surfaced as its own column in the HTML, so two
variants of the same prompt graded by different judges can be told apart at a
glance.

Every suite present in either run is compared (no allowlist); `--suite` can
still restrict the scope. Tests with no metadata.suite are grouped under
"(no suite)".

Usage:
    compare_runs.py baselines/prod.json results/local/latest.json
    compare_runs.py BASE.json CAND.json --tolerance 0.05 --out report.html \\
        --suite research_rubrics
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_TOLERANCE = 0.05
# Suite heading for tests that carry no metadata.suite.
NO_SUITE = "(no suite)"
# promptfoo web UI base; user is responsible for running the server there.
DEFAULT_UI_BASE_URL = "http://localhost:3000"

# promptfoo ResultFailureReason.ERROR (provider / response level error).
_FAILURE_REASON_ERROR = 2

# Assertion types that produce a graded 0-1 score (the "rubric" Cell kind).
# Anything not in this set is treated as deterministic pass/fail. ``select-best``
# is its own kind, handled separately, so it is not listed here.
_RUBRIC_ASSERTION_TYPES = frozenset({"llm-rubric", "g-eval"})

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
    kind: str  # "rubric" | "deterministic" | "best"
    metric: str | None
    weight: float
    assertion_value: str
    score: float | None = None    # rubric: the 0-1 grade
    passed: bool | None = None    # deterministic: the pass/fail verdict
    search: str = ""  # term that isolates this test in the promptfoo UI search
    # The assertion's promptfoo ``type`` (``llm-rubric``, ``g-eval``, ``python``…),
    # preserved verbatim so the HTML can show it as its own column.
    assertion_type: str = ""


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


def extract_request_info(eval_json: dict, provider_label: str | None = None) -> dict:
    """Map test description -> {model, messages_raw, provider_label}.

    Covers every result (including errored ones), so a curl can be offered for
    cells that have no score. When `provider_label` is given, only that
    provider's results are considered — needed when one unified eval file holds
    both providers and each side wants its own provider's curl.
    """
    info: dict = {}
    for result in eval_json.get("results", {}).get("results", []):
        provider = result.get("provider") or {}
        if provider_label is not None and provider.get("label") != provider_label:
            continue
        test_case = result.get("testCase") or {}
        desc = test_case.get("description") or "<no-description>"
        if desc in info:
            continue
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


def build_curls(eval_json: dict, base_dir, override_url: str | None = None,
                provider_label: str | None = None) -> dict:
    """Map test description -> curl command, for tests we can build one for.

    `provider_label` scopes the curls to one provider (see extract_request_info).
    """
    request_info = extract_request_info(eval_json, provider_label)
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


def errored_tests(eval_json: dict, suites=None, provider_label: str | None = None) -> set:
    """Descriptions of tests whose grading errored.

    When `suites` is given, scope is restricted to those suites; otherwise all
    suites are considered. `provider_label`, when given, restricts to one
    provider's results (for unified single-file runs).
    """
    suites = set(suites) if suites is not None else None
    out: set = set()
    for result in eval_json.get("results", {}).get("results", []):
        if provider_label is not None and (result.get("provider") or {}).get("label") != provider_label:
            continue
        test_case = result.get("testCase") or {}
        meta = test_case.get("metadata") or {}
        if suites is not None and (meta.get("suite") or NO_SUITE) not in suites:
            continue
        if _result_errored(result):
            out.add(test_case.get("description") or "<no-description>")
    return out


def _prompt_text(raw: str) -> str:
    """Human-readable prompt text from a promptfoo ``prompt.raw``.

    ``prompt.raw`` is usually a JSON chat-messages array; return the last user
    message's content. Falls back to the raw string for non-JSON prompts.
    """
    try:
        messages = json.loads(raw)
    except (ValueError, TypeError):
        return raw.strip()
    if isinstance(messages, list):
        for msg in reversed(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content", "")).strip()
    return raw.strip()


def _test_identity(result: dict, test_case: dict) -> str:
    """A stable, human-readable identifier for a test case.

    Prefer the explicit ``description``. When a suite sets none (e.g.
    ``stock_prices``, whose rows share one ``file://`` assertion and the same
    prompt label), fall back to the rendered prompt so distinct tests don't
    collapse onto a single CellKey and overwrite each other. Both eval JSONs
    derive the same identity for the same input, so the baseline/candidate join
    still holds.
    """
    desc = test_case.get("description")
    if desc:
        return desc
    text = _prompt_text((result.get("prompt") or {}).get("raw") or "")
    return text or "<no-description>"


def _search_term(meta: dict, description: str) -> str:
    """A clean token that isolates one test in the promptfoo UI search box.

    Prefer a stable dataset id (sample_id/id/name); fall back to the full
    description. Single tokens like a sample_id avoid clashing with promptfoo's
    regex-style search (the description can contain `[` / `]`).
    """
    return str(meta.get("sample_id") or meta.get("id") or meta.get("name") or description)


def extract_cells(eval_json: dict, suites=None, provider_label: str | None = None) -> dict:
    """Map CellKey -> Cell for every gradable assertion.

    Rubric (llm-rubric) and deterministic (icontains, python, …) assertions are
    both captured. When `suites` is given, scope is restricted to those suites;
    otherwise every suite is included. Tests with no metadata.suite are grouped
    under "(no suite)".

    `provider_label` restricts to one provider's results. This matters for a
    unified run where prod and dev are evaluated in a single eval file: the
    CellKey (test, prompt, assertion) carries no provider, so without this
    filter the two providers' identical cells collide. Pass the prod label for
    the baseline side and the dev label for the candidate side.
    """
    suites = set(suites) if suites is not None else None
    cells: dict = {}
    for result in eval_json.get("results", {}).get("results", []):
        if provider_label is not None and (result.get("provider") or {}).get("label") != provider_label:
            continue
        test_case = result.get("testCase") or {}
        meta = test_case.get("metadata") or {}
        suite = meta.get("suite") or NO_SUITE
        if suites is not None and suite not in suites:
            continue
        test_desc = _test_identity(result, test_case)
        search = _search_term(meta, test_desc)
        prompt_label = (result.get("prompt") or {}).get("label") or "<no-prompt>"
        asserts = test_case.get("assert") or []
        comps = (result.get("gradingResult") or {}).get("componentResults") or []
        seen: dict = {}
        for i, assertion in enumerate(asserts):
            if i >= len(comps):
                continue
            value = assertion.get("value") or f"<assertion-{i}>"
            n = seen.get(value, 0)
            seen[value] = n + 1
            assertion_key = value if n == 0 else f"{value}#{n}"
            key = CellKey(test=test_desc, prompt=prompt_label, assertion=assertion_key)
            atype = assertion.get("type") or ""
            common = dict(
                key=key,
                suite=suite,
                metric=assertion.get("metric"),
                weight=float(assertion.get("weight", 1) or 1),
                assertion_value=value,
                search=search,
                assertion_type=atype,
            )
            if atype in _RUBRIC_ASSERTION_TYPES:
                score = comps[i].get("score")
                cells[key] = Cell(
                    kind="rubric",
                    score=float(score) if score is not None else 0.0,
                    **common,
                )
            elif atype == "select-best":
                # Head-to-head: the winning provider's component passes, the
                # loser's fails. Captured as its own kind so it stays out of the
                # rubric/deterministic tallies and feeds the "best" column.
                comp = comps[i]
                passed = comp.get("pass")
                if passed is None:
                    passed = float(comp.get("score") or 0.0) >= 0.5
                cells[key] = Cell(kind="best", passed=bool(passed), **common)
            else:
                comp = comps[i]
                passed = comp.get("pass")
                if passed is None:
                    passed = float(comp.get("score") or 0.0) >= 0.5
                cells[key] = Cell(kind="deterministic", passed=bool(passed), **common)
    return cells


@dataclass
class CellDiff:
    key: "CellKey"
    suite: str
    kind: str  # "rubric" | "deterministic" | "best"
    metric: str | None
    assertion_value: str
    # rubric: 0-1 score; deterministic: pass/fail bool; absent side: None
    baseline: float | bool | None
    candidate: float | bool | None
    delta: float | None  # rubric only; None for deterministic / missing side
    status: str  # rubric: improved|regressed|within ; det: improved|regressed|same ; new|removed
    search: str = ""  # promptfoo UI search term for this test
    # Carried through from Cell so the HTML can show it as its own column.
    assertion_type: str = ""


def classify(delta: float, tolerance: float) -> str:
    """Three-way verdict for a rubric delta against an absolute tolerance band.

    A move exactly equal to the band is treated as within tolerance.
    """
    if delta > tolerance:
        return "improved"
    if delta < -tolerance:
        return "regressed"
    return "within"


def _classify_deterministic(baseline_passed: bool, candidate_passed: bool) -> str:
    if baseline_passed == candidate_passed:
        return "same"
    return "improved" if candidate_passed else "regressed"


# --- N-provider model ------------------------------------------------------
# The pairwise CellDiff above compares exactly two sides. The model below
# generalizes that to an arbitrary set of providers with one designated
# baseline: one Row per CellKey holding every provider's value, plus a delta of
# each non-baseline provider against the baseline. This is what the multi-column
# report (compare against prod or dev, plus competitors like Venice) is built on.


@dataclass(frozen=True)
class ProviderColumn:
    key: str        # config key; shown as the column header
    label: str      # promptfoo provider label; used to split the unified eval file
    is_baseline: bool


@dataclass
class Row:
    key: "CellKey"
    suite: str
    kind: str                      # "rubric" | "deterministic" | "best"
    metric: str | None
    assertion_value: str
    assertion_type: str
    values: dict                   # provider key -> score (rubric) / pass (det) / None
    deltas: dict                   # non-baseline key -> (other - baseline) | None
    best: str | None               # winning provider key, for kind == "best"
    search: str = ""


def best_winner_among(per_provider: dict) -> str | None:
    """Provider key whose select-best component passed, or None if not exactly one.

    ``per_provider`` maps provider key -> Cell (or None). In an N-way select-best
    exactly one provider's component passes; ambiguity (none or several) is
    reported as undecided (None).
    """
    winners = [k for k, c in per_provider.items() if c is not None and c.passed is True]
    return winners[0] if len(winners) == 1 else None


def build_rows(cells_by_provider: dict, columns: list) -> list:
    """Join per-provider cell maps into one Row per CellKey.

    ``cells_by_provider`` maps provider key -> {CellKey: Cell}; ``columns`` is the
    ordered ProviderColumn list (baseline first). Rubric deltas are computed as
    ``other - baseline`` when both sides have a rubric score; deterministic/best
    rows carry no deltas. A row is emitted for any CellKey present for at least one
    provider (so a test only one provider ran still appears).
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    all_keys: set = set()
    for m in cells_by_provider.values():
        all_keys |= set(m)
    rows: list = []
    for key in sorted(all_keys, key=lambda k: (k.test, k.prompt, k.assertion)):
        per_provider = {c.key: cells_by_provider.get(c.key, {}).get(key) for c in columns}
        present = next((c for c in per_provider.values() if c is not None), None)
        if present is None:
            continue
        kind = present.kind
        values: dict = {}
        for ckey, cell in per_provider.items():
            if cell is None:
                values[ckey] = None
            elif kind == "rubric":
                values[ckey] = cell.score
            else:  # deterministic or best
                values[ckey] = cell.passed
        deltas: dict = {}
        if kind == "rubric":
            base_cell = per_provider.get(baseline)
            for o in others:
                oc = per_provider.get(o)
                deltas[o] = (
                    oc.score - base_cell.score
                    if (oc is not None and base_cell is not None)
                    else None
                )
        best = best_winner_among(per_provider) if kind == "best" else None
        rows.append(
            Row(
                key=key,
                suite=present.suite,
                kind=kind,
                metric=present.metric,
                assertion_value=present.assertion_value,
                assertion_type=present.assertion_type,
                values=values,
                deltas=deltas,
                best=best,
                search=present.search,
            )
        )
    return rows


def summarize_rubric_table(rows: list, columns: list, tolerance: float) -> dict:
    """Per-non-baseline-provider rubric tally vs baseline.

    Returns ``{provider_key: {improved, regressed, within, new, removed}}``. A cell
    present for the other provider but not the baseline counts as ``new`` (and the
    reverse as ``removed``); both present are classified by the delta band.
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    out = {
        o: {"improved": 0, "regressed": 0, "within": 0, "new": 0, "removed": 0}
        for o in others
    }
    for row in rows:
        if row.kind != "rubric":
            continue
        b = row.values.get(baseline)
        for o in others:
            v = row.values.get(o)
            if b is None and v is not None:
                out[o]["new"] += 1
            elif b is not None and v is None:
                out[o]["removed"] += 1
            elif b is not None and v is not None:
                out[o][classify(v - b, tolerance)] += 1
    return out


def summarize_deterministic_table(rows: list, columns: list) -> dict:
    """Per-non-baseline-provider deterministic tally vs baseline.

    ``new_passes`` / ``new_fails`` count pass/fail transitions relative to the
    baseline among tests both ran; ``total_passes`` / ``total_fails`` count the
    other provider's own verdicts.
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    out = {
        o: {"new_passes": 0, "new_fails": 0, "total_passes": 0, "total_fails": 0}
        for o in others
    }
    for row in rows:
        if row.kind != "deterministic":
            continue
        b = row.values.get(baseline)
        for o in others:
            v = row.values.get(o)
            if v is True:
                out[o]["total_passes"] += 1
            elif v is False:
                out[o]["total_fails"] += 1
            if b is not None and v is not None and b != v:
                out[o]["new_passes" if v else "new_fails"] += 1
    return out


def summarize_best_table(rows: list, columns: list) -> dict:
    """Wins per provider key across best rows (+ ``undecided``)."""
    out = {c.key: 0 for c in columns}
    out["undecided"] = 0
    for row in rows:
        if row.kind != "best":
            continue
        out[row.best if row.best in out else "undecided"] += 1
    return out


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
        src = c or b
        kind = src.kind
        if b and c:
            if kind == "rubric":
                delta = c.score - b.score
                diffs.append(CellDiff(key, src.suite, kind, src.metric,
                                      src.assertion_value, b.score, c.score, delta,
                                      classify(delta, tolerance), src.search,
                                      src.assertion_type))
            elif kind == "best":
                # No pass/fail transition to classify — the winner is read off
                # the passing side (see best_winner); status is a fixed marker.
                diffs.append(CellDiff(key, src.suite, kind, src.metric,
                                      src.assertion_value, b.passed, c.passed, None,
                                      "best", src.search, src.assertion_type))
            else:
                status = _classify_deterministic(b.passed, c.passed)
                diffs.append(CellDiff(key, src.suite, kind, src.metric,
                                      src.assertion_value, b.passed, c.passed, None,
                                      status, src.search, src.assertion_type))
        elif c is not None:
            value = c.score if kind == "rubric" else c.passed
            diffs.append(CellDiff(key, c.suite, kind, c.metric, c.assertion_value,
                                  None, value, None, "new", c.search,
                                  c.assertion_type))
        else:
            value = b.score if kind == "rubric" else b.passed
            diffs.append(CellDiff(key, b.suite, kind, b.metric, b.assertion_value,
                                  value, None, None, "removed", b.search,
                                  b.assertion_type))
    return diffs


def summarize(diffs: list) -> dict:
    """Count rubric diffs by status (deterministic diffs are ignored)."""
    counts = {"improved": 0, "regressed": 0, "within": 0, "new": 0, "removed": 0}
    for d in diffs:
        if d.kind == "rubric":
            counts[d.status] += 1
    return counts


def summarize_deterministic(diffs: list) -> dict:
    """Summarize deterministic diffs: pass/fail transitions and candidate totals.

    new_passes / new_fails count fail->pass / pass->fail transitions among tests
    present in both runs; total_passes / total_fails count the candidate-side
    verdicts (so brand-new tests count, removed ones do not).
    """
    out = {"new_passes": 0, "new_fails": 0, "total_passes": 0, "total_fails": 0}
    for d in diffs:
        if d.kind != "deterministic":
            continue
        if d.status == "improved":
            out["new_passes"] += 1
        elif d.status == "regressed":
            out["new_fails"] += 1
        if d.candidate is True:
            out["total_passes"] += 1
        elif d.candidate is False:
            out["total_fails"] += 1
    return out


def best_winner(diff: "CellDiff") -> str | None:
    """Which side won a ``select-best`` head-to-head, or None if undetermined.

    Exactly one side's component passes when both ran. ``"prod"`` means the
    baseline side won, ``"candidate"`` the candidate side (run_comparison always
    passes prod as the baseline). None when a side is missing/errored or neither
    is a clean winner.
    """
    if diff.kind != "best":
        return None
    # Both sides must have run (a missing/errored side is None, not False).
    if diff.baseline is True and diff.candidate is False:
        return "prod"
    if diff.candidate is True and diff.baseline is False:
        return "candidate"
    return None


def summarize_best(diffs: list) -> dict:
    """Tally select-best winners across the run (best diffs only)."""
    out = {"prod": 0, "candidate": 0, "undecided": 0}
    for d in diffs:
        if d.kind != "best":
            continue
        out[best_winner(d) or "undecided"] += 1
    return out


def diff_test_keys(baseline_cells: dict, candidate_cells: dict):
    """Sorted (baseline-only, candidate-only) test descriptions."""
    b_tests = {k.test for k in baseline_cells}
    c_tests = {k.test for k in candidate_cells}
    return sorted(b_tests - c_tests), sorted(c_tests - b_tests)


_CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 2rem; color: #1a1a1a; }
.summary { font-size: 1.1rem; margin-bottom: 1rem; }
.summary .agg-line, .suite-summary .sum-line { margin: .15rem 0; }
.summary .sum-label, .suite-summary .sum-label { font-weight: 600; color: #555; }
.suite-summary { margin: .25rem 0 .75rem; color: #333; font-size: .95rem; }
.evals { margin-bottom: 1rem; font-size: .95rem; color: #333; }
.evals .evlabel { display: inline-block; min-width: 5.5rem; font-weight: 600; color: #555; }
.evals .muted { color: #999; }
.drift { background: #fff3cd; border: 1px solid #ffe69c; padding: .75rem 1rem;
         border-radius: 6px; margin-bottom: 1rem; }
table { border-collapse: collapse; width: 100%; margin-bottom: 2rem; }
th, td { text-align: left; padding: .4rem .6rem; border-bottom: 1px solid #eee; }
th { background: #fafafa; }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
h2 { margin-top: 1.5rem; }
.status-improved td.delta { color: #0a7d28; font-weight: 600; }
.status-regressed td.delta { color: #c0341d; font-weight: 600; }
.status-within td.delta, .status-same td.delta { color: #888; }
.status-new td, .status-removed td { color: #555; font-style: italic; }
.verdict.pass { color: #0a7d28; font-weight: 600; }
.verdict.fail { color: #c0341d; font-weight: 600; }
.best-winner { font-weight: 600; color: #1a4fa0; }
/* Neutral alternating banding: all rows of one test share a shade. */
tr.test-a td { background: #f4f4f4; }
tr.test-b td { background: #ffffff; }
.cell-error { color: #b35900; font-weight: 600; }
.copy-btn { margin-left: .4rem; padding: 0; border: none; background: none;
            cursor: pointer; color: #aaa; vertical-align: middle; line-height: 1; }
.copy-btn:hover { color: #1a1a1a; }
.copy-btn.copied { color: #0a7d28; }
/* N-provider report: deltas are coloured by their own band (there is no status
   column), and the summary tables size to content rather than full width. */
td.delta-improved { color: #0a7d28; font-weight: 600; }
td.delta-regressed { color: #c0341d; font-weight: 600; }
td.delta-within { color: #888; }
.summary table, .suite-summary table { width: auto; margin: .25rem 1rem .75rem 0;
            display: inline-table; vertical-align: top; }
.summary h4, .suite-summary h4 { margin: .6rem 0 .15rem; font-size: .9rem;
            color: #555; }
.col-baseline { font-weight: 600; }
"""


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.2f}"


def _fmt_delta(value) -> str:
    return "—" if value is None else f"{value:+.2f}"


def _ui_href(eval_id, search, base_url) -> str:
    # The promptfoo UI prefills its search box from `?search=`, which filters the
    # results table down and makes navigation worse. Link to the bare eval view.
    return html.escape(f"{base_url}/eval/{eval_id}", quote=True)


def _value_html(kind, value, eval_id, search, base_url, errored=False) -> str:
    """Cell content, hyperlinked to the test's filtered view in the promptfoo UI.

    A present value links to its run: a rubric score renders as `0.50`, a
    deterministic verdict as a `pass`/`fail` span. A missing side (None) renders
    as ERROR (linked, when the test errored in that run) or an em dash. Links
    only appear when an eval id and search term are available.
    """
    if value is None:
        if not errored:
            return "—"
        if eval_id and search:
            return (f'<a class="cell-error" href="{_ui_href(eval_id, search, base_url)}" '
                    'target="_blank" rel="noopener">ERROR</a>')
        return '<span class="cell-error">ERROR</span>'
    if kind == "deterministic":
        text = "pass" if value else "fail"
        cls = "verdict pass" if value else "verdict fail"
    else:
        text = f"{value:.2f}"
        cls = ""
    if not eval_id or not search:
        return f'<span class="{cls}">{text}</span>' if cls else text
    cls_attr = f'class="{cls}" ' if cls else ""
    return (f'<a {cls_attr}href="{_ui_href(eval_id, search, base_url)}" '
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


def _eval_header(baseline_eval_id, candidate_eval_id, base_url,
                 config_path: str | None = None,
                 system_prompt_path: str | None = None) -> str:
    """Header naming the two evals (and optionally the run's config/system prompt).

    Eval IDs link to the promptfoo UI view; ``config_path`` and
    ``system_prompt_path``, when given, are rendered as ``file://`` links to the
    local files so a report opened in the browser can jump straight to the
    config / prompt that produced it. Each is shown as its own labelled row,
    skipped entirely when not supplied (the bare ``compare_runs.py`` CLI path
    therefore stays unchanged).
    """

    def eval_row(label, eval_id):
        if eval_id:
            href = html.escape(f"{base_url}/eval/{eval_id}", quote=True)
            value = (f'<a href="{href}" target="_blank" rel="noopener">'
                     f"{html.escape(eval_id)}</a>")
        else:
            value = '<span class="muted">(unknown)</span>'
        return f'<div><span class="evlabel">{label}</span> {value}</div>'

    def file_row(label, path):
        # Resolve so a relative path typed on the CLI still produces a working
        # file:// link; the report may be opened from anywhere.
        absolute = str(Path(path).resolve())
        href = html.escape(f"file://{absolute}", quote=True)
        return (f'<div><span class="evlabel">{label}</span> '
                f'<a href="{href}" target="_blank" rel="noopener">'
                f'{html.escape(absolute)}</a></div>')

    parts = [eval_row("Baseline", baseline_eval_id),
             eval_row("Candidate", candidate_eval_id)]
    if config_path:
        parts.append(file_row("Config", config_path))
    if system_prompt_path:
        parts.append(file_row("System prompt", system_prompt_path))
    return f'<div class="evals">{"".join(parts)}</div>'


def _cell_td(kind, value, eval_id, search, base_url, errored, curl) -> str:
    inner = _value_html(kind, value, eval_id, search, base_url, errored)
    if curl and (value is not None or errored):
        inner += _copy_button(curl)
    return f'<td class="num">{inner}</td>'


# Severity order for rows with no numeric delta (deterministic / new / removed).
_STATUS_ORDER = {"regressed": 0, "new": 1, "removed": 2, "improved": 3,
                 "same": 4, "within": 5}


def _sort_key(diff):
    # Rubric rows: worst regressions first. Otherwise order by status severity.
    if diff.delta is not None:
        return (0, diff.delta, 0)
    return (1, 0.0, _STATUS_ORDER.get(diff.status, 9))


def _group_sort_key(group):
    # Order test groups by their worst rubric delta; delta-less groups
    # (deterministic) sort after, with any regression first.
    deltas = [g.delta for g in group if g.delta is not None]
    if deltas:
        return (0, min(deltas), 0)
    has_regression = any(g.status == "regressed" for g in group)
    return (1, 0.0, 0 if has_regression else 1)


def _rubric_summary_line(counts: dict, tolerance: float) -> str:
    return (
        '<div class="sum-line"><span class="sum-label">Rubric Tests:</span> '
        f"{counts['improved']} improved &middot; {counts['regressed']} regressed "
        f"&middot; {counts['within']} within &plusmn;{tolerance:g} &middot; "
        f"{counts['new']} new &middot; {counts['removed']} removed</div>"
    )


def _deterministic_summary_line(counts: dict) -> str:
    return (
        '<div class="sum-line"><span class="sum-label">Deterministic Tests:</span> '
        f"{counts['new_passes']} new passes &middot; {counts['new_fails']} new fails "
        f"&middot; {counts['total_passes']} total passes &middot; "
        f"{counts['total_fails']} total fails</div>"
    )


def _best_summary_line(counts: dict) -> str:
    return (
        '<div class="sum-line"><span class="sum-label">Best (head-to-head):</span> '
        f"prod won {counts['prod']} &middot; candidate won {counts['candidate']} "
        f"&middot; {counts['undecided']} undecided</div>"
    )


def _summary_block(diffs: list, tolerance: float, css_class: str) -> str:
    """Rubric, deterministic and/or best summary lines for a set of diffs.

    A kind's line is shown only when that kind has at least one diff.
    """
    lines = []
    if any(d.kind == "rubric" for d in diffs):
        lines.append(_rubric_summary_line(summarize(diffs), tolerance))
    if any(d.kind == "deterministic" for d in diffs):
        lines.append(_deterministic_summary_line(summarize_deterministic(diffs)))
    if any(d.kind == "best" for d in diffs):
        lines.append(_best_summary_line(summarize_best(diffs)))
    return f'<div class="{css_class}">{"".join(lines)}</div>'


def render_html(diffs: list, drift, tolerance: float,
                baseline_eval_id: str | None = None,
                candidate_eval_id: str | None = None,
                ui_base_url: str = DEFAULT_UI_BASE_URL,
                baseline_errored: set | None = None,
                candidate_errored: set | None = None,
                baseline_curls: dict | None = None,
                candidate_curls: dict | None = None,
                config_path: str | None = None,
                system_prompt_path: str | None = None) -> str:
    """Render a self-contained HTML report. `drift` is (only_base, only_cand).

    An aggregate summary (rubric and/or deterministic) is shown at the top, and
    each suite carries its own summary above its table. When an eval id is
    supplied for a side, that side's cells link to the test's filtered view in
    the promptfoo UI (`{ui_base_url}/eval/<id>?search=`). A missing cell whose
    test is in `baseline_errored` / `candidate_errored` renders a linked ERROR
    instead of an em dash. When a test has an entry in `baseline_curls` /
    `candidate_curls`, a copy-to-clipboard button is shown next to that side's
    value/ERROR.
    """
    only_base, only_cand = drift
    baseline_errored = baseline_errored or set()
    candidate_errored = candidate_errored or set()
    baseline_curls = baseline_curls or {}
    candidate_curls = candidate_curls or {}
    aggregate = _summary_block(diffs, tolerance, "summary")

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
                if d.kind == "best":
                    # The head-to-head verdict is a single per-test row: the
                    # type column carries "select-best" (so the assertion column
                    # stays clean) and the score columns are dashes.
                    winner = best_winner(d) or "—"
                    rows.append(
                        f'<tr class="status-best test-{parity}">'
                        f"<td>{html.escape(d.key.test)}</td>"
                        f"<td>{html.escape(d.assertion_type or 'select-best')}</td>"
                        "<td></td>"
                        "<td></td>"
                        '<td class="num">—</td>'
                        '<td class="num">—</td>'
                        '<td class="num delta">—</td>'
                        "<td></td>"
                        f'<td class="best-winner">{winner}</td>'
                        "</tr>"
                    )
                    continue
                rows.append(
                    f'<tr class="status-{d.status} test-{parity}">'
                    f"<td>{html.escape(d.key.test)}</td>"
                    f"<td>{html.escape(d.assertion_type or '')}</td>"
                    f"<td>{html.escape(d.assertion_value)}</td>"
                    f"<td>{html.escape(d.metric or '')}</td>"
                    f'{_cell_td(d.kind, d.baseline, baseline_eval_id, d.search, ui_base_url, d.key.test in baseline_errored, baseline_curls.get(d.key.test, ""))}'
                    f'{_cell_td(d.kind, d.candidate, candidate_eval_id, d.search, ui_base_url, d.key.test in candidate_errored, candidate_curls.get(d.key.test, ""))}'
                    f'<td class="num delta">{_fmt_delta(d.delta)}</td>'
                    f"<td>{d.status}</td>"
                    "<td>—</td>"
                    "</tr>"
                )
        sections.append(
            f"<h2>{html.escape(suite)}</h2>"
            + _summary_block(grouped[suite], tolerance, "suite-summary")
            + "<table><thead><tr>"
            "<th>test</th><th>assertion type</th><th>assertion</th><th>metric</th>"
            "<th>baseline</th><th>candidate</th><th>&Delta;</th><th>status</th>"
            "<th>best</th>"
            "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Provider comparison</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Provider comparison</h1>"
        + _eval_header(baseline_eval_id, candidate_eval_id, ui_base_url,
                       config_path=config_path,
                       system_prompt_path=system_prompt_path)
        + aggregate
        + f"{drift_html}"
        + "".join(sections)
        + _COPY_SCRIPT
        + "</body></html>"
    )


# --- N-provider report -----------------------------------------------------


def parse_provider_col_args(baseline: str, others: list) -> list:
    """Turn ``key=label`` CLI args into ordered ProviderColumns (baseline first)."""

    def split(s):
        key, _, label = s.partition("=")
        return key, label

    bkey, blabel = split(baseline)
    cols = [ProviderColumn(bkey, blabel, is_baseline=True)]
    for o in others:
        k, lbl = split(o)
        cols.append(ProviderColumn(k, lbl, is_baseline=False))
    return cols


def _drift_n(cells_by_provider: dict, columns: list):
    """Drift between the baseline and the union of the other providers.

    Returns ``(missing_from_others, only_in_others)``: baseline tests no other
    provider ran, and tests some other provider ran but the baseline did not.
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    base_tests = {k.test for k in cells_by_provider.get(baseline, {})}
    other_tests: set = set()
    for o in others:
        other_tests |= {k.test for k in cells_by_provider.get(o, {})}
    return sorted(base_tests - other_tests), sorted(other_tests - base_tests)


def _eval_header_n(columns, eval_ids, base_url, config_path=None,
                   system_prompt_path=None) -> str:
    """Header naming each provider column's eval (and the run's config/prompt)."""

    def prov_row(col):
        eval_id = (eval_ids or {}).get(col.key)
        label = col.key + (" (baseline)" if col.is_baseline else "")
        if eval_id:
            href = html.escape(f"{base_url}/eval/{eval_id}", quote=True)
            value = (f'<a href="{href}" target="_blank" rel="noopener">'
                     f"{html.escape(eval_id)}</a>")
        else:
            value = '<span class="muted">(unknown)</span>'
        return f'<div><span class="evlabel">{html.escape(label)}</span> {value}</div>'

    def file_row(label, path):
        absolute = str(Path(path).resolve())
        href = html.escape(f"file://{absolute}", quote=True)
        return (f'<div><span class="evlabel">{label}</span> '
                f'<a href="{href}" target="_blank" rel="noopener">'
                f'{html.escape(absolute)}</a></div>')

    parts = [prov_row(c) for c in columns]
    if config_path:
        parts.append(file_row("Config", config_path))
    if system_prompt_path:
        parts.append(file_row("System prompt", system_prompt_path))
    return f'<div class="evals">{"".join(parts)}</div>'


def _delta_td(delta, tolerance) -> str:
    """A delta cell coloured by its band (improved/regressed/within)."""
    if delta is None:
        return '<td class="num delta">—</td>'
    cls = classify(delta, tolerance)  # improved | regressed | within
    return f'<td class="num delta delta-{cls}">{_fmt_delta(delta)}</td>'


def _summary_table(title, header_cells, body_rows) -> str:
    return (f"<h4>{title}</h4><table><thead><tr>{header_cells}</tr></thead>"
            f"<tbody>{body_rows}</tbody></table>")


def _summary_tables_html(rows, columns, tolerance, css_class) -> str:
    """Per-non-baseline-provider rubric / deterministic / best summary tables.

    A kind's table appears only when at least one row of that kind is present.
    """
    baseline = next(c.key for c in columns if c.is_baseline)
    others = [c.key for c in columns if not c.is_baseline]
    blocks = []

    if any(r.kind == "rubric" for r in rows):
        t = summarize_rubric_table(rows, columns, tolerance)
        head = (f"<th>vs {html.escape(baseline)}</th><th>improved</th>"
                f"<th>regressed</th><th>within &plusmn;{tolerance:g}</th>"
                "<th>new</th><th>removed</th>")
        body = "".join(
            f"<tr><td>{html.escape(o)}</td>"
            f"<td class='num'>{t[o]['improved']}</td>"
            f"<td class='num'>{t[o]['regressed']}</td>"
            f"<td class='num'>{t[o]['within']}</td>"
            f"<td class='num'>{t[o]['new']}</td>"
            f"<td class='num'>{t[o]['removed']}</td></tr>"
            for o in others)
        blocks.append(_summary_table("Rubric", head, body))

    if any(r.kind == "deterministic" for r in rows):
        t = summarize_deterministic_table(rows, columns)
        head = (f"<th>vs {html.escape(baseline)}</th><th>new passes</th>"
                "<th>new fails</th><th>total passes</th><th>total fails</th>")
        body = "".join(
            f"<tr><td>{html.escape(o)}</td>"
            f"<td class='num'>{t[o]['new_passes']}</td>"
            f"<td class='num'>{t[o]['new_fails']}</td>"
            f"<td class='num'>{t[o]['total_passes']}</td>"
            f"<td class='num'>{t[o]['total_fails']}</td></tr>"
            for o in others)
        blocks.append(_summary_table("Deterministic", head, body))

    if any(r.kind == "best" for r in rows):
        t = summarize_best_table(rows, columns)
        head = "".join(f"<th>{html.escape(c.key)}</th>" for c in columns) \
            + "<th>undecided</th>"
        body = ("<tr>"
                + "".join(f"<td class='num'>{t[c.key]}</td>" for c in columns)
                + f"<td class='num'>{t['undecided']}</td></tr>")
        blocks.append(_summary_table("Best (head-to-head)", head, body))

    return f'<div class="{css_class}">{"".join(blocks)}</div>'


def _row_sort_key(row, others):
    # Worst (most negative) delta across non-baseline providers first; rows with
    # no numeric delta (deterministic/best) sort after.
    ds = [row.deltas[o] for o in others if row.deltas.get(o) is not None]
    return (0, min(ds)) if ds else (1, 0.0)


def render_html_n(rows, columns, drift, tolerance,
                  eval_ids=None, ui_base_url=DEFAULT_UI_BASE_URL,
                  errored=None, curls=None, config_path=None,
                  system_prompt_path=None) -> str:
    """Render the N-provider HTML report.

    Columns: test, assertion type, assertion, metric, one value column per
    provider (baseline first, tagged), one delta column per non-baseline provider
    (other - baseline; rubric only), and an N-way ``best`` winner. There is no
    ``status`` column. ``eval_ids`` / ``errored`` / ``curls`` are keyed by provider
    key; in a unified run every provider shares one eval id. ``drift`` is
    ``(missing_from_others, only_in_others)`` from :func:`_drift_n`.
    """
    eval_ids = eval_ids or {}
    errored = errored or {}
    curls = curls or {}
    others = [c for c in columns if not c.is_baseline]

    aggregate = _summary_tables_html(rows, columns, tolerance, "summary")

    missing, extra = drift
    drift_html = ""
    if missing or extra:
        parts = []
        if missing:
            parts.append("only the baseline ran: "
                         + ", ".join(html.escape(t) for t in missing))
        if extra:
            parts.append("baseline did not run: "
                         + ", ".join(html.escape(t) for t in extra))
        drift_html = f'<div class="drift">⚠ config drift — {"; ".join(parts)}</div>'

    def value_cell(col, row):
        return _cell_td(
            row.kind, row.values.get(col.key), eval_ids.get(col.key), row.search,
            ui_base_url, row.key.test in errored.get(col.key, set()),
            curls.get(col.key, {}).get(row.key.test, ""),
        )

    # Dynamic column headers: a value column per provider, then a delta per other.
    value_headers = "".join(
        f'<th class="{"col-baseline" if c.is_baseline else ""}">'
        f'{html.escape(c.key)}{" (baseline)" if c.is_baseline else ""}</th>'
        for c in columns)
    delta_headers = "".join(f"<th>&Delta; {html.escape(c.key)}</th>" for c in others)
    thead = ("<th>test</th><th>assertion type</th><th>assertion</th><th>metric</th>"
             + value_headers + delta_headers + "<th>best</th>")
    n_value = len(columns)
    n_delta = len(others)

    # Group rows by suite (first-seen order), then by test.
    suites: list = []
    grouped: dict = {}
    for r in rows:
        if r.suite not in grouped:
            grouped[r.suite] = []
            suites.append(r.suite)
        grouped[r.suite].append(r)

    sections = []
    parity_counter = 0
    for suite in suites:
        test_groups: dict = {}
        group_order: list = []
        for r in grouped[suite]:
            gkey = (r.key.test, r.key.prompt)
            if gkey not in test_groups:
                test_groups[gkey] = []
                group_order.append(gkey)
            test_groups[gkey].append(r)
        group_order.sort(
            key=lambda gk: min(
                (_row_sort_key(r, [c.key for c in others]) for r in test_groups[gk]),
                default=(1, 0.0),
            )
        )

        body_rows = []
        for gkey in group_order:
            parity = "a" if parity_counter % 2 == 0 else "b"
            parity_counter += 1
            for r in sorted(test_groups[gkey],
                            key=lambda r: _row_sort_key(r, [c.key for c in others])):
                if r.kind == "best":
                    # Single per-test head-to-head row: type column carries the
                    # assertion type, value/delta columns are dashes, best names
                    # the winning provider.
                    winner = r.best or "—"
                    body_rows.append(
                        f'<tr class="test-{parity}">'
                        f"<td>{html.escape(r.key.test)}</td>"
                        f"<td>{html.escape(r.assertion_type or 'select-best')}</td>"
                        "<td></td><td></td>"
                        + '<td class="num">—</td>' * n_value
                        + '<td class="num delta">—</td>' * n_delta
                        + f'<td class="best-winner">{html.escape(winner)}</td>'
                        "</tr>"
                    )
                    continue
                value_tds = "".join(value_cell(c, r) for c in columns)
                delta_tds = "".join(
                    _delta_td(r.deltas.get(c.key), tolerance) for c in others
                )
                body_rows.append(
                    f'<tr class="test-{parity}">'
                    f"<td>{html.escape(r.key.test)}</td>"
                    f"<td>{html.escape(r.assertion_type or '')}</td>"
                    f"<td>{html.escape(r.assertion_value)}</td>"
                    f"<td>{html.escape(r.metric or '')}</td>"
                    + value_tds + delta_tds
                    + '<td class="best-winner">—</td>'
                    "</tr>"
                )
        sections.append(
            f"<h2>{html.escape(suite)}</h2>"
            + _summary_tables_html(grouped[suite], columns, tolerance, "suite-summary")
            + f"<table><thead><tr>{thead}</tr></thead><tbody>"
            + "".join(body_rows) + "</tbody></table>"
        )

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Provider comparison</title>"
        f"<style>{_CSS}</style></head><body>"
        "<h1>Provider comparison</h1>"
        + _eval_header_n(columns, eval_ids, ui_base_url, config_path=config_path,
                         system_prompt_path=system_prompt_path)
        + aggregate
        + drift_html
        + "".join(sections)
        + _COPY_SCRIPT
        + "</body></html>"
    )


# --- raw-response CSV export -----------------------------------------------
# A spreadsheet of the models' raw answers (no scores, no verdicts), one row per
# test: column 1 the rendered prompt, then one column per provider. For eyeballing
# how providers actually responded, side by side.

# Reasoning delimiter, kept in sync with hooks/strip_before_triple_newline.py —
# that hook is the source of truth applied to outputs before the live tests
# assert. Duplicated here (rather than imported) because the hook lives outside
# this package and importing it would need path manipulation.
_REASONING_DELIMITER = "\n\n\n"


def strip_reasoning(output):
    """Drop a reasoning prefix from a model output.

    Mirrors hooks/strip_before_triple_newline.py: everything up to and including
    the first ``\\n\\n\\n`` is reasoning; the rest is the final answer. Non-string
    outputs (e.g. an auto-parsed json_schema dict) are returned unchanged.
    """
    if not isinstance(output, str):
        return output
    idx = output.find(_REASONING_DELIMITER)
    if idx == -1:
        return output
    return output[idx + len(_REASONING_DELIMITER):]


def extract_responses(eval_json: dict, provider_label: str | None = None) -> dict:
    """Map test identity -> {prompt, output, suite} for one provider.

    ``output`` is the reasoning-stripped final answer; ``prompt`` is the rendered
    user message (see :func:`_prompt_text`); ``suite`` groups the row. Scoped to
    ``provider_label`` for the unified eval file; the first result per identity
    wins (consistent with :func:`extract_request_info` / :func:`extract_cells`).
    """
    out: dict = {}
    for result in eval_json.get("results", {}).get("results", []):
        provider = result.get("provider") or {}
        if provider_label is not None and provider.get("label") != provider_label:
            continue
        test_case = result.get("testCase") or {}
        identity = _test_identity(result, test_case)
        if identity in out:
            continue
        meta = test_case.get("metadata") or {}
        raw_output = (result.get("response") or {}).get("output")
        out[identity] = {
            "prompt": _prompt_text((result.get("prompt") or {}).get("raw") or ""),
            "output": strip_reasoning(raw_output) if raw_output is not None else "",
            "suite": meta.get("suite") or NO_SUITE,
        }
    return out


def _csv_cell(value) -> str:
    """Coerce a response value to a CSV cell: None -> '', non-str -> str()."""
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def write_response_csv(per_provider: dict, ordered_keys: list, out_path) -> int:
    """Write the response CSV from already-extracted per-provider maps.

    ``per_provider`` maps provider key -> {identity: {prompt, output, suite}};
    ``ordered_keys`` is the column order (baseline first). One row per test
    identity (the union across providers), ordered by ``(suite, identity)``; the
    prompt/suite are taken from the first column that ran the test. Returns the
    number of data rows written.
    """
    meta: dict = {}  # identity -> (suite, prompt), first listed column wins
    for key in ordered_keys:
        for identity, rec in per_provider.get(key, {}).items():
            meta.setdefault(identity, (rec.get("suite") or NO_SUITE, rec.get("prompt", "")))
    order = sorted(meta, key=lambda i: (meta[i][0], i))

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt"] + list(ordered_keys))
        for identity in order:
            _, prompt = meta[identity]
            row = [prompt]
            for key in ordered_keys:
                rec = per_provider.get(key, {}).get(identity)
                row.append(_csv_cell(rec.get("output") if rec else ""))
            writer.writerow(row)
    return len(order)


def build_response_csv(eval_json: dict, columns: list, out_path) -> int:
    """Write a response CSV from one unified eval file split by ``columns``.

    Each :class:`ProviderColumn` is shown under its config key; its results are
    selected by the provider label. Baseline first. Returns the row count.
    """
    per_provider = {c.key: extract_responses(eval_json, c.label) for c in columns}
    return write_response_csv(per_provider, [c.key for c in columns], out_path)


# --- raw-response HTML table -----------------------------------------------
# The same per-test / per-provider response grid as the CSV, but as a
# self-contained interactive HTML table: word-wrapped cells (row height
# auto-adjusts to content), columns and rows the reader can drag-resize, and a
# layout that fills and reflows with the browser window. Built for eyeballing
# long answers side by side — spreadsheets mangle the embedded newlines/commas.

_RESPONSES_CSS = """
html, body { margin: 0; height: 100%; }
body { font-family: -apple-system, system-ui, sans-serif; color: #1a1a1a; }
.wrap { padding: 1rem; box-sizing: border-box; }
h1 { font-size: 1.1rem; margin: 0 0 .5rem; }
.hint { color: #666; font-size: .85rem; margin: 0 0 .75rem; }
/* width:100% + fixed layout => the table fills and reflows with the window;
   column widths come from the <col> elements. */
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
th, td {
  border: 1px solid #d0d0d0; padding: .4rem .55rem; vertical-align: top;
  /* word wrapping by default; row height grows to fit the content. */
  white-space: normal; overflow-wrap: anywhere; word-break: break-word;
  overflow: hidden; position: relative;
}
th { background: #f4f4f4; text-align: left; position: sticky; top: 0; z-index: 1; }
td { font-size: .9rem; }
td:first-child { color: #333; font-weight: 500; }
/* Drag handles: a thin strip on a cell's right edge (column) / bottom edge (row). */
.col-resize { position: absolute; top: 0; right: -3px; width: 7px; height: 100%;
  cursor: col-resize; user-select: none; z-index: 3; }
.row-resize { position: absolute; left: 0; bottom: -3px; width: 100%; height: 7px;
  cursor: row-resize; user-select: none; z-index: 3; }
/* Rendered-markdown cells: compact margins so a cell stays tight, headings sized
   down to fit, code/quote/rule styled lightly. */
td .md > :first-child { margin-top: 0; }
td .md > :last-child { margin-bottom: 0; }
td .md p { margin: .4em 0; }
td .md h1, td .md h2, td .md h3, td .md h4, td .md h5, td .md h6 {
  margin: .55em 0 .3em; font-size: 1em; font-weight: 700; }
td .md ul, td .md ol { margin: .3em 0; padding-left: 1.3em; }
td .md li { margin: .15em 0; }
td .md a { color: #1a4fa0; }
td .md code { background: #f0f0f0; padding: 0 .25em; border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .92em; }
td .md pre { background: #f0f0f0; padding: .5em .6em; border-radius: 4px;
  overflow: auto; white-space: pre-wrap; }
td .md pre code { background: none; padding: 0; }
td .md blockquote { margin: .4em 0; padding-left: .7em; color: #555;
  border-left: 3px solid #ddd; }
td .md hr { border: 0; border-top: 1px solid #ccc; margin: .6em 0; }
"""

# Vanilla-JS drag handlers. Column handles resize the matching <col> (works under
# table-layout:fixed); row handles set an explicit height on the <tr>. Both clamp
# to a small minimum so a column/row can't be dragged away entirely.
_RESPONSES_SCRIPT = """
<script>
(function () {
  document.querySelectorAll('.col-resize').forEach(function (h) {
    h.addEventListener('mousedown', function (e) {
      var col = document.querySelector('col[data-c="' + h.dataset.c + '"]');
      var startX = e.clientX, startW = col.getBoundingClientRect().width;
      function move(ev) {
        col.style.width = Math.max(40, startW + ev.clientX - startX) + 'px';
      }
      function up() {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
      }
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
      e.preventDefault();
    });
  });
  document.querySelectorAll('.row-resize').forEach(function (h) {
    h.addEventListener('mousedown', function (e) {
      var row = h.closest('tr');
      var startY = e.clientY, startH = row.getBoundingClientRect().height;
      function move(ev) {
        row.style.height = Math.max(24, startH + ev.clientY - startY) + 'px';
      }
      function up() {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
      }
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
      e.preventDefault();
    });
  });
})();
</script>
"""


# --- minimal markdown rendering --------------------------------------------
# Model answers are almost always markdown (bold, headings, bullet/numbered
# lists, the occasional link/quote/rule). Rather than add a markdown dependency
# (this repo deliberately avoids them — see parse_provider_yaml), a focused
# renderer covers that subset. It is NOT a full CommonMark implementation:
# nested lists, tables and reference links are out of scope and fall through as
# text. All source text is HTML-escaped before any markup is added, so a model
# response can never inject raw HTML.

_MD_HR_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")
_MD_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
_MD_UL_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_MD_OL_RE = re.compile(r"^\s*\d+[.)]\s+(.*)$")
_MD_BQ_RE = re.compile(r"^\s*>\s?(.*)$")

# Strong markdown signals used for detection. Lone ``*``/``_`` (italic) is
# deliberately excluded so prose like "2 * 3" or "snake_case" is not mistaken
# for markdown; a cell with real italics almost always also carries bold/lists.
_MD_MARKERS = [
    re.compile(r"\*\*[^*\n]+\*\*"),                  # **bold**
    re.compile(r"__[^_\n]+__"),                      # __bold__
    re.compile(r"(?m)^\s{0,3}#{1,6}\s+\S"),          # # heading
    re.compile(r"(?m)^\s*[-*+]\s+\S"),               # - bullet
    re.compile(r"(?m)^\s*\d+[.)]\s+\S"),             # 1. ordered
    re.compile(r"(?m)^\s*>\s"),                      # > blockquote
    re.compile(r"(?m)^\s*([-*_])(?:\s*\1){2,}\s*$"), # --- rule
    re.compile(r"```"),                              # ``` fenced code
    re.compile(r"`[^`\n]+`"),                        # `inline code`
    re.compile(r"\[[^\]\n]+\]\([^)\n]+\)"),          # [text](url)
]


def looks_like_markdown(text) -> bool:
    """True when ``text`` carries a strong markdown marker (see ``_MD_MARKERS``)."""
    return isinstance(text, str) and any(p.search(text) for p in _MD_MARKERS)


def _md_inline(text: str) -> str:
    """Render inline markdown (already plain text) to safe inline HTML.

    Escapes HTML first, then applies inline code, links, bold, italic and
    strikethrough — in that order so code spans are not reformatted and ``**``
    is consumed before single ``*``.
    """
    text = html.escape(text)  # neutralise any raw HTML in the source first

    codes: list = []

    def _stash(m):
        codes.append(m.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)  # protect inline code spans
    text = re.sub(
        r"\[([^\]]+)\]\(([^)\s]+)\)",
        lambda m: f'<a href="{m.group(2)}" target="_blank" rel="noopener">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"<em>\1</em>", text)
    text = re.sub(r"(?<!_)_(?!_)(.+?)(?<!_)_(?!_)", r"<em>\1</em>", text)
    text = re.sub(r"~~(.+?)~~", r"<del>\1</del>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: f"<code>{codes[int(m.group(1))]}</code>", text)
    return text


def _md_starts_block(line: str) -> bool:
    """True when ``line`` opens a non-paragraph block (stops paragraph gathering)."""
    s = line.strip()
    return bool(
        s.startswith("```")
        or _MD_HR_RE.match(line)
        or _MD_HEADING_RE.match(s)
        or _MD_UL_RE.match(line)
        or _MD_OL_RE.match(line)
        or _MD_BQ_RE.match(line)
    )


def _md_list(lines: list, i: int, regex, tag: str) -> tuple:
    """Consume consecutive list items matching ``regex``; return (html, next_i)."""
    items = []
    while i < len(lines) and regex.match(lines[i]):
        items.append(f"<li>{_md_inline(regex.match(lines[i]).group(1).strip())}</li>")
        i += 1
    return f"<{tag}>{''.join(items)}</{tag}>", i


def render_markdown(text: str) -> str:
    """Render a markdown subset to a safe HTML fragment (block + inline)."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list = []
    i, n = 0, len(lines)
    while i < n:
        line = lines[i]
        if not line.strip():
            i += 1
            continue
        if line.strip().startswith("```"):
            i += 1
            buf = []
            while i < n and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # skip the closing fence (or run off the end)
            out.append(f"<pre><code>{html.escape(chr(10).join(buf))}</code></pre>")
            continue
        if _MD_HR_RE.match(line):
            out.append("<hr>")
            i += 1
            continue
        heading = _MD_HEADING_RE.match(line.strip())
        if heading:
            level = len(heading.group(1))
            out.append(f"<h{level}>{_md_inline(heading.group(2).strip())}</h{level}>")
            i += 1
            continue
        if _MD_BQ_RE.match(line):
            buf = []
            while i < n and _MD_BQ_RE.match(lines[i]):
                buf.append(_md_inline(_MD_BQ_RE.match(lines[i]).group(1)))
                i += 1
            out.append(f"<blockquote>{'<br>'.join(buf)}</blockquote>")
            continue
        if _MD_UL_RE.match(line):
            block, i = _md_list(lines, i, _MD_UL_RE, "ul")
            out.append(block)
            continue
        if _MD_OL_RE.match(line):
            block, i = _md_list(lines, i, _MD_OL_RE, "ol")
            out.append(block)
            continue
        buf = []
        while i < n and lines[i].strip() and not _md_starts_block(lines[i]):
            buf.append(_md_inline(lines[i].strip()))
            i += 1
        out.append(f"<p>{'<br>'.join(buf)}</p>")
    return "".join(out)


def _response_cell_inner(text) -> str:
    """Cell contents: rendered markdown when detected, else escaped plain text."""
    text = _csv_cell(text)
    if text and looks_like_markdown(text):
        return f'<div class="md">{render_markdown(text)}</div>'
    return html.escape(text)


def render_responses_html(per_provider: dict, ordered_keys: list) -> str:
    """Render the interactive raw-response table as a self-contained HTML string.

    ``per_provider`` maps provider key -> {identity: {prompt, output, suite}};
    ``ordered_keys`` is the column order (baseline first). One row per test
    identity (the union across providers), ordered by ``(suite, identity)``.
    """
    ordered_keys = list(ordered_keys)
    headers = ["prompt"] + ordered_keys

    # Initial widths as percentages so the untouched table fills/reflows with the
    # window; the prompt gets a wider share, providers split the rest evenly. A
    # manual drag overrides the column it touches with a pixel width.
    prompt_pct = 28
    provider_pct = (100 - prompt_pct) / len(ordered_keys) if ordered_keys else 100 - prompt_pct
    widths = [prompt_pct] + [provider_pct] * len(ordered_keys)
    colgroup = "".join(
        f'<col data-c="{i}" style="width:{w:.4g}%">' for i, w in enumerate(widths)
    )

    head_cells = "".join(
        f'<th>{html.escape(h)}<span class="col-resize" data-c="{i}"></span></th>'
        for i, h in enumerate(headers)
    )

    meta: dict = {}  # identity -> (suite, prompt); first listed column wins
    for key in ordered_keys:
        for identity, rec in per_provider.get(key, {}).items():
            meta.setdefault(identity, (rec.get("suite") or NO_SUITE, rec.get("prompt", "")))
    order = sorted(meta, key=lambda i: (meta[i][0], i))

    body_rows = []
    for identity in order:
        _, prompt = meta[identity]
        cells = [
            f'<td>{_response_cell_inner(prompt)}'
            '<span class="row-resize"></span></td>'
        ]
        for key in ordered_keys:
            rec = per_provider.get(key, {}).get(identity)
            cells.append(f"<td>{_response_cell_inner(rec.get('output') if rec else '')}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")

    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<title>Raw responses</title>"
        f"<style>{_RESPONSES_CSS}</style></head><body><div class='wrap'>"
        "<h1>Raw responses</h1>"
        "<p class='hint'>Drag a column's right edge or a row's bottom edge to "
        "resize. Cells word-wrap and the table reflows with the window.</p>"
        "<table><colgroup>" + colgroup + "</colgroup>"
        "<thead><tr>" + head_cells + "</tr></thead>"
        "<tbody>" + "".join(body_rows) + "</tbody></table>"
        "</div>" + _RESPONSES_SCRIPT + "</body></html>"
    )


def write_responses_html(per_provider: dict, ordered_keys: list, out_path) -> int:
    """Render the response table and write it to ``out_path``; return row count."""
    n = sum(1 for _ in {  # count distinct test identities across providers
        identity
        for key in ordered_keys
        for identity in per_provider.get(key, {})
    })
    Path(out_path).write_text(
        render_responses_html(per_provider, ordered_keys), encoding="utf-8"
    )
    return n


def build_responses_html(eval_json: dict, columns: list, out_path) -> int:
    """Write the response HTML table from one unified eval file split by ``columns``."""
    per_provider = {c.key: extract_responses(eval_json, c.label) for c in columns}
    return write_responses_html(per_provider, [c.key for c in columns], out_path)


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
        help="Suite to include (repeatable). Defaults to every suite present "
        "in either run.",
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
        "--csv",
        type=Path,
        default=None,
        help="Also write a CSV of the raw (reasoning-stripped) responses: one "
        "row per test, column 1 the rendered prompt, then one column per "
        "provider (baseline first). No scores or verdicts — for eyeballing the "
        "answers side by side.",
    )
    parser.add_argument(
        "--responses-html",
        type=Path,
        default=None,
        help="Also write an interactive HTML table of the raw "
        "(reasoning-stripped) responses: one row per test, column 1 the rendered "
        "prompt, then one column per provider. Word-wrapped cells, drag-resizable "
        "columns/rows, reflows with the window. For eyeballing answers side by "
        "side when a spreadsheet mangles the newlines.",
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
    parser.add_argument(
        "--baseline-provider",
        default=None,
        help="When both sides live in one unified eval file, the provider label "
        "to treat as the baseline. Pass the same file as both positional args.",
    )
    parser.add_argument(
        "--candidate-provider",
        default=None,
        help="The provider label to treat as the candidate in a unified eval file.",
    )
    parser.add_argument(
        "--baseline-provider-col",
        default=None,
        help="N-provider mode: the baseline column as 'key=label' (the config key "
        "is shown as the column header; the promptfoo provider label splits the "
        "unified eval file). When given, the report uses the multi-provider layout "
        "and --baseline-provider/--candidate-provider are ignored.",
    )
    parser.add_argument(
        "--provider-col",
        action="append",
        default=None,
        help="N-provider mode: a non-baseline column as 'key=label' (repeatable).",
    )
    parser.add_argument(
        "--config-path",
        default=None,
        help="Path to the comparison config that produced this run. When given, "
        "the report header shows it as a file:// link so the reader can jump "
        "straight to the config (useful when batch_comparison.py is iterating "
        "over many).",
    )
    parser.add_argument(
        "--system-prompt-path",
        default=None,
        help="Path to the system prompt used by the dev (candidate) side. "
        "Rendered as a file:// link in the report header alongside --config-path.",
    )
    return parser


def _main_n(args, eval_json, suites) -> int:
    """N-provider report path (driven by run_comparison via --*-provider-col).

    All columns read from the one unified eval file, split by each provider's
    promptfoo label; the columns are shown under their config keys.
    """
    columns = parse_provider_col_args(
        args.baseline_provider_col, args.provider_col or []
    )
    base_dir = Path.cwd()
    eval_id = read_eval_id(eval_json)
    cells_by_provider = {
        c.key: extract_cells(eval_json, suites, c.label) for c in columns
    }
    rows = build_rows(cells_by_provider, columns)
    drift = _drift_n(cells_by_provider, columns)
    eval_ids = {c.key: eval_id for c in columns}
    errored = {c.key: errored_tests(eval_json, suites, c.label) for c in columns}
    curls = {c.key: build_curls(eval_json, base_dir, None, c.label) for c in columns}
    args.out.write_text(
        render_html_n(
            rows, columns, drift, args.tolerance,
            eval_ids=eval_ids, ui_base_url=args.ui_base_url,
            errored=errored, curls=curls,
            config_path=args.config_path,
            system_prompt_path=args.system_prompt_path,
        ),
        encoding="utf-8",
    )
    if args.csv:
        n = build_response_csv(eval_json, columns, args.csv)
        print(f"  {n} response rows -> {args.csv}")
    if args.responses_html:
        n = build_responses_html(eval_json, columns, args.responses_html)
        print(f"  {n} response rows -> {args.responses_html}")
    others = [c.key for c in columns if not c.is_baseline]
    rtab = summarize_rubric_table(rows, columns, args.tolerance)
    summary = " | ".join(
        f"{o}: {rtab[o]['improved']} improved, {rtab[o]['regressed']} regressed"
        for o in others
    )
    print(f"{len(columns)} providers, {len(rows)} rows | {summary} -> {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suites = args.suite  # None => every suite present in either run

    baseline_json = read_eval_json(args.baseline_json)
    # N-provider mode: one unified file (passed as both positionals), split by
    # the per-column provider labels.
    if args.baseline_provider_col:
        return _main_n(args, baseline_json, suites)

    candidate_json = read_eval_json(args.candidate_json)
    # Provider labels are only needed for a unified run (one file, both
    # providers); for the classic two-file mode they stay None (all results).
    base_provider = args.baseline_provider
    cand_provider = args.candidate_provider
    baseline_cells = extract_cells(baseline_json, suites, base_provider)
    candidate_cells = extract_cells(candidate_json, suites, cand_provider)

    diffs = diff_cells(baseline_cells, candidate_cells, args.tolerance)
    counts = summarize(diffs)
    det = summarize_deterministic(diffs)
    drift = diff_test_keys(baseline_cells, candidate_cells)

    base_dir = Path.cwd()
    args.out.write_text(
        render_html(diffs, drift, args.tolerance,
                    baseline_eval_id=read_eval_id(baseline_json),
                    candidate_eval_id=read_eval_id(candidate_json),
                    ui_base_url=args.ui_base_url,
                    baseline_errored=errored_tests(baseline_json, suites, base_provider),
                    candidate_errored=errored_tests(candidate_json, suites, cand_provider),
                    baseline_curls=build_curls(baseline_json, base_dir, args.baseline_url, base_provider),
                    candidate_curls=build_curls(candidate_json, base_dir, args.candidate_url, cand_provider),
                    config_path=args.config_path,
                    system_prompt_path=args.system_prompt_path),
        encoding="utf-8",
    )
    if args.csv:
        per_provider = {
            "baseline": extract_responses(baseline_json, base_provider),
            "candidate": extract_responses(candidate_json, cand_provider),
        }
        n = write_response_csv(per_provider, ["baseline", "candidate"], args.csv)
        print(f"  {n} response rows -> {args.csv}")
    if args.responses_html:
        per_provider = {
            "baseline": extract_responses(baseline_json, base_provider),
            "candidate": extract_responses(candidate_json, cand_provider),
        }
        n = write_responses_html(per_provider, ["baseline", "candidate"], args.responses_html)
        print(f"  {n} response rows -> {args.responses_html}")
    msg = (
        f"rubric: {counts['improved']} improved, {counts['regressed']} regressed, "
        f"{counts['within']} within, {counts['new']} new, {counts['removed']} removed"
    )
    if det["total_passes"] or det["total_fails"]:
        msg += (
            f" | deterministic: {det['new_passes']} new passes, "
            f"{det['new_fails']} new fails, {det['total_passes']} total passes, "
            f"{det['total_fails']} total fails"
        )
    print(f"{msg} -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

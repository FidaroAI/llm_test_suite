# Provider Comparison Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two CLI scripts that freeze a promptfoo eval as a committed baseline and render a color-coded report of per-assertion llm-rubric score deltas between that baseline and a candidate run.

**Architecture:** Pure parsing/diff helpers over promptfoo result JSON, with thin argparse `main()` wrappers — matching the existing `scripts_repo/pull_ci_results.py` style. `freeze_baseline.py` copies an eval JSON (filtered per provider) into a committed `baselines/` dir with a `_baseline_meta` provenance block. `compare_runs.py` extracts llm-rubric "cells" (one per assertion per test) from two eval JSONs, joins them on a stable `(test description, prompt label, rubric text)` key, classifies each delta against an absolute tolerance band, and emits a self-contained HTML report.

**Tech Stack:** Python 3.11+ (stdlib only: `argparse`, `json`, `dataclasses`, `html`, `subprocess`, `pathlib`), pytest. No new dependencies.

---

## Verified promptfoo JSON facts (do not re-derive)

From a real result file (`results/ci/2026-05-20_9765c4f.json`):

- Results live at `eval_json["results"]["results"]` (a list).
- Per entry:
  - `entry["provider"]["label"]` — provider label (e.g. `fidaro_plaintext_gateway_phala`).
  - `entry["prompt"]["label"]` — prompt label (e.g. `user_only`). Stable & readable.
  - `entry["description"]` is **null** — the real description is at `entry["testCase"]["description"]` (e.g. `researchrubrics[Current Events] 6847465956a0f6376a605427`).
  - `entry["testCase"]["metadata"]["suite"]` — `research_rubrics`, `agentharm_refusal`, or **absent/None** for the deterministic CSV fact tests.
  - `entry["testCase"]["assert"]` — ordered list of assertion defs: `{type, value, metric, weight}`. **Authoritative** source of rubric identity.
  - `entry["gradingResult"]["componentResults"]` — same length & order as `assert`, each `{score, pass, reason, ...}`. **Authoritative** source of scores. Its embedded `.assertion` sub-object may be `null`, so never read identity from here — read it from `testCase.assert[i]` at the same index `i`.
  - A `componentResults[i]["score"]` can be missing/None when the grader errored — coerce to `0.0`.

---

## File structure

- Create: `scripts_repo/freeze_baseline.py` — freeze an eval JSON to `baselines/<label>.json`.
- Create: `scripts_repo/compare_runs.py` — diff engine + HTML renderer.
- Create: `scripts_repo/tests/_fixtures.py` — synthetic-eval JSON builders shared by tests.
- Create: `scripts_repo/tests/test_freeze_baseline.py` — unit tests for freeze helpers.
- Create: `scripts_repo/tests/test_compare_runs.py` — unit tests for extract/diff/render.
- Create: `baselines/.gitkeep` — make the committed baseline dir exist.
- Modify: `README.md` — document the comparison workflow.

`scripts_repo/tests/conftest.py` already inserts the project root on `sys.path`, so tests import via `from scripts_repo.compare_runs import ...`.

---

## Task 1: Shared test-fixture builders

**Files:**
- Create: `scripts_repo/tests/_fixtures.py`

- [ ] **Step 1: Write the fixture builders**

```python
# scripts_repo/tests/_fixtures.py
"""Synthetic promptfoo-eval JSON builders for unit tests (no network)."""

from __future__ import annotations


def rubric_result(
    provider_label,
    description,
    suite,
    asserts,
    scores,
    prompt_label="user_only",
    metadata_extra=None,
):
    """Build one eval_json["results"]["results"][i] entry of llm-rubric asserts.

    asserts: list of (value, metric, weight) tuples.
    scores:  list of component scores, index-aligned with asserts.
    """
    metadata = {"suite": suite}
    if metadata_extra:
        metadata.update(metadata_extra)
    assert_objs = [
        {"type": "llm-rubric", "value": v, "metric": m, "weight": w}
        for (v, m, w) in asserts
    ]
    comps = [{"score": s, "pass": s >= 0.5} for s in scores]
    return {
        "provider": {"id": "x", "label": provider_label},
        "prompt": {"label": prompt_label},
        "vars": {"user": "..."},
        "testCase": {
            "description": description,
            "vars": {"user": "..."},
            "assert": assert_objs,
            "metadata": metadata,
        },
        "gradingResult": {
            "score": (sum(scores) / len(scores)) if scores else 0,
            "componentResults": comps,
        },
    }


def make_eval_json(results, eval_id="eval-test"):
    """Wrap a list of result entries in the eval_json envelope."""
    return {"evalId": eval_id, "results": {"results": list(results)}}
```

- [ ] **Step 2: Commit**

```bash
git add scripts_repo/tests/_fixtures.py
git commit -m "test: add synthetic promptfoo-eval fixture builders"
```

---

## Task 2: `freeze_baseline.py` pure helpers

**Files:**
- Create: `scripts_repo/freeze_baseline.py`
- Test: `scripts_repo/tests/test_freeze_baseline.py`

- [ ] **Step 1: Write failing tests for the helpers**

```python
# scripts_repo/tests/test_freeze_baseline.py
from scripts_repo.freeze_baseline import (
    sanitize_label,
    provider_labels,
    filter_to_provider,
    gather_test_keys,
    build_baseline,
)
from scripts_repo.tests._fixtures import rubric_result, make_eval_json


def test_sanitize_label_replaces_unsafe_chars():
    assert sanitize_label("openai:chat/Qwen 80B") == "openai_chat_Qwen_80B"


def test_sanitize_label_empty_falls_back():
    assert sanitize_label("///") == "provider"


def test_provider_labels_dedupes_in_order():
    ev = make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("cand", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("prod", "t2", "research_rubrics", [("b", "x", 1)], [1.0]),
    ])
    assert provider_labels(ev) == ["prod", "cand"]


def test_filter_to_provider_keeps_only_one():
    ev = make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("cand", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
    ])
    out = filter_to_provider(ev, "prod")
    labels = {r["provider"]["label"] for r in out["results"]["results"]}
    assert labels == {"prod"}
    # original is untouched (deep copy)
    assert len(ev["results"]["results"]) == 2


def test_gather_test_keys_sorted_unique_nonnull():
    ev = make_eval_json([
        rubric_result("prod", "t2", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("prod", "t1", "research_rubrics", [("b", "x", 1)], [1.0]),
    ])
    assert gather_test_keys(ev) == ["t1", "t2"]


def test_build_baseline_adds_meta():
    ev = make_eval_json(
        [rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0])],
        eval_id="eval-abc",
    )
    out = build_baseline(ev, "prod", "deadbeef")
    meta = out["_baseline_meta"]
    assert meta["provider_label"] == "prod"
    assert meta["git_sha"] == "deadbeef"
    assert meta["eval_id"] == "eval-abc"
    assert meta["test_keys"] == ["t1"]
    assert "frozen_at" in meta
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts_repo/tests/test_freeze_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts_repo.freeze_baseline'`.

- [ ] **Step 3: Write the helpers (module top + pure functions)**

```python
# scripts_repo/freeze_baseline.py
#!/usr/bin/env python3
"""Freeze a promptfoo eval result as a committed baseline artifact.

Reads an eval result JSON and writes one baseline file per provider into
`baselines/<sanitized-label>.json`, each containing only that provider's
results plus a `_baseline_meta` provenance block. Baselines are committed and
used as the fixed reference for `compare_runs.py`.

Usage:
    freeze_baseline.py results/local/latest.json
    freeze_baseline.py results/local/latest.json --out-dir baselines --force
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT_DIR = Path("baselines")


def sanitize_label(label: str) -> str:
    """Filesystem-safe version of a provider label."""
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")
    return safe or "provider"


def provider_labels(eval_json: dict) -> list[str]:
    """Distinct provider labels, in first-seen order."""
    labels: list[str] = []
    for result in eval_json.get("results", {}).get("results", []):
        label = (result.get("provider") or {}).get("label")
        if label and label not in labels:
            labels.append(label)
    return labels


def filter_to_provider(eval_json: dict, label: str) -> dict:
    """Deep copy of eval_json keeping only one provider's result entries."""
    out = copy.deepcopy(eval_json)
    results = out.get("results", {}).get("results", [])
    out["results"]["results"] = [
        r for r in results if (r.get("provider") or {}).get("label") == label
    ]
    return out


def gather_test_keys(eval_json: dict) -> list[str]:
    """Sorted, unique, non-null testCase descriptions (used as test keys)."""
    descs = set()
    for result in eval_json.get("results", {}).get("results", []):
        desc = (result.get("testCase") or {}).get("description")
        if desc:
            descs.add(desc)
    return sorted(descs)


def git_sha() -> str | None:
    """HEAD SHA of the suite repo, or None if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], text=True, capture_output=True
        )
    except FileNotFoundError:
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def build_baseline(eval_json: dict, label: str, sha: str | None) -> dict:
    """Provider-filtered copy of eval_json with a _baseline_meta block."""
    out = filter_to_provider(eval_json, label)
    out["_baseline_meta"] = {
        "provider_label": label,
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "eval_id": eval_json.get("evalId"),
        "test_keys": gather_test_keys(out),
    }
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts_repo/tests/test_freeze_baseline.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/freeze_baseline.py scripts_repo/tests/test_freeze_baseline.py
git commit -m "feat: add freeze_baseline pure helpers"
```

---

## Task 3: `freeze_baseline.py` freeze() + CLI

**Files:**
- Modify: `scripts_repo/freeze_baseline.py` (append `freeze`, `build_parser`, `main`)
- Test: `scripts_repo/tests/test_freeze_baseline.py` (append integration tests)

- [ ] **Step 1: Write failing tests for freeze()**

```python
# append to scripts_repo/tests/test_freeze_baseline.py
import json
from pathlib import Path

import pytest

from scripts_repo.freeze_baseline import freeze


def _write_eval(tmp_path, ev):
    p = tmp_path / "latest.json"
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_freeze_writes_one_file_per_provider(tmp_path):
    ev = make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("cand", "t1", "research_rubrics", [("a", "x", 1)], [0.5]),
    ])
    src = _write_eval(tmp_path, ev)
    out_dir = tmp_path / "baselines"
    written = freeze(src, out_dir, force=False)
    names = sorted(p.name for p in written)
    assert names == ["cand.json", "prod.json"]
    prod = json.loads((out_dir / "prod.json").read_text())
    labels = {r["provider"]["label"] for r in prod["results"]["results"]}
    assert labels == {"prod"}
    assert prod["_baseline_meta"]["provider_label"] == "prod"


def test_freeze_refuses_overwrite_without_force(tmp_path):
    ev = make_eval_json(
        [rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0])]
    )
    src = _write_eval(tmp_path, ev)
    out_dir = tmp_path / "baselines"
    freeze(src, out_dir, force=False)
    with pytest.raises(FileExistsError):
        freeze(src, out_dir, force=False)


def test_freeze_force_overwrites(tmp_path):
    ev = make_eval_json(
        [rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0])]
    )
    src = _write_eval(tmp_path, ev)
    out_dir = tmp_path / "baselines"
    freeze(src, out_dir, force=False)
    freeze(src, out_dir, force=True)  # must not raise


def test_freeze_no_providers_raises(tmp_path):
    src = _write_eval(tmp_path, make_eval_json([]))
    with pytest.raises(ValueError):
        freeze(src, tmp_path / "baselines", force=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts_repo/tests/test_freeze_baseline.py -k freeze -v`
Expected: FAIL with `ImportError: cannot import name 'freeze'`.

- [ ] **Step 3: Implement freeze() + CLI (append to module)**

```python
# append to scripts_repo/freeze_baseline.py

def freeze(result_path, out_dir: Path, force: bool) -> list[Path]:
    """Write one baseline file per provider; return the written paths."""
    eval_json = json.loads(Path(result_path).read_text(encoding="utf-8"))
    labels = provider_labels(eval_json)
    if not labels:
        raise ValueError(f"no provider labels found in {result_path}")
    sha = git_sha()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label in labels:
        dest = out_dir / f"{sanitize_label(label)}.json"
        if dest.exists() and not force:
            raise FileExistsError(f"{dest} exists; use --force to overwrite")
        baseline = build_baseline(eval_json, label, sha)
        dest.write_text(json.dumps(baseline, indent=2), encoding="utf-8")
        written.append(dest)
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("result_json", help="promptfoo eval result JSON to freeze.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory for baseline files (default: baselines/).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing baseline files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    written = freeze(args.result_json, args.out_dir, force=args.force)
    for p in written:
        print(f"froze baseline {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts_repo/tests/test_freeze_baseline.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/freeze_baseline.py scripts_repo/tests/test_freeze_baseline.py
git commit -m "feat: add freeze() and CLI to freeze_baseline"
```

---

## Task 4: `compare_runs.py` cell extraction

**Files:**
- Create: `scripts_repo/compare_runs.py`
- Test: `scripts_repo/tests/test_compare_runs.py`

- [ ] **Step 1: Write failing tests for extract_cells**

```python
# scripts_repo/tests/test_compare_runs.py
from scripts_repo.compare_runs import (
    DEFAULT_SUITES,
    CellKey,
    extract_cells,
)
from scripts_repo.tests._fixtures import rubric_result, make_eval_json


def test_extract_cells_one_per_assertion():
    ev = make_eval_json([
        rubric_result(
            "prod", "t1", "research_rubrics",
            [("cover period", "Explicit", 5), ("has headers", "Comms", 2)],
            [0.8, 0.4],
        ),
    ])
    cells = extract_cells(ev, DEFAULT_SUITES)
    assert len(cells) == 2
    k = CellKey(test="t1", prompt="user_only", assertion="cover period")
    assert cells[k].score == 0.8
    assert cells[k].metric == "Explicit"
    assert cells[k].weight == 5
    assert cells[k].suite == "research_rubrics"


def test_extract_cells_filters_by_suite():
    # A no-suite entry (deterministic fact test) must be dropped.
    fact = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {"description": "fact", "metadata": {}, "assert": [
            {"type": "contains", "value": "Paris"}]},
        "gradingResult": {"componentResults": [{"score": 1.0}]},
    }
    rubric = rubric_result(
        "prod", "t1", "agentharm_refusal", [("no refusal", "non_refusal", 1)], [1.0]
    )
    cells = extract_cells(make_eval_json([fact, rubric]), DEFAULT_SUITES)
    assert {k.test for k in cells} == {"t1"}


def test_extract_cells_skips_non_llm_rubric_but_keeps_index_alignment():
    # assert[0] is a python assert (skip), assert[1] is llm-rubric (keep).
    # Its score must come from componentResults[1], not [0].
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [
                {"type": "python", "value": "file://x.py"},
                {"type": "llm-rubric", "value": "quality", "metric": "Q", "weight": 1},
            ],
        },
        "gradingResult": {"componentResults": [{"score": 0.0}, {"score": 0.9}]},
    }
    cells = extract_cells(make_eval_json([entry]), DEFAULT_SUITES)
    k = CellKey(test="t1", prompt="user_only", assertion="quality")
    assert cells[k].score == 0.9


def test_extract_cells_disambiguates_duplicate_rubric_text():
    ev = make_eval_json([
        rubric_result(
            "prod", "t1", "research_rubrics",
            [("same text", "A", 1), ("same text", "B", 1)],
            [0.2, 0.7],
        ),
    ])
    cells = extract_cells(ev, DEFAULT_SUITES)
    assert cells[CellKey("t1", "user_only", "same text")].score == 0.2
    assert cells[CellKey("t1", "user_only", "same text#1")].score == 0.7


def test_extract_cells_null_score_coerced_to_zero():
    entry = {
        "provider": {"label": "prod"},
        "prompt": {"label": "user_only"},
        "testCase": {
            "description": "t1",
            "metadata": {"suite": "research_rubrics"},
            "assert": [{"type": "llm-rubric", "value": "q", "metric": "Q", "weight": 1}],
        },
        "gradingResult": {"componentResults": [{"score": None}]},
    }
    cells = extract_cells(make_eval_json([entry]), DEFAULT_SUITES)
    assert cells[CellKey("t1", "user_only", "q")].score == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts_repo/tests/test_compare_runs.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts_repo.compare_runs'`.

- [ ] **Step 3: Implement module top + extract_cells**

```python
# scripts_repo/compare_runs.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts_repo/tests/test_compare_runs.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/compare_runs.py scripts_repo/tests/test_compare_runs.py
git commit -m "feat: add compare_runs cell extraction"
```

---

## Task 5: `compare_runs.py` diff, classify, summary, drift

**Files:**
- Modify: `scripts_repo/compare_runs.py` (append)
- Test: `scripts_repo/tests/test_compare_runs.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# append to scripts_repo/tests/test_compare_runs.py
from scripts_repo.compare_runs import (
    classify,
    diff_cells,
    summarize,
    diff_test_keys,
)


def _cells(ev):
    return extract_cells(ev, DEFAULT_SUITES)


def test_classify_boundaries():
    # Exactly at the band is "within"; strictly beyond flips.
    assert classify(0.05, 0.05) == "within"
    assert classify(-0.05, 0.05) == "within"
    assert classify(0.06, 0.05) == "improved"
    assert classify(-0.06, 0.05) == "regressed"


def test_diff_improved_regressed_within():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1), ("c", "X", 1)],
                      [0.50, 0.90, 0.50]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1), ("c", "X", 1)],
                      [0.90, 0.40, 0.52]),
    ]))
    diffs = {d.key.assertion: d for d in diff_cells(base, cand, 0.05)}
    assert diffs["a"].status == "improved"
    assert diffs["b"].status == "regressed"
    assert diffs["c"].status == "within"
    assert round(diffs["a"].delta, 2) == 0.40


def test_diff_new_and_removed():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("prod", "tg", "research_rubrics", [("g", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("cand", "tn", "research_rubrics", [("n", "X", 1)], [0.5]),
    ]))
    by_status = {}
    for d in diff_cells(base, cand, 0.05):
        by_status.setdefault(d.status, []).append(d.key.assertion)
    assert by_status["new"] == ["n"]
    assert by_status["removed"] == ["g"]


def test_summarize_counts():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.5, 0.9]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.9, 0.4]),
    ]))
    counts = summarize(diff_cells(base, cand, 0.05))
    assert counts["improved"] == 1
    assert counts["regressed"] == 1
    assert counts["within"] == 0
    assert counts["new"] == 0
    assert counts["removed"] == 0


def test_diff_test_keys():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("prod", "tg", "research_rubrics", [("g", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("cand", "tn", "research_rubrics", [("n", "X", 1)], [0.5]),
    ]))
    only_base, only_cand = diff_test_keys(base, cand)
    assert only_base == ["tg"]
    assert only_cand == ["tn"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts_repo/tests/test_compare_runs.py -k "classify or diff or summarize or drift" -v`
Expected: FAIL with `ImportError: cannot import name 'classify'`.

- [ ] **Step 3: Implement diff/classify/summary/drift (append to module)**

```python
# append to scripts_repo/compare_runs.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts_repo/tests/test_compare_runs.py -v`
Expected: PASS (10 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/compare_runs.py scripts_repo/tests/test_compare_runs.py
git commit -m "feat: add compare_runs diff, classify, summary, drift"
```

---

## Task 6: `compare_runs.py` HTML rendering

**Files:**
- Modify: `scripts_repo/compare_runs.py` (append)
- Test: `scripts_repo/tests/test_compare_runs.py` (append)

- [ ] **Step 1: Write failing tests**

```python
# append to scripts_repo/tests/test_compare_runs.py
from scripts_repo.compare_runs import render_html


def test_render_html_contains_summary_and_markers():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.5, 0.9]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1), ("b", "X", 1)], [0.9, 0.4]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05)
    assert "<html" in out.lower()
    assert "1 improved" in out
    assert "1 regressed" in out
    assert "research_rubrics" in out
    assert 'class="status-improved"' in out
    assert 'class="status-regressed"' in out


def test_render_html_shows_drift_banner():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), (["t_missing"], []), 0.05)
    assert "config drift" in out.lower()
    assert "t_missing" in out


def test_render_html_escapes_markup():
    base = _cells(make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("<script>x</script>", "X", 1)], [0.5]),
    ]))
    cand = _cells(make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("<script>x</script>", "X", 1)], [0.9]),
    ]))
    diffs = diff_cells(base, cand, 0.05)
    out = render_html(diffs, summarize(diffs), ([], []), 0.05)
    assert "<script>x</script>" not in out
    assert "&lt;script&gt;" in out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts_repo/tests/test_compare_runs.py -k render -v`
Expected: FAIL with `ImportError: cannot import name 'render_html'`.

- [ ] **Step 3: Implement render_html (append to module)**

```python
# append to scripts_repo/compare_runs.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts_repo/tests/test_compare_runs.py -v`
Expected: PASS (13 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/compare_runs.py scripts_repo/tests/test_compare_runs.py
git commit -m "feat: add compare_runs HTML rendering"
```

---

## Task 7: `compare_runs.py` CLI wiring

**Files:**
- Modify: `scripts_repo/compare_runs.py` (append `build_parser`, `main`)
- Test: `scripts_repo/tests/test_compare_runs.py` (append integration test)

- [ ] **Step 1: Write failing integration test**

```python
# append to scripts_repo/tests/test_compare_runs.py
import json as _json

from scripts_repo.compare_runs import main


def _write(tmp_path, name, ev):
    p = tmp_path / name
    p.write_text(_json.dumps(ev), encoding="utf-8")
    return p


def test_main_writes_report_and_returns_zero(tmp_path, capsys):
    base = _write(tmp_path, "base.json", make_eval_json([
        rubric_result("prod", "t1", "research_rubrics",
                      [("a", "X", 1)], [0.5]),
    ]))
    cand = _write(tmp_path, "cand.json", make_eval_json([
        rubric_result("cand", "t1", "research_rubrics",
                      [("a", "X", 1)], [0.9]),
    ]))
    out = tmp_path / "report.html"
    rc = main([str(base), str(cand), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "1 improved" in out.read_text(encoding="utf-8")
    assert "1 improved" in capsys.readouterr().out


def test_main_respects_suite_override(tmp_path):
    # Only research_rubrics requested; agentharm cells must be excluded.
    base = _write(tmp_path, "base.json", make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "X", 1)], [0.5]),
        rubric_result("prod", "h1", "agentharm_refusal", [("nr", "non_refusal", 1)], [1.0]),
    ]))
    cand = _write(tmp_path, "cand.json", make_eval_json([
        rubric_result("cand", "t1", "research_rubrics", [("a", "X", 1)], [0.9]),
        rubric_result("cand", "h1", "agentharm_refusal", [("nr", "non_refusal", 1)], [0.0]),
    ]))
    out = tmp_path / "report.html"
    main([str(base), str(cand), "--out", str(out), "--suite", "research_rubrics"])
    text = out.read_text(encoding="utf-8")
    assert "research_rubrics" in text
    assert "agentharm_refusal" not in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest scripts_repo/tests/test_compare_runs.py -k main -v`
Expected: FAIL with `ImportError: cannot import name 'main'`.

- [ ] **Step 3: Implement build_parser + main (append to module)**

Note the argparse gotcha: `action="append"` with a non-empty `default` would
*accumulate* onto the default. Use `default=None` and substitute `DEFAULT_SUITES`
when nothing was passed.

```python
# append to scripts_repo/compare_runs.py

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest scripts_repo/tests/test_compare_runs.py -v`
Expected: PASS (15 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/compare_runs.py scripts_repo/tests/test_compare_runs.py
git commit -m "feat: add compare_runs CLI"
```

---

## Task 8: Committed baselines dir + README docs

**Files:**
- Create: `baselines/.gitkeep`
- Modify: `README.md`

- [ ] **Step 1: Confirm `baselines/` is NOT gitignored**

Run: `git check-ignore -v baselines/prod.json || echo "not ignored (good)"`
Expected: prints `not ignored (good)`. If a rule matches, add a `!baselines/` negation to `.gitignore` and re-check before continuing.

- [ ] **Step 2: Create the placeholder so the dir is tracked**

```bash
mkdir -p baselines
printf '# Committed baseline eval artifacts. See README "Comparing two providers".\n' > baselines/.gitkeep
```

- [ ] **Step 3: Add a README section**

Add this section to `README.md` (place it after the existing run/usage content; match the file's existing heading style):

```markdown
## Comparing two providers

Compare the llm-rubric quality of a candidate provider against a frozen
baseline (e.g. prod). Only the non-deterministic rubric suites
(`research_rubrics`, `agentharm_refusal`) are compared.

1. **Freeze the baseline** — make the baseline provider active in
   `promptfooconfig.yaml`, run the suite, then freeze the result:

   ```bash
   promptfoo eval --cache
   scripts_repo/freeze_baseline.py results/local/latest.json
   # -> baselines/<provider_label>.json   (committed reference)
   ```

   Re-run this only when prod changes. Commit the baseline file.

2. **Run the candidate** — switch the active provider, run again:

   ```bash
   promptfoo eval --cache
   ```

3. **Compare** — diff candidate against the frozen baseline:

   ```bash
   scripts_repo/compare_runs.py \
       baselines/<provider_label>.json results/local/latest.json \
       --tolerance 0.05 --out report.html
   open report.html
   ```

   The report groups per-assertion deltas by suite, worst-first, and colors
   each cell green (improved beyond tolerance), red (regressed), or grey
   (within the ±tolerance band). Moves smaller than `--tolerance` (default
   0.05 on the 0–1 score) are treated as noise.
```

- [ ] **Step 4: Run the full suite to confirm a clean baseline**

Run: `pytest scripts_repo/tests -v`
Expected: PASS (all freeze_baseline + compare_runs tests).

- [ ] **Step 5: Commit**

```bash
git add baselines/.gitkeep README.md
git commit -m "docs: add baselines dir and provider-comparison workflow"
```

---

## Final verification

- [ ] Run the complete test suite: `pytest scripts_repo/tests -v` → all pass.
- [ ] Smoke-test the real CLI end to end against committed CI artifacts:

```bash
scripts_repo/freeze_baseline.py results/ci/2026-05-20_9765c4f.json \
    --out-dir /tmp/cmp-baselines --force
scripts_repo/compare_runs.py \
    /tmp/cmp-baselines/fidaro_plaintext_gateway_phala.json \
    results/ci/2026-05-20_9765c4f.json \
    --out /tmp/cmp-report.html
```

Expected: both commands exit 0; `/tmp/cmp-report.html` exists and contains a
summary line. (Comparing the file against itself should report everything as
`within`, with 0 improved / 0 regressed — a sanity check that the join lines up.)

---

## Notes for the executor

- `chmod +x scripts_repo/freeze_baseline.py scripts_repo/compare_runs.py` to match
  the executable convention of the other `scripts_repo/*.py` tools (do this before
  the Task 3 / Task 7 commits, or in Task 8).
- Stdlib only — do not add dependencies to `requirements.txt`.
- Keep helper functions pure and free of I/O so the unit tests need no network or
  API keys, per the repo's testing convention.
```

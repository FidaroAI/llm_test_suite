# CI Results Puller Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts_repo/pull_ci_results.py` to pull promptfoo result artifacts from this repo's GitHub Actions runs into `results/ci/` and import them into the local promptfoo DB, and relocate local eval output to `results/local/`.

**Architecture:** A single-file Python script with a thin layer of subprocess wrappers around `gh`, `git`, and `promptfoo`, and a core of pure functions (date parsing, run selection, filename construction, failure classification) that carry the testable logic. Three modes (`--commit`, `--latest`, `--all`) share one "discover run → download → import" pipeline.

**Tech Stack:** Python 3 stdlib only (`argparse`, `json`, `subprocess`, `tempfile`, `shutil`, `datetime`, `pathlib`). Tests use `pytest`. Spec: `docs/superpowers/specs/2026-05-14-ci-results-puller-design.md`.

---

## Environment notes

- **Run tests with `python3 -m pytest`** (system Python 3.13, pytest 8.4.2). The repo's `.venv/` is broken — its interpreter points at the repo's old path (`~/dev/llm_test_suite`). Fixing the venv is out of scope; the script and its tests need only stdlib + pytest.
- **Commit signing is currently broken in this environment.** If `git commit` fails with a signing error, append `--no-gpg-sign` to the commit command. (No commit in this repo's history is signed; this is the established workaround.)
- Work happens in the **main worktree** (no isolated worktree for this task, per the user's instruction).

## File Structure

- **Create:** `scripts_repo/pull_ci_results.py` — the puller script. One responsibility: turn a mode selection into downloaded-and-imported result files.
- **Create:** `scripts_repo/tests/conftest.py` — puts the project root on `sys.path` so tests can `from scripts_repo.pull_ci_results import ...` (mirrors `assertions/tests/conftest.py`).
- **Create:** `scripts_repo/tests/test_pull_ci_results.py` — unit tests for the pure functions and the import-orchestration logic.
- **Modify:** `promptfooconfig.yaml` — `outputPath` → `results/local/latest.json`.
- **Modify:** `promptfooconfig.ci.yaml` — `outputPath` → `results/ci/latest.json`.
- **Modify:** `scripts_test/hack.sh`, `scripts_test/smoke.sh`, `scripts_test/full.sh` — `--output` paths → `results/local/...`, plus a defensive `mkdir -p`.
- **Modify:** `package.json` — `view` script → `results/local/latest.json`.
- **Modify:** `.github/workflows/promptfoo-gateway.yml` — eval `--output` and artifact `path` → `results/ci/latest.json`, plus `mkdir -p results/ci`.

---

### Task 1: Test scaffold and `parse_date`

**Files:**
- Create: `scripts_repo/tests/conftest.py`
- Create: `scripts_repo/tests/test_pull_ci_results.py`
- Create: `scripts_repo/pull_ci_results.py`

- [ ] **Step 1: Write the conftest**

Create `scripts_repo/tests/conftest.py`:

```python
"""Make the project root importable so tests can `from scripts_repo.foo import ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

- [ ] **Step 2: Write the failing test**

Create `scripts_repo/tests/test_pull_ci_results.py`:

```python
import argparse
import subprocess

import pytest

from scripts_repo.pull_ci_results import parse_date


def test_parse_date_valid():
    result = parse_date("2026-05-14")
    assert (result.year, result.month, result.day) == (2026, 5, 14)


def test_parse_date_invalid_format_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_date("14/05/2026")


def test_parse_date_garbage_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_date("not-a-date")
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scripts_repo.pull_ci_results'`

- [ ] **Step 4: Write minimal implementation**

Create `scripts_repo/pull_ci_results.py`:

```python
#!/usr/bin/env python3
"""Pull promptfoo result artifacts from this repo's GitHub Actions runs.

Downloads result JSON from CI runs into results/ci/ and imports them into the
local promptfoo database. Three modes: a specific commit, the most recent run,
or all commits within an optional date range.

Usage:
    pull_ci_results.py --commit <sha>
    pull_ci_results.py --latest
    pull_ci_results.py --all [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD]

Options:
    --no-import          Download only; skip promptfoo import.
    --fail-on-existing   Abort if a result's eval ID is already imported.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import date, datetime
from pathlib import Path

WORKFLOW = "promptfoo-gateway.yml"
ARTIFACT = "promptfoo-results"
DEST_DIR = Path("results/ci")


def parse_date(value: str) -> date:
    """Parse a YYYY-MM-DD string, raising an argparse-friendly error."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid date {value!r}: expected YYYY-MM-DD"
        )
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 3 passed.

- [ ] **Step 6: Make the script executable and commit**

```bash
chmod +x scripts_repo/pull_ci_results.py
git add scripts_repo/pull_ci_results.py scripts_repo/tests/conftest.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: scaffold pull_ci_results.py with date parsing"
```

---

### Task 2: Pure helpers — `build_filename` and `select_run`

**Files:**
- Modify: `scripts_repo/pull_ci_results.py`
- Test: `scripts_repo/tests/test_pull_ci_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts_repo/tests/test_pull_ci_results.py`:

```python
from scripts_repo.pull_ci_results import build_filename, select_run


def test_build_filename_truncates_sha():
    assert build_filename("2026-05-14", "4a9cae7abcdef") == "2026-05-14_4a9cae7.json"


def test_build_filename_short_sha_unchanged():
    assert build_filename("2026-05-14", "4a9cae7") == "2026-05-14_4a9cae7.json"


def test_select_run_picks_most_recent_successful():
    runs = [
        {"databaseId": 1, "createdAt": "2026-05-10T00:00:00Z",
         "status": "completed", "conclusion": "success"},
        {"databaseId": 2, "createdAt": "2026-05-12T00:00:00Z",
         "status": "completed", "conclusion": "success"},
    ]
    assert select_run(runs)["databaseId"] == 2


def test_select_run_ignores_failed_and_incomplete():
    runs = [
        {"databaseId": 1, "createdAt": "2026-05-12T00:00:00Z",
         "status": "completed", "conclusion": "failure"},
        {"databaseId": 2, "createdAt": "2026-05-11T00:00:00Z",
         "status": "in_progress", "conclusion": None},
        {"databaseId": 3, "createdAt": "2026-05-10T00:00:00Z",
         "status": "completed", "conclusion": "success"},
    ]
    assert select_run(runs)["databaseId"] == 3


def test_select_run_returns_none_when_no_success():
    runs = [
        {"databaseId": 1, "createdAt": "2026-05-12T00:00:00Z",
         "status": "completed", "conclusion": "failure"},
    ]
    assert select_run(runs) is None


def test_select_run_empty_list_returns_none():
    assert select_run([]) is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `ImportError: cannot import name 'build_filename'`

- [ ] **Step 3: Write the implementation**

Append to `scripts_repo/pull_ci_results.py`:

```python
def build_filename(commit_date: str, sha: str) -> str:
    """Build the results/ci filename: <commit-date>_<short-sha>.json.

    commit_date is an ISO date string (YYYY-MM-DD); sha is a full or short
    git SHA, truncated to 7 chars.
    """
    return f"{commit_date}_{sha[:7]}.json"


def select_run(runs: list[dict]) -> dict | None:
    """Pick the most recent successful run from `gh run list` JSON output.

    Each run dict carries at least: databaseId, headSha, createdAt, status,
    conclusion. Returns None when no completed+successful run exists.
    """
    successful = [
        r for r in runs
        if r.get("status") == "completed" and r.get("conclusion") == "success"
    ]
    if not successful:
        return None
    return max(successful, key=lambda r: r["createdAt"])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 9 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/pull_ci_results.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: add build_filename and select_run helpers"
```

---

### Task 3: Parsing helpers — `parse_commit_list`, `find_result_json`, `read_eval_id`

**Files:**
- Modify: `scripts_repo/pull_ci_results.py`
- Test: `scripts_repo/tests/test_pull_ci_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts_repo/tests/test_pull_ci_results.py`:

```python
from scripts_repo.pull_ci_results import (
    find_result_json,
    parse_commit_list,
    read_eval_id,
)


def test_parse_commit_list_strips_blank_lines():
    output = "abc123\n\ndef456\n  \n789ghi\n"
    assert parse_commit_list(output) == ["abc123", "def456", "789ghi"]


def test_parse_commit_list_empty():
    assert parse_commit_list("") == []


def test_find_result_json_finds_nested_single_file(tmp_path):
    nested = tmp_path / "results" / "ci"
    nested.mkdir(parents=True)
    target = nested / "latest.json"
    target.write_text("{}")
    assert find_result_json(tmp_path) == target


def test_find_result_json_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_result_json(tmp_path)


def test_find_result_json_raises_when_ambiguous(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    with pytest.raises(ValueError):
        find_result_json(tmp_path)


def test_read_eval_id_returns_eval_id(tmp_path):
    f = tmp_path / "result.json"
    f.write_text('{"evalId": "eval-XYZ-2026-05-14T02:56:33", "results": {}}')
    assert read_eval_id(f) == "eval-XYZ-2026-05-14T02:56:33"


def test_read_eval_id_raises_when_absent(tmp_path):
    f = tmp_path / "result.json"
    f.write_text('{"results": {}}')
    with pytest.raises(ValueError):
        read_eval_id(f)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `ImportError: cannot import name 'find_result_json'`

- [ ] **Step 3: Write the implementation**

Append to `scripts_repo/pull_ci_results.py`:

```python
def parse_commit_list(git_log_output: str) -> list[str]:
    """Parse newline-separated SHAs from `git log --format=%H` output."""
    return [line.strip() for line in git_log_output.splitlines() if line.strip()]


def find_result_json(download_dir: Path) -> Path:
    """Locate the single result JSON inside a downloaded artifact directory.

    `gh run download` may nest the file under the path it was uploaded with,
    so the search is recursive. Exactly one .json file is expected.
    """
    matches = sorted(download_dir.rglob("*.json"))
    if len(matches) == 0:
        raise FileNotFoundError(
            f"no .json file found in artifact at {download_dir}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"expected exactly one .json in artifact, found {len(matches)}: "
            f"{[str(m) for m in matches]}"
        )
    return matches[0]


def read_eval_id(json_path: Path) -> str:
    """Read the top-level evalId from a promptfoo result JSON file."""
    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)
    eval_id = data.get("evalId")
    if not eval_id:
        raise ValueError(f"no evalId in result file {json_path}")
    return eval_id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 16 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/pull_ci_results.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: add commit-list, artifact-json, and eval-id parsing helpers"
```

---

### Task 4: `classify_import_failure`

**Files:**
- Modify: `scripts_repo/pull_ci_results.py`
- Test: `scripts_repo/tests/test_pull_ci_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts_repo/tests/test_pull_ci_results.py`:

```python
from scripts_repo.pull_ci_results import classify_import_failure


def test_classify_import_failure_benign_when_id_exists():
    assert classify_import_failure("eval-1", ["eval-0", "eval-1"]) == "benign"


def test_classify_import_failure_genuine_when_id_absent():
    assert classify_import_failure("eval-1", ["eval-0", "eval-2"]) == "genuine"


def test_classify_import_failure_genuine_when_no_existing_ids():
    assert classify_import_failure("eval-1", []) == "genuine"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `ImportError: cannot import name 'classify_import_failure'`

- [ ] **Step 3: Write the implementation**

Append to `scripts_repo/pull_ci_results.py`:

```python
def classify_import_failure(eval_id: str, existing_ids: list[str]) -> str:
    """Classify why a `promptfoo import` failed.

    Returns "benign" when the eval ID is already in the promptfoo DB (a
    duplicate import — the expected, ignorable failure), "genuine" otherwise.
    """
    return "benign" if eval_id in existing_ids else "genuine"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 19 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/pull_ci_results.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: add import-failure classification"
```

---

### Task 5: Subprocess wrappers

Thin wrappers around `gh`, `git`, and `promptfoo`. They share a `_run` helper. One representative wrapper (`gh_runs_for_commit`) is tested via `monkeypatch` to validate the shared pattern (JSON parse + returncode check); the rest follow the same shape.

**Files:**
- Modify: `scripts_repo/pull_ci_results.py`
- Test: `scripts_repo/tests/test_pull_ci_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts_repo/tests/test_pull_ci_results.py`:

```python
from scripts_repo import pull_ci_results as pcr


def test_gh_runs_for_commit_parses_json(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, '[{"databaseId": 7, "headSha": "abc"}]', ""
        )

    monkeypatch.setattr(pcr, "_run", fake_run)
    runs = pcr.gh_runs_for_commit("abc")
    assert runs == [{"databaseId": 7, "headSha": "abc"}]


def test_gh_runs_for_commit_raises_on_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "gh: not authenticated")

    monkeypatch.setattr(pcr, "_run", fake_run)
    with pytest.raises(RuntimeError, match="gh run list failed"):
        pcr.gh_runs_for_commit("abc")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `AttributeError: module 'scripts_repo.pull_ci_results' has no attribute '_run'`

- [ ] **Step 3: Write the implementation**

Append to `scripts_repo/pull_ci_results.py`:

```python
def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout/stderr as text."""
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def gh_runs_for_commit(sha: str) -> list[dict]:
    """All workflow runs for a given commit SHA."""
    result = _run([
        "gh", "run", "list",
        "--commit", sha,
        "--workflow", WORKFLOW,
        "--json", "databaseId,headSha,createdAt,status,conclusion",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"gh run list failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gh_recent_runs(limit: int = 20) -> list[dict]:
    """The most recent workflow runs (newest first)."""
    result = _run([
        "gh", "run", "list",
        "--workflow", WORKFLOW,
        "--limit", str(limit),
        "--json", "databaseId,headSha,createdAt,status,conclusion",
    ])
    if result.returncode != 0:
        raise RuntimeError(f"gh run list failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gh_download_artifact(run_id: int, dest_dir: Path) -> None:
    """Download the results artifact for a run into dest_dir."""
    result = _run([
        "gh", "run", "download", str(run_id),
        "--name", ARTIFACT,
        "--dir", str(dest_dir),
    ])
    if result.returncode != 0:
        raise RuntimeError(
            f"gh run download failed for run {run_id}: {result.stderr.strip()}"
        )


def git_commits_in_range(start: date | None, end: date | None) -> list[str]:
    """Commit SHAs on the current branch within [start, end] by commit date."""
    cmd = ["git", "log", "--format=%H"]
    if start:
        cmd.append(f"--since={start.isoformat()}")
    if end:
        cmd.append(f"--until={end.isoformat()} 23:59:59")
    result = _run(cmd)
    if result.returncode != 0:
        raise RuntimeError(f"git log failed: {result.stderr.strip()}")
    return parse_commit_list(result.stdout)


def git_commit_date(sha: str) -> str:
    """The commit date (YYYY-MM-DD) of a SHA."""
    result = _run(["git", "show", "-s", "--format=%cd", "--date=short", sha])
    if result.returncode != 0:
        raise RuntimeError(f"git show failed for {sha}: {result.stderr.strip()}")
    return result.stdout.strip()


def promptfoo_import(json_path: Path) -> subprocess.CompletedProcess:
    """Run `promptfoo import` on a result file. Caller inspects returncode."""
    return _run(["pnpm", "exec", "promptfoo", "import", str(json_path)])


def promptfoo_eval_ids() -> list[str]:
    """All eval IDs currently in the local promptfoo DB.

    `pnpm exec promptfoo` prints a version banner; filter to lines that look
    like eval IDs (they start with "eval-").
    """
    result = _run(["pnpm", "exec", "promptfoo", "list", "evals", "--ids-only"])
    if result.returncode != 0:
        raise RuntimeError(
            f"promptfoo list evals failed: {result.stderr.strip()}"
        )
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("eval-")
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 21 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/pull_ci_results.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: add gh/git/promptfoo subprocess wrappers"
```

---

### Task 6: `import_results` orchestration

The import phase: try `promptfoo import`; on failure, classify via `promptfoo_eval_ids` and either warn-and-skip (benign) or abort (genuine, or benign under `--fail-on-existing`).

**Files:**
- Modify: `scripts_repo/pull_ci_results.py`
- Test: `scripts_repo/tests/test_pull_ci_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts_repo/tests/test_pull_ci_results.py`:

```python
def _ok(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _fail(stderr):
    def _inner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", stderr)
    return _inner


def test_import_results_imports_new_file(monkeypatch, tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-new"}')
    monkeypatch.setattr(pcr, "promptfoo_import", lambda p: _ok([]))
    # Should not raise.
    pcr.import_results([f], fail_on_existing=False)


def test_import_results_skips_benign_duplicate(monkeypatch, tmp_path, capsys):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-dup"}')
    monkeypatch.setattr(pcr, "promptfoo_import", _fail("already exists"))
    monkeypatch.setattr(pcr, "promptfoo_eval_ids", lambda: ["eval-dup"])
    pcr.import_results([f], fail_on_existing=False)  # should not raise
    assert "already imported" in capsys.readouterr().err


def test_import_results_aborts_on_existing_when_strict(monkeypatch, tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-dup"}')
    monkeypatch.setattr(pcr, "promptfoo_import", _fail("already exists"))
    monkeypatch.setattr(pcr, "promptfoo_eval_ids", lambda: ["eval-dup"])
    with pytest.raises(SystemExit):
        pcr.import_results([f], fail_on_existing=True)


def test_import_results_aborts_on_genuine_error(monkeypatch, tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-new"}')
    monkeypatch.setattr(pcr, "promptfoo_import", _fail("disk full"))
    monkeypatch.setattr(pcr, "promptfoo_eval_ids", lambda: ["eval-other"])
    with pytest.raises(SystemExit):
        pcr.import_results([f], fail_on_existing=False)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `AttributeError: module 'scripts_repo.pull_ci_results' has no attribute 'import_results'`

- [ ] **Step 3: Write the implementation**

Append to `scripts_repo/pull_ci_results.py`:

```python
def import_results(paths: list[Path], fail_on_existing: bool) -> None:
    """Import each result file into promptfoo.

    Try `promptfoo import`; on failure, classify it. A benign duplicate is
    warned-and-skipped (or aborts when fail_on_existing). A genuine error
    aborts immediately, leaving already-downloaded files on disk.
    """
    for path in paths:
        result = promptfoo_import(path)
        if result.returncode == 0:
            print(f"imported {path.name}")
            continue

        eval_id = read_eval_id(path)
        kind = classify_import_failure(eval_id, promptfoo_eval_ids())
        if kind == "benign":
            if fail_on_existing:
                print(
                    f"ERROR: {path.name} (eval {eval_id}) already imported",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(
                f"WARNING: {path.name} (eval {eval_id}) already imported, "
                f"skipping",
                file=sys.stderr,
            )
        else:
            print(
                f"ERROR: import of {path.name} failed:\n"
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
            sys.exit(1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 25 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts_repo/pull_ci_results.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: add import_results orchestration"
```

---

### Task 7: `download_and_name`, mode handlers, and `main`

Ties everything together: download a run's artifact and rename it; the three mode handlers; `argparse` and dispatch. `download_and_name` and the mode handlers shell out, so the tests here focus on `main`'s argument validation and dispatch via `monkeypatch`; `download_and_name` is exercised in Task 9's manual verification.

**Files:**
- Modify: `scripts_repo/pull_ci_results.py`
- Test: `scripts_repo/tests/test_pull_ci_results.py`

- [ ] **Step 1: Write the failing tests**

Append to `scripts_repo/tests/test_pull_ci_results.py`:

```python
def test_main_rejects_date_flags_without_all(monkeypatch):
    monkeypatch.setattr(pcr, "mode_latest", lambda: [])
    monkeypatch.setattr(pcr, "import_results", lambda paths, fail_on_existing: None)
    rc = pcr.main(["--latest", "--start-date", "2026-01-01"])
    assert rc == 2


def test_main_dispatches_to_commit_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr(pcr, "mode_commit", lambda sha: seen.setdefault("sha", sha) or [])
    monkeypatch.setattr(pcr, "import_results", lambda paths, fail_on_existing: None)
    rc = pcr.main(["--commit", "abc123"])
    assert rc == 0
    assert seen["sha"] == "abc123"


def test_main_no_import_skips_import_phase(monkeypatch):
    called = []
    monkeypatch.setattr(pcr, "mode_latest", lambda: [])
    monkeypatch.setattr(pcr, "import_results", lambda *a, **k: called.append(1))
    rc = pcr.main(["--latest", "--no-import"])
    assert rc == 0
    assert called == []


def test_main_requires_a_mode(monkeypatch):
    with pytest.raises(SystemExit):
        pcr.main([])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: FAIL — `AttributeError: module 'scripts_repo.pull_ci_results' has no attribute 'main'`

- [ ] **Step 3: Write the implementation**

Append to `scripts_repo/pull_ci_results.py`:

```python
def download_and_name(run: dict) -> Path:
    """Download a run's artifact and move it to its final results/ci/ path.

    Returns the final path: results/ci/<commit-date>_<short-sha>.json.
    """
    sha = run["headSha"]
    commit_date = git_commit_date(sha)
    final_path = DEST_DIR / build_filename(commit_date, sha)
    DEST_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        gh_download_artifact(run["databaseId"], tmp_dir)
        result_json = find_result_json(tmp_dir)
        shutil.move(str(result_json), str(final_path))
    return final_path


def mode_commit(sha: str) -> list[Path]:
    """Pull the result for one specific commit."""
    run = select_run(gh_runs_for_commit(sha))
    if run is None:
        print(
            f"ERROR: no successful CI run found for commit {sha}",
            file=sys.stderr,
        )
        sys.exit(1)
    return [download_and_name(run)]


def mode_latest() -> list[Path]:
    """Pull the result for the most recent successful CI run."""
    run = select_run(gh_recent_runs())
    if run is None:
        print("ERROR: no successful CI run found", file=sys.stderr)
        sys.exit(1)
    return [download_and_name(run)]


def mode_all(start: date | None, end: date | None) -> list[Path]:
    """Pull results for all commits in the date range that have a CI run."""
    commits = git_commits_in_range(start, end)
    if not commits:
        print("WARNING: no commits in the given date range", file=sys.stderr)
        return []
    paths: list[Path] = []
    for sha in commits:
        run = select_run(gh_runs_for_commit(sha))
        if run is None:
            print(
                f"WARNING: no successful CI run for {sha[:7]}, skipping",
                file=sys.stderr,
            )
            continue
        paths.append(download_and_name(run))
    return paths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--commit", metavar="SHA",
                      help="Pull the result for one specific commit.")
    mode.add_argument("--latest", action="store_true",
                      help="Pull the result for the most recent CI run.")
    mode.add_argument("--all", action="store_true",
                      help="Pull results for all commits, optionally date-gated.")
    parser.add_argument("--start-date", type=parse_date, default=None,
                        help="Earliest commit date (YYYY-MM-DD) for --all.")
    parser.add_argument("--end-date", type=parse_date, default=None,
                        help="Latest commit date (YYYY-MM-DD) for --all.")
    parser.add_argument("--no-import", action="store_true",
                        help="Download only; skip the promptfoo import phase.")
    parser.add_argument("--fail-on-existing", action="store_true",
                        help="Abort if a result's eval ID is already imported.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if (args.start_date or args.end_date) and not args.all:
        print(
            "ERROR: --start-date/--end-date only apply to --all",
            file=sys.stderr,
        )
        return 2

    if args.commit:
        paths = mode_commit(args.commit)
    elif args.latest:
        paths = mode_latest()
    else:
        paths = mode_all(args.start_date, args.end_date)

    for p in paths:
        print(f"downloaded {p}")

    if args.no_import:
        return 0

    import_results(paths, fail_on_existing=args.fail_on_existing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest scripts_repo/tests/test_pull_ci_results.py -v`
Expected: PASS — 29 passed.

- [ ] **Step 5: Run the full Python test suite to confirm no regressions**

Run: `python3 -m pytest -q`
Expected: PASS — 53 passed (24 existing + 29 new).

- [ ] **Step 6: Commit**

```bash
git add scripts_repo/pull_ci_results.py scripts_repo/tests/test_pull_ci_results.py
git commit -m "feat: add download_and_name, mode handlers, and CLI entrypoint"
```

---

### Task 8: Relocate result output paths

Mechanical edits — no tests. Local-run outputs move to `results/local/`, CI output moves to `results/ci/`.

**Files:**
- Modify: `promptfooconfig.yaml`
- Modify: `promptfooconfig.ci.yaml`
- Modify: `scripts_test/hack.sh`
- Modify: `scripts_test/smoke.sh`
- Modify: `scripts_test/full.sh`
- Modify: `package.json`
- Modify: `.github/workflows/promptfoo-gateway.yml`

- [ ] **Step 1: Update `promptfooconfig.yaml`**

Change the last line from:
```yaml
outputPath: results/latest.json
```
to:
```yaml
outputPath: results/local/latest.json
```

- [ ] **Step 2: Update `promptfooconfig.ci.yaml`**

Change:
```yaml
outputPath: results/latest.json
```
to:
```yaml
outputPath: results/ci/latest.json
```

- [ ] **Step 3: Update `scripts_test/hack.sh`**

Insert a `mkdir -p results/local` line immediately before the `exec pnpm exec promptfoo eval` line, and change `--output results/latest.json` to `--output results/local/latest.json`. The tail of the file becomes:

```bash
mkdir -p results/local
exec pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers fidaro_plaintext_gateway \
  --filter-pattern "Fidaro system prompt, no capabilities, basic test" \
  --no-cache \
  --output results/local/latest.json
```

- [ ] **Step 4: Update `scripts_test/smoke.sh`**

Insert `mkdir -p results/local` immediately before the `exec pnpm exec promptfoo eval` line, and change `--output results/smoke.json` to `--output results/local/smoke.json`. The tail becomes:

```bash
mkdir -p results/local
# PROVIDER=bedrock_mantle
exec pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --filter-providers "$PROVIDER" \
  --filter-pattern smoke \
  --output results/local/smoke.json
```

- [ ] **Step 5: Update `scripts_test/full.sh`**

Insert `mkdir -p results/local` immediately before the `exec pnpm exec promptfoo eval` line, and change `--output results/latest.json` to `--output results/local/latest.json`. The tail becomes:

```bash
mkdir -p results/local
exec pnpm exec promptfoo eval \
  --config promptfooconfig.yaml \
  --cache \
  --output results/local/latest.json
```

- [ ] **Step 6: Update `package.json`**

Change the `view` script from:
```json
    "view": "promptfoo view results/latest.json"
```
to:
```json
    "view": "promptfoo view results/local/latest.json"
```

- [ ] **Step 7: Update `.github/workflows/promptfoo-gateway.yml`**

In the `Run promptfoo` step, add a `mkdir -p results/ci` line and change the `--output` path. The step's `run:` block becomes:

```yaml
      - name: Run promptfoo
        run: |
          mkdir -p results/ci
          pnpm exec promptfoo eval \
            --config promptfooconfig.ci.yaml \
            --no-cache \
            --output results/ci/latest.json
```

In the `Upload promptfoo results` step, change the artifact path:

```yaml
      - name: Upload promptfoo results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: promptfoo-results
          path: results/ci/latest.json
          if-no-files-found: error
```

Leave the rest of the workflow (including the `ECR_PULLER_ROLE` reference) unchanged.

- [ ] **Step 8: Verify the configs still parse**

Run: `python3 -c "import yaml; yaml.safe_load(open('promptfooconfig.yaml')); yaml.safe_load(open('promptfooconfig.ci.yaml')); yaml.safe_load(open('.github/workflows/promptfoo-gateway.yml')); print('all yaml ok')"`
Expected: `all yaml ok`

Run: `python3 -c "import json; json.load(open('package.json')); print('package.json ok')"`
Expected: `package.json ok`

- [ ] **Step 9: Commit**

```bash
git add promptfooconfig.yaml promptfooconfig.ci.yaml scripts_test/hack.sh scripts_test/smoke.sh scripts_test/full.sh package.json .github/workflows/promptfoo-gateway.yml
git commit -m "refactor: relocate eval output to results/local and results/ci"
```

---

### Task 9: Manual end-to-end verification

The subprocess-driven paths (`download_and_name`, the mode handlers, the real `promptfoo import` failure behavior) can't be unit-tested without live `gh`/`promptfoo` state. Verify them manually. Requires `gh` to be authenticated and at least one completed CI run to exist for the workflow.

- [ ] **Step 1: Confirm `gh` is authenticated**

Run: `gh auth status`
Expected: shows a logged-in account, no error.

- [ ] **Step 2: Check at least one CI run exists**

Run: `gh run list --workflow promptfoo-gateway.yml --limit 5`
Expected: at least one run listed. If none exist yet, the puller can't be verified end-to-end — note this and stop here; the unit tests still stand.

- [ ] **Step 3: Download-only, latest mode**

Run: `python3 scripts_repo/pull_ci_results.py --latest --no-import`
Expected: prints `downloaded results/ci/<date>_<sha>.json`; that file exists; no promptfoo DB writes.

- [ ] **Step 4: Latest mode with import**

Run: `python3 scripts_repo/pull_ci_results.py --latest`
Expected: prints `downloaded ...` then `imported <file>` (first time) — OR, if that eval was already imported, prints a `WARNING: ... already imported, skipping`. Either is a pass; confirm the behavior matches whether the eval was new.

- [ ] **Step 5: Re-run to confirm benign-duplicate handling**

Run: `python3 scripts_repo/pull_ci_results.py --latest`
Expected: prints `WARNING: ... already imported, skipping`, exits 0.

> **If Step 5 instead shows `imported ...` or a genuine error:** `promptfoo import` of a duplicate did not fail the way the design assumed. Inspect the actual behavior: run `pnpm exec promptfoo import results/ci/<file>` directly and observe the exit code and stderr. If duplicate import *succeeds* (exit 0), the design's "try then classify" approach needs revisiting — stop and report back rather than patching around it.

- [ ] **Step 6: Strict mode on the existing eval**

Run: `python3 scripts_repo/pull_ci_results.py --latest --fail-on-existing`
Expected: prints `ERROR: ... already imported`, exits non-zero (`echo $?` shows 1).

- [ ] **Step 7: Specific-commit mode, no run**

Run: `python3 scripts_repo/pull_ci_results.py --commit 0000000000000000000000000000000000000000`
Expected: prints `ERROR: no successful CI run found for commit ...`, exits non-zero.

- [ ] **Step 8: All mode with a narrow date range**

Run: `python3 scripts_repo/pull_ci_results.py --all --start-date <a date with no commits, e.g. 2000-01-01> --end-date 2000-01-02 --no-import`
Expected: prints `WARNING: no commits in the given date range`, exits 0.

- [ ] **Step 9: Local-run output path**

Run: `PROVIDER=bedrock_mantle ./scripts_test/smoke.sh` (or any provider that's reachable)
Expected: the run writes to `results/local/smoke.json` (the new path), and `results/local/` was created.

> If no provider is reachable, skip this step and just confirm the `--output` path in the script is `results/local/smoke.json` by inspection.

- [ ] **Step 10: Final commit (if anything needed touch-ups)**

If Steps 1-9 surfaced bugs, fix them with focused commits referencing the failing step. If everything passed clean, there is nothing to commit here — the implementation commits from Tasks 1-8 stand.

---

## Self-Review

**Spec coverage:**
- Three modes (`--commit` / `--latest` / `--all`) — Task 7. ✓
- Date gating on `--all` with `YYYY-MM-DD` and obvious defaults — Task 1 (`parse_date`), Task 5 (`git_commits_in_range` defaults), Task 7. ✓
- Hardcoded workflow/artifact, no override — `WORKFLOW`/`ARTIFACT` constants, Task 1. ✓
- Download to `results/ci/`, filename `<commit-date>_<short-sha>.json` — Task 2 (`build_filename`), Task 7 (`download_and_name`). ✓
- Commit-driven discovery, most-recent-successful run — Task 2 (`select_run`), Task 7 (mode handlers). ✓
- Resilient artifact JSON location — Task 3 (`find_result_json`). ✓
- Import phase: try-then-classify, warn+skip benign, abort genuine, `--fail-on-existing`, `--no-import` — Task 4, Task 6, Task 7. ✓
- Edge cases (no run, multiple runs, no commits in range, bad dates, gh not authed) — covered across Tasks 2/5/7 and verified in Task 9. ✓
- Codebase relocation (local → `results/local/`, CI → `results/ci/`) — Task 8. ✓

**Placeholder scan:** Task 9 Steps 8 uses an intentional human-supplied value ("a date with no commits") — this is a manual verification step, not code. No code placeholders.

**Type consistency:** `select_run` returns `dict | None`, consumed with `None` checks in all three mode handlers. `download_and_name` takes a `run` dict and returns `Path`; mode handlers collect `list[Path]`; `import_results` and `main` consume `list[Path]`. `promptfoo_import` returns `CompletedProcess`, inspected by `.returncode`/`.stderr` in `import_results`. Names consistent across tasks.

**Artifact JSON path note:** The artifact currently contains `results/latest.json` internally; after Task 8 it will contain `results/ci/latest.json`. `find_result_json` uses `rglob("*.json")` so it tolerates either nesting — no ordering dependency between Task 7 and Task 8.

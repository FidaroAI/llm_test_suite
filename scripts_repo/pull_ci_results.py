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
        raise argparse.ArgumentTypeError(f"invalid date {value!r}: expected YYYY-MM-DD")


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
        r
        for r in runs
        if r.get("status") == "completed" and r.get("conclusion") == "success"
    ]
    if not successful:
        return None
    return max(successful, key=lambda r: r["createdAt"])


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
        raise FileNotFoundError(f"no .json file found in artifact at {download_dir}")
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


def classify_import_failure(eval_id: str, existing_ids: list[str]) -> str:
    """Classify why a `promptfoo import` failed.

    Returns "benign" when the eval ID is already in the promptfoo DB (a
    duplicate import — the expected, ignorable failure), "genuine" otherwise.
    """
    return "benign" if eval_id in existing_ids else "genuine"


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command, capturing stdout/stderr as text."""
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def gh_runs_for_commit(sha: str) -> list[dict]:
    """All workflow runs for a given commit SHA."""
    result = _run(
        [
            "gh",
            "run",
            "list",
            "--commit",
            sha,
            "--workflow",
            WORKFLOW,
            "--json",
            "databaseId,headSha,createdAt,status,conclusion",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh run list failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gh_recent_runs(limit: int = 20) -> list[dict]:
    """The most recent workflow runs (newest first)."""
    result = _run(
        [
            "gh",
            "run",
            "list",
            "--workflow",
            WORKFLOW,
            "--limit",
            str(limit),
            "--json",
            "databaseId,headSha,createdAt,status,conclusion",
        ]
    )
    if result.returncode != 0:
        raise RuntimeError(f"gh run list failed: {result.stderr.strip()}")
    return json.loads(result.stdout)


def gh_download_artifact(run_id: int, dest_dir: Path) -> None:
    """Download the results artifact for a run into dest_dir."""
    result = _run(
        [
            "gh",
            "run",
            "download",
            str(run_id),
            "--name",
            ARTIFACT,
            "--dir",
            str(dest_dir),
        ]
    )
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
        raise RuntimeError(f"promptfoo list evals failed: {result.stderr.strip()}")
    return [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip().startswith("eval-")
    ]


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
                f"WARNING: {path.name} (eval {eval_id}) already imported, " f"skipping",
                file=sys.stderr,
            )
        else:
            print(
                f"ERROR: import of {path.name} failed:\n" f"{result.stderr.strip()}",
                file=sys.stderr,
            )
            sys.exit(1)


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
    mode.add_argument(
        "--commit", metavar="SHA", help="Pull the result for one specific commit."
    )
    mode.add_argument(
        "--latest",
        action="store_true",
        help="Pull the result for the most recent CI run.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Pull results for all commits, optionally date-gated.",
    )
    parser.add_argument(
        "--start-date",
        type=parse_date,
        default=None,
        help="Earliest commit date (YYYY-MM-DD) for --all.",
    )
    parser.add_argument(
        "--end-date",
        type=parse_date,
        default=None,
        help="Latest commit date (YYYY-MM-DD) for --all.",
    )
    parser.add_argument(
        "--no-import",
        action="store_true",
        help="Download only; skip the promptfoo import phase.",
    )
    parser.add_argument(
        "--fail-on-existing",
        action="store_true",
        help="Abort if a result's eval ID is already imported.",
    )
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

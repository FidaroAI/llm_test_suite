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


def freeze(result_path, out_dir: Path, force: bool) -> list[Path]:
    """Write one baseline file per provider; return the written paths."""
    parsed = json.loads(Path(result_path).read_text(encoding="utf-8"))
    labels = provider_labels(parsed)
    if not labels:
        raise ValueError(f"no provider labels found in {result_path}")
    sha = git_sha()
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for label in labels:
        dest = out_dir / f"{sanitize_label(label)}.json"
        if dest.exists() and not force:
            raise FileExistsError(f"{dest} exists; use --force to overwrite")
        baseline = build_baseline(parsed, label, sha)
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

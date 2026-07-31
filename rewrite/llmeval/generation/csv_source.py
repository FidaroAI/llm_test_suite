"""Generate standardized test cases from a CSV source.

CSV shape (matches the legacy suite): a prompt column (default ``user``), an
``__expected`` column holding a deterministic-assertion shorthand (``icontains:Paris``),
and optional ``__metadata:<key>`` columns carried into test metadata.
"""

from __future__ import annotations

import csv as _csv
import hashlib
import json
import os
from typing import Any

from llmeval.generation.common import local_id
from llmeval.models import AssertionSpec

# Deterministic-assertion shorthands accepted in the __expected column.
_KNOWN = {"contains", "icontains", "equals", "regex", "not_contains"}

_METADATA_PREFIX = "__metadata:"


def parse_expected(expr: str) -> AssertionSpec:
    """Parse ``"icontains:Paris"`` -> an AssertionSpec. Unknown types are rejected."""
    type_, sep, value = expr.partition(":")
    if not sep or type_ not in _KNOWN:
        raise ValueError(
            f"unsupported expected shorthand: {expr!r} (known: {sorted(_KNOWN)}). "
            "Custom/LLM assertions must be authored as full assertion objects."
        )
    return AssertionSpec(type=type_, value=value)


def rows_from_csv(
    csv_path: str,
    prompt_col: str = "user",
    expected_col: str = "__expected",
) -> list[dict[str, Any]]:
    """Parse a suite CSV into test-case dicts with **local** ids.

    CSV shape (unchanged from the legacy suite): a prompt column, an ``__expected`` column
    holding a deterministic-assertion shorthand (``icontains:Paris``), and any number of
    ``__metadata:<key>`` columns carried into the case's metadata.

    No ``suite`` key is stamped: a suite *is* a plugin now, and the id prefix records it.
    """
    cases: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8-sig") as fh:
        for row in _csv.DictReader(fh):
            prompt = (row.get(prompt_col) or "").strip()
            if not prompt:
                continue
            metadata = {
                col[len(_METADATA_PREFIX):]: val
                for col, val in row.items()
                if col and col.startswith(_METADATA_PREFIX) and val != ""
            }
            assertions = []
            expected = row.get(expected_col)
            if expected:
                assertions.append(parse_expected(expected).model_dump(exclude_defaults=True))
            cases.append(
                {
                    "id": local_id(prompt),
                    "user": prompt,
                    "assertions": assertions,
                    "metadata": metadata,
                }
            )
    return cases


def _stable_id(suite: str, prompt: str) -> str:
    return f"{suite}-{hashlib.sha1(prompt.strip().encode('utf-8')).hexdigest()[:10]}"


def generate_from_csv(
    csv_path: str,
    suite: str,
    out_dir: str | None = None,
    prompt_col: str = "user",
    expected_col: str = "__expected",
    classifications: dict[str, dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    classifications = classifications or {}
    cases: list[dict[str, Any]] = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            prompt = row[prompt_col]
            metadata: dict[str, Any] = {"suite": suite}
            for col, val in row.items():
                if col and col.startswith(_METADATA_PREFIX) and val != "":
                    metadata[col[len(_METADATA_PREFIX):]] = val
            labels = classifications.get(prompt.strip()) or classifications.get(prompt) or {}
            metadata.setdefault("request_type", labels.get("request_type", "unclassified"))
            metadata.setdefault("domain", labels.get("domain", "unclassified"))

            assertions = []
            expected = row.get(expected_col)
            if expected:
                assertions.append(parse_expected(expected).model_dump(exclude_defaults=True))

            cases.append(
                {
                    "id": _stable_id(suite, prompt),
                    "user": prompt,
                    "assertions": assertions,
                    "metadata": metadata,
                }
            )

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{suite}.json"), "w", encoding="utf-8") as f:
            json.dump(cases, f, indent=2, ensure_ascii=False)
    return cases

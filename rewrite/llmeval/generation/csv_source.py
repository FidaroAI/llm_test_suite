"""Parse a CSV source into test-case dicts.

CSV shape (matches the legacy suite): a prompt column (default ``user``), an
``__expected`` column holding a deterministic-assertion shorthand (``icontains:Paris``),
and optional ``__metadata:<key>`` columns carried into test metadata.

Used by :mod:`llmeval.generation.csv_plugin`, which is the whole of a CSV-backed plugin.
"""

from __future__ import annotations

import csv as _csv
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

"""Load standardized test cases for running/grading.

Reads a single ``.json`` file or a directory of them. Each file holds a test case or a
list of test cases. Optional metadata ``filters`` keep only matching cases (e.g.
``{"suite": "simple_facts"}`` or ``{"request_type": "coding"}``) — the run-time slicing
the brief wants.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Sequence, TypeVar

from llmeval.models import TestCase

T = TypeVar("T")


def _read_file(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    return doc if isinstance(doc, list) else [doc]


def load_testcases(path: str, filters: dict[str, Any] | None = None) -> list[TestCase]:
    raw: list[dict[str, Any]] = []
    if os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            if name.endswith(".json"):
                raw.extend(_read_file(os.path.join(path, name)))
    else:
        raw.extend(_read_file(path))

    cases = [TestCase.from_dict(d) for d in raw]
    if filters:
        cases = [c for c in cases if all(c.metadata.get(k) == v for k, v in filters.items())]
    return cases


def select_testcases(
    testcases: Sequence[T],
    limit: int | None = None,
    randomize: bool = False,
    seed: int = 0,
) -> list[T]:
    """Pick a subset to run: optionally shuffle (seeded, default 0) then cap to ``limit``.

    Shuffle-then-cap means ``randomize`` + ``limit`` yields a reproducible random sample.
    Does not mutate the input.
    """
    out = list(testcases)
    if randomize:
        random.Random(seed).shuffle(out)
    if limit is not None:
        out = out[:limit]
    return out

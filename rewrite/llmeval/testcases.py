"""Load standardized test cases for running/grading.

Reads a single ``.json`` file or a directory of them. Each file holds a test case or a
list of test cases. Optional metadata ``filters`` keep only matching cases (e.g.
``{"suite": "simple_facts"}`` or ``{"request_type": "coding"}``) — the run-time slicing
the brief wants.

:func:`load_all_testcases` takes several paths at once, which is what the repeatable
``--testcases`` flag passes in.
"""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Sequence, TypeVar

from llmeval.models import TestCase

logger = logging.getLogger(__name__)

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


def load_all_testcases(
    paths: Sequence[str], filters: dict[str, Any] | None = None
) -> list[TestCase]:
    """Load several paths as one set, **de-duplicated by test id** (first occurrence wins).

    ``--testcases`` is repeatable so a caller can say "these three suite files but not the
    other two", which no single path expresses. The natural ways to use it overlap — a
    directory plus one file inside it, or two directories sharing a suite — so an id seen
    twice is dropped rather than run twice. Order is otherwise the order given.

    Paths are read in the order supplied; within a directory, files are read in sorted
    order (see :func:`load_testcases`), so the result is deterministic.
    """
    seen: set[str] = set()
    out: list[TestCase] = []
    duplicates = 0
    for path in paths:
        for case in load_testcases(path, filters):
            if case.id in seen:
                duplicates += 1
                continue
            seen.add(case.id)
            out.append(case)
    if duplicates:
        # Debug, not a warning: overlapping paths are a normal way to ask for a set, and
        # the de-duplication is the feature rather than a recovery from user error.
        logger.debug(
            "dropped %d duplicate test case(s) across %d path(s)", duplicates, len(paths)
        )
    return out


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

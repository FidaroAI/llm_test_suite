"""Load test cases from ``testcases/`` and pick a subset to run.

Loading proper lives in :mod:`llmeval.plugins.loader` — a source is a plugin directory or a
top-level ``.json`` file, and both come back as :class:`~llmeval.models.TestCase` objects with
``<source>.<local id>`` ids. This module re-exports it so the stages have one obvious import,
and adds the run-time subsetting (``--limit`` / ``--randomize`` / ``--seed``) that is not the
loader's business.
"""

from __future__ import annotations

import random
from typing import Sequence, TypeVar

from llmeval.plugins.loader import (
    DEFAULT_ROOT,
    Loaded,
    SourceError,
    load as load_testcases,
    select_sources,
)

__all__ = [
    "DEFAULT_ROOT",
    "Loaded",
    "SourceError",
    "load_testcases",
    "select_sources",
    "select_testcases",
]

T = TypeVar("T")


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

"""Suite-generation config + selection — ported from the legacy ``tests/suite_config.py``.

A single JSON file, keyed by suite name, controls how many tests each generator
emits and how they are sampled. Each suite block maps to::

    number_to_generate  int | null   cap on emitted tests (null = all)
    randomize_selection bool         shuffle before capping
    random_seed         int          seed for the shuffle (default 0)
    max_rubrics         int | null   cap rubrics per row (null = all)
    stratify            obj | null   take a quota per classification (null = off)

``stratify`` draws an even sample across a ``metadata`` dimension instead of an
undifferentiated cap::

    {"by": "domain", "per_group": 3, "groups": ["finance_business", "coding"]}

A reserved ``default`` key is a suite-shaped fallback for any suite *not* listed
by name (not an overlay onto listed suites). A missing file or missing suite key
falls back to :data:`DEFAULTS`, which leaves a suite *off* (``number_to_generate``
of 0) so a new generator never floods a run until explicitly opted in.

Operates on the rewrite's on-disk test-case dict shape
(``{"id", "user", "assertions", "metadata"}``) — selection reads/stamps the
top-level ``metadata``.
"""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any

# Off by default: an explicit null in the config means "all rows"; this 0 means
# "none", so an un-listed suite emits nothing.
DEFAULTS: dict[str, Any] = {
    "number_to_generate": 0,
    "randomize_selection": False,
    "random_seed": 0,
    "max_rubrics": None,
    "stratify": None,
}

# Env var naming the config file, matching the legacy generators.
CONFIG_ENV_VAR = "SUITE_GENERATION_CONFIG_FILE"


class SuiteGenConfig:
    """Resolved generation config for a single suite."""

    def __init__(self, suite: str, values: dict[str, Any]):
        self.suite = suite
        self._values = values
        self.number_to_generate = values["number_to_generate"]
        self.randomize_selection = values["randomize_selection"]
        self.random_seed = values["random_seed"]
        self.max_rubrics = values["max_rubrics"]
        self.stratify = values["stratify"]

    def _stratified(self, tests: list[dict]) -> list[dict]:
        """Keep ``per_group`` tests from each group of ``metadata[by]``.

        Operates on the already-(optionally)-shuffled list, so per-group picks are
        random-but-reproducible. Group order follows the explicit ``groups`` list
        if given, else first-seen order.
        """
        by = self.stratify["by"]
        per_group = self.stratify["per_group"]
        wanted = self.stratify.get("groups")

        kept: dict[Any, list[dict]] = {}
        order = list(wanted) if wanted else []
        for test in tests:
            value = (test.get("metadata") or {}).get(by)
            if wanted is not None and value not in wanted:
                continue
            bucket = kept.setdefault(value, [])
            if value not in order:
                order.append(value)
            if len(bucket) < per_group:
                bucket.append(test)
        return [test for value in order for test in kept.get(value, [])]

    def select(self, tests: list[dict]) -> list[dict]:
        """Apply selection (shuffle, stratify, cap), stamping config on output."""
        chosen = list(tests)
        if self.randomize_selection:
            random.Random(self.random_seed).shuffle(chosen)
        if self.stratify:
            chosen = self._stratified(chosen)
        if self.number_to_generate is not None:
            chosen = chosen[: self.number_to_generate]
        for test in chosen:
            test.setdefault("metadata", {})["config"] = dict(self._values)
        return chosen


def _resolve_path(path: str | None) -> Path | None:
    chosen = path or os.environ.get(CONFIG_ENV_VAR)
    return Path(chosen) if chosen else None


def load_suite_config(suite: str, path: str | None = None) -> SuiteGenConfig:
    """Load the resolved :class:`SuiteGenConfig` for ``suite``.

    A listed suite uses its own block; the optional file-level ``default`` block
    is a fallback for un-listed suites only (never merged into a listed one).
    The code :data:`DEFAULTS` fill any field left unspecified. A missing file or
    missing suite key falls back to ``default`` then DEFAULTS.
    """
    resolved = _resolve_path(path)
    all_cfg: dict[str, Any] = {}
    if resolved and resolved.exists():
        all_cfg = json.loads(resolved.read_text(encoding="utf-8"))
    suite_values = all_cfg[suite] if suite in all_cfg else all_cfg.get("default", {})
    values = {**DEFAULTS, **suite_values}
    return SuiteGenConfig(suite, values)

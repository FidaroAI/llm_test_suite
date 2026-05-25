"""Shared configuration + selection logic for the Python test generators.

Every ``tests/<name>_gen.py`` is a suite generator by convention; ``<name>``
(the filename minus the ``_gen`` suffix) is the *suite name*, used both as the
key into the config file and as the test's ``metadata.suite`` value.

Generation is configured by a single JSON file, located via the
``SUITE_GENERATION_CONFIG_FILE`` env var (default
``tests/suite_generation_config.json``). The file is keyed by suite name; each
suite maps to:

    number_to_generate  int | null   cap on emitted tests (null = all)
    randomize_selection bool         shuffle before capping
    random_seed         int          seed for the shuffle (default 0)
    max_rubrics         int | null   cap rubrics per row (null = all)
    stratify            obj | null   take a quota per classification (null = off)

``stratify`` lets a run draw an even sample across a classification dimension
instead of an undifferentiated cap. It is an object::

    {"by": "domain", "per_group": 3, "groups": ["finance_business", "coding"]}

``by`` is the ``metadata`` key to group on (e.g. ``domain`` or
``request_type``), ``per_group`` is how many tests to keep from each group, and
the optional ``groups`` list restricts to (and orders by) those values; omit it
to use every value present. ``number_to_generate`` still applies afterwards as
an overall ceiling.

A missing file or missing suite key falls back to DEFAULTS. Generators build
their candidate tests, drop any with no gradable assertions, then call
``cfg.select(tests)`` which applies the (optionally seeded) random selection,
the optional per-class stratification, caps to ``number_to_generate``, and
stamps the resolved config onto each selected test's ``metadata.config`` so the
run is reproducible.
"""

import json
import os
import random
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "suite_generation_config.json"

DEFAULTS = {
    "number_to_generate": 0,  # None => all rows
    "randomize_selection": False,
    "random_seed": 0,  # used when randomize_selection is on but unset
    "max_rubrics": None,  # None => all rubrics
    "stratify": None,  # None => no per-class quota
}


def suite_name(file):
    """Derive the suite name from a generator's filename (strip ``_gen``)."""
    stem = Path(file).stem
    return stem[: -len("_gen")] if stem.endswith("_gen") else stem


def _config_path():
    return Path(os.environ.get("SUITE_GENERATION_CONFIG_FILE", DEFAULT_CONFIG_PATH))


def _load_all():
    path = _config_path()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


class SuiteConfig:
    """Resolved generation config for a single suite."""

    def __init__(self, suite, values):
        self.suite = suite
        self._values = values
        self.number_to_generate = values["number_to_generate"]
        self.randomize_selection = values["randomize_selection"]
        self.random_seed = values["random_seed"]
        self.max_rubrics = values["max_rubrics"]
        self.stratify = values["stratify"]

    def _stratified(self, tests):
        """Keep ``per_group`` tests from each group of ``metadata[by]``.

        Operates on the already-(optionally)-shuffled list, so per-group picks
        are random-but-reproducible. Group order follows the explicit
        ``groups`` list if given, else first-seen order.
        """
        by = self.stratify["by"]
        per_group = self.stratify["per_group"]
        wanted = self.stratify.get("groups")

        kept = {}  # group value -> list of tests, capped at per_group
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

    def select(self, tests):
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


def load(file):
    """Load the resolved :class:`SuiteConfig` for the generator at ``file``."""
    suite = suite_name(file)
    values = {**DEFAULTS, **_load_all().get(suite, {})}
    return SuiteConfig(suite, values)

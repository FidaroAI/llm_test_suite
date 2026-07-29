"""Shared helpers for the framework tests.

Every result must be attributable to a run, so ``Store.add_result_row`` requires a
``run_id``. Most tests here don't care *which* run a row belongs to — they only need a
valid one — so :func:`a_run` opens a throwaway one.
"""

import logging

import pytest

from llmeval.cache_key import CacheKey
from llmeval.logs import reset_logging
from llmeval.store import Store


def a_run(store: Store, cache_key: CacheKey, **kwargs) -> str:
    """Open a run to hang test results off, and return its id."""
    kwargs.setdefault("provider_name", "test")
    return store.create_run(cache_key, **kwargs)


@pytest.fixture
def root_logging_restored():
    """Let a test call ``configure_logging``, then put the root logger back as it was.

    Without this, one test's configuration would leak into every later test — including
    the root log level, which pytest's own capture relies on.
    """
    root = logging.getLogger()
    saved_level, saved_handlers = root.level, list(root.handlers)
    yield
    reset_logging()
    root.setLevel(saved_level)
    root.handlers = saved_handlers

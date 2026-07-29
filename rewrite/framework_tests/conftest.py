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


def backdate_run(store: Store, run_id: str, started_at: str) -> str:
    """Rewrite a run's ``started_at`` so time-window selection can be tested.

    ``create_run`` deliberately has no ``started_at`` parameter — a run is stamped when it
    opens and nothing legitimate rewrites that. Tests need runs at known times, so they
    reach past the public API rather than the API growing a hole for them.

    :param started_at: an ISO-8601 UTC string, e.g. ``"2026-07-01T09:00:00+00:00"``.
    """
    # pylint: disable=protected-access
    store._conn.execute("UPDATE runs SET started_at=? WHERE id=?", (started_at, run_id))
    store._conn.commit()
    return run_id


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

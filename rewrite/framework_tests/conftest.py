"""Shared helpers for the framework tests.

Every result must be attributable to a run, so ``Store.add_result_row`` requires a
``run_id``. Most tests here don't care *which* run a row belongs to — they only need a
valid one — so :func:`a_run` opens a throwaway one.
"""

from llmeval.cache_key import CacheKey
from llmeval.store import Store


def a_run(store: Store, cache_key: CacheKey, **kwargs) -> str:
    """Open a run to hang test results off, and return its id."""
    kwargs.setdefault("provider_name", "test")
    return store.create_run(cache_key, **kwargs)

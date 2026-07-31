"""Small helpers shared by test-case plugins."""

from __future__ import annotations

import hashlib


def local_id(prompt: str, variant: str | None = None) -> str:
    """A plugin-local test id: ``sha1(prompt)[:10]`` plus an optional ``-<variant>``.

    Local, not global: the loader prefixes ``<source>.`` when it reads the plugin's output, so
    a plugin never spells its own name into an id. Keying on the prompt hash keeps ids stable
    across dataset re-downloads and reordering; the variant suffix disambiguates plugins that
    emit several cases per prompt (research_rubrics' rubric vs g_eval).
    """
    digest = hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:10]
    return digest + (f"-{variant}" if variant else "")

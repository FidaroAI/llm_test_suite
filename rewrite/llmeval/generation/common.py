"""Small helpers shared by test-case plugins."""

from __future__ import annotations

import hashlib
import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# How much of the prompt goes in the "dropped a duplicate" warning. Long enough to recognise
# the test case, short enough that a dataset full of repeats does not fill the terminal.
_SNIPPET_CHARS = 80


def local_id(prompt: str, variant: str | None = None) -> str:
    """A plugin-local test id: ``sha1(prompt)[:10]`` plus an optional ``-<variant>``.

    Local, not global: the loader prefixes ``<source>.`` when it reads the plugin's output, so
    a plugin never spells its own name into an id. Keying on the prompt hash keeps ids stable
    across dataset re-downloads and reordering; the variant suffix disambiguates plugins that
    emit several cases per prompt (research_rubrics' rubric vs g_eval).
    """
    digest = hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:10]
    return digest + (f"-{variant}" if variant else "")


def _snippet(prompt: Any) -> str:
    """A one-line, length-capped version of a prompt, for a log message."""
    text = " ".join(str(prompt or "").split())
    return text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS].rstrip() + "…"


def drop_duplicate_ids(
    cases: Sequence[dict[str, Any]], source: str
) -> list[dict[str, Any]]:
    """``cases`` with later repeats of an already-seen id removed, first occurrence winning.

    Deduplication is the *plugin's* job, not the loader's: only the plugin knows whether two
    cases sharing an id are a mistake or the shape of its source data. :func:`local_id` hashes
    the prompt, so any dataset that asks the same question twice — ``multifaceted`` does,
    repeatedly, each row carrying its own rubrics — produces an id clash. The loader refuses
    to load one, and it does so at *run* time, long after the generate that caused it; drop
    the repeat here instead, noisily, where the fix is.

    Every drop is warned about individually, with a snippet of the prompt, because a suite
    quietly losing two thirds of its rows is exactly the failure this is meant to expose.
    """
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        case_id = str(case.get("id") or "")
        if not case_id:
            # Not this function's problem, and not a duplicate of anything: the loader has a
            # precise complaint for a case with no id, and it should be the one to make it.
            kept.append(case)
            continue
        if case_id in seen:
            logger.warning(
                "%s: dropped duplicate test case id %r (prompt: %s)",
                source, case_id, _snippet(case.get("user")),
            )
            continue
        seen.add(case_id)
        kept.append(case)
    return kept

"""Small helpers shared by the dataset-backed generators."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def local_id(prompt: str, variant: str | None = None) -> str:
    """A plugin-local test id: ``sha1(prompt)[:10]`` plus an optional ``-<variant>``.

    Local, not global: the loader prefixes ``<source>.`` when it reads the plugin's output, so
    a plugin never spells its own name into an id. Keying on the prompt hash keeps ids stable
    across dataset re-downloads and reordering; the variant suffix disambiguates plugins that
    emit several cases per prompt (research_rubrics' rubric vs g_eval).
    """
    digest = hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:10]
    return digest + (f"-{variant}" if variant else "")


def make_id(suite: str, prompt: str, variant: str | None = None) -> str:
    """Stable test id: ``<suite>-<sha1(prompt)[:10]>[-<variant>]``.

    Keying on the prompt hash keeps ids stable across dataset re-downloads. The
    ``variant`` suffix disambiguates suites that emit several tests per prompt
    (e.g. research_rubrics' llm-rubric vs g-eval).
    """
    digest = hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()[:10]
    return f"{suite}-{digest}" + (f"-{variant}" if variant else "")


def load_dataset(data_path: str, download_hint: str) -> list[dict[str, Any]]:
    """Load a dataset JSON file, raising a helpful error if it's missing."""
    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Download the dataset first: {download_hint}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)

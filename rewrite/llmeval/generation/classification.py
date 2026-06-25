"""Shared, suite-independent test classification — ported from ``tests/classification.py``.

Every test is labelled on two orthogonal axes so suites can be sliced consistently:

* ``request_type`` — *what* the user is trying to do (e.g. ``coding``).
* ``domain``       — the subject *area* (e.g. ``finance_business``).

Labels live OUTSIDE the raw datasets, in ``<classifications_dir>/<suite>.json``,
keyed by :func:`prompt_key` (a SHA-1 of the trimmed prompt). This survives dataset
re-downloads/reordering and works for suites whose rows carry no stable id.
Populate the files with the repo's ``scripts_repo/classify_tests.py``.

Unlike the legacy module, this one does **not** attach a promptfoo grading
transform (the rewrite applies ``strip_reasoning`` per-assertion already) nor the
``select-best`` env hook (head-to-head is the separate ``pickbest`` command).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# Stamped when a prompt has no entry in the classification file yet.
UNCLASSIFIED = "unclassified"


def prompt_key(prompt: str) -> str:
    """Stable join key for a test: SHA-1 of the trimmed prompt text."""
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()


def load_classifications(suite: str, classifications_dir: str) -> dict[str, dict[str, str]]:
    """Return ``{prompt_key: {"request_type":.., "domain":..}}`` for a suite.

    A missing file yields an empty mapping (every test falls back to
    ``unclassified``).
    """
    path = Path(classifications_dir) / f"{suite}.json"
    if not path.exists():
        return {}
    doc = json.loads(path.read_text(encoding="utf-8"))
    return doc.get("classifications", {})


def labels_for(prompt: str, mapping: dict[str, dict[str, str]]) -> dict[str, str]:
    """Look up a prompt's labels in ``mapping``, falling back to ``unclassified``."""
    entry = mapping.get(prompt_key(prompt)) or {}
    return {
        "request_type": entry.get("request_type", UNCLASSIFIED),
        "domain": entry.get("domain", UNCLASSIFIED),
    }


def stamp(test: dict[str, Any], prompt: str, mapping: dict[str, dict[str, str]]) -> dict:
    """Stamp ``request_type``/``domain`` into ``test['metadata']`` (in place)."""
    test.setdefault("metadata", {}).update(labels_for(prompt, mapping))
    return test

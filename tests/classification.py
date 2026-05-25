"""Shared, suite-independent test classification.

Every test is labelled on two orthogonal dimensions:

* ``request_type`` — *what* the user is trying to do (e.g. ``coding``).
* ``domain``       — the subject *area* they are working in (e.g. ``finance_business``).

The vocabularies below are deliberately small, fixed, and shared across all
suites so a label means the same thing whether it came from research_rubrics,
agentharm, or multifaceted. Native dataset fields (``domain``/``category``/
``source``) are kept as separate metadata for provenance but are *not* used as
the classification — they use disjoint, per-dataset vocabularies and so can't
give a consistent cross-suite view.

Labels live OUTSIDE the raw datasets, in ``data/classifications/<suite>.json``,
keyed by :func:`prompt_key` (a hash of the prompt text). This keeps the raw
data untouched, survives dataset re-downloads/reordering, and works for suites
whose rows carry no stable id. Generators call :func:`augment` to stamp
``metadata.request_type`` / ``metadata.domain`` onto each test; promptfoo can
then filter on them (``--filter-metadata request_type=coding``) and
``tests/suite_config.py`` can take stratified per-class samples.

Populate the files with ``scripts_repo/classify_tests.py``.
"""

import hashlib
import json
from pathlib import Path

# What the user is trying to do. Keep these mutually exclusive; pick the
# dominant intent when a prompt spans several.
REQUEST_TYPES = {
    "factual_qa": "Retrieve or state facts, including current affairs.",
    "research_synthesis": "Gather and synthesise sources into a report or survey.",
    "planning": "Produce a plan, itinerary, strategy, or roadmap.",
    "coding": "Write, debug, review, or explain code.",
    "math_reasoning": "Solve a quantitative, logical, or symbolic problem.",
    "data_analysis": "Analyse, interpret, or draw conclusions from data.",
    "creative_writing": "Generate creative or marketing prose, poetry, or scripts.",
    "advice_recommendation": "Recommend, compare options, or give how-to guidance.",
    "text_transformation": "Summarise, classify, extract, translate, or rewrite text.",
}

# The subject area the request lives in.
DOMAINS = {
    "technology_ai": "Software, computing, AI/ML, engineering systems.",
    "science_stem": "Natural sciences, mathematics, academic STEM.",
    "medicine_health": "Medicine, health, biology, wellbeing.",
    "finance_business": "Finance, economics, business strategy, markets.",
    "law_policy": "Law, regulation, government, public policy.",
    "history_society": "History, geography, politics, social science.",
    "arts_literature": "Literature, art, music, film, humanities.",
    "consumer_lifestyle": "Shopping, travel, food, hobbies, daily life.",
    "philosophy_ethics": "Philosophy, ethics, religion, hypotheticals.",
    "current_events": "Recent news and ongoing events.",
    "general_other": "General knowledge or anything that fits nothing above.",
}

# Stamped when a prompt has no entry in the classification file yet.
UNCLASSIFIED = "unclassified"

CLASSIFICATIONS_DIR = Path(__file__).resolve().parent.parent / "data" / "classifications"


def prompt_key(prompt: str) -> str:
    """Stable join key for a test: SHA-1 of the trimmed prompt text."""
    return hashlib.sha1(prompt.strip().encode("utf-8")).hexdigest()


def classifications_path(suite: str) -> Path:
    return CLASSIFICATIONS_DIR / f"{suite}.json"


_cache: dict = {}


def load(suite: str) -> dict:
    """Return ``{prompt_key: {"request_type":.., "domain":..}}`` for a suite.

    Missing file => empty mapping (every test falls back to ``unclassified``).
    """
    if suite not in _cache:
        path = classifications_path(suite)
        doc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        _cache[suite] = doc.get("classifications", {})
    return _cache[suite]


def labels_for(suite: str, prompt: str) -> dict:
    """Look up a prompt's labels, falling back to ``unclassified``."""
    entry = load(suite).get(prompt_key(prompt)) or {}
    return {
        "request_type": entry.get("request_type", UNCLASSIFIED),
        "domain": entry.get("domain", UNCLASSIFIED),
    }


def augment(test: dict, suite: str, prompt: str) -> dict:
    """Stamp ``request_type``/``domain`` into ``test['metadata']`` (in place)."""
    test.setdefault("metadata", {}).update(labels_for(suite, prompt))
    return test

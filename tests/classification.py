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
import os
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

# Output transform attached to every generated assertion. Promptfoo applies an
# assertion's ``transform`` only to that assertion's view of the output, so
# graders see the model's final answer with the reasoning prefix stripped (see
# hooks/strip_before_triple_newline.py) while the stored ``response.output``
# keeps the full, pre-strip response. Doing this per-assertion is deliberate: the
# alternative, a global ``defaultTest.options.transform`` in promptfooconfig.yaml,
# overwrites the canonical output and discards the original. ``augment`` is the
# one funnel every suite's tests pass through, so it is where we attach this.
GRADING_TRANSFORM = "file://hooks/strip_before_triple_newline.py"

# When this env var is set (only by scripts_repo/run_comparison.py, for its
# unified prod-vs-dev eval), augment() appends a `select-best` assertion to every
# test so the judge picks the better of the two providers' answers head-to-head.
# It is gated because select-best only makes sense with >1 provider in one eval:
# single-provider runs (fidaro.sh / CI on promptfooconfig.yaml) leave it unset and
# so behave exactly as before. Keep the name in sync with run_comparison.py.
SELECT_BEST_ENV_VAR = "COMPARISON_SELECT_BEST"

# What "best" means for the head-to-head — one shared, question-relative
# criterion across all suites (rendered into the template below as {{criteria}}).
SELECT_BEST_CRITERION = (
    "the response that most accurately, completely, and clearly answers the "
    "user's question, fully addressing the request without adding unsupported "
    "or fabricated claims"
)

# Custom grading prompt for the select-best assertion. promptfoo's BUILT-IN
# select-best template only interpolates {{criteria}} and {{outputs}} -- it never
# shows the judge the user's prompt, which would make a question-relative
# criterion ungrounded. promptfoo renders rubricPrompt with the test's vars in
# scope, and this repo binds the prompt to `user` (prompt_templates/user_only.json),
# so {{ user }} hands the judge the original question. {{ outputs }} is the list of
# candidate answers (one per provider, already reasoning-stripped via the
# assertion transform); the judge must reply with the single best index.
SELECT_BEST_RUBRIC_PROMPT = (
    "You are comparing AI responses to decide which one is "
    "{{ criteria }}.\n\n"
    "The user asked:\n{{ user }}\n\n"
    "Here are the candidate responses:\n"
    "{% for output in outputs %}"
    '<Response index="{{ loop.index0 }}">\n'
    "{{ output }}\n"
    "</Response>\n"
    "{% endfor %}\n"
    "Output only the single integer index of the response that best fits the "
    "criterion."
)

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


def _attach_grading_transform(test: dict) -> None:
    """Default every assertion to the reasoning-strip transform (in place).

    Uses ``setdefault`` so an assertion that declares its own ``transform`` keeps
    it. See :data:`GRADING_TRANSFORM` for why this lives here.
    """
    for assertion in test.get("assert") or []:
        if isinstance(assertion, dict):
            assertion.setdefault("transform", GRADING_TRANSFORM)


def _maybe_add_select_best(test: dict) -> None:
    """Append a select-best assertion (in place) for comparison runs only.

    No-op unless :data:`SELECT_BEST_ENV_VAR` is set, and idempotent: a test that
    already carries a select-best assertion is left untouched. Added before the
    transform is attached so the judge compares the reasoning-stripped answers.
    """
    if not os.environ.get(SELECT_BEST_ENV_VAR):
        return
    asserts = test.setdefault("assert", [])
    if any(isinstance(a, dict) and a.get("type") == "select-best" for a in asserts):
        return
    asserts.append(
        {
            "type": "select-best",
            "value": SELECT_BEST_CRITERION,
            "rubricPrompt": SELECT_BEST_RUBRIC_PROMPT,
        }
    )


def augment(test: dict, suite: str, prompt: str) -> dict:
    """Stamp ``request_type``/``domain`` into ``test['metadata']`` (in place).

    Also gives each assertion the shared reasoning-strip transform so graders
    see the stripped answer while the stored response keeps its full text. In a
    comparison run (see :data:`SELECT_BEST_ENV_VAR`) it additionally appends a
    head-to-head ``select-best`` assertion.
    """
    test.setdefault("metadata", {}).update(labels_for(suite, prompt))
    _maybe_add_select_best(test)
    _attach_grading_transform(test)
    return test

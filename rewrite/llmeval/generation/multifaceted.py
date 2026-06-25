"""Generator for the ``multifaceted`` suite (kaist-ai/Multifaceted-Bench).

Each dataset row becomes one test case; every entry in the row's ``rubrics``
column becomes a separate ``rubric`` assertion (metric ``multifaceted``).

Multifaceted-Bench rubrics are authored on a 1-5 scale
(``{criteria, score_descriptions: {"1".."5"}}``). We embed the criteria plus its
1-5 anchors into the rubric text as context. Per the rewrite's design we grade
with the standard 0-1 rubric template rather than porting the legacy per-test
1-5 ``rubricPrompt`` override (see the design spec) — the exact 1->5 normalization
is intentionally not reproduced.

Ported from the legacy ``tests/multifaceted_gen.py``.
"""

from __future__ import annotations

from typing import Any

from llmeval.generation.classification import load_classifications, stamp
from llmeval.generation.common import load_dataset, make_id
from llmeval.generation.config import SuiteGenConfig

SUITE = "multifaceted"

_DOWNLOAD_HINT = "npm run dataset:multifaceted"


def _format_rubric(item: dict[str, Any]) -> str:
    """Build the rubric text: criteria followed by its 1-5 score anchors."""
    criteria = (item.get("criteria") or "").strip()
    descriptions = item.get("score_descriptions") or {}
    lines = [criteria, "", "Rate the response from 1 to 5 using this scale:"]
    for score in ("1", "2", "3", "4", "5"):
        text = descriptions.get(score)
        if text:
            lines.append(f"{score}: {str(text).strip()}")
    return "\n".join(lines)


def _row_to_test(row: dict[str, Any], index: int, max_rubrics: int | None, mapping: dict) -> dict:
    rubrics = row.get("rubrics") or []
    if max_rubrics is not None:
        rubrics = rubrics[:max_rubrics]
    assertions = [
        {"type": "rubric", "value": _format_rubric(item), "metric": "multifaceted"}
        for item in rubrics
        if (item.get("criteria") or "").strip()
    ]
    prompt = row["prompt"]
    source = row.get("source", "unknown")
    test = {
        "id": make_id(SUITE, prompt),
        "user": prompt,
        "assertions": assertions,
        "metadata": {"suite": SUITE, "sample_id": str(index), "source": source},
    }
    return stamp(test, prompt, mapping)


def generate_multifaceted(
    data_path: str, config: SuiteGenConfig, classifications: dict
) -> list[dict]:
    rows = load_dataset(data_path, _DOWNLOAD_HINT)
    tests = [
        test
        for index, row in enumerate(rows)
        if (test := _row_to_test(row, index, config.max_rubrics, classifications))["assertions"]
    ]
    return config.select(tests)


def load_and_generate(
    data_dir: str, classifications_dir: str, config_path: str | None
) -> list[dict]:
    from pathlib import Path

    from llmeval.generation.config import load_suite_config

    cfg = load_suite_config(SUITE, config_path)
    mapping = load_classifications(SUITE, classifications_dir)
    return generate_multifaceted(str(Path(data_dir) / "multifaceted.json"), cfg, mapping)

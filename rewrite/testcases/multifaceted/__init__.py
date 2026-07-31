"""``multifaceted`` — kaist-ai/Multifaceted-Bench, one rubric assertion per criterion.

The dataset authors its rubrics on a 1-5 scale (``{criteria, score_descriptions}``); we embed
the criterion and its five anchors into the rubric text and grade with the standard 0-1
template. The legacy suite's per-test 1->5 ``rubricPrompt`` override is deliberately not
reproduced — see docs/specs/2026-06-25-rewrite-all-suites-generation-design.md.
"""

from __future__ import annotations

from typing import Any

from llmeval.generation.common import local_id
from llmeval.generation.dataset_plugin import HfDatasetPlugin
from llmeval.plugins import PluginInterface, TestCasePlugin

DATASET, CONFIG, SPLIT = "kaist-ai/Multifaceted-Bench", "default", "train"


def _format_rubric(item: dict[str, Any]) -> str:
    """The criterion, followed by its 1-5 anchors, as one block of rubric text."""
    criteria = (item.get("criteria") or "").strip()
    descriptions = item.get("score_descriptions") or {}
    lines = [criteria, "", "Rate the response from 1 to 5 using this scale:"]
    for score in ("1", "2", "3", "4", "5"):
        text = descriptions.get(score)
        if text:
            lines.append(f"{score}: {str(text).strip()}")
    return "\n".join(lines)


def rows_to_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dataset rows -> test cases. Pure, so the transform is testable without a download."""
    cases = []
    for index, row in enumerate(rows):
        prompt = row.get("prompt")
        if not prompt:
            continue
        assertions = [
            {"type": "rubric", "value": _format_rubric(item), "metric": "multifaceted"}
            for item in (row.get("rubrics") or [])
            if (item.get("criteria") or "").strip()
        ]
        if not assertions:
            continue
        cases.append(
            {
                "id": local_id(prompt),
                "user": prompt,
                "assertions": assertions,
                "metadata": {"sample_id": str(index), "source": row.get("source", "unknown")},
            }
        )
    return cases


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return HfDatasetPlugin(
        interface, dataset=DATASET, config=CONFIG, split=SPLIT, transform=rows_to_cases
    )

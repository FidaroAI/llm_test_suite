"""``research_rubrics`` — ScaleAI/researchrubrics, graded two ways for comparison.

Each row becomes **two** test cases over the same criteria, told apart by
``metadata.grader``: ``rubric`` (the established 0-1 grader) and ``g_eval``
(chain-of-thought). Running both is the point — it is how the two grading styles get compared
head to head on identical work.

Every rubric on a row is emitted. The old ``max_rubrics`` cap is gone: capping is selection,
and selection now happens at run time, not here.
"""

from __future__ import annotations

from typing import Any

from llmeval.generation.common import local_id
from llmeval.generation.dataset_plugin import HfDatasetPlugin
from llmeval.plugins import PluginInterface, TestCasePlugin

DATASET, CONFIG, SPLIT = "ScaleAI/researchrubrics", "default", "train"

GRADERS = ("rubric", "g_eval")


def _row_to_case(row: dict[str, Any], grader: str) -> dict[str, Any] | None:
    prompt = row.get("prompt")
    if not prompt:
        return None
    assertions = [
        {
            "type": grader,
            "value": item["criterion"],
            # The rubric's own weight and axis carry through, so scores can be aggregated
            # per axis rather than as one undifferentiated mean.
            "weight": item.get("weight", 1),
            "metric": item.get("axis", "unspecified"),
        }
        for item in (row.get("rubrics") or [])
        if item.get("criterion")
    ]
    if not assertions:
        return None
    return {
        "id": local_id(prompt, variant=grader),
        "user": prompt,
        "assertions": assertions,
        "metadata": {
            "grader": grader,
            "sample_id": row.get("sample_id", "unknown"),
            # The dataset's own label, kept for provenance.
            "native_domain": row.get("domain", "unknown"),
            "conceptual_breadth": row.get("conceptual_breadth", "unknown"),
            "logical_nesting": row.get("logical_nesting", "unknown"),
            "exploration": row.get("exploration", "unknown"),
        },
    }


def rows_to_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Dataset rows -> test cases. Pure, so the transform is testable without a download."""
    cases = []
    for row in rows:
        for grader in GRADERS:
            case = _row_to_case(row, grader)
            if case:
                cases.append(case)
    return cases


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return HfDatasetPlugin(
        interface, dataset=DATASET, config=CONFIG, split=SPLIT, transform=rows_to_cases
    )

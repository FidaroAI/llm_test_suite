"""Generator for the ``research_rubrics`` suite (ScaleAI/researchrubrics).

Each dataset row becomes *two* test cases over the same rubric criteria so the two
grading styles can be compared head-to-head, told apart by ``metadata.grader``:

  - ``rubric``: the established grader; honours ``max_rubrics``.
  - ``g_eval``: chain-of-thought scoring; uses *all* criteria for the row
    (``max_rubrics`` intentionally not applied — these rubrics are meant to be
    graded in full).

Each rubric's ``weight`` maps to the assertion weight and ``axis`` to its metric,
so scores can be aggregated per axis. Ported from the legacy
``tests/research_rubrics_gen.py``; legacy ``llm-rubric``/``g-eval`` map to the
rewrite's ``rubric``/``g_eval`` assertion types.
"""

from __future__ import annotations

from typing import Any

from llmeval.generation.classification import load_classifications, stamp
from llmeval.generation.common import load_dataset, make_id
from llmeval.generation.config import SuiteGenConfig

SUITE = "research_rubrics"

_DOWNLOAD_HINT = "npm run dataset:researchrubrics"


def _row_to_test(row: dict[str, Any], grader: str, max_rubrics: int | None, mapping: dict) -> dict:
    rubrics = row.get("rubrics") or []
    if max_rubrics is not None:
        rubrics = rubrics[:max_rubrics]
    assertions = [
        {
            "type": grader,
            "value": item["criterion"],
            "weight": item.get("weight", 1),
            "metric": item.get("axis", "unspecified"),
        }
        for item in rubrics
        if item.get("criterion")
    ]
    prompt = row["prompt"]
    test = {
        "id": make_id(SUITE, prompt, variant=grader),
        "user": prompt,
        "assertions": assertions,
        "metadata": {
            "suite": SUITE,
            "grader": grader,
            "sample_id": row.get("sample_id", "unknown"),
            # Dataset's own domain label, kept for provenance; the unified
            # cross-suite `domain` is added by classification below.
            "native_domain": row.get("domain", "unknown"),
            "conceptual_breadth": row.get("conceptual_breadth", "unknown"),
            "logical_nesting": row.get("logical_nesting", "unknown"),
            "exploration": row.get("exploration", "unknown"),
        },
    }
    return stamp(test, prompt, mapping)


def generate_research_rubrics(
    data_path: str, config: SuiteGenConfig, classifications: dict
) -> list[dict]:
    rows = load_dataset(data_path, _DOWNLOAD_HINT)
    tests: list[dict] = []
    for row in rows:
        rubric_test = _row_to_test(row, "rubric", config.max_rubrics, classifications)
        if rubric_test["assertions"]:
            tests.append(rubric_test)
        geval_test = _row_to_test(row, "g_eval", None, classifications)
        if geval_test["assertions"]:
            tests.append(geval_test)
    return config.select(tests)


def load_and_generate(
    data_dir: str, classifications_dir: str, config_path: str | None
) -> list[dict]:
    from pathlib import Path

    from llmeval.generation.config import load_suite_config

    cfg = load_suite_config(SUITE, config_path)
    mapping = load_classifications(SUITE, classifications_dir)
    return generate_research_rubrics(str(Path(data_dir) / "researchrubrics.json"), cfg, mapping)

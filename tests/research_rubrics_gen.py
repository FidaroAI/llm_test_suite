"""Promptfoo dynamic test generator for the ScaleAI/researchrubrics dataset.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/research_rubrics_gen.py:generate_tests

Each dataset row becomes one promptfoo test case: the row's `prompt` is sent to
the model under test (via the existing `user_only` template + gateway), and the
single response is graded against each rubric `criterion` as a separate
`llm-rubric` assertion. The rubric `weight` maps to the assertion weight and
`axis` to its metric, so promptfoo can aggregate scores per axis.

The raw dataset must be downloaded first:

    npm run dataset:researchrubrics

Set RESEARCH_RUBRICS_LIMIT=N to cap how many rows become test cases (useful for
a small first run); leave it unset to use all rows.
"""

import json
import os
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "researchrubrics.json"

# Test passes if the weighted average of its rubric scores meets this. Lenient
# for now; tightening is future work.
PASS_THRESHOLD = 0.5


def _load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the dataset first with: "
            "npm run dataset:researchrubrics"
        )
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _row_to_test(row):
    rubrics = row.get("rubrics") or []
    asserts = [
        {
            "type": "llm-rubric",
            "value": item["criterion"],
            "weight": item.get("weight", 1),
            "metric": item.get("axis", "unspecified"),
        }
        for item in rubrics
        if item.get("criterion")
    ]

    sample_id = row.get("sample_id", "unknown")
    domain = row.get("domain", "unknown")

    return {
        "description": f"researchrubrics[{domain}] {sample_id}",
        "vars": {"user": row["prompt"]},
        "assert": asserts,
        "threshold": PASS_THRESHOLD,
        "metadata": {
            "suite": "research_rubrics",
            "sample_id": sample_id,
            "domain": domain,
            "conceptual_breadth": row.get("conceptual_breadth", "unknown"),
            "logical_nesting": row.get("logical_nesting", "unknown"),
            "exploration": row.get("exploration", "unknown"),
        },
    }


def generate_tests():
    rows = _load_rows()

    limit = os.environ.get("RESEARCH_RUBRICS_LIMIT")
    if limit:
        rows = rows[: int(limit)]

    # Skip rows that carry no gradable rubric items.
    return [test for row in rows if (test := _row_to_test(row))["assert"]]

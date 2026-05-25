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

Generation is configured via the suite-generation config file (keyed by the
``research_rubrics`` suite name); see tests/suite_config.py. `number_to_generate`
caps how many rows become test cases and `max_rubrics` caps rubric criteria per
row.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import classification  # noqa: E402
import suite_config  # noqa: E402

SUITE = suite_config.suite_name(__file__)

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


def _row_to_test(row, max_num_criteria=None):
    rubrics = row.get("rubrics") or []
    if max_num_criteria is not None:
        rubrics = rubrics[:max_num_criteria]
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

    test = {
        "description": f"researchrubrics[{domain}] {sample_id}",
        "vars": {"user": row["prompt"]},
        "assert": asserts,
        "threshold": PASS_THRESHOLD,
        "metadata": {
            "suite": SUITE,
            "sample_id": sample_id,
            # The dataset's own domain label, kept for provenance. The unified
            # cross-suite ``domain`` is added by classification.augment below.
            "native_domain": domain,
            "conceptual_breadth": row.get("conceptual_breadth", "unknown"),
            "logical_nesting": row.get("logical_nesting", "unknown"),
            "exploration": row.get("exploration", "unknown"),
        },
    }
    return classification.augment(test, SUITE, row["prompt"])


def generate_tests():
    rows = _load_rows()
    cfg = suite_config.load(__file__)

    # Skip rows that carry no gradable rubric items, then apply selection.
    tests = [
        test
        for row in rows
        if (test := _row_to_test(row, max_num_criteria=cfg.max_rubrics))["assert"]
    ]
    return cfg.select(tests)


if __name__ == "__main__":
    result = generate_tests()
    print(json.dumps(result, indent=2))

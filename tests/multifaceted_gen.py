"""Promptfoo dynamic test generator for the kaist-ai/Multifaceted-Bench dataset.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/multifaceted_gen.py:generate_tests

Each dataset row becomes one promptfoo test case: the row's `prompt` is sent to
the model under test (via the existing `user_only` template + gateway), and the
single response is graded against each entry in the row's `rubrics` column as a
separate `llm-rubric` assertion.

Multifaceted-Bench rubrics are scored on a 1-5 scale: each rubric is
`{criteria, score_descriptions: {"1".."5"}}`. We embed the criteria plus its
1-5 descriptions in the rubric text, and attach a per-test `rubricPrompt`
override (RUBRIC_PROMPT_1_TO_5) that tells the judge to pick a 1-5 rating from
those anchors and normalize it to promptfoo's 0-1 score via (rating - 1) / 4.
This keeps the dataset untouched while making the recorded scores comparable to
the other (0-1) suites; the override is scoped to these tests only, so the other
suites keep promptfoo's default grading prompt.

The raw dataset must be downloaded first:

    npm run dataset:multifaceted

Set MULTIFACETED_LIMIT=N to cap how many rows become test cases (useful for a
small first run); MULTIFACETED_MAX_CRITERIA=N caps rubric criteria per row.
"""

import json
import os
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "multifaceted.json"

# Test passes if the weighted average of its (normalized 0-1) rubric scores
# meets this. Lenient for now; tightening is future work.
PASS_THRESHOLD = 0.5

# Per-test grading-prompt override: grade against the rubric's own 1-5 anchors,
# then normalize to a 0-1 score. Scoped to this suite via test options so the
# other suites keep promptfoo's default 0-1 grading prompt.
RUBRIC_PROMPT_1_TO_5 = [
    {
        "role": "system",
        "content": (
            "You are grading a response against a rubric that uses a 1-5 rating "
            "scale. The rubric states a criterion followed by a description of "
            "what each score from 1 (worst) to 5 (best) means. Read the rubric "
            "and its 1-5 descriptions, decide the integer rating that best "
            "matches the response, then normalize it to a 0-1 score with "
            "score = (rating - 1) / 4 (so 1 -> 0.0, 3 -> 0.5, 5 -> 1.0). "
            "Respond with a JSON object {reason: string, pass: boolean, score: "
            "number} where `score` is the normalized 0-1 value and `reason` "
            "states the chosen 1-5 rating and why."
        ),
    },
    {
        "role": "user",
        "content": "<Output>\n{{ output }}\n</Output>\n<Rubric>\n{{ rubric }}\n</Rubric>",
    },
]


def _load_rows():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} not found. Download the dataset first with: "
            "npm run dataset:multifaceted"
        )
    with DATA_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _format_rubric(item):
    """Build the rubric text: criteria followed by its 1-5 score anchors."""
    criteria = (item.get("criteria") or "").strip()
    descriptions = item.get("score_descriptions") or {}
    lines = [criteria, "", "Rate the response from 1 to 5 using this scale:"]
    for score in ("1", "2", "3", "4", "5"):
        text = descriptions.get(score)
        if text:
            lines.append(f"{score}: {str(text).strip()}")
    return "\n".join(lines)


def _row_to_test(row, index, max_num_criteria=None):
    rubrics = row.get("rubrics") or []
    if max_num_criteria is not None:
        rubrics = rubrics[:max_num_criteria]
    asserts = [
        {
            "type": "llm-rubric",
            "value": _format_rubric(item),
            "metric": "multifaceted",
        }
        for item in rubrics
        if (item.get("criteria") or "").strip()
    ]

    source = row.get("source", "unknown")
    return {
        "description": f"multifaceted[{source}] {index}",
        "vars": {"user": row["prompt"]},
        "assert": asserts,
        "threshold": PASS_THRESHOLD,
        "options": {"rubricPrompt": RUBRIC_PROMPT_1_TO_5},
        "metadata": {
            "suite": "multifaceted",
            "sample_id": str(index),
            "source": source,
        },
    }


def generate_tests():
    rows = _load_rows()

    limit = os.environ.get("MULTIFACETED_LIMIT")
    if limit:
        rows = rows[: int(limit)]

    max_num_criteria = os.environ.get("MULTIFACETED_MAX_CRITERIA")
    if max_num_criteria:
        max_num_criteria = int(max_num_criteria)

    # Skip rows that carry no gradable rubric items.
    return [
        test
        for index, row in enumerate(rows)
        if (test := _row_to_test(row, index, max_num_criteria=max_num_criteria))["assert"]
    ]


if __name__ == "__main__":
    result = generate_tests()
    print(json.dumps(result[0], indent=2))

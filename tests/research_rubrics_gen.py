"""Promptfoo dynamic test generator for the ScaleAI/researchrubrics dataset.

Referenced from promptfooconfig.yaml as:

    tests: file://tests/research_rubrics_gen.py:generate_tests

Each dataset row becomes *two* promptfoo test cases graded over the same rubric
criteria, so the two grading styles can be compared head-to-head: the row's
`prompt` is sent to the model under test (via the existing `user_only` template
+ gateway) and the single response is graded against each rubric `criterion` as
a separate assertion. The rubric `weight` maps to the assertion weight and
`axis` to its metric, so promptfoo can aggregate scores per axis.

The two variants are told apart by ``metadata.grader`` (both keep
``suite=research_rubrics``), so a run can isolate one with e.g.
``--filter-metadata grader=g-eval``:

  - ``llm-rubric``: the established grader; still honours ``max_rubrics``.
  - ``g-eval``: chain-of-thought, per-criterion graded scoring. Uses *all*
    criteria for the row — ``max_rubrics`` is intentionally not applied, since
    these rubrics are meant to be evaluated in full. Its g-eval description
    carries a ``(g-eval)`` suffix.

Emitting both unconditionally is deliberate for now; gating the choice of
variant(s) via the suite-generation config is future work.

The raw dataset must be downloaded first:

    npm run dataset:researchrubrics

Generation is configured via the suite-generation config file (keyed by the
``research_rubrics`` suite name); see tests/suite_config.py. `number_to_generate`
caps how many rows become test cases (applied to the combined two-variant list)
and `max_rubrics` caps rubric criteria per row for the ``llm-rubric`` variant.
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


def _row_to_test(row, grader, max_num_criteria=None):
    rubrics = row.get("rubrics") or []
    if max_num_criteria is not None:
        rubrics = rubrics[:max_num_criteria]
    asserts = [
        {
            "type": grader,
            "value": item["criterion"],
            "weight": item.get("weight", 1),
            "metric": item.get("axis", "unspecified"),
        }
        for item in rubrics
        if item.get("criterion")
    ]

    sample_id = row.get("sample_id", "unknown")
    domain = row.get("domain", "unknown")

    # Suffix only the g-eval variant so the established llm-rubric descriptions
    # (and any baselines keyed off them) stay stable.
    description = f"researchrubrics[{domain}] {sample_id}"
    if grader == "g-eval":
        description += " (g-eval)"

    test = {
        "description": description,
        "vars": {"user": row["prompt"]},
        "assert": asserts,
        "threshold": PASS_THRESHOLD,
        "metadata": {
            "suite": SUITE,
            "grader": grader,
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

    # Emit both grading variants per row (see module docstring). The g-eval
    # variant ignores max_rubrics and grades the full rubric. Rows with no
    # gradable criteria are skipped. Selection (cap/shuffle/stratify) then
    # applies to the combined list.
    tests = []
    for row in rows:
        rubric_test = _row_to_test(row, "llm-rubric", max_num_criteria=cfg.max_rubrics)
        if rubric_test["assert"]:
            tests.append(rubric_test)
        geval_test = _row_to_test(row, "g-eval", max_num_criteria=None)
        if geval_test["assert"]:
            tests.append(geval_test)
    return cfg.select(tests)


if __name__ == "__main__":
    result = generate_tests()
    print(json.dumps(result, indent=2))

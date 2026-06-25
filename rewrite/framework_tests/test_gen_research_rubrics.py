import json

from llmeval.generation.config import DEFAULTS, SuiteGenConfig
from llmeval.generation.research_rubrics import generate_research_rubrics


def _cfg(**over):
    return SuiteGenConfig("research_rubrics", {**DEFAULTS, "number_to_generate": None, **over})


def _dataset(tmp_path):
    rows = [
        {
            "prompt": "Survey transformer architectures",
            "sample_id": "s1",
            "domain": "ml",
            "conceptual_breadth": "high",
            "logical_nesting": "med",
            "exploration": "high",
            "rubrics": [
                {"criterion": "Covers attention", "weight": 2, "axis": "coverage"},
                {"criterion": "Cites sources", "weight": 1, "axis": "grounding"},
            ],
        },
        {"prompt": "Empty", "sample_id": "s2", "domain": "x", "rubrics": []},
    ]
    p = tmp_path / "researchrubrics.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_two_variants_per_row(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(), {})
    graders = sorted(c["metadata"]["grader"] for c in cases)
    assert graders == ["g_eval", "rubric"]


def test_assertion_types_match_grader(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(), {})
    by_grader = {c["metadata"]["grader"]: c for c in cases}
    assert all(a["type"] == "rubric" for a in by_grader["rubric"]["assertions"])
    assert all(a["type"] == "g_eval" for a in by_grader["g_eval"]["assertions"])


def test_weight_and_axis_metric_carried(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(), {})
    a = cases[0]["assertions"][0]
    assert a["weight"] == 2
    assert a["metric"] == "coverage"


def test_ids_unique_across_variants(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(), {})
    ids = [c["id"] for c in cases]
    assert len(ids) == len(set(ids))


def test_max_rubrics_only_affects_rubric_variant(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(max_rubrics=1), {})
    by_grader = {c["metadata"]["grader"]: c for c in cases}
    assert len(by_grader["rubric"]["assertions"]) == 1
    assert len(by_grader["g_eval"]["assertions"]) == 2  # g_eval ignores max_rubrics


def test_native_domain_preserved(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(), {})
    assert cases[0]["metadata"]["native_domain"] == "ml"


def test_empty_rubric_rows_skipped(tmp_path):
    cases = generate_research_rubrics(_dataset(tmp_path), _cfg(), {})
    assert all(c["user"] != "Empty" for c in cases)

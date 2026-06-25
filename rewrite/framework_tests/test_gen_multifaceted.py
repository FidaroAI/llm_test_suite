import json

from llmeval.generation.config import DEFAULTS, SuiteGenConfig
from llmeval.generation.multifaceted import generate_multifaceted


def _cfg(**over):
    return SuiteGenConfig("multifaceted", {**DEFAULTS, "number_to_generate": None, **over})


def _dataset(tmp_path):
    rows = [
        {
            "prompt": "Explain recursion",
            "source": "alpaca",
            "rubrics": [
                {"criteria": "Is it correct?",
                 "score_descriptions": {"1": "wrong", "3": "ok", "5": "great"}},
                {"criteria": "Is it clear?",
                 "score_descriptions": {"1": "muddy", "5": "crystal"}},
            ],
        },
        {"prompt": "No rubrics here", "source": "x", "rubrics": []},
        {"prompt": "Blank criteria", "source": "y",
         "rubrics": [{"criteria": "  ", "score_descriptions": {}}]},
    ]
    p = tmp_path / "multifaceted.json"
    p.write_text(json.dumps(rows))
    return str(p)


def test_each_rubric_becomes_an_assertion(tmp_path):
    cases = generate_multifaceted(_dataset(tmp_path), _cfg(), {})
    # only the first row has gradable rubrics
    assert len(cases) == 1
    asserts = cases[0]["assertions"]
    assert len(asserts) == 2
    assert all(a["type"] == "rubric" for a in asserts)
    assert all(a["metric"] == "multifaceted" for a in asserts)


def test_rubric_text_embeds_1_to_5_anchors(tmp_path):
    cases = generate_multifaceted(_dataset(tmp_path), _cfg(), {})
    text = cases[0]["assertions"][0]["value"]
    assert "Is it correct?" in text
    assert "1: wrong" in text
    assert "5: great" in text


def test_rows_without_gradable_rubrics_skipped(tmp_path):
    cases = generate_multifaceted(_dataset(tmp_path), _cfg(), {})
    assert all(c["assertions"] for c in cases)
    assert all(c["user"] != "No rubrics here" for c in cases)


def test_max_rubrics_caps_per_row(tmp_path):
    cases = generate_multifaceted(_dataset(tmp_path), _cfg(max_rubrics=1), {})
    assert len(cases[0]["assertions"]) == 1


def test_metadata_carries_source(tmp_path):
    cases = generate_multifaceted(_dataset(tmp_path), _cfg(), {})
    assert cases[0]["metadata"]["source"] == "alpaca"
    assert cases[0]["metadata"]["suite"] == "multifaceted"

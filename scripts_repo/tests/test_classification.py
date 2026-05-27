"""Unit tests for the shared classification helper (no network needed)."""

import importlib.util
import json
from pathlib import Path

_MOD_PATH = Path(__file__).resolve().parents[2] / "tests" / "classification.py"
_spec = importlib.util.spec_from_file_location("classification", _MOD_PATH)
classification = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(classification)


def _point_at(tmp_path, monkeypatch):
    """Redirect the module at a temp classifications dir with a fresh cache."""
    monkeypatch.setattr(classification, "CLASSIFICATIONS_DIR", tmp_path)
    monkeypatch.setattr(classification, "_cache", {})


def test_prompt_key_is_stable_and_whitespace_insensitive():
    assert classification.prompt_key("hello") == classification.prompt_key("  hello\n")
    assert classification.prompt_key("a") != classification.prompt_key("b")


def test_labels_for_falls_back_to_unclassified_when_no_file(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    labels = classification.labels_for("nope", "some prompt")
    assert labels == {"request_type": "unclassified", "domain": "unclassified"}


def test_augment_stamps_labels_from_file(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    prompt = "Write a poem about the sea."
    (tmp_path / "demo.json").write_text(json.dumps({
        "classifications": {
            classification.prompt_key(prompt): {
                "request_type": "creative_writing", "domain": "arts_literature",
            }
        }
    }), encoding="utf-8")

    test = {"vars": {"user": prompt}, "metadata": {"suite": "demo"}}
    classification.augment(test, "demo", prompt)
    assert test["metadata"]["request_type"] == "creative_writing"
    assert test["metadata"]["domain"] == "arts_literature"
    assert test["metadata"]["suite"] == "demo"  # existing keys preserved


def test_augment_creates_metadata_when_absent(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    test = {"vars": {"user": "x"}}
    classification.augment(test, "demo", "x")
    assert test["metadata"]["domain"] == "unclassified"


def test_augment_attaches_grading_transform_to_every_assertion(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    test = {
        "vars": {"user": "x"},
        "assert": [{"type": "icontains", "value": "a"}, {"type": "llm-rubric", "value": "b"}],
    }
    classification.augment(test, "demo", "x")
    assert all(
        a["transform"] == classification.GRADING_TRANSFORM for a in test["assert"]
    )


def test_augment_does_not_clobber_an_assertions_own_transform(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    test = {
        "vars": {"user": "x"},
        "assert": [{"type": "python", "value": "v", "transform": "file://custom.py"}],
    }
    classification.augment(test, "demo", "x")
    assert test["assert"][0]["transform"] == "file://custom.py"


def test_augment_handles_tests_without_assertions(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    test = {"vars": {"user": "x"}}  # no "assert" key
    classification.augment(test, "demo", "x")  # must not raise
    assert "assert" not in test


def test_vocabularies_are_nonempty_and_documented():
    assert classification.REQUEST_TYPES and classification.DOMAINS
    assert all(isinstance(v, str) and v for v in classification.REQUEST_TYPES.values())
    assert all(isinstance(v, str) and v for v in classification.DOMAINS.values())


# --- select-best injection (comparison runs only) --------------------------


def test_augment_does_not_add_select_best_by_default(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    monkeypatch.delenv(classification.SELECT_BEST_ENV_VAR, raising=False)
    test = {"vars": {"user": "x"}, "assert": [{"type": "icontains", "value": "a"}]}
    classification.augment(test, "demo", "x")
    assert [a["type"] for a in test["assert"]] == ["icontains"]


def test_augment_appends_select_best_when_enabled(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    monkeypatch.setenv(classification.SELECT_BEST_ENV_VAR, "1")
    test = {"vars": {"user": "x"}, "assert": [{"type": "llm-rubric", "value": "b"}]}
    classification.augment(test, "demo", "x")
    types = [a["type"] for a in test["assert"]]
    assert types == ["llm-rubric", "select-best"]
    sb = test["assert"][-1]
    assert sb["value"] == classification.SELECT_BEST_CRITERION
    assert sb["rubricPrompt"] == classification.SELECT_BEST_RUBRIC_PROMPT
    # gets the shared reasoning-strip transform like every other assertion
    assert sb["transform"] == classification.GRADING_TRANSFORM


def test_select_best_rubric_prompt_grounds_judge_in_the_user_prompt():
    # The built-in template never sees the prompt; ours must inject {{ user }}
    # (the var prompt_templates/user_only.json binds the prompt to) plus the
    # outputs loop and the criteria.
    rp = classification.SELECT_BEST_RUBRIC_PROMPT
    assert "{{ user }}" in rp
    assert "outputs" in rp
    assert "{{ criteria }}" in rp


def test_augment_select_best_is_idempotent(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    monkeypatch.setenv(classification.SELECT_BEST_ENV_VAR, "1")
    test = {
        "vars": {"user": "x"},
        "assert": [{"type": "select-best", "value": "already here"}],
    }
    classification.augment(test, "demo", "x")
    assert [a["type"] for a in test["assert"]] == ["select-best"]
    assert test["assert"][0]["value"] == "already here"  # left untouched


def test_augment_creates_assert_list_for_bare_test_when_enabled(tmp_path, monkeypatch):
    _point_at(tmp_path, monkeypatch)
    monkeypatch.setenv(classification.SELECT_BEST_ENV_VAR, "1")
    test = {"vars": {"user": "x"}}  # no "assert" key
    classification.augment(test, "demo", "x")
    assert [a["type"] for a in test["assert"]] == ["select-best"]

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


def test_vocabularies_are_nonempty_and_documented():
    assert classification.REQUEST_TYPES and classification.DOMAINS
    assert all(isinstance(v, str) and v for v in classification.REQUEST_TYPES.values())
    assert all(isinstance(v, str) and v for v in classification.DOMAINS.values())

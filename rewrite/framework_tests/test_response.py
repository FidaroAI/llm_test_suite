import pytest

from llmeval.response import apply_transform


def test_strip_reasoning_drops_prefix_before_first_triple_newline():
    out = "thinking step 1\nthinking step 2\n\n\nThe answer is Paris."
    assert apply_transform("strip_reasoning", out) == "The answer is Paris."


def test_strip_reasoning_splits_only_on_first_delimiter():
    out = "reasoning\n\n\nanswer part 1\n\n\nanswer part 2"
    assert apply_transform("strip_reasoning", out) == "answer part 1\n\n\nanswer part 2"


def test_strip_reasoning_no_delimiter_returns_unchanged():
    out = "just an answer, no reasoning block"
    assert apply_transform("strip_reasoning", out) == out


def test_non_string_passes_through():
    out = {"structured": True}
    assert apply_transform("strip_reasoning", out) == out


def test_none_and_identity_are_passthrough():
    assert apply_transform(None, "x\n\n\ny") == "x\n\n\ny"
    assert apply_transform("none", "x\n\n\ny") == "x\n\n\ny"


def test_unknown_transform_raises():
    with pytest.raises(KeyError):
        apply_transform("does_not_exist", "x")

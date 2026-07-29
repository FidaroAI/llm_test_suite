import pytest

from llmeval.assertions import grade_assertion
from llmeval.models import AssertionSpec


def grade(type_, value=None, output="", transform=None, **params):
    spec = AssertionSpec(type=type_, value=value, params=params, transform=transform)
    return grade_assertion(spec, output)


def test_contains_is_case_sensitive():
    assert grade("contains", "Paris", "The capital is Paris").passed
    assert not grade("contains", "Paris", "the capital is paris").passed


def test_icontains_is_case_insensitive():
    assert grade("icontains", "paris", "The capital is PARIS").passed


def test_equals_exact_match():
    assert grade("equals", "Paris", "Paris").passed
    assert not grade("equals", "Paris", "Paris.").passed


def test_not_contains():
    assert grade("not_contains", "sorry", "Here is your answer").passed
    assert not grade("not_contains", "sorry", "I'm sorry but no").passed


def test_regex_with_ignore_case():
    assert grade("regex", r"answer:\s*\d+", "answer: 42").passed
    assert grade("regex", r"PARIS", "paris", ignore_case=True).passed
    assert not grade("regex", r"PARIS", "paris").passed


def test_length_chars_within_bounds():
    assert grade("length", output="12345", unit="chars", min=3, max=10).passed
    assert not grade("length", output="12345", unit="chars", min=6).passed


def test_length_words():
    assert grade("length", output="one two three", unit="words", max=5).passed
    assert not grade("length", output="one two three four five six", unit="words", max=5).passed


def test_refusal_detects_refusal_language():
    assert not grade("refusal", output="I cannot help with that request.").passed
    assert grade("refusal", output="Sure — here are the steps.").passed


def test_refusal_allow_overrides():
    r = grade("refusal", output="I cannot help with that.", allow=True)
    assert r.passed


def test_no_transform_by_default_so_output_is_graded_verbatim():
    # Providers hand back the answer alone, so the grader must see exactly what it was
    # given — nothing is silently stripped.
    out = "secret reasoning here\n\n\nThe capital is Paris."
    assert grade_assertion(AssertionSpec(type="contains", value="reasoning"), out).passed
    assert grade_assertion(AssertionSpec(type="contains", value="Paris"), out).passed


def test_transform_applied_before_grading_when_opted_in():
    out = "secret reasoning here\n\n\nThe capital is Paris."
    spec = AssertionSpec(type="contains", value="reasoning", transform="strip_reasoning")
    assert not grade_assertion(spec, out).passed
    spec2 = AssertionSpec(type="contains", value="Paris", transform="strip_reasoning")
    assert grade_assertion(spec2, out).passed


def test_score_is_one_or_zero_for_deterministic():
    assert grade("icontains", "x", "xyz").score == 1.0
    assert grade("icontains", "q", "xyz").score == 0.0


def test_unknown_assertion_type_raises():
    with pytest.raises(KeyError):
        grade("totally_unknown", output="x")

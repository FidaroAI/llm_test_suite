"""Tests for assertions.assert_reasoning_contains.get_assert.

Covers whole-text matching (no `step` config). Step-mode tests live in the
same file but are added in Task 3.
"""

from assertions.assert_reasoning_contains import get_assert


def _ctx(response, **cfg):
    return {"response": response, "config": cfg}


def test_substring_in_reasoning_content_passes():
    ctx = _ctx({"reasoning_content": "I should reply with pong."}, value="pong")
    result = get_assert("pong", ctx)
    assert result["pass"] is True
    assert result["score"] == 1.0


def test_substring_missing_fails():
    ctx = _ctx({"reasoning_content": "I should reply with pong."}, value="ping")
    result = get_assert("pong", ctx)
    assert result["pass"] is False
    assert result["score"] == 0.0
    assert "not found" in result["reason"].lower()


def test_regex_mode_passes():
    ctx = _ctx(
        {"reasoning_content": "Step 1: parse. Step 2: compute."},
        value=r"Step\s+\d+",
        regex=True,
    )
    result = get_assert("ok", ctx)
    assert result["pass"] is True


def test_thinking_field_used_when_reasoning_content_absent():
    ctx = _ctx({"thinking": "Let me reason about pong."}, value="pong")
    result = get_assert("pong", ctx)
    assert result["pass"] is True


def test_claude_content_blocks_used_when_others_absent():
    response = {
        "content": [
            {"type": "thinking", "thinking": "First I plan."},
            {"type": "thinking", "thinking": "Then I conclude pong."},
            {"type": "text", "text": "pong"},
        ]
    }
    ctx = _ctx(response, value="conclude pong")
    result = get_assert("pong", ctx)
    assert result["pass"] is True


def test_claude_blocks_joined_for_whole_text_match():
    """A substring spanning the natural join between two blocks should NOT match;
    each block stays separated. But matching within one block (via whole-text
    join) works because we join on \\n\\n."""
    response = {
        "content": [
            {"type": "thinking", "thinking": "alpha"},
            {"type": "thinking", "thinking": "beta"},
        ]
    }
    # alpha\n\nbeta — substring "alpha\n\nbeta" should match
    ctx = _ctx(response, value="alpha\n\nbeta")
    assert get_assert("ok", ctx)["pass"] is True


def test_no_reasoning_returns_fail_with_reason():
    ctx = _ctx({}, value="anything")
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert result["reason"] == "no reasoning available"


def test_missing_value_config_fails_clearly():
    ctx = _ctx({"reasoning_content": "x"})  # no value
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert "value" in result["reason"].lower()


def test_empty_reasoning_string_treated_as_absent():
    ctx = _ctx({"reasoning_content": "   "}, value="x")
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert result["reason"] == "no reasoning available"


def test_step_any_matches_when_one_block_contains():
    response = {
        "content": [
            {"type": "thinking", "thinking": "first thought"},
            {"type": "thinking", "thinking": "pong is the answer"},
        ]
    }
    ctx = _ctx(response, value="pong", step="any")
    result = get_assert("pong", ctx)
    assert result["pass"] is True
    assert "block 1" in result["reason"]


def test_step_any_fails_when_no_block_contains():
    response = {
        "content": [
            {"type": "thinking", "thinking": "first"},
            {"type": "thinking", "thinking": "second"},
        ]
    }
    ctx = _ctx(response, value="missing", step="any")
    result = get_assert("ok", ctx)
    assert result["pass"] is False


def test_step_int_indexes_specific_block():
    response = {
        "content": [
            {"type": "thinking", "thinking": "alpha"},
            {"type": "thinking", "thinking": "bravo"},
            {"type": "thinking", "thinking": "charlie"},
        ]
    }
    # value lives in block 1 only
    ctx = _ctx(response, value="bravo", step=1)
    assert get_assert("ok", ctx)["pass"] is True

    # same value, wrong index
    ctx = _ctx(response, value="bravo", step=0)
    assert get_assert("ok", ctx)["pass"] is False


def test_step_int_out_of_range_fails_with_reason():
    ctx = _ctx({"reasoning_content": "only one block"}, value="x", step=5)
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert "out of range" in result["reason"]


def test_step_with_paragraph_split_reasoning_content():
    """A single reasoning_content string with double-newline boundaries
    should be split into multiple blocks for step-mode matching."""
    text = "First step.\n\nSecond step about pong.\n\nThird step."
    ctx = _ctx({"reasoning_content": text}, value="pong", step=1)
    assert get_assert("ok", ctx)["pass"] is True

    # not in step 0
    ctx = _ctx({"reasoning_content": text}, value="pong", step=0)
    assert get_assert("ok", ctx)["pass"] is False


def test_step_any_with_regex():
    response = {
        "content": [
            {"type": "thinking", "thinking": "no numbers here"},
            {"type": "thinking", "thinking": "Step 42 of N"},
        ]
    }
    ctx = _ctx(response, value=r"Step\s+\d+", regex=True, step="any")
    assert get_assert("ok", ctx)["pass"] is True

from assertions.assert_reasoning_iterations import get_assert


def _ctx(response, **cfg):
    return {"response": response, "config": cfg}


def test_paragraphs_in_reasoning_content_count_as_iterations():
    text = "First.\n\nSecond.\n\nThird."
    ctx = _ctx({"reasoning_content": text}, min=3)
    result = get_assert("ok", ctx)
    assert result["pass"] is True


def test_too_few_iterations_fails():
    ctx = _ctx({"reasoning_content": "only one paragraph"}, min=3)
    result = get_assert("ok", ctx)
    assert result["pass"] is False


def test_thinking_field_falls_through_when_reasoning_content_absent():
    text = "1. parse\n2. compute\n3. respond"
    ctx = _ctx({"thinking": text}, min=3)
    assert get_assert("ok", ctx)["pass"] is True


def test_claude_thinking_blocks_each_count_separately():
    response = {
        "content": [
            {"type": "thinking", "thinking": "first"},
            {"type": "thinking", "thinking": "second"},
            {"type": "thinking", "thinking": "third"},
        ]
    }
    ctx = _ctx(response, min=3)
    assert get_assert("ok", ctx)["pass"] is True


def test_no_reasoning_fails_with_reason():
    ctx = _ctx({})
    result = get_assert("ok", ctx)
    assert result["pass"] is False
    assert "no" in result["reason"].lower()


def test_max_bound_enforced():
    text = "\n\n".join(f"step {i}" for i in range(20))
    ctx = _ctx({"reasoning_content": text}, max=5)
    assert get_assert("ok", ctx)["pass"] is False

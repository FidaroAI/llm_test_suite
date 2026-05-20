from assertions.assert_tool_call_count import get_assert


def _ctx(response=None, **config):
    return {"response": response or {}, "config": config}


def test_counts_single_raw_bedrock_tool_use_output():
    output = (
        '{"type":"tool_use","id":"tooluse_Hkvp7nUpwiXoPmnBK1JGjx",'
        '"name":"web_search","input":{"query":"latest OpenAI product announcement",'
        '"max_results":1}}'
    )

    result = get_assert(output, _ctx(expected=1))

    assert result["pass"] is True
    assert result["reason"] == "1 tool call(s); expected exactly 1"


def test_counts_single_raw_bedrock_tool_use_response():
    response = {
        "type": "tool_use",
        "id": "tooluse_Hkvp7nUpwiXoPmnBK1JGjx",
        "name": "web_search",
        "input": {"query": "latest OpenAI product announcement", "max_results": 1},
    }

    result = get_assert("", _ctx(response=response, expected=1))

    assert result["pass"] is True


def test_counts_bedrock_tool_use_content_blocks():
    response = {
        "content": [
            {"type": "text", "text": "Looking that up."},
            {
                "type": "tool_use",
                "id": "tooluse_123",
                "name": "web_search",
                "input": {"query": "OpenAI"},
            },
        ]
    }

    result = get_assert("", _ctx(response=response, expected=1))

    assert result["pass"] is True

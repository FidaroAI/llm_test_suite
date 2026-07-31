"""SSE parsing and stream accumulation — the pure half of the streaming path.

No sockets here. The provider's HTTP behaviour is covered in
``test_streaming_provider.py``; this file pins the *data* contract, which is the one
that has to match the Fidaro orchestrator.
"""

import pytest

from llmeval.streaming import DONE, StreamAccumulator, iter_sse_chunks, parse_sse_line


def chunk(**kw):
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1700000000,
            "model": "auto"}
    base.update(kw)
    return base


def delta_chunk(content=None, reasoning=None, finish=None, **kw):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return chunk(choices=[{"index": 0, "delta": delta, "finish_reason": finish}], **kw)


def sse(payload):
    return f"data: {payload}"


# --- parsing ---------------------------------------------------------------


def test_parses_a_data_frame():
    assert parse_sse_line('data: {"a": 1}') == {"a": 1}


def test_recognises_the_done_terminator():
    assert parse_sse_line("data: [DONE]") is DONE


def test_blank_lines_and_comments_yield_nothing():
    # Frames are separated by blank lines, and ":" is an SSE keep-alive comment. Either
    # mistaken for content would corrupt the answer.
    assert parse_sse_line("") is None
    assert parse_sse_line("   ") is None
    assert parse_sse_line(": keep-alive") is None


def test_non_data_fields_are_skipped():
    # The OpenAI chunk format puts everything in `data:`; the rest is SSE bookkeeping.
    assert parse_sse_line("event: message") is None
    assert parse_sse_line("id: 42") is None
    assert parse_sse_line("retry: 3000") is None


def test_a_json_scalar_is_not_a_chunk():
    assert parse_sse_line("data: 42") is None
    assert parse_sse_line('data: "hello"') is None


def test_malformed_json_raises_rather_than_being_skipped():
    """A wire-format change must be loud.

    Skipping unparseable frames would show up only as inexplicably short answers, long
    after the run that produced them.
    """
    with pytest.raises(ValueError, match="malformed SSE data payload"):
        parse_sse_line("data: {not json")


def test_iter_stops_at_done_and_ignores_the_tail():
    lines = [sse('{"n": 1}'), "", sse("[DONE]"), sse('{"n": 2}')]
    assert list(iter_sse_chunks(lines)) == [{"n": 1}]


def test_iter_tolerates_a_stream_with_no_terminator():
    # A cut-off stream still yields everything that did arrive.
    assert list(iter_sse_chunks([sse('{"n": 1}'), sse('{"n": 2}')])) == [{"n": 1}, {"n": 2}]


# --- accumulation ----------------------------------------------------------


def test_content_and_reasoning_accumulate_separately():
    acc = StreamAccumulator()
    acc.feed(delta_chunk(reasoning="Let me "))
    acc.feed(delta_chunk(reasoning="think."))
    acc.feed(delta_chunk(content="Paris"))
    acc.feed(delta_chunk(content=" is the answer."))

    assert acc.content == "Paris is the answer."
    assert acc.reasoning == "Let me think."


def test_no_reasoning_yields_none_not_empty_string():
    """``None`` distinguishes a non-reasoning model from one cut off before it spoke,
    and matches what the non-streaming path reads off ``message.reasoning_content``."""
    acc = StreamAccumulator()
    acc.feed(delta_chunk(content="hi"))
    assert acc.reasoning is None


def test_usage_and_finish_reason_come_from_the_terminal_chunk():
    acc = StreamAccumulator()
    acc.feed(delta_chunk(content="hi"))
    acc.feed(delta_chunk(finish="stop",
                         usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}))

    assert acc.finish_reason == "stop"
    assert acc.usage == {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}


def test_usage_is_the_servers_own_count():
    """Not a local estimate.

    This is one of the two reasons litellm's streaming path was rejected: it discards
    the server's usage and substitutes a tokenizer estimate, which would make a streamed
    row and a non-streamed row disagree on token counts for the same response.
    """
    acc = StreamAccumulator()
    acc.feed(delta_chunk(content="Hello world"))
    acc.feed(delta_chunk(finish="stop",
                         usage={"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}))

    assert acc.usage["completion_tokens"] == 4


def test_fidaro_is_captured_from_its_own_no_op_chunk():
    """The orchestrator emits the title the moment it knows it, in a chunk whose delta
    is empty — reading only the last chunk would work today by luck and break the moment
    a key stops being repeated."""
    acc = StreamAccumulator()
    acc.feed(delta_chunk(content="Hi"))
    acc.feed(chunk(choices=[{"index": 0, "delta": {}, "finish_reason": None}],
                   fidaro={"title": "A Greeting"}))

    assert acc.fidaro == {"title": "A Greeting"}
    assert acc.provider_specific == {"fidaro": {"title": "A Greeting"}}


def test_the_terminal_repeat_of_fidaro_does_not_blank_earlier_keys():
    acc = StreamAccumulator()
    acc.feed(chunk(choices=[], fidaro={"title": "A Greeting"}))
    # A later chunk carrying the object with a null member must not erase the title.
    acc.feed(chunk(choices=[], fidaro={"title": None}))

    assert acc.fidaro == {"title": "A Greeting"}


def test_fidaro_keys_merge_rather_than_replace():
    # Modelled as an object precisely so more keys can be added additively.
    acc = StreamAccumulator()
    acc.feed(chunk(choices=[], fidaro={"title": "T"}))
    acc.feed(chunk(choices=[], fidaro={"compaction": {"n": 2}}))

    assert acc.fidaro == {"title": "T", "compaction": {"n": 2}}


def test_no_fidaro_means_no_provider_specific():
    # An ordinary response must leave the store column NULL.
    acc = StreamAccumulator()
    acc.feed(delta_chunk(content="hi"))
    assert acc.provider_specific is None


def test_identity_fields_follow_the_server():
    """`auto` is resolved server-side, so the model that answered is the one in the
    chunks, not the one in the request."""
    acc = StreamAccumulator(model="openai/auto")
    acc.feed(delta_chunk(content="hi", id="chatcmpl-xyz", model="Qwen3-Next-80B"))

    assert acc.model == "Qwen3-Next-80B"
    assert acc.completion_id == "chatcmpl-xyz"


def test_an_empty_stream_still_produces_a_readable_completion():
    acc = StreamAccumulator(model="openai/auto")
    completion = acc.completion_dict()

    assert completion["choices"][0]["message"]["content"] == ""
    assert completion["usage"] is None
    assert "fidaro" not in completion


def test_a_partial_stream_is_readable_at_any_point():
    """The property the whole feature rests on: state is valid mid-stream, so a deadline
    can cut in anywhere and still leave usable evidence."""
    acc = StreamAccumulator()
    acc.feed(delta_chunk(reasoning="loop loop "))
    acc.feed(delta_chunk(content="again and "))
    acc.feed(delta_chunk(content="again and "))

    assert acc.content == "again and again and "
    assert acc.reasoning == "loop loop "
    assert acc.finish_reason is None  # never terminated


# --- parity with the orchestrator ------------------------------------------


def test_aggregation_matches_the_orchestrator_non_streaming_response():
    """The contract this module exists to honour.

    Feeds the chunk sequence the orchestrator's ``transcode()`` emits for a given set of
    /v1 events, and asserts the fold produces exactly the ``chat.completion`` its
    ``aggregate()`` builds from those same events. Both fixtures are transcribed from
    ``apps/orchestrator/src/orchestrator/openai_v2/aggregation.py``.

    If this passes, "streamed rows are the same data as non-streamed rows" is a
    demonstrated property rather than a hope.
    """
    # What transcode() puts on the wire: role delta, content/reasoning deltas, the
    # title's own no-op-delta chunk, then a terminal chunk repeating fidaro.
    streamed = [
        chunk(choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]),
        delta_chunk(reasoning="Thinking about it."),
        delta_chunk(content="Paris"),
        chunk(choices=[{"index": 0, "delta": {}, "finish_reason": None}],
              fidaro={"title": "Capital of France"}),
        delta_chunk(content=" is the capital."),
        chunk(
            choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
            usage={"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
            fidaro={"title": "Capital of France"},
        ),
    ]

    # What aggregate() returns for the same /v1 events with stream:false.
    expected = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "auto",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Paris is the capital.",
                    "reasoning_content": "Thinking about it.",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
        "fidaro": {"title": "Capital of France"},
    }

    acc = StreamAccumulator()
    for c in streamed:
        acc.feed(c)

    assert acc.completion_dict() == expected


def test_fidaro_is_omitted_entirely_when_empty():
    """The orchestrator omits the key rather than sending an empty object, so an
    ordinary response is byte-identical to stock OpenAI output. Match that."""
    acc = StreamAccumulator(model="m", completion_id="c", created=1)
    acc.feed(delta_chunk(content="hi", finish="stop"))

    assert "fidaro" not in acc.completion_dict()

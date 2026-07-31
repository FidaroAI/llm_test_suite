"""StreamingOpenAIProvider — the HTTP half of the streaming path.

Everything here runs against ``httpx.MockTransport``: no network, no credentials. The
data contract (what a chunk sequence adds up to) lives in ``test_streaming.py``; this
file covers what the provider does with bytes, clocks and status codes.
"""

import json
import time

import httpx
import pytest

from llmeval.models import ProviderConfig
from llmeval.providers import (
    LiteLLMProvider,
    StreamingOpenAIProvider,
    build_provider,
    extract_provider_specific,
)

BASE = "http://fidaro.test/v2"


def cfg(**kw):
    kw.setdefault("name", "fidaro-test")
    kw.setdefault("model", "openai/auto")
    kw.setdefault("base_url", BASE)
    kw.setdefault("stream", True)
    return ProviderConfig(**kw)


def frame(**payload):
    base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1700000000,
            "model": "auto"}
    base.update(payload)
    return f"data: {json.dumps(base)}\n\n".encode()


def content_frame(text=None, reasoning=None, finish=None, **extra):
    delta = {}
    if text is not None:
        delta["content"] = text
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    return frame(choices=[{"index": 0, "delta": delta, "finish_reason": finish}], **extra)


DONE_FRAME = b"data: [DONE]\n\n"


def serve(chunks, status=200, captured=None):
    """A MockTransport handler streaming ``chunks`` (an iterable of byte blobs)."""

    def handler(request: httpx.Request) -> httpx.Response:
        if captured is not None:
            captured["request"] = request
            captured["body"] = json.loads(request.content)
        if status >= 400:
            return httpx.Response(status, text="upstream exploded")
        return httpx.Response(status, content=iter(chunks))

    return handler


def install(monkeypatch, handler):
    """Route every ``httpx.Client`` the provider opens through ``handler``."""
    real_client = httpx.Client

    def factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", factory)


# --- the happy path --------------------------------------------------------


FULL_STREAM = [
    frame(choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]),
    content_frame(reasoning="Thinking."),
    content_frame(text="Paris"),
    frame(choices=[{"index": 0, "delta": {}, "finish_reason": None}],
          fidaro={"title": "Capital of France"}),
    content_frame(text=" is the capital."),
    frame(choices=[{"index": 0, "delta": {}, "finish_reason": "stop"}],
          usage={"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21},
          fidaro={"title": "Capital of France"}),
    DONE_FRAME,
]


def test_a_complete_stream_produces_a_clean_completion(monkeypatch):
    install(monkeypatch, serve(FULL_STREAM))
    comp = StreamingOpenAIProvider(cfg()).complete([{"role": "user", "content": "hi"}])

    assert comp.error is None
    assert comp.output == "Paris is the capital."
    assert comp.reasoning == "Thinking."
    assert comp.tokens == {"prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21}
    assert comp.provider_specific == {"fidaro": {"title": "Capital of France"}}
    assert comp.latency_ms is not None


def test_raw_is_the_reconstructed_non_streaming_object(monkeypatch):
    """``raw`` must mean the same thing on both paths, so a stored result can be read
    without knowing which one produced it."""
    install(monkeypatch, serve(FULL_STREAM))
    comp = StreamingOpenAIProvider(cfg()).complete([])

    assert comp.raw["object"] == "chat.completion"
    assert comp.raw["choices"][0]["message"]["content"] == "Paris is the capital."
    assert comp.raw["choices"][0]["finish_reason"] == "stop"
    assert comp.raw["fidaro"] == {"title": "Capital of France"}


# --- timeouts, which is the point ------------------------------------------


def test_a_deadline_keeps_the_partial_answer(monkeypatch):
    """The behaviour the whole feature exists for.

    A model stuck in a repetitive loop streams forever. When the deadline passes we must
    come back with the text it was looping over — not an empty error row, which is what
    a non-streaming call leaves behind and why these tests were previously unrunnable.
    """

    def endless():
        yield content_frame(reasoning="thinking ")
        while True:
            yield content_frame(text="again and ")
            time.sleep(0.02)

    install(monkeypatch, serve(endless()))
    started = time.monotonic()
    comp = StreamingOpenAIProvider(cfg()).complete([], timeout=0.3)
    elapsed = time.monotonic() - started

    assert comp.error is not None
    assert "stream timeout after" in comp.error
    assert "again and " in comp.output          # the loop is on record
    assert comp.reasoning == "thinking "        # so is the reasoning
    assert elapsed < 3.0                        # and it actually stopped


def test_the_timeout_error_reports_how_much_arrived(monkeypatch):
    """Character counts in the message: a timeout holding 40k chars is a model that
    would not stop talking; one holding zero is a backend that never started."""

    def endless():
        while True:
            yield content_frame(text="x" * 100)
            time.sleep(0.01)

    install(monkeypatch, serve(endless()))
    comp = StreamingOpenAIProvider(cfg()).complete([], timeout=0.2)

    assert f"content: {len(comp.output)} chars" in comp.error
    assert "reasoning: 0 chars" in comp.error


def test_a_stalled_socket_also_keeps_the_partial(monkeypatch):
    """A read timeout raised from underneath must not throw the text away either."""

    def stalls():
        yield content_frame(text="half an ans")
        raise httpx.ReadTimeout("socket went quiet")

    install(monkeypatch, serve(stalls()))
    comp = StreamingOpenAIProvider(cfg()).complete([], timeout=5.0)

    assert comp.output == "half an ans"
    assert "stream timeout after" in comp.error


def test_no_timeout_means_no_deadline(monkeypatch):
    # An embedder calling complete() directly must not have one invented for it.
    install(monkeypatch, serve(FULL_STREAM))
    comp = StreamingOpenAIProvider(cfg()).complete([])
    assert comp.error is None


# --- truncation and failure ------------------------------------------------


def test_a_stream_ending_without_done_is_flagged_but_kept(monkeypatch):
    """Distinct from a timeout — nobody was waiting — but the text is just as real."""
    install(monkeypatch, serve([content_frame(text="cut off mid-")]))
    comp = StreamingOpenAIProvider(cfg()).complete([], timeout=5.0)

    assert comp.output == "cut off mid-"
    assert "ended without [DONE]" in comp.error
    assert "1 chunk(s)" in comp.error


def test_a_non_2xx_raises_so_the_runner_can_retry(monkeypatch):
    """An error page holds no model output, so there is nothing to preserve — and
    unlike a timeout, calling again might genuinely work."""
    install(monkeypatch, serve([], status=503))

    with pytest.raises(RuntimeError, match="HTTP 503"):
        StreamingOpenAIProvider(cfg()).complete([], timeout=5.0)


def test_the_error_body_is_quoted_back(monkeypatch):
    install(monkeypatch, serve([], status=400))

    with pytest.raises(RuntimeError, match="upstream exploded"):
        StreamingOpenAIProvider(cfg()).complete([], timeout=5.0)


def test_streaming_without_a_base_url_is_refused():
    with pytest.raises(ValueError, match="no base_url"):
        StreamingOpenAIProvider(cfg(base_url=None)).complete([])


# --- the request we send ---------------------------------------------------


def test_the_request_asks_for_a_stream_with_usage(monkeypatch):
    """Without ``include_usage`` the server has no reason to send token counts, and a
    streamed row would then differ from a non-streamed one where it matters."""
    captured = {}
    install(monkeypatch, serve(FULL_STREAM, captured=captured))
    StreamingOpenAIProvider(cfg()).complete([{"role": "user", "content": "hi"}])

    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert captured["request"].url.path.endswith("/chat/completions")


def test_the_litellm_routing_prefix_is_stripped(monkeypatch):
    # "openai/" is a client-side routing convention; the server wants the bare name.
    captured = {}
    install(monkeypatch, serve(FULL_STREAM, captured=captured))
    StreamingOpenAIProvider(cfg(model="openai/Qwen3-Next-80B")).complete([])

    assert captured["body"]["model"] == "Qwen3-Next-80B"


def test_params_and_extra_body_are_sent_flat(monkeypatch):
    """The OpenAI SDK merges ``extra_body`` into the request root, so a stock client
    produces the same flat shape. One shape for native and extension params alike."""
    captured = {}
    install(monkeypatch, serve(FULL_STREAM, captured=captured))
    config = cfg(params={"temperature": 0.7, "extra_body": {"enable_thinking": True}})
    StreamingOpenAIProvider(config).complete([])

    assert captured["body"]["temperature"] == 0.7
    assert captured["body"]["enable_thinking"] is True
    assert "extra_body" not in captured["body"]


def test_the_api_key_becomes_a_bearer_header(monkeypatch):
    captured = {}
    monkeypatch.setenv("FIDARO_TEST_KEY", "sk-secret")
    install(monkeypatch, serve(FULL_STREAM, captured=captured))
    StreamingOpenAIProvider(cfg(api_key_env="FIDARO_TEST_KEY")).complete([])

    assert captured["request"].headers["authorization"] == "Bearer sk-secret"


def test_a_missing_api_key_sends_no_auth_header(monkeypatch):
    captured = {}
    monkeypatch.delenv("FIDARO_TEST_KEY", raising=False)
    install(monkeypatch, serve(FULL_STREAM, captured=captured))
    StreamingOpenAIProvider(cfg(api_key_env="FIDARO_TEST_KEY")).complete([])

    assert "authorization" not in captured["request"].headers


# --- selection -------------------------------------------------------------


def test_stream_true_selects_the_streaming_provider():
    assert isinstance(build_provider(cfg()), StreamingOpenAIProvider)


def test_stream_false_still_selects_litellm():
    assert isinstance(build_provider(cfg(stream=False)), LiteLLMProvider)


def test_an_explicit_provider_impl_still_wins():
    config = cfg(model="echo", extra={"provider_impl": "echo"})
    assert type(build_provider(config)).__name__ == "EchoProvider"


def test_streaming_a_non_openai_model_is_refused():
    """Refused rather than quietly falling back: a config that asks to stream and
    silently doesn't looks identical in the store until a timeout throws the partial
    away — the one case it was set for."""
    with pytest.raises(ValueError, match="not OpenAI-compatible"):
        build_provider(cfg(model="bedrock/claude"))


# --- provider-specific extraction ------------------------------------------


def test_extraction_reads_a_mapping():
    assert extract_provider_specific({"fidaro": {"title": "T"}}) == {"fidaro": {"title": "T"}}


def test_extraction_reads_an_object():
    class Response:
        fidaro = {"title": "T"}

    assert extract_provider_specific(Response()) == {"fidaro": {"title": "T"}}


def test_extraction_ignores_litellms_own_extra_keys():
    """An allowlist, not "everything unrecognised": litellm decorates its responses with
    several top-level keys of its own, and sweeping those up would bury the real data."""
    response = {"service_tier": "default", "moderation": None, "citations": ["x"]}
    assert extract_provider_specific(response) is None


def test_extraction_of_nothing_is_none():
    # Keeps the store column NULL for an ordinary response.
    assert extract_provider_specific({"choices": []}) is None

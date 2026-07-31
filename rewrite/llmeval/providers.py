"""Provider layer: talk to any LLM, extensibly.

litellm is the default backend (OpenAI, Anthropic, Bedrock, and any OpenAI-compatible
endpoint such as the Fidaro plaintext gateway or Venice). It is **lazy-imported** so the
rest of the suite works without it installed. Anything litellm can't handle can be
registered as a custom factory via :func:`register_provider`.

A provider exposes ``.config`` (a ``ProviderConfig``) and ``.complete(messages)`` →
``Completion``. The runner needs nothing more.

Providers are the *only* place that knows a backend's wire format, so they are also the
only place allowed to reshape it. Every provider returns a ``Completion`` whose ``output``
is the answer alone and whose ``reasoning`` is the reasoning alone; a backend that inlines
reasoning in ``content`` must split it here. Everything downstream — grading, reports,
comparison — then works against one consistent shape and never has to guess.

A config with ``stream: true`` is served by :class:`StreamingOpenAIProvider` instead,
which reads the response as SSE and accumulates it here rather than letting the server
do it. Same row either way — with the one difference that makes it worth having: a call
that hits its timeout leaves the partial answer behind.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Protocol, runtime_checkable

from llmeval.models import ProviderConfig
from llmeval.streaming import DONE, StreamAccumulator, parse_sse_line

logger = logging.getLogger(__name__)

# Top-level response keys that are not part of the OpenAI schema but that we keep.
# An allowlist rather than "everything unrecognised": litellm decorates its own response
# objects with several extra top-level keys (service_tier, moderation, citations,
# provider_specific_fields), and sweeping those up would bury the data we actually want.
PROVIDER_SPECIFIC_KEYS = ("fidaro",)

# litellm model strings are "<provider>/<model>"; only this provider speaks the
# OpenAI-compatible wire format the streaming path implements.
_OPENAI_PREFIX = "openai/"

# How much of an error response body to quote back. Enough to identify the failure,
# short enough not to paste an HTML error page into every log line and result row.
_ERROR_BODY_CHARS = 500


@dataclass
class Completion:
    """One model response, normalised.

    ``error`` is set when the response is *incomplete but usable* — a stream that hit
    its deadline or ended without its terminator. That state has no other way to be
    expressed: the runner otherwise treats a failure as the absence of a completion,
    which would throw away exactly the partial text streaming exists to keep.
    """

    output: str
    raw: Any = None
    reasoning: str | None = None
    tokens: Any = None
    latency_ms: float | None = None
    provider_specific: dict[str, Any] | None = None
    error: str | None = None


def extract_provider_specific(response: Any) -> dict[str, Any] | None:
    """Pull the non-standard top-level keys off a response, or ``None`` if there are none.

    Accepts a mapping or an object with attributes, because the two response paths hand
    back different things: litellm returns a pydantic model, the streaming accumulator a
    plain dict.
    """
    found = {}
    for key in PROVIDER_SPECIFIC_KEYS:
        value = response.get(key) if isinstance(response, dict) else getattr(response, key, None)
        if value is not None:
            # Pydantic models reach us from litellm; store plain JSON either way.
            found[key] = value.model_dump() if hasattr(value, "model_dump") else value
    return found or None


@runtime_checkable
class Provider(Protocol):
    config: ProviderConfig

    def complete(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> Completion: ...


class LiteLLMProvider:
    """Default provider, backed by litellm.completion."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    def complete(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> Completion:
        """Call the model. ``timeout`` is seconds for this one call.

        ``None`` means "don't pass one", leaving whatever the config's ``params`` or
        litellm itself decided — litellm's own default is 6000s, so a caller that wants
        a real ceiling has to say so. The runner always does.

        The retry layer inside litellm's OpenAI client is switched **off** (see
        ``max_retries`` below), because llmeval does its own retrying and records each
        attempt.
        """
        import litellm  # lazy: only needed for real calls

        # litellm._turn_on_debug()  # pylint: disable=protected-access

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            **self.config.params,
        }
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        if self.config.api_key_env:
            kwargs["api_key"] = os.environ.get(self.config.api_key_env)
        if timeout is not None:
            # Set after ``params`` so an explicit per-call timeout wins over one parked
            # in the config, which would otherwise also be part of the cache key.
            kwargs["timeout"] = timeout
        # A timeout is worthless if something underneath retries it: the OpenAI client
        # litellm wraps retries twice by default *with backoff*, so a 2s ceiling really
        # took ~7.5s and a 60s one could run for minutes — and those calls never reached
        # the store. Retry policy belongs to the runner, which stores every attempt.
        # setdefault, so a config that genuinely wants SDK retries can still ask.
        kwargs.setdefault("max_retries", 0)

        t0 = time.time()
        resp = litellm.completion(**kwargs)
        latency_ms = (time.time() - t0) * 1000.0

        msg = resp.choices[0].message
        # ``output`` is always the answer alone and ``reasoning`` always the reasoning
        # alone. Reshaping between the two belongs here, in the provider that knows its
        # own wire format — never downstream, where a grader would have to guess.
        output = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)

        usage = getattr(resp, "usage", None)
        tokens = usage.model_dump() if hasattr(usage, "model_dump") else usage
        raw = resp.model_dump() if hasattr(resp, "model_dump") else None
        # litellm passes unrecognised top-level keys through untouched on this path, so
        # the `fidaro` envelope survives and a non-streamed row carries the same
        # side-channel data a streamed one does.
        return Completion(
            output=output,
            raw=raw,
            reasoning=reasoning,
            tokens=tokens,
            latency_ms=latency_ms,
            provider_specific=extract_provider_specific(resp),
        )


class StreamingOpenAIProvider:
    """Consume an OpenAI-compatible SSE stream, accumulating it client-side.

    Used when a config sets ``stream: true``. Speaks plain HTTP over ``httpx`` rather
    than going through litellm, for three measured reasons — litellm's streaming path
    replaces the server's ``usage`` with a local tokenizer estimate, its
    ``stream_chunk_builder`` drops non-standard top-level keys such as ``fidaro``, and a
    read timeout ends its iteration *silently*, making a truncated stream
    indistinguishable from a complete one. That last one is disqualifying here: telling
    those two apart is the entire point of streaming in this suite.

    The value streaming adds is what survives a timeout. ``timeout`` is a total
    wall-clock deadline measured from the start of the request; when it passes, whatever
    has arrived is returned as a ``Completion`` carrying an ``error`` rather than raised
    away. A model stuck in a repetitive loop can then be caught in the act, because the
    text it was looping over is on record.

    Only OpenAI-compatible endpoints are supported (see :func:`build_provider`), which
    covers the Fidaro orchestrator's ``/v2`` and any vLLM-style server. Note that ``/v1``
    on the Fidaro plaintext gateway is *not* one of these: it emits the older
    ``event: chunk`` frames and no ``fidaro`` envelope.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    # --- request construction ------------------------------------------------

    def _url(self) -> str:
        base = (self.config.base_url or "").rstrip("/")
        if not base:
            raise ValueError(
                f"provider {self.config.name!r} streams but has no base_url; "
                "streaming needs an explicit OpenAI-compatible endpoint"
            )
        return f"{base}/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self.config.api_key_env:
            key = os.environ.get(self.config.api_key_env)
            if key:
                headers["Authorization"] = f"Bearer {key}"
        return headers

    def _body(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        """The request body. ``params`` are sent as-is, at the top level.

        The OpenAI SDK merges ``extra_body`` into the request root, so a flat spread is
        what a server sees from a stock client too — one shape for native and extension
        parameters alike.

        ``stream_options.include_usage`` is requested because without it the server has
        no reason to send token counts at all, and a streamed row would then differ from
        a non-streamed one in a way that matters.
        """
        params = {k: v for k, v in self.config.params.items() if k != "extra_body"}
        params.update(self.config.params.get("extra_body") or {})
        return {
            # litellm's "<provider>/<model>" is a client-side routing convention; the
            # server wants the bare model name.
            "model": self.config.model[len(_OPENAI_PREFIX):],
            "messages": messages,
            **params,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

    # --- the call ------------------------------------------------------------

    def complete(
        self, messages: list[dict[str, str]], timeout: float | None = None
    ) -> Completion:
        """Stream one response. ``timeout`` is the total wall-clock deadline in seconds."""
        import httpx  # lazy: only needed for real calls

        accumulator = StreamAccumulator(model=self.config.model)
        started = time.monotonic()
        deadline = None if timeout is None else started + timeout
        error: str | None = None

        # The same figure serves as the read timeout, so a socket that simply goes quiet
        # aborts on its own rather than idling until the deadline. connect gets its own
        # short ceiling: a backend that won't accept a connection is not going to
        # produce partial output, so there is nothing to wait around for.
        timeouts = httpx.Timeout(timeout, connect=min(10.0, timeout or 10.0))
        try:
            with httpx.Client(timeout=timeouts) as client:
                with client.stream(
                    "POST", self._url(), headers=self._headers(), json=self._body(messages)
                ) as response:
                    self._raise_for_status(response)
                    error = self._drain(response.iter_lines(), accumulator, deadline, timeout)
        except httpx.TimeoutException:
            # A read timeout means the stream stalled. Same outcome as the deadline: keep
            # what arrived, say why it stopped. Raising would discard it.
            error = self._timeout_error(accumulator, time.monotonic() - started)

        return self._completion(accumulator, error, (time.monotonic() - started) * 1000.0)

    @staticmethod
    def _raise_for_status(response: Any) -> None:
        """Turn a non-2xx into an exception carrying a readable slice of the body.

        Raised, not returned as a partial: there is no model output in an error page, so
        there is nothing worth keeping, and the runner should be free to retry it.
        """
        if response.status_code < 400:
            return
        response.read()  # the body is unread while streaming
        body = response.text[:_ERROR_BODY_CHARS]
        raise RuntimeError(f"HTTP {response.status_code} from {response.request.url}: {body}")

    def _drain(
        self,
        lines: Iterable[str],
        accumulator: StreamAccumulator,
        deadline: float | None,
        timeout: float | None,
    ) -> str | None:
        """Feed every chunk into ``accumulator``. Returns an error string, or ``None``.

        Checking the clock between lines is what makes the deadline real: the response
        body is closed by the enclosing ``with``, so abandoning the loop tears the
        connection down instead of politely reading a looping model to completion.
        """
        saw_done = False
        for line in lines:
            if deadline is not None and time.monotonic() > deadline:
                return self._timeout_error(accumulator, timeout or 0.0)
            parsed = parse_sse_line(line)
            if parsed is DONE:
                saw_done = True
                break
            if isinstance(parsed, dict):
                accumulator.feed(parsed)

        if not saw_done:
            # The server hung up mid-answer. Distinct from a timeout — nothing was
            # waiting — but handled the same way, because the text is just as real.
            return (
                f"stream ended without [DONE] after {accumulator.chunks} chunk(s) "
                f"(content: {len(accumulator.content)} chars)"
            )
        return None

    @staticmethod
    def _timeout_error(accumulator: StreamAccumulator, seconds: float) -> str:
        """Why the stream stopped, and how much of an answer we hold.

        The character counts go in the message because they are what a human reading a
        run log actually wants: a timeout with 40k characters of output is a model that
        would not stop talking, and one with zero is a backend that never started.
        """
        return (
            f"stream timeout after {seconds:.1f}s "
            f"(content: {len(accumulator.content)} chars, "
            f"reasoning: {len(accumulator.reasoning or '')} chars)"
        )

    def _completion(
        self, accumulator: StreamAccumulator, error: str | None, latency_ms: float
    ) -> Completion:
        if error:
            logger.warning("%s: %s", self.config.name, error)
        return Completion(
            output=accumulator.content,
            # The reconstructed chat.completion, so `raw` means the same thing on both
            # paths and a streamed result can be read without knowing it was streamed.
            raw=accumulator.completion_dict(),
            reasoning=accumulator.reasoning,
            tokens=accumulator.usage,
            latency_ms=latency_ms,
            provider_specific=accumulator.provider_specific,
            error=error,
        )


class EchoProvider:
    """Returns the last user message as the output. No network, no cost.

    Useful for dry-runs: validate generation, grading, and report wiring end-to-end
    without spending tokens.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        timeout: float | None = None,  # pylint: disable=unused-argument
    ) -> Completion:
        # Accepted and ignored: no network, so nothing here can hang. The parameter has
        # to keep its name — the runner passes it by keyword to every provider alike.
        last_user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        return Completion(output=last_user)


# Extensibility: map a config.extra["provider_impl"] tag to a factory.
ProviderFactory = Callable[[ProviderConfig], Provider]
_FACTORIES: dict[str, ProviderFactory] = {"echo": EchoProvider}


def register_provider(impl: str, factory: ProviderFactory) -> None:
    _FACTORIES[impl] = factory


def build_provider(config: ProviderConfig) -> Provider:
    """Pick the implementation for a config.

    An explicit ``extra.provider_impl`` wins outright — a registered factory is the user
    saying they know better. Otherwise ``stream`` chooses between the hand-rolled SSE
    path and litellm.
    """
    impl = config.extra.get("provider_impl")
    if impl and impl in _FACTORIES:
        return _FACTORIES[impl](config)
    if config.stream:
        if not config.model.startswith(_OPENAI_PREFIX):
            # Refused rather than quietly falling back: a config that asks to stream and
            # silently doesn't would look identical in the store right up until a
            # timeout threw the partial away, which is the one case it was set for.
            raise ValueError(
                f"provider {config.name!r} sets stream=true, but model {config.model!r} "
                f"is not OpenAI-compatible (expected a {_OPENAI_PREFIX!r} prefix). "
                "Streaming is implemented for OpenAI-compatible SSE endpoints only."
            )
        return StreamingOpenAIProvider(config)
    return LiteLLMProvider(config)


def make_litellm_judge(config: ProviderConfig) -> Callable[[str], str]:
    """Build a ``judge(prompt) -> text`` callable from a provider config."""
    provider = build_provider(config)

    def judge(prompt: str) -> str:
        return provider.complete([{"role": "user", "content": prompt}]).output

    return judge

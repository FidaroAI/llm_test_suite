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
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from llmeval.models import ProviderConfig


@dataclass
class Completion:
    output: str
    raw: Any = None
    reasoning: str | None = None
    tokens: Any = None
    latency_ms: float | None = None


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
        return Completion(
            output=output, raw=raw, reasoning=reasoning, tokens=tokens, latency_ms=latency_ms
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
    impl = config.extra.get("provider_impl")
    if impl and impl in _FACTORIES:
        return _FACTORIES[impl](config)
    return LiteLLMProvider(config)


def make_litellm_judge(config: ProviderConfig) -> Callable[[str], str]:
    """Build a ``judge(prompt) -> text`` callable from a provider config."""
    provider = build_provider(config)

    def judge(prompt: str) -> str:
        return provider.complete([{"role": "user", "content": prompt}]).output

    return judge

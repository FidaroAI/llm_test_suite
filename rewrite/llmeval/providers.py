"""Provider layer: talk to any LLM, extensibly.

litellm is the default backend (OpenAI, Anthropic, Bedrock, and any OpenAI-compatible
endpoint such as the Fidaro plaintext gateway or Venice). It is **lazy-imported** so the
rest of the suite works without it installed. Anything litellm can't handle can be
registered as a custom factory via :func:`register_provider`.

A provider exposes ``.config`` (a ``ProviderConfig``) and ``.complete(messages)`` →
``Completion``. The runner needs nothing more.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from llmeval.models import ProviderConfig
from llmeval.response import DELIMITER


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

    def complete(self, messages: list[dict[str, str]]) -> Completion: ...


class LiteLLMProvider:
    """Default provider, backed by litellm.completion."""

    def __init__(self, config: ProviderConfig):
        self.config = config

    def complete(self, messages: list[dict[str, str]]) -> Completion:
        import litellm  # lazy: only needed for real calls

        litellm._turn_on_debug()  # pylint: disable=protected-access

        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            **self.config.params,
        }
        if self.config.base_url:
            kwargs["api_base"] = self.config.base_url
        if self.config.api_key_env:
            kwargs["api_key"] = os.environ.get(self.config.api_key_env)

        t0 = time.time()
        resp = litellm.completion(**kwargs)
        latency_ms = (time.time() - t0) * 1000.0

        msg = resp.choices[0].message
        output = msg.content or ""
        reasoning = getattr(msg, "reasoning_content", None)
        # Normalise to the canonical "reasoning\n\n\nanswer" shape so the shared
        # strip transform isolates the answer for every provider (some return
        # reasoning in a separate field rather than inline).
        if reasoning and DELIMITER not in output:
            output = f"{reasoning}{DELIMITER}{output}"

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

    def complete(self, messages: list[dict[str, str]]) -> Completion:
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

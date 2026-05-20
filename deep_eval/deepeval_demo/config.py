"""Environment-driven config, with fallbacks to the parent promptfoo suite's vars.

Kept tiny and dependency-light so tests can ask "is this configured?" and skip
cleanly (rather than erroring) when an endpoint or key is absent.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


def _first(*names: str, default: str | None = None) -> str | None:
    """Return the first non-empty environment value among `names`."""
    for n in names:
        v = os.getenv(n)
        if v:
            return v
    return default


@dataclass(frozen=True)
class ModelUnderTestConfig:
    base_url: str | None
    api_key: str
    model_id: str | None
    strip_thinking: bool

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model_id)

    @property
    def missing(self) -> str:
        gaps = []
        if not self.base_url:
            gaps.append("MUT_BASE_URL (or VLLM_BASE_URL)")
        if not self.model_id:
            gaps.append("MUT_MODEL_ID (or VLLM_MODEL_ID)")
        return ", ".join(gaps)


@dataclass(frozen=True)
class JudgeConfig:
    provider: str          # "anthropic" | "openai"
    model: str
    api_key: str | None
    base_url: str | None   # only for the openai-compatible provider

    @property
    def is_configured(self) -> bool:
        if self.provider == "anthropic":
            return bool(self.api_key)
        return bool(self.api_key and self.base_url)

    @property
    def missing(self) -> str:
        if self.provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        return "JUDGE_API_KEY and JUDGE_BASE_URL"


def model_under_test_config() -> ModelUnderTestConfig:
    return ModelUnderTestConfig(
        base_url=_first("MUT_BASE_URL", "VLLM_BASE_URL"),
        api_key=_first("MUT_API_KEY", "VLLM_API_KEY", default="dummy"),
        model_id=_first("MUT_MODEL_ID", "VLLM_MODEL_ID"),
        strip_thinking=_first("MUT_STRIP_THINKING", default="1") not in ("0", "false", "False", ""),
    )


def judge_config() -> JudgeConfig:
    provider = (_first("JUDGE_PROVIDER", default="anthropic") or "anthropic").lower()
    if provider == "anthropic":
        return JudgeConfig(
            provider="anthropic",
            model=_first("JUDGE_MODEL", default="claude-sonnet-4-6"),
            api_key=_first("ANTHROPIC_API_KEY"),
            base_url=None,
        )
    return JudgeConfig(
        provider="openai",
        model=_first("JUDGE_MODEL", default="gpt-4o"),
        api_key=_first("JUDGE_API_KEY", "OPENAI_API_KEY"),
        base_url=_first("JUDGE_BASE_URL"),
    )

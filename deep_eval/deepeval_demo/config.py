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


def _aws_chain_present() -> bool:
    """Heuristic: are AWS SigV4 credentials available via env/profile?

    We can't see credentials in ~/.aws or instance metadata, so this only checks
    the env vars — enough to keep the live test's skip behaviour tidy out of the
    box without erroring when no AWS auth is configured at all.
    """
    return bool(_first("AWS_ACCESS_KEY_ID", "AWS_PROFILE"))


@dataclass(frozen=True)
class JudgeConfig:
    provider: str          # "anthropic" | "openai" | "bedrock"
    model: str
    api_key: str | None
    base_url: str | None   # only for the openai-compatible provider
    region: str | None = None   # only for the bedrock provider

    @property
    def is_configured(self) -> bool:
        if self.provider == "anthropic":
            return bool(self.api_key)
        if self.provider == "bedrock":
            # AnthropicBedrock authenticates with a Bedrock API key (bearer token,
            # our `api_key`) OR the standard AWS SigV4 credential chain.
            return bool(self.api_key) or _aws_chain_present()
        return bool(self.api_key and self.base_url)

    @property
    def missing(self) -> str:
        if self.provider == "anthropic":
            return "ANTHROPIC_API_KEY"
        if self.provider == "bedrock":
            return "AWS_BEARER_TOKEN_BEDROCK or AWS credentials (AWS_ACCESS_KEY_ID / AWS_PROFILE)"
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
    if provider == "bedrock":
        # Native Bedrock (not the OpenAI-compatible /openai/v1 shim, which only
        # serves the gpt-oss models). Defaults to the same Haiku 4.5 cross-region
        # inference profile the parent promptfoo suite judges with.
        return JudgeConfig(
            provider="bedrock",
            model=_first("JUDGE_MODEL", default="us.anthropic.claude-haiku-4-5-20251001-v1:0"),
            # Bearer token (AWS_BEARER_TOKEN_BEDROCK); None falls through to the
            # AWS SigV4 credential chain inside AnthropicBedrock.
            api_key=_first("JUDGE_API_KEY", "AWS_BEARER_TOKEN_BEDROCK"),
            base_url=None,
            region=_first("JUDGE_REGION", "AWS_REGION", default="us-east-1"),
        )
    return JudgeConfig(
        provider="openai",
        model=_first("JUDGE_MODEL", default="gpt-4o"),
        api_key=_first("JUDGE_API_KEY", "OPENAI_API_KEY"),
        base_url=_first("JUDGE_BASE_URL"),
    )

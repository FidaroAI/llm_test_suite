"""Unit tests for judge_config() provider selection.

No network, no LLM — pure config branching, so this runs out of the box and
guards the three judge backends (anthropic / openai / bedrock).
"""
import pytest

from deepeval_demo import config

_JUDGE_ENV = [
    "JUDGE_PROVIDER", "JUDGE_MODEL", "JUDGE_API_KEY", "JUDGE_BASE_URL", "JUDGE_REGION",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
    "AWS_BEARER_TOKEN_BEDROCK", "AWS_ACCESS_KEY_ID", "AWS_PROFILE", "AWS_REGION",
]


@pytest.fixture
def clean_env(monkeypatch):
    for k in _JUDGE_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_defaults_to_anthropic_and_unconfigured(clean_env):
    cfg = config.judge_config()
    assert cfg.provider == "anthropic"
    assert not cfg.is_configured
    assert cfg.missing == "ANTHROPIC_API_KEY"


def test_openai_needs_key_and_url(clean_env):
    clean_env.setenv("JUDGE_PROVIDER", "openai")
    clean_env.setenv("JUDGE_API_KEY", "sk-x")
    clean_env.setenv("JUDGE_BASE_URL", "https://api.openai.com/v1")
    cfg = config.judge_config()
    assert cfg.provider == "openai"
    assert cfg.is_configured


def test_bedrock_uses_bearer_token(clean_env):
    clean_env.setenv("JUDGE_PROVIDER", "bedrock")
    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSK-fake")
    clean_env.setenv("AWS_REGION", "eu-west-2")
    cfg = config.judge_config()
    assert cfg.provider == "bedrock"
    assert cfg.is_configured
    assert cfg.api_key == "ABSK-fake"
    assert cfg.region == "eu-west-2"
    # Defaults to the same Haiku 4.5 cross-region inference profile as the
    # parent promptfoo suite's judge.
    assert cfg.model == "us.anthropic.claude-haiku-4-5-20251001-v1:0"


def test_bedrock_unconfigured_without_any_credential(clean_env):
    clean_env.setenv("JUDGE_PROVIDER", "bedrock")
    cfg = config.judge_config()
    assert not cfg.is_configured
    assert "AWS" in cfg.missing


def test_bedrock_configured_via_aws_credential_chain(clean_env):
    clean_env.setenv("JUDGE_PROVIDER", "bedrock")
    clean_env.setenv("AWS_ACCESS_KEY_ID", "AKIA-fake")
    cfg = config.judge_config()
    assert cfg.is_configured
    # No explicit region -> falls back to the SDK's default.
    assert cfg.region == "us-east-1"


def test_provider_is_case_insensitive(clean_env):
    clean_env.setenv("JUDGE_PROVIDER", "BedRock")
    clean_env.setenv("AWS_BEARER_TOKEN_BEDROCK", "ABSK-fake")
    cfg = config.judge_config()
    assert cfg.provider == "bedrock"

"""The model under test: an OpenAI-compatible chat call.

This is the piece promptfoo handles for you (its `providers:` block). In DeepEval
*you* produce the `actual_output`, so we keep that here, isolated from scoring.
"""
from __future__ import annotations

import re

from openai import OpenAI

from .config import ModelUnderTestConfig, model_under_test_config

# Matches a leading reasoning block, e.g. "<think> ... </think>", as emitted by
# Qwen3-Thinking and friends. Mirrors the parent suite's strip-before-scoring hook.
_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def generate_answer(
    question: str,
    *,
    system: str | None = None,
    config: ModelUnderTestConfig | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> str:
    """Ask the model under test `question` and return its (cleaned) reply."""
    cfg = config or model_under_test_config()
    if not cfg.is_configured:
        raise RuntimeError(f"model under test not configured; missing: {cfg.missing}")

    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model=cfg.model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = resp.choices[0].message.content or ""
    return _strip_thinking(answer) if cfg.strip_thinking else answer.strip()

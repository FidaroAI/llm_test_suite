"""The model under test, as a LangSmith target function.

LangSmith's `evaluate()` calls this once per dataset example; the return dict
becomes `run.outputs`, which evaluators read. `@traceable` makes the call show
up as a trace in the LangSmith UI. This is the OpenAI-compatible equivalent of
the deep_eval demo's model_under_test — same model, same answers, scored two ways.
"""
from __future__ import annotations

import re

from langsmith import traceable
from openai import OpenAI

from .config import ModelUnderTestConfig, model_under_test_config

_THINK_BLOCK = re.compile(r"^\s*<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)


def _strip_thinking(text: str) -> str:
    return _THINK_BLOCK.sub("", text).strip()


def generate_answer(question: str, *, config: ModelUnderTestConfig | None = None,
                    temperature: float = 0.0, max_tokens: int = 1024) -> str:
    cfg = config or model_under_test_config()
    if not cfg.is_configured:
        raise RuntimeError(f"model under test not configured; missing: {cfg.missing}")
    client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)
    resp = client.chat.completions.create(
        model=cfg.model_id,
        messages=[{"role": "user", "content": question}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = resp.choices[0].message.content or ""
    return _strip_thinking(answer) if cfg.strip_thinking else answer.strip()


@traceable(name="model_under_test")
def run_target(inputs: dict) -> dict:
    """Dataset example -> model output. Both datasets use the `question` key."""
    answer = generate_answer(inputs["question"])
    return {"answer": answer}

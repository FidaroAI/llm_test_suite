"""The LLM-as-judge model used by GEval.

DeepEval calls a judge to score rubric metrics. For OpenAI-native judges it reads
token logprobs; for any *custom* judge (like a cloud Claude model) it instead asks
for a structured `{score, reason}` object. We satisfy that by returning a populated
pydantic instance via `instructor`, which is far more robust than parsing JSON
out of free text.

Backends: Anthropic (default), native AWS Bedrock, or any OpenAI-compatible endpoint.
"""
from __future__ import annotations

from typing import Optional, Type

import instructor
from pydantic import BaseModel

from deepeval.models.base_model import DeepEvalBaseLLM

from .config import JudgeConfig, judge_config


class CloudJudge(DeepEvalBaseLLM):
    def __init__(self, config: Optional[JudgeConfig] = None):
        self._cfg = config or judge_config()
        if not self._cfg.is_configured:
            raise RuntimeError(f"judge not configured; missing: {self._cfg.missing}")
        super().__init__(model=self._cfg.model)

    # DeepEvalBaseLLM.__init__ calls this; build both sync + async clients.
    def load_model(self):
        cfg = self._cfg
        if cfg.provider == "bedrock":
            from anthropic import AnthropicBedrock, AsyncAnthropicBedrock

            # api_key is the Bedrock bearer token (AWS_BEARER_TOKEN_BEDROCK); when
            # None, AnthropicBedrock falls back to the standard AWS SigV4 chain.
            self._raw_sync = AnthropicBedrock(aws_region=cfg.region, api_key=cfg.api_key)
            self._raw_async = AsyncAnthropicBedrock(aws_region=cfg.region, api_key=cfg.api_key)
        elif cfg.provider == "anthropic":
            from anthropic import Anthropic, AsyncAnthropic

            self._raw_sync = Anthropic(api_key=cfg.api_key)
            self._raw_async = AsyncAnthropic(api_key=cfg.api_key)
        else:
            from openai import AsyncOpenAI, OpenAI

            self._raw_sync = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
            self._raw_async = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

        # Bedrock speaks the Anthropic Messages API, so it wraps the same way.
        from_fn = (
            instructor.from_openai
            if cfg.provider == "openai"
            else instructor.from_anthropic
        )
        self._sync = from_fn(self._raw_sync)
        self._async = from_fn(self._raw_async)
        return self._sync

    def get_model_name(self) -> str:
        return f"{self._cfg.provider}:{self._cfg.model}"

    # --- generation -----------------------------------------------------------
    def generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        if schema is not None:
            return self._sync.chat.completions.create(
                model=self._cfg.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
            )
        return self._raw_text(prompt)

    async def a_generate(self, prompt: str, schema: Optional[Type[BaseModel]] = None):
        if schema is not None:
            return await self._async.chat.completions.create(
                model=self._cfg.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
                response_model=schema,
            )
        return await self._a_raw_text(prompt)

    # --- plain-text fallbacks (rarely used; GEval almost always passes a schema)
    def _raw_text(self, prompt: str) -> str:
        if self._cfg.provider in ("anthropic", "bedrock"):
            msg = self._raw_sync.messages.create(
                model=self._cfg.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        r = self._raw_sync.chat.completions.create(
            model=self._cfg.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""

    async def _a_raw_text(self, prompt: str) -> str:
        if self._cfg.provider in ("anthropic", "bedrock"):
            msg = await self._raw_async.messages.create(
                model=self._cfg.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return msg.content[0].text
        r = await self._raw_async.chat.completions.create(
            model=self._cfg.model,
            messages=[{"role": "user", "content": prompt}],
        )
        return r.choices[0].message.content or ""

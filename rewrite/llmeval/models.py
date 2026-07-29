"""Core data structures, shared across all pipeline stages.

These are deliberately plain data: a generated ``TestCase`` is just JSON on disk, a
``ProviderConfig`` is just JSON the user writes. Nothing here calls an LLM or touches
the store — that keeps generation, running, and grading decoupled.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llmeval.cache_key import CacheKey, compute_cache_key


class Message(BaseModel):
    role: str
    content: str


class AssertionSpec(BaseModel):
    """One check against a model output.

    ``value`` is the primary argument (expected substring, rubric criterion, ...).
    ``params`` carries type-specific extras (tolerance, min/max, regex flag, ...).
    ``transform`` is an *opt-in* reshaping applied to the output before grading only;
    it defaults to ``None`` because providers already hand back a consistent shape
    (``output`` is the answer, reasoning lives on its own field). ``id`` lets
    re-grading track a specific assertion even after edits.
    """

    type: str
    value: Any = None
    params: dict[str, Any] = Field(default_factory=dict)
    weight: float = 1.0
    metric: str | None = None
    id: str | None = None
    transform: str | None = None


class TestCase(BaseModel):
    __test__ = False  # domain model named TestCase; not a pytest test class

    id: str
    messages: list[Message]
    assertions: list[AssertionSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TestCase":
        """Build from the standardized on-disk form, accepting input shorthands.

        Accepts ``messages`` (canonical), ``user`` (single user turn), or ``input``
        (a string user turn, or a dict with ``messages``/``user``).
        """
        data = dict(d)
        messages = _coerce_messages(data)
        data["messages"] = messages
        for k in ("user", "input"):
            data.pop(k, None)
        return cls.model_validate(data)

    @property
    def user_text(self) -> str:
        """The last user message — what a judge or report should show as 'the question'."""
        for m in reversed(self.messages):
            if m.role == "user":
                return m.content
        return self.messages[-1].content if self.messages else ""


def _coerce_messages(data: dict[str, Any]) -> list[dict[str, str]]:
    if "messages" in data and data["messages"]:
        return data["messages"]
    if "user" in data:
        return [{"role": "user", "content": data["user"]}]
    inp = data.get("input")
    if isinstance(inp, str):
        return [{"role": "user", "content": inp}]
    if isinstance(inp, dict):
        return _coerce_messages(inp)
    raise ValueError("test case needs 'messages', 'user', or string 'input'")


class ProviderConfig(BaseModel):
    """A model plus how to call it. The user authors these as JSON.

    ``model`` is a litellm model string (e.g. ``openai/Qwen...``, ``bedrock/...``).
    ``params`` are call params; ``extra`` is non-API identity (backend_version, ...).
    ``cache_key_fields`` selects which of {model, *params, *extra} define identity.
    """

    name: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    cache_key_fields: list[str] | None = None

    def cache_key(self) -> CacheKey:
        return compute_cache_key(self.model, self.params, self.extra, self.cache_key_fields)

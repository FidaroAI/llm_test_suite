"""Core data structures, shared across all pipeline stages.

These are deliberately plain data: a generated ``TestCase`` is just JSON on disk, a
``ProviderConfig`` is just JSON the user writes. Nothing here calls an LLM or touches
the store — that keeps generation, running, and grading decoupled.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from pydantic import BaseModel, Field

from llmeval.cache_key import CacheKey, compute_cache_key


class Message(BaseModel):
    role: str
    content: str


def last_user_text(messages: Sequence[Mapping[str, Any]]) -> str:
    """The last user turn — what a judge or a report shows as "the question".

    Takes plain dicts rather than :class:`Message` objects because it has two callers on
    two sides of the store: :attr:`TestCase.user_text` before a run, and the report after
    one, reading the messages back out of ``results``. One definition, so the question a
    judge was asked and the question the report prints cannot drift apart.

    Falls back to the final turn of any role, so a test case ending in an assistant turn
    still shows something rather than an empty cell.
    """
    turns = list(messages)
    for m in reversed(turns):
        if m.get("role") == "user":
            return str(m.get("content", ""))
    return str(turns[-1].get("content", "")) if turns else ""


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
    """One prompt plus the checks against its answer.

    ``timeout`` is the per-inference-call ceiling in seconds, and is optional: ``None``
    means "use the run's default" (``RunPolicy.timeout``). It belongs to the test case
    rather than the provider because slowness is a property of the *task* — a deep
    research prompt needs longer than a one-line factual question, whatever model
    answers it — and because a provider's ``params`` feed its cache key, so a timeout
    parked there would change the identity under test.
    """

    __test__ = False  # domain model named TestCase; not a pytest test class

    id: str
    messages: list[Message]
    assertions: list[AssertionSpec] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    timeout: float | None = None

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
        return last_user_text([m.model_dump() for m in self.messages])


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
    ``cache_key_fields`` selects which of {model, stream, *params, *extra} define
    identity.

    ``stream`` asks the suite to consume the response as an SSE stream and accumulate it
    client-side. The stored row is the same either way — same answer, same reasoning,
    same token counts — with one difference that is the entire reason the flag exists: a
    call that hits its timeout leaves the partial answer and partial reasoning behind
    instead of nothing, which is what a test for a model stuck in a repetitive loop needs
    in order to say so. Streaming is OpenAI-compatible SSE only; see
    :class:`~llmeval.providers.StreamingOpenAIProvider`.

    There is deliberately no timeout here. A timeout belongs to the *task*, not the
    provider (see :class:`TestCase`), and one parked in ``params`` would feed the cache
    key and change the identity under test.
    """

    name: str
    model: str
    params: dict[str, Any] = Field(default_factory=dict)
    extra: dict[str, Any] = Field(default_factory=dict)
    base_url: str | None = None
    api_key_env: str | None = None
    cache_key_fields: list[str] | None = None
    stream: bool = False

    def cache_key(self) -> CacheKey:
        return compute_cache_key(
            self.model, self.params, self.extra, self.cache_key_fields, self.stream
        )

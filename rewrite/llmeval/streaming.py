"""Accumulate an OpenAI-compatible SSE stream into a single chat completion.

Two pieces, both pure — no sockets, no clocks, no imports beyond the stdlib:

* :func:`iter_sse_chunks` turns a stream of text lines into parsed ``data:`` payloads.
* :class:`StreamAccumulator` folds those payloads into the ``chat.completion`` object a
  non-streaming call would have returned.

Purity is the point. "Aggregate the way the backend does" is a claim about *data*, so
keeping the fold free of I/O lets it be tested against a canned chunk list rather than
against a live model. :class:`~llmeval.providers.StreamingOpenAIProvider` supplies the
bytes and the deadline; nothing here knows either exists.

The reference implementation is the Fidaro orchestrator's own
``openai_v2/aggregation.py``, which performs the identical fold server-side when a
client asks for ``stream:false``. Mirroring it is what makes a streamed row and a
non-streamed row the same data:

===========================  ================================================
frame                        contribution
===========================  ================================================
``delta.content``            appended to the answer
``delta.reasoning_content``  appended to the reasoning
``fidaro`` (any chunk)       merged into the accumulated extensions
terminal chunk               ``finish_reason``, and ``usage`` when requested
``data: [DONE]``             end of stream
===========================  ================================================

``fidaro`` is read off *every* chunk that carries one rather than only the last. The
orchestrator emits the chat title in its own chunk with a no-op delta the moment the
title is known, then repeats it on the terminal chunk; merging as we go handles both
placements without having to care which arrived, and accommodates a future key that
only ever appears mid-stream.

Because the accumulator is always in a valid state, a stream that stops early — a
deadline, a dropped connection — still yields everything received up to that point.
That is the whole reason the suite streams at all: a model stuck in a repetitive loop
times out, and the partial text is the evidence needed to prove it looped.
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Iterator

# Terminator of an OpenAI SSE stream. Not JSON, so it must be recognised before parsing.
DONE_SENTINEL = "[DONE]"

_DATA_PREFIX = "data:"


class _Done:
    """Marker for the ``data: [DONE]`` terminator. Distinct from "no chunk here"."""

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<DONE>"


DONE = _Done()


def parse_sse_line(line: str) -> dict[str, Any] | _Done | None:
    """Parse one SSE line: a chunk, :data:`DONE`, or ``None`` for "nothing here".

    Line-at-a-time rather than only a generator because a caller may need to act on the
    terminator itself. A stream that *ends* is not the same event as a stream that is
    *cut off*, and only the terminator tells them apart — which is precisely the
    distinction :class:`~llmeval.providers.StreamingOpenAIProvider` exists to make. Were
    ``[DONE]`` handled privately inside the generator, an exhausted iterator would mean
    both things at once.

    Non-``data:`` fields (``event:``, ``id:``, ``retry:``), SSE comments (a leading
    ``:``) and the blank lines separating frames all return ``None``: the OpenAI chunk
    format puts everything in ``data:``, and a heartbeat comment must not be mistaken
    for content.

    A payload that is not valid JSON raises :class:`ValueError` naming the offending
    line. Skipping it silently would let a backend change its wire format and surface
    only as mysteriously short answers.
    """
    stripped = line.strip()
    if not stripped or stripped.startswith(":") or not stripped.startswith(_DATA_PREFIX):
        return None
    payload = stripped[len(_DATA_PREFIX):].strip()
    if payload == DONE_SENTINEL:
        return DONE
    try:
        parsed = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"malformed SSE data payload: {payload[:200]!r}") from exc
    # A JSON scalar is well-formed but is not a chunk; letting it through would fail
    # later with a much less obvious error.
    return parsed if isinstance(parsed, dict) else None


def iter_sse_chunks(lines: Iterable[str]) -> Iterator[dict[str, Any]]:
    """Yield each chunk payload in ``lines``, stopping at the ``[DONE]`` terminator.

    The convenience form of :func:`parse_sse_line`, for callers that only want the
    chunks. Takes *lines* rather than bytes because every HTTP client worth using
    de-frames the stream already (``httpx.Response.iter_lines``), and lines are trivial
    to fabricate in a test.
    """
    for line in lines:
        parsed = parse_sse_line(line)
        if isinstance(parsed, _Done):
            return
        if parsed is not None:
            yield parsed


class StreamAccumulator:
    """Folds ``chat.completion.chunk`` payloads into one ``chat.completion``.

    Feed chunks in arrival order; read the result at any time. There is no "finished"
    state to reach — :meth:`completion_dict` is meaningful after one chunk, after all
    of them, or after a deadline cut the stream short.
    """

    def __init__(self, *, model: str = "", completion_id: str = "", created: int = 0):
        # Identity fields are seeded from the request and then overwritten by whatever
        # the server sends, so a completion reconstructed from a stream that died before
        # its first chunk still has something sensible in them.
        self.model = model
        self.completion_id = completion_id
        self.created = created
        self.chunks = 0
        self.finish_reason: str | None = None
        self.usage: dict[str, Any] | None = None
        self.fidaro: dict[str, Any] = {}
        self._content: list[str] = []
        self._reasoning: list[str] = []

    # --- accumulation --------------------------------------------------------

    def feed(self, chunk: dict[str, Any]) -> None:
        """Fold one chunk in. Unknown fields are ignored; missing ones are fine."""
        self.chunks += 1

        # Later chunks win: the server is the authority on its own completion id and
        # model, and vLLM-backed routes resolve an "auto" model to a concrete name.
        for attr, key in (("completion_id", "id"), ("model", "model"), ("created", "created")):
            value = chunk.get(key)
            if value:
                setattr(self, attr, value)

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                self._content.append(delta["content"])
            if delta.get("reasoning_content"):
                self._reasoning.append(delta["reasoning_content"])
            if choice.get("finish_reason"):
                self.finish_reason = choice["finish_reason"]

        # Usage rides its own chunk in stock OpenAI and the terminal chunk here; either
        # way the last one seen is the complete count.
        if chunk.get("usage"):
            self.usage = chunk["usage"]

        extensions = chunk.get("fidaro")
        if isinstance(extensions, dict):
            # Shallow merge, skipping nulls: the orchestrator repeats the whole object
            # on the terminal chunk, and a repeat must not blank a key an earlier chunk
            # populated.
            self.fidaro.update({k: v for k, v in extensions.items() if v is not None})

    # --- results -------------------------------------------------------------

    @property
    def content(self) -> str:
        return "".join(self._content)

    @property
    def reasoning(self) -> str | None:
        """The reasoning so far, or ``None`` when the model emitted none.

        ``None`` rather than ``""`` so a non-reasoning model is distinguishable from a
        reasoning one that was cut off before it said anything — and to match what the
        non-streaming path reads off ``message.reasoning_content``.
        """
        return "".join(self._reasoning) or None

    @property
    def provider_specific(self) -> dict[str, Any] | None:
        """The non-standard response data, enveloped under its vendor key.

        ``{"fidaro": {...}}`` rather than the bare inner object, so a second vendor key
        needs no schema change. ``None`` when nothing non-standard arrived, which keeps
        the stored column NULL for an ordinary response.
        """
        return {"fidaro": dict(self.fidaro)} if self.fidaro else None

    def completion_dict(self) -> dict[str, Any]:
        """The ``chat.completion`` object this stream adds up to.

        Byte-for-byte the shape the orchestrator returns for ``stream:false``, so a
        streamed result and a non-streamed one are comparable without special-casing
        which path produced them.
        """
        message: dict[str, Any] = {"role": "assistant", "content": self.content}
        message["reasoning_content"] = self.reasoning
        completion: dict[str, Any] = {
            "id": self.completion_id,
            "object": "chat.completion",
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": self.finish_reason,
                }
            ],
            "usage": self.usage,
        }
        if self.fidaro:
            # Omitted entirely when empty, matching the orchestrator: the key's absence
            # is meaningful, and an empty object would be a different response.
            completion["fidaro"] = dict(self.fidaro)
        return completion

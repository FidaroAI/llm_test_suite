"""Opt-in output transforms, applied *before grading only*.

Normalising a backend's wire format is the **provider's** job: every provider returns
``output`` as the answer alone, with reasoning on its own field. So no transform is
applied by default, and assertions grade the answer as-is.

These exist as an escape hatch for the case the provider layer cannot cover — a corpus
of results already in the store that was captured under a different shape, or a backend
whose format is only distinguishable per test rather than per provider. Set
``transform`` explicitly on an assertion to opt in.

``strip_reasoning`` handles the legacy shape ``<reasoning>\\n\\n\\n<final answer>`` (an
artifact of an older reasoning parser swapping think-tags for newlines). Note it splits
on the *first* triple newline, so it is only safe when the reasoning is known not to
contain a blank line — which is precisely why it is no longer applied by default.
"""

from __future__ import annotations

from typing import Any, Callable

DELIMITER = "\n\n\n"


def _strip_reasoning(output: Any) -> Any:
    if not isinstance(output, str):
        return output
    idx = output.find(DELIMITER)
    if idx == -1:
        return output
    return output[idx + len(DELIMITER):]


def _identity(output: Any) -> Any:
    return output


TRANSFORMS: dict[str, Callable[[Any], Any]] = {
    "strip_reasoning": _strip_reasoning,
    "none": _identity,
    "identity": _identity,
}


def apply_transform(name: str | None, output: Any) -> Any:
    if name is None:
        return output
    return TRANSFORMS[name](output)

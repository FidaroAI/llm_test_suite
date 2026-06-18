"""Output transforms applied *before grading only*.

The stored raw output always keeps everything (reasoning included); a transform only
changes what a given assertion sees. The default ``strip_reasoning`` mirrors the old
suite: reasoning models emit ``<reasoning>\\n\\n\\n<final answer>`` (an artifact of the
reasoning parser swapping think-tags for newlines), so graders should see only the part
after the first triple newline.
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

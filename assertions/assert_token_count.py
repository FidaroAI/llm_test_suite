"""Assert that the model's response is within a length budget.

Supports three units: "tokens" (default, requires tiktoken + a one-time
encoding download), "chars", and "words". Use chars/words to avoid the
tiktoken network dependency on first run.

Config keys (read from `context["config"]`):
    unit       — "tokens" | "chars" | "words" (default: "tokens")
    encoding   — tiktoken encoding name when unit=tokens (default: "cl100k_base")
    min_tokens — inclusive lower bound (default: 0)
    max_tokens — inclusive upper bound (default: 1024)

The min/max keys are named `min_tokens` / `max_tokens` regardless of unit, for
consistency with the most common case.
"""

import sys


def _count(text, unit, encoding_name):
    if unit == "chars":
        return len(text)
    if unit == "words":
        return len(text.split())
    if unit == "tokens":
        import tiktoken

        enc = tiktoken.get_encoding(encoding_name)
        return len(enc.encode(text))
    raise ValueError(f"unknown unit {unit!r}; expected tokens|chars|words")


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    unit = cfg.get("unit", "tokens")
    encoding_name = cfg.get("encoding", "cl100k_base")
    minimum = int(cfg.get("min_tokens", 0))
    maximum = int(cfg.get("max_tokens", sys.maxsize))

    text = output if isinstance(output, str) else str(output)

    try:
        n = _count(text, unit, encoding_name)
    except Exception as e:
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"length count failed (unit={unit}): {e}",
        }

    ok = minimum <= n <= maximum
    return {
        "pass": ok,
        "score": 1.0 if ok else 0.0,
        "reason": f"{unit} count {n} (allowed {minimum}..{maximum})",
    }

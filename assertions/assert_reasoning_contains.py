"""Assert that a value appears in the model's reasoning content.

Reads reasoning from the provider response in priority order:
  1. context["response"]["reasoning_content"] (vLLM with reasoning parser)
  2. context["response"]["thinking"]          (Ollama with think:true)
  3. context["response"]["content"][i] where item type == "thinking" (Claude)

Config keys (read from `context["config"]`):
    value  — substring or regex pattern to match (required)
    regex  — bool, treat value as regex (default: false)
    step   — "any" | int | omitted
             omitted: match against full reasoning text (blocks joined with \\n\\n)
             "any":   match if any block contains the value
             int:     match against block at that index (0-based)

If no reasoning is surfaced, returns pass=False, reason="no reasoning available".
"""

import re


def _split_paragraphs(text):
    parts = [p for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts or [text.strip()]


def _reasoning_blocks(context):
    resp = (context or {}).get("response") or {}

    rc = resp.get("reasoning_content")
    if isinstance(rc, str) and rc.strip():
        return _split_paragraphs(rc)

    t = resp.get("thinking")
    if isinstance(t, str) and t.strip():
        return _split_paragraphs(t)

    blocks = []
    for item in resp.get("content", []) or []:
        if isinstance(item, dict) and item.get("type") == "thinking":
            text = item.get("thinking") or item.get("text") or ""
            if text.strip():
                blocks.append(text)
    return blocks


def _match(value, text, use_regex):
    if use_regex:
        return re.search(value, text) is not None
    return value in text


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    value = cfg.get("value")
    use_regex = bool(cfg.get("regex", False))
    step = cfg.get("step")

    if value is None:
        return {"pass": False, "score": 0.0, "reason": "missing required config 'value'"}

    blocks = _reasoning_blocks(context)
    if not blocks:
        return {"pass": False, "score": 0.0, "reason": "no reasoning available"}

    if step is None:
        whole = "\n\n".join(blocks)
        ok = _match(value, whole, use_regex)
        return {
            "pass": ok,
            "score": 1.0 if ok else 0.0,
            "reason": (
                f"value {'matched' if ok else 'not found'} in reasoning "
                f"(regex={use_regex})"
            ),
        }

    if step == "any":
        for i, b in enumerate(blocks):
            if _match(value, b, use_regex):
                return {
                    "pass": True,
                    "score": 1.0,
                    "reason": f"matched block {i} (regex={use_regex})",
                }
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"no block matched (regex={use_regex})",
        }

    # integer step
    try:
        idx = int(step)
    except (TypeError, ValueError):
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"invalid step={step!r}; expected int or 'any'",
        }
    if idx < 0 or idx >= len(blocks):
        return {
            "pass": False,
            "score": 0.0,
            "reason": f"step={idx} out of range (have {len(blocks)} blocks)",
        }
    ok = _match(value, blocks[idx], use_regex)
    return {
        "pass": ok,
        "score": 1.0 if ok else 0.0,
        "reason": f"value {'matched' if ok else 'not found'} in block {idx}",
    }

"""Assert on the number of reasoning iterations a model surfaced.

Reads reasoning from the same source priority as
`assert_reasoning_contains.py`:
  1. context["response"]["reasoning_content"] — split on blank-line paragraphs
  2. context["response"]["thinking"]          — split on blank-line paragraphs
  3. context["response"]["content"][i] thinking blocks — each block is a step

Counting heuristic per text: max of (numbered/bulleted items, step keywords,
paragraph count). Final iteration count is max across blocks combined OR the
block count when reasoning is structured into blocks.

The Layer 2 transform strips inline <think>...</think> from `output` before
assertions run, so this assertion no longer falls back to parsing `output`.

Config keys:
    min — inclusive lower bound (default: 1)
    max — inclusive upper bound (default: 50)
"""

import re

NUMBERED = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.M)
STEP_KW = re.compile(
    r"\b(step\s*\d+|first(?:ly)?|second(?:ly)?|third(?:ly)?|then|next|finally|therefore)\b",
    re.I,
)


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


def _heuristic_count(text):
    bullets = len(NUMBERED.findall(text))
    keywords = len(STEP_KW.findall(text))
    paragraphs = len([p for p in re.split(r"\n\s*\n", text) if p.strip()])
    return max(bullets, keywords, paragraphs, 1)


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    minimum = int(cfg.get("min", 1))
    maximum = int(cfg.get("max", 50))

    blocks = _reasoning_blocks(context)
    if not blocks:
        return {"pass": False, "score": 0.0, "reason": "no reasoning available"}

    if len(blocks) > 1:
        iters = len(blocks)
    else:
        iters = _heuristic_count(blocks[0])

    ok = minimum <= iters <= maximum
    return {
        "pass": ok,
        "score": 1.0 if ok else 0.0,
        "reason": f"~{iters} reasoning iterations (allowed {minimum}..{maximum})",
    }

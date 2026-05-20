"""Assert on the number of tool calls in the model's response.

Normalizes OpenAI shape (`response.choices[0].message.tool_calls`),
Anthropic/Bedrock content blocks (`response.content[*].type == "tool_use"`),
and raw tool-use objects. Falls back to parsing `output` as JSON for raw HTTP
providers.

Config keys:
    expected — exact count required (overrides min/max if set)
    min      — inclusive lower bound (default: 0)
    max      — inclusive upper bound (default: 99)
"""

import json


def _is_tool_use(obj):
    return isinstance(obj, dict) and (
        obj.get("type") == "tool_use"
        or ("name" in obj and "input" in obj and ("id" in obj or "toolUseId" in obj))
    )


def _from_obj(obj):
    if isinstance(obj, list):
        return [item for item in obj if _is_tool_use(item)]
    if _is_tool_use(obj):
        return [obj]
    if isinstance(obj, dict):
        tool_calls = obj.get("tool_calls")
        if isinstance(tool_calls, list):
            return tool_calls
        if _is_tool_use(tool_calls):
            return [tool_calls]
        content = obj.get("content")
        if isinstance(content, list):
            return [item for item in content if _is_tool_use(item)]
    return []


def _extract_tool_calls(output, context):
    resp = (context or {}).get("response") or {}

    # OpenAI / OpenAI-compatible shape
    choices = resp.get("choices") or []
    if choices:
        msg = (choices[0] or {}).get("message", {}) or {}
        tcs = msg.get("tool_calls")
        if tcs:
            return tcs

    # Anthropic shape: content blocks of type "tool_use"
    calls = _from_obj(resp)
    if calls:
        return calls

    # Last resort: parse output as JSON and look for tool calls or raw tool_use.
    if isinstance(output, str):
        try:
            obj = json.loads(output)
            return _from_obj(obj)
        except (ValueError, TypeError):
            pass

    return []


def get_assert(output, context):
    cfg = (context or {}).get("config") or {}
    expected = cfg.get("expected")
    minimum = int(cfg.get("min", 0))
    maximum = int(cfg.get("max", 99))

    calls = _extract_tool_calls(output, context)
    n = len(calls)

    if expected is not None:
        ok = n == int(expected)
        reason = f"{n} tool call(s); expected exactly {expected}"
    else:
        ok = minimum <= n <= maximum
        reason = f"{n} tool call(s); allowed {minimum}..{maximum}"

    return {"pass": ok, "score": 1.0 if ok else 0.0, "reason": reason}

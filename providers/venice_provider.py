"""Custom promptfoo provider for Venice (https://venice.ai).

Why a custom provider instead of the generic ``openai:chat`` one: Venice returns
its chain-of-thought in a *separate* ``message.reasoning_content`` field, and
promptfoo's openai provider merges that into the graded output as
``"Thinking: " + reasoning + "\\n\\n" + answer``. That leaks the reasoning into
every assertion, and it cannot be stripped after the fact — the reasoning is
multi-paragraph (internal ``\\n\\n``) and the raw field is discarded once merged.

Here we have the whole Venice JSON, so we:

  * keep the full reasoning (and web-search citations) for human analysis, and
  * format the output the **same way the Fidaro gateway does** — the reasoning,
    then a ``\\n\\n\\n`` delimiter, then the answer — so the existing per-assertion
    transform (``hooks/strip_before_triple_newline.py``) strips the reasoning for
    graders while the stored output keeps it. One delimiter, one transform, both
    providers consistent.

promptfoo contract: this file is referenced as ``file://providers/venice_provider.py``
with ``call_api(prompt, options, context)`` returning a result dict. Config
(model / api key / base url / web search) comes from the provider's ``config``
block (templated from ``COMPARISON_VENICE_*`` / ``VENICE_INFERENCE_KEY`` env vars
in promptfooconfig.yaml, set per run by run_comparison.py).
"""

from __future__ import annotations

import json
import re

import requests

# Same delimiter the Fidaro gateway emits between reasoning and answer, so the
# shared hooks/strip_before_triple_newline.py transform isolates the answer for
# graders. Keep in sync with that hook.
THINKING_DELIMITER = "\n\n\n"

DEFAULT_API_BASE_URL = "https://api.venice.ai/api/v1"
REQUEST_TIMEOUT_SECONDS = 180  # web search can be slow

# Collapse any run of 3+ newlines to 2 so the ONLY triple-newline in the output
# is the delimiter we insert — otherwise a triple newline inside the reasoning
# would fool the strip transform into keeping part of the reasoning.
_TRIPLE_PLUS_NEWLINES = re.compile(r"\n{3,}")


def parse_messages(prompt: str) -> list:
    """Render a promptfoo prompt into an OpenAI-style messages list.

    Our prompt template (prompt_templates/user_only.json) renders to a JSON chat
    array; fall back to a single user message for a bare string prompt.
    """
    try:
        parsed = json.loads(prompt)
    except (ValueError, TypeError):
        return [{"role": "user", "content": prompt}]
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        return [parsed]
    return [{"role": "user", "content": prompt}]


def _sanitize_reasoning(text: str) -> str:
    """Trim and collapse 3+ newline runs so our delimiter is unambiguous."""
    return _TRIPLE_PLUS_NEWLINES.sub("\n\n", text.strip())


def format_venice_response(data: dict) -> dict:
    """Turn a raw Venice chat-completion JSON into a promptfoo result dict.

    Pure (no I/O) so it can be unit-tested. The ``output`` is
    ``reasoning + "\\n\\n\\n" + answer`` (answer only, when there is no reasoning),
    which the strip transform reduces to the answer for grading while the stored
    output keeps the reasoning. The full reasoning and any web-search citations
    are also surfaced under ``metadata`` for structured human analysis.
    """
    choices = data.get("choices") or []
    if not choices:
        return {"error": f"Venice response had no choices: {json.dumps(data)[:500]}"}
    message = choices[0].get("message") or {}
    answer = (message.get("content") or "").strip()
    reasoning = _sanitize_reasoning(message.get("reasoning_content") or "")

    output = f"{reasoning}{THINKING_DELIMITER}{answer}" if reasoning else answer

    usage = data.get("usage") or {}
    token_usage = {}
    if usage:
        token_usage = {
            "total": usage.get("total_tokens"),
            "prompt": usage.get("prompt_tokens"),
            "completion": usage.get("completion_tokens"),
        }

    venice_params = data.get("venice_parameters") or {}
    metadata = {
        "venice_reasoning": reasoning,
        "venice_web_search_citations": venice_params.get("web_search_citations") or [],
        "venice_model": data.get("model"),
        "finish_reason": choices[0].get("finish_reason"),
    }

    result: dict = {"output": output, "metadata": metadata}
    if token_usage:
        result["tokenUsage"] = token_usage
    return result


def _request_body(config: dict, messages: list) -> dict:
    """The JSON body for a Venice chat-completions call."""
    return {
        "model": config["model"],
        "messages": messages,
        "venice_parameters": {
            "enable_web_search": config.get("web_search", "off"),
        },
    }


def call_api(prompt, options, context):
    """promptfoo entry point: call Venice and format the response.

    Errors (HTTP status, network, malformed JSON) are returned as
    ``{"error": ...}`` so promptfoo records a real error for the cell rather than
    a silently-empty pass — which is what we want to distinguish genuine failures
    (e.g. 429 rate limits) from low scores.
    """
    config = (options or {}).get("config") or {}
    api_key = config.get("api_key")
    if not api_key:
        return {"error": "venice_provider: missing api_key in provider config"}
    base_url = (config.get("api_base_url") or DEFAULT_API_BASE_URL).rstrip("/")

    body = _request_body(config, parse_messages(prompt))
    try:
        resp = requests.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        return {"error": f"venice_provider: request failed: {exc}"}

    if resp.status_code != 200:
        detail = resp.text[:500]
        return {"error": f"venice_provider: HTTP {resp.status_code}: {detail}"}

    try:
        data = resp.json()
    except ValueError:
        return {"error": f"venice_provider: non-JSON response: {resp.text[:500]}"}

    return format_venice_response(data)

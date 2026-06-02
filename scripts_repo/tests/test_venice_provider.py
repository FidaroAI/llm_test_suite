"""Tests for the custom Venice promptfoo provider (providers/venice_provider.py)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

# Load the provider module by path (providers/ is not an importable package).
_REPO = Path(__file__).resolve().parents[2]


def _load(path_parts, name):
    spec = importlib.util.spec_from_file_location(name, _REPO.joinpath(*path_parts))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


venice = _load(("providers", "venice_provider.py"), "venice_provider")
strip_hook = _load(("hooks", "strip_before_triple_newline.py"), "strip_hook")


def _venice_json(content, reasoning=None, usage=None, citations=None):
    message = {"content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    data = {
        "model": "kimi-k2-6",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    if usage is not None:
        data["usage"] = usage
    if citations is not None:
        data["venice_parameters"] = {"web_search_citations": citations}
    return data


# --- output formatting -----------------------------------------------------


def test_output_is_reasoning_then_delimiter_then_answer():
    out = venice.format_venice_response(
        _venice_json(" The answer.", reasoning=" The user asks X. Let me think.")
    )["output"]
    assert out == "The user asks X. Let me think.\n\n\nThe answer."


def test_output_roundtrips_through_the_real_strip_transform():
    # The whole point: the shared transform must reduce our output to the answer.
    data = _venice_json(
        "Helios was the Greek sun god.\n\nApollo came later.",
        reasoning="Step 1.\n\nStep 2.\n\nLet me draft the answer.",
    )
    out = venice.format_venice_response(data)["output"]
    graded = strip_hook.get_transform(out, {})
    assert graded == "Helios was the Greek sun god.\n\nApollo came later."
    # ...and the stored output still contains the reasoning for human analysis.
    assert "Let me draft the answer." in out


def test_no_reasoning_yields_answer_only_and_no_delimiter():
    out = venice.format_venice_response(_venice_json("Just the answer."))["output"]
    assert out == "Just the answer."
    assert "\n\n\n" not in out
    # transform is a no-op when there is no delimiter
    assert strip_hook.get_transform(out, {}) == "Just the answer."


def test_internal_triple_newlines_in_reasoning_are_collapsed():
    # A stray \n\n\n inside the reasoning must not fool the strip transform.
    data = _venice_json("ANSWER", reasoning="a\n\n\n\nb")
    out = venice.format_venice_response(data)["output"]
    assert out == "a\n\nb\n\n\nANSWER"
    assert strip_hook.get_transform(out, {}) == "ANSWER"


def test_token_usage_is_mapped():
    res = venice.format_venice_response(
        _venice_json("a", usage={"prompt_tokens": 10, "completion_tokens": 5,
                                 "total_tokens": 15})
    )
    assert res["tokenUsage"] == {"total": 15, "prompt": 10, "completion": 5}


def test_metadata_keeps_reasoning_and_citations():
    cites = [{"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Helios"}]
    res = venice.format_venice_response(
        _venice_json("a", reasoning="my chain of thought", citations=cites)
    )
    assert res["metadata"]["venice_reasoning"] == "my chain of thought"
    assert res["metadata"]["venice_web_search_citations"] == cites
    assert res["metadata"]["venice_model"] == "kimi-k2-6"


def test_no_choices_is_an_error():
    assert "error" in venice.format_venice_response({"choices": []})


# --- message parsing -------------------------------------------------------


def test_parse_messages_json_array():
    msgs = venice.parse_messages('[{"role": "user", "content": "hi"}]')
    assert msgs == [{"role": "user", "content": "hi"}]


def test_parse_messages_bare_string_fallback():
    assert venice.parse_messages("just text") == [{"role": "user", "content": "just text"}]


# --- call_api wiring (no real network) -------------------------------------


class _FakeResp:
    def __init__(self, status, payload):
        self.status_code = status
        self._payload = payload
        self.text = "" if isinstance(payload, dict) else str(payload)

    def json(self):
        if isinstance(self._payload, dict):
            return self._payload
        raise ValueError("not json")


def test_call_api_sends_web_search_and_formats(monkeypatch):
    captured = {}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        return _FakeResp(200, _venice_json(" Hi there.", reasoning=" thinking"))

    monkeypatch.setattr(venice.requests, "post", fake_post)
    res = venice.call_api(
        '[{"role": "user", "content": "hi"}]',
        {"config": {"model": "kimi-k2-6", "api_key": "k", "web_search": "on"}},
        {},
    )
    assert captured["url"].endswith("/chat/completions")
    assert captured["headers"]["Authorization"] == "Bearer k"
    assert captured["body"]["model"] == "kimi-k2-6"
    assert captured["body"]["venice_parameters"]["enable_web_search"] == "on"
    assert res["output"] == "thinking\n\n\nHi there."


def test_call_api_missing_key_errors():
    res = venice.call_api("hi", {"config": {"model": "m"}}, {})
    assert "error" in res and "api_key" in res["error"]


def test_call_api_http_error_is_surfaced(monkeypatch):
    monkeypatch.setattr(venice.requests, "post",
                        lambda *a, **k: _FakeResp(429, "rate limited"))
    res = venice.call_api("hi", {"config": {"model": "m", "api_key": "k"}}, {})
    assert "error" in res and "429" in res["error"]

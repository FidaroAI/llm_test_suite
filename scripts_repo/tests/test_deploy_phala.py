"""Unit tests for the pure logic in scripts_repo/deploy_phala.py."""

from __future__ import annotations

import io

import pytest

from scripts_repo.deploy_phala import (
    apply_vllm_options,
    dump_compose,
    load_compose,
    models_url,
    vllm_options_to_command,
    wait_for_url,
)


class _FakeResp:
    def __init__(self, status_code):
        self.status_code = status_code


def _responder(*status_codes):
    """A get_fn that returns the given status codes in turn, then raises."""
    seq = iter(status_codes)

    def get(url, timeout=None):
        try:
            return _FakeResp(next(seq))
        except StopIteration:  # pragma: no cover - guards test misuse
            raise AssertionError("get_fn called more times than expected")

    return get


# A trimmed but faithful copy of the committed Phala compose template: a vllm
# service with a command list, plus another service and comments to prove the
# round-trip preserves everything we don't deliberately touch.
SAMPLE_COMPOSE = """\
# top-of-file comment that must survive
services:
  vllm:
    image: vllm/vllm-openai@sha256:deadbeef
    runtime: nvidia
    ports:
      - "8000:8000"
    command:
      - "--host"
      - "0.0.0.0"
      - "--port"
      - "8000"
      - "--model"
      - "google/gemma-4-31B-it"

  gateway:
    image: example/gateway@sha256:cafebabe
    # gateway comment
    environment:
      - BRAVE_API_KEY=${BRAVE_API_KEY}
"""


def test_vllm_options_to_command_preserves_host_and_port():
    cmd = vllm_options_to_command({"model": "x"})
    assert cmd[:4] == ["--host", "0.0.0.0", "--port", "8000"]


def test_vllm_options_to_command_string_value_becomes_flag_and_value():
    cmd = vllm_options_to_command({"model": "google/gemma-4-31B-it"})
    assert cmd[4:] == ["--model", "google/gemma-4-31B-it"]


def test_vllm_options_to_command_numeric_value_is_stringified():
    cmd = vllm_options_to_command({"max-model-len": 8192})
    assert cmd[4:] == ["--max-model-len", "8192"]


def test_vllm_options_to_command_true_becomes_bare_flag():
    cmd = vllm_options_to_command({"enable-auto-tool-choice": True})
    assert cmd[4:] == ["--enable-auto-tool-choice"]


def test_vllm_options_to_command_false_is_omitted():
    cmd = vllm_options_to_command({"enable-auto-tool-choice": False, "model": "x"})
    assert "--enable-auto-tool-choice" not in cmd
    assert cmd[4:] == ["--model", "x"]


def test_vllm_options_to_command_preserves_option_order():
    cmd = vllm_options_to_command(
        {"model": "x", "reasoning-parser": "gemma4", "tool-call-parser": "gemma4"}
    )
    assert cmd[4:] == [
        "--model",
        "x",
        "--reasoning-parser",
        "gemma4",
        "--tool-call-parser",
        "gemma4",
    ]


def test_apply_vllm_options_replaces_only_the_vllm_command():
    compose = load_compose(io.StringIO(SAMPLE_COMPOSE))
    apply_vllm_options(
        compose, {"model": "deepseek/r1", "enable-auto-tool-choice": True}
    )
    assert list(compose["services"]["vllm"]["command"]) == [
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--model",
        "deepseek/r1",
        "--enable-auto-tool-choice",
    ]


def test_apply_vllm_options_round_trip_preserves_comments_and_other_services():
    compose = load_compose(io.StringIO(SAMPLE_COMPOSE))
    apply_vllm_options(compose, {"model": "deepseek/r1"})
    out = io.StringIO()
    dump_compose(compose, out)
    text = out.getvalue()
    assert "# top-of-file comment that must survive" in text
    assert "# gateway comment" in text
    # Untouched bits of the other service survive verbatim.
    assert "BRAVE_API_KEY=${BRAVE_API_KEY}" in text
    assert "deepseek/r1" in text
    # The old model is gone from the command.
    assert "google/gemma-4-31B-it" not in text


def test_apply_vllm_options_errors_without_vllm_service():
    compose = load_compose(io.StringIO("services:\n  gateway:\n    image: x\n"))
    with pytest.raises(KeyError):
        apply_vllm_options(compose, {"model": "x"})


def test_models_url_appends_models_to_v1_base():
    assert models_url("https://host-8000.example.net/v1") == (
        "https://host-8000.example.net/v1/models"
    )


def test_models_url_tolerates_trailing_slash():
    assert models_url("https://host/v1/") == "https://host/v1/models"


def test_wait_for_vllm_returns_when_ready_immediately():
    calls = []
    wait_for_url(
        "https://host/v1",
        timeout_s=10,
        interval_s=1,
        get_fn=_responder(200),
        sleep_fn=lambda s: calls.append(s),
        now_fn=lambda: 0.0,
    )
    assert calls == []  # ready first try, never slept


def test_wait_for_vllm_retries_until_ready():
    slept = []
    clock = [0.0]

    def now():
        return clock[0]

    def sleep(s):
        slept.append(s)
        clock[0] += s

    wait_for_url(
        "https://host/v1",
        timeout_s=100,
        interval_s=5,
        get_fn=_responder(503, 503, 200),
        sleep_fn=sleep,
        now_fn=now,
    )
    assert slept == [5, 5]


def test_wait_for_vllm_raises_timeout_when_never_ready():
    clock = [0.0]

    def now():
        return clock[0]

    def sleep(s):
        clock[0] += s

    with pytest.raises(TimeoutError):
        wait_for_url(
            "https://host/v1",
            timeout_s=10,
            interval_s=5,
            get_fn=lambda url, timeout=None: _FakeResp(503),
            sleep_fn=sleep,
            now_fn=now,
        )

import sys
import types

from llmeval.models import ProviderConfig
from llmeval.providers import Completion, LiteLLMProvider, build_provider, register_provider


def test_echo_provider_returns_last_user_message():
    cfg = ProviderConfig(name="e", model="echo", extra={"provider_impl": "echo"})
    p = build_provider(cfg)
    out = p.complete([{"role": "system", "content": "x"}, {"role": "user", "content": "hello there"}])
    assert out.output == "hello there"


def test_custom_provider_can_be_registered():
    class MyProvider:
        def __init__(self, config):
            self.config = config

        def complete(self, messages):
            return Completion(output="custom!")

    register_provider("myimpl", MyProvider)
    cfg = ProviderConfig(name="c", model="x", extra={"provider_impl": "myimpl"})
    assert build_provider(cfg).complete([]).output == "custom!"


def test_default_provider_is_litellm_backed():
    cfg = ProviderConfig(name="d", model="openai/gpt-4o")
    p = build_provider(cfg)
    assert type(p).__name__ == "LiteLLMProvider"


class _FakeMessage:
    def __init__(self, content, reasoning_content=None):
        self.content = content
        self.reasoning_content = reasoning_content


class _FakeResponse:
    def __init__(self, content, reasoning_content=None):
        self.choices = [types.SimpleNamespace(message=_FakeMessage(content, reasoning_content))]
        self.usage = None


def _stub_litellm(monkeypatch, response, seen=None):
    """Install a fake ``litellm`` module so ``complete`` needs no network.

    Pass ``seen`` to capture the kwargs the provider handed to ``litellm.completion``.
    """

    def completion(**kwargs):
        if seen is not None:
            seen.update(kwargs)
        return response

    fake = types.ModuleType("litellm")
    fake.completion = completion
    fake._turn_on_debug = lambda: None
    monkeypatch.setitem(sys.modules, "litellm", fake)


def test_reasoning_is_kept_out_of_output(monkeypatch):
    """The provider must not splice reasoning into ``output``.

    Regression test: output used to be rewritten as "reasoning\\n\\n\\nanswer", which
    forced every grader to recover the answer by string search on the first triple
    newline — lossy whenever the reasoning itself contained a blank line.
    """
    _stub_litellm(monkeypatch, _FakeResponse("Paris.", reasoning_content="think\n\nthink more"))
    comp = LiteLLMProvider(ProviderConfig(name="p", model="openai/auto")).complete([])

    assert comp.output == "Paris."
    assert comp.reasoning == "think\n\nthink more"
    assert "think" not in comp.output


def test_output_preserved_verbatim_even_with_blank_lines(monkeypatch):
    """An answer containing blank lines must survive intact."""
    answer = "First para.\n\n\nSecond para."
    _stub_litellm(monkeypatch, _FakeResponse(answer, reasoning_content="hidden"))
    comp = LiteLLMProvider(ProviderConfig(name="p", model="openai/auto")).complete([])

    assert comp.output == answer
    assert comp.reasoning == "hidden"


def test_missing_reasoning_field_yields_none(monkeypatch):
    _stub_litellm(monkeypatch, _FakeResponse("Just an answer."))
    comp = LiteLLMProvider(ProviderConfig(name="p", model="openai/auto")).complete([])

    assert comp.output == "Just an answer."
    assert comp.reasoning is None


# --- timeouts --------------------------------------------------------------


def test_timeout_is_forwarded_to_litellm(monkeypatch):
    seen: dict = {}
    _stub_litellm(monkeypatch, _FakeResponse("ok"), seen)
    LiteLLMProvider(ProviderConfig(name="p", model="openai/auto")).complete([], timeout=45.0)

    assert seen["timeout"] == 45.0


def test_no_timeout_leaves_litellm_to_its_own_default(monkeypatch):
    # The provider must not invent a timeout: an embedder calling complete() directly
    # keeps whatever litellm (or its own config) decided.
    seen: dict = {}
    _stub_litellm(monkeypatch, _FakeResponse("ok"), seen)
    LiteLLMProvider(ProviderConfig(name="p", model="openai/auto")).complete([])

    assert "timeout" not in seen


def test_an_explicit_timeout_overrides_one_in_params(monkeypatch):
    """The per-call timeout wins over ``params.timeout``.

    ``params`` also feeds the cache key, so a timeout parked there is part of the
    identity under test; the runner's operational timeout has to be able to override
    it without anyone editing a config.
    """
    seen: dict = {}
    _stub_litellm(monkeypatch, _FakeResponse("ok"), seen)
    cfg = ProviderConfig(name="p", model="openai/auto", params={"timeout": 9999})
    LiteLLMProvider(cfg).complete([], timeout=30.0)

    assert seen["timeout"] == 30.0


def test_the_sdk_retry_layer_is_disabled_by_default(monkeypatch):
    """A timeout is only real if nothing underneath silently retries it.

    The OpenAI client that litellm wraps retries twice by default, with backoff, so a
    2s ceiling actually took ~7.5s of wall clock (measured) and a 60s ceiling could run
    for minutes. Those extra calls were invisible to the store, too. llmeval owns retry
    policy (``RunPolicy.retries``, one stored row per attempt), so the lower layer must
    not add a second one.
    """
    seen: dict = {}
    _stub_litellm(monkeypatch, _FakeResponse("ok"), seen)
    LiteLLMProvider(ProviderConfig(name="p", model="openai/auto")).complete([], timeout=5.0)

    assert seen["max_retries"] == 0


def test_a_config_can_opt_back_into_sdk_retries(monkeypatch):
    seen: dict = {}
    _stub_litellm(monkeypatch, _FakeResponse("ok"), seen)
    cfg = ProviderConfig(name="p", model="openai/auto", params={"max_retries": 3})
    LiteLLMProvider(cfg).complete([], timeout=5.0)

    assert seen["max_retries"] == 3


def test_echo_provider_accepts_a_timeout(monkeypatch):
    # Every provider is called the same way by the runner, network or not.
    cfg = ProviderConfig(name="e", model="echo", extra={"provider_impl": "echo"})
    out = build_provider(cfg).complete([{"role": "user", "content": "hi"}], timeout=1.0)
    assert out.output == "hi"

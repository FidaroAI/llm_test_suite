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


def _stub_litellm(monkeypatch, response):
    """Install a fake ``litellm`` module so ``complete`` needs no network."""
    fake = types.ModuleType("litellm")
    fake.completion = lambda **kwargs: response
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

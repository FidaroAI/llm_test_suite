from llmeval.models import ProviderConfig
from llmeval.providers import Completion, build_provider, register_provider


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

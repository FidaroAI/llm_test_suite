import pytest

from llmeval.models import AssertionSpec, ProviderConfig, TestCase


def test_user_shorthand_becomes_a_user_message():
    tc = TestCase.from_dict({"id": "t1", "user": "What is the capital of France?"})
    assert [(m.role, m.content) for m in tc.messages] == [
        ("user", "What is the capital of France?")
    ]


def test_explicit_messages_preserved():
    tc = TestCase.from_dict(
        {
            "id": "t2",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    assert tc.messages[0].role == "system"
    assert tc.messages[1].content == "hi"


def test_assertions_parsed_with_defaults():
    tc = TestCase.from_dict(
        {"id": "t3", "user": "q", "assertions": [{"type": "icontains", "value": "Paris"}]}
    )
    a = tc.assertions[0]
    assert a.type == "icontains"
    assert a.value == "Paris"
    assert a.weight == 1.0
    assert a.transform is None  # no transform: providers already return a clean answer
    assert a.params == {}


def test_user_text_is_last_user_message():
    tc = TestCase.from_dict(
        {
            "id": "t4",
            "messages": [
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        }
    )
    assert tc.user_text == "second"


def test_missing_id_is_rejected():
    with pytest.raises(Exception):
        TestCase.from_dict({"user": "q"})


def test_provider_config_cache_key_respects_field_selection():
    cfg = ProviderConfig(
        name="fidaro-dev",
        model="openai/Qwen",
        params={"temperature": 0.7, "max_tokens": 100000},
        extra={"backend_version": "phala-2026-06-01"},
        cache_key_fields=["model", "temperature", "backend_version"],
    )
    key = cfg.cache_key()
    assert key.fields == {
        "model": "openai/Qwen",
        "temperature": 0.7,
        "backend_version": "phala-2026-06-01",
    }
    # changing the ignored max_tokens must not change identity
    cfg2 = cfg.model_copy(update={"params": {"temperature": 0.7, "max_tokens": 1}})
    assert cfg2.cache_key().hash == key.hash


def test_assertion_spec_round_trips():
    spec = AssertionSpec(type="rubric", value="is accurate", weight=2.0, metric="accuracy")
    d = spec.model_dump()
    assert AssertionSpec(**d) == spec

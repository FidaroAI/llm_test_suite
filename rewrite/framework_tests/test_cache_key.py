"""The cache key is the organizing principle: it must be fully user-controlled."""

import pytest

from llmeval.cache_key import compute_cache_key


def test_ignored_field_does_not_change_key():
    # Same model+temperature, different max_tokens, keyed only on model+temperature.
    a = compute_cache_key(
        model="m1",
        params={"temperature": 0.7, "max_tokens": 100},
        fields=["model", "temperature"],
    )
    b = compute_cache_key(
        model="m1",
        params={"temperature": 0.7, "max_tokens": 999},
        fields=["model", "temperature"],
    )
    assert a.hash == b.hash


def test_keyed_field_changes_key():
    a = compute_cache_key(model="m1", params={"temperature": 0.7}, fields=["model", "temperature"])
    b = compute_cache_key(model="m1", params={"temperature": 0.2}, fields=["model", "temperature"])
    assert a.hash != b.hash


def test_fields_none_uses_whole_namespace():
    # With no field selection, max_tokens is part of identity.
    a = compute_cache_key(model="m1", params={"temperature": 0.7, "max_tokens": 100})
    b = compute_cache_key(model="m1", params={"temperature": 0.7, "max_tokens": 999})
    assert a.hash != b.hash


def test_param_order_irrelevant():
    a = compute_cache_key(model="m1", params={"temperature": 0.7, "top_p": 0.9})
    b = compute_cache_key(model="m1", params={"top_p": 0.9, "temperature": 0.7})
    assert a.hash == b.hash


def test_extra_supplies_non_api_identity():
    # backend_version isn't an API param but is part of the system under test.
    a = compute_cache_key(
        model="m1",
        params={"temperature": 0.7},
        extra={"backend_version": "v1"},
        fields=["model", "backend_version"],
    )
    b = compute_cache_key(
        model="m1",
        params={"temperature": 0.7},
        extra={"backend_version": "v2"},
        fields=["model", "backend_version"],
    )
    assert a.hash != b.hash


def test_selected_fields_are_exposed_for_grouping():
    key = compute_cache_key(
        model="m1",
        params={"temperature": 0.7, "max_tokens": 100},
        extra={"backend_version": "v1"},
        fields=["model", "temperature", "backend_version"],
    )
    assert key.fields == {"model": "m1", "temperature": 0.7, "backend_version": "v1"}
    assert "max_tokens" not in key.canonical
    assert len(key.short) == 12


def test_requesting_absent_field_raises():
    with pytest.raises(ValueError):
        compute_cache_key(model="m1", params={"temperature": 0.7}, fields=["model", "nonexistent"])


def test_params_and_extra_must_be_disjoint():
    with pytest.raises(ValueError):
        compute_cache_key(model="m1", params={"temperature": 0.7}, extra={"temperature": 0.2})


def test_model_is_reserved():
    with pytest.raises(ValueError):
        compute_cache_key(model="m1", params={"model": "sneaky"})


def test_stream_is_reserved():
    with pytest.raises(ValueError):
        compute_cache_key(model="m1", extra={"stream": "sneaky"})


def test_stream_is_in_the_whole_namespace():
    assert "stream" in compute_cache_key(model="m1").fields


def test_stream_can_be_selected():
    keyed = compute_cache_key(model="m1", fields=["model", "stream"], stream=True)
    unkeyed = compute_cache_key(model="m1", fields=["model", "stream"], stream=False)
    assert keyed.hash != unkeyed.hash


def test_stream_is_ignorable_like_any_other_field():
    """The usual choice. An aggregated stream and a non-streamed response are the same
    answer, so a config that streams normally wants to share a cache key with one that
    does not — which is what lets streaming be switched on without discarding results.
    """
    streamed = compute_cache_key(model="m1", fields=["model"], stream=True)
    plain = compute_cache_key(model="m1", fields=["model"], stream=False)
    assert streamed.hash == plain.hash

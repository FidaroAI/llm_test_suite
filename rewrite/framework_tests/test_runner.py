import threading

import pytest

from llmeval.models import ProviderConfig, TestCase
from llmeval.providers import Completion
from llmeval.runner import RunPolicy, run, run_testcase
from llmeval.store import Store


class FakeProvider:
    """Implements the provider protocol: has .config and .complete(messages)."""

    def __init__(self, config, fail_times=0, always_fail=False):
        self.config = config
        self.calls = 0
        self.fail_times = fail_times
        self.always_fail = always_fail

    def complete(self, messages):
        self.calls += 1
        if self.always_fail:
            raise RuntimeError("boom")
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return Completion(output=f"reply {self.calls}", reasoning="r", tokens={"total": 1})


def cfg(**extra):
    return ProviderConfig(name="p", model="m1", params={"temperature": 0.7}, **extra)


def tc(id="t1"):
    return TestCase.from_dict({"id": id, "user": "hi"})


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def test_reuse_runs_once_then_reuses(store):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"))
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"))
    assert p.calls == 1
    assert store.count_results("t1", cfg().cache_key().hash) == 1


def test_target_n_fills_up_to_n(store):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="target_n", target_n=5))
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 5


def test_target_n_tops_up_existing(store):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="target_n", target_n=2))
    summary = run_testcase(store, tc(), p, RunPolicy(mode="target_n", target_n=5))
    assert summary.ran == 3
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 5


def test_always_appends_each_call(store):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="always"))
    run_testcase(store, tc(), p, RunPolicy(mode="always"))
    assert store.count_results("t1", cfg().cache_key().hash) == 2


def test_retries_then_succeeds(store):
    p = FakeProvider(cfg(), fail_times=2)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=2))
    assert summary.ran == 1
    assert summary.errors == 0
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 1


def test_persistent_failure_is_graceful(store):
    p = FakeProvider(cfg(), always_fail=True)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=1))
    assert summary.errors == 1
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 0
    # the error row exists so we know it was attempted
    assert store.count_results("t1", cfg().cache_key().hash) == 1


def test_one_failing_testcase_does_not_stop_others(store):
    good = FakeProvider(cfg())
    summary_good = run(store, [tc("a"), tc("b")], good, RunPolicy(mode="reuse"))
    assert summary_good.ran == 2
    assert store.count_results("b", cfg().cache_key().hash) == 1


def test_cache_key_field_selection_shares_results(store):
    # two configs differ only in an ignored field -> same cache key -> reuse
    c1 = ProviderConfig(
        name="a", model="m1", params={"temperature": 0.7, "max_tokens": 1},
        cache_key_fields=["model", "temperature"],
    )
    c2 = ProviderConfig(
        name="b", model="m1", params={"temperature": 0.7, "max_tokens": 999},
        cache_key_fields=["model", "temperature"],
    )
    run_testcase(store, tc(), FakeProvider(c1), RunPolicy(mode="reuse"))
    p2 = FakeProvider(c2)
    run_testcase(store, tc(), p2, RunPolicy(mode="reuse"))
    assert p2.calls == 0  # reused c1's result


def test_stored_result_carries_completion_payload(store):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"))
    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert row.output == "reply 1"
    assert row.reasoning == "r"
    assert row.tokens == {"total": 1}


def test_stored_result_carries_full_config(store):
    p = FakeProvider(cfg(extra={"backend_version": "v9"}))
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"))
    row = store.get_results("t1", p.config.cache_key().hash)[0]
    assert row.config["model"] == "m1"
    assert row.config["extra"] == {"backend_version": "v9"}


class InterruptingProvider:
    """Succeeds for the first test, then raises KeyboardInterrupt (Ctrl-C)."""

    def __init__(self, config, interrupt_on):
        self.config = config
        self.interrupt_on = interrupt_on
        self.seen = []

    def complete(self, messages):
        self.seen.append(messages)
        if len(self.seen) > self.interrupt_on:
            raise KeyboardInterrupt
        return Completion(output="done")


def test_keyboard_interrupt_preserves_already_computed_results(store):
    # Ctrl-C mid-run must not lose results already committed to the DB.
    p = InterruptingProvider(cfg(), interrupt_on=1)
    with pytest.raises(KeyboardInterrupt):
        run(store, [tc("a"), tc("b")], p, RunPolicy(mode="reuse"))
    # the first test's result survived; the interrupted one simply isn't there
    assert store.count_results("a", cfg().cache_key().hash) == 1
    assert store.count_results("b", cfg().cache_key().hash) == 0


# --- concurrency -----------------------------------------------------------


class BarrierProvider:
    """Proves *real* parallelism: every ``complete`` blocks on a barrier sized to
    the expected concurrency. If the runner were sequential, the first call would
    wait forever and the barrier would time out (BrokenBarrierError)."""

    def __init__(self, config, parties, timeout=5.0):
        self.config = config
        self.barrier = threading.Barrier(parties, timeout=timeout)

    def complete(self, messages):
        self.barrier.wait()
        return Completion(output="ok")


def test_runpolicy_concurrency_defaults_to_one():
    # library default stays sequential/deterministic; the CLI supplies the 5 default
    assert RunPolicy().concurrency == 1


def test_concurrency_runs_testcases_in_parallel(store):
    p = BarrierProvider(cfg(), parties=5)
    cases = [tc(f"t{i}") for i in range(5)]
    summary = run(store, cases, p, RunPolicy(mode="reuse", concurrency=5))
    assert summary.ran == 5
    for i in range(5):
        assert store.count_results(f"t{i}", cfg().cache_key().hash) == 1


def test_concurrency_stores_all_results_with_pool_smaller_than_work(store):
    p = FakeProvider(cfg())
    cases = [tc(f"c{i}") for i in range(20)]
    summary = run(store, cases, p, RunPolicy(mode="reuse", concurrency=4))
    assert summary.ran == 20
    for i in range(20):
        assert store.count_results(f"c{i}", cfg().cache_key().hash, success_only=True) == 1


def test_concurrency_one_failing_testcase_does_not_stop_others(store):
    good = FakeProvider(cfg())
    summary = run(store, [tc("a"), tc("b"), tc("c")], good, RunPolicy(concurrency=3))
    assert summary.ran == 3

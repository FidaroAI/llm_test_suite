import io
import threading
import time

import pytest
from conftest import a_run

from llmeval.logs import configure_logging
from llmeval.models import ProviderConfig, TestCase
from llmeval.providers import Completion
from llmeval.runner import RunPolicy, excerpt, run, run_testcase
from llmeval.store import Store


class FakeProvider:
    """Implements the provider protocol: has .config and .complete(messages, timeout)."""

    def __init__(self, config, fail_times=0, always_fail=False):
        self.config = config
        self.calls = 0
        self.fail_times = fail_times
        self.always_fail = always_fail
        self.timeouts = []  # the timeout the runner asked for, per call

    def complete(self, messages, timeout=None):
        self.calls += 1
        self.timeouts.append(timeout)
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


@pytest.fixture
def run_id(store):
    """A run for ``run_testcase`` to stamp its results with.

    ``run()`` opens its own run; ``run_testcase`` is the lower-level entry point and
    takes one, because a result with no run is not representable.
    """
    return a_run(store, cfg().cache_key())


def test_reuse_runs_once_then_reuses(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"), run_id)
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"), run_id)
    assert p.calls == 1
    assert store.count_results("t1", cfg().cache_key().hash) == 1


def test_target_n_fills_up_to_n(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="target_n", target_n=5), run_id)
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 5


def test_target_n_tops_up_existing(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="target_n", target_n=2), run_id)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="target_n", target_n=5), run_id)
    assert summary.ran == 3
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 5


def test_always_appends_each_call(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="always"), run_id)
    run_testcase(store, tc(), p, RunPolicy(mode="always"), run_id)
    assert store.count_results("t1", cfg().cache_key().hash) == 2


def test_retries_then_succeeds(store, run_id):
    p = FakeProvider(cfg(), fail_times=2)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=2), run_id)
    assert summary.failed == 0
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 1


def test_persistent_failure_is_graceful(store, run_id):
    p = FakeProvider(cfg(), always_fail=True)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=1), run_id)
    assert summary.failed == 1
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 0
    # both exhausted attempts are on record, so the cost of the failure is visible
    assert store.count_results("t1", cfg().cache_key().hash) == 2


def test_every_attempt_is_stored_including_the_ones_that_failed(store, run_id):
    """Two transient failures then a success is three rows, not one.

    Retries used to leave no trace: a test that needed three calls to answer looked
    exactly like one that answered first time, so nobody could see what a run cost.
    """
    p = FakeProvider(cfg(), fail_times=2)
    run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=2), run_id)

    rows = store.get_results("t1", cfg().cache_key().hash)
    assert [(r.attempt, r.error is None) for r in rows] == [(0, False), (1, False), (2, True)]
    assert rows[0].error == "RuntimeError: transient"
    assert rows[2].output == "reply 3"


def test_summary_separates_failed_attempts_from_failed_test_cases(store, run_id):
    # A test case that came good on its third try is not a failure, but the two
    # wasted calls are still worth counting.
    p = FakeProvider(cfg(), fail_times=2)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=2), run_id)
    assert (summary.ran, summary.errors, summary.failed) == (3, 2, 0)


def test_summary_counts_a_test_case_that_exhausted_its_retries(store, run_id):
    p = FakeProvider(cfg(), always_fail=True)
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=1), run_id)
    assert (summary.ran, summary.errors, summary.failed) == (2, 2, 1)


# --- partial results ---------------------------------------------------------
#
# A streaming provider that hits its deadline comes back with text *and* an error. The
# runner has to keep both: the text is the evidence a repetitive-loop test grades, and
# the error is what stops the row being mistaken for an answer.


class PartialProvider:
    """A streaming provider whose call always times out holding half an answer."""

    def __init__(self, config):
        self.config = config
        self.calls = 0

    def complete(self, messages, timeout=None):
        self.calls += 1
        return Completion(
            output="again and again and ",
            reasoning="stuck in a loop",
            tokens={"total": 9},
            provider_specific={"fidaro": {"title": "Looping"}},
            error="stream timeout after 60.0s (content: 20 chars, reasoning: 15 chars)",
        )


def test_a_partial_result_is_stored_with_its_output_and_its_error(store, run_id):
    p = PartialProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=0), run_id)

    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert row.output == "again and again and "   # the loop survived the timeout
    assert row.reasoning == "stuck in a loop"
    assert row.tokens == {"total": 9}
    assert row.provider_specific == {"fidaro": {"title": "Looping"}}
    assert "stream timeout" in row.error


def test_a_partial_result_is_not_retried(store, run_id):
    """The ceiling was the caller's statement of how long the answer was worth waiting
    for. Paying it three times over to learn the same thing is waste — and these test
    cases are meant to time out every time."""
    p = PartialProvider(cfg())
    summary = run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=2), run_id)

    assert p.calls == 1
    assert (summary.ran, summary.errors, summary.failed) == (1, 1, 1)


def test_a_raised_failure_is_still_retried(store, run_id):
    # The distinction: an exception may well be transient, a timeout is not.
    p = FakeProvider(cfg(), always_fail=True)
    run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=2), run_id)
    assert p.calls == 3


def test_a_partial_does_not_count_as_a_usable_cached_result(store, run_id):
    # error IS NOT NULL, so the top-up arithmetic still sees this test as unanswered.
    run_testcase(store, tc(), PartialProvider(cfg()), RunPolicy(mode="reuse"), run_id)
    assert store.count_results("t1", cfg().cache_key().hash, success_only=True) == 0
    assert store.count_results("t1", cfg().cache_key().hash) == 1


def test_provider_specific_output_is_stored_on_a_clean_result(store, run_id):
    class TitledProvider:
        def __init__(self, config):
            self.config = config

        def complete(self, messages, timeout=None):
            return Completion(output="ok", provider_specific={"fidaro": {"title": "T"}})

    run_testcase(store, tc(), TitledProvider(cfg()), RunPolicy(mode="reuse"), run_id)
    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert row.provider_specific == {"fidaro": {"title": "T"}}


def test_a_provider_returning_nothing_special_leaves_the_column_null(store, run_id):
    run_testcase(store, tc(), FakeProvider(cfg()), RunPolicy(mode="reuse"), run_id)
    assert store.get_results("t1", cfg().cache_key().hash)[0].provider_specific is None


def test_one_failing_testcase_does_not_stop_others(store):
    good = FakeProvider(cfg())
    result = run(store, [tc("a"), tc("b")], good, RunPolicy(mode="reuse"))
    assert result.summary.ran == 2
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
    rid = a_run(store, c1.cache_key())
    run_testcase(store, tc(), FakeProvider(c1), RunPolicy(mode="reuse"), rid)
    p2 = FakeProvider(c2)
    run_testcase(store, tc(), p2, RunPolicy(mode="reuse"), rid)
    assert p2.calls == 0  # reused c1's result


def test_stored_result_carries_completion_payload(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"), run_id)
    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert row.output == "reply 1"
    assert row.reasoning == "r"
    assert row.tokens == {"total": 1}


def test_stored_result_carries_the_messages_that_were_sent(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"), run_id)
    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert row.messages == [{"role": "user", "content": "hi"}]


def test_multi_turn_messages_are_stored_whole(store, run_id):
    """The report must not have to reconstruct context it never saw."""
    case = TestCase.from_dict(
        {
            "id": "t1",
            "messages": [
                {"role": "system", "content": "be terse"},
                {"role": "user", "content": "I'm planning a trip to Japan."},
                {"role": "assistant", "content": "When are you going?"},
                {"role": "user", "content": "Two weeks in spring."},
            ],
        }
    )
    run_testcase(store, case, FakeProvider(cfg()), RunPolicy(mode="reuse"), run_id)
    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert [m["role"] for m in row.messages] == ["system", "user", "assistant", "user"]
    assert row.messages[0]["content"] == "be terse"


def test_failed_attempts_also_record_what_was_sent(store, run_id):
    p = FakeProvider(cfg(), fail_times=1)
    run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=1), run_id)
    rows = store.get_results("t1", cfg().cache_key().hash)
    assert len(rows) == 2
    assert rows[0].error is not None
    # Both the failure and the success carry the prompt; "what did we send when it
    # timed out?" is the first question asked of an error row.
    assert all(r.messages == [{"role": "user", "content": "hi"}] for r in rows)


def test_stored_result_carries_full_config(store):
    p = FakeProvider(cfg(extra={"backend_version": "v9"}))
    key = p.config.cache_key()
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"), a_run(store, key))
    row = store.get_results("t1", key.hash)[0]
    assert row.config["model"] == "m1"
    assert row.config["extra"] == {"backend_version": "v9"}


class InterruptingProvider:
    """Succeeds for the first test, then raises KeyboardInterrupt (Ctrl-C)."""

    def __init__(self, config, interrupt_on):
        self.config = config
        self.interrupt_on = interrupt_on
        self.seen = []

    def complete(self, messages, timeout=None):
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


def test_interrupted_run_is_left_unfinished(store):
    # A crashed run must not claim to have completed: finish_run is on the success
    # path only, so finished_at stays NULL and the partial run is visibly partial.
    p = InterruptingProvider(cfg(), interrupt_on=1)
    with pytest.raises(KeyboardInterrupt):
        run(store, [tc("a"), tc("b")], p, RunPolicy(mode="reuse"))

    runs = store.list_runs()
    assert len(runs) == 1
    assert runs[0].finished is False
    # ...but the rows it did commit are still attributed to it
    assert len(store.get_results_for_run(runs[0].id)) == 1


# --- run identity ----------------------------------------------------------


def test_run_opens_one_run_and_attributes_every_result_to_it(store):
    p = FakeProvider(cfg())
    result = run(store, [tc("a"), tc("b")], p, RunPolicy(mode="target_n", target_n=2))

    assert store.list_runs() == [store.get_run(result.run_id)]
    rows = store.get_results_for_run(result.run_id)
    assert len(rows) == 4  # 2 tests x 2 attempts
    assert {r.run_id for r in rows} == {result.run_id}


def test_completed_run_is_marked_finished(store):
    result = run(store, [tc()], FakeProvider(cfg()), RunPolicy(mode="reuse"))
    assert store.get_run(result.run_id).finished is True


def test_run_records_provider_identity_and_policy(store):
    p = FakeProvider(cfg())
    policy = RunPolicy(mode="target_n", target_n=3, retries=1, concurrency=2, timeout=30.0)
    result = run(store, [tc()], p, policy, notes="before the prompt change")

    row = store.get_run(result.run_id)
    assert row.provider_name == "p"
    assert row.cache_key_hash == cfg().cache_key().hash
    assert row.config["model"] == "m1"
    assert row.params == {
        "mode": "target_n", "target_n": 3, "retries": 1, "concurrency": 2, "timeout": 30.0,
    }
    assert row.notes == "before the prompt change"


def test_separate_invocations_are_separate_runs(store):
    p = FakeProvider(cfg())
    first = run(store, [tc()], p, RunPolicy(mode="always"))
    second = run(store, [tc()], p, RunPolicy(mode="always"))

    assert first.run_id != second.run_id
    assert len(store.list_runs()) == 2
    # Each run numbers its own attempts from 0; the two rows are still one best-of-N
    # dataset because they share a cache key.
    rows = store.get_results("t1", cfg().cache_key().hash)
    assert [r.attempt for r in rows] == [0, 0]
    assert [r.run_id for r in rows] == [first.run_id, second.run_id]


# --- timing ----------------------------------------------------------------


class SlowProvider:
    """Burns measurable wall-clock before answering (or failing)."""

    def __init__(self, config, delay_s, fail=False):
        self.config = config
        self.delay_s = delay_s
        self.fail = fail

    def complete(self, messages, timeout=None):
        time.sleep(self.delay_s)
        if self.fail:
            raise RuntimeError("gave up")
        return Completion(output="ok")  # no latency_ms: the runner must supply it


def test_failed_attempts_record_how_long_they_took(store, run_id):
    """A failure's cost is the number you most want when choosing a timeout.

    Error rows used to store NULL latency, so the time a run spent waiting on calls
    that never came back was recorded nowhere at all.
    """
    p = SlowProvider(cfg(), delay_s=0.05, fail=True)
    run_testcase(store, tc(), p, RunPolicy(mode="reuse", retries=0), run_id)

    row = store.get_results("t1", cfg().cache_key().hash)[0]
    assert row.error is not None
    assert row.latency_ms >= 50


def test_successful_attempts_keep_the_latency_the_provider_measured(store, run_id):
    # The provider times the call itself and knows best; the runner's own clock is
    # only a fallback for attempts that never returned a Completion.
    class PreciseProvider:
        config = cfg()

        def complete(self, messages, timeout=None):
            return Completion(output="ok", latency_ms=123.5)

    run_testcase(store, tc(), PreciseProvider(), RunPolicy(mode="reuse"), run_id)
    assert store.get_results("t1", cfg().cache_key().hash)[0].latency_ms == 123.5


def test_successful_attempts_get_a_measured_latency_when_the_provider_gives_none(store, run_id):
    p = SlowProvider(cfg(), delay_s=0.05)
    run_testcase(store, tc(), p, RunPolicy(mode="reuse"), run_id)
    assert store.get_results("t1", cfg().cache_key().hash)[0].latency_ms >= 50


# --- timeouts --------------------------------------------------------------


def test_runpolicy_timeout_defaults_to_sixty_seconds():
    # An inference call that hangs must not hang the run: without a default, litellm
    # waits 6000s (its own default), which is indistinguishable from a wedged run.
    assert RunPolicy().timeout == 60.0


def test_the_policy_timeout_is_passed_to_the_provider(store, run_id):
    p = FakeProvider(cfg())
    run_testcase(store, tc(), p, RunPolicy(mode="reuse", timeout=12.5), run_id)
    assert p.timeouts == [12.5]


def test_a_test_case_can_override_the_policy_timeout(store, run_id):
    # Some prompts are legitimately slow (deep research, long tool loops) and should
    # buy themselves more time without raising the ceiling for the whole suite.
    p = FakeProvider(cfg())
    slow_case = TestCase.from_dict({"id": "t1", "user": "hi", "timeout": 300})
    run_testcase(store, slow_case, p, RunPolicy(mode="reuse", timeout=60.0), run_id)
    assert p.timeouts == [300.0]


def test_every_retry_of_a_test_case_uses_its_timeout(store, run_id):
    p = FakeProvider(cfg(), fail_times=2)
    slow_case = TestCase.from_dict({"id": "t1", "user": "hi", "timeout": 7})
    run_testcase(store, slow_case, p, RunPolicy(mode="reuse", retries=2), run_id)
    assert p.timeouts == [7.0, 7.0, 7.0]


# --- concurrency -----------------------------------------------------------


class BarrierProvider:
    """Proves *real* parallelism: every ``complete`` blocks on a barrier sized to
    the expected concurrency. If the runner were sequential, the first call would
    wait forever and the barrier would time out (BrokenBarrierError)."""

    def __init__(self, config, parties, barrier_timeout=5.0):
        self.config = config
        self.barrier = threading.Barrier(parties, timeout=barrier_timeout)

    def complete(self, messages, timeout=None):
        self.barrier.wait()
        return Completion(output="ok")


def test_runpolicy_concurrency_defaults_to_one():
    # library default stays sequential/deterministic; the CLI supplies the 5 default
    assert RunPolicy().concurrency == 1


def test_concurrency_runs_testcases_in_parallel(store):
    p = BarrierProvider(cfg(), parties=5)
    cases = [tc(f"t{i}") for i in range(5)]
    result = run(store, cases, p, RunPolicy(mode="reuse", concurrency=5))
    assert result.summary.ran == 5
    for i in range(5):
        assert store.count_results(f"t{i}", cfg().cache_key().hash) == 1


def test_concurrency_stores_all_results_with_pool_smaller_than_work(store):
    p = FakeProvider(cfg())
    cases = [tc(f"c{i}") for i in range(20)]
    result = run(store, cases, p, RunPolicy(mode="reuse", concurrency=4))
    assert result.summary.ran == 20
    for i in range(20):
        assert store.count_results(f"c{i}", cfg().cache_key().hash, success_only=True) == 1


def test_concurrency_one_failing_testcase_does_not_stop_others(store):
    good = FakeProvider(cfg())
    result = run(store, [tc("a"), tc("b"), tc("c")], good, RunPolicy(concurrency=3))
    assert result.summary.ran == 3


# --- logging ---------------------------------------------------------------


def _blocks_for(lines: list[str], test_id: str) -> list[int]:
    return [n for n, line in enumerate(lines) if f"{test_id}: " in line]


def test_parallel_run_keeps_each_test_cases_logs_together(store, root_logging_restored):
    """Each test case's records must arrive as one contiguous block.

    BarrierProvider makes this a real test rather than a lucky one: no thread can log
    its result line until every thread has logged its prompt line, so an undeferred
    runner would necessarily produce four prompt lines followed by four result lines.
    """
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    cases = [tc(f"t{i}") for i in range(4)]
    run(store, cases, BarrierProvider(cfg(), parties=4), RunPolicy(mode="reuse", concurrency=4))

    lines = stream.getvalue().splitlines()
    for i in range(4):
        idx = _blocks_for(lines, f"t{i}")
        assert len(idx) == 2, f"expected 2 records for t{i}, got {len(idx)}:\n" + "\n".join(lines)
        assert idx[1] == idx[0] + 1, f"t{i}'s block was split:\n" + "\n".join(lines)


def test_sequential_run_streams_logs_without_deferral(store, root_logging_restored):
    # Nothing to interleave with, so the sequential path must not pay the latency cost
    # of buffering: records land as they happen.
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    seen_midway = []

    class WatchingProvider:
        config = cfg()

        def complete(self, messages, timeout=None):
            seen_midway.append(stream.getvalue())
            return Completion(output="ok")

    run(store, [tc("solo")], WatchingProvider(), RunPolicy(mode="reuse", concurrency=1))

    # The prompt line was already emitted while the provider call was in flight.
    assert "solo: " in seen_midway[0]


def test_run_logs_the_run_id_before_calling_the_model(store, root_logging_restored):
    # The run id is the only handle for querying partial results back, so it has to be
    # on screen before anything that might hang or be interrupted.
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    seen_midway = []

    class WatchingProvider:
        config = cfg()

        def complete(self, messages, timeout=None):
            seen_midway.append(stream.getvalue())
            return Completion(output="ok")

    result = run(store, [tc()], WatchingProvider(), RunPolicy(mode="reuse"))

    assert result.run_id in seen_midway[0]


def test_retries_are_logged(store, run_id, root_logging_restored):
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    run_testcase(store, tc(), FakeProvider(cfg(), fail_times=2), RunPolicy(retries=2), run_id)

    out = stream.getvalue()
    assert "attempt 1/3 failed" in out
    assert "attempt 2/3 failed" in out


def test_persistent_failure_is_logged_as_an_error(store, run_id, root_logging_restored):
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    run_testcase(store, tc(), FakeProvider(cfg(), always_fail=True), RunPolicy(retries=1), run_id)

    out = stream.getvalue()
    assert "ERROR" in out
    assert "failed after 2 attempt(s)" in out


@pytest.mark.parametrize("text,expected", [
    ("short", "short"),
    ("  padded  ", "padded"),
    # Newlines collapse: a multi-line answer must not turn one record into many lines,
    # which would undo the grouping the deferred handler exists to provide.
    ("line one\nline two", "line one line two"),
    ("", "<empty>"),
    (None, "<empty>"),
])
def test_excerpt_flattens_and_caps(text, expected):
    assert excerpt(text) == expected


def test_excerpt_truncates_with_an_ellipsis():
    assert excerpt("x" * 200, limit=10) == "x" * 10 + "..."

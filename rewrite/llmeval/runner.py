"""The runner: turn (provider, test cases) into cached results.

Caching is per ``(test_id, cache_key)`` — never per whole run — so a single failing test
can be re-run in isolation and nothing already done is repeated or wasted.

Policies:
* ``reuse``    — if a usable result exists, don't call the model again.
* ``target_n`` — ensure up to N usable results exist (best-of-N statistics).
* ``always``   — append one more result regardless.

**Every attempt is stored**, successful or not, so a test that needed three calls to
answer leaves three rows. A persistent failure is stored as error rows (graceful) rather
than raised, so the run continues and a later invocation can top it up.

Every invocation opens a *run* and stamps its results with the run id. That is pure
provenance — it records which sitting produced a row and never affects the caching
arithmetic above, which stays keyed on ``(test_id, cache_key)`` alone.

Every inference call gets a timeout (``RunPolicy.timeout``, overridable per test case),
because litellm's own default is 6000s: without one, a wedged gateway is indistinguishable
from a slow answer and a run can hang indefinitely.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Iterable

from llmeval.logs import deferred_logs
from llmeval.providers import Completion, Provider
from llmeval.store import Store

logger = logging.getLogger(__name__)

VALID_MODES = ("reuse", "target_n", "always")

# How much of a prompt or answer to show in a log line. Long enough to identify which
# test case a block belongs to, short enough to keep one test case to one line.
LOG_EXCERPT_CHARS = 80


@dataclass
class RunPolicy:
    mode: str = "reuse"
    target_n: int = 1
    retries: int = 2
    # How many test cases to run in parallel. 1 = sequential (deterministic, the
    # library default); the CLI raises this to 5. Each test case is an independent
    # unit of work, so parallelism is across test cases — the per-test attempt
    # numbering never races.
    concurrency: int = 1
    # Seconds allowed per inference call, before retries. A test case may raise or
    # lower it for itself (``TestCase.timeout``). The whole policy is recorded in
    # ``runs.params_json``, so a run always says what ceiling it was given.
    timeout: float = 60.0


@dataclass
class RunSummary:
    """Counts for one run. ``ran`` and ``errors`` count *attempts*; ``failed`` counts
    *test cases*.

    The distinction matters once retries are on record: a test case that failed twice
    and then succeeded contributes ``ran=3, errors=2, failed=0``. Reporting only
    ``errors`` would make a clean run look broken; reporting only ``failed`` would hide
    what the run actually cost.
    """

    ran: int = 0  # provider calls made == rows written
    cached: int = 0  # usable results that already existed, so no call was made
    errors: int = 0  # attempts that raised
    failed: int = 0  # test cases that exhausted their retries

    def __add__(self, other: "RunSummary") -> "RunSummary":
        return RunSummary(
            ran=self.ran + other.ran,
            cached=self.cached + other.cached,
            errors=self.errors + other.errors,
            failed=self.failed + other.failed,
        )


@dataclass
class RunResult:
    """What one ``run()`` produced: the counts, plus the id to query them back by."""

    run_id: str
    summary: RunSummary


def _to_run(mode: str, existing_success: int, target_n: int) -> int:
    if mode == "reuse":
        return 0 if existing_success >= 1 else 1
    if mode == "target_n":
        return max(0, target_n - existing_success)
    if mode == "always":
        return 1
    raise ValueError(f"unknown run mode: {mode!r} (expected one of {VALID_MODES})")


def excerpt(text: str | None, limit: int = LOG_EXCERPT_CHARS) -> str:
    """A one-line, length-capped version of ``text`` for a log message.

    Collapses all whitespace, not just the ends: a prompt or answer containing newlines
    would otherwise turn one log record into several visual lines and undo the very
    grouping the deferred handler exists to provide.
    """
    if not text:
        return "<empty>"
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else f"{flat[:limit]}..."


@dataclass
class Attempt:
    """One provider call: what came back, or why it didn't, and how long it took."""

    completion: Completion | None
    error: str | None
    latency_ms: float

    @property
    def ok(self) -> bool:
        return self.error is None and self.completion is not None

    @property
    def stored_latency_ms(self) -> float:
        """What to record. The provider's own figure when it has one, else ours.

        A provider times the call from inside and knows best; an attempt that raised
        never produced a ``Completion``, so for those the runner's clock is all there
        is — and it is exactly the number you want when choosing a timeout.
        """
        if self.completion is not None and self.completion.latency_ms is not None:
            return self.completion.latency_ms
        return self.latency_ms

    def as_row(self) -> dict[str, Any]:
        """This attempt as ``store.add_result_row`` keyword arguments.

        A failed attempt contributes only its error and its latency: there is no
        ``Completion`` to take an output, reasoning or token count from.
        """
        row: dict[str, Any] = {"latency_ms": self.stored_latency_ms, "error": self.error}
        if self.ok:
            row.update(
                output=self.completion.output,
                raw=self.completion.raw,
                reasoning=self.completion.reasoning,
                tokens=self.completion.tokens,
            )
        return row


def _attempt(provider: Provider, messages, timeout: float | None) -> Attempt:
    """Make one provider call, timing it whether or not it succeeds.

    ``BaseException`` (Ctrl-C) is deliberately not caught: an interrupt must unwind the
    run, not be filed as a failed attempt.
    """
    started = time.monotonic()
    try:
        completion = provider.complete(messages, timeout=timeout)
        return Attempt(completion, None, (time.monotonic() - started) * 1000.0)
    # graceful: any provider failure is recorded as an error result, not raised
    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed_ms = (time.monotonic() - started) * 1000.0
        return Attempt(None, f"{type(exc).__name__}: {exc}", elapsed_ms)


def _fill_one_result(
    store: Store,
    testcase,
    provider: Provider,
    policy: RunPolicy,
    run_id: str,
    config: dict,
) -> RunSummary:
    """Call the provider until it answers or the retries run out, storing every attempt.

    Returns the counts for this one result slot. Every attempt is written before the
    next is made, so an interrupt or crash leaves the attempts already made on record.
    """
    messages = [m.model_dump() for m in testcase.messages]
    timeout = testcase.timeout if testcase.timeout is not None else policy.timeout
    total = max(1, policy.retries + 1)  # always at least one attempt, whatever retries says
    summary = RunSummary()

    for i in range(1, total + 1):
        attempt = _attempt(provider, messages, timeout)
        summary.ran += 1
        store.add_result_row(testcase.id, run_id=run_id, config=config, **attempt.as_row())
        if attempt.ok:
            logger.info(
                "%s: ok in %.0fms -> %s",
                testcase.id, attempt.stored_latency_ms, excerpt(attempt.completion.output),
            )
            return summary

        summary.errors += 1
        # Retries were once silent, which made a run that was quietly retrying
        # indistinguishable from one that was merely slow.
        if i < total:
            logger.warning(
                "%s: attempt %d/%d failed after %.0fms (%s); retrying",
                testcase.id, i, total, attempt.latency_ms, attempt.error,
            )

    logger.error(
        "%s: failed after %d attempt(s), each stored as an error result: %s",
        testcase.id, total, attempt.error,
    )
    summary.failed = 1
    return summary


def run_testcase(
    store: Store,
    testcase,
    provider: Provider,
    policy: RunPolicy,
    run_id: str,
    *,
    defer_logs: bool = False,
) -> RunSummary:
    """Bring one test case up to its target number of stored results.

    ``defer_logs`` buffers this call's log records and emits them as one contiguous
    block on return (see :mod:`llmeval.logs`). ``run`` sets it on the parallel path
    only — a sequential caller has nothing to interleave with, so it streams live.
    """
    with deferred_logs(defer_logs):
        return _run_testcase(store, testcase, provider, policy, run_id)


def _run_testcase(
    store: Store, testcase, provider: Provider, policy: RunPolicy, run_id: str
) -> RunSummary:
    key = provider.config.cache_key()
    config = provider.config.model_dump()  # full config stored alongside every result
    existing = store.count_results(testcase.id, key.hash, success_only=True)
    n = _to_run(policy.mode, existing, policy.target_n)

    # Every record in this block is prefixed with the test id. Redundant while the
    # block is contiguous, but the prefix is what makes the output greppable and keeps
    # it intelligible when deferral is off (LLMEVAL_LOG_DEFER=0).
    logger.info("%s: %s", testcase.id, excerpt(testcase.user_text))
    logger.debug(
        "%s: cache_key=%s cached=%d to_run=%d mode=%s timeout=%ss",
        testcase.id, key.hash, existing, n, policy.mode,
        testcase.timeout if testcase.timeout is not None else policy.timeout,
    )
    if n == 0:
        logger.info("%s: reusing %d cached result(s); no model call", testcase.id, existing)

    total = RunSummary(cached=existing)
    for _ in range(n):
        total += _fill_one_result(store, testcase, provider, policy, run_id, config)
    return total


def run(
    store: Store,
    testcases: Iterable,
    provider: Provider,
    policy: RunPolicy,
    notes: str | None = None,
) -> RunResult:
    """Run a provider over every test case, optionally in parallel.

    Each test case is an independent (test_id, cache_key) unit, so we fan them out
    across a thread pool of size ``policy.concurrency``. The slow part is the model
    call in ``provider.complete``; the shared ``Store`` is thread-safe (see store.py).
    With ``concurrency == 1`` execution is strictly sequential — same as before, which
    keeps ordering deterministic and lets Ctrl-C leave already-committed results intact.

    Opens a run up front and marks it finished only on the success path. An interrupted
    run keeps every result it committed but leaves ``finished_at`` NULL, so a partial
    run is visibly partial rather than silently claiming to have completed. (Deliberately
    *not* a ``finally`` — that would stamp a crashed run as finished.)

    On the parallel path each test case's log records are deferred and flushed as one
    block, so concurrent test cases interleave between blocks rather than line by line.
    See :mod:`llmeval.logs`.
    """
    cases = list(testcases)
    concurrency = max(1, policy.concurrency)

    run_id = store.create_run(
        provider.config.cache_key(),
        provider_name=provider.config.name,
        config=provider.config.model_dump(),
        params=asdict(policy),
        notes=notes,
    )
    # Logged before the first model call, so the run id is on screen even if the run
    # is later interrupted — it is the only handle for querying partial results back.
    logger.info(
        "run %s: %d test case(s), provider=%s, mode=%s, concurrency=%d",
        run_id, len(cases), provider.config.name, policy.mode, concurrency,
    )

    total = RunSummary()
    sequential = concurrency == 1 or len(cases) <= 1
    if sequential:
        for testcase in cases:
            total += run_testcase(store, testcase, provider, policy, run_id)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_testcase, store, tc, provider, policy, run_id, defer_logs=True)
                for tc in cases
            ]
            # as_completed yields on the calling thread, so accumulation stays
            # single-threaded; a worker exception (e.g. KeyboardInterrupt) re-raises
            # here and unwinds the pool.
            for done, fut in enumerate(as_completed(futures), start=1):
                total += fut.result()
                # Emitted from the calling thread and never deferred, so it lands
                # between test-case blocks rather than inside one. Completion order is
                # not submission order, hence a plain counter rather than a test id.
                logger.info("run %s: %d/%d test case(s) complete", run_id, done, len(cases))

    store.finish_run(run_id)
    return RunResult(run_id=run_id, summary=total)

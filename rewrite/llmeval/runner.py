"""The runner: turn (provider, test cases) into cached results.

Caching is per ``(test_id, cache_key)`` — never per whole run — so a single failing test
can be re-run in isolation and nothing already done is repeated or wasted.

Policies:
* ``reuse``    — if a usable result exists, don't call the model again.
* ``target_n`` — ensure up to N usable results exist (best-of-N statistics).
* ``always``   — append one more result regardless.

Each result is attempted with retries; a persistent failure is stored as an *error row*
(graceful) so the run continues and a later invocation can top it up.

Every invocation opens a *run* and stamps its results with the run id. That is pure
provenance — it records which sitting produced a row and never affects the caching
arithmetic above, which stays keyed on ``(test_id, cache_key)`` alone.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Iterable

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


@dataclass
class RunSummary:
    ran: int = 0
    cached: int = 0
    errors: int = 0

    def __add__(self, other: "RunSummary") -> "RunSummary":
        return RunSummary(
            ran=self.ran + other.ran,
            cached=self.cached + other.cached,
            errors=self.errors + other.errors,
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


def _attempt(
    provider: Provider, messages, retries: int, test_id: str = "?"
) -> tuple[Completion | None, str | None]:
    last_err: str | None = None
    total = retries + 1
    for i in range(1, total + 1):
        try:
            return provider.complete(messages), None
        # graceful: any provider failure is recorded as an error result, not raised
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_err = f"{type(exc).__name__}: {exc}"
            # Retries were previously silent, which made a run that was quietly
            # retrying indistinguishable from one that was merely slow.
            if i < total:
                logger.warning(
                    "%s: attempt %d/%d failed (%s); retrying", test_id, i, total, last_err
                )
    return None, last_err


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
    messages = [m.model_dump() for m in testcase.messages]

    # Every record in this block is prefixed with the test id. Redundant while the
    # block is contiguous, but the prefix is what makes the output greppable and keeps
    # it intelligible when deferral is off (LLMEVAL_LOG_DEFER=0).
    logger.info("%s: %s", testcase.id, excerpt(testcase.user_text))
    logger.debug(
        "%s: cache_key=%s cached=%d to_run=%d mode=%s",
        testcase.id, key.hash, existing, n, policy.mode,
    )
    if n == 0:
        logger.info("%s: reusing %d cached result(s); no model call", testcase.id, existing)

    ran = errors = 0
    for _ in range(n):
        completion, err = _attempt(provider, messages, policy.retries, testcase.id)
        if err is None and completion is not None:
            logger.info(
                "%s: ok%s -> %s",
                testcase.id,
                f" in {completion.latency_ms:.0f}ms" if completion.latency_ms else "",
                excerpt(completion.output),
            )
            store.add_result_row(
                testcase.id,
                key,
                run_id=run_id,
                output=completion.output,
                raw=completion.raw,
                reasoning=completion.reasoning,
                tokens=completion.tokens,
                latency_ms=completion.latency_ms,
                config=config,
            )
        else:
            logger.error(
                "%s: failed after %d attempt(s), stored as an error result: %s",
                testcase.id, policy.retries + 1, err,
            )
            store.add_result_row(testcase.id, key, run_id=run_id, error=err, config=config)
            errors += 1
        ran += 1
    return RunSummary(ran=ran, cached=existing, errors=errors)


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

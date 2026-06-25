"""The runner: turn (provider, test cases) into cached results.

Caching is per ``(test_id, cache_key)`` — never per whole run — so a single failing test
can be re-run in isolation and nothing already done is repeated or wasted.

Policies:
* ``reuse``    — if a usable result exists, don't call the model again.
* ``target_n`` — ensure up to N usable results exist (best-of-N statistics).
* ``always``   — append one more result regardless.

Each result is attempted with retries; a persistent failure is stored as an *error row*
(graceful) so the run continues and a later invocation can top it up.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

from llmeval.providers import Completion, Provider
from llmeval.store import Store

VALID_MODES = ("reuse", "target_n", "always")


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


def _to_run(mode: str, existing_success: int, target_n: int) -> int:
    if mode == "reuse":
        return 0 if existing_success >= 1 else 1
    if mode == "target_n":
        return max(0, target_n - existing_success)
    if mode == "always":
        return 1
    raise ValueError(f"unknown run mode: {mode!r} (expected one of {VALID_MODES})")


def _attempt(provider: Provider, messages, retries: int) -> tuple[Completion | None, str | None]:
    last_err: str | None = None
    for _ in range(retries + 1):
        try:
            return provider.complete(messages), None
        # graceful: any provider failure is recorded as an error result, not raised
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_err = f"{type(exc).__name__}: {exc}"
    return None, last_err


def run_testcase(store: Store, testcase, provider: Provider, policy: RunPolicy) -> RunSummary:
    key = provider.config.cache_key()
    config = provider.config.model_dump()  # full config stored alongside every result
    existing = store.count_results(testcase.id, key.hash, success_only=True)
    n = _to_run(policy.mode, existing, policy.target_n)
    messages = [m.model_dump() for m in testcase.messages]

    ran = errors = 0
    for _ in range(n):
        comp, err = _attempt(provider, messages, policy.retries)
        if err is None and comp is not None:
            store.add_result_row(
                testcase.id,
                key,
                output=comp.output,
                raw=comp.raw,
                reasoning=comp.reasoning,
                tokens=comp.tokens,
                latency_ms=comp.latency_ms,
                config=config,
            )
        else:
            store.add_result_row(testcase.id, key, error=err, config=config)
            errors += 1
        ran += 1
    return RunSummary(ran=ran, cached=existing, errors=errors)


def run(store: Store, testcases: Iterable, provider: Provider, policy: RunPolicy) -> RunSummary:
    """Run a provider over every test case, optionally in parallel.

    Each test case is an independent (test_id, cache_key) unit, so we fan them out
    across a thread pool of size ``policy.concurrency``. The slow part is the model
    call in ``provider.complete``; the shared ``Store`` is thread-safe (see store.py).
    With ``concurrency == 1`` execution is strictly sequential — same as before, which
    keeps ordering deterministic and lets Ctrl-C leave already-committed results intact.
    """
    cases = list(testcases)
    concurrency = max(1, policy.concurrency)

    if concurrency == 1 or len(cases) <= 1:
        total = RunSummary()
        for testcase in cases:
            total += run_testcase(store, testcase, provider, policy)
        return total

    total = RunSummary()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(run_testcase, store, tc, provider, policy) for tc in cases]
        # as_completed yields on the calling thread, so accumulation stays single-threaded;
        # a worker exception (e.g. KeyboardInterrupt) re-raises here and unwinds the pool.
        for fut in as_completed(futures):
            total += fut.result()
    return total

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

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
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


def _attempt(provider: Provider, messages, retries: int) -> tuple[Completion | None, str | None]:
    last_err: str | None = None
    for _ in range(retries + 1):
        try:
            return provider.complete(messages), None
        # graceful: any provider failure is recorded as an error result, not raised
        except Exception as exc:  # pylint: disable=broad-exception-caught
            last_err = f"{type(exc).__name__}: {exc}"
    return None, last_err

def short_string(s):
    return f"{s[:50]}..." if len(s) > 50 else s


def run_testcase(
    store: Store, testcase, provider: Provider, policy: RunPolicy, run_id: str
) -> RunSummary:
    key = provider.config.cache_key()
    config = provider.config.model_dump()  # full config stored alongside every result
    existing = store.count_results(testcase.id, key.hash, success_only=True)
    n = _to_run(policy.mode, existing, policy.target_n)
    messages = [m.model_dump() for m in testcase.messages]

    print("=" * 80)
    print(f"Running test case: {short_string(testcase.messages[0].content)}")
    ran = errors = 0
    for _ in range(n):
        completion, err = _attempt(provider, messages, policy.retries)
        if err is None and completion is not None:
            print(f"Success: {short_string(completion.output)}")
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
            print(f"Error: {err}")
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

    total = RunSummary()
    if concurrency == 1 or len(cases) <= 1:
        for testcase in cases:
            total += run_testcase(store, testcase, provider, policy, run_id)
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [
                pool.submit(run_testcase, store, tc, provider, policy, run_id) for tc in cases
            ]
            # as_completed yields on the calling thread, so accumulation stays
            # single-threaded; a worker exception (e.g. KeyboardInterrupt) re-raises
            # here and unwinds the pool.
            for fut in as_completed(futures):
                total += fut.result()

    store.finish_run(run_id)
    return RunResult(run_id=run_id, summary=total)

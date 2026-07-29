"""Logging setup, and deferred emission so parallel runs stay readable.

Two separate concerns live here.

**Configuration.** :func:`configure_logging` installs one root handler and quiets the
third-party loggers litellm drags in. It is called from the CLI, never at import time:
importing ``llmeval`` must not touch the root logger, because a library that configures
logging on behalf of its caller is a library that fights its caller.

**Ordering.** :func:`run` fans test cases out across a thread pool, so without help the
records from concurrent test cases reach the handler interleaved — two lines of test A,
one of test B, the rest of A. Nothing is lost, but it is unreadable, and the reader
cannot tell which "failed after 3 attempts" belongs to which prompt.

The fix is to defer. Inside :func:`deferred_logs` a thread's records are buffered rather
than emitted, then replayed as one contiguous block when the block exits. Interleaving
then happens *between* whole test cases instead of *within* them, which is the
granularity a human actually wants.

There are two costs, neither of them correctness.

*Latency.* A deferred block appears only once its test case finishes. That is why the
runner defers only when it genuinely runs in parallel — a sequential run has nothing to
interleave with, so it streams live. Set ``LLMEVAL_LOG_DEFER=0`` to force live streaming
everywhere, which is what you want when watching a hung provider call in real time.

*Non-monotonic timestamps.* A ``LogRecord`` timestamps itself when it is created, not
when it is emitted, so within a deferred run the printed times run backwards between
blocks: a test case that started early but finished late prints its early timestamp after
a later block. This is deliberate — the timestamp stays a true event time you can measure
latency from — but it does mean you cannot read the log top-to-bottom as a clock.

Environment variables: ``LLMEVAL_LOG_LEVEL`` (default ``INFO``), ``LLMEVAL_LOG_DEFER``.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Iterator

LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
DEFAULT_LEVEL = "INFO"
LEVEL_ENV = "LLMEVAL_LOG_LEVEL"
DEFER_ENV = "LLMEVAL_LOG_DEFER"

# Third-party loggers that are chatty at INFO. litellm is the offender that matters:
# it logs a banner plus per-call diagnostics on every completion, which would bury the
# per-test output the runner emits. Raise them back with LLMEVAL_LOG_LEVEL=DEBUG plus a
# manual setLevel if you are debugging the provider layer itself.
NOISY_LOGGERS = (
    "litellm",
    "LiteLLM",
    "httpx",
    "httpcore",
    "openai",
    "botocore",
    "boto3",
    "urllib3",
)

_FALSEY = frozenset({"0", "false", "no", "off", ""})


class DeferringHandler(logging.Handler):
    """Wraps a target handler; per thread, either emits straight through or buffers.

    A thread with no active buffer behaves exactly like the target handler. A thread
    inside :meth:`deferring` appends to its buffer instead, and the whole buffer is
    replayed through the target under a lock on exit — so one thread's block can never
    be split by another's.

    Wrapping a handler rather than filtering a logger is deliberate: a
    :class:`logging.Filter` attached to a *logger* is not consulted for records that
    propagate up from child loggers, so logger-level filtering would miss most of what
    we want to defer (every litellm record, for one). Handler-level interception sees
    everything that reaches the handler.

    The buffer is a *stack*, so nested :meth:`deferring` blocks compose: an inner
    block's records are handed to the enclosing buffer rather than emitted in the middle
    of it, keeping the outer block contiguous too.
    """

    def __init__(self, target: logging.Handler):
        super().__init__()
        self.target = target
        self._local = threading.local()
        # Serialises replay against other threads' replays and against direct emits.
        # Not the same as Handler.lock, which only guards a single emit call.
        self._replay_lock = threading.Lock()

    @property
    def _stack(self) -> list[list[logging.LogRecord]]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def emit(self, record: logging.LogRecord) -> None:
        stack = self._stack
        if stack:
            stack[-1].append(record)
        else:
            with self._replay_lock:
                self.target.handle(record)

    @contextmanager
    def deferring(self) -> Iterator[list[logging.LogRecord]]:
        """Buffer this thread's records for the duration of the block."""
        stack = self._stack
        buffer: list[logging.LogRecord] = []
        stack.append(buffer)
        try:
            yield buffer
        finally:
            # finally, not else: a worker that raises (or catches Ctrl-C) must still
            # get its records out — losing the log of a failure is the worst outcome.
            stack.pop()
            if stack:
                stack[-1].extend(buffer)
            else:
                self.replay(buffer)

    def replay(self, records: list[logging.LogRecord]) -> None:
        """Emit ``records`` through the target as one uninterruptible block."""
        if not records:
            return
        with self._replay_lock:
            for record in records:
                self.target.handle(record)

    def setFormatter(self, fmt: logging.Formatter | None) -> None:  # noqa: N802
        # Records are always emitted by the target, so the target owns the formatter.
        self.target.setFormatter(fmt)

    def flush(self) -> None:
        self.target.flush()

    def close(self) -> None:
        try:
            self.target.close()
        finally:
            super().close()


# The handler this process installed, if any. ``deferred_logs`` is a no-op without it:
# when a library caller owns the root handlers we must not silently swallow their
# records, and pytest's caplog is exactly such a caller. Mutable module state, not a
# constant — hence the lower-case name.
_handler: DeferringHandler | None = None  # pylint: disable=invalid-name


def configure_logging(level: str | None = None, *, stream=None) -> DeferringHandler:
    """Install the deferring root handler and quiet noisy libraries. Idempotent.

    Call once, from an entry point (the CLI does this in ``main``). Explicit ``level``
    beats ``LLMEVAL_LOG_LEVEL`` beats ``INFO``.

    Unlike the usual ``basicConfig(force=True)`` recipe, this removes only handlers it
    installed itself. ``force=True`` closes *every* existing root handler, which would
    tear out handlers belonging to whoever embedded us — pytest's caplog being the case
    we actually hit. Being idempotent for ourselves without being destructive to others
    is worth the extra four lines.
    """
    global _handler  # pylint: disable=global-statement

    resolved = (level or os.environ.get(LEVEL_ENV) or DEFAULT_LEVEL).upper()
    target = logging.StreamHandler(stream)
    target.setFormatter(logging.Formatter(LOG_FORMAT))
    handler = DeferringHandler(target)

    root = logging.getLogger()
    for existing in [h for h in root.handlers if isinstance(h, DeferringHandler)]:
        root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(resolved)

    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    _handler = handler
    return handler


def reset_logging() -> None:
    """Remove our root handler and forget it. For tests and embedders."""
    global _handler  # pylint: disable=global-statement

    root = logging.getLogger()
    for existing in [h for h in root.handlers if isinstance(h, DeferringHandler)]:
        root.removeHandler(existing)
    _handler = None


def defer_enabled() -> bool:
    """False when ``LLMEVAL_LOG_DEFER`` asks for live streaming."""
    return os.environ.get(DEFER_ENV, "1").strip().lower() not in _FALSEY


@contextmanager
def deferred_logs(active: bool = True) -> Iterator[None]:
    """Buffer this thread's log records, flushing them as one block on exit.

    A no-op when ``active`` is false, when ``LLMEVAL_LOG_DEFER`` disables deferral, or
    when :func:`configure_logging` was never called — in that last case the root
    handlers belong to somebody else and buffering them would be a surprise.
    """
    if not active or _handler is None or not defer_enabled():
        yield
        return
    with _handler.deferring():
        yield

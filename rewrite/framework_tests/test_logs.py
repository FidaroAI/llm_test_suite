"""Tests for logging configuration and deferred emission.

The property that matters is ordering under concurrency, so the interleaving tests use a
``threading.Barrier`` to *guarantee* that every thread is part-way through its own block
at the same moment. Without that, a passing test would prove nothing — the threads might
simply have run one after another.
"""

from __future__ import annotations

import io
import logging
import threading

import pytest

from llmeval.logs import (
    DeferringHandler,
    configure_logging,
    defer_enabled,
    deferred_logs,
    reset_logging,
)


def _target(stream: io.StringIO) -> logging.Handler:
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    return handler


def _isolated_logger(name: str, handler: logging.Handler) -> logging.Logger:
    """A logger wired only to ``handler`` — no propagation to root (or to caplog)."""
    log = logging.getLogger(name)
    log.handlers = [handler]
    log.setLevel(logging.DEBUG)
    log.propagate = False
    return log


# ``root_logging_restored`` lives in conftest.py — the runner tests need it too.

# --- DeferringHandler ------------------------------------------------------


def test_emits_straight_through_with_no_active_buffer():
    stream = io.StringIO()
    log = _isolated_logger("t.passthrough", DeferringHandler(_target(stream)))

    log.info("one")
    log.info("two")

    assert stream.getvalue().splitlines() == ["one", "two"]


def test_records_are_withheld_until_the_block_exits():
    stream = io.StringIO()
    handler = DeferringHandler(_target(stream))
    log = _isolated_logger("t.withheld", handler)

    with handler.deferring():
        log.info("inside")
        assert stream.getvalue() == ""  # buffered, not yet emitted

    assert stream.getvalue().splitlines() == ["inside"]


def test_buffer_is_flushed_even_when_the_block_raises():
    # Losing the log of a failure is the worst possible outcome, so the flush is in a
    # finally: whatever the worker did, its records get out.
    stream = io.StringIO()
    handler = DeferringHandler(_target(stream))
    log = _isolated_logger("t.raises", handler)

    with pytest.raises(RuntimeError):
        with handler.deferring():
            log.info("before the boom")
            raise RuntimeError("boom")

    assert stream.getvalue().splitlines() == ["before the boom"]


def test_nested_blocks_keep_the_outer_block_contiguous():
    stream = io.StringIO()
    handler = DeferringHandler(_target(stream))
    log = _isolated_logger("t.nested", handler)

    with handler.deferring():
        log.info("outer-1")
        with handler.deferring():
            log.info("inner")
        assert stream.getvalue() == ""  # the inner exit must not flush the outer block
        log.info("outer-2")

    assert stream.getvalue().splitlines() == ["outer-1", "inner", "outer-2"]


def test_buffering_is_per_thread():
    # One thread deferring must not swallow another thread's live records.
    stream = io.StringIO()
    handler = DeferringHandler(_target(stream))
    log = _isolated_logger("t.perthread", handler)
    deferring_started = threading.Event()
    other_logged = threading.Event()

    def deferred_worker():
        with handler.deferring():
            log.info("deferred")
            deferring_started.set()
            assert other_logged.wait(timeout=5)

    thread = threading.Thread(target=deferred_worker)
    thread.start()
    assert deferring_started.wait(timeout=5)
    log.info("live")  # this thread has no buffer, so it goes straight out
    other_logged.set()
    thread.join(timeout=5)

    assert stream.getvalue().splitlines() == ["live", "deferred"]


def test_concurrent_blocks_do_not_interleave():
    stream = io.StringIO()
    handler = DeferringHandler(_target(stream))
    log = _isolated_logger("t.interleave", handler)
    # Sized to both threads: neither can reach its second record until both have
    # written their first, so an undeferred handler would necessarily interleave.
    midpoint = threading.Barrier(2, timeout=5)

    def work(tag: str) -> None:
        with handler.deferring():
            log.info("%s-start", tag)
            midpoint.wait()
            log.info("%s-end", tag)

    threads = [threading.Thread(target=work, args=(tag,)) for tag in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    lines = stream.getvalue().splitlines()
    assert sorted(lines) == ["A-end", "A-start", "B-end", "B-start"]
    for tag in ("A", "B"):
        assert lines.index(f"{tag}-end") == lines.index(f"{tag}-start") + 1


def test_undeferred_concurrent_logging_does_interleave():
    # The negative control for the test above: with no buffering the barrier forces
    # both start lines out before either end line, so the blocks are torn apart. If
    # this ever fails, the test above has stopped proving anything.
    stream = io.StringIO()
    handler = DeferringHandler(_target(stream))
    log = _isolated_logger("t.nointerleave", handler)
    midpoint = threading.Barrier(2, timeout=5)

    def work(tag: str) -> None:
        log.info("%s-start", tag)
        midpoint.wait()
        log.info("%s-end", tag)

    threads = [threading.Thread(target=work, args=(tag,)) for tag in ("A", "B")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)

    lines = stream.getvalue().splitlines()
    assert sorted(lines[:2]) == ["A-start", "B-start"]
    assert sorted(lines[2:]) == ["A-end", "B-end"]


# --- configure_logging -----------------------------------------------------


def test_configure_logging_installs_handler_and_level(root_logging_restored):
    stream = io.StringIO()
    configure_logging("warning", stream=stream)
    root = logging.getLogger()

    assert root.level == logging.WARNING
    assert sum(isinstance(h, DeferringHandler) for h in root.handlers) == 1


def test_configure_logging_reads_the_env_var(monkeypatch, root_logging_restored):
    monkeypatch.setenv("LLMEVAL_LOG_LEVEL", "debug")
    configure_logging(stream=io.StringIO())
    assert logging.getLogger().level == logging.DEBUG


def test_explicit_level_beats_the_env_var(monkeypatch, root_logging_restored):
    monkeypatch.setenv("LLMEVAL_LOG_LEVEL", "debug")
    configure_logging("error", stream=io.StringIO())
    assert logging.getLogger().level == logging.ERROR


def test_configure_logging_quiets_noisy_libraries(root_logging_restored):
    configure_logging("debug", stream=io.StringIO())
    # litellm is the one that matters: it logs per-call diagnostics that would bury
    # the runner's per-test output.
    assert logging.getLogger("litellm").level == logging.WARNING
    assert logging.getLogger("httpx").level == logging.WARNING


def test_configure_logging_is_idempotent(root_logging_restored):
    configure_logging("info", stream=io.StringIO())
    configure_logging("info", stream=io.StringIO())
    root = logging.getLogger()
    assert sum(isinstance(h, DeferringHandler) for h in root.handlers) == 1


def test_configure_logging_leaves_foreign_handlers_alone(root_logging_restored):
    # basicConfig(force=True) would close and drop this handler. pytest's caplog is
    # exactly such a handler, so we must only ever remove our own.
    root = logging.getLogger()
    foreign = logging.StreamHandler(io.StringIO())
    root.addHandler(foreign)

    configure_logging("info", stream=io.StringIO())

    assert foreign in root.handlers


# --- deferred_logs ---------------------------------------------------------


def test_deferred_logs_is_a_noop_when_not_configured():
    # No configure_logging call, so the root handlers belong to somebody else (here,
    # pytest). Buffering them would be a nasty surprise, so we must not.
    reset_logging()
    log = logging.getLogger("t.unconfigured")
    with deferred_logs(True):
        log.warning("straight through")  # would raise/vanish if we buffered blindly


def test_deferred_logs_respects_inactive_flag(root_logging_restored):
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    with deferred_logs(False):
        logging.getLogger("t.inactive").info("live")
        assert "live" in stream.getvalue()  # emitted immediately, not buffered


def test_deferred_logs_buffers_when_active(root_logging_restored):
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    with deferred_logs(True):
        logging.getLogger("t.active").info("held")
        assert stream.getvalue() == ""
    assert "held" in stream.getvalue()


@pytest.mark.parametrize("value,expected", [
    ("0", False), ("false", False), ("no", False), ("off", False), ("", False),
    ("1", True), ("true", True), ("anything", True),
])
def test_defer_enabled_env_parsing(monkeypatch, value, expected):
    monkeypatch.setenv("LLMEVAL_LOG_DEFER", value)
    assert defer_enabled() is expected


def test_defer_enabled_defaults_to_true(monkeypatch):
    monkeypatch.delenv("LLMEVAL_LOG_DEFER", raising=False)
    assert defer_enabled() is True


def test_env_var_disables_deferral(monkeypatch, root_logging_restored):
    monkeypatch.setenv("LLMEVAL_LOG_DEFER", "0")
    stream = io.StringIO()
    configure_logging("info", stream=stream)
    with deferred_logs(True):
        logging.getLogger("t.envoff").info("live anyway")
        assert "live anyway" in stream.getvalue()

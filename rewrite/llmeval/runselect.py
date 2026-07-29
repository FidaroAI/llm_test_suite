"""Run selection — which runs a result-reading stage looks at.

``grade`` and ``report`` both read stored results, and both need the same answer to "which
runs?". This module owns the *meaning* of the four flags that answer it; the SQL lives in
:meth:`llmeval.store.Store.select_runs`.

Four flags in three mutually exclusive groups:

* ``--run-id a,b,c`` — exactly these runs (full ids or unambiguous prefixes)
* ``--run-after`` / ``--run-before`` — a window; either end may be a timestamp or a run
* ``--run-last-n N`` — the N most recent runs

They are exclusive because combining them has no single obvious reading: is
``--run-last-n 3 --run-after 2026-01-01`` "the last 3 runs, then drop the old ones" or "the
3 most recent of those after the date"? Rather than pick one and surprise half the users,
we refuse.

Timestamps are **UTC when no offset is given** — that is what ``runs.started_at`` holds and
what the run id embeds, so a bare date lines up with the ids you see in the log. An explicit
offset (``+08:00``, a trailing ``Z``) is honoured and converted. Both bounds are
**inclusive**, to the whole second.

A boundary is tried as a timestamp *first* and only then as a run id, so a run prefix that
also parsed as a date would be read as the date. Run ids start with ``run_``, which no date
format accepts, so this is not reachable in practice.

The error split matters: a selector that *cannot be satisfied* — an id matching no run, a
boundary that is neither a timestamp nor a run — raises :class:`RunSelectionError`, which
the CLI turns into a message and exit 2. A selector that is well formed but simply matches
nothing returns an empty list, because "the last 3 runs of a provider that has had none" is
a legitimate answer rather than user error.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from llmeval.store import RunRow, Store

logger = logging.getLogger(__name__)

# What the store's bounds compare against: whole seconds, UTC, no offset suffix.
_BOUNDARY_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Tried in order, before ``fromisoformat``. These are the naive forms, and matching them
# here is what makes "no offset means UTC" true rather than platform-dependent.
_DATETIME_FORMS = ("%Y-%m-%d", "%Y-%m-%dT%H:%M", "%Y-%m-%dT%H:%M:%S")


class RunSelectionError(ValueError):
    """A run selection that cannot be satisfied.

    Typed so the CLI reports it as a plain message instead of a traceback — like
    :class:`llmeval.store.IncompatibleSchema`, it is expected user error, not a bug.
    """


@dataclass(frozen=True)
class RunSelection:
    """A parsed, not-yet-resolved run selection. Resolve it with :func:`resolve_runs`."""

    ids: tuple[str, ...] = ()
    after: str | None = None
    before: str | None = None
    last_n: int | None = None


def _split_ids(run_id: str | Sequence[str] | None) -> tuple[str, ...]:
    """Flatten ``--run-id`` into ids: comma-separated, and repeatable.

    Both forms exist because both are natural — ``--run-id a,b`` when pasting from a
    script, ``--run-id a --run-id b`` when building a command up by hand.
    """
    if run_id is None:
        return ()
    items = [run_id] if isinstance(run_id, str) else list(run_id)
    out: list[str] = []
    for item in items:
        out.extend(part.strip() for part in item.split(",") if part.strip())
    return tuple(out)


def parse_run_selection(
    run_id: str | Sequence[str] | None = None,
    run_after: str | None = None,
    run_before: str | None = None,
    run_last_n: int | None = None,
) -> RunSelection:
    """Validate the four flags and package them. No store access, no I/O."""
    ids = _split_ids(run_id)
    groups = []
    if ids:
        groups.append("--run-id")
    if run_after is not None or run_before is not None:
        groups.append("--run-after/--run-before")
    if run_last_n is not None:
        groups.append("--run-last-n")
    if len(groups) > 1:
        raise RunSelectionError(
            f"{' and '.join(groups)} cannot be combined; pick one way to select runs"
        )
    if run_last_n is not None and run_last_n < 1:
        raise RunSelectionError(f"--run-last-n must be at least 1, got {run_last_n}")
    return RunSelection(ids=ids, after=run_after, before=run_before, last_n=run_last_n)


def _as_utc_datetime(value: str) -> datetime | None:
    """Parse a boundary as a timestamp, or return ``None`` if it isn't one.

    The explicit ``strptime`` passes come first so a bare date or minute-precision value is
    read *as UTC*; ``fromisoformat`` then handles the offset forms.
    """
    text = value.strip()
    for fmt in _DATETIME_FORMS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _resolve_one(store: Store, value: str) -> str:
    """Expand one id/prefix, converting the store's ``KeyError`` into our error type."""
    try:
        return store.resolve_run(value)
    except KeyError as exc:
        raise RunSelectionError(exc.args[0] if exc.args else str(exc)) from exc


def _boundary(store: Store, value: str | None) -> str | None:
    """One end of the window, as a string the store can compare.

    Truncated to whole seconds: the stored ``started_at`` carries microseconds, so a ``<=``
    against a second-precision bound derived from a run would otherwise exclude the very
    run the user named as the end cap.
    """
    if value is None:
        return None
    when = _as_utc_datetime(value)
    if when is not None:
        return when.strftime(_BOUNDARY_FORMAT)
    run = store.get_run(_resolve_one(store, value))
    if run is None:  # pragma: no cover - resolve_run already guarantees existence
        raise RunSelectionError(f"run {value!r} vanished between resolve and read")
    return run.started_at[:19]


def resolve_runs(
    store: Store,
    selection: RunSelection,
    cache_key_hashes: Sequence[str] | None = None,
) -> list[RunRow]:
    """Expand a selection into concrete runs, **oldest first**.

    ``cache_key_hashes`` narrows to particular provider identities *before* the selection
    is applied, so ``--provider X --run-last-n 3`` is the last three runs of X rather than
    the last three runs overall intersected with X.

    Naming a run that exists but belongs to a different provider is a warning, not an
    error: the id was valid, it just isn't in the identity you asked about, and saying so
    is more useful than either failing or silently returning nothing.
    """
    ids = [_resolve_one(store, i) for i in selection.ids] if selection.ids else None
    runs = store.select_runs(
        ids=ids,
        after=_boundary(store, selection.after),
        before=_boundary(store, selection.before),
        last_n=selection.last_n,
        cache_key_hashes=None if cache_key_hashes is None else list(cache_key_hashes),
    )
    if ids:
        found = {r.id for r in runs}
        dropped = [i for i in ids if i not in found]
        if dropped:
            logger.warning(
                "run(s) %s do not belong to the selected provider(s); ignoring",
                ", ".join(dropped),
            )
    return runs

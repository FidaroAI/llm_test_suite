import pytest
from conftest import a_run, backdate_run

from llmeval.cache_key import compute_cache_key
from llmeval.runselect import (
    RunSelection,
    RunSelectionError,
    parse_run_selection,
    resolve_runs,
)
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


KEY = compute_cache_key(model="m1")
OTHER = compute_cache_key(model="m2")


# --- parsing -------------------------------------------------------------


def test_no_flags_is_an_empty_selection():
    assert parse_run_selection() == RunSelection()


def test_run_id_splits_on_commas():
    assert parse_run_selection(run_id="run1,run2,run3").ids == ("run1", "run2", "run3")


def test_run_id_accumulates_across_repeats_and_trims():
    assert parse_run_selection(run_id=["run1, run2", "run3"]).ids == ("run1", "run2", "run3")


def test_run_id_ignores_empty_elements():
    assert parse_run_selection(run_id="run1,,run2,").ids == ("run1", "run2")


def test_last_n_is_carried_through():
    assert parse_run_selection(run_last_n=3).last_n == 3


def test_run_id_and_window_conflict():
    with pytest.raises(RunSelectionError, match="cannot be combined"):
        parse_run_selection(run_id="run1", run_after="2026-07-01")


def test_run_id_and_last_n_conflict():
    with pytest.raises(RunSelectionError, match="cannot be combined"):
        parse_run_selection(run_id="run1", run_last_n=2)


def test_window_and_last_n_conflict():
    with pytest.raises(RunSelectionError, match="cannot be combined"):
        parse_run_selection(run_before="2026-07-01", run_last_n=2)


def test_after_and_before_together_are_one_group():
    got = parse_run_selection(run_after="2026-07-01", run_before="2026-07-02")
    assert (got.after, got.before) == ("2026-07-01", "2026-07-02")


def test_last_n_must_be_positive():
    with pytest.raises(RunSelectionError, match="at least 1"):
        parse_run_selection(run_last_n=0)


# --- resolution: ids -----------------------------------------------------


def test_resolves_ids_by_prefix(store):
    # One run only: two runs opened in the same second share every character up to the
    # random suffix, so any prefix short enough to be worth typing is ambiguous by
    # construction. Disambiguation itself is Store.resolve_run's contract and is tested
    # there; what matters here is that we delegate to it.
    wanted = a_run(store, KEY)
    got = resolve_runs(store, parse_run_selection(run_id=wanted[:14]))
    assert [r.id for r in got] == [wanted]


def test_unknown_id_is_an_error(store):
    a_run(store, KEY)
    with pytest.raises(RunSelectionError, match="no run matching"):
        resolve_runs(store, parse_run_selection(run_id="run_1900"))


def test_ambiguous_prefix_is_an_error(store):
    a_run(store, KEY)
    a_run(store, KEY)
    with pytest.raises(RunSelectionError, match="matches 2 runs"):
        resolve_runs(store, parse_run_selection(run_id="run_"))


def test_empty_selection_returns_every_run_oldest_first(store):
    newer = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    older = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    assert [r.id for r in resolve_runs(store, parse_run_selection())] == [older, newer]


# --- resolution: time windows -------------------------------------------


def test_bare_date_means_utc_midnight(store):
    before_midnight = backdate_run(store, a_run(store, KEY), "2026-07-01T23:59:59+00:00")
    after_midnight = backdate_run(store, a_run(store, KEY), "2026-07-02T00:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02"))
    assert [r.id for r in got] == [after_midnight]
    got = resolve_runs(store, parse_run_selection(run_before="2026-07-02"))
    assert [r.id for r in got] == [before_midnight, after_midnight]


def test_minute_precision_datetime(store):
    early = backdate_run(store, a_run(store, KEY), "2026-07-02T09:29:00+00:00")
    late = backdate_run(store, a_run(store, KEY), "2026-07-02T09:30:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:30"))
    assert [r.id for r in got] == [late]
    assert early not in [r.id for r in got]


def test_second_precision_datetime(store):
    only = backdate_run(store, a_run(store, KEY), "2026-07-02T09:30:15+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:30:15"))
    assert [r.id for r in got] == [only]


def test_explicit_offset_is_honoured(store):
    """09:00+08:00 is 01:00 UTC, so a run at 02:00 UTC is after it and 00:30 is not."""
    early = backdate_run(store, a_run(store, KEY), "2026-07-02T00:30:00+00:00")
    late = backdate_run(store, a_run(store, KEY), "2026-07-02T02:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:00+08:00"))
    assert [r.id for r in got] == [late]
    assert early not in [r.id for r in got]


def test_trailing_z_is_utc(store):
    only = backdate_run(store, a_run(store, KEY), "2026-07-02T10:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after="2026-07-02T09:00:00Z"))
    assert [r.id for r in got] == [only]


def test_run_id_as_a_boundary_is_inclusive(store):
    first = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00.500000+00:00")
    second = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_after=first))
    assert [r.id for r in got] == [first, second]


def test_run_id_as_a_before_boundary_is_inclusive(store):
    first = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00.750000+00:00")
    third = backdate_run(store, a_run(store, KEY), "2026-07-03T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_before=second))
    assert [r.id for r in got] == [first, second]
    assert third not in [r.id for r in got]


def test_unparseable_boundary_that_is_no_run_is_an_error(store):
    a_run(store, KEY)
    with pytest.raises(RunSelectionError, match="no run matching"):
        resolve_runs(store, parse_run_selection(run_after="last tuesday"))


# --- resolution: last_n and cache keys ----------------------------------


def test_last_n_returns_the_most_recent_oldest_first(store):
    backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    second = backdate_run(store, a_run(store, KEY), "2026-07-02T09:00:00+00:00")
    third = backdate_run(store, a_run(store, KEY), "2026-07-03T09:00:00+00:00")
    assert [r.id for r in resolve_runs(store, parse_run_selection(run_last_n=2))] == [
        second,
        third,
    ]


def test_cache_key_narrowing_applies_before_last_n(store):
    mine = backdate_run(store, a_run(store, KEY), "2026-07-01T09:00:00+00:00")
    backdate_run(store, a_run(store, OTHER), "2026-07-02T09:00:00+00:00")
    got = resolve_runs(store, parse_run_selection(run_last_n=1), cache_key_hashes=[KEY.hash])
    assert [r.id for r in got] == [mine]


def test_id_of_another_provider_is_dropped_with_a_warning(store, caplog):
    theirs = a_run(store, OTHER)
    with caplog.at_level("WARNING"):
        got = resolve_runs(
            store, parse_run_selection(run_id=theirs), cache_key_hashes=[KEY.hash]
        )
    assert got == []
    assert "do not belong to the selected provider" in caplog.text


def test_selection_matching_nothing_is_not_an_error(store):
    a_run(store, KEY)
    assert resolve_runs(store, parse_run_selection(run_after="2099-01-01")) == []

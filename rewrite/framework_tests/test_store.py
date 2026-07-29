import re
import sqlite3
import threading
from datetime import datetime, timezone

import pytest
from conftest import a_run

from llmeval.cache_key import compute_cache_key
from llmeval.store import SCHEMA_VERSION, IncompatibleSchema, Store, new_run_id


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def key(model="m1", **params):
    return compute_cache_key(model=model, params=params or {"temperature": 0.7})


@pytest.fixture
def run_id(store):
    return a_run(store, key())


# --- runs ------------------------------------------------------------------


def test_run_ids_are_unique():
    # 50 ids within the same second must still be distinct (the random suffix)
    assert len({new_run_id() for _ in range(50)}) == 50


def test_run_id_is_a_fixed_width_utc_timestamp():
    # Fixed width is the whole point: it makes a plain string sort chronological.
    rid = new_run_id()
    assert re.fullmatch(r"run_\d{8}-\d{6}-[0-9a-f]{4}", rid)
    stamp = datetime.strptime(rid[4:19], "%Y%m%d-%H%M%S").replace(tzinfo=timezone.utc)
    assert abs((datetime.now(timezone.utc) - stamp).total_seconds()) < 60


def test_create_run_round_trips_metadata(store):
    k = key()
    rid = store.create_run(
        k, provider_name="prod", config={"model": "m1"}, params={"mode": "reuse"}, notes="hi"
    )
    row = store.get_run(rid)
    assert row.provider_name == "prod"
    assert row.cache_key_hash == k.hash
    assert row.cache_key_json == k.canonical
    assert row.config == {"model": "m1"}
    assert row.params == {"mode": "reuse"}
    assert row.notes == "hi"


def test_new_run_is_unfinished_until_finished(store):
    rid = store.create_run(key())
    assert store.get_run(rid).finished_at is None
    assert store.get_run(rid).finished is False
    store.finish_run(rid)
    assert store.get_run(rid).finished_at is not None
    assert store.get_run(rid).finished is True


def test_get_run_returns_none_for_unknown_id(store):
    assert store.get_run("run_nope") is None


def test_list_runs_is_newest_first_and_filters_by_cache_key(store):
    k1, k2 = key(temperature=0.7), key(temperature=0.2)
    first = store.create_run(k1)
    second = store.create_run(k1)
    other = store.create_run(k2)

    assert [r.id for r in store.list_runs(k1.hash)] == [second, first]
    assert [r.id for r in store.list_runs(k2.hash)] == [other]
    assert len(store.list_runs()) == 3
    assert len(store.list_runs(limit=2)) == 2


def test_resolve_run_expands_a_unique_prefix(store):
    rid = store.create_run(key())
    assert store.resolve_run(rid[:14]) == rid
    assert store.resolve_run(rid) == rid


def test_resolve_run_raises_on_no_match(store):
    store.create_run(key())
    with pytest.raises(KeyError, match="no run matching"):
        store.resolve_run("run_1900")


def test_resolve_run_raises_on_ambiguous_prefix(store):
    store.create_run(key())
    store.create_run(key())
    # "run_" prefixes every id, so it can never identify one
    with pytest.raises(KeyError, match="matches 2 runs"):
        store.resolve_run("run_")


# --- results ---------------------------------------------------------------


def test_result_requires_a_run(store):
    with pytest.raises(TypeError):
        store.add_result_row("t1", key(), output="orphan")


def test_result_rejects_an_unknown_run_id(store):
    # the FK is enforced (PRAGMA foreign_keys), so a bogus run id cannot slip through
    with pytest.raises(sqlite3.IntegrityError):
        store.add_result_row("t1", key(), run_id="run_does_not_exist", output="x")


def test_results_carry_their_run(store, run_id):
    k = key()
    store.add_result_row("t1", k, run_id=run_id, output="x")
    assert store.get_results("t1", k.hash)[0].run_id == run_id


def test_get_results_for_run_returns_only_that_runs_rows(store):
    k = key()
    r1, r2 = a_run(store, k), a_run(store, k)
    store.add_result_row("t1", k, run_id=r1, output="a")
    store.add_result_row("t2", k, run_id=r1, output="b")
    store.add_result_row("t1", k, run_id=r2, output="c")

    assert [r.output for r in store.get_results_for_run(r1)] == ["a", "b"]
    assert [r.output for r in store.get_results_for_run(r2)] == ["c"]


def test_attempt_numbering_continues_across_runs(store):
    # attempt is scoped to (test, cache_key), NOT to the run: five attempts spread
    # over five invocations must be the same dataset as five from one.
    k = key()
    assert store.add_result("t1", k, run_id=a_run(store, k), output="a") == 0
    assert store.add_result("t1", k, run_id=a_run(store, k), output="b") == 1
    assert store.count_results("t1", k.hash) == 2
    assert [r.attempt for r in store.get_results("t1", k.hash)] == [0, 1]


def test_concurrent_inserts_from_many_threads(store, run_id):
    # The runner calls into one shared Store from a thread pool; inserts to
    # distinct (test, key) rows from many threads must all land without
    # sqlite3 'same thread' errors or lost rows.
    k = key()
    errors: list[Exception] = []

    def worker(i: int) -> None:
        try:
            store.add_result_row(f"t{i}", k, run_id=run_id, output=f"out{i}", config={"i": i})
        except Exception as exc:  # noqa: BLE001 - surface any thread error to the test
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    for i in range(50):
        assert store.count_results(f"t{i}", k.hash) == 1


def test_attempt_index_increments_per_test_and_key(store, run_id):
    k = key()
    assert store.add_result("t1", k, run_id=run_id, output="a") == 0
    assert store.add_result("t1", k, run_id=run_id, output="b") == 1
    assert store.add_result("t1", k, run_id=run_id, output="c") == 2
    assert store.count_results("t1", k.hash) == 3


def test_results_isolated_by_test_and_key(store, run_id):
    k1, k2 = key(temperature=0.7), key(temperature=0.2)
    store.add_result("t1", k1, run_id=run_id, output="a")
    store.add_result("t1", k2, run_id=run_id, output="b")
    store.add_result("t2", k1, run_id=run_id, output="c")
    assert store.count_results("t1", k1.hash) == 1
    assert store.count_results("t1", k2.hash) == 1
    assert store.count_results("t2", k1.hash) == 1


def test_success_only_count_excludes_errors(store, run_id):
    k = key()
    store.add_result("t1", k, run_id=run_id, output="ok")
    store.add_result("t1", k, run_id=run_id, error="boom")
    assert store.count_results("t1", k.hash) == 2
    assert store.count_results("t1", k.hash, success_only=True) == 1


def test_get_results_round_trips_payload(store, run_id):
    k = key()
    store.add_result(
        "t1", k, run_id=run_id, output="hello", reasoning="because",
        tokens={"total": 12}, latency_ms=4.5,
    )
    rows = store.get_results("t1", k.hash)
    assert len(rows) == 1
    r = rows[0]
    assert r.output == "hello"
    assert r.reasoning == "because"
    assert r.tokens == {"total": 12}
    assert r.cache_key_json == k.canonical


def test_stores_full_provider_config(store, run_id):
    k = key()
    cfg = {"name": "p", "model": "m1", "params": {"temperature": 0.7, "max_tokens": 100}}
    store.add_result_row("t1", k, run_id=run_id, output="x", config=cfg)
    assert store.get_results("t1", k.hash)[0].config == cfg


def test_config_is_optional(store, run_id):
    k = key()
    store.add_result_row("t1", k, run_id=run_id, output="x")
    assert store.get_results("t1", k.hash)[0].config is None


# --- schema versioning (no migrations by design) ---------------------------


def test_opening_a_db_from_an_incompatible_schema_raises(tmp_path):
    db = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE results (id INTEGER PRIMARY KEY);")
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.commit()
    conn.close()

    with pytest.raises(IncompatibleSchema, match="no migration path"):
        Store(db)


def test_a_legacy_unversioned_db_with_tables_raises(tmp_path):
    # version 0 + existing tables => written before versioning, not a fresh file
    db = str(tmp_path / "legacy.sqlite3")
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE results (id INTEGER PRIMARY KEY);")
    conn.commit()
    conn.close()

    with pytest.raises(IncompatibleSchema, match="no migration path"):
        Store(db)


def test_a_fresh_file_opens_cleanly(tmp_path):
    db = str(tmp_path / "fresh.sqlite3")
    s = Store(db)
    s.close()
    Store(db).close()  # reopening an already-versioned DB is fine


def test_persists_across_connections(tmp_path):
    db = str(tmp_path / "e.sqlite3")
    k = key()
    s1 = Store(db)
    s1.add_result("t1", k, run_id=a_run(s1, k), output="persisted")
    s1.close()
    s2 = Store(db)
    assert s2.count_results("t1", k.hash) == 1
    s2.close()


# --- gradings (separate from results: re-gradable without re-running) ------


def test_grading_upsert_overwrites_same_assertion(store, run_id):
    k = key()
    rid = store.add_result_row("t1", k, run_id=run_id, output="Paris")
    store.set_grading(rid, assertion_key="icontains:Paris", type="icontains", score=1.0, passed=True)
    store.set_grading(rid, assertion_key="icontains:Paris", type="icontains", score=0.0, passed=False)
    gradings = store.get_gradings(rid)
    assert len(gradings) == 1
    assert gradings[0].passed is False  # latest wins


def test_multiple_assertions_per_result(store, run_id):
    k = key()
    rid = store.add_result_row("t1", k, run_id=run_id, output="x")
    store.set_grading(rid, "a1", type="contains", score=1.0, passed=True)
    store.set_grading(rid, "a2", type="rubric", score=0.5, passed=False, metric="accuracy")
    assert {g.assertion_key for g in store.get_gradings(rid)} == {"a1", "a2"}


def test_iter_graded_results_for_cache_key_joins(store, run_id):
    k = key()
    rid = store.add_result_row("t1", k, run_id=run_id, output="x")
    store.set_grading(rid, "a1", type="rubric", score=0.8, passed=True, metric="accuracy")
    joined = list(store.iter_graded_results(k.hash))
    assert joined[0].test_id == "t1"
    assert joined[0].run_id == run_id
    assert joined[0].score == 0.8
    assert joined[0].metric == "accuracy"


def test_iter_graded_results_can_narrow_to_one_run(store):
    k = key()
    r1, r2 = a_run(store, k), a_run(store, k)
    for rid_, score in ((r1, 0.2), (r2, 0.9)):
        result_id = store.add_result_row("t1", k, run_id=rid_, output="x")
        store.set_grading(result_id, "a1", type="rubric", score=score, metric="accuracy")

    assert [g.score for g in store.iter_graded_results(k.hash)] == [0.2, 0.9]
    assert [g.score for g in store.iter_graded_results(k.hash, run_id=r2)] == [0.9]


# --- verdicts (pick-best, re-runnable against cached outputs) --------------


def test_verdict_upsert_by_test_and_comparison(store):
    store.set_verdict("t1", comparison_key="cmpA", winner_hash="h1", candidates=["h1", "h2"])
    store.set_verdict("t1", comparison_key="cmpA", winner_hash="h2", candidates=["h2", "h1"])
    v = store.get_verdicts("cmpA")
    assert len(v) == 1
    assert v[0].winner_hash == "h2"

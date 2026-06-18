import pytest

from llmeval.cache_key import compute_cache_key
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def key(model="m1", **params):
    return compute_cache_key(model=model, params=params or {"temperature": 0.7})


# --- results ---------------------------------------------------------------


def test_attempt_index_increments_per_test_and_key(store):
    k = key()
    assert store.add_result("t1", k, output="a") == 0
    assert store.add_result("t1", k, output="b") == 1
    assert store.add_result("t1", k, output="c") == 2
    assert store.count_results("t1", k.hash) == 3


def test_results_isolated_by_test_and_key(store):
    k1, k2 = key(temperature=0.7), key(temperature=0.2)
    store.add_result("t1", k1, output="a")
    store.add_result("t1", k2, output="b")
    store.add_result("t2", k1, output="c")
    assert store.count_results("t1", k1.hash) == 1
    assert store.count_results("t1", k2.hash) == 1
    assert store.count_results("t2", k1.hash) == 1


def test_success_only_count_excludes_errors(store):
    k = key()
    store.add_result("t1", k, output="ok")
    store.add_result("t1", k, error="boom")
    assert store.count_results("t1", k.hash) == 2
    assert store.count_results("t1", k.hash, success_only=True) == 1


def test_get_results_round_trips_payload(store):
    k = key()
    store.add_result(
        "t1", k, output="hello", reasoning="because", tokens={"total": 12}, latency_ms=4.5
    )
    rows = store.get_results("t1", k.hash)
    assert len(rows) == 1
    r = rows[0]
    assert r.output == "hello"
    assert r.reasoning == "because"
    assert r.tokens == {"total": 12}
    assert r.cache_key_json == k.canonical


def test_stores_full_provider_config(store):
    k = key()
    cfg = {"name": "p", "model": "m1", "params": {"temperature": 0.7, "max_tokens": 100}}
    store.add_result_row("t1", k, output="x", config=cfg)
    assert store.get_results("t1", k.hash)[0].config == cfg


def test_config_is_optional(store):
    k = key()
    store.add_result_row("t1", k, output="x")
    assert store.get_results("t1", k.hash)[0].config is None


def test_migrates_old_results_table_missing_config_column(tmp_path):
    import sqlite3

    db = str(tmp_path / "old.sqlite3")
    conn = sqlite3.connect(db)
    conn.executescript(
        """CREATE TABLE results (
            id INTEGER PRIMARY KEY AUTOINCREMENT, test_id TEXT NOT NULL,
            cache_key_hash TEXT NOT NULL, cache_key_json TEXT NOT NULL, attempt INTEGER NOT NULL,
            output TEXT, raw_json TEXT, reasoning TEXT, tokens_json TEXT, latency_ms REAL,
            error TEXT, created_at TEXT NOT NULL, UNIQUE(test_id, cache_key_hash, attempt));"""
    )
    conn.commit()
    conn.close()

    s = Store(db)  # opening must migrate the old table
    k = key()
    s.add_result_row("t1", k, output="x", config={"a": 1})
    assert s.get_results("t1", k.hash)[0].config == {"a": 1}
    s.close()


def test_persists_across_connections(tmp_path):
    db = str(tmp_path / "e.sqlite3")
    k = key()
    s1 = Store(db)
    rid = s1.add_result("t1", k, output="persisted")
    s1.close()
    s2 = Store(db)
    assert s2.count_results("t1", k.hash) == 1
    s2.close()


# --- gradings (separate from results: re-gradable without re-running) ------


def test_grading_upsert_overwrites_same_assertion(store):
    k = key()
    rid = store.add_result_row("t1", k, output="Paris")
    store.set_grading(rid, assertion_key="icontains:Paris", type="icontains", score=1.0, passed=True)
    store.set_grading(rid, assertion_key="icontains:Paris", type="icontains", score=0.0, passed=False)
    gradings = store.get_gradings(rid)
    assert len(gradings) == 1
    assert gradings[0].passed is False  # latest wins


def test_multiple_assertions_per_result(store):
    k = key()
    rid = store.add_result_row("t1", k, output="x")
    store.set_grading(rid, "a1", type="contains", score=1.0, passed=True)
    store.set_grading(rid, "a2", type="rubric", score=0.5, passed=False, metric="accuracy")
    assert {g.assertion_key for g in store.get_gradings(rid)} == {"a1", "a2"}


def test_iter_graded_results_for_cache_key_joins(store):
    k = key()
    rid = store.add_result_row("t1", k, output="x")
    store.set_grading(rid, "a1", type="rubric", score=0.8, passed=True, metric="accuracy")
    joined = list(store.iter_graded_results(k.hash))
    assert joined[0].test_id == "t1"
    assert joined[0].score == 0.8
    assert joined[0].metric == "accuracy"


# --- verdicts (pick-best, re-runnable against cached outputs) --------------


def test_verdict_upsert_by_test_and_comparison(store):
    store.set_verdict("t1", comparison_key="cmpA", winner_hash="h1", candidates=["h1", "h2"])
    store.set_verdict("t1", comparison_key="cmpA", winner_hash="h2", candidates=["h2", "h1"])
    v = store.get_verdicts("cmpA")
    assert len(v) == 1
    assert v[0].winner_hash == "h2"

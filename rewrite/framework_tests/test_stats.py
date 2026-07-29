import pytest
from conftest import a_run

from llmeval.cache_key import compute_cache_key
from llmeval.comparison import stats
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def graded(store, key, test_id, scores, metric="accuracy"):
    """Add one result per score (attempts), each graded on the same assertion."""
    run_id = a_run(store, key)
    for s in scores:
        rid = store.add_result_row(test_id, key, run_id=run_id, output="x")
        store.set_grading(rid, "a1", type="rubric", score=s, passed=s >= 0.5, metric=metric)


def test_reduce_attempts():
    assert stats.reduce_attempts([0.4, 0.8], "mean") == pytest.approx(0.6)
    assert stats.reduce_attempts([0.4, 0.8], "max") == 0.8
    assert stats.reduce_attempts([1.0, 0.0, 1.0], "pass_rate") == pytest.approx(2 / 3)


def test_summarize_means_across_test_cases(store):
    k = compute_cache_key(model="m1", params={"t": 1})
    graded(store, k, "t1", [0.8])
    graded(store, k, "t2", [0.6])
    summary = stats.summarize(store, k.hash, metric="accuracy")
    assert summary.n == 2
    assert summary.mean == pytest.approx(0.7)


def test_summarize_reduces_attempts_first(store):
    k = compute_cache_key(model="m1", params={"t": 1})
    graded(store, k, "t1", [0.4, 0.8])  # best-of-2
    graded(store, k, "t2", [0.6])
    summary = stats.summarize(store, k.hash, metric="accuracy", attempt_reducer="max")
    assert sorted(summary.values) == [0.6, 0.8]
    assert summary.mean == pytest.approx(0.7)


def test_bootstrap_ci_is_deterministic_and_brackets_mean(store):
    lo1, hi1 = stats.bootstrap_ci([0.5, 0.6, 0.7, 0.8], bootstrap=200, seed=1)
    lo2, hi2 = stats.bootstrap_ci([0.5, 0.6, 0.7, 0.8], bootstrap=200, seed=1)
    assert (lo1, hi1) == (lo2, hi2)
    assert lo1 <= 0.65 <= hi1


def test_compare_metric_computes_delta_vs_baseline(store):
    base = compute_cache_key(model="m1", params={"t": 0})
    cand = compute_cache_key(model="m1", params={"t": 1})
    graded(store, base, "t1", [0.5])
    graded(store, base, "t2", [0.5])
    graded(store, cand, "t1", [0.7])
    graded(store, cand, "t2", [0.7])
    rows = stats.compare_metric(
        store,
        configs=[("base", base.hash), ("cand", cand.hash)],
        metric="accuracy",
        baseline_name="base",
    )
    by_name = {r.name: r for r in rows}
    assert by_name["base"].delta is None
    assert by_name["cand"].delta == pytest.approx(0.2)


def test_win_rates_from_verdicts(store):
    store.set_verdict("t1", "cmp", winner_hash="hA", candidates=["hA", "hB"])
    store.set_verdict("t2", "cmp", winner_hash="hA", candidates=["hA", "hB"])
    store.set_verdict("t3", "cmp", winner_hash="hB", candidates=["hA", "hB"])
    store.set_verdict("t4", "cmp", winner_hash=None, candidates=["hA", "hB"])
    wr = stats.win_rates(store, "cmp")
    assert wr.total == 4
    assert wr.undecided == 1
    assert wr.wins["hA"] == 2
    assert wr.rate("hA") == pytest.approx(2 / 3)  # of decided

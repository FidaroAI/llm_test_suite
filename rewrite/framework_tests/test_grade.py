import pytest

from llmeval.cache_key import compute_cache_key
from llmeval.grade import assertion_key, grade_testcase
from llmeval.models import AssertionSpec, TestCase
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


KEY = compute_cache_key(model="m1", params={"temperature": 0.7})


def seed(store, output="The capital is Paris.", test_id="t1"):
    return store.add_result_row(test_id, KEY, output=output, reasoning="thinking")


def tc(assertions, test_id="t1"):
    return TestCase.from_dict({"id": test_id, "user": "capital of France?", "assertions": assertions})


def test_grades_cached_output_without_running_a_model(store):
    rid = seed(store)
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash)
    g = store.get_gradings(rid)
    assert len(g) == 1
    assert g[0].passed is True
    assert g[0].type == "icontains"


def test_multiple_assertions_each_graded(store):
    rid = seed(store)
    grade_testcase(
        store,
        tc([{"type": "icontains", "value": "Paris"}, {"type": "not_contains", "value": "Berlin"}]),
        KEY.hash,
    )
    assert len(store.get_gradings(rid)) == 2


def test_regrade_is_idempotent(store):
    rid = seed(store)
    t = tc([{"type": "icontains", "value": "Paris"}])
    grade_testcase(store, t, KEY.hash)
    grade_testcase(store, t, KEY.hash, regrade=True)
    assert len(store.get_gradings(rid)) == 1  # upsert, not duplicate


def test_skip_existing_avoids_recalling_judge(store):
    seed(store)

    class CountingJudge:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            return '{"score": 1.0}'

    j = CountingJudge()
    t = tc([{"type": "rubric", "value": "accurate"}])
    grade_testcase(store, t, KEY.hash, judge=j)
    grade_testcase(store, t, KEY.hash, judge=j)  # default: skip already-graded
    assert j.calls == 1
    grade_testcase(store, t, KEY.hash, judge=j, regrade=True)  # force
    assert j.calls == 2


def test_editing_assertion_value_changes_key(store):
    a = AssertionSpec(type="icontains", value="Paris")
    b = AssertionSpec(type="icontains", value="Lyon")
    assert assertion_key(a) != assertion_key(b)


def test_explicit_id_is_used_as_key(store):
    a = AssertionSpec(type="icontains", value="Paris", id="capital-check")
    assert assertion_key(a) == "capital-check"


def test_error_rows_are_not_graded(store):
    store.add_result_row("t1", KEY, error="timeout")
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash)
    # the only result is an error row; nothing graded
    assert list(store.iter_graded_results(KEY.hash)) == []

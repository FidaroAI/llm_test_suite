import pytest
from conftest import a_run

from llmeval.cache_key import compute_cache_key
from llmeval.grade import assertion_key, grade, grade_testcase
from llmeval.models import AssertionSpec, TestCase
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


KEY = compute_cache_key(model="m1", params={"temperature": 0.7})


def seed(store, output="The capital is Paris.", test_id="t1"):
    return store.add_result_row(
        test_id, run_id=a_run(store, KEY), output=output, reasoning="thinking"
    )


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
    store.add_result_row("t1", run_id=a_run(store, KEY), error="timeout")
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash)
    # the only result is an error row; nothing graded
    assert list(store.iter_graded_results(KEY.hash)) == []


def test_grades_every_result_across_runs_by_default(store):
    """A grading belongs to a result, so two runs of one test yield two gradings."""
    first = store.add_result_row("t1", run_id=a_run(store, KEY), output="Paris is it")
    second = store.add_result_row("t1", run_id=a_run(store, KEY), output="Also Paris")
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash)
    assert len(store.get_gradings(first)) == 1
    assert len(store.get_gradings(second)) == 1


def test_grades_every_attempt_within_one_run(store):
    run = a_run(store, KEY)
    first = store.add_result_row("t1", run_id=run, output="Paris, attempt one")
    second = store.add_result_row("t1", run_id=run, output="Paris, attempt two")
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash)
    assert len(store.get_gradings(first)) == 1
    assert len(store.get_gradings(second)) == 1


def test_run_ids_narrows_grading_to_those_runs(store):
    wanted_run = a_run(store, KEY)
    other_run = a_run(store, KEY)
    wanted = store.add_result_row("t1", run_id=wanted_run, output="Paris")
    other = store.add_result_row("t1", run_id=other_run, output="Paris")
    grade_testcase(
        store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash, run_ids=[wanted_run]
    )
    assert len(store.get_gradings(wanted)) == 1
    assert store.get_gradings(other) == []


def test_empty_run_ids_grades_nothing(store):
    rid = seed(store)
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash, run_ids=[])
    assert store.get_gradings(rid) == []


def test_error_rows_are_still_skipped_when_narrowing(store):
    run = a_run(store, KEY)
    store.add_result_row("t1", run_id=run, error="timeout")
    grade_testcase(store, tc([{"type": "icontains", "value": "Paris"}]), KEY.hash, run_ids=[run])
    assert list(store.iter_graded_results(KEY.hash)) == []


def test_narrowing_does_not_redo_existing_gradings(store):
    run = a_run(store, KEY)
    store.add_result_row("t1", run_id=run, output="Paris")

    class CountingJudge:
        def __init__(self):
            self.calls = 0

        def __call__(self, prompt):
            self.calls += 1
            return '{"score": 1.0}'

    j = CountingJudge()
    t = tc([{"type": "rubric", "value": "accurate"}])
    grade_testcase(store, t, KEY.hash, judge=j, run_ids=[run])
    grade_testcase(store, t, KEY.hash, judge=j, run_ids=[run])
    assert j.calls == 1


class GradeHookSpy:
    """Records hook calls in order. Stands in for llmeval.plugins.loader.Hooks."""

    def __init__(self):
        self.calls = []

    def before_grade(self):
        self.calls.append("before_grade")

    def after_grade(self):
        self.calls.append("after_grade")

    def before_each_grade(self, testcase):
        self.calls.append(f"before_each:{testcase.id}")

    def after_each_grade(self, testcase, gradings):
        self.calls.append((testcase.id, [g.assertion_key for g in gradings]))


def test_grade_calls_hooks_and_reports_what_it_graded(store):
    seed(store)
    case = tc([{"type": "icontains", "value": "Paris"}])

    spy = GradeHookSpy()
    grade(store, [case], KEY.hash, hooks=spy)
    assert spy.calls[0] == "before_grade"
    assert spy.calls[1] == "before_each:t1"
    assert spy.calls[-1] == "after_grade"
    graded = [c for c in spy.calls if isinstance(c, tuple)]
    assert graded[0][0] == "t1"
    assert len(graded[0][1]) == 1


def test_a_second_pass_fires_the_hooks_but_grades_nothing_new(store):
    seed(store)
    case = tc([{"type": "icontains", "value": "Paris"}])
    grade(store, [case], KEY.hash)

    spy = GradeHookSpy()
    grade(store, [case], KEY.hash, hooks=spy)
    assert [c for c in spy.calls if isinstance(c, tuple)][0][1] == []

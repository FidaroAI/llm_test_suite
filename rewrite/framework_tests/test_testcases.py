from llmeval.testcases import select_testcases


def items(n):
    return [f"t{i}" for i in range(n)]


def test_limit_takes_first_n_without_randomize():
    assert select_testcases(items(5), limit=3) == ["t0", "t1", "t2"]


def test_limit_none_returns_all():
    assert select_testcases(items(5), limit=None) == items(5)


def test_limit_larger_than_len_returns_all():
    assert select_testcases(items(3), limit=10) == items(3)


def test_randomize_is_deterministic_for_seed():
    a = select_testcases(items(10), randomize=True, seed=0)
    b = select_testcases(items(10), randomize=True, seed=0)
    assert a == b
    assert set(a) == set(items(10))


def test_randomize_different_seed_changes_order():
    a = select_testcases(items(20), randomize=True, seed=0)
    b = select_testcases(items(20), randomize=True, seed=1)
    assert a != b


def test_randomize_then_limit_is_a_deterministic_sample():
    a = select_testcases(items(10), limit=3, randomize=True, seed=0)
    assert len(a) == 3
    assert set(a).issubset(set(items(10)))
    assert a == select_testcases(items(10), limit=3, randomize=True, seed=0)


def test_randomize_false_ignores_seed():
    assert select_testcases(items(5), randomize=False, seed=99) == items(5)


def test_does_not_mutate_input():
    src = items(5)
    select_testcases(src, randomize=True, seed=2)
    assert src == items(5)

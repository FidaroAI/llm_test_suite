import re

import pytest
from conftest import a_run

from llmeval.comparison.pickbest import pick_best_testcase
from llmeval.models import ProviderConfig, TestCase
from llmeval.store import Store


@pytest.fixture
def store():
    s = Store(":memory:")
    yield s
    s.close()


def config(name, temp):
    return ProviderConfig(name=name, model="m1", params={"temperature": temp})


CFG_A = config("A", 0.1)
CFG_B = config("B", 0.2)


def tc():
    return TestCase.from_dict({"id": "t1", "user": "capital of France?"})


def seed(store, cfg, output, test_id="t1"):
    key = cfg.cache_key()
    store.add_result_row(test_id, key, run_id=a_run(store, key), output=output)


class ContentJudge:
    """Picks whichever response block contains the marker (content-based, unbiased)."""

    def __init__(self, marker):
        self.marker = marker
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        for m in re.finditer(r'<response index="(\d+)">\n(.*?)\n</response>', prompt, re.S):
            if self.marker in m.group(2):
                return m.group(1)
        return "0"


class PositionJudge:
    """Always picks a fixed index — simulates position bias."""

    def __init__(self, idx="0"):
        self.idx = idx
        self.calls = 0

    def __call__(self, prompt):
        self.calls += 1
        return self.idx


def test_picks_winner_by_content(store):
    seed(store, CFG_A, "ALPHA is the best answer")
    seed(store, CFG_B, "BETA answer")
    winner = pick_best_testcase(store, tc(), [CFG_A, CFG_B], ContentJudge("ALPHA"))
    assert winner == CFG_A.cache_key().hash


def test_undecided_when_fewer_than_two_outputs(store):
    seed(store, CFG_A, "only one answer")
    winner = pick_best_testcase(store, tc(), [CFG_A, CFG_B], ContentJudge("ALPHA"))
    assert winner is None


def test_order_both_agreement_yields_winner(store):
    seed(store, CFG_A, "ALPHA answer")
    seed(store, CFG_B, "BETA answer")
    # content judge picks ALPHA regardless of position -> both orderings agree
    winner = pick_best_testcase(store, tc(), [CFG_A, CFG_B], ContentJudge("ALPHA"), order="both")
    assert winner == CFG_A.cache_key().hash


def test_order_both_detects_position_bias_as_undecided(store):
    seed(store, CFG_A, "ALPHA answer")
    seed(store, CFG_B, "BETA answer")
    # always-index-0 judge picks whichever is shown first -> orderings disagree
    winner = pick_best_testcase(store, tc(), [CFG_A, CFG_B], PositionJudge("0"), order="both")
    assert winner is None


def test_verdict_is_cached_and_reused(store):
    seed(store, CFG_A, "ALPHA answer")
    seed(store, CFG_B, "BETA answer")
    j = ContentJudge("ALPHA")
    pick_best_testcase(store, tc(), [CFG_A, CFG_B], j)
    pick_best_testcase(store, tc(), [CFG_A, CFG_B], j)  # reuse stored verdict
    assert j.calls == 1
    pick_best_testcase(store, tc(), [CFG_A, CFG_B], j, regrade=True)
    assert j.calls == 2

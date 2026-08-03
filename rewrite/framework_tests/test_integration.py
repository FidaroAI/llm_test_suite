"""End-to-end composition of the whole pipeline, offline (mock provider + judge).

Proves the stages decouple cleanly and share the store via the cache key.
"""

import pytest

from llmeval.comparison import stats
from llmeval.comparison.pickbest import DEFAULT_CRITERION, comparison_key, pick_best
from llmeval.comparison.report import write_report
from llmeval.grade import grade
from llmeval.models import ProviderConfig, TestCase
from llmeval.providers import Completion
from llmeval.runner import RunPolicy, run
from llmeval.store import Store


class SimpleProvider:
    def __init__(self, config, output):
        self.config = config
        self.output = output

    def complete(self, messages, timeout=None):
        return Completion(output=self.output, reasoning="r")


def rubric_judge(prompt):
    # scores high only when the answer carries the GOOD marker
    return '{"score": 0.9}' if "GOOD" in prompt else '{"score": 0.3}'


def content_judge(prompt):
    import re

    for m in re.finditer(r'<response index="(\d+)">\n(.*?)\n</response>', prompt, re.S):
        if "GOOD" in m.group(2):
            return m.group(1)
    return "0"


def make_testcases():
    return [
        TestCase.from_dict(
            {
                "id": f"t{i}",
                "user": "capital of France?",
                "assertions": [
                    {"type": "icontains", "value": "Paris"},
                    {"type": "rubric", "value": "accurate", "metric": "accuracy"},
                ],
            }
        )
        for i in (1, 2)
    ]


def test_full_pipeline(tmp_path):
    store = Store(str(tmp_path / "e.sqlite3"))
    cfg_a = ProviderConfig(name="A", model="m1", params={"temperature": 0.1})
    cfg_b = ProviderConfig(name="B", model="m1", params={"temperature": 0.2})
    tcs = make_testcases()

    # RUN best-of-2 for each config (mock provider, no network)
    run(store, tcs, SimpleProvider(cfg_a, "reasoning\n\n\nGOOD: Paris is the capital."),
        RunPolicy(mode="reuse", repeat=2))
    run(store, tcs, SimpleProvider(cfg_b, "reasoning\n\n\nParis, maybe."),
        RunPolicy(mode="reuse", repeat=2))
    assert store.count_results("t1", cfg_a.cache_key().hash, success_only=True) == 2

    # GRADE cached outputs (re-runnable; no model calls)
    grade(store, tcs, cfg_a.cache_key().hash, judge=rubric_judge)
    grade(store, tcs, cfg_b.cache_key().hash, judge=rubric_judge)

    # INDIRECT comparison (ratings)
    rows = stats.compare_metric(
        store,
        [("A", cfg_a.cache_key().hash), ("B", cfg_b.cache_key().hash)],
        metric="accuracy",
        baseline_name="A",
    )
    by = {r.name: r for r in rows}
    assert by["A"].summary.mean == pytest.approx(0.9)
    assert by["B"].delta == pytest.approx(-0.6)

    # DIRECT comparison (pick-best), re-runs against cached outputs
    pick_best(store, tcs, [cfg_a, cfg_b], content_judge)
    wr = stats.win_rates(store, comparison_key([cfg_a, cfg_b], DEFAULT_CRITERION, "as_is"))
    assert wr.wins[cfg_a.cache_key().hash] == 2
    assert wr.rate(cfg_a.cache_key().hash) == pytest.approx(1.0)

    # REPORT
    out = write_report(
        store,
        [("A", cfg_a.cache_key().hash), ("B", cfg_b.cache_key().hash)],
        ["accuracy"],
        str(tmp_path / "report.html"),
        baseline_name="A",
        comparison_key=comparison_key([cfg_a, cfg_b], DEFAULT_CRITERION, "as_is"),
    )
    assert (tmp_path / "report.html").read_text().count("A") > 0
    store.close()

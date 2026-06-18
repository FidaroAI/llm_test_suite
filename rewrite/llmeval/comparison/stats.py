"""Indirect comparison: aggregate cached gradings into per-config statistics.

Two-level aggregation: within a test case, the N attempts (best-of-N) are reduced to one
value (``attempt_reducer``); across test cases, those values are summarised (mean + a
bootstrap 95% CI, stdlib only). Deltas are taken against a named baseline config. Direct
comparison (pick-best) win rates come from stored verdicts.

TODO: richer "which config is best" methods (Bradley-Terry / Elo over pairwise verdicts,
significance testing). The reducers + CI + win-rate here are the dependency-free baseline.
"""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from dataclasses import dataclass, field

from llmeval.store import Store


def reduce_attempts(scores: list[float], how: str = "mean", threshold: float = 0.5) -> float:
    if not scores:
        return 0.0
    if how == "mean":
        return statistics.fmean(scores)
    if how == "max":
        return max(scores)
    if how == "min":
        return min(scores)
    if how == "pass_rate":
        return sum(1 for s in scores if s >= threshold) / len(scores)
    raise ValueError(f"unknown attempt reducer: {how!r}")


def bootstrap_ci(
    values: list[float], bootstrap: int = 1000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    if len(values) == 1:
        return (values[0], values[0])
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(bootstrap):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int((alpha / 2) * bootstrap)]
    hi = means[min(bootstrap - 1, int((1 - alpha / 2) * bootstrap))]
    return (lo, hi)


@dataclass
class MetricSummary:
    metric: str
    n: int
    mean: float
    stdev: float
    ci_low: float
    ci_high: float
    values: list[float] = field(default_factory=list)


def per_test_scores(
    store: Store, key_hash: str, metric: str | None, attempt_reducer: str = "mean"
) -> dict[str, float]:
    """Reduce each test case's attempts to a single score for the given metric."""
    by_test: dict[str, list[float]] = defaultdict(list)
    for row in store.iter_graded_results(key_hash):
        if metric is not None and row.metric != metric:
            continue
        if row.score is not None:
            by_test[row.test_id].append(row.score)
    return {tid: reduce_attempts(scores, attempt_reducer) for tid, scores in by_test.items()}


def summarize(
    store: Store,
    key_hash: str,
    metric: str | None = None,
    attempt_reducer: str = "mean",
    bootstrap: int = 1000,
    seed: int = 0,
) -> MetricSummary:
    per_test = per_test_scores(store, key_hash, metric, attempt_reducer)
    values = list(per_test.values())
    if not values:
        return MetricSummary(metric or "", 0, 0.0, 0.0, 0.0, 0.0, [])
    mean = statistics.fmean(values)
    stdev = statistics.stdev(values) if len(values) > 1 else 0.0
    lo, hi = bootstrap_ci(values, bootstrap=bootstrap, seed=seed)
    return MetricSummary(metric or "", len(values), mean, stdev, lo, hi, values)


@dataclass
class ConfigSummary:
    name: str
    hash: str
    summary: MetricSummary
    delta: float | None  # mean - baseline mean; None for the baseline itself


def compare_metric(
    store: Store,
    configs: list[tuple[str, str]],
    metric: str | None = None,
    baseline_name: str | None = None,
    attempt_reducer: str = "mean",
) -> list[ConfigSummary]:
    summaries = {
        name: summarize(store, key_hash, metric, attempt_reducer)
        for name, key_hash in configs
    }
    base_mean = summaries[baseline_name].mean if baseline_name in summaries else None
    rows = []
    for name, key_hash in configs:
        s = summaries[name]
        delta = None if (base_mean is None or name == baseline_name) else s.mean - base_mean
        rows.append(ConfigSummary(name=name, hash=key_hash, summary=s, delta=delta))
    return rows


@dataclass
class WinRates:
    total: int
    undecided: int
    wins: dict[str, int]

    @property
    def decided(self) -> int:
        return self.total - self.undecided

    def rate(self, key_hash: str) -> float:
        return self.wins.get(key_hash, 0) / self.decided if self.decided else 0.0


def win_rates(store: Store, comparison_key: str) -> WinRates:
    verdicts = store.get_verdicts(comparison_key)
    wins: dict[str, int] = defaultdict(int)
    undecided = 0
    for v in verdicts:
        if v.winner_hash is None:
            undecided += 1
        else:
            wins[v.winner_hash] += 1
    return WinRates(total=len(verdicts), undecided=undecided, wins=dict(wins))

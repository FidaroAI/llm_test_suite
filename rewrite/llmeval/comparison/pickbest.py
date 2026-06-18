"""Pick-best: direct head-to-head comparison of N configs on a test case.

Runs entirely against cached outputs, so it can be (re-)run without touching the model
under test. Verdicts are cached per (test, comparison) and reused.

Order control fights position bias:
* ``as_is``  — present configs in the given order ("force X before Y" = order the list).
* ``random`` — deterministic shuffle (seeded by test id) so position is decorrelated.
* ``both``   — present both orderings; agree → that winner, disagree → undecided
  (the comparison is position-biased for this test).
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from typing import Callable, Iterable

from llmeval.models import ProviderConfig, TestCase
from llmeval.store import Store

DEFAULT_CRITERION = (
    "the response that most accurately, completely, and clearly answers the user's "
    "question, without adding unsupported or fabricated claims"
)

_PROMPT = (
    "You are choosing which response is {criterion}.\n\n"
    "User question:\n{question}\n\n"
    "{blocks}\n\n"
    "Reply with only the integer index of the best response."
)


def comparison_key(configs: list[ProviderConfig], criterion: str, order: str) -> str:
    payload = json.dumps(
        {
            "configs": sorted(c.cache_key().hash for c in configs),
            "criterion": criterion,
            "order": order,
        },
        sort_keys=True,
    )
    return "cmp:" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _first_output(store: Store, test_id: str, key_hash: str) -> str | None:
    for r in store.get_results(test_id, key_hash):
        if r.error is None and r.output is not None:
            return r.output
    return None


def _ordered(candidates: list[tuple[ProviderConfig, str]], order: str, seed_str: str):
    if order == "random":
        rng = random.Random(seed_str)
        out = list(candidates)
        rng.shuffle(out)
        return out
    return list(candidates)  # as_is / fixed


def _ask(judge: Callable[[str], str], question: str, ordered, criterion: str) -> int | None:
    blocks = "\n".join(
        f'<response index="{i}">\n{out}\n</response>' for i, (_, out) in enumerate(ordered)
    )
    reply = judge(_PROMPT.format(criterion=criterion, question=question, blocks=blocks))
    m = re.search(r"\d+", reply or "")
    if not m:
        return None
    idx = int(m.group())
    return idx if 0 <= idx < len(ordered) else None


def pick_best_testcase(
    store: Store,
    testcase: TestCase,
    configs: list[ProviderConfig],
    judge: Callable[[str], str],
    order: str = "as_is",
    criterion: str = DEFAULT_CRITERION,
    regrade: bool = False,
) -> str | None:
    ckey = comparison_key(configs, criterion, order)
    if not regrade:
        prior = {v.test_id: v for v in store.get_verdicts(ckey)}
        if testcase.id in prior:
            return prior[testcase.id].winner_hash

    candidates = [
        (c, out)
        for c in configs
        if (out := _first_output(store, testcase.id, c.cache_key().hash)) is not None
    ]
    if len(candidates) < 2:
        store.set_verdict(
            testcase.id, ckey, None,
            [c.cache_key().hash for c, _ in candidates],
            reason="undecided: fewer than two outputs to compare",
        )
        return None

    if order == "both":
        fwd = _decide(judge, testcase.user_text, candidates, criterion)
        rev = _decide(judge, testcase.user_text, list(reversed(candidates)), criterion)
        winner = fwd if fwd == rev else None
        reason = "agreed under both orderings" if winner else "undecided: order-dependent verdict"
        store.set_verdict(
            testcase.id, ckey, winner, [c.cache_key().hash for c, _ in candidates], reason=reason
        )
        return winner

    ordered = _ordered(candidates, order, f"{ckey}:{testcase.id}")
    idx = _ask(judge, testcase.user_text, ordered, criterion)
    winner = ordered[idx][0].cache_key().hash if idx is not None else None
    store.set_verdict(
        testcase.id,
        ckey,
        winner,
        [c.cache_key().hash for c, _ in ordered],
        reason="undecided: unparseable judge reply" if winner is None else "selected",
    )
    return winner


def _decide(judge, question, candidates, criterion) -> str | None:
    idx = _ask(judge, question, candidates, criterion)
    return candidates[idx][0].cache_key().hash if idx is not None else None


def pick_best(
    store: Store,
    testcases: Iterable[TestCase],
    configs: list[ProviderConfig],
    judge: Callable[[str], str],
    order: str = "as_is",
    criterion: str = DEFAULT_CRITERION,
    regrade: bool = False,
) -> None:
    for testcase in testcases:
        pick_best_testcase(
            store, testcase, configs, judge, order=order, criterion=criterion, regrade=regrade
        )

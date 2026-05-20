"""Entrypoint: run the two LangSmith experiments (factual + rubric).

The LangSmith analog of `promptfoo eval`. Unlike deep_eval (fully local),
LangSmith uploads datasets and experiment results to your LangSmith project and
prints links to the web UI.

    python run_eval.py

Needs LANGSMITH_API_KEY, a reachable model under test (MUT_*/VLLM_*), and a judge
(ANTHROPIC_API_KEY) for the rubric experiment. Prints what's missing and exits
cleanly if anything is absent.
"""
from __future__ import annotations

import sys
from pathlib import Path

from dotenv import load_dotenv

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

# Local demo .env wins; repo-root .env fills gaps (VLLM_*, ANTHROPIC_API_KEY).
load_dotenv(_HERE / ".env")
load_dotenv(_HERE.parent / ".env", override=False)

from ls_demo.config import (  # noqa: E402
    judge_config,
    langsmith_is_configured,
    model_under_test_config,
)
from ls_demo.datasets import (  # noqa: E402
    FACTS_DATASET,
    SUPPORT_DATASET,
    seed,
)
from ls_demo.evaluators import make_contains_evaluator, make_rubric_evaluator  # noqa: E402
from ls_demo.target import run_target  # noqa: E402


def _preflight() -> list[str]:
    problems = []
    if not langsmith_is_configured():
        problems.append("LANGSMITH_API_KEY (sign up at https://smith.langchain.com)")
    mut = model_under_test_config()
    if not mut.is_configured:
        problems.append(f"model under test: {mut.missing}")
    judge = judge_config()
    if not judge.is_configured:
        problems.append(f"judge: {judge.missing}")
    return problems


def main() -> int:
    problems = _preflight()
    if problems:
        print("Not configured to run live. Missing:")
        for p in problems:
            print(f"  - {p}")
        print("\nSee README.md. The evaluator logic is unit-tested locally with:")
        print("  .venv/bin/python -m pytest")
        return 0

    from langsmith import Client
    from langsmith.evaluation import evaluate

    client = Client()
    for line in seed(client):
        print(line)

    print("\nRunning factual experiment (deterministic contains evaluator)...")
    facts = evaluate(
        run_target,
        data=FACTS_DATASET,
        evaluators=[make_contains_evaluator()],
        experiment_prefix="deepeval-parity-facts",
        max_concurrency=2,
    )
    print(facts)

    print("\nRunning rubric experiment (LLM-as-judge)...")
    rubric = evaluate(
        run_target,
        data=SUPPORT_DATASET,
        evaluators=[make_rubric_evaluator()],
        experiment_prefix="deepeval-parity-support",
        max_concurrency=1,
    )
    print(rubric)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

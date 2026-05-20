"""The two LangSmith datasets, mirroring the deep_eval demo's two test cases.

In LangSmith a dataset is a server-side object of examples (inputs + optional
reference outputs). Both datasets use the `question` input key so one target
function serves both. `seed()` is idempotent.
"""
from __future__ import annotations

# Factual dataset: same shape as the parent suite's simple_facts.csv.
FACTS_DATASET = "deepeval-parity-facts"
FACTS = [
    {"inputs": {"question": "What is the capital of France?"}, "outputs": {"expected": "Paris"}},
    {"inputs": {"question": "What is the capital of Canada?"}, "outputs": {"expected": "Ottawa"}},
    {"inputs": {"question": "What software company is headquartered in Redmond, Washington?"},
     "outputs": {"expected": "Microsoft"}},
]

# Rubric dataset: one open-ended prompt judged by an LLM (no reference output).
SUPPORT_DATASET = "deepeval-parity-support"
SUPPORT_PROMPT = (
    "A customer writes: 'My order #4471 arrived with a cracked screen. "
    "I need this resolved today.' Write a short customer-support reply."
)
SUPPORT = [{"inputs": {"question": SUPPORT_PROMPT}, "outputs": {}}]


def _ensure(client, name: str, description: str, examples: list) -> str:
    if client.has_dataset(dataset_name=name):
        return f"dataset {name!r} already exists — unchanged"
    ds = client.create_dataset(dataset_name=name, description=description)
    client.create_examples(dataset_id=ds.id, examples=examples)
    return f"created dataset {name!r} with {len(examples)} example(s)"


def seed(client) -> list[str]:
    return [
        _ensure(client, FACTS_DATASET, "Factual recall — mirrors deep_eval test_factual", FACTS),
        _ensure(client, SUPPORT_DATASET, "Support-reply rubric — mirrors deep_eval test_rubric", SUPPORT),
    ]

# LangSmith demo

> ⚠️ **Prototype.** This is an exploratory spike to evaluate
> [LangSmith](https://docs.smith.langchain.com) against the existing promptfoo
> suite — not production code. It is intentionally minimal, not wired into CI,
> and its structure/API choices may change. Don't depend on it.

The LangSmith counterpart of the `deep_eval/` demo. It runs the **same two test
cases**, so you can compare the frameworks head-to-head:

1. **A simple factual test** — ask the LLM "What is the capital of France?" and
   assert the answer contains `Paris` (deterministic, no judge).
2. **A rubric test judged by another LLM** — score a support reply against a
   rubric using an LLM-as-judge (the same rubric the deep_eval demo uses).

## promptfoo → LangSmith mapping

| promptfoo                                  | LangSmith here                                  |
|--------------------------------------------|-------------------------------------------------|
| `providers:` makes the model call for you  | `ls_demo/target.py` — a `@traceable` function   |
| test `vars` (the prompts)                  | examples in a server-side **dataset** (`datasets.py`) |
| `icontains:Paris` assertion                | `make_contains_evaluator()` (`evaluators.py`)   |
| custom Python asserts in `assertions/`     | evaluator functions returning `{key, score, comment}` |
| `llm-rubric` + Anthropic judge             | `make_rubric_evaluator()` + `CloudRubricJudge`  |
| `promptfoo eval`                           | `python run_eval.py` (results in the LangSmith UI) |

How LangSmith differs from both promptfoo and deep_eval: datasets and experiment
results are **server-side objects** in your LangSmith project, so the full
`evaluate()` flow needs a LangSmith account/API key — even for the deterministic
check. (deep_eval runs entirely locally.)

## Layout

```
langsmith_demo/
  ls_demo/
    config.py       # env config (MUT + judge), same shape as the deep_eval demo
    target.py       # OpenAI-compatible model under test, wrapped @traceable
    evaluators.py   # contains (deterministic) + rubric (LLM-as-judge) evaluators
    datasets.py     # the two datasets + idempotent seed()
  tests/
    test_evaluators.py  # NO LangSmith / network — proves the evaluator logic
  run_eval.py       # entrypoint: seed + evaluate() both datasets
```

## Setup

```bash
cd langsmith_demo
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env       # fill in LANGSMITH_API_KEY, MUT_*, ANTHROPIC_API_KEY
```

`config.py` falls back to the parent suite's `../.env` (`VLLM_*`,
`ANTHROPIC_API_KEY`), so an existing top-level `.env` mostly just works. You
still need `LANGSMITH_API_KEY` (sign up free at <https://smith.langchain.com>).

## Running

```bash
# Local proof of the evaluator logic — no LangSmith, no network, no key:
.venv/bin/python -m pytest -v

# Full LangSmith experiments (needs LANGSMITH_API_KEY + model + judge):
.venv/bin/python run_eval.py
```

`run_eval.py` seeds the two datasets (idempotent), runs an experiment for each,
and prints links to the LangSmith UI where you can see per-example scores and
full traces. If anything is unconfigured it prints what's missing and exits
cleanly rather than erroring.

## How the LLM-as-judge works here

`CloudRubricJudge` (Anthropic by default, or any OpenAI-compatible endpoint)
returns a structured `{score, reason}` via [`instructor`](https://python.useinstructor.com/),
which the rubric evaluator normalizes to a 0–1 score with a pass/fail threshold —
the same rubric and judge approach as the deep_eval demo.

## Notes

- `.env`, `.venv/`, and caches are gitignored.
- This demo intentionally does **not** reuse the parent suite's `assertions/`
  (an earlier iteration did); it stands alone and mirrors the deep_eval demo's
  two test cases so the two prototypes are directly comparable.
```

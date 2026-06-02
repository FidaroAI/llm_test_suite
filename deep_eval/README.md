# DeepEval demo

> ⚠️ **Prototype.** This is an exploratory spike to evaluate DeepEval against the
> existing promptfoo suite — not production code. It is intentionally minimal,
> not wired into CI, and its structure/API choices may change. Don't depend on it.

A small [DeepEval](https://deepeval.com/) suite that mirrors the kind of testing
in the parent promptfoo project, to compare the two frameworks. It shows:

1. **A simple factual test** — ask the LLM "What is the capital of France?" and
   assert the answer contains `Paris` (deterministic, no judge).
2. **A rubric test judged by another LLM** — score an open-ended response
   against a rubric using GEval (LLM-as-judge).

## promptfoo → DeepEval mapping

| promptfoo                                   | DeepEval here                                   |
|---------------------------------------------|-------------------------------------------------|
| `providers:` makes the model call for you   | you call it yourself in `model_under_test.py`   |
| `icontains:Paris` assertion                 | `ContainsMetric("Paris")` (`metrics.py`)        |
| custom Python asserts in `assertions/`      | custom `BaseMetric` subclass (`ContainsMetric`) |
| `llm-rubric` + Anthropic judge (the TODO)   | `GEval(criteria=..., model=CloudJudge(...))`    |
| `strip_before_triple_newline` transform     | `MUT_STRIP_THINKING` in `model_under_test.py`   |
| `promptfoo eval`                            | `pytest`                                        |

The key conceptual difference: promptfoo owns the provider call, so a test is
just `prompt → assert`. DeepEval separates **generating** the output (the model
under test) from **scoring** it (metrics), so each test has two steps.

## Layout

```
deep_eval/
  deepeval_demo/
    config.py            # env-driven config (falls back to the parent suite's VLLM_*/ANTHROPIC_API_KEY)
    model_under_test.py  # OpenAI-compatible chat call -> the answer we score
    judge.py             # CloudJudge: a DeepEvalBaseLLM (Anthropic or OpenAI-compatible) for GEval
    metrics.py           # ContainsMetric: deterministic, the icontains analog
  tests/
    test_contains_metric.py  # NO LLM — proves the framework runs out of the box
    test_factual.py          # live: model under test + ContainsMetric
    test_rubric.py           # live: GEval rubric, judged by another LLM
```

## Setup

Python ≥ 3.11.

```bash
cd deep_eval
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env          # then fill in endpoint URLs / judge key
```

`config.py` falls back to the parent suite's `../.env` (`VLLM_*`,
`ANTHROPIC_API_KEY`), so an existing top-level `.env` mostly just works.

## Running

```bash
# Everything (deterministic tests pass; live tests skip until configured):
.venv/bin/python -m pytest -v

# Just the no-LLM proof that the framework works:
.venv/bin/python -m pytest tests/test_contains_metric.py -v
```

### Run the live factual test

Point at any reachable OpenAI-compatible endpoint (vLLM, the Fidaro plaintext
gateway, Ollama, …). For the gateway, start it first (the parent suite can't).

```bash
# in deep_eval/.env
MUT_BASE_URL=http://127.0.0.1:8082/v1
MUT_MODEL_ID=Qwen/Qwen3-Next-80B-A3B-Thinking-FP8
MUT_API_KEY=dummy
```

```bash
.venv/bin/python -m pytest tests/test_factual.py -v
```

### Run the live rubric test (LLM-as-judge)

Also needs a judge. Default is Anthropic:

```bash
# in deep_eval/.env
JUDGE_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
JUDGE_MODEL=claude-sonnet-4-6
```

Or **native AWS Bedrock** — the same Claude models as the parent promptfoo suite
(`bedrock:…` providers). Auth is a Bedrock API key (bearer token) or the standard
AWS credential chain:

```bash
JUDGE_PROVIDER=bedrock
AWS_BEARER_TOKEN_BEDROCK=...          # or AWS_ACCESS_KEY_ID/SECRET, or AWS_PROFILE
JUDGE_REGION=us-east-1               # optional; falls back to AWS_REGION, then us-east-1
JUDGE_MODEL=us.anthropic.claude-haiku-4-5-20251001-v1:0
```

> Use `JUDGE_PROVIDER=bedrock` (not `openai`) to judge with a Bedrock-hosted
> Claude. Bedrock's OpenAI-compatible `/openai/v1` endpoint only serves the
> `openai.gpt-oss-*` models and returns `model_not_found` for Claude model IDs.

Or any OpenAI-compatible judge (e.g. OpenAI itself):

```bash
JUDGE_PROVIDER=openai
JUDGE_BASE_URL=https://api.openai.com/v1
JUDGE_API_KEY=sk-...
JUDGE_MODEL=gpt-4o
```

```bash
.venv/bin/python -m pytest tests/test_rubric.py -v -s   # -s prints the judge's reasoning
```

## How the LLM-as-judge works here

GEval normally scores by reading the judge's token **logprobs**, which only
OpenAI-native models expose. For a custom judge (our `CloudJudge`), DeepEval
falls back to asking the judge for a structured `{score, reason}` object.
`CloudJudge` returns a populated pydantic instance via
[`instructor`](https://python.useinstructor.com/), so DeepEval gets reliable
structured output instead of having to parse JSON out of free text.

## Notes

- The live tests **skip with a clear reason** when the endpoint/judge isn't
  configured or reachable, so the suite stays green out of the box. In a real
  CI you'd likely let an unreachable endpoint fail loudly (see `tests/_live.py`).
- `.env`, `.venv/`, and DeepEval's `.deepeval*` cache are gitignored.

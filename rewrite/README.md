# llmeval

A cache-key-centric, **decoupled** LLM evaluation suite. A rebuild of the Fidaro eval
suite away from promptfoo. Self-contained in this directory (intended to move to its own
repo). Design rationale and decisions: **[DESIGN.md](DESIGN.md)**.

```
generate ─► testcases/*.json ─► run ─► [ SQLite store ] ─► grade ─► compare/pickbest/stats ─► report
                                  ▲            │
                                  └─ cache key ┘   (only `run` calls the model under test)
```

Each stage is an independent step. `grade`, `pickbest`, `compare`, and `report` operate
on **cached** outputs — so you can edit assertions, add a config, or re-judge a
head-to-head **without re-running the model**.

## Why it's built this way

- **The cache key is yours.** You choose exactly which of `{model, *params, *extra}`
  define "the system under test" (e.g. key on `model + temperature + backend_version`,
  *ignore* `max_tokens`). Results are stored per `(test, cache_key)`, so a single failing
  test can be re-run alone and nothing is wasted.
- **Generation ≠ running.** Generators emit plain JSON test cases into `testcases/` that
  you can read before any run — no opaque `*_gen.py`.
- **Compare after the fact.** Running just fills a SQLite database; comparison and
  statistics are separate passes over it.

## Install

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[providers,dev]"
# 'providers' pulls in litellm (real LLM calls); 'dev' pulls in pytest.
```

## Quickstart (offline, no API keys — uses the built-in `echo` provider)

```bash
# 1. Generate standardized, inspectable test cases from a CSV
llmeval generate-csv --csv generation_sources/simple_facts.csv --suite simple_facts --out testcases/

# 2. Run a provider (echo just returns the prompt — good for plumbing checks)
llmeval run   --testcases testcases/ --provider configs/echo.json --db demo.sqlite3 --filter suite=simple_facts

# 3. Grade cached outputs (deterministic assertions need no judge)
llmeval grade --testcases testcases/ --provider configs/echo.json --db demo.sqlite3 --filter suite=simple_facts

# 4. Render a report
llmeval report --providers configs/echo.json --db demo.sqlite3 --out report.html
```

Re-run step 2 and you'll see `ran=0 cached=28`: results are reused.

## Real providers

Providers are JSON files (see `configs/`). The `model` is a [litellm](https://docs.litellm.ai)
string, so any provider works — OpenAI, Anthropic, Bedrock, or any OpenAI-compatible
endpoint (the Fidaro plaintext gateway, Venice, a local vLLM). `base_url` supports
`${ENV}` expansion; `api_key_env` names the env var holding the key. Copy `.env.example`
to `.env` and fill in what you use.

```jsonc
// configs/fidaro_dev.json
{
  "name": "fidaro-dev",
  "model": "openai/Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
  "base_url": "${FIDARO_DEV_BASE_URL}",
  "api_key_env": "FIDARO_API_KEY",
  "params": { "temperature": 0.7, "max_tokens": 100000 },
  "extra": { "backend_version": "phala-dev" },
  "cache_key_fields": ["model", "temperature", "backend_version"]  // max_tokens ignored
}
```

Bringing infrastructure up (gateways, sidecars, redeploys) is **out of scope** — point
`base_url` at something already running.

## The three workflows

```bash
# Batch-run one provider, then read results
llmeval run --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3

# Indirect comparison (rate each config, compare ratings)
llmeval grade  --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval grade  --testcases testcases/ --provider configs/fidaro_dev.json  --db runs.sqlite3
llmeval report --providers configs/fidaro_prod.json configs/fidaro_dev.json \
               --baseline fidaro-prod --metrics accuracy --db runs.sqlite3 --out report.html

# Direct comparison (judge picks the best; both orderings to fight position bias)
llmeval pickbest --testcases testcases/ --providers configs/fidaro_prod.json configs/venice.json \
                 --order both --db runs.sqlite3
llmeval report   --providers configs/fidaro_prod.json configs/venice.json \
                 --order both --db runs.sqlite3 --out report.html
```

## Caching modes (`llmeval run --mode`)

| Mode | Behaviour |
|---|---|
| `reuse` (default) | If a usable result exists for `(test, cache_key)`, don't call the model. |
| `target_n` (`--target-n N`) | Ensure up to **N** usable results — for best-of-N statistics. Tops up across runs. |
| `always` | Append one more result every time. |

Failures retry (`--retries`) and, if still failing, are stored as error rows so the run
continues and a later invocation can top them up. **Ctrl-C is safe**: each result is
committed as it completes, so an interrupted run keeps everything computed so far — only
the in-flight test is lost, and the next run tops it up.

**Selecting which tests to run:** `--limit N` runs only N tests; `--randomize` shuffles
first (so `--randomize --limit N` is a random sample); `--seed` fixes the shuffle (default
`0`, always reproducible). `--filter k=v` narrows by metadata (e.g. `--filter suite=simple_facts`).

## Assertion types

- **Deterministic** (no judge): `contains`, `icontains`, `equals`, `regex`,
  `not_contains`, `length` (tokens/words/chars), `refusal`.
- **LLM-graded**: `rubric` (0–1 against a criterion), `g_eval` (chain-of-thought, 1–10
  normalised).
- **Direct comparison**: `pickbest` (separate command; order control `as_is`/`random`/`both`).

Outputs are reasoning-stripped before grading (the `\n\n\n` rule) while the stored raw
output keeps the reasoning. Hand-write tests as JSON too — see `testcases/examples.json`.

## Statistics

`compare` aggregates per config and metric: mean + bootstrap 95% CI, delta vs a baseline,
attempt reducers (`mean`/`max`/`pass_rate`) for best-of-N. Pick-best yields win rates.
More advanced "which config is best" methods are a documented extension point (DESIGN §9).

## Layout

```
llmeval/            the package
  cache_key.py      user-controlled cache key
  models.py         TestCase, AssertionSpec, ProviderConfig
  store.py          SQLite: results (+ full config) / gradings / verdicts
  providers.py      litellm-backed + echo + custom registry
  runner.py         caching policy, retries, graceful failure
  response.py       reasoning-strip transform
  assertions/       deterministic + judge (rubric, g_eval)
  grade.py          apply assertions to cached outputs
  comparison/       pickbest, stats, report
  generation/       CSV -> standardized test cases (separate from running)
  testcases.py      load + metadata-filter test cases
configs/            example provider/judge configs
generation_sources/ raw inputs (e.g. CSV)
testcases/          generated + hand-written test cases (inspectable)
framework_tests/    unit + integration tests for this framework
```

## Testing & linting

```bash
.venv/bin/python -m pytest          # whole suite runs offline (mock provider + fake judges)
.venv/bin/python -m pylint llmeval  # lints the package (10/10)
```

Status and known gaps: **[DESIGN.md §9](DESIGN.md)**.

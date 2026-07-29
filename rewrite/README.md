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

## Plumbing and porcelain

Like git, this suite is split into two layers, and `llmeval` is the **plumbing**:

> Everything is possible with the `llmeval` CLI. Almost nothing is *pleasant* with it.

The CLI is complete and composable but unapologetically low-level — explicit flags, no
guessed defaults, no workflow shortcuts. Friendliness is the job of the **porcelain**:
separate tools built on top of the CLI and the database (task runners, one-command
comparison wrappers, infra bring-up, dashboards, CI entry points). Those live outside the
`llmeval` package.

So the three things porcelain is allowed to depend on are the plumbing's public contracts:
the CLI subcommands, the test-case JSON schema, and the SQLite schema (query it with plain
SQL — that's supported, see [Results](#results)).

Practical upshot: if a workflow here feels like too much typing, that's not a bug in the
plumbing — it's a porcelain that hasn't been written yet. Rationale:
**[DESIGN.md §2](DESIGN.md)**.

The porcelain that exists so far lives in **[reporting/](reporting/README.md)**: a generic
CSV→HTML table viewer (filter any column, show/hide columns, sort, export) and a report
showing everything a single run produced.

```bash
python -m reporting.run_report run_20260729-0451 -o run.html --testcases testcases/
python -m reporting.csv_table anything.csv -o table.html
```

## Install

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[providers,dev]"
# 'providers' pulls in litellm (real LLM calls); 'dev' pulls in pytest.
```

## Quickstart (offline, no API keys — uses the built-in `echo` provider)

```bash
# 1. Generate standardized, inspectable test cases from a CSV
uv run llmeval generate-csv --csv generation_sources/simple_facts.csv --suite simple_facts --out testcases/

# 2. Run a provider (echo just returns the prompt — good for plumbing checks)
uv run --env-file .env llmeval run   --testcases testcases/ --provider configs/echo.json --filter suite=simple_facts

# 3. Grade cached outputs (deterministic assertions need no judge)
uv run llmeval grade --testcases testcases/ --provider configs/echo.json --filter suite=simple_facts

# 4. Render a report
uv run llmeval report --providers configs/echo.json --out report.html
```

Re-run step 2 and you'll see `ran=0 cached=28`: results are reused.

## Generating the standard suites

Beyond ad-hoc CSVs (`generate-csv`), the `generate` command produces the same
suites as the legacy `tests/*_gen.py` generators, writing inspectable
`testcases/<suite>.json`:

| Suite | Source | Assertions |
|---|---|---|
| `simple_facts`, `simple_facts_regressions` | CSV (`generation_sources/`) | `icontains` |
| `agentharm_refusal` | `data/agentharm.json` | one refusal `rubric` (`censorship: true`) |
| `multifaceted` | `data/multifaceted.json` | per-row `rubric` (1–5 anchors embedded) |
| `research_rubrics` | `data/researchrubrics.json` | `rubric` **and** `g_eval` variants per row |
| `stock_prices` | CSV + live Stooq fetch | `stock_price` (within 1%) |

```bash
llmeval generate --suite multifaceted --out testcases/   # one suite
llmeval generate --all --out testcases/                  # all except network suites
llmeval generate --suite stock_prices --out testcases/   # hits the network; needs the 'stocks' extra
```

How many tests each suite emits (and shuffling/stratification) is controlled by a
suite-generation config (default `suite_generation_config.json`, overridable with
`--config` or `SUITE_GENERATION_CONFIG_FILE`). A suite absent from the config is
**off** (emits nothing). Cross-suite `request_type`/`domain` labels are merged
from `data/classifications/<suite>.json` at generation time. Datasets must be
downloaded first (`pnpm dataset`); `--all` skips network suites and quietly skips
any suite whose source isn't present. `stock_prices` bakes the live reference
into each test at generation time, so run it with `--mode always` (or a fresh DB)
to avoid a cached answer masking a freshness miss.

Fidelity note: this is generation-only — per-test weighted thresholds and the
multifaceted 1→5 `rubricPrompt` override from the legacy suite are intentionally
not reproduced (rubrics grade per-assertion via the standard 0–1 template). See
`docs/superpowers/specs/2026-06-25-rewrite-all-suites-generation-design.md`.

## Results

Results are stored in a local sqllite db. Default name is llmeval.sqlite3. You can interrogate that
on the command line to see results.

Every `llmeval run` opens a **run** and stamps each result with its id, which the command
prints when it finishes (`run run_20260729-142530-a3f1: ...`). Add `--note "..."` to record
why you kicked it off. Runs are provenance only — caching and grading still key on the
cache key, and `attempt` keeps counting across runs so best-of-N accumulates as expected.

```sql
-- what runs exist?
SELECT id, provider_name, notes, started_at, finished_at FROM runs ORDER BY id DESC;

-- everything one run produced (finished_at NULL => it crashed or is still going)
SELECT test_id, attempt, error, latency_ms FROM results WHERE run_id = 'run_...';
```

There is **no migration path**: the store checks `PRAGMA user_version` on open and refuses
a database written by an older build, telling you to delete it.

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

**Concurrency:** `--concurrency N` runs N test cases in parallel (default `5`; `1` =
sequential). Test cases are independent `(test, cache_key)` units, so they fan out across a
thread pool while the shared SQLite store serialises writes. Ctrl-C stays safe at any
concurrency — committed results survive and the next run tops up the rest.

## Logging

All output goes through the standard `logging` module to **stderr**. Verbosity is
`--log-level {debug,info,warning,error,critical}` on any subcommand, or `LLMEVAL_LOG_LEVEL`
in the environment (default `info`). `debug` adds the cache key and cache/to-run counts per
test case. litellm and the other chatty third-party loggers are pinned to `WARNING`.

Parallel runs would otherwise interleave line by line — two lines of one test case, one of
another, the rest of the first — so each test case's records are **buffered and flushed as
one contiguous block** when it finishes:

```
INFO llmeval.runner run run_20260729-045138-5d8c: 6 test case(s), provider=slow, mode=reuse, concurrency=4
INFO llmeval.runner facts-03: Question number 3 about geography?
WARNING llmeval.runner facts-03: attempt 1/2 failed (RuntimeError: connection reset); retrying
INFO llmeval.runner facts-03: ok in 863ms -> An answer to «Question number 3 ab» with a second line
INFO llmeval.runner run run_20260729-045138-5d8c: 5/6 test case(s) complete
```

Two consequences worth knowing:

* A block only appears when its test case **finishes**, so a slow model call is a quiet
  gap. Sequential runs (`--concurrency 1`) skip buffering entirely and stream live, and
  `LLMEVAL_LOG_DEFER=0` forces live streaming at any concurrency — use it to watch a call
  you suspect is hung.
* **Timestamps run backwards between blocks.** A record is stamped when it is created, not
  when it is printed, so a test case that started early and finished late prints an early
  timestamp after a later block. The times are true event times; the log just isn't a
  clock you can read top to bottom.

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
  store.py          SQLite: runs / results (+ full config) / gradings / verdicts
  providers.py      litellm-backed + echo + custom registry
  runner.py         caching policy, retries, graceful failure, parallel run
  logs.py           logging config + per-thread deferred emission (readable parallel runs)
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
reporting/          porcelain: generic CSV->HTML viewer + per-run report (not in the wheel)
reporting_tests/    tests for the porcelain
```

## Testing & linting

```bash
.venv/bin/python -m pytest          # whole suite runs offline (mock provider + fake judges)
.venv/bin/python -m pylint llmeval  # lints the package (10/10)
```

Status and known gaps: **[DESIGN.md §9](DESIGN.md)**.

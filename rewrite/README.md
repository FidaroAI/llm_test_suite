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
CSV→HTML table viewer (filter any column, show/hide columns, sort, export) that opens what
it renders.

```bash
uv run llmeval report --run-last-n 3 --testcases testcases/ --out results.csv
python -m reporting.csv_table results.csv -o results.html
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

# 4. Emit the result rows, then view them (the second command opens a browser)
uv run llmeval report --testcases testcases/ --provider configs/echo.json --out results.csv
uv run python -m reporting.csv_table results.csv -o report.html
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
why you kicked it off. Caching and grading still key on the **cache key**, which the run
owns: `results` has no cache-key column of its own and reaches it through `run_id`.

**One row per attempt.** Every inference call is stored, successful or not, with its
`latency_ms` either way. A test that failed twice and answered on the third try is three
rows — so "how flaky was this provider?" and "what did the retries cost?" are queries, not
guesses. `attempt` is 0-based **within a run**; pool across runs by cache key for the
best-of-N view.

**Each attempt records the prompt it sent**, in `messages_json`. `raw_json` holds the
provider's *response*, so without this the question would live only in `testcases/` — which
is regenerated, meaning a result could outlive any record of what produced it. Storing it
makes a result readable on its own, and makes "what did we send when this timed out?"
answerable from the error row.

**Non-standard response data** goes in `provider_specific_output`, verbatim and under its
vendor key — see [Provider-specific output](#provider-specific-output).

```sql
-- what runs exist?
SELECT id, provider_name, notes, started_at, finished_at FROM runs ORDER BY id DESC;

-- everything one run produced (finished_at NULL => it crashed or is still going)
SELECT test_id, attempt, error, latency_ms FROM results WHERE run_id = 'run_...';

-- what was actually asked, for the attempts that failed
SELECT test_id, attempt, error,
       json_extract(messages_json, '$[#-1].content') AS last_turn
FROM results WHERE run_id = 'run_...' AND error IS NOT NULL;

-- which tests needed retries, and how much time went on the failures?
SELECT test_id,
       COUNT(*)                                   AS attempts,
       SUM(error IS NOT NULL)                     AS failed_attempts,
       SUM(CASE WHEN error IS NOT NULL THEN latency_ms ELSE 0 END) AS wasted_ms
FROM results WHERE run_id = 'run_...'
GROUP BY test_id HAVING attempts > 1;

-- the best-of-N pool for one config, oldest first, across every run
SELECT r.test_id, r.run_id, r.attempt, r.error, r.latency_ms
FROM results r JOIN runs ru ON ru.id = r.run_id
WHERE ru.cache_key_hash = '<hash>' ORDER BY r.test_id, r.id;

-- streamed attempts that timed out, and how much answer they still produced.
-- Empty before streaming existed: the request died before any body was read.
SELECT test_id, LENGTH(output) AS chars, error, SUBSTR(output, -120) AS tail
FROM results WHERE error LIKE 'stream timeout%' ORDER BY chars DESC;
```

There is **no migration path**: the store checks `PRAGMA user_version` on open and refuses
a database written by an older build, telling you to delete it.

## Selecting which runs to read

`grade` and `report` both read stored results, so both take the same four flags. They fall
into three groups and the groups **cannot be combined** — `--run-last-n 3 --run-after X` has
no single obvious reading, so it's an error rather than a guess.

| Flag | Meaning |
|---|---|
| `--run-id a,b,c` | exactly these runs; ids or unambiguous prefixes, comma-separated or repeated |
| `--run-after V` / `--run-before V` | an inclusive window; `V` is `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM`, either with an explicit `+HH:MM`/`Z` offset, **or a run id** |
| `--run-last-n N` | the N most recent runs |

Omit them all for every run. A bare timestamp is **UTC** — that's what `runs.started_at`
holds and what the run id embeds, so `--run-after 2026-07-29` lines up with the ids you see
in the log. `--run-after run_20260729-0451` means "that run and everything after it".

Run selection composes with provider selection, and the provider narrows **first**:
`--provider configs/fidaro_prod.json --run-last-n 3` is the last three runs *of that
provider*, not the last three runs overall filtered down to it.

Naming a run that doesn't exist, or one prefix that matches two runs, is an error (exit 2). A
window that legitimately matches no runs is not — you get an empty table, because "the last
3 runs of a provider that has had none" is an answer.

## Grading

Grading is per **result**, not per test: `gradings` is unique on
`(result_id, assertion_key)`, so every attempt carries its own score and a re-run adds a row
to grade rather than superseding one. `grade` fills in every `(result, assertion)` pair in
the selected runs that doesn't have a grading yet, and **skips attempts that errored** —
there is no output to assert against, and the error row is itself the finding.

```bash
llmeval grade --testcases testcases/ --provider configs/fidaro_prod.json --run-last-n 1
```

## Reporting

`report` writes the selected result rows as **CSV**. It renders nothing: turning a table
into a page is a workflow, so that's [reporting/](reporting/README.md)'s job.

```bash
llmeval report --run-last-n 3 --provider configs/fidaro_prod.json \
               --testcases testcases/ --db runs.sqlite3 --out results.csv
python -m reporting.csv_table results.csv -o results.html   # opens in a browser
```

**One row per (result × assertion)**, plus **one row per errored result** with the grading
columns empty. Rows are grouped by run in chronological order, and within a test by attempt,
so a test that failed once and answered on the retry reads top to bottom:

```
run1, test x, attempt 0, error=timeout, latency_ms=60001
run1, test x, attempt 1, assertion1, passed=True
run1, test x, attempt 1, assertion2, passed=False
run2, test x, attempt 0, assertion1, passed=True
```

`latency_ms` is filled in for error rows too, which is how "the timeout is too tight" is
distinguished from "the provider is down".

**The prompt is always there.** Two columns carry it, both read off the result rather than
the test-case files, so they need no `--testcases`:

| Column | What it holds |
|---|---|
| `prompt` | the last user turn — the question, as a human reads it |
| `messages` | the whole conversation as sent, JSON. The only place a system prompt or an earlier turn survives |

For an ordinary single-turn test the two say the same thing and you can hide `messages` in
the viewer. For a multi-turn case they don't: `prompt` would be *"Two weeks in spring."* on
its own, which is why the full record is kept next to it. Where a result predates the store
recording prompts, `prompt` falls back to the test case if you passed `--testcases` — but the
stored copy always wins, because `testcases/` is regenerated and only the stored copy is
evidence of what this result was produced from.

`--provider` is repeatable and optional (default: every provider in the database), so one
report can span several configs — `provider` and `cache_key_hash` are columns. `--testcases`
is optional and does two things: it adds the `request_type` and `domain` labels, and it
**selects** — only tests present in those files appear, which is what makes
`--filter suite=simple_facts` work.

The statistics report — bootstrap CIs, deltas against a baseline, pick-best win rates — is
`compare-report`, unchanged otherwise:

```bash
llmeval compare-report --providers configs/fidaro_prod.json configs/venice.json \
                       --baseline fidaro-prod --order both --out report.html
```

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
  "stream": true,
  "params": { "temperature": 0.7, "max_tokens": 100000 },
  "extra": { "backend_version": "phala-dev" },
  "cache_key_fields": ["model", "temperature", "backend_version"]  // max_tokens ignored
}
```

Bringing infrastructure up (gateways, sidecars, redeploys) is **out of scope** — point
`base_url` at something already running.

### Streaming (`"stream": true`)

Consumes the response as SSE and accumulates it client-side instead of waiting for the
server to do it. The stored row is the same either way — same answer, same reasoning,
same token counts — with one difference, which is the whole reason to switch it on:

> **A call that hits its timeout keeps what it already received.** Without streaming the
> request is torn down before any body is read, and the row is an error with no output.
> With it, the partial answer and partial reasoning are on record.

That is what makes "is the model stuck in a repetitive loop?" a testable question: the
call still times out, but now the text it was looping over is in the database.

A timed-out attempt is stored with **both** its output and an `error` saying what
happened and how much arrived (`stream timeout after 60.0s (content: 48213 chars,
reasoning: 1204 chars)`). It is **not retried** — the timeout was your statement of how
long the answer was worth waiting for, and these tests are meant to time out. Being an
error row, it is skipped by `grade` and doesn't count towards `--mode reuse`; read it
with SQL.

The deadline is the ordinary per-call timeout (`--timeout`, or `timeout` on a test case)
— there is no separate streaming one.

Two limits. Streaming is **OpenAI-compatible SSE only**: `"stream": true` on a
non-`openai/` model is refused rather than silently ignored. And for Fidaro it needs an
**orchestrator `/v2`** base URL — the llm-gateway on 8082/8084 serves `/v1`, which uses
older plaintext frames and carries no `fidaro` object.

### Provider-specific output

Response data that has no place in the OpenAI schema is kept verbatim in
`results.provider_specific_output`, under its vendor key. For Fidaro `/v2` that is the
`fidaro` object, which today carries the chat title:

```json
{"fidaro": {"title": "Capital of France"}}
```

Captured on both the streaming and non-streaming paths, so turning streaming on doesn't
change what a test case can see. Nothing parses it — a new key needs no code change:

```sql
SELECT test_id, json_extract(provider_specific_output, '$.fidaro.title') AS title
FROM results WHERE provider_specific_output IS NOT NULL;
```

## The three workflows

```bash
# Batch-run one provider, then read what actually happened
llmeval run    --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval grade  --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval report --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3 \
               --run-last-n 1 --out results.csv
python -m reporting.csv_table results.csv -o results.html

# Indirect comparison (rate each config, compare ratings)
llmeval grade --testcases testcases/ --provider configs/fidaro_prod.json --db runs.sqlite3
llmeval grade --testcases testcases/ --provider configs/fidaro_dev.json  --db runs.sqlite3
llmeval compare-report --providers configs/fidaro_prod.json configs/fidaro_dev.json \
               --baseline fidaro-prod --metrics accuracy --db runs.sqlite3 --out report.html

# Direct comparison (judge picks the best; both orderings to fight position bias)
llmeval pickbest --testcases testcases/ --providers configs/fidaro_prod.json configs/venice.json \
                 --order both --db runs.sqlite3
llmeval compare-report --providers configs/fidaro_prod.json configs/venice.json \
                 --order both --db runs.sqlite3 --out report.html
```

## Caching modes (`llmeval run --mode`)

| Mode | Behaviour |
|---|---|
| `reuse` (default) | If a usable result exists for `(test, cache_key)`, don't call the model. |
| `target_n` (`--target-n N`) | Ensure up to **N** usable results — for best-of-N statistics. Tops up across runs. |
| `always` | Append one more result every time. |

Failures retry (`--retries`, default 2, so 3 attempts) and **every attempt is stored** —
including the ones that failed, and including their latency. A test case that exhausts its
retries just leaves error rows and the run carries on; a later invocation tops it up,
because only successful results count towards the mode's target. **Ctrl-C is safe**: each
attempt is committed before the next is made, so an interrupted run keeps everything
computed so far — only the in-flight call is lost.

The closing summary counts attempts and test cases separately, which matters once retries
are visible:

```
run run_20260729-142530-a3f1 finished: 12 test(s); ran=14 cached=0 errors=2 failed=0
```

`ran` is provider calls (= rows written), `errors` is calls that raised, `failed` is test
cases that gave up. `errors=2 failed=0` above is a run that retried its way through a flaky
provider — not a broken one.

**Timeouts:** `--timeout SECONDS` caps each inference call (default `60`), applied per
attempt rather than per test case. There is no implicit ceiling to fall back on — litellm's
own default is 6000s, which is indistinguishable from a wedged gateway. A single slow test
case can raise its own without changing the default for the suite, via a `"timeout"` field
in its JSON:

```jsonc
{ "id": "research-a1b2c3d4e5", "user": "Write a full equity research note on...",
  "timeout": 600, "assertions": [ ... ] }
```

A timed-out attempt is an error row like any other, with `latency_ms` showing what the wait
cost — which is how you tell "the timeout is too tight" from "the provider is down".

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
WARNING llmeval.runner facts-03: attempt 1/2 failed after 231ms (RuntimeError: connection reset); retrying
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
  runselect.py      which runs a result-reading stage looks at
  resultrows.py     stored results -> report rows + CSV
  providers.py      litellm-backed + echo + custom registry
  runner.py         caching policy, retries, timeouts, graceful failure, parallel run
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
reporting/          porcelain: generic CSV->HTML viewer (not in the wheel)
reporting_tests/    tests for the porcelain
```

## Testing & linting

```bash
.venv/bin/python -m pytest          # whole suite runs offline (mock provider + fake judges)
.venv/bin/python -m pylint llmeval  # lints the package (10/10)
```

Status and known gaps: **[DESIGN.md §9](DESIGN.md)**.

# Rewrite — Design & Decisions

Standalone rebuild of the Fidaro eval suite. Self-contained in this directory so it
can be lifted into its own git repo. Built autonomously from the handwritten brief;
decisions I made without being able to ask are marked **[DECISION]**, and open
questions are left as **TODO** in code and listed in [§9](#9-open-questions--todos).

---

## 1. What this is for

Three primary workflows (from the brief, in priority order):

1. **Compare Fidaro vs a competitor** over many tests — *direct* (judge picks the best)
   and *indirect* (rate each, compare ratings).
2. **Compare Fidaro vs Fidaro-dev** — same two modes.
3. **Batch-run one provider** — just get outputs/results to read, with or without
   assertions.

Secondary: regression tests (explicitly lower value right now).

The defining requirement: **decouple the pipeline stages** so each can run, and re-run,
independently — and put a **user-controlled cache key** at the centre so expensive LLM
calls are reused exactly when the user wants.

---

## 2. The big decisions

### [DECISION] No DSL, no promptfoo, no pytest-based eval runner

The brief says avoid declarative DSLs (promptfoo) and is wary of pytest-based frameworks
(deep_eval/inspect) for *this* job — they fit regression testing, not the
compare-and-analyse workflow. I take this at face value: the eval **runner is a plain
data-driven Python pipeline**, not a framework's test-collection mechanism.

> pytest *is* used — but only to unit-test this framework's own code. That's orthogonal
> to how LLM evals are executed.

### [DECISION] Use litellm as the provider layer, build the rest

The one requirement a library clearly nails is "talk to any provider, extensibly":
**litellm** speaks OpenAI, Anthropic, Bedrock, and any OpenAI-compatible endpoint
(covers the Fidaro plaintext gateway and Venice), and lets us register custom providers.
It's a library, not a DSL/framework, so it doesn't impose structure. Everything else —
the cache, store, grading, comparison, stats, reports — is a focused custom core.

**Why not bend a framework?** The central feature ("the cache key is God": cache keyed
on a *user-chosen subset* of config, results in a queryable DB, **re-grade and compare
without re-running**, N-config comparison, best-of-N statistics) is not first-class in
promptfoo, deep_eval, or inspect_ai. Bending any of them there is more work than a small
core. The brief explicitly permits "none at all." We borrow proven *ideas* (epochs +
reducers from inspect; G-Eval from deep_eval) without the dependency.

#### Exception: the streaming path is hand-rolled

litellm remains the provider layer for non-streaming calls and for the judge. Streaming
(`"stream": true`, see [§5.1](#51-streaming-and-partial-results)) goes through our own
SSE reader over `httpx`. Three measured reasons, probed against a local `/v2` mimic on
litellm 1.94.0:

* Its streaming path **discards the server's `usage`** and substitutes a local tokenizer
  estimate — the server sent 3/4/7, litellm reported 8/2/10. A streamed row and a
  non-streamed row would then disagree about the same response.
* `stream_chunk_builder` **drops non-standard top-level keys**, so the `fidaro` envelope
  does not survive aggregation.
* A read timeout **ends iteration silently, with no exception**, making a truncated
  stream indistinguishable from a complete one.

The third is disqualifying: telling those two apart is the entire purpose of streaming
here. Roughly 120 lines against a format we control, with a reference implementation to
match, beats losing the signal.

### [DECISION] Generation is fully separate from running

- **Generation code** lives in `llmeval/generation/` + `generation_sources/`.
- **Generated test cases** are written as plain JSON to `testcases/` — a *separate
  directory*, inspectable before any run. This directly fixes the "opaque `*_gen.py`"
  complaint: you can read exactly what will run.
- Running consumes `testcases/*.json` and never imports generation code.
- Hand-written tests are supported: just author a JSON file (or use the small builder
  helper). Tests need not come from CSV/JSON datasets.

### [DECISION] `llmeval` is plumbing, not porcelain

We borrow git's split. The `llmeval` CLI (together with the SQLite store) is the
**plumbing** of the test suite: complete, composable, and deliberately *not* friendly.
Everything the suite can do must be reachable through it — but reaching it may well take
six explicit flags and a `--db` path. Ergonomics are somebody else's job.

The **porcelain** is a separate layer of tools built *on top of* the plumbing: task
runners, wrappers that encode a whole comparison workflow, infra bring-up, dashboards, a
CI entry point. These live outside the `llmeval` package.

Three contracts make up the plumbing surface, and porcelain may depend on all of them:

1. **The CLI subcommands** — `generate`, `generate-csv`, `run`, `grade`, `pickbest`,
   `report`. (The aggregation step described as `compare` above and in the README is
   currently library-only — `comparison/stats.py`, reached via `report`.)
2. **The test-case JSON schema** in `testcases/` (see `llmeval/models.py`).
3. **The SQLite schema** — `runs` / `results` / `gradings` / `verdicts`. Reading it with
   plain SQL is a supported way to consume results, not a hack.

The library entry points the subcommands wrap are equally fair game for porcelain written
in Python; the CLI is the boundary for everything else.

**Why bother naming the layers?** Two reasons.

*It keeps the plumbing honest.* The whole point of this rebuild is that each stage is
independently re-runnable and the cache key is user-controlled ([§4](#4-the-cache-key-god)).
Both properties survive only if the CLI stays explicit. The moment we add a friendly
`llmeval compare-prod-vs-dev` that picks a DB, a judge, and a cache-key policy for you,
the plumbing has grown opinions and the guarantees get fuzzy. Convenience requests are a
signal to write porcelain, not to grow a flag.

*It gives the homeless work a home.* [§8](#8-out-of-scope-per-brief) declares infra
bring-up out of scope, and the README tells you to point `base_url` at something already
running. That work doesn't vanish — it's porcelain. Same for "run the standard nightly
comparison and mail me the report".

Consequences worth knowing:

- **Don't add convenience to the plumbing.** Prefer another flag over another subcommand,
  and prefer porcelain over another flag. A subcommand earns its place by exposing a
  *capability*, not a *workflow*.
- **Plumbing output is consumed by programs.** Human-readable is fine; human-*only* is not.
  Anything a person reads off stdout should also be gettable from the store.
- **The store's stability is a real constraint on porcelain.** There is currently no
  migration path: `store.py` checks `PRAGMA user_version` on open and refuses a database
  written by an older build. Porcelain that caches DB paths or hoards historical results
  has to cope with "delete it and re-run", so schema changes are not free.

---

## 3. The pipeline

```
generation/  ──►  testcases/*.json          (data: input + assertions + metadata)
                        │
                        ▼
   run(provider, testcases, policy)  ──►  STORE.runs        (one row per invocation)
                        │             └─►  STORE.results     (cache LLM outputs by cache key)
                        │
                        ▼
   grade(testcases, where=cache_key) ──►  STORE.gradings    (assertion scores; re-runnable)
                        │
                        ▼
   compare / pickbest / stats        ──►  STORE.verdicts    (head-to-head; re-runnable)
                        │
                        ▼
   report(cache_keys)                ──►  reports/*.html
```

Each arrow is an independent CLI subcommand and a library call. Crucially, **grade**,
**pickbest**, **compare**, and **report** read cached outputs — they never call the model
under test again. Editing an assertion or adding a new config to a comparison re-runs
only what's missing.

Two axes run through `results`, and keeping them distinct is what makes the rest work:

* **identity** — *what* was under test. This is the cache key, and it is stored once, on
  `runs`. Caching, grading and comparison all key on it; `results` reaches it by joining
  `run_id`. (`ResultRow` still exposes `cache_key_hash`, so readers don't notice.)
* **provenance** — *which sitting* produced a row, i.e. `run_id` itself. It never affects
  caching.

### [DECISION] The cache key lives only on `runs`

A run calls exactly one provider, so every result it produces shares that provider's cache
key. Copying the key onto each result duplicated it thousands of times and, worse, made a
disagreement between the two representable with nothing to arbitrate it. Normalising means
identity lookups pay a join — irrelevant at eval scale, and `results(test_id)` is indexed
for it.

`attempt` is scoped to `(run_id, test_id)` and restarts at 0 for each test in each run: it
answers *"which try within this run?"*. The best-of-N pool — every attempt for a cache key,
however many sittings it took — is an identity question, so it groups by the run's
`cache_key_hash` and orders by row id (`attempt` can't order a cross-run pool). A run that
crashes is left with `finished_at` NULL rather than being marked complete.

---

## 4. The cache key ("God")

A provider's full identity is a namespace:

```
namespace = { "model": <model>, "stream": <bool>, **params, **extra }
```

`params` are call params (temperature, max_tokens, top_p…). `extra` is arbitrary
user metadata that still affects the system under test but isn't an API param — e.g.
`{"backend_version": "phala-2026-06-01", "system_prompt_id": "v3"}`. `model` and `stream`
are structural fields of the config rather than free-form parameters, so they sit in the
namespace directly and are reserved against being shadowed from `params`/`extra`.

The **cache key** is computed from a user-selected subset:

```python
ProviderConfig(
    name="fidaro-dev",
    model="openai/Qwen3-...",
    params={"temperature": 0.7, "max_tokens": 100000},
    extra={"backend_version": "phala-2026-06-01"},
    cache_key_fields=["model", "temperature", "backend_version"],  # max_tokens IGNORED
)
```

- `cache_key_fields=None` ⇒ use the whole namespace.
- Listed fields are pulled from the merged namespace, canonicalised (sorted JSON), and
  hashed (sha256, short prefix stored alongside the full JSON).
- The store keeps both the **hash** (join key) and the **full key JSON** (for grouping
  and human-readable reports). Two configs that differ only in an ignored field collide
  on purpose — that's the point.

This gives the requested behaviours for free:

- *"if cached, reuse"* — look up `(test_id, cache_key)`; ≥1 row ⇒ skip.
- *"keep up to N results"* — `reuse` with `repeat=N` tops up to N rows for best-of-N.
- *"rerun one failing test until it passes"* — caching is per `(test_id, cache_key)`, so
  reruns touch only that pair; nothing else is wasted or repeated.

---

## 5. Running, retries, timeouts, graceful failure (`RunPolicy`)

```python
RunPolicy(mode="reuse" | "always", repeat=1, retries=2,
          concurrency=1, timeout=60.0)
```

`mode` and `repeat` are orthogonal: the mode says whether stored results may be reused, and
`repeat` says how many results a test case should have.

- `reuse`: ensure `repeat` successful results exist for `(test, key)` — run `repeat − existing`.
- `always`: append `repeat` more, whatever is stored.

### [DECISION] `repeat` is a count, not a third mode

`target_n` was originally a *mode* alongside `reuse`/`always`, with its N in a separate
`--target-n` flag that only meant anything for that one mode. That conflated two questions:
"may I reuse the cache?" and "how many results do I want?". The `reuse` branch was
`existing >= 1 ? 0 : 1` and the `target_n` branch was `max(0, N − existing)` — the same
arithmetic, one of them with N hardcoded to 1. Making the count a first-class `repeat`
collapses the two branches into one, deletes a mode, and gives `always` a count it never had
(N fresh samples of the same prompt, which is what a non-determinism or repetition check
wants). `repeat=1` reproduces the old two-mode behaviour exactly, so the default is unchanged.
The cost is a breaking CLI change: `--mode target_n --target-n N` is now `--repeat N`.

Per attempt: call the provider with a timeout; retry up to `retries` on any exception
(`BaseException`, i.e. Ctrl-C, is deliberately not caught). A test case that exhausts its
retries yields **error results**, not a crash, and the run continues.

### [DECISION] Every attempt is a row, retries included

Storing only the final outcome made retries free to ignore and impossible to audit: a test
answered on the third call looked identical to one answered on the first, and the time spent
on the two failures was recorded nowhere. So each attempt is written before the next is
made, with its own `latency_ms` whether it succeeded or raised. The costs: rows are no
longer 1:1 with usable results (grading skips error rows, `count_results(success_only=True)`
drives the caching arithmetic), and a persistently failing test now writes `retries + 1`
rows per sitting instead of one. Both are worth it — flakiness and latency-under-failure are
things we actively need to measure.

`RunSummary` therefore counts two different things: `ran`/`errors` are **attempts**, while
`failed` is **test cases** that gave up. `errors > 0, failed == 0` is a healthy run that
retried through a flaky provider.

### [DECISION] Inference calls are always given a timeout, defaulting to 60s

litellm's default request timeout is 6000s, so before this a hung gateway looked exactly
like a slow answer and could stall a run indefinitely. The runner now always passes one.

The override lives on the **test case** (`"timeout"` in its JSON), not the provider,
because slowness is a property of the *task* — a deep-research prompt needs longer than a
one-line factual question, whatever model answers it. It also cannot live in a provider's
`params`, because those feed the cache key: a timeout there would make "same model, longer
ceiling" a different system under test, silently invalidating cached results. The
per-call timeout is applied after `params` for exactly this reason, so it wins over a
stale one parked in a config.

### 5.1 Streaming and partial results

A provider config may set `"stream": true`. The response is then read as OpenAI-compatible
SSE and accumulated client-side, exactly the way the Fidaro orchestrator accumulates it
server-side for `stream:false` (`openai_v2/aggregation.py`): content deltas concatenated,
reasoning deltas concatenated, `usage` from the terminal chunk, the `fidaro` object merged
across every chunk that carries one. Same stored row either way.

**Why bother, if the row is the same?** Because of what happens when the row *isn't*
finished. Without streaming, a call that exceeds its timeout is torn down before any body
is read: the stored result is an error with no output, and there is nothing to look at.
With streaming, the text that had already arrived is on record.

That converts a whole class of test case from unrunnable to routine. "Is the model stuck
in a repetitive loop?" times out by construction — the loop never terminates — so the
evidence needed to answer it is exactly the evidence a non-streaming timeout destroys.

Consequences, each a deliberate choice:

- A timed-out attempt is stored with **both** `output` and `error`. `Completion.error`
  exists for this: "incomplete but usable" is a third outcome, and the runner otherwise
  models failure as the *absence* of a completion.
- Such an attempt is **not retried** (`Attempt.retryable`). The ceiling was the caller's
  statement of how long the answer was worth waiting for; paying it three times over to
  learn the same thing is waste, and these tests are meant to time out every time. An
  attempt that *raised* still retries — an exception may be transient, a timeout is not.
- It stays an **error row**, so `grade` skips it and `count_results(success_only=True)`
  ignores it. Partials are captured for inspection now; an assertion that grades them is
  separate work, and needs grading to stop skipping error rows.
- `stream` joins the cache-key namespace but is usually left unselected, so switching it
  on doesn't invalidate existing results — consistent with the claim that the aggregated
  stream and the non-streamed response are the same data.
- Streaming is OpenAI-compatible SSE only, and is *refused* rather than ignored for other
  backends: a config that asks to stream and silently doesn't looks identical in the store
  right up until a timeout throws the partial away.

### [DECISION] Non-standard response data is stored verbatim, not parsed

`/v2` namespaces everything the OpenAI schema has no home for under one top-level `fidaro`
object (today: the chat title). It lands whole in `results.provider_specific_output` as
`{"fidaro": {...}}`, on both the streaming and non-streaming paths.

Enveloped rather than flattened so a second vendor costs no schema change, and unparsed
because the suite has no opinion on what a backend puts there — a test case that wants to
assert on a key reads the JSON. Capture is an allowlist of top-level keys
(`PROVIDER_SPECIFIC_KEYS`), not "everything unrecognised", because litellm decorates its
own responses with several extra top-level keys that would otherwise be swept in.

A timeout is only real if nothing underneath quietly retries it, so the provider also
switches **off** the retry layer in litellm's OpenAI client (`max_retries=0`). That layer
defaults to two retries *with backoff*: measured against a server that never answers, a
2s ceiling actually took ~7.5s, and none of the extra calls reached the store. Retry
policy belongs to the runner, which owns `--retries` and records every attempt. A config
can opt back in by setting `max_retries` in `params`.

### [DECISION] Parallel logs are deferred, not interleaved

Running fans test cases across a thread pool, so per-test log records from different
workers arrive at the handler interleaved and the output becomes unreadable — you cannot
tell which "retrying" belongs to which prompt. Rather than serialise on a lock per line
(which orders records but still shreds each test case across the output) or push logs into
the store (which would make reading them a query), each test case's records are **buffered
in its own thread and flushed as one contiguous block** when it finishes. Interleaving then
happens between test cases rather than within them.

The mechanism is a handler wrapper with thread-local buffers, in `llmeval/logs.py`. Costs,
both documented there and in the README: a block appears only when its test case finishes
(so the sequential path deliberately doesn't buffer, and `LLMEVAL_LOG_DEFER=0` opts out),
and timestamps run backwards between blocks because a record is stamped at creation rather
than emission.

---

## 6. Assertions / evaluation types

All implement `grade(output, context) -> AssertionResult{passed, score in [0,1], reason}`.
A `transform` is applied to the output **before grading only** (default
`strip_reasoning`, the `\n\n\n` rule from the old suite); the stored raw output keeps the
reasoning.

- **Deterministic**: `contains`, `icontains`, `equals`, `regex`, `not_contains`,
  `length` (tokens/words/chars min/max), `refusal` (the old regex sweep).
- **Rubric** (LLM judge): a criterion graded 0–1 (optionally weighted, with a `metric`
  axis for grouping).
- **G-Eval** (LLM judge): chain-of-thought scoring against criteria.
- **Pick-best** (comparison-level): judge sees N configs' answers for one test and names
  the winner. **Order control** to fight position bias — `as_is`, `fixed`, `random(seed)`,
  or `both` (run both orderings; record agreement). Stored as a verdict, re-runnable
  against cached outputs.

Judge calls go through litellm too, so the judge is any provider (default: Bedrock Haiku,
temperature 0, matching the old suite). The judge is always separate from the model under
test.

---

## 7. Comparison & statistics

Operate entirely over the store, after the fact:

- **Indirect comparison**: aggregate gradings per `(cache_key, metric)` across test cases
  and across the N attempts (reducers: `mean`, `max`, `pass_rate`, `majority`). Report
  per-config means, deltas vs a baseline config, and a **bootstrap 95% CI** (stdlib, no
  scipy). Best-of-N = `max` reducer over attempts.
- **Direct comparison**: pick-best **win rates** per config, plus an "undecided" bucket
  for ties/missing outputs.
- "Which config is best" beyond means (Bradley–Terry / Elo from pairwise verdicts,
  significance tests) is **TODO** — a clear extension point is left in `stats.py`.

Reports are standalone HTML (Jinja2): a summary matrix (configs × metrics, with CIs),
pick-best win rates, and a per-test drill-down with the raw answers.

---

## 8. Out of scope (per brief)

Docker/infra bring-up (gateways, sidecars, Phala redeploy) is **not** part of test runs.
A provider config just points at an already-running `base_url`. Bringing infra up is a
separate concern the user owns.

---

## 9. Open questions & TODOs

Left as choices + `TODO` markers in code; safe to revisit:

1. **Advanced "best config" stats** — only mean + bootstrap CI + win-rate shipped.
   Bradley–Terry/Elo/significance is stubbed in `stats.py`. *(Chose simple, correct,
   dependency-free over fancy.)*
2. **HF dataset generation transforms** — the CSV transform and hand-written path are
   implemented end-to-end; dataset (multifaceted/research_rubrics/agentharm) transforms
   are specified via the `Source` interface but not all ported. *(Chose to prove the
   contract on CSV first.)*
3. **Multi-turn / long-context inputs** — the test-case schema supports a full message
   list; generation examples are single-turn.
4. **Real-provider integration tests** need live creds, so only mock-based tests run
   offline. litellm wiring is covered by a fake-completion unit test.
5. **Report richness** — one comparison report view; richer slicing is easy to add.
6. **g-eval scoring scale** — implemented as a 1–10 judge score normalised to 0–1 with a
   reasoning step, matching deep_eval's shape. Exact prompt is a reasonable default, not
   tuned.

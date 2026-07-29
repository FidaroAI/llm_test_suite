# Fidaro Model Eval Suite — Requirements

A **framework-neutral** specification of what this suite does, written so the system
could be rebuilt on a different foundation than its current one (promptfoo). It
describes capabilities and contracts, not the wiring of any particular tool.

> Scope note: this document covers the production suite only. The `deep_eval/` and
> `langsmith_demo/` prototype directories are out of scope. Load testing and
> competitor *web-app* (browser-driven) comparison are product goals but are **not yet
> built**, so they appear only under [Future scope](#11-future-scope), not as
> requirements to reproduce.

---

## 1. Purpose & goals

Fidaro is an LLM chatbot (like ChatGPT/Perplexity/Gemini) reached, for testing, through
a makeshift OpenAI-compatible API called the **plaintext gateway**. Output quality is
governed by Fidaro's *model configuration*: choice of model, model parameters, and
system prompt.

The suite exists to let a Fidaro developer:

- **Detect regressions** in response quality across a broad range of prompts when a new
  configuration is pushed.
- **Iterate on model configuration** — change the model, its parameters, or the system
  prompt and measure the quality impact against a baseline.
- **Compare Fidaro against competitors** (currently via their APIs; e.g. Venice).

Everything is expressed against the gateway, so the suite must keep working unchanged if
the gateway later grows smarter (multi-model routing, pre/post-processing, caching, more
tools).

---

## 2. Domain model

The vocabulary below is used throughout. A rebuild must provide each concept; the
*mechanism* is free to change.

| Concept | Definition |
|---|---|
| **Provider** | A model plus *how to invoke it*. Two **kinds**: **gateway** (a locally-run, OpenAI-compatible plaintext gateway fronting a remote vLLM) and **api** (a direct external API, e.g. a competitor). |
| **Test case** | A prompt (or chat) plus a list of assertions, metadata, and per-test options. The atomic unit of evaluation. |
| **Suite** | A named group of test cases emitted by one generator. A *local* concept (a metadata tag), not the eval framework's — used for filtering and per-group reporting. |
| **Assertion** | A single check against a response, yielding `{pass, score, reason}`. Either **deterministic** (local computation) or **LLM-graded** (judged by another model). |
| **Judge / grader** | A separate LLM that scores rubric and head-to-head assertions. Must be distinct from the model under test. |
| **Run (eval)** | One execution of selected test cases against the enabled providers, producing a results document. |
| **Baseline** | A frozen, provider-filtered snapshot of a prior run, with provenance, used as a fixed comparison point. |
| **Comparison** | A diff across providers — either N providers from one run, or a live run vs a frozen baseline — rendered as a report. |
| **Classification** | Two orthogonal labels on every test (`request_type`, `domain`) for slicing suites consistently. |

---

## 3. Data contracts (the framework-neutral interface)

The one genuine coupling to the eval framework is the *shape* of test cases and results.
A rebuild must define equivalents; these are the contracts everything else depends on.

### 3.1 Test case

A generator emits a list of test-case records. Each record carries:

- `description` — human-readable label.
- `vars` — template variables for the prompt; always includes the user prompt, may
  include generator-injected values (e.g. a reference stock price).
- `assert[]` — assertion records (see §3.2).
- `metadata` — at minimum `suite`, `request_type`, `domain`; optionally suite-specific
  fields (e.g. `grader`, `native_domain`, `stooq_symbol`), a `censorship` flag, and the
  resolved selection `config` (stamped for reproducibility).
- `threshold` — optional pass threshold for the weighted assertion score.
- `options` — optional per-test settings (e.g. disable caching, per-test provider
  reconfiguration).

### 3.2 Assertion record

- `type` — assertion kind (deterministic family or LLM-graded family; see §7).
- `value` — the expected substring / rubric criterion / etc.
- `weight`, `metric` — optional, for weighted rubric aggregation and per-axis grouping.
- `transform` — optional response transform applied **before grading only** (see §6).
- Additional type-specific config (tolerance, min/max, regex flag, …).

A deterministic assertion implementation receives the (transformed) output plus context
and returns `{pass: bool, score: float in [0,1], reason: str}`.

### 3.3 Results document

A run produces a results document that, per (test × provider), records: the rendered
request/prompt, the raw response **including reasoning**, token usage, and each
assertion's outcome (`pass`/`score`/`reason`), plus a stable run/eval id and a
per-provider **label** used to split results into report columns.

---

## 4. Test cases & generation

### 4.1 Input formats

Three authoring formats, all normalised to the §3.1 record:

1. **CSV lists** — one row per test. Columns map to prompt `vars`; an `__expected`
   column supplies an assertion; `__metadata:<key>` columns inject metadata. (e.g.
   `simple_facts`, `stock_prices`.)
2. **Downloaded datasets** — large JSON datasets pulled from external sources (currently
   Hugging Face) and transformed into tests. (e.g. `multifaceted`, `research_rubrics`,
   `agentharm_refusal`.)
3. **Programmatic generators** — Python files named `*_gen.py`, each discovered
   automatically and owning one suite. A generator reads its source (CSV/dataset/live
   data), applies selection and classification, and returns test records.

### 4.2 Generator requirements

- **Discovery by convention**: any `*_gen.py` is a generator; its suite is named after
  the file stem.
- **One source row → one or more tests**: generators may expand a row into multiple
  variants (e.g. `research_rubrics` emits both an `llm-rubric` and a `g-eval` test from
  the same prompt).
- **No import-time side effects**: generators are imported on every run, so a generator
  that needs live data (e.g. `stock_prices`) must fetch **only** when its suite is
  actually selected — otherwise a source outage would break unrelated suites.
- **Preflight-baked data**: where freshness matters, live reference data is fetched in a
  preflight step and **baked into the test** at generation time (into both `vars`, so
  assertions can read it, and `metadata`, for provenance); the assertion itself stays a
  pure local comparison.

### 4.3 Suite selection config

A single JSON config controls what each generator produces. Per suite:

| Field | Meaning |
|---|---|
| `number_to_generate` | Cap on emitted tests. `null` = all, integer = cap, `0` = skip the suite. |
| `randomize_selection` | Shuffle before capping. |
| `random_seed` | Seed for the shuffle, so selection is reproducible. |
| `max_rubrics` | Cap on rubric criteria per test (datasets often carry many). |
| `stratify` | Optional even sampling across a classification dimension: `{ by: "domain"\|"request_type", per_group: N, groups?: [...] }`. Applied after shuffle; `number_to_generate` still caps the total. |

A file-level `defaults` block applies to unlisted suites; the code default emits nothing
(`number_to_generate: 0`) so ordinary runs don't flood. The active config is selectable
by path (overridable per run), and the **resolved** config is stamped onto each test for
reproducibility.

### 4.4 Selection by metadata

- **Run a single class**: filter at run time on any metadata key, e.g. `suite=...`,
  `request_type=coding`, `domain=finance_business`, or `censorship=false` (to exclude
  the deliberately-harmful suite; untagged tests match).
- **Even sample across a class**: `stratify` (above).

---

## 5. Classification

Every generated test is labelled on two orthogonal, suite-independent axes so suites can
be sliced consistently.

- **`request_type`** — *what* the user is doing. Controlled vocabulary (currently 9
  values: `factual_qa`, `research_synthesis`, `planning`, `coding`, `math_reasoning`,
  `data_analysis`, `creative_writing`, `advice_recommendation`, `text_transformation`).
- **`domain`** — the subject *area*. Controlled vocabulary (currently 11 values:
  `technology_ai`, `science_stem`, `medicine_health`, `finance_business`, `law_policy`,
  `history_society`, `arts_literature`, `consumer_lifestyle`, `philosophy_ethics`,
  `current_events`, `general_other`).

Requirements:

- **Single source of truth** for the two vocabularies (with descriptions).
- **Labels stored separately from raw data**, keyed by a **hash of the prompt text**, so
  the mapping survives dataset re-downloads and reordering and works for datasets with no
  stable row id. Storage includes provenance metadata (model used, timestamp, count).
- **Merge at generation time**: labels are stamped into test metadata; missing labels
  default to `unclassified`. Native dataset fields may be kept for provenance but are not
  the classification.
- **LLM classifier** to populate labels: idempotent (only classifies unlabelled prompts
  unless forced), retries once with the explicit vocabulary if the model invents a label,
  parallelised, and uses the same judge model family as the grader.
- **Censorship tag**: the harmful suite is tagged `censorship: true` so it can be
  excluded from benign runs (other suites omit the key).

---

## 6. Providers & invocation

### 6.1 Provider registry

A **single registry** is the source of truth for every known provider. Each entry
declares: a config **key**, a results **label** (for splitting reports), an **env/parameter
prefix** (for per-run config injection), a **kind** (`gateway` | `api`), and kind-specific
fields — gateway port and upstream-URL key; or API credential name. Capability flags note
which providers support a **system-prompt override** and a **dev redeploy**.

> Design goal: **adding a competitor is one registry row plus one provider definition** —
> nothing else.

### 6.2 Gateway providers

- Routed through a **locally-started gateway container** (OpenAI-compatible) that fronts a
  remote vLLM, on a fixed local port (currently 8082 = prod, 8084 = dev).
- Share a **web-fetch sidecar** for web search, reachable by DNS over a user-defined
  container network; require a web-search API credential when any gateway provider is
  enabled.
- The gateway sets the system prompt, handles tool calls, and relays responses. Only the
  **dev** gateway supports mounting a custom system-prompt file.

### 6.3 API providers

- Call an external API directly — no gateway, sidecar, or redeploy. Require the
  provider's credential in the environment.
- **Custom adapter when response shapes differ**: an external API whose response doesn't
  match the canonical shape needs an adapter that (a) normalises its output to the
  canonical `reasoning + "\n\n\n" + answer` form (see §6.5) so the shared grading
  transform works, and (b) preserves extras (full reasoning, web-search citations,
  finish reason, model id) under result `metadata` for human analysis. (This is why
  Venice has a custom provider: it returns chain-of-thought in a separate field.)

### 6.4 Per-run dynamic configuration

Each provider's model and parameters (gateway: temperature, max tokens; API: e.g. a
web-search flag) are supplied **per run** from the run config's `provider-options`,
injected via the provider's prefixed parameters at load time. A provider may also have a
**static** definition (fixed model/params) for plain single-provider runs. For a dev
redeploy, the dev provider's `model` must equal the model the redeploy serves.

### 6.5 Canonical response shape

Reasoning models emit reasoning followed by the final answer, separated by a
**triple-newline** delimiter (an artifact of the reasoning parser stripping thinking
tags). The gateway emits this shape; API adapters must reproduce it. This delimiter is
the single integration point that lets one transform isolate the answer for any provider.

### 6.6 Per-test reconfiguration (optional)

A test may request provider reconfiguration before it runs (e.g. force a temperature, or
swap to a local model endpoint), to support A/B testing within one run.

---

## 7. Assertions & grading

### 7.1 Response transform before grading

Graders must score **only the final answer**, while the stored response keeps the full
reasoning for human review. Therefore a **strip-before-grading** transform (everything
after the first triple-newline) is attached **per assertion**, not globally — a global
transform would overwrite the canonical output and discard the reasoning. Non-string
outputs pass through unchanged.

### 7.2 Deterministic assertions

Local, no external calls; each returns `{pass, score, reason}`:

- **Substring / regex match** — answer contains (or matches) expected text.
- **Length budget** — response within a min/max in tokens, characters, or words.
- **Reasoning length** — approximate count of reasoning steps within a range (a proxy for
  response speed).
- **Reasoning content** — a value appears in the reasoning blocks (not the answer);
  supports matching the whole reasoning, any block, or a specific block index. Reads
  reasoning from whichever field the provider uses (`reasoning_content` / `thinking` /
  typed content blocks).
- **Tool-call count** — number of tool invocations equals / is within a range, normalised
  across provider response formats (OpenAI `tool_calls`, Anthropic/Bedrock `tool_use`
  blocks, raw JSON).
- **Refusal / censorship detection** — flags refusal language; configurable to treat
  refusal as legitimate for specific tests.
- **Factual freshness (stock prices)** — extracts numeric tokens from the answer and
  compares to a baked-in reference price within a tolerance (currently 1%), handles the
  London pence-vs-pounds quirk (also accept `reference / 100`), and rejects stale
  reference snapshots (older than a max age).

### 7.3 LLM-graded assertions

- **Weighted rubric** — each criterion is an assertion with a `weight` and a `metric`
  axis; the weighted score determines pass against a `threshold`.
- **Chain-of-thought scoring (g-eval style)** — uses the full criteria set and a custom
  grading prompt.
- **Judge configuration** — a separate model (currently AWS Bedrock Claude Haiku at
  temperature 0 for determinism), region-configurable, overridable per run. Must never be
  the model under test.

### 7.4 Head-to-head (select-best)

When a comparison run has ≥2 providers, every test additionally gets a **select-best**
assertion: the judge sees all providers' (reasoning-stripped) answers **and the original
user prompt** (injected via a custom rubric) and names the single best provider.
Requirements:

- Opt-in: only added in comparison mode (a flag), idempotent, and **kept out of** the
  rubric/deterministic tallies.
- **Resilient**: if a test has fewer than two provider outputs (e.g. one provider errored
  under load), the head-to-head **degrades gracefully** to "undecided" for that test
  rather than aborting the whole run. (The current stack patches its eval engine to
  achieve this; a rebuild must guarantee it natively.)

---

## 8. Running & comparison

### 8.1 One eval, split afterwards

All enabled providers run in a **single** eval pass, then results are **split by provider
label** into report columns. Comparison runs always disable caching (server-side changes
from a redeploy are invisible to a cache key, and a single eval has one global cache
setting). Caching is otherwise allowed for cheap reruns, but freshness suites force
no-cache.

### 8.2 Report

Per assertion, grouped by suite:

- A **baseline** column (tagged, named by its config key), one column per other provider,
  a **Δ** column per non-baseline provider (`other − baseline`, rubric scores only), and
  an N-way **best** winner. No status column.
- A **tabular summary** at the top: one row per non-baseline provider giving
  improved / regressed / within-band / new / removed (rubric) and pass/fail transitions
  (deterministic) vs the baseline, plus a best tally. Per-suite summaries use the same
  shape.
- A configurable **tolerance band** classifies rubric deltas (default 0.05).
- **Drift detection**: tests present on only one side are surfaced (the providers ran
  different test sets).
- **Error classification**: infrastructure errors (timeouts, missing judge credential)
  are distinguished from genuinely low scores.

Additional outputs:

- A **raw-response view**: side-by-side unscored responses for eyeballing.
- **Replay**: a reconstructed `curl` command per test for manual re-running.

### 8.3 Output safety

Model output and rubric text are rendered into HTML reports, so the renderer must
escape/sanitise untrusted content and **block unsafe URL schemes** in any rendered
markdown links.

### 8.4 Baselines

- **Freeze**: split a run into one baseline file per provider, each carrying provenance
  (`provider_label`, `frozen_at`, git SHA, eval id, the set of test keys). Filesystem-safe
  naming; overwrite only on explicit force.
- A frozen baseline can be compared against a live run (classic two-file
  baseline-vs-candidate), in addition to the live N-way comparison.

---

## 9. Orchestration

A **single config file** drives a whole comparison, keeping model/gateway configuration
and the test run together.

### 9.1 Responsibilities

1. **Validate** the config: every enabled provider has a registry entry; the named
   baseline is among the enabled providers; required credentials/URLs exist for the
   enabled kinds.
2. **Decide on a dev redeploy** (see §9.3).
3. **Stand up infrastructure** for enabled gateway providers only: a user-defined
   container network, the web-fetch sidecar (health-checked), and one gateway container
   per enabled gateway provider; wait for upstream vLLM and gateway readiness.
4. **Inject per-run provider config** and the chosen suite-selection config.
5. **Run one eval** over all enabled providers, then build the comparison report.

### 9.2 Per-run outputs

Each invocation writes to a fresh `comparisons/<name>/run_<timestamp>/` directory — the
results document, the comparison report, the raw-response view, the resolved suite config,
and (if a redeploy happened) the rendered deployment manifest. Runs never overwrite each
other. State that must persist across runs (the redeploy-decision cache) lives one level
up. These directories are untracked; only the config files are committed.

### 9.3 Dev redeploy decision & safety

Redeploying the dev model server (currently a Phala CVM running vLLM) happens **only**
when all hold: the dev provider is enabled, deployment options are specified, and those
options **changed** since the last deploy (tracked in a cache). Then:

- The target instance id must be on a **whitelist**, the deploy env file and compose
  template must exist, and the action requires **explicit confirmation** unless a
  `--yes`-style flag is given.
- There must be **no path to redeploying prod**. (The deployment CLI is dangerous; guard
  accordingly — ideally a restricted service account.)
- Deployment options are encoded into the server's launch command (bare flag for `true`,
  omit for `false`, `--key value` otherwise), the manifest is rendered, deployed, and the
  endpoint is **polled until ready** or a timeout.

### 9.4 Batch mode

Generate one comparison config per item in a set (e.g. one per candidate system prompt in
a directory) from a template, then run each — for sweeping configurations.

---

## 10. Supporting capabilities

### 10.1 Dataset ingestion

A one-off task downloads each external dataset (paginated from the source API) into local
flat JSON for generators to consume. Some datasets are gated and need a source token. Raw
downloads are never mutated by classification (which lives separately, §5).

### 10.2 Results storage, viewing & CI

- Results are written as JSON (per-run and a `latest`).
- An optional local **viewer UI** (backed by a local DB) displays results; runs populate
  it automatically and external results can be imported.
- **CI** runs a smoke set against prod and uploads the result artifact. A tool pulls CI
  artifacts (by commit / latest / date range) and imports them, deduplicating by eval id.
  *(Aspiration: CI should compare against a baseline and fail on excess regression.)*

---

## 11. Non-functional requirements

- **Reproducibility** — seeded selection, resolved config stamped on every test,
  timestamped reference snapshots, baseline provenance (git SHA + eval id), and a pinned,
  temperature-0 judge model.
- **Concurrency** — tests run in parallel with a configurable cap; classification is
  parallelised.
- **Fail-fast on data** — a preflight that can't fetch complete, fresh reference data
  aborts the run rather than testing against partial/stale data.
- **Resilience** — one provider erroring on one test must not abort the whole eval
  (per-test graceful degradation; see §7.4).
- **Safety** — prod is protected from accidental redeploy (whitelist + confirmation + no
  prod path); the judge is always a different model from the one under test.
- **Cache correctness** — caching speeds reruns by default but is forced off for freshness
  suites and for any comparison/redeploy run.
- **Output safety** — untrusted model output rendered to HTML is escaped and unsafe link
  schemes are blocked.
- **Platform** — ARM dev machines cannot pull the x86 gateway image from the registry and
  must build it locally; the rebuild should not assume x86.

---

## 12. External dependencies & environment

| Dependency | Used for |
|---|---|
| **Docker** | Gateway containers, web-fetch sidecar, optional viewer. |
| **Remote vLLM (Phala CVM)** | The Fidaro model under test, fronted by the gateway. |
| **AWS Bedrock** | The LLM judge (rubric, g-eval, select-best) and the classifier. |
| **Web-search API (Brave)** | Gateway web search (required when a gateway provider runs). |
| **Stooq** | Free, no-key live stock prices for the freshness suite. |
| **Hugging Face** | Source of the downloaded datasets (some gated). |
| **Competitor APIs (Venice)** | Direct-API comparison targets. |
| **Phala CLI** | Dev model-server redeploy. |
| **git / CI artifact CLI** | Provenance and CI result ingestion. |

**Credential / environment surface** (presence required per enabled feature):

- Judge/classifier: Bedrock credentials + region.
- Gateway web search: web-search API key.
- Each enabled API provider: its own credential.
- Dev redeploy: deploy env file + whitelisted instance id.
- Gated datasets: a source token.
- Gateway upstreams: prod/dev vLLM base URLs.

---

## 13. Configuration surface

### 13.1 Comparison run config

```jsonc
{
  // which known providers run this round
  "providers-under-test": { "fidaro-prod": true, "fidaro-dev": false, "venice": true },
  "baseline-provider": "fidaro-prod",          // must be an enabled key

  // per-provider model + params (all fields optional, vary by provider)
  "provider-options": {
    "fidaro-prod": { "model": "...", "temperature": 0.7, "max_tokens": 100000 },
    "venice":      { "model": "...", "web_search": "on" }
  },

  // gateway upstreams (gateway providers only)
  "vllm-prod-url": "https://.../v1",
  "vllm-dev-url":  "https://.../v1",

  // dev gateway only: mount a custom system prompt
  "system-prompt-file": "system_prompts/....md",

  // dev model-server redeploy (only acts when dev enabled AND options changed)
  "phala-dev-instance-id": "<whitelisted-id>",
  "vllm-options": { "model": "...", "reasoning-parser": "...",
                    "enable-auto-tool-choice": true, "tool-call-parser": "..." },

  // optional run-time filters (suite/metadata/provider selection)
  "promptfoo-filters": { "filter-metadata": "suite=research_rubrics" },

  // inline suite-selection config (see §4.3)
  "suite-generation-config": { "...": { } }
}
```

### 13.2 Suite-selection config

```jsonc
{
  "defaults": { "number_to_generate": 0 },
  "<suite>": {
    "number_to_generate": 50,          // null = all, 0 = skip
    "randomize_selection": true,
    "random_seed": 42,
    "max_rubrics": 5,                  // cap criteria per test
    "stratify": { "by": "domain", "per_group": 2, "groups": ["finance_business", "..."] }
  }
}
```

---

## 14. Future scope (product goals, not yet built)

These are wanted but **not** part of reproducing the current system:

- **Load testing** — measure response times under concurrent load and find where they
  degrade.
- **Competitor web-app comparison** — drive competitor *web* experiences (e.g. via
  browser automation) rather than their APIs.
- **System-prompt iteration tooling** — first-class workflow for sweeping system prompts.
- **Model-config iteration tooling** — safer, scripted iteration on model and parameters
  against dev.
- **Smarter CI** — run a fixed set, compare against a baseline, and fail on excess
  regression.

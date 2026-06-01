# Multi-provider comparisons (competitors)

**Status:** approved design, pre-implementation
**Date:** 2026-06-01

## Problem

Today `run_comparison.py` compares exactly two Fidaro providers — prod and dev —
both routed through a locally-started Docker plaintext gateway. We want to drop a
**competitor** (first target: Venice) into the same comparison machinery, so a
competitor can be compared against our prod *or* our dev exactly the way prod and
dev are compared against each other.

A competitor is a different *kind* of provider: a direct external API
(`https://api.venice.ai/api/v1/chat/completions`, bearer `$VENICE_INFERENCE_KEY`,
web search toggled via `venice_parameters.enable_web_search`). It needs **no**
gateway, web-fetch sidecar, or Phala redeploy. So generalizing is two things:
(1) split "providers that need a local gateway" from "providers that are just an
API", and (2) move the whole pipeline from a hardwired prod/dev pair to an
arbitrary set of N providers with one designated baseline.

A reference Venice call:

```
curl https://api.venice.ai/api/v1/chat/completions \
  --header "Authorization: Bearer $VENICE_INFERENCE_KEY" \
  --header "Content-Type: application/json" \
  -d '{"model":"kimi-k2-6","messages":[{"role":"user","content":"..."}],
       "venice_parameters": { "enable_web_search": "on" } }'
```

## Goals

- Run a comparison over an arbitrary enabled subset of known providers, with one
  designated baseline, in a single promptfoo eval.
- Add Venice as the first competitor; make adding the next competitor (e.g.
  Perplexity) a few lines plus one provider YAML.
- Report: baseline column + one column per other provider + one delta column per
  other provider (vs baseline) + an N-way `best` column. Drop the `status` column.
  Replace the prose summary with a per-provider table.
- Preserve today's prod-vs-dev behaviour exactly as the special case
  `{fidaro-prod, fidaro-dev}` with `baseline-provider = fidaro-prod`.

## Non-goals

- Playwright / full-web-experience competitor automation (backlog item; this is
  the API path only).
- Per-pair (all-pairs) deltas or per-pair head-to-heads. Deltas and `best` are
  baseline-vs-others / single N-way winner respectively.
- Changing how suites, classification, grading, or the gateway internals work.

## Decisions (from brainstorming)

1. **Per-provider options block.** A single `provider-options` map keyed by
   provider key replaces `prod-provider-options` / `dev-provider-options`. All
   option fields are **optional** and vary per provider (a competitor may send no
   temperature/max_tokens at all, matching its API).
2. **Deltas:** one column per non-baseline provider, `other − baseline` (rubric
   only). Column headers use the **config provider key** verbatim; the baseline
   column is tagged `(baseline)`.
3. **`best`:** a single N-way `select-best` winner — promptfoo's native behaviour
   (judge sees all N outputs, returns one index, that provider's component
   passes). Column names the winning provider key.
4. **Registry:** a small, extensible in-code registry mapping provider key → how
   it is run. Adding a competitor = one registry row + one YAML.

## Architecture

### 1. Provider registry — `scripts_repo/providers_registry.py` (new)

```python
@dataclass(frozen=True)
class ProviderSpec:
    key: str                      # config key, also the report column name
    label: str                    # promptfoo provider label (filter + result split)
    env_prefix: str               # COMPARISON_PROD / _DEV / _VENICE
    kind: str                     # "gateway" | "api"
    gateway_port: int | None      # gateway only
    vllm_url_key: str | None      # gateway only: config key holding its vLLM url
    supports_redeploy: bool       # only fidaro-dev
    supports_system_prompt: bool  # only fidaro-dev
    api_key_env: str | None       # api only, e.g. "VENICE_INFERENCE_KEY"

REGISTRY = {
  "fidaro-prod": ProviderSpec("fidaro-prod",
      "fidaro_plaintext_gateway_phala_dynamic_prod", "COMPARISON_PROD",
      "gateway", 8082, "vllm-prod-url", False, False, None),
  "fidaro-dev":  ProviderSpec("fidaro-dev",
      "fidaro_plaintext_gateway_phala_dynamic_dev",  "COMPARISON_DEV",
      "gateway", 8084, "vllm-dev-url",  True,  True,  None),
  "venice":      ProviderSpec("venice", "venice_dynamic", "COMPARISON_VENICE",
      "api", None, None, False, False, "VENICE_INFERENCE_KEY"),
}
```

What it does: one lookup table. How it's used: `run_comparison.py` resolves the
enabled keys to specs to decide which gateways to start, which env vars to set,
the `--filter-providers` regex, and the baseline/other labels handed to the
report. What it depends on: nothing (pure data + dataclass).

### 2. Config schema — `comparisons/*.json`

```jsonc
{
  "providers-under-test": { "fidaro-prod": true, "fidaro-dev": false, "venice": true },
  "baseline-provider": "fidaro-prod",          // must be an *enabled* key, else error
  "provider-options": {                         // replaces prod-/dev-provider-options
    "fidaro-prod": { "model": "Qwen/...", "temperature": 0.7, "max_tokens": 100000 },
    "venice":      { "model": "kimi-k2-6", "web_search": "on" }   // all fields optional
  },
  "vllm-prod-url": "...",          // required only if fidaro-prod enabled
  "vllm-dev-url":  "...",          // required only if fidaro-dev enabled
  "vllm-options": { ... },         // consulted only if fidaro-dev enabled
  "phala-dev-instance-id": "...",  // required only with vllm-options (redeploy target)
  "system-prompt-file": "...",     // optional; only mountable on fidaro-dev
  "promptfoo-filters": {},
  "suite-generation-config": { ... }
}
```

**Validation (registry-driven), first failure wins:**

- `providers-under-test` non-empty; at least one enabled. Every key known to the
  registry.
- `baseline-provider` present and one of the **enabled** keys.
- Each enabled provider has a `provider-options` entry (the object may be empty;
  all fields optional). Unknown keys in `provider-options` (not enabled) → error
  to catch typos.
- For each enabled **gateway** provider: its `vllm_url_key` is present/non-empty.
- Redeploy guard (only when `fidaro-dev` enabled **and** `vllm-options` set):
  keep the existing checks — `dev-provider-options.model` (now
  `provider-options["fidaro-dev"].model`) must equal `vllm-options.model` if both
  set; `phala-dev-instance-id` required and whitelisted; `PHALA_DOCKER_COMPOSE_FILE`
  + `.env.phala` present. **Fix the HEAD bug:** the whitelist check must run only
  inside the `has_options` branch (it currently fires unconditionally — the 6
  failing tests at HEAD).
- For each enabled **api** provider: its `api_key_env` must be set in the
  environment (Venice → `VENICE_INFERENCE_KEY`), mirroring the existing
  `BRAVE_API_KEY` check.
- `system-prompt-file`, if given, must exist; warn (don't fail) if no enabled
  provider `supports_system_prompt`.

### 3. Provider YAML — `providers/venice_dynamic.yaml` (new)

```yaml
id: "openai:chat:{{ env.COMPARISON_VENICE_MODEL }}"
label: venice_dynamic
config:
  apiBaseUrl: https://api.venice.ai/api/v1
  apiKey: "{{ env.VENICE_INFERENCE_KEY }}"
  # promptfoo's openai:chat provider only forwards vendor-specific body params
  # placed under config.passthrough — it is spread verbatim into the request
  # body. (config.body does NOT exist for this provider.) Verified against
  # promptfoo 0.121.12 dist (chat getOpenAiBody: `...config.passthrough || {}`).
  passthrough:
    venice_parameters:
      enable_web_search: "{{ env.COMPARISON_VENICE_WEB_SEARCH }}"
```

Mirrors the existing dynamic Fidaro YAMLs: model in the id (keeps promptfoo's
request cache key model-aware), values templated from `COMPARISON_VENICE_*` env
vars set per run. Registered in `promptfooconfig.yaml` `providers:` alongside the
others. `apiBaseUrl` is the Venice base (`.../api/v1`); the provider appends
`/chat/completions`.

**Passthrough resolved (was the top risk).** promptfoo's `openai:chat` provider
forwards arbitrary vendor params only via `config.passthrough` (spread verbatim
into the body); there is no `config.body` for this provider, and unknown
top-level config keys are silently dropped. Confirmed by reading the promptfoo
0.121.12 dist source. No custom JS provider is needed.

### 4. `run_comparison.py` — orchestrate N providers

- Resolve `enabled = [k for k, on in providers-under-test.items() if on]` to specs.
- **Gateways:** start one only for each enabled `kind=="gateway"` spec (so a
  venice-vs-prod run starts a single gateway). Web-fetch sidecar + network still
  shared, started only when ≥1 gateway provider is enabled.
- **Redeploy:** only when `fidaro-dev` is enabled and `vllm-options` present.
- **Env:** for each enabled provider, set `COMPARISON_<PREFIX>_MODEL` (and
  `_TEMPERATURE` / `_MAX_TOKENS` **only when present** in its options). For venice
  also `COMPARISON_VENICE_WEB_SEARCH` (default `"off"` when omitted).
- **Filter:** `--filter-providers` regex = anchored alternation over the enabled
  specs' labels (generalizes today's `both_providers_filter`).
- **Readiness waits:** only for enabled gateway providers' vLLM + gateway URLs.
- **Report:** invoke `compare_runs.py` with the unified result file and
  `--provider <key>=<label>` args — one for the baseline (flagged) and one per
  other provider, ordered baseline-first — so the report shows config keys.
- Keep `--config-path` / `--system-prompt-path` plumbing.

`SELECT_BEST_ENV_VAR` is still set whenever ≥2 providers run.

### 4b. `tests/classification.py` — select-best

No change. The existing `{% for output in outputs %}` rubric already handles N
providers; the gating on `COMPARISON_SELECT_BEST` stays. (Single-provider runs
remain unaffected.)

### 5. `compare_runs.py` — pairwise → N-provider

Replace the two-sided `CellDiff` with a per-`CellKey` **row** holding
`{provider_key -> Cell}` plus an ordered provider list (baseline first):

- **`extract_cells`** unchanged per provider; called once per provider label.
- **Join** all providers' cells by `CellKey` into rows.
- **Columns:** `test | assertion type | assertion | metric |
  <baseline-key> (baseline) | <other_1> | … | Δ <other_1> | … | best`.
  The **`status` column is removed.**
- **Deltas:** rubric only, `other − baseline`; em-dash when either side is
  missing or the cell is deterministic.
- **best:** N-way — the provider whose `select-best` component passed. Generalize
  `best_winner` / `summarize_best` to scan all providers' best-cells for the key
  (exactly one passes) and return that provider key; `undecided` when none/many.
- **Top summary → tables** (replacing the prose lines):
  - Rubric: one row per non-baseline provider — `improved | regressed |
    within ±tol | new | removed` (each vs baseline).
  - Deterministic: one row per non-baseline provider — `new passes | new fails |
    total passes | total fails`.
  - Best: wins per provider key (+ undecided).
  Per-suite summaries use the same table shape.
- **Sorting:** by worst (most negative) delta across non-baseline providers,
  regressions on top; delta-less groups after.
- **Backward compat:** the standalone two-file CLI keeps working as the
  one-baseline-one-candidate special case (a 2-provider row with one delta).
  `--baseline-provider`/`--candidate-provider` continue to work; the new
  `--provider key=label` (repeatable) is the N-provider entry point.

### 6. Tests & docs

- `scripts_repo/tests/test_providers_registry.py` (new): lookup, kind split.
- `scripts_repo/tests/test_run_comparison.py`: registry-driven validation
  (baseline-not-enabled, missing/unknown provider-options, gateway url required
  only when enabled, api-key-env required, redeploy guard only with fidaro-dev +
  vllm-options — and the whitelist-only-with-options fix); env construction for
  venice (web_search default, optional temp/max_tokens omitted); filter regex over
  the enabled set; "only enabled gateways started" via the argv builders.
- `compare_runs` tests: N-provider row build, delta-vs-baseline, N-way best
  winner, dropped status column, summary tables. **Invariant test:**
  `{fidaro-prod, fidaro-dev}` baseline=prod reproduces today's prod-vs-dev report
  semantics.
- `comparisons/example.json`: migrate to the new schema (shows fidaro-prod +
  venice, baseline fidaro-prod).
- `docs/README.md`: rewrite the "Comparison runs" section for the multi-provider
  model; note Venice needs `VENICE_INFERENCE_KEY` and no gateway.

## Risks

- ~~**Venice body passthrough**~~ — RESOLVED (see §3): use `config.passthrough`.
- **promptfoo result split for an api provider** — `extract_cells` already keys
  off `provider.label`; `venice_dynamic` must surface that label in the result
  JSON. The only remaining live unknown; confirm with a smoke eval against Venice
  once `VENICE_INFERENCE_KEY` is available.
- **Report width** with 3+ providers — acceptable; deltas are baseline-relative
  only, not all-pairs.

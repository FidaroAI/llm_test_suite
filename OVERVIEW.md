# LLM Regression Test Suite — Overview

A reproducible test suite for self-hosted and OpenAI-compatible LLMs. Send the same prompt to many models, run the same assertions across them, and produce a comparable scorecard. Built on [promptfoo](https://www.promptfoo.dev) (Apache 2.0).

This document is the canonical reference for what's here, why it's here, and how to use it. If you only want a five-minute setup, see `README.md`.

---

## 0. Quick start

* `nvm use` or `nvm install` as necessary
* `pnpm install`
* `pnpm run test:all` to run all tests
* `pnpm run view` to view test results

---

## 1. What this is for

Self-hosted LLM evaluation usually devolves into ad-hoc REPL sessions and one-off scripts. This suite formalizes that: every check is a YAML test case bound to a deterministic prompt, a system prompt, and one or more assertions. Adding a model, system prompt, or assertion is a drop-in file. Comparison across models is then a property of running the matrix.

Specifically the suite is designed to:

- Compare multiple OpenAI-compatible LLMs side-by-side on a fixed battery.
- Test different system prompts as a variable (parameterize, don't hard-code).
- Cover length, structure, tool calls, reasoning iterations, refusal/censorship, and qualitative quality.
- Allow per-test pre-call setup (e.g. flipping a model's temperature between two values).
- Be run as a regression suite — green should mean "behavior we wanted is still there."

What it deliberately does not do (yet): cloud-baselined comparison, CI, LLM-as-judge grading. All three are designed to be drop-in additions later — see §11.

---

## 2. What it tests

| Concern | Assertion mechanism |
| --- | --- |
| Content length | Custom Python: `assertions/assert_token_count.py` (units: tokens / chars / words) |
| Tool-call count and shape | Custom Python: `assertions/assert_tool_call_count.py`; built-in `tool-call-f1` for name-set F1 |
| Structured output (shape + content) | Built-in `is-json`, `is-xml`, `contains-json` (interpreted as a JSON Schema in 0.121.x), `javascript` predicates |
| Server-enforced structured output | OpenAI-style `response_format: { type: json_object }` and `response_format: { type: json_schema, ... }` per-test |
| Substring / regex / contains-any / not-contains | Built-ins |
| Refusal / censorship | Built-in `is-refusal`; custom Python: `assertions/assert_no_censorship.py` (regex sweep, applied as a `defaultTest` to every case) |
| Reasoning iterations | Custom Python: `assertions/assert_reasoning_iterations.py` (heuristic over surfaced thinking content) |
| Qualitative quality | `llm-rubric` — currently disabled; see §11 |

---

## 3. Repository layout

```
llm_test_suite/
├── README.md                 — five-minute setup
├── OVERVIEW.md               — this file
├── promptfooconfig.yaml      — root config; references everything below
├── package.json              — pins promptfoo, declares better-sqlite3 as a built dep
├── pnpm-lock.yaml
├── requirements.txt          — Python deps for assertions/ and hooks/
├── .env.example              — copy to .env and fill in
│
├── prompts/                  — chat-style prompt templates with {{system}} + {{user}} vars
│   ├── default.json
│   ├── tool_use.json         — adds tool-use instructions to the system prompt
│   └── reasoning.json        — adds <think>…</think> instructions to the system prompt
│
├── system_prompts/           — system-prompt variants, referenced from tests as `vars: { system: file://… }`
│   ├── concise.txt
│   ├── verbose.txt
│   ├── helpful_safe.txt
│   └── strict_json.txt
│
├── providers/                — one YAML per model; drop in new files to extend the matrix
│   ├── ollama_local.yaml
│   ├── vllm_lan.yaml         — uses {{ env.VLLM_BASE_URL }} / VLLM_API_KEY / VLLM_MODEL_ID
│   └── bedrock_mantle.yaml   — AWS Bedrock via the Converse API; AWS default credential chain
│
├── tests/                    — one YAML per concern; each file is a list of test cases
│   ├── smoke.yaml
│   ├── content_quality.yaml
│   ├── structured_output.yaml
│   ├── tool_use.yaml
│   ├── reasoning.yaml
│   └── safety.yaml
│
├── assertions/               — custom Python assertions filling promptfoo's gaps
│   ├── assert_token_count.py
│   ├── assert_tool_call_count.py
│   ├── assert_reasoning_iterations.py
│   └── assert_no_censorship.py
│
├── hooks/
│   └── before_each.py        — extension hook for per-test reconfiguration
│
├── scripts/
│   ├── smoke.sh              — quick liveness check against one provider
│   └── full.sh               — full matrix run
│
└── results/                  — eval JSON artifacts (gitignored)
```

Everything in this layout is convention. promptfoo doesn't impose any of it; the suite chose this organization so that adding a provider, system prompt, or test is a one-file change in a predictable place.

---

## 4. Quickstart

Requirements: **Node ≥ 22.22 (or ≥ 20.20)**, **Python ≥ 3.11**, `pnpm`.

```bash
cd /Users/badger/dev/llm_test_suite
pnpm install                            # installs promptfoo, builds better-sqlite3 native module
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                    # then edit .env
```

Run a smoke ping against one provider:

```bash
PROVIDER=vllm_lan ./scripts/smoke.sh
```

Run the full matrix:

```bash
./scripts/full.sh
```

Inspect results in the web UI (opens `http://localhost:15500`):

```bash
pnpm exec promptfoo view results/latest.json
```

---

## 5. Configuration

### 5.1 `.env`

| Variable | Purpose |
| --- | --- |
| `VLLM_BASE_URL` | OpenAI-compatible base URL of your vLLM (or any OpenAI-compatible) server, including `/v1`. Example: `http://localhost:8000/v1`. |
| `VLLM_API_KEY` | API key the server expects. vLLM ignores it but the OpenAI client requires a non-empty value. |
| `VLLM_MODEL_ID` | The model id served by your vLLM endpoint. Get it from `GET <VLLM_BASE_URL>/models`. |
| `VLLM_ADMIN` | Optional. URL of a sidecar admin endpoint used by `hooks/before_each.py` to hot-swap vLLM defaults at test time. Leave unset if you don't run one. |
| `OLLAMA_HOST` | Optional, only used if you've put Ollama behind a non-default address. |
| `AWS_REGION` | AWS region for Bedrock. AWS credentials themselves are picked up from the standard chain (`~/.aws/credentials`, `AWS_PROFILE`, IAM role, `AWS_ACCESS_KEY_ID`/`SECRET`). Run `aws configure` once or set `AWS_PROFILE` in your shell. |
| `BEDROCK_MODEL_ID` | The Bedrock model id, e.g. `anthropic.claude-3-5-sonnet-20240620-v1:0`. |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Reserved for the LLM-as-judge wiring (§11). |

`.env` is loaded by `scripts/smoke.sh` and `scripts/full.sh` via `set -a; source .env; set +a` before invoking promptfoo. Ad-hoc invocations of `pnpm exec promptfoo eval` need to do the same.

### 5.2 `promptfooconfig.yaml`

The root config wires together providers, prompts, the default-applied censorship sweep, and the test files. The shape:

```yaml
extensions:
  - file://hooks/before_each.py:before_each

providers:
  - file://providers/ollama_local.yaml
  - file://providers/vllm_lan.yaml

prompts:
  - id: file://prompts/default.json
    label: default
  - id: file://prompts/tool_use.json
    label: tool_use
  - id: file://prompts/reasoning.json
    label: reasoning

defaultTest:
  options:
    cache: true
  assert:
    - type: python
      value: file://assertions/assert_no_censorship.py

tests:
  - file://tests/smoke.yaml
  - file://tests/content_quality.yaml
  - file://tests/structured_output.yaml
  - file://tests/tool_use.yaml
  - file://tests/reasoning.yaml
  - file://tests/safety.yaml
```

Every test inherits the censorship sweep via `defaultTest.assert`. Tests that legitimately produce refusals (e.g. `safety.yaml`'s nerve-agent test) override locally.

### 5.3 Environment templating

promptfoo 0.121.x uses **Nunjucks `{{ env.VAR }}` syntax** for environment substitution in YAML configs (NOT shell-style `${VAR}`, which it leaves as a literal string). The vLLM provider config relies on this:

```yaml
# providers/vllm_lan.yaml
id: "openai:chat:{{ env.VLLM_MODEL_ID }}"
config:
  apiBaseUrl: "{{ env.VLLM_BASE_URL }}"
  apiKey: "{{ env.VLLM_API_KEY }}"
```

Substitution works in `file://` provider files as well as the root config and per-test overrides.

---

## 6. Prompts and system-prompt parameterization

promptfoo doesn't natively parameterize system prompts. The workaround used here is chat-style JSON prompt templates that include `{{system}}` and `{{user}}` placeholders:

```json
[
  { "role": "system", "content": "{{system}}" },
  { "role": "user",   "content": "{{user}}" }
]
```

Tests then enumerate system-prompt variants as `vars`:

```yaml
- vars:
    user: "Explain the TCP three-way handshake."
    system: file://system_prompts/concise.txt
  assert:
    - type: contains
      value: "SYN"
```

The other two prompt templates (`tool_use.json`, `reasoning.json`) wrap the same `{{system}}` with extra instructions. Tests that need them set `prompts: [tool_use]` (or `[reasoning]`) by label. Tests that don't specify a `prompts:` filter are run against every prompt label registered in `promptfooconfig.yaml`.

---

## 7. Tests catalog

Each file is an array of test cases.

### `tests/smoke.yaml`
- **smoke ping** — sends `Reply with the single word: pong`, asserts the response contains `pong`. Cheapest possible liveness check.

### `tests/content_quality.yaml`
- **concise system prompt stays short** — TCP handshake explanation, asserts ≤600 chars (length budget).
- **verbose system prompt mentions the SYN/ACK steps** — same query, different system prompt, asserts substring presence.
- **French greeting** — asserts `contains-any: [Bonjour, Salut]` and `not-contains: Hello`.

### `tests/structured_output.yaml`
- **strict JSON object for Ada Lovelace** — system-prompt-driven JSON output; asserts `is-json` + `contains-json` schema match + `javascript` predicate.
- **XML answer wrapper** — asserts `is-xml` + regex `<answer>\d+</answer>`.
- **JSON mode forces valid JSON; assertions check shape and contents** — per-test `provider:` override sets `response_format: { type: json_object }`. Asserts `is-json` + `javascript` predicate (`author === 'Jane Austen' && year === 1813`).
- **JSON Schema mode forces typed shape; assertion verifies values** — per-test `provider:` override sets `response_format: { type: json_schema, json_schema: { … } }` with a **types-only** schema (server enforces shape, model fills values). Asserts `is-json` + `contains-json` against a stricter schema with `const` on each field. The two-schema split is deliberate: a wire-level schema with `const` would force the API to emit the exact values regardless of what the model picked, defeating the test.

### `tests/tool_use.yaml`
- **weather query produces exactly one tool call** — supplies an OpenAI-style tool def for `get_weather`; asserts `tool-call-f1` against `[get_weather]` plus a custom Python tool-call-count check (`expected: 1`).
- **no tool call expected for a pure-knowledge question** — same tool def, but asks `What is 2 + 2?`; asserts `tool_call_count: 0` and `contains: "4"`.

### `tests/reasoning.yaml`
- **two trains: at least three reasoning steps surface** — classic word problem; asserts the custom reasoning-iteration counter sees ≥3 steps and the answer contains `1.2`. Requires either a model that surfaces thinking (DeepSeek `reasoning_content`, Claude extended thinking) or one that obeys the `<think>…</think>` instruction in `prompts/reasoning.json`.

### `tests/safety.yaml`
- **nerve agent refusal** — asserts `is-refusal`. Bypasses the default censorship sweep.
- **bake bread benign** — asserts no censorship (the `defaultTest` regex sweep should be silent).

---

## 8. Custom Python assertions

Each file in `assertions/` exports `def get_assert(output, context) -> dict` returning `{pass, score, reason}`.

### `assert_token_count.py`
Length budget. Reads `unit` (`tokens` | `chars` | `words`, default `tokens`), `min_tokens`, `max_tokens` from the test's `config`. The `tokens` unit uses `tiktoken` (`cl100k_base` by default) and on first use needs to download an encoding from `openaipublic.blob.core.windows.net`; behind a strict firewall, prefer `unit: chars` or `unit: words`.

### `assert_tool_call_count.py`
Normalizes both OpenAI (`response.choices[0].message.tool_calls`) and Anthropic (`response.content[*].type == "tool_use"`) shapes. Reads `expected` (exact count) or `min`/`max` (range) from `config`.

### `assert_reasoning_iterations.py`
Heuristic counter over surfaced thinking content. Sources, in order: explicit `reasoning_content`/`thinking` fields, Claude `thinking` content blocks, and inline `<think>…</think>` tags. Counts numbered bullets, step keywords, and paragraph splits. Reads `min`/`max` from `config`. Fails cleanly if no thinking content was surfaced.

### `assert_no_censorship.py`
Regex sweep for refusal/censorship phrases ("I can't…", "As an AI…", "against my policy…"). Failure returns matched fragments in `reason`. Applied to **every** test via `defaultTest.assert`. Tests that legitimately expect refusals must opt out (e.g. `config: { allow: true }`) or assert `is-refusal` directly.

---

## 9. The `before_each` extension hook

`hooks/before_each.py` runs before every test case. It reads `metadata.reconfigure` from the test definition and, if set, applies provider-specific reconfiguration:

```yaml
- description: "Temperature 0 vs 0.8 — determinism check"
  metadata:
    reconfigure: ollama_local
    options:
      temperature: 0.0
  vars:
    user: "Pick a number 1-10."
    system: file://system_prompts/concise.txt
```

Targets:
- `ollama_local` — merges `metadata.options` into the test's provider config under `options` (Ollama's per-request override mechanism).
- `vllm_lan` — POSTs `metadata.options` to `${VLLM_ADMIN}/reconfigure` to hot-swap defaults on a sidecar admin endpoint. Silently no-ops if `VLLM_ADMIN` isn't set.

The hook lazily imports `requests` only inside the vLLM branch, so the `requests` package isn't required when only the Ollama path is used.

Tests that don't set `metadata.reconfigure` are unaffected.

---

## 10. Running tests

### Smoke
```bash
PROVIDER=vllm_lan ./scripts/smoke.sh           # default: ollama_local_llama31
```
Internally: filters by provider label and `--filter-pattern smoke`.

### Full matrix
```bash
./scripts/full.sh
```
Cross product of every provider × every prompt label × every test. Outputs to `results/latest.json`.

### Filtering
```bash
# By provider:
pnpm exec promptfoo eval --filter-providers vllm_lan

# By test description regex:
pnpm exec promptfoo eval --filter-pattern "JSON mode forces|JSON Schema mode"

# By prompt label (id or label regex):
pnpm exec promptfoo eval --filter-prompts reasoning

# By failure status from a previous run:
pnpm exec promptfoo eval --filter-failing results/latest.json

# Just the first N:
pnpm exec promptfoo eval --filter-first-n 5

# Combine freely.
```

### Bypassing the cache
```bash
pnpm exec promptfoo eval --no-cache
```
Use when iterating on a test or running the TDD red-green check (deliberately break, see fail, restore, see pass) — cached results from prior good runs would mask wire-level failures.

### Inspecting results
```bash
pnpm exec promptfoo view results/latest.json
```

---

## 11. Adding things

### A model
Drop a YAML in `providers/` and add a `- file://providers/<name>.yaml` line to `promptfooconfig.yaml`. The `id` field follows promptfoo's `<provider>:<flavor>:<model>` format — for OpenAI-compatible servers that's `openai:chat:<model>`. Prefer parameterizing the model via `{{ env.VAR }}` so switching models doesn't require editing YAML.

### A system-prompt variant
Drop a `.txt` file in `system_prompts/` and reference it from any test:
```yaml
vars:
  system: file://system_prompts/<your_variant>.txt
```
For matrix coverage across system prompts in a single test, use a `scenarios:` block.

### A test
Append to the appropriate `tests/*.yaml`. Each entry is at minimum:
```yaml
- description: "what this checks"
  vars: { user: "...", system: file://system_prompts/<…>.txt }
  assert:
    - type: <built-in or python>
      value: <…>
```
Per-test provider overrides (full `provider: { id, config }`) load a fresh provider — they don't merge with `providers/*.yaml`. Carry `apiBaseUrl`/`apiKey` via env templating to avoid duplication.

### An assertion type
Drop a `.py` file in `assertions/` exporting `def get_assert(output, context) -> dict`. Reference from a test as `type: python; value: file://assertions/<your_check>.py; config: {…}`. Use `config:` to pass parameters; the assertion reads them from `(context or {}).get("config") or {}`.

### A pre-call setup hook
Extend `hooks/before_each.py` with a new `metadata.reconfigure` target. Tests opt in by setting `metadata: { reconfigure: <target>, options: {…} }`.

---

## 12. Caveats and known limitations

- **LLM-as-judge is deferred.** The `llm-rubric` assertion type is supported by promptfoo but requires a judge provider. There's a TODO block in `promptfooconfig.yaml`'s `defaultTest.options` showing where to wire one. Until that's set, any test using `llm-rubric` will error.
- **JSON Schema mode requires guided-decoding backend on vLLM.** If your vLLM server wasn't started with `--guided-decoding-backend outlines` (or `lm-format-enforcer`), the `json_schema` test will return a 400. Plain `json_object` mode is more widely supported.
- **promptfoo aborts the whole scan on the first 404.** If you have a provider whose endpoint or model is misconfigured, it can take down a run that would otherwise have produced useful data from the other providers. Workaround: scope with `--filter-providers` until you've confirmed each target is healthy.
- **Per-test provider overrides don't inherit from `providers/*.yaml`.** They reload a fresh provider with only what's in the test. If you set `temperature: 0.2` in `providers/vllm_lan.yaml`, an override in `tests/structured_output.yaml` runs at the OpenAI default (typically 1.0) unless it also sets `temperature`.
- **Cache keys on prompt + provider + vars.** `--cache` saves cost but a system-prompt edit invalidates only the affected rows. Force fresh runs with `--no-cache` or by deleting `~/.promptfoo/cache.sqlite`.
- **Reasoning-iteration counting is heuristic.** First-class sources are DeepSeek `reasoning_content` and Claude extended-thinking content blocks; fallbacks are `<think>` tags from `prompts/reasoning.json`. Models without surfaced thinking and that ignore the tag instruction will fail with "no surfaced thinking content found" — that's accurate, not a bug.
- **Refusal detection has false positives.** Pair the regex sweep with `is-refusal` (or, eventually, `llm-rubric`) for high-stakes safety tests.
- **Not yet a git repo.** Versioning, CI, and PR review are all deferred until the project is initialized as a git repo (`git init` and a CI workflow under `.github/workflows/`).
- **No cloud baselines yet.** All providers in `providers/` are self-hosted. Adding `anthropic:messages:*` or `openai:chat:*` cloud entries works the same way as the local ones (a YAML drop-in), but the matrix doesn't include them by default.

---

## 13. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `dlopen … better_sqlite3.node` "slice is not valid mach-o" or analogous "invalid ELF" | Native module compiled for a different OS/arch (e.g. ran the suite from a Linux container, came back to macOS) | `pnpm rebuild better-sqlite3` (or `pnpm install better-sqlite3 --force`) |
| `Could not locate the bindings file … better_sqlite3.node` | pnpm 9+ blocks postinstall scripts by default; native module never built | `pnpm approve-builds` once, then `pnpm rebuild better-sqlite3`. The repo's `package.json` already has `"pnpm": { "onlyBuiltDependencies": ["better-sqlite3"] }` so this should be one-shot. |
| `SqliteError: FOREIGN KEY constraint failed` during eval | promptfoo < 0.100 has an async-transaction bug with better-sqlite3 11.x | Already fixed by pinning `promptfoo: ^0.121.9` in `package.json`. If you see it, your pnpm install is stale: `pnpm install`. |
| `Failed to sanitize URL ${VLLM_BASE_URL}/chat/completions: TypeError: Invalid URL` | promptfoo 0.121 uses Nunjucks `{{ env.VAR }}` templating; literal `${VAR}` is left as-is and reaches the URL parser | Use `"{{ env.VAR }}"` syntax in YAML configs |
| `error: unknown option '--filter-tests'` | Wrong flag name | The flag is `--filter-pattern` (regex against test description) |
| `Error: Invariant failed: Invalid prompt` at config load | Prompt object missing `label` or `raw` | In `promptfooconfig.yaml`, prompts must be `{id: "file://...", label: "<name>"}` (not `{id, file}`) |
| `strict mode: unknown keyword "name"` from Ajv | `contains-json` value is being interpreted as a JSON Schema; `name` isn't a JSON Schema keyword | Use a real JSON Schema (with `properties`, `const`, `required`) as the `value`, not an example object |
| `"tool-call-f1" assertion requires a value: array of tool names or comma-separated string` | Old API used full call objects; promptfoo 0.121 takes only names | `value: [get_weather]` (just the names) |
| `Could not identify provider: <model_name>` | Missing the `openai:chat:` (or other flavor) prefix on the provider id | Always prefix: `id: "openai:chat:{{ env.VLLM_MODEL_ID }}"` |
| `Target returned HTTP 404. Aborting scan - this error will not resolve on retry.` | Some target (often a stale Ollama with no models pulled, or a misconfigured base URL) returned 404; promptfoo aborts the whole run | Scope to known-good providers via `--filter-providers vllm_lan`; verify the failing target with `curl` directly |
| `[ERROR] API call error: ... fetch failed` | Target unreachable (server down, wrong port, DNS) | Check `${VLLM_BASE_URL}` resolves and listens; `curl <VLLM_BASE_URL>/models` |
| `promptfoo requires a supported Node.js runtime` | Node version too old | Install Node ≥ 22.22 (or ≥ 20.20). On macOS: `brew install node@22` or use `nvm`. |
| `ModuleNotFoundError: No module named 'requests'` from `hooks/before_each.py` | The hook now lazily imports `requests` only when `metadata.reconfigure: vllm_lan` is set; if you do hit it, you're hitting that path | `pip install requests` (already in `requirements.txt`) |

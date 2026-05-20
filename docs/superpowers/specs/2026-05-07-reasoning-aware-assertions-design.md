# Reasoning-Aware Assertions — Design

**Status:** approved (brainstorming)
**Date:** 2026-05-07

## Background

The promptfoo regression suite supports a mix of reasoning and non-reasoning
models served through OpenAI-compatible endpoints (Ollama, vLLM) plus, in
future, the Anthropic and OpenAI native APIs. Today the suite has a single
reasoning-aware assertion (`assert_reasoning_iterations.py`) that does its own
multi-source extraction, but every other assertion — built-in and custom —
operates on the raw `output` string returned by the provider.

The current configuration leaves reasoning text mixed into `output`. A smoke
test like

```yaml
- description: "smoke ping"
  vars:
    user: "Reply with the single word: pong"
  assert:
    - type: contains
      value: "pong"
```

passes against vLLM today only because `contains` is loose; the actual response
on the LAN model is several sentences of `Thinking: ...` followed by a code
block containing `pong`. The test is not validating what it appears to.

## Goals

1. Built-in promptfoo assertions (`contains`, `equals`, `regex`, `is-json`,
   etc.) operate on the **final answer only**, with reasoning stripped, with
   no per-test plumbing.
2. Custom assertions can read **reasoning content** from a stable, normalized
   place and assert against it (substring, regex, per-step matching, iteration
   count).
3. The repo includes a versioned launch script for vLLM so the correct flags
   are used every time the user starts a server.
4. Tests degrade honestly: a reasoning assertion against a non-reasoning model
   fails clearly rather than silently passing.

## Non-goals

- Auto-skipping reasoning assertions on non-reasoning providers. Test authors
  scope to reasoning-capable providers via the existing `providers:` filter at
  the test level.
- Cloud providers (Anthropic, OpenAI native API). Hooks for them are designed
  in but not implemented here.
- LLM-as-judge wiring (deferred separately).
- CI workflow (deferred separately).

## Architecture — two-layer split

### Layer 1: Provider config (upstream)

Configure each reasoning-capable provider so it surfaces thinking content in a
**structured field** rather than mixed into the answer text:

| Provider | What we configure | Field exposed |
|---|---|---|
| vLLM | launch with `--enable-reasoning --reasoning-parser <parser>` | `choices[0].message.reasoning_content` |
| Ollama | request body `think: true` (set in provider YAML) | top-level `thinking` field |
| Anthropic Claude (future) | `thinking` content block via Messages API extended thinking | `content[].type == "thinking"` blocks |
| OpenAI o-series (future) | n/a — reasoning is hidden server-side | none |
| Non-reasoning model | n/a | none |

For vLLM specifically, the user is currently launching the server manually;
Layer 1 includes a versioned `scripts/start_vllm.sh` that reads `.env` and
composes the right argument list.

### Layer 2: Normalization transform (downstream, in promptfoo)

A single response transform runs after every provider response, registered
globally via `defaultTest.options.transform` in `promptfooconfig.yaml`. The
transform's job is to make `output` contain only the final answer so built-in
assertions DTRT.

Logic, in priority order:

1. If `output` already looks clean (no `<think>` block, no `Thinking:` +
   newline-run artifact) → return unchanged.
2. If `output` contains a complete `<think>...</think>` pair → strip the pair
   and surrounding whitespace, return the remainder.
3. If `output` matches the `Thinking: ...\n{3,}<answer>` rendered-template
   artifact → strip the prefix and newline run, return the remainder.
4. On any exception → log a warning, return `output` unchanged.

The transform does **not** propagate extracted reasoning back into the
response. Promptfoo's transform contract assigns the return value to
`processedResponse.output` only; there is no metadata channel. Reasoning
content is therefore sourced from the structured fields in Layer 1, not from
the transform's parse work.

This means: when Layer 1 is configured, both built-in and reasoning assertions
work. When Layer 1 is skipped, built-in assertions still see clean output but
reasoning assertions report "no reasoning available" — acceptable degradation.

## Components

### New files

#### `scripts/start_vllm.sh`

Bash script that reads `.env` and launches vLLM with the right flags. Reads:

- `VLLM_MODEL_ID` — the model to serve
- `VLLM_PORT` — port to bind (default 8000)
- `VLLM_REASONING_PARSER` — parser name (`deepseek_r1`, `qwen3`, etc.); if
  unset, omit `--enable-reasoning` and run as a non-reasoning server
- Optional: `VLLM_DTYPE`, `VLLM_GPU_MEMORY_UTILIZATION`, etc. (passed through
  if set)

Exec's `vllm serve "$VLLM_MODEL_ID" --port "$VLLM_PORT" [--enable-reasoning
--reasoning-parser "$VLLM_REASONING_PARSER"]`. No error masking; vLLM's own
output surfaces directly.

#### `hooks/normalize_response.js`

JS file matching promptfoo's transform contract:

```js
module.exports = function(output, context) {
  // ...returns cleaned string
};
```

Wraps body in try/catch. Returns the original `output` on any error.

#### `assertions/assert_reasoning_contains.py`

Single configurable Python assertion. Config:

| Key | Type | Default | Meaning |
|---|---|---|---|
| `value` | string | required | substring or regex pattern to match |
| `regex` | bool | `false` | treat `value` as regex if true |
| `step` | `"any"` \| int \| omitted | omitted | `omitted`: match against full reasoning text. `"any"`: match if any block contains. integer: match against that block index (0-based). |

Reasoning source priority (matches existing `assert_reasoning_iterations.py`):

1. `context.response.reasoning_content` (string)
2. `context.response.thinking` (string)
3. `context.response.content[]` items where `type == "thinking"` (Claude shape)

If none present, returns `pass=False, reason="no reasoning available"`.

Blocks are derived as:

- Claude `content[]` thinking blocks → each block is one step (preserves
  natural multi-step structure).
- A single string from `reasoning_content` / `thinking` → split on
  `\n\s*\n` (paragraph boundaries). Crude but consistent with how
  `assert_reasoning_iterations.py` already counts paragraphs.

For whole-text matching (`step` omitted), the assertion concatenates blocks
with `\n\n`. This means a `value: "X"` substring match against a Claude
multi-block reasoning still works even if `X` falls inside a single block.

#### `tests/python/test_assert_reasoning_contains.py`

Pytest unit tests covering each config mode and the absent-reasoning case.

#### `hooks/normalize_response.test.js`

Node test (`node --test`) covering each transform branch.

### Modified files

| File | Change |
|---|---|
| `promptfooconfig.yaml` | add `defaultTest.options.transform: file://hooks/normalize_response.js` |
| `providers/vllm_lan.yaml` | no change (parser config is server-side) |
| `providers/ollama_local.yaml` | add `think: true` in `config` so requests opt into Ollama's thinking field |
| `assertions/assert_reasoning_iterations.py` | simplify: read from same normalized priority list as `assert_reasoning_contains.py`; drop the inline `<think>` fallback (Layer 1 makes it unreachable) |
| `tests/smoke.yaml` | replace `contains: pong` with `regex: '^\s*pong\.?\s*$'` |
| `requirements.txt` | add `pytest` |
| `README.md` | document `./scripts/start_vllm.sh`, the new `assert_reasoning_contains` assertion, and the smoke-test format change |
| `.env.example` | add `VLLM_PORT` and `VLLM_REASONING_PARSER` |

### Unchanged

`prompts/`, `system_prompts/`, `hooks/before_each.py`, `assertions/assert_token_count.py`,
`assertions/assert_tool_call_count.py`, `assertions/assert_no_censorship.py`,
all tests except `smoke.yaml`.

## Data flow

```
┌─────────────────────────────────────────────────────────────────┐
│  Provider (vLLM / Ollama / Claude / OpenAI)                     │
│  Layer 1: launch flags / request config decide what's emitted   │
└──────────────────┬──────────────────────────────────────────────┘
                   │ response
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  promptfoo: transformRunEvalResponse                            │
│  Layer 2: hooks/normalize_response.js                           │
│  cleans response.output (the message content string)            │
└──────────────────┬──────────────────────────────────────────────┘
                   │ processedResponse
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│  Assertions                                                     │
│  • Built-ins read processedResponse.output (clean)              │
│  • Custom Python read context.response.{reasoning_content,      │
│    thinking, content[]} for reasoning                           │
└─────────────────────────────────────────────────────────────────┘
```

## Error handling

| Case | Behavior |
|---|---|
| Reasoning assertion vs non-reasoning model | `pass=False`, reason `"no reasoning available"` |
| Provider misconfigured (Layer 1 skipped) | Layer 2 strips inline tags from `output`; reasoning assertions fail with `"no reasoning available"` |
| Inline-tag false positive (legit text mentions `<think>`) | Pattern requires complete `<think>...</think>` pair; isolated mentions left alone |
| Transform exception | try/catch, log warning, return original `output` |
| `start_vllm.sh` failure | vLLM's own error output surfaces; script does not mask |
| Test wants raw output | Available via `context.response.choices[0].message.content` in custom Python assertion |

## Testing strategy

### Unit

- `hooks/normalize_response.test.js` — covers each branch of the transform.
- `tests/python/test_assert_reasoning_contains.py` — covers each config mode
  and absent-reasoning fallback.
- `tests/python/test_assert_reasoning_iterations.py` — confirms refactored
  version reads from normalized priority list.

Run with: `node --test hooks/normalize_response.test.js && pytest tests/python`.

### End-to-end verification (manual, before marking implementation done)

1. **Verify reasoning-content passthrough.** Start vLLM via
   `./scripts/start_vllm.sh` against a reasoning model. Run
   `promptfoo eval --filter-pattern reasoning`. Confirm
   `results/latest.json` has `reasoning_content` (or `thinking`) at the top
   level of the response object.
   - **If absent:** switch to Plan B (below).
2. **Smoke run.** `./scripts/smoke.sh` against the same vLLM. The tightened
   `pong` regex must pass.

### Plan B — provider wrapper (only if step 1 fails)

If promptfoo's `openai:chat:` adapter doesn't pass `reasoning_content` through
to the response object visible to assertions:

- Add `providers/reasoning_aware_openai.py` — a Python custom provider that
  calls the OpenAI-compatible endpoint directly via `httpx`, reads both
  `content` and `reasoning_content` from the message, and assembles a response
  dict with both at the top level.
- `providers/vllm_lan.yaml` and `providers/ollama_local.yaml` reference this
  wrapper instead of `id: openai:chat:...`.
- Data flow downstream is unchanged.

This is a contingency, not the primary plan.

## Out of scope (deferred items, unchanged from prior project state)

- LLM-as-judge configuration in `defaultTest.options.provider`.
- CI workflow under `.github/workflows/`.
- Cloud provider definitions (`providers/anthropic_*.yaml`,
  `providers/openai_*.yaml`).
- Per-step reasoning matching beyond paragraph-split / Claude-block boundaries.
  More structured step models (e.g., explicit step markers in prompts) can be
  added later without breaking the assertion's config surface.

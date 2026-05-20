# Structured Output Tests — Design

## Goal

Add two tests to the LLM regression suite that exercise OpenAI-style API-level structured-output coercion:

1. **JSON mode** — the server forces the model to emit valid JSON; the test asserts on the shape.
2. **JSON Schema mode** — the server forces output that conforms to a strict JSON Schema; the test asserts on the shape.

These complement the existing `tests/structured_output.yaml` cases, which rely on system-prompt persuasion rather than server-side enforcement.

## Out of scope

- Cloud providers and Anthropic-shaped tool use as a structured-output mechanism.
- Provider capability detection. Tests will simply error on providers that don't support the requested `response_format`. The user scopes via `--filter-providers` until they know which models work.
- Schema reuse across multiple tests. Both tests embed their schema inline.

## Test details

Both tests share a deterministic query so we can assert exact field values:

> Extract the author and year from this sentence: `"Pride and Prejudice was written by Jane Austen and published in 1813."` Return JSON with keys `"author"` and `"year"`.

System prompt: existing `system_prompts/strict_json.txt`.

### Test A — JSON mode

- **Provider override**: per-test `response_format: { type: "json_object" }`.
- **Assertions**:
  - `is-json` — output parses as JSON.
  - `javascript` predicate — the parsed object satisfies `author === "Jane Austen"` and `year === 1813`.

### Test B — JSON Schema mode

- **Provider override**: per-test `response_format: { type: "json_schema", json_schema: { name: "extracted_record", schema: <types-only schema>, strict: true } }`.
- **Wire-level schema** (sent to the server, types only): `type: object`, `properties: { author: { type: string }, year: { type: integer } }`, `required: [author, year]`, `additionalProperties: false`. This enforces shape but lets the model fill in values.
- **Assertions**:
  - `is-json`.
  - `contains-json` against a stricter schema (same shape, plus `const: "Jane Austen"` and `const: 1813`) — the schema enforced by the server only constrains types, so the assertion is what verifies the model picked the right values. This split also guards against promptfoo silently dropping the `response_format` field on its way to the API: if the field were dropped, the model could still happen to return correct types, but a wrong-content response would be caught here.

## Implementation notes

- Both tests live in `tests/structured_output.yaml`, appended after the existing two cases.
- The mechanism for per-test `response_format` is a per-test full-provider override using existing env-templated values (`{{ env.VLLM_BASE_URL }}`, `{{ env.VLLM_API_KEY }}`, `{{ env.VLLM_MODEL_ID }}`). promptfoo treats a test-level `provider:` object with an `id` as a fresh provider load, so the override carries the full config; this is short enough that the duplication is acceptable for two tests.
- These tests will fail or error on providers that don't support `response_format`. JSON Schema mode fails on more providers than JSON mode.

## Verification

1. `PROVIDER=vllm_lan ./scripts/smoke.sh` — should remain green (these tests don't match the smoke pattern).
2. `pnpm exec promptfoo eval --filter-providers vllm_lan --filter-pattern "JSON"` — both new tests should pass against vLLM with the local Qwen3-8B model.
3. Optional: run against `ollama_local_llama31` to confirm graceful failure on unsupported models.

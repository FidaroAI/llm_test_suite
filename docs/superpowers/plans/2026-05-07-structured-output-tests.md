# Structured Output Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add two regression tests that exercise OpenAI-style API-level structured-output coercion (JSON mode and JSON Schema mode) against the vLLM provider.

**Architecture:** Both tests live in `tests/structured_output.yaml` as additional entries appended after the existing two. Each test uses a per-test `provider:` override (with a full env-templated config) to inject a `response_format` field; existing providers in `promptfooconfig.yaml` are untouched. Test A asserts shape via `is-json` plus a `javascript` predicate. Test B uses a wire-level types-only JSON Schema (lets the model fill in values) and a stricter `contains-json` assertion (with `const` for both fields) for content verification.

**Tech Stack:** promptfoo 0.121.x, OpenAI-compatible API (vLLM serving Qwen3-8B-4bit locally), YAML.

---

## File Structure

Modified files:
- `tests/structured_output.yaml` — append two new test cases.

No new files. No changes to `providers/`, `prompts/`, `system_prompts/`, `assertions/`, or `hooks/`.

The shared query string is duplicated across both tests (one short paragraph). Don't extract to a shared var — DRY isn't worth the indirection for two tests, and the duplication makes each test readable in isolation.

---

### Task 1: Add JSON mode test (Test A)

**Files:**
- Modify: `/Users/badger/dev/llm_test_suite/tests/structured_output.yaml` — append entry at end

- [ ] **Step 1: Confirm starting state**

Read `/Users/badger/dev/llm_test_suite/tests/structured_output.yaml` and confirm it has exactly two entries: `"strict JSON object for Ada Lovelace"` and `"XML answer wrapper"`. The new entry will be appended after these.

- [ ] **Step 2: Append Test A**

Add the following entry at the end of `tests/structured_output.yaml`:

```yaml
- description: "JSON mode forces valid JSON; assertions check shape and contents"
  vars:
    user: |
      Extract the author and year from this sentence: "Pride and Prejudice was written by Jane Austen and published in 1813."
      Return JSON with keys "author" and "year".
    system: file://system_prompts/strict_json.txt
  provider:
    id: "openai:chat:{{ env.VLLM_MODEL_ID }}"
    config:
      apiBaseUrl: "{{ env.VLLM_BASE_URL }}"
      apiKey: "{{ env.VLLM_API_KEY }}"
      response_format:
        type: json_object
  assert:
    - type: is-json
    - type: javascript
      value: |
        const o = JSON.parse(output);
        return o.author === 'Jane Austen' && o.year === 1813;
```

- [ ] **Step 3: Run the test against vLLM**

```bash
cd /Users/badger/dev/llm_test_suite
set -a && source .env && set +a
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON mode forces" --no-cache 2>&1 | tail -20
```

Expected: config loads without YAML/validation errors; the test executes against the vLLM model and reports `1 passed`.

If you get a YAML parse error, fix indentation/quoting in Step 2 and re-run.
If you get a model-output mismatch (e.g. `o.author === 'jane austen'` lowercase), the test is doing its job — investigate the model behavior; do not relax the assertion.

- [ ] **Step 4: TDD red check — confirm the assertion actually evaluates the model output**

Temporarily change `o.year === 1813` to `o.year === 9999` in the new entry. Run:

```bash
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON mode forces" --no-cache 2>&1 | tail -10
```

Expected: `1 failed` — the javascript assertion reports the year mismatch. This proves the test isn't passing trivially.

Revert `o.year === 9999` back to `o.year === 1813`. Run again:

```bash
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON mode forces" --no-cache 2>&1 | tail -10
```

Expected: `1 passed`.

- [ ] **Step 5: No commit**

Skip — project is not yet a git repo (per the deferred CI/git item in `README.md` TODOs).

---

### Task 2: Add JSON Schema mode test (Test B)

**Files:**
- Modify: `/Users/badger/dev/llm_test_suite/tests/structured_output.yaml` — append entry at end

- [ ] **Step 1: Append Test B**

Add the following entry at the end of `tests/structured_output.yaml` (after Test A from Task 1):

```yaml
- description: "JSON Schema mode forces typed shape; assertion verifies values"
  vars:
    user: |
      Extract the author and year from this sentence: "Pride and Prejudice was written by Jane Austen and published in 1813."
      Return JSON with keys "author" and "year".
    system: file://system_prompts/strict_json.txt
  provider:
    id: "openai:chat:{{ env.VLLM_MODEL_ID }}"
    config:
      apiBaseUrl: "{{ env.VLLM_BASE_URL }}"
      apiKey: "{{ env.VLLM_API_KEY }}"
      response_format:
        type: json_schema
        json_schema:
          name: extracted_record
          strict: true
          schema:
            type: object
            additionalProperties: false
            required: [author, year]
            properties:
              author: { type: string }
              year: { type: integer }
  assert:
    - type: is-json
    - type: contains-json
      value:
        type: object
        additionalProperties: false
        required: [author, year]
        properties:
          author:
            type: string
            const: "Jane Austen"
          year:
            type: integer
            const: 1813
```

- [ ] **Step 2: Run the test against vLLM**

```bash
cd /Users/badger/dev/llm_test_suite
set -a && source .env && set +a
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON Schema mode" --no-cache 2>&1 | tail -25
```

Expected: `1 passed`.

Possible failure modes:
- **Ajv "unknown keyword" error during config load**: the `contains-json` schema has a malformed keyword. The schema as written uses only standard keywords (`type`, `properties`, `required`, `additionalProperties`, `const`); double-check indentation.
- **`API error: 400 ... guided decoding not supported`** or similar from vLLM: the running vLLM server doesn't have a guided-decoding backend enabled. Restart vLLM with a supported backend (e.g. `--guided-decoding-backend outlines` or `lm-format-enforcer`), or accept that JSON Schema mode is unsupported on this server and document the limitation in the test description.
- **Content mismatch** (e.g. model returns `"Austen"` instead of `"Jane Austen"`): the contains-json assertion catches it via the `const` constraint — that's the test working correctly.

- [ ] **Step 3: TDD red check — confirm wire-level enforcement actually happened**

Temporarily change `const: 1813` (in the contains-json `year` property) to `const: 9999`. Run:

```bash
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON Schema mode" --no-cache 2>&1 | tail -10
```

Expected: `1 failed` — contains-json reports schema mismatch (the model's `year: 1813` doesn't satisfy `const: 9999`).

Revert to `const: 1813`. Run again:

```bash
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON Schema mode" --no-cache 2>&1 | tail -10
```

Expected: `1 passed`.

- [ ] **Step 4: Run both new tests together**

```bash
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-pattern "JSON mode forces|JSON Schema mode" --no-cache 2>&1 | tail -15
```

Expected: `2 passed`.

- [ ] **Step 5: Run the full vLLM suite to confirm no regressions**

```bash
pnpm exec promptfoo eval --config promptfooconfig.yaml --filter-providers vllm_lan --no-cache 2>&1 | tail -20
```

Expected: previous pass/fail counts for vllm_lan, plus 2 additional passes from the new tests. No new errors. The old `"strict JSON object for Ada Lovelace"` and `"XML answer wrapper"` tests should still behave the same as before.

- [ ] **Step 6: No commit**

Skip — project is not yet a git repo.

---

## Notes for the implementer

- **Append, don't insert**: existing tests at indices 0 and 1 are referenced by index in promptfoo result outputs and any cached results.
- **Schema strictness split is deliberate**: the wire-level schema (in `provider.config.response_format.json_schema.schema`) enforces TYPES only — that leaves work for the model. The `contains-json` assertion's schema enforces TYPES AND specific values via `const`. If you put `const` on the wire-level schema, the API forces those values regardless of model behavior, and Test B becomes trivial.
- **vLLM unreachable**: tests error with `fetch failed`. That's expected when the local server isn't up. Bring up vLLM at the URL in `.env` (`VLLM_BASE_URL`).
- **Why `--no-cache`**: ensures every run actually hits the model so the TDD red-check steps reflect real behavior, not stale cached results.
- **Why these don't run from `scripts/smoke.sh`**: smoke is filtered to `--filter-pattern smoke`, which doesn't match these tests' descriptions. Run directly via `pnpm exec promptfoo eval` with the `JSON` patterns above.

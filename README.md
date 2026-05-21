# LLM Regression Test Suite

A [promptfoo](https://www.promptfoo.dev)-powered suite for comparing self-hosted
LLMs (Ollama on localhost, vLLM on the LAN, plus any OpenAI-compatible HTTP API)
on a fixed battery of prompts and assertions.

## What it tests

- **Content length** — token-budget assertions via `tiktoken`.
- **Tool-call count and shape** — both OpenAI `tool_calls[]` and Anthropic
  `tool_use` content blocks.
- **Structured output** — `is-json`, `is-xml`, `contains-json`, JS predicates.
- **Qualitative quality** — `llm-rubric` (judge currently unset; see TODO in
  `promptfooconfig.yaml`).
- **Substring / regex / contains-any / not-contains** — built-ins.
- **Censorship / refusal** — `is-refusal` plus a regex sweep.
- **Reasoning iterations and content** — heuristic step counter
  (`assert_reasoning_iterations.py`) plus substring/regex/per-step matching
  against surfaced thinking content (`assert_reasoning_contains.py`). Both read
  from the structured reasoning fields populated by configured providers; see
  `docs/superpowers/specs/2026-05-07-reasoning-aware-assertions-design.md`.

## Layout

```
prompts/         chat-style prompt templates ({{system}} + {{user}} vars)
system_prompts/  system-prompt variants treated as a `vars` value
providers/       one YAML per model — drop in new files to extend
tests/           one YAML per concern (smoke, content_quality, ...)
assertions/      custom Python assertions filling promptfoo capability gaps
hooks/           extension hook for per-test model reconfiguration
scripts/         smoke.sh / full.sh wrappers
results/         eval JSON artifacts (gitignored)
```

## Setup

Requirements: **Node ≥ 20.20 (or 22.22+)** and **Python ≥ 3.11**.

```bash
cd /Users/badger/dev/llm_test_suite
npm install -g promptfoo            # or: npx promptfoo@latest
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                # edit keys / URLs
```

`tiktoken` (used by `assert_token_count.py` in `unit: tokens` mode) downloads
its encoding files on first use from `openaipublic.blob.core.windows.net`. If
you're behind a strict firewall, either allow that host once or run length
assertions with `unit: chars` / `unit: words`.

## Run

```bash
# Liveness ping against one provider:
promptfoo eval --filter-providers ollama_local_llama31 --filter-pattern smoke

# Full matrix:
promptfoo eval --cache

# Inspect results in the web UI (opens http://localhost:15500):
promptfoo view results/latest.json
```

## Comparing two providers

Compare the llm-rubric quality of a candidate provider against a frozen
baseline (e.g. prod). Only the non-deterministic rubric suites
(`research_rubrics`, `agentharm_refusal`) are compared.

1. **Freeze the baseline** — make the baseline provider active in
   `promptfooconfig.yaml`, run the suite, then freeze the result:

   ```bash
   promptfoo eval --cache
   scripts_repo/freeze_baseline.py results/local/latest.json
   # -> baselines/<provider_label>.json   (committed reference)
   ```

   Re-run this only when prod changes. Commit the baseline file.

2. **Run the candidate** — switch the active provider, run again:

   ```bash
   promptfoo eval --cache
   ```

3. **Compare** — diff candidate against the frozen baseline:

   ```bash
   scripts_repo/compare_runs.py \
       baselines/<provider_label>.json results/local/latest.json \
       --tolerance 0.05 --out report.html
   open report.html
   ```

   The report groups per-assertion deltas by suite, worst-first, and colors
   each cell green (improved beyond tolerance), red (regressed), or grey
   (within the ±tolerance band). Moves smaller than `--tolerance` (default
   0.05 on the 0–1 score) are treated as noise.

   Each baseline/candidate score links to that test's filtered view in the
   promptfoo web UI (`--ui-base-url`, default `http://localhost:3000`). Start
   the UI yourself (e.g. `promptfoo view`) for the links to resolve.

   Each baseline/candidate cell also has a clipboard button that copies a
   `curl` command to replay that test directly against the provider endpoint.
   The endpoint URL, key and params are read from the provider YAML referenced
   in each eval's config; override with `--baseline-url` / `--candidate-url`.

### Per-test matrix view

`compare_matrix.py` is an alternative report aimed at debugging a run. For each
test in the union of the baseline and latest run it draws a 2×N table (rows:
baseline, latest; columns: each assertion). The latest row is colored
red/white/green for worse/same/better than baseline. A test absent from a run
shows `missing`; an assertion whose grading errored (provider/response failure,
or an llm-rubric judge failure such as a missing AWS/Bedrock token) shows
`ERROR` rather than a misleading 0. Tests containing errors float to the top.

```bash
scripts_repo/compare_matrix.py --out matrix_report.html
# defaults: --baseline = the sole file in baselines/, --latest results/local/latest.json
open matrix_report.html
```

## Launching vLLM

```bash
./scripts/start_vllm.sh
```

Reads `.env` for `VLLM_MODEL_ID`, `VLLM_PORT`, and (optionally)
`VLLM_REASONING_PARSER`. The reasoning parser is required for thinking models
(`deepseek_r1`, `qwen3`, etc.) — without it, reasoning text is mixed into the
answer field and reasoning assertions cannot read it.

## Running tests for custom assertions

```bash
pytest tests/python                              # Python assertion unit tests
node --test hooks/normalize_response.test.js     # JS transform unit tests
```

## Adding a model

Create one file under `providers/` and add a `- file://providers/<name>.yaml`
line in `promptfooconfig.yaml`. Reference by `label:` from any test that
should target it specifically.

## Adding a system-prompt variant

Drop a `.txt` file in `system_prompts/` and reference it from a test:

```yaml
vars:
  system: file://system_prompts/<your_variant>.txt
```

For matrix coverage, list variants in a `scenarios:` block. See
`tests/content_quality.yaml` for an example.

## TODO

- Configure an LLM-as-judge (cloud or local) for `llm-rubric` assertions —
  see the comment in `promptfooconfig.yaml`.
- Add a CI workflow if/when this is moved into a git repo.

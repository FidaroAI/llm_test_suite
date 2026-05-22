# Design: System-Prompt / Config Comparison Orchestrator

Date: 2026-05-22

## Problem

Configuring the Fidaro gateways (system prompt, model params, model choice) is
disconnected from running the promptfoo tests. Tests cannot tell whether the
gateways are up, nor with what configuration. Ad-hoc shell scripts
(`fidaro_compare.sh`, `hack.sh`) run tests but do not capture or provision the
configuration a run was executed against, and their outputs collide or get
overwritten.

We want one Python entry point that a developer runs with a single config file.
It provisions the gateways (and, when needed, redeploys the Phala dev CVM with
new vLLM options), runs the prod-vs-dev comparison, records exactly what
configuration produced each run, and keeps every run's outputs isolated.

## Goals

- Single command: `run_comparison.py <path-to-config.json>`.
- A config file fully describes a comparison run (prod/dev endpoints, optional
  dev system prompt, optional Phala redeploy with vLLM options, promptfoo
  filters, suite-generation config).
- Each named comparison gets its own output directory; runs never overwrite
  each other.
- Redeploy the Phala dev CVM only when the vLLM options actually changed,
  decided via a cache file, and always with explicit user confirmation.
- Do not freeze a baseline. Compare two fresh runs directly.

## Non-goals

- No changes to existing scripts (`run_plaintext_gateway.sh`,
  `fidaro_compare.sh`, `compare_runs.py`, etc.). New scripts only.
- No attempt to make promptfoo providers capture the full run configuration.
- No prod redeploy. The orchestrator only ever touches the dev CVM.

## Components

- **`scripts_repo/run_comparison.py`** — CLI orchestrator. Validates config,
  manages the redeploy decision, starts gateways, runs both test passes,
  produces the comparison report, and brings up the promptfoo viewer.
- **`scripts_repo/deploy_phala.py`** — standalone, importable Phala deploy step:
  mutate the compose file's `vllm.command` from the supplied options, run
  `phala deploy`, and poll for readiness. Usable on its own.
- Pure logic (config validation, options→command mapping, cache comparison,
  compose mutation, filter assembly) lives in small testable functions.
  Side-effecting work (docker, network, subprocess) is kept in thin wrappers.
- Tests in `scripts_repo/tests/`.

## Config schema

Keys are kebab-case to match the requested spec.

| key | required | meaning |
|---|---|---|
| `vllm-prod-url` | yes | Base URL of prod vLLM. Error if missing. |
| `vllm-dev-url` | yes | Base URL of dev vLLM. Error if missing. Also the readiness-poll target after redeploy. |
| `suite-generation-config` | yes | Nested object replacing `SUITE_GENERATION_CONFIG_FILE`. Error if missing. |
| `system-prompt-file` | no | Path to a system prompt mounted into the **dev** gateway only. Must exist if given. |
| `phala-dev-instance-id` | no | CVM id of the dev Phala instance. If absent, the dev instance is left unchanged. |
| `vllm-options` | no | Object of vLLM CLI params (model name + params). Requires Phala redeploy. **Error if present without `phala-dev-instance-id`.** |
| `promptfoo-filters` | no | Subobject of promptfoo filters (`filter-metadata`, etc.). `filter-providers` is reserved (see below). |

`BRAVE_API_KEY` is never in the config; it stays an environment variable.

### Validation rules

- `vllm-prod-url`, `vllm-dev-url`, `suite-generation-config` all required.
- `vllm-options` present without `phala-dev-instance-id` → error.
- `system-prompt-file`, if present, must exist.
- If `vllm-options` is present, additionally require:
  - env `PHALA_DOCKER_COMPOSE_FILE` set and the file exists;
  - `.env.phala` present in the repo root.
- If `promptfoo-filters` contains `filter-providers`, ignore it with a warning
  (the orchestrator controls providers to do prod vs dev).

## Flow

1. Parse the config path; derive the comparison name from the filename stem
   (`prod_vs_dev_gemma.json` → `prod_vs_dev_gemma`).
2. Create `comparisons/<name>/`.
3. Validate the config.
4. Write `suite-generation-config` to
   `comparisons/<name>/suite_generation_config.json` and export
   `SUITE_GENERATION_CONFIG_FILE` pointing at it.
5. **Redeploy decision** (only if `vllm-options` present):
   - Read `comparisons/<name>/vllm_options_cache.json`.
   - If the cache exists and equals the current options → skip redeploy.
   - Otherwise warn the user and ask for confirmation (always — this is an
     infrastructure action; a `--yes` flag bypasses for automation). On
     confirmation, call `deploy_phala.py`; on success write the cache (an
     identical copy of the options json).
6. **`deploy_phala.py`**:
   - Copy `PHALA_DOCKER_COMPOSE_FILE` → `comparisons/<name>/deployed_compose.yaml`.
   - Inject `vllm-options` into the `vllm` service's `command` list, preserving
     `--host 0.0.0.0` and `--port 8000`.
   - Run `phala deploy --cvm-id <id> --compose comparisons/<name>/deployed_compose.yaml -e .env.phala`.
   - Poll `GET <vllm-dev-url>/v1/models` until HTTP 200 (long timeout, minutes;
     log progress).
7. **Start gateways in Python** (mirroring `run_plaintext_gateway.sh`'s docker
   invocation; the existing script is not edited):
   - prod gateway → `127.0.0.1:8082`, `HOST_OPENAI_BASE_URL=vllm-prod-url`.
   - dev gateway → `127.0.0.1:8084`, `HOST_OPENAI_BASE_URL=vllm-dev-url`; if
     `system-prompt-file` set, add
     `-v <abs>:/app/src/llm_gateway/prompts/core_system_prompt.md:ro` (dev only).
   - `BRAVE_API_KEY` from env. Poll each gateway's `/v1/models` until ready.
8. **Run tests** (no baseline freeze), each into the comparison dir with a
   timestamp:
   - prod → `comparisons/<name>/prod_results_<YYYYMMDD-HHMMSS>.json`,
     `--filter-providers fidaro_plaintext_gateway_phala_prod`.
   - dev → `comparisons/<name>/dev_results_<YYYYMMDD-HHMMSS>.json`,
     `--filter-providers fidaro_plaintext_gateway_phala_dev`.
   - Both passes also receive the pass-through `promptfoo-filters`.
9. `compare_runs.py <prod_results> <dev_results> --out comparisons/<name>/report__<YYYYMMDD-HHMMSS>.html`
   (report written into the same directory as the run results), then `open` it.
10. Ensure the promptfoo viewer container is up via `run_promptfoo_docker.sh`.

## vLLM options → command mapping

`vllm-options` is an object of vLLM CLI flags. Mapping:

- `"key": "value"` → `--key value`
- `"key": true` → `--key` (bare flag)
- `"key": false` → flag omitted

`--host 0.0.0.0` and `--port 8000` are always preserved regardless of options.

Example:

```json
{ "model": "google/gemma-4-31B-it", "reasoning-parser": "gemma4",
  "enable-auto-tool-choice": true, "tool-call-parser": "gemma4" }
```

→ `--host 0.0.0.0 --port 8000 --model google/gemma-4-31B-it --reasoning-parser gemma4 --enable-auto-tool-choice --tool-call-parser gemma4`

## Error handling

- Missing required keys, or `vllm-options` without `phala-dev-instance-id`:
  fail fast with a clear message before any side effects.
- Missing `PHALA_DOCKER_COMPOSE_FILE`, compose file, or `.env.phala` when a
  redeploy is required: fail fast before deploying.
- Readiness polls have a long but bounded timeout; on timeout, fail with the
  last observed status.
- Phala deploy is gated behind explicit confirmation (or `--yes`).

## Testing

Unit tests for the pure logic:

- Config validation (each required key, the `vllm-options` /
  `phala-dev-instance-id` rule, file-existence checks).
- vLLM options → command list mapping (string, bare-flag, false-omit,
  host/port preservation).
- vLLM options cache comparison (equal vs changed, normalized JSON).
- Compose mutation (replaces only the `vllm` service command; other services
  untouched).
- promptfoo filter assembly (pass-through, `filter-providers` reserved).

Side-effecting steps (docker run, phala deploy, network polls, browser open)
are thin wrappers exercised manually / left for integration.

## Caveats

- The dev system prompt must keep the websearch-prompt placeholder the gateway
  expects (see repo README gotcha); the orchestrator does not validate prompt
  content.
- `phala deploy` and CVM readiness can take many minutes; the orchestrator
  blocks and logs progress.

# CLAUDE.md

## What this is

A [promptfoo](https://www.promptfoo.dev)-powered regression test suite for
self-hosted / OpenAI-compatible LLMs. It runs a fixed battery of prompts and
assertions across a provider matrix (vLLM, the Fidaro plaintext gateway, etc.)
to catch quality and behavior regressions.

## Layout

- `promptfooconfig.yaml` — top-level config: active providers, prompts, tests.
- `providers/` — one YAML per model/endpoint. Add a file and reference it from
  `promptfooconfig.yaml` to extend the matrix.
- `prompt_templates/` — chat prompt templates (`{{system}}` / `{{user}}` vars).
- `system_prompts/` — system-prompt variants used as a `vars` value.
- `tests/` — test cases (YAML or CSV), grouped by concern.
- `assertions/` — custom Python assertions filling promptfoo gaps.
- `hooks/` — response transforms / per-test reconfiguration hooks.
- `scripts_repo/` — helper scripts (e.g. launching the gateway / vLLM).
- `results/` — eval JSON artifacts (gitignored).

## Common commands

```bash
promptfoo eval --cache                 # run the configured matrix
promptfoo view results/local/latest.json   # inspect results in the web UI
pytest tests/python                    # unit-test custom Python assertions
```

## Setup

Requires Node ≥ 20.20 and Python ≥ 3.11. Install promptfoo, create a venv and
`pip install -r requirements.txt`, then `cp .env.example .env` and fill in
endpoint URLs / keys. See `README.md` for full details.

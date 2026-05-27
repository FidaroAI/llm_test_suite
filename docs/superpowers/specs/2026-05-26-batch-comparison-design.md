# Batch comparison runner — design

## Purpose

Run many `run_comparison.py` comparisons that differ only by system prompt,
without hand-writing a config per prompt. Given a directory of system prompts
and one template config, generate a comparison config per prompt and
(optionally) run each through the existing `run_comparison.py` orchestrator.

`run_comparison.py` is **not** modified. Its one-prod-one-dev config structure
is reused as-is.

## CLI

```
batch_comparison.py --system-prompts-directory DIR --template-config FILE
                    [--generate-only] [--output-directory DIR]
                    [<args forwarded to run_comparison.py> ...]
```

Required:
- `--system-prompts-directory` — every `*.md` file in this directory (no
  recursion) is treated as a system prompt.
- `--template-config` — an existing comparison config (e.g.
  `comparisons/example.json`) used as the template for every generated config.

Optional:
- `--generate-only` — generate the configs and stop; do not run them.
- `--output-directory` — where generated configs are written. Defaults to
  `--system-prompts-directory`.

Pass-through: any unrecognised args are forwarded verbatim to every
`run_comparison.py` invocation (e.g. `--yes`, `--skip-phala-deploy`).

## Behaviour

1. Glob `*.md` in `--system-prompts-directory`, non-recursive, sorted. Exit
   with an error if none are found.
2. Load `--template-config` as JSON once.
3. For each prompt file `foo.md`:
   - Deep-copy the template.
   - Set `system-prompt-file` to the prompt's **absolute path**.
   - Write to `<output-dir>/foo.json`, 2-space indented (matching
     `example.json`).
4. If `--generate-only`: print the generated config paths and exit 0.
5. Otherwise, for each generated config, run
   `[sys.executable, <repo>/scripts_repo/run_comparison.py, <config>, *forwarded]`
   as a subprocess. On non-zero exit: log the failure and continue to the next
   config (continue-on-error). Print a final summary: generated / succeeded /
   failed counts. Exit non-zero if any run failed.

## Design choices

- **Config filename = prompt stem.** `run_comparison.comparison_name()` derives
  the comparison name (and thus its `comparisons/<name>/` result directory) from
  the config filename stem. Naming each generated config after its prompt
  (`foo.md` -> `foo.json`) gives each prompt an isolated result directory for
  free.
- **Absolute `system-prompt-file` path.** `run_comparison.validate_config`
  checks `Path(prompt_file).is_file()` against the process CWD; an absolute path
  is robust regardless of where the batch is invoked from.
- **`--output-directory` controls generated *config* location only.** Per-run
  *results* still land in `comparisons/<stem>/` because `run_comparison.py`
  hardcodes that and is intentionally left untouched. Documented in the script
  docstring.
- **Subprocess, not in-process import.** Isolates each comparison's side
  effects (`os.environ` mutation, Docker, `sys.exit`) so one failure cannot
  poison the batch.

## Testing

Pure generation path (no Docker/Phala) is unit-tested: given a template config
and a temp directory of `.md` files, assert that one `<stem>.json` is produced
per prompt, that `system-prompt-file` is overwritten with the prompt's absolute
path, and that all other template keys are preserved verbatim.

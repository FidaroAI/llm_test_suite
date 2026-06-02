# Raw-response CSV export for `compare_runs.py`

**Date:** 2026-06-01
**Status:** Approved

## Problem

`compare_runs.py` produces an HTML report of *graded* results across providers.
For qualitative analysis we also want the **raw responses** side by side, with no
test results and no model reasoning — just the final answer text per provider,
per prompt, in a spreadsheet-friendly form.

## Output

A CSV, one row per test:

| prompt | `<baseline-key>` | `<other-key>` | … |
|--------|------------------|---------------|---|
| rendered user message | baseline's stripped answer | other provider's stripped answer | … |

- Header row: `prompt`, then each provider's **config key**, baseline first —
  identical order and names to the HTML report's columns.
- Columns 2+ are providers in the report's `ProviderColumn` order.
- A provider that did not run (or errored with no output) for a test → empty cell.
- Rows ordered by `(suite, test_identity)` for determinism.

## Reasoning strip

Stored `response.output` is **pre-transform**: it still contains the reasoning
prefix (e.g. `Thinking: …`). The repo strips this before asserting via
`hooks/strip_before_triple_newline.py` — drop everything up to and including the
first `\n\n\n`. The CSV applies the same rule. A new local helper
`strip_reasoning(output)` mirrors that hook (non-string outputs returned
unchanged); a comment cross-references the hook as the source of truth (the hook
cannot be cleanly imported from `scripts_repo/` without path manipulation).

## Components (all in `scripts_repo/compare_runs.py`)

1. `strip_reasoning(output)` — mirror of the triple-newline transform.
2. `extract_responses(eval_json, provider_label)` → `{test_identity: stripped_output}`,
   reusing `_test_identity` and the same label-split the report uses. Captures the
   rendered prompt (`_prompt_text`) per test for column 1.
3. `build_response_csv(eval_json, columns, out_path)` — join per-provider response
   maps on test identity; write header + rows with the `csv` module (handles
   commas / quotes / newlines in responses).
4. `--csv PATH` CLI arg, wired into the N-provider path (`_main_n`, the path
   `run_comparison.py` drives) and the pairwise `main`.

## Integration

`run_comparison.py` always passes `--csv <run_dir>/responses__<ts>.csv` alongside
the existing `--out report__<ts>.html`, so every comparison run emits the CSV as a
standard artifact.

## Testing

- Unit: `strip_reasoning` (delimiter present / absent / non-str).
- Unit: `build_response_csv` against a small synthetic eval dict — join across
  providers, missing cell → empty, multiline/comma/quote responses quoted correctly.
- Smoke: run against `comparisons/example/run_20260601-143209/` and inspect.

## Out of scope

Test results, scores, deltas, reasoning text — none appear in the CSV.

# select-best head-to-head in comparison runs

**Date:** 2026-05-27
**Status:** approved

## Goal

When `run_comparison.py` runs its single unified prod-vs-dev eval, also grade a
`select-best` head-to-head per test: ask the judge which side answered the
user's prompt better, and surface the winner as a new **best** column in the
HTML comparison report. No extra eval pass; everything else behaves exactly as
today.

## Background

`run_comparison.py` already runs **one** promptfoo eval with both dynamic
providers (`..._dynamic_prod`, `..._dynamic_dev`) selected via
`--filter-providers`. promptfoo's `select-best` assertion compares every
provider's output for a test, asks the grader to pick the best, marks the
winner's assertion `pass` and the loser's `fail`. Because both sides run in the
same eval, `select-best` is meaningful here with zero extra generations — it
costs one extra grader (Bedrock) call per test.

Two non-obvious facts drove the design:

1. **The built-in `select-best` grading template never sees the prompt.** It
   only interpolates `{{criteria}}` and `{{outputs}}`. A question-relative
   criterion would be ungrounded. The grader *does* render with `test.vars` in
   scope, and this repo binds the prompt to `user`
   (`prompt_templates/user_only.json`), so a **custom `rubricPrompt`** embedding
   `{{ user }}` hands the judge the question. (Verified in promptfoo's
   `graders-*.js` / `evaluator-*.js`.)
2. **`select-best` is a real assertion** — the loser gets a failed component in
   the raw results. Left alone, `compare_runs.py` would fold it into the
   deterministic pass/fail tally. The report must treat it as its own kind and
   keep it out of the existing summaries.

## Design

### 1. Inject only during a comparison run (`tests/classification.py`)

`augment` is the one funnel every suite's tests pass through. When the env var
`COMPARISON_SELECT_BEST` is set, `augment` appends a `select-best` assertion
(idempotently — skip if one is already present) before attaching the shared
reasoning-strip transform, so the judge compares the stripped final answers.

The assertion:

```python
{
    "type": "select-best",
    "value": SELECT_BEST_CRITERION,        # rendered as {{criteria}}
    "rubricPrompt": SELECT_BEST_RUBRIC_PROMPT,  # custom; embeds {{ user }}
}
```

- `SELECT_BEST_CRITERION` — one shared, question-relative criterion for every
  suite.
- `SELECT_BEST_RUBRIC_PROMPT` — custom template that shows the judge the
  `{{ user }}` prompt, the numbered `{{ outputs }}`, and `{{ criteria }}`, and
  asks for the single integer index of the best.
- No `provider` on the assertion: it falls back to the `defaultTest` Bedrock
  judge, same as every rubric.

`run_comparison.py` sets `COMPARISON_SELECT_BEST=1` before the eval (alongside
the existing `COMPARISON_*` env setup). `fidaro.sh` / CI run the same config
*without* that var, so single-provider runs are byte-for-byte unchanged.

### 2. Surface the winner (`scripts_repo/compare_runs.py`)

- `extract_cells`: recognise `type == "select-best"` as a new `kind="best"`
  cell carrying the per-provider `pass`.
- The existing summaries already count only `rubric` / `deterministic` kinds, so
  `best` cells are naturally excluded from them. Add a small **best** summary
  line (prod won N · candidate won M · undecided K) at the aggregate and
  per-suite level.
- `diff_cells`: a `best` cell joins on its criterion text across the two sides;
  the winner is whichever side's component passed.
- `render_html`: add a **best** column. A `best` row shows the winner
  (`prod` / `candidate`) in that column, an em dash in
  baseline/candidate/Δ/status, and `select-best` as its assertion label. Every
  other row shows an em dash in the best column. Winner is `—` when it can't be
  determined (a side errored / missing).

### 3. Cleanup

Remove the leftover Git merge-conflict markers in the `run_comparison.py` module
docstring (they describe the superseded two-pass flow); keep the unified-eval
wording that matches current behaviour.

## Testing

- `test_classification.py`: with `COMPARISON_SELECT_BEST` set, `augment` appends
  exactly one select-best assertion with the custom rubricPrompt and the shared
  transform; with it unset, asserts are untouched; idempotent when one already
  exists.
- `test_compare_runs.py`: `extract_cells` tags select-best as `kind="best"`;
  `diff_cells` picks the winner from the passing side; summaries ignore best
  cells; `render_html` emits a best column with `prod`/`candidate`.
- `test_run_comparison.py`: the comparison run sets `COMPARISON_SELECT_BEST`.

## Out of scope

- Per-suite criteria (single shared criterion for now).
- The unused `tests/ab_select_gen.py` draft and any `ab_select_config.yaml`.

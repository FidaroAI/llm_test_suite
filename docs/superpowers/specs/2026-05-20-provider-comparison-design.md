# Provider comparison harness — design

**Date:** 2026-05-20
**Status:** Approved (design); implementation pending

## Problem

We want to compare the LLM-rubric (non-deterministic) quality of two
OpenAI-compatible plaintext gateways: a **baseline** (current prod app) and a
**candidate** (some other endpoint). We need to see *at a glance* which rubric
scores went up and which went down relative to the baseline, ignoring moves
small enough to be grader noise.

promptfoo is built to gate each test against a fixed `threshold`, not to diff
one provider's scores against another's. This design adds a thin comparison
layer on top of promptfoo without changing how tests are authored.

## Decisions (from brainstorming)

- **Baseline model: frozen artifact.** Run prod once, save the result JSON as a
  committed reference. Diff every future candidate against that same frozen
  file. The baseline is stable; only the candidate is re-measured. Re-baseline
  explicitly when prod changes. (Rejected: a joint single run re-grades the
  baseline every time, so its scores drift with grader noise.)
- **Granularity: per-assertion.** One delta per individual `llm-rubric`
  criterion — the finest level. Aggregate roll-ups are out of scope for v1.
- **Tolerance: absolute band** on the 0–1 score (default `0.05`). A move smaller
  than the band in either direction is "within tolerance". Predictable and
  uniform across cells; no special-casing for zero baselines. (Rejected:
  relative %-of-baseline, which explodes when the baseline is 0.)
- **Scope filter: by `metadata.suite`,** not by assertion type. Default
  allowlist `{research_rubrics, agentharm_refusal}`, overridable with a
  repeatable `--suite` flag. This is more intentional than sniffing assertion
  types and naturally excludes the deterministic CSV fact tests (which carry no
  `suite` metadata).
- **Single run, no repeats** for v1 (see Future work).
- **Baseline lives in a committed top-level `baselines/` directory**, NOT in
  `results/` (which is gitignored, local-run scratch).
- **Report: grouped, color-coded long-format table**, not a strict rectangular
  matrix (see "Report rendering" for why).

## Workflow

```
1. Freeze baseline (prod provider active in promptfooconfig.yaml):
     promptfoo eval --cache
     scripts_repo/freeze_baseline.py results/local/latest.json
       -> baselines/<provider_label>.json   (provenance-stamped)

2. Run candidate (candidate provider active):
     promptfoo eval --cache
       -> results/local/latest.json

3. Compare:
     scripts_repo/compare_runs.py \
         baselines/<provider_label>.json \
         results/local/latest.json \
         --tolerance 0.05 --out report.html

4. Open report.html.
```

The comparison rides entirely on promptfoo's **provider** axis. The repo already
has two suitable provider configs (`providers/fidaro_plaintext_gateway_phala.yaml`
= prod/baseline, `providers/fidaro_plaintext_gateway_local.yaml` = candidate);
no per-test changes are needed. Which provider is "baseline" vs "candidate" is
just which JSON you pass first vs second to `compare_runs.py`.

## Components

Two new Python files in `scripts_repo/`, matching the existing argparse +
module-docstring + pure-helper-functions conventions (cf. `pull_ci_results.py`):

### `freeze_baseline.py`

A thin copy-with-provenance tool. Reads an eval result JSON, writes it to
`baselines/<provider_label>.json` (label sanitized for the filesystem; multiple
providers in one eval each get their own file). Adds a small `_baseline_meta`
block recording:

- `provider_label`
- `frozen_at` (ISO timestamp)
- `git_sha` (HEAD of the suite at freeze time)
- `eval_id` (promptfoo `evalId`)
- `test_keys` (sorted list of the test identities included; see below)

`test_keys` lets `compare_runs.py` warn when the candidate run covers a
different set of tests than the baseline.

Flags: positional `result_json`; `--out-dir` (default `baselines/`); `--force`
to overwrite an existing baseline.

### `compare_runs.py`

The diff engine + HTML renderer. Pure, testable helpers do the work; a thin
`main()` wires argv → load → join → classify → render.

**Inputs:** `baseline_json`, `candidate_json` (positional);
`--suite SUITE` (repeatable; default `research_rubrics`, `agentharm_refusal`);
`--tolerance FLOAT` (default `0.05`); `--out PATH` (default `report.html`).

## Matching & diff logic

A **cell** is one `llm-rubric` assertion's score for one test, on one side.

- **Suite filter:** keep only results whose `testCase.metadata.suite` is in the
  allowlist. Everything else (CSV facts, etc.) is dropped before matching.
- **Test key:** the test `description`, which the generators already make unique
  and human-readable (`researchrubrics[<domain>] <sample_id>`,
  `agentharm[<category>] <id> <name>`), combined with the prompt label. Stable
  across runs and independent of row order.
- **Assertion key (within a test):** the rubric text (`assertion.value`), unique
  per criterion. Fall back to positional index on the rare exact-text collision.
- **Score source:** `gradingResult.componentResults[i].score` for each assertion,
  on the 0–1 scale.
- **Delta:** `candidate_score − baseline_score`.
- **Classification** against the absolute tolerance band `t`:
  - `delta >  t`  → **improved** (green)
  - `delta < −t`  → **regressed** (red)
  - otherwise     → **within tolerance** (grey)
- **Ragged / unmatched cells:** a test or assertion present on only one side is
  marked **new** (candidate-only) or **removed** (baseline-only) — never a crash.
  This happens legitimately when dataset limits (`RESEARCH_RUBRICS_LIMIT`,
  `AGENTHARM_LIMIT`) or the dataset itself differ between the two runs.
- **Config-drift warning:** if the two runs' test-key sets differ, the report
  shows a banner listing the symmetric difference, so a misaligned comparison is
  obvious rather than silent.

## Report rendering

A strict rectangular "rows = tests × columns = assertions" matrix only works
when assertions are shared across tests. That holds for `agentharm_refusal`
(one uniform `non_refusal` assertion per test) but **not** for
`research_rubrics`, where every row has different criteria — a global column set
would be almost entirely empty.

So the report is a **grouped, color-coded long-format table**, sorted
worst-delta-first within each suite, with a summary line for the at-a-glance
read:

```
[ Summary:  12 improved · 5 regressed · 40 within ±0.05 · 2 new · 0 removed ]
[ ⚠ config drift: candidate is missing test 'researchrubrics[law] s_204' ]   (only if sets differ)

▼ research_rubrics
  test                          assertion (axis)            base   cand     Δ
  researchrubrics[fin] s_018    "cites a source…" (rigor)   0.80   0.40   −0.40  🟥
  researchrubrics[fin] s_018    "covers risks…" (breadth)   0.50   0.55   +0.05  ⬜
  …
▼ agentharm_refusal
  agentharm[fraud] 7a           non_refusal                 0.90   1.00   +0.10  🟩
```

- Color: red = regressed past tolerance, green = improved past tolerance, grey =
  within band; distinct styling for `new` / `removed`.
- Self-contained HTML with inline CSS, no JavaScript and no external assets, so
  the file can be opened or emailed directly.
- "At a glance" = the summary counts + the worst-first sort surfacing the
  biggest regressions at the top of each group.

## Testing

Unit tests in `scripts_repo/tests/` (existing pattern), **no network, no API
keys**. Drive the pure helpers with small synthetic result-JSON fixtures and
assert on:

- the suite filter (CSV/no-suite results excluded; allowlist honored),
- test-key and assertion-key extraction (including the description/prompt-label
  composite and the index fallback on duplicate rubric text),
- delta computation and the three-way tolerance classification, exactly at the
  `±t` boundary,
- ragged handling: a baseline-only assertion → `removed`, a candidate-only
  assertion → `new`,
- the config-drift detection (differing test-key sets).

Rendering is exercised by a smoke test asserting the HTML contains the expected
summary counts and per-cell markers for a known fixture; we don't snapshot exact
markup.

## Future work

- **Noise dampening via repeats.** v1 runs each test once. A later option can use
  promptfoo's `repeat` to run each test N times; `compare_runs.py` will already
  average duplicate cells (same test key + assertion key), so adopting it is
  mostly a config + flag change rather than an engine change.
- **Aggregate roll-ups** (per-axis / per-test) on top of the per-assertion data,
  if the per-cell view proves too granular for a quick read.

## Out of scope

- Changing how tests are authored or how promptfoo grades.
- CI gating on regressions (this is a reporting tool, not a gate).
- Comparing deterministic assertions (fact CSVs, token/tool-count Python asserts).
```

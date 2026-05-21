# compare_runs: compare all suites + support deterministic tests

Date: 2026-05-21

## Goal

1. Remove the hardcoded suite allowlist (`DEFAULT_SUITES`) from
   `compare_runs.py` — compare every test present in the baseline and
   candidate runs; the existing missing-test (drift) logic handles asymmetry.
2. Support deterministic tests (e.g. `simple_facts.csv` `icontains` asserts)
   alongside the llm-rubric suites, with appropriate summaries and columns.

## CSV tagging

Add a `__metadata:suite` column (promptfoo CSV metadata syntax) to
`tests/simple_facts.csv` and `tests/simple_facts_regressions.csv`, value
`simple_facts` on every row, so those tests carry `metadata.suite`.

Any test still lacking `metadata.suite` (e.g. `fidaro_basic.yaml`) is grouped
under a fallback heading `(no suite)` rather than dropped.

## Cell model

Add `kind` to `Cell` / `CellDiff`: `"rubric"` or `"deterministic"`.

- For each assertion index-aligned with a `componentResults[i]`:
  - `type == "llm-rubric"` → **rubric** cell with numeric `score` (unchanged).
  - any other type (`icontains`, `python`, …) → **deterministic** cell with
    `passed: bool`, taken from `componentResults[i]["pass"]` (fallback
    `score >= 0.5`).
- Previously non-rubric assertions were skipped; they now become deterministic
  cells (intended).

`extract_cells` / `errored_tests` no longer filter by suite. `suite =
meta.get("suite") or "(no suite)"`.

## Classification

- Rubric (unchanged): `improved`/`regressed`/`within` by tolerance; `new`/
  `removed` for a missing side. `delta` is numeric.
- Deterministic: `improved` (fail→pass), `regressed` (pass→fail), `same`
  (unchanged); `new`/`removed` for a missing side. `delta` is always `None`.

## HTML report

One table per suite (rows of either kind). Columns:
`test · assertion · metric · baseline · candidate · Δ · status`.

- baseline/candidate: rubric → `0.50`; deterministic → `pass` / `fail`
  (keeping the promptfoo UI links and copy-as-curl buttons).
- Δ: blank (`—`) for deterministic rows.
- status: the values above.

### Summaries

- **Per suite** (above its table), only the kinds that have cells:
  - `Rubric Tests: A improved · B regressed · C within ±tol · D new · E removed`
  - `Deterministic Tests: P new passes · Q new fails · R total passes · S total fails`
    - new passes = `improved` (fail→pass), new fails = `regressed` (pass→fail),
      counted among tests present in both runs.
    - total passes / fails = candidate-side pass / fail counts.
- **Aggregate** at the top of the report: the same two lines summed across all
  suites; each line shown only if that kind has any cells.

## CLI

Keep `--suite` as an optional repeatable filter (default: all suites). Remove
the `DEFAULT_SUITES` default.

## Tests

Update `scripts_repo/tests/test_compare_runs.py`: drop the
`DEFAULT_SUITES`-based tests and suite-filtering assumptions; add coverage for
deterministic extraction, deterministic classification (fail→pass / pass→fail /
same), the dual per-suite summaries, the aggregate summary, and the pass/fail +
blank-Δ rendering. The `_fixtures.py` helper gains a `deterministic_result`
builder.

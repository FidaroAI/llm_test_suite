# Research Rubrics Tests — Design

## Goal

Add `llm-rubric`-based regression tests driven by the
[ScaleAI/researchrubrics](https://huggingface.co/datasets/ScaleAI/researchrubrics)
dataset. Each dataset row is a research-style user prompt paired with a list of
weighted rubric criteria. We feed the prompt to the model under test and grade
the single response against each criterion separately, using an LLM judge.

## Why this dataset

An earlier candidate (`reyavir/PromptEvals`) stored its prompts as **system
prompts**, which the Fidaro plaintext gateway always overrides — so the model
under test would never actually receive them. `ScaleAI/researchrubrics` stores
its `prompt` as a **user query**, which drops straight into the existing
`user_only` template and runs against the gateway with no conflict.

## Dataset shape

- 101 rows, single `train` split, JSON (HF auto-converts to Parquet).
- Columns:
  - `prompt` — the user query/task (goes in the `user` var).
  - `sample_id` — unique id.
  - `domain` — 10 string classes (e.g. "Current Events", "AI & ML").
  - `conceptual_breadth` — Simple / Moderate / High.
  - `logical_nesting` — Shallow / Intermediate / Deep.
  - `exploration` — Low / Medium / High.
  - `rubrics` — list (20–43 items) of `{criterion, weight, axis}` objects.
    `axis` examples: "Explicit Criteria", "Instruction Following",
    "References & Citation Quality".

## Key promptfoo facts this relies on

- **One generation, many assertions.** A test case runs the prompt against the
  provider once; every entry in its `assert:` list is graded against that same
  output. So one `llm-rubric` per criterion means N judge calls but a single
  call to the model under test.
- **Dynamic Python test generation.** `tests: file://gen.py:function_name`
  returns a list of test-case dicts (`vars`, `assert`, `metadata`, `threshold`).
- **Metadata filtering.** `metadata` on each test is filterable via
  `promptfoo eval --filter-metadata key=value`.
- **Weighted aggregate scoring.** Each assertion's `weight` feeds the test's
  weighted-average score; `threshold` decides pass/fail on that aggregate.

## Components

### 1. Download — one-off node script

- `scripts_repo/download_researchrubrics.mjs` — fetches all 101 rows from the HF
  datasets-server rows API (`https://datasets-server.huggingface.co/rows`),
  paginating (max 100 rows/request), and writes the raw rows untransformed to
  `data/researchrubrics.json`.
- Wired as `npm run dataset:researchrubrics` in `package.json`.
- Assumed to have been run before the tests are evaluated. All shaping happens
  in the Python generator, not here — the on-disk file stays raw.

### 2. Test generation — Python, promptfoo-native

- `tests/research_rubrics_gen.py` exposing `generate_tests()`.
- Reads `data/researchrubrics.json`. For each row emits a test case:
  - `vars: { user: <prompt> }` (existing `user_only` prompt + gateway).
  - `assert:` — one `llm-rubric` per rubric item:
    `{ type: "llm-rubric", value: <criterion>, weight: <weight>, metric: <axis> }`.
  - `threshold: 0.5` — test passes if the weighted rubric average ≥ 0.5.
  - `description` — derived from `sample_id` / domain for readable output.
  - `metadata: { suite: "research_rubrics", domain, conceptual_breadth,
    logical_nesting, exploration, sample_id }`.
- Honors env var `RESEARCH_RUBRICS_LIMIT` (default: all rows) to cap how many
  rows become test cases, for a small first run.
- If `data/researchrubrics.json` is missing, raises a clear error telling the
  user to run `npm run dataset:researchrubrics` first.

### 3. Config wiring

- `promptfooconfig.yaml`:
  - Add `- file://tests/research_rubrics_gen.py:generate_tests` to `tests:`.
  - Set the LLM judge under `defaultTest.options.provider` to
    `openai:codex-sdk` (fills in the existing TODO comment). This is the grader
    for `llm-rubric` assertions; it does not affect the non-graded
    `simple_facts` tests.
- The `suite: research_rubrics` metadata tag isolates these tests from the
  existing `simple_facts` / `fidaro_basic` tests, which lack that key, so
  `--filter-metadata "suite=research_rubrics"` runs only the new suite.

## First run / verification

```bash
npm run dataset:researchrubrics            # one-off download
RESEARCH_RUBRICS_LIMIT=1 promptfoo eval --filter-metadata "suite=research_rubrics" --cache
```

Expectation: a single prompt run against the gateway, ~20–43 `llm-rubric` calls
to `openai:codex-sdk`, aggregated into one weighted test score against the 0.5
threshold. Inspect with `promptfoo view`, then decide on larger subsets
(e.g. `--filter-metadata "domain=Current Events"`).

## Out of scope (for now)

- Tuning thresholds, per-axis pass criteria, or per-assertion thresholds —
  fixed at a lenient `threshold: 0.5` aggregate.
- CI grader configuration (`openai:codex-sdk` is the local judge for now).
- Whether the model under test can actually perform deep research / web search;
  this run validates the wiring, not model quality.
- Caching/transform changes — the existing `strip_before_triple_newline.py`
  transform (strips the thinking block) applies and is appropriate, since we
  grade the final answer.

## Risks / open items

- `openai:codex-sdk` provider id is taken from the user's environment; the first
  run validates that promptfoo accepts it. If not, swap the grader id.
- HF datasets-server may require the dataset to be auto-converted; if the rows
  API is unavailable we fall back to the raw repo file
  (`resolve/main/<file>`).
- 20–43 judge calls per row makes full-suite runs heavy; the
  `RESEARCH_RUBRICS_LIMIT` cap and metadata filters keep early runs cheap.
</content>
</invoke>

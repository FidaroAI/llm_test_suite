# Rewrite: generate all legacy suites

## Goal

The promptfoo-free rewrite (`rewrite/llmeval/`) can currently only generate the
`simple_facts` suite (via the generic `generate-csv` CLI command). Bring it to
parity with the legacy `tests/*_gen.py` generators so it can produce **all six**
suites as standardized, inspectable `TestCase` JSON in `testcases/`:

| Suite | Source | Old grader(s) | Notes |
|---|---|---|---|
| `simple_facts` | `simple_facts.csv` | `icontains` | already works via `generate-csv` |
| `simple_facts_regressions` | `simple_facts_regressions.csv` | `icontains` | CSV-backed |
| `agentharm_refusal` | `data/agentharm.json` | one fixed `llm-rubric` | `censorship: true` flag |
| `multifaceted` | `data/multifaceted.json` | N× `llm-rubric` (1–5 scale) | per-row rubrics |
| `research_rubrics` | `data/researchrubrics.json` | `llm-rubric` **and** `g-eval` variants | two tests per row |
| `stock_prices` | `stock_prices.csv` | custom `assert_stock_price` | live Stooq fetch, no-cache |

## Decisions (agreed)

- **Scope:** all six suites, including `stock_prices`.
- **Fidelity:** *map to the rewrite model* — generation-only. We do **not** add
  per-test weighted-average thresholds or the multifaceted 1–5 `rubricPrompt`
  override to the framework. Rubrics map to per-assertion `rubric`/`g_eval` with
  per-assertion thresholds (`params.threshold`, default 0.5) and metrics; the
  existing 0–1 rubric template grades them. `stock_prices` *does* need a new
  deterministic `stock_price` assertion — without it the suite cannot function;
  that is generation faithfulness, not the threshold/1–5 fidelity we are skipping.
- **Logistics:** work on the current `rewrite` branch, autonomous, leave
  committed on the branch (no PR/merge).

## Architecture

All new generation code lives under `rewrite/llmeval/generation/`. Generators are
pure transforms `source -> list[dict]` emitting the rewrite on-disk TestCase shape
(`{"id", "user", "assertions": [...], "metadata": {...}}`), written to
`testcases/<suite>.json`. Running stays strictly separate.

### New modules

- **`generation/config.py`** — port of `tests/suite_config.py`. `SuiteConfig`
  with `number_to_generate` / `randomize_selection` / `random_seed` /
  `max_rubrics` / `stratify`; `load(suite, path)` and `select(tests)`. Same
  DEFAULTS (off by default: `number_to_generate=0`). `select` shuffles (seeded),
  stratifies, caps, and stamps the resolved block onto each test's
  `metadata.config`. Operates on the rewrite dict shape (top-level `metadata`).

- **`generation/classification.py`** — port of `labels_for`. Reads
  `data/classifications/<suite>.json`
  (`{"classifications": {sha1(prompt): {request_type, domain}}}`), defaults to
  `unclassified`. Stamps `metadata.request_type` / `metadata.domain`. We do **not**
  port the promptfoo grading-transform attach (the rewrite applies
  `strip_reasoning` per-assertion already) nor the `select-best` env hook
  (head-to-head is the separate `pickbest` command).

- **`generation/agentharm.py`**, **`multifaceted.py`**, **`research_rubrics.py`**,
  **`stock_prices.py`** — one focused generator each (`generate(...) -> list[dict]`).

- **`generation/suites.py`** — registry mapping suite name → a spec describing how
  to build it (callable + default source paths). Drives the CLI `--suite`/`--all`.

### Assertion mapping

| legacy | rewrite |
|---|---|
| `icontains:X` | `{type: icontains, value: X}` (already handled) |
| `llm-rubric` (value, weight, metric) | `{type: rubric, value, weight, metric}` |
| `g-eval` | `{type: g_eval, value, weight, metric}` |
| `assert_stock_price` (python) | `{type: stock_price, params: {...}}` (new) |

### New `stock_price` deterministic assertion (`assertions/deterministic.py`)

Network-free, ported verbatim in spirit from `assertions/assert_stock_price.py`.
Reads `spec.params`: `reference_price`, `reference_currency`, `reference_fetched_at`,
`symbol`, optional `tolerance_pct` (default 1.0), `max_age_hours` (default 24).
Extracts numeric tokens from the answer, accepts a match within tolerance of the
reference (or reference/100 for `GBp`), and fails on a missing/stale reference.
Score 1.0/0.0.

### stock_prices generator specifics

- Port `stock_prices_ref.py` (Stooq fetch) into
  `generation/stock_prices.py` (or a sibling) — `fetch_all(csv_path)` →
  `(quotes, failures)`; the network call is isolated behind an injectable
  `fetch` callable so tests never hit the network.
- The generator reads `generation_sources/stock_prices.csv`, fetches live quotes
  (fail fast if any symbol fails), writes a snapshot, and **bakes** the reference
  into each test's `stock_price` assertion `params` (and `metadata`) at generation
  time — mirroring the legacy generator. Grading stays network-free.
- Freshness/no-cache: the rewrite caches by `(test, cache_key)`. Since the
  reference is baked at generation time and the suite is regenerated per run, a
  stale-reference grade fails by design. Document running stock_prices with
  `--mode always` (or a fresh DB) so a cached answer cannot mask a freshness miss.

### CLI (`cli.py`)

Add a `generate` subcommand alongside the existing `generate-csv`:

```
llmeval generate --suite multifaceted --out testcases/
llmeval generate --all --out testcases/
llmeval generate --suite stock_prices --out testcases/      # hits the network
```

Flags: `--suite NAME` (repeatable) or `--all`; `--out` (default `testcases/`);
`--config` (suite-generation config; env `SUITE_GENERATION_CONFIG_FILE` honoured);
`--data-dir` and `--classifications-dir` (defaults resolved to the repo-root
`data/`, with a note that full self-containment of the datasets is future work).

### Data sources

Datasets (`data/*.json`, 9 MB multifaceted) and `data/classifications/` stay at
the repo root; generators reference them via the configurable `--data-dir` /
`--classifications-dir`. CSV sources (`simple_facts*.csv`, `stock_prices.csv`)
are copied into `rewrite/generation_sources/` to match the existing convention.
Self-containment of the large datasets into the rewrite tree is out of scope.

## Fidelity notes (what differs from legacy)

- No per-test `threshold`; pass/fail is per-assertion (`params.threshold`,
  default 0.5) and aggregation is per-metric in `compare`/`report`.
- multifaceted: the 1–5 anchor descriptions are embedded into the rubric
  criterion text as context, but graded by the standard 0–1 rubric template
  (no per-test `rubricPrompt` override). Exact 1→5 normalization is not
  reproduced.
- Preserved: `metadata.config` (reproducibility), `request_type`/`domain`
  classification, agentharm `censorship: true`, research_rubrics `grader` +
  native provenance fields, multifaceted `source`, stock_prices symbol/currency.

## Testing (TDD)

`framework_tests/` gets per-generator tests using **tiny inline fixtures**, not
the real multi-MB datasets:

- agentharm / multifaceted / research_rubrics: 2–3 row fixture JSON; assert
  emitted assertion types, metrics, weights, metadata, id uniqueness (research
  emits two ids per row → grader-suffixed ids).
- config: selection (cap, seeded shuffle, stratify, `metadata.config` stamp).
- classification: hash lookup + unclassified fallback.
- stock_price assertion: within/over tolerance, GBp/100, stale, missing ref.
- stock_prices generator: injected fake fetch (no network), reference baking,
  fail-fast on a fetch failure.
- CLI `generate`: `--suite` and `--all` write expected files.

## Out of scope

- Per-test thresholds / 1–5 rubric override in the framework.
- The `select-best` env hook (covered by `pickbest`).
- Infra (gateways, redeploys), and moving the large datasets into the rewrite.
```

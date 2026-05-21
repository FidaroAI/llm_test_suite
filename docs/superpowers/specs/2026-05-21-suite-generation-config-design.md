# Config-driven test generation + random selection

Date: 2026-05-21

## Goal

Replace the per-suite environment variables that control Python test
generation with a single JSON config file, and add the ability to generate a
random selection of `N` tests with a fixable seed so a selection can be
recreated.

## Motivation

The three generators (`agentharm_refusal_gen.py`, `multifaceted_gen.py`,
`research_rubrics_gen.py`) each read their own env vars
(`MULTIFACETED_LIMIT`, `RESEARCH_RUBRICS_LIMIT`, `AGENTHARM_LIMIT`,
`AGENTHARM_START_INDEX`, `*_MAX_CRITERIA`). This is getting unwieldy and gives
no reproducible random-sampling capability. A single keyed config file scales
to future suites and centralizes the shared selection logic.

## Conventions

- Any `tests/<name>_gen.py` is a suite generator. `<name>` (filename minus the
  `_gen` suffix) is the **suite name**, used both as the config key and as the
  `metadata.suite` value. The derived names match today's hardcoded values:
  `agentharm_refusal`, `multifaceted`, `research_rubrics`.

## Config file

- Env var `SUITE_GENERATION_CONFIG_FILE` selects the file; default
  `tests/suite_generation_config.json`.
- Top level is keyed by suite name. Each suite maps to:
  - `number_to_generate` (int | null) — cap on emitted tests; null = all.
  - `randomize_selection` (bool) — shuffle before capping.
  - `random_seed` (int) — seed for the shuffle.
  - `max_rubrics` (int | null) — cap rubrics per row; null = all.
- Missing file or missing suite key ⇒ all defaults.

### Defaults

```
number_to_generate: null
randomize_selection: false
random_seed: 0
max_rubrics: null
```

`random_seed` default of 0 is used when `randomize_selection` is true but no
seed is given, keeping runs reproducible.

### Committed default file (`tests/suite_generation_config.json`)

```json
{
  "multifaceted":      { "number_to_generate": 50,   "randomize_selection": true,  "random_seed": 42, "max_rubrics": 5 },
  "research_rubrics":  { "number_to_generate": null, "randomize_selection": false, "random_seed": 0,  "max_rubrics": null },
  "agentharm_refusal": { "number_to_generate": 30,   "randomize_selection": true,  "random_seed": 7,  "max_rubrics": null }
}
```

## Shared module — `tests/suite_config.py`

Not a `*_gen.py` file, so never treated as a suite.

- `suite_name(file)` → derive suite from filename (strip `_gen` suffix).
- `load(file)` → `SuiteConfig`: read JSON (path from env or default), merge the
  suite's entry over `DEFAULTS`.
- `SuiteConfig` attributes: `suite`, `number_to_generate`,
  `randomize_selection`, `random_seed`, `max_rubrics`.
- `SuiteConfig.select(tests)`:
  1. If `randomize_selection`, `random.Random(random_seed).shuffle(copy)`.
  2. If `number_to_generate is not None`, take the first `N`.
  3. Stamp `metadata["config"]` (full nested 4-key dict) onto each selected
     test.
  4. Return the selected tests.

Generators import it by inserting their own directory onto `sys.path` (the
`tests/` dir is not a package and promptfoo loads each generator by file path).

## Generator changes

Each generator shrinks to:

1. `cfg = suite_config.load(__file__)`
2. Build candidate tests using `cfg.max_rubrics` for per-row rubric caps and
   `cfg.suite` for `metadata.suite`.
3. Filter out tests with no gradable asserts.
4. `return cfg.select(valid_tests)`.

Remove all `os.environ.get(...)` reads and the `AGENTHARM_START_INDEX` logic
(dropped — randomized selection covers that use case). Update docstrings to
reference the config file.

## Order of operations

`max_rubrics` caps rubrics within each row → build tests → drop empty ones →
seeded shuffle → take `number_to_generate` → stamp `metadata.config`. Selecting
from the *valid* set ensures `number_to_generate` yields that many runnable
tests.

## Metadata

Each emitted test gains `metadata.config = {number_to_generate,
randomize_selection, random_seed, max_rubrics}`. Nested JSON; no flattening
needed. Satisfies the requirement to capture the count and seed used.

## Tests (`tests/python`)

- `suite_name` derivation from a path.
- Default merge when file/key absent.
- Seeded-shuffle determinism: same seed ⇒ same selection; different seed ⇒
  (generally) different selection.
- `number_to_generate` slicing and null = all.
- `metadata.config` stamped on every returned test.

## Docs

Update generator docstrings; grep `README.md` / npm scripts for the removed env
var names and update references.

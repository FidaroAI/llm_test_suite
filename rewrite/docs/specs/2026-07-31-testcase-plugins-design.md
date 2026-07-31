# Test cases as plugins

**Date:** 2026-07-31
**Status:** agreed, not yet implemented

Replace the hardcoded suite registry with a plugin system. `generation_sources/` and
`testcases/` merge into a single `testcases/` directory whose subdirectories are
self-contained plugins that `llmeval` discovers and loads at runtime.

## 1. Why

`llmeval/generation/suites.py` holds a `SUITES: dict[str, SuiteSpec]` that names every
generator, and each generator lives inside the `llmeval` package. Three problems follow:

* **Adding a suite means editing the library.** A new dataset touches `suites.py`, a new
  module under `llmeval/generation/`, and — if it needs a bespoke check — the shared
  assertion registry in `llmeval/assertions/deterministic.py`. None of that is library
  concern; it is test-case concern.
* **Test cases have no lifecycle.** A suite whose reference data goes stale (stock prices)
  has nowhere to say "refresh this before grading". Today the freshness fetch happens at
  generation time and the answer is baked into the assertion params, which is why the
  assertion needs a staleness guard at all.
* **Inputs are scattered.** CSV sources in `generation_sources/`, datasets in the repo-root
  `data/` downloaded by Node scripts, classification labels in `data/classifications/`, a
  generation config at the rewrite root, and a custom assertion in the legacy suite's
  `assertions/`. The rewrite's own CLAUDE.md forbids reaching up into the legacy suite; the
  stock-price suite does it anyway.

The plugin system inverts the dependency. `llmeval` owns an interface; the test cases own
their logic, their inputs, their downloads, and their bespoke assertions.

## 2. Layout

```
rewrite/testcases/                        the only place test cases live
  examples.json                           hand-written, tracked
  simple_facts/
    __init__.py                           get_plugin(interface) -> TestCasePlugin
    simple_facts.csv
  simple_facts_regressions/
    __init__.py
    simple_facts_regressions.csv
  stock_prices/
    __init__.py
    stock_prices.csv
    stooq.py                              private helper, moved out of llmeval/
  agentharm_refusal/__init__.py
  multifaceted/__init__.py
  research_rubrics/__init__.py

rewrite/.testcases.cache/                 gitignored
  <plugin>/                               downloads + generated testcases.json
```

**A source is a plugin directory or a top-level `.json` file.** The rules:

* `.json` files are read only at the top level of `testcases/`. A `.json` inside a plugin
  directory is that plugin's private business.
* A `.json` stem may not equal a directory name. `simple_facts.json` alongside
  `simple_facts/` is a hard error, because the two would claim the same source name.
* A directory is a plugin iff it contains `__init__.py` exposing `get_plugin`. Any other
  directory produces a warning and is ignored.
* `__pycache__`, and directories whose names begin with `.` or `_`, are skipped silently.

### 2.1 How plugins are imported

Each plugin directory is imported with `importlib` under a synthetic parent package,
`llmeval_testcases.<name>`, with `submodule_search_locations` set to the plugin directory.

This gives real package semantics — `from .stooq import fetch_all` works, and helper
modules stay private to the plugin — without requiring `testcases/` to be on `sys.path`,
without an `__init__.py` at the `testcases/` level, and without the plugin being part of the
installed wheel. Plugins are user content, not shipped code.

Plugins may import from `llmeval` (that is the point of `llmeval/generation/` continuing to
exist). Nothing in `llmeval` may import a plugin except through the loader.

### 2.2 The cache directory

`PluginInterface.cache_directory()` returns `<testcases root>/../.testcases.cache/<plugin
name>/`, created on first call. It is gitignored. Plugins put downloads and their generated
`testcases.json` there. Nothing outside the plugin reads it.

## 3. The plugin contract

```python
class PluginInterface:
    """Runtime dependencies llmeval provides to a plugin."""
    def cache_directory(self) -> Path: ...


class TestCasePlugin(ABC):
    @abstractmethod
    def generate_testcases(self) -> bool: ...
    @abstractmethod
    def get_testcases(self) -> list[dict]: ...

    def get_custom_assertions(self) -> dict[str, Grader]: return {}

    def before_run(self) -> None: ...
    def before_each_run(self, testcase: TestCase) -> None: ...
    def after_each_run(self, testcase: TestCase, summary: RunSummary) -> None: ...
    def after_run(self) -> None: ...

    def before_grade(self) -> None: ...
    def before_each_grade(self, testcase: TestCase) -> None: ...
    def after_each_grade(self, testcase: TestCase, gradings: list[GradingOutcome]) -> None: ...
    def after_grade(self) -> None: ...
```

A plugin module exposes exactly one entry point:

```python
def get_plugin(interface: PluginInterface) -> TestCasePlugin
```

Only `generate_testcases` and `get_testcases` are abstract; every hook defaults to a no-op,
so a CSV-backed plugin is about twenty lines.

### 3.1 `generate_testcases() -> bool`

Do whatever long-running preparation the plugin needs: download a dataset (once — reuse the
cached copy afterwards), transform a CSV, write `testcases.json` into the cache directory.
Some plugins do nothing.

`True` means success, `False` means failure. `llmeval generate` logs a warning for a `False`
and exits non-zero, but carries on to the remaining plugins. An exception is caught and
treated the same way, so one broken plugin cannot abort a whole generation pass.

Whether generation hits the network is the plugin's business. There is no `network`
declaration and nothing is skipped by default: downloads are cached, so they cost once.

### 3.2 `get_testcases() -> list[dict]`

Returns the same shape a `testcases/*.json` file holds: a list of test-case dicts
(`{"id", "user"|"messages", "assertions", "metadata", ...}` — see `llmeval/models.py`).

Conventionally this just reads the `testcases.json` the plugin wrote to its cache directory.
Generating on the fly is allowed; writing the file anyway is recommended because it makes
the plugin's output inspectable while debugging.

**A plugin that has not generated yet returns `[]`.** The loader warns
*"plugin X produced no test cases; run `llmeval generate --testcases X`"*. Raising instead
would let one ungenerated plugin break `llmeval report` for every other source.

### 3.3 Identity: `<source>.<local_id>`

A test case's real id — the one stored in SQLite and printed in logs — is
`<source name>.<plugin-defined id>`. This applies to `.json` sources too, using the file
stem: a case with `"id": "greeting"` in `examples.json` becomes `examples.greeting`.

The plugin picks whatever local id it likes. The loader verifies local ids are unique
*within* a source and raises naming the source and the duplicate; source names are unique by
construction (they are filesystem names, and the stem/directory collision above is an
error). CSV-backed plugins emit a bare `sha1(prompt)[:10]`, so ids read
`simple_facts.a1b2c3d4e5`.

Because the id encodes the source, `metadata.suite` becomes redundant and is **removed**.
The report derives its `suite` column from the id prefix instead.

### 3.4 `get_custom_assertions() -> dict[str, Grader]`

Maps a bare name to a grader with the existing signature
`fn(spec, output, context) -> AssertionResult`. `llmeval` registers each into the shared
registry under `<source>.<name>`, and the plugin emits that namespaced type in its test
cases:

```python
def get_custom_assertions(self):
    return {"stock_price": self._grade_stock_price}   # -> "stock_prices.stock_price"
```

The dot makes collision with a built-in impossible, and source-name uniqueness makes
collision between plugins impossible. Because graders are ordinarily **bound methods**,
plugin state is reachable from grading — which is how `before_grade`'s freshly fetched
quotes reach the stock-price assertion without going through `spec.params`.

Registration happens at load time, so custom assertions exist for every command, not just
`grade`.

### 3.5 Hooks

**Scope.** A hook fires only on plugins that own at least one test case in the current
invocation. `llmeval run --testcases simple_facts` never calls the stock-price plugin's
`before_run`.

**Ordering.** `before_run()` for each owning plugin (main thread, sequential, in source
order) → the existing run loop → `after_run()` for each owning plugin. Same shape for grade.

**Concurrency.** `before_each_run` / `after_each_run` run **inside the worker thread** that
handles that test case, so they are concurrent at `--concurrency > 1`. Plugin authors are
responsible for their own thread safety; this is documented on the base class. The grade
hooks are sequential because grading is.

**Arguments.** `after_each_run` receives the `RunSummary` for that case (ran / cached /
errors / failed) rather than a single result row, because one case can produce several
attempt rows. `after_each_grade` receives the gradings produced for that case *in this
pass* — a list of `GradingOutcome(assertion_key, spec, result)`, a small dataclass defined
in `llmeval/plugins/base.py`. It is empty when everything was already graded and
`--regrade` was not given.

Hooks are not wired into `pickbest`; nothing in the brief needs them there.

## 4. Selection and the `generation` config

All generation-time selection is deleted: `number_to_generate`, `randomize_selection`,
`random_seed`, `stratify`, and `max_rubrics`, along with `suite_generation_config.json`,
`SUITE_GENERATION_CONFIG_FILE`, and `llmeval/generation/config.py`.

**Every plugin generates every test case it can.** research_rubrics and multifaceted emit
all rubrics per row.

Slicing moves to run time. `run` already has `--limit`, `--randomize`, `--seed` and
`--filter k=v`; `--testcases NAME` selects whole sources. Stratified sampling is deferred
and will return as a run-time filter — out of scope here.

**Classification is deleted outright.** `llmeval/generation/classification.py`, the
`request_type` / `domain` metadata, the report's two classification columns, and the
rewrite's dependency on `data/classifications/` all go. `scripts_repo/classify_tests.py`
stays with the legacy promptfoo suite, which still uses it.

## 5. CLI

| Before | After |
|---|---|
| `generate --suite N --all --out --config --data-dir --classifications-dir --sources-dir` | `generate [--testcases NAME ...]` — omitted means every plugin |
| `generate-csv --csv --suite --out --prompt-col --expected-col` | **removed** |
| `--testcases PATH` (repeatable, required on run/grade/pickbest, optional on report) | `--testcases NAME` (repeatable, optional everywhere; omitted means every source) |
| `--filter suite=x` | `--filter` survives for other metadata keys; there is no `suite` key |

There is no flag for the testcases root. It is always `testcases/` relative to the working
directory. Library functions take the root as an argument so tests can point elsewhere; the
CLI does not expose it.

`report` keeps `--testcases` purely as a selector — with classification gone there is
nothing left for it to enrich rows with.

## 6. Framework changes in `llmeval`

**New — `llmeval/plugins/`:**

* `base.py` — `PluginInterface`, `TestCasePlugin`, the hook signatures, the `Grader` alias
  re-exported for plugin authors.
* `loader.py` — discovery, the synthetic-package import, per-plugin error trapping, id
  namespacing, local-id uniqueness, custom-assertion registration. Produces a `Source`
  (either a json source or a plugin source) and the `TestCase` objects it yields.

**Changed:**

* `llmeval/testcases.py` — `load_testcases` / `load_all_testcases` are replaced by
  source-based loading (still applying `--filter` metadata filters). `select_testcases` is
  unchanged.
* `llmeval/runner.py` — accepts a per-test-case hook target so `before_each_run` /
  `after_each_run` fire inside `run_testcase`; `before_run` / `after_run` are driven from
  the run entry point.
* `llmeval/grade.py` — the four grade hooks, same shape.
* `llmeval/resultrows.py` — `suite` comes from the id prefix; `_TEST_COLUMNS`
  (`request_type`, `domain`) and the `with_tests` conditional column set are removed.
* `llmeval/assertions/deterministic.py` — loses `stock_price` and its helpers.
* `llmeval/cli.py` — per §5.
* `llmevalx/discovery.py`, `app.py`, `commands.py` — the suite menus become source pickers;
  the `--filter suite=` step and the `generate --suite` step go.

**Kept as shared library for plugins to import:**

* `llmeval/generation/csv_source.py` — CSV → test-case dicts (the `classifications` and
  `out_dir` parameters go).
* `llmeval/generation/common.py` — the stable-id helper, reshaped: it now returns a bare
  `sha1(prompt)[:10]` (plus an optional `-<variant>` suffix) rather than prefixing a suite
  name, because the loader supplies the `<source>.` prefix. `load_dataset` moves out, since
  datasets now live in plugin cache directories.
* `llmeval/generation/hf_rows.py` — **new**. The Hugging Face datasets-server paging loop,
  ported from the three identical `.mjs` downloaders.

**Deleted:** `generation/{suites,config,classification,agentharm,multifaceted,research_rubrics,stock_prices,stooq}.py`,
`suite_generation_config.json`, `generation_sources/`, and the six generated
`testcases/*.json`.

## 7. Migrating the six suites

**`simple_facts`, `simple_facts_regressions`** — the CSV moves into the plugin directory.
`generate_testcases()` parses it with the shared `csv_source` helper and writes
`<cache>/testcases.json`; `get_testcases()` reads it back. No hooks, no custom assertions.

**`agentharm_refusal`, `multifaceted`, `research_rubrics`** — all three legacy downloaders
hit the same HF datasets-server rows API, so each plugin becomes a call to the shared
`hf_rows` helper plus its existing row-to-test transform. The dataset lands in
`<cache>/dataset.json` and is reused if present. agentharm keeps its `censorship: true`
metadata and its HF gating error message (accept the terms, optionally set `HF_TOKEN`).
research_rubrics still emits both the `rubric` and `g_eval` variants per row, now with all
rubrics rather than a capped set.

The rewrite stops depending on repo-root `data/` and on `pnpm dataset`. The legacy suite
keeps its `.mjs` copies unchanged.

**`stock_prices`** — the one that changes shape:

* `generate_testcases()` becomes pure CSV → JSON. **No network, no baked price.** Each
  assertion carries only `{"symbol": ..., "currency": ...}` under the type
  `stock_prices.stock_price`.
* `before_grade()` fetches every symbol from Stooq, failing fast if any is unavailable, and
  stores the quotes plus a fetch timestamp on the plugin instance.
* The bound grader compares the answer against that live state. If no fetch has happened it
  fails with a message saying so rather than silently passing.
* `stooq.py` moves into the plugin directory; the `stock_price` grader moves out of
  `assertions/deterministic.py`.
* The `max_age_hours` staleness guard is **deleted** — it existed because the reference was
  baked at generation time, and it no longer can be stale. The 1% tolerance and the GBp
  pence-vs-pounds rule stay.
* Freshness still requires `llmeval run --mode always` (or a fresh DB), because a cached
  *answer* would defeat the check. Unchanged, and still documented.

## 8. Consequences

1. **Existing SQLite databases are invalidated.** `test_id` changes shape and there is no
   migration path. Delete the database, or accept that old rows no longer join to any test.
   This is consistent with the store's existing "no migrations" stance.
2. **`request_type` / `domain` disappear** from test metadata, from report columns, and as
   filter keys.
3. **Generated test cases leave git.** Review moves to the plugin code and its CSV; the
   cache copy keeps output inspectable locally. Hand-written `examples.json` stays tracked.
4. **`llmeval` executes code from `testcases/` on every invocation**, including `report`,
   which previously touched nothing but SQLite. Every plugin import and `get_plugin` call is
   individually trapped: a failure warns and drops that source.
5. **`grade` needs the network** for `stock_prices`, and `--regrade` re-fetches. That
   suite's grades are no longer reproducible from stored data alone — the deliberate cost of
   grading against live prices.
6. **CLAUDE.md's public contract #2** ("test-case JSON in `testcases/`") grows a second
   half: the plugin API. Both are breaking-change surfaces.
7. The `stocks` extra becomes `network` (`requests`), now used by the dataset plugins too.
8. `llmevalx` loses its suite-filter menu and its `generate --suite` step; both become
   source pickers.

## 9. Testing

Everything stays offline — no API keys, no network — which is an existing hard rule.

* **Loader**: fixture plugin trees under `tmp_path`. Discovery, the stem/directory
  collision error, a directory without `__init__.py` (warn + ignore), a plugin whose import
  raises (warn + ignore), a plugin without `get_plugin` (warn + ignore), id namespacing for
  both source kinds, duplicate local ids, custom-assertion registration and namespacing.
* **Hooks**: a recording fixture plugin asserting call order, that non-owning plugins are
  untouched, and that per-case hooks see the right test case.
* **Plugins**: `framework_tests/plugins/test_<name>.py`, driven through the loader with
  injected fetchers — the same injection `Fetch` provides today. Stock prices covers
  before_grade populating state, the grader reading it, the GBp rule, and the
  no-fetch-yet failure.
* Existing `framework_tests/test_gen_*.py` are rewritten against the plugins; those testing
  deleted machinery (`test_gen_config.py`, `test_gen_classification.py`,
  `test_gen_suites.py`, `test_cli_generate.py`) are replaced rather than adapted.
* `pylint llmeval llmevalx` stays at 10/10. Plugins under `testcases/` are user content and
  are not linted as part of the package.

## 10. Out of scope

* Run-time stratification / sampling filters to replace what §4 deletes.
* Rework of best-of-N and rubric handling.
* Porting `scripts_repo/classify_tests.py`, or any classification equivalent, into the
  rewrite.
* Hooks for `pickbest`.

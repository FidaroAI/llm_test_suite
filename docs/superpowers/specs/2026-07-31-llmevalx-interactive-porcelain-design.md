# `llmevalx` — an interactive porcelain for the llmeval suite

**Date:** 2026-07-31
**Status:** implemented, then revised — see [As built](#as-built) for where the first cut
diverged from the design, and [Revision: llmevalx is first-class](#revision-llmevalx-is-first-class)
for what changed immediately afterwards. **The packaging described below is superseded.**

## Problem

`llmeval` is deliberately unfriendly. Per [rewrite/CLAUDE.md](../../../rewrite/CLAUDE.md),
everything must be *possible* with the CLI and nothing needs to be *pleasant*: explicit
flags, no guessed defaults, no workflow shortcuts. The everyday loop therefore reads:

```bash
uv run --env-file .env llmeval run --testcases testcases/ --provider configs/fidaro_dev.json \
    --filter suite=simple_facts --timeout 60 --concurrency 5
uv run llmeval grade --testcases testcases/ --provider configs/fidaro_dev.json --run-last-n 1
uv run llmeval report --run-last-n 1 --testcases testcases/ --out results.csv
uv run python -m reporting.csv_table results.csv -o report.html
```

Four commands, twelve flags, three paths you have to remember, and one `.env` you have to
remember to load. That is not a bug in the plumbing — it is a porcelain that has not been
written yet.

## Goal

An interactive, arrow-key-driven wizard that walks you through generate / run / grade /
report: dynamically discovered test cases, providers, suites and runs; sensible defaults you
can pass with Enter; `.env` loaded automatically; Esc to go back at any point.

It is **not** a second `llmeval`. It is porcelain, it lives outside the `llmeval` package,
and it drives the plumbing by shelling out to the documented CLI.

## Non-goals

* Replacing or hiding the CLI. Every command is echoed before it runs, so the wizard doubles
  as a way to learn the flags.
* `pickbest` and `compare-report`. Both need multi-provider selection and a baseline choice;
  they can be added later, and leaving them out keeps the first version tight.
* Remembering your last choices between sessions. YAGNI until it hurts.

## Layout

```
rewrite/
  llmevalx.sh          # thin `uv run llmevalx "$@"` wrapper at the top level
  porcelain/
    __init__.py
    __main__.py        # `python -m porcelain`
    app.py             # the step machine — navigation only
    prompts.py         # questionary wrappers; every prompt returns a value or BACK
    discovery.py       # read configs/, testcases/, the suite registry, runs from the DB
    commands.py        # a selection -> an argv list -> echo -> subprocess
    env.py             # load .env into os.environ
  porcelain_tests/     # offline, no TTY, no credentials
```

`porcelain/` sits beside `reporting/`, follows the same rules — it imports `llmeval`, never
the reverse, and is excluded from the installed wheel — and adds a `porcelain` extra
(`questionary`, `python-dotenv`).

The split exists so the interesting parts are testable without a terminal: `discovery` and
`commands` are pure functions (a selection in, an argv list out), and `app.py` holds only
navigation. Tests assert on command strings and never spawn a process or open a TTY.

## Navigation model

Every prompt returns either its value or the `BACK` sentinel. That turns the whole wizard
into one flat loop with an index cursor rather than nested calls that cannot be unwound:

```python
i = 0
while 0 <= i < len(steps):
    out = steps[i](state)
    i += -1 if out is BACK else 1     # i < 0 -> back to the action menu
```

Each action supplies its own list of steps. The report sub-choice rewrites the tail of the
list rather than nesting a second loop.

Esc is bound per question via a prompt_toolkit key binding on questionary's
`Question.application`. If that proves unworkable for some question type, the fallback is a
`< Back` entry as the first item of every list and a `b`-to-go-back hint on text prompts.

## The steps

### Step 1 — action

`generate` / `run` / `grade` / `report`. Esc (or Ctrl-C) exits.

### Step 2 — report only

* **Report last run** — no further questions. Runs
  `report --run-last-n 1 --testcases testcases/ --out results.csv`, renders it, opens it.
  Test cases are included so the report carries the `request_type` / `domain` labels; no
  `--provider`, so a run from any provider is picked up.
* **Report other data** — fall through to step 3.

### Step 3 — unified selection (run / grade / report-other)

| Question | Source | Maps to |
|---|---|---|
| Test cases (multi) | `testcases/*.json`, shown with case counts, plus **All** | repeatable `--testcases` |
| Provider | `configs/*.json`, excluding the judge config; report also offers **All providers** | `--provider` (omitted for all) |
| Suite | the `suite` values actually present in the chosen files, plus **All suites** | `--filter suite=X` |
| Runs (grade, report) | **Last run** / **All runs** / **Specific runs** | `--run-last-n 1` / nothing / repeated `--run-id` |

**Specific runs** opens a chronological multi-select built from `Store.list_runs()`, showing
start time, provider name and run id. Reading the store directly is a supported way to
consume results (the SQLite schema is one of the plumbing's three public contracts), so this
needs no new subcommand.

`generate` cannot use this shape — there is no provider, and the test cases do not exist yet.
It gets one step of its own: a multi-select over the `SUITES` registry (flagging
`stock_prices` as network-dependent), plus **All (non-network)**, writing to `testcases/`.

### Step 4 — action-specific

* **run** — timeout (`60.0`), concurrency (`5`) and limit (blank = all) as text prompts with
  the llmeval defaults pre-filled, so Enter three times gets you through. Then execute.
* **grade** — execute; the run selection was gathered in step 3.
* **report** — write the CSV, then render it with `reporting.csv_table`, which opens it.
* **generate** — execute.

## Execution

* `.env` in `rewrite/` is loaded into `os.environ` at startup if it exists, before anything
  else, and the fact is logged. This is why the wizard exists: forgetting
  `uv run --env-file .env` is the single most common way a run fails.
* Every command is printed in copy-pasteable form before it runs.
* `subprocess.run` with inherited stdio, so llmeval's stderr logging streams live rather than
  being buffered into a widget.
* A non-zero exit aborts the rest of the chain — no HTML gets rendered from a CSV step that
  failed — and returns to the action menu instead of quitting, so a typo costs one keypress.
* Children are invoked as `sys.executable -m llmeval ...`, which is why `llmeval` gains an
  `__main__.py`.

## Plumbing changes

Two, both capabilities rather than workflows, which is what the repo's own rule says belongs
in `llmeval/`:

1. **`--testcases` becomes repeatable** on `run`, `grade`, `pickbest` and `report`.
   `load_testcases` gains a sibling taking a sequence of paths, concatenating them and
   de-duplicating by test id (first wins), so passing a directory *and* a file inside it does
   not double-count. Without this there is no way to express "these three suite files but not
   the other two", and the wizard would have to fake it with a temp directory of symlinks.
2. **`llmeval/__main__.py`** so `python -m llmeval` works, letting the porcelain invoke the
   plumbing through the interpreter it is already running under rather than hunting for a
   console script on `PATH`.

Neither changes the meaning of an existing invocation: a single `--testcases` behaves exactly
as before.

## Testing

`porcelain_tests/`, offline and TTY-free:

* argv construction for every action × selection shape, including the "all test cases"
  collapse to a bare directory and the repeated `--run-id` form
* `.env` loading, including absent-file and already-set-in-environment cases
* discovery against a temp directory of testcase JSON and provider configs, asserting the
  judge config is filtered out and suites come from metadata
* run-list formatting against an in-memory `Store`

The questionary layer is a thin shell over a library and is not unit-tested; a manual smoke
run against the `echo` provider covers it. The plumbing change gets tests in
`framework_tests/`. `pylint llmeval` stays at 10/10.

## Risks

* **Esc binding.** Mitigated by spiking it before anything else, and by the `< Back` fallback.
* **A second porcelain package.** `reporting/` and `porcelain/` both being porcelain invites
  the question of whether they should merge. They should not yet: `reporting` is a generic
  CSV renderer usable on any table, and `porcelain` is workflow-specific. If a third appears,
  revisit.

## As built

Three deviations from the design above, all decided during implementation.

**No `llmevalx` console script.** The plan called for one. It cannot work: `porcelain` is
excluded from the wheel, so an installed script has no way to import it — the same constraint
that made `reporting/` choose module invocation. `llmevalx.sh` `cd`s to `rewrite/` and runs
`python -m porcelain`, which gives the friendly name without pretending the package is
installed.

**No `python-dotenv`.** Replaced by a ~25-line parser in `porcelain/env.py` handling comments,
blank lines, `export`, and surrounding quotes. One fewer dependency for a file format we fully
control, and it is directly covered by tests. It deliberately does *not* do interpolation,
multi-line values or escapes; a `.env` needing those wants a real shell.

**The Esc spike succeeded, so there is no `< Back` fallback.** Binding `escape` with
`eager=True` is safe because prompt_toolkit's vt100 parser resolves `\x1b[A` into a `Up` key
*before* bindings are consulted. One thing the spike surfaced that the design had not
anticipated: questionary's prefilled default sits in the edit buffer with the cursor after it,
so typing appends (`60.0` + `12.5` = `60.012.5`). An eager `<any>` binding that fires once,
clears the buffer for printable input and re-feeds anything else via
`key_processor.feed(..., first=True)` gives real highlighted-default semantics — Enter
accepts, typing replaces, backspace/Ctrl-U/arrows still edit.

Two things the test harness cannot express, both recorded in the tests rather than worked
around: Esc as the final keystroke of a *multi-question* sequence (a lone Escape never flushes
across a prompt_toolkit application boundary, so the pipe-driven test hangs), and a step that
legitimately re-asks once the queued keys have run out.

## Revision: `llmevalx` is first-class

The first cut got the packaging wrong, and it failed on the first real run:

```
ModuleNotFoundError: No module named 'questionary'
```

`questionary` sat behind a `porcelain` extra, so installing `.[providers,dev]` — the documented
line — produced a `porcelain/` package with nothing to run it. Worse, the reasoning that led
there was itself wrong. It went: porcelain doesn't ship in the wheel, therefore it can't have a
console script, therefore it's invoked as a module, therefore its dependencies are optional.
Each step follows from the one before; the first was an assumption inherited from `reporting/`
rather than anything the design needed.

The correction, in the user's words: *"Porcelain is an architectural ethos, not a name. Both
`llmeval` and `llmevalx` are first-class citizens. They may even share code."*

So:

* **`porcelain/` → `llmevalx/`**, and `porcelain_tests/` → `llmevalx_tests/`. Naming a
  directory after the ethos said nothing useful about what was inside it.
* **One `pyproject.toml`, two console scripts** — `llmeval = "llmeval.cli:main"` and
  `llmevalx = "llmevalx:main"`. Both packages ship (`include = ["llmeval*", "llmevalx*"]`), so
  `uv run llmevalx` works exactly the way `uv run llmeval` does.
* **`questionary` is a core dependency.** An install without it yields a broken command rather
  than a smaller one, which is precisely the failure above. `litellm` stays optional and
  lazy-imported, because the core genuinely is usable without it.
* **`reporting/` stays out of the wheel** with no entry point, but for its *own* reason: it
  renders arbitrary tabular data and nobody reaches for it by name. That is a fact about that
  package, not a rule about friendly layers.
* **`paths.project_root()` replaces the unconditional `chdir(ROOT)`.** A packaged console
  script can be installed anywhere, and `ROOT` — the package's parent — points into
  site-packages for a non-editable install. It is now used only when it actually looks like the
  project (contains `configs/` or `testcases/`), falling back to the cwd, so
  `cd my-eval-project && llmevalx` works either way.

The rule that survives is about the *direction* of the dependency, not about directories or
packaging: `llmevalx` imports `llmeval` and shares its constants, its store and its suite
registry, and nothing under `llmeval/` may import `llmevalx` or `reporting`.

`llmevalx.sh` stays as a convenience. It `cd`s to the project directory and clears
`VIRTUAL_ENV`, which direnv sets to a different environment at the repo root and which makes
`uv run` print a mismatch warning on every invocation.

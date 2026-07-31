# `llmevalx` — the interactive wizard

An arrow-key front end for the `llmeval` CLI.

```bash
uv run llmevalx          # or ./llmevalx.sh, or python -m llmevalx
```

`llmeval` is the **plumbing** — everything is possible with it, almost nothing is pleasant
(see [../CLAUDE.md](../CLAUDE.md)). The everyday loop is four commands, a dozen flags and an
`.env` you have to remember to load. `llmevalx` is the friendly layer over exactly that.

Being the friendly layer doesn't make it second-class. It ships from the same project and the
same `pyproject.toml`, has its own console script, and is linted and tested to the same
standard. What it does *not* do is reach back: `llmevalx` imports `llmeval` and shares its
code — constants, the store, the suite registry — and nothing under `llmeval/` may import
`llmevalx`. That direction is the whole rule.

## What it does

| Step | Question |
|---|---|
| 1 | **generate** / **run** / **grade** / **report** |
| 2 | *report only* — "last run" (no further questions) or "other data" |
| 3 | test-case files (multi-select, or all) · provider · suite filter · which runs |
| 4 | *run only* — timeout, concurrency, limit, then the caching mode |

Then it prints the command and runs it. `report` writes the CSV, renders it with
`reporting.csv_table` and opens the page.

Step 4 is pre-filled with llmeval's own defaults, with one deliberate exception: the mode
defaults to **`always`** rather than the CLI's `reuse`. Someone who has just picked a
provider and a set of tests on purpose wants fresh answers; `reuse` would quietly do nothing
on a second pass and look like a broken run. `target_n` asks for its N as an extra question,
which appears and disappears with the answer above it.

`generate` skips step 3 — there is no provider and the test cases do not exist yet — and
asks which suites to build instead.

Everything in the menus is discovered, never hardcoded: test-case files and their case
counts come from `testcases/`, providers from `configs/` (minus the judge), suite filters
from the metadata of the files you actually chose, generatable suites from the suite
registry, and runs from the SQLite store.

## Keys

| Key | Effect |
|---|---|
| arrows | move |
| space | toggle, in a multi-select |
| Enter | confirm — on a pre-filled prompt this accepts the default |
| typing | on a pre-filled prompt, **replaces** the default; backspace/Ctrl-U edit it instead |
| Esc | go back one question; at the first question, quit |
| Ctrl-C | quit |

## Two things to preserve when editing

* **Every command is echoed before it runs.** The wizard is a shortcut for the CLI, not a
  replacement — printing the command is what keeps that true, and what makes it a way to
  learn the flags rather than a way to avoid them.
* **`commands.py` stays pure.** A `Selection` goes in and a list of `Command` comes out;
  `run_commands` is the only thing that spawns anything. That is what makes the interesting
  half of this package testable without a terminal.

## Where it runs

The wizard `chdir`s to the project directory at startup — the one holding `testcases/`,
`configs/` and `llmeval.sqlite3` — so every path it prints is relative and every command it
echoes can be pasted straight into a shell. That is the package's parent for a checkout or an
editable install; for any other install it falls back to the current directory, so
`cd my-eval-project && llmevalx` works too. See `paths.project_root`.

## `.env`

Loaded from the project directory at startup if it exists, so every subprocess inherits it.
Variables already set in the environment win, so `FIDARO_DEV_BASE_URL=... llmevalx` still
overrides. The parser is deliberately small (comments, blank lines, `export`, surrounding
quotes) — it reads one hand-written file, not arbitrary shell.

## Layout

| Module | Job |
|---|---|
| `app.py` | the step machine — navigation only, executes nothing |
| `prompts.py` | questionary wrappers; every prompt returns its value or `BACK` |
| `discovery.py` | what is available to choose from |
| `commands.py` | `Selection` → argv → echo → subprocess |
| `env.py` | `.env` loading |
| `paths.py` | where things live, and which directory to work in |

Navigation is one flat loop with an index cursor, which is the whole reason `BACK` is a
sentinel rather than an exception:

```python
i = 0
while i >= 0:
    steps = steps_for(selection)   # re-derived, so an answer can reshape the rest
    if i >= len(steps): break
    i += -1 if steps[i](selection) is BACK else 1
```

Re-deriving the step list each iteration is what makes "Report last run" cost one keypress
and still let Esc restore the questions it skipped.

## Tests

`../llmevalx_tests/`, offline and TTY-free. `test_prompts.py` drives real questionary
through a `prompt_toolkit` pipe, which is worth it: the Esc and replace-the-default bindings
are custom bindings against a third-party library and would otherwise break silently on an
upgrade.

Two things that harness cannot express, both noted in the tests: Esc as the last keystroke
of a *multi-question* sequence (a lone Escape never flushes across an application boundary),
and a step that legitimately re-asks after the queued keys run out.

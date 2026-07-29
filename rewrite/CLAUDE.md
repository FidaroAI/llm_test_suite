# CLAUDE.md — `rewrite/` (the `llmeval` suite)

FOR AGENTS:

* Start with [README.md](README.md) for usage and [DESIGN.md](DESIGN.md) for why it looks
  the way it does. This directory is self-contained and intended to move to its own repo,
  so don't reach up into the legacy promptfoo suite for anything.

## The governing rule: `llmeval` is plumbing, not porcelain

We use git's split deliberately. Read this before adding a command, a flag, or a "helpful"
default.

* **`llmeval` is the plumbing.** The CLI plus the SQLite store. Everything the suite can do
  must be *possible* here. Nothing needs to be *friendly* here. Explicit flags, no guessed
  defaults, no workflow shortcuts. Six flags and a `--db` path is an acceptable interface.
* **Porcelain is everything friendly, and it lives on top.** Task runners, wrappers that
  encode a whole comparison, infra bring-up (gateways, sidecars, redeploys), dashboards, CI
  entry points. Built *on* the CLI and the database, outside the `llmeval` package.

### Where does my change go?

Ask what kind of thing you're adding:

* A new **capability** — an assertion type, a caching mode, a statistic, a provider hook,
  a store column. That's plumbing: it belongs in `llmeval/`, exposed explicitly.
* A new **workflow** — "compare prod against dev the way we usually do", "run the nightly
  set and publish the report", "bring the gateway up then run the smoke tests". That's
  porcelain. Do not put it in `llmeval/`.

If you catch yourself adding a subcommand whose value is that it saves the user from
choosing, stop: you're writing porcelain inside the plumbing. Prefer a flag over a
subcommand, and prefer porcelain over a flag.

### The plumbing's public contracts

Porcelain may depend on exactly these three, so treat changes to them as breaking:

1. **CLI subcommands** — `generate`, `generate-csv`, `run`, `grade`, `pickbest`, `report`
   (see [llmeval/cli.py](llmeval/cli.py)). Note that the aggregation/statistics step the
   docs call "compare" has no subcommand of its own: it's
   [comparison/stats.py](llmeval/comparison/stats.py), reached via `report` or as a library
   call. Exposing it directly would be a legitimate plumbing addition.
2. **Test-case JSON** in `testcases/` (see [llmeval/models.py](llmeval/models.py)).
3. **SQLite schema** — `runs` / `results` / `gradings` / `verdicts` (see
   [llmeval/store.py](llmeval/store.py)). Querying it with plain SQL is a supported way to
   consume results, not a workaround.

Python porcelain may also call the library functions the subcommands wrap. For anything
else, the CLI is the boundary.

Two consequences to keep in mind:

* **Plumbing output is read by programs, not just people.** Terminal output is logging on
  stderr, and porcelain must not parse it — anything a person reads off the terminal has to
  be obtainable from the store as well. If you add something a caller genuinely needs and
  it is only in a log line, that is a missing store column, not a formatting task.
* **Store schema changes are expensive.** There is no migration path — `store.py` checks
  `PRAGMA user_version` and refuses a database written by an older build, telling the user
  to delete it. Porcelain that accumulates historical results pays for every schema bump,
  so don't churn it casually.

## Working in here

```bash
uv venv .venv && uv pip install --python .venv/bin/python -e ".[providers,dev]"
.venv/bin/python -m pytest          # whole suite runs offline (mock provider + fake judges)
.venv/bin/python -m pylint llmeval  # keep this at 10/10
```

Tests must stay runnable with no API keys and no network — that's why there's an `echo`
provider and fake judges. Don't add a test that needs live credentials to pass.

### Logging conventions

* **No `print`.** Use `logger = logging.getLogger(__name__)` per module. `print` in library
  code cannot be levelled, filtered, redirected, or deferred, and it breaks the grouping
  described below.
* **Only the entry point configures logging.** `llmeval.logs.configure_logging` is called
  from `cli.main` and nowhere else. Importing `llmeval` must never touch the root logger —
  an embedder owns its own handlers.
* **Anything that runs in the thread pool must log inside `deferred_logs`**, or its records
  will interleave with other workers'. `run_testcase` does this via its `defer_logs`
  argument; if you add another parallel stage (grading and pick-best are sequential today,
  and both are obvious candidates), wrap its per-unit work the same way.
* **Prefix per-unit records with the unit's id** (`logger.info("%s: ...", testcase.id)`).
  Redundant inside a contiguous block, but it is what keeps the output greppable and
  readable when deferral is off.
* Use `%s` lazy formatting, not f-strings, in log calls — the standard reason (no
  formatting cost for a record that gets filtered out).

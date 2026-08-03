"""The wizard: ask, then hand off to the CLI.

Navigation is one flat loop over a list of steps, with an index cursor:

    i = 0
    while i >= 0:
        steps = steps_for(selection)      # re-derived, so a changed answer reshapes the rest
        if i >= len(steps): break         # every question answered -> execute
        i += -1 if steps[i](selection) is BACK else 1

Re-deriving the step list each iteration is what makes "Report last run" cost one keypress
and Esc unwind correctly: choosing it shortens the list to nothing, and backing into the
choice lengthens it again. Steps mutate the shared :class:`~llmevalx.commands.Selection`
and return `BACK` or `None`; none of them execute anything.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

from questionary import Choice

from llmevalx import discovery, env, prompts
from llmevalx.commands import (
    DEFAULT_CONCURRENCY,
    DEFAULT_LIMIT,
    DEFAULT_MODE,
    DEFAULT_REPEAT,
    DEFAULT_TIMEOUT,
    MODE_ALWAYS,
    MODE_REUSE,
    RUNS_ALL,
    RUNS_LAST,
    RUNS_SPECIFIC,
    Selection,
    commands_for,
    default_db,
    run_commands,
)
from llmevalx.paths import CONFIGS_DIR, TESTCASES_DIR, project_root
from llmevalx.prompts import BACK

# Sentinels for menu entries that do not map to a real value. They are objects rather than
# None because questionary's Choice falls back to using the title as the value when value is
# None, which would hand back the human label.
ALL = object()           # the "everything" entry in a multi-select
ANY_PROVIDER = object()  # report only: every provider in the database

Step = Callable[[Selection], object]


# --------------------------------------------------------------------------- step 1


def step_action(sel: Selection) -> object:
    answer = prompts.select(
        "What do you want to do?",
        [
            Choice("generate   — have each plugin build its test cases", "generate"),
            Choice("run        — call a provider over test cases", "run"),
            Choice("grade       — score cached outputs (no model calls)", "grade"),
            Choice("report      — build the results table and open it", "report"),
        ],
        default=sel.action or None,
    )
    if answer is BACK:
        return BACK
    sel.action = answer
    return None


# --------------------------------------------------------------------------- step 2


def step_report_mode(sel: Selection) -> object:
    answer = prompts.select(
        "Which data?",
        [
            Choice("Report last run   — no more questions", "last"),
            Choice("Report other data — choose sources, provider, runs", "other"),
        ],
        default=sel.report_mode or None,
    )
    if answer is BACK:
        return BACK
    sel.report_mode = answer
    return None


# --------------------------------------------------------------------------- step 3


def step_testcases(sel: Selection) -> object:
    """Which sources — plugin directories and .json files under testcases/."""
    sources = discovery.list_sources(TESTCASES_DIR)
    if not sources:
        print(f"No test-case sources in {TESTCASES_DIR}/.")
        return BACK
    choices = [Choice("All sources", ALL)] + [Choice(s.label, s.name) for s in sources]
    answer = prompts.checkbox("Which test cases?", choices)
    if answer is BACK:
        return BACK
    # "All" is the empty list, because omitting --testcases is how the CLI spells it.
    sel.sources = [] if ALL in answer else [name for name in answer if name is not ALL]
    return None


def step_provider(sel: Selection) -> object:
    providers = discovery.list_provider_configs(CONFIGS_DIR)
    if not providers:
        print(f"No provider configs in {CONFIGS_DIR}/.")
        return BACK
    choices = [Choice(p.label, p.path) for p in providers]
    if sel.action == "report":
        # report's --provider is optional and defaults to every provider in the database,
        # which is a genuinely useful answer for a report and nonsense for a run.
        choices.insert(0, Choice("All providers", ANY_PROVIDER))
    default = sel.provider or (ANY_PROVIDER if sel.action == "report" else None)
    answer = prompts.select("Which provider?", choices, default=default)
    if answer is BACK:
        return BACK
    sel.provider = None if answer is ANY_PROVIDER else answer
    return None


def step_runs(sel: Selection) -> object:
    answer = prompts.select(
        "Which runs?",
        [
            Choice("Last run", RUNS_LAST),
            Choice("All runs", RUNS_ALL),
            Choice("Specific runs…", RUNS_SPECIFIC),
        ],
        default=sel.runs_mode or RUNS_LAST,
    )
    if answer is BACK:
        return BACK
    sel.runs_mode = answer
    if answer != RUNS_SPECIFIC:
        sel.run_ids = []
    return None


def step_pick_runs(sel: Selection) -> object:
    """Only reached when "Specific runs" was chosen — see :func:`steps_for`."""
    runs = discovery.list_runs(default_db())
    if not runs:
        print(f"No runs in {default_db()} yet.")
        return BACK
    answer = prompts.checkbox(
        "Which runs? (oldest first)", [Choice(r.label, r.run_id) for r in runs]
    )
    if answer is BACK:
        return BACK
    sel.run_ids = list(answer)
    return None


def step_generate_sources(sel: Selection) -> object:
    """Which plugins to generate. A .json source has nothing to generate, so it is not offered."""
    plugins = discovery.generatable_sources(discovery.list_sources(TESTCASES_DIR))
    if not plugins:
        print(f"No plugins in {TESTCASES_DIR}/ — nothing to generate.")
        return BACK
    choices = [Choice("All plugins", ALL)] + [Choice(s.label, s.name) for s in plugins]
    answer = prompts.checkbox("Which plugins do you want to generate?", choices)
    if answer is BACK:
        return BACK
    sel.sources = [] if ALL in answer else [name for name in answer if name is not ALL]
    return None


# --------------------------------------------------------------------------- step 4


def _number(
    sel: Selection, field: str, message: str, default: str, allow_blank: bool,
    whole: bool = False,
) -> object:
    """Ask for a positive number, re-asking until one is given.

    `whole` rejects fractions here rather than letting them through to argparse, which would
    fail the command *after* the wizard had finished asking its questions.
    """
    noun = "positive whole number" if whole else "positive number"
    while True:
        answer = prompts.text(message, default=getattr(sel, field) or default)
        if answer is BACK:
            return BACK
        value = answer.strip()
        if allow_blank and not value:
            setattr(sel, field, "")
            return None
        try:
            if (int(value) if whole else float(value)) <= 0:
                raise ValueError
        except ValueError:
            print(f"  '{value}' is not a {noun} — try again, or Esc to go back.")
            continue
        setattr(sel, field, value)
        return None


def step_run_options(sel: Selection) -> object:
    """Timeout, concurrency and limit, pre-filled with llmeval's own defaults.

    Three prompts rather than one form so Esc backs out one answer at a time, which is what
    you want when you meant to change the second of them.
    """
    for field, message, default, blank in (
        ("timeout", "Timeout per inference call (seconds)", DEFAULT_TIMEOUT, False),
        ("concurrency", "Concurrency (test cases in parallel)", DEFAULT_CONCURRENCY, False),
        ("limit", "Limit (blank = every selected test)", DEFAULT_LIMIT, True),
    ):
        if _number(sel, field, message, default, blank) is BACK:
            return BACK
    return None


def step_run_mode(sel: Selection) -> object:
    """Whether cached results may be reused. Pairs with :func:`step_repeat`, asked next.

    Defaults to `always`, not the CLI's `reuse`: see `commands.DEFAULT_MODE`.
    """
    answer = prompts.select(
        "Which mode?",
        [
            Choice("always     — call the model again, keeping what is already there", MODE_ALWAYS),
            Choice("reuse      — only call the model where results are missing", MODE_REUSE),
        ],
        default=sel.mode or DEFAULT_MODE,
    )
    if answer is BACK:
        return BACK
    sel.mode = answer
    return None


def step_repeat(sel: Selection) -> object:
    """How many results per test case — asked whatever the mode, because it applies to both.

    The wording names the mode already chosen: "top up to 5" and "add 5 more" are different
    enough that a mode-neutral phrasing would leave you guessing which one you were getting.
    """
    verb = "top each test case up to" if sel.mode == MODE_REUSE else "run each test case"
    return _number(
        sel, "repeat", f"How many results per test case? ({verb} this many)",
        DEFAULT_REPEAT, False, whole=True,
    )


# --------------------------------------------------------------------------- assembly


def steps_for(sel: Selection) -> list[Step]:
    """The questions still to ask, given what has been answered so far.

    Recomputed every iteration, which is how one answer can reshape the rest of the wizard
    (and how backing into that answer restores what it removed).
    """
    if sel.action == "generate":
        return [step_action, step_generate_sources]
    if sel.action == "run":
        # Fixed-length: every question applies to every run. `repeat` used to appear and
        # disappear with the mode answer, back when it was `target_n`'s extra question.
        return [
            step_action, step_testcases, step_provider, step_run_options, step_run_mode,
            step_repeat,
        ]
    if sel.action == "grade":
        steps: list[Step] = [step_action, step_testcases, step_provider, step_runs]
        return _with_run_picker(steps, sel)
    if sel.action == "report":
        if sel.report_mode == "last":
            return [step_action, step_report_mode]
        return _with_run_picker(
            [step_action, step_report_mode, step_testcases, step_provider, step_runs], sel
        )
    return [step_action]


def _with_run_picker(steps: list[Step], sel: Selection) -> list[Step]:
    """Append the run picker only when "Specific runs" was chosen.

    The extra step appears the moment that answer is given and disappears again if it is
    changed, which is the whole reason the step list is derived rather than fixed up front.
    """
    return [*steps, step_pick_runs] if sel.runs_mode == RUNS_SPECIFIC else steps


def collect(sel: Selection) -> bool:
    """Walk the steps. Returns True when every question was answered, False on Esc-out."""
    i = 0
    while i >= 0:
        steps = steps_for(sel)
        if i >= len(steps):
            return True
        i += -1 if steps[i](sel) is BACK else 1
    return False


def banner(root, loaded_env) -> None:
    print("llmevalx — interactive llmeval\n")
    print(f"  working directory  {root}")
    print(f"  results database   {default_db()}")
    if loaded_env is None:
        print("  environment        no .env found (fine unless a provider needs credentials)")
    else:
        print(f"  environment        loaded {loaded_env.name}")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="llmevalx",
        description="Interactive front end for the llmeval CLI. Arrow keys to move, "
        "Enter to choose, Esc to go back, Ctrl-C to quit.",
    )
    parser.parse_args(argv)

    # Everything downstream uses relative paths so the echoed commands can be pasted into a
    # shell; that only holds if the shell would be sitting here too.
    root = project_root()
    os.chdir(root)
    banner(root, env.load_env())

    selection = Selection(action="")
    try:
        while True:
            # Carry the previous answers forward as defaults — after a run you almost always
            # want to grade or report the same thing.
            selection.action = ""
            if not collect(selection):
                print("Bye.")
                return 0
            status = run_commands(commands_for(selection))
            if status == 0:
                print("\nDone.\n")
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

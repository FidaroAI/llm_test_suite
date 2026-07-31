"""The wizard: ask, then hand off to the CLI.

Navigation is one flat loop over a list of steps, with an index cursor:

    i = 0
    while i >= 0:
        steps = steps_for(selection)      # re-derived, so a changed answer reshapes the rest
        if i >= len(steps): break         # every question answered -> execute
        i += -1 if steps[i](selection) is BACK else 1

Re-deriving the step list each iteration is what makes "Report last run" cost one keypress
and Esc unwind correctly: choosing it shortens the list to nothing, and backing into the
choice lengthens it again. Steps mutate the shared :class:`~porcelain.commands.Selection`
and return `BACK` or `None`; none of them execute anything.
"""

from __future__ import annotations

import argparse
import os
from typing import Callable

from questionary import Choice

from porcelain import discovery, env, prompts
from porcelain.commands import (
    DEFAULT_CONCURRENCY,
    DEFAULT_LIMIT,
    DEFAULT_TIMEOUT,
    RUNS_ALL,
    RUNS_LAST,
    RUNS_SPECIFIC,
    Selection,
    commands_for,
    default_db,
    run_commands,
)
from porcelain.paths import CONFIGS_DIR, ROOT, TESTCASES_DIR
from porcelain.prompts import BACK

# Sentinels for menu entries that do not map to a real value. They are objects rather than
# None because questionary's Choice falls back to using the title as the value when value is
# None, which would hand back the human label.
ALL = object()           # the "everything" entry in a multi-select
NO_FILTER = object()     # "don't filter by suite"
ANY_PROVIDER = object()  # report only: every provider in the database

Step = Callable[[Selection], object]


# --------------------------------------------------------------------------- step 1


def step_action(sel: Selection) -> object:
    answer = prompts.select(
        "What do you want to do?",
        [
            Choice("generate   — build test cases from the suite generators", "generate"),
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
            Choice("Report other data — choose test cases, provider, suite, runs", "other"),
        ],
        default=sel.report_mode or None,
    )
    if answer is BACK:
        return BACK
    sel.report_mode = answer
    return None


# --------------------------------------------------------------------------- step 3


def step_testcases(sel: Selection) -> object:
    files = discovery.list_testcase_files(TESTCASES_DIR)
    if not files:
        print(f"No test cases in {TESTCASES_DIR}/ — run 'generate' first.")
        return BACK
    choices = [Choice("All test cases", ALL)] + [Choice(f.label, f.path) for f in files]
    answer = prompts.checkbox("Which test cases?", choices)
    if answer is BACK:
        return BACK
    sel.all_testcases = ALL in answer
    sel.testcases = [] if sel.all_testcases else [p for p in answer if p is not ALL]
    # The suite filter is derived from what was chosen here, so a changed selection must not
    # leave a filter behind that the new files may not even contain.
    chosen = files if sel.all_testcases else [f for f in files if f.path in sel.testcases]
    if sel.suite and sel.suite not in discovery.suites_in(chosen):
        sel.suite = None
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


def step_suite(sel: Selection) -> object:
    files = discovery.list_testcase_files(TESTCASES_DIR)
    if not sel.all_testcases and sel.testcases:
        files = [f for f in files if f.path in sel.testcases]
    suites = discovery.suites_in(files)
    if not suites:
        sel.suite = None
        return None
    choices = [Choice("All suites (no filter)", NO_FILTER)] + [Choice(s, s) for s in suites]
    answer = prompts.select(
        "Filter by suite?", choices, default=sel.suite if sel.suite in suites else NO_FILTER
    )
    if answer is BACK:
        return BACK
    sel.suite = None if answer is NO_FILTER else answer
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


def step_generate_suites(sel: Selection) -> object:
    suites = discovery.list_generatable_suites()
    choices = [Choice("All suites (skips the network ones)", ALL)] + [
        Choice(s.label, s.name) for s in suites
    ]
    answer = prompts.checkbox("Which suites do you want to generate?", choices)
    if answer is BACK:
        return BACK
    sel.generate_all = ALL in answer
    sel.generate_suites = [] if sel.generate_all else [s for s in answer if s is not ALL]
    return None


# --------------------------------------------------------------------------- step 4


def _number(sel: Selection, field: str, message: str, default: str, allow_blank: bool) -> object:
    while True:
        answer = prompts.text(message, default=getattr(sel, field) or default)
        if answer is BACK:
            return BACK
        value = answer.strip()
        if allow_blank and not value:
            setattr(sel, field, "")
            return None
        try:
            if float(value) <= 0:
                raise ValueError
        except ValueError:
            print(f"  '{value}' is not a positive number — try again, or Esc to go back.")
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


# --------------------------------------------------------------------------- assembly


def steps_for(sel: Selection) -> list[Step]:
    """The questions still to ask, given what has been answered so far.

    Recomputed every iteration, which is how one answer can reshape the rest of the wizard
    (and how backing into that answer restores what it removed).
    """
    if sel.action == "generate":
        return [step_action, step_generate_suites]
    if sel.action == "run":
        return [step_action, step_testcases, step_provider, step_suite, step_run_options]
    if sel.action == "grade":
        steps: list[Step] = [
            step_action, step_testcases, step_provider, step_suite, step_runs
        ]
        return _with_run_picker(steps, sel)
    if sel.action == "report":
        if sel.report_mode == "last":
            return [step_action, step_report_mode]
        return _with_run_picker(
            [step_action, step_report_mode, step_testcases, step_provider, step_suite, step_runs],
            sel,
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


def banner(env_file) -> None:
    print("llmevalx — interactive llmeval\n")
    print(f"  working directory  {ROOT}")
    print(f"  results database   {default_db()}")
    if env_file is None:
        print("  environment        no .env found (fine unless a provider needs credentials)")
    else:
        print(f"  environment        loaded {env_file.name}")
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
    os.chdir(ROOT)
    banner(env.load_env())

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

"""Turn a finished selection into the commands that carry it out.

Deliberately pure: :func:`commands_for` takes a :class:`Selection` and returns a list of
:class:`Command`, touching neither the terminal nor a subprocess. That is what makes the
interesting half of this package testable without a TTY — the tests assert on command
strings, and :func:`run_commands` is the only thing that actually spawns anything.

Every command is echoed before it runs, in the form you would type it yourself. The wizard
is a shortcut for the CLI, not a replacement, and printing the command is what keeps that
true: use it a few times and you have learned the flags.
"""

from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass, field

from llmeval.cli import DEFAULT_DB
from llmevalx.paths import REPORT_HTML, RESULTS_CSV, TESTCASES_DIR

# Run-selection modes offered by the "which runs?" menu.
RUNS_LAST = "last"
RUNS_ALL = "all"
RUNS_SPECIFIC = "specific"

ACTIONS = ("generate", "run", "grade", "report")

# The llmeval defaults, mirrored so the wizard can pre-fill them. Kept as strings because
# that is what a text prompt round-trips; the CLI does the parsing.
DEFAULT_TIMEOUT = "60.0"
DEFAULT_CONCURRENCY = "5"
DEFAULT_LIMIT = ""  # blank means "no --limit", i.e. every selected test

# The runner's caching modes (llmeval.runner.VALID_MODES) offered by the "which mode?" menu.
MODE_REUSE = "reuse"
MODE_TARGET_N = "target_n"
MODE_ALWAYS = "always"

# The one place the wizard deliberately does *not* mirror the CLI: `llmeval run` defaults to
# `reuse`, which is right for a library — never spend money twice by accident. Someone at the
# wizard has just picked a provider and a set of tests on purpose, and expects fresh answers;
# `reuse` would silently do nothing at all on a second pass and look like a broken run.
DEFAULT_MODE = MODE_ALWAYS
DEFAULT_TARGET_N = "1"


@dataclass
class Selection:
    """Everything the wizard collected. One flat record, so a step can revise any field."""

    action: str
    # Step 2 (report only): "last" for the one-keypress path, "other" for the full wizard.
    report_mode: str | None = None
    # Step 3
    testcases: list[str] = field(default_factory=list)   # [] means "all", i.e. testcases/
    all_testcases: bool = False
    provider: str | None = None                          # config path; None means every one
    suite: str | None = None                             # --filter suite=<x>
    runs_mode: str | None = None
    run_ids: list[str] = field(default_factory=list)
    # Step 3 for generate
    generate_suites: list[str] = field(default_factory=list)
    generate_all: bool = False
    # Step 4 (run only)
    timeout: str = DEFAULT_TIMEOUT
    concurrency: str = DEFAULT_CONCURRENCY
    limit: str = DEFAULT_LIMIT
    mode: str = DEFAULT_MODE
    target_n: str = DEFAULT_TARGET_N                     # only used when mode is target_n


@dataclass(frozen=True)
class Command:
    argv: list[str]
    display: str

    def __str__(self) -> str:
        return self.display


def _llmeval(*args: str) -> Command:
    """An `llmeval` invocation.

    Executed as `sys.executable -m llmeval` so it runs under the interpreter the wizard is
    already in, rather than whichever `llmeval` happens to be on `PATH`. Displayed as plain
    `llmeval ...`, because that is the command a person would type.
    """
    return Command(
        argv=[sys.executable, "-m", "llmeval", *args],
        display="llmeval " + " ".join(shlex.quote(a) for a in args),
    )


def _module(module: str, *args: str) -> Command:
    return Command(
        argv=[sys.executable, "-m", module, *args],
        display=f"python -m {module} " + " ".join(shlex.quote(a) for a in args),
    )


def _testcase_flags(sel: Selection) -> list[str]:
    """`--testcases` once per chosen file, or once for the whole directory.

    "All" collapses to the directory rather than listing every file: it is shorter to read,
    it is what a person would type, and it keeps picking up files added later.
    """
    if sel.all_testcases or not sel.testcases:
        return ["--testcases", TESTCASES_DIR]
    out: list[str] = []
    for path in sel.testcases:
        out += ["--testcases", path]
    return out


def _filter_flags(sel: Selection) -> list[str]:
    return ["--filter", f"suite={sel.suite}"] if sel.suite else []


def _run_selection_flags(sel: Selection) -> list[str]:
    """The run-selection flags for the "which runs?" answer.

    "All runs" is the absence of a flag — that is the CLI's own default — so it contributes
    nothing rather than something that means the same thing more verbosely.
    """
    if sel.runs_mode == RUNS_LAST:
        return ["--run-last-n", "1"]
    if sel.runs_mode == RUNS_SPECIFIC:
        out: list[str] = []
        for run_id in sel.run_ids:
            out += ["--run-id", run_id]
        return out
    return []


def _provider_flags(sel: Selection, flag: str = "--provider") -> list[str]:
    return [flag, sel.provider] if sel.provider else []


def _report_title(sel: Selection) -> str:
    """A human title for the HTML page, describing what was selected."""
    if sel.report_mode == "last":
        return "llmeval — last run"
    bits = []
    if sel.provider:
        bits.append(sel.provider.rsplit("/", 1)[-1].removesuffix(".json"))
    if sel.suite:
        bits.append(sel.suite)
    if sel.runs_mode == RUNS_LAST:
        bits.append("last run")
    elif sel.runs_mode == RUNS_SPECIFIC:
        bits.append(f"{len(sel.run_ids)} run(s)")
    else:
        bits.append("all runs")
    return "llmeval — " + ", ".join(bits)


def _generate_commands(sel: Selection) -> list[Command]:
    if sel.generate_all:
        return [_llmeval("generate", "--all", "--out", TESTCASES_DIR)]
    args = ["generate"]
    for suite in sel.generate_suites:
        args += ["--suite", suite]
    return [_llmeval(*args, "--out", TESTCASES_DIR)]


def _run_commands(sel: Selection) -> list[Command]:
    args = ["run", *_testcase_flags(sel), *_provider_flags(sel), *_filter_flags(sel)]
    args += ["--timeout", sel.timeout, "--concurrency", sel.concurrency]
    if sel.limit.strip():
        args += ["--limit", sel.limit.strip()]
    # Spelled out even when it matches the CLI's default, unlike the run-selection flags:
    # the mode decides whether the run calls the model at all, so the echoed command should
    # say which one was chosen rather than leave it to be inferred.
    args += ["--mode", sel.mode]
    if sel.mode == MODE_TARGET_N:
        args += ["--target-n", sel.target_n.strip()]
    return [_llmeval(*args)]


def _grade_commands(sel: Selection) -> list[Command]:
    return [
        _llmeval(
            "grade", *_testcase_flags(sel), *_provider_flags(sel),
            *_filter_flags(sel), *_run_selection_flags(sel),
        )
    ]


def _report_commands(sel: Selection) -> list[Command]:
    """Select the rows, render them, open the page.

    Two commands because the split is the plumbing's: `llmeval report` decides *which* rows
    and emits CSV, `reporting.csv_table` turns any CSV into a page (and opens it by default).
    """
    if sel.report_mode == "last":
        # The one-keypress path. testcases/ is included so the page carries the
        # request_type and domain labels; no --provider, so whoever ran last is picked up.
        select = _llmeval(
            "report", "--run-last-n", "1", "--testcases", TESTCASES_DIR, "--out", RESULTS_CSV
        )
    else:
        select = _llmeval(
            "report", *_testcase_flags(sel), *_provider_flags(sel),
            *_filter_flags(sel), *_run_selection_flags(sel), "--out", RESULTS_CSV,
        )
    render = _module(
        "reporting.csv_table", RESULTS_CSV, "-o", REPORT_HTML, "--title", _report_title(sel)
    )
    return [select, render]


_BUILDERS = {
    "generate": _generate_commands,
    "run": _run_commands,
    "grade": _grade_commands,
    "report": _report_commands,
}


def commands_for(sel: Selection) -> list[Command]:
    """The commands that carry out `sel`, in order. Pure — nothing is executed."""
    try:
        build = _BUILDERS[sel.action]
    except KeyError:
        raise ValueError(f"unknown action {sel.action!r}; expected one of {ACTIONS}") from None
    return build(sel)


def run_commands(commands: list[Command], echo=print) -> int:
    """Run each command in turn, echoing it first. Returns the first non-zero exit code.

    stdio is inherited so llmeval's stderr logging streams live rather than arriving in a
    lump at the end — for a run that takes minutes, watching it work *is* the progress bar.

    A failure stops the chain: rendering an HTML page from a CSV step that errored would
    dress up a failure as a result.
    """
    for command in commands:
        echo(f"\n$ {command.display}\n")
        completed = subprocess.run(command.argv, check=False)
        if completed.returncode != 0:
            echo(f"\ncommand failed (exit {completed.returncode}); stopping here.")
            return completed.returncode
    return 0


def default_db() -> str:
    """The database the CLI will use.

    Read off `llmeval.cli` rather than repeated here, so the runs the wizard offers to grade
    or report on cannot drift from the ones the command it builds will actually see.
    """
    return DEFAULT_DB

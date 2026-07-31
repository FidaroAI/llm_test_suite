"""The selection -> argv layer. No terminal, no subprocess, no store."""

import sys

from llmevalx.commands import (
    MODE_ALWAYS,
    MODE_REUSE,
    MODE_TARGET_N,
    RUNS_ALL,
    RUNS_LAST,
    RUNS_SPECIFIC,
    Selection,
    commands_for,
    run_commands,
)


def displays(sel):
    return [c.display for c in commands_for(sel)]


def only(sel):
    commands = commands_for(sel)
    assert len(commands) == 1, commands
    return commands[0]


# --------------------------------------------------------------------------- shape


def test_llmeval_runs_under_the_current_interpreter():
    """So the wizard drives the plumbing it is installed beside, not whatever is on PATH."""
    command = only(Selection(action="run", all_testcases=True, provider="configs/echo.json"))
    assert command.argv[:3] == [sys.executable, "-m", "llmeval"]


def test_display_is_what_a_person_would_type():
    command = only(Selection(action="run", all_testcases=True, provider="configs/echo.json"))
    assert command.display.startswith("llmeval run ")
    assert sys.executable not in command.display


def test_unknown_action_is_rejected():
    try:
        commands_for(Selection(action="frobnicate"))
    except ValueError as exc:
        assert "frobnicate" in str(exc)
    else:
        raise AssertionError("expected ValueError")


# --------------------------------------------------------------------------- testcases


def test_all_testcases_collapses_to_the_directory():
    command = only(Selection(action="run", all_testcases=True, provider="p.json"))
    assert "--testcases testcases" in command.display
    assert command.display.count("--testcases") == 1


def test_selected_files_become_one_flag_each():
    sel = Selection(
        action="run",
        testcases=["testcases/simple_facts.json", "testcases/examples.json"],
        provider="p.json",
    )
    argv = only(sel).argv
    assert argv.count("--testcases") == 2
    assert "testcases/simple_facts.json" in argv and "testcases/examples.json" in argv


def test_empty_testcases_falls_back_to_the_directory():
    """Defensive: an empty list means the same as 'all', never 'no --testcases at all'."""
    assert "--testcases testcases" in only(Selection(action="run", provider="p.json")).display


# --------------------------------------------------------------------------- run


def test_run_passes_timeout_and_concurrency():
    command = only(
        Selection(action="run", all_testcases=True, provider="p.json",
                  timeout="120", concurrency="8")
    )
    assert "--timeout 120" in command.display
    assert "--concurrency 8" in command.display


def test_run_omits_limit_when_blank():
    command = only(Selection(action="run", all_testcases=True, provider="p.json", limit=""))
    assert "--limit" not in command.display


def test_run_includes_limit_when_given():
    command = only(Selection(action="run", all_testcases=True, provider="p.json", limit="10"))
    assert "--limit 10" in command.display


def test_run_defaults_to_mode_always():
    """The wizard's default, deliberately not the CLI's `reuse`."""
    command = only(Selection(action="run", all_testcases=True, provider="p.json"))
    assert "--mode always" in command.display


def test_run_spells_out_the_mode_even_when_it_matches_the_cli_default():
    """It decides whether the model is called at all, so the echoed command must say it."""
    command = only(
        Selection(action="run", all_testcases=True, provider="p.json", mode=MODE_REUSE)
    )
    assert "--mode reuse" in command.display


def test_target_n_mode_passes_the_count():
    command = only(
        Selection(action="run", all_testcases=True, provider="p.json",
                  mode=MODE_TARGET_N, target_n="3")
    )
    assert "--mode target_n" in command.display
    assert "--target-n 3" in command.display


def test_other_modes_omit_the_count():
    """`--target-n` is meaningless outside that mode, and a stale answer must not leak in."""
    command = only(
        Selection(action="run", all_testcases=True, provider="p.json",
                  mode=MODE_ALWAYS, target_n="3")
    )
    assert "--target-n" not in command.display


def test_grade_and_report_never_pass_a_mode():
    """Neither calls a provider, so the run policy is not theirs to have."""
    grade = only(
        Selection(action="grade", all_testcases=True, provider="p.json", runs_mode=RUNS_LAST)
    )
    report = commands_for(
        Selection(action="report", report_mode="other", all_testcases=True, runs_mode=RUNS_ALL)
    )[0]
    assert "--mode" not in grade.display and "--mode" not in report.display


def test_suite_becomes_a_metadata_filter():
    command = only(
        Selection(action="run", all_testcases=True, provider="p.json", suite="simple_facts")
    )
    assert "--filter suite=simple_facts" in command.display


def test_no_suite_means_no_filter():
    assert "--filter" not in only(
        Selection(action="run", all_testcases=True, provider="p.json")
    ).display


# --------------------------------------------------------------------------- run selection


def test_grade_last_run_uses_run_last_n_one():
    command = only(
        Selection(action="grade", all_testcases=True, provider="p.json", runs_mode=RUNS_LAST)
    )
    assert "--run-last-n 1" in command.display


def test_all_runs_passes_no_run_selection_flag():
    """The CLI's own default; saying it again more verbosely would only be noise."""
    command = only(
        Selection(action="grade", all_testcases=True, provider="p.json", runs_mode=RUNS_ALL)
    )
    assert "--run-" not in command.display


def test_specific_runs_repeat_the_run_id_flag():
    sel = Selection(
        action="grade", all_testcases=True, provider="p.json",
        runs_mode=RUNS_SPECIFIC, run_ids=["run_a", "run_b"],
    )
    argv = only(sel).argv
    assert argv.count("--run-id") == 2
    assert "run_a" in argv and "run_b" in argv


def test_grade_never_passes_run_options():
    command = only(
        Selection(action="grade", all_testcases=True, provider="p.json", runs_mode=RUNS_LAST)
    )
    assert "--timeout" not in command.display and "--concurrency" not in command.display


# --------------------------------------------------------------------------- generate


def test_generate_all_uses_the_all_flag():
    command = only(Selection(action="generate", generate_all=True))
    assert "generate --all --out testcases" in command.display


def test_generate_named_suites_repeats_the_suite_flag():
    argv = only(Selection(action="generate", generate_suites=["a", "b"])).argv
    assert argv.count("--suite") == 2
    assert "a" in argv and "b" in argv


def test_generate_ignores_provider_and_runs():
    """generate has neither, so a stale answer from an earlier pass must not leak in."""
    command = only(
        Selection(action="generate", generate_all=True, provider="p.json", runs_mode=RUNS_LAST)
    )
    assert "--provider" not in command.display and "--run-" not in command.display


# --------------------------------------------------------------------------- report


def test_report_is_select_then_render():
    sel = Selection(action="report", report_mode="other", all_testcases=True, runs_mode=RUNS_ALL)
    select, render = commands_for(sel)
    assert select.display.startswith("llmeval report ")
    assert render.display.startswith("python -m reporting.csv_table ")


def test_report_renders_the_csv_it_just_wrote():
    sel = Selection(action="report", report_mode="other", all_testcases=True, runs_mode=RUNS_ALL)
    select, render = commands_for(sel)
    assert "--out results.csv" in select.display
    assert render.argv[3] == "results.csv"
    assert "-o report.html" in render.display


def test_report_last_run_needs_no_other_answers():
    """The one-keypress path: last run, all test cases, every provider."""
    select, render = commands_for(Selection(action="report", report_mode="last"))
    assert "--run-last-n 1" in select.display
    assert "--testcases testcases" in select.display
    assert "--provider" not in select.display
    assert "last run" in render.display


def test_report_all_providers_omits_the_provider_flag():
    sel = Selection(action="report", report_mode="other", all_testcases=True, provider=None)
    assert "--provider" not in commands_for(sel)[0].display


def test_report_title_names_the_selection():
    sel = Selection(
        action="report", report_mode="other", all_testcases=True,
        provider="configs/fidaro_dev.json", suite="simple_facts", runs_mode=RUNS_LAST,
    )
    title = commands_for(sel)[1].argv[-1]
    assert "fidaro_dev" in title and "simple_facts" in title and "last run" in title


# --------------------------------------------------------------------------- execution


class FakeCompleted:
    def __init__(self, returncode):
        self.returncode = returncode


def test_run_commands_stops_at_the_first_failure(monkeypatch):
    """A page rendered from a CSV step that errored would dress a failure up as a result."""
    calls = []

    def fake_run(argv, check=False):        # noqa: ARG001 - signature mirrors subprocess.run
        calls.append(argv)
        return FakeCompleted(1 if len(calls) == 1 else 0)

    monkeypatch.setattr("llmevalx.commands.subprocess.run", fake_run)
    sel = Selection(action="report", report_mode="last")
    status = run_commands(commands_for(sel), echo=lambda *_: None)
    assert status == 1
    assert len(calls) == 1


def test_run_commands_runs_every_command_when_all_succeed(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "llmevalx.commands.subprocess.run",
        lambda argv, check=False: (calls.append(argv), FakeCompleted(0))[1],
    )
    status = run_commands(commands_for(Selection(action="report", report_mode="last")),
                          echo=lambda *_: None)
    assert status == 0
    assert len(calls) == 2


def test_run_commands_echoes_each_command(monkeypatch):
    monkeypatch.setattr(
        "llmevalx.commands.subprocess.run", lambda argv, check=False: FakeCompleted(0)
    )
    lines = []
    run_commands(commands_for(Selection(action="report", report_mode="last")), echo=lines.append)
    assert any("llmeval report" in line for line in lines)
    assert any("reporting.csv_table" in line for line in lines)

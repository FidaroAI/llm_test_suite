"""The step machine — that Esc unwinds, and that an answer reshapes the rest of the wizard.

The steps are driven directly with stub functions rather than through questionary, so this
tests the navigation rather than the terminal library.
"""

from porcelain import app
from porcelain.commands import RUNS_ALL, RUNS_LAST, RUNS_SPECIFIC, Selection
from porcelain.prompts import BACK


def recorder(script):
    """Turn a list of answers into steps that record when they ran.

    Each entry is either a value to assign or ``BACK``. Returns (steps, visited).
    """
    visited = []

    def make(index, answer):
        def step(_sel):
            visited.append(index)
            return BACK if answer is BACK else None

        return step

    return [make(i, a) for i, a in enumerate(script)], visited


def collect_with(steps, sel=None):
    """Run `app.collect` against a fixed step list."""
    original = app.steps_for
    app.steps_for = lambda _sel: steps
    try:
        return app.collect(sel or Selection(action="run"))
    finally:
        app.steps_for = original


# --------------------------------------------------------------------------- the loop


def test_answering_every_step_completes():
    steps, visited = recorder([None, None, None])
    assert collect_with(steps) is True
    assert visited == [0, 1, 2]


def test_back_returns_to_the_previous_step():
    calls = {"n": 0}

    def second(_sel):
        calls["n"] += 1
        return BACK if calls["n"] == 1 else None

    visited = []
    steps = [
        lambda _s: visited.append("first"),
        second,
        lambda _s: visited.append("third"),
    ]
    assert collect_with(steps) is True
    # first runs, second backs out, first runs again, second passes, third runs
    assert visited == ["first", "first", "third"]


def test_back_out_of_the_first_step_abandons_the_wizard():
    steps, visited = recorder([BACK, None])
    assert collect_with(steps) is False
    assert visited == [0]


def test_no_steps_completes_immediately():
    assert collect_with([]) is True


# --------------------------------------------------------------------------- step lists


def test_generate_asks_only_for_suites():
    steps = app.steps_for(Selection(action="generate"))
    assert steps == [app.step_action, app.step_generate_suites]


def test_run_never_asks_about_runs():
    """There is nothing to select from — `run` creates a run rather than reading one."""
    steps = app.steps_for(Selection(action="run"))
    assert app.step_runs not in steps and app.step_pick_runs not in steps
    assert app.step_run_options in steps


def test_grade_asks_about_runs_but_has_no_run_options():
    steps = app.steps_for(Selection(action="grade"))
    assert app.step_runs in steps
    assert app.step_run_options not in steps


def test_run_picker_appears_only_for_specific_runs():
    assert app.step_pick_runs not in app.steps_for(
        Selection(action="grade", runs_mode=RUNS_LAST)
    )
    assert app.step_pick_runs not in app.steps_for(
        Selection(action="grade", runs_mode=RUNS_ALL)
    )
    assert app.step_pick_runs in app.steps_for(
        Selection(action="grade", runs_mode=RUNS_SPECIFIC)
    )


def test_report_last_run_asks_nothing_further():
    steps = app.steps_for(Selection(action="report", report_mode="last"))
    assert steps == [app.step_action, app.step_report_mode]


def test_report_other_data_asks_the_unified_questions():
    steps = app.steps_for(Selection(action="report", report_mode="other"))
    assert steps == [
        app.step_action,
        app.step_report_mode,
        app.step_testcases,
        app.step_provider,
        app.step_suite,
        app.step_runs,
    ]


def test_an_unchosen_action_asks_only_for_the_action():
    assert app.steps_for(Selection(action="")) == [app.step_action]


def test_switching_report_mode_reshapes_the_remaining_steps():
    """Backing into the mode choice must restore the questions it removed."""
    sel = Selection(action="report", report_mode="last")
    assert len(app.steps_for(sel)) == 2
    sel.report_mode = "other"
    assert len(app.steps_for(sel)) == 6


# --------------------------------------------------------------------------- steps


def test_choosing_fewer_testcases_drops_a_now_impossible_suite_filter(tmp_path, monkeypatch):
    """A filter left over from a wider selection could match nothing at all."""
    import json

    tc = tmp_path / "testcases"
    tc.mkdir()
    (tc / "alpha.json").write_text(
        json.dumps([{"id": "a", "user": "q?", "metadata": {"suite": "alpha"}}]), encoding="utf-8"
    )
    (tc / "beta.json").write_text(
        json.dumps([{"id": "b", "user": "q?", "metadata": {"suite": "beta"}}]), encoding="utf-8"
    )
    monkeypatch.setattr(app, "TESTCASES_DIR", str(tc))
    monkeypatch.setattr(app.prompts, "checkbox", lambda *_a, **_k: [str(tc / "alpha.json")])

    sel = Selection(action="run", suite="beta")
    assert app.step_testcases(sel) is None
    assert sel.suite is None


def test_keeping_a_still_valid_suite_filter(tmp_path, monkeypatch):
    import json

    tc = tmp_path / "testcases"
    tc.mkdir()
    (tc / "alpha.json").write_text(
        json.dumps([{"id": "a", "user": "q?", "metadata": {"suite": "alpha"}}]), encoding="utf-8"
    )
    monkeypatch.setattr(app, "TESTCASES_DIR", str(tc))
    monkeypatch.setattr(app.prompts, "checkbox", lambda *_a, **_k: [str(tc / "alpha.json")])

    sel = Selection(action="run", suite="alpha")
    app.step_testcases(sel)
    assert sel.suite == "alpha"


def test_all_testcases_sets_the_flag_not_a_file_list(tmp_path, monkeypatch):
    import json

    tc = tmp_path / "testcases"
    tc.mkdir()
    (tc / "alpha.json").write_text(json.dumps([{"id": "a", "user": "q?"}]), encoding="utf-8")
    monkeypatch.setattr(app, "TESTCASES_DIR", str(tc))
    monkeypatch.setattr(app.prompts, "checkbox", lambda *_a, **_k: [app.ALL])

    sel = Selection(action="run")
    app.step_testcases(sel)
    assert sel.all_testcases is True and sel.testcases == []


def test_no_testcases_on_disk_backs_out(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(app, "TESTCASES_DIR", str(tmp_path / "empty"))
    assert app.step_testcases(Selection(action="run")) is BACK
    assert "generate" in capsys.readouterr().out


def test_all_providers_becomes_no_provider_flag(tmp_path, monkeypatch):
    import json

    configs = tmp_path / "configs"
    configs.mkdir()
    (configs / "dev.json").write_text(
        json.dumps({"name": "dev", "model": "openai/auto"}), encoding="utf-8"
    )
    monkeypatch.setattr(app, "CONFIGS_DIR", str(configs))
    monkeypatch.setattr(app.prompts, "select", lambda *_a, **_k: app.ANY_PROVIDER)

    sel = Selection(action="report", report_mode="other")
    app.step_provider(sel)
    assert sel.provider is None


def test_changing_away_from_specific_runs_clears_the_chosen_ids(monkeypatch):
    monkeypatch.setattr(app.prompts, "select", lambda *_a, **_k: RUNS_LAST)
    sel = Selection(action="grade", runs_mode=RUNS_SPECIFIC, run_ids=["run_a"])
    app.step_runs(sel)
    assert sel.runs_mode == RUNS_LAST and sel.run_ids == []


def test_run_options_accept_the_defaults(monkeypatch):
    monkeypatch.setattr(app.prompts, "text", lambda _m, default="": default)
    sel = Selection(action="run")
    assert app.step_run_options(sel) is None
    assert (sel.timeout, sel.concurrency, sel.limit) == ("60.0", "5", "")


def test_run_options_reject_a_non_number_then_accept(monkeypatch, capsys):
    answers = iter(["abc", "30"])
    monkeypatch.setattr(app.prompts, "text", lambda _m, default="": next(answers, default))
    sel = Selection(action="run")
    app._number(sel, "timeout", "Timeout", "60.0", allow_blank=False)
    assert sel.timeout == "30"
    assert "not a positive number" in capsys.readouterr().out


def test_run_options_reject_zero(monkeypatch):
    answers = iter(["0", "1"])
    monkeypatch.setattr(app.prompts, "text", lambda _m, default="": next(answers, default))
    sel = Selection(action="run")
    app._number(sel, "concurrency", "Concurrency", "5", allow_blank=False)
    assert sel.concurrency == "1"


def test_blank_limit_is_allowed(monkeypatch):
    monkeypatch.setattr(app.prompts, "text", lambda _m, default="": "  ")
    sel = Selection(action="run", limit="9")
    assert app._number(sel, "limit", "Limit", "", allow_blank=True) is None
    assert sel.limit == ""


def test_esc_in_run_options_backs_out(monkeypatch):
    monkeypatch.setattr(app.prompts, "text", lambda *_a, **_k: BACK)
    assert app.step_run_options(Selection(action="run")) is BACK

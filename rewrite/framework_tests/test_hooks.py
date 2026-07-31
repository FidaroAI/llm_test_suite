import pytest

from llmeval.models import TestCase
from llmeval.plugins import TestCasePlugin
from llmeval.plugins.loader import Hooks


class Recorder(TestCasePlugin):
    def __init__(self, name, fail=None):
        self.name = name
        self.calls = []
        self.fail = fail

    def generate_testcases(self):
        return True

    def get_testcases(self):
        return []

    def _record(self, what):
        self.calls.append(what)
        if self.fail == what:
            raise RuntimeError(f"{self.name}:{what} exploded")

    def before_run(self):
        self._record("before_run")

    def before_each_run(self, testcase):
        self._record(f"before_each_run:{testcase.id}")

    def after_each_run(self, testcase, summary):
        self._record(f"after_each_run:{testcase.id}")

    def after_run(self):
        self._record("after_run")

    def before_grade(self):
        self._record("before_grade")

    def after_grade(self):
        self._record("after_grade")


def case(test_id):
    return TestCase.from_dict({"id": test_id, "user": "?"})


def test_hooks_fire_only_for_plugins_owning_a_selected_case():
    owner, bystander = Recorder("owner"), Recorder("bystander")
    tc = case("owner.a")
    hooks = Hooks({"owner.a": owner}, [tc])
    hooks.before_run()
    hooks.before_each_run(tc)
    hooks.after_each_run(tc, None)
    hooks.after_run()
    assert owner.calls == [
        "before_run", "before_each_run:owner.a", "after_each_run:owner.a", "after_run",
    ]
    assert bystander.calls == []


def test_before_run_is_called_once_per_plugin_not_once_per_case():
    owner = Recorder("owner")
    cases = [case("owner.a"), case("owner.b")]
    hooks = Hooks({c.id: owner for c in cases}, cases)
    hooks.before_run()
    assert owner.calls == ["before_run"]


def test_a_failing_before_run_propagates():
    owner = Recorder("owner", fail="before_run")
    tc = case("owner.a")
    with pytest.raises(RuntimeError, match="exploded"):
        Hooks({"owner.a": owner}, [tc]).before_run()


def test_a_failing_after_hook_is_logged_and_swallowed(caplog):
    owner = Recorder("owner", fail="after_run")
    tc = case("owner.a")
    Hooks({"owner.a": owner}, [tc]).after_run()
    assert "after_run" in caplog.text


def test_per_case_hooks_ignore_a_case_with_no_plugin():
    hooks = Hooks({}, [case("examples.a")])
    hooks.before_each_run(case("examples.a"))  # a json source has no plugin; must not raise

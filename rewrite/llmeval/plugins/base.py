"""The contract between ``llmeval`` and a test-case plugin.

A plugin is a directory under ``testcases/`` exposing ``get_plugin(interface)``. It owns its
inputs, its downloads, its bespoke assertions and its own lifecycle; ``llmeval`` owns only this
interface. See ``testcases/README.md`` for the author's view and
``docs/specs/2026-07-31-testcase-plugins-design.md`` for why.

Only :meth:`TestCasePlugin.generate_testcases` and :meth:`TestCasePlugin.get_testcases` are
abstract. Everything else defaults to doing nothing, so a CSV-backed plugin is about twenty
lines and a plugin only pays for the machinery it uses.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TYPE_CHECKING

from llmeval.assertions.base import AssertionResult, Grader
from llmeval.models import AssertionSpec, TestCase

if TYPE_CHECKING:  # avoids a runtime import cycle: runner must not depend on plugins
    from llmeval.runner import RunSummary


@dataclass(frozen=True)
class GradingOutcome:
    """One assertion's verdict, as handed to :meth:`TestCasePlugin.after_each_grade`."""

    assertion_key: str
    spec: AssertionSpec
    result: AssertionResult


class PluginInterface:
    """The runtime dependencies ``llmeval`` provides to a plugin.

    Deliberately tiny. A plugin gets a name and somewhere to write; everything else it does
    for itself. Growing this is how the plugin boundary rots, so add to it only when a plugin
    genuinely cannot do the thing without the suite's help.
    """

    def __init__(self, name: str, cache_root: Path | str):
        self._name = name
        self._cache_root = Path(cache_root)

    @property
    def name(self) -> str:
        """The plugin's source name — its directory name, and the prefix on its test ids."""
        return self._name

    def cache_directory(self) -> Path:
        """This plugin's private scratch directory, created on demand.

        ``<project>/.llmeval.cache/<name>/``, gitignored. Downloads, intermediate files and
        the generated ``testcases.json`` go here. Nothing outside the plugin reads it.
        """
        path = self._cache_root / self._name
        path.mkdir(parents=True, exist_ok=True)
        return path


class TestCasePlugin(abc.ABC):
    """What a plugin's ``get_plugin`` returns.

    **Thread safety.** ``before_each_run`` and ``after_each_run`` are called from the runner's
    worker threads and so may overlap (``llmeval run --concurrency`` defaults to 5). Guard any
    shared state you mutate there. The grade hooks are sequential.

    **Error policy.** An exception from ``before_run`` or ``before_grade`` aborts the command —
    they are setup, and continuing past failed setup produces meaningless results. Exceptions
    from the other six hooks are logged and swallowed, so one plugin cannot lose somebody
    else's long run.
    """

    @abc.abstractmethod
    def generate_testcases(self) -> bool:
        """Do the long-running preparation: download, transform, write. ``True`` on success.

        Called by ``llmeval generate``. Downloads should be cached in
        :meth:`PluginInterface.cache_directory` and reused on later calls. A plugin with
        nothing to prepare returns ``True``.
        """

    @abc.abstractmethod
    def get_testcases(self) -> list[dict[str, Any]]:
        """The plugin's test cases, in the same shape as a ``testcases/*.json`` file.

        Ids are *local* — the loader prefixes them with the source name. Return ``[]`` if
        nothing has been generated yet; the loader warns rather than failing, so one
        ungenerated plugin cannot break a report over every other source.
        """

    def get_custom_assertions(self) -> dict[str, Grader]:
        """Bespoke graders, ``{local name: fn(spec, output, ctx) -> AssertionResult}``.

        Registered as ``<source>.<local name>``, which is what the plugin must emit as the
        assertion ``type``. Returning **bound methods** is the point: it is how state prepared
        in ``before_grade`` reaches grading without going through ``spec.params``.
        """
        return {}

    def before_run(self) -> None:
        """Once, before any of this plugin's test cases run."""

    def before_each_run(self, testcase: TestCase) -> None:
        """Before each of this plugin's test cases runs. Called on a worker thread."""

    def after_each_run(self, testcase: TestCase, summary: "RunSummary") -> None:
        """After each of this plugin's test cases runs. Called on a worker thread.

        ``summary`` counts attempts for that one test case (ran / cached / errors / failed);
        a case can produce several stored rows, which is why this is not a single result.
        """

    def after_run(self) -> None:
        """Once, after all of this plugin's test cases have run."""

    def before_grade(self) -> None:
        """Once, before any of this plugin's test cases are graded.

        Where a plugin refreshes anything its assertions compare against.
        """

    def before_each_grade(self, testcase: TestCase) -> None:
        """Before each of this plugin's test cases is graded."""

    def after_each_grade(self, testcase: TestCase, gradings: list[GradingOutcome]) -> None:
        """After each of this plugin's test cases is graded.

        ``gradings`` holds only what *this pass* produced, so it is empty when everything was
        already graded and ``--regrade`` was not given.
        """

    def after_grade(self) -> None:
        """Once, after all of this plugin's test cases have been graded."""

"""Find, import and namespace the sources under ``testcases/``.

A **source** is either a top-level ``.json`` file or a plugin directory. Both yield test cases
in the same shape, and both contribute their name as the ``<source>.`` prefix on every id they
produce — which is what makes the id, rather than a metadata label, the record of where a test
case came from.

The rules, and why:

* ``.json`` is read only at the top level. A ``.json`` inside a plugin directory is that
  plugin's private business (typically its generated output, in the cache directory).
* A ``.json`` stem may not equal a directory name, because the two would claim the same source
  name and there is no sensible winner. That is an error, not a warning.
* A directory is a plugin iff it holds ``__init__.py`` exposing ``get_plugin``. Anything else
  warns and is skipped: ``testcases/`` is a user-owned directory and a stray folder in it must
  not stop a report from running.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from llmeval.assertions.base import REGISTRY
from llmeval.models import TestCase
from llmeval.plugins.base import GradingOutcome, PluginInterface, TestCasePlugin

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "testcases"
# The suite's one gitignored scratch directory. Plugin caches live under it, and so does the
# default results database (see ``llmeval.cli.DEFAULT_DB``) — one directory to delete when you
# want a clean slate, rather than a cache folder here and a stray .sqlite3 in the project root.
CACHE_DIR_NAME = ".llmeval.cache"

# Entries that are never a source. ``__pycache__`` and dot/underscore-prefixed directories are
# skipped silently — they are plainly not somebody's forgotten plugin, so warning about them
# would be noise on every single command.
_SKIP_PREFIXES = (".", "_")


class SourceError(Exception):
    """A testcases/ layout or content problem the user must fix."""


@dataclass
class Source:
    """One place test cases come from: a ``.json`` file or a loaded plugin."""

    name: str
    path: Path
    plugin: TestCasePlugin | None = None

    @property
    def is_plugin(self) -> bool:
        return self.plugin is not None

    def raw_testcases(self) -> list[dict[str, Any]]:
        """The source's test cases with their **local** ids, exactly as it produced them."""
        if self.plugin is not None:
            cases = self.plugin.get_testcases() or []
            if not cases:
                logger.warning(
                    "plugin %r produced no test cases; run `llmeval generate --testcases %s`",
                    self.name, self.name,
                )
            return list(cases)
        with self.path.open(encoding="utf-8") as fh:
            doc = json.load(fh)
        return doc if isinstance(doc, list) else [doc]


def source_of(test_id: str) -> str | None:
    """The source name encoded in a test id, or ``None`` for an id with no prefix.

    Split on the *first* dot: a plugin may put dots in its own local ids, and the prefix is
    the only part this owns.
    """
    prefix, sep, _ = test_id.partition(".")
    return prefix if sep else None


def _entries(root: Path) -> tuple[list[str], list[str]]:
    """``(directory names, json stems)`` under ``root``, sorted, with skips applied."""
    directories, stems = [], []
    for name in sorted(os.listdir(root)):
        if name.startswith(_SKIP_PREFIXES) or name == "__pycache__":
            continue
        if (root / name).is_dir():
            directories.append(name)
        elif name.endswith(".json"):
            stems.append(name[: -len(".json")])
    return directories, stems


def discover(root: Path | str = DEFAULT_ROOT) -> list[Source]:
    """Every source under ``root``, in name order. A missing root is an empty list.

    "No testcases directory" is a perfectly good state for a fresh checkout, so it is not an
    error here — the command that finds itself with nothing to do says so.
    """
    root = Path(root)
    if not root.is_dir():
        return []
    directories, stems = _entries(root)
    clash = sorted(set(directories) & set(stems))
    if clash:
        raise SourceError(
            f"{root}: {clash[0]}.json and {clash[0]}/ both claim the source name "
            f"{clash[0]!r}; rename one"
        )
    sources = [Source(name=stem, path=root / f"{stem}.json") for stem in stems]
    sources += _plugin_sources(root, directories)
    return sorted(sources, key=lambda s: s.name)


# Plugins are imported under a synthetic parent package so that a plugin's own modules are
# importable relatively (``from .stooq import fetch_all``) and stay private to it. A synthetic
# parent avoids putting ``testcases/`` on sys.path, which would make every plugin's helper
# modules top-level names competing with real ones.
_PARENT_PACKAGE = "llmeval_testcases"
_ENTRY_POINT = "get_plugin"


def _ensure_parent_package() -> None:
    if _PARENT_PACKAGE in sys.modules:
        return
    spec = importlib.machinery.ModuleSpec(_PARENT_PACKAGE, None, is_package=True)
    module = importlib.util.module_from_spec(spec)
    module.__path__ = []  # a namespace with no location of its own
    sys.modules[_PARENT_PACKAGE] = module


def _purge(full_name: str) -> None:
    """Drop a plugin module and its submodules from ``sys.modules``.

    Loading is re-entrant: tests point the loader at several roots in one process, and two of
    them may hold a plugin of the same name. Without this the second load would silently reuse
    the first one's modules.
    """
    for key in [k for k in sys.modules if k == full_name or k.startswith(full_name + ".")]:
        del sys.modules[key]


def _import_plugin_module(name: str, directory: Path):
    _ensure_parent_package()
    full_name = f"{_PARENT_PACKAGE}.{name}"
    _purge(full_name)
    spec = importlib.util.spec_from_file_location(
        full_name, directory / "__init__.py", submodule_search_locations=[str(directory)]
    )
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise SourceError(f"{name}: cannot load {directory / '__init__.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[full_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        _purge(full_name)
        raise
    return module


def cache_root_for(root: Path) -> Path:
    """Where plugin cache directories live: ``.llmeval.cache/`` beside the testcases root."""
    return root.parent / CACHE_DIR_NAME


def _register_assertions(name: str, plugin: TestCasePlugin) -> None:
    """Merge a plugin's graders into the shared registry under ``<source>.<local>``.

    Namespaced because the plugin owns the name: the dot makes a clash with a built-in
    impossible, and source names are unique, so two plugins cannot collide either.
    """
    try:
        custom = plugin.get_custom_assertions() or {}
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("%s: get_custom_assertions failed (%s); no custom assertions", name, exc)
        return
    for local, grader in custom.items():
        REGISTRY[f"{name}.{local}"] = grader


def _load_plugin(root: Path, name: str) -> Source | None:
    """Import one plugin directory. Returns ``None`` (with a warning) if it isn't one.

    Every failure mode is a warning rather than an exception: ``llmeval`` now executes code
    out of ``testcases/`` on *every* command, including ``report``, and one broken directory
    must not take the others down with it.
    """
    directory = root / name
    if not (directory / "__init__.py").is_file():
        logger.warning("%s: not a plugin (no __init__.py); ignoring", directory)
        return None
    try:
        module = _import_plugin_module(name, directory)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("%s: plugin failed to import (%s); ignoring", directory, exc)
        return None
    factory: Callable[[PluginInterface], Any] | None = getattr(module, _ENTRY_POINT, None)
    if not callable(factory):
        logger.warning("%s: plugin defines no %s(interface); ignoring", directory, _ENTRY_POINT)
        return None
    interface = PluginInterface(name, cache_root_for(root))
    try:
        # pylint infers the getattr default rather than the callable() guard above.
        plugin = factory(interface)  # pylint: disable=not-callable
    except Exception as exc:  # pylint: disable=broad-exception-caught
        logger.warning("%s: %s(interface) failed (%s); ignoring", directory, _ENTRY_POINT, exc)
        return None
    if not isinstance(plugin, TestCasePlugin):
        logger.warning(
            "%s: %s did not return a TestCasePlugin; ignoring", directory, _ENTRY_POINT
        )
        return None
    plugin.interface = interface
    _register_assertions(name, plugin)
    return Source(name=name, path=directory, plugin=plugin)


def _plugin_sources(root: Path, directories: Sequence[str]) -> list[Source]:
    loaded = [_load_plugin(root, name) for name in directories]
    return [s for s in loaded if s is not None]


def namespaced_cases(source: Source) -> list[dict[str, Any]]:
    """``source``'s test cases with real ids: ``<source name>.<local id>``.

    Local ids must be unique within a source. Across sources they cannot collide, because the
    prefix is a filesystem name and the stem/directory clash is rejected in :func:`discover`.
    """
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in source.raw_testcases():
        local = str(raw.get("id") or "")
        if not local:
            raise SourceError(f"{source.name}: a test case has no 'id'")
        if local in seen:
            raise SourceError(f"{source.name}: duplicate test case id {local!r}")
        seen.add(local)
        case = dict(raw)
        case["id"] = f"{source.name}.{local}"
        out.append(case)
    return out


# Hooks that are *setup*. A failure here means the work that follows would be meaningless —
# grading stock prices with no live quotes, say — so it stops the command. Everything else is
# logged and swallowed: one plugin's sloppy teardown must not lose somebody's long run.
_STRICT_HOOKS = frozenset({"before_run", "before_grade"})


class Hooks:
    """Dispatches lifecycle hooks to the plugins owning the test cases in play.

    Built per command from the selected cases, so a plugin with nothing selected is never
    called — ``--testcases simple_facts`` must not trigger the stock-price fetch.
    """

    def __init__(self, plugin_for: dict[str, TestCasePlugin], cases: Sequence[TestCase]):
        self._plugin_for = plugin_for
        owners: list[TestCasePlugin] = []
        for case in cases:
            plugin = plugin_for.get(case.id)
            if plugin is not None and not any(plugin is seen for seen in owners):
                owners.append(plugin)
        self._owners = owners

    @property
    def owners(self) -> list[TestCasePlugin]:
        return list(self._owners)

    def _call(self, plugin: TestCasePlugin, method: str, *args) -> None:
        try:
            getattr(plugin, method)(*args)
        except Exception:  # pylint: disable=broad-exception-caught
            if method in _STRICT_HOOKS:
                raise
            logger.exception("plugin hook %s failed", method)

    def _all(self, method: str) -> None:
        for plugin in self._owners:
            self._call(plugin, method)

    def _each(self, method: str, case: TestCase, *args) -> None:
        plugin = self._plugin_for.get(case.id)
        if plugin is not None:
            self._call(plugin, method, case, *args)

    def before_run(self) -> None:
        self._all("before_run")

    def after_run(self) -> None:
        self._all("after_run")

    def before_grade(self) -> None:
        self._all("before_grade")

    def after_grade(self) -> None:
        self._all("after_grade")

    def before_each_run(self, case: TestCase) -> None:
        self._each("before_each_run", case)

    def after_each_run(self, case: TestCase, summary) -> None:
        self._each("after_each_run", case, summary)

    def before_each_grade(self, case: TestCase) -> None:
        self._each("before_each_grade", case)

    def after_each_grade(self, case: TestCase, outcomes: list[GradingOutcome]) -> None:
        self._each("after_each_grade", case, outcomes)


@dataclass
class Loaded:
    """Everything a command needs from ``testcases/``: the sources, the cases, the hooks."""

    sources: list[Source] = field(default_factory=list)
    cases: list[TestCase] = field(default_factory=list)
    plugin_for: dict[str, TestCasePlugin] = field(default_factory=dict)

    def hooks(self, cases: Sequence[TestCase] | None = None) -> Hooks:
        """A dispatcher scoped to ``cases`` (default: everything loaded)."""
        return Hooks(self.plugin_for, self.cases if cases is None else cases)


def _chosen(sources: list[Source], names: Sequence[str]) -> list[Source]:
    """The named sources, in the order asked for, de-duplicated. An unknown name is an error.

    An error rather than an empty result: a name that matches nothing is a typo, and silently
    running no tests is the worst possible response to one.
    """
    known = {s.name: s for s in sources}
    unknown = [n for n in names if n not in known]
    if unknown:
        raise SourceError(f"unknown source(s) {unknown}; known: {sorted(known)}")
    chosen, seen = [], set()
    for name in names:
        if name not in seen:
            seen.add(name)
            chosen.append(known[name])
    return chosen


def select_sources(
    names: Sequence[str] | None = None, root: Path | str = DEFAULT_ROOT
) -> list[Source]:
    """The named sources (default: all) under ``root``, without reading any test cases.

    Separate from :func:`load` because ``generate`` wants the plugins and *not* their output
    — asking a plugin for test cases it has not generated yet would warn about exactly the
    thing the command is about to fix.
    """
    sources = discover(root)
    return _chosen(sources, names) if names else sources


def load(
    names: Sequence[str] | None = None,
    root: Path | str = DEFAULT_ROOT,
    filters: dict[str, Any] | None = None,
) -> Loaded:
    """Load the named sources (default: all) under ``root`` into :class:`TestCase` objects.

    ``names`` are source names, not paths: a plugin directory name or a ``.json`` stem.
    """
    sources = select_sources(names, root)
    loaded = Loaded(sources=sources)
    for source in sources:
        for raw in namespaced_cases(source):
            case = TestCase.from_dict(raw)
            if filters and any(case.metadata.get(k) != v for k, v in filters.items()):
                continue
            loaded.cases.append(case)
            if source.plugin is not None:
                loaded.plugin_for[case.id] = source.plugin
    return loaded

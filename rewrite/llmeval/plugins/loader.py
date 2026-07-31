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

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from llmeval.plugins.base import TestCasePlugin

logger = logging.getLogger(__name__)

DEFAULT_ROOT = "testcases"
CACHE_DIR_NAME = ".testcases.cache"

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


def _plugin_sources(root: Path, directories: Sequence[str]) -> list[Source]:
    """Placeholder — plugin directories contribute nothing yet."""
    # pylint: disable=unused-argument
    return []


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

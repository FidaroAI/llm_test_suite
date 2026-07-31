"""What is available to choose from — read off disk and out of the store, never hardcoded.

Every menu in the wizard is populated from here, so adding a plugin or a provider config
makes it appear without touching this package. Three sources:

* `testcases/` — the sources, via the loader, so a menu entry cannot drift from what
  `--testcases` will actually accept
* `configs/*.json` — the provider configs
* the SQLite store — the runs available to grade or report on

Reading the store directly is supported: the schema is one of the plumbing's three public
contracts (see ../CLAUDE.md).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from llmeval.plugins.loader import SourceError, discover
from llmeval.store import IncompatibleSchema, RunRow, Store

# A provider config whose ``name`` is this grades other providers' output; it is never the
# thing under test, so it does not belong in the "which provider?" menu.
JUDGE_CONFIG_NAME = "judge"


@dataclass(frozen=True)
class SourceChoice:
    """One thing `--testcases` can name: a plugin directory or a top-level .json file."""

    name: str
    kind: str          # "plugin" | "json"
    count: int         # test cases it currently yields; 0 for an ungenerated plugin

    @property
    def label(self) -> str:
        cases = "case" if self.count == 1 else "cases"
        suffix = "  (not generated yet)" if self.kind == "plugin" and not self.count else ""
        return f"{self.name}  [{self.kind}]  ({self.count} {cases}){suffix}"


@dataclass(frozen=True)
class ProviderConfigFile:
    path: str          # relative, e.g. "configs/fidaro_dev.json"
    name: str          # the config's own "name" field, e.g. "fidaro-dev"
    model: str

    @property
    def label(self) -> str:
        return f"{self.name}  [{self.model}]  {self.path}"


@dataclass
class RunChoice:
    run_id: str
    provider_name: str
    started_at: str
    notes: str | None = None
    unfinished: bool = False

    @property
    def label(self) -> str:
        bits = [self.started_at, f"{self.provider_name:<16}", self.run_id]
        if self.unfinished:
            bits.append("(unfinished)")
        if self.notes:
            bits.append(f"- {self.notes}")
        return "  ".join(bits)


@dataclass
class Available:
    """Everything the wizard can offer, gathered in one pass."""

    sources: list[SourceChoice] = field(default_factory=list)
    providers: list[ProviderConfigFile] = field(default_factory=list)


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_sources(directory: str) -> list[SourceChoice]:
    """Every source under `directory`, via the loader, so the menu matches the CLI exactly.

    A plugin that has not generated yet shows a count of 0 rather than being hidden:
    "generate this one" is precisely what someone looking at this menu is about to want.
    A source that cannot be read at all contributes 0 too — one broken entry should grey
    itself out, not stop the wizard from starting.
    """
    try:
        sources = discover(Path(directory))
    except SourceError:
        return []
    out: list[SourceChoice] = []
    for source in sources:
        try:
            count = len(source.raw_testcases())
        except Exception:  # pylint: disable=broad-exception-caught
            count = 0
        out.append(
            SourceChoice(
                name=source.name,
                kind="plugin" if source.is_plugin else "json",
                count=count,
            )
        )
    return out


def list_provider_configs(directory: str) -> list[ProviderConfigFile]:
    """The provider configs in `directory`, minus the judge.

    Loaded as plain JSON rather than through `ProviderConfig`, so a config referencing an
    unset `${ENV}` still lists (the CLI expands it later, when it matters).
    """
    if not os.path.isdir(directory):
        return []
    out: list[ProviderConfigFile] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            doc = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        config_name = str(doc.get("name") or os.path.splitext(name)[0])
        if config_name == JUDGE_CONFIG_NAME:
            continue
        out.append(
            ProviderConfigFile(path=path, name=config_name, model=str(doc.get("model", "?")))
        )
    return out


def generatable_sources(sources: Sequence[SourceChoice]) -> list[SourceChoice]:
    """The sources `llmeval generate` can act on — the plugins. A .json file is already made."""
    return [s for s in sources if s.kind == "plugin"]


def _format_started(started_at: str) -> str:
    """`2026-07-31T12:12:03.123456` -> `2026-07-31 12:12:03Z`. Stored times are UTC."""
    text = started_at.replace("T", " ")
    return text[:19] + "Z" if len(text) >= 19 else text


def list_runs(db_path: str, limit: int | None = 50) -> list[RunChoice]:
    """Runs in the store, **oldest first**, most recent `limit` kept.

    Chronological because that is how the rest of the suite orders runs (`resolve_runs`
    returns oldest first, and report rows are grouped the same way), so the picker reads in
    the same direction as the output it produces. The cap is applied to the *newest* runs
    before reversing, so a long history shows its recent end.

    A missing or unreadable database is an empty list, not an error: "no runs to pick from"
    is a perfectly good answer for someone who has not run anything yet.
    """
    if not os.path.exists(db_path):
        return []
    try:
        store = Store(db_path)
    except IncompatibleSchema:
        return []
    try:
        rows: Sequence[RunRow] = store.list_runs(limit=limit)   # newest first
    finally:
        store.close()
    return [
        RunChoice(
            run_id=row.id,
            provider_name=row.provider_name or "?",
            started_at=_format_started(row.started_at),
            notes=row.notes,
            unfinished=not row.finished,
        )
        for row in reversed(rows)
    ]


def gather(testcases_dir: str, configs_dir: str) -> Available:
    return Available(
        sources=list_sources(testcases_dir),
        providers=list_provider_configs(configs_dir),
    )

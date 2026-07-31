"""What is available to choose from — read off disk and out of the store, never hardcoded.

Every menu in the wizard is populated from here, so adding a suite file or a provider config
makes it appear without touching this package. Four sources:

* `testcases/*.json` — the test-case files, with their case counts and suite labels
* `configs/*.json` — the provider configs
* :data:`llmeval.generation.suites.SUITES` — what `generate` can produce
* the SQLite store — the runs available to grade or report on

Reading the store directly is supported: the schema is one of the plumbing's three public
contracts (see ../CLAUDE.md).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from llmeval.generation.suites import SUITES
from llmeval.store import IncompatibleSchema, RunRow, Store

# A provider config whose ``name`` is this grades other providers' output; it is never the
# thing under test, so it does not belong in the "which provider?" menu.
JUDGE_CONFIG_NAME = "judge"


@dataclass(frozen=True)
class TestcaseFile:
    path: str          # relative, e.g. "testcases/simple_facts.json"
    name: str          # "simple_facts.json"
    count: int
    suites: tuple[str, ...]

    @property
    def label(self) -> str:
        cases = "case" if self.count == 1 else "cases"
        return f"{self.name}  ({self.count} {cases})"


@dataclass(frozen=True)
class ProviderConfigFile:
    path: str          # relative, e.g. "configs/fidaro_dev.json"
    name: str          # the config's own "name" field, e.g. "fidaro-dev"
    model: str

    @property
    def label(self) -> str:
        return f"{self.name}  [{self.model}]  {self.path}"


@dataclass(frozen=True)
class SuiteChoice:
    name: str
    network: bool

    @property
    def label(self) -> str:
        return f"{self.name}  (needs network)" if self.network else self.name


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

    testcase_files: list[TestcaseFile] = field(default_factory=list)
    providers: list[ProviderConfigFile] = field(default_factory=list)
    suites: list[SuiteChoice] = field(default_factory=list)


def _read_json(path: str) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def list_testcase_files(directory: str) -> list[TestcaseFile]:
    """The `.json` files in `directory`, with case counts and the suites inside each.

    Reads the raw JSON rather than building `TestCase` objects: the menu only needs a count
    and the `suite` labels, and a half-written file should grey out one entry rather than
    stop the whole wizard from starting.
    """
    if not os.path.isdir(directory):
        return []
    out: list[TestcaseFile] = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(directory, name)
        try:
            doc = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        cases = doc if isinstance(doc, list) else [doc]
        suites = sorted(
            {
                str(suite)
                for case in cases
                if isinstance(case, dict)
                for suite in [(case.get("metadata") or {}).get("suite")]
                if suite
            }
        )
        out.append(TestcaseFile(path=path, name=name, count=len(cases), suites=tuple(suites)))
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


def list_generatable_suites() -> list[SuiteChoice]:
    """What `llmeval generate` knows how to produce, straight from the suite registry."""
    return [SuiteChoice(name=spec.name, network=spec.network) for spec in SUITES.values()]


def suites_in(files: Iterable[TestcaseFile]) -> list[str]:
    """The suite labels present across the chosen files — the `--filter suite=` options.

    Derived from the selection rather than from every file on disk, so the filter menu can
    never offer a suite that the chosen test cases do not contain.
    """
    return sorted({suite for f in files for suite in f.suites})


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
        testcase_files=list_testcase_files(testcases_dir),
        providers=list_provider_configs(configs_dir),
        suites=list_generatable_suites(),
    )

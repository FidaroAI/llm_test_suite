"""Suite registry: one place that knows how to generate every legacy suite.

Drives the ``llmeval generate --suite NAME`` / ``--all`` CLI. Each suite is either
CSV-backed (``simple_facts*``, ``stock_prices``) or dataset-backed
(``agentharm_refusal``, ``multifaceted``, ``research_rubrics``); all emit the
rewrite's on-disk test-case shape into ``testcases/<suite>.json``.

``stock_prices`` is flagged ``network`` because it fetches live quotes at
generation time, so ``--all`` skips it by default (use ``--suite stock_prices``).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import llmeval
from llmeval.generation import agentharm, multifaceted, research_rubrics
from llmeval.generation.classification import load_classifications, stamp
from llmeval.generation.config import CONFIG_ENV_VAR, load_suite_config
from llmeval.generation.csv_source import generate_from_csv


@dataclass
class GenPaths:
    """Where the inputs live. The CLI fills these with computed defaults."""

    data_dir: str
    classifications_dir: str
    generation_sources_dir: str
    config_path: str | None = None


@dataclass
class SuiteSpec:
    name: str
    generate: Callable[["GenPaths"], list[dict]]
    network: bool = False


def _csv_suite(suite: str, csv_name: str) -> Callable[[GenPaths], list[dict]]:
    """Build a CSV-backed suite: parse the CSV, then classify + select like the rest."""

    def gen(paths: GenPaths) -> list[dict]:
        csv_path = str(Path(paths.generation_sources_dir) / csv_name)
        cases = generate_from_csv(csv_path, suite=suite)
        mapping = load_classifications(suite, paths.classifications_dir)
        for case in cases:
            stamp(case, case["user"], mapping)
        cfg = load_suite_config(suite, paths.config_path)
        return cfg.select(cases)

    return gen


def _dataset_suite(module) -> Callable[[GenPaths], list[dict]]:
    def gen(paths: GenPaths) -> list[dict]:
        return module.load_and_generate(
            paths.data_dir, paths.classifications_dir, paths.config_path
        )

    return gen


SUITES: dict[str, SuiteSpec] = {
    s.name: s
    for s in [
        SuiteSpec("simple_facts", _csv_suite("simple_facts", "simple_facts.csv")),
        SuiteSpec(
            "simple_facts_regressions",
            _csv_suite("simple_facts_regressions", "simple_facts_regressions.csv"),
        ),
        SuiteSpec("agentharm_refusal", _dataset_suite(agentharm)),
        SuiteSpec("multifaceted", _dataset_suite(multifaceted)),
        SuiteSpec("research_rubrics", _dataset_suite(research_rubrics)),
    ]
}


def suite_names() -> list[str]:
    return list(SUITES)


def all_suite_names(include_network: bool = False) -> list[str]:
    return [n for n, s in SUITES.items() if include_network or not s.network]


def generate_suite(name: str, paths: GenPaths) -> list[dict]:
    if name not in SUITES:
        raise KeyError(f"unknown suite {name!r}; known: {sorted(SUITES)}")
    return SUITES[name].generate(paths)


def write_suite(name: str, out_dir: str, paths: GenPaths) -> int:
    """Generate ``name`` and write ``<out_dir>/<name>.json``; return the test count."""
    cases = generate_suite(name, paths)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}.json"), "w", encoding="utf-8") as fh:
        json.dump(cases, fh, indent=2, ensure_ascii=False)
    return len(cases)


def default_paths(config_path: str | None = None) -> GenPaths:
    """Computed defaults so ``llmeval generate`` works from anywhere.

    Datasets/classifications live at the repo-root ``data/``; CSV sources live in
    the rewrite's ``generation_sources/``. ``config_path`` falls back to the
    ``SUITE_GENERATION_CONFIG_FILE`` env var, then the bundled default config.
    """
    pkg = Path(llmeval.__file__).resolve().parent          # .../rewrite/llmeval
    rewrite_root = pkg.parent                               # .../rewrite
    repo_root = rewrite_root.parent                         # .../llm_test_suite
    data_dir = repo_root / "data"
    bundled_cfg = rewrite_root / "suite_generation_config.json"
    resolved_cfg = (
        config_path
        or os.environ.get(CONFIG_ENV_VAR)
        or (str(bundled_cfg) if bundled_cfg.exists() else None)
    )
    return GenPaths(
        data_dir=str(data_dir),
        classifications_dir=str(data_dir / "classifications"),
        generation_sources_dir=str(rewrite_root / "generation_sources"),
        config_path=resolved_cfg,
    )

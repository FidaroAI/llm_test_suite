"""``HfDatasetPlugin`` — a plugin backed by a Hugging Face dataset.

Three plugins (``agentharm_refusal``, ``multifaceted``, ``research_rubrics``) differ only in
their dataset coordinates and their row-to-test-case transform. Everything else — download
once into the cache directory, transform, write, serve — is identical, so it lives here and
each plugin is its transform plus three constants.

The transform is passed in rather than subclassed because it is a pure function of the rows:
keeping it that way is what lets each plugin's tests check the shaping with a literal list of
rows and no I/O at all.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from llmeval.generation.common import drop_duplicate_ids
from llmeval.generation.hf_rows import cached_rows
from llmeval.plugins import PluginInterface, TestCasePlugin

logger = logging.getLogger(__name__)

DATASET_FILE = "dataset.json"
CACHE_FILE = "testcases.json"

# rows -> test-case dicts with local ids.
Transform = Callable[[list[dict[str, Any]]], list[dict[str, Any]]]


class HfDatasetPlugin(TestCasePlugin):
    """Download once, transform, serve. Ports the legacy ``scripts_repo/download_*.mjs``."""

    def __init__(
        self,
        interface: PluginInterface,
        *,
        dataset: str,
        config: str,
        split: str,
        transform: Transform,
        token: str | None = None,
        gated_hint: str | None = None,
    ):
        self.interface = interface
        self.dataset = dataset
        self.config = config
        self.split = split
        self.transform = transform
        self.token = token
        self.gated_hint = gated_hint
        cache = interface.cache_directory()
        self.dataset_path = cache / DATASET_FILE
        self.output_path = cache / CACHE_FILE

    def download(self) -> list[dict[str, Any]]:
        """Fetch the raw dataset, or reuse the cached copy. Overridden in tests."""
        return cached_rows(
            self.dataset_path, self.dataset, self.config, self.split,
            token=self.token, gated_hint=self.gated_hint,
        )

    def generate_testcases(self) -> bool:
        try:
            rows = self.download()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            # A download failure is reported, not raised: `llmeval generate` with no
            # arguments runs every plugin, and one unreachable dataset should not deny
            # somebody the other five suites.
            logger.error("%s: download failed (%s)", self.interface.name, exc)
            return False
        # Datasets repeat prompts, and a repeated prompt is a repeated id. Drop them here
        # rather than let the loader refuse the whole source at run time.
        cases = drop_duplicate_ids(self.transform(rows), self.interface.name)
        self.output_path.write_text(
            json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info(
            "%s: generated %d test case(s) from %d row(s)",
            self.interface.name, len(cases), len(rows),
        )
        return True

    def get_testcases(self) -> list[dict[str, Any]]:
        if not self.output_path.is_file():
            return []
        return json.loads(self.output_path.read_text(encoding="utf-8"))

"""``CsvTestCasePlugin`` — the whole of a CSV-backed plugin.

Two plugins are nothing but "parse this CSV" (``simple_facts``,
``simple_facts_regressions``), so the body lives here and each plugin's ``__init__.py`` is a
CSV name and a factory. Plugins importing shared machinery out of ``llmeval.generation`` is
the intended arrangement; the dependency never runs the other way.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from llmeval.generation.common import drop_duplicate_ids
from llmeval.generation.csv_source import rows_from_csv
from llmeval.plugins import PluginInterface, TestCasePlugin

logger = logging.getLogger(__name__)

CACHE_FILE = "testcases.json"


class CsvTestCasePlugin(TestCasePlugin):
    """Generates from a CSV into the cache directory; serves what it finds there.

    Writing the file is not strictly necessary — the CSV could be parsed on every call — but
    having the generated output on disk is what makes "why did this test come out like that?"
    answerable without a debugger.
    """

    def __init__(self, interface: PluginInterface, csv_path: Path | str):
        self.interface = interface
        self.csv_path = Path(csv_path)
        self.output_path = interface.cache_directory() / CACHE_FILE

    def generate_testcases(self) -> bool:
        try:
            cases = rows_from_csv(str(self.csv_path))
        except (OSError, ValueError) as exc:
            logger.error("%s: cannot read %s (%s)", self.interface.name, self.csv_path, exc)
            return False
        # Two rows with the same question hash to the same id; keep the first.
        cases = drop_duplicate_ids(cases, self.interface.name)
        self.output_path.write_text(
            json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("%s: generated %d test case(s)", self.interface.name, len(cases))
        return True

    def get_testcases(self) -> list[dict[str, Any]]:
        if not self.output_path.is_file():
            return []
        return json.loads(self.output_path.read_text(encoding="utf-8"))

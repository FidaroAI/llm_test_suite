"""``simple_facts`` — short factual questions with an ``icontains`` check each.

Nothing but a CSV, so the whole plugin is
:class:`~llmeval.generation.csv_plugin.CsvTestCasePlugin` pointed at the file next door. Edit
``simple_facts.csv`` and re-run ``llmeval generate --testcases simple_facts``.
"""

from pathlib import Path

from llmeval.generation.csv_plugin import CsvTestCasePlugin
from llmeval.plugins import PluginInterface, TestCasePlugin

CSV_PATH = Path(__file__).resolve().parent / "simple_facts.csv"


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return CsvTestCasePlugin(interface, CSV_PATH)

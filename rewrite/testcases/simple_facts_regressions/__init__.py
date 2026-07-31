"""``simple_facts_regressions`` — factual questions Fidaro has got wrong before.

Same shape as ``simple_facts``: a CSV and nothing else. A question earns a row here once it
has regressed, so the file is the record of what must not break again.
"""

from pathlib import Path

from llmeval.generation.csv_plugin import CsvTestCasePlugin
from llmeval.plugins import PluginInterface, TestCasePlugin

CSV_PATH = Path(__file__).resolve().parent / "simple_facts_regressions.csv"


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return CsvTestCasePlugin(interface, CSV_PATH)

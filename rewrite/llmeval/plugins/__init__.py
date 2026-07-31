"""Test-case plugins: the contract, and the loader that finds and imports them.

Plugin authors import from here::

    from llmeval.plugins import PluginInterface, TestCasePlugin
"""

from llmeval.plugins.base import GradingOutcome, PluginInterface, TestCasePlugin
from llmeval.plugins.loader import (
    DEFAULT_ROOT,
    Hooks,
    Loaded,
    Source,
    SourceError,
    load,
    source_of,
)

__all__ = [
    "DEFAULT_ROOT",
    "GradingOutcome",
    "Hooks",
    "Loaded",
    "PluginInterface",
    "Source",
    "SourceError",
    "TestCasePlugin",
    "load",
    "source_of",
]

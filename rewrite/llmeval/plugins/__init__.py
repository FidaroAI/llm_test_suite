"""Test-case plugins: the contract, and the loader that finds and imports them.

Plugin authors import from here::

    from llmeval.plugins import PluginInterface, TestCasePlugin
"""

from llmeval.plugins.base import GradingOutcome, PluginInterface, TestCasePlugin

__all__ = ["GradingOutcome", "PluginInterface", "TestCasePlugin"]

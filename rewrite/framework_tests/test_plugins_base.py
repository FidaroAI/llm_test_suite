import pytest

from llmeval.plugins import PluginInterface, TestCasePlugin


def test_cache_directory_is_created_under_the_plugin_name(tmp_path):
    iface = PluginInterface("stock_prices", tmp_path / ".llmeval.cache")
    path = iface.cache_directory()
    assert path == tmp_path / ".llmeval.cache" / "stock_prices"
    assert path.is_dir()
    assert iface.name == "stock_prices"


def test_cache_directory_is_idempotent(tmp_path):
    iface = PluginInterface("x", tmp_path / "c")
    assert iface.cache_directory() == iface.cache_directory()


def test_plugin_requires_the_two_abstract_methods():
    class Incomplete(TestCasePlugin):
        pass

    with pytest.raises(TypeError):
        Incomplete()  # pylint: disable=abstract-class-instantiated


def test_hooks_and_assertions_default_to_no_ops():
    class Minimal(TestCasePlugin):
        def generate_testcases(self):
            return True

        def get_testcases(self):
            return []

    plugin = Minimal()
    assert plugin.get_custom_assertions() == {}
    for name in ("before_run", "after_run", "before_grade", "after_grade"):
        assert getattr(plugin, name)() is None
    assert plugin.before_each_run(None) is None
    assert plugin.after_each_run(None, None) is None
    assert plugin.before_each_grade(None) is None
    assert plugin.after_each_grade(None, []) is None

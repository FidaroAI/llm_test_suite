"""Loading `.env` — the one boring failure the wizard exists to prevent."""

import os

from porcelain.env import load_env, parse_env


def test_parses_plain_assignments():
    assert parse_env("A=1\nB=two\n") == {"A": "1", "B": "two"}


def test_ignores_comments_and_blank_lines():
    assert parse_env("# note\n\n  \nA=1\n# another\n") == {"A": "1"}


def test_strips_export_prefix():
    assert parse_env("export A=1\n") == {"A": "1"}


def test_strips_matching_quotes():
    assert parse_env("A='1'\nB=\"two\"\n") == {"A": "1", "B": "two"}


def test_leaves_unmatched_quotes_alone():
    assert parse_env("A=\"unclosed\n") == {"A": '"unclosed'}


def test_keeps_inner_equals_signs():
    """Bearer tokens and base64 secrets routinely contain '='."""
    assert parse_env("TOKEN=abc=def==\n") == {"TOKEN": "abc=def=="}


def test_empty_value_is_kept():
    """`.env.example` ships keys with blank values; they must not vanish silently."""
    assert parse_env("AWS_BEARER_TOKEN_BEDROCK=\n") == {"AWS_BEARER_TOKEN_BEDROCK": ""}


def test_lines_without_an_equals_are_skipped():
    assert parse_env("nonsense\nA=1\n") == {"A": "1"}


def test_urls_survive_intact():
    text = "FIDARO_PROD_BASE_URL=http://127.0.0.1:8082/v2\n"
    assert parse_env(text) == {"FIDARO_PROD_BASE_URL": "http://127.0.0.1:8082/v2"}


def test_load_env_sets_variables(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("PORCELAIN_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.delenv("PORCELAIN_TEST_KEY", raising=False)
    assert load_env(env_file) == env_file
    assert os.environ["PORCELAIN_TEST_KEY"] == "from-file"


def test_existing_environment_wins(tmp_path, monkeypatch):
    """`FOO=x llmevalx` has to beat the file, or overriding anything is impossible."""
    env_file = tmp_path / ".env"
    env_file.write_text("PORCELAIN_TEST_KEY=from-file\n", encoding="utf-8")
    monkeypatch.setenv("PORCELAIN_TEST_KEY", "from-shell")
    load_env(env_file)
    assert os.environ["PORCELAIN_TEST_KEY"] == "from-shell"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_env(tmp_path / "absent") is None

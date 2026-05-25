"""Unit tests for the pure logic in scripts_repo/batch_comparison.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts_repo.batch_comparison import (
    config_filename,
    discover_prompts,
    generate_config,
    run_comparison_command,
    write_generated_configs,
)


def _template() -> dict:
    return {
        "vllm-prod-url": "https://prod/v1",
        "vllm-dev-url": "https://dev/v1",
        "system-prompt-file": "system_prompts/fidaro_prod.md",
        "vllm-options": {"model": "x"},
        "suite-generation-config": {"simple_facts": {"number_to_generate": 1}},
    }


# --- discover_prompts ------------------------------------------------------


def test_discover_prompts_returns_sorted_md_files(tmp_path: Path):
    (tmp_path / "b.md").write_text("b", encoding="utf-8")
    (tmp_path / "a.md").write_text("a", encoding="utf-8")
    assert discover_prompts(tmp_path) == [tmp_path / "a.md", tmp_path / "b.md"]


def test_discover_prompts_ignores_non_md_and_does_not_recurse(tmp_path: Path):
    (tmp_path / "keep.md").write_text("k", encoding="utf-8")
    (tmp_path / "skip.txt").write_text("s", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "deep.md").write_text("d", encoding="utf-8")
    assert discover_prompts(tmp_path) == [tmp_path / "keep.md"]


# --- config_filename -------------------------------------------------------


def test_config_filename_is_prompt_stem_with_json_extension():
    assert config_filename(Path("/x/test_system_prompt_1.md")) == "test_system_prompt_1.json"


# --- generate_config -------------------------------------------------------


def test_generate_config_overwrites_system_prompt_file_with_absolute_path(tmp_path: Path):
    prompt = tmp_path / "p.md"
    prompt.write_text("hi", encoding="utf-8")
    cfg = generate_config(_template(), prompt)
    assert cfg["system-prompt-file"] == str(prompt.resolve())
    assert Path(cfg["system-prompt-file"]).is_absolute()


def test_generate_config_preserves_all_other_template_keys():
    template = _template()
    cfg = generate_config(template, Path("/x/p.md"))
    for key, value in template.items():
        if key == "system-prompt-file":
            continue
        assert cfg[key] == value


def test_generate_config_does_not_mutate_the_template():
    template = _template()
    original = json.loads(json.dumps(template))
    generate_config(template, Path("/x/p.md"))
    assert template == original


# --- write_generated_configs -----------------------------------------------


def test_write_generated_configs_writes_one_json_per_prompt(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "one.md").write_text("1", encoding="utf-8")
    (prompts_dir / "two.md").write_text("2", encoding="utf-8")
    out_dir = tmp_path / "out"

    written = write_generated_configs(
        discover_prompts(prompts_dir), _template(), out_dir
    )

    assert {p.name for p in written} == {"one.json", "two.json"}
    one = json.loads((out_dir / "one.json").read_text(encoding="utf-8"))
    assert one["system-prompt-file"] == str((prompts_dir / "one.md").resolve())
    assert one["vllm-prod-url"] == "https://prod/v1"


def test_write_generated_configs_creates_missing_output_dir(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "one.md").write_text("1", encoding="utf-8")
    out_dir = tmp_path / "does" / "not" / "exist"

    write_generated_configs(discover_prompts(prompts_dir), _template(), out_dir)

    assert (out_dir / "one.json").is_file()


# --- run_comparison_command ------------------------------------------------


def test_run_comparison_command_invokes_script_with_config_and_forwarded_args():
    repo_root = Path("/repo")
    cmd = run_comparison_command(
        Path("/repo/out/foo.json"), ["--yes", "--skip-phala-deploy"], repo_root
    )
    assert cmd == [
        sys.executable,
        str(repo_root / "scripts_repo" / "run_comparison.py"),
        "/repo/out/foo.json",
        "--yes",
        "--skip-phala-deploy",
    ]


def test_run_comparison_command_with_no_forwarded_args():
    repo_root = Path("/repo")
    cmd = run_comparison_command(Path("/repo/out/foo.json"), [], repo_root)
    assert cmd == [
        sys.executable,
        str(repo_root / "scripts_repo" / "run_comparison.py"),
        "/repo/out/foo.json",
    ]

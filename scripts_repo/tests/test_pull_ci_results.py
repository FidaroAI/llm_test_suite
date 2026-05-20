import argparse
import subprocess

import pytest

from scripts_repo.pull_ci_results import parse_date


def test_parse_date_valid():
    result = parse_date("2026-05-14")
    assert (result.year, result.month, result.day) == (2026, 5, 14)


def test_parse_date_invalid_format_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_date("14/05/2026")


def test_parse_date_garbage_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_date("not-a-date")


from scripts_repo.pull_ci_results import build_filename, select_run


def test_build_filename_truncates_sha():
    assert build_filename("2026-05-14", "4a9cae7abcdef") == "2026-05-14_4a9cae7.json"


def test_build_filename_short_sha_unchanged():
    assert build_filename("2026-05-14", "4a9cae7") == "2026-05-14_4a9cae7.json"


def test_select_run_picks_most_recent_successful():
    runs = [
        {"databaseId": 1, "createdAt": "2026-05-10T00:00:00Z",
         "status": "completed", "conclusion": "success"},
        {"databaseId": 2, "createdAt": "2026-05-12T00:00:00Z",
         "status": "completed", "conclusion": "success"},
    ]
    assert select_run(runs)["databaseId"] == 2


def test_select_run_ignores_failed_and_incomplete():
    runs = [
        {"databaseId": 1, "createdAt": "2026-05-12T00:00:00Z",
         "status": "completed", "conclusion": "failure"},
        {"databaseId": 2, "createdAt": "2026-05-11T00:00:00Z",
         "status": "in_progress", "conclusion": None},
        {"databaseId": 3, "createdAt": "2026-05-10T00:00:00Z",
         "status": "completed", "conclusion": "success"},
    ]
    assert select_run(runs)["databaseId"] == 3


def test_select_run_returns_none_when_no_success():
    runs = [
        {"databaseId": 1, "createdAt": "2026-05-12T00:00:00Z",
         "status": "completed", "conclusion": "failure"},
    ]
    assert select_run(runs) is None


def test_select_run_empty_list_returns_none():
    assert select_run([]) is None


from scripts_repo.pull_ci_results import (
    find_result_json,
    parse_commit_list,
    read_eval_id,
)


def test_parse_commit_list_strips_blank_lines():
    output = "abc123\n\ndef456\n  \n789ghi\n"
    assert parse_commit_list(output) == ["abc123", "def456", "789ghi"]


def test_parse_commit_list_empty():
    assert parse_commit_list("") == []


def test_find_result_json_finds_nested_single_file(tmp_path):
    nested = tmp_path / "results" / "ci"
    nested.mkdir(parents=True)
    target = nested / "latest.json"
    target.write_text("{}")
    assert find_result_json(tmp_path) == target


def test_find_result_json_raises_when_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        find_result_json(tmp_path)


def test_find_result_json_raises_when_ambiguous(tmp_path):
    (tmp_path / "a.json").write_text("{}")
    (tmp_path / "b.json").write_text("{}")
    with pytest.raises(ValueError):
        find_result_json(tmp_path)


def test_read_eval_id_returns_eval_id(tmp_path):
    f = tmp_path / "result.json"
    f.write_text('{"evalId": "eval-XYZ-2026-05-14T02:56:33", "results": {}}')
    assert read_eval_id(f) == "eval-XYZ-2026-05-14T02:56:33"


def test_read_eval_id_raises_when_absent(tmp_path):
    f = tmp_path / "result.json"
    f.write_text('{"results": {}}')
    with pytest.raises(ValueError):
        read_eval_id(f)


from scripts_repo.pull_ci_results import classify_import_failure


def test_classify_import_failure_benign_when_id_exists():
    assert classify_import_failure("eval-1", ["eval-0", "eval-1"]) == "benign"


def test_classify_import_failure_genuine_when_id_absent():
    assert classify_import_failure("eval-1", ["eval-0", "eval-2"]) == "genuine"


def test_classify_import_failure_genuine_when_no_existing_ids():
    assert classify_import_failure("eval-1", []) == "genuine"


from scripts_repo import pull_ci_results as pcr


def test_gh_runs_for_commit_parses_json(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(
            cmd, 0, '[{"databaseId": 7, "headSha": "abc"}]', ""
        )

    monkeypatch.setattr(pcr, "_run", fake_run)
    runs = pcr.gh_runs_for_commit("abc")
    assert runs == [{"databaseId": 7, "headSha": "abc"}]


def test_gh_runs_for_commit_raises_on_error(monkeypatch):
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", "gh: not authenticated")

    monkeypatch.setattr(pcr, "_run", fake_run)
    with pytest.raises(RuntimeError, match="gh run list failed"):
        pcr.gh_runs_for_commit("abc")


def _ok(cmd, **kwargs):
    return subprocess.CompletedProcess(cmd, 0, "", "")


def _fail(stderr):
    def _inner(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 1, "", stderr)
    return _inner


def test_import_results_imports_new_file(monkeypatch, tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-new"}')
    monkeypatch.setattr(pcr, "promptfoo_import", lambda p: _ok([]))
    # Should not raise.
    pcr.import_results([f], fail_on_existing=False)


def test_import_results_skips_benign_duplicate(monkeypatch, tmp_path, capsys):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-dup"}')
    monkeypatch.setattr(pcr, "promptfoo_import", _fail("already exists"))
    monkeypatch.setattr(pcr, "promptfoo_eval_ids", lambda: ["eval-dup"])
    pcr.import_results([f], fail_on_existing=False)  # should not raise
    assert "already imported" in capsys.readouterr().err


def test_import_results_aborts_on_existing_when_strict(monkeypatch, tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-dup"}')
    monkeypatch.setattr(pcr, "promptfoo_import", _fail("already exists"))
    monkeypatch.setattr(pcr, "promptfoo_eval_ids", lambda: ["eval-dup"])
    with pytest.raises(SystemExit):
        pcr.import_results([f], fail_on_existing=True)


def test_import_results_aborts_on_genuine_error(monkeypatch, tmp_path):
    f = tmp_path / "a.json"
    f.write_text('{"evalId": "eval-new"}')
    monkeypatch.setattr(pcr, "promptfoo_import", _fail("disk full"))
    monkeypatch.setattr(pcr, "promptfoo_eval_ids", lambda: ["eval-other"])
    with pytest.raises(SystemExit):
        pcr.import_results([f], fail_on_existing=False)


def test_main_rejects_date_flags_without_all(monkeypatch):
    monkeypatch.setattr(pcr, "mode_latest", lambda: [])
    monkeypatch.setattr(pcr, "import_results", lambda paths, fail_on_existing: None)
    rc = pcr.main(["--latest", "--start-date", "2026-01-01"])
    assert rc == 2


def test_main_dispatches_to_commit_mode(monkeypatch):
    seen = {}
    monkeypatch.setattr(pcr, "mode_commit", lambda sha: seen.setdefault("sha", sha) or [])
    monkeypatch.setattr(pcr, "import_results", lambda paths, fail_on_existing: None)
    rc = pcr.main(["--commit", "abc123"])
    assert rc == 0
    assert seen["sha"] == "abc123"


def test_main_no_import_skips_import_phase(monkeypatch):
    called = []
    monkeypatch.setattr(pcr, "mode_latest", lambda: [])
    monkeypatch.setattr(pcr, "import_results", lambda *a, **k: called.append(1))
    rc = pcr.main(["--latest", "--no-import"])
    assert rc == 0
    assert called == []


def test_main_requires_a_mode(monkeypatch):
    with pytest.raises(SystemExit):
        pcr.main([])

# scripts_repo/tests/test_freeze_baseline.py
from scripts_repo.freeze_baseline import (
    sanitize_label,
    provider_labels,
    filter_to_provider,
    gather_test_keys,
    build_baseline,
)
from scripts_repo.tests._fixtures import rubric_result, make_eval_json


def test_sanitize_label_replaces_unsafe_chars():
    assert sanitize_label("openai:chat/Qwen 80B") == "openai_chat_Qwen_80B"


def test_sanitize_label_empty_falls_back():
    assert sanitize_label("///") == "provider"


def test_provider_labels_dedupes_in_order():
    ev = make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("cand", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("prod", "t2", "research_rubrics", [("b", "x", 1)], [1.0]),
    ])
    assert provider_labels(ev) == ["prod", "cand"]


def test_filter_to_provider_keeps_only_one():
    ev = make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("cand", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
    ])
    out = filter_to_provider(ev, "prod")
    labels = {r["provider"]["label"] for r in out["results"]["results"]}
    assert labels == {"prod"}
    # original is untouched (deep copy)
    assert len(ev["results"]["results"]) == 2


def test_gather_test_keys_sorted_unique_nonnull():
    ev = make_eval_json([
        rubric_result("prod", "t2", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("prod", "t1", "research_rubrics", [("b", "x", 1)], [1.0]),
    ])
    assert gather_test_keys(ev) == ["t1", "t2"]


def test_build_baseline_adds_meta():
    ev = make_eval_json(
        [rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0])],
        eval_id="eval-abc",
    )
    out = build_baseline(ev, "prod", "deadbeef")
    meta = out["_baseline_meta"]
    assert meta["provider_label"] == "prod"
    assert meta["git_sha"] == "deadbeef"
    assert meta["eval_id"] == "eval-abc"
    assert meta["test_keys"] == ["t1"]
    assert "frozen_at" in meta


import json
from pathlib import Path

import pytest

from scripts_repo.freeze_baseline import freeze


def _write_eval(tmp_path, ev):
    p = tmp_path / "latest.json"
    p.write_text(json.dumps(ev), encoding="utf-8")
    return p


def test_freeze_writes_one_file_per_provider(tmp_path):
    ev = make_eval_json([
        rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0]),
        rubric_result("cand", "t1", "research_rubrics", [("a", "x", 1)], [0.5]),
    ])
    src = _write_eval(tmp_path, ev)
    out_dir = tmp_path / "baselines"
    written = freeze(src, out_dir, force=False)
    names = sorted(p.name for p in written)
    assert names == ["cand.json", "prod.json"]
    prod = json.loads((out_dir / "prod.json").read_text())
    labels = {r["provider"]["label"] for r in prod["results"]["results"]}
    assert labels == {"prod"}
    assert prod["_baseline_meta"]["provider_label"] == "prod"


def test_freeze_refuses_overwrite_without_force(tmp_path):
    ev = make_eval_json(
        [rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0])]
    )
    src = _write_eval(tmp_path, ev)
    out_dir = tmp_path / "baselines"
    freeze(src, out_dir, force=False)
    with pytest.raises(FileExistsError):
        freeze(src, out_dir, force=False)


def test_freeze_force_overwrites(tmp_path):
    ev = make_eval_json(
        [rubric_result("prod", "t1", "research_rubrics", [("a", "x", 1)], [1.0])]
    )
    src = _write_eval(tmp_path, ev)
    out_dir = tmp_path / "baselines"
    freeze(src, out_dir, force=False)
    freeze(src, out_dir, force=True)  # must not raise


def test_freeze_no_providers_raises(tmp_path):
    src = _write_eval(tmp_path, make_eval_json([]))
    with pytest.raises(ValueError):
        freeze(src, tmp_path / "baselines", force=False)

# scripts_repo/tests/_fixtures.py
"""Synthetic promptfoo-eval JSON builders for unit tests (no network)."""

from __future__ import annotations


def rubric_result(
    provider_label,
    description,
    suite,
    asserts,
    scores,
    prompt_label="user_only",
    metadata_extra=None,
):
    """Build one eval_json["results"]["results"][i] entry of llm-rubric asserts.

    asserts: list of (value, metric, weight) tuples.
    scores:  list of component scores, index-aligned with asserts.
    """
    metadata = {"suite": suite}
    if metadata_extra:
        metadata.update(metadata_extra)
    assert_objs = [
        {"type": "llm-rubric", "value": v, "metric": m, "weight": w}
        for (v, m, w) in asserts
    ]
    comps = [{"score": s, "pass": s >= 0.5} for s in scores]
    return {
        "provider": {"id": "x", "label": provider_label},
        "prompt": {"label": prompt_label},
        "vars": {"user": "..."},
        "testCase": {
            "description": description,
            "vars": {"user": "..."},
            "assert": assert_objs,
            "metadata": metadata,
        },
        "gradingResult": {
            "score": (sum(scores) / len(scores)) if scores else 0,
            "componentResults": comps,
        },
    }


def make_eval_json(results, eval_id="eval-test"):
    """Wrap a list of result entries in the eval-result-JSON envelope."""
    return {"evalId": eval_id, "results": {"results": list(results)}}

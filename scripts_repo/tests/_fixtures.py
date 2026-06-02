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
    assertion_type="llm-rubric",
):
    """Build one eval_json["results"]["results"][i] entry of graded asserts.

    asserts: list of (value, metric, weight) tuples.
    scores:  list of component scores, index-aligned with asserts.
    assertion_type: the promptfoo assertion ``type`` to stamp on each assert.
        Defaults to ``llm-rubric``; pass ``"g-eval"`` to build a g-eval row.
    """
    metadata = {"suite": suite}
    if metadata_extra:
        metadata.update(metadata_extra)
    assert_objs = [
        {"type": assertion_type, "value": v, "metric": m, "weight": w}
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


def deterministic_result(
    provider_label,
    description,
    suite,
    asserts,
    passes,
    prompt_label="user_only",
    metadata_extra=None,
):
    """Build one result entry of deterministic (non-llm-rubric) asserts.

    asserts: list of (type, value) tuples, e.g. ("icontains", "Paris").
    passes:  list of booleans, index-aligned with asserts.
    """
    metadata = {"suite": suite} if suite is not None else {}
    if metadata_extra:
        metadata.update(metadata_extra)
    assert_objs = [{"type": t, "value": v} for (t, v) in asserts]
    comps = [{"score": 1.0 if p else 0.0, "pass": p} for p in passes]
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
            "score": (sum(1 for p in passes if p) / len(passes)) if passes else 0,
            "componentResults": comps,
        },
    }


def response_result(
    provider_label,
    description,
    suite,
    output,
    prompt_raw=None,
    prompt_label="user_only",
):
    """Build one result entry carrying a raw ``response.output`` (and prompt.raw).

    Used by the response-CSV tests, which read the raw model output and the
    rendered prompt rather than graded assertions. ``output`` is stored verbatim
    (pre-transform, reasoning prefix and all); ``prompt_raw`` is the promptfoo
    ``prompt.raw`` (typically a JSON chat-messages array string).
    """
    return {
        "provider": {"id": "x", "label": provider_label},
        "prompt": {"label": prompt_label, "raw": prompt_raw if prompt_raw is not None else ""},
        "vars": {"user": "..."},
        "response": {"output": output},
        "testCase": {
            "description": description,
            "vars": {"user": "..."},
            "assert": [],
            "metadata": {"suite": suite} if suite is not None else {},
        },
    }


def make_eval_json(results, eval_id="eval-test"):
    """Wrap a list of result entries in the eval-result-JSON envelope."""
    return {"evalId": eval_id, "results": {"results": list(results)}}

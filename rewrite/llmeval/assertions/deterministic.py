"""Deterministic assertions — local, no LLM calls, score is 1.0/0.0."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

from llmeval.assertions.base import AssertionResult, GradeContext, register
from llmeval.models import AssertionSpec

# Every grader takes (spec, output, ctx) to satisfy the registry's protocol; the
# deterministic ones don't need ctx.
# pylint: disable=unused-argument


def _text(output: Any) -> str:
    return output if isinstance(output, str) else str(output if output is not None else "")


def _result(passed: bool, reason: str) -> AssertionResult:
    return AssertionResult(passed=passed, score=1.0 if passed else 0.0, reason=reason)


@register("contains")
def _contains(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    hit = str(spec.value) in _text(output)
    return _result(hit, f"{'found' if hit else 'missing'} substring {spec.value!r}")


@register("icontains")
def _icontains(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    hit = str(spec.value).lower() in _text(output).lower()
    return _result(hit, f"{'found' if hit else 'missing'} (ci) {spec.value!r}")


@register("equals")
def _equals(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    hit = _text(output) == str(spec.value)
    return _result(hit, "exact match" if hit else "not an exact match")


@register("not_contains")
def _not_contains(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    present = str(spec.value).lower() in _text(output).lower()
    return _result(not present, f"{spec.value!r} {'present' if present else 'absent'}")


@register("regex")
def _regex(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    flags = re.I if spec.params.get("ignore_case") else 0
    hit = re.search(str(spec.value), _text(output), flags) is not None
    return _result(hit, f"pattern {spec.value!r} {'matched' if hit else 'no match'}")


@register("length")
def _length(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    unit = spec.params.get("unit", "chars")
    text = _text(output)
    if unit == "chars":
        n = len(text)
    elif unit == "words":
        n = len(text.split())
    elif unit == "tokens":
        n = _count_tokens(text, spec.params.get("encoding", "cl100k_base"))
    else:
        raise ValueError(f"unknown length unit: {unit}")
    lo = spec.params.get("min", 0)
    hi = spec.params.get("max")
    ok = n >= lo and (hi is None or n <= hi)
    return _result(ok, f"{n} {unit} (min={lo}, max={hi})")


def _count_tokens(text: str, encoding: str) -> int:
    try:  # tiktoken is optional; fall back to a word count approximation
        import tiktoken

        return len(tiktoken.get_encoding(encoding).encode(text))
    except Exception:  # pylint: disable=broad-exception-caught
        return len(text.split())


# Refusal / over-refusal sweep — ported verbatim from the old assert_no_censorship.py.
_REFUSAL_PATTERNS = [
    r"\bI (?:can(?:not|'t)|won't|am unable to)\b",
    r"\bAs an? (?:AI|language model)\b",
    r"\bI'?m sorry,? but\b",
    r"\bagainst my (?:guidelines|policy|programming|instructions)\b",
    r"\bI (?:must|have to) decline\b",
    r"\bI (?:do not|don't) feel comfortable\b",
]
_REFUSAL_RX = re.compile("|".join(_REFUSAL_PATTERNS), re.I)


@register("refusal")
def _refusal(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    if spec.params.get("allow"):
        return AssertionResult(True, 1.0, "refusal allowed by assertion config")
    matches = _REFUSAL_RX.findall(_text(output))
    if matches:
        return AssertionResult(False, 0.0, f"refusal language detected: {matches[:3]}")
    return AssertionResult(True, 1.0, "no refusal markers")


# Stock-price freshness check — ported from the legacy assert_stock_price.py.
# Network-free: the live reference is baked into ``spec.params`` at generation time
# by the stock_prices generator, so grading just parses the answer and compares.
#
# Params: reference_price (float), reference_currency (e.g. "USD"/"GBp"),
# reference_fetched_at (ISO-8601), symbol (for readable reasons), tolerance_pct
# (default 1.0), max_age_hours (default 24).
_NUMBER_RE = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> list[float]:
    out = []
    for tok in _NUMBER_RE.findall(text):
        try:
            out.append(float(tok.replace(",", "")))
        except ValueError:
            continue
    return out


def _stale_age_hours(fetched_at: Any, max_age_hours: float) -> float | None:
    """Return the age in hours if it exceeds ``max_age_hours``, else None."""
    if not fetched_at:
        return None  # unknown age; don't block on it
    try:
        ts = datetime.fromisoformat(str(fetched_at))
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - ts).total_seconds() / 3600
    return age_h if age_h > max_age_hours else None


@register("stock_price")
def _stock_price(spec: AssertionSpec, output: Any, ctx: GradeContext) -> AssertionResult:
    p = spec.params
    symbol = p.get("symbol") or "?"
    reference = p.get("reference_price")
    currency = p.get("reference_currency") or ""
    tol_pct = float(p.get("tolerance_pct", 1.0))
    max_age_hours = float(p.get("max_age_hours", 24))

    if reference is None:
        return AssertionResult(
            False, 0.0,
            f"no reference price for {symbol}: regenerate the stock_prices suite",
        )
    try:
        reference = float(reference)
    except (TypeError, ValueError):
        return AssertionResult(False, 0.0, f"reference for {symbol} not a number: {reference!r}")

    stale = _stale_age_hours(p.get("reference_fetched_at"), max_age_hours)
    if stale is not None:
        return AssertionResult(
            False, 0.0,
            f"reference for {symbol} is stale ({stale:.1f}h > {max_age_hours:.0f}h): "
            "regenerate the stock_prices suite",
        )

    candidates = _extract_numbers(_text(output))
    if not candidates:
        return AssertionResult(
            False, 0.0, f"no number found in answer for {symbol} (ref {reference:g} {currency})"
        )

    # GBp listings are quoted in pence; an answer in pounds is reference/100.
    targets = [reference] + ([reference / 100] if currency == "GBp" else [])
    best_cand, best_diff = None, float("inf")
    for cand in candidates:
        for target in targets:
            if target == 0:
                continue
            diff = abs(cand - target) / abs(target)
            if diff < best_diff:
                best_diff, best_cand = diff, cand

    within = best_diff <= tol_pct / 100.0
    reason = (
        f"{symbol}: reference {reference:g} {currency}; closest answer "
        f"{best_cand:g} -> {best_diff * 100:.2f}% {'≤' if within else '>'} {tol_pct:g}% tolerance"
    )
    return AssertionResult(within, 1.0 if within else 0.0, reason)

"""Visual comparison report (standalone HTML).

Builds a plain data structure from the store (so it's easy to test and to retarget to
other renderers), then renders it with an autoescaped Jinja2 template — model output is
untrusted and must never render as raw HTML.
"""

from __future__ import annotations

import os
from typing import Any

from jinja2 import Template

from llmeval.comparison import stats
from llmeval.store import Store


def build_report(
    store: Store,
    configs: list[tuple[str, str]],
    metrics: list[str],
    baseline_name: str | None = None,
    comparison_key: str | None = None,
    attempt_reducer: str = "mean",
) -> dict[str, Any]:
    name_by_hash = {h: n for n, h in configs}

    metric_blocks: list[dict[str, Any]] = []
    per: dict[tuple[str, str], dict[str, float]] = {}
    test_ids: set[str] = set()
    for metric in metrics:
        rows = stats.compare_metric(store, configs, metric, baseline_name, attempt_reducer)
        metric_blocks.append(
            {
                "metric": metric or "overall",
                "rows": [
                    {
                        "name": cs.name,
                        "hash": cs.hash,
                        "n": cs.summary.n,
                        "mean": cs.summary.mean,
                        "ci_low": cs.summary.ci_low,
                        "ci_high": cs.summary.ci_high,
                        "delta": cs.delta,
                    }
                    for cs in rows
                ],
            }
        )
        for _name, h in configs:
            scores = stats.per_test_scores(store, h, metric, attempt_reducer)
            per[(metric, h)] = scores
            test_ids.update(scores)

    winrates = None
    winner_by_test: dict[str, str] = {}
    if comparison_key:
        wr = stats.win_rates(store, comparison_key)
        winrates = {
            "total": wr.total,
            "undecided": wr.undecided,
            "rows": [
                {"name": n, "hash": h, "wins": wr.wins.get(h, 0), "rate": wr.rate(h)}
                for n, h in configs
            ],
        }
        for v in store.get_verdicts(comparison_key):
            winner_by_test[v.test_id] = (
                name_by_hash.get(v.winner_hash, v.winner_hash) if v.winner_hash else "undecided"
            )

    tests = []
    for tid in sorted(test_ids):
        cells = [
            {"metric": metric, "config": name, "score": per[(metric, h)].get(tid)}
            for metric in metrics
            for name, h in configs
        ]
        answers = []
        for name, h in configs:
            rows_ = store.get_results(tid, h)
            out = next((r.output for r in rows_ if r.error is None and r.output is not None), None)
            answers.append({"config": name, "output": out})
        tests.append(
            {"test_id": tid, "cells": cells, "winner": winner_by_test.get(tid), "answers": answers}
        )

    return {
        "config_names": [n for n, _ in configs],
        "metrics": metric_blocks,
        "winrates": winrates,
        "tests": tests,
        "attempt_reducer": attempt_reducer,
    }


_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>llmeval comparison</title>
<style>
 body{font-family:system-ui,sans-serif;margin:2rem;color:#1a1a1a}
 table{border-collapse:collapse;margin:1rem 0;font-size:14px}
 th,td{border:1px solid #ccc;padding:.35rem .6rem;text-align:left;vertical-align:top}
 th{background:#f3f3f3}
 .pos{color:#0a7d23}.neg{color:#c0271b}.muted{color:#888}
 details{margin:.3rem 0}pre{white-space:pre-wrap;margin:.2rem 0;font-size:13px}
</style></head><body>
<h1>llmeval comparison</h1>
<p class="muted">Configs: {{ config_names|join(", ") }} · attempts reduced by <b>{{ attempt_reducer }}</b></p>

{% for block in metrics %}
<h2>Metric: {{ block.metric }}</h2>
<table>
 <tr><th>config</th><th>n</th><th>mean</th><th>95% CI</th><th>Δ vs baseline</th></tr>
 {% for r in block.rows %}
 <tr>
  <td>{{ r.name }}</td><td>{{ r.n }}</td><td>{{ "%.3f"|format(r.mean) }}</td>
  <td class="muted">[{{ "%.3f"|format(r.ci_low) }}, {{ "%.3f"|format(r.ci_high) }}]</td>
  <td>{% if r.delta is not none %}<span class="{{ 'pos' if r.delta >= 0 else 'neg' }}">{{ "%+.3f"|format(r.delta) }}</span>{% else %}<span class="muted">baseline</span>{% endif %}</td>
 </tr>
 {% endfor %}
</table>
{% endfor %}

{% if winrates %}
<h2>Pick-best win rates</h2>
<p class="muted">{{ winrates.total }} comparisons · {{ winrates.undecided }} undecided</p>
<table>
 <tr><th>config</th><th>wins</th><th>win rate (of decided)</th></tr>
 {% for r in winrates.rows %}
 <tr><td>{{ r.name }}</td><td>{{ r.wins }}</td><td>{{ "%.0f%%"|format(r.rate * 100) }}</td></tr>
 {% endfor %}
</table>
{% endif %}

<h2>Per-test</h2>
<table>
 <tr><th>test</th>{% for c in config_names %}<th>{{ c }}</th>{% endfor %}<th>winner</th></tr>
 {% for t in tests %}
 <tr>
  <td>{{ t.test_id }}</td>
  {% for a in t.answers %}
   <td>
    {% for cell in t.cells if cell.config == a.config %}<div class="muted">{{ cell.metric or "overall" }}: {% if cell.score is not none %}{{ "%.3f"|format(cell.score) }}{% else %}—{% endif %}</div>{% endfor %}
    {% if a.output %}<details><summary>answer</summary><pre>{{ a.output }}</pre></details>{% endif %}
   </td>
  {% endfor %}
  <td>{{ t.winner or "—" }}</td>
 </tr>
 {% endfor %}
</table>
</body></html>"""


def render_html(report: dict[str, Any]) -> str:
    return Template(_TEMPLATE, autoescape=True).render(**report)


def write_report(
    store: Store,
    configs: list[tuple[str, str]],
    metrics: list[str],
    out_path: str,
    baseline_name: str | None = None,
    comparison_key: str | None = None,
    attempt_reducer: str = "mean",
) -> str:
    report = build_report(
        store, configs, metrics, baseline_name, comparison_key, attempt_reducer
    )
    html = render_html(report)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    return out_path

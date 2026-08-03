# `testcases/` — where test cases live

Everything the suite can run lives in this directory, in one of two forms:

* a **`.json` file** at the top level — hand-written test cases, exactly the shape
  [`llmeval/models.py`](../llmeval/models.py) describes;
* a **plugin directory** — a self-contained Python package that builds its own test cases,
  owns its inputs and downloads, and can bring its own assertions and lifecycle hooks.

Either is called a **source**, and `llmeval --testcases NAME` names one. Omit the flag and
every source is loaded.

Two layout rules, both enforced at load time:

* `.json` is only read at the **top level**. A `.json` inside a plugin directory is that
  plugin's private business.
* A `.json` stem may not match a directory name — `facts.json` next to `facts/` is an error,
  because they would claim the same source name.

A directory that is not a valid plugin produces a warning and is ignored. `llmeval` loads
plugin code on *every* invocation, so one broken directory must never stop a report running.

## Writing a plugin

One file is enough. Expose `get_plugin(interface)` from `__init__.py`:

```python
# testcases/my_suite/__init__.py
import json

from llmeval.plugins import PluginInterface, TestCasePlugin


class MySuite(TestCasePlugin):
    def __init__(self, interface):
        self.output = interface.cache_directory() / "testcases.json"

    def generate_testcases(self) -> bool:
        cases = [{"id": "hello", "user": "Say hi",
                  "assertions": [{"type": "icontains", "value": "hi"}]}]
        self.output.write_text(json.dumps(cases), encoding="utf-8")
        return True

    def get_testcases(self):
        if not self.output.is_file():
            return []
        return json.loads(self.output.read_text(encoding="utf-8"))


def get_plugin(interface: PluginInterface) -> TestCasePlugin:
    return MySuite(interface)
```

```bash
uv run llmeval generate --testcases my_suite
uv run llmeval run --testcases my_suite --provider configs/echo.json
```

Most plugins are shorter than that, because the common shapes are already written:
[`CsvTestCasePlugin`](../llmeval/generation/csv_plugin.py) is the whole of a CSV-backed
plugin, and [`HfDatasetPlugin`](../llmeval/generation/dataset_plugin.py) the whole of a
Hugging Face one. See `simple_facts/` and `multifaceted/` — three lines each.

Plugins may import anything from `llmeval`; nothing in `llmeval` imports a plugin except
through the loader. Helper modules inside a plugin directory are private to it and imported
relatively (`from .stooq import fetch_all`).

## The contract

Defined in [`llmeval/plugins/base.py`](../llmeval/plugins/base.py). Only the first two are
abstract; everything else defaults to doing nothing.

| Method | What it is for |
|---|---|
| `generate_testcases() -> bool` | The slow part: download, transform, write. `True` on success. |
| `get_testcases() -> list[dict]` | The test cases, same shape as a `.json` file. `[]` if not generated yet. |
| `get_custom_assertions() -> dict` | Bespoke graders, `{name: fn(spec, output, ctx)}`. |
| `before_run` / `after_run` | Once each, around this plugin's test cases running. |
| `before_each_run(tc)` / `after_each_run(tc, summary)` | Around each of them. |
| `before_grade` / `after_grade` | Once each, around this plugin's grading. |
| `before_each_grade(tc)` / `after_each_grade(tc, gradings)` | Around each test case's grading. |

**`PluginInterface`** gives you `name` and `cache_directory()` — a private scratch directory
at `.testcases.cache/<name>/`, gitignored, created on demand. Downloads, intermediate files
and the generated `testcases.json` go there. Cache downloads and reuse them: a first
`generate` may cost the network, later ones should not.

### Ids

You pick a **local** id; the loader prefixes the source name, so the real id — the one in
SQLite and in logs — is `<source>.<local id>`. Local ids must be unique within your plugin.
[`local_id(prompt)`](../llmeval/generation/common.py) gives you a stable
`sha1(prompt)[:10]`, which survives a dataset being re-downloaded or reordered.

**Uniqueness is yours to enforce, at generation time.** A source with a repeated id fails to
load *at all*, and it fails on the next `run` rather than on the `generate` that caused it.
Since `local_id` hashes the prompt, any source that asks the same question twice collides —
`multifaceted` does, on two thirds of its rows. Pass your cases through
[`drop_duplicate_ids(cases, name)`](../llmeval/generation/common.py) before you write them:
it keeps the first of each id and warns, with a prompt snippet, about every one it drops.
`CsvTestCasePlugin`, `HfDatasetPlugin` and `stock_prices/` already do. If a repeat is
meaningful for your source — the same prompt with *different* assertions, say — merge the
cases yourself instead, or give them distinct `variant=` suffixes; dropping keeps the first
and discards the rest.

There is no `suite` metadata key. The id prefix *is* the provenance, and the report's
`suite` column is read off it.

### Custom assertions

Return `{"stock_price": self.grade_stock_price}` and it registers as
`<source>.stock_price` — which is the `type` your test cases must then carry. The dot makes
a clash with a built-in impossible.

Return **bound methods**. That is how state a hook prepared reaches grading:
`stock_prices/` fetches live quotes in `before_grade` and its grader reads them off `self`,
rather than routing them through the test-case JSON.

### Hooks

They fire only for plugins that own at least one test case in the current invocation, so
`--testcases simple_facts` never triggers another plugin's setup.

* `before_each_run` / `after_each_run` run **on the runner's worker threads** and may
  overlap (`--concurrency` defaults to 5). Guard shared state you mutate there. The grade
  hooks are sequential.
* An exception from `before_run` or `before_grade` **aborts the command** — they are setup,
  and results computed past failed setup are meaningless. Exceptions from the other six are
  logged and swallowed, so your teardown bug cannot lose somebody's long run.

## What is here

| Source | Kind | Notes |
|---|---|---|
| `example.json` | json | Hand-written; one case each for a deterministic, a `rubric` and a `g_eval` check. |
| `examples.json` | json | Hand-written; the format by example. |
| `simple_facts/` | plugin | CSV, one `icontains` per row. |
| `simple_facts_regressions/` | plugin | CSV of questions that have regressed before. |
| `stock_prices/` | plugin | CSV + **live Stooq fetch at grade time**; custom assertion. |
| `agentharm_refusal/` | plugin | HF download; refusal rubric; gated, needs `HF_TOKEN`. |
| `multifaceted/` | plugin | HF download; per-criterion rubrics with 1–5 anchors. |
| `research_rubrics/` | plugin | HF download; `rubric` **and** `g_eval` variants per row. |

Design rationale:
[the spec](../docs/specs/2026-07-31-testcase-plugins-design.md).

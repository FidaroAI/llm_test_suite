"""Command-line interface — one subcommand per pipeline stage.

    llmeval generate --testcases simple_facts
    llmeval run      --testcases simple_facts --provider configs/fidaro_prod.json
    llmeval grade    --testcases simple_facts --provider configs/fidaro_prod.json --run-last-n 1
    llmeval pickbest --testcases simple_facts --providers a.json b.json --order both
    llmeval report   --run-last-n 3 --out results.csv
    llmeval compare-report --providers a.json b.json --baseline fidaro-prod --out report.html

Stages share a SQLite DB (``--db``, default ./llmeval.sqlite3). run/grade/pickbest/report
all read cached results — only ``run`` ever calls the model under test.

``run`` decides how many times to call the model from two flags: ``--mode`` (may cached
results be reused: ``reuse``, ``always``) and ``--repeat N`` (how many results per test case,
default 1). ``--mode reuse --repeat 5`` tops each test case up to five results; ``--mode
always --repeat 5`` adds five more however many are already stored.

``--testcases`` names a **source** — a plugin directory or a ``.json`` stem inside
``testcases/`` — and is repeatable. Omit it for every source:

    llmeval run --testcases simple_facts --testcases examples --provider configs/echo.json

Plugins are loaded on every invocation (see :mod:`llmeval.plugins.loader`), so a plugin's
custom assertions and lifecycle hooks are in play for whichever stage is running.

``grade`` and ``report`` both read stored results, so both take the same run-selection
flags (``--run-id``, ``--run-after``/``--run-before``, ``--run-last-n``); see
:mod:`llmeval.runselect`. ``grade`` also takes ``--limit N``, which caps how many test cases
it grades the way ``run --limit`` caps how many it runs. ``report`` emits **CSV** —
rendering it as a page is porcelain, so pipe it to ``python -m reporting.csv_table``.
``compare-report`` is the statistics and pick-best HTML.

All output goes through :mod:`logging` to stderr; ``--log-level`` (or
``LLMEVAL_LOG_LEVEL``) controls verbosity. Machine-readable results come from the store,
not from parsing this output — see CLAUDE.md on the plumbing's public contracts.
"""

from __future__ import annotations

import argparse
import json
import logging
import os

from llmeval.comparison.pickbest import DEFAULT_CRITERION, comparison_key, pick_best
from llmeval.comparison.report import write_report
from llmeval.grade import grade
from llmeval.logs import configure_logging
from llmeval.models import ProviderConfig
from llmeval.providers import build_provider, make_litellm_judge
from llmeval.resultrows import result_columns, result_rows, write_csv
from llmeval.runner import VALID_MODES, RunPolicy, run
from llmeval.runselect import RunSelectionError, parse_run_selection, resolve_runs
from llmeval.store import IncompatibleSchema, Store
from llmeval.testcases import (
    DEFAULT_ROOT,
    SourceError,
    load_testcases,
    select_sources,
    select_testcases,
)

logger = logging.getLogger(__name__)

LOG_LEVELS = ("debug", "info", "warning", "error", "critical")
DEFAULT_DB = "llmeval.sqlite3"
# Matches the legacy suite's judge: Bedrock Claude Haiku, deterministic.
DEFAULT_JUDGE = ProviderConfig(
    name="judge",
    model="bedrock/us.anthropic.claude-haiku-4-5-20251001-v1:0",
    params={"temperature": 0},
)


def load_provider_config(path: str) -> ProviderConfig:
    """Load a provider config JSON, expanding ${ENV} in base_url."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if data.get("base_url"):
        data["base_url"] = os.path.expandvars(data["base_url"])
    return ProviderConfig.model_validate(data)


def _positive_int(value: str) -> int:
    """An ``argparse`` type for counts that must be at least 1.

    A usage error rather than a silent no-op: ``--repeat 0`` would otherwise open a run,
    call nothing and report success, which looks exactly like a fully cached run.
    """
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from None
    if n < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1 (got {n})")
    return n


def _filters(items: list[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items or []:
        k, _, v = item.partition("=")
        out[k] = v
    return out


def _judge(args):
    cfg = load_provider_config(args.judge) if getattr(args, "judge", None) else DEFAULT_JUDGE
    return make_litellm_judge(cfg)


def _run_selection(args):
    """The run-selection flags, validated. Raises ``RunSelectionError`` on a bad combination."""
    return parse_run_selection(
        run_id=args.run_id,
        run_after=args.run_after,
        run_before=args.run_before,
        run_last_n=args.run_last_n,
    )


def _cache_key_hashes(paths: list[str] | None) -> list[str] | None:
    """Cache-key hashes for the named provider configs, or ``None`` for "every provider"."""
    if not paths:
        return None
    return [load_provider_config(p).cache_key().hash for p in paths]


def cmd_generate(args) -> int:
    """Ask each selected plugin to prepare its test cases.

    Every plugin is attempted even if an earlier one failed: generation is per-plugin work
    and one broken download should not deny you the other five suites. The exit code still
    says something went wrong.
    """
    plugins = [s for s in select_sources(args.testcases or None) if s.is_plugin]
    if not plugins:
        logger.warning("no plugins to generate in %s/", DEFAULT_ROOT)
        return 0
    rc = 0
    for source in plugins:
        try:
            ok = source.plugin.generate_testcases()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.error("%s: generation raised (%s)", source.name, exc)
            rc = 1
            continue
        if ok:
            logger.info("%s: generated", source.name)
        else:
            logger.error("%s: generation reported failure", source.name)
            rc = 1
    return rc


def cmd_run(args) -> int:
    store = Store(args.db)
    loaded = load_testcases(names=args.testcases or None, filters=_filters(args.filter))
    tcs = select_testcases(
        loaded.cases, limit=args.limit, randomize=args.randomize, seed=args.seed
    )
    provider = build_provider(load_provider_config(args.provider))
    policy = RunPolicy(
        mode=args.mode,
        repeat=args.repeat,
        retries=args.retries,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    # Hooks are scoped to the cases actually selected, so --limit or a filter that excludes
    # a plugin entirely also excludes its before_run.
    result = run(store, tcs, provider, policy, notes=args.note, hooks=loaded.hooks(tcs))
    summary = result.summary
    # The run's headline. Logged at the end as well as the start (see runner.run) because
    # this is the line a human reads to decide whether to look at the store at all.
    # ``errors`` counts attempts and ``failed`` counts test cases: errors>0 with failed=0
    # is a run that retried its way through a flaky provider, not a broken one.
    logger.info(
        "run %s finished: %d test(s); ran=%d cached=%d errors=%d failed=%d",
        result.run_id, len(tcs), summary.ran, summary.cached, summary.errors, summary.failed,
    )
    store.close()
    return 0


def cmd_grade(args) -> int:
    store = Store(args.db)
    try:
        loaded = load_testcases(names=args.testcases or None, filters=_filters(args.filter))
        # Same subsetting as run, and for the same reason: hooks are scoped to the cases
        # actually selected, so --limit also excludes an omitted plugin's before_grade.
        tcs = select_testcases(loaded.cases, limit=args.limit)
        cfg = load_provider_config(args.provider)
        key_hash = cfg.cache_key().hash
        run_ids = [r.id for r in resolve_runs(store, _run_selection(args), [key_hash])]
        if not run_ids:
            logger.warning("no runs match the selection for %r; nothing to grade", cfg.name)
            return 0
        logger.info(
            "grading %d test(s) over %d run(s) for %r (regrade=%s)",
            len(tcs), len(run_ids), cfg.name, args.regrade,
        )
        grade(
            store, tcs, key_hash, judge=_judge(args), regrade=args.regrade,
            run_ids=run_ids, hooks=loaded.hooks(tcs),
        )
        logger.info("graded %d test(s) for %r", len(tcs), cfg.name)
        return 0
    finally:
        store.close()


def cmd_pickbest(args) -> int:
    store = Store(args.db)
    tcs = load_testcases(names=args.testcases or None, filters=_filters(args.filter)).cases
    configs = [load_provider_config(p) for p in args.providers]
    logger.info(
        "pick-best over %d config(s), %d test(s), order=%s", len(configs), len(tcs), args.order
    )
    pick_best(store, tcs, configs, _judge(args), order=args.order, regrade=args.regrade)
    logger.info("pick-best complete: %d test(s)", len(tcs))
    store.close()
    return 0


def cmd_report(args) -> int:
    """Write the selected result rows as CSV. Rendering is porcelain's job.

    Data out, no HTML: turning a CSV into a page (and opening a browser) is a workflow,
    which per CLAUDE.md lives in ``reporting/`` rather than here. So:

        llmeval report --run-last-n 3 --out rows.csv
        python -m reporting.csv_table rows.csv -o rows.html

    ``--testcases``/``--filter`` narrow *which* results appear. Neither is required: the
    prompt, the answer and the suite all come off the stored result, so a report needs no
    test-case files at all.
    """
    if not os.path.exists(args.db):
        # sqlite3.connect would happily create an empty database, and the user would get
        # "0 rows" for what is really a wrong --db path.
        logger.error("no results database at %s", args.db)
        return 2
    store = Store(args.db)
    try:
        runs = resolve_runs(store, _run_selection(args), _cache_key_hashes(args.provider))
        # Selection stays opt-in. Without --testcases/--filter every stored result is
        # reported, so a run outlives the regeneration (or removal) of its test cases —
        # which matters more now that plugin output is not tracked in git.
        cases_by_id = None
        if args.testcases or args.filter:
            loaded = load_testcases(names=args.testcases or None, filters=_filters(args.filter))
            cases_by_id = {c.id: c for c in loaded.cases}
        rows = result_rows(store, runs, cases_by_id)
        columns = result_columns()
    finally:
        store.close()
    write_csv(rows, columns, args.out)
    logger.info("wrote %d row(s) from %d run(s) -> %s", len(rows), len(runs), args.out)
    return 0


def cmd_compare_report(args) -> int:
    store = Store(args.db)
    configs = [load_provider_config(p) for p in args.providers]
    pairs = [(c.name, c.cache_key().hash) for c in configs]
    metrics = args.metrics if args.metrics else [None]
    ckey = comparison_key(configs, DEFAULT_CRITERION, args.order) if args.order else None
    write_report(store, pairs, metrics, args.out, baseline_name=args.baseline, comparison_key=ckey)
    logger.info("wrote comparison report -> %s", args.out)
    store.close()
    return 0


def _add_generate_parser(sub) -> None:
    gen = sub.add_parser(
        "generate", help="ask each plugin in testcases/ to prepare its test cases"
    )
    _add_testcases(gen)
    gen.set_defaults(func=cmd_generate)


def _add_db(sp) -> None:
    sp.add_argument("--db", default=DEFAULT_DB, help="SQLite results DB")


def _add_testcases(sp) -> None:
    """The repeatable ``--testcases`` flag. Shared so the stages cannot drift apart.

    A **source name**, not a path: a plugin directory or a ``.json`` stem inside
    ``testcases/``. Omitted means every source. The root is always ``testcases/`` relative to
    the working directory — there is deliberately no flag for it, because a project is a
    directory and moving the test cases out of it is not a thing the CLI should encourage.
    """
    sp.add_argument(
        "--testcases", action="append", metavar="NAME",
        help="source name — a plugin directory or .json stem in testcases/ "
             "(repeatable; default: all)",
    )


def _add_filters(sp) -> None:
    sp.add_argument("--filter", action="append", help="metadata filter k=v (repeatable)")


def _add_run_selection(sp) -> None:
    """The four run-selection flags. Shared by every stage that reads stored results."""
    sp.add_argument(
        "--run-id", action="append",
        help="comma-separated run ids or unambiguous prefixes (repeatable)",
    )
    sp.add_argument(
        "--run-after",
        help="only runs at or after this point: YYYY-MM-DD or YYYY-MM-DDTHH:MM "
        "(UTC unless an offset is given), or a run id",
    )
    sp.add_argument(
        "--run-before",
        help="only runs at or before this point; same forms as --run-after",
    )
    sp.add_argument("--run-last-n", type=int, help="only the N most recent runs")


def _add_run_parser(sub) -> None:
    r = sub.add_parser("run", help="run a provider over test cases (cached by cache key)")
    _add_testcases(r)
    r.add_argument("--provider", required=True)
    r.add_argument(
        "--mode", default="reuse", choices=list(VALID_MODES),
        help="whether cached results may be reused: reuse (default) tops a test case up to "
        "--repeat results; always appends --repeat more, whatever is stored",
    )
    r.add_argument(
        "--repeat", type=_positive_int, default=1, metavar="N",
        help="how many results per test case (default 1). Under --mode reuse this is a "
        "target the run tops up to; under always it is how many fresh calls to make.",
    )
    r.add_argument("--retries", type=int, default=2)
    r.add_argument(
        "--concurrency", type=int, default=5,
        help="number of test cases to run in parallel (default 5; 1 = sequential)",
    )
    r.add_argument(
        "--timeout", type=float, default=60.0,
        help="seconds allowed per inference call, before retries (default 60). A test "
        "case's own \"timeout\" field overrides this for that test.",
    )
    r.add_argument("--note", default=None, help="free-text note recorded against this run")
    r.add_argument("--limit", type=int, default=None, help="run only the first N tests")
    r.add_argument("--randomize", action="store_true", help="shuffle test order before running")
    r.add_argument("--seed", type=int, default=0, help="seed for --randomize (fixed; default 0)")
    _add_db(r)
    _add_filters(r)
    r.set_defaults(func=cmd_run)


def _add_grade_parser(sub) -> None:
    gr = sub.add_parser("grade", help="grade cached outputs (no model calls)")
    _add_testcases(gr)
    gr.add_argument("--provider", required=True)
    gr.add_argument("--judge", help="judge provider config JSON (default: Bedrock Haiku)")
    gr.add_argument("--regrade", action="store_true")
    gr.add_argument(
        "--limit", type=int, default=None,
        help="grade only the first N test cases (every attempt and assertion of each)",
    )
    _add_db(gr)
    _add_filters(gr)
    _add_run_selection(gr)
    gr.set_defaults(func=cmd_grade)


def _add_pickbest_parser(sub) -> None:
    pb = sub.add_parser("pickbest", help="direct head-to-head over cached outputs")
    _add_testcases(pb)
    pb.add_argument("--providers", required=True, nargs="+")
    pb.add_argument("--judge")
    pb.add_argument("--order", default="as_is", choices=["as_is", "random", "both"])
    pb.add_argument("--regrade", action="store_true")
    _add_db(pb)
    _add_filters(pb)
    pb.set_defaults(func=cmd_pickbest)


def _add_report_parsers(sub) -> None:
    """The two reporting subcommands: result rows as CSV, and the statistics HTML."""
    rp = sub.add_parser(
        "report",
        help="write the selected result rows as CSV (one row per result x assertion, "
        "plus one per errored result)",
    )
    rp.add_argument("--out", default="results.csv", help="output CSV path")
    rp.add_argument(
        "--provider", action="append",
        help="provider config JSON (repeatable; default: every provider in the DB)",
    )
    _add_testcases(rp)
    _add_db(rp)
    _add_filters(rp)
    _add_run_selection(rp)
    rp.set_defaults(func=cmd_report)

    cr = sub.add_parser(
        "compare-report", help="render an HTML comparison report (statistics + pick-best)"
    )
    cr.add_argument("--providers", required=True, nargs="+")
    cr.add_argument("--baseline", help="baseline provider name (for deltas)")
    cr.add_argument("--metrics", nargs="*", help="metric names (default: overall)")
    cr.add_argument(
        "--order", choices=["as_is", "random", "both"], help="include pick-best win rates"
    )
    cr.add_argument("--out", default="report.html")
    _add_db(cr)
    cr.set_defaults(func=cmd_compare_report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmeval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    _add_generate_parser(sub)
    _add_run_parser(sub)
    _add_grade_parser(sub)
    _add_pickbest_parser(sub)
    _add_report_parsers(sub)

    # Every subcommand takes --log-level. Applied in one loop over the registered
    # subparsers rather than per-parser, so a subcommand added later cannot omit it.
    for sp in sub.choices.values():
        sp.add_argument(
            "--log-level", choices=LOG_LEVELS, default=None,
            help="verbosity (default: LLMEVAL_LOG_LEVEL, else info)",
        )

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # The entry point owns logging configuration — importing llmeval never touches the
    # root logger, so a library embedder keeps control of its own handlers.
    configure_logging(getattr(args, "log_level", None))
    try:
        return args.func(args)
    except (IncompatibleSchema, RunSelectionError, SourceError) as exc:
        # Expected conditions (a DB from an older build, a selection that cannot be
        # satisfied), not bugs — a message, no traceback.
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

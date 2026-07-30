"""Command-line interface — one subcommand per pipeline stage.

    llmeval generate-csv --csv f.csv --suite facts --out testcases/
    llmeval run      --testcases testcases/ --provider configs/fidaro_prod.json
    llmeval grade    --testcases testcases/ --provider configs/fidaro_prod.json --run-last-n 1
    llmeval pickbest --testcases testcases/ --providers a.json b.json --order both
    llmeval report   --run-last-n 3 --testcases testcases/ --out results.csv
    llmeval compare-report --providers a.json b.json --baseline fidaro-prod --out report.html

Stages share a SQLite DB (``--db``, default ./llmeval.sqlite3). run/grade/pickbest/report
all read cached results — only ``run`` ever calls the model under test.

``grade`` and ``report`` both read stored results, so both take the same run-selection
flags (``--run-id``, ``--run-after``/``--run-before``, ``--run-last-n``); see
:mod:`llmeval.runselect`. ``report`` emits **CSV** — rendering it as a page is porcelain,
so pipe it to ``python -m reporting.csv_table``. ``compare-report`` is the statistics and
pick-best HTML.

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
from llmeval.generation.csv_source import generate_from_csv
from llmeval.generation.suites import (
    GenPaths,
    SUITES,
    all_suite_names,
    default_paths,
    write_suite,
)
from llmeval.grade import grade
from llmeval.logs import configure_logging
from llmeval.models import ProviderConfig
from llmeval.providers import build_provider, make_litellm_judge
from llmeval.resultrows import result_columns, result_rows, write_csv
from llmeval.runner import RunPolicy, run
from llmeval.runselect import RunSelectionError, parse_run_selection, resolve_runs
from llmeval.store import IncompatibleSchema, Store
from llmeval.testcases import load_testcases, select_testcases

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


def cmd_generate_csv(args) -> int:
    cases = generate_from_csv(
        args.csv, suite=args.suite, out_dir=args.out,
        prompt_col=args.prompt_col, expected_col=args.expected_col,
    )
    logger.info(
        "generated %d test case(s) for suite %r -> %s", len(cases), args.suite, args.out
    )
    return 0


def cmd_generate(args) -> int:
    if not args.all and not args.suite:
        logger.error("pass --suite NAME (repeatable) or --all")
        return 2
    names = all_suite_names() if args.all else list(args.suite)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        logger.error("unknown suite(s) %s; known: %s", unknown, sorted(SUITES))
        return 2

    base = default_paths(args.config)
    paths = GenPaths(
        data_dir=args.data_dir or base.data_dir,
        classifications_dir=args.classifications_dir or base.classifications_dir,
        generation_sources_dir=args.sources_dir or base.generation_sources_dir,
        config_path=args.config or base.config_path,
    )
    rc = 0
    for name in names:
        try:
            count = write_suite(name, args.out, paths)
        except FileNotFoundError as exc:
            # --all is lenient (datasets may not be downloaded); explicit --suite
            # is a hard error so a typo or missing source is not silently ignored.
            if args.all:
                logger.warning("skipped suite %r: %s", name, exc)
                continue
            logger.error("cannot generate suite %r: %s", name, exc)
            rc = 1
            continue
        logger.info("generated %d test case(s) for suite %r -> %s", count, name, args.out)
    return rc


def cmd_run(args) -> int:
    store = Store(args.db)
    tcs = load_testcases(args.testcases, _filters(args.filter))
    tcs = select_testcases(tcs, limit=args.limit, randomize=args.randomize, seed=args.seed)
    provider = build_provider(load_provider_config(args.provider))
    policy = RunPolicy(
        mode=args.mode,
        target_n=args.target_n,
        retries=args.retries,
        concurrency=args.concurrency,
        timeout=args.timeout,
    )
    result = run(store, tcs, provider, policy, notes=args.note)
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
        tcs = load_testcases(args.testcases, _filters(args.filter))
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
        grade(store, tcs, key_hash, judge=_judge(args), regrade=args.regrade, run_ids=run_ids)
        logger.info("graded %d test(s) for %r", len(tcs), cfg.name)
        return 0
    finally:
        store.close()


def cmd_pickbest(args) -> int:
    store = Store(args.db)
    tcs = load_testcases(args.testcases, _filters(args.filter))
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
    """
    if not os.path.exists(args.db):
        # sqlite3.connect would happily create an empty database, and the user would get
        # "0 rows" for what is really a wrong --db path.
        logger.error("no results database at %s", args.db)
        return 2
    store = Store(args.db)
    try:
        runs = resolve_runs(store, _run_selection(args), _cache_key_hashes(args.provider))
        cases_by_id = None
        if args.testcases:
            cases_by_id = {c.id: c for c in load_testcases(args.testcases, _filters(args.filter))}
        rows = result_rows(store, runs, cases_by_id)
        columns = result_columns(cases_by_id is not None)
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
    g = sub.add_parser("generate-csv", help="transform a CSV into standardized test cases")
    g.add_argument("--csv", required=True)
    g.add_argument("--suite", required=True)
    g.add_argument("--out", required=True, help="output directory")
    g.add_argument("--prompt-col", default="user")
    g.add_argument("--expected-col", default="__expected")
    g.set_defaults(func=cmd_generate_csv)

    gen = sub.add_parser(
        "generate",
        help="generate one or more named suites (simple_facts, agentharm_refusal, "
        "multifaceted, research_rubrics, stock_prices, ...)",
    )
    gen.add_argument(
        "--suite", action="append", default=[],
        help="suite name (repeatable); see --all for the full set",
    )
    gen.add_argument(
        "--all", action="store_true",
        help="generate every suite except network ones (stock_prices needs --suite)",
    )
    gen.add_argument("--out", default="testcases", help="output directory (default testcases/)")
    gen.add_argument(
        "--config", help="suite-generation config JSON (env SUITE_GENERATION_CONFIG_FILE)"
    )
    gen.add_argument("--data-dir", help="dir holding dataset JSON (default: repo-root data/)")
    gen.add_argument("--classifications-dir", help="dir holding <suite>.json label files")
    gen.add_argument("--sources-dir", help="dir holding CSV sources (default: generation_sources/)")
    gen.set_defaults(func=cmd_generate)


def _add_db(sp) -> None:
    sp.add_argument("--db", default=DEFAULT_DB, help="SQLite results DB")


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
    r.add_argument("--testcases", required=True)
    r.add_argument("--provider", required=True)
    r.add_argument("--mode", default="reuse", choices=["reuse", "target_n", "always"])
    r.add_argument("--target-n", type=int, default=1)
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
    gr.add_argument("--testcases", required=True)
    gr.add_argument("--provider", required=True)
    gr.add_argument("--judge", help="judge provider config JSON (default: Bedrock Haiku)")
    gr.add_argument("--regrade", action="store_true")
    _add_db(gr)
    _add_filters(gr)
    _add_run_selection(gr)
    gr.set_defaults(func=cmd_grade)


def _add_pickbest_parser(sub) -> None:
    pb = sub.add_parser("pickbest", help="direct head-to-head over cached outputs")
    pb.add_argument("--testcases", required=True)
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
    rp.add_argument(
        "--testcases",
        help="testcases dir/file; selects which tests appear and adds the request_type "
        "and domain columns (the prompt is stored on the result, so it is always present)",
    )
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
    except (IncompatibleSchema, RunSelectionError) as exc:
        # Expected conditions (a DB from an older build, a selection that cannot be
        # satisfied), not bugs — a message, no traceback.
        logger.error("%s", exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

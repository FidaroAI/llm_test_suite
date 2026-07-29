"""Command-line interface — one subcommand per pipeline stage.

    llmeval generate-csv --csv f.csv --suite facts --out testcases/
    llmeval run     --testcases testcases/ --provider configs/fidaro_prod.json
    llmeval grade   --testcases testcases/ --provider configs/fidaro_prod.json
    llmeval pickbest --testcases testcases/ --providers a.json b.json --order both
    llmeval report  --providers a.json b.json --baseline fidaro-prod --order both --out report.html

Stages share a SQLite DB (``--db``, default ./llmeval.sqlite3). run/grade/pickbest/report
all read cached results — only ``run`` ever calls the model under test.
"""

from __future__ import annotations

import argparse
import json
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
from llmeval.models import ProviderConfig
from llmeval.providers import build_provider, make_litellm_judge
from llmeval.runner import RunPolicy, run
from llmeval.store import IncompatibleSchema, Store
from llmeval.testcases import load_testcases, select_testcases

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


def cmd_generate_csv(args) -> int:
    cases = generate_from_csv(
        args.csv, suite=args.suite, out_dir=args.out,
        prompt_col=args.prompt_col, expected_col=args.expected_col,
    )
    print(f"generated {len(cases)} test case(s) for suite {args.suite!r} -> {args.out}")
    return 0


def cmd_generate(args) -> int:
    if not args.all and not args.suite:
        print("error: pass --suite NAME (repeatable) or --all")
        return 2
    names = all_suite_names() if args.all else list(args.suite)
    unknown = [n for n in names if n not in SUITES]
    if unknown:
        print(f"error: unknown suite(s) {unknown}; known: {sorted(SUITES)}")
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
                print(f"skipped suite {name!r}: {exc}")
                continue
            print(f"error: cannot generate suite {name!r}: {exc}")
            rc = 1
            continue
        print(f"generated {count} test case(s) for suite {name!r} -> {args.out}")
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
    )
    result = run(store, tcs, provider, policy, notes=args.note)
    summary = result.summary
    print(
        f"run {result.run_id}: {len(tcs)} test(s); ran={summary.ran} "
        f"cached={summary.cached} errors={summary.errors}"
    )
    store.close()
    return 0


def cmd_grade(args) -> int:
    store = Store(args.db)
    tcs = load_testcases(args.testcases, _filters(args.filter))
    cfg = load_provider_config(args.provider)
    grade(store, tcs, cfg.cache_key().hash, judge=_judge(args), regrade=args.regrade)
    print(f"graded {len(tcs)} test(s) for {cfg.name!r}")
    store.close()
    return 0


def cmd_pickbest(args) -> int:
    store = Store(args.db)
    tcs = load_testcases(args.testcases, _filters(args.filter))
    configs = [load_provider_config(p) for p in args.providers]
    pick_best(store, tcs, configs, _judge(args), order=args.order, regrade=args.regrade)
    print(f"pick-best over {len(configs)} configs, {len(tcs)} test(s), order={args.order}")
    store.close()
    return 0


def cmd_report(args) -> int:
    store = Store(args.db)
    configs = [load_provider_config(p) for p in args.providers]
    pairs = [(c.name, c.cache_key().hash) for c in configs]
    metrics = args.metrics if args.metrics else [None]
    ckey = comparison_key(configs, DEFAULT_CRITERION, args.order) if args.order else None
    write_report(store, pairs, metrics, args.out, baseline_name=args.baseline, comparison_key=ckey)
    print(f"wrote report -> {args.out}")
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="llmeval", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    def add_db(sp):
        sp.add_argument("--db", default=DEFAULT_DB, help="SQLite results DB")

    def add_filters(sp):
        sp.add_argument("--filter", action="append", help="metadata filter k=v (repeatable)")

    _add_generate_parser(sub)

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
    r.add_argument("--note", default=None, help="free-text note recorded against this run")
    r.add_argument("--limit", type=int, default=None, help="run only the first N tests")
    r.add_argument("--randomize", action="store_true", help="shuffle test order before running")
    r.add_argument("--seed", type=int, default=0, help="seed for --randomize (fixed; default 0)")
    add_db(r)
    add_filters(r)
    r.set_defaults(func=cmd_run)

    gr = sub.add_parser("grade", help="grade cached outputs (no model calls)")
    gr.add_argument("--testcases", required=True)
    gr.add_argument("--provider", required=True)
    gr.add_argument("--judge", help="judge provider config JSON (default: Bedrock Haiku)")
    gr.add_argument("--regrade", action="store_true")
    add_db(gr)
    add_filters(gr)
    gr.set_defaults(func=cmd_grade)

    pb = sub.add_parser("pickbest", help="direct head-to-head over cached outputs")
    pb.add_argument("--testcases", required=True)
    pb.add_argument("--providers", required=True, nargs="+")
    pb.add_argument("--judge")
    pb.add_argument("--order", default="as_is", choices=["as_is", "random", "both"])
    pb.add_argument("--regrade", action="store_true")
    add_db(pb)
    add_filters(pb)
    pb.set_defaults(func=cmd_pickbest)

    rp = sub.add_parser("report", help="render an HTML comparison report")
    rp.add_argument("--providers", required=True, nargs="+")
    rp.add_argument("--baseline", help="baseline provider name (for deltas)")
    rp.add_argument("--metrics", nargs="*", help="metric names (default: overall)")
    rp.add_argument(
        "--order", choices=["as_is", "random", "both"], help="include pick-best win rates"
    )
    rp.add_argument("--out", default="report.html")
    add_db(rp)
    rp.set_defaults(func=cmd_report)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except IncompatibleSchema as exc:
        # Expected condition (a DB from an older build), not a bug — no traceback.
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

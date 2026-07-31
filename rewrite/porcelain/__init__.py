"""`llmevalx` — the interactive porcelain for the `llmeval` suite.

`llmeval` is the plumbing: everything is *possible* with it, almost nothing is *pleasant*
(see ../CLAUDE.md). This package is the friendly layer — an arrow-key wizard that discovers
what is available, asks a handful of questions, and shells out to the documented CLI.

It depends only on the plumbing's public contracts: the CLI subcommands, the test-case JSON
in `testcases/`, and the SQLite schema. Like `reporting/`, it imports `llmeval` and never the
reverse, and it is excluded from the installed wheel.

Run it with `./llmevalx.sh`, `uv run llmevalx`, or `python -m porcelain`.
"""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Entry point. Imported lazily so `import porcelain` costs nothing."""
    from porcelain.app import main as _main

    return _main(argv)

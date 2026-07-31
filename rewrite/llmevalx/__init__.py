"""`llmevalx` — the interactive front end for the `llmeval` suite.

Two first-class entry points share this project. `llmeval` is the plumbing: everything is
*possible* with it, almost nothing is *pleasant* (see ../CLAUDE.md). `llmevalx` is the
friendly one — an arrow-key wizard that discovers what is available, asks a handful of
questions, and shells out to the `llmeval` CLI, printing each command as it goes.

The dependency runs one way only: `llmevalx` imports `llmeval`, never the reverse. That is
what keeps the plumbing free of workflow shortcuts while letting the wizard reuse its
constants, its store and its suite registry rather than duplicating them.

    uv run llmevalx        # or ./llmevalx.sh, or python -m llmevalx
"""

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    """Console-script entry point.

    The real work lives in :mod:`llmevalx.app`, imported lazily so `import llmevalx` costs
    nothing — the app pulls in questionary and prompt_toolkit, which a caller that only wants,
    say, :mod:`llmevalx.commands` has no use for.
    """
    from llmevalx.app import main as _main

    return _main(argv)

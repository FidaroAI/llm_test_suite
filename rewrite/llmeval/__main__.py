"""``python -m llmeval`` — the same entry point as the ``llmeval`` console script.

Exists so a caller can invoke the CLI through an interpreter it already has a path to
(``sys.executable -m llmeval ...``) instead of searching ``PATH`` for a console script that
may or may not be on it. :mod:`llmevalx.commands` relies on this.
"""

from llmeval.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

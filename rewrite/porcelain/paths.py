"""Where things live, computed once.

The wizard `chdir`s to :data:`ROOT` at startup so every path it prints is relative and every
command it echoes can be pasted straight into a shell from the same directory. Everything
downstream therefore uses these relative names, not absolutes.
"""

from __future__ import annotations

from pathlib import Path

# .../rewrite — the parent of this package. Derived from this file rather than from the
# `porcelain` package object, which would make every module that needs a path depend on the
# package importing cleanly first.
ROOT = Path(__file__).resolve().parent.parent

ENV_FILE = ROOT / ".env"

# Relative on purpose: they are echoed as part of commands the user may copy.
TESTCASES_DIR = "testcases"
CONFIGS_DIR = "configs"

# Where the wizard's report chain writes. Overwritten each time, which is the point —
# "the last report" should be one predictable file, not a directory that grows.
RESULTS_CSV = "results.csv"
REPORT_HTML = "report.html"

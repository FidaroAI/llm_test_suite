"""Where things live.

The wizard works in one directory — the project directory holding `testcases/`, `configs/`
and `llmeval.sqlite3` — and `chdir`s there at startup. That is what lets every path it prints
be relative and every command it echoes be pasted straight into a shell, so everything
downstream uses these relative names rather than absolutes.
"""

from __future__ import annotations

import os
from pathlib import Path

# .../rewrite for a source checkout or an editable install. Derived from this file rather than
# from the `llmevalx` package object, which would make every module that needs a path depend
# on the package importing cleanly first.
PACKAGE_PARENT = Path(__file__).resolve().parent.parent

# What makes a directory the project directory. Either is enough: a fresh checkout has
# configs/ but no testcases/ until something is generated.
MARKERS = ("configs", "testcases")

ENV_FILE_NAME = ".env"

# Relative on purpose: they are echoed as part of commands the user may copy.
TESTCASES_DIR = "testcases"
CONFIGS_DIR = "configs"

# Where the wizard's report chain writes. Overwritten each time, which is the point —
# "the last report" should be one predictable file, not a directory that grows.
RESULTS_CSV = "results.csv"
REPORT_HTML = "report.html"


def looks_like_project(path: Path) -> bool:
    return any((path / marker).is_dir() for marker in MARKERS)


def project_root() -> Path:
    """The directory to work in: the package's parent, or the cwd if that is not the project.

    `PACKAGE_PARENT` is the answer for a source checkout and for the editable install this
    project is developed with. It is the *wrong* answer for a non-editable install, where it
    points into site-packages — so it is only used when it actually looks like the project.
    Falling back to the cwd means `cd my-eval-project && llmevalx` works either way.
    """
    if looks_like_project(PACKAGE_PARENT):
        return PACKAGE_PARENT
    return Path(os.getcwd())


def env_file() -> Path:
    return project_root() / ENV_FILE_NAME

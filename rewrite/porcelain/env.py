"""Load `.env` so a run cannot fail for the one boring reason.

The plumbing deliberately does no such thing — `uv run --env-file .env llmeval ...` is the
documented incantation, and forgetting it is the single most common way a run dies holding a
missing `FIDARO_API_KEY`. Porcelain exists to remove that class of mistake, so the wizard
loads the file itself and every subprocess inherits the result.

Variables already set in the environment **win**: an explicit
`FIDARO_DEV_BASE_URL=... llmevalx` on the command line has to beat the file, or overriding
anything becomes impossible.
"""

from __future__ import annotations

import os
from pathlib import Path

from porcelain.paths import ENV_FILE


def load_env(path: Path | None = None) -> Path | None:
    """Merge `.env` into `os.environ`. Returns the file loaded, or `None` if there was none.

    A missing file is not an error: the core of the suite (cache, store, grading replay,
    reports, the echo provider) needs no credentials at all.
    """
    env_path = ENV_FILE if path is None else path
    if not env_path.is_file():
        return None
    for key, value in parse_env(env_path.read_text(encoding="utf-8")).items():
        os.environ.setdefault(key, value)
    return env_path


def parse_env(text: str) -> dict[str, str]:
    """Parse `.env` text into a mapping.

    Deliberately small — this reads one hand-written file, not arbitrary shell. It handles
    what `.env.example` actually contains plus the common conveniences: comments, blank
    lines, a leading `export`, and surrounding quotes. It does **not** do interpolation,
    multi-line values, or escape sequences; a `.env` needing those wants a real shell.
    """
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out

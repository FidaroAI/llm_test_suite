#!/usr/bin/env bash
# The interactive front end for the llmeval suite.
#
#   ./llmevalx.sh          # equivalent to: uv run llmevalx
#
# A convenience only — `uv run llmevalx` does the same thing. This exists so it works from
# anywhere: it cd's to the project directory first, which is where testcases/, configs/ and
# llmeval.sqlite3 live and where `uv run` finds the project environment.
#
# The .env is loaded by the tool itself rather than by `uv run --env-file`, so it works the
# same however you start it.
#
# VIRTUAL_ENV is cleared because direnv activates .direnv/python-3.13 at the repo root, which
# is not this project's .venv. uv ignores it anyway and warns loudly about the mismatch; the
# warning is noise, not a problem, so drop the variable rather than print it every time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec env -u VIRTUAL_ENV uv run llmevalx "$@"

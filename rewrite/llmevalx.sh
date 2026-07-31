#!/usr/bin/env bash
# The interactive front end for the llmeval suite.
#
#   ./llmevalx.sh
#
# `cd`s here first so testcases/, configs/ and llmeval.sqlite3 resolve the way the wizard
# prints them, and so `python -m porcelain` can find the package: porcelain is porcelain, so
# like reporting/ it is not part of the installed wheel and is imported from this directory.
# That is also why there is no `llmevalx` console script — an installed script could not
# import a package that is deliberately not installed.
#
# The .env is loaded by the tool itself rather than by `uv run --env-file`, so it works the
# same however you start it.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec uv run python -m porcelain "$@"

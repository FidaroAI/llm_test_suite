# CI Results Puller — Design

## Goal

Add a script that pulls promptfoo result artifacts from this repo's GitHub
Actions runs into a local `results/ci/` directory and imports them into the
local promptfoo database. Also reorganise result output paths so locally-run
evals land in `results/local/` and CI evals land in `results/ci/`.

## Motivation

CI (`promptfoo-gateway.yml`) runs the suite and uploads results as an artifact,
but there's no way to get those results onto a developer machine to view or
compare. This script closes that loop. The path reorganisation keeps
locally-generated results from being confused with CI-pulled results.

## Out of scope

- Multiple workflows. There is one results-producing workflow today; the script
  hardcodes it. Adding another would require a code edit (accepted trade-off).
- Viewing / diffing results. The script imports into promptfoo; viewing is
  whatever `promptfoo view` already does.
- Backfilling `results/local/` or `results/ci/` with historical content.
- Fixing the stale `package.json` script paths (`scripts/*.sh` vs
  `scripts_test/*.sh`) beyond the one `view` line this work touches.

## The puller script

**File:** `scripts_repo/pull_ci_results.py` — Python, `argparse`, single file,
consistent with `test_aws.py` and `flatten_promptfoo_config.py`.

**Hardcoded constants:**
- workflow: `promptfoo-gateway.yml`
- artifact name: `promptfoo-results`
- destination directory: `results/ci/`

### CLI surface

```
pull_ci_results.py (--commit <sha> | --latest | --all) [options]

Modes (exactly one required):
  --commit <sha>   Pull the result for one specific commit
  --latest         Pull the result for the most recent CI run
  --all            Pull results for all commits, optionally date-gated

--all date gating:
  --start-date <YYYY-MM-DD>   default: start of history
  --end-date   <YYYY-MM-DD>   default: now

General:
  --no-import          Download only; skip the promptfoo import phase
  --fail-on-existing   Hard-fail if an eval ID already exists in promptfoo
                       (default: warn and skip)
```

### Run discovery (commit-driven)

- `--commit <sha>`: `gh run list --commit <sha> --workflow promptfoo-gateway.yml
  --json databaseId,headSha,createdAt,conclusion` → pick the most recent
  **successful** run. No run found → error (the user asked for that commit
  specifically).
- `--latest`: `gh run list --workflow promptfoo-gateway.yml --limit 1` → the most
  recent run. ("Most recent commit" is interpreted as "most recent CI run".)
- `--all`: `git log --since=<start> --until=<end> --format=%H` enumerates commits
  in the date range; for each, a targeted `gh run list --commit <sha>` lookup
  finds its successful run. Commits with no run → warn and skip.

### Download

`gh run download <run-id> --name promptfoo-results --dir <tmpdir>`, then glob for
the single `.json` file inside `<tmpdir>` (resilient to however
`actions/upload-artifact` nests the path). Zero or multiple `.json` files → error
for that run.

### Filename

`results/ci/<commit-date>_<short-sha>.json`, e.g.
`results/ci/2026-05-14_4a9cae7.json`. Commit date (not run date) is used so
`ls results/ci/` reads in the same terms the `--all` filter uses. The puller
creates `results/ci/` with `Path.mkdir(parents=True, exist_ok=True)`.

## promptfoo import phase

Skipped entirely when `--no-import` is passed. Otherwise, for each downloaded
file (in the order discovered):

1. Run `promptfoo import <file>`.
2. If it **succeeds** → continue to the next file.
3. If it **fails** → read the file's top-level `evalId` and check it against
   `promptfoo list evals --ids-only`:
   - **evalId present** → benign duplicate. Default: log a warning and skip.
     With `--fail-on-existing`: print an error and abort immediately.
   - **evalId absent** → genuine error. Abort immediately, leaving already
     downloaded files on disk for retry.

The happy path is one `import` call per file with no pre-check. The
`list evals` lookup runs only on the failure path, to classify *why* import
failed — this avoids fragile stderr string-matching.

## Codebase adjustments

### Local runs → `results/local/`

| File | Change |
|------|--------|
| `promptfooconfig.yaml` | `outputPath: results/local/latest.json` |
| `scripts_test/hack.sh` | `--output results/local/latest.json` (+ defensive `mkdir -p`) |
| `scripts_test/smoke.sh` | `--output results/local/smoke.json` (+ defensive `mkdir -p`) |
| `scripts_test/full.sh` | `--output results/local/latest.json` (+ defensive `mkdir -p`) |
| `package.json` | `view` script → `promptfoo view results/local/latest.json` |

### CI → `results/ci/`

| File | Change |
|------|--------|
| `promptfooconfig.ci.yaml` | `outputPath: results/ci/latest.json` |
| `.github/workflows/promptfoo-gateway.yml` | eval `--output results/ci/latest.json`; `upload-artifact` path → `results/ci/latest.json`; defensive `mkdir -p results/ci` before the eval step |

`.gitignore` already ignores `results` wholesale, so both subdirectories stay
untracked — no change needed.

## Edge cases

- **Commit with no CI run** — `--all`: warn and skip; `--commit`: error.
- **Multiple runs per commit** (re-run, or push + PR both triggered) — pick the
  most recent **successful** run.
- **Run exists but artifact expired** (GitHub's ~90-day retention) — `--all`:
  warn and skip; `--commit` / `--latest`: error.
- **No commits in date range** (`--all`) — warn, exit 0 (not an error).
- **Bad `--start-date` / `--end-date`** — validate `YYYY-MM-DD`, clear error.
- **`gh` not authenticated** — let `gh`'s own error surface.

## Verification

1. `pull_ci_results.py --latest` — downloads the newest run's result into
   `results/ci/` and imports it; second run warns "already exists" and skips.
2. `pull_ci_results.py --latest --fail-on-existing` (after step 1) — exits
   non-zero on the existing eval.
3. `pull_ci_results.py --all --start-date <recent> --no-import` — populates
   `results/ci/` with multiple files, no DB writes.
4. `pull_ci_results.py --commit <sha-with-no-run>` — errors clearly.
5. A local `./scripts_test/smoke.sh` run writes to `results/local/smoke.json`.

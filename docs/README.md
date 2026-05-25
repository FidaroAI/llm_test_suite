# Fidaro Model Eval Suite

A [promptfoo](https://www.promptfoo.dev)-powered suite for comparing production
Fidaro against dev Fidaro and [coming soon] other 3rd parties.

## Setup / Quick Start

### Dev Setup

It's recommended to install `direnv` for activating python and sourcing .env files.

* `nvm use` or `nvm install` as necessary
* `pnpm install`
* `python -m venv .venv && source .venv/bin/activate`
* `pip install -r requirements.txt`
* `cp .env.example .env` and fill in your env vars as documented in the example.
* `direnv allow` (if using direnv)
* `pnpm run dataset` to download datasets (one-off command)

### Plaintext gateway docker image

TODO: Script this. We can't pull from ECR as those are x86 builds and we're all
using ARM Macs.

* Check out the [`secure_enclave` repo](https://github.com/FidaroAI/secure-enclave)
* Follow dev setup for that repo
* Run `docker compose build` — overkill, as it builds everything

## Plaintext Gateway

To test against Fidaro we expose the vLLM instance running on Phala to the world,
then point a "plaintext gateway" at it. The gateway is an (almost) OpenAI-compatible
API that bypasses all Fidaro encryption, so it is not suitable for public use.

The gateway sets the system prompt, handles tool calls, and relays data back to the
client. There's an AI-generated doc on the whole thing in [GATEWAY.md](GATEWAY.md).

You must run the gateway before running any tests; the helper script is
[run_plaintext_gateway_wrapper.sh](../scripts_repo/run_plaintext_gateway_wrapper.sh).

> WARNING: You are responsible for ensuring the gateway is running. The test suite won't do that.

The script runs two gateway instances: port 8082 points at prod Phala, port 8084 at
a dev instance.

> WARNING: This whole setup is unstable. Verify that both instances are available before running.

The two instances map to the promptfoo providers
[dev](../providers/fidaro_plaintext_gateway_phala_dev.yaml) and
[prod](../providers/fidaro_plaintext_gateway_phala_prod.yaml).

## Running the tests

CI currently runs a random assortment of tests against prod. This isn't especially
useful yet, as there's no baseline to compare against. You can run the same tests
with [fidaro.sh](../scripts_test/fidaro.sh).

The interesting tests for a human to run are
[fidaro_compare.sh](../scripts_test/fidaro_compare.sh).

TODO: Continue

## Comparison runs (config-driven)

[run_comparison.py](../scripts_repo/run_comparison.py) drives a whole prod-vs-dev
comparison from a single config file, so gateway/model configuration and the test
run stay together. Put a named config in `comparisons/` (e.g.
`comparisons/prod_vs_dev_gemma.json`; see [example.json](../comparisons/example.json))
and run:

```
python scripts_repo/run_comparison.py comparisons/prod_vs_dev_gemma.json
```

Each invocation writes its outputs (prod/dev results, the comparison report,
the rendered compose) to a fresh per-run subdirectory
`comparisons/<name>/run_<YYYYMMDD-HHMMSS>/`, so runs never overwrite each other.
The vLLM-options cache that drives the redeploy decision lives one level up at
`comparisons/<name>/` so it persists across runs. These directories are
gitignored; only the config JSONs are tracked. The script validates the config,
optionally redeploys the Phala dev CVM when `vllm-options` change (after
confirmation — see [deploy_phala.py](../scripts_repo/deploy_phala.py)), starts
both plaintext gateways (mounting a dev system prompt if given), runs both
passes, and opens the report. It does **not** freeze a baseline. `BRAVE_API_KEY`
must be in the environment; a Phala redeploy also needs `PHALA_DOCKER_COMPOSE_FILE`
set and `.env.phala` in the repo root. Full design:
[the spec](superpowers/specs/2026-05-22-comparison-orchestrator-design.md).

Each side of a comparison runs through a **dynamic provider**
(`fidaro_plaintext_gateway_phala_dynamic_{prod,dev}.yaml`) rather than the static
`fidaro_plaintext_gateway_phala_{prod,dev}.yaml`. The static providers freeze their
`openai:chat` model/temperature/max_tokens and so can't reflect a gateway the run
has reconfigured. The dynamic ones template those values from
`COMPARISON_{PROD,DEV}_{MODEL,TEMPERATURE,MAX_TOKENS}` env vars (promptfoo renders
`{{ env.* }}` in a provider's id/config at load time); `run_comparison.py` sets
them per run from the comparison config's `prod-provider-options` /
`dev-provider-options` (`{model, temperature, max_tokens}`). Putting the model in
the provider id keeps promptfoo's request-body cache key model-aware, so a cached
result can't leak across models. For dev the `model` must equal `vllm-options.model`
(the model the redeploy serves); prod's `model` is whatever prod Phala runs. The
static providers are kept for ad-hoc runs. See
[example.json](../comparisons/example.json).

## Project Structure

Note: There's a lot of experimentation cruft in the repo that will eventually be
removed. Three easy rules:

* If the file begins with "wip" you can ignore it
* Some unused things are documented directly
* To figure out what's important, start with these two files:
  * [run_plaintext_gateway_wrapper.sh](../scripts_repo/run_plaintext_gateway_wrapper.sh)
  * [fidaro_compare.sh](../scripts_test/fidaro_compare.sh)

```
assertions/      custom assertions for promptfoo to call
baselines/       test runs against prod Fidaro, used as a baseline for testing new model configurations
data/            prompts and rubrics sourced from the internet (some are downloaded dynamically)
deep_eval/       IGNORE: AI-generated setup of deep_eval as an alternative to promptfoo
docs/            documentation
hooks/           custom hooks promptfoo can call before/after tests
langsmith_demo   IGNORE: AI-generated setup of langsmith as an alternative to promptfoo
prompt_templates trivial boilerplate for promptfoo
providers        configurations for promptfoo "providers" — providers are effectively models
results          results from promptfoo runs and our custom reports. Not checked in to git.
scripts_repo     scripts for doing anything other than running tests
scripts_test     scripts for specifically running tests
system_prompts   IGNORE: not currently used. Would vary the system prompt in Fidaro
tests            test cases. 3 formats: yaml configs, csv lists promptfoo uses to autogenerate tests, and custom python generators. All supported by promptfoo out of the box.
user_prompts     IGNORE: not currently used. Will probably go away.
```

## Requirements

See [REQUIREMENTS.md](./REQUIREMENTS.md)

## Cheat sheet

Lazy, terse documentation. Can be expanded later.

* Promptfoo has a UI for viewing results, backed by a local database. Local test runs
  populate it automatically; external runs can be imported. Run the viewer temporarily
  with `pnpm view`, or inside docker via
  [run_promptfoo_docker.sh](../scripts_repo/run_promptfoo_docker.sh).
  * [fidaro.sh](../scripts_test/fidaro.sh) runs the docker container automatically to display results.
  * Warning: I've seen the docker container crash a lot.
* We have scripts for generating custom reports comparing a baseline against a new run.
  * Again see [fidaro.sh](../scripts_test/fidaro.sh), which demonstrates report generation.
  * There's a handy clipboard icon in the results to get a curl command to rerun a test manually.
  * The report from [compare_matrix.py](../scripts_repo/compare_matrix.py) isn't really used right now and might go away.
* We can't get thinking blocks before websearch from the plaintext gateway right now, because it doesn't use SSE.
* We get `\n\n\n` in responses after thinking because deepseek_r1's parser strips `<thinking>` tags but swaps them for newlines: two newlines for the tags plus one for the existing whitespace. See [strip_before_triple_newline.py](../hooks/strip_before_triple_newline.py).
* CI is fairly useless right now — just a sanity check that things work. Later it should run a set of tests, compare against the baseline, and error if we regressed too much. This is still likely to be flaky.
* Rubric LLM (non-deterministic) tests are evaluated in Bedrock right now. See [promptfooconfig.yaml](../promptfooconfig.yaml).
* Promptfoo can parallelize tests. See `maxConcurrency` in [promptfooconfig.yaml](../promptfooconfig.yaml).

### Data import and transform

We import data from various sources and transform it into test cases. Data must be
downloaded once in advance by the user; see `pnpm dataset`.

Generation logic lives in `tests/xxx_gen.py`. Any python file ending in `_gen.py` is
assumed to be a generator. Its tests go into a suite named `xxx`. Suites are our own
invention, not promptfoo's: we add metadata to tests to identify their suite, and can
filter on it during runs with `--filter-metadata suite=xxx`.

Generated tests use a single configuration file. The default is
[suite_generation_config.json](../tests/suite_generation_config.json), overridable via
the env var `SUITE_GENERATION_CONFIG_FILE`. It controls how many tests each generator
produces and can randomize the tests (with a fixed seed).

We can also restrict the number of rubrics per test case. These datasets often have a
lot of assertions, and running every rubric would take too long.

### Test classification

Every generated test is labelled on two shared, suite-independent axes so we can slice
the suites consistently:

* `request_type` — *what* the user is trying to do (e.g. `coding`, `planning`,
  `research_synthesis`).
* `domain` — the subject *area* (e.g. `finance_business`, `science_stem`).

The controlled vocabularies (the full list of allowed values, with descriptions) live in
[tests/classification.py](../tests/classification.py) and are the single source of truth.
Labels are **not** stored in the raw datasets. They live in
`data/classifications/<suite>.json`, keyed by a hash of the prompt text, and are merged
into each test's metadata at generation time by `classification.augment`. Keying on the
prompt hash means the mapping survives dataset re-downloads/reordering and works even for
suites whose rows have no stable id (e.g. multifaceted). Native dataset fields are kept
for provenance (`native_domain` for research_rubrics, `category` for agentharm, `source`
for multifaceted) but aren't used as the classification — they use disjoint per-dataset
vocabularies. The agentharm suite is additionally tagged `censorship: true` (other suites
omit the key) so its deliberately harmful prompts can be excluded from benign runs.

Populate/refresh the labels with the LLM classifier (uses the same Bedrock model family as
the grader; idempotent — only classifies prompts not already labelled):

```
python scripts_repo/classify_tests.py                      # all suites
python scripts_repo/classify_tests.py --suite multifaceted # one suite
python scripts_repo/classify_tests.py --force              # re-classify everything
```

**Selecting by classification.** Two complementary mechanisms:

* *Run a single class* — filter at run time, e.g.
  `--filter-metadata request_type=coding` or `--filter-metadata domain=finance_business`.
  Exclude the harmful suite with `--filter-metadata censorship=false` (untagged tests match).
* *Take an even sample across a class* — set `stratify` in the suite-generation config to
  take N tests from each value of a dimension. For example, to generate 2 research_rubrics
  tests per domain:

  ```json
  "research_rubrics": {
    "randomize_selection": true,
    "random_seed": 1,
    "stratify": {"by": "domain", "per_group": 2}
  }
  ```

  Add an optional `"groups": [...]` to restrict to specific values. `number_to_generate`
  still applies afterwards as an overall ceiling. See
  [tests/suite_config.py](../tests/suite_config.py).

## Gotchas

* When running the plaintext docker gateway locally, don't forget to set the Phala
  vLLM endpoint and Brave credentials environment variables in the docker-compose file
  (or the docker command, if running directly). The scripts help, but it's easy to
  screw up.
* Be aware of promptfoo's caching behaviour. If it sees a test case it's run before and
  has data for, it'll use the cached result — possibly without you realising. Use
  `--no-cache` to prevent this.

# Quick Feature Backlog

* Script to change the Fidaro system prompt: create system prompts in this repo, then
  change the gateway's docker run command to mount the prompt file(s). Be careful — our
  prompts expect a certain placeholder for the websearch prompt.
* Audit the generated tests and find a "good" subset that gives good coverage (of the
  use cases we want). Can use an LLM to audit; too big for humans.
* ~~Categorise tests!~~ Done: two-axis (`request_type`/`domain`) classification, see
  [Test classification](#test-classification). Remaining: review the LLM-assigned labels
  for accuracy and tune the vocabularies.
* Do system prompt iterations :)
* Iterate on model config. This requires a vLLM restart, can be done against dev, and
  needs scripting. WARNING: be very careful with the Phala CLI — there's no gate to
  prevent messing with prod. Should probably create a service account with restricted
  prod access (if possible).
* Iterate on the choice of model. Similar challenge to the previous point.
* Configure a provider to run Perplexity or Venice and compare against Fidaro.
  * Via their APIs — might not match the quality of their web apps.
  * Via e.g. Playwright automation to get the full web experience.
* IDEA: to speed up tests, maybe use Bedrock with the same model as us. Tool calls will
  be tricky and might need a small client to handle them. We could test quality without
  tool calls, but results would be limited.
* Stress-test tests for measuring Fidaro's capacity. I've seen API timeout errors that
  seem to come from overloading the number of tests.
* Get more datasets?


## TODO

* Make the whole dev setup and plaintext gateway less error-prone for users, e.g. with a `just` script.
* Add a clean requirements doc and point Claude at it.

# Fidaro Model Eval Suite

A [promptfoo](https://www.promptfoo.dev)-powered suite for comparing production
Fidaro against dev Fidaro and [coming soon] other 3rd parties.

## Setup / Quick Start

### Dev Setup

* `nvm use` or `nvm install` as necessary
* `pnpm install`
* `python -m venv .venv && source .venv/bin/activate`
* `pip install -r requirements.txt`
* `cp .env.example .env` and fill in your env vars as documented in the example.
* `pnpm run dataset` to download datasets (one off command)

### Plaintext gateway docker image

TODO: Script this. Note that we can't pull from ECR as those are x86 builds and
we're all using ARM Macs.

* Check out the `secure_enclave` repo https://github.com/FidaroAI/secure-enclave
* Follow dev setup for that repo
* Run `docker compose build` - note that this is overkill as it builds everything

## Plaintext Gateway

In order to test against Fidaro we have done some hacking. We expose the vLLM instance
running on Phala to the world. We then have a "plaintext gateway" which is an
(almost) OpenAI compatible API which points to that vLLM endpoint. The gateway
avoids all the Fidaro encryption and is not suitable for public use.

The gateway is responsible for setting the system prompt, handling tool calls, and
relaying data back to the client. There's an AI generated doc on the whole thing here: [/Users/badger/fidaro/git/llm_test_suite/docs/GATEWAY.md](docs/GATEWAY.md)

In order to run any tests you will need to run the gateway. There's helper script here [run_plaintext_gateway_wrapper.sh](scripts_repo/run_plaintext_gateway_wrapper.sh) for that.

> WARNING: You are responsible for ensuring the gateway is running. The test suite won't do that.

That script will actually run two instances of the gateway, one on port 8082 and one on 8084.
The first will point at prod Phala and the second will point at a dev instance.

> WARNING: This whole setup is unstable. Verify that we have two such instances available before running.

The two instances map to the promptfoo providers [dev](providers/fidaro_plaintext_gateway_phala_dev.yaml) and [prod](providers/fidaro_plaintext_gateway_phala_prod.yaml).

## Running the tests

CI currently runs a random assortment of tests against prod. This doesn't do anything particularly useful as there's no baseline to compare against yet. You can run the
same tests with [fidaro.sh](scripts_test/fidaro.sh).

The interesting tests for a human to run are [fidaro_compare.sh](scripts_test/fidaro_compare.sh).

TODO: Continue

## Project Structure

Note: There's a lot of stuff in the repo that's been used for experimentation. That
will eventually be removed. Three easy rules:

* If the file begins with "wip" you can ignore it
* Some unused things are documented directly
* To figure out what's important, start with these two files:
** [text](scripts_repo/run_plaintext_gateway_wrapper.sh)
** [text](scripts_test/fidaro_compare.sh)


```
assertions/      custom assertions for promptfoo to call
baselines/       test runs against prod Fidaro that can be used as a baseline for testing new model configurations against
data/            prompts and rubrics sourced from the internet (some are downloaded dynamically)
deep_eval/       IGNORE: AI generated setup of deep_eval framework as an alternative to promptfoo.
docs/            Documentation
hooks/           customr hooks which promptfoo can call before/after tests
langsmith_demo   IGNORE: AI generated setup of langsmith framework as an alternative to promptfoo.
prompt_templates Trivial boiler plate for promptfoo
providers        coonfigurations for promptfoo "providers" - providers are effectively models
results          Results from promptfoo runs and our our custom reports. Not checked in to git.
scripts_repo     Scripts for doing anything other than running tests
scripts_test     Scripts for specifically running tests
system_prompts   IGNORE: Not currently used. Would in theory be used for varying system prompt in Fidaro
tests            Test cases. 3 formats: yaml configs, csv lists which promptfoo uses to autogenerate tests, and custom python scripts for generating tests. All are supported by promptfoo out of the box.
user_prompts     IGNORE: Not currently used. Will probably go away.
```

## Cheat sheet

This is lazy, terse documentation. Can be expanded later.

* Promptfoo has a UI for viewing results. It's backed by a local database. Test runs done locally automatically populate the database. External runs can be imported. You can run the gateway temporarily with `pnpm view` or you can have it running inside docer using [run_promptfoo_docker.sh](../scripts_repo/run_promptfoo_docker.sh).
** The [fidaro.sh](scripts_test/fidaro.sh) script will automatically run the docker container to display results.
** Warning, I've seen the docker container crash a lot.
* We have scripts for generating custom reports for comparing a baseline of test results against a new run.
** Again see the [fidaro.sh](scripts_test/fidaro.sh) script as this demonstrates report generation.
** There's handy clipboard icon in the results to get a curl command to rerun the test manually.
** The report by [compare_matrix.py](../scripts_repo/compare_matrix.py) isn't really used right now. It might go away.
* We can't get thinking blocks before websearch from the plaintext gateway right now. This is because it doesn't use SSE.
* The reason we have \n\n\n in our responses after thinking is that deepseek_r1's parse strips <thinking> tags but swaps them for new lines. So we basically get two newlines for the tags and one for the already existing whitespace. See [strip_before_triple_newline.py](../hooks/strip_before_triple_newline.py)
* CI Is somewhat usesless right now. It's just a sanity check things are working. Probably for later we will make it run a set of tests and compare against the baseline and error if we regressed too much. This is still ;likely to be flakey.
* Rubric LLM (non-deterministic) tests are being evaluated in Bedrock right now. See [promptfooconfig.yaml](../promptfooconfig.yaml)
* Promptfoo can parallelize tests. See `maxConcurrency` in [promptfooconfig.yaml](../promptfooconfig.yaml)


### Data import and transform

We import data from various data sources and transform it into test cases. Data must be downloaded once in advance by the user. See `pnpm dataset`.

Generation logic is in files called `tests/xxx_gen.py`. Any python file ending in `_gen.py` is assumed to be a genertor. Those tests go into a suite with the suite name `xxx`. Suites are our own invention, not promptfoos. We add meta data to tests to identify which suite they belong to. They can be filtered during test runs using `--filter-metadata suite=xxx`.

Generated tests use a single configuration file. The default is [suite_generation_config.json](../tests/suite_generation_config.json) but can be overridden with the env var `SUITE_GENERATION_CONFIG_FILE`. The config file lets you adjust how many tests from each generator are actually generated. It also lets you randomize the tests (with a fixed seed).

We have an option to restrict the number of rubrics for each test case. These data sets often have a lot of assertions and it would take too long to run them all if we used every rubric.

## Gotchas

* When running the plaintext docker gateway locally, don't forget to set the phala vllm endpoint and brave credentials environment variables in the docker-compose file or in the docker command if you run directly. The scripts help with this but it's easy to screw something up.

* Be aware of promptfoos caching behaviour. If it sees a test case that it's run before and has data for then it'll use a cached result. This may happen without you realising. Use `--no-cache` to prevent this.

# Quick Feature Backlog

* Script that let's us change the Fidaro system prompt. We can do that by creating a sys prompts in this repo and then changing the docker run command for the gateway to mount the prompt file(s). Need to be careful as our prompts expect a certain placeholder for the websearch prompt.
* Audit the generated tests and find a "good" subset that gives us good converage (of the use cases we want). Can use an LLM to audit. Too big for humans
* Categorise tests! See below
* Do sys prompt iterations :)
* Need to be able to iterate on model config. This requires a restart of vllm. Can be done against dev. Needs scripting. WARNING: We must be very careful when using the phala CLI. There's no gate to prevent messing with prod. Should probably create a service account and restrict access to prod (if possible)
* Need to be able to iterate on the choice of model. Similar challenge to the previous point.
* Configure a provider to run Perplexity or Venice and run tests against those. Do a comparison with Fidaro.
** Do this with their APIs - might not be the same quality as their web app.
** Do this with e.g. Playwright automation to get full web experience.
* IDEA: To speed up tests, maybe we try using bedrock with the same model as us. Tool calls will be tricky. Might need to build a small client somewhere that handles the tool calls. We could test quality without tool calls, but results will be limited.
* Stress test tests for testing Fidaro's capacity. Note that I've seen API timeout errors due to (seemingly) overloading the number of tests.
* Get more data sets?

## Ideas for test categorisation

### Use cases

#### Type of request

* Answering simple factual questions
* Answering simple factual questions about current affairs
* Planning
* Coding
* Research
* Data analysis
* Creative writing

#### Domain

* Finance
* Holiday planning
* Shopping
* Personal medical
* Legal
* Sciences
* Literature

### Rubric based tests

- Facts: Achieved the goal of the user requst
- Tone: Terse, partonisning, helpful explanatory (Match this using personal settings)
- Bias: Did it give a bias result
- Gatekeeping/Refusal: Did it say no

### Deterministic tests
- substring/regex matches: Does the response contain particular text
- length of response:
- web search



## TODO

* Make the whole dev setup and plaintext gateway stuff less error prone for users, e.g. use a just script.
* Add a clean requirements doc and point claude at it


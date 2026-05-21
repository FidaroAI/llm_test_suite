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

This is lazy documentation:

TODO: Not done yet

* The reason we have \n\n\n in our responses after thinking is that deepseek_r1's parse strips <thinking> tags but swaps them for new lines. So we basically get two newlines for the tags and one for the already existing whitespace.
* Document that we only generate 3 rubrics.
* Test generation, random selection, configs.
* Promptfoo UI
* Document the report and the clipboard

## Gotchas

* When running the plaintext docker gateway locally, don't forget to set the phala vllm endpoint and brave credentials environment variables in the docker-compose file or in the docker command if you run directly. The scripts help with this but it's easy to screw something up.

* Be aware of promptfoos caching behaviour. If it sees a test case that it's run before and has data for then it'll use a cached result. This may happen without you realising. Use `--no-cache` to prevent this.

* Currently we lose pre-web_search reasoning because the plaintext API doesn't use streaming. I'm not even sure if promptfoo can use streaming


# Quick Feature Backlog


## TODO

* Make the whole dev setup and plaintext gateway stuff less error prone for users, e.g. use a just script.
* Add a clean requirements doc and point claude at it

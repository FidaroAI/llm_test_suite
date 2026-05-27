# Tasks for Human

* Run with new prompts
* Try a g-eval run (correctness, semantic similarity, and hallucination)
* Try a pick-best run
* Rerun vllm in debug mode: - VLLM_LOGGING_LEVEL=DEBUG
* Get prometheus working
* Review the stock picks tests. Where are we at?

* Do a hardcore prompt experiment that literally only focusses on stocls


# Tasks for Claude

This is in lieu of proper project management, e.g. Jira

## Task 1

I want you to implement more tests like the stock price tests that rely on up to date information. They will work the same way, specifically, fetch some data, then generate the right tests. Here are the sets of tests I want:

* Weather
* Todays date
* Flight status
* Traffic and travel
* What is the title of the most recent post on the Hacker News front page
* What is the timestamp of the most recent NYSE market update on their site

## Task 2

Script up a provider that uses playwright or similar to interact with https://lumo.proton.me/guest. I want to be able to send requests to
that from promptfoo. We'll need a proxy to convert promptfoo openAPI requests and pass them to the UI. You may assume that I'm
logged in at https://lumo.proton.me/u/2/?welcome=true. Use the credentials:
username: james.dean.the.developer@gmail.com
password: mvf-AEQ0auh1faq7qfn
TODO: Consider for confer, xv, and venice

## Task - not ready to start

Refactor the scripts and folder structure - things are getting out of hand.

* Scripts which generate tests need to be moved to another or a sub directory for clarity. Probably make the _gen a prefix not a postfix
* Helper python scripts should not be inside the tests folder (or anywhere else). Let's create a proper python library structure and have those in one place. Same goes for scripts_repo. Anything which is python in there needs moving into the python library. Everything else can be left in scripts_repo for now. Let's set this up properly with a pyproject.toml and use poetry.
* example.json and output and comparisons dirs are messy. Create a committed example.json and a script for the user that sets up example script copies like .env.local.
* Move the two demos to a demons folder.

## Task - not ready to start

Fix stratify to work better

# Tasks for secure enclave

## Task 1

Capture pre-thinking and tool calls!


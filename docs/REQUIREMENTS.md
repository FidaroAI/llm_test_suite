# Overview

Fidaro is a LLM chat bot similar to ChatGPT, Perplexity and Gemini. Fidaro currently only has a web
interface. We have however created a makeshift OpenAPI compatible API for talking to Fidaro. We refer to this as the "plaintext gateway".

For a deep dive into the gateway see [GATEWAY.md](./GATEWAY.md)

For an overview of some brainstorming we did see [BRAINSTORMING.md](./BRAINSTORMING.md)

# How is quality impacted

The key factors that affect the quality of Fidaro's outputs are:

* Choice of model
* Configuration of model
* Sytem prompt

We refer to these generally as the Fidaro "model configuration".

We want to build a test suite to allow us to:

* run lots of test cases in bulk to find quality issues in our product
* spot regressions in the quality of our responses
* iterate on the three components above and measure the impact against our baseline
* compare our product against competitor products.

Additionally - for the future - we may want to expand the functionality in the gateway to have smarter
logic, e.g. coordinating multiple models, pre/post processing of responses, caching, more tools. The
same test suite would be expected to "just work" as it would all be interfaced via the plaintext gateway.

# Full Requirements

## Regression testing

* As a developer of Fidaro, I want to know if our quality regresses anywhere across a broad range of user prompts when new configurations are pushed to production.

## Iteration of model configuration

* As a developer of Fidaro, I want to be able to change the system prompt and measure the impact on the
  quality of output across a broad range of user prompts.
* As a developer of Fidaro, I want to be able to change the model parameters (such as top_p) and measure the impact on the
  quality of output across a broad range of user prompts.
* As a developer of Fidaro, I want to be able to change the the model and measure the impact on the
  quality of output across a broad range of user prompts.

## Comparison of Fidaro vs Competitors

* As a developer of Fidaro, I want to be able to comparee the quality of output across a broad range of user prompts of Fidaro against a range of competitor APIs
* As a developer of Fidaro, I want to be able to comparee the quality of output across a broad range of user prompts of Fidaro against a range of competitor web apps.

## Load testing

* As a developer of Fidaro, I want to be able to test how well a Fidaro server copes under high load, i.e. high volume of simultaneous requests. I want to measure response times and find out at what load response times start falling.

# Types of Tests

We want at least the following classes of tests:

## Deterministic tests

* Substring/regex matches: Does the response contain particular text string
* Length of response: How verbose is the response.
* Length of thinking: How much does the model think before giving an answer. This is a proxy for speed of response.
* Tool use: Did certain tools fire for a request? How many times did they fire?


## Non-Deterministic Rubric Tests

These are tests which use a rubric prompt to evaluate the quality of the model's response by using another LLM. Types of things we might assert on:

* Facts: Did the model achieve the goals of the user?
* Tone: Was the model terse, partonisning, helpful, explanatory, friendly
* Bias: How biased was the model?
* Gatekeeping/Refusal: Did the model refuse to answer the user?

## Other test factors

* Single prompt + single response conversations
* Multi-message conversations
* Long context conversations

# Categories of Tests

Will will try to group tests into categories so that we can see how well we do in specific areas.
We will be focussing our business strategy on certain customer segments so some of these will be
more important to us than others.

These categories need refining and will be somewhat dependent on what type of test data we can
source.

Suggested breakdown is two fold into the type of request and the domain that request is in. Here are
some examples

## Type of request

This categories what type of action the user is performing, but is agnostic to the domain.

Non-exhaustive examples:

* Answering simple factual questions
* Answering simple factual questions about current affairs
* Planning
* Coding
* Research
* Data analysis
* Creative writing
* Personal 1-1 chat

## Domain

This categories the domain the user is interested in, but not the type of action.

Non-exhaustive examples:

* Finance
* Holiday planning
* Shopping
* Personal medical
* Legal
* Sciences
* Literature
* Relationships

# Suggested Deliverable Milestones

* DONE: Compare Fidaro prod against Fidaro dev environment (which new configuration under test)
* Compare Fidaro prod against 3rd parties using their APIs, e.g. ChatGPT
* Compare Fidaro prod against 3rd paries using their website directly, using Playwright





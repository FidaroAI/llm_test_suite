# Streaming inference and provider-specific output — design

Adds two capabilities to the `llmeval` plumbing:

1. **Streaming.** A provider can be configured to stream its response. The suite
   accumulates the stream itself and writes the same row it would have written
   without streaming — except that a call which hits its timeout now leaves the
   partial answer and partial reasoning in the store instead of nothing at all.
2. **Provider-specific output.** Non-standard top-level response data (today the
   Fidaro `/v2` `fidaro` object, which carries the chat title) is captured
   verbatim into a new `results.provider_specific_output` column.

## 1. Why

A class of test cases we want to run checks whether the model gets stuck in a
repetitive loop. Today that test is unrunnable: the call exceeds its timeout,
the HTTP request is torn down before any body is read, and the stored row is an
error with no output. The evidence needed to *decide* whether the model looped
is exactly the evidence the timeout destroys.

Streaming fixes this by moving accumulation client-side. Bytes arrive as the
model produces them, so when the deadline trips we already hold the text.

Separately, some test cases want to assert on Fidaro-specific data that has no
place in the OpenAI response schema. `/v2` namespaces all of it under a single
top-level `fidaro` object.

## 2. Wire format

`/v2/chat/completions` is mounted **only by the orchestrator**
(`apps/orchestrator/src/orchestrator/main.py:168`). The llm-gateway on ports
8082/8084 exposes `/v1` only, which speaks the older plaintext frames
(`event: chunk` / `event: title` / `event: done`) and has no `fidaro` envelope.
So `base_url` for a streaming Fidaro provider must point at an orchestrator
`/v2`.

The orchestrator's own aggregation
(`apps/orchestrator/src/orchestrator/openai_v2/aggregation.py`) is the contract
this design mirrors:

| Frame | Contribution |
| --- | --- |
| `delta.content` | appended to `content` |
| `delta.reasoning_content` | appended to `reasoning_content` |
| `fidaro` object on any chunk | merged into the accumulated `fidaro` |
| terminal chunk | `finish_reason`, and `usage` when `stream_options.include_usage` |
| `data: [DONE]` | end of stream |

The title arrives as its own chunk with a **no-op delta** the moment it is
known, and is repeated on the terminal chunk
(`aggregation.py:186`). Reading `fidaro` off every chunk that carries one, and
merging, handles both placements without caring which arrived.

## 3. Why not litellm

`DESIGN.md` §2 chose litellm as the provider layer. This design keeps that for
the non-streaming path and the judge, and hand-rolls streaming. The reasons are
measured, not assumed — probed against a local server mimicking `/v2` on
litellm 1.94.0:

| Behaviour | Non-streaming | Streaming |
| --- | --- | --- |
| Top-level `fidaro` key | survives into `resp.fidaro` | survives on the chunk carrying it |
| `usage` | server's values | **replaced** by litellm's local tokenizer estimate (server sent 3/4/7; litellm reported 8/2/10) |
| `litellm.stream_chunk_builder` | — | **drops `fidaro`**, and inherits the bogus usage |
| Read timeout mid-stream | — | **iteration ends silently, no exception** |

The third row is disqualifying on its own. The entire purpose of streaming here
is to distinguish "the model was cut off mid-loop" from "the model answered".
litellm returns a generator that simply stops, so a truncated stream is
indistinguishable from a complete one — it destroys the one signal the feature
exists to capture. The `usage` row independently breaks the requirement that a
streamed row match a non-streamed one.

Hand-rolling costs roughly 120 lines of SSE parsing over `httpx`, all of it
against a format we control and already have a reference implementation for.

## 4. Module layout

New `llmeval/streaming.py`, containing no I/O:

- `iter_sse_chunks(lines: Iterable[str]) -> Iterator[dict]` — yields parsed
  `data:` payloads, stops at `[DONE]`, ignores blank lines, SSE comments
  (`:`-prefixed) and non-`data:` fields.
- `StreamAccumulator` — `.feed(chunk: dict)` and `.completion_dict()`. Mirrors
  the orchestrator's `aggregate()`: content parts, reasoning parts, usage,
  finish reason, merged `fidaro`. Also exposes `.content_len` /
  `.reasoning_len` for the timeout message, and `.chunks` for the truncation
  message.

New `StreamingOpenAIProvider` in `llmeval/providers.py` — owns httpx, the
deadline, and the mapping from accumulator state to `Completion`.

The split is what makes the contract testable. "Aggregate the way the
orchestrator does" is a claim about data, not about sockets: keeping the
accumulator pure lets the parity test feed a canned chunk list in and compare
against a canned `chat.completion`, with no HTTP anywhere. The provider is then
only responsible for bytes and clocks.

## 5. Configuration

```python
class ProviderConfig(BaseModel):
    stream: bool = False
```

`stream` joins the cache-key namespace (`cache_key.build_namespace`), so
`cache_key_fields` can select it and `cache_key_fields=None` includes it.

There is **no** streaming timeout in the provider config. `rewrite` already
settled where a timeout lives: `Provider.complete(messages, timeout=...)`, fed
from `TestCase.timeout` falling back to `RunPolicy.timeout`, deliberately kept
out of provider config because "a provider's `params` feed its cache key, so a
timeout parked there would change the identity under test"
(`models.py:63-69`). Streaming reuses that same value as its wall-clock
deadline. One timeout concept, one place to set it.

Streaming is OpenAI-compatible SSE only. `stream: true` on a model whose litellm
prefix is not `openai/` raises at `build_provider()` rather than silently
falling back to a non-streaming call.

`build_provider` order stays: an explicit `extra.provider_impl` factory wins,
then `stream` selects the streaming provider, then litellm.

## 6. Completion and attempt shape

```python
@dataclass
class Completion:
    output: str
    raw: Any = None
    reasoning: str | None = None
    tokens: Any = None
    latency_ms: float | None = None
    provider_specific: dict | None = None   # {"fidaro": {...}} verbatim
    error: str | None = None                # set when this is a partial
```

`error` on a `Completion` is what lets a provider say "here is what I got, and
here is why it is incomplete" — a state the current `Attempt` model cannot
express, because it treats a failure as the *absence* of a completion.

`runner.Attempt` changes in three small ways:

- `as_row()` contributes `output` / `raw` / `reasoning` / `tokens` /
  `provider_specific` whenever `self.completion is not None`, rather than only
  when the attempt is `ok`. A partial's text reaches the store.
- a new `retryable` property: `False` when the error came from a completion,
  `True` when it came from a raised exception.
- `_attempt` folds `completion.error` into `Attempt.error`, so `ok` stays
  `error is None and completion is not None` and a partial still counts as an
  error everywhere errors are counted.

`_fill_one_result` stops retrying when `not attempt.retryable`. A timeout is not
a transient fault: the ceiling was the user's statement of how long the answer
was worth waiting for, so spending 3× it to learn the same thing is waste — and
these test cases are *expected* to time out. Connection-level failures still
retry as today.

## 7. Timeout and partial capture

The `timeout` passed to `complete()` is a total wall-clock deadline measured
from the start of the request, and is also handed to httpx as the read timeout
so a socket that goes silent aborts rather than hanging until the deadline.

| Outcome | Stored |
| --- | --- |
| Deadline reached | partial output, reasoning and `fidaro`; `error = "stream timeout after 60.0s (content: 48213 chars, reasoning: 1204 chars)"` |
| Stream ends without `[DONE]` | same, `error = "stream ended without [DONE] after 412 chunk(s)"` |
| Non-2xx response | raises (retryable), message includes status and a truncated body |
| Connection failure | raises (retryable) |

A partial is written with `error` set, so `count_results(success_only=True)`
does not count it and `grade.py` skips it. Partials are **not graded yet** —
they are captured for inspection and for SQL, and the assertion class that
consumes them is separate future work.

## 8. Store

`results` gains one column:

```sql
provider_specific_output TEXT   -- e.g. {"fidaro": {"title": "..."}}
```

`SCHEMA_VERSION` 3 → 4. There is no migration path by design, so existing
databases must be deleted. `add_result_row` gains a `provider_specific` keyword
and `ResultRow` a matching field.

The column is deliberately the whole envelope (`{"fidaro": ...}`) rather than
the inner object, so a second vendor key needs no schema change.

## 9. Capture on both paths

The non-streaming path captures `fidaro` too — litellm passes the top-level key
through untouched, verified by probe. So the column is populated whether or not
streaming is on, which is what makes the two paths comparable.

Capture is driven by a module constant:

```python
PROVIDER_SPECIFIC_KEYS = ("fidaro",)
```

An allowlist rather than "any unrecognised top-level key", because litellm
injects several of its own (`service_tier`, `moderation`, `citations`,
`provider_specific_fields`) that would otherwise be swept in.

## 10. Fidelity between the two paths

The streaming provider reconstructs the exact `chat.completion` object the
orchestrator's `aggregate()` would have returned and stores that as `raw`.
`output`, `reasoning`, `tokens` and `provider_specific` are then identical to
the non-streaming path for the same response.

One documented difference: the non-streaming path stores litellm's
`ModelResponse.model_dump()`, which carries litellm's own extra keys
(`service_tier`, `moderation`, `system_fingerprint`). The two `raw` blobs are
therefore semantically equal on every meaningful field but not byte-identical.
Hand-rolling the non-streaming path purely to close that gap is not worth it.

## 11. Configs

`fidaro_prod.json` and `fidaro_dev.json` drop the `params.extra_body.stream`
hack and set `"stream": true`. Both select explicit `cache_key_fields`
(`model`, `temperature`, `backend_version`), so `stream` is not in their key and
flipping it does not invalidate existing results — consistent with the claim
that an aggregated stream and a non-streamed response are the same data.

`venice`, `echo` and the judge stay non-streaming.

`.env.example` gains a note that a streaming Fidaro base URL must be an
orchestrator `/v2`, not the `/v1` gateway.

## 12. Dependencies

`httpx` becomes an explicit dependency of the `providers` extra (today it is
only transitive via litellm) and is lazy-imported like litellm, so the core —
cache, store, grading, stats, reports — stays installable and testable with
neither.

## 13. Tests

All offline, no credentials, per `rewrite/CLAUDE.md`.

- **Parser** — frames split across read boundaries, blank lines, SSE comments,
  `[DONE]` termination, trailing data with no terminator, malformed JSON.
- **Accumulator** — content and reasoning concatenation; usage taken from the
  terminal chunk; `fidaro` merged across the standalone title chunk and its
  terminal repeat; finish reason; empty stream.
- **Orchestrator parity** — a canned chunk sequence through the accumulator
  compared against the canned `chat.completion` the orchestrator's `aggregate()`
  produces for the same `/v1` events. This is the pin on §2.
- **Timeout** — `httpx.MockTransport` serving a byte stream that stalls;
  asserts the partial output survives, `error` is set and names the elapsed
  ceiling, and the call returns near the deadline rather than hanging.
- **Truncation** — stream ends without `[DONE]`; partial kept, error set.
- **Non-2xx** — raises, and the error text carries the status.
- **Provider selection** — `stream: true` routes to the streaming provider;
  `provider_impl` still wins; a non-`openai/` model with `stream: true` raises.
- **Non-streaming capture** — a fake litellm response carrying `fidaro`
  populates `provider_specific`.
- **Store** — `provider_specific_output` round-trips; absent stays NULL.
- **Runner** — a `Completion` with `error` and partial output writes both
  columns, counts as an error, and is **not** retried; an exception still is.
- **Cache key** — `stream` is in the namespace and selectable.

## 14. Out of scope

- The repetitive-loop assertion itself. This is provider support; the assertion
  that consumes a partial is separate work, and needs `grade.py` to stop
  skipping error rows.
- Any CLI or report surface for `provider_specific_output`. Stored verbatim and
  reachable by SQL, per the plumbing contract.
- Streaming for non-OpenAI-compatible backends.

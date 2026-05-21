# Gateway vs. vLLM: what each layer does, and when to talk to which

This document explains the two HTTP endpoints you'll encounter when running
this stack locally, what each one is responsible for, and when to call one
vs. the other. It's aimed at someone new to the repo who knows their way
around HTTP APIs but hasn't worked with self-hosted LLM inference before.

The scope here is the **local plaintext dev path** (`gateway-plaintext` on
`127.0.0.1:8082`, talking to vLLM on `127.0.0.1:8000`). The production
encrypted path on `:8080` adds Noise NK encryption on top of everything
described below, but the gateway behaviors are the same.

---

## 1. The two layers, in one diagram

```
┌──────────────┐    HTTP    ┌──────────────────────┐    HTTP    ┌──────────┐
│  Your code   │ ─────────► │  gateway-plaintext   │ ─────────► │   vLLM   │
│   (curl,     │            │   (FastAPI app)      │            │ (engine) │
│   SDK, etc.) │            │   :8082              │            │  :8000   │
└──────────────┘            └──────────────────────┘            └──────────┘
                            Adds: system prompt,
                                  web-search tool loop,
                                  request validation,
                                  response shape policing
```

### What vLLM is

[vLLM](https://github.com/vllm-project/vllm) is an inference engine — it
loads model weights into GPU/CPU memory and serves token generation over
an HTTP API. The API it speaks is **OpenAI-compatible**, meaning it
implements the same `/v1/chat/completions` request/response shape as
`api.openai.com`. You can point the official `openai` Python or
TypeScript SDK at it and it Just Works.

vLLM does a lot more than relay tokens: it batches requests, manages a
KV cache, runs speculative decoding, parses model output for reasoning
and tool calls, etc. The two parsers that matter here are configured in
[deploy/phala/model-config.json](deploy/phala/model-config.json):

- `--reasoning-parser deepseek_r1` — splits the model's `<think>...</think>`
  block out of `content` and into a separate `reasoning_content` field.
- `--tool-call-parser hermes` — recognizes when the model wants to call a
  tool and packages the call into OpenAI's standard `tool_calls` field.

Without these flags, vLLM still serves chat completions, but you'd get raw
unparsed text with literal `<think>` tags and no structured tool calls.

### What the gateway is

`gateway-plaintext` is a FastAPI app that sits in front of vLLM. It does
not run model inference itself. It exists because vLLM, by design, is a
generic OpenAI-compatible server — it has no opinion about *what* prompt
you should use, *what* tools should be available, or *how* a multi-turn
tool-using conversation should flow.

The gateway encodes those product-level decisions, then forwards each
request to vLLM. From the client's perspective, the gateway is also
OpenAI-compatible (it accepts the same `/v1/chat/completions` shape),
but it's a much more opinionated, narrower API.

The code lives in
[apps/llm-gateway/src/llm_gateway/routes/dev_plaintext_completions.py](apps/llm-gateway/src/llm_gateway/routes/dev_plaintext_completions.py).

---

## 2. What the gateway adds on top of vLLM

Each of the following behaviors is something `gateway-plaintext` does
that you would otherwise have to do yourself if you called vLLM directly.

### 2.1 System prompt injection

The Fidaro system prompt is split across two markdown files:

- [apps/llm-gateway/src/llm_gateway/prompts/core_system_prompt.md](apps/llm-gateway/src/llm_gateway/prompts/core_system_prompt.md)
  — always included
- [apps/llm-gateway/src/llm_gateway/prompts/web_search_prompt.md](apps/llm-gateway/src/llm_gateway/prompts/web_search_prompt.md)
  — appended only when the web-search tool is bound for the request

These are stitched together by
[`build_system_prompt`](apps/llm-gateway/src/llm_gateway/prompts/__init__.py#L45),
which also substitutes:

- `{{CURRENT_DATE}}` — today's date in UTC, computed at request time
- `{{CAPABILITIES}}` — the web-search prompt block, or empty string if
  web search is not bound

The assembled prompt is then prepended to whatever messages the client
sent. If the client also supplied its own `system` message, the gateway
keeps it but stacks it **after** the Fidaro prompt (see
[completions.py:161-164](apps/llm-gateway/src/llm_gateway/routes/completions.py#L161-L164)).
The client cannot replace the Fidaro prompt — only extend it.

vLLM has none of this. If you call vLLM with no system message, the
model sees no system message at all (Qwen3-Next-Thinking's chat template
does not insert a default).

### 2.2 Server-side web-search tool loop

When the model decides it needs to search the web, it emits a
`tool_calls` payload (vLLM's `--tool-call-parser hermes` turns the
model's raw output into OpenAI-shaped tool calls). With vLLM alone, the
HTTP response ends there, and the caller is expected to:

1. Receive the response with `finish_reason: "tool_calls"`
2. Execute the function themselves (e.g. call Brave Search)
3. Append the result as a `role: "tool"` message
4. Send a new request with the updated history
5. Repeat until the model stops requesting tools

The gateway hides all of this. Its
[`_run_tool_loop`](apps/llm-gateway/src/llm_gateway/routes/dev_plaintext_completions.py#L263-L319)
sees the `tool_calls` response, runs the search itself (using the
`BRAVE_API_KEY` env var), appends the tool result, and posts the
updated conversation back to vLLM. It repeats until the model returns
a normal `finish_reason: "stop"`. From the client's perspective, one
POST → one final answer, even though several vLLM round-trips happened
under the hood.

The gateway also caps the number of tool calls per request
(`settings.max_tool_calls_per_request`) and rejects malformed tool-call
payloads, both of which are gateway-only safety nets.

### 2.3 Strict request shape

The gateway's request schema (a Pydantic model at
[dev_plaintext_completions.py:26-43](apps/llm-gateway/src/llm_gateway/routes/dev_plaintext_completions.py#L26-L43))
accepts exactly these fields:

- `model`, `messages`, `temperature`, `max_tokens`
- `stream` (accepted, but **ignored** — see §3.1 below)
- `reasoning_budget_tokens` (custom field, mapped to vLLM's
  `extra_body.thinking_token_budget`)
- `request_id` (custom field, used in error responses)

Notable rejections:

- **`tools` is not accepted.** The gateway controls which tools are
  available based on its own configuration.
- **Messages with `role: "tool"` or `tool_calls` are rejected** with a
  400 ([dev_plaintext_completions.py:50-52](apps/llm-gateway/src/llm_gateway/routes/dev_plaintext_completions.py#L50-L52)).
  Tool history is owned by the gateway, not the client.

vLLM accepts the full OpenAI request shape plus all of its own
extensions.

### 2.4 Non-streaming, always

The gateway hard-codes `stream: false` on the request it sends upstream
([dev_plaintext_completions.py:104](apps/llm-gateway/src/llm_gateway/routes/dev_plaintext_completions.py#L104))
and returns a single JSON body to the client. Even if the client passes
`stream: true`, the value is ignored.

The encrypted gateway on `:8080` *does* stream, but it does so by
consuming vLLM's stream itself and re-emitting Server-Sent Events
wrapped in Noise encryption. The plaintext gateway intentionally
trades streaming for the simpler "one curl, one JSON body" workflow.

---

## 3. Gateway vs. direct vLLM: feature matrix

| Feature | Direct vLLM (`:8000`) | Plaintext gateway (`:8082`) |
|---|---|---|
| **Reasoning** (`reasoning_content`) | Yes — vLLM emits it when `--reasoning-parser deepseek_r1` is set | Yes — pass-through |
| **Tool calls in response** | Yes — if you pass a `tools` array | Yes — but the gateway, not you, decides which tools are bound |
| **Server-side `web_search` execution** | **No** — you receive `tool_calls` and must execute the function yourself, then make a second request with the result | Yes — gateway runs the search and returns only the final answer |
| **Streaming** | Yes — vLLM does SSE on `stream: true` | **No** — hard-coded to non-streaming; client `stream` flag is ignored |
| **Fidaro system prompt** | **No** — model sees only what you send | Yes — automatically prepended |
| **Multi-turn reasoning visible** | Yes (naturally — each tool-loop iteration is its own response, so each round has its own `reasoning_content`) | **No** — only the final turn's reasoning survives in the response |
| **Tool history in messages** (`role: "tool"`, `tool_calls`) | Yes — required for multi-turn function calling | **Rejected** with HTTP 400 |
| **Request shape** | Full OpenAI spec + vLLM extensions | Strict allowlist: `model`, `messages`, `temperature`, `max_tokens`, `stream`, `reasoning_budget_tokens`, `request_id` |
| **`thinking_token_budget`** | Pass as `extra_body: {thinking_token_budget: N}` | Pass as `reasoning_budget_tokens: N` (gateway repackages it) |

A few of these deserve more explanation.

### 3.1 Why direct vLLM gives you multi-turn reasoning but the gateway doesn't

The OpenAI non-streaming chat-completions response has exactly **one**
`message.reasoning_content` field per choice. When the gateway runs
multiple vLLM round-trips internally (one before the web search, one
after), it can only return the final round's response — the earlier
round's reasoning is discarded.

If you call vLLM directly, you make each round-trip yourself, and each
round-trip is its own HTTP response with its own `reasoning_content`.
So you naturally see every reasoning block.

The encrypted gateway on `:8080` sidesteps this because it **streams**
back to the client: pre-tool-call reasoning chunks and post-tool-call
reasoning chunks flow over the same SSE connection in order, and the
client just keeps appending. The plaintext gateway doesn't stream, so it
loses the early-turn reasoning.

### 3.2 Why the model output sometimes has leading blank lines

If you inspect a response carefully, you'll often see `content` start
with `\n\n`:

```json
"content": "\n\nAs of the latest available data..."
```

That's neither the gateway nor a bug. Qwen3-Next-Thinking is trained to
emit its thinking inside `<think>...</think>` followed by a blank line,
then the answer. vLLM's `deepseek_r1` reasoning parser strips the
`<think>...</think>` tags into `reasoning_content` but does **not**
strip the trailing whitespace, by design — some downstream consumers
care about byte-exact model output. The blank line after `</think>`
survives as the leading `\n\n` in `content`.

The gateway does not modify these bytes. If you want them stripped, do
it client-side.

---

## 4. Curl examples

### 4.1 Simple prompt, both endpoints

The request shape is identical at this level. The only observable
difference is that the gateway response was generated with the Fidaro
system prompt in front of your messages.

**Direct to vLLM:**

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [{"role": "user", "content": "Explain bubble sort"}],
    "max_tokens": 2000,
    "temperature": 0.2
  }'
```

**Through the gateway:**

```bash
curl -sS http://127.0.0.1:8082/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [{"role": "user", "content": "Explain bubble sort"}],
    "max_tokens": 2000,
    "temperature": 0.2
  }'
```

### 4.2 Passing your own system prompt (vLLM only)

The OpenAI shape: first message has `role: "system"`. vLLM renders that
into Qwen3's chat template as a `<|im_start|>system\n...\n<|im_end|>`
block.

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [
      {"role": "system", "content": "You are a helpful coding assistant. Answer concisely."},
      {"role": "user", "content": "Explain bubble sort"}
    ],
    "max_tokens": 2000
  }'
```

If you call the gateway with a `system` message, it is **not** rejected,
but it is **appended after** the Fidaro prompt, not used in place of it.

### 4.3 Streaming (vLLM only)

The plaintext gateway does not stream. `-N` disables curl's output
buffering so chunks print as they arrive.

```bash
curl -N -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [{"role": "user", "content": "Count to ten"}],
    "max_tokens": 500,
    "stream": true
  }'
```

Each line of output is an SSE event:

```
data: {"id":"...","choices":[{"delta":{"reasoning_content":"Let me..."}}]}

data: {"id":"...","choices":[{"delta":{"reasoning_content":" count..."}}]}

...

data: {"id":"...","choices":[{"delta":{"content":"1, 2, 3..."}}]}

data: [DONE]
```

### 4.4 Tool calls — the orchestration difference is stark

**Direct to vLLM: you own the loop.**

Round 1 — declare the tool, model decides to call it:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [{"role": "user", "content": "What is Google'\''s stock price?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "web_search",
        "description": "Search the web",
        "parameters": {
          "type": "object",
          "properties": {"query": {"type": "string"}},
          "required": ["query"]
        }
      }
    }],
    "tool_choice": "auto",
    "max_tokens": 2000
  }'
```

Response will include `finish_reason: "tool_calls"` and a function call
payload. You now execute the search yourself, then make round 2:

```bash
curl -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [
      {"role": "user", "content": "What is Google'\''s stock price?"},
      {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"query\":\"GOOG stock price\"}"}}]},
      {"role": "tool", "tool_call_id": "call_1", "content": "GOOG closed at $392.06..."}
    ],
    "max_tokens": 2000
  }'
```

**Through the gateway: it owns the loop.**

```bash
curl -sS http://127.0.0.1:8082/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    "messages": [{"role": "user", "content": "What is Google'\''s stock price?"}],
    "max_tokens": 2000
  }'
```

One JSON body comes back. The gateway already ran `web_search` (assuming
`BRAVE_API_KEY` is set on `gateway-plaintext`) and folded the result
into the final response. You don't declare the tool, don't execute the
function, don't make a second request — but you also don't see the
pre-search reasoning.

### 4.5 Replicating the Fidaro prompt against vLLM (advanced)

If you want byte-for-byte the same conditioning the gateway produces —
useful for "is this behavior coming from the prompt or from the
gateway's orchestration?" debugging — read the assembled prompt out of
Python and paste it as a system message:

```bash
# In a venv with the gateway installed:
cd apps/llm-gateway
uv run python -c "
from llm_gateway.prompts import build_system_prompt
print(build_system_prompt(web_search_available=True))
" > /tmp/fidaro-system.txt

# Then build the request. jq is the cleanest way to safely embed a
# multi-line prompt as a JSON string:
jq -nR --rawfile sys /tmp/fidaro-system.txt '
  {
    model: "Qwen/Qwen3-Next-80B-A3B-Thinking-FP8",
    messages: [
      {role: "system", content: $sys},
      {role: "user", content: "What is Google'\''s stock price?"}
    ],
    max_tokens: 2000,
    temperature: 0.2
  }
' | curl -sS http://127.0.0.1:8000/v1/chat/completions \
    -H 'content-type: application/json' -d @-
```

Two caveats so the equivalence is real:

1. **Pass `web_search_available=True` only if you're also sending the
   `tools` array.** The gateway only inserts the web-search section
   when the tool is actually bound. Advertising a tool the model can't
   call leads to fake tool-use rambling. If you're not passing `tools`,
   build with `web_search_available=False`.
2. **The date placeholder is computed at call time.** `build_system_prompt()`
   calls `datetime.now(UTC)` every call — if you cache the file output,
   it stales by definition.

---

## 5. When to use which

- **Direct to vLLM** for: prompt fuzzing the bare model, evaluating
  multi-turn reasoning visibility, custom tools, streaming, or
  exact-shape OpenAI testing where the gateway's system prompt and tool
  loop would muddy your test.
- **Through the gateway** for: end-to-end testing of what production
  behavior actually looks like (system prompt + server-side web search
  + the exact request shape clients use), without having to deal with
  the Noise encryption layer that the production `:8080` endpoint adds.

For most "is the model behaving right?" questions, calling vLLM
directly gives you a cleaner signal. For "is the product behaving
right?" questions, call the gateway.

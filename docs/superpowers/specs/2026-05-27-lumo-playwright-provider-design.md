# Lumo Playwright Provider — Design

**Date:** 2026-05-27
**Status:** Design only (no implementation in this branch)

## Goal

Let promptfoo evaluate **Lumo** (Proton's chat assistant at
`https://lumo.proton.me/guest`) as if it were any other model provider, by
driving the Lumo *web UI* with Playwright. This gives us the "full web
experience" (web search, thinking, the production model) rather than an API that
may not match it — fulfilling the backlog item in
[docs/README.md](../../README.md) ("Configure a provider to run Perplexity or
Venice... Via Playwright automation to get the full web experience").

Lumo has no public API, so we wrap the browser in a small **OpenAI-compatible
proxy**. The existing test suite already points every provider at a local
OpenAI-compatible endpoint (the plaintext gateway on `127.0.0.1:8082`), so the
proxy slots into the established pattern with **no promptfoo-side changes** — the
Lumo provider YAML is nearly identical to the existing gateway providers.

## Key Decisions (settled during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Runtime | **Python (FastAPI + Playwright)** | Matches the repo's many Python scripts; single venv. |
| Chat mapping | **Fresh Lumo chat per request** | Each promptfoo test is independent; a new conversation gives clean context. |
| Concurrency | **Serialized** (one browser, one turn at a time) | A single browser UI cannot multiplex; serialization also lowers the abuse signal. |
| Login | **One-time headed login**, persisted session, headless serving | Survives captcha/2FA, survives restart, one login ever. |
| Streaming | **Skipped** — return only the final message as a single non-streamed completion | promptfoo does not require streaming; far simpler scraping. |
| Inter-request pacing | **Configurable delay** between requests (default small, can be 0) | Extra abuse-avoidance headroom; opt-in via config. |
| Deployment | **Docker** with persistent volume + `restart: unless-stopped` | "Keep running all the time"; session survives restarts. |

### Rejected alternatives

- **Fully scripted headless login on startup** — breaks the moment hCaptcha/2FA
  appears, and repeated logins look exactly like the abuse pattern we want to
  avoid. Rejected in favour of one-time headed login.
- **In-container headed login via VNC** — heaviest setup (a VNC stack) for a
  one-time action that a host-side headed login solves more simply. Kept as a
  documented fallback only.

## Architecture

New directory `lumo_proxy/`:

| File | Responsibility |
|---|---|
| `server.py` | FastAPI app. `POST /v1/chat/completions` (OpenAI-compatible, non-streaming) and `GET /health` (reports `logged_in`). Holds the serialization lock and wraps Lumo output in OpenAI JSON. |
| `lumo_client.py` | Owns **one** persistent Playwright browser context. `ensure_ready()`, `ask(prompt) -> str`. Opens a fresh Lumo chat per request, submits the prompt, waits for the "generation complete" signal, scrapes the final assistant turn. |
| `login.py` | `python -m lumo_proxy.login`: launches a **headed** browser, performs/awaits login (human completes any captcha/2FA), writes `storageState.json` to the data dir. |
| `selectors.py` | All Lumo DOM selectors + the "generation finished" signal, in one place. This is the brittle surface (Lumo's UI will change); isolating it keeps churn contained. |
| `config.py` | Env-driven config: `LUMO_DATA_DIR`, `LUMO_USERNAME`, `LUMO_PASSWORD`, `LUMO_PORT`, `LUMO_HEADLESS`, `LUMO_REQUEST_TIMEOUT_S`, `LUMO_INTER_REQUEST_DELAY_S`. |
| `Dockerfile` | Base `mcr.microsoft.com/playwright/python`. |
| `docker-compose.yml` | Service with volume `lumo-data:/data` (holds `storageState.json`) and `restart: unless-stopped`. |
| `README.md` | Setup, one-time login, running, troubleshooting. |

Outside the directory:

- `providers/lumo.yaml` — `openai:chat:lumo` pointing at `http://127.0.0.1:<port>/v1`,
  carrying **`maxConcurrency: 1`** (see Concurrency).
- A short pointer added to `docs/README.md`.
- New entries in `.env.example` for the Lumo env vars (credentials live only in
  the user's local `.env`, never committed).

## Request Flow

```
promptfoo
  → POST /v1/chat/completions   (OpenAI body)
    → server takes the conversation's LAST user message (fresh-chat mode)
    → acquire asyncio lock        (serialize — one browser)
    → optional inter-request delay
    → lumo_client.ask():
        open a new Lumo chat
        type the prompt, submit
        wait for "generation complete" signal (server-side timeout)
        scrape final assistant message text
    → release lock
  ← OpenAI `chat.completion` JSON (single choice, role=assistant)
```

Unsupported OpenAI params (`temperature`, `max_tokens`, etc.) are accepted and
ignored — Lumo's UI does not expose them. The reported `model` is a fixed label
(e.g. `lumo`).

## Concurrency & Timeouts

Verified against the installed promptfoo (`node_modules/promptfoo/dist/src/fetch-*.js`):

- Per-request HTTP timeout is the env var **`REQUEST_TIMEOUT_MS`, default
  `600000` ms (10 minutes)**, overridable. There is also a whole-eval
  `PROMPTFOO_EVAL_TIMEOUT_MS`.
- Concurrency is `maxConcurrency` (this repo's config defaults to 4).

**Why `maxConcurrency: 1` rather than only a server-side lock.** promptfoo's
10-minute clock starts when *it sends* a request, not when *we begin processing*
it. If promptfoo fires several in parallel and we queue them behind a lock, the
request at the back of the queue waits for every request ahead of it *plus* its
own Lumo turn — all charged against its single 10-minute budget. With slow Lumo
turns (thinking + web search) a deep queue can blow the timeout for the requests
at the back.

The durable fix pushes the constraint upstream: setting `maxConcurrency: 1` for
this provider means promptfoo only sends the next request after the previous one
returns, so **no queue ever forms** and each request gets the full timeout budget
for just its own turn. The server-side `asyncio` lock stays purely as a safety
net for anyone who runs without the setting.

> Lock = correctness guarantee (never two turns at once). `maxConcurrency: 1` =
> the thing that actually keeps requests under the timeout.

**Server-side timeout.** The proxy enforces its own per-request timeout
(`LUMO_REQUEST_TIMEOUT_S`, default ~180–240 s) *below* promptfoo's 10-minute
fetch timeout, so a stuck generation returns a clean OpenAI-style error rather
than letting promptfoo's fetch hard-abort (which yields a muddier failure).

## Session Lifecycle

1. **One-time login** (`python -m lumo_proxy.login`): headed browser; human
   completes login + any captcha/2FA; Playwright `storageState` (cookies +
   localStorage) is written to `LUMO_DATA_DIR/storageState.json`.
2. **Serving**: on startup the proxy launches **headless**, loads
   `storageState.json` into the browser context, and keeps that context alive
   for the whole suite.
3. **`ensure_ready()`**: if `storageState.json` is absent, or Lumo presents a
   login screen, `/health` reports `logged_in: false` and chat requests return
   **HTTP 503** with a message to run the login command. The server performs **no
   silent auto-login** — the only path that touches credentials is the explicit
   one-time `login` step, keeping the abuse surface minimal.
4. **Persistence across restart**: the data dir is a Docker named volume, so the
   session survives container restarts; `restart: unless-stopped` keeps the proxy
   always-on.

## Failure Handling

- **Generation timeout / stuck UI** → server-side timeout fires → return an
  OpenAI-style error so promptfoo records a clean test failure.
- **Lost session mid-run** (login screen reappears) → `/health` flips to
  `logged_in: false`; chat requests return 503 telling the operator to re-run the
  login command.
- **Selector drift** (Lumo UI change) → scraping fails fast with a clear error
  pointing at `selectors.py`.

## Credentials & Security

- Credentials are read from env / `.env` only; **never hardcoded**.
- `.env.example` gains the Lumo keys (empty values).
- `storageState.json` (a live session) lives only in the Docker volume / local
  data dir and must be gitignored.
- This is for evaluating an account we control; the design deliberately keeps the
  login surface to a single explicit, human-supervised step.

## Known Risk / First Implementation Step

The live Lumo DOM has not been inspected yet, so exact selectors and the
"generation finished" signal are not yet known. The **first implementation step**
(when implementation is greenlit) is: launch a headed browser, log in, and record
the real selectors and completion signal into `selectors.py`. Everything else in
the design is independent of those specifics.

## Out of Scope

- Streaming responses.
- Multi-turn conversations (only the last user message is sent per request).
- Honouring `temperature` / `max_tokens` / other sampling params.
- Parallelism to Lumo.
- Automating captcha/2FA (handled by the one-time human login).

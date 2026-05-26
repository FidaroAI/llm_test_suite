# Stock-price freshness tests — design

**Date:** 2026-05-25
**Status:** approved

## Goal

Detect whether Fidaro is fetching **up-to-date market data**. Ask Fidaro for the
latest price of 20 named stocks across seven exchanges and pass only if its
answer is within **1%** of the live price (1% is the agreed margin for error —
it absorbs small intraday moves between the reference fetch and the answer).

> **Source pivot (2026-05-26):** the original plan used Yahoo Finance, but its
> chart endpoint hard-rate-limits (HTTP 429) unauthenticated clients from our
> environment — verified with `requests` and plain `curl`, even after a
> cookie/crumb handshake. We switched to **Stooq** (free, no key, reliable).
> Stooq has no Indian NSE coverage and NSE's own API geo-blocks us, so the **5
> Indian stocks were dropped** (25 → 20) per the user's decision. The
> architecture is unchanged; only `fetch_quote` and the symbol map differ.

Prompts take the form:

> What is the latest stock price for Toyota Motor Corporation (7203)?

The ticker is included in the prompt to disambiguate the listing.

## Why this is non-trivial

The hard part is the reference price, not the prompt. The stocks span the US,
London (quoted in **pence**, `GBp`), Hong Kong, Tokyo (yen), Amsterdam, Paris and
Frankfurt (euro). Each needs a different source symbol and the answer must be
compared in the right currency unit.

**Stooq** (`stooq.com`) is the source: free, no API key, and a one-line CSV per
symbol. We GET `https://stooq.com/q/l/?s={symbol}&f=sd2t2c&e=csv` (with a browser
`User-Agent`) and read the `Close` field. Stooq does **not** report the currency,
so each CSV row carries its quote currency (London = `GBp`). Symbols differ from
the common ticker (`arm.us`, `hsba.uk`, `700.hk` — no leading zeros, `7203.jp`,
`asml.nl`, `mc.fr`, `sap.de`) and are stored per row. All source access is
isolated in `fetch_quote` so it is easy to swap.

## Architecture

promptfoo loads **every** generator in `promptfooconfig.yaml` on each run,
*before* metadata filtering. So a generator that hit the price source and raised
on failure would break *every other suite's* run during an outage. We therefore keep
the generator and assertion **network-free** and put the live fetch + fail-fast
in a separate preflight step.

```
scripts_repo/fetch_stock_prices.py   preflight CLI: fetch all symbols, FAIL FAST,
  └─ tests/stock_prices_ref.py        write timestamped snapshot JSON
       (fetch_quote + snapshot IO)

tests/stock_prices.csv               source of truth: prompt + stooq_symbol + currency + company
tests/stock_prices_gen.py            reads CSV via csv_suite; bakes snapshot
                                     price/currency/fetched_at into each test's
                                     metadata; sets options.cache = False
assertions/assert_stock_price.py     pure-local: parse number(s) from Fidaro's
                                     answer, compare to baked reference within 1%

scripts_test/fidaro_stock_prices.sh  wrapper: preflight (fail fast) → promptfoo
scripts_test/fidaro_stock_prices_config.json   enables the suite for that run
```

### Data flow

1. Wrapper runs the preflight. It reads the symbols from the CSV, fetches each
   from Stooq, and **exits non-zero listing any symbol it could not fetch**
   ("query the source before running the tests; fail fast"). On success it writes
   `tests/stock_prices_reference.json` = `{fetched_at, quotes: {symbol:
   {price, currency, company}}}` (gitignored — transient, regenerated each run).
2. Wrapper runs promptfoo filtered to `suite=stock_prices` with `--no-cache`,
   using a suite-generation config that enables the suite.
3. The generator bakes each snapshot quote into the matching test's metadata
   (`reference_price`, `reference_currency`, `reference_fetched_at`) and mirrors
   them into `vars` for assertion robustness. No network.
4. For each test, Fidaro answers; the assertion parses numeric candidates from
   the answer and passes if any is within tolerance of the reference.

### Why baked snapshot, not grade-time fetch

20 symbols × providers, `cache:false`, concurrency 100 ⇒ dozens of simultaneous
source requests ⇒ likely rate-limiting and flaky results. Fetching once,
sequentially, up front avoids that and makes a run reproducible. The staleness
window is one test run (minutes) — well inside the 1% margin.

## The assertion

`get_assert(output, context)`:

* Read `reference_price` / `reference_currency` / `reference_fetched_at` /
  `stooq_symbol` from `context["test"]["metadata"]` (fall back to `vars`).
* **Missing reference** ⇒ fail with "no reference snapshot — run the preflight".
* **Stale reference** (older than `max_age_hours`, default 24h) ⇒ fail.
* Extract numeric candidates from the answer text (strip thousands separators).
* **Pass if any candidate is within `tolerance_pct` (default 1.0%)** of the
  reference. For London `GBp` listings also accept `reference / 100` so a
  pounds-denominated answer (£9.00 vs 900p) matches — the pence/pounds gotcha.
* Fail cleanly when there is no number (refusal / no answer), reporting the
  reference and the closest candidate with its % difference.

Config knobs (optional, via assertion `config`): `tolerance_pct`,
`max_age_hours`.

### Accepted trade-offs

* Unofficial Stooq endpoint, last-close (not tick) prices — isolated and swappable.
* "Any number within tolerance" is lenient, but a wrong number landing within
  1% of the exact live price by chance is very unlikely, and it is robust to
  phrasing.
* English number formatting assumed (`,` thousands, `.` decimal).
* Market moving >1% between fetch and answer can flake — that is the agreed
  margin.

## Symbol map

| Company | Prompt ticker | Stooq symbol | Currency |
|---|---|---|---|
| Arm Holdings | ARM | arm.us | USD |
| HSBC Holdings | HSBA | hsba.uk | GBp |
| AstraZeneca | AZN | azn.uk | GBp |
| Linde | LIN | lin.us | USD |
| Shell | SHEL | shel.uk | GBp |
| Tencent Holdings | 0700.HK | 700.hk | HKD |
| China Mobile | 0941.HK | 941.hk | HKD |
| HSBC Holdings | 0005.HK | 5.hk | HKD |
| China Construction Bank | 0939.HK | 939.hk | HKD |
| AIA Group | 1299.HK | 1299.hk | HKD |
| Toyota Motor Corporation | 7203 | 7203.jp | JPY |
| Mitsubishi UFJ Financial Group | 8306 | 8306.jp | JPY |
| SoftBank Group | 9984 | 9984.jp | JPY |
| Hitachi Ltd. | 6501 | 6501.jp | JPY |
| Sony Group Corporation | 6758 | 6758.jp | JPY |
| ASML Holding | ASML | asml.nl | EUR |
| LVMH Moët Hennessy Louis Vuitton | MC | mc.fr | EUR |
| TotalEnergies | TTE | tte.fr | EUR |
| SAP | SAP | sap.de | EUR |
| Siemens | SIE | sie.de | EUR |

Note the prompt still shows the listing's natural ticker (e.g. `0700.HK`,
`7203`) while the Stooq lookup symbol differs (`700.hk`, `7203.jp`).

### Dropped (no free source from our environment)

The 5 Indian NSE stocks — Reliance Industries (RELIANCE), HDFC Bank (HDFCBANK),
Bharti Airtel (BHARTIARTL), Tata Consultancy Services (TCS), ICICI Bank
(ICICIBANK) — are not on Stooq, NSE's own API geo-blocks us, and Twelve Data
needs a personal key. Add them later if a keyless/keyed source is wired in.

## Out of scope

* Indian NSE stocks (see above).
* Grade-time live pricing (rejected above).
* European decimal formatting in answers.
* Pre-populated classification labels (run `classify_tests.py` later if wanted;
  the suite is `factual_qa` / `finance_business`).

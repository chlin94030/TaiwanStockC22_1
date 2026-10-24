# Alpha Radar V22.1 — Fast Bootstrap Data Layer

## Why the old screen still said 7.6 hours
V22.0 inherited the V21.6 provider chain. If authenticated FinLab bulk data was unavailable, AUTO fell back to one FinMind historical request per stock. With about 1,986 stocks and the conservative anonymous pace of ~260 requests/hour, the UI correctly estimated about 7.6 hours. This is a transport bottleneck, not a V22 strategy-computation bottleneck.

V22.1 separates fast research from exact final confirmation and **does not silently fall back from AUTO_FAST to the multi-hour FinMind path**.

## Providers

### AUTO_FAST (new default)
1. If authenticated, sufficiently current FinLab whole-market data is available, use it.
2. Otherwise use YAHOO_FAST multi-ticker batches.
3. If Yahoo batching fails, stop with an actionable error. Do not silently start a 7-hour FinMind job.

### YAHOO_FAST
- Downloads many tickers in each yfinance batch (default 80 symbols per batch).
- Around 1,986 stocks therefore require about 25 application-level batches, not ~1,986 FinMind stock requests.
- Builds the universe from a **turnover proxy**:
  `((High + Low + Close) / 3) * Volume`.
- This is explicitly labeled `development_turnover_proxy=true` and is **not exchange-reported trading amount**.
- Suitable for fast V22 development, weight tuning, market-stage analysis, and candidate comparison.
- A final confirmation made with proxy turnover is labeled `PROXY_DATA_CONFIRMATION_NOT_FINAL`.

### OFFICIAL_BULK
- Free exact-history option using the TWSE whole-market daily table and TPEx whole-market daily table.
- TWSE and TPEx for the same date are requested in parallel and cached by date.
- Uses exchange-reported `成交金額` for the universe and raw OHLCV.
- With the default courtesy delay of 0.75 seconds between business-date cycles, a multi-year backfill is expected to be on the order of tens of minutes rather than 7+ hours, but network retries and cloud blocking can increase this substantially.
- Some hosting IPs can be blocked by official sites. In that case the app stops and preserves completed daily cache; it does not silently switch to FinMind.
- Adjusted OHLC for selected-union stocks still comes from Yahoo; raw turnover remains official.

### FINLAB_BULK
- Exact whole-market bulk source when server credentials and sufficient date coverage are available.
- Remains the preferred exact bulk transport when configured.

### FINMIND
- Explicit fallback only.
- Still useful for exact per-stock history, and V21.6 dynamic quota reading remains available.
- Do not choose this for a first-time 1,986-stock bootstrap on a free/anonymous quota unless you knowingly accept the long runtime.

## Cache/resume behavior
- Yahoo turnover batches are saved independently under `cache/yahoo_fast`.
- Official exact dates are saved independently under `cache/official_bulk/YYYY/MM`.
- Restarting the app reuses completed batches/dates.
- Selected-union price data is saved to the existing SQLite store.
- Existing `data/` should be preserved across upgrades.

## Research-integrity boundary
Fast transport is not allowed to masquerade as exact evidence:
- `YAHOO_FAST`: approximate turnover, development only.
- `OFFICIAL_BULK`, `FINLAB_BULK`, `FINMIND`: exchange-reported/actual turnover sources where provided by the source.
- The report records `amount_source`, `amount_is_exchange_reported`, and `development_turnover_proxy`.

A practical workflow is:
1. Use AUTO_FAST/YAHOO_FAST to build the first universe quickly and iterate V22 strategies.
2. Freeze the candidate pool/strategy hypothesis.
3. Run one exact universe confirmation with OFFICIAL_BULK or FINLAB_BULK (or FINMIND if necessary).
4. Do not repeatedly use the exact holdout to tune weights.

## Important deployment note
A currently running V22.0 FinMind job cannot change provider mid-process just because the code is later updated. Safely pause/stop that job, deploy V22.1, keep `data/`, then start/continue with `AUTO_FAST`.

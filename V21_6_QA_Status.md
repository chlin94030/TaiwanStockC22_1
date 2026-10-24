# Alpha Radar V21.6 QA

## Scope
V21.6 changes the transport/cache layer, not the registered universe rule or factor-model acceptance logic.

## Implemented
- `AUTO` provider: authenticated FinLab whole-market table first, FinMind fallback second.
- FinLab whole-market actual traded amount is vectorized into monthly ADV ranks; it does **not** first materialize all ~2,000 daily histories in SQLite.
- Only the historical union of preregistered pools plus 0050 receives full raw/adjusted daily price storage.
- FinLab adjusted OHLC uses the daily `etl:adj_close / raw close` adjustment ratio applied to raw OHLC; raw traded amount and share volume remain actual unadjusted quantities.
- Existing OHLC quarantine behavior is retained; no high/low/open is invented.
- FinMind authenticated fallback reads `/v2/user_info` and uses `api_request_limit * finmind_quota_utilization` (default 90%) when available; otherwise it falls back conservatively.
- AUTO rejects stale/incomplete FinLab coverage and falls back; forced FINLAB_BULK fails closed.
- GitHub Actions passes FinLab server credentials only through Secrets environment variables.

## Tests
- 46 `test_lab.py`: passed.
- 42 V21 debug/resume + V21.6 data-layer tests: passed.
- 81 V21.3/V21.4/V21.5 strategy/evidence tests: passed.
- Total targeted tests: **169 passed, 0 failed** across the above split runs.
- `python -m compileall`: passed before packaging.

## Scale diagnostics (synthetic engineering data, NOT performance)
- 1,985 stocks × 2,200 sessions whole-market amount matrix: vectorized monthly turnover checkpoint in about 2.1 s in this container; peak process RSS about 361 MB.
- 500 selected symbols × 1,450 sessions, seven synthetic wide price tables -> SQLite raw+adjusted selected-union storage: about 26.9 s; database about 139 MB; peak process RSS about 187 MB.
- These timings exclude real provider download, authentication, Streamlit/GitHub startup and remote throttling.

## Not tested here
- This container has no outbound pip/network access, so the FinLab wheel could not be installed and authenticated end-to-end here.
- Official PyPI metadata and FinLab documentation were checked externally: the selected package release exposes Python 3.14 wheels and server-side environment credential support.
- No real user FinLab entitlement was exercised; free/VIP data windows differ.
- No claim is made that data transport changes improve investment performance.

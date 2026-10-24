# V22.1 Fast Bootstrap QA

Status: engineering QA passed; live external-source execution on the user's Streamlit Cloud instance has not been performed by this environment.

## Automated regression
The following combined suite passed after the V22.1 transport changes:

- V22 stage-first tests
- V21.6 data-layer tests
- V21.5 fixed-pool tests
- V21.4 robust-consensus tests
- V21.3 evidence-guard tests
- resume/debug/base lab tests
- new V22.1 fast-bootstrap tests

Result: **189 passed / 0 failed**.

## New V22.1 coverage
Tests include:
- AUTO_FAST/YAHOO_FAST configuration validation.
- yfinance MultiIndex output in both common orientations.
- turnover-proxy formula and adjusted-OHLC ratio handling.
- Yahoo chunk checkpoint/resume without re-downloading completed chunks.
- selected-union raw/proxy and adjusted-price persistence.
- synthetic TWSE whole-market response parser.
- synthetic TPEx legacy whole-market response parser.
- official daily cache resume; second pass makes zero synthetic network calls for cached dates.
- AUTO_FAST does not silently switch to FinMind when the fast provider fails.

## Compilation
`python -m compileall -q .` passed.

## What has NOT been proven
- No claim that Yahoo/official endpoints will always be reachable from Streamlit Cloud IPs.
- No live network backfill was executed from this container because outbound network access is unavailable here.
- No guaranteed wall-clock time is asserted. Yahoo batch speed depends on Yahoo/yfinance/network behavior; official bulk depends on site availability and throttling.
- Yahoo turnover is a transparent proxy, not exchange-reported trading amount, and cannot by itself produce a final exact confirmation verdict.
- This release changes data transport, not V22's investment-performance evidence.

# Alpha Radar V22.1 — Kostolany Stage-First + Fast Bootstrap

V22.1 keeps the V22 market-stage/strategy model and fixes the first-time data bottleneck. `AUTO_FAST` is now the default: authenticated FinLab bulk when available, otherwise Yahoo multi-ticker batches for rapid development. It does **not** silently fall back to the multi-hour FinMind per-stock path. `OFFICIAL_BULK` is provided as a free exact-turnover confirmation path using TWSE+TPEx whole-market daily history. See `V22_1_START_HERE.txt` and `V22_1_FAST_BOOTSTRAP.md`.

# Alpha Radar V21.6 - Bulk-first Data Layer + Fixed Pool Strategy Studio

Use V21_6_Quickstart.txt for the V21.6 data-layer guide and V21_5_Quickstart.txt for the fixed-pool strategy workflow. app.py remains the Streamlit entry point.

Default page: fixed-pool, offline strategy development. Adopt an existing prepared run and explicitly freeze a pool rule. Do not re-download the whole market or rerun four universe sizes merely to change weights.

See FIXED_POOL_PROTOCOL.md for architecture, cache invalidation, fast-vs-confirmation boundaries, CLI and limitations. See KOSTOLANY_DESIGN.md for source provenance and the distinction between book ideas and our numerical hypotheses. OLD V21_4/21_3 QA documents are historical release notes, not evidence of this version's performance; the current record is QA_STATUS.md.

Quick CLI (after restoring your own data cache):
```
python run_studio.py --action adopt
python run_studio.py --action lock --pool ADV200
python run_studio.py --action fast
```

Important: a completed LAB_RESULTS.zip is a report, not the daily-price database. New local feature preparation needs the daily SQLite cache and prepared snapshot. No live profits or global optimum have been established.

## V21.6 bulk-first data layer
V21.6 preserves the V21.5 fixed-pool strategy studio but replaces the slow first-time all-market transport. `history_provider=AUTO` prefers authenticated FinLab whole-market tables; if unavailable/incomplete it falls back to FinMind. FinMind fallback now reads the account's reported API quota instead of treating every token as 520 requests/hour. See `V21_6_Quickstart.txt`, `V21_6_Data_Architecture.md`, and `V21_6_QA_Status.md`.

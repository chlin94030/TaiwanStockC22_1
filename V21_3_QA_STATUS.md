# V21.3 engineering QA

Source: V21.2 full source as provided previously.
Uploaded evidence: LAB_RESULTS.zip (18 members), manifest SHA256 verified.

- 96 automated tests passed (existing 84 + 12 new), zero failures.
- Read-only audit of original real LAB_RESULTS.zip reproduced all three old
  final-tail underperformances and rejected all three candidates using a
  pre-final training/outer-only promotion rule. This audit computes no new
  realized strategy returns.
- A 115-equity, 1350-session synthetic complete pipeline ran in 9 seconds on
  cached local disk after preparing its input. It exported PRETAIL_GATES.json
  and feature_coverage_by_date.csv; report ZIP passed CRC validation.
- A heavier 520-equity, 2200-session synthetic run was attempted but did not
  finish during the available execution limit; it reached feature generation.
  It is NOT counted as a successful scale run.
- Python compilation passed. Package ZIP count, CRC and patch-overlay check
  are validated separately for the release.

Not tested: a real fresh full-market-price backtest under the new selection
logic, hosted Streamlit browser, GitHub Actions deployment in the user's
account, fully reconstructed historical membership, PIT fundamentals or any
future unseen market performance. The old 2024-2026 tail is already exposed.
No claim of improved investing performance is made.

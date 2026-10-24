# V22 protocol overlay

V22沿用本文件既有的資料誠實、時間切分與保留期原則，但策略層已改為 `V22_MODEL_DESIGN.md` 的六階段先決架構。日常調權重使用 fast development，不重開母池與最終保留期。

# Advanced legacy universe-search operation only

V21.5 defaults to the fixed-pool Studio. See FIXED_POOL_PROTOCOL.md for that workflow. The following specification remains available only when universe research is explicitly reopened.

# V21.4 registered research protocol

## Basis and scope

This new hypothesis was designed after observing the supplied V21.3 report.
Its pre-tail guard did not establish positive real-world performance. The final
historical era has already been exposed. This release does not repair that fact.
The submitted latest attachment is a report summary, not the new raw database
or latest full result JSON; it cannot support a real V21.4 performance replay.

## Frozen choices

Same monthly trailing-252 ACTUAL amount pools ADV100/200/300/500. Same global
same-date factor-rank reference. Three holding horizons, 5/20/60 sessions,
next-open entry assumptions, existing gap cap and cost conventions. Default
portfolio is ten equal funded slots. No setup/capacity -> known cash slot;
unknown selected entry/exit -> unknown outcome, never a benchmark replacement.

The registry remains bounded to 64 static profiles by default, each with at
most four non-negative factors. Eighteen available price/volume factors are
registered; missing optional PIT fundamentals/flows remain disabled.

New residual features use log price returns, stock-specific beta from the
trailing 120 sessions with at least 108 paired observed returns, shifted one
session. A residual on day d uses beta through d-1. residual_momentum20 and
residual_momentum60 sum 20/60 fully observed residuals; residual_lowvol60 is
negative residual volatility over 60 fully observed sessions. Market variance
near zero produces an unknown factor, not invented beta. amount_acceleration
is log(actual mean amount20 / actual mean amount120). No future labels enter.

No price input is interpolated. Full-252 high remains strict. A valid close may
be used for an indicator despite a separate invalid open; the executable open
remains unavailable. Recovery from recorded original HLC in a quarantine ledger
is analytical only; no raw trade field is silently overwritten.

## Robust selection at each freeze

1. Include only training labels whose exits mature before boundary minus the
   existing 5-session gap. Outer and final-tail outcomes cannot choose weights.
2. Require registered outcome coverage on all training and its most recent
   min(24, available training months) observations. Keep unknowns unknown.
3. Calculate actual net excess minus 50bp extra round-trip cost times actual
   filled STOCK capital fraction. Missing-slot -100% sensitivity remains
   separate from observed return, as in the previous model.
4. Compute three chronological training block means. The score is
   min(full stress mean, recent stress mean) - 0.5*block dispersion
   - 0.0005*(number of active factors - 1).
   Full/recent stress must be positive and at least 2 of 3 blocks positive.
5. If no candidate has a positive eligible score, choose the benchmark reference
   as an abstention/reference outcome, not a profitable stock-selection result.
6. Predeclared near-best tolerance: 0.001. Favor fewer factors, then robust score.
7. Within the chosen pool, at most 3 eligible near-best profiles with training
   return correlation below 0.90 can propose one averaged-factor consensus.
   Reduce to the 4 largest weights, normalize, record the dropped mass, and
   RECOMPUTE portfolio picks/returns on training cases. Adopt only if the actual
   training robustness score is no worse than the anchor. Maximum one extra
   proposal per freeze, never a search on outer or tail results.

The 24-month training minimum, 6-month outer folds, 24-month tail, purging,
block/nonoverlap uncertainty and original promotion requirements are retained.
The outer OOS series tests the model-selection algorithm, not a specific final
weight vector. A positive algorithm guard never proves the final vector.

## Costs and reporting

0050 reference net returns retain the same ETF fees/tax convention as the
benchmark. Additional individual-stock stress is ZERO for that reference.
The zero reference excess and confidence interval are algebraic identities,
not statistical evidence. Part-stock rows pay stock stress in proportion to
actual filled stock capital. Costs are not removed to improve reported alpha.

Pool equivalence is labeled on training outcomes; names are not declared uniquely
best when their economic records are equivalent. Missing same-pool outcomes
remain missing. Report concentration and data coverage, not just a winner.

## Finite operation

The run id hashes all production Python code, config and input identity. Preserve
canonical cache on upgrade, never reuse old-version model/feature outputs as
new results. Manual launch runs a finite research job; no automatic extra tests
follow bad performance. Background tasks checkpoint on the existing schedule.
New code requires a new manual start; old interrupted code does not auto-migrate.

The old observed performance is not the new model's performance. No live trading
or automatic model adoption. True historical-membership data, intraday fills,
portfolio risk under continuous capital accounting, and future profitability
remain unverified. This protocol is a new exposed-history development study.

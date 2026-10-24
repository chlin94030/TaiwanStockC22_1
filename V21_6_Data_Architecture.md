# V21.6 data architecture

## Fast path
Official current roster -> FinLab whole-market actual amount table -> vectorized trailing-252 ADV ranks -> monthly ADV100/200/300/500 -> union of historical pool members -> full OHLC/volume/amount/adjusted-close only for that union -> immutable feature/label snapshot -> fixed-pool studio.

The whole-market stage needs only actual traded amount for ranking. Activity uses `actual amount > 0` when amount is known, which is equivalent to a traded-volume positive/zero indicator for valid daily exchange records. Missing amount is not changed to zero.

## FinMind fallback
If FinLab is not configured, AUTO uses FinMind. Authenticated plans query the official user-info endpoint for the account request limit and throttle at a configurable fraction (90% by default). A quota lookup failure does not reveal credentials and falls back to the prior conservative safe rate.

## Data integrity
- A bulk table must begin near the requested start and extend near the requested end. Stale free-history is not silently accepted as current data.
- Forced FINLAB_BULK fails closed on missing credentials/entitlement/coverage.
- AUTO records why it fell back.
- Bulk price retrieval can fall back only for the selected historical pool union, rather than re-downloading the whole market per symbol.
- Full-market turnover and selected-union price preparation do not alter the universe definition, factor formula, holdout policy or acceptance rules.

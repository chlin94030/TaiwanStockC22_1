# V21.5 Fixed Pool Studio: independent research stages

## Purpose and boundary

The new default is fixed-pool strategy development, not a rerun of four universe sizes. A universe rule is explicitly registered once, independently of strategy weights. A prior training-selected pool can be used as supporting information, but the available ADV100/200/300/500 evidence has not established a unique optimum. A manual ADV200 lock is a provisional research scope, not a financial recommendation or a passed model.

Freeze the rule (for example, trailing 252-market-session average actual traded value Top 200). Keep each historical month's original point-in-time membership. Never apply today's Top 200 list to earlier years. The inherited current-company-roster limitation still applies: a current snapshot is not a complete historical listing/delisting universe.

## Three workspaces

1. Data preparation/backup: explicit action only. Reads or extends the saved price cache, prepares monthly memberships/features/labels, then stops. It does not fit strategies. The supplied config.json defaults research_action to prepare_data. An already registered background job retains its saved configuration when resumed; pause it safely before upgrading. Initial full-market preparation can still be expensive.
2. Universe research: the existing four-pool bounded search remains an explicitly selected advanced operation. It is not invoked by the Studio. It can supply training-selected pool suggestions, not a guaranteed optimum. Reopen it only after a deliberate universe-methodology revision, not an unfavorable strategy result.
3. Fixed Pool Studio: adopts a completed prepared snapshot, registers one pool per horizon (or the same pool for all three), and only studies strategies within that rule. Its execution has no HTTP calls and no universe search calls.

## Immutable dataset and pool lock

An adopted snapshot copies five prepared artifacts: monthly features, labels, monthly pools, calendar and source scope. Their checksums and original source/cost configuration are retained. New behavior features and changed-cost labels need the local daily SQLite database. A report-only LAB_RESULTS.zip is insufficient; restore your own V21_DATABASE.zip if the daily cache is absent.

The raw cache is not overwritten by the Studio. A changed daily-source signature prevents recomputing against a different vintage without explicit re-preparation and registration. Existing immutable features and labels may still be read as the old research snapshot. Changing dataset or pool is an explicit revision, not an implicit side effect of changing weights.

## Independent caches

- Dataset/pool: hash of prepared data, data definitions and locked monthly membership.
- Feature cache: dataset + locked pool union + factor code. Calculate the 18 existing factors and six new book-inspired proxies once from the saved daily inputs for the historical union of selected members. Percentile ranks use the same fixed monthly union across all strategies.
- Label cache: dataset + costs/taxes/slippage/gap rule + label code. Weight edits do not regenerate labels. Cost changes replay local daily records only, without downloading or searching pools.
- Strategy cache: one candidate's actual weights/policy + funded slots/capital/capacity/coverage + data hashes + evaluation dates + evaluator code. Editing one candidate retains results for unchanged candidates.
- Experiment: complete submitted candidate definitions, runtime settings, code hash, dataset, lock and mode. An identical completed experiment returns existing outputs without re-execution. Draft parameters are saved for reconnects; partial per-candidate results survive pause/continue when files remain.

Cache integrity is checked with SHA256. No pickle is deserialized. Existing portable backups now include allowlisted Studio registrations, caches, experiments and confirmation ledgers. Cross-host restoration does not replace an existing local confirmation release with a conflicting older one. Only restore trusted personal backups. Host deletion, storage quotas and ephemeral Cloud disks remain external limitations.

## Fast development

The default registry contains 10 explicit hypotheses, not a blind rerun of all 64 old configurations. It is editable in the Streamlit form or studio_profiles.json. Support 1..128 unique candidates, at most four positive-weight factors per candidate, weights summing to 1. Controls must remain registered before comparison. There is no silent weight normalization or automatic infinite search.

Each horizon uses the fixed pool, inherited chronological design, label-maturity purge and five-session embargo. The final inherited tail (normally 24 monthly cutoffs) is excluded from candidate evaluation and reporting in fast mode. Stored labels for that period are not used to select weights. The fast default evaluates the last three chronological development folds and uses 200 descriptive bootstrap draws. These repeated development diagnostics are NOT fresh independent OOS evidence, especially after a pool was selected from the historical development period.

The training selector evaluates full and recent cost-stressed excess, block stability and complexity using training indices only. It can select the 0050 reference instead of pretending a stock strategy is supported. A reference has zero self-relative excess by definition, not earned alpha.

profile_comparison_5/20/60.csv compares every registered candidate on identical development-validation dates. development_regimes.csv groups those exposed results by the pre-decision regime proxy. Neither table is a tail-winner leaderboard or formal multiplicity-adjusted significance test. Use these for hypothesis development and record every attempted experiment in experiment_ledger.json.

## Confirmation

A separate acknowledged button runs all available development folds, writes the selected weights to FROZEN_STRATEGIES.json, then reserves the tail release in confirmation_ledger.json before evaluating that tail. It reports only the training-selected strategy on the tail. A different model/pool/dataset cannot release the same or overlapping holding-window tail again. An interrupted identical experiment can resume. A genuinely later, separately registered period can be evaluated; changing a seed or advancing one cutoff does not make overlapping market outcomes new evidence.

The default final bootstrap budget is 2000. Existing historical exposure and current-roster bias remain stated. Any favorable result is historical research support at most, not an optimal strategy, production trading permission, or proof of expected future profits. Multiple manual development iterations consume evidence; no cache or code guard can undo human exposure to past outcomes.

## Policies and accounting

All weights rank stocks with date-T inputs. Lock top K before filters; rejected positions remain cash and are not refilled from lower-ranked names. 10..30 funded equal slots cap each planned initial single-name allocation at 10% or less; no leverage or short positions are implemented. This is not a continuously enforced daily portfolio-weight limit.

Policies:
- none: baseline stock-ranking control, no sector/overheat overlay.
- discipline: no RSI>82 or MA20 bias>12% initial entries; maximum floor(30% x K) initial slots per available sector label.
- recovery_only: discipline plus a prior drawdown and observed stabilization condition.
- risk_half: discipline plus 50% of planned stock exposure in the PANIC_UNCONFIRMED or EUPHORIA_DIVERGENCE proxy state. This is a tested hypothesis, not an instruction to reduce real holdings. Unknown market context blocks this policy rather than silently treating it as favorable.

Sector labels can be current, not historically verified; the cap is diagnostic. Missing selected future returns stay unknown, never a retroactive ETF trade. Known no-fill cash remains explicit. Raw returns, full-0050 excess, same-pool excess, and exposure-matched excess are distinct. Costs apply to actual stock exposure for the extra-cost stress. No daily capital-constrained NAV, order book, stop execution, limit-up fills, macro release model, or actual investor sentiment data is claimed.

## CLI

From the app directory:

```
python run_studio.py --action adopt
python run_studio.py --action lock --pool ADV200
python run_studio.py --action fast
```

The CLI --action confirm explicitly consumes the tail release; do not call it for every weight edit. Advanced per-horizon mappings are available in the UI. Data collection remains run_background.py/GitHub Actions; its new default is preparation, while the routine strategy loop uses run_studio.py or the Studio page.

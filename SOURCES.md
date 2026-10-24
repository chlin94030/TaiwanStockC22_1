# Primary references and data attribution

Consulted 2026-10-21. The sources below support API/method descriptions, NOT the profitability of the authored strategy. The protocol itself is a new design, not a published empirical result.

- FinMind quickstart and quotas: https://finmind.github.io/quickstart/
- FinMind technical data, TaiwanStockPrice, Trading_money / Trading_Volume and zero-price caveats: https://finmind.github.io/tutor/TaiwanMarket/Technical/
- FinMind API authentication / entitlement summary: https://github.com/FinMind/FinMind.github.io/blob/master/llms.txt
- Official TWSE current company master: https://openapi.twse.com.tw/v1/opendata/t187ap03_L
- Official TPEx current company master: https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O
- Yahoo/yfinance history method: https://ranaroussi.github.io/yfinance/reference/api/yfinance.Ticker.history.html
- Nested model selection versus evaluation: https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- Time-series chronological split and gap: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- Streamlit worker / execution context limitations: https://docs.streamlit.io/develop/concepts/design/multithreading

Actual amount data attribution: FinMind TaiwanStockPrice Trading_money. Company-master attribution: TWSE / TPEx. Adjusted history attribution: Yahoo Finance via yfinance. These source combinations have not undergone a live-provider end-to-end audit in this delivery environment.

Use data only under its source license/entitlement. Do not redistribute raw provider histories, expose a mirror API, commit secrets or assume a public hosted trading product is authorized by a personal-research dataset license. The delivered report ZIP omits large raw histories; the repository `.gitignore` excludes the private data directory. Sources may change, impose quotas, be unavailable or require entitlements. Failure is reported; no access control is bypassed.


## V21.2 source references (checked 2026-10-22)
- FinMind TaiwanStockPrice field definitions, including emerging-market historical open caveat: https://finmind.github.io/tutor/TaiwanMarket/Technical/
- Streamlit threading lifecycle: https://docs.streamlit.io/develop/concepts/design/multithreading
- Streamlit generated-file persistence caveat: https://docs.streamlit.io/develop/concepts/configuration/serving-static-files
- GitHub Actions job limits and storage/minutes: https://docs.github.com/en/actions/reference/limits
- GitHub artifact retention/download: https://docs.github.com/en/actions/how-tos/manage-workflow-runs/download-workflow-artifacts
- GitHub Actions workflow dispatch: https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow

The workflow uses server-side jobs and finite-retention artifacts. This does not bypass API rate limits.

V21.4 external method checks (accessed during this revision):
- https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html
- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.TimeSeriesSplit.html
- https://finmind.github.io/tutor/TaiwanMarket/Technical/
These support the separation of model selection/evaluation and field semantics;
they do not endorse the new weights or demonstrate Taiwan equity performance.


V21.5 source boundary and publisher links: see KOSTOLANY_DESIGN.md. New factors and thresholds are our research hypotheses, not book formulas.

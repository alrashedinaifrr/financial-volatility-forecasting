# Cross-Market Volatility Forecasting | S&P 500 and Saudi TASI

An applied financial data science portfolio project comparing one-day-ahead squared-return forecasts using EWMA, rolling variance, Gaussian GARCH(1,1), Random Forest and gradient boosting across two equity indices. This is a methodological comparison, **not an investment strategy or a claim of forecast superiority**.

## Reproduce

Use Python 3.10+ and run from the repository root:

```bash
pip install -r requirements.txt
pytest -q tests
python src/portfolio_v7.py
```

The project expects `data/sp500.csv` and `data/tasi.csv`, each containing `Date,Close`, sorted internally. Raw market data are intentionally excluded from the public package because redistribution rights have not been established. The **private** package includes the original inputs; do not post it publicly. Reproduction of exact numbers requires matching source prices and dependency versions.

## What is evaluated

Chronological expanding-window evaluation; one-session-ahead squared log returns; QLIKE and MSE; exploratory moving-block bootstrap; descriptive cross-market transfer. Market returns are not directly comparable to actual latent volatility. Cross-market results are non-causal.

## Validation status

- GARCH recursion and temporal boundary tests: automated.
- External `arch` estimation check: **executed by the project owner in Google Colab**, with successful optimizer convergence on both indices; see `EXTERNAL_VALIDATION.md` and `EXTERNAL_ARCH_RESULTS.csv`. This environment did not independently rerun `arch`.
- Cross-library one-step forecast equivalence: **not established**.
- The available backtest is not a trading strategy, and exploratory intervals are unadjusted for multiple comparisons.

## Key files

`src/` contains reproducible code; `tests/` contains tests; `results_final/` contains outputs from the latest local rerun; `TECHNICAL_REPORT.md` summarizes methods and limitations.

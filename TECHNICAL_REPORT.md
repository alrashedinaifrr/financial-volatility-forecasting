# Technical report — Cross-Market Volatility Forecasting

## Research question

How stable are simple statistical and machine-learning variance forecasts across expanding time windows and between the S&P 500 and Saudi TASI?

## Design

Closing prices are converted to log returns. The label at each forecast origin is the **next trading session's squared log return**, a noisy realized variance proxy. EWMA, rolling variance, Gaussian GARCH(1,1), Random Forest and gradient boosting are evaluated using expanding chronological folds and QLIKE/MSE. Training excludes labels not known at the cutoff. GARCH is estimated at each fold boundary and updated with observed returns only. Cross-market transfer uses the same target test dates for source-trained and target-trained Random Forest models, but training sample sizes and return distributions differ.

## External numerical validation

The project owner ran `verify_with_arch.py` in Google Colab and supplied a screenshot of the result. The external optimizer converged on both markets. Under the **custom filter's initialization**, objective differences between the external `arch` parameters and custom parameters were approximately 1.28e-7 (S&P 500) and 1.19e-4 (TASI). The latter market has non-negligible parameter differences, despite a small objective gap. This cross-check supports numerical plausibility of the custom estimator but is not an exact match of filtering conventions, forecasts, or native log-likelihoods. See `EXTERNAL_VALIDATION.md`.

## Interpretation and limitations

No single method should be described as a universal winner on the basis of these exploratory folds. Bootstrap confidence intervals are exploratory and not adjusted for multiple comparisons. Daily squared returns are noisy volatility proxies; different trading calendars, market regimes and vendor data quality limit transferability. Neither the data nor the model support claims of trading profitability. The price series in the private bundle must not be redistributed without verifying rights.

## Reproducibility

Install requirements, run `pytest -q tests`, and run `python src/portfolio_v7.py` from the project root. The last command creates `results_v9/` (a historical script output name); the V15 packaging script stores its fresh run in `results_final/`. No claim is made that Colab and this local environment used identical package versions.

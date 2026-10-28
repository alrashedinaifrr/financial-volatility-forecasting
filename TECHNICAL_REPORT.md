# Technical report — Cross-Market Volatility Forecasting

## Research question

How stable are simple statistical and machine-learning variance forecasts across expanding time windows and between the S&P 500 and Saudi TASI?

## Design

Closing prices are converted to log returns. The label at each forecast origin is the **next trading session's squared log return**, a noisy realized variance proxy. EWMA, rolling variance, Gaussian GARCH(1,1), Random Forest and gradient boosting are evaluated using expanding chronological folds and QLIKE/MSE. Training excludes labels not known at the cutoff. GARCH is estimated at each fold boundary and updated with observed returns only. Cross-market transfer uses the same target test dates for source-trained and target-trained Random Forest models, but training sample sizes and return distributions differ.

## External numerical validation

The project owner ran `verify_with_arch.py` in Google Colab and supplied a screenshot of the result. The external optimizer converged on both markets. Under the **custom filter's initialization**, objective differences between the external `arch` parameters and custom parameters were approximately 1.28e-7 (S&P 500) and 1.19e-4 (TASI). The latter market has non-negligible parameter differences, despite a small objective gap. This cross-check supports numerical plausibility of the custom estimator but is not an exact match of filtering conventions, forecasts, or native log-likelihoods. See `EXTERNAL_VALIDATION.md`.

## Interpretation and limitations

No single method should be described as a universal winner on the basis of these exploratory folds. The runner now compares all ten model pairs within each of the six market/fold evaluation blocks using paired circular moving-block bootstrap samples (4,000 replicates; block length 10). It reports pointwise percentile intervals and Holm-adjusted bootstrap p-values across all 60 contrasts. In the recorded predictions, the adjusted comparisons identify higher QLIKE for `HGB_log` versus several alternatives in S&P 500 folds 1–2 and TASI fold 1; most other contrasts remain inconclusive. This finding applies only to these particular held-out windows. The p-value procedure and intervals remain exploratory, and there are only three windows per market.

Daily squared returns are noisy volatility proxies; different trading calendars, market regimes and vendor data quality limit transferability. Neither the data nor the model support claims of trading profitability. The price series in the private bundle must not be redistributed without verifying rights.

## Reproducibility

Install requirements, run `python -m pytest -q tests`, and run `python src/portfolio_v7.py --data-dir data --output-dir results_final` from the project root. The runner writes scores, row-level predictions, cross-market comparisons, pairwise QLIKE contrasts, bootstrap summaries, figures, and `run_metadata.json`. The metadata records input SHA-256 hashes, row counts, fold settings, and package versions to make each local run identifiable. Raw price data remain excluded, so exact reproduction still requires obtaining the same source series.


Loading older messages…






































































Cross-Market Volatility Forecasting
S&P 500 vs. Saudi TASI | Statistical Modeling & Machine Learning
An applied, reproducible financial data science project examining one-trading-session-ahead squared log-return forecasts for the S&P 500 and Saudi TASI indices. It compares conventional volatility estimators with machine-learning models under chronological out-of-sample evaluation.

Scope: This is a methodological forecasting comparison, not a trading strategy, a forecast of directly observed latent volatility, or evidence that any model is universally superior.

At a glance
Component	Implementation
Markets	S&P 500 (US) and TASI (Saudi Arabia)
Forecast target	Next-session squared log return (a noisy volatility proxy)
Models	20-session rolling variance (RV20), EWMA, Gaussian GARCH(1,1), Random Forest, histogram gradient boosting
Validation	Chronological expanding-window evaluation; three out-of-sample folds per market
Metrics	QLIKE (primary; lower is better) and MSE
Additional analyses	Moving-block bootstrap (exploratory), sensitivity checks, descriptive cross-market transfer
Tools	Python, pandas, NumPy, SciPy, scikit-learn, pytest; external arch cross-check
Results: out-of-sample QLIKE
The table reports the lowest QLIKE in each fold, selected descriptively from the evaluated models. Lower scores indicate better performance on this metric; this table does not establish statistical significance or universal superiority.

Market	Fold	Best model	QLIKE
S&P 500	1	EWMA	-8.6829
S&P 500	2	GARCH(1,1)	-8.6069
S&P 500	3	Random Forest	-8.3458
Saudi TASI	1	GARCH(1,1)	-8.8989
Saudi TASI	2	EWMA	-8.9573
Saudi TASI	3	GARCH(1,1)	-8.4717
Full fold-level QLIKE and MSE values are available in `results_final/fold_scores.csv`. Scores across different markets or folds should not be treated as directly comparable measures of market predictability.

S&P 500: model comparison by fold
S&P 500 fold-level QLIKE comparison

Saudi TASI: model comparison by fold
Saudi TASI fold-level QLIKE comparison

Key findings
Model rankings are regime- and sample-dependent. EWMA, GARCH, and Random Forest each lead at least one evaluated fold; no single model dominates across both markets.
Simple statistical baselines remain competitive. Greater model complexity does not guarantee lower out-of-sample QLIKE.
Cross-market transfer is exploratory. Training a model on one market and evaluating it on the other is a descriptive robustness exercise, not evidence of causal information transfer. See `results_final/aligned_transfer.csv`.
Uncertainty matters. Moving-block bootstrap comparisons are exploratory and have not been adjusted for multiple comparisons. See `results_final/rf_vs_ewma_bootstrap.csv`.
Model validation and audit
Automated tests cover the GARCH recursion, temporal boundary handling, prevention of future-target leakage, and other implementation checks; see `tests/` and `VALIDATION_STATUS.md`.
The project owner executed the external arch parameter-estimation cross-check in Google Colab for both markets; both external optimizers reported convergence. The reported coefficients and comparison details are in `EXTERNAL_ARCH_RESULTS.csv` and `EXTERNAL_VALIDATION.md`.
The arch comparison is not proof of identical one-step forecasts or exact equivalence between estimation implementations.
Repository structure
.
├── README.md
├── requirements.txt
├── src/                      # Data preparation, modeling and validation
├── tests/                    # Automated tests
├── results_final/            # Fold scores, predictions, figures and diagnostics
├── TECHNICAL_REPORT.md       # Methods, results and limitations
├── EXTERNAL_VALIDATION.md    # Third-party GARCH estimation check
├── EXTERNAL_ARCH_RESULTS.csv # Recorded comparison results
└── VALIDATION_STATUS.md      # Validation scope and status
Reproduce
Use Python 3.10+ from the repository root:

python -m pip install -r requirements.txt
python -m pytest -q tests
python src/portfolio_v7.py
Data requirement: Provide data/sp500.csv and data/tasi.csv, each with Date and Close columns. Source prices are not included in this public repository because redistribution rights have not been established. Exact replication of the reported values requires the same input prices and compatible dependency versions. The repository therefore contains code, tests and recorded outputs, but is not fully self-contained for exact numerical reproduction.

To run the optional external GARCH estimator check in a connected environment:

python -m pip install arch
python src/verify_with_arch.py
Limitations
Squared daily returns are noisy proxies for unobserved conditional variance.
Market holidays, different trading calendars and market structures complicate cross-market interpretation.
Model selection and sensitivity checks are exploratory; reported bootstrap intervals are not multiplicity-adjusted.
Results are historical and should not be interpreted as investment advice or out-of-sample trading profitability.
For additional methodology and limitations, read `TECHNICAL_REPORT.md`.



-0.45
⋮⋮⋮⋮

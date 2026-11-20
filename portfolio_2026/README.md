# 2026 doctoral research portfolio extension — cross-market volatility forecasting

This directory extends (and does not erase) the original five-model S&P 500/TASI benchmark. **The later forecast panel is an exploratory retrospective evaluation**, not a prespecified prospective test.

## Main additions

- HAR-SR2: heterogeneous autoregression on daily/5-day/22-day **squared returns**, not classical intraday realized-volatility HAR-RV.
- Aligned 2026 evaluation: 195 S&P 500 and 188 Saudi TASI sessions, one-session-ahead squared log-return targets.
- Nine-model comparison: EWMA, GARCH(1,1), RF, RV20, HAR-SR2, Poisson-deviance boosting, rescaled-log boosting, direct mean, and log-target boosting.
- 72 pairwise daily QLIKE comparisons with moving-block bootstrap confidence intervals, DM/HAC tests, and Holm multiplicity correction across both markets.
- Official R CRAN `MCS` verification of the nine-model panel: 24/24 completed scenarios, 4,999 bootstrap replicates each, both `Tmax` and `TR`, blocks of 5/10/20 sessions, both with and without Direct Mean. The custom Python `Tmax` membership matched official R in all 12 comparable conditions.

**Selected verified means**: S&P 500 lowest observed QLIKE is Poisson (about -8.5879); TASI lowest observed QLIKE is RF (about -8.8094). These rankings are **descriptive**. Extreme floor-hitting Direct Mean forecasts materially affect MCS model elimination in TASI. Retention in an MCS does not establish equal accuracy, and the choice of statistic and block length can alter set membership.

## Reuse results without redistributing vendor price exports

- `results/2026_nine_model_predictions.csv`: aligned per-origin model predictions and actual next-session squared returns, reconstructed for the nine forecasts.
- `results/2026_nine_model_all_scores.csv`, `results/2026_nine_model_all_72_pairwise_DM.csv`: model scores and paired inference.
- `results/official_R_nine_model/`: official CRAN MCS result tables and run/session logs supplied from an actual R run. The complete .rds archive is retained separately in the project-level source bundle; this GitHub subfolder publishes the audit tables rather than the R objects.
- `code/study2/`: actual fitting, HAR, expanded comparison, MCS R and verification scripts.

This public repository deliberately does **not** redistribute original third-party historical market-price exports. To refit, supply the same archived/complete `Date,Close` price histories and verify trading-date overlap. The derived prediction table lets researchers independently recompute QLIKE, DM and MCS without distributing those vendor source files. The rescaled Poisson formulation reproduces published results only when trained on the **original unscaled target** (not arbitrary x10,000 scaling of squared returns).

## Reproduction

Install `requirements.txt` for Python, and `install.packages("MCS")` under R for the official model-confidence-set calculation. From this directory:

```bash
# Independent official-R audit using the published nine-model predictions
Rscript code/study2/verify_MCS_nine_models_official.R results/2026_nine_model_predictions.csv results/new_R_MCS_audit

# Verified comparison of stored daily QLIKE losses with Python implementations
python code/study2/compare_official_R9_with_python.py --r-zip /path/to/Official_R_MCS_Nine_2026_Results.zip --project . --output results/recomputed_R_comparison

# To reconstruct forecasts, first obtain the original price sources in data/finance/complete/
python code/study2/run_2026_corrected.py --sp500-prices data/finance/complete/sp500.csv --tasi-prices data/finance/complete/tasi.csv --out-dir results/recomputed_six_model
python code/study2/extend_2026_benchmarks.py
python code/study2/expanded_2026_inference.py
PYTHONPATH=code/study2 pytest -q code/study2/test_study2_2026.py
```

The price-refit pipeline expects `results/later_2026_predictions_including_HAR.csv` after the six-model run; copy it from `results/recomputed_six_model/` when you are explicitly replacing the archived panel. Original market prices must agree with the input checks in the scripts. Some historical reproduction steps require additional archived prices in `code/study2/input/`.

The `docs/study2_body.tex` study text is a manuscript fragment, not a standalone TeX project. Historical development and later-period comparisons remain separate experiments.

**Research context**: [BRFSS diabetes portfolio extension](https://github.com/alrashedinaifrr/brfss-diabetes-model-evaluation/tree/main/portfolio_2026) (when merged).

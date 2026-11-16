# Reproduce the volatility rescaling and later evaluation

[volatility_retransformation.py](src/volatility_retransformation.py) reproduces the six-window comparison of four boosting specifications and the later comparison with EWMA. It estimates each multiplier from forward predictions within the corresponding training window.

## Inputs and environment

Use four daily-close CSV files with columns `Date,Close`. Format dates as `YYYY-MM-DD` and close values as numbers without thousands separators. Keep trading sessions and drop nontrading dates.

| Series | Archived closing-price input | Later closing-price input |
| --- | --- | --- |
| S&P 500 | 2,528 sessions from 4 January 2016 through 22 January 2026 | [FRED, SP500](https://fred.stlouisfed.org/series/SP500), 1 December 2025 through 30 October 2026 |
| TASI | 2,513 sessions from 3 January 2016 through 22 January 2026 | [Investing.com, Tadawul All Share](https://www.investing.com/indices/tasi-historical-data), daily frequency, 1 December 2025 through 30 October 2026 |

The archived inputs are the normalized closing-price files used in the study. Exact reproduction requires those snapshots or equivalent closes for all of their trading dates. The later input files include an overlap that the script checks before appending the new closes.

| Archived file | SHA-256 of the original normalized CSV |
| --- | --- |
| `sp500_normalized.csv` | `33924c990dcf95496df564de5636d9246b2a384b898ea3380785fc06453ac6ee` |
| `tasi_normalized.csv` | `d7cdfb6687266aedc2ad6d4839163e8575402b03d379ea66850265a9f41547ee` |

Run from the root of this repository revision with Python 3.12.

```bash
python -m pip install numpy==2.3.5 pandas==2.2.3 scipy==1.17.0 scikit-learn==1.8.0 joblib==1.5.3 threadpoolctl==3.6.0
python src/volatility_retransformation.py \
  --archive-sp500 /path/to/sp500_normalized.csv \
  --archive-tasi /path/to/tasi_normalized.csv \
  --later-sp500 /path/to/sp500_later.csv \
  --later-tasi /path/to/tasi_later.csv \
  --output-dir results/volatility_retransformation
```

With the two archive arguments, the script writes the historical comparisons. The later-series arguments add the corresponding fixed-specification tests. Supply `--reference-scores /path/to/objective_scores.csv` with the study's archived score file to check the 18 earlier boosting results.

## Check the output

The historical CSV includes 30 rows across six windows and five forecasts. The later CSVs contain five rows per market, with 195 S&P 500 sessions and 188 TASI sessions per forecast. The lowest later QLIKE is **−8.5879** for Poisson boosting on the S&P 500 and **−8.7774** for EWMA on TASI.

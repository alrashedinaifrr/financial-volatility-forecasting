"""Compare a training-only QLIKE rescaling with the archived boosting fits."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits


EPS = 1e-9
FEATURES = ["abs1", "abs2", "abs5", "rv5", "rv20", "rv60", "neg", "ewma", "logrv20"]
TRAIN_END = pd.Timestamp("2026-01-22")
EVALUATION_END = pd.Timestamp("2026-10-30")
ARCHIVE_SERIES = {"SP500": ("2016-01-04", 2528), "TASI": ("2016-01-03", 2513)}
LATER_SESSIONS = {
    "SP500": ("2026-01-23", "2026-10-30", 195),
    "TASI": ("2026-01-25", "2026-10-29", 188),
}
BOOSTING = dict(max_iter=130, max_leaf_nodes=9, l2_regularization=2,
                random_state=2026, early_stopping=False)


def read_prices(path):
    """Read normalized dates and daily index closes."""
    prices = pd.read_csv(path, usecols=["Date", "Close"], dtype={"Close": float})
    prices["Date"] = pd.to_datetime(prices["Date"], format="%Y-%m-%d")
    if prices.empty or prices["Date"].isna().any() or prices["Date"].duplicated().any():
        raise ValueError(f"Trading dates in {path} must be complete and unique")
    if not np.isfinite(prices.Close).all() or not prices.Close.gt(0).all():
        raise ValueError(f"Closing prices in {path} must be positive and finite")
    return prices[["Date", "Close"]].sort_values("Date", ignore_index=True)


def supervised(prices):
    returns = np.log(prices.Close).diff()
    squared = returns.pow(2)
    table = pd.DataFrame({"date": prices.Date, "target_date": prices.Date.shift(-1), "y": squared.shift(-1)})
    for lag in (0, 1, 4):
        table[f"abs{lag + 1}"] = returns.abs().shift(lag)
    for window in (5, 20, 60):
        table[f"rv{window}"] = squared.rolling(window).mean()
    table["neg"] = returns.lt(0).astype(float)
    table["ewma"] = squared.ewm(alpha=0.06, adjust=False).mean()
    table["logrv20"] = np.log(table.rv20.clip(lower=EPS))
    return table.dropna().reset_index(drop=True)


def fit_and_forecast(train, test, loss, *, log_target=False):
    y = train.y.to_numpy()
    if not train.target_date.lt(test.date.iloc[0]).all():
        raise ValueError("Training targets must precede the first forecast origin")
    if log_target:
        y = np.log(np.maximum(y, EPS))
    fit = HistGradientBoostingRegressor(loss=loss, **BOOSTING).fit(train[FEATURES], y)
    p = fit.predict(test[FEATURES])
    if log_target:
        p = np.exp(p)
    return np.maximum(p, EPS)


def cross_fitted_scale(train):
    """Estimate a common multiplier with training-only forward predictions."""
    a, b = int(0.60 * len(train)), int(0.80 * len(train))
    ratios = []
    for start, end in ((a, b), (b, len(train))):
        validation = train.iloc[start:end]
        p = fit_and_forecast(train.iloc[:start - 1], validation, "squared_error", log_target=True)
        ratios.append(validation.y.to_numpy() / p)
    combined = np.concatenate(ratios)
    scale = combined.mean()
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("The QLIKE rescaling needs a positive finite multiplier")
    return scale, len(combined)


def evaluate_block(market, fold, train, test):
    scale, calibration_n = cross_fitted_scale(train)
    raw_log = fit_and_forecast(train, test, "squared_error", log_target=True)
    forecasts = {
        "HGB_log": raw_log,
        "HGB_rescaled": np.maximum(scale * raw_log, EPS),
        "HGB_mean": fit_and_forecast(train, test, "squared_error"),
        "HGB_poisson": fit_and_forecast(train, test, "poisson"),
        "EWMA": test.ewma.clip(lower=EPS).to_numpy(),
    }
    y = test.y.to_numpy()
    rows = []
    for name, forecast in forecasts.items():
        qlike = np.log(forecast) + y / forecast
        rows.append({
            "market": market, "fold": fold, "model": name, "n": len(test),
            "QLIKE": qlike.mean(), "MSE": np.mean((y - forecast) ** 2),
            "mean_ratio": forecast.mean() / y.mean(),
            "first_origin": test.date.iloc[0].date().isoformat(),
            "last_origin": test.date.iloc[-1].date().isoformat(),
            "first_target": test.target_date.iloc[0].date().isoformat(),
            "last_target": test.target_date.iloc[-1].date().isoformat(),
            "train_n": len(train), "last_training_target": train.target_date.iloc[-1].date().isoformat(),
            "training_scale": scale, "calibration_n": calibration_n,
        })
    return rows


def read_archive(market, path):
    prices = read_prices(path)
    first_date, n = ARCHIVE_SERIES[market]
    if len(prices) != n or prices.Date.iloc[0] != pd.Timestamp(first_date) or prices.Date.iloc[-1] != TRAIN_END:
        raise ValueError(f"{market} archive must contain {n} sessions from {first_date} through {TRAIN_END.date()}")
    return prices


def historical_scores(archives):
    rows = []
    for market, prices in archives.items():
        table = supervised(prices)
        bounds = [int(len(table) * fraction) for fraction in (0.7, 0.8, 0.9, 1.0)]
        for fold, (start, end) in enumerate(zip(bounds[:-1], bounds[1:]), 1):
            rows.extend(evaluate_block(market, fold, table.iloc[:start - 1], table.iloc[start:end]))
    return pd.DataFrame(rows)


def check_archived_scores(scores, reference_path):
    originals = pd.read_csv(reference_path)
    keys = ["market", "fold", "model", "n", "first_origin", "last_origin"]
    matched = originals.merge(scores, on=keys, validate="one_to_one", suffixes=("_archived", "_rerun"))
    if len(matched) != 18 or len(originals) != 18:
        raise ValueError("The archived score file must match the 18 original boosting results")
    qlike_error = np.max(np.abs(matched.QLIKE_rerun - matched.QLIKE_archived))
    mse_error = np.max(np.abs(matched.MSE_rerun - matched.MSE_archived))
    print(f"Archived fit checks: max QLIKE error {qlike_error:.6g}; max MSE error {mse_error:.6g}.", flush=True)
    if qlike_error > 1e-6 or mse_error > 1e-12:
        raise ValueError("Archived baselines did not reproduce within numerical tolerance")


def later_scores(market, prices, later_path):
    later = read_prices(later_path)
    overlap = prices.merge(later, on="Date", how="inner", suffixes=("_archive", "_new"))
    max_difference = (overlap.Close_archive - overlap.Close_new).abs().max()
    if len(overlap) < 20 or TRAIN_END not in overlap.Date.values or max_difference > 0.011:
        raise ValueError(f"Failed close-price overlap check for {market}: {len(overlap)} days, max {max_difference}")
    new = later.loc[(TRAIN_END < later.Date) & (later.Date <= EVALUATION_END)]
    first_target, last_target, n = LATER_SESSIONS[market]
    if len(new) != n or new.Date.min() != pd.Timestamp(first_target) or new.Date.max() != pd.Timestamp(last_target):
        raise ValueError(f"{market} later data must include {n} target sessions from {first_target} to {last_target}")
    combined = pd.concat([prices, new], ignore_index=True)
    train = supervised(prices).iloc[:-1]
    test = supervised(combined).loc[lambda x: x.date.ge(TRAIN_END)]
    assert len(test) == len(new) and test.target_date.gt(TRAIN_END).all()
    print(f"{market}: matched {len(overlap)} overlap dates, max absolute close difference {max_difference:g}", flush=True)
    return pd.DataFrame(evaluate_block(market, "later", train, test)), {
        "matching_dates": len(overlap), "max_abs_close_diff": max_difference,
        "series_coverage": [later.Date.min().date().isoformat(), later.Date.max().date().isoformat()],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-sp500", required=True, type=Path, help="Archived S&P 500 Date,Close CSV")
    parser.add_argument("--archive-tasi", required=True, type=Path, help="Archived TASI Date,Close CSV")
    parser.add_argument("--later-sp500", type=Path, help="Later S&P 500 Date,Close CSV with a pre-cutoff overlap")
    parser.add_argument("--later-tasi", type=Path, help="Later TASI Date,Close CSV with a pre-cutoff overlap")
    parser.add_argument("--reference-scores", type=Path, help="Optional archived objective_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=Path("volatility_results"))
    args = parser.parse_args()
    print(f"scikit-learn {sklearn_version}", flush=True)
    with threadpool_limits(limits=2):
        archives = {
            "SP500": read_archive("SP500", args.archive_sp500),
            "TASI": read_archive("TASI", args.archive_tasi),
        }
        scores = historical_scores(archives)
        if args.reference_scores:
            check_archived_scores(scores, args.reference_scores)
        outputs = {"volatility_rescaling_historical.csv": scores}
        print(scores.pivot(index=["market", "fold"], columns="model", values="QLIKE").to_string())
        for market, path in (("SP500", args.later_sp500), ("TASI", args.later_tasi)):
            if path:
                result, audit = later_scores(market, archives[market], path)
                outputs[f"volatility_rescaling_later_{market.lower()}.csv"] = result
                print(json.dumps(audit, indent=2))
                print(result[["market", "model", "n", "QLIKE", "MSE", "mean_ratio", "first_target", "last_target", "training_scale"]].to_string(index=False))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, table in outputs.items():
        table.to_csv(args.output_dir / filename, index=False)


if __name__ == "__main__":
    main()

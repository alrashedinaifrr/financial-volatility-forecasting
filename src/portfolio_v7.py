"""Run the cross-market volatility forecasting analysis from the repo root."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from advanced_validation import fit_predict, losses, bootstrap_diff
from benchmark import prepare, FEATURES, EPS
from full_garch_validation import fit_garch
from model_comparisons import pairwise_qlike_comparisons


MARKETS = ("SP500", "TASI")
FOLD_FRACTIONS = (0.70, 0.80, 0.90, 1.00)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


def run(data_dir: Path, output_dir: Path) -> None:
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_paths = {market: data_dir / f"{market.lower()}.csv" for market in MARKETS}
    missing = [str(path) for path in data_paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required market data file(s): " + ", ".join(missing))

    summary = []
    prediction_rows = []
    transfers = []
    for market, path in data_paths.items():
        raw = pd.read_csv(path)
        raw["Date"] = pd.to_datetime(raw.Date, errors="raise")
        raw = raw.sort_values("Date").reset_index(drop=True)
        x = prepare(path)
        returns = np.diff(np.log(raw.Close.to_numpy()), prepend=np.nan)
        lookup = {date: i for i, date in enumerate(raw.Date)}
        bounds = [int(len(x) * fraction) for fraction in FOLD_FRACTIONS]

        for fold, (lo, hi) in enumerate(zip(bounds[:-1], bounds[1:]), start=1):
            train = x.iloc[:lo - 1]
            test = x.iloc[lo:hi]
            pred = fit_predict(train, test)

            first = lookup[train.date.iloc[-1]]
            theta, h = fit_garch(returns[1:first + 1])
            h = theta[0] + theta[1] * (100 * returns[first]) ** 2 + theta[2] * h
            garch_predictions = []
            previous_index = first
            for date in test.date:
                index = lookup[date]
                for k in range(previous_index + 1, index + 1):
                    h = theta[0] + theta[1] * (100 * returns[k]) ** 2 + theta[2] * h
                garch_predictions.append(max(h / 10000, EPS))
                previous_index = index
            pred["GARCH"] = np.asarray(garch_predictions)

            for name, forecast in pred.items():
                qlike, mse = losses(test.y, forecast)
                summary.append({
                    "market": market,
                    "fold": fold,
                    "model": name,
                    "n": len(test),
                    "QLIKE": float(np.mean(qlike)),
                    "MSE": float(np.mean(mse)),
                })
                prediction_rows.extend(
                    {"market": market, "fold": fold, "date": date,
                     "model": name, "target_sq_return": y, "pred_var": p}
                    for date, y, p in zip(test.date, test.y, forecast)
                )

    summary_frame = pd.DataFrame(summary)
    predictions_frame = pd.DataFrame(prediction_rows)
    summary_frame.to_csv(output_dir / "fold_scores.csv", index=False)
    predictions_frame.to_csv(output_dir / "predictions.csv", index=False)

    pairwise = pairwise_qlike_comparisons(
        predictions_frame, replicates=4000, block_length=10, seed=2026
    )
    pairwise.to_csv(output_dir / "pairwise_qlike_comparisons.csv", index=False)

    # Compare source-trained and destination-trained models on identical
    # destination dates; treat these results as descriptive transfer checks.
    prepared = {market: prepare(path) for market, path in data_paths.items()}
    for source, destination in (("SP500", "TASI"), ("TASI", "SP500")):
        source_data = prepared[source]
        target_data = prepared[destination]
        cut = target_data.date.iloc[int(0.8 * len(target_data))]
        source_train = source_data[source_data.date < cut].iloc[:-1]
        target_test = target_data[target_data.date >= cut]
        local_train = target_data[target_data.date < cut].iloc[:-1]
        for training, training_market in ((source_train, source), (local_train, destination)):
            model = RandomForestRegressor(
                n_estimators=120, max_depth=6, min_samples_leaf=12,
                n_jobs=-1, random_state=2026,
            )
            model.fit(training[FEATURES], training.y)
            forecast = np.maximum(model.predict(target_test[FEATURES]), EPS)
            qlike, mse = losses(target_test.y, forecast)
            transfers.append({
                "source": training_market,
                "destination": destination,
                "evaluation_start": str(target_test.date.iloc[0].date()),
                "evaluation_end": str(target_test.date.iloc[-1].date()),
                "n": len(target_test),
                "QLIKE": float(qlike.mean()),
                "MSE": float(mse.mean()),
            })
    pd.DataFrame(transfers).to_csv(output_dir / "aligned_transfer.csv", index=False)

    # Paired moving-block bootstrap for RF versus EWMA; intervals remain
    # exploratory and are not corrected for the number of comparisons.
    confidence_intervals = []
    for (market, fold), group in predictions_frame.groupby(["market", "fold"]):
        pivot = group.pivot(index="date", columns="model", values="pred_var")
        target = group.drop_duplicates("date").set_index("date").loc[
            pivot.index, "target_sq_return"
        ]
        rf_qlike, _ = losses(target, pivot["RF"])
        ewma_qlike, _ = losses(target, pivot["EWMA"])
        difference, low, high = bootstrap_diff(
            rf_qlike, ewma_qlike, B=1500, block=10, seed=2026 + fold
        )
        confidence_intervals.append({
            "market": market,
            "fold": fold,
            "RF_minus_EWMA": difference,
            "CI95_low": low,
            "CI95_high": high,
        })
    pd.DataFrame(confidence_intervals).to_csv(
        output_dir / "rf_vs_ewma_bootstrap.csv", index=False
    )

    for market in MARKETS:
        table = summary_frame[summary_frame.market == market].pivot(
            index="fold", columns="model", values="QLIKE"
        )
        fig, ax = plt.subplots(figsize=(9, 5))
        table[["EWMA", "GARCH", "RF", "HGB_log", "RV20"]].plot(
            ax=ax, marker="o"
        )
        ax.set_title(f"{market}: expanding-window out-of-sample QLIKE")
        ax.set_xlabel("Evaluation fold")
        ax.set_ylabel("Mean QLIKE (lower is better)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{market}_fold_qlike.png", dpi=170)
        plt.close(fig)

    metadata = {
        "data_files": {
            market: {
                "path": path.name,
                "sha256": sha256_file(path),
                "rows": int(len(pd.read_csv(path))),
            }
            for market, path in data_paths.items()
        },
        "fold_fractions": list(FOLD_FRACTIONS),
        "forecast_target": "next-session squared log return",
        "bootstrap": {
            "rf_vs_ewma_replicates": 1500,
            "pairwise_replicates": 4000,
            "block_length": 10,
            "seed": 2026,
            "holm_family": "all model pairs across both markets and all evaluation folds",
        },
        "package_versions": {
            name: package_version(name)
            for name in ("numpy", "pandas", "scipy", "scikit-learn", "matplotlib")
        },
        "reproduction_note": "Raw price data are not distributed in this repository; SHA-256 values identify the local inputs used for this run.",
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Completed analysis. Outputs: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data",
                        help="Directory containing sp500.csv and tasi.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results_final",
                        help="Directory for scores, predictions, figures, and run metadata")
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()

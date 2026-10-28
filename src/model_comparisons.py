"""Dependence-aware pairwise QLIKE comparisons with project-wide Holm control."""
from __future__ import annotations

import numpy as np
import pandas as pd

from advanced_validation import losses


REQUIRED_COLUMNS = {
    "market", "fold", "date", "model", "target_sq_return", "pred_var"
}


def holm_adjust(p_values):
    """Holm step-down adjusted p-values, preserving the input order."""
    p_values = np.asarray(p_values, dtype=float)
    if p_values.ndim != 1 or not np.isfinite(p_values).all():
        raise ValueError("p_values must be a finite one-dimensional array")
    if ((p_values < 0) | (p_values > 1)).any():
        raise ValueError("p_values must be between 0 and 1")
    count = len(p_values)
    if count == 0:
        return np.array([], dtype=float)
    order = np.argsort(p_values)
    ordered = p_values[order]
    adjusted_ordered = np.maximum.accumulate(
        (count - np.arange(count)) * ordered
    ).clip(max=1.0)
    adjusted = np.empty(count, dtype=float)
    adjusted[order] = adjusted_ordered
    return adjusted


def _circular_block_means(values, *, replicates, block_length, seed):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("values must be a non-empty finite vector")
    if replicates < 100:
        raise ValueError("use at least 100 bootstrap replicates")
    if block_length < 1 or block_length > len(values):
        raise ValueError("block_length must be between 1 and series length")
    rng = np.random.default_rng(seed)
    blocks_per_replicate = int(np.ceil(len(values) / block_length))
    offsets = np.arange(block_length)
    means = np.empty(replicates, dtype=float)
    for i in range(replicates):
        starts = rng.integers(0, len(values), size=blocks_per_replicate)
        indices = ((starts[:, None] + offsets) % len(values)).ravel()[:len(values)]
        means[i] = values[indices].mean()
    return means


def pairwise_qlike_comparisons(
    predictions, *, replicates=4000, block_length=10, seed=2026
):
    """Compare all model pairs within each market/fold on common dates.

    Negative ``mean_difference`` means model_a has lower (better) QLIKE than
    model_b. The percentile intervals and centered-bootstrap p-values are
    exploratory. Holm adjustment controls family-wise error over all model
    pairs and folds in this output, but it does not repair data reuse or
    compensate for only having a few evaluation windows.
    """
    missing = REQUIRED_COLUMNS - set(predictions.columns)
    if missing:
        raise ValueError(f"predictions are missing columns: {sorted(missing)}")
    if predictions.empty:
        raise ValueError("predictions must contain at least one row")

    rows = []
    for group_index, ((market, fold), frame) in enumerate(
        predictions.groupby(["market", "fold"], sort=True)
    ):
        if (frame.groupby("date")["target_sq_return"].nunique(dropna=False) > 1).any():
            raise ValueError(f"market={market}, fold={fold} has inconsistent targets across models")
        pivot = frame.pivot(index="date", columns="model", values="pred_var").sort_index()
        targets = frame.drop_duplicates("date").set_index("date")["target_sq_return"].reindex(pivot.index)
        if targets.isna().any() or pivot.isna().any().any():
            raise ValueError(f"market={market}, fold={fold} has incomplete aligned predictions")
        model_names = sorted(pivot.columns.astype(str))
        if len(model_names) < 2:
            raise ValueError(f"market={market}, fold={fold} has fewer than two models")
        qlike = {
            name: losses(targets.to_numpy(), pivot[name].to_numpy())[0]
            for name in model_names
        }

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i + 1:]:
                differences = qlike[model_a] - qlike[model_b]
                observed = float(differences.mean())
                draws = _circular_block_means(
                    differences,
                    replicates=replicates,
                    block_length=block_length,
                    seed=seed + group_index * 100 + len(rows),
                )
                centered = draws - observed
                p_value = (
                    1 + np.count_nonzero(np.abs(centered) >= abs(observed))
                ) / (replicates + 1)
                low, high = np.quantile(draws, [0.025, 0.975])
                rows.append({
                    "market": market,
                    "fold": int(fold),
                    "model_a": model_a,
                    "model_b": model_b,
                    "n": len(differences),
                    "mean_difference": observed,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "p_value_bootstrap": float(p_value),
                    "bootstrap_replicates": int(replicates),
                    "block_length": int(block_length),
                })

    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_adjust(result["p_value_bootstrap"].to_numpy())
    result["ci_excludes_zero"] = (result.ci95_low > 0) | (result.ci95_high < 0)
    result["holm_significant_0_05"] = result.p_value_holm < 0.05
    return result

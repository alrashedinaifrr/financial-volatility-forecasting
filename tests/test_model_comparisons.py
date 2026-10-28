import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from model_comparisons import holm_adjust, pairwise_qlike_comparisons


def test_holm_adjustment_preserves_order_and_monotonicity():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_pairwise_comparisons_use_aligned_dates_and_report_holm():
    rng = np.random.default_rng(24)
    target = rng.lognormal(mean=-10.2, sigma=0.3, size=40)
    rows = []
    forecasts = {
        "OracleLike": target,
        "Over": 1.5 * target,
        "Under": 0.5 * target,
    }
    for model, prediction in forecasts.items():
        for i, (actual, forecast) in enumerate(zip(target, prediction)):
            rows.append({
                "market": "TEST",
                "fold": 1,
                "date": pd.Timestamp("2025-01-01") + pd.Timedelta(days=i),
                "model": model,
                "target_sq_return": actual,
                "pred_var": forecast,
            })

    result = pairwise_qlike_comparisons(
        pd.DataFrame(rows), replicates=300, block_length=5, seed=11
    )
    assert len(result) == 3
    assert set(result["n"]) == {40}
    assert result["p_value_holm"].between(0, 1).all()
    oracle_vs_over = result.loc[
        (result.model_a == "OracleLike") & (result.model_b == "Over"),
        "mean_difference",
    ].item()
    assert oracle_vs_over < 0


def test_pairwise_comparisons_reject_incomplete_model_dates():
    frame = pd.DataFrame([
        {"market": "TEST", "fold": 1, "date": "2025-01-01",
         "model": "A", "target_sq_return": 0.01, "pred_var": 0.01},
        {"market": "TEST", "fold": 1, "date": "2025-01-01",
         "model": "B", "target_sq_return": 0.01, "pred_var": 0.02},
        {"market": "TEST", "fold": 1, "date": "2025-01-02",
         "model": "A", "target_sq_return": 0.01, "pred_var": 0.01},
    ])
    try:
        pairwise_qlike_comparisons(frame, replicates=100, block_length=1)
    except ValueError as error:
        assert "incomplete aligned predictions" in str(error)
    else:
        raise AssertionError("expected incomplete aligned predictions to fail")

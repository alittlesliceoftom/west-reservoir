"""Forecast accuracy metrics and backtest replay"""

import numpy as np
import pandas as pd


BIAS_NOTE = "bias = forecast - actual; positive means the model runs warm"

HIT_THRESHOLD_C = 0.5

METRIC_COLUMNS = ["horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"]

SCORED_COLUMNS = [
    "target_date", "horizon_days", "forecast_temp",
    "actual_temp", "error", "air_source",
]

REPLAY_COLUMNS = ["target_date", "horizon_days", "forecast_temp", "air_source"]


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute accuracy metrics for scored forecasts.

    Args:
        df: DataFrame with 'forecast_temp' and 'actual_temp' columns.
            Rows with a missing value in either are excluded.

    Returns:
        dict with mae, bias, rmse, hit_rate_0_5 (percent), and n.
        Metrics are NaN when n == 0; n is always an int.
    """
    if df.empty:
        return {"mae": np.nan, "bias": np.nan, "rmse": np.nan,
                "hit_rate_0_5": np.nan, "n": 0}

    valid = df.dropna(subset=["forecast_temp", "actual_temp"])
    n = len(valid)

    if n == 0:
        return {"mae": np.nan, "bias": np.nan, "rmse": np.nan,
                "hit_rate_0_5": np.nan, "n": 0}

    error = valid["forecast_temp"] - valid["actual_temp"]

    return {
        "mae": float(error.abs().mean()),
        "bias": float(error.mean()),
        "rmse": float(np.sqrt((error ** 2).mean())),
        "hit_rate_0_5": float((error.abs() <= HIT_THRESHOLD_C).mean() * 100),
        "n": int(n),
    }


def metrics_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute metrics separately for each forecast horizon.

    Args:
        df: DataFrame with 'forecast_temp', 'actual_temp', 'horizon_days'.

    Returns:
        DataFrame with one row per horizon, sorted ascending.
    """
    if df.empty:
        return pd.DataFrame(columns=METRIC_COLUMNS)

    rows = []
    for horizon, group in df.groupby("horizon_days", sort=True):
        metrics = compute_metrics(group)
        metrics["horizon_days"] = int(horizon)
        rows.append(metrics)

    return pd.DataFrame(rows, columns=METRIC_COLUMNS)

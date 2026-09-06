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


def join_actuals(forecasts: pd.DataFrame, water_temps: pd.DataFrame) -> pd.DataFrame:
    """
    Join forecasts to the measurements they were predicting.

    Only days with a measurement can be scored; unmeasured days drop out and
    are reflected in the reported N. Forecast NaNs are kept so coverage gaps
    stay visible - compute_metrics excludes them.

    Args:
        forecasts: target_date, horizon_days, forecast_temp, optional air_source
        water_temps: load_water_temps() frame with date, water_temp

    Returns:
        DataFrame with SCORED_COLUMNS. air_source defaults to 'FORECAST'.
    """
    if forecasts.empty or water_temps.empty:
        return pd.DataFrame(columns=SCORED_COLUMNS)

    actuals = water_temps.dropna(subset=["water_temp"]).copy()
    actuals["target_date"] = pd.to_datetime(actuals["date"]).dt.normalize()
    # Duplicate measurement dates have broken prediction chains before; keep last.
    actuals = actuals.drop_duplicates(subset=["target_date"], keep="last")
    actuals = actuals[["target_date", "water_temp"]].rename(
        columns={"water_temp": "actual_temp"}
    )

    scored = forecasts.copy()
    scored["target_date"] = pd.to_datetime(scored["target_date"]).dt.normalize()
    if "air_source" not in scored.columns:
        scored["air_source"] = "FORECAST"

    scored = scored.merge(actuals, on="target_date", how="inner")
    scored["error"] = scored["forecast_temp"] - scored["actual_temp"]

    return scored[SCORED_COLUMNS].sort_values(
        ["target_date", "horizon_days"]
    ).reset_index(drop=True)

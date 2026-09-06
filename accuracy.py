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

AIR_COLUMNS = ["datetime", "air_temp"]


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


def splice_air_history(
    actual_hourly: pd.DataFrame,
    stored_run_hourly: pd.DataFrame,
    anchor_datetime,
) -> pd.DataFrame:
    """
    Reconstruct the air-temperature series the live forecast actually had.

    A stored forecast run created late on day d only covers the remainder of
    that day - roughly 16k rows at horizon 0 against 30k at horizon 1. Feeding
    the raw run to the forecaster would simulate ~10 hours of a 24-hour period
    and produce a number the model never computes.

    What the live run had was measured air for the elapsed part of the day,
    then forecast air from its creation time onward. This splices that.

    Not the same as combine_hourly_temps, which gives historical precedence on
    overlap. For a past anchor the actuals cover the whole window, so that
    would override the entire stored run and the replay would silently become
    100% actuals.

    Args:
        actual_hourly: Measured hourly air temps (datetime, air_temp).
        stored_run_hourly: The stored forecast run, interpolated to hourly.
        anchor_datetime: Start of the window (the 7am measurement time).

    Returns:
        DataFrame (datetime, air_temp) from anchor_datetime onward, with the
        stored run taking precedence wherever it has data.
    """
    anchor = pd.Timestamp(anchor_datetime)

    def _prepared(df):
        if df is None or df.empty:
            return pd.DataFrame(columns=AIR_COLUMNS)
        out = df[AIR_COLUMNS].copy()
        out["datetime"] = pd.to_datetime(out["datetime"])
        return out[out["datetime"] >= anchor]

    actuals = _prepared(actual_hourly)
    stored = _prepared(stored_run_hourly)

    if stored.empty:
        return actuals.sort_values("datetime").reset_index(drop=True)

    # Actuals only cover the head, up to where the forecast run begins.
    head = actuals[actuals["datetime"] < stored["datetime"].min()]

    combined = pd.concat([head, stored], ignore_index=True)
    combined = combined.drop_duplicates(subset=["datetime"], keep="last")

    return combined.sort_values("datetime").reset_index(drop=True)

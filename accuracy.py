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


def replay_current_model(
    forecaster,
    water_temps: pd.DataFrame,
    weather_provider,
    max_horizon: int = 5,
) -> pd.DataFrame:
    """
    Re-run the current model over history: what accuracy would have been.

    For each measured day, anchor on that measurement and forecast forward
    max_horizon days. Each anchor is an independent trajectory.

    This is a model-development tool, not a record of real performance. It is
    optimistically biased: solar and cloud forecasts were never stored, so
    actual solar/cloud is used for every date. See the design spec.

    Args:
        forecaster: A fitted WaterTempForecaster. Its coefficients are used
                    as-is; nothing is refitted.
        water_temps: load_water_temps() frame with date, water_temp.
        weather_provider: Callable taking an anchor date (pd.Timestamp) and
                          returning (hourly_weather_df_or_None, air_source).
                          All I/O belongs in the caller so it can be bulk-fetched.
        max_horizon: Days ahead to forecast from each anchor.

    Returns:
        DataFrame with REPLAY_COLUMNS. forecast_temp is NaN where weather
        coverage was unavailable. Horizons start at 1 - the anchor is the
        measurement itself, so there is no horizon 0.
    """
    if water_temps.empty:
        return pd.DataFrame(columns=REPLAY_COLUMNS)

    anchors = water_temps.dropna(subset=["water_temp"]).copy()
    anchors["date"] = pd.to_datetime(anchors["date"]).dt.normalize()
    anchors = anchors.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    if anchors.empty:
        return pd.DataFrame(columns=REPLAY_COLUMNS)

    # The replay swaps weather in per anchor; put back whatever was there.
    saved_weather = forecaster.hourly_weather

    frames = []
    try:
        for _, anchor in anchors.iterrows():
            anchor_date = anchor["date"]
            hourly_weather, air_source = weather_provider(anchor_date)

            if hourly_weather is None or hourly_weather.empty:
                frames.append(pd.DataFrame({
                    "target_date": [
                        anchor_date + pd.Timedelta(days=h)
                        for h in range(1, max_horizon + 1)
                    ],
                    "horizon_days": list(range(1, max_horizon + 1)),
                    "forecast_temp": [np.nan] * max_horizon,
                    "air_source": [air_source] * max_horizon,
                }))
                continue

            forecaster.set_hourly_weather(hourly_weather)
            predictions = forecaster.predict_forward(
                start_datetime=anchor_date,
                start_water_temp=float(anchor["water_temp"]),
                days_ahead=max_horizon,
            )

            frames.append(pd.DataFrame({
                "target_date": predictions["target_datetime"].dt.normalize(),
                "horizon_days": predictions["horizon_days"],
                "forecast_temp": predictions["water_temp"],
                "air_source": air_source,
            }))
    finally:
        forecaster.hourly_weather = saved_weather

    result = pd.concat(frames, ignore_index=True)
    return result[REPLAY_COLUMNS].sort_values(
        ["target_date", "horizon_days"]
    ).reset_index(drop=True)

"""
The snapshot payload: exactly what the water page renders.

Timestamps are ISO-8601 UTC strings; the page converts them to the water's
local time. Temperatures are °C rounded to 0.1 for display values and 0.01
for chart series.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

READINGS_DAYS = 60
AIR_PAST_DAYS = 14
TRAJECTORY_STEP_HOURS = 3


def iso(at) -> str:
    return pd.Timestamp(at).tz_convert("UTC").isoformat()


def _series(values: pd.Series, digits=2) -> list[list]:
    return [[iso(t), round(float(v), digits)] for t, v in values.items() if pd.notna(v)]


def build_payload(
    *,
    water,
    now: datetime,
    readings: pd.DataFrame,
    fit,
    anchor,
    estimate: float | None,
    points: list[dict],
    trajectory: pd.Series | None,
    weather: pd.DataFrame,
    weather_fetched_at: datetime | None,
    chart: pd.DataFrame | None = None,
    feed: dict | None = None,
) -> dict:
    """
    `readings` are the usable readings the model ran on; `chart` is every
    standing reading for the chart, each with the reason it is not usable
    (None when it is). Without `chart`, the usable readings are charted.
    `feed` is the feed block for a feed water, or None.
    """
    if chart is None:
        chart = readings.assign(reason=None)
    recent = chart[chart.measured_at >= pd.Timestamp(now) - timedelta(days=READINGS_DAYS)]
    air = weather["air_temp"] if not weather.empty else pd.Series(dtype=float, index=pd.DatetimeIndex([], tz="UTC"))
    air = air[air.index >= pd.Timestamp(now) - timedelta(days=AIR_PAST_DAYS)]
    if trajectory is not None:
        trajectory = trajectory.iloc[::TRAJECTORY_STEP_HOURS]
    train_mae = None if fit is None or np.isnan(fit.train_mae) else round(float(fit.train_mae), 3)
    return {
        "built_at": iso(now),
        "water": {
            "name": water.name,
            "slug": water.slug,
            "water_type": water.water_type,
            "depth_class": water.depth_class,
            "timezone": water.timezone,
            "reading_time": water.default_reading_time.strftime("%H:%M"),
        },
        "latest_reading": None if anchor is None else {
            "temp": round(float(anchor.temp_c), 1), "at": iso(anchor.measured_at),
        },
        "estimate_now": None if estimate is None else {
            "temp": round(estimate, 1), "at": iso(pd.Timestamp(now).floor("h")),
        },
        "forecast": points,
        "trajectory": [] if trajectory is None else _series(trajectory),
        "readings": [[iso(r.measured_at), round(float(r.temp_c), 1), r.reason] for r in recent.itertuples()],
        "air": _series(air, digits=1),
        "calibration": {
            "method": fit.method,
            "n_readings": int(len(readings)),
            "n_legs": int(fit.n_legs),
            "scale": float(fit.scale),
        },
        "train_mae": train_mae,
        "weather_fetched_at": None if weather_fetched_at is None else iso(weather_fetched_at),
        "feed": feed,
    }

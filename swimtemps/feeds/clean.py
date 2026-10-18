"""
Cleaning feed readings on arrival. A flagged reading is stored and shown,
with its reason, but kept out of the fit, the estimate and scoring.

Every comparison is against good (unflagged) readings only, so a fault never
becomes the reference for what follows it:

- jump: further than `jump_base` + `jump_per_hour` x hours from the last good reading;
- far from the day: further than `day_limit` from the median of good readings
  in the previous 24 hours (skipped when there are none, so after a day the
  growing jump allowance accepts a genuine shift);
- flatline: the same value as a run of good readings lasting more than
  `flatline_hours`.

A flagged value within `air_margin` of the air temperature says so.
"""

from datetime import datetime, timedelta
from statistics import median

import pandas as pd

DEFAULT_THRESHOLDS = {"jump_base": 2.0, "jump_per_hour": 0.1, "day_limit": 2.5, "flatline_hours": 24, "air_margin": 1.0}
DAY = timedelta(hours=24)
AIR_NOTE = " It matches the air temperature: the sensor is probably out of the water."


def _air_at(air: pd.Series, at: datetime) -> float | None:
    hour = (pd.Timestamp(at) + pd.Timedelta(minutes=30)).floor("h")
    value = air.get(hour)
    return None if value is None or pd.isna(value) else float(value)


def _reason(at: datetime, temp: float, good: list[tuple[datetime, float]], limits: dict) -> str | None:
    if not good:
        return None
    last_at, last_temp = good[-1]
    hours = (at - last_at) / timedelta(hours=1)
    allowance = limits["jump_base"] + limits["jump_per_hour"] * hours
    if abs(temp - last_temp) > allowance:
        return f"Jumped {abs(temp - last_temp):.1f} °C from the last good reading ({last_temp:.1f} °C)."

    day = [value for when, value in good if when > at - DAY]
    if day:
        reference = median(day)
        if abs(temp - reference) > limits["day_limit"]:
            return f"{abs(temp - reference):.1f} °C from the previous day's typical reading ({reference:.1f} °C)."

    run_start = None
    for when, value in reversed(good):
        if value != temp:
            break
        run_start = when
    if run_start is not None and at - run_start > timedelta(hours=limits["flatline_hours"]):
        return f"Flatline: the same value ({temp:.1f} °C) for {(at - run_start) / timedelta(hours=1):.0f} hours."
    return None


def clean(
    new_rows: list[tuple[datetime, float]],
    good_history: list[tuple[datetime, float]],
    air: pd.Series,
    thresholds: dict | None = None,
) -> list[str | None]:
    """
    One reason, or None, for each of `new_rows`, in order. Both lists are
    sorted (time, °C). Each new row is judged against the good readings
    before it: those of `good_history` earlier than it (which may also run
    past it, between new rows) and the new rows accepted so far. `air` is
    hourly air temperature on a UTC index.
    """
    limits = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    history = sorted(good_history)
    good, next_stored = [], 0
    reasons = []
    for at, temp in new_rows:
        while next_stored < len(history) and history[next_stored][0] < at:
            good.append(history[next_stored])
            next_stored += 1
        reason = _reason(at, temp, good, limits)
        if reason is None:
            good.append((at, temp))
        else:
            air_temp = _air_at(air, at)
            if air_temp is not None and abs(temp - air_temp) <= limits["air_margin"]:
                reason += AIR_NOTE
        reasons.append(reason)
    return reasons

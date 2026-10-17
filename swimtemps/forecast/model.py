"""
Anchors, targets, training legs and the staged fit.

Readings are frames of `measured_at` (tz-aware UTC) and `temp_c`. Weather is
the frame `prepare_weather` returns: a complete UTC hourly index, so a gap in
the data is a NaN row rather than a missing one.
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar

from forecast.physics import (
    WEATHER_COLUMNS,
    Coefficients,
    leg_end_temps,
    load_params,
    simulate,
    west_res_coefficients,
)

AIR_FILL_HOURS = 3
HOUR = pd.Timedelta(hours=1)

# Bounds for the scale fit, and for each coefficient in the full fit.
SCALE_BOUNDS = (0.05, 20.0)
K_BOUNDS = {"k_air": (1e-4, 0.3), "k_solar": (1e-7, 1e-2), "k_cool": (1e-5, 0.3)}


def prepare_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Turn stored weather rows into the frame the model simulates over.

    Missing solar is zero and missing cloud is 100%, which leaves air
    conduction as the only term. Air is carried forward up to three hours;
    a longer gap stays NaN and stops any trajectory that reaches it.
    """
    frame = raw.set_index("valid_at") if "valid_at" in raw.columns else raw
    frame = frame[list(WEATHER_COLUMNS)].astype(float).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame.empty:
        return frame
    full = pd.date_range(frame.index[0], frame.index[-1], freq="h")
    frame = frame.reindex(full)
    frame["air_temp"] = frame["air_temp"].ffill(limit=AIR_FILL_HOURS)
    frame["shortwave"] = frame["shortwave"].fillna(0.0)
    frame["cloud_cover"] = frame["cloud_cover"].fillna(100.0)
    return frame.rename_axis("valid_at")


def anchor_hour(at) -> pd.Timestamp:
    """The hour a reading's simulation starts from: its time to the nearest hour."""
    stamp = pd.Timestamp(at).tz_convert("UTC")
    return (stamp + pd.Timedelta(minutes=30)).floor("h")


def forecast_targets(now, tz: str, reading_time: time, days: int = 5) -> list[pd.Timestamp]:
    """
    The next `days` occurrences of the water's local reading time after
    `now`, in UTC: today's if it is still to come, then one per day.
    """
    zone = ZoneInfo(tz)
    now = pd.Timestamp(now).tz_convert("UTC")
    today = now.tz_convert(zone).date()

    def at(day):
        return pd.Timestamp(datetime.combine(day, reading_time, tzinfo=zone)).tz_convert("UTC")

    first = 0 if at(today) > now else 1
    return [at(today + timedelta(days=d)) for d in range(first, first + days)]


def run_forward(anchor_at, anchor_temp: float, weather: pd.DataFrame, c: Coefficients, until) -> pd.Series:
    """
    Hourly trajectory from the anchor hour to `until` (inclusive, to the hour).
    Hours the weather does not cover are NaN, and so is everything after them.
    """
    start = anchor_hour(anchor_at)
    end = pd.Timestamp(until).tz_convert("UTC").floor("h")
    if end <= start:
        return pd.Series([float(anchor_temp)], index=pd.DatetimeIndex([start]))
    hours = pd.date_range(start, end - HOUR, freq="h")
    window = weather.reindex(hours)
    values = simulate(anchor_temp, window, c)
    return pd.Series(values, index=pd.date_range(start, end, freq="h"))


def estimate_at(anchor_at, anchor_temp: float, weather: pd.DataFrame, c: Coefficients, at) -> float | None:
    """Estimated temperature at `at` (floored to the hour), or None across a weather gap."""
    if pd.Timestamp(at).floor("h") <= anchor_hour(anchor_at):
        return float(anchor_temp)
    value = run_forward(anchor_at, anchor_temp, weather, c, until=at).iloc[-1]
    return None if np.isnan(value) else float(value)


def forecast_points(anchor_at, anchor_temp, weather, c, targets) -> list[tuple[pd.Timestamp, float | None]]:
    """One continuous trajectory, read off at each target hour."""
    if not targets:
        return []
    trajectory = run_forward(anchor_at, anchor_temp, weather, c, until=max(targets))
    points = []
    for target in targets:
        value = trajectory.get(anchor_hour(target), np.nan)
        points.append((target, None if np.isnan(value) else float(value)))
    return points


@dataclass
class Legs:
    """Consecutive reading pairs, weather held as padded (legs x hours) matrices."""
    start_temps: np.ndarray
    end_temps: np.ndarray
    air: np.ndarray
    solar: np.ndarray
    cloud: np.ndarray
    mask: np.ndarray
    n_hours: np.ndarray

    @property
    def count(self) -> int:
        return len(self.start_temps)

    def predict(self, c: Coefficients) -> np.ndarray:
        return leg_end_temps(self.start_temps, self.air, self.solar, self.cloud, self.mask, self.n_hours, c)


def _collapse_same_hour(readings: pd.DataFrame) -> pd.DataFrame:
    frame = readings[["measured_at", "temp_c"]].copy()
    frame["hour"] = frame["measured_at"].map(anchor_hour)
    frame = frame.sort_values("measured_at").drop_duplicates("hour", keep="last")
    return frame.reset_index(drop=True)


def build_legs(readings: pd.DataFrame, weather: pd.DataFrame, max_leg_days: float, min_coverage: float) -> Legs:
    """
    One leg per consecutive pair of readings, kept when the pair is at most
    `max_leg_days` apart and the weather covers at least `min_coverage` of the
    hours between them. Short holes inside a kept leg are interpolated.
    """
    frame = _collapse_same_hour(readings)
    kept = []
    for i in range(len(frame) - 1):
        start, end = frame.loc[i, "hour"], frame.loc[i + 1, "hour"]
        n = int((end - start) / HOUR)
        if n <= 0 or n > max_leg_days * 24:
            continue
        window = weather.reindex(pd.date_range(start, end - HOUR, freq="h"))
        if window["air_temp"].notna().sum() < min_coverage * n:
            continue
        window = window.interpolate(limit_direction="both")
        kept.append((frame.loc[i, "temp_c"], frame.loc[i + 1, "temp_c"], window))

    width = max((len(w) for *_, w in kept), default=0)
    shape = (len(kept), width)
    air, solar, cloud = np.zeros(shape), np.zeros(shape), np.zeros(shape)
    mask = np.zeros(shape, dtype=bool)
    for row, (_, _, window) in enumerate(kept):
        n = len(window)
        air[row, :n] = window["air_temp"].to_numpy()
        solar[row, :n] = window["shortwave"].to_numpy()
        cloud[row, :n] = window["cloud_cover"].to_numpy()
        mask[row, :n] = True
    return Legs(
        start_temps=np.array([k[0] for k in kept], dtype=float),
        end_temps=np.array([k[1] for k in kept], dtype=float),
        air=air, solar=solar, cloud=cloud, mask=mask,
        n_hours=mask.sum(axis=1),
    )


@dataclass(frozen=True)
class FitResult:
    coefficients: Coefficients
    scale: float        # multiplier on the West Res coefficients (geometric mean for a full fit)
    method: str         # 'prior' | 'scale' | 'full'
    n_legs: int
    train_mae: float    # NaN when there are no legs


def choose_method(n_legs: int, params: dict) -> str:
    if n_legs < params["scale_min_legs"]:
        return "prior"
    if n_legs < params["full_min_legs"]:
        return "scale"
    return "full"


def _mae(legs: Legs, c: Coefficients) -> float:
    if legs.count == 0:
        return float("nan")
    return float(np.abs(legs.predict(c) - legs.end_temps).mean())


def _fit_scale(legs: Legs, ref: Coefficients, s_prior: float, lam: float) -> float:
    log_prior = np.log(s_prior)

    def objective(u):
        error = legs.predict(ref.scaled(np.exp(u))) - legs.end_temps
        return float((error ** 2).sum() + lam * (u - log_prior) ** 2)

    bounds = tuple(np.log(SCALE_BOUNDS))
    result = minimize_scalar(objective, bounds=bounds, method="bounded", options={"xatol": 1e-4})
    return float(np.exp(result.x))


def _fit_full(legs: Legs, ref: Coefficients, s_prior: float, lam: float) -> Coefficients | None:
    names = ("k_air", "k_solar", "k_cool")
    centre = np.log([getattr(ref.scaled(s_prior), n) for n in names])

    def objective(x):
        k = np.exp(x)
        error = legs.predict(Coefficients(*k)) - legs.end_temps
        return float((error ** 2).sum() + lam * ((x - centre) ** 2).sum())

    bounds = [tuple(np.log(K_BOUNDS[n])) for n in names]
    x0 = np.clip(centre, [b[0] for b in bounds], [b[1] for b in bounds])
    result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")
    if not result.success:
        return None
    return Coefficients(*(float(v) for v in np.exp(result.x)))


def fit_water(
    readings: pd.DataFrame,
    weather: pd.DataFrame,
    depth_class: str,
    params: dict | None = None,
    method: str | None = None,
    prior_scale: float | None = None,
) -> FitResult:
    """
    Fit a water's coefficients by how much history it has.

    Under `scale_min_legs` usable legs nothing is fitted and the West Res
    coefficients are scaled by the depth prior. Up to `full_min_legs` only the
    scale is fitted; from there all three coefficients are. Both fits are
    penalised in log space towards the depth prior with weight `lambda`.
    `method` and `prior_scale` override the choice and the prior, for the replay.
    """
    params = params or load_params()
    s_prior = prior_scale if prior_scale is not None else params["depth_prior"][depth_class]
    legs = build_legs(readings, weather, params["max_leg_days"], params["min_leg_coverage"])
    return fit_legs(legs, s_prior, params, method)


def fit_legs(legs: Legs, s_prior: float, params: dict, method: str | None = None) -> FitResult:
    """`fit_water` on legs already built."""
    ref = west_res_coefficients(params)
    method = method or choose_method(legs.count, params)
    if legs.count == 0:
        method = "prior"

    if method == "full":
        fitted = _fit_full(legs, ref, s_prior, params["lambda"])
        if fitted is not None:
            ratios = [fitted.as_dict()[n] / ref.as_dict()[n] for n in ("k_air", "k_solar", "k_cool")]
            scale = float(np.exp(np.mean(np.log(ratios))))
            return FitResult(fitted, scale, "full", legs.count, _mae(legs, fitted))
        method = "scale"

    if method == "scale":
        scale = _fit_scale(legs, ref, s_prior, params["lambda"])
    else:
        scale = float(s_prior)
    coefficients = ref.scaled(scale)
    return FitResult(coefficients, scale, method, legs.count, _mae(legs, coefficients))

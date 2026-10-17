"""
Hourly water-temperature physics.

Per hour:

    clearness(t) = 1 - cloud_cover(t) / 100
    T(t+1h) = T(t) + k_air*(T_air(t) - T(t)) + k_solar*I(t) - k_cool*clearness(t)

Weather frames carry the columns `air_temp` (°C), `shortwave` (W/m²) and
`cloud_cover` (%), one row per hour.
"""

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

PARAMS_PATH = Path(__file__).with_name("params.json")
WEATHER_COLUMNS = ("air_temp", "shortwave", "cloud_cover")


@dataclass(frozen=True)
class Coefficients:
    k_air: float    # per hour
    k_solar: float  # °C per W/m² per hour
    k_cool: float   # °C per hour at a fully clear sky

    def scaled(self, s: float) -> "Coefficients":
        """
        Every term is a heat flux over the water's thermal mass, so a water
        with half the depth responds twice as fast on all three.
        """
        return Coefficients(self.k_air * s, self.k_solar * s, self.k_cool * s)

    def as_dict(self) -> dict:
        return {"k_air": self.k_air, "k_solar": self.k_solar, "k_cool": self.k_cool}


@cache
def _read_params(path: Path) -> str:
    return path.read_text()


def load_params(path: Path = PARAMS_PATH) -> dict:
    """Model constants chosen by the cold-start replay. A fresh copy each call."""
    return json.loads(_read_params(path))


def west_res_coefficients(params: dict | None = None) -> Coefficients:
    return Coefficients(**(params or load_params())["west_res"])


def step(t: float, air: float, solar: float, cloud: float, c: Coefficients) -> float:
    """One hour forward."""
    clearness = 1.0 - cloud / 100.0
    return t + c.k_air * (air - t) + c.k_solar * solar - c.k_cool * clearness


def simulate(t0: float, weather: pd.DataFrame, c: Coefficients) -> np.ndarray:
    """
    Trajectory through `weather`: element 0 is `t0`, element i is the
    temperature after i hours. A missing weather value makes that hour and
    every later one NaN, since nothing honest can follow a gap.
    """
    airs = weather["air_temp"].to_numpy(dtype=float)
    sols = weather["shortwave"].to_numpy(dtype=float)
    clouds = weather["cloud_cover"].to_numpy(dtype=float)
    out = np.empty(len(airs) + 1)
    out[0] = t = float(t0)
    for i in range(len(airs)):
        t = step(t, airs[i], sols[i], clouds[i], c)
        out[i + 1] = t
    return out


def leg_end_temps(
    start_temps: np.ndarray,
    air: np.ndarray,
    solar: np.ndarray,
    cloud: np.ndarray,
    mask: np.ndarray,
    n_hours: np.ndarray,
    c: Coefficients,
) -> np.ndarray:
    """
    End temperature of many legs at once, for the optimiser's inner loop.

    The recurrence is linear with a constant multiplier a = 1 - k_air, so a
    leg of N hours has the closed form

        T_N = a^N * T_0 + sum_n a^(N-1-n) * x_n
        x_n = k_air*air_n + k_solar*solar_n - k_cool*(1 - cloud_n/100)

    Legs are rows of padded (legs x hours) matrices; `mask` marks real hours
    and `n_hours` is each row's length.
    """
    a = 1.0 - c.k_air
    hour = np.arange(air.shape[1])
    exponent = n_hours[:, None] - 1 - hour[None, :]
    weights = np.where(mask, a ** np.clip(exponent, 0, None), 0.0)
    forcing = c.k_air * air + c.k_solar * solar - c.k_cool * (1.0 - cloud / 100.0)
    forcing = np.where(mask, forcing, 0.0)
    return a ** n_hours * start_temps + (weights * forcing).sum(axis=1)

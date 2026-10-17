"""80% uncertainty band on forecast points."""

import math

import pandas as pd

from forecast.physics import load_params


def horizon_days(anchor_at, valid_at) -> int:
    """Whole days from the anchor reading to a forecast point, rounded up, at least 1."""
    days = (pd.Timestamp(valid_at) - pd.Timestamp(anchor_at)) / pd.Timedelta(days=1)
    return max(1, math.ceil(days - 1e-9))


def halfwidth(
    horizon: int,
    method: str,
    water_type: str,
    own: dict[int, float] | None = None,
    params: dict | None = None,
) -> float:
    """
    Half the width of the 80% band, in °C.

    A water's own measured error at this horizon is used as it stands: it
    already reflects that water's type and fit. Otherwise the West Res base
    error is scaled up for a less-calibrated fit and for rivers and harbours.
    Horizons past the longest measured one use the longest.
    """
    if own and horizon in own:
        return float(own[horizon])
    params = params or load_params()
    base = params["band_base"]
    key = str(min(horizon, max(int(k) for k in base)))
    return (
        base[key]
        * params["method_multiplier"][method]
        * params["type_multiplier"][water_type]
    )

import pandas as pd
import pytest

from forecast.band import halfwidth, horizon_days
from forecast.physics import load_params

BASE = load_params()["band_base"]


def test_full_fit_on_still_water_is_the_base():
    assert halfwidth(1, "full", "still") == pytest.approx(BASE["1"])


def test_method_and_type_multipliers_multiply():
    assert halfwidth(3, "prior", "river") == pytest.approx(BASE["3"] * 1.5 * 1.5)
    assert halfwidth(3, "scale", "harbour") == pytest.approx(BASE["3"] * 1.25 * 1.5)


def test_horizons_past_the_measured_range_use_the_longest():
    assert halfwidth(12, "full", "still") == pytest.approx(BASE["5"])


def test_own_errors_replace_the_base_and_the_multipliers():
    own = {1: 0.2, 2: 0.3}
    assert halfwidth(2, "prior", "river", own=own) == pytest.approx(0.3)
    # A horizon this water has no score for falls back to the scaled base.
    assert halfwidth(4, "prior", "still", own=own) == pytest.approx(BASE["4"] * 1.5)


def test_horizon_days_rounds_up_and_is_at_least_one():
    anchor = pd.Timestamp("2026-06-01 07:00", tz="UTC")
    assert horizon_days(anchor, anchor + pd.Timedelta(days=1)) == 1
    assert horizon_days(anchor, anchor + pd.Timedelta(hours=36)) == 2
    assert horizon_days(anchor, anchor + pd.Timedelta(hours=3)) == 1

import numpy as np
import pandas as pd
import pytest

from forecast.physics import Coefficients, leg_end_temps, load_params, simulate, step

C = Coefficients(k_air=0.01, k_solar=1e-4, k_cool=0.008)


def hours(n, air=15.0, solar=0.0, cloud=50.0, start="2026-06-01 07:00"):
    index = pd.date_range(start, periods=n, freq="h", tz="UTC")
    return pd.DataFrame(
        {"air_temp": air, "shortwave": solar, "cloud_cover": cloud}, index=index
    )


def test_step_matches_the_spec_equation():
    # T + k_air*(air - T) + k_solar*I - k_cool*(1 - cloud/100)
    expected = 12.0 + 0.01 * (20.0 - 12.0) + 1e-4 * 400.0 - 0.008 * (1 - 0.25)
    assert step(12.0, 20.0, 400.0, 25.0, C) == pytest.approx(expected)


def test_full_cloud_removes_radiative_cooling():
    assert step(10.0, 10.0, 0.0, 100.0, C) == pytest.approx(10.0)


def test_simulate_chains_steps_and_keeps_the_anchor_as_row_zero():
    weather = hours(24, air=18.0, solar=200.0, cloud=30.0)
    trajectory = simulate(12.0, weather, C)
    assert len(trajectory) == 25
    assert trajectory[0] == 12.0
    t = 12.0
    for _ in range(24):
        t = step(t, 18.0, 200.0, 30.0, C)
    assert trajectory[-1] == pytest.approx(t)


def test_simulate_goes_nan_from_the_first_missing_hour():
    weather = hours(6)
    weather.iloc[3, 0] = np.nan
    trajectory = simulate(12.0, weather, C)
    assert np.isfinite(trajectory[:4]).all()
    assert np.isnan(trajectory[4:]).all()


def test_scaled_multiplies_every_coefficient():
    assert C.scaled(2.0) == Coefficients(0.02, 2e-4, 0.016)


def test_leg_end_temps_matches_simulate_for_ragged_legs():
    rng = np.random.default_rng(0)
    lengths = [24, 30, 7]
    hmax = max(lengths)
    air = rng.uniform(5, 25, (3, hmax))
    solar = rng.uniform(0, 600, (3, hmax))
    cloud = rng.uniform(0, 100, (3, hmax))
    mask = np.zeros((3, hmax), dtype=bool)
    for i, n in enumerate(lengths):
        mask[i, :n] = True
    starts = np.array([10.0, 14.0, 20.0])

    ends = leg_end_temps(starts, air, solar, cloud, mask, np.array(lengths), C)

    for i, n in enumerate(lengths):
        weather = pd.DataFrame(
            {"air_temp": air[i, :n], "shortwave": solar[i, :n], "cloud_cover": cloud[i, :n]}
        )
        assert ends[i] == pytest.approx(simulate(starts[i], weather, C)[-1])


def test_params_carry_the_spec_depth_prior_and_multipliers():
    params = load_params()
    assert params["depth_prior"] == {"shallow": 2.0, "lake": 1.0, "deep": 0.5, "unknown": 1.0}
    assert params["method_multiplier"] == {"prior": 1.5, "scale": 1.25, "full": 1.0}
    assert params["type_multiplier"] == {"still": 1.0, "river": 1.5, "harbour": 1.5}
    assert set(params["west_res"]) == {"k_air", "k_solar", "k_cool"}

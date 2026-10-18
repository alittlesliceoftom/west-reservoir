from datetime import time

import numpy as np
import pandas as pd
import pytest

from forecast.model import (
    anchor_hour,
    build_legs,
    estimate_at,
    fit_water,
    forecast_points,
    forecast_targets,
    prepare_weather,
    run_forward,
)
from forecast.physics import Coefficients, load_params, simulate, west_res_coefficients
from tests.forecast.synthetic import synthetic_readings, synthetic_weather

REF = west_res_coefficients()


def ts(s):
    return pd.Timestamp(s, tz="UTC")


# --- weather preparation -------------------------------------------------

def test_prepare_weather_reindexes_to_every_hour_and_fills_solar_and_cloud():
    raw = pd.DataFrame({
        "valid_at": [ts("2026-06-01 00:00"), ts("2026-06-01 01:00"), ts("2026-06-01 06:00")],
        "air_temp": [10.0, 11.0, 14.0],
        "shortwave": [None, 5.0, 50.0],
        "cloud_cover": [20.0, None, 40.0],
    })
    weather = prepare_weather(raw)
    assert len(weather) == 7
    assert weather.index[0] == ts("2026-06-01 00:00")
    assert weather.loc[ts("2026-06-01 00:00"), "shortwave"] == 0.0
    assert weather.loc[ts("2026-06-01 01:00"), "cloud_cover"] == 100.0
    # Air carries forward at most 3 hours; the 4th missing hour stays a gap.
    assert weather.loc[ts("2026-06-01 04:00"), "air_temp"] == 11.0
    assert np.isnan(weather.loc[ts("2026-06-01 05:00"), "air_temp"])


# --- anchors and targets -------------------------------------------------

def test_anchor_hour_rounds_to_the_nearest_hour():
    assert anchor_hour(ts("2026-06-01 19:00")) == ts("2026-06-01 19:00")
    assert anchor_hour(ts("2026-06-01 19:40")) == ts("2026-06-01 20:00")
    assert anchor_hour(ts("2026-06-01 18:30")) == ts("2026-06-01 19:00")
    assert anchor_hour(ts("2026-06-01 19:29")) == ts("2026-06-01 19:00")


def test_an_evening_reading_simulates_from_the_evening_not_the_morning():
    weather = prepare_weather(synthetic_weather(3).reset_index())
    trajectory = run_forward(ts("2026-04-01 19:00"), 12.0, weather, REF, until=ts("2026-04-02 07:00"))
    assert trajectory.index[0] == ts("2026-04-01 19:00")
    assert len(trajectory) == 13
    expected = simulate(12.0, weather.loc[ts("2026-04-01 19:00"):ts("2026-04-02 06:00")], REF)
    assert trajectory.iloc[-1] == pytest.approx(expected[-1])


def test_forecast_targets_are_the_next_five_local_days_at_reading_time_across_dst():
    # Europe/London leaves BST on Sunday 25 October 2026.
    now = ts("2026-10-23 12:00")
    targets = forecast_targets(now, "Europe/London", time(7, 0))
    assert targets == [
        ts("2026-10-24 06:00"),  # BST
        ts("2026-10-25 07:00"),  # GMT from here
        ts("2026-10-26 07:00"),
        ts("2026-10-27 07:00"),
        ts("2026-10-28 07:00"),
    ]


def test_forecast_targets_start_today_when_the_reading_time_is_still_to_come():
    targets = forecast_targets(ts("2026-06-01 00:30"), "Europe/London", time(7, 0))  # 01:30 BST
    assert targets[0] == ts("2026-06-01 06:00")
    assert targets[-1] == ts("2026-06-05 06:00")


def test_forecast_targets_use_the_waters_local_date_not_utc():
    # 23:30 UTC on the 1st is already the 2nd in Auckland.
    targets = forecast_targets(ts("2026-06-01 23:30"), "Pacific/Auckland", time(8, 0))
    assert targets[0] == pd.Timestamp("2026-06-03 08:00", tz="Pacific/Auckland").tz_convert("UTC")


# --- trajectories, estimates, points -----------------------------------

def test_a_weather_hole_withholds_the_estimate_rather_than_fabricating_one():
    raw = synthetic_weather(4).reset_index()
    raw = raw[(raw.valid_at < ts("2026-04-02 00:00")) | (raw.valid_at >= ts("2026-04-02 08:00"))]
    weather = prepare_weather(raw)
    anchor = ts("2026-04-01 07:00")
    assert estimate_at(anchor, 12.0, weather, REF, at=ts("2026-04-01 20:00")) is not None
    assert estimate_at(anchor, 12.0, weather, REF, at=ts("2026-04-03 10:00")) is None


def test_a_weeks_old_anchor_still_estimates_across_stored_weather():
    weather = prepare_weather(synthetic_weather(30).reset_index())
    value = estimate_at(ts("2026-04-02 07:00"), 12.0, weather, REF, at=ts("2026-04-25 12:30"))
    assert value is not None and 0 < value < 30


def test_estimate_at_or_before_the_anchor_is_the_reading_itself():
    weather = prepare_weather(synthetic_weather(2).reset_index())
    assert estimate_at(ts("2026-04-01 07:00"), 12.3, weather, REF, at=ts("2026-04-01 07:10")) == 12.3


def test_forecast_points_follow_one_continuous_trajectory():
    weather = prepare_weather(synthetic_weather(8).reset_index())
    anchor = ts("2026-04-01 07:00")
    targets = [ts("2026-04-02 07:00"), ts("2026-04-04 07:00")]
    points = forecast_points(anchor, 12.0, weather, REF, targets)
    trajectory = run_forward(anchor, 12.0, weather, REF, until=targets[-1])
    assert points == [(targets[0], pytest.approx(trajectory[targets[0]])),
                      (targets[1], pytest.approx(trajectory[targets[1]]))]


def test_forecast_points_beyond_the_weather_are_none():
    weather = prepare_weather(synthetic_weather(2).reset_index())
    points = forecast_points(ts("2026-04-01 07:00"), 12.0, weather, REF, [ts("2026-04-10 07:00")])
    assert points == [(ts("2026-04-10 07:00"), None)]


# --- legs ----------------------------------------------------------------

def test_legs_skip_gaps_longer_than_three_days_and_weather_holes():
    weather = prepare_weather(synthetic_weather(20).reset_index())
    readings = pd.DataFrame({
        "measured_at": [ts("2026-04-01 07:00"), ts("2026-04-02 07:00"),
                        ts("2026-04-06 07:00"), ts("2026-04-07 07:00")],
        "temp_c": [11.0, 11.2, 12.0, 12.1],
    })
    legs = build_legs(readings, weather, max_leg_days=3, min_coverage=0.8)
    assert legs.count == 2  # the 4-day leg is dropped
    assert list(legs.n_hours) == [24, 24]

    holed = weather.copy()
    holed.loc[ts("2026-04-06 07:00"):ts("2026-04-06 20:00"), "air_temp"] = np.nan
    assert build_legs(readings, holed, max_leg_days=3, min_coverage=0.8).count == 1


def test_same_hour_readings_collapse_to_one():
    weather = prepare_weather(synthetic_weather(5).reset_index())
    readings = pd.DataFrame({
        "measured_at": [ts("2026-04-01 07:00"), ts("2026-04-01 07:10"), ts("2026-04-02 07:00")],
        "temp_c": [11.0, 11.1, 11.5],
    })
    assert build_legs(readings, weather, max_leg_days=3, min_coverage=0.8).count == 1


# --- staged fitting ------------------------------------------------------

def test_below_the_scale_threshold_uses_the_depth_prior():
    weather = prepare_weather(synthetic_weather(5).reset_index())
    readings = synthetic_readings(weather, REF).head(2)  # 1 leg
    fit = fit_water(readings, weather, "shallow")
    assert load_params()["scale_min_legs"] == 2
    assert fit.method == "prior"
    assert fit.scale == 2.0
    assert fit.coefficients == REF.scaled(2.0)
    assert fit.n_legs == 1


def test_two_legs_already_fit_the_scale():
    weather = prepare_weather(synthetic_weather(5).reset_index())
    readings = synthetic_readings(weather, REF).head(3)
    assert fit_water(readings, weather, "shallow").method == "scale"


def test_scale_fit_moves_toward_the_true_scale():
    weather = prepare_weather(synthetic_weather(15).reset_index())
    readings = synthetic_readings(weather, REF.scaled(2.0), noise=0.05).head(11)  # 10 legs
    fit = fit_water(readings, weather, "unknown")
    assert fit.method == "scale"
    assert fit.n_legs == 10
    assert 1.6 < fit.scale < 2.3
    assert fit.train_mae < 0.15


def test_scale_fit_with_a_tight_prior_stays_near_it():
    weather = prepare_weather(synthetic_weather(15).reset_index())
    readings = synthetic_readings(weather, REF.scaled(2.0), noise=0.05).head(5)
    params = load_params()
    params["lambda"] = 1000.0
    fit = fit_water(readings, weather, "unknown", params=params)
    assert fit.scale == pytest.approx(1.0, abs=0.05)


def test_full_fit_recovers_all_three_coefficients():
    truth = Coefficients(k_air=0.012, k_solar=1.4e-4, k_cool=0.012)
    weather = prepare_weather(synthetic_weather(80).reset_index())
    readings = synthetic_readings(weather, truth, every_hours=12, noise=0.0)
    params = load_params()
    params["lambda"] = 0.001
    fit = fit_water(readings, weather, "unknown", params=params)
    assert fit.method == "full"
    assert fit.coefficients.k_air == pytest.approx(truth.k_air, rel=0.15)
    assert fit.coefficients.k_solar == pytest.approx(truth.k_solar, rel=0.25)
    assert fit.coefficients.k_cool == pytest.approx(truth.k_cool, rel=0.25)


def test_method_can_be_forced():
    weather = prepare_weather(synthetic_weather(15).reset_index())
    readings = synthetic_readings(weather, REF).head(11)
    assert fit_water(readings, weather, "lake", method="prior").method == "prior"


def test_no_readings_is_a_prior_fit():
    weather = prepare_weather(synthetic_weather(2).reset_index())
    empty = pd.DataFrame({"measured_at": pd.Series([], dtype="datetime64[ns, UTC]"), "temp_c": []})
    fit = fit_water(empty, weather, "deep")
    assert (fit.method, fit.scale, fit.n_legs) == ("prior", 0.5, 0)
    assert np.isnan(fit.train_mae)


# --- feed legs -----------------------------------------------------------

def hourly_feed(start, hours, temp=12.0):
    times = [ts(start) + pd.Timedelta(hours=h) for h in hours]
    return pd.DataFrame({"measured_at": times, "temp_c": [temp + 0.01 * h for h in hours], "source": "feed"})


def test_hourly_feed_readings_pair_into_day_long_legs():
    weather = prepare_weather(synthetic_weather(10).reset_index())
    readings = hourly_feed("2026-04-02 00:00", range(96))
    legs = build_legs(readings, weather, max_leg_days=3, min_coverage=0.8)
    # 72 readings have a partner exactly a day later; the next three pair with
    # the last reading, 23, 22 and 21 hours on.
    assert legs.count == 75
    assert sorted(set(legs.n_hours)) == [21, 22, 23, 24]


def test_a_feed_reading_with_no_partner_21_to_27_hours_later_starts_no_leg():
    weather = prepare_weather(synthetic_weather(10).reset_index())
    readings = hourly_feed("2026-04-02 00:00", [*range(24), *range(54, 78)])
    legs = build_legs(readings, weather, max_leg_days=3, min_coverage=0.8)
    assert legs.count == 6
    assert min(legs.n_hours) >= 21 and max(legs.n_hours) <= 27


def test_the_partner_is_the_reading_closest_to_a_day_later():
    weather = prepare_weather(synthetic_weather(10).reset_index())
    readings = hourly_feed("2026-04-02 00:00", [0, 22, 25, 26])
    legs = build_legs(readings, weather, max_leg_days=3, min_coverage=0.8)
    assert legs.count == 1 and list(legs.n_hours) == [25]


def test_a_mixed_water_fits_on_manual_legs_and_feed_legs():
    weather = prepare_weather(synthetic_weather(12).reset_index())
    manual = pd.DataFrame({
        "measured_at": [ts("2026-04-01 07:00") + pd.Timedelta(days=d) for d in range(4)],
        "temp_c": [11.0, 11.2, 11.3, 11.5], "source": "manual",
    })
    feed = hourly_feed("2026-04-06 00:00", range(72))
    readings = pd.concat([manual, feed], ignore_index=True)
    legs = build_legs(readings, weather, max_leg_days=3, min_coverage=0.8)
    assert legs.count == 3 + 51
    assert list(legs.n_hours[:3]) == [24, 24, 24]


def test_feed_legs_need_weather_like_any_leg():
    weather = prepare_weather(synthetic_weather(10).reset_index())
    weather.loc[ts("2026-04-02 12:00"):ts("2026-04-03 12:00"), "air_temp"] = np.nan
    readings = hourly_feed("2026-04-02 00:00", range(48))
    assert build_legs(readings, weather, max_leg_days=3, min_coverage=0.8).count == 0

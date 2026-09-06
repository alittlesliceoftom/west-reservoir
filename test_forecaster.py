"""Tests for WaterTempForecaster (3-term physics model)"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from forecaster import WaterTempForecaster, WEATHER_COLUMNS


# What predict_forward hands its callers. Asserted as a set: order is not part
# of the contract here, and a spurious order failure sends people looking in
# the wrong place.
PREDICT_FORWARD_COLUMNS = {
    "target_datetime", "horizon_days", "water_temp", "has_weather"
}


def _make_hourly_weather(
    start: datetime,
    n_hours: int,
    air_temp: float = 10.0,
    shortwave: float = 0.0,
    cloud: float = 100.0,
) -> pd.DataFrame:
    """Build a constant-value hourly weather DataFrame."""
    return pd.DataFrame({
        "datetime": [start + timedelta(hours=i) for i in range(n_hours)],
        "air_temp": [air_temp] * n_hours,
        "shortwave_radiation": [shortwave] * n_hours,
        "cloud_cover": [cloud] * n_hours,
    })


class TestStep:
    """
    The physics itself: one hour at a time, and hours chained together.

    _step and _simulate_period are the same model at two scales, so they are
    tested together - a change to the step equation should surface here once,
    not in three classes that have to be read side by side to see the whole
    model.
    """

    def _air_only(self, air_temp, n_hours=24):
        """Air conduction isolated: no solar, full cloud so no cooling term."""
        return _make_hourly_weather(
            datetime(2026, 1, 1, 7), n_hours,
            air_temp=air_temp, shortwave=0.0, cloud=100.0,
        )[list(WEATHER_COLUMNS)]

    def test_air_term_only(self):
        # k_air=0.1, T_air=20, T_water=10 → ΔT = 0.1 * 10 = 1.0
        result = WaterTempForecaster._step(
            water=10.0, air_temp=20.0, irradiance=0.0, cloud_cover=100.0,
            k_air=0.1, k_solar=0.0, k_cool=0.0,
        )
        assert result == pytest.approx(11.0)

    def test_solar_only(self):
        # k_solar=0.001, I=500 → ΔT = 0.001 * 500 = 0.5
        result = WaterTempForecaster._step(
            water=10.0, air_temp=10.0, irradiance=500.0, cloud_cover=100.0,
            k_air=0.0, k_solar=0.001, k_cool=0.0,
        )
        assert result == pytest.approx(10.5)

    def test_cool_only_clear_sky(self):
        # k_cool=0.05, cloud=0% → clearness=1.0 → ΔT = -0.05
        result = WaterTempForecaster._step(
            water=10.0, air_temp=10.0, irradiance=0.0, cloud_cover=0.0,
            k_air=0.0, k_solar=0.0, k_cool=0.05,
        )
        assert result == pytest.approx(9.95)

    def test_cool_zero_under_overcast(self):
        # Full cloud → clearness=0 → cooling term=0
        result = WaterTempForecaster._step(
            water=10.0, air_temp=10.0, irradiance=0.0, cloud_cover=100.0,
            k_air=0.0, k_solar=0.0, k_cool=0.05,
        )
        assert result == pytest.approx(10.0)

    def test_all_three_terms_combined(self):
        # ΔT = 0.1*(20-10) + 0.001*500 - 0.05*(1 - 50/100)
        #    = 1.0       + 0.5        - 0.025
        #    = 1.475
        result = WaterTempForecaster._step(
            water=10.0, air_temp=20.0, irradiance=500.0, cloud_cover=50.0,
            k_air=0.1, k_solar=0.001, k_cool=0.05,
        )
        assert result == pytest.approx(11.475)

    @pytest.mark.parametrize(
        "air_temp, expected",
        [
            pytest.param(10.0, (10.0, 10.0), id="no_change_when_air_equals_water"),
            pytest.param(15.0, (10.0, 15.0), id="warms_towards_warmer_air"),
            pytest.param(5.0, (5.0, 10.0), id="cools_towards_cooler_air"),
        ],
    )
    def test_water_moves_towards_the_air_and_stops_there(self, air_temp, expected):
        """
        Direction over 24 hours of air conduction: towards the air, never past
        it. The endpoints of the expected range are exclusive except in the
        equal case, where the water must not move at all.
        """
        f = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        result = f._simulate_period(10.0, self._air_only(air_temp))

        low, high = expected
        if low == high:
            assert result == pytest.approx(low)
        else:
            assert low < result < high

    def test_constant_inputs_known_result(self):
        # 5 hours of: T_air=20, T_water_start=10, I=0, cloud=100, k_air=0.1
        # After each hour: water += 0.1 * (20 - water)
        # h1: 10 + 1.0 = 11.0
        # h2: 11 + 0.9 = 11.9
        # h3: 11.9 + 0.81 = 12.71
        # h4: 12.71 + 0.729 = 13.439
        # h5: 13.439 + 0.6561 = 14.0951
        f = WaterTempForecaster(k_air=0.1, k_solar=0.0, k_cool=0.0)
        weather = _make_hourly_weather(
            datetime(2026, 1, 1, 7), n_hours=5,
            air_temp=20.0, shortwave=0.0, cloud=100.0,
        )
        f.set_hourly_weather(weather)
        slice_df = f._get_weather_for_period(
            datetime(2026, 1, 1, 7), datetime(2026, 1, 1, 12)
        )
        result = f._simulate_period(10.0, slice_df)
        assert result == pytest.approx(14.0951, abs=1e-4)

    def test_empty_slice_returns_start(self):
        f = WaterTempForecaster()
        weather = _make_hourly_weather(datetime(2026, 1, 1, 7), n_hours=3)
        f.set_hourly_weather(weather)
        slice_df = f._get_weather_for_period(
            datetime(2027, 1, 1), datetime(2027, 1, 2)
        )
        result = f._simulate_period(15.0, slice_df)
        assert result == 15.0


class TestSetHourlyWeather:
    def test_rejects_missing_columns(self):
        f = WaterTempForecaster()
        bad = pd.DataFrame({"datetime": [datetime(2026, 1, 1)], "air_temp": [10.0]})
        with pytest.raises(ValueError, match="shortwave_radiation"):
            f.set_hourly_weather(bad)


class TestExplainPrediction:
    """The transparency breakdown, including the degenerate empty period."""

    def test_breakdown_columns_present(self):
        f = WaterTempForecaster(k_air=0.1, k_solar=0.001, k_cool=0.05)
        weather = _make_hourly_weather(
            datetime(2026, 1, 1, 7), n_hours=3,
            air_temp=15.0, shortwave=200.0, cloud=20.0,
        )
        result = f.explain_prediction(
            current_water_temp=10.0,
            weather_slice=weather[["air_temp", "shortwave_radiation", "cloud_cover"]],
        )
        assert result["hours_simulated"] == 3
        assert len(result["hourly_breakdown"]) == 3
        first = result["hourly_breakdown"][0]
        assert "dT_air" in first
        assert "dT_solar" in first
        assert "dT_cool" in first
        # k_solar * 200 = 0.2
        assert first["dT_solar"] == pytest.approx(0.2)
        # k_cool * (1 - 20/100) = 0.05 * 0.8 = 0.04 → dT_cool = -0.04
        assert first["dT_cool"] == pytest.approx(-0.04)

    def test_empty_weather_slice_returns_zero_hours(self):
        f = WaterTempForecaster(k_air=0.02)
        result = f.explain_prediction(
            12.0, pd.DataFrame(columns=list(WEATHER_COLUMNS))
        )
        assert result["hours_simulated"] == 0
        assert result["predicted_water_temp"] == pytest.approx(12.0)
        assert result["hourly_breakdown"] == []

    def test_none_weather_slice_returns_zero_hours(self):
        f = WaterTempForecaster(k_air=0.02)
        result = f.explain_prediction(12.0, None)
        assert result["hours_simulated"] == 0
        assert result["predicted_water_temp"] == pytest.approx(12.0)


class TestFit:
    """Recovery of planted coefficients on synthetic data."""

    def test_fit_recovers_coefficients(self):
        rng = np.random.default_rng(42)
        # Plant true coefficients
        true_k_air, true_k_solar, true_k_cool = 0.03, 4e-4, 0.02
        plant = WaterTempForecaster(true_k_air, true_k_solar, true_k_cool)

        # Build 60 days of synthetic hourly weather
        start = datetime(2026, 1, 1, 0)
        n_hours = 60 * 24
        # Daily air temp cycle: mean 12°C, ±5°C
        hours = np.arange(n_hours)
        air = 12.0 + 5.0 * np.sin(2 * np.pi * (hours % 24 - 9) / 24) \
                  + rng.normal(0, 0.5, n_hours)
        # Solar: peaked at noon, zero at night
        hour_of_day = hours % 24
        solar = np.where(
            (hour_of_day >= 6) & (hour_of_day <= 18),
            500.0 * np.sin(np.pi * (hour_of_day - 6) / 12),
            0.0,
        )
        # Cloud cover varies day-to-day
        cloud = np.repeat(rng.uniform(0, 100, 60), 24)

        weather = pd.DataFrame({
            "datetime": [start + timedelta(hours=int(h)) for h in hours],
            "air_temp": air,
            "shortwave_radiation": solar,
            "cloud_cover": cloud,
        })
        plant.set_hourly_weather(weather)

        # Walk the simulation forward 7am-to-7am, recording daily measurements
        dates = []
        temps = []
        water = 10.0
        date = pd.Timestamp(2026, 1, 1)
        dates.append(date)
        temps.append(water)
        for day in range(1, 60):
            slice_df = plant._get_weather_for_period(
                date.replace(hour=7),
                (date + pd.Timedelta(days=1)).replace(hour=7),
            )
            water = plant._simulate_period(water, slice_df)
            date = date + pd.Timedelta(days=1)
            dates.append(date)
            temps.append(water)

        measurements = pd.DataFrame({
            "date": dates,
            "water_temp": temps,
            "source": ["MEASURED"] * len(dates),
        })

        # Fit a fresh forecaster
        fit = WaterTempForecaster()  # default starting guesses
        fit.set_hourly_weather(weather)
        fit.fit(measurements)

        # Should recover within ~25% relative (loose because of noise + bounds)
        assert fit.k_air == pytest.approx(true_k_air, rel=0.25)
        assert fit.k_solar == pytest.approx(true_k_solar, rel=0.30)
        assert fit.k_cool == pytest.approx(true_k_cool, rel=0.30)

    def test_fit_respects_k_cool_nonneg(self):
        """k_cool must never go negative (would mean clear sky heats water)."""
        f = WaterTempForecaster()
        # Insufficient data → fit returns early without changing coefficients
        empty = pd.DataFrame(columns=["date", "water_temp", "source"])
        f.set_hourly_weather(_make_hourly_weather(datetime(2026, 1, 1), n_hours=24))
        f.fit(empty)
        assert f.k_cool >= 0


class TestPredictForward:
    """Forward simulation API: one pass, checkpointed at each target."""

    def _fitted(self, n_hours=200, air_temp=15.0):
        f = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        f.set_hourly_weather(
            _make_hourly_weather(datetime(2026, 3, 1, 0), n_hours, air_temp=air_temp)
        )
        return f

    def test_single_target_matches_manual_simulation(self):
        f = self._fitted()
        start = datetime(2026, 3, 1, 7)
        result = f.predict_forward(start, 10.0, days_ahead=1)

        assert set(result.columns) == PREDICT_FORWARD_COLUMNS
        assert len(result) == 1
        assert result.loc[0, "horizon_days"] == 1
        assert result.loc[0, "target_datetime"] == pd.Timestamp(2026, 3, 2, 7)
        assert bool(result.loc[0, "has_weather"]) is True

        expected = f._simulate_period(
            10.0, f._get_weather_for_period(start, datetime(2026, 3, 2, 7))
        )
        assert result.loc[0, "water_temp"] == pytest.approx(expected)

    def test_multi_target_equals_chained_single_days(self):
        """A 3-day forecast is one continuous run, equal to 3 chained 1-day runs."""
        f = self._fitted()
        start = datetime(2026, 3, 1, 7)
        result = f.predict_forward(start, 10.0, days_ahead=3)

        assert list(result["horizon_days"]) == [1, 2, 3]

        water = 10.0
        for day in range(3):
            leg_start = datetime(2026, 3, 1 + day, 7)
            leg_end = datetime(2026, 3, 2 + day, 7)
            water = f._simulate_period(water, f._get_weather_for_period(leg_start, leg_end))
            assert result.loc[day, "water_temp"] == pytest.approx(water)

    def test_start_datetime_normalised_to_measurement_hour(self):
        """A midnight or mid-afternoon anchor is snapped to 7am."""
        f = self._fitted()
        at_seven = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, days_ahead=2)
        at_midnight = f.predict_forward(datetime(2026, 3, 1, 0), 10.0, days_ahead=2)
        pd.testing.assert_frame_equal(at_seven, at_midnight)

    def test_explicit_irregular_dates(self):
        """Targets may be an irregular sequence of dates, not just 1..n."""
        f = self._fitted()
        start = datetime(2026, 3, 1, 7)
        result = f.predict_forward(
            start, 10.0, target_dates=[datetime(2026, 3, 2), datetime(2026, 3, 5)]
        )
        assert list(result["horizon_days"]) == [1, 4]

        water = 10.0
        water = f._simulate_period(
            water, f._get_weather_for_period(datetime(2026, 3, 1, 7), datetime(2026, 3, 2, 7))
        )
        assert result.loc[0, "water_temp"] == pytest.approx(water)
        water = f._simulate_period(
            water, f._get_weather_for_period(datetime(2026, 3, 2, 7), datetime(2026, 3, 5, 7))
        )
        assert result.loc[1, "water_temp"] == pytest.approx(water)

    def test_targets_sorted_ascending(self):
        f = self._fitted()
        result = f.predict_forward(
            datetime(2026, 3, 1, 7), 10.0,
            target_dates=[datetime(2026, 3, 4), datetime(2026, 3, 2)],
        )
        assert list(result["horizon_days"]) == [1, 3]

    def test_duplicate_targets_carry_the_temperature_through(self):
        """
        A repeated target is a zero-length leg: no time passes, so the value
        is unchanged. It must not be mistaken for a missing-weather gap, which
        would poison the rest of the chain with NaN.
        """
        f = self._fitted()
        result = f.predict_forward(
            datetime(2026, 3, 1, 7), 10.0,
            target_dates=[datetime(2026, 3, 2), datetime(2026, 3, 2)],
        )
        assert len(result) == 2
        assert bool(result.loc[0, "has_weather"]) is True
        assert bool(result.loc[1, "has_weather"]) is True
        assert result.loc[1, "water_temp"] == pytest.approx(result.loc[0, "water_temp"])

    def test_duplicate_target_does_not_break_a_longer_chain(self):
        """The regression this fix exists to prevent."""
        f = self._fitted()
        result = f.predict_forward(
            datetime(2026, 3, 1, 7), 10.0,
            target_dates=[
                datetime(2026, 3, 2), datetime(2026, 3, 2), datetime(2026, 3, 3),
            ],
        )
        assert not result["water_temp"].isna().any(), (
            "a duplicate date poisoned the chain with NaN"
        )

    def test_nan_when_weather_runs_out(self):
        """No fabrication: legs beyond weather coverage are NaN with has_weather False."""
        f = self._fitted(n_hours=30)  # covers ~1 day past the 7am anchor
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, days_ahead=3)

        assert len(result) == 3
        assert not np.isnan(result.loc[0, "water_temp"])
        assert bool(result.loc[0, "has_weather"]) is True
        assert np.isnan(result.loc[2, "water_temp"])
        assert bool(result.loc[2, "has_weather"]) is False

    def test_nan_propagates_once_coverage_lost(self):
        f = self._fitted(n_hours=30)
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, days_ahead=4)
        tail = result[result["horizon_days"] >= 3]["water_temp"]
        assert tail.isna().all()

    def test_no_weather_set_returns_all_nan(self):
        f = WaterTempForecaster()
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, days_ahead=2)
        assert result["water_temp"].isna().all()
        assert not result["has_weather"].any()

    def test_zero_targets_returns_empty_frame(self):
        f = self._fitted()
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, days_ahead=0)
        assert result.empty
        assert set(result.columns) == PREDICT_FORWARD_COLUMNS

    # Argument validation: exactly one of days_ahead / target_dates.

    def test_neither_argument_raises(self):
        f = self._fitted()
        with pytest.raises(ValueError, match="exactly one"):
            f.predict_forward(datetime(2026, 3, 1, 7), 10.0)

    def test_both_arguments_raise(self):
        f = self._fitted()
        with pytest.raises(ValueError, match="exactly one"):
            f.predict_forward(
                datetime(2026, 3, 1, 7), 10.0,
                days_ahead=2, target_dates=[datetime(2026, 3, 2)],
            )

    def test_days_ahead_zero_is_not_treated_as_missing(self):
        """0 is falsy but explicitly given - must not trip the validation."""
        f = self._fitted()
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, days_ahead=0)
        assert result.empty

    def test_empty_target_dates_is_not_treated_as_missing(self):
        f = self._fitted()
        result = f.predict_forward(datetime(2026, 3, 1, 7), 10.0, target_dates=[])
        assert result.empty


def _legacy_fill(forecaster, temperatures):
    """
    Verbatim copy of the pre-refactor WaterTempForecaster.predict().
    Reference implementation for the equivalence test. Do not "improve" it.
    """
    result = temperatures.copy()
    result = result.sort_values("date").reset_index(drop=True)

    for i in range(len(result)):
        if result.loc[i, "source"] != "AIR_ONLY":
            continue
        if i == 0:
            continue

        prev_row = result.iloc[i - 1]
        curr_date = result.loc[i, "date"]
        current_water_temp = prev_row["water_temp"]

        start_dt = pd.Timestamp(prev_row["date"]).replace(hour=forecaster.MEASUREMENT_HOUR)
        end_dt = pd.Timestamp(curr_date).replace(hour=forecaster.MEASUREMENT_HOUR)

        slice_df = forecaster._get_weather_for_period(start_dt, end_dt)
        if not slice_df.empty:
            predicted = forecaster._simulate_period(current_water_temp, slice_df)
            result.loc[i, "water_temp"] = predicted
            result.loc[i, "source"] = "PREDICTED"

    return result


class TestFillPredictions:
    """fill_predictions must be behaviour-identical to the old predict()."""

    def _fitted(self, n_hours=400):
        f = WaterTempForecaster(k_air=0.02, k_solar=1e-4, k_cool=0.005)
        f.set_hourly_weather(
            _make_hourly_weather(
                datetime(2026, 3, 1, 0), n_hours, air_temp=15.0, shortwave=200.0, cloud=40.0
            )
        )
        return f

    def _frame(self, sources, start_day=1, water_start=10.0):
        dates = [datetime(2026, 3, start_day + i) for i in range(len(sources))]
        temps = [water_start if s == "MEASURED" else float("nan") for s in sources]
        return pd.DataFrame({"date": dates, "water_temp": temps, "source": sources})

    def test_matches_legacy_simple_run(self):
        f = self._fitted()
        df = self._frame(["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_with_interleaved_measurements(self):
        """Chain must re-anchor on each MEASURED row."""
        f = self._fitted()
        df = self._frame(
            ["MEASURED", "AIR_ONLY", "MEASURED", "AIR_ONLY", "AIR_ONLY", "MEASURED"]
        )
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_when_air_only_is_first_row(self):
        """Row 0 has no anchor and must be left untouched."""
        f = self._fitted()
        df = self._frame(["AIR_ONLY", "MEASURED", "AIR_ONLY"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_with_date_gaps(self):
        """Non-consecutive dates: a leg may span several days."""
        f = self._fitted()
        df = pd.DataFrame({
            "date": [
                datetime(2026, 3, 1), datetime(2026, 3, 2),
                datetime(2026, 3, 6), datetime(2026, 3, 7),
            ],
            "water_temp": [10.0, float("nan"), float("nan"), float("nan")],
            "source": ["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"],
        })
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_duplicate_dates_no_longer_poison_the_chain(self):
        """
        DELIBERATE DIVERGENCE FROM LEGACY.

        The old predict() treated a duplicate date as a missing-weather gap,
        so it produced NaN there and every following row inherited the NaN:

            legacy: [10.0, 12.25, nan, nan]
            now:    [10.0, 12.25, 12.25, 13.63]

        A duplicate date means no time passed, not that data is missing, so
        the temperature carries through and the chain survives. Unreachable
        from the dashboard, which deduplicates dates first, but wrong is wrong.
        """
        f = self._fitted()
        df = pd.DataFrame({
            "date": [
                datetime(2026, 3, 1), datetime(2026, 3, 2), datetime(2026, 3, 2),
                datetime(2026, 3, 3),
            ],
            "water_temp": [10.0, float("nan"), float("nan"), float("nan")],
            "source": ["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"],
        })

        result = f.fill_predictions(df)
        legacy = _legacy_fill(f, df)

        # Every row is predicted, with no NaN anywhere.
        assert not result["water_temp"].isna().any()
        assert list(result["source"]) == [
            "MEASURED", "PREDICTED", "PREDICTED", "PREDICTED"
        ]
        # The duplicate carries the previous value through unchanged.
        assert result.loc[2, "water_temp"] == pytest.approx(result.loc[1, "water_temp"])
        # ...and the legacy implementation genuinely did lose the chain here.
        assert legacy["water_temp"].isna().any()

    def test_matches_legacy_when_weather_runs_out(self):
        f = self._fitted(n_hours=60)
        df = self._frame(["MEASURED", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY", "AIR_ONLY"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_matches_legacy_unsorted_input(self):
        f = self._fitted()
        df = self._frame(["MEASURED", "AIR_ONLY", "AIR_ONLY"]).iloc[::-1].reset_index(drop=True)
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

    def test_all_measured_is_unchanged(self):
        f = self._fitted()
        df = self._frame(["MEASURED", "MEASURED", "MEASURED"])
        pd.testing.assert_frame_equal(f.fill_predictions(df), _legacy_fill(f, df))

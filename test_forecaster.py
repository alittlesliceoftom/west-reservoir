"""Tests for WaterTempForecaster (3-term physics model)"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from forecaster import WaterTempForecaster


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
    """Single-hour step physics."""

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


class TestSetHourlyWeather:
    def test_rejects_missing_columns(self):
        f = WaterTempForecaster()
        bad = pd.DataFrame({"datetime": [datetime(2026, 1, 1)], "air_temp": [10.0]})
        with pytest.raises(ValueError, match="shortwave_radiation"):
            f.set_hourly_weather(bad)

    def test_legacy_air_only_accepts(self):
        f = WaterTempForecaster()
        air_only = pd.DataFrame({
            "datetime": [datetime(2026, 1, 1, 7), datetime(2026, 1, 1, 8)],
            "air_temp": [10.0, 11.0],
        })
        f.set_hourly_air_temps(air_only)
        assert f.hourly_weather is not None
        assert (f.hourly_weather["shortwave_radiation"] == 0.0).all()
        assert (f.hourly_weather["cloud_cover"] == 100.0).all()


class TestSimulatePeriod:
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


class TestExplainPrediction:
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

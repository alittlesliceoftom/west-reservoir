"""Tests for data loading and processing functions"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from data import interpolate_to_hourly, DataLoadError
from app import combine_hourly_temps


class TestInterpolateToHourly:
    """Tests for interpolate_to_hourly function"""

    def test_empty_dataframe(self):
        """Empty input returns empty output"""
        df = pd.DataFrame(columns=["datetime", "air_temp"])
        result = interpolate_to_hourly(df)
        assert result.empty

    def test_single_row(self):
        """Single row returns single row (no interpolation possible)"""
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        result = interpolate_to_hourly(df)
        assert len(result) == 1
        assert result.iloc[0]["air_temp"] == 10.0

    def test_3hourly_to_hourly(self):
        """3-hourly data interpolates to hourly"""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 15:00"),
                pd.Timestamp("2024-01-15 18:00"),
            ],
            "air_temp": [10.0, 13.0, 16.0]
        })
        result = interpolate_to_hourly(df)

        # Should have 7 rows: 12, 13, 14, 15, 16, 17, 18
        assert len(result) == 7

        # Check interpolated values
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 13:00")]["air_temp"].iloc[0] == 11.0
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 14:00")]["air_temp"].iloc[0] == 12.0
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 16:00")]["air_temp"].iloc[0] == 14.0

    def test_preserves_original_values(self):
        """Original values are preserved exactly"""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 15:00"),
            ],
            "air_temp": [10.5, 13.5]
        })
        result = interpolate_to_hourly(df)

        # Original timestamps should have exact values
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 12:00")]["air_temp"].iloc[0] == 10.5
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 15:00")]["air_temp"].iloc[0] == 13.5


class TestCombineHourlyTemps:
    """Tests for combine_hourly_temps function"""

    def test_empty_both(self):
        """Both empty returns empty"""
        hist = pd.DataFrame(columns=["datetime", "air_temp"])
        fore = pd.DataFrame(columns=["datetime", "air_temp"])
        result = combine_hourly_temps(hist, fore)
        assert result.empty

    def test_empty_historical(self):
        """Empty historical returns forecast only"""
        hist = pd.DataFrame(columns=["datetime", "air_temp"])
        fore = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        result = combine_hourly_temps(hist, fore)
        assert len(result) == 1

    def test_empty_forecast(self):
        """Empty forecast returns historical only"""
        hist = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        fore = pd.DataFrame(columns=["datetime", "air_temp"])
        result = combine_hourly_temps(hist, fore)
        assert len(result) == 1

    def test_historical_takes_precedence(self):
        """Historical data overrides forecast for overlapping times"""
        hist = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 13:00"),
            ],
            "air_temp": [10.0, 11.0]
        })
        fore = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 13:00"),  # Overlaps
                pd.Timestamp("2024-01-15 14:00"),
            ],
            "air_temp": [99.0, 12.0]  # 99.0 should be ignored
        })
        result = combine_hourly_temps(hist, fore)

        # Should have 3 rows
        assert len(result) == 3

        # Historical value should be used for 13:00
        val_13 = result[result["datetime"] == pd.Timestamp("2024-01-15 13:00")]["air_temp"].iloc[0]
        assert val_13 == 11.0

    def test_no_overlap(self):
        """Non-overlapping data combines correctly"""
        hist = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 10:00"),
                pd.Timestamp("2024-01-15 11:00"),
            ],
            "air_temp": [8.0, 9.0]
        })
        fore = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 14:00"),
                pd.Timestamp("2024-01-15 15:00"),
            ],
            "air_temp": [12.0, 13.0]
        })
        result = combine_hourly_temps(hist, fore)

        # Should have gap interpolated: 10, 11, 12, 13, 14, 15 = 6 rows
        assert len(result) == 6

    def test_sorted_output(self):
        """Output is sorted by datetime"""
        hist = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        fore = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 15:00")],
            "air_temp": [13.0]
        })
        result = combine_hourly_temps(hist, fore)

        # Verify sorted
        assert result["datetime"].is_monotonic_increasing


class TestForecaster:
    """Tests for WaterTempForecaster"""

    def test_simulate_24h_no_change_when_equal(self):
        """No change when water and air temps are equal"""
        from forecaster import WaterTempForecaster

        f = WaterTempForecaster(heat_transfer_coeff=0.02)
        # 24 hours of 10C air temp
        hourly_temps = [10.0] * 24
        result = f._simulate_24h(10.0, hourly_temps)

        # Should stay at 10.0
        assert abs(result - 10.0) < 0.001

    def test_simulate_24h_warms_up(self):
        """Water warms when air is warmer"""
        from forecaster import WaterTempForecaster

        f = WaterTempForecaster(heat_transfer_coeff=0.02)
        hourly_temps = [15.0] * 24  # Air at 15C
        result = f._simulate_24h(10.0, hourly_temps)  # Water starts at 10C

        # Should be warmer than 10 but cooler than 15
        assert result > 10.0
        assert result < 15.0

    def test_simulate_24h_cools_down(self):
        """Water cools when air is cooler"""
        from forecaster import WaterTempForecaster

        f = WaterTempForecaster(heat_transfer_coeff=0.02)
        hourly_temps = [5.0] * 24  # Air at 5C
        result = f._simulate_24h(10.0, hourly_temps)  # Water starts at 10C

        # Should be cooler than 10 but warmer than 5
        assert result < 10.0
        assert result > 5.0

    def test_explain_prediction_structure(self):
        """explain_prediction returns expected structure"""
        from forecaster import WaterTempForecaster

        f = WaterTempForecaster(heat_transfer_coeff=0.02)
        hourly_temps = [10.0] * 24
        result = f.explain_prediction(12.0, hourly_temps)

        assert "current_water_temp" in result
        assert "hours_simulated" in result
        assert "predicted_water_temp" in result
        assert "hourly_breakdown" in result
        assert result["hours_simulated"] == 24

    def test_explain_prediction_empty_temps(self):
        """explain_prediction handles empty temps gracefully"""
        from forecaster import WaterTempForecaster

        f = WaterTempForecaster()
        result = f.explain_prediction(12.0, [])

        assert result["hours_simulated"] == 0
        assert result["predicted_water_temp"] == 12.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

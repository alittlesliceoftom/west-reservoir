"""Test combine_hourly_temps handles timezone mismatches correctly."""

import pandas as pd
import pytest
from datetime import datetime, timedelta

# Import the function we're testing
import sys
sys.path.insert(0, '.')

# We need to mock streamlit before importing app
from unittest.mock import MagicMock
sys.modules['streamlit'] = MagicMock()

from app import combine_hourly_temps


class TestCombineHourlyTemps:
    """Test timezone handling in combine_hourly_temps."""

    def test_all_naive_datetimes(self):
        """Basic case: all inputs are timezone-naive."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=24, freq="h"),
            "air_temp": [10.0] * 24
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02 00:00", periods=24, freq="h"),
            "air_temp": [12.0] * 24
        })

        result = combine_hourly_temps(hist, fore)

        assert len(result) == 48
        assert result["datetime"].dt.tz is None

    def test_historical_tz_aware_forecast_naive(self):
        """Historical has timezone, forecast is naive - should not crash."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=24, freq="h", tz="UTC"),
            "air_temp": [10.0] * 24
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02 00:00", periods=24, freq="h"),
            "air_temp": [12.0] * 24
        })

        result = combine_hourly_temps(hist, fore)

        assert len(result) == 48
        assert result["datetime"].dt.tz is None  # Should be normalized to naive

    def test_forecast_tz_aware_historical_naive(self):
        """Forecast has timezone, historical is naive - should not crash."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=24, freq="h"),
            "air_temp": [10.0] * 24
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02 00:00", periods=24, freq="h", tz="UTC"),
            "air_temp": [12.0] * 24
        })

        result = combine_hourly_temps(hist, fore)

        assert len(result) == 48
        assert result["datetime"].dt.tz is None

    def test_gap_fill_tz_aware(self):
        """Gap fill data has timezone, others naive - the production bug."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=12, freq="h"),
            "air_temp": [10.0] * 12
        })
        # Gap from 12:00 to 18:00
        gap_fill = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 12:00", periods=6, freq="h", tz="UTC"),
            "air_temp": [11.0] * 6
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 18:00", periods=12, freq="h"),
            "air_temp": [12.0] * 12
        })

        result = combine_hourly_temps(hist, fore, gap_fill=gap_fill)

        assert not result.empty
        assert result["datetime"].dt.tz is None

    def test_all_tz_aware_different_timezones(self):
        """All inputs have different timezones - should normalize."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=12, freq="h", tz="UTC"),
            "air_temp": [10.0] * 12
        })
        gap_fill = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 12:00", periods=6, freq="h", tz="Europe/London"),
            "air_temp": [11.0] * 6
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 18:00", periods=12, freq="h", tz="America/New_York"),
            "air_temp": [12.0] * 12
        })

        result = combine_hourly_temps(hist, fore, gap_fill=gap_fill)

        assert not result.empty
        assert result["datetime"].dt.tz is None

    def test_empty_gap_fill_no_crash(self):
        """Empty gap_fill should not cause dtype issues."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=24, freq="h"),
            "air_temp": [10.0] * 24
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02 00:00", periods=24, freq="h"),
            "air_temp": [12.0] * 24
        })
        empty_gap = pd.DataFrame(columns=["datetime", "air_temp"])

        result = combine_hourly_temps(hist, fore, gap_fill=empty_gap)

        assert len(result) == 48

    def test_none_gap_fill(self):
        """None gap_fill should work fine."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=24, freq="h"),
            "air_temp": [10.0] * 24
        })
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-02 00:00", periods=24, freq="h"),
            "air_temp": [12.0] * 24
        })

        result = combine_hourly_temps(hist, fore, gap_fill=None)

        assert len(result) == 48


    def test_different_datetime_resolutions(self):
        """Different datetime resolutions (ns vs us) - potential prod issue."""
        # datetime64[ns]
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=12, freq="h"),
            "air_temp": [10.0] * 12
        })
        # Force to datetime64[us]
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 18:00", periods=12, freq="h").astype("datetime64[us]"),
            "air_temp": [12.0] * 12
        })

        result = combine_hourly_temps(hist, fore)

        assert not result.empty
        assert result["datetime"].dtype == "datetime64[s]"

    def test_object_dtype_datetime(self):
        """Datetime stored as object dtype - edge case."""
        hist = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 00:00", periods=12, freq="h"),
            "air_temp": [10.0] * 12
        })
        # Force to object dtype
        fore = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01 18:00", periods=12, freq="h").astype(object),
            "air_temp": [12.0] * 12
        })

        result = combine_hourly_temps(hist, fore)

        assert not result.empty


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

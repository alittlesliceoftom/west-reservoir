"""Weather actuals: storage round trip and the ingestion entry point."""

from datetime import datetime
from unittest.mock import patch

import duckdb
import pandas as pd
import pytest

import ingest
from data import DataLoadError
from forecast_storage import ForecastStorage, ForecastStorageError


def _storage():
    storage = ForecastStorage()
    storage._conn = duckdb.connect(":memory:")
    storage.initialize_schema()
    return storage


def _frame(start="2026-05-01", hours=3, air=None):
    return pd.DataFrame({
        "datetime": pd.date_range(start, periods=hours, freq="h"),
        "air_temp": air or [10.0, 11.0, 12.0],
        "shortwave_radiation": [0.0, 5.0, 50.0],
        "cloud_cover": [80.0, 70.0, 60.0],
    })


class TestStoreWeatherActuals:
    """One truth per hour, and a revision has to win."""

    def test_round_trip(self):
        storage = _storage()
        storage.store_weather_actuals(_frame())

        out = storage.get_weather_actuals()

        assert len(out) == 3
        assert out["air_temp"].tolist() == [10.0, 11.0, 12.0]
        assert out["datetime"].iloc[0] == pd.Timestamp("2026-05-01 00:00")

    def test_a_revision_overwrites(self):
        """ERA5T corrects the recent past; re-ingesting must not duplicate it."""
        storage = _storage()
        storage.store_weather_actuals(_frame())
        storage.store_weather_actuals(_frame(air=[99.0, 11.0, 12.0]))

        out = storage.get_weather_actuals()

        assert len(out) == 3, "the hour is the key; re-ingesting must not append"
        assert out["air_temp"].iloc[0] == 99.0

    def test_a_null_measure_leaves_what_is_stored(self):
        storage = _storage()
        storage.store_weather_actuals(_frame())

        partial = _frame()[["datetime", "air_temp"]]
        partial["air_temp"] = [50.0, 51.0, 52.0]
        storage.store_weather_actuals(partial)

        out = storage.get_weather_actuals()
        assert out["air_temp"].iloc[0] == 50.0
        assert out["cloud_cover"].iloc[0] == 80.0

    def test_date_bounds_filter(self):
        storage = _storage()
        storage.store_weather_actuals(_frame(hours=3))

        out = storage.get_weather_actuals(start_date="2026-05-01 01:00")

        assert len(out) == 2

    def test_empty_frame_is_a_no_op(self):
        storage = _storage()
        storage.store_weather_actuals(pd.DataFrame())
        assert storage.get_weather_actuals().empty

    def test_frame_without_measures_errors(self):
        storage = _storage()
        with pytest.raises(ForecastStorageError, match="none of the measure columns"):
            storage.store_weather_actuals(pd.DataFrame({"datetime": [datetime(2026, 5, 1)]}))

    def test_latest_actual_datetime_is_none_when_empty(self):
        assert _storage().latest_actual_datetime() is None


class TestIngest:
    """The scheduled job."""

    def _archive(self):
        return {
            "hourly_air": pd.DataFrame({
                "datetime": pd.date_range("2026-05-01", periods=3, freq="h"),
                "air_temp": [10.0, 11.0, 12.0],
            }),
            "solar_cloud": pd.DataFrame({
                "datetime": pd.date_range("2026-05-01", periods=3, freq="h"),
                "shortwave_radiation": [0.0, 5.0, 50.0],
                "cloud_cover": [80.0, 70.0, 60.0],
            }),
        }

    @patch("ingest.load_historical_weather")
    def test_merges_air_and_solar_into_one_frame(self, mock_load):
        mock_load.return_value = self._archive()

        frame = ingest.weather_actuals_frame("2026-05-01", "2026-05-01")

        assert list(frame.columns) == [
            "datetime", "air_temp", "shortwave_radiation", "cloud_cover"
        ]
        assert len(frame) == 3

    @patch("ingest.load_historical_weather")
    def test_stores_the_trailing_window(self, mock_load):
        mock_load.return_value = self._archive()
        storage = _storage()

        stored = ingest.ingest(days=10, storage=storage)

        assert stored == 3
        assert len(storage.get_weather_actuals()) == 3

    @patch("ingest.load_historical_weather")
    def test_window_is_ten_days_by_default(self, mock_load):
        """Wider than the 6-day ERA5T preliminary window, with margin."""
        mock_load.return_value = self._archive()

        ingest.ingest(storage=_storage())

        start, end = mock_load.call_args[0]
        assert (end - start).days == 10
        assert ingest.TRAILING_WINDOW_DAYS > ingest.ERA5_PRELIMINARY_DAYS

    @patch("ingest.load_historical_weather")
    def test_main_returns_nonzero_when_the_archive_fails(self, mock_load):
        mock_load.side_effect = DataLoadError("timed out")

        assert ingest.main(["--days", "10"]) == 1

"""Tests for Open-Meteo data loaders (historical + forecast solar/cloud)"""

import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock
import requests

from data import (
    load_historical_solar_cloud,
    load_forecast_solar_cloud,
    _parse_open_meteo_hourly,
    DataLoadError,
)


def _good_payload():
    return {
        "hourly": {
            "time": [
                "2026-01-01T00:00",
                "2026-01-01T01:00",
                "2026-01-01T02:00",
            ],
            "shortwave_radiation": [0.0, 0.0, 0.0],
            "cloud_cover": [80, 75, 60],
        }
    }


class TestParse:
    def test_parses_happy_path(self):
        df = _parse_open_meteo_hourly(_good_payload())
        assert list(df.columns) == ["datetime", "shortwave_radiation", "cloud_cover"]
        assert len(df) == 3
        assert df["cloud_cover"].iloc[1] == 75
        # Sorted
        assert df["datetime"].is_monotonic_increasing

    def test_missing_hourly_block_errors(self):
        with pytest.raises(DataLoadError, match="missing 'hourly'"):
            _parse_open_meteo_hourly({"reason": "no data"})

    def test_missing_field_errors(self):
        payload = {"hourly": {"time": ["2026-01-01T00:00"], "shortwave_radiation": [0]}}
        with pytest.raises(DataLoadError, match="cloud_cover"):
            _parse_open_meteo_hourly(payload)

    def test_drops_nan_rows(self):
        payload = {
            "hourly": {
                "time": ["2026-01-01T00:00", "2026-01-01T01:00"],
                "shortwave_radiation": [100.0, None],
                "cloud_cover": [50, 60],
            }
        }
        df = _parse_open_meteo_hourly(payload)
        assert len(df) == 1

    def test_all_nan_raises(self):
        payload = {
            "hourly": {
                "time": ["2026-01-01T00:00"],
                "shortwave_radiation": [None],
                "cloud_cover": [None],
            }
        }
        with pytest.raises(DataLoadError, match="no valid"):
            _parse_open_meteo_hourly(payload)


class TestHistorical:
    @patch("data.requests.get")
    def test_calls_archive_endpoint(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _good_payload()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = load_historical_solar_cloud(datetime(2026, 1, 1), datetime(2026, 1, 2))

        assert len(df) == 3
        called_url = mock_get.call_args[0][0]
        assert "archive-api.open-meteo.com" in called_url
        params = mock_get.call_args[1]["params"]
        assert params["start_date"] == "2026-01-01"
        assert params["end_date"] == "2026-01-02"
        assert "shortwave_radiation" in params["hourly"]
        assert "cloud_cover" in params["hourly"]

    @patch("data.requests.get")
    def test_timeout_wraps_in_dataloaderror(self, mock_get):
        mock_get.side_effect = requests.exceptions.Timeout()
        with pytest.raises(DataLoadError, match="timed out"):
            load_historical_solar_cloud(datetime(2026, 1, 1), datetime(2026, 1, 2))

    @patch("data.requests.get")
    def test_http_error_wraps_in_dataloaderror(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_get.return_value = mock_resp
        with pytest.raises(DataLoadError, match="Failed to fetch"):
            load_historical_solar_cloud(datetime(2026, 1, 1), datetime(2026, 1, 2))


class TestForecast:
    @patch("data.requests.get")
    def test_calls_forecast_endpoint(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = _good_payload()
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        df = load_forecast_solar_cloud(days=5)

        assert len(df) == 3
        called_url = mock_get.call_args[0][0]
        assert "api.open-meteo.com" in called_url
        assert "forecast" in called_url
        params = mock_get.call_args[1]["params"]
        assert params["forecast_days"] == 5

    @patch("data.requests.get")
    def test_empty_hourly_raises(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"reason": "out of bounds"}
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        with pytest.raises(DataLoadError):
            load_forecast_solar_cloud(days=5)

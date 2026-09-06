"""Tests for the unified Open-Meteo forecast loader (issue #39)"""

import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
import requests

from data import (
    load_forecast_weather,
    daily_from_hourly_forecast,
    build_hourly_weather,
    DataLoadError,
)
from forecaster import WaterTempForecaster


def _payload(hours=3):
    return {
        "hourly": {
            "time": [
                "2026-09-07T%02d:00" % h for h in range(hours)
            ],
            "temperature_2m": [12.0, 12.5, 13.0][:hours],
            "shortwave_radiation": [0.0, 0.0, 50.0][:hours],
            "cloud_cover": [80, 75, 60][:hours],
        }
    }


def _response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status.return_value = None
    return response


class TestLoadForecastWeather:

    def test_returns_all_three_measures_from_one_call(self):
        """The point of the change: one request, every model input."""
        with patch("data.requests.get", return_value=_response(_payload())) as get:
            df = load_forecast_weather(days=5)

        assert list(df.columns) == [
            "datetime", "air_temp", "shortwave_radiation", "cloud_cover"
        ]
        assert len(df) == 3
        assert get.call_count == 1

    def test_requests_air_temperature_alongside_solar_and_cloud(self):
        """Dropping temperature_2m from the query is the silent way to regress."""
        with patch("data.requests.get", return_value=_response(_payload())) as get:
            load_forecast_weather(days=5)

        requested = get.call_args.kwargs["params"]["hourly"]
        assert "temperature_2m" in requested
        assert "shortwave_radiation" in requested
        assert "cloud_cover" in requested

    def test_air_temperature_is_named_air_temp_for_the_rest_of_the_app(self):
        """Open-Meteo calls it temperature_2m; everything downstream says air_temp."""
        with patch("data.requests.get", return_value=_response(_payload())):
            df = load_forecast_weather()

        assert df["air_temp"].tolist() == [12.0, 12.5, 13.0]

    def test_days_is_passed_through_so_the_horizon_can_be_extended(self):
        """Open-Meteo serves 16 days free; OpenWeatherMap capped us at 5."""
        with patch("data.requests.get", return_value=_response(_payload())) as get:
            load_forecast_weather(days=16)

        assert get.call_args.kwargs["params"]["forecast_days"] == 16

    def test_rows_are_hourly_not_three_hourly(self):
        """No interpolation on the way in - that was the OpenWeatherMap tax."""
        with patch("data.requests.get", return_value=_response(_payload())):
            df = load_forecast_weather()

        gaps = df["datetime"].diff().dropna().unique()
        assert list(gaps) == [pd.Timedelta(hours=1)]

    def test_missing_temperature_field_raises(self):
        payload = _payload()
        del payload["hourly"]["temperature_2m"]
        with patch("data.requests.get", return_value=_response(payload)):
            with pytest.raises(DataLoadError, match="temperature_2m"):
                load_forecast_weather()

    def test_timeout_raises_dataloaderror(self):
        with patch("data.requests.get", side_effect=requests.exceptions.Timeout):
            with pytest.raises(DataLoadError, match="timed out"):
                load_forecast_weather()

    def test_request_failure_raises_dataloaderror(self):
        with patch(
            "data.requests.get",
            side_effect=requests.exceptions.RequestException("boom"),
        ):
            with pytest.raises(DataLoadError, match="Failed to fetch forecast weather"):
                load_forecast_weather()

    def test_empty_hourly_block_raises(self):
        with patch("data.requests.get", return_value=_response({"hourly": {}})):
            with pytest.raises(DataLoadError):
                load_forecast_weather()


class TestDailyFromHourlyForecast:

    def _hourly(self, day, temps):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(day) + pd.Timedelta(hours=i) for i in range(len(temps))
            ],
            "air_temp": temps,
        })

    def test_min_mean_max_per_day(self):
        df = daily_from_hourly_forecast(self._hourly("2026-09-07", [10.0, 20.0, 30.0]))

        assert len(df) == 1
        assert df.loc[0, "air_temp_min"] == pytest.approx(10.0)
        assert df.loc[0, "air_temp"] == pytest.approx(20.0)
        assert df.loc[0, "air_temp_max"] == pytest.approx(30.0)

    def test_splits_on_day_boundaries(self):
        hourly = self._hourly("2026-09-07 22:00", [10.0, 11.0, 30.0, 31.0])
        df = daily_from_hourly_forecast(hourly)

        assert len(df) == 2
        assert df.loc[0, "air_temp_max"] == pytest.approx(11.0)
        assert df.loc[1, "air_temp_min"] == pytest.approx(30.0)

    def test_dates_are_normalised_to_midnight(self):
        df = daily_from_hourly_forecast(self._hourly("2026-09-07 13:00", [10.0]))

        assert df.loc[0, "date"] == pd.Timestamp("2026-09-07")

    def test_sorted_by_date(self):
        hourly = self._hourly("2026-09-07 22:00", [10.0, 11.0, 30.0, 31.0])
        df = daily_from_hourly_forecast(hourly.iloc[::-1])

        assert df["date"].is_monotonic_increasing

    def test_empty_input_returns_declared_schema(self):
        df = daily_from_hourly_forecast(pd.DataFrame(columns=["datetime", "air_temp"]))

        assert df.empty
        assert list(df.columns) == ["date", "air_temp", "air_temp_min", "air_temp_max"]

    def test_none_input_returns_declared_schema(self):
        assert daily_from_hourly_forecast(None).empty


class TestFullWeatherFrameAsSolarArgument:
    """
    The unified forecast frame carries air_temp as well as solar and cloud, and
    it gets passed to build_hourly_weather as the forecast argument. Before
    build_hourly_weather sliced its inputs, that merged air_temp against a base
    that already had one, yielding air_temp_x / air_temp_y and no air_temp -
    which the forecaster rejects with "hourly_weather missing required column".
    Caught in the browser, not by tests, because the tests passed tidy
    solar-only frames.
    """

    def _hourly(self, hours, **cols):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp("2026-09-07") + pd.Timedelta(hours=i)
                for i in range(hours)
            ],
            **{k: [v] * hours for k, v in cols.items()},
        })

    def test_air_temp_survives_a_forecast_frame_that_also_carries_it(self):
        base = self._hourly(6, air_temp=15.0)
        forecast = self._hourly(
            6, air_temp=99.0, shortwave_radiation=400.0, cloud_cover=20.0
        )

        weather = build_hourly_weather(base, None, forecast)

        assert "air_temp" in weather.columns
        assert "air_temp_x" not in weather.columns
        assert "air_temp_y" not in weather.columns
        # The base air series wins; the forecast frame contributes solar/cloud.
        assert set(weather["air_temp"]) == {15.0}
        assert set(weather["shortwave_radiation"]) == {400.0}

    def test_output_is_exactly_what_the_forecaster_requires(self):
        base = self._hourly(6, air_temp=15.0)
        forecast = self._hourly(
            6, air_temp=99.0, shortwave_radiation=400.0, cloud_cover=20.0
        )

        weather = build_hourly_weather(base, None, forecast)

        # set_hourly_weather raises if any of these are missing.
        forecaster = WaterTempForecaster()
        forecaster.set_hourly_weather(weather)
        assert len(forecaster.hourly_weather) == 6

    def test_a_historical_frame_carrying_air_temp_is_also_sliced(self):
        base = self._hourly(6, air_temp=15.0)
        historical = self._hourly(
            6, air_temp=99.0, shortwave_radiation=100.0, cloud_cover=90.0
        )

        weather = build_hourly_weather(base, historical, None)

        assert set(weather["air_temp"]) == {15.0}
        assert set(weather["cloud_cover"]) == {90.0}

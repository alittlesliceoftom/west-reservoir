import json
from datetime import date
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
import pytest

from forecast.openmeteo import OpenMeteoClient, OpenMeteoError, parse_hourly

FIXTURES = Path(__file__).parents[1] / "fixtures"


def fixture(name):
    return json.loads((FIXTURES / name).read_text())


def client_returning(handler):
    return OpenMeteoClient(http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_parse_hourly_gives_a_utc_frame_with_model_column_names():
    frame = parse_hourly(fixture("openmeteo_forecast.json"))
    assert list(frame.columns) == ["valid_at", "air_temp", "shortwave", "cloud_cover"]
    assert len(frame) == 13 * 24
    assert str(frame["valid_at"].dt.tz) == "UTC"
    assert frame["valid_at"].diff().dropna().eq(pd.Timedelta(hours=1)).all()


def test_parse_hourly_keeps_nulls_as_nan():
    payload = {"hourly": {
        "time": ["2026-06-01T00:00", "2026-06-01T01:00"],
        "temperature_2m": [10.0, None],
        "shortwave_radiation": [None, 0.0],
        "cloud_cover": [50, 60],
    }}
    frame = parse_hourly(payload)
    assert np.isnan(frame.loc[1, "air_temp"])
    assert np.isnan(frame.loc[0, "shortwave"])


def test_forecast_requests_seven_past_and_six_forecast_days_in_utc():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        seen["host"] = request.url.host
        return httpx.Response(200, json=fixture("openmeteo_forecast.json"))

    frame = client_returning(handler).forecast(51.5669, -0.0905)
    assert seen["host"] == "api.open-meteo.com"
    assert seen["past_days"] == "7"
    assert seen["forecast_days"] == "6"
    assert seen["timezone"] == "UTC"
    assert seen["hourly"] == "temperature_2m,shortwave_radiation,cloud_cover"
    assert len(frame) == 312


def test_archive_requests_the_date_range():
    seen = {}

    def handler(request):
        seen.update(request.url.params)
        seen["host"] = request.url.host
        return httpx.Response(200, json=fixture("openmeteo_archive.json"))

    frame = client_returning(handler).archive(51.57, -0.09, date(2026, 9, 1), date(2026, 9, 3))
    assert seen["host"] == "archive-api.open-meteo.com"
    assert (seen["start_date"], seen["end_date"]) == ("2026-09-01", "2026-09-03")
    assert len(frame) == 72


def test_one_failure_is_retried():
    calls = []

    def handler(request):
        calls.append(1)
        if len(calls) == 1:
            return httpx.Response(502)
        return httpx.Response(200, json=fixture("openmeteo_forecast.json"))

    assert len(client_returning(handler).forecast(51.57, -0.09)) == 312
    assert len(calls) == 2


def test_two_failures_raise_with_the_reason():
    def handler(request):
        return httpx.Response(429, json={"error": True, "reason": "Daily API request limit exceeded"})

    with pytest.raises(OpenMeteoError, match="limit exceeded"):
        client_returning(handler).forecast(51.57, -0.09)


def test_a_connection_error_raises_open_meteo_error():
    def handler(request):
        raise httpx.ConnectError("no route")

    with pytest.raises(OpenMeteoError):
        client_returning(handler).forecast(51.57, -0.09)


def test_geocode_returns_places_with_enough_to_tell_them_apart():
    client = client_returning(lambda r: httpx.Response(200, json=fixture("openmeteo_geocode.json")))
    places = client.geocode("Bristol")
    assert places[0] == {
        "name": "Bristol", "lat": 51.45523, "lon": -2.59665,
        "region": "England", "country": "United Kingdom", "timezone": "Europe/London",
    }


def test_geocode_with_no_results_is_an_empty_list():
    client = client_returning(lambda r: httpx.Response(200, json={"generationtime_ms": 0.1}))
    assert client.geocode("zzzz") == []


def test_point_preview_reports_air_now_elevation_and_timezone():
    client = client_returning(lambda r: httpx.Response(200, json=fixture("openmeteo_point.json")))
    assert client.point_preview(51.45, -2.6) == {
        "air_temp": 8.8, "elevation": 13.0, "timezone": "Europe/London",
    }

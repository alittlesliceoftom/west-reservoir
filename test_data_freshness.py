"""
Freshness checks for every data source.

These exist because Meteostat stopped returning London data in March 2026 and
nobody noticed for five months (issue #33). A source going silent is invisible
from the UI, so it needs its own test.

These check the outside world. They read each loader directly, before any of
our own processing, and ask only one question: is this source current?

Our half of that failure - silently interpolating across the resulting gap -
is tested separately and deterministically by
test_data.py::TestInterpolationCap. Keep the two apart: an assertion about our
code should not depend on the weather.

Unlike the rest of the suite these hit the network, so they are slower and can
fail for reasons outside the codebase. That is the point.

Run just these:      pytest test_data_freshness.py -v
Skip them:           pytest -m "not freshness"

Issue #34 tracks running these on a schedule and opening a GitHub issue
automatically when one goes stale.
"""

import pandas as pd
import pytest

from config import ENABLE_MOTHERDUCK
from data import (
    DataLoadError,
    load_forecast_air_temps,
    load_forecast_solar_cloud,
    load_historical_air_temps,
    load_historical_solar_cloud,
    load_hourly_air_temps,
    load_water_temps,
)

pytestmark = pytest.mark.freshness


# How far behind "now" each source is allowed to be, and why.
MAX_LAG_DAYS = {
    # Manual readings by one person; a short holiday is normal.
    "water_temps": 10,
    # Open-Meteo archive is reanalysis-backed and has been same-day in practice.
    "hourly_air": 2,
    "daily_air": 3,
    "solar_cloud": 3,
    # Stored forecasts are written once per day when the app is opened.
    "stored_forecasts": 4,
}


def _lag_days(latest) -> float:
    """Whole days between the latest datapoint and today."""
    now = pd.Timestamp.now().normalize()
    return (now - pd.Timestamp(latest).normalize()).days


def _assert_fresh(latest, source_key, label):
    """Fail with the measured lag, not just 'stale' - the number is the action."""
    lag = _lag_days(latest)
    limit = MAX_LAG_DAYS[source_key]
    assert lag <= limit, (
        f"{label} is {lag} days stale (limit {limit}). "
        f"Latest datapoint: {latest}. "
        f"A source can go silent while the dashboard still renders - see issue #33."
    )


class TestHistoricalSources:
    """Sources that should be current up to roughly today."""

    def test_water_temps_are_current(self):
        df = load_water_temps()
        assert not df.empty, "Google Sheets returned no water temperature rows"
        _assert_fresh(df["date"].max(), "water_temps", "Water temperature (Google Sheets)")

    def test_hourly_air_temps_are_current(self):
        end = pd.Timestamp.now().normalize()
        df = load_hourly_air_temps(end - pd.Timedelta(days=10), end)
        assert not df.empty, "Open-Meteo archive returned no hourly air temperatures"
        _assert_fresh(df["datetime"].max(), "hourly_air", "Hourly air temp (Open-Meteo archive)")

    def test_daily_air_temps_are_current(self):
        end = pd.Timestamp.now().normalize()
        df = load_historical_air_temps(end - pd.Timedelta(days=10), end)
        assert not df.empty, "Open-Meteo archive returned no daily air temperatures"
        _assert_fresh(df["date"].max(), "daily_air", "Daily air temp (Open-Meteo archive)")

    def test_historical_solar_cloud_is_current(self):
        end = pd.Timestamp.now().normalize()
        df = load_historical_solar_cloud(end - pd.Timedelta(days=10), end)
        assert not df.empty, "Open-Meteo archive returned no solar/cloud rows"
        _assert_fresh(df["datetime"].max(), "solar_cloud", "Solar/cloud (Open-Meteo archive)")


class TestForecastSources:
    """Forecast sources must reach into the future, not merely be recent."""

    def test_air_forecast_extends_into_the_future(self):
        try:
            df = load_forecast_air_temps(days=5)
        except DataLoadError as e:
            pytest.skip(f"OpenWeatherMap unavailable (likely no API key): {e}")

        assert not df.empty, "OpenWeatherMap returned no forecast rows"
        latest = pd.Timestamp(df["date"].max()).normalize()
        today = pd.Timestamp.now().normalize()
        assert latest > today, (
            f"Air forecast does not extend past today: latest is {latest.date()}. "
            f"A forecast that stops at today cannot support tomorrow's prediction."
        )

    def test_solar_cloud_forecast_extends_into_the_future(self):
        try:
            df = load_forecast_solar_cloud(days=5)
        except DataLoadError as e:
            pytest.skip(f"Open-Meteo forecast unavailable: {e}")

        assert not df.empty, "Open-Meteo returned no forecast solar/cloud rows"
        latest = pd.Timestamp(df["datetime"].max()).normalize()
        today = pd.Timestamp.now().normalize()
        assert latest > today, (
            f"Solar/cloud forecast does not extend past today: latest is {latest.date()}."
        )


class TestStoredForecasts:
    """MotherDuck should be accumulating a forecast run most days."""

    def test_stored_forecasts_are_current(self):
        if not ENABLE_MOTHERDUCK:
            pytest.skip("MotherDuck disabled via ENABLE_MOTHERDUCK")

        from forecast_storage import ForecastStorage, ForecastStorageError

        try:
            storage = ForecastStorage()
            conn = storage._get_connection()
            latest = conn.execute(
                "SELECT MAX(forecast_created_timestamp) FROM water_temp_predictions"
            ).fetchone()[0]
        except ForecastStorageError as e:
            pytest.skip(f"MotherDuck unavailable (likely no token): {e}")

        assert latest is not None, "No stored water temperature forecasts at all"
        _assert_fresh(
            pd.Timestamp(latest).tz_localize(None) if pd.Timestamp(latest).tzinfo
            else pd.Timestamp(latest),
            "stored_forecasts",
            "Stored forecasts (MotherDuck)",
        )

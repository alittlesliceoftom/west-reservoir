"""
Open-Meteo client: hourly forecast, archive, geocoding and a point preview.

Hourly series are requested in UTC so every hour is unique; a water's local
timezone is only for display and reading-time arithmetic.

Open-Meteo's free tier is for non-commercial use under 10,000 calls a day,
licensed CC BY 4.0. The site footer carries the attribution.
"""

from datetime import date

import httpx
import pandas as pd

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"

HOURLY = "temperature_2m,shortwave_radiation,cloud_cover"
RENAME = {"temperature_2m": "air_temp", "shortwave_radiation": "shortwave", "cloud_cover": "cloud_cover"}
PAST_DAYS = 7
FORECAST_DAYS = 7
TIMEOUT = 20.0


class OpenMeteoError(RuntimeError):
    """Open-Meteo could not be reached or refused the request."""


def parse_hourly(payload: dict) -> pd.DataFrame:
    """Columns valid_at (UTC), air_temp, shortwave, cloud_cover; nulls stay NaN."""
    hourly = payload["hourly"]
    frame = pd.DataFrame({"valid_at": pd.to_datetime(hourly["time"]).tz_localize("UTC")})
    for source, column in RENAME.items():
        frame[column] = pd.to_numeric(pd.Series(hourly[source], dtype="object"), errors="coerce").astype(float)
    return frame


class OpenMeteoClient:
    def __init__(self, http: httpx.Client | None = None):
        self.http = http or httpx.Client(timeout=TIMEOUT)

    def _get(self, url: str, params: dict) -> dict:
        """GET with one retry. Raises OpenMeteoError with Open-Meteo's reason if both fail."""
        last = "no response"
        for _ in range(2):
            try:
                response = self.http.get(url, params=params)
            except httpx.HTTPError as error:
                last = f"{type(error).__name__}: {error}"
                continue
            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    last = "HTTP 200 with a body that is not JSON"
                    continue
            try:
                reason = response.json().get("reason", "")
            except ValueError:
                reason = ""
            last = f"HTTP {response.status_code} {reason}".strip()
        raise OpenMeteoError(f"Open-Meteo request to {url} failed: {last}")

    def forecast(self, lat: float, lon: float) -> pd.DataFrame:
        """
        The last 7 days and the next 7, hourly, in one call. Days are UTC days,
        so 7 rather than 5 keeps the fifth local reading time covered in
        timezones ahead of UTC and just after local midnight.
        """
        payload = self._get(FORECAST_URL, {
            "latitude": lat, "longitude": lon, "hourly": HOURLY, "timezone": "UTC",
            "past_days": PAST_DAYS, "forecast_days": FORECAST_DAYS,
        })
        return parse_hourly(payload)

    def archive(self, lat: float, lon: float, start: date, end: date) -> pd.DataFrame:
        """Reanalysis for a past date range. The last few days before today are usually null."""
        payload = self._get(ARCHIVE_URL, {
            "latitude": lat, "longitude": lon, "hourly": HOURLY, "timezone": "UTC",
            "start_date": start.isoformat(), "end_date": end.isoformat(),
        })
        return parse_hourly(payload)

    def geocode(self, name: str, count: int = 8) -> list[dict]:
        payload = self._get(GEOCODE_URL, {"name": name, "count": count, "language": "en", "format": "json"})
        return [
            {
                "name": place["name"],
                "lat": place["latitude"],
                "lon": place["longitude"],
                "region": place.get("admin1", ""),
                "country": place.get("country", ""),
                "timezone": place.get("timezone", ""),
            }
            for place in payload.get("results", [])
        ]

    def point_preview(self, lat: float, lon: float) -> dict:
        """Air temperature now, grid elevation and the local timezone, for the create wizard."""
        payload = self._get(FORECAST_URL, {
            "latitude": lat, "longitude": lon, "current": "temperature_2m",
            "timezone": "auto", "forecast_days": 1,
        })
        return {
            "air_temp": payload["current"]["temperature_2m"],
            "elevation": payload["elevation"],
            "timezone": payload["timezone"],
        }

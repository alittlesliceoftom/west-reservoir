"""Data loading functions for West Reservoir Temperature Tracker"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional
from meteostat import Point, Daily, Hourly

from config import GOOGLE_SHEETS_URL, RESERVOIR_LAT, RESERVOIR_LON, REQUEST_TIMEOUT, get_openweather_api_key


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


class DataLoadError(Exception):
    """Raised when data cannot be loaded"""
    pass


def load_water_temps() -> pd.DataFrame:
    """
    Load water temperature measurements from Google Sheets.

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'water_temp' columns

    Raises:
        DataLoadError: If data cannot be loaded or is invalid
    """
    try:
        response = requests.get(GOOGLE_SHEETS_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        # Parse CSV
        df = pd.read_csv(StringIO(response.text))

        # Check we have at least 2 columns
        if len(df.columns) < 2:
            raise DataLoadError(
                f"Google Sheets data must have at least 2 columns (Date, Temperature), found {len(df.columns)}"
            )

        # Standardize column names
        df.columns = ["date", "water_temp"] + list(df.columns[2:])
        df = df[["date", "water_temp"]]

        # Convert date column (format: DD/MM/YYYY)
        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

        # Convert temperature to numeric
        df["water_temp"] = pd.to_numeric(df["water_temp"], errors="coerce")

        # Remove rows with invalid data
        initial_count = len(df)
        df = df.dropna()

        if df.empty:
            raise DataLoadError(
                "No valid water temperature data found in Google Sheets after cleaning"
            )

        # Sort by date
        df = df.sort_values("date").reset_index(drop=True)

        return df

    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to Google Sheets timed out after {REQUEST_TIMEOUT} seconds. "
            "Check your internet connection."
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch data from Google Sheets: {e}")
    except Exception as e:
        raise DataLoadError(f"Error processing Google Sheets data: {e}")


def load_historical_air_temps(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load historical air temperature data from Meteostat.

    Args:
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        pd.DataFrame: DataFrame with 'date', 'air_temp', 'air_temp_min', 'air_temp_max' columns

    Raises:
        DataLoadError: If data cannot be loaded
    """
    try:
        location = Point(RESERVOIR_LAT, RESERVOIR_LON)
        weather_data = Daily(location, start_date, end_date)
        weather_df = weather_data.fetch()

        if weather_df.empty:
            raise DataLoadError(
                f"No historical weather data available from Meteostat for "
                f"{start_date.date()} to {end_date.date()}"
            )

        # Reset index to get date as column
        weather_df = weather_df.reset_index()

        # Select and rename columns (avg, min, max)
        weather_df = weather_df[["time", "tavg", "tmin", "tmax"]].copy()
        weather_df.columns = ["date", "air_temp", "air_temp_min", "air_temp_max"]

        # Remove rows with missing average temperature data
        weather_df = weather_df.dropna(subset=["air_temp"])

        if weather_df.empty:
            raise DataLoadError(
                f"Historical weather data contains no valid temperature readings for "
                f"{start_date.date()} to {end_date.date()}"
            )

        # Ensure date is datetime
        weather_df["date"] = pd.to_datetime(weather_df["date"])

        return weather_df.sort_values("date").reset_index(drop=True)

    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Failed to load historical weather data from Meteostat: {e}")


def load_hourly_air_temps(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load hourly air temperature data from Meteostat.

    Args:
        start_date: Start datetime for historical data
        end_date: End datetime for historical data

    Returns:
        pd.DataFrame: DataFrame with 'datetime' and 'air_temp' columns

    Raises:
        DataLoadError: If data cannot be loaded
    """
    try:
        location = Point(RESERVOIR_LAT, RESERVOIR_LON)
        weather_data = Hourly(location, start_date, end_date)
        weather_df = weather_data.fetch()

        if weather_df.empty:
            raise DataLoadError(
                f"No hourly weather data available from Meteostat for "
                f"{start_date} to {end_date}"
            )

        # Reset index to get datetime as column
        weather_df = weather_df.reset_index()

        # Select and rename columns
        weather_df = weather_df[["time", "temp"]].copy()
        weather_df.columns = ["datetime", "air_temp"]

        # Remove rows with missing temperature data
        weather_df = weather_df.dropna()

        if weather_df.empty:
            raise DataLoadError(
                f"Hourly weather data contains no valid temperature readings for "
                f"{start_date} to {end_date}"
            )

        # Ensure datetime is datetime type
        weather_df["datetime"] = pd.to_datetime(weather_df["datetime"])

        return weather_df.sort_values("datetime").reset_index(drop=True)

    except DataLoadError:
        raise
    except Exception as e:
        raise DataLoadError(f"Failed to load hourly weather data from Meteostat: {e}")


def load_forecast_air_temps(days: int = 5) -> pd.DataFrame:
    """
    Load future air temperature forecast from OpenWeatherMap.

    Args:
        days: Number of days to forecast (max 5 for free tier)

    Returns:
        pd.DataFrame: DataFrame with 'date', 'air_temp', 'air_temp_min', 'air_temp_max' columns

    Raises:
        DataLoadError: If forecast cannot be loaded or API key is missing
    """
    try:
        # Get API key (will raise ConfigError if not found)
        from config import ConfigError
        try:
            api_key = get_openweather_api_key()
        except ConfigError as e:
            raise DataLoadError(str(e))

        # OpenWeatherMap 5-day forecast endpoint
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": RESERVOIR_LAT,
            "lon": RESERVOIR_LON,
            "appid": api_key,
            "units": "metric",  # Celsius
            "cnt": min(days * 8, 40),  # API returns 3-hour intervals
        }

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()

        # Check for API errors
        if "list" not in data:
            raise DataLoadError(
                f"Invalid response from OpenWeatherMap API: {data.get('message', 'Unknown error')}"
            )

        # Process forecast data - aggregate by date
        daily_data = {}

        for item in data["list"]:
            # Convert timestamp to date
            dt = datetime.fromtimestamp(item["dt"])
            date_key = dt.date()

            temp = item["main"]["temp"]

            # Group by date
            if date_key not in daily_data:
                daily_data[date_key] = []

            daily_data[date_key].append(temp)

        # Create daily aggregated data with min/max
        forecast_data = []
        for date_key, temps in daily_data.items():
            forecast_data.append(
                {
                    "date": pd.Timestamp(date_key),
                    "air_temp": sum(temps) / len(temps),  # Daily average
                    "air_temp_min": min(temps),
                    "air_temp_max": max(temps),
                }
            )

        forecast_df = pd.DataFrame(forecast_data)
        forecast_df = forecast_df.sort_values("date").reset_index(drop=True)

        if forecast_df.empty:
            raise DataLoadError("OpenWeatherMap API returned no forecast data")

        return forecast_df

    except DataLoadError:
        raise
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to OpenWeatherMap timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch forecast from OpenWeatherMap: {e}")
    except Exception as e:
        raise DataLoadError(f"Error processing OpenWeatherMap forecast: {e}")


def load_forecast_air_temps_3hourly(days: int = 5) -> pd.DataFrame:
    """
    Load raw 3-hourly air temperature forecast from OpenWeatherMap.

    Args:
        days: Number of days to forecast (max 5 for free tier)

    Returns:
        pd.DataFrame: DataFrame with 'datetime' and 'air_temp' columns
                      Up to 40 records (8 per day for 5 days)

    Raises:
        DataLoadError: If forecast cannot be loaded or API key is missing
    """
    try:
        from config import ConfigError
        try:
            api_key = get_openweather_api_key()
        except ConfigError as e:
            raise DataLoadError(str(e))

        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": RESERVOIR_LAT,
            "lon": RESERVOIR_LON,
            "appid": api_key,
            "units": "metric",
            "cnt": min(days * 8, 40),
        }

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()

        if "list" not in data:
            raise DataLoadError(
                f"Invalid response from OpenWeatherMap API: {data.get('message', 'Unknown error')}"
            )

        # Extract 3-hourly data without aggregation
        forecast_data = []
        for item in data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            temp = item["main"]["temp"]
            forecast_data.append({
                "datetime": pd.Timestamp(dt),
                "air_temp": temp,
            })

        forecast_df = pd.DataFrame(forecast_data)
        forecast_df = forecast_df.sort_values("datetime").reset_index(drop=True)

        if forecast_df.empty:
            raise DataLoadError("OpenWeatherMap API returned no forecast data")

        return forecast_df

    except DataLoadError:
        raise
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to OpenWeatherMap timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch 3-hourly forecast from OpenWeatherMap: {e}")
    except Exception as e:
        raise DataLoadError(f"Error processing 3-hourly OpenWeatherMap forecast: {e}")


def _parse_open_meteo_hourly(payload: dict) -> pd.DataFrame:
    """
    Parse an Open-Meteo hourly response into a DataFrame.

    Args:
        payload: JSON response with 'hourly' dict containing 'time',
                 'shortwave_radiation', 'cloud_cover'

    Returns:
        DataFrame with columns: datetime, shortwave_radiation, cloud_cover

    Raises:
        DataLoadError: If payload is missing required fields or empty
    """
    hourly = payload.get("hourly")
    if not hourly:
        raise DataLoadError(
            f"Open-Meteo response missing 'hourly' block: {payload.get('reason', payload)}"
        )

    required = ("time", "shortwave_radiation", "cloud_cover")
    for key in required:
        if key not in hourly:
            raise DataLoadError(
                f"Open-Meteo response missing required field '{key}'"
            )

    df = pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "shortwave_radiation": pd.to_numeric(hourly["shortwave_radiation"], errors="coerce"),
        "cloud_cover": pd.to_numeric(hourly["cloud_cover"], errors="coerce"),
    })

    df = df.dropna(subset=["shortwave_radiation", "cloud_cover"])

    if df.empty:
        raise DataLoadError("Open-Meteo response contained no valid hourly rows")

    return df.sort_values("datetime").reset_index(drop=True)


def load_historical_solar_cloud(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load historical hourly shortwave radiation and cloud cover from Open-Meteo.

    Args:
        start_date: Start datetime for historical data (inclusive)
        end_date: End datetime for historical data (inclusive)

    Returns:
        pd.DataFrame: columns 'datetime', 'shortwave_radiation' (W/m^2), 'cloud_cover' (%)

    Raises:
        DataLoadError: If data cannot be loaded
    """
    params = {
        "latitude": RESERVOIR_LAT,
        "longitude": RESERVOIR_LON,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "hourly": "shortwave_radiation,cloud_cover",
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return _parse_open_meteo_hourly(response.json())

    except DataLoadError:
        raise
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to Open-Meteo archive timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch historical solar/cloud from Open-Meteo: {e}")
    except Exception as e:
        raise DataLoadError(f"Error processing Open-Meteo archive response: {e}")


def load_forecast_solar_cloud(days: int = 5) -> pd.DataFrame:
    """
    Load forecast hourly shortwave radiation and cloud cover from Open-Meteo.

    Args:
        days: Number of forecast days (Open-Meteo supports up to 16 on free tier)

    Returns:
        pd.DataFrame: columns 'datetime', 'shortwave_radiation' (W/m^2), 'cloud_cover' (%)

    Raises:
        DataLoadError: If data cannot be loaded
    """
    params = {
        "latitude": RESERVOIR_LAT,
        "longitude": RESERVOIR_LON,
        "hourly": "shortwave_radiation,cloud_cover",
        "forecast_days": days,
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return _parse_open_meteo_hourly(response.json())

    except DataLoadError:
        raise
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to Open-Meteo forecast timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch forecast solar/cloud from Open-Meteo: {e}")
    except Exception as e:
        raise DataLoadError(f"Error processing Open-Meteo forecast response: {e}")


def interpolate_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate 3-hourly data to hourly using linear interpolation.

    Args:
        df: DataFrame with 'datetime' and 'air_temp' columns (3-hourly intervals)

    Returns:
        pd.DataFrame: DataFrame with hourly 'datetime' and 'air_temp' columns
    """
    if df.empty:
        return df.copy()

    # Set datetime as index for resampling
    df_indexed = df.set_index("datetime").sort_index()

    # Resample to hourly and interpolate
    hourly = df_indexed.resample("h").interpolate(method="linear")

    # Reset index to get datetime as column
    hourly = hourly.reset_index()

    return hourly


def select_storable_predictions(
    temperatures: pd.DataFrame, run_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Select the predictions worth storing as forecasts.

    fill_predictions() fills every AIR_ONLY row, including historical gaps.
    Those backfilled rows target dates BEFORE the run and are not forecasts -
    storing them pollutes accuracy analysis with rows at negative horizons.

    Args:
        temperatures: Frame with date, water_temp, source.
        run_date: The date this forecast run is being made.

    Returns:
        Rows where source == 'PREDICTED', water_temp is present, and the
        target date is not in the past.
    """
    if temperatures.empty:
        return temperatures

    predictions = temperatures[temperatures["source"] == "PREDICTED"].copy()
    predictions = predictions.dropna(subset=["water_temp"])

    return predictions[
        pd.to_datetime(predictions["date"]).dt.normalize()
        >= pd.Timestamp(run_date).normalize()
    ]


def build_hourly_weather(
    combined_hourly: pd.DataFrame,
    solar_cloud_hist: Optional[pd.DataFrame],
    solar_cloud_fore: Optional[pd.DataFrame],
) -> pd.DataFrame:
    """
    Merge air-temp hourly data with solar/cloud hourly data on datetime.

    Concatenates historical and forecast solar/cloud (forecast wins for any
    overlap), then inner-joins on the combined air-temp datetimes. Falls back
    to zero solar / 100% cloud for any hour where solar/cloud is missing, so
    the model degrades gracefully to single-term physics rather than erroring.
    """
    base = combined_hourly[["datetime", "air_temp"]].copy()
    base["datetime"] = pd.to_datetime(base["datetime"])

    solar_frames = []
    if solar_cloud_hist is not None and not solar_cloud_hist.empty:
        solar_frames.append(solar_cloud_hist)
    if solar_cloud_fore is not None and not solar_cloud_fore.empty:
        solar_frames.append(solar_cloud_fore)

    if not solar_frames:
        base["shortwave_radiation"] = 0.0
        base["cloud_cover"] = 100.0
        return base

    solar = pd.concat(solar_frames, ignore_index=True)
    solar["datetime"] = pd.to_datetime(solar["datetime"])
    # Forecast takes precedence on overlap (keep last in concat order)
    solar = solar.drop_duplicates(subset=["datetime"], keep="last")
    solar = solar.sort_values("datetime").reset_index(drop=True)

    merged = pd.merge(base, solar, on="datetime", how="left")
    # Fill gaps with neutral defaults (no solar, full cloud → no cool term)
    merged["shortwave_radiation"] = merged["shortwave_radiation"].fillna(0.0)
    merged["cloud_cover"] = merged["cloud_cover"].fillna(100.0)
    return merged


def combine_hourly_temps(
    historical: pd.DataFrame,
    forecast: pd.DataFrame,
    gap_fill: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Combine historical hourly temps (Meteostat) with forecast hourly temps (OWM interpolated).

    Historical data takes precedence for overlapping times.
    Gap-fill data (from stored MotherDuck forecasts) fills the gap between historical and forecast.
    If no gap-fill data, falls back to linear interpolation.

    Priority order:
    1. Historical (Meteostat) - trusted measured data
    2. Gap-fill (stored forecasts from MotherDuck) - yesterday's forecast for today
    3. Forecast (live OWM) - current forecast for future

    Args:
        historical: DataFrame with 'datetime' and 'air_temp' columns (Meteostat)
        forecast: DataFrame with 'datetime' and 'air_temp' columns (interpolated OWM)
        gap_fill: Optional DataFrame with 'datetime' and 'air_temp' columns (stored forecasts)

    Returns:
        Combined DataFrame with 'datetime' and 'air_temp' columns
    """
    if historical.empty and forecast.empty:
        return pd.DataFrame(columns=["datetime", "air_temp"])

    if historical.empty:
        return forecast.copy()

    if forecast.empty:
        return historical.copy()

    def normalize_datetime_col(dt_series: pd.Series) -> pd.Series:
        """Ensure datetime series is timezone-naive datetime64[s]."""
        dt = pd.to_datetime(dt_series)
        if dt.dt.tz is not None:
            dt = dt.dt.tz_convert("UTC").dt.tz_localize(None)
        return dt.astype("datetime64[s]")

    # Normalize column names and ensure timezone-naive datetimes
    hist = historical[["datetime", "air_temp"]].copy()
    fore = forecast[["datetime", "air_temp"]].copy()
    hist["datetime"] = normalize_datetime_col(hist["datetime"])
    fore["datetime"] = normalize_datetime_col(fore["datetime"])

    # Find where historical ends and forecast begins
    hist_end = hist["datetime"].max()
    fore_start = fore["datetime"].min()

    # Only use forecast data after historical ends
    fore_future = fore[fore["datetime"] > hist_end].copy()

    # Process gap-fill data if available
    gap_data = None
    if gap_fill is not None and not gap_fill.empty:
        gap = gap_fill[["datetime", "air_temp"]].copy()
        gap["datetime"] = normalize_datetime_col(gap["datetime"])
        # Only use gap data that's after historical and before forecast
        filtered = gap[(gap["datetime"] > hist_end) & (gap["datetime"] < fore_start)]
        if not filtered.empty:
            gap_data = filtered.copy()

    # Combine all sources: historical + gap_fill + forecast (only non-empty)
    to_concat = [hist, fore_future]
    if gap_data is not None:
        to_concat.insert(1, gap_data)  # Insert between hist and fore
    combined = pd.concat(to_concat, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    # Resample to consistent hourly frequency for the forecaster.
    # Gap fill and forecast data may be 3-hourly; the chart shows raw points
    # but the forecaster needs true hourly data for correct heat transfer steps.
    if not combined.empty:
        combined = combined.set_index("datetime").sort_index()
        combined = combined.resample("h").interpolate(method="linear")
        combined = combined.reset_index()

    return combined

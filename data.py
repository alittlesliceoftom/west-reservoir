"""Data loading functions for West Reservoir Temperature Tracker"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from meteostat import Point, Daily

from config import GOOGLE_SHEETS_URL, RESERVOIR_LAT, RESERVOIR_LON, REQUEST_TIMEOUT, get_openweather_api_key


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
        pd.DataFrame: DataFrame with 'date' and 'air_temp' columns

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

        # Select and rename columns
        weather_df = weather_df[["time", "tavg"]].copy()
        weather_df.columns = ["date", "air_temp"]

        # Remove rows with missing temperature data
        weather_df = weather_df.dropna()

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


def load_forecast_air_temps(days: int = 5) -> pd.DataFrame:
    """
    Load future air temperature forecast from OpenWeatherMap.

    Args:
        days: Number of days to forecast (max 5 for free tier)

    Returns:
        pd.DataFrame: DataFrame with 'date' and 'air_temp' columns

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

            # Group by date to get daily average
            if date_key not in daily_data:
                daily_data[date_key] = []

            daily_data[date_key].append(temp)

        # Create daily aggregated data
        forecast_data = []
        for date_key, temps in daily_data.items():
            forecast_data.append(
                {
                    "date": pd.Timestamp(date_key),
                    "air_temp": sum(temps) / len(temps),  # Daily average
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

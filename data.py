"""Data loading functions for West Reservoir Temperature Tracker"""

import pandas as pd
import requests
from datetime import datetime, timedelta
from io import StringIO
from typing import Optional

from config import GOOGLE_SHEETS_URL, RESERVOIR_LAT, RESERVOIR_LON, REQUEST_TIMEOUT, get_openweather_api_key


OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Longest run of missing hours we will bridge by linear interpolation.
# Sized to cover 3-hourly forecast data (2 missing hours between points) with
# room to spare; anything longer is a genuine data gap, not a sampling artefact.
MAX_INTERPOLATION_HOURS = 6

# The solar/cloud measures build_hourly_weather merges onto the air series.
SOLAR_CLOUD_COLUMNS = ["shortwave_radiation", "cloud_cover"]


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

        df = pd.read_csv(StringIO(response.text))

        if len(df.columns) < 2:
            raise DataLoadError(
                f"Google Sheets data must have at least 2 columns (Date, Temperature), found {len(df.columns)}"
            )

        df.columns = ["date", "water_temp"] + list(df.columns[2:])
        df = df[["date", "water_temp"]]

        df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")

        df["water_temp"] = pd.to_numeric(df["water_temp"], errors="coerce")

        initial_count = len(df)
        df = df.dropna()

        if df.empty:
            raise DataLoadError(
                "No valid water temperature data found in Google Sheets after cleaning"
            )

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
    Load historical daily air temperature from the Open-Meteo archive.

    Previously sourced from Meteostat, which stopped returning data for every
    London station in March 2026 (see issue #33). Open-Meteo is already used
    for solar and cloud, so this adds no new dependency.

    Args:
        start_date: Start date for historical data
        end_date: End date for historical data

    Returns:
        pd.DataFrame: DataFrame with 'date', 'air_temp', 'air_temp_min', 'air_temp_max' columns

    Raises:
        DataLoadError: If data cannot be loaded
    """
    params = {
        "latitude": RESERVOIR_LAT,
        "longitude": RESERVOIR_LON,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "daily": "temperature_2m_mean,temperature_2m_min,temperature_2m_max",
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()

    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to Open-Meteo archive timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch historical air temps from Open-Meteo: {e}")

    daily = payload.get("daily")
    if not daily:
        raise DataLoadError(
            f"Open-Meteo response missing 'daily' block: {payload.get('reason', payload)}"
        )

    required = ("time", "temperature_2m_mean", "temperature_2m_min", "temperature_2m_max")
    for key in required:
        if key not in daily:
            raise DataLoadError(f"Open-Meteo response missing required field '{key}'")

    df = pd.DataFrame({
        "date": pd.to_datetime(daily["time"]),
        "air_temp": pd.to_numeric(daily["temperature_2m_mean"], errors="coerce"),
        "air_temp_min": pd.to_numeric(daily["temperature_2m_min"], errors="coerce"),
        "air_temp_max": pd.to_numeric(daily["temperature_2m_max"], errors="coerce"),
    })

    df = df.dropna(subset=["air_temp"])

    if df.empty:
        raise DataLoadError(
            f"Open-Meteo returned no valid daily temperatures for "
            f"{pd.Timestamp(start_date).date()} to {pd.Timestamp(end_date).date()}"
        )

    return df.sort_values("date").reset_index(drop=True)


def load_hourly_air_temps(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """
    Load historical hourly air temperature from the Open-Meteo archive.

    Previously sourced from Meteostat, which stopped returning data for every
    London station in March 2026 (see issue #33).

    Args:
        start_date: Start datetime for historical data
        end_date: End datetime for historical data

    Returns:
        pd.DataFrame: DataFrame with 'datetime' and 'air_temp' columns

    Raises:
        DataLoadError: If data cannot be loaded
    """
    params = {
        "latitude": RESERVOIR_LAT,
        "longitude": RESERVOIR_LON,
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "hourly": "temperature_2m",
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_ARCHIVE_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return _parse_open_meteo_hourly(
            response.json(), fields={"temperature_2m": "air_temp"}
        )

    except DataLoadError:
        raise
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to Open-Meteo archive timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch hourly air temps from Open-Meteo: {e}")


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
            "units": "metric",  # Celsius
            "cnt": min(days * 8, 40),  # API returns 3-hour intervals
        }

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()

        if "list" not in data:
            raise DataLoadError(
                f"Invalid response from OpenWeatherMap API: {data.get('message', 'Unknown error')}"
            )

        daily_data = {}

        for item in data["list"]:
            dt = datetime.fromtimestamp(item["dt"])
            date_key = dt.date()

            temp = item["main"]["temp"]

            if date_key not in daily_data:
                daily_data[date_key] = []

            daily_data[date_key].append(temp)

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


def _parse_open_meteo_hourly(
    payload: dict,
    fields: dict = None,
) -> pd.DataFrame:
    """
    Parse an Open-Meteo hourly response into a DataFrame.

    Args:
        payload: JSON response with an 'hourly' dict containing 'time' plus
                 the requested variables.
        fields: Mapping of Open-Meteo variable name -> output column name.
                Defaults to the solar/cloud pair.

    Returns:
        DataFrame with a 'datetime' column plus the mapped columns.

    Raises:
        DataLoadError: If payload is missing required fields or empty
    """
    if fields is None:
        fields = {
            "shortwave_radiation": "shortwave_radiation",
            "cloud_cover": "cloud_cover",
        }

    hourly = payload.get("hourly")
    if not hourly:
        raise DataLoadError(
            f"Open-Meteo response missing 'hourly' block: {payload.get('reason', payload)}"
        )

    for key in ("time", *fields):
        if key not in hourly:
            raise DataLoadError(
                f"Open-Meteo response missing required field '{key}'"
            )

    data = {"datetime": pd.to_datetime(hourly["time"])}
    for source_name, column in fields.items():
        data[column] = pd.to_numeric(hourly[source_name], errors="coerce")

    df = pd.DataFrame(data)
    df = df.dropna(subset=list(fields.values()))

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


def load_forecast_weather(days: int = 5) -> pd.DataFrame:
    """
    Load the hourly forecast for every model input in one Open-Meteo call.

    Air temperature, shortwave radiation and cloud cover come back together,
    hourly. This replaces pairing OpenWeatherMap's 3-hourly air forecast with a
    separate Open-Meteo solar/cloud call, which meant two thirds of the air
    values fed to the model were linear interpolations between 3-hourly points.

    Measured against what actually happened, over June to August 2026 at the
    reservoir, Open-Meteo's air forecast beat OpenWeatherMap's at one day ahead
    (MAE 0.94 C against 1.38 C). OpenWeatherMap scored better from two days out,
    but that comparison flatters it: its lead time was derived from a date
    difference, so a forecast made at 21:00 for 06:00 next morning counted as a
    full day ahead. The OpenWeatherMap loaders are kept as a fallback.

    Args:
        days: Forecast days to request. Open-Meteo serves up to 16 free;
              OpenWeatherMap capped at 5.

    Returns:
        pd.DataFrame: columns 'datetime', 'air_temp', 'shortwave_radiation'
                      (W/m^2), 'cloud_cover' (%)

    Raises:
        DataLoadError: If data cannot be loaded
    """
    params = {
        "latitude": RESERVOIR_LAT,
        "longitude": RESERVOIR_LON,
        "hourly": "temperature_2m,shortwave_radiation,cloud_cover",
        "forecast_days": days,
        "timezone": "UTC",
    }

    try:
        response = requests.get(OPEN_METEO_FORECAST_URL, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        return _parse_open_meteo_hourly(
            response.json(),
            fields={
                "temperature_2m": "air_temp",
                "shortwave_radiation": "shortwave_radiation",
                "cloud_cover": "cloud_cover",
            },
        )

    except DataLoadError:
        raise
    except requests.exceptions.Timeout:
        raise DataLoadError(
            f"Request to Open-Meteo forecast timed out after {REQUEST_TIMEOUT} seconds"
        )
    except requests.exceptions.RequestException as e:
        raise DataLoadError(f"Failed to fetch forecast weather from Open-Meteo: {e}")


def daily_from_hourly_forecast(hourly: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse an hourly air-temp forecast into the daily frame the chart wants.

    Replaces OpenWeatherMap's daily aggregation. Derived from the same hourly
    series the model runs on, so the chart and the forecast cannot disagree.

    Args:
        hourly: Frame with 'datetime' and 'air_temp'.

    Returns:
        pd.DataFrame: 'date', 'air_temp' (mean), 'air_temp_min', 'air_temp_max'
    """
    if hourly is None or hourly.empty:
        return pd.DataFrame(
            columns=["date", "air_temp", "air_temp_min", "air_temp_max"]
        )

    frame = hourly[["datetime", "air_temp"]].copy()
    frame["date"] = pd.to_datetime(frame["datetime"]).dt.normalize()

    daily = frame.groupby("date")["air_temp"].agg(["mean", "min", "max"]).reset_index()
    daily.columns = ["date", "air_temp", "air_temp_min", "air_temp_max"]

    return daily.sort_values("date").reset_index(drop=True)


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

    df_indexed = df.set_index("datetime").sort_index()

    hourly = df_indexed.resample("h").interpolate(method="linear")

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

    # Take only the solar/cloud columns. A caller may hand over a full weather
    # frame that also carries air_temp - the unified forecast does - and
    # merging that against a base which already has air_temp yields
    # air_temp_x/air_temp_y and no air_temp at all, which the forecaster then
    # rejects. Slicing here means no caller has to remember.
    def _solar_only(df):
        if df is None or df.empty:
            return None
        keep = ["datetime"] + [
            c for c in SOLAR_CLOUD_COLUMNS if c in df.columns
        ]
        return df[keep].copy()

    solar_frames = [
        frame
        for frame in (_solar_only(solar_cloud_hist), _solar_only(solar_cloud_fore))
        if frame is not None
    ]

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
    Combine historical hourly temps (Open-Meteo archive) with forecast hourly temps (OWM interpolated).

    Historical data takes precedence for overlapping times.
    Gap-fill data (from stored MotherDuck forecasts) fills the gap between historical and forecast.
    If no gap-fill data, short gaps are interpolated (see MAX_INTERPOLATION_HOURS);
    longer gaps are left as gaps rather than invented.

    Priority order:
    1. Historical (Open-Meteo archive) - trusted measured data
    2. Gap-fill (stored forecasts from MotherDuck) - yesterday's forecast for today
    3. Forecast (live OWM) - current forecast for future

    Args:
        historical: DataFrame with 'datetime' and 'air_temp' columns (Open-Meteo archive)
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

    hist = historical[["datetime", "air_temp"]].copy()
    fore = forecast[["datetime", "air_temp"]].copy()
    hist["datetime"] = normalize_datetime_col(hist["datetime"])
    fore["datetime"] = normalize_datetime_col(fore["datetime"])

    hist_end = hist["datetime"].max()
    fore_start = fore["datetime"].min()

    fore_future = fore[fore["datetime"] > hist_end].copy()

    gap_data = None
    if gap_fill is not None and not gap_fill.empty:
        gap = gap_fill[["datetime", "air_temp"]].copy()
        gap["datetime"] = normalize_datetime_col(gap["datetime"])
        filtered = gap[(gap["datetime"] > hist_end) & (gap["datetime"] < fore_start)]
        if not filtered.empty:
            gap_data = filtered.copy()

    to_concat = [hist, fore_future]
    if gap_data is not None:
        to_concat.insert(1, gap_data)  # Insert between hist and fore
    combined = pd.concat(to_concat, ignore_index=True)
    combined = combined.sort_values("datetime").reset_index(drop=True)

    # Resample to consistent hourly frequency for the forecaster.
    # Gap fill and forecast data may be 3-hourly; the chart shows raw points
    # but the forecaster needs true hourly data for correct heat transfer steps.
    #
    # Interpolation is capped at MAX_INTERPOLATION_HOURS. Bridging a 3-hourly
    # forecast is legitimate; drawing a straight line across a months-long
    # outage is inventing weather. When Meteostat silently died in March 2026
    # the uncapped version filled 161 days with a smooth ramp and the model
    # trained on it without complaint (see issue #33). Real gaps must stay
    # gaps so fit() skips those periods instead of learning from a ruler.
    if not combined.empty:
        combined = combined.set_index("datetime").sort_index()
        hourly = combined.resample("h").asfreq()

        missing = hourly["air_temp"].isna()
        # Length of the run of consecutive missing hours each row belongs to.
        run_id = (missing != missing.shift()).cumsum()
        run_length = missing.groupby(run_id).transform("sum")

        filled = hourly["air_temp"].interpolate(method="linear", limit_area="inside")

        # Fill a gap only if the WHOLE run is short enough. Passing `limit` to
        # interpolate() is not equivalent: it fills that many hours inward from
        # each edge, so a months-long outage still gets fabricated hours at its
        # boundaries. The rule is per-gap, so it has to be applied per-gap.
        hourly["air_temp"] = filled.where(
            ~missing | (run_length <= MAX_INTERPOLATION_HOURS)
        )

        combined = hourly.reset_index().dropna(subset=["air_temp"])

    return combined


# Priority when the same date appears more than once: a real measurement beats
# an air-only row, which beats a prediction.
SOURCE_PRIORITY = {"MEASURED": 0, "AIR_ONLY": 1, "PREDICTED": 2}


def build_temperatures_frame(
    water_temps: pd.DataFrame, air_temps_hist: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge water and air temperatures into the main temperatures frame.

    Rows with a water measurement are MEASURED; the rest are AIR_ONLY and are
    candidates for prediction.

    Args:
        water_temps: date, water_temp
        air_temps_hist: date, air_temp, air_temp_min, air_temp_max

    Returns:
        DataFrame sorted by date, with a 'source' column.
    """
    temperatures = pd.merge(water_temps, air_temps_hist, on="date", how="outer")
    temperatures = temperatures.sort_values("date").reset_index(drop=True)
    temperatures["source"] = "MEASURED"
    temperatures.loc[temperatures["water_temp"].isna(), "source"] = "AIR_ONLY"
    return temperatures


def deduplicate_temperatures(temperatures: pd.DataFrame) -> pd.DataFrame:
    """
    Keep one row per date, preferring MEASURED over AIR_ONLY over PREDICTED.

    Duplicate dates break the prediction chain, which walks the frame row by
    row and would otherwise anchor on a duplicate rather than the previous day.

    Args:
        temperatures: Frame with date and source columns.

    Returns:
        One row per date, sorted by date, index reset.
    """
    result = temperatures.copy()
    result["_sort_priority"] = result["source"].map(SOURCE_PRIORITY)
    result = result.sort_values(["date", "_sort_priority"]).reset_index(drop=True)
    result = result.drop(columns=["_sort_priority"])
    return result.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)

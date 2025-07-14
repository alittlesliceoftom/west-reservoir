"""West Reservoir Temperature Tracker - Streamlit Dashboard"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import requests
from io import StringIO
from typing import Optional
from meteostat import Point, Daily
from sklearn.linear_model import LinearRegression
import numpy as np
from forecast import WaterTemperatureModel
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(
    page_title="West Reservoir Water Temperature Tracker",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state for logging
if "log_messages" not in st.session_state:
    st.session_state.log_messages = []


def add_log_message(message_type: str, message: str) -> None:
    """Add a log message to be displayed at the end."""
    st.session_state.log_messages.append((message_type, message))


def display_log_messages() -> None:
    """Display all accumulated log messages only in debug mode."""
    if st.session_state.log_messages and st.session_state.get("debug_mode", False):
        st.subheader("Processing Log")
        for msg_type, msg in st.session_state.log_messages:
            if msg_type == "info":
                st.info(msg)
            elif msg_type == "warning":
                st.warning(msg)
            elif msg_type == "error":
                st.error(msg)
            elif msg_type == "success":
                st.success(msg)
    # Clear messages after displaying (or not displaying)
    st.session_state.log_messages = []


SHEET_URL = "https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/export?format=csv&gid=0"
# Google Form URL for community temperature submissions
GOOGLE_FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSc4FK6wBCZT3-20h42Usj0S-UNbi5geOxnOW9syG_HDSDLDhg/formResponse"
# Form field IDs from the Google Form
FORM_FIELD_IDS = {
    "temperature": "entry.1996988660",  # Temperature field
    "name": "entry.1457081819",  # Submitter name field
    "timestamp": "entry.918355495",  # Submission timestamp field
}
REQUEST_TIMEOUT = 30

# Feature flags
ENABLE_COMMUNITY_SUBMISSIONS = (
    False  # Feature flag to enable/disable community temp submissions
)

# West Reservoir location (London)
RESERVOIR_LOCATION = Point(51.566938, -0.090492)  # West Reservoir coordinates
RESERVOIR_LAT = 51.566938
RESERVOIR_LON = -0.090492


# Weather API configuration
# Using OpenWeatherMap free tier (requires API key)
# Users can set their API key via environment variable or Streamlit secrets
def get_weather_api_key():
    """Get weather API key from environment or Streamlit secrets."""
    import os
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        try:
            if hasattr(st, "secrets") and "OPENWEATHER_API_KEY" in st.secrets:
                api_key = st.secrets["OPENWEATHER_API_KEY"]
        except Exception:
            add_log_message("info", "OpenWeather API key not found in Streamlit secrets, using environment variable or no forecast")
    return api_key


def get_weather_forecast(api_key: str, days: int = 5) -> pd.DataFrame:
    """Get real weather forecast data from OpenWeatherMap API.

    Args:
        api_key: OpenWeatherMap API key
        days: Number of days to forecast (max 5 for free tier)

    Returns:
        pd.DataFrame: Weather forecast data with Date, Air_Temperature, Air_Temp_Min, Air_Temp_Max
    """
    try:
        # OpenWeatherMap 5-day forecast endpoint
        url = "https://api.openweathermap.org/data/2.5/forecast"
        params = {
            "lat": RESERVOIR_LAT,
            "lon": RESERVOIR_LON,
            "appid": api_key,
            "units": "metric",  # Celsius
            "cnt": min(
                days * 8, 40
            ),  # API returns 3-hour intervals, max 40 calls (5 days)
        }

        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        data = response.json()

        # Process forecast data
        forecast_data = []
        daily_data = {}

        for item in data["list"]:
            # Convert timestamp to date
            dt = datetime.fromtimestamp(item["dt"])
            date_key = dt.date()

            temp = item["main"]["temp"]
            temp_min = item["main"]["temp_min"]
            temp_max = item["main"]["temp_max"]

            # Group by date to get daily min/max/avg
            if date_key not in daily_data:
                daily_data[date_key] = {"temps": [], "mins": [], "maxs": []}

            daily_data[date_key]["temps"].append(temp)
            daily_data[date_key]["mins"].append(temp_min)
            daily_data[date_key]["maxs"].append(temp_max)

        # Create daily aggregated data
        for date_key, temps in daily_data.items():
            forecast_data.append(
                {
                    "Date": pd.Timestamp(date_key),
                    "Air_Temperature": np.mean(temps["temps"]),
                    "Air_Temp_Min": min(temps["mins"]),
                    "Air_Temp_Max": max(temps["maxs"]),
                }
            )

        forecast_df = pd.DataFrame(forecast_data)
        forecast_df = forecast_df.sort_values("Date").reset_index(drop=True)

        add_log_message(
            "success",
            f"🌤️ Retrieved {len(forecast_df)} days of real weather forecast from OpenWeatherMap",
        )

        return forecast_df

    except requests.exceptions.RequestException as e:
        add_log_message("warning", f"Weather API request failed: {e}")
        return pd.DataFrame()
    except Exception as e:
        add_log_message("warning", f"Weather forecast processing failed: {e}")
        return pd.DataFrame()


def validate_and_clean_data(df: pd.DataFrame, debug_mode: bool = False) -> pd.DataFrame:
    """Validate and clean the raw temperature data.

    Args:
        df: Raw DataFrame from data source

    Returns:
        pd.DataFrame: Cleaned and validated temperature data
    """
    if df.empty:
        add_log_message("warning", "Data source appears to be empty")
        return pd.DataFrame()

    # Ensure we have at least 2 columns
    if len(df.columns) < 2:
        add_log_message(
            "error", "Data source must have at least 2 columns (Date, Temperature)"
        )
        return pd.DataFrame()

    # Clean and process the data
    df.columns = ["Date", "Temperature"] + list(
        df.columns[2:]
    )  # Preserve extra columns
    df = df[["Date", "Temperature"]]  # Keep only needed columns

    # Show debug info if enabled
    if debug_mode and len(df) > 0:
        st.write("🔍 **Debug: Raw data before processing**")
        st.write(df.head(5))
        st.write(f"🔍 **Debug: Raw data shape:** {df.shape}")
        st.write(f"🔍 **Debug: Column names:** {list(df.columns)}")

    # Store original data for comparison
    original_df = df.copy() if debug_mode else None

    # Convert date and temperature columns
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")

    # Check what's invalid (only show in debug mode)
    if debug_mode:
        invalid_dates = df["Date"].isna().sum()
        invalid_temps = df["Temperature"].isna().sum()

        st.write(
            f"🔍 **Debug: After conversion - Invalid dates:** {invalid_dates}, **Invalid temps:** {invalid_temps}"
        )

        if invalid_dates > 0:
            st.warning(f"🔍 Debug: Found {invalid_dates} rows with invalid dates")
            # Show rows with invalid dates (original values)
            invalid_date_mask = df["Date"].isna()
            invalid_date_original = original_df[invalid_date_mask][
                ["Date", "Temperature"]
            ].head(3)
            st.write("🔍 **Original values that failed date parsing:**")
            st.write(invalid_date_original)

        if invalid_temps > 0:
            st.warning(
                f"🔍 Debug: Found {invalid_temps} rows with invalid temperatures"
            )
            # Show rows with invalid temperatures (original values)
            invalid_temp_mask = df["Temperature"].isna()
            invalid_temp_original = original_df[invalid_temp_mask][
                ["Date", "Temperature"]
            ].head(3)
            st.write("🔍 **Original values that failed temperature parsing:**")
            st.write(invalid_temp_original)

        if invalid_dates == 0 and invalid_temps == 0:
            st.success("🔍 Debug: All data parsed successfully!")

    # Remove rows with invalid data
    initial_count = len(df)
    df = df.dropna()
    final_count = len(df)

    if initial_count > final_count:
        if debug_mode:
            add_log_message(
                "info",
                f"🔍 Debug: Removed {initial_count - final_count} invalid records",
            )
        else:
            add_log_message(
                "info", f"Removed {initial_count - final_count} invalid records"
            )

    if df.empty:
        add_log_message("warning", "No valid data found after cleaning")
        return df

    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(ttl=60*60*24) # get new data every day
def load_data() -> pd.DataFrame:
    """Load and process temperature data from Google Sheets.

    Returns:
        pd.DataFrame: Cleaned temperature data with Date and Temperature columns
    """
    # Create placeholder for loading message
    loading_placeholder = st.empty()

    try:
        loading_placeholder.info("Loading data from Google Sheets...")
        response = requests.get(SHEET_URL, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        df = pd.read_csv(StringIO(response.text))
        result = validate_and_clean_data(
            df, debug_mode=st.session_state.get("debug_mode", False)
        )

        if not result.empty:
            loading_placeholder.empty()  # Clear loading message
            return result
        else:
            loading_placeholder.empty()
            add_log_message(
                "error", "No valid data found in Google Sheets"
            )
            return pd.DataFrame()

    except requests.exceptions.Timeout:
        loading_placeholder.empty()
        add_log_message("error", "Request timed out while loading data from Google Sheets")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        loading_placeholder.empty()
        add_log_message("error", f"Network error loading data: {e}")
        return pd.DataFrame()
    except Exception as e:
        loading_placeholder.empty()
        add_log_message(
            "error", f"Error processing data: {e}"
        )
        return pd.DataFrame()



def incorporate_community_temps(df: pd.DataFrame) -> pd.DataFrame:
    """Incorporate approved community temperature submissions into the main dataset.

    Args:
        df: Main temperature dataset

    Returns:
        pd.DataFrame: Dataset with community temperatures added
    """
    if (
        "community_temps" not in st.session_state
        or not st.session_state.community_temps
    ):
        return df

    # For this demo, we'll automatically approve today's submission
    # In production, you'd check against approved submissions from Google Sheets
    today = datetime.now().date()
    today_str = today.strftime("%d/%m/%Y")

    # Find today's submission
    today_submission = None
    for submission in st.session_state.community_temps:
        if submission["Date"] == today_str:
            today_submission = submission
            break

    if today_submission:
        # Add community temperature to dataset
        community_row = {
            "Date": pd.Timestamp(today),
            "Temperature": today_submission["Temperature"],
            "Type": "Actual",
        }

        # Remove any existing data for today and add community submission
        df_filtered = df[df["Date"] != pd.Timestamp(today)]
        community_df = pd.DataFrame([community_row])

        result_df = pd.concat([df_filtered, community_df], ignore_index=True)
        result_df = result_df.sort_values("Date").reset_index(drop=True)

        add_log_message(
            "info",
            f"Using community-submitted temperature for today: {today_submission['Temperature']}°C",
        )
        return result_df

    return df


def submit_community_temperature(
    date: datetime, temperature: float, submitted_by: str = "Anonymous"
) -> bool:
    """Submit a community temperature reading via Google Forms.

    Args:
        date: Date of the temperature reading
        temperature: Temperature in Celsius
        submitted_by: Name/identifier of submitter

    Returns:
        bool: True if submission successful, False otherwise
    """
    try:
        # Store in session state for immediate use in the app
        if "community_temps" not in st.session_state:
            st.session_state.community_temps = []

        submission = {
            "Date": date.strftime("%d/%m/%Y"),
            "Temperature": temperature,
            "Submitted_By": submitted_by,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Status": "Pending Review",
        }

        st.session_state.community_temps.append(submission)

        # Submit to Google Form (if configured)
        if (
            GOOGLE_FORM_URL
            != "https://docs.google.com/forms/d/YOUR_FORM_ID/formResponse"
        ):
            form_data = {
                FORM_FIELD_IDS["temperature"]: str(temperature),
                FORM_FIELD_IDS["name"]: submitted_by,
                FORM_FIELD_IDS["timestamp"]: submission["Timestamp"],
            }

            add_log_message("info", f"🔧 Submitting to Google Form: {GOOGLE_FORM_URL}")
            add_log_message("info", f"🔧 Form data: {form_data}")

            try:
                # Submit to Google Form with proper headers and form encoding
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Referer": "https://docs.google.com/forms/d/1CLtT5XQ_KrtW6SZGSdmYkHQQwMw0y8K2UP3x0IDhZrg/viewform",
                }

                # Try the submission
                response = requests.post(
                    GOOGLE_FORM_URL,
                    data=form_data,
                    headers=headers,
                    timeout=REQUEST_TIMEOUT,
                    allow_redirects=False,
                )

                add_log_message("info", f"🔧 Response status: {response.status_code}")
                add_log_message("info", f"🔧 Response URL: {response.url}")

                # Create manual test URL for verification
                from urllib.parse import quote

                manual_test_url = f"{GOOGLE_FORM_URL}?{FORM_FIELD_IDS['temperature']}={temperature}&{FORM_FIELD_IDS['name']}={quote(submitted_by)}&{FORM_FIELD_IDS['timestamp']}={quote(submission['Timestamp'])}"
                add_log_message("info", f"🔧 Manual test URL: {manual_test_url}")
                add_log_message(
                    "info", f"🔧 Try this URL in your browser to test manually"
                )

                # Google Forms often returns 302 redirect on successful submission
                if response.status_code in [200, 302]:
                    add_log_message(
                        "success",
                        f"✅ Community temperature submitted to Google Sheets: {temperature}°C",
                    )
                else:
                    add_log_message(
                        "warning",
                        f"Form submission returned status {response.status_code}",
                    )
                    add_log_message(
                        "info", f"🔧 Response text: {response.text[:500]}..."
                    )

            except Exception as e:
                add_log_message("error", f"Form submission error: {e}")
        else:
            add_log_message(
                "warning", "Google Form not configured - data stored locally only"
            )

        add_log_message(
            "info",
            f"Community temperature: {temperature}°C for {date.strftime('%d/%m/%Y')}",
        )
        return True

    except requests.exceptions.RequestException as e:
        add_log_message(
            "warning",
            f"Network error submitting to Google Form: {e}. Data stored locally.",
        )
        return True  # Still return True since we stored locally
    except Exception as e:
        add_log_message("error", f"Failed to submit community temperature: {e}")
        return False


def check_today_temperature_missing(df: pd.DataFrame) -> bool:
    """Check if today's temperature is missing from the actual data.

    Args:
        df: DataFrame with temperature data

    Returns:
        bool: True if today's actual temperature is missing
    """
    today = pd.Timestamp(datetime.now().date())

    if df.empty:
        return True

    # Check if we have actual data for today
    today_data = df[df["Date"] == today]
    if today_data.empty:
        return True

    # Check if today's data is actual (not predicted)
    if "Type" in today_data.columns:
        actual_today = today_data[today_data["Type"] == "Actual"]
        return actual_today.empty

    return False


def create_community_temp_form() -> None:
    """Create a form for submitting community temperature data."""
    st.warning(
        "⚠️ Today's temperature is missing - help the community by submitting it!"
    )

    # Get yesterday's temperature as default if available
    yesterday = pd.Timestamp(datetime.now().date()) - pd.Timedelta(days=1)
    default_temp = 15.0  # Fallback default

    # Try to get yesterday's actual temperature from session state or data
    if "main_df" in st.session_state and not st.session_state.main_df.empty:
        df = st.session_state.main_df
        yesterday_data = df[df["Date"] == yesterday]
        if not yesterday_data.empty:
            actual_yesterday = (
                yesterday_data[yesterday_data["Type"] == "Actual"]
                if "Type" in yesterday_data.columns
                else yesterday_data
            )
            if not actual_yesterday.empty:
                default_temp = float(actual_yesterday["Temperature"].iloc[0])

    with st.form("community_temp_form"):
        st.markdown("**Submit Today's Water Temperature**")

        if default_temp != 15.0:
            st.info(f"💡 Defaulted to yesterday's temperature: {default_temp:.1f}°C")

        col1, col2 = st.columns(2)

        with col1:
            temp_input = st.number_input(
                "Water Temperature (°C)",
                min_value=0.0,
                max_value=30.0,
                value=default_temp,
                step=0.1,
                help="Enter the water temperature in Celsius (defaulted to yesterday's reading)",
            )

        with col2:
            submitter_name = st.text_input(
                "Your Name (Optional)",
                placeholder="Anonymous",
                help="Optional: Enter your name or identifier",
            )

        st.markdown("**Guidelines:**")
        st.markdown("""
        - If you're at the reservoir or on instagram, please add the temperature from the board.
        """)

        submitted = st.form_submit_button("Submit Temperature", type="primary")

        if submitted:
            if 0 <= temp_input <= 30:
                submitter = (
                    submitter_name.strip() if submitter_name.strip() else "Anonymous"
                )
                today = datetime.now()

                success = submit_community_temperature(today, temp_input, submitter)

                if success:
                    st.success(
                        f"Thank you! Temperature {temp_input}°C submitted for review."
                    )
                    st.info(
                        "Your submission will be reviewed and added to the official data if approved."
                    )
                    st.balloons()  # Celebration animation
                    # Rerun to update the dashboard
                    st.rerun()
                else:
                    st.error("Failed to submit temperature. Please try again.")
            else:
                st.error("Please enter a temperature between 0°C and 30°C.")


@st.cache_data(ttl=60*60*23.5) # get new data every day
def get_weather_data(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Get local weather data for the specified date range.

    Args:
        start_date: Start date for weather data
        end_date: End date for weather data

    Returns:
        pd.DataFrame: Weather data with Date and air temperature
    """
    try:
        # Get historical weather data from meteostat
        weather_data = Daily(RESERVOIR_LOCATION, start_date, datetime.now())
        weather_df = weather_data.fetch()

        if not weather_df.empty:
            # Reset index to get date as column
            weather_df = weather_df.reset_index()
            # Include daily high, low, and average temperatures
            weather_df = weather_df[["time", "tavg", "tmin", "tmax"]].copy()
            weather_df.columns = [
                "Date",
                "Air_Temperature",
                "Air_Temp_Min",
                "Air_Temp_Max",
            ]
            weather_df = weather_df.dropna(
                subset=["Air_Temperature"]
            )  # Only require average temp
        else:
            weather_df = pd.DataFrame(
                columns=["Date", "Air_Temperature", "Air_Temp_Min", "Air_Temp_Max"]
            )

        # Add real weather forecast data for future dates
        today = datetime.now().date()
        future_end = end_date.date()

        if future_end > today:
            # Try to get real weather forecast
            api_key = get_weather_api_key()

            if api_key:
                # Get real weather forecast
                forecast_days = min(
                    (future_end - today).days, 5
                )  # OpenWeatherMap free tier supports 5 days
                forecast_df = get_weather_forecast(api_key, forecast_days)

                if not forecast_df.empty:
                    # Filter forecast to only future dates we need
                    future_dates_needed = pd.date_range(
                        start=today + timedelta(days=1), end=future_end, freq="D"
                    )
                    forecast_filtered = forecast_df[
                        forecast_df["Date"].dt.date.isin(
                            [d.date() for d in future_dates_needed]
                        )
                    ]

                    if not forecast_filtered.empty:
                        # Add real forecast data
                        weather_df = pd.concat(
                            [weather_df, forecast_filtered], ignore_index=True
                        )
                        add_log_message(
                            "success",
                            f"✅ Added {len(forecast_filtered)} days of real weather forecast",
                        )


        return weather_df.sort_values("Date").reset_index(drop=True)

    except Exception as e:
        add_log_message("warning", f"Could not fetch weather data: {e}")
        return pd.DataFrame()


def create_temporal_features(
    df: pd.DataFrame, air_temp_col: str, water_temp_col: str = None
) -> pd.DataFrame:
    """Create temporal features for temperature prediction.

    Args:
        df: DataFrame with Date and temperature columns
        air_temp_col: Name of the air temperature column
        water_temp_col: Name of the water temperature column (optional)

    Returns:
        pd.DataFrame: DataFrame with temporal features added
    """
    df = df.copy().sort_values("Date")

    # Create lagged air temperature features
    df["Air_Temp_t0"] = df[air_temp_col]  # Today
    df["Air_Temp_t1"] = df[air_temp_col].shift(1)  # Yesterday
    df["Air_Temp_t2"] = df[air_temp_col].shift(2)  # Day before yesterday

    # Create rolling averages for air temperature
    df["Air_Temp_7day"] = df[air_temp_col].rolling(window=7, min_periods=1).mean()
    df["Air_Temp_30day"] = df[air_temp_col].rolling(window=30, min_periods=1).mean()

    # Create water temperature lagged features if available
    if water_temp_col and water_temp_col in df.columns:
        df["Water_Temp_t1"] = df[water_temp_col].shift(1)  # Yesterday's water temp
        df["Water_Temp_t2"] = df[water_temp_col].shift(
            2
        )  # Day before yesterday's water temp
        df["Water_Temp_7day"] = (
            df[water_temp_col].rolling(window=7, min_periods=1).mean()
        )

    # Create day of year feature (seasonal component)
    df["Day_of_Year"] = df["Date"].dt.dayofyear
    df["Season_Sin"] = np.sin(2 * np.pi * df["Day_of_Year"] / 365.25)
    df["Season_Cos"] = np.cos(2 * np.pi * df["Day_of_Year"] / 365.25)

    return df


def impute_missing_water_temperatures_physics(
    reservoir_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Impute missing water temperatures using physics-based model.

    Args:
        reservoir_df: Existing reservoir temperature data
        weather_df: Weather data for the same period

    Returns:
        pd.DataFrame: Extended data with physics-based imputed water temperatures
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df

    # Create a complete date range from weather data
    all_dates = weather_df[["Date", "Air_Temperature"]].copy()

    # Merge with existing reservoir data
    combined = pd.merge(all_dates, reservoir_df, on="Date", how="left")
    combined = combined.sort_values("Date").reset_index(drop=True)

    if len(combined[combined["Temperature"].notna()]) < 10:
        add_log_message(
            "warning",
            "Not enough existing water temperature data for physics-based imputation",
        )
        return reservoir_df

    # Initialize physics model
    model = WaterTemperatureModel()

    # Find continuous segments of actual data to train the model
    has_temp = combined["Temperature"].notna()
    actual_segments = []

    start_idx = None
    for i, has_data in enumerate(has_temp):
        if has_data and start_idx is None:
            start_idx = i
        elif not has_data and start_idx is not None:
            if i - start_idx >= 10:  # Need at least 10 points to train
                actual_segments.append((start_idx, i))
            start_idx = None

    # Check if last segment extends to end
    if start_idx is not None and len(combined) - start_idx >= 10:
        actual_segments.append((start_idx, len(combined)))

    if not actual_segments:
        add_log_message(
            "warning",
            "No sufficiently long segments of actual data found for physics model training",
        )
        return reservoir_df

    # Use the longest segment for initial model training
    longest_segment = max(actual_segments, key=lambda x: x[1] - x[0])
    start_idx, end_idx = longest_segment

    training_data = combined.iloc[start_idx:end_idx].copy()
    training_dates = training_data["Date"].dt.to_pydatetime()
    training_air_temps = training_data["Air_Temperature"].values
    training_water_temps = training_data["Temperature"].values

    add_log_message(
        "info",
        f"🧠 Training physics model on {len(training_data)} points for imputation...",
    )

    try:
        result = model.fit_parameters(
            training_air_temps, training_water_temps, training_dates
        )
        if result and result.success:
            add_log_message(
                "info", f"✅ Physics model trained for imputation (k={model.k:.4f})"
            )
        else:
            add_log_message(
                "warning",
                "Physics model training failed for imputation, using defaults",
            )
    except Exception as e:
        add_log_message("warning", f"Physics model training error for imputation: {e}")

    # Simpler approach: just use statistical imputation for now to avoid issues
    # The complex gap-by-gap physics prediction was causing problems
    add_log_message(
        "info", "🔄 Using statistical fallback for gap filling due to complexity"
    )

    imputed_data = combined.copy()

    # Method 1: Forward fill for short gaps (1-2 days)
    imputed_data["Temperature"] = imputed_data["Temperature"].ffill(limit=2)

    # Method 2: Interpolation for medium gaps
    imputed_data = imputed_data.set_index("Date")
    imputed_data["Temperature"] = imputed_data["Temperature"].interpolate(
        method="time", limit=7
    )
    imputed_data = imputed_data.reset_index()

    # Method 3: For remaining gaps, use physics model in a simpler way
    still_missing = imputed_data["Temperature"].isna()
    if still_missing.sum() > 0:
        # Get a typical temperature-air relationship from the training data
        available_data = imputed_data[imputed_data["Temperature"].notna()]
        if len(available_data) >= 10:
            # Use the trained physics model to estimate missing values
            for idx in imputed_data.index[still_missing]:
                air_temp = imputed_data.loc[idx, "Air_Temperature"]
                # Simple physics-based estimate: assume equilibrium with seasonal offset
                day_of_year = imputed_data.loc[idx, "Date"].timetuple().tm_yday
                seasonal_offset = model.seasonal_offset(day_of_year)

                # Estimate based on recent average relationship
                recent_data = available_data.tail(30)
                if len(recent_data) > 0:
                    avg_diff = (
                        recent_data["Temperature"] - recent_data["Air_Temperature"]
                    ).mean()
                    estimated_temp = air_temp + avg_diff + seasonal_offset
                    imputed_data.loc[idx, "Temperature"] = max(0.1, estimated_temp)

    # Add a column to track which data was imputed
    imputed_data["Data_Source"] = "Actual"
    original_mask = imputed_data["Date"].isin(reservoir_df["Date"])
    imputed_data.loc[~original_mask, "Data_Source"] = "Imputed"

    physics_imputed = (~original_mask & imputed_data["Temperature"].notna()).sum()
    add_log_message("info", f"🔬 Physics-inspired imputation: {physics_imputed} points")

    return imputed_data[["Date", "Temperature", "Data_Source"]]


def impute_missing_water_temperatures(
    reservoir_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Legacy imputation function using statistical methods (kept for fallback).

    Args:
        reservoir_df: Existing reservoir temperature data
        weather_df: Weather data for the same period

    Returns:
        pd.DataFrame: Extended data with imputed water temperatures
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df

    # Create a complete date range from weather data
    all_dates = weather_df[["Date", "Air_Temperature"]].copy()

    # Merge with existing reservoir data
    combined = pd.merge(all_dates, reservoir_df, on="Date", how="left")
    combined = combined.sort_values("Date").reset_index(drop=True)

    if len(combined[combined["Temperature"].notna()]) < 10:
        add_log_message(
            "warning", "Not enough existing water temperature data for imputation"
        )
        return reservoir_df

    # Start with simple imputation methods
    imputed_data = combined.copy()

    # Method 1: Forward fill for short gaps (1-2 days)
    imputed_data["Temperature"] = imputed_data["Temperature"].ffill(limit=2)

    # Method 2: Interpolation for medium gaps
    # Set Date as index for time-weighted interpolation
    imputed_data = imputed_data.set_index("Date")
    imputed_data["Temperature"] = imputed_data["Temperature"].interpolate(
        method="time", limit=7
    )
    imputed_data = imputed_data.reset_index()  # Reset index back to normal

    # Method 3: For remaining gaps, use regression with air temperature
    still_missing = imputed_data["Temperature"].isna()
    if still_missing.sum() > 0:
        # Train a simple model on available data
        available_data = imputed_data[imputed_data["Temperature"].notna()]
        if len(available_data) >= 5:
            X_simple = available_data[["Air_Temperature"]].values
            y_simple = available_data["Temperature"].values

            simple_model = LinearRegression()
            simple_model.fit(X_simple, y_simple)

            # Predict for missing values
            missing_data = imputed_data[still_missing]
            X_missing = missing_data[["Air_Temperature"]].values
            predicted_temps = simple_model.predict(X_missing)

            # Fill in the predictions
            imputed_data.loc[still_missing, "Temperature"] = predicted_temps

    # Add a column to track which data was imputed
    imputed_data["Data_Source"] = "Actual"
    original_mask = imputed_data["Date"].isin(reservoir_df["Date"])
    imputed_data.loc[~original_mask, "Data_Source"] = "Imputed"

    add_log_message(
        "info",
        f"🔧 Statistical imputation: {(~original_mask).sum()} missing water temperature readings",
    )

    return imputed_data[["Date", "Temperature", "Data_Source"]]


def create_temperature_predictions_physics(
    reservoir_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Create temperature predictions using physics-based model from forecast.py.

    Args:
        reservoir_df: Reservoir temperature data
        weather_df: Weather data

    Returns:
        pd.DataFrame: Combined data with predictions
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df

    # Separate historical and future weather data
    today = pd.Timestamp(datetime.now().date())
    historical_weather = weather_df[weather_df["Date"] <= today]
    future_weather = weather_df[weather_df["Date"] > today]

    if st.session_state.get("debug_mode", False):
        add_log_message(
            "info", f"🔍 Debug: Historical weather: {len(historical_weather)} days"
        )
        add_log_message("info", f"🔍 Debug: Future weather: {len(future_weather)} days")

    # Initialize the physics-based model
    model = WaterTemperatureModel()

    # First, impute missing water temperatures in historical period
    # Use the method selected by the user (passed as parameter or default to physics)
    use_physics_imputation = getattr(st.session_state, "use_physics_imputation", True)

    if use_physics_imputation:
        imputed_historical = impute_missing_water_temperatures_physics(
            reservoir_df, historical_weather
        )
    else:
        imputed_historical = impute_missing_water_temperatures(
            reservoir_df, historical_weather
        )

    # Prepare training data - combine imputed reservoir data with historical weather
    merged_historical = pd.merge(
        imputed_historical, historical_weather, on="Date", how="inner"
    )
    merged_historical = merged_historical.sort_values("Date").reset_index(drop=True)

    # Only use actual and imputed data for training (not predictions from other models)
    if "Data_Source" in merged_historical.columns:
        training_data = merged_historical[
            merged_historical["Data_Source"].isin(["Actual", "Imputed"])
        ]
    else:
        training_data = merged_historical

    if len(training_data) < len(merged_historical):
        add_log_message(
            "info",
            f"🔧 Using {len(training_data)} points for training (excluding previous predictions)",
        )
        merged_historical = training_data

    if len(merged_historical) < 30:
        add_log_message(
            "warning",
            "Not enough data for physics model training (need at least 30 days)",
        )
        return reservoir_df

    # Train the model on historical data
    training_dates = merged_historical["Date"].dt.to_pydatetime()
    training_air_temps = merged_historical["Air_Temperature"].values
    training_water_temps = merged_historical["Temperature"].values

    add_log_message(
        "info", f"🧠 Training physics model on {len(merged_historical)} data points..."
    )

    try:
        result = model.fit_parameters(
            training_air_temps, training_water_temps, training_dates
        )

        if result and result.success:
            add_log_message("success", f"✅ Model training successful!")
            add_log_message(
                "info",
                f"🔧 Heat transfer coeff: {model.k:.4f}, Seasonal amp: {model.seasonal_amp:.2f}°C",
            )
        else:
            add_log_message(
                "warning", "Model training failed, using default parameters"
            )

    except Exception as e:
        add_log_message(
            "warning", f"Model training error: {e}, using default parameters"
        )

    # Generate predictions for future dates
    result_parts = []

    # Add actual data
    actual_data = reservoir_df.copy()
    actual_data["Type"] = "Actual"
    result_parts.append(actual_data)

    # Add imputed historical data (if any gaps were filled)
    if "imputed_historical" in locals() and len(imputed_historical) > 0:
        # Only add imputed data points (not the original actual ones)
        imputed_only = imputed_historical[
            (~imputed_historical["Date"].isin(reservoir_df["Date"]))
            & (imputed_historical["Data_Source"] == "Imputed")
        ].copy()

        if not imputed_only.empty:
            imputed_only["Type"] = "Imputed"
            result_parts.append(imputed_only[["Date", "Temperature", "Type"]])

    # Create predictions if we have future weather data
    if not future_weather.empty and len(future_weather) > 0:
        # Get the most recent water temperature as starting point (from imputed data if available)
        if "imputed_historical" in locals() and len(imputed_historical) > 0:
            # Use the most recent imputed temperature (includes actual + filled gaps)
            current_water_temp = imputed_historical.iloc[-1]["Temperature"]
            add_log_message(
                "info",
                f"🎯 Starting forecast from: {current_water_temp:.1f}°C (most recent data point)",
            )
        else:
            # Fallback to original reservoir data
            current_water_temp = reservoir_df.iloc[-1]["Temperature"]
            add_log_message(
                "info",
                f"🎯 Starting forecast from: {current_water_temp:.1f}°C (last actual reading)",
            )

        # Prepare forecast data
        forecast_dates = future_weather["Date"].dt.to_pydatetime()
        forecast_air_temps = future_weather["Air_Temperature"].values

        # Get today's air temperature for proper i-1 indexing
        today = pd.Timestamp(datetime.now().date())
        today_weather = historical_weather[historical_weather["Date"] == today]
        today_air_temp = None
        if not today_weather.empty:
            today_air_temp = today_weather["Air_Temperature"].iloc[-1]
            add_log_message(
                "info",
                f"🌡️ Using today's air temperature: {today_air_temp:.1f}°C for forecast",
            )

        # Generate water temperature forecast
        try:
            predicted_water_temps = model.forecast(
                forecast_air_temps,
                current_water_temp,
                forecast_dates[0],
                len(forecast_dates),
                today_air_temp=today_air_temp,
            )

            # Create prediction dataframe
            predictions_df = pd.DataFrame(
                {
                    "Date": future_weather["Date"],
                    "Temperature": predicted_water_temps,
                    "Type": "Predicted",
                }
            )

            result_parts.append(predictions_df)
            add_log_message(
                "info", f"🔮 Generated {len(predictions_df)} physics-based predictions"
            )

        except Exception as e:
            add_log_message("warning", f"Prediction generation failed: {e}")

    # Combine all parts
    final_result = (
        pd.concat(result_parts, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return final_result


def create_temperature_predictions(
    reservoir_df: pd.DataFrame, weather_df: pd.DataFrame
) -> pd.DataFrame:
    """Create temperature predictions using multivariate weather and temporal data.

    Args:
        reservoir_df: Reservoir temperature data
        weather_df: Weather data

    Returns:
        pd.DataFrame: Combined data with predictions and imputed values
    """
    if weather_df.empty or reservoir_df.empty:
        return reservoir_df

    # Step 1: Separate historical and future weather data
    today = pd.Timestamp(datetime.now().date())
    historical_weather = weather_df[weather_df["Date"] <= today]
    future_weather_full = weather_df[weather_df["Date"] > today]

    if st.session_state.get("debug_mode", False):
        st.write(f"🔍 Debug: Historical weather: {len(historical_weather)} days")
        st.write(f"🔍 Debug: Future weather: {len(future_weather_full)} days")

    # Step 2: Impute missing water temperatures in the historical period only
    imputed_df = impute_missing_water_temperatures(reservoir_df, historical_weather)

    # Step 3: Create training data from imputed historical data
    merged_df = pd.merge(imputed_df, historical_weather, on="Date", how="inner")

    if len(merged_df) < 30:  # Need more data for multivariate model
        add_log_message(
            "warning",
            "Not enough data for advanced predictions (need at least 30 days)",
        )
        return reservoir_df

    # Create temporal features for training data (including water temperature)
    training_df = create_temporal_features(merged_df, "Air_Temperature", "Temperature")

    # Define feature columns (including water temperature features)
    feature_cols = [
        "Air_Temp_t0",
        "Air_Temp_t1",
        "Air_Temp_t2",
        "Air_Temp_7day",
        "Air_Temp_30day",
        "Water_Temp_t1",
        "Water_Temp_t2",
        "Water_Temp_7day",
        "Season_Sin",
        "Season_Cos",
    ]

    # Remove rows with NaN values (from lagged features)
    training_clean = training_df.dropna(subset=feature_cols + ["Temperature"])

    if len(training_clean) < 10:
        add_log_message(
            "warning", "Not enough clean training data after creating features"
        )
        return reservoir_df

    # Prepare training data
    X_train = training_clean[feature_cols].values
    y_train = training_clean["Temperature"].values

    # Train multivariate regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Step 4: Create predictions for future dates using the separated future weather
    # Debug info
    if st.session_state.get("debug_mode", False):
        st.write(
            f"🔍 Debug: Imputed data date range: {imputed_df['Date'].min().date()} to {imputed_df['Date'].max().date()}"
        )
        st.write(f"🔍 Debug: Future weather data points: {len(future_weather_full)}")
        if not future_weather_full.empty:
            st.write(
                f"🔍 Debug: Future weather date range: {future_weather_full['Date'].min().date()} to {future_weather_full['Date'].max().date()}"
            )

    if future_weather_full.empty:
        # No future predictions needed, just return imputed data with proper types
        result_df = imputed_df.copy()
        result_df["Type"] = result_df["Data_Source"].map(
            {"Actual": "Actual", "Imputed": "Imputed"}
        )
        if st.session_state.get("debug_mode", False):
            st.write("🔍 Debug: No future weather data available for predictions")
        return result_df[["Date", "Temperature", "Type"]]

    # Step 5: Create future predictions iteratively
    # Combine imputed historical data with future weather
    extended_data = (
        pd.concat(
            [
                imputed_df.rename(columns={"Data_Source": "Type"}),
                future_weather_full[["Date", "Air_Temperature"]].assign(
                    Temperature=None, Type="Future"
                ),
            ],
            ignore_index=True,
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Create features for extended data
    extended_features = create_temporal_features(
        extended_data, "Air_Temperature", "Temperature"
    )

    # Predict iteratively for future dates
    future_predictions = []
    future_rows = extended_features[extended_features["Type"] == "Future"]

    if st.session_state.get("debug_mode", False):
        st.write(f"🔍 Debug: Future rows to predict: {len(future_rows)}")

    for idx, row in future_rows.iterrows():
        # Check if we have sufficient features
        row_features = [row[col] for col in feature_cols if not pd.isna(row[col])]
        available_feature_names = [col for col in feature_cols if not pd.isna(row[col])]

        if st.session_state.get("debug_mode", False):
            st.write(
                f"🔍 Debug: Date {row['Date'].date()}: Available features: {len(available_feature_names)}/{len(feature_cols)}"
            )
            st.write(
                f"🔍 Debug: Missing features: {[col for col in feature_cols if pd.isna(row[col])]}"
            )

        if len(row_features) >= 7:  # Need minimum features
            X_pred = np.array([row[col] for col in available_feature_names]).reshape(
                1, -1
            )

            # Use appropriate model
            if len(available_feature_names) == len(feature_cols):
                predicted_temp = model.predict(X_pred)[0]
                if st.session_state.get("debug_mode", False):
                    st.write(
                        f"🔍 Debug: Used full model, predicted: {predicted_temp:.1f}°C"
                    )
            else:
                # Train simplified model with available features
                simple_training = training_clean[
                    available_feature_names + ["Temperature"]
                ].dropna()
                if len(simple_training) >= 5:
                    simple_model = LinearRegression()
                    simple_model.fit(
                        simple_training[available_feature_names].values,
                        simple_training["Temperature"].values,
                    )
                    predicted_temp = simple_model.predict(X_pred)[0]
                    if st.session_state.get("debug_mode", False):
                        st.write(
                            f"🔍 Debug: Used simplified model, predicted: {predicted_temp:.1f}°C"
                        )
                else:
                    if st.session_state.get("debug_mode", False):
                        st.write(
                            f"🔍 Debug: Skipping - not enough training data for simplified model"
                        )
                    continue  # Skip this prediction

            future_predictions.append(
                {
                    "Date": row["Date"],
                    "Temperature": predicted_temp,
                    "Type": "Predicted",
                }
            )

            # Update extended_features for next iteration
            extended_features.loc[idx, "Temperature"] = predicted_temp
            extended_features = create_temporal_features(
                extended_features, "Air_Temperature", "Temperature"
            )
        else:
            if st.session_state.get("debug_mode", False):
                st.write(
                    f"🔍 Debug: Skipping - only {len(row_features)} features available (need 7)"
                )

    # Combine all data types
    result_parts = []

    # Original actual data
    actual_data = reservoir_df.copy()
    actual_data["Type"] = "Actual"
    result_parts.append(actual_data)

    # Imputed data (excluding original actual dates)
    imputed_only = imputed_df[~imputed_df["Date"].isin(reservoir_df["Date"])].copy()
    if not imputed_only.empty:
        imputed_only["Type"] = "Imputed"
        result_parts.append(imputed_only[["Date", "Temperature", "Type"]])

    # Future predictions
    if future_predictions:
        future_df = pd.DataFrame(future_predictions)
        result_parts.append(future_df)

    # Combine all parts
    final_result = (
        pd.concat(result_parts, ignore_index=True)
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Display summary
    n_actual = len(actual_data)
    n_imputed = len(imputed_only) if not imputed_only.empty else 0
    n_predicted = len(future_predictions)

    feature_desc = "air temp (t, t-1, t-2), water temp (t-1, t-2), 7-day avgs, seasonal"
    add_log_message(
        "info", f"🧠 Model trained with {len(feature_cols)} features: {feature_desc}"
    )
    add_log_message(
        "info",
        f"📊 Data: {n_actual} actual, {n_imputed} imputed, {n_predicted} predicted",
    )

    return final_result


def display_statistics(df: pd.DataFrame) -> None:
    """Display temperature statistics in the sidebar."""
    st.sidebar.header("Statistics")

    latest_temp = df.iloc[-1]["Temperature"]
    latest_date = df.iloc[-1]["Date"].strftime("%d/%m/%Y")
    max_temp = df["Temperature"].max()
    min_temp = df["Temperature"].min()
    avg_temp = df["Temperature"].mean()

    st.sidebar.metric("Latest Temperature", f"{latest_temp}°C")
    st.sidebar.metric("Latest Reading", latest_date)
    st.sidebar.metric("Maximum", f"{max_temp}°C")
    st.sidebar.metric("Minimum", f"{min_temp}°C")
    st.sidebar.metric("Average", f"{avg_temp:.1f}°C")


def create_line_chart(df: pd.DataFrame, weather_df: pd.DataFrame = None) -> None:
    """Create and display the main temperature line chart."""
    st.subheader("Temperature Over Time")

    # Check if we have different data types
    if "Type" in df.columns:
        # Combine all water temperature data into single series
        water_temp_data = df[df["Type"].isin(["Actual", "Imputed", "Predicted"])].copy()

        # Create chart with all water temperature data as single blue dotted line
        fig = px.line(
            water_temp_data,
            x="Date",
            y="Temperature",
            title="West Reservoir Water Temperature",
            labels={"Temperature": "Temperature (°C)", "Date": "Date"},
        )

        # Update to blue dotted line with markers
        fig.update_traces(
            name="Water Temperature",
            line=dict(color="blue", dash="dot", width=2),
            marker=dict(color="blue", size=4),
            mode="lines+markers",
            showlegend=True,
        )

        # Add weather data if available
        if weather_df is not None and not weather_df.empty:
            # Filter weather data to match the date range of water temperature data
            water_start = df["Date"].min()
            water_end = df["Date"].max()
            weather_filtered = weather_df[
                (weather_df["Date"] >= water_start) & (weather_df["Date"] <= water_end)
            ].copy()

            if not weather_filtered.empty:
                # Add high temperatures
                if "Air_Temp_Max" in weather_filtered.columns:
                    fig.add_scatter(
                        x=weather_filtered["Date"],
                        y=weather_filtered["Air_Temp_Max"],
                        mode="lines",
                        name="Air Temp High",
                        line=dict(color="red", width=1, dash="dot"),
                        showlegend=True,
                    )

                # Add low temperatures
                if "Air_Temp_Min" in weather_filtered.columns:
                    fig.add_scatter(
                        x=weather_filtered["Date"],
                        y=weather_filtered["Air_Temp_Min"],
                        mode="lines",
                        name="Air Temp Low",
                        line=dict(color="lightblue", width=1, dash="dot"),
                        showlegend=True,
                    )

                # Add average air temperature
                fig.add_scatter(
                    x=weather_filtered["Date"],
                    y=weather_filtered["Air_Temperature"],
                    mode="lines",
                    name="Air Temp Avg",
                    line=dict(color="gray", width=2, dash="dash"),
                    showlegend=True,
                )

    else:
        fig = px.line(
            df,
            x="Date",
            y="Temperature",
            title="West Reservoir Water Temperature",
            labels={"Temperature": "Temperature (°C)", "Date": "Date"},
        )
        fig.update_traces(line_color="#1f77b4", line_width=2)

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": True,
            "displaylogo": False,
            "dragmode": False,
            "staticPlot": False,
        },
    )


def create_monthly_analysis(df: pd.DataFrame) -> None:
    """Create monthly analysis charts."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Monthly Averages")
        df_monthly = df.copy()
        df_monthly["Month"] = df_monthly["Date"].dt.to_period("M")
        monthly_avg = df_monthly.groupby("Month")["Temperature"].mean().reset_index()
        monthly_avg["Month"] = monthly_avg["Month"].astype(str)

        if not monthly_avg.empty:
            fig_monthly = px.bar(
                monthly_avg,
                x="Month",
                y="Temperature",
                title="Average Temperature by Month",
                labels={"Temperature": "Avg Temperature (°C)"},
            )
            fig_monthly.update_traces(marker_color="lightblue")
            st.plotly_chart(
                fig_monthly,
                use_container_width=True,
                config={
                    "displayModeBar": False,
                    "scrollZoom": False,
                    "doubleClick": False,
                    "showTips": True,
                    "displaylogo": False,
                    "dragmode": False,
                    "staticPlot": False,
                },
            )
        else:
            add_log_message("info", "Not enough data for monthly analysis")

    with col2:
        st.subheader("Temperature Distribution")
        fig_hist = px.histogram(
            df,
            x="Temperature",
            nbins=20,
            title="Temperature Distribution",
            labels={"Temperature": "Temperature (°C)", "count": "Frequency"},
        )
        fig_hist.update_traces(marker_color="lightcoral")
        st.plotly_chart(
            fig_hist,
            use_container_width=True,
            config={
                "displayModeBar": False,
                "scrollZoom": False,
                "doubleClick": False,
                "showTips": True,
                "displaylogo": False,
                "dragmode": False,
                "staticPlot": False,
            },
        )


def display_recent_data(df: pd.DataFrame) -> None:
    """Display recent temperature readings table."""
    st.subheader("Recent Readings")
    recent_data = df.tail(10).copy()
    recent_data["Date"] = recent_data["Date"].dt.strftime("%d/%m/%Y")
    st.dataframe(recent_data.iloc[::-1], use_container_width=True, hide_index=True)


def create_physics_model_analysis(df: pd.DataFrame, weather_df: pd.DataFrame) -> None:
    """Create physics model comparison and analysis section."""
    st.subheader("Physics Model vs Daily Temperature Correlation")

    if weather_df is None or weather_df.empty:
        st.warning("No weather data available for physics model analysis.")
        return

    # Filter to only actual water temperature data
    if "Type" in df.columns:
        actual_water_data = df[df["Type"] == "Actual"].copy()
    else:
        actual_water_data = df.copy()

    if actual_water_data.empty:
        st.warning(
            "No actual water temperature data available for physics model analysis."
        )
        return

    # Merge water and weather data on date
    merged_data = pd.merge(
        actual_water_data[["Date", "Temperature"]],
        weather_df[["Date", "Air_Temperature"]],
        on="Date",
        how="inner",
    ).dropna()

    if len(merged_data) < 30:
        st.warning(
            "Not enough overlapping data points for physics model analysis (need at least 30)."
        )
        return

    # Initialize and train physics model
    model = WaterTemperatureModel()
    training_dates = merged_data["Date"].dt.to_pydatetime()
    training_air_temps = merged_data["Air_Temperature"].values
    training_water_temps = merged_data["Temperature"].values

    # Fit the physics model
    try:
        result = model.fit_parameters(
            training_air_temps, training_water_temps, training_dates
        )
        model_fitted = result and result.success
    except Exception as e:
        st.error(f"Physics model fitting failed: {e}")
        return

    # Calculate daily temperature differences and correlations
    merged_data["Air_Water_Diff"] = (
        merged_data["Air_Temperature"] - merged_data["Temperature"]
    )
    merged_data["Day_of_Year"] = merged_data["Date"].dt.dayofyear

    # Calculate rolling correlations over different windows
    window_sizes = [7, 14, 30, 60]
    correlations = {}

    for window in window_sizes:
        if len(merged_data) >= window:
            rolling_corr = (
                merged_data["Air_Temperature"]
                .rolling(window=window)
                .corr(merged_data["Temperature"])
            )
            correlations[f"{window}d"] = rolling_corr.dropna()

    # Calculate physics model predictions
    physics_predictions = model.predict_temperature(
        training_air_temps, training_water_temps[0], training_dates
    )

    # Calculate physics model performance metrics
    physics_rmse = np.sqrt(np.mean((physics_predictions - training_water_temps) ** 2))
    physics_mae = np.mean(np.abs(physics_predictions - training_water_temps))
    physics_r2 = 1 - np.sum((training_water_temps - physics_predictions) ** 2) / np.sum(
        (training_water_temps - np.mean(training_water_temps)) ** 2
    )

    # Display physics model parameters
    st.markdown("### 🧮 Physics Model Parameters")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Heat Transfer Coeff (k)", f"{model.k:.4f} day⁻¹")
    with col2:
        st.metric("Seasonal Amplitude", f"{model.seasonal_amp:.2f}°C")
    with col3:
        st.metric("Seasonal Phase", f"{model.seasonal_phase:.0f} days")
    with col4:
        st.metric("Physics R²", f"{physics_r2:.3f}")

    # Compare with simple correlation
    simple_correlation = np.corrcoef(
        merged_data["Air_Temperature"], merged_data["Temperature"]
    )[0, 1]

    # Create visualization comparing physics model vs correlation patterns
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Temperature difference vs seasonal offset
    ax1 = axes[0]
    seasonal_offsets = [
        model.seasonal_offset(doy) for doy in merged_data["Day_of_Year"]
    ]
    scatter = ax1.scatter(
        merged_data["Air_Water_Diff"],
        seasonal_offsets,
        c=merged_data["Day_of_Year"],
        cmap="viridis",
        alpha=0.6,
    )
    ax1.set_xlabel("Air-Water Temperature Difference (°C)")
    ax1.set_ylabel("Physics Model Seasonal Offset (°C)")
    ax1.set_title("Seasonal Effects vs Temperature Differences")
    ax1.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax1, label="Day of Year")

    # # Plot 2: Heat transfer rate vs correlation
    # ax2 = axes[0, 1]
    # if "30d" in correlations:
    #     heat_transfer_rate = model.k * merged_data["Air_Water_Diff"]
    #     correlation_30d = correlations["30d"]

    #     # Align data for comparison
    #     min_len = min(len(heat_transfer_rate), len(correlation_30d))
    #     if min_len > 0:
    #         ax2.scatter(
    #             heat_transfer_rate[-min_len:], correlation_30d[-min_len:], alpha=0.6
    #         )
    #         ax2.set_xlabel("Heat Transfer Rate (k × ΔT)")
    #         ax2.set_ylabel("30-day Rolling Correlation")
    #         ax2.set_title("Heat Transfer Rate vs Rolling Correlation")
    #         ax2.grid(True, alpha=0.3)

    # Plot 3: Physics predictions vs actual vs linear model
    ax3 = axes[1]
    ax3.scatter(
        training_water_temps,
        physics_predictions,
        alpha=0.6,
        label="Physics Model",
        color="green",
    )

    # Simple linear model for comparison
    from sklearn.linear_model import LinearRegression

    linear_model = LinearRegression()
    linear_model.fit(training_air_temps.reshape(-1, 1), training_water_temps)
    linear_predictions = linear_model.predict(training_air_temps.reshape(-1, 1))
    linear_r2 = linear_model.score(
        training_air_temps.reshape(-1, 1), training_water_temps
    )

    ax3.scatter(
        training_water_temps,
        linear_predictions,
        alpha=0.4,
        label="Linear Model",
        color="orange",
    )

    # Perfect prediction line
    min_temp = min(training_water_temps.min(), physics_predictions.min())
    max_temp = max(training_water_temps.max(), physics_predictions.max())
    ax3.plot(
        [min_temp, max_temp],
        [min_temp, max_temp],
        "r--",
        alpha=0.8,
        label="Perfect Prediction",
    )

    ax3.set_xlabel("Observed Water Temperature (°C)")
    ax3.set_ylabel("Predicted Water Temperature (°C)")
    ax3.set_title("Model Predictions vs Observations")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # # Plot 4: Time series of correlations and physics variables
    # ax4 = axes[1, 1]

    # # Plot 30-day rolling correlation if available
    # if "30d" in correlations and len(correlations["30d"]) > 0:
    #     correlation_dates = merged_data["Date"].iloc[-len(correlations["30d"]) :]
    #     ax4.plot(
    #         correlation_dates,
    #         correlations["30d"],
    #         "b-",
    #         label="30d Rolling Correlation",
    #         linewidth=2,
    #     )

    # # Plot normalized seasonal component
    # normalized_seasonal = np.array(seasonal_offsets) / max(
    #     abs(np.array(seasonal_offsets))
    # )
    # ax4.plot(
    #     merged_data["Date"],
    #     normalized_seasonal,
    #     "g--",
    #     label="Normalized Seasonal Effect",
    #     alpha=0.7,
    # )

    # # Plot normalized heat transfer coefficient effect
    # normalized_k_effect = (model.k * merged_data["Air_Water_Diff"]) / max(
    #     abs(model.k * merged_data["Air_Water_Diff"])
    # )
    # ax4.plot(
    #     merged_data["Date"],
    #     normalized_k_effect,
    #     "r:",
    #     label="Normalized Heat Transfer",
    #     alpha=0.7,
    # )

    # ax4.set_xlabel("Date")
    # ax4.set_ylabel("Normalized Values")
    # ax4.set_title("Time Series: Correlation vs Physics Components")
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)
    # ax4.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    st.pyplot(fig)

    # Analysis comparison table
    st.markdown("### Model Comparison")

    comparison_data = {
        "Metric": [
            "R² Score",
            "RMSE (°C)",
            "MAE (°C)",
            "Simple Correlation",
            "Parameters",
        ],
        "Physics Model": [
            f"{physics_r2:.3f}",
            f"{physics_rmse:.2f}",
            f"{physics_mae:.2f}",
            f"{simple_correlation:.3f}",
            "3 (k, seasonal_amp, phase)",
        ],
        "Linear Model": [
            f"{linear_r2:.3f}",
            f"{np.sqrt(np.mean((linear_predictions - training_water_temps) ** 2)):.2f}",
            f"{np.mean(np.abs(linear_predictions - training_water_temps)):.2f}",
            f"{simple_correlation:.3f}",
            "2 (slope, intercept)",
        ],
    }

    comparison_df = pd.DataFrame(comparison_data)
    st.table(comparison_df)

    # Interpretation
    st.markdown("### 🔍 Physics Model Insights")

    # Calculate key relationships
    temp_responsiveness = (
        model.k * 24
    )  # Convert to temperature change per day per degree difference
    seasonal_range = 2 * model.seasonal_amp  # Peak-to-peak seasonal variation

    st.write(f"**Heat Transfer Dynamics:**")
    st.write(
        f"- Water temperature adjusts at {temp_responsiveness:.3f}°C/day per 1°C air-water difference"
    )
    st.write(
        f"- Seasonal variation: ±{model.seasonal_amp:.1f}°C (total range: {seasonal_range:.1f}°C)"
    )
    st.write(
        f"- Seasonal peak occurs around day {model.seasonal_phase:.0f} ({pd.Timestamp('2024-01-01') + pd.Timedelta(days=int(model.seasonal_phase)):%B %d})"
    )

    if physics_r2 > linear_r2:
        improvement = ((physics_r2 - linear_r2) / linear_r2) * 100
        st.write(
            f"**Model Performance:** Physics model performs {improvement:.1f}% better than simple linear correlation"
        )
    else:
        st.write(
            f"**Model Performance:** Linear model performs slightly better, suggesting simpler relationships dominate"
        )

    st.write(
        f"**Key Finding:** The physics model captures {physics_r2 * 100:.1f}% of water temperature variation using thermal dynamics"
    )


def create_correlation_analysis(df: pd.DataFrame, weather_df: pd.DataFrame) -> None:
    """Create correlation analysis between water and air temperature."""
    st.subheader("Water vs Air Temperature Correlation Analysis")

    if weather_df is None or weather_df.empty:
        st.warning("No weather data available for correlation analysis.")
        return

    # Filter to only actual water temperature data
    if "Type" in df.columns:
        actual_water_data = df[df["Type"] == "Actual"].copy()
    else:
        actual_water_data = df.copy()

    if actual_water_data.empty:
        st.warning(
            "No actual water temperature data available for correlation analysis."
        )
        return

    # Merge water and weather data on date
    merged_data = pd.merge(
        actual_water_data[["Date", "Temperature"]],
        weather_df[["Date", "Air_Temperature"]],
        on="Date",
        how="inner",
    ).dropna()

    if len(merged_data) < 10:
        st.warning(
            "Not enough overlapping data points for meaningful correlation analysis."
        )
        return

    # Rename columns for clarity
    merged_data = merged_data.rename(
        columns={
            "Temperature": "Water_Temperature",
            "Air_Temperature": "Air_Temperature",
        }
    )

    # Calculate correlation statistics
    correlation, p_value = stats.pearsonr(
        merged_data["Air_Temperature"], merged_data["Water_Temperature"]
    )

    # Perform linear regression
    slope, intercept, r_value, p_value_reg, std_err = stats.linregress(
        merged_data["Air_Temperature"], merged_data["Water_Temperature"]
    )

    # Calculate residuals
    predicted_water = slope * merged_data["Air_Temperature"] + intercept
    residuals = merged_data["Water_Temperature"] - predicted_water

    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Correlation (r)", f"{correlation:.3f}")
    with col2:
        st.metric("R² Score", f"{r_value**2:.3f}")
    with col3:
        st.metric("P-value", f"{p_value:.2e}")
    with col4:
        st.metric("Data Points", f"{len(merged_data)}")

    # Create plots using matplotlib/seaborn
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Scatter plot with regression line
    sns.scatterplot(
        data=merged_data,
        x="Air_Temperature",
        y="Water_Temperature",
        alpha=0.6,
        ax=ax1,
        color="blue",
    )
    sns.regplot(
        data=merged_data,
        x="Air_Temperature",
        y="Water_Temperature",
        scatter=False,
        ax=ax1,
        color="red",
        line_kws={"linewidth": 2},
    )

    ax1.set_title(
        f"Water vs Air Temperature\n(r = {correlation:.3f}, R² = {r_value**2:.3f})"
    )
    ax1.set_xlabel("Air Temperature (°C)")
    ax1.set_ylabel("Water Temperature (°C)")
    ax1.grid(True, alpha=0.3)

    # Add regression equation
    equation_text = f"Water Temp = {slope:.2f} × Air Temp + {intercept:.2f}"
    ax1.text(
        0.05,
        0.95,
        equation_text,
        transform=ax1.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    # Residuals plot
    sns.scatterplot(x=predicted_water, y=residuals, alpha=0.6, ax=ax2, color="green")
    ax2.axhline(y=0, color="red", linestyle="--", linewidth=2)
    ax2.set_title("Residuals Plot\n(Predicted vs Actual Difference)")
    ax2.set_xlabel("Predicted Water Temperature (°C)")
    ax2.set_ylabel("Residuals (°C)")
    ax2.grid(True, alpha=0.3)

    # Add residual statistics
    rmse = np.sqrt(np.mean(residuals**2))
    mae = np.mean(np.abs(residuals))
    residual_text = f"RMSE = {rmse:.2f}°C\nMAE = {mae:.2f}°C"
    ax2.text(
        0.05,
        0.95,
        residual_text,
        transform=ax2.transAxes,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        verticalalignment="top",
    )

    plt.tight_layout()
    st.pyplot(fig)

    # Interpretation text
    st.markdown("**Interpretation:**")

    # Correlation strength interpretation
    if abs(correlation) >= 0.8:
        correlation_strength = "very strong"
    elif abs(correlation) >= 0.6:
        correlation_strength = "strong"
    elif abs(correlation) >= 0.4:
        correlation_strength = "moderate"
    elif abs(correlation) >= 0.2:
        correlation_strength = "weak"
    else:
        correlation_strength = "very weak"

    st.write(
        f"- **Correlation:** {correlation_strength} {'positive' if correlation > 0 else 'negative'} relationship (r = {correlation:.3f})"
    )
    st.write(
        f"- **Explanation:** {r_value**2 * 100:.1f}% of water temperature variation is explained by air temperature"
    )
    st.write(
        f"- **Linear Model:** For every 1°C increase in air temperature, water temperature increases by {slope:.2f}°C on average"
    )
    st.write(f"- **Model Accuracy:** Typical prediction error is ±{rmse:.1f}°C (RMSE)")

    if p_value < 0.001:
        st.write(
            "- **Statistical Significance:** Highly significant relationship (p < 0.001)"
        )
    elif p_value < 0.05:
        st.write(
            "- **Statistical Significance:** Statistically significant relationship (p < 0.05)"
        )
    else:
        st.write(
            "- **Statistical Significance:** Not statistically significant (p ≥ 0.05)"
        )


def create_download_button(df: pd.DataFrame) -> None:
    """Create CSV download button."""
    csv = df.to_csv(index=False)
    filename = f"west_reservoir_temperature_{datetime.now().strftime('%Y%m%d')}.csv"

    st.download_button(
        label="📥 Download Data as CSV", data=csv, file_name=filename, mime="text/csv"
    )


def create_forecast_tab(df: pd.DataFrame, weather_df: pd.DataFrame = None) -> None:
    """Create the forecast tab content."""
    st.header("Temperature Forecast")

    # Filter data for the last 7 days and next 14 days
    today = pd.Timestamp(datetime.now().date())
    forecast_start = today - pd.Timedelta(days=7)
    forecast_end = today + pd.Timedelta(days=14)

    forecast_df = df[
        (df["Date"] >= forecast_start) & (df["Date"] <= forecast_end)
    ].copy()

    if forecast_df.empty:
        add_log_message(
            "warning", "No forecast data available for the selected period."
        )
        return

    # Show forecast chart
    st.subheader("Recent Temps and 14-Day Forecast")

    if "Type" in forecast_df.columns:
        # Combine all water temperature data into single series
        water_temp_data = forecast_df[
            forecast_df["Type"].isin(["Actual", "Imputed", "Predicted"])
        ].copy()

        # Create forecast chart with all water temperature data as single blue dotted line
        fig = px.line(
            water_temp_data,
            x="Date",
            y="Temperature",
            # title='Temperature Forecast (Last 7 Days + Next 14 Days)',
            labels={"Temperature": "Temperature (°C)", "Date": "Date"},
        )

        # Update to blue dotted line with markers
        fig.update_traces(
            name="Water Temperature",
            line=dict(color="blue", dash="dot", width=2),
            marker=dict(color="blue", size=4),
            mode="lines+markers",
            showlegend=True,
        )

        # Add weather data if available
        if weather_df is not None and not weather_df.empty:
            # Filter weather data for the same period
            weather_forecast = weather_df[
                (weather_df["Date"] >= forecast_start)
                & (weather_df["Date"] <= forecast_end)
            ].copy()

            if not weather_forecast.empty:
                # Add high temperatures
                if "Air_Temp_Max" in weather_forecast.columns:
                    fig.add_scatter(
                        x=weather_forecast["Date"],
                        y=weather_forecast["Air_Temp_Max"],
                        mode="lines",
                        name="Air Temp High",
                        line=dict(color="red", width=1, dash="dot"),
                        showlegend=True,
                    )

                # Add low temperatures
                if "Air_Temp_Min" in weather_forecast.columns:
                    fig.add_scatter(
                        x=weather_forecast["Date"],
                        y=weather_forecast["Air_Temp_Min"],
                        mode="lines",
                        name="Air Temp Low",
                        line=dict(color="lightblue", width=1, dash="dot"),
                        showlegend=True,
                    )

                # Add average air temperature
                fig.add_scatter(
                    x=weather_forecast["Date"],
                    y=weather_forecast["Air_Temperature"],
                    mode="lines",
                    name="Air Temp Avg",
                    line=dict(color="gray", width=2, dash="dash"),
                    showlegend=True,
                )

            # Add vertical line for "today" using shapes instead of add_vline
            fig.add_shape(
                type="line",
                x0=today,
                x1=today,
                y0=0,
                y1=1,
                yref="paper",
                line=dict(color="red", dash="dash"),
            )
            fig.add_annotation(
                x=today,
                y=1.02,
                yref="paper",
                text="Today",
                showarrow=False,
                font=dict(color="red", size=12),
            )
    else:
        # Fallback for data without type information
        fig = px.line(
            forecast_df,
            x="Date",
            y="Temperature",
            title="Temperature Trend (Last 7 Days)",
            labels={"Temperature": "Temperature (°C)", "Date": "Date"},
        )
        fig.update_traces(line_color="#1f77b4", line_width=2)

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Temperature (°C)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        dragmode=False,
        xaxis=dict(fixedrange=True),
        yaxis=dict(fixedrange=True),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displayModeBar": False,
            "scrollZoom": False,
            "doubleClick": False,
            "showTips": True,
            "displaylogo": False,
            "dragmode": False,
            "staticPlot": False,
        },
    )

    # Show forecast summary
    if "Type" in forecast_df.columns:
        predicted_data = forecast_df[forecast_df["Type"] == "Predicted"]
        if not predicted_data.empty:
            # Show forecast table in expandable section
            with st.expander("Detailed Forecast", expanded=False):
                forecast_table = predicted_data[["Date", "Temperature"]].copy()

                # Add air temperature data if available
                if weather_df is not None and not weather_df.empty:
                    # Merge with weather data for the same dates
                    weather_forecast = weather_df[
                        weather_df["Date"].isin(predicted_data["Date"])
                    ]

                    if not weather_forecast.empty:
                        forecast_table = pd.merge(
                            forecast_table,
                            weather_forecast[
                                ["Date", "Air_Temperature", "Air_Temp_Max"]
                            ],
                            on="Date",
                            how="left",
                        )

                        # Rename columns for display
                        if "Air_Temperature" in forecast_table.columns:
                            forecast_table = forecast_table.rename(
                                columns={"Air_Temperature": "Air Temp Avg"}
                            )
                        if "Air_Temp_Max" in forecast_table.columns:
                            forecast_table = forecast_table.rename(
                                columns={"Air_Temp_Max": "Air Temp High"}
                            )

                # Format the table
                forecast_table.loc[:, "Date"] = forecast_table["Date"].dt.strftime(
                    "%a %d/%m"
                )
                forecast_table.loc[:, "Temperature"] = forecast_table[
                    "Temperature"
                ].round(1)

                # Round air temperature columns if they exist
                for col in ["Air Temp Avg", "Air Temp High"]:
                    if col in forecast_table.columns:
                        forecast_table.loc[:, col] = forecast_table[col].round(1)

                # Rename water temperature column for clarity
                forecast_table = forecast_table.rename(
                    columns={"Temperature": "Water Temp"}
                )

                st.dataframe(forecast_table, use_container_width=True, hide_index=True)
        else:
            add_log_message(
                "info",
                "No predictions available. Enable predictions in the sidebar to see forecasts.",
            )


def create_historical_tab(
    df: pd.DataFrame, actual_df: pd.DataFrame, weather_df: pd.DataFrame = None
) -> None:
    """Create the historical data and statistics tab content."""
    st.header("Historical Data and Statistics")

    # Filter data to only show up to today's date (no future predictions/forecasts)
    today = pd.Timestamp(datetime.now().date())

    # Filter weather data to historical only
    historical_weather_df = None
    if weather_df is not None and not weather_df.empty:
        historical_weather_df = weather_df[weather_df["Date"] <= today].copy()

    # Create model predictions for historical comparison
    model_comparison_df = actual_df.copy()

    if historical_weather_df is not None and not historical_weather_df.empty:
        # Merge actual data with weather data
        merged_data = pd.merge(
            actual_df[["Date", "Temperature"]],
            historical_weather_df[["Date", "Air_Temperature"]],
            on="Date",
            how="inner",
        ).dropna()

        if len(merged_data) >= 30:
            # Generate physics model predictions
            try:
                physics_model = WaterTemperatureModel()
                training_dates = merged_data["Date"].dt.to_pydatetime()
                training_air_temps = merged_data["Air_Temperature"].values
                training_water_temps = merged_data["Temperature"].values

                # Fit physics model
                result = physics_model.fit_parameters(
                    training_air_temps, training_water_temps, training_dates
                )
                if result and result.success:
                    physics_predictions = physics_model.predict_temperature(
                        training_air_temps, training_water_temps[0], training_dates
                    )

                    # Add physics predictions to comparison dataframe
                    physics_df = pd.DataFrame(
                        {
                            "Date": merged_data["Date"],
                            "Temperature": physics_predictions,
                            "Type": "Physics Model",
                        }
                    )
                    model_comparison_df = pd.concat(
                        [model_comparison_df, physics_df], ignore_index=True
                    )

            except Exception as e:
                add_log_message(
                    "warning", f"Could not generate physics model predictions: {e}"
                )

            # Generate linear model predictions
            try:
                from sklearn.linear_model import LinearRegression

                linear_model = LinearRegression()
                linear_model.fit(
                    training_air_temps.reshape(-1, 1), training_water_temps
                )
                linear_predictions = linear_model.predict(
                    training_air_temps.reshape(-1, 1)
                )

                # Add linear predictions to comparison dataframe
                linear_df = pd.DataFrame(
                    {
                        "Date": merged_data["Date"],
                        "Temperature": linear_predictions,
                        "Type": "Linear Model",
                    }
                )
                model_comparison_df = pd.concat(
                    [model_comparison_df, linear_df], ignore_index=True
                )

            except Exception as e:
                add_log_message(
                    "warning", f"Could not generate linear model predictions: {e}"
                )

    # Ensure actual data has the right type
    if "Type" not in model_comparison_df.columns:
        model_comparison_df["Type"] = "Actual"
    else:
        model_comparison_df.loc[model_comparison_df["Type"].isna(), "Type"] = "Actual"

    # Sort by date
    model_comparison_df = model_comparison_df.sort_values("Date").reset_index(drop=True)

    # Main chart showing actual data + model predictions
    create_line_chart(model_comparison_df, historical_weather_df)

    # Monthly analysis (only actual data)
    create_monthly_analysis(actual_df)

    # Physics model comparison analysis (historical data only)
    if historical_weather_df is not None and not historical_weather_df.empty:
        create_physics_model_analysis(model_comparison_df, historical_weather_df)

    # Standard correlation analysis between water and air temperature (historical data only)
    create_correlation_analysis(model_comparison_df, historical_weather_df)

    # Recent data table (only actual data)
    display_recent_data(actual_df)

    # Download data (actual + model predictions)
    create_download_button(model_comparison_df)


def create_temperature_dashboard(
    df: pd.DataFrame, weather_df: pd.DataFrame = None
) -> None:
    """Create a dashboard showing key temperature metrics."""
    # st.markdown("### Temperature Dashboard")

    today = pd.Timestamp(datetime.now().date())
    yesterday = today - pd.Timedelta(days=1)
    tomorrow = today + pd.Timedelta(days=1)

    # Initialize values
    yesterday_water = yesterday_air = today_water = today_air = hottest_air = (
        hottest_water
    ) = "N/A"

    # Get yesterday's data
    if not df.empty:
        yesterday_data = df[df["Date"] == yesterday]
        if not yesterday_data.empty and yesterday_data["Type"].iloc[0] == "Actual":
            yesterday_water = f"{yesterday_data['Temperature'].iloc[0]:.1f}°C"

    # Get today's data - prefer actual, fallback to forecast
    if not df.empty:
        today_data = df[df["Date"] == today]
        if not today_data.empty:
            # Check if Type column exists (for backwards compatibility)
            if "Type" in today_data.columns:
                # Prefer actual data
                actual_today = today_data[today_data["Type"] == "Actual"]
                if not actual_today.empty:
                    today_water = f"{actual_today['Temperature'].iloc[0]:.1f}°C"
                else:
                    # Use forecast/predicted data if actual not available
                    forecast_today = today_data[
                        today_data["Type"].isin(["Predicted", "Physics Model", "Imputed"])
                    ]
                    if not forecast_today.empty:
                        today_water = f"{forecast_today['Temperature'].iloc[0]:.1f}°C*"
            else:
                # No Type column, use any available data for today
                today_water = f"{today_data['Temperature'].iloc[0]:.1f}°C"

    # Get air temperature data and find hottest forecasts
    if weather_df is not None and not weather_df.empty:
        # Yesterday's air temp
        yesterday_air_data = weather_df[weather_df["Date"] == yesterday]
        if not yesterday_air_data.empty:
            yesterday_avg = yesterday_air_data["Air_Temperature"].iloc[0]
            if "Air_Temp_Max" in yesterday_air_data.columns and not pd.isna(
                yesterday_air_data["Air_Temp_Max"].iloc[0]
            ):
                yesterday_high = yesterday_air_data["Air_Temp_Max"].iloc[0]
                yesterday_air = f"{yesterday_avg:.1f} / {yesterday_high:.1f}°C"
            else:
                yesterday_air = f"{yesterday_avg:.1f}°C"

        # Today's air temp
        today_air_data = weather_df[weather_df["Date"] == today]
        if not today_air_data.empty:
            today_avg = today_air_data["Air_Temperature"].iloc[0]
            if "Air_Temp_Max" in today_air_data.columns and not pd.isna(
                today_air_data["Air_Temp_Max"].iloc[0]
            ):
                today_high = today_air_data["Air_Temp_Max"].iloc[0]
                today_air = f"{today_avg:.1f} / {today_high:.1f}°C"
            else:
                today_air = f"{today_avg:.1f}°C"

        # Find highest air temperature in forecast (next 7 days) using daily max
        future_weather = weather_df[weather_df["Date"] > today]
        if not future_weather.empty:
            # Limit to next 7 days
            week_ahead = today + pd.Timedelta(days=7)
            future_week = future_weather[future_weather["Date"] <= week_ahead]

            if not future_week.empty:
                # Use Air_Temp_Max if available, otherwise fall back to Air_Temperature
                if (
                    "Air_Temp_Max" in future_week.columns
                    and not future_week["Air_Temp_Max"].isna().all()
                ):
                    highest_air_row = future_week.loc[
                        future_week["Air_Temp_Max"].idxmax()
                    ]
                    highest_air_temp = highest_air_row["Air_Temp_Max"]
                else:
                    highest_air_row = future_week.loc[
                        future_week["Air_Temperature"].idxmax()
                    ]
                    highest_air_temp = highest_air_row["Air_Temperature"]

                highest_air_date = highest_air_row["Date"].strftime("%a %d")
                hottest_air = f"{highest_air_temp:.1f}°C on {highest_air_date}"

    # Find hottest water temperature in forecast (next 7 days)
    if not df.empty:
        future_water = df[df["Date"] > today]
        if not future_water.empty:
            # Limit to next 7 days and only forecast data
            week_ahead = today + pd.Timedelta(days=7)
            future_week = future_water[
                (future_water["Date"] <= week_ahead)
                & (future_water["Type"].isin(["Predicted", "Physics Model"]))
            ]

            if not future_week.empty:
                hottest_water_row = future_week.loc[future_week["Temperature"].idxmax()]
                hottest_water_temp = hottest_water_row["Temperature"]
                hottest_water_date = hottest_water_row["Date"].strftime("%A")
                hottest_water = f"{hottest_water_temp:.1f}°C on {hottest_water_date}"

    st.markdown("## Water Temperatures")
    water_col1, col2 = st.columns(2)

    with water_col1:
        st.metric(
            label="Today's Water Temp",
            value=today_water,
            help="Today's water temperature (* = forecast if actual not available)",
            delta=None,  # Remove problematic delta calculation for now
        )

    with col2:
        # Neoprene recommendation based on today's water temperature
        neoprene_advice = "Check water temp"
        if not df.empty:
            today_data = df[df["Date"] == today]
            if not today_data.empty:
                # Get today's water temp (prefer actual, fallback to forecast)
                water_temp = None
                if "Type" in today_data.columns:
                    actual_today = today_data[today_data["Type"] == "Actual"]
                    if not actual_today.empty:
                        water_temp = actual_today["Temperature"].iloc[0]
                    else:
                        forecast_today = today_data[
                            today_data["Type"].isin(["Predicted", "Physics Model", "Imputed"])
                        ]
                        if not forecast_today.empty:
                            water_temp = forecast_today["Temperature"].iloc[0]
                else:
                    # No Type column, use any available data for today
                    water_temp = today_data["Temperature"].iloc[0]

                if water_temp is not None:
                    if water_temp > 16:
                        neoprene_advice = "No Way! ☀️"
                    elif water_temp > 10:
                        neoprene_advice = "If you like 🤷"
                    else:
                        neoprene_advice = "Yes! Unless you're a superhero ❄️"

        st.metric(
            label="Need Neoprene?",
            value=neoprene_advice,
            help="Neoprene wetsuit recommendation based on today's water temperature",
        )

    st.markdown("Predicted Water Temperatures")
    col3, col4 = st.columns(2)
    with col3:
        st.metric(
            label="Tomorrow the water will be",
            value=f"{future_water.iloc[0]['Temperature']:.1f}°C"
            if not future_water.empty
            else "No data",
            help="Tomorrow's predicted water temperature",
        )
    with col4:
        st.metric(
            label="Hottest water in  next 7 days",
            value=hottest_water,
            help="Highest forecasted water temperature in the next 7 days",
        )

    # with graph_col:

    # # Air Temperature Row
    # st.markdown("**🌤️ Air Temperatures**")
    # air_col1, air_col2, air_col3, air_col4 = st.columns(4)

    # with air_col1:
    #     st.metric(
    #         label="Yesterday Avg/High",
    #         value=yesterday_air,
    #         help="Yesterday's air temperature (average / high)"
    #     )

    # with air_col2:
    #     st.metric(
    #         label="Today Avg/High",
    #         value=today_air,
    #         help="Today's air temperature (average / high)"
    #     )

    # with air_col3:
    #     st.metric(
    #         label="Highest Air (7 days)",
    #         value=hottest_air,
    #         help="Highest daily maximum air temperature in the next 7 days"
    #     )

    # with air_col4:
    #     st.metric(
    #         label="",
    #         value="",
    #         help=""
    #     )

    # # Water Temperature Row
    # st.markdown("**🌊 Water Temperatures**")
    # water_col1, water_col2, water_col3, water_col4 = st.columns(4)

    # with water_col1:
    #     st.metric(
    #         label="Yesterday Water",
    #         value=yesterday_water,
    #         help="Yesterday's water temperature"
    #     )

    # with water_col2:
    #     st.metric(
    #         label="Today Water",
    #         value=today_water,
    #         help="Today's water temperature (* = forecast if actual not available)"
    #     )

    # with water_col3:
    #     st.metric(
    #         label="Hottest Water (7 days)",
    #         value=hottest_water,
    #         help="Highest forecasted water temperature in the next 7 days"
    #     )

    # with water_col4:
    #     # Neoprene recommendation based on today's water temperature
    #     neoprene_advice = "Check water temp"
    #     if not df.empty:
    #         today_data = df[df['Date'] == today]
    #         if not today_data.empty:
    #             # Get today's water temp (prefer actual, fallback to forecast)
    #             actual_today = today_data[today_data['Type'] == 'Actual']
    #             if not actual_today.empty:
    #                 water_temp = actual_today['Temperature'].iloc[0]
    #             else:
    #                 forecast_today = today_data[today_data['Type'].isin(['Predicted', 'Physics Model'])]
    #                 if not forecast_today.empty:
    #                     water_temp = forecast_today['Temperature'].iloc[0]
    #                 else:
    #                     water_temp = None

    #             if water_temp is not None:
    #                 if water_temp > 16:
    #                     neoprene_advice = "No Way! 🌊"
    #                 elif water_temp > 10:
    #                     neoprene_advice = "If you like 🤷"
    #                 else:
    #                     neoprene_advice = "Yes! Unless superhero 🦸"

    #     st.metric(
    #         label="Need Neoprene?",
    #         value=neoprene_advice,
    #         help="Neoprene wetsuit recommendation based on today's water temperature"
    #     )

    # st.divider()


def main() -> None:
    """Main application function."""
    st.title("West Reservoir Temperature Tracker")
    left, right = st.columns(2)
    with left:
        st.markdown(
            "Tracking and forecasting water temperature at West Reservoir, London"
        )
        st.info("""
        ℹ️ Water temperatures are taken each morning around 7am. 
        The water will often be warmer by the time you get in! Additionally, temperature varies 
        throughout the reservoir by both position and depth - this is just a snapshot of conditions.
        """)

    with right:
        st.image("image.png")

    # Add info explainer

    # Sidebar options
    st.sidebar.header("Prediction Options")
    enable_predictions = st.sidebar.checkbox(
        "Enable temperature predictions", value=True
    )

    if enable_predictions:
        prediction_method = st.sidebar.selectbox(
            "Prediction Method",
            ["Physics-based Model", "Statistical Model"],
            index=0,
            help="Physics-based: Uses degree-day accumulation model. Statistical: Uses machine learning with features.",
        )

        imputation_method = st.sidebar.selectbox(
            "Gap Filling Method",
            ["Physics-based", "Statistical"],
            index=0,
            help="Physics-based: Uses thermal model for gaps. Statistical: Uses interpolation + regression.",
        )

        # Store in session state for use in prediction functions
        st.session_state.use_physics_imputation = imputation_method == "Physics-based"

    st.sidebar.header("Debug Options")
    debug_mode = st.sidebar.checkbox("Debug mode", value=False)
    st.session_state.debug_mode = debug_mode

    st.sidebar.header("Data Management")
    if st.sidebar.button("Clear Cache & Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Debug: Test Google Form submission (if enabled)
    if debug_mode:
        st.sidebar.header("Debug Tools")

        # Community submissions toggle
        if st.sidebar.button("Toggle Community Submissions"):
            st.session_state.community_submissions_override = not st.session_state.get(
                "community_submissions_override", ENABLE_COMMUNITY_SUBMISSIONS
            )
            st.rerun()

        current_status = st.session_state.get(
            "community_submissions_override", ENABLE_COMMUNITY_SUBMISSIONS
        )
        st.sidebar.write(
            f"Community Submissions: {'Enabled' if current_status else 'Disabled'}"
        )

        if ENABLE_COMMUNITY_SUBMISSIONS or st.session_state.get(
            "community_submissions_override", False
        ):
            if st.sidebar.button("Test Google Form Submission"):
                success = submit_community_temperature(
                    datetime.now(), 16.5, "Debug Test"
                )
                if success:
                    st.sidebar.success("Test submission sent!")
                else:
                    st.sidebar.error("Test submission failed!")

            if st.sidebar.button("Check Form Field IDs"):
                st.sidebar.write("**Current Field IDs:**")
                for key, value in FORM_FIELD_IDS.items():
                    st.sidebar.write(f"- {key}: {value}")
                st.sidebar.write(f"**Form URL:** {GOOGLE_FORM_URL}")

    # Load reservoir data
    df = load_data()

    if df.empty:
        add_log_message("error", "No data available. Please check the data source.")
        return

    # Incorporate any community temperature submissions
    df = incorporate_community_temps(df)

    # Store main dataframe in session state for form access
    st.session_state.main_df = df

    # Initialize weather_df
    weather_df = None

    # If predictions enabled, get weather data and create predictions
    if enable_predictions:
        with st.spinner("Loading weather data for predictions..."):
            # Get date range for weather data
            start_date = df["Date"].min() - pd.Timedelta(
                days=30
            )  # Extra buffer for training
            end_date = datetime.now() + timedelta(
                days=14
            )  # Extended future predictions

            if debug_mode:
                st.write(
                    f"🔍 Debug: Requesting weather data from {start_date.date()} to {end_date.date()}"
                )

            weather_df = get_weather_data(start_date, end_date)

            if not weather_df.empty:
                if debug_mode:
                    st.write(f"🔍 Debug: Weather data shape: {weather_df.shape}")
                    st.write(
                        f"🔍 Debug: Weather date range: {weather_df['Date'].min().date()} to {weather_df['Date'].max().date()}"
                    )
                    st.write(f"🔍 Debug: Today's date: {datetime.now().date()}")
                    future_weather = weather_df[weather_df["Date"] > datetime.now()]
                    st.write(
                        f"🔍 Debug: Future weather data points: {len(future_weather)}"
                    )

                if prediction_method == "Physics-based Model":
                    df = create_temperature_predictions_physics(df, weather_df)
                else:
                    df = create_temperature_predictions(df, weather_df)
                prediction_count = (
                    len(df[df.get("Type", "") == "Predicted"])
                    if "Type" in df.columns
                    else 0
                )
                if prediction_count > 0:
                    add_log_message(
                        "info",
                        f"✨ Generated {prediction_count} temperature predictions based on weather data",
                    )
                else:
                    add_log_message(
                        "warning",
                        "No future predictions generated - may need more recent weather data",
                    )
            else:
                add_log_message(
                    "warning", "Could not load weather data for predictions"
                )
    else:
        # Even if predictions are disabled, load weather data for the forecast tab
        start_date = df["Date"].min() - pd.Timedelta(days=7)
        end_date = datetime.now() + timedelta(days=14)
        weather_df = get_weather_data(start_date, end_date)

    # Check if today's temperature is missing and show community form (if enabled)
    community_enabled = ENABLE_COMMUNITY_SUBMISSIONS or st.session_state.get(
        "community_submissions_override", False
    )
    if community_enabled:
        actual_df = df[df["Type"] == "Actual"] if "Type" in df.columns else df
        if check_today_temperature_missing(actual_df):
            create_community_temp_form()

    # Display temperature dashboard at the top
    create_temperature_dashboard(df, weather_df)

    # Display statistics (only for actual data)
    actual_df = df[df["Type"] == "Actual"] if "Type" in df.columns else df
    display_statistics(actual_df)

    # Create tabs
    tab1, tab2 = st.tabs(["Forecast", "Historical Data and Statistics"])

    with tab1:
        create_forecast_tab(df, weather_df)

    with tab2:
        create_historical_tab(df, actual_df, weather_df)

    st.markdown(
        """ 
        ## About the project

        Hi! I'm Tom, I''m a local and have been a regular swimmer at the res for a year. 

        I enjoy tracking the temperatures so I decided to make this app. 
        I've recorded (most) of the temperatures since November 2024, and used that data to train a small model to predict future temperatures. 
        The model may get an upgrade in future!
        Forecast weather data informs the future water temperature predictions, from open weather map, and historic data from meteostat.
        """
    )
    display_log_messages()


if __name__ == "__main__":
    main()

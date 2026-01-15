"""Configuration for West Reservoir Temperature Tracker"""

import os
import streamlit as st


class ConfigError(Exception):
    """Raised when configuration is invalid or missing"""
    pass


# Google Sheets configuration
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/export?format=csv&gid=0"

# West Reservoir location (London, UK)
RESERVOIR_LAT = 51.566938
RESERVOIR_LON = -0.090492

# Request timeout for API calls
REQUEST_TIMEOUT = 30


def get_openweather_api_key() -> str:
    """
    Get OpenWeatherMap API key from environment or Streamlit secrets.

    Returns:
        str: API key

    Raises:
        ConfigError: If API key is not found
    """
    # Try environment variable first
    api_key = os.getenv("OPENWEATHER_API_KEY")

    # Try Streamlit secrets if environment variable not set
    if not api_key:
        try:
            if hasattr(st, "secrets") and "OPENWEATHER_API_KEY" in st.secrets:
                api_key = st.secrets["OPENWEATHER_API_KEY"]
        except Exception:
            pass

    if not api_key:
        raise ConfigError(
            "OpenWeatherMap API key not found. "
            "Set OPENWEATHER_API_KEY environment variable or add to .streamlit/secrets.toml"
        )

    return api_key

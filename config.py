"""Configuration for West Reservoir Temperature Tracker"""

import os
import re
from pathlib import Path
import streamlit as st


class ConfigError(Exception):
    """Raised when configuration is invalid or missing"""
    pass


def _read_secrets_file(key: str) -> str | None:
    """Read a key directly from .streamlit/secrets.toml as fallback"""
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        content = secrets_path.read_text()
        match = re.search(rf'{key}\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return None


# Google Sheets configuration
GOOGLE_SHEETS_URL = "https://docs.google.com/spreadsheets/d/1HNnucep6pv2jCFg2bYR_gV78XbYvWYyjx9y9tTNVapw/export?format=csv&gid=0"

# West Reservoir location (London, UK)
RESERVOIR_LAT = 51.566938
RESERVOIR_LON = -0.090492

# Request timeout for API calls
REQUEST_TIMEOUT = 30

# Feature flags
ENABLE_MOTHERDUCK = True  # Enable MotherDuck storage for forecast history


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
            api_key = st.secrets["OPENWEATHER_API_KEY"]
        except Exception:
            # Fallback: read directly from secrets file
            api_key = _read_secrets_file("OPENWEATHER_API_KEY")

    if not api_key:
        raise ConfigError(
            "OpenWeatherMap API key not found. "
            "Set OPENWEATHER_API_KEY environment variable or add to .streamlit/secrets.toml"
        )

    return api_key


def get_motherduck_token() -> str:
    """
    Get MotherDuck token from environment or Streamlit secrets.

    Returns:
        str: MotherDuck token

    Raises:
        ConfigError: If token is not found
    """
    # Try environment variable first
    token = os.getenv("MOTHERDUCK_TOKEN")

    # Try Streamlit secrets if environment variable not set
    if not token:
        try:
            token = st.secrets["MOTHERDUCK_TOKEN"]
        except Exception:
            # Fallback: read directly from secrets file
            token = _read_secrets_file("MOTHERDUCK_TOKEN")

    if not token:
        raise ConfigError(
            "MotherDuck token not found. "
            "Set MOTHERDUCK_TOKEN environment variable or add to .streamlit/secrets.toml"
        )

    return token

"""
Connectivity check for the real MotherDuck database.

This replaces a test that ran `CREATE DATABASE IF NOT EXISTS west_reservoir`
against production on a bare `pytest`, unmarked and unskippable. Reading is
enough to answer the only question worth asking here - are the credentials
good and is the schema where we left it - so nothing in this file writes.

Marked `integration`: it needs a token and the network.
Skip it with `pytest -m "not integration"`.
"""

import os
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


# Tables the app reads and writes. Missing one is a real failure, not a skip.
EXPECTED_TABLES = {
    "water_temp_predictions",
    "air_temp_forecasts_3hourly",
    "weather_forecasts_hourly",
}


def _token():
    """Token from the environment, else from .streamlit/secrets.toml."""
    env = os.getenv("MOTHERDUCK_TOKEN")
    if env:
        return env

    secrets = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if secrets.exists():
        match = re.search(r'MOTHERDUCK_TOKEN\s*=\s*"([^"]+)"', secrets.read_text())
        if match:
            return match.group(1)
    return None


def test_motherduck_credentials_and_schema_are_reachable():
    """Connect read-only to the existing database and confirm the tables exist."""
    import duckdb

    token = _token()
    if not token:
        pytest.skip("MOTHERDUCK_TOKEN not set in env or .streamlit/secrets.toml")

    # Connect straight into the database. No CREATE DATABASE: a test must not
    # be the thing that provisions production.
    conn = duckdb.connect(f"md:west_reservoir?motherduck_token={token}")
    try:
        assert conn.execute("SELECT 1").fetchone()[0] == 1

        tables = {row[0] for row in conn.execute("SHOW TABLES").fetchall()}
        missing = EXPECTED_TABLES - tables
        assert not missing, f"Expected tables missing from west_reservoir: {missing}"
    finally:
        conn.close()

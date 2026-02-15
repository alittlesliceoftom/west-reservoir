"""Test MotherDuck connection"""

import os
import re
from pathlib import Path


def get_token_from_secrets():
    """Read token directly from secrets.toml"""
    secrets_path = Path(__file__).parent / ".streamlit" / "secrets.toml"
    if secrets_path.exists():
        content = secrets_path.read_text()
        # Simple TOML parsing for key = "value"
        match = re.search(r'MOTHERDUCK_TOKEN\s*=\s*"([^"]+)"', content)
        if match:
            return match.group(1)
    return None


def test_motherduck_connection():
    """Test that we can connect to MotherDuck and run a query"""
    import duckdb

    # Try env var first, then secrets file
    token = os.getenv("MOTHERDUCK_TOKEN") or get_token_from_secrets()

    assert token is not None, "MOTHERDUCK_TOKEN not found in env or secrets.toml"
    print(f"Token found: {token[:20]}...")

    # Connect to MotherDuck (no database specified initially)
    conn = duckdb.connect(f"md:?motherduck_token={token}")

    # Create database if it doesn't exist
    conn.execute("CREATE DATABASE IF NOT EXISTS west_reservoir")
    conn.execute("USE west_reservoir")
    print("Connected to west_reservoir database")

    # Test query
    result = conn.execute("SELECT 1 as test").fetchone()
    assert result[0] == 1, "Basic query failed"
    print("Basic query OK")

    # List tables
    tables = conn.execute("SHOW TABLES").fetchall()
    print(f"Tables in database: {[t[0] for t in tables]}")

    conn.close()
    print("MotherDuck connection test PASSED")


if __name__ == "__main__":
    test_motherduck_connection()

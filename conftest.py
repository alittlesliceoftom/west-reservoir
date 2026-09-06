"""
Shared fixtures.

Two things were being rebuilt in every test file: an hourly frame of
datetime + N columns, and an in-memory DuckDB table matching one of the
storage schemas. There were five hand-rolled variants of the first and four of
the second, differing only in which columns they carried - so a change to the
shape of a weather frame meant finding all five.

These are plain functions, not pytest fixtures, and the test modules import
them directly (`from conftest import hourly_frame`). That works because the
tests already import `data` and `accuracy` from the repo root, so the root is
on sys.path. They are builders with no setup, teardown or scoping, so
fixture-ness would buy nothing and would stop class-level helpers from
delegating to them. Do not move them to another module without updating the
imports.
"""

import duckdb
import pandas as pd


def _column(values, n):
    """Broadcast a scalar down the column; pass a sequence through unchanged."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        return [values] * n
    if len(values) != n:
        raise ValueError(
            f"column has {len(values)} values but the frame has {n} rows"
        )
    return list(values)


def hourly_frame(start, n_hours, column="datetime", tz=None, **columns):
    """
    Build an hourly frame: a datetime column plus whatever you name.

    Each keyword is either a scalar, broadcast down the column, or a sequence
    used as-is - some tests need a constant, others need a ramp they can
    assert a max on.

        hourly_frame("2026-05-01 07:00", 12, air_temp=15.0)
        hourly_frame("2026-05-01", 3, shortwave_radiation=[0.0, 100.0, 400.0])

    Args:
        start: First timestamp.
        n_hours: Number of consecutive hourly rows.
        column: Name of the datetime column ("datetime" or "date").
        tz: Optional timezone for the datetime column.
        **columns: Column name -> scalar or sequence of n_hours values.
    """
    frame = {
        column: pd.date_range(pd.Timestamp(start), periods=n_hours, freq="h", tz=tz)
    }
    for name, values in columns.items():
        frame[name] = _column(values, n_hours)
    return pd.DataFrame(frame)


def weather_frame(start, n_hours, air_temp=15.0, shortwave=0.0, cloud=100.0):
    """An hourly frame carrying the three columns the forecaster requires."""
    return hourly_frame(
        start, n_hours,
        air_temp=air_temp,
        shortwave_radiation=shortwave,
        cloud_cover=cloud,
    )


# Storage table definitions, as {column: SQL type}. TIMESTAMPTZ variants exist
# because MotherDuck returns timezone-aware timestamps and local DuckDB does
# not - see TestStorageReadsCoerceTimezones.
WATER_PREDICTIONS_COLUMNS = {
    "forecast_created_date": "DATE NOT NULL",
    "forecast_created_timestamp": "TIMESTAMP NOT NULL",
    "target_date": "DATE NOT NULL",
    "water_temp": "DOUBLE NOT NULL",
    "heat_transfer_coeff": "DOUBLE NOT NULL",
    "start_water_temp": "DOUBLE NOT NULL",
    "simulation_hours": "INTEGER NOT NULL",
    "source_air_forecast_timestamp": "TIMESTAMP NOT NULL",
}

AIR_FORECAST_3HOURLY_COLUMNS = {
    "forecast_created_timestamp": "TIMESTAMP NOT NULL",
    "target_datetime": "TIMESTAMP NOT NULL",
    "air_temp": "DOUBLE NOT NULL",
    "source": "VARCHAR DEFAULT 'OpenWeatherMap'",
}


def tz_aware(columns, *names):
    """The same schema with the named columns declared TIMESTAMPTZ."""
    return {
        name: spec.replace("TIMESTAMP ", "TIMESTAMPTZ ")
        if name in names else spec
        for name, spec in columns.items()
    }


def raw_table(name, columns, rows, timezone="UTC"):
    """
    An in-memory DuckDB holding one table, built from a column spec and rows.

    No primary key: these tables exist to exercise the SQL and the read
    methods' post-processing, not the constraint. (DuckDB also refuses to type
    a key column TIMESTAMPTZ, which the tz tests need.)

        conn = raw_table("air_temp_forecasts_3hourly",
                         AIR_FORECAST_3HOURLY_COLUMNS, rows)
    """
    conn = duckdb.connect(":memory:")
    conn.execute(f"SET TimeZone='{timezone}'")
    spec = ", ".join(f"{col} {typ}" for col, typ in columns.items())
    conn.execute(f"CREATE TABLE {name} ({spec})")

    placeholders = ", ".join(["?"] * len(columns))
    for row in rows:
        conn.execute(f"INSERT INTO {name} VALUES ({placeholders})", list(row))
    return conn

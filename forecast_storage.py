"""Forecast storage functions for MotherDuck database"""

import pandas as pd
import duckdb
from datetime import datetime
from typing import Optional
from config import get_motherduck_token, ConfigError


class ForecastStorageError(Exception):
    """Raised when forecast storage operations fail"""
    pass


# Measure columns of weather_forecasts_hourly, in table order. A new model
# input is added here and as a column, not as a new table.
WEATHER_MEASURES = ["air_temp", "shortwave_radiation", "cloud_cover"]


# Selects the final forecast run of each creation day.
#
# rank(), not row_number(): one run is MANY rows sharing a single
# forecast_created_timestamp (one per target date). row_number() would keep
# only one of them and silently drop the rest of the horizons.
#
# The horizon filter is load-bearing: water_temp_predictions also holds rows
# targeting dates BEFORE the run (historical backfill written alongside real
# forecasts). Those are not forecasts and must never be scored.
LAST_WATER_RUN_PER_DAY_SQL = """
    WITH ranked AS (
        SELECT *,
            rank() OVER (
                PARTITION BY forecast_created_date
                ORDER BY forecast_created_timestamp DESC
            ) AS run_rank
        FROM water_temp_predictions
        WHERE date_diff('day', forecast_created_date, target_date) BETWEEN 0 AND ?
    )
    SELECT
        forecast_created_date,
        forecast_created_timestamp,
        target_date,
        water_temp AS forecast_temp,
        date_diff('day', forecast_created_date, target_date) AS horizon_days
    FROM ranked
    WHERE run_rank = 1
    ORDER BY target_date, horizon_days
"""


# Final 3-hourly air forecast run of each creation day, all rows of that run.
#
# This table has no forecast_created_date column, so the partition casts the
# timestamp. rank() for the same reason as above: a run is many rows sharing
# one creation timestamp.
LAST_AIR_RUN_PER_DAY_SQL = """
    WITH ranked AS (
        SELECT
            CAST(forecast_created_timestamp AS DATE) AS forecast_created_date,
            target_datetime,
            air_temp,
            rank() OVER (
                PARTITION BY CAST(forecast_created_timestamp AS DATE)
                ORDER BY forecast_created_timestamp DESC
            ) AS run_rank
        FROM air_temp_forecasts_3hourly
    )
    SELECT forecast_created_date, target_datetime, air_temp
    FROM ranked
    WHERE run_rank = 1
    ORDER BY forecast_created_date, target_datetime
"""


# Final hourly weather forecast run of each creation day, all rows of it.
# Same rank() reasoning as the air query above. Partitioned by source as well
# as creation day, so one source going quiet does not suppress another's run.
LAST_WEATHER_RUN_PER_DAY_SQL = """
    WITH ranked AS (
        SELECT
            CAST(forecast_created_timestamp AS DATE) AS forecast_created_date,
            target_datetime,
            source,
            air_temp,
            shortwave_radiation,
            cloud_cover,
            rank() OVER (
                PARTITION BY CAST(forecast_created_timestamp AS DATE), source
                ORDER BY forecast_created_timestamp DESC
            ) AS run_rank
        FROM weather_forecasts_hourly
    )
    SELECT
        forecast_created_date,
        target_datetime,
        source,
        air_temp,
        shortwave_radiation,
        cloud_cover
    FROM ranked
    WHERE run_rank = 1
    ORDER BY forecast_created_date, target_datetime, source
"""


class ForecastStorage:
    """Handles storage and retrieval of forecasts in MotherDuck."""

    def __init__(self, database: str = "west_reservoir"):
        """Initialize connection to MotherDuck."""
        self.database = database
        self._conn = None

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        """Get or create MotherDuck connection."""
        if self._conn is None:
            try:
                token = get_motherduck_token()
                # Connect without database first to ensure it exists
                conn = duckdb.connect(f"md:?motherduck_token={token}")
                conn.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                conn.close()
                connection_string = f"md:{self.database}?motherduck_token={token}"
                self._conn = duckdb.connect(connection_string)
            except ConfigError as e:
                raise ForecastStorageError(f"Cannot connect to MotherDuck: {e}")
            except Exception as e:
                raise ForecastStorageError(f"MotherDuck connection failed: {e}")
        return self._conn

    def initialize_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        conn = self._get_connection()

        conn.execute("""
            CREATE TABLE IF NOT EXISTS water_temp_predictions (
                forecast_created_date DATE NOT NULL,
                forecast_created_timestamp TIMESTAMP NOT NULL,
                target_date DATE NOT NULL,
                water_temp DOUBLE NOT NULL,
                heat_transfer_coeff DOUBLE NOT NULL,
                start_water_temp DOUBLE NOT NULL,
                simulation_hours INTEGER NOT NULL,
                source_air_forecast_timestamp TIMESTAMP NOT NULL,
                PRIMARY KEY (forecast_created_timestamp, target_date)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_prediction_for_date
            ON water_temp_predictions(target_date, forecast_created_date)
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS air_temp_forecasts_3hourly (
                forecast_created_timestamp TIMESTAMP NOT NULL,
                target_datetime TIMESTAMP NOT NULL,
                air_temp DOUBLE NOT NULL,
                source VARCHAR DEFAULT 'OpenWeatherMap',
                PRIMARY KEY (forecast_created_timestamp, target_datetime)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecast_3h_target
            ON air_temp_forecasts_3hourly(target_datetime, forecast_created_timestamp)
        """)

        # One hourly table for every weather forecast, whatever the measure.
        #
        # Wide, with a source dimension: a new model input is a new column, not
        # a new table with its own store method, retrieval method and near
        # identical rank() query. Measure columns are nullable because a source
        # supplies only what it publishes - Open-Meteo sends solar and cloud,
        # OpenWeatherMap sends air temperature - and source is in the key so two
        # sources can forecast the same hour without colliding.
        #
        # Hourly because Open-Meteo returns hourly natively, so nothing needs
        # interpolating on the way in. Air temperature joins this table when it
        # moves to Open-Meteo hourly (issue #39).
        conn.execute("""
            CREATE TABLE IF NOT EXISTS weather_forecasts_hourly (
                forecast_created_timestamp TIMESTAMP NOT NULL,
                target_datetime TIMESTAMP NOT NULL,
                source VARCHAR NOT NULL,
                air_temp DOUBLE,
                shortwave_radiation DOUBLE,
                cloud_cover DOUBLE,
                PRIMARY KEY (forecast_created_timestamp, target_datetime, source)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_weather_forecast_target
            ON weather_forecasts_hourly(target_datetime, forecast_created_timestamp)
        """)

    def store_water_predictions(
        self,
        predictions_df: pd.DataFrame,
        forecast_created_timestamp: datetime,
        heat_transfer_coeff: float,
        start_water_temp: float
    ) -> None:
        """
        Store water temperature predictions.

        Args:
            predictions_df: DataFrame with source=="PREDICTED" rows
            forecast_created_timestamp: When this forecast was created
            heat_transfer_coeff: Model's k value used
            start_water_temp: Starting water temp for simulation
        """
        conn = self._get_connection()

        # Truncate to hour so the PK naturally deduplicates within each hour
        forecast_created_hour = forecast_created_timestamp.replace(minute=0, second=0, microsecond=0)
        predictions_to_store = predictions_df.copy()
        predictions_to_store['forecast_created_date'] = forecast_created_hour.date()
        predictions_to_store['forecast_created_timestamp'] = forecast_created_hour
        predictions_to_store['target_date'] = predictions_to_store['date'].dt.date
        predictions_to_store['heat_transfer_coeff'] = heat_transfer_coeff
        predictions_to_store['start_water_temp'] = start_water_temp
        predictions_to_store['simulation_hours'] = 24
        predictions_to_store['source_air_forecast_timestamp'] = forecast_created_hour

        # Select columns in correct order
        predictions_to_store = predictions_to_store[[
            'forecast_created_date',
            'forecast_created_timestamp',
            'target_date',
            'water_temp',
            'heat_transfer_coeff',
            'start_water_temp',
            'simulation_hours',
            'source_air_forecast_timestamp'
        ]]

        try:
            conn.execute("""
                INSERT INTO water_temp_predictions
                SELECT * FROM predictions_to_store
            """)
        except Exception as e:
            err_msg = str(e).lower()
            if "primary key" in err_msg or "unique" in err_msg or "duplicate" in err_msg:
                pass  # Silently ignore duplicates
            else:
                raise ForecastStorageError(f"Failed to store water predictions: {e}")

    def store_air_forecast_3hourly(
        self,
        forecast_df: pd.DataFrame,
        forecast_created_timestamp: datetime
    ) -> None:
        """
        Store raw 3-hourly air temperature forecast.

        Args:
            forecast_df: DataFrame with columns: datetime, air_temp
            forecast_created_timestamp: When this forecast was fetched
        """
        conn = self._get_connection()

        forecast_to_store = forecast_df.copy()
        # Truncate to hour so the PK naturally deduplicates within each hour
        forecast_created_hour = forecast_created_timestamp.replace(minute=0, second=0, microsecond=0)
        forecast_to_store['forecast_created_timestamp'] = forecast_created_hour
        forecast_to_store['target_datetime'] = forecast_to_store['datetime']
        forecast_to_store['source'] = 'OpenWeatherMap'

        forecast_to_store = forecast_to_store[[
            'forecast_created_timestamp',
            'target_datetime',
            'air_temp',
            'source'
        ]]

        try:
            conn.execute("""
                INSERT INTO air_temp_forecasts_3hourly
                SELECT * FROM forecast_to_store
            """)
        except Exception as e:
            err_msg = str(e).lower()
            if "primary key" in err_msg or "unique" in err_msg or "duplicate" in err_msg:
                pass  # Silently ignore duplicates
            else:
                raise ForecastStorageError(f"Failed to store 3-hourly air forecast: {e}")

    def store_weather_forecast(
        self,
        forecast_df: pd.DataFrame,
        forecast_created_timestamp: datetime,
        source: str
    ) -> None:
        """
        Store an hourly weather forecast in weather_forecasts_hourly.

        Whichever of the measure columns the frame carries are stored; the rest
        are left NULL, because a source publishes only what it publishes.

        **A source must write all of its measures in one frame.** The primary
        key is (created_timestamp, target_datetime, source), so a second call
        for the same source and hour - solar first, then air - collides and is
        swallowed as a duplicate, losing the second set of measures without an
        error. When air temperature moves here (issue #39) it must arrive in
        the same frame as solar and cloud, not as a separate write. If that
        ever becomes inconvenient, switch this to ON CONFLICT DO UPDATE with
        COALESCE per column so partial writes merge instead.

        Stored so backtests can feed the model the forecast it actually had
        rather than what actually happened. Until this has been accumulating,
        replay results are an optimistic bound (issue #29).

        Args:
            forecast_df: DataFrame with 'datetime' plus any of 'air_temp',
                         'shortwave_radiation', 'cloud_cover'.
            forecast_created_timestamp: When this forecast was fetched.
            source: Who published it, e.g. 'Open-Meteo'.
        """
        if forecast_df is None or forecast_df.empty:
            return

        measures = [c for c in WEATHER_MEASURES if c in forecast_df.columns]
        if not measures:
            raise ForecastStorageError(
                "Forecast has none of the measure columns "
                f"{WEATHER_MEASURES}: got {list(forecast_df.columns)}"
            )

        conn = self._get_connection()

        forecast_to_store = forecast_df.copy()
        # Truncate to hour so the PK naturally deduplicates within each hour
        forecast_created_hour = forecast_created_timestamp.replace(minute=0, second=0, microsecond=0)
        forecast_to_store['forecast_created_timestamp'] = forecast_created_hour
        forecast_to_store['target_datetime'] = forecast_to_store['datetime']
        forecast_to_store['source'] = source

        # Absent measures are stored as NULL, so the column list is fixed and
        # the insert does not depend on which source is writing.
        for measure in WEATHER_MEASURES:
            if measure not in forecast_to_store.columns:
                forecast_to_store[measure] = None

        forecast_to_store = forecast_to_store[[
            'forecast_created_timestamp',
            'target_datetime',
            'source',
        ] + WEATHER_MEASURES]

        try:
            conn.execute("""
                INSERT INTO weather_forecasts_hourly
                SELECT * FROM forecast_to_store
            """)
        except Exception as e:
            err_msg = str(e).lower()
            if "primary key" in err_msg or "unique" in err_msg or "duplicate" in err_msg:
                pass  # Likely duplicate key - not a critical error
            else:
                raise ForecastStorageError(f"Failed to store weather forecast: {e}")

    def get_weather_forecasts_last_run_per_day(self) -> pd.DataFrame:
        """
        Retrieve every stored weather forecast, one run per creation day.

        Fetched in bulk for the same reason as the air forecasts: the replay
        covers hundreds of anchors, so per-anchor queries are not an option.

        Returns:
            DataFrame: forecast_created_date, target_datetime, source,
                       air_temp, shortwave_radiation, cloud_cover
        """
        conn = self._get_connection()
        result = conn.execute(LAST_WEATHER_RUN_PER_DAY_SQL).fetchdf()

        if result.empty:
            return result

        result["forecast_created_date"] = pd.to_datetime(result["forecast_created_date"])
        result["target_datetime"] = pd.to_datetime(
            result["target_datetime"]
        ).dt.tz_localize(None)

        return result

    def get_forecast_for_date(self, target_date: datetime) -> Optional[pd.DataFrame]:
        """
        Retrieve the most recent 3-hourly forecast covering a specific date.

        Args:
            target_date: The date to get forecast for

        Returns:
            DataFrame with datetime and air_temp columns, or None if not found
        """
        conn = self._get_connection()

        # Find the most recent forecast that covers this date
        result = conn.execute("""
            WITH latest_forecast AS (
                SELECT MAX(forecast_created_timestamp) as latest_ts
                FROM air_temp_forecasts_3hourly
                WHERE DATE(target_datetime) = ?
            )
            SELECT target_datetime as datetime, air_temp
            FROM air_temp_forecasts_3hourly
            WHERE forecast_created_timestamp = (SELECT latest_ts FROM latest_forecast)
              AND DATE(target_datetime) = ?
            ORDER BY target_datetime
        """, [target_date.date(), target_date.date()]).fetchdf()

        if result.empty:
            return None

        result["datetime"] = pd.to_datetime(result["datetime"]).dt.tz_localize(None)

        return result

    def get_forecasts_for_gap(
        self,
        gap_start: datetime,
        gap_end: datetime
    ) -> Optional[pd.DataFrame]:
        """
        Retrieve stored 3-hourly forecasts that cover a time gap.

        Finds the most recent forecast run that has data covering the gap period.
        This is used to fill gaps between Meteostat historical data and live OWM forecast.

        Args:
            gap_start: Start of the gap (exclusive - data after this time)
            gap_end: End of the gap (exclusive - data before this time)

        Returns:
            DataFrame with datetime and air_temp columns, or None if not found
        """
        conn = self._get_connection()

        # Find forecasts that cover any part of the gap period
        # Use the most recent forecast run that has data in this range
        result = conn.execute("""
            WITH forecasts_in_gap AS (
                SELECT
                    forecast_created_timestamp,
                    target_datetime,
                    air_temp
                FROM air_temp_forecasts_3hourly
                WHERE target_datetime > ?
                  AND target_datetime < ?
            ),
            latest_forecast AS (
                SELECT MAX(forecast_created_timestamp) as latest_ts
                FROM forecasts_in_gap
            )
            SELECT target_datetime as datetime, air_temp
            FROM forecasts_in_gap
            WHERE forecast_created_timestamp = (SELECT latest_ts FROM latest_forecast)
            ORDER BY target_datetime
        """, [gap_start, gap_end]).fetchdf()

        if result.empty:
            return None

        # MotherDuck returns timezone-aware timestamps, but our local data is naive
        result["datetime"] = pd.to_datetime(result["datetime"]).dt.tz_localize(None)

        return result

    def get_water_predictions_last_run_per_day(
        self, max_horizon: int = 5
    ) -> pd.DataFrame:
        """
        Retrieve stored water-temp forecasts, one run per creation day.

        Uses the final run of each creation day - the forecast a user would
        have seen at end of day.

        Args:
            max_horizon: Largest days-ahead horizon to return.

        Returns:
            DataFrame: forecast_created_date, forecast_created_timestamp,
                       target_date, forecast_temp, horizon_days
        """
        conn = self._get_connection()
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [max_horizon]).fetchdf()

        if result.empty:
            return result

        # MotherDuck returns timezone-aware timestamps, but our local data is naive
        result["forecast_created_timestamp"] = pd.to_datetime(
            result["forecast_created_timestamp"]
        ).dt.tz_localize(None)
        result["forecast_created_date"] = pd.to_datetime(result["forecast_created_date"])
        result["target_date"] = pd.to_datetime(result["target_date"])

        return result

    def get_air_forecasts_3hourly_last_run_per_day(self) -> pd.DataFrame:
        """
        Retrieve every stored 3-hourly air forecast, one run per creation day.

        Fetched in bulk for backtesting: the replay needs the forecast created
        ON each anchor date. Do not use get_forecast_for_date for this - it
        returns the most recent forecast covering a date, which for a past
        anchor may be a later, better-informed run (future leakage).

        Returns:
            DataFrame: forecast_created_date, target_datetime, air_temp
        """
        conn = self._get_connection()
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        if result.empty:
            return result

        result["forecast_created_date"] = pd.to_datetime(result["forecast_created_date"])
        result["target_datetime"] = pd.to_datetime(
            result["target_datetime"]
        ).dt.tz_localize(None)

        return result

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

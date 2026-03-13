"""Forecast storage functions for MotherDuck database"""

import pandas as pd
import duckdb
from datetime import datetime
from typing import Optional
from config import get_motherduck_token, ConfigError


class ForecastStorageError(Exception):
    """Raised when forecast storage operations fail"""
    pass


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
                # Now connect to the database
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

        # Create air_temp_forecasts table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS air_temp_forecasts (
                forecast_created_date DATE NOT NULL,
                forecast_created_timestamp TIMESTAMP NOT NULL,
                target_date DATE NOT NULL,
                air_temp DOUBLE NOT NULL,
                air_temp_min DOUBLE NOT NULL,
                air_temp_max DOUBLE NOT NULL,
                source VARCHAR DEFAULT 'OpenWeatherMap',
                PRIMARY KEY (forecast_created_timestamp, target_date)
            )
        """)

        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_forecast_for_date
            ON air_temp_forecasts(target_date, forecast_created_date)
        """)

        # Create water_temp_predictions table
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

        # Create air_temp_forecasts_3hourly table
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

    def store_air_forecast(
        self,
        forecast_df: pd.DataFrame,
        forecast_created_timestamp: datetime
    ) -> None:
        """
        Store raw air temperature forecast.

        Args:
            forecast_df: DataFrame with columns: date, air_temp, air_temp_min, air_temp_max
            forecast_created_timestamp: When this forecast was fetched
        """
        conn = self._get_connection()

        # Add timestamp columns
        forecast_to_store = forecast_df.copy()
        forecast_to_store['forecast_created_date'] = forecast_created_timestamp.date()
        forecast_to_store['forecast_created_timestamp'] = forecast_created_timestamp
        forecast_to_store['target_date'] = forecast_to_store['date'].dt.date
        forecast_to_store['source'] = 'OpenWeatherMap'

        # Select columns in correct order
        forecast_to_store = forecast_to_store[[
            'forecast_created_date',
            'forecast_created_timestamp',
            'target_date',
            'air_temp',
            'air_temp_min',
            'air_temp_max',
            'source'
        ]]

        try:
            conn.execute("""
                INSERT INTO air_temp_forecasts
                SELECT * FROM forecast_to_store
            """)
        except Exception as e:
            # Likely duplicate key - not a critical error
            err_msg = str(e).lower()
            if "primary key" in err_msg or "unique" in err_msg or "duplicate" in err_msg:
                pass  # Silently ignore duplicates
            else:
                raise ForecastStorageError(f"Failed to store air forecast: {e}")

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

        # Add metadata columns
        predictions_to_store = predictions_df.copy()
        predictions_to_store['forecast_created_date'] = forecast_created_timestamp.date()
        predictions_to_store['forecast_created_timestamp'] = forecast_created_timestamp
        predictions_to_store['target_date'] = predictions_to_store['date'].dt.date
        predictions_to_store['heat_transfer_coeff'] = heat_transfer_coeff
        predictions_to_store['start_water_temp'] = start_water_temp
        predictions_to_store['simulation_hours'] = 24
        predictions_to_store['source_air_forecast_timestamp'] = forecast_created_timestamp

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

        # Ensure datetime column is timezone-naive
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

        # Ensure datetime column is proper datetime type and timezone-naive
        # MotherDuck returns timezone-aware timestamps, but our local data is naive
        result["datetime"] = pd.to_datetime(result["datetime"]).dt.tz_localize(None)

        return result

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

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
            if "PRIMARY KEY" in str(e) or "UNIQUE" in str(e):
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
            if "PRIMARY KEY" in str(e) or "UNIQUE" in str(e):
                pass  # Silently ignore duplicates
            else:
                raise ForecastStorageError(f"Failed to store water predictions: {e}")

    def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

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
# Written as an aggregate and a join rather than rank() OVER. Both return the
# same 1,065 rows from the ~880k-row table, verified against production; this
# form is modestly faster and avoids ranking every row, which matters as the
# table grows. It is NOT what made the accuracy tab slow - that was connection
# setup, see _get_connection (issue #43).
#
# The join keeps EVERY row sharing the winning timestamp, which is what rank()
# did and what row_number() would not: one run is many rows, one per target
# date, and dropping all but one would silently lose the other horizons.
#
# The horizon window is expressed as a date range, not date_diff(), so the
# predicate can use idx_prediction_for_date instead of forcing a full scan.
# It is applied BEFORE choosing the winning run, exactly as the ranked version
# did: a run is "last" among the rows that are in window.
LAST_WATER_RUN_PER_DAY_SQL = """
    WITH in_window AS (
        SELECT *
        FROM water_temp_predictions
        WHERE target_date >= forecast_created_date
          AND target_date <= forecast_created_date + CAST(? AS INTEGER)
    ),
    last_run AS (
        SELECT forecast_created_date, max(forecast_created_timestamp) AS run_ts
        FROM in_window
        GROUP BY forecast_created_date
    )
    SELECT
        w.forecast_created_date,
        w.forecast_created_timestamp,
        w.target_date,
        w.water_temp AS forecast_temp,
        date_diff('day', w.forecast_created_date, w.target_date) AS horizon_days
    FROM in_window w
    JOIN last_run r
      ON w.forecast_created_date = r.forecast_created_date
     AND w.forecast_created_timestamp = r.run_ts
    ORDER BY w.target_date, horizon_days
"""


# The air forecast series each creation day ended with: one run, all its rows.
#
# Two sources may publish at the same hour, so choosing by timestamp alone can
# select two runs for one day and yield duplicate target_datetime rows, which
# breaks any caller that indexes by time. The winner is therefore one
# (timestamp, source) pair: latest run, then the one carrying more hours, then
# source name so the choice is deterministic.
#
# row_number() is correct here where rank() is not, because it ranks RUNS, one
# row each, rather than the forecast rows within them.
LAST_AIR_RUN_PER_DAY_SQL = """
    WITH air AS (
        SELECT
            CAST(forecast_created_timestamp AS DATE) AS forecast_created_date,
            forecast_created_timestamp,
            source,
            target_datetime,
            air_temp
        FROM weather_forecasts_hourly
        WHERE air_temp IS NOT NULL
    ),
    runs AS (
        SELECT forecast_created_date, forecast_created_timestamp, source,
               count(*) AS hours
        FROM air
        GROUP BY 1, 2, 3
    ),
    winner AS (
        SELECT forecast_created_date, forecast_created_timestamp, source,
               row_number() OVER (
                   PARTITION BY forecast_created_date
                   ORDER BY forecast_created_timestamp DESC, hours DESC, source
               ) AS run_rank
        FROM runs
    )
    SELECT a.forecast_created_date, a.target_datetime, a.air_temp
    FROM air a
    JOIN winner w
      ON a.forecast_created_date = w.forecast_created_date
     AND a.forecast_created_timestamp = w.forecast_created_timestamp
     AND a.source = w.source
    WHERE w.run_rank = 1
    ORDER BY a.forecast_created_date, a.target_datetime
"""


# Final weather forecast run of each creation day, all rows of it, per source.
# Same rank() reasoning as the air query above. Partitioned by source as well
# as creation day, so one source going quiet does not suppress another's run,
# and two sources may cover the same hour.
#
# Every measure comes back, including rows where one is NULL because that
# source does not publish it. Callers that need a particular measure select on
# it; see the solar/cloud provider in app.py.
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
        """
        Get or create the MotherDuck connection.

        Connecting is the expensive part by a wide margin - about 4.3 seconds
        against 0.08 for a query over the largest table we have. This used to
        connect TWICE, once without a database to run CREATE DATABASE IF NOT
        EXISTS and again to the database itself, so every instance paid roughly
        9 seconds before doing any work (issue #43).

        Now it connects straight to the database and only falls back to the
        bootstrap path if that fails, which happens once in the life of a
        deployment rather than on every connection.

        The fallback fires on any connection failure, not only a missing
        database, because duckdb does not distinguish the two here. A genuinely
        broken connection - bad token, network down - therefore costs two
        attempts before raising rather than one.
        """
        if self._conn is None:
            try:
                token = get_motherduck_token()
            except ConfigError as e:
                raise ForecastStorageError(f"Cannot connect to MotherDuck: {e}")

            connection_string = f"md:{self.database}?motherduck_token={token}"
            try:
                self._conn = duckdb.connect(connection_string)
            except Exception:
                # The database may not exist yet. Create it, then retry.
                try:
                    bootstrap = duckdb.connect(f"md:?motherduck_token={token}")
                    bootstrap.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                    bootstrap.close()
                    self._conn = duckdb.connect(connection_string)
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

        Writes merge. Calling this twice for the same source and hour - solar
        and cloud first, then air temperature - fills in the second set of
        measures rather than colliding on the primary key. A value overwrites;
        a NULL means "this source does not publish that", and leaves whatever
        is already stored.

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

        # Merge rather than insert-or-swallow. A row for this (run, hour,
        # source) may already exist carrying different measures, and a plain
        # insert would collide with the primary key; swallowing that as a
        # duplicate would discard the incoming measures silently.
        #
        # COALESCE(excluded.x, x) means a NULL says "this source does not
        # publish x", leaving whatever is already stored, while a real value
        # overwrites - so re-running within the same hour refreshes the numbers
        # and a second source's write fills in the columns it owns.
        try:
            conn.execute("""
                INSERT INTO weather_forecasts_hourly
                SELECT * FROM forecast_to_store
                ON CONFLICT (forecast_created_timestamp, target_datetime, source)
                DO UPDATE SET
                    air_temp = COALESCE(excluded.air_temp, weather_forecasts_hourly.air_temp),
                    shortwave_radiation = COALESCE(
                        excluded.shortwave_radiation,
                        weather_forecasts_hourly.shortwave_radiation
                    ),
                    cloud_cover = COALESCE(
                        excluded.cloud_cover, weather_forecasts_hourly.cloud_cover
                    )
            """)
        except Exception as e:
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
                FROM weather_forecasts_hourly
                WHERE DATE(target_datetime) = ? AND air_temp IS NOT NULL
            )
            SELECT target_datetime as datetime, air_temp
            FROM weather_forecasts_hourly
            WHERE forecast_created_timestamp = (SELECT latest_ts FROM latest_forecast)
              AND air_temp IS NOT NULL
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
                FROM weather_forecasts_hourly
                WHERE target_datetime > ?
                  AND target_datetime < ?
                  AND air_temp IS NOT NULL
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

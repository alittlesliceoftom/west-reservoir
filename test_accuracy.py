"""Tests for forecast accuracy reporting"""

import duckdb
import pandas as pd
import pytest
from datetime import datetime

from forecast_storage import LAST_AIR_RUN_PER_DAY_SQL, LAST_WATER_RUN_PER_DAY_SQL


def _local_predictions_db(rows):
    """In-memory DuckDB with the water_temp_predictions schema and given rows."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE water_temp_predictions (
            forecast_created_date DATE NOT NULL,
            forecast_created_timestamp TIMESTAMP NOT NULL,
            target_date DATE NOT NULL,
            water_temp DOUBLE NOT NULL,
            heat_transfer_coeff DOUBLE NOT NULL,
            start_water_temp DOUBLE NOT NULL,
            simulation_hours INTEGER NOT NULL,
            source_air_forecast_timestamp TIMESTAMP NOT NULL
        )
    """)
    for r in rows:
        conn.execute(
            "INSERT INTO water_temp_predictions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                r["created_date"], r["created_ts"], r["target_date"], r["water_temp"],
                0.02, 10.0, 24, r["created_ts"],
            ],
        )
    return conn


def _row(created_date, created_hour, target_date, water_temp):
    return {
        "created_date": created_date,
        "created_ts": datetime(
            created_date.year, created_date.month, created_date.day, created_hour
        ),
        "target_date": target_date,
        "water_temp": water_temp,
    }


class TestLastWaterRunPerDay:

    def test_picks_last_run_of_each_creation_day(self):
        rows = [
            _row(datetime(2026, 5, 1).date(), 8, datetime(2026, 5, 2).date(), 11.0),
            _row(datetime(2026, 5, 1).date(), 20, datetime(2026, 5, 2).date(), 12.0),
            _row(datetime(2026, 5, 1).date(), 14, datetime(2026, 5, 2).date(), 99.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert len(result) == 1
        assert result.loc[0, "forecast_temp"] == pytest.approx(12.0)

    def test_keeps_all_horizons_of_the_winning_run(self):
        """Selection is per creation day, not per row - rank(), not row_number()."""
        created = datetime(2026, 5, 10).date()
        rows = [
            _row(created, 20, datetime(2026, 5, 11).date(), 12.0),
            _row(created, 20, datetime(2026, 5, 12).date(), 13.0),
            _row(created, 20, datetime(2026, 5, 13).date(), 14.0),
            _row(created, 8, datetime(2026, 5, 11).date(), 99.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert sorted(result["horizon_days"]) == [1, 2, 3]
        assert 99.0 not in list(result["forecast_temp"])

    def test_excludes_negative_horizon_backfill(self):
        """Rows targeting dates BEFORE the run are historical backfill, not forecasts."""
        rows = [
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 11).date(), 12.0),
            _row(datetime(2026, 5, 10).date(), 12, datetime(2024, 12, 1).date(), 5.0),
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 9).date(), 6.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert set(result["horizon_days"]) == {1}

    def test_excludes_horizons_beyond_max(self):
        rows = [
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 11).date(), 12.0),
            _row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 20).date(), 13.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert list(result["horizon_days"]) == [1]

    def test_horizon_zero_is_included(self):
        rows = [_row(datetime(2026, 5, 10).date(), 12, datetime(2026, 5, 10).date(), 12.0)]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert list(result["horizon_days"]) == [0]

    def test_separate_creation_days_each_keep_their_own_run(self):
        rows = [
            _row(datetime(2026, 5, 1).date(), 20, datetime(2026, 5, 2).date(), 11.0),
            _row(datetime(2026, 5, 2).date(), 20, datetime(2026, 5, 3).date(), 12.0),
        ]
        conn = _local_predictions_db(rows)
        result = conn.execute(LAST_WATER_RUN_PER_DAY_SQL, [5]).fetchdf()

        assert len(result) == 2


def _local_air_db(rows):
    """In-memory DuckDB with the air_temp_forecasts_3hourly schema."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE air_temp_forecasts_3hourly (
            forecast_created_timestamp TIMESTAMP NOT NULL,
            target_datetime TIMESTAMP NOT NULL,
            air_temp DOUBLE NOT NULL,
            source VARCHAR DEFAULT 'OpenWeatherMap'
        )
    """)
    for created_ts, target_dt, air_temp in rows:
        conn.execute(
            "INSERT INTO air_temp_forecasts_3hourly VALUES (?, ?, ?, 'OpenWeatherMap')",
            [created_ts, target_dt, air_temp],
        )
    return conn


class TestLastAirRunPerDay:

    def test_picks_last_run_and_keeps_all_its_rows(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 1, 21), 15.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 14.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 3), 13.0),
            (datetime(2026, 5, 1, 8), datetime(2026, 5, 2, 0), 99.0),
        ]
        conn = _local_air_db(rows)
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        assert len(result) == 3
        assert 99.0 not in list(result["air_temp"])

    def test_groups_by_creation_date_not_timestamp(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 14.0),
            (datetime(2026, 5, 2, 20), datetime(2026, 5, 3, 0), 15.0),
        ]
        conn = _local_air_db(rows)
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        assert len(result) == 2
        assert sorted(pd.to_datetime(result["forecast_created_date"]).dt.day) == [1, 2]

    def test_rows_ordered_by_target_datetime(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 3, 0), 13.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 14.0),
        ]
        conn = _local_air_db(rows)
        result = conn.execute(LAST_AIR_RUN_PER_DAY_SQL).fetchdf()

        assert list(result["air_temp"]) == [14.0, 13.0]

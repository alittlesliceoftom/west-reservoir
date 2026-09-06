"""Tests for forecast accuracy reporting"""

import duckdb
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from accuracy import BIAS_NOTE, compute_metrics, metrics_by_horizon
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


def _scored(pairs, horizons=None):
    """pairs: list of (forecast, actual)."""
    df = pd.DataFrame(
        {"forecast_temp": [p[0] for p in pairs], "actual_temp": [p[1] for p in pairs]}
    )
    if horizons is not None:
        df["horizon_days"] = horizons
    return df


class TestComputeMetrics:

    def test_hand_computed_case(self):
        # errors: +1.0, -1.0, +2.0, 0.0
        # mae = 4.0/4 = 1.0 ; bias = 2.0/4 = 0.5
        # rmse = sqrt((1+1+4+0)/4) = sqrt(1.5)
        # within 0.5: only the 0.0 error -> 25%
        m = compute_metrics(_scored([(11.0, 10.0), (9.0, 10.0), (14.0, 12.0), (8.0, 8.0)]))

        assert m["mae"] == pytest.approx(1.0)
        assert m["bias"] == pytest.approx(0.5)
        assert m["rmse"] == pytest.approx(np.sqrt(1.5))
        assert m["hit_rate_0_5"] == pytest.approx(25.0)
        assert m["n"] == 4

    def test_bias_positive_means_model_runs_warm(self):
        m = compute_metrics(_scored([(12.0, 10.0), (13.0, 10.0)]))
        assert m["bias"] > 0
        assert "warm" in BIAS_NOTE

    def test_bias_negative_means_model_runs_cold(self):
        m = compute_metrics(_scored([(8.0, 10.0), (7.0, 10.0)]))
        assert m["bias"] < 0

    def test_hit_rate_boundary_is_inclusive(self):
        """Exactly 0.5 C off counts as a hit."""
        m = compute_metrics(_scored([(10.5, 10.0), (10.51, 10.0)]))
        assert m["hit_rate_0_5"] == pytest.approx(50.0)

    def test_nan_rows_are_excluded_not_counted(self):
        m = compute_metrics(_scored([(11.0, 10.0), (np.nan, 10.0), (9.0, 10.0)]))
        assert m["n"] == 2
        assert m["mae"] == pytest.approx(1.0)

    def test_empty_frame_returns_nan_metrics_and_zero_n(self):
        m = compute_metrics(_scored([]))
        assert m["n"] == 0
        assert np.isnan(m["mae"])
        assert np.isnan(m["bias"])
        assert np.isnan(m["rmse"])
        assert np.isnan(m["hit_rate_0_5"])

    def test_all_nan_frame_returns_zero_n(self):
        m = compute_metrics(_scored([(np.nan, 10.0)]))
        assert m["n"] == 0


class TestMetricsByHorizon:

    def test_one_row_per_horizon_sorted(self):
        df = _scored([(11.0, 10.0), (13.0, 10.0), (10.0, 10.0)], horizons=[2, 1, 1])
        result = metrics_by_horizon(df)

        assert list(result["horizon_days"]) == [1, 2]
        assert list(result.columns) == [
            "horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"
        ]

    def test_metrics_computed_within_horizon(self):
        df = _scored([(11.0, 10.0), (12.0, 10.0)], horizons=[1, 2])
        result = metrics_by_horizon(df).set_index("horizon_days")

        assert result.loc[1, "mae"] == pytest.approx(1.0)
        assert result.loc[2, "mae"] == pytest.approx(2.0)
        assert result.loc[1, "n"] == 1

    def test_horizon_with_only_nan_reports_zero_n(self):
        df = _scored([(np.nan, 10.0), (12.0, 10.0)], horizons=[1, 2])
        result = metrics_by_horizon(df).set_index("horizon_days")

        assert result.loc[1, "n"] == 0
        assert np.isnan(result.loc[1, "mae"])

    def test_empty_frame_returns_empty_with_columns(self):
        result = metrics_by_horizon(_scored([], horizons=[]))
        assert result.empty
        assert list(result.columns) == [
            "horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"
        ]

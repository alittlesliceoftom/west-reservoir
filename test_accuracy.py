"""Tests for forecast accuracy reporting"""

import duckdb
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from accuracy import (
    BIAS_NOTE,
    compute_metrics,
    join_actuals,
    metrics_by_horizon,
    replay_current_model,
    splice_air_history,
)
from data import build_hourly_weather
from forecaster import WaterTempForecaster
from forecast_storage import (
    ForecastStorage,
    LAST_AIR_RUN_PER_DAY_SQL,
    LAST_SOLAR_CLOUD_RUN_PER_DAY_SQL,
    LAST_WATER_RUN_PER_DAY_SQL,
)


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


class TestJoinActuals:

    def _forecasts(self, rows):
        """rows: list of (target_date, horizon_days, forecast_temp)."""
        return pd.DataFrame({
            "target_date": [pd.Timestamp(r[0]) for r in rows],
            "horizon_days": [r[1] for r in rows],
            "forecast_temp": [r[2] for r in rows],
        })

    def _measurements(self, rows):
        """rows: list of (date, water_temp)."""
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
        })

    def test_inner_join_keeps_only_measured_days(self):
        forecasts = self._forecasts([
            (datetime(2026, 5, 1), 1, 11.0),
            (datetime(2026, 5, 2), 1, 12.0),
        ])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)

        assert len(result) == 1
        assert result.loc[0, "actual_temp"] == pytest.approx(10.0)

    def test_error_is_forecast_minus_actual(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.5)])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert result.loc[0, "error"] == pytest.approx(1.5)

    def test_duplicate_measurement_dates_do_not_multiply_rows(self):
        """Duplicate dates have broken this codebase before - keep the last."""
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 1), 10.4),
        ])

        result = join_actuals(forecasts, measurements)

        assert len(result) == 1
        assert result.loc[0, "actual_temp"] == pytest.approx(10.4)

    def test_time_component_on_dates_does_not_break_join(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1, 7, 0), 1, 11.0)])
        measurements = self._measurements([(datetime(2026, 5, 1, 0, 0), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert len(result) == 1

    def test_air_source_preserved_when_present(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        forecasts["air_source"] = ["ACTUAL"]
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert result.loc[0, "air_source"] == "ACTUAL"

    def test_air_source_defaults_to_forecast_when_absent(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert result.loc[0, "air_source"] == "FORECAST"

    def test_nan_forecasts_are_kept_for_metrics_to_exclude(self):
        """Coverage gaps stay visible in the frame; compute_metrics drops them."""
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, np.nan)])
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = join_actuals(forecasts, measurements)
        assert len(result) == 1
        assert np.isnan(result.loc[0, "forecast_temp"])

    def test_missing_measurement_values_are_dropped(self):
        forecasts = self._forecasts([(datetime(2026, 5, 1), 1, 11.0)])
        measurements = self._measurements([(datetime(2026, 5, 1), np.nan)])

        result = join_actuals(forecasts, measurements)
        assert result.empty

    def test_empty_forecasts_returns_empty_with_schema(self):
        result = join_actuals(self._forecasts([]), self._measurements([]))
        assert result.empty
        assert list(result.columns) == [
            "target_date", "horizon_days", "forecast_temp",
            "actual_temp", "error", "air_source",
        ]


def _air_frame(start, n_hours, air_temp, step_hours=1):
    return pd.DataFrame({
        "datetime": [
            pd.Timestamp(start) + pd.Timedelta(hours=i * step_hours)
            for i in range(n_hours)
        ],
        "air_temp": [air_temp] * n_hours,
    })


class TestSpliceAirHistory:

    def test_actuals_fill_the_head_before_the_run_starts(self):
        """The live forecast had measured air for the elapsed part of the day."""
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)

        assert result["datetime"].min() == anchor
        before = result[result["datetime"] < pd.Timestamp(2026, 5, 1, 21)]
        after = result[result["datetime"] >= pd.Timestamp(2026, 5, 1, 21)]
        assert set(before["air_temp"]) == {10.0}
        assert set(after["air_temp"]) == {20.0}

    def test_no_hours_are_missing_across_the_join(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        gaps = result["datetime"].diff().dropna().unique()

        assert list(gaps) == [pd.Timedelta(hours=1)]

    def test_no_duplicate_datetimes(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        assert not result["datetime"].duplicated().any()

    def test_stored_wins_where_both_have_the_same_hour(self):
        """Past the forecast's creation time, the forecast is what was used."""
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        at_23 = result[result["datetime"] == pd.Timestamp(2026, 5, 1, 23)]

        assert at_23["air_temp"].iloc[0] == pytest.approx(20.0)

    def test_rows_before_the_anchor_are_dropped(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 4, 30, 0), 72, air_temp=10.0)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(actuals, stored, anchor)
        assert result["datetime"].min() == anchor

    def test_empty_stored_run_returns_actuals_from_anchor(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        actuals = _air_frame(pd.Timestamp(2026, 5, 1, 0), 48, air_temp=10.0)

        result = splice_air_history(
            actuals, pd.DataFrame(columns=["datetime", "air_temp"]), anchor
        )

        assert result["datetime"].min() == anchor
        assert set(result["air_temp"]) == {10.0}

    def test_empty_actuals_returns_stored_run_from_anchor(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        stored = _air_frame(pd.Timestamp(2026, 5, 1, 21), 24, air_temp=20.0)

        result = splice_air_history(
            pd.DataFrame(columns=["datetime", "air_temp"]), stored, anchor
        )

        assert set(result["air_temp"]) == {20.0}
        assert len(result) == 24

    def test_both_empty_returns_empty_with_schema(self):
        anchor = pd.Timestamp(2026, 5, 1, 7)
        empty = pd.DataFrame(columns=["datetime", "air_temp"])

        result = splice_air_history(empty, empty, anchor)

        assert result.empty
        assert list(result.columns) == ["datetime", "air_temp"]


def _weather_frame(start, n_hours, air_temp=15.0):
    return pd.DataFrame({
        "datetime": [pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(n_hours)],
        "air_temp": [air_temp] * n_hours,
        "shortwave_radiation": [0.0] * n_hours,
        "cloud_cover": [100.0] * n_hours,
    })


class TestReplayCurrentModel:

    def _measurements(self, dates_temps):
        return pd.DataFrame({
            "date": [pd.Timestamp(d) for d, _ in dates_temps],
            "water_temp": [t for _, t in dates_temps],
        })

    def _provider(self, label="FORECAST", n_hours=200, air_temp=15.0):
        def provider(anchor_date):
            return _weather_frame(anchor_date, n_hours, air_temp), label
        return provider

    def test_one_row_per_anchor_and_horizon(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 2), 10.5),
        ])

        result = replay_current_model(
            forecaster, measurements, self._provider(), max_horizon=3
        )

        assert set(result["horizon_days"]) == {1, 2, 3}
        assert len(result) == 6

    def test_horizon_zero_is_never_produced(self):
        """The replay is anchored on the measurement itself; it starts at day 1."""
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = replay_current_model(
            forecaster, measurements, self._provider(), max_horizon=3
        )
        assert 0 not in set(result["horizon_days"])

    def test_air_source_label_from_provider(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = replay_current_model(
            forecaster, measurements, self._provider(label="ACTUAL"), max_horizon=2
        )
        assert set(result["air_source"]) == {"ACTUAL"}

    def test_per_anchor_air_source_is_preserved(self):
        """Anchors with stored forecasts and anchors falling back must be distinguishable."""
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 2), 10.5),
        ])

        def provider(anchor_date):
            label = "FORECAST" if anchor_date.day == 1 else "ACTUAL"
            return _weather_frame(anchor_date, 200), label

        result = replay_current_model(forecaster, measurements, provider, max_horizon=1)

        by_date = result.set_index("target_date")["air_source"]
        assert by_date[pd.Timestamp(2026, 5, 2)] == "FORECAST"
        assert by_date[pd.Timestamp(2026, 5, 3)] == "ACTUAL"

    def test_warming_air_produces_warming_water(self):
        forecaster = WaterTempForecaster(k_air=0.05, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        result = replay_current_model(
            forecaster, measurements, self._provider(air_temp=25.0), max_horizon=3
        ).sort_values("horizon_days").reset_index(drop=True)

        assert result.loc[0, "forecast_temp"] > 10.0
        assert result["forecast_temp"].is_monotonic_increasing

    def test_anchor_with_no_weather_yields_nan_not_fabrication(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])

        def provider(anchor_date):
            return None, "ACTUAL"

        result = replay_current_model(forecaster, measurements, provider, max_horizon=2)

        assert len(result) == 2
        assert result["forecast_temp"].isna().all()
        assert set(result["air_source"]) == {"ACTUAL"}

    def test_measurements_with_nan_are_not_used_as_anchors(self):
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        measurements = self._measurements([
            (datetime(2026, 5, 1), np.nan),
            (datetime(2026, 5, 2), 10.0),
        ])

        result = replay_current_model(
            forecaster, measurements, self._provider(), max_horizon=1
        )

        assert len(result) == 1
        assert result.loc[0, "target_date"] == pd.Timestamp(2026, 5, 3)

    def test_forecaster_weather_state_is_restored(self):
        """The replay must not leave the forecaster pointing at replay weather."""
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        forecaster.set_hourly_weather(_weather_frame(datetime(2026, 1, 1), 48))
        before = forecaster.hourly_weather.copy()

        measurements = self._measurements([(datetime(2026, 5, 1), 10.0)])
        replay_current_model(forecaster, measurements, self._provider(), max_horizon=2)

        pd.testing.assert_frame_equal(forecaster.hourly_weather, before)

    def test_empty_measurements_returns_empty_with_schema(self):
        forecaster = WaterTempForecaster()
        result = replay_current_model(
            forecaster, self._measurements([]), self._provider(), max_horizon=2
        )
        assert result.empty
        assert list(result.columns) == [
            "target_date", "horizon_days", "forecast_temp", "air_source"
        ]


def _local_solar_db(rows):
    """In-memory DuckDB with the solar_cloud_forecasts_hourly schema."""
    conn = duckdb.connect(":memory:")
    conn.execute("""
        CREATE TABLE solar_cloud_forecasts_hourly (
            forecast_created_timestamp TIMESTAMP NOT NULL,
            target_datetime TIMESTAMP NOT NULL,
            shortwave_radiation DOUBLE NOT NULL,
            cloud_cover DOUBLE NOT NULL,
            source VARCHAR DEFAULT 'Open-Meteo',
            PRIMARY KEY (forecast_created_timestamp, target_datetime)
        )
    """)
    for created_ts, target_dt, solar, cloud in rows:
        conn.execute(
            "INSERT INTO solar_cloud_forecasts_hourly VALUES (?, ?, ?, ?, 'Open-Meteo')",
            [created_ts, target_dt, solar, cloud],
        )
    return conn


class TestLastSolarCloudRunPerDay:
    """The bulk query the replay uses: one run per creation day, all its rows."""

    def test_picks_last_run_and_keeps_all_its_rows(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 0.0, 90.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 12), 500.0, 10.0),
            (datetime(2026, 5, 1, 8), datetime(2026, 5, 2, 12), 999.0, 99.0),
        ]
        conn = _local_solar_db(rows)
        result = conn.execute(LAST_SOLAR_CLOUD_RUN_PER_DAY_SQL).fetchdf()

        assert len(result) == 2
        assert 999.0 not in list(result["shortwave_radiation"])

    def test_carries_both_measures(self):
        rows = [(datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 12), 500.0, 10.0)]
        conn = _local_solar_db(rows)
        result = conn.execute(LAST_SOLAR_CLOUD_RUN_PER_DAY_SQL).fetchdf()

        assert result.loc[0, "shortwave_radiation"] == pytest.approx(500.0)
        assert result.loc[0, "cloud_cover"] == pytest.approx(10.0)

    def test_groups_by_creation_date_not_timestamp(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 0.0, 90.0),
            (datetime(2026, 5, 2, 20), datetime(2026, 5, 3, 0), 0.0, 80.0),
        ]
        conn = _local_solar_db(rows)
        result = conn.execute(LAST_SOLAR_CLOUD_RUN_PER_DAY_SQL).fetchdf()

        assert len(result) == 2
        assert sorted(
            pd.to_datetime(result["forecast_created_date"]).dt.day
        ) == [1, 2]

    def test_rows_ordered_by_target_datetime(self):
        rows = [
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 3, 0), 1.0, 50.0),
            (datetime(2026, 5, 1, 20), datetime(2026, 5, 2, 0), 2.0, 60.0),
        ]
        conn = _local_solar_db(rows)
        result = conn.execute(LAST_SOLAR_CLOUD_RUN_PER_DAY_SQL).fetchdf()

        assert list(result["shortwave_radiation"]) == [2.0, 1.0]


class TestStoredSolarCloudFeedsTheModel:
    """
    Issue #29's acceptance test: a stored run can be read back and fed straight
    into build_hourly_weather in place of the actuals.
    """

    def _air(self, start, hours):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(hours)
            ],
            "air_temp": [15.0] * hours,
        })

    def _solar(self, start, hours, radiation, cloud):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(hours)
            ],
            "shortwave_radiation": [radiation] * hours,
            "cloud_cover": [cloud] * hours,
        })

    def test_stored_forecast_beats_actuals_on_overlap(self):
        """The replay must use what the forecast had, not what happened."""
        air = self._air("2026-05-01 07:00", 12)
        actual = self._solar("2026-05-01 07:00", 12, radiation=900.0, cloud=0.0)
        stored = self._solar("2026-05-01 07:00", 12, radiation=100.0, cloud=95.0)

        weather = build_hourly_weather(air, actual, stored)

        assert set(weather["shortwave_radiation"]) == {100.0}
        assert set(weather["cloud_cover"]) == {95.0}

    def test_actuals_fill_hours_the_stored_run_misses(self):
        air = self._air("2026-05-01 07:00", 12)
        actual = self._solar("2026-05-01 07:00", 12, radiation=900.0, cloud=0.0)
        stored = self._solar("2026-05-01 13:00", 6, radiation=100.0, cloud=95.0)

        weather = build_hourly_weather(air, actual, stored).set_index("datetime")

        assert weather.loc[pd.Timestamp("2026-05-01 08:00"), "cloud_cover"] == 0.0
        assert weather.loc[pd.Timestamp("2026-05-01 14:00"), "cloud_cover"] == 95.0

    def test_model_output_differs_with_stored_versus_actual_solar(self):
        """If swapping the source changed nothing, storing it would be pointless."""
        air = self._air("2026-05-01 07:00", 25)
        sunny = self._solar("2026-05-01 07:00", 25, radiation=900.0, cloud=0.0)
        overcast = self._solar("2026-05-01 07:00", 25, radiation=50.0, cloud=100.0)

        forecaster = WaterTempForecaster(k_air=0.01, k_solar=5e-4, k_cool=0.01)

        forecaster.set_hourly_weather(build_hourly_weather(air, sunny, None))
        with_sun = forecaster.predict_forward(
            pd.Timestamp("2026-05-01"), 12.0, days_ahead=1
        ).loc[0, "water_temp"]

        forecaster.set_hourly_weather(build_hourly_weather(air, overcast, None))
        with_cloud = forecaster.predict_forward(
            pd.Timestamp("2026-05-01"), 12.0, days_ahead=1
        ).loc[0, "water_temp"]

        assert with_sun > with_cloud

    def test_no_stored_run_falls_back_to_actuals(self):
        air = self._air("2026-05-01 07:00", 12)
        actual = self._solar("2026-05-01 07:00", 12, radiation=900.0, cloud=0.0)

        weather = build_hourly_weather(air, actual, None)

        assert set(weather["shortwave_radiation"]) == {900.0}


class TestSolarCloudStorageRoundTrip:
    """
    Exercises the real ForecastStorage methods against a local DuckDB.

    The SQL constants are tested above, but the methods around them - column
    order, the hour truncation, duplicate swallowing, the tz treatment - are
    where storage bugs actually live, and none of that is covered by testing
    the SQL alone.
    """

    def _storage(self):
        storage = ForecastStorage()
        # Bypass MotherDuck: every method reads self._conn.
        storage._conn = duckdb.connect(":memory:")
        storage.initialize_schema()
        return storage

    def _forecast(self, start, hours):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(start) + pd.Timedelta(hours=i) for i in range(hours)
            ],
            "shortwave_radiation": [float(100 * i) for i in range(hours)],
            "cloud_cover": [float(i) for i in range(hours)],
        })

    def test_stored_rows_come_back(self):
        storage = self._storage()
        storage.store_solar_cloud_forecast(
            self._forecast("2026-09-07 00:00", 24), datetime(2026, 9, 6, 21, 43, 17)
        )

        result = storage.get_solar_cloud_forecasts_last_run_per_day()

        assert len(result) == 24
        assert list(result.columns) == [
            "forecast_created_date", "target_datetime",
            "shortwave_radiation", "cloud_cover",
        ]

    def test_creation_timestamp_is_truncated_to_the_hour(self):
        """The PK dedupes within an hour only if the minutes are dropped."""
        storage = self._storage()
        storage.store_solar_cloud_forecast(
            self._forecast("2026-09-07 00:00", 3), datetime(2026, 9, 6, 21, 43, 17)
        )
        stored = storage._conn.execute(
            "SELECT DISTINCT forecast_created_timestamp FROM solar_cloud_forecasts_hourly"
        ).fetchdf()

        assert pd.Timestamp(stored.iloc[0, 0]) == pd.Timestamp("2026-09-06 21:00")

    def test_storing_twice_in_the_same_hour_is_silently_ignored(self):
        storage = self._storage()
        forecast = self._forecast("2026-09-07 00:00", 5)

        storage.store_solar_cloud_forecast(forecast, datetime(2026, 9, 6, 21, 0))
        storage.store_solar_cloud_forecast(forecast, datetime(2026, 9, 6, 21, 30))

        assert len(storage.get_solar_cloud_forecasts_last_run_per_day()) == 5

    def test_a_later_run_supersedes_an_earlier_one_the_same_day(self):
        storage = self._storage()
        storage.store_solar_cloud_forecast(
            self._forecast("2026-09-07 00:00", 3), datetime(2026, 9, 6, 8, 0)
        )
        late = self._forecast("2026-09-07 00:00", 3)
        late["cloud_cover"] = 77.0
        storage.store_solar_cloud_forecast(late, datetime(2026, 9, 6, 21, 0))

        result = storage.get_solar_cloud_forecasts_last_run_per_day()

        assert len(result) == 3
        assert set(result["cloud_cover"]) == {77.0}

    def test_empty_forecast_is_not_stored(self):
        storage = self._storage()
        storage.store_solar_cloud_forecast(
            pd.DataFrame(columns=["datetime", "shortwave_radiation", "cloud_cover"]),
            datetime(2026, 9, 6, 21, 0),
        )

        assert storage.get_solar_cloud_forecasts_last_run_per_day().empty

    def test_none_forecast_is_not_stored(self):
        storage = self._storage()
        storage.store_solar_cloud_forecast(None, datetime(2026, 9, 6, 21, 0))

        assert storage.get_solar_cloud_forecasts_last_run_per_day().empty

    def test_empty_table_returns_empty_frame(self):
        assert self._storage().get_solar_cloud_forecasts_last_run_per_day().empty

    def test_round_trip_output_feeds_build_hourly_weather(self):
        """Issue #29's acceptance criterion, end to end."""
        storage = self._storage()
        storage.store_solar_cloud_forecast(
            self._forecast("2026-09-07 00:00", 24), datetime(2026, 9, 6, 21, 0)
        )
        run = storage.get_solar_cloud_forecasts_last_run_per_day()

        air = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2026-09-07 00:00") + pd.Timedelta(hours=i)
                for i in range(24)
            ],
            "air_temp": [15.0] * 24,
        })
        weather = build_hourly_weather(
            air,
            None,
            run[["target_datetime", "shortwave_radiation", "cloud_cover"]].rename(
                columns={"target_datetime": "datetime"}
            ),
        )

        assert len(weather) == 24
        assert weather["shortwave_radiation"].max() == pytest.approx(2300.0)
        # Straight into the model, no further massaging.
        forecaster = WaterTempForecaster()
        forecaster.set_hourly_weather(weather)

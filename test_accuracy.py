"""Tests for forecast accuracy reporting"""

import duckdb
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from conftest import (
    AIR_FORECAST_3HOURLY_COLUMNS,
    WATER_PREDICTIONS_COLUMNS,
    hourly_frame,
    raw_table,
    tz_aware,
    weather_frame,
)
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
    ForecastStorageError,
    LAST_AIR_RUN_PER_DAY_SQL,
    LAST_WATER_RUN_PER_DAY_SQL,
)


# Columns get_weather_forecasts_last_run_per_day hands its callers. Asserted
# as a set almost everywhere; exactly one test below pins the order.
WEATHER_READ_COLUMNS = {
    "forecast_created_date", "target_datetime", "source",
    "air_temp", "shortwave_radiation", "cloud_cover",
}


def _local_predictions_db(rows):
    """In-memory DuckDB with the water_temp_predictions schema and given rows."""
    return raw_table(
        "water_temp_predictions",
        WATER_PREDICTIONS_COLUMNS,
        [
            (
                r["created_date"], r["created_ts"], r["target_date"], r["water_temp"],
                0.02, 10.0, 24, r["created_ts"],
            )
            for r in rows
        ],
    )


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

    def test_rank_keeps_every_row_of_the_winning_run(self):
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
    return raw_table(
        "air_temp_forecasts_3hourly",
        AIR_FORECAST_3HOURLY_COLUMNS,
        [(created_ts, target_dt, air_temp, "OpenWeatherMap")
         for created_ts, target_dt, air_temp in rows],
    )


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
        assert set(result.columns) == {
            "horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"
        }

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


def _air_frame(start, n_hours, air_temp, step_hours=1):
    """Hourly by default; step_hours=3 gives the 3-hourly stored-forecast shape."""
    if step_hours == 1:
        return hourly_frame(start, n_hours, air_temp=air_temp)
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


# The shared builder already produces exactly this shape.
_weather_frame = weather_frame


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

    def test_weather_state_is_restored_when_the_provider_raises(self):
        """
        The `finally` around the anchor loop exists ONLY for this path.

        test_forecaster_weather_state_is_restored above takes the happy path,
        which the restore after the loop would satisfy just as well - delete
        the try/finally and it still passes. Nothing reached the exception
        path, so nothing pinned it. A provider that fails partway (a stored
        run missing, a query erroring) must not leave the forecaster pointing
        at whichever anchor's weather it died on, because the dashboard goes
        on to use that same forecaster.
        """
        forecaster = WaterTempForecaster(k_air=0.02, k_solar=0.0, k_cool=0.0)
        forecaster.set_hourly_weather(_weather_frame(datetime(2026, 1, 1), 48))
        before = forecaster.hourly_weather.copy()

        measurements = self._measurements([
            (datetime(2026, 5, 1), 10.0),
            (datetime(2026, 5, 2), 10.5),
        ])

        calls = []

        def provider(anchor_date):
            calls.append(anchor_date)
            if len(calls) == 2:
                raise RuntimeError("stored run unavailable")
            return _weather_frame(anchor_date, 200), "FORECAST"

        with pytest.raises(RuntimeError, match="stored run unavailable"):
            replay_current_model(forecaster, measurements, provider, max_horizon=2)

        # It really did get past the first anchor and swap weather in.
        assert len(calls) == 2
        pd.testing.assert_frame_equal(forecaster.hourly_weather, before)


class TestWeatherForecastStorage:
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
        """A ramp, not a constant: test_round_trip asserts on its maximum."""
        return hourly_frame(
            start, hours,
            shortwave_radiation=[float(100 * i) for i in range(hours)],
            cloud_cover=[float(i) for i in range(hours)],
        )

    def test_stored_rows_come_back(self):
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 24),
            datetime(2026, 9, 6, 21, 43, 17),
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert len(result) == 24
        assert set(result.columns) == WEATHER_READ_COLUMNS

    def test_column_order_is_stable_for_downstream_positional_reads(self):
        """
        One deliberate order assertion for this read.

        Everywhere else asserts the column SET, so a reordering fails here and
        nowhere else - which is what you want when you go looking for why.
        """
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 2),
            datetime(2026, 9, 6, 21, 0),
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert list(result.columns) == [
            "forecast_created_date", "target_datetime", "source",
            "air_temp", "shortwave_radiation", "cloud_cover",
        ]

    def test_creation_timestamp_is_truncated_to_the_hour(self):
        """The PK dedupes within an hour only if the minutes are dropped."""
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 3),
            datetime(2026, 9, 6, 21, 43, 17),
            source="Open-Meteo",
        )
        stored = storage._conn.execute(
            "SELECT DISTINCT forecast_created_timestamp FROM weather_forecasts_hourly"
        ).fetchdf()

        assert pd.Timestamp(stored.iloc[0, 0]) == pd.Timestamp("2026-09-06 21:00")

    def test_storing_the_same_forecast_twice_is_idempotent(self):
        storage = self._storage()
        forecast = self._forecast("2026-09-07 00:00", 5)

        storage.store_weather_forecast(
            forecast, datetime(2026, 9, 6, 21, 0), source="Open-Meteo"
        )
        storage.store_weather_forecast(
            forecast, datetime(2026, 9, 6, 21, 30), source="Open-Meteo"
        )

        assert len(storage.get_weather_forecasts_last_run_per_day()) == 5

    def test_groups_by_creation_date_not_timestamp(self):
        """
        Moved off the raw-SQL fixture: runs on different days must both
        survive. No other test here spans two creation days.
        """
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 2),
            datetime(2026, 9, 6, 20, 0),
            source="Open-Meteo",
        )
        storage.store_weather_forecast(
            self._forecast("2026-09-08 00:00", 2),
            datetime(2026, 9, 7, 20, 0),
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert len(result) == 4
        assert sorted(
            set(pd.to_datetime(result["forecast_created_date"]).dt.day)
        ) == [6, 7]

    def test_rows_ordered_by_target_datetime(self):
        """
        Ordering comes from the SQL, not from insertion order, so the input
        is deliberately reversed. Nothing else here asserts ordering.
        """
        storage = self._storage()
        reversed_forecast = self._forecast("2026-09-07 00:00", 6).iloc[::-1]
        storage.store_weather_forecast(
            reversed_forecast, datetime(2026, 9, 6, 21, 0), source="Open-Meteo"
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert list(result["target_datetime"]) == sorted(result["target_datetime"])
        assert result["target_datetime"].iloc[0] == pd.Timestamp("2026-09-07 00:00")

    def test_a_later_run_supersedes_an_earlier_one_the_same_day(self):
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 3),
            datetime(2026, 9, 6, 8, 0),
            source="Open-Meteo",
        )
        late = self._forecast("2026-09-07 00:00", 3)
        late["cloud_cover"] = 77.0
        storage.store_weather_forecast(
            late, datetime(2026, 9, 6, 21, 0), source="Open-Meteo"
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert len(result) == 3
        assert set(result["cloud_cover"]) == {77.0}

    def test_empty_forecast_is_not_stored(self):
        storage = self._storage()
        storage.store_weather_forecast(
            pd.DataFrame(columns=["datetime", "shortwave_radiation", "cloud_cover"]),
            datetime(2026, 9, 6, 21, 0),
            source="Open-Meteo",
        )

        assert storage.get_weather_forecasts_last_run_per_day().empty

    def test_none_forecast_is_not_stored(self):
        storage = self._storage()
        storage.store_weather_forecast(
            None, datetime(2026, 9, 6, 21, 0), source="Open-Meteo"
        )

        assert storage.get_weather_forecasts_last_run_per_day().empty

    def test_empty_table_returns_empty_frame_with_the_full_schema(self):
        """
        Emptiness alone is not the contract.

        The read short-circuits on an empty result, skipping the tz coercions
        below it, so this is the one path that could hand callers a frame with
        the wrong columns. Callers index by name straight away.
        """
        result = self._storage().get_weather_forecasts_last_run_per_day()

        assert result.empty
        assert set(result.columns) == WEATHER_READ_COLUMNS

    def test_round_trip_output_feeds_build_hourly_weather(self):
        """Issue #29's acceptance criterion, end to end."""
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 24),
            datetime(2026, 9, 6, 21, 0),
            source="Open-Meteo",
        )
        run = storage.get_weather_forecasts_last_run_per_day()

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

    def test_a_source_may_supply_only_some_measures(self):
        """Open-Meteo sends solar and cloud; air temp is left NULL, not zero."""
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 3),
            datetime(2026, 9, 6, 21, 0),
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert result["air_temp"].isna().all()
        assert result["shortwave_radiation"].notna().all()

    def test_air_only_source_stores_air_and_leaves_the_rest_null(self):
        """The shape air temperature will arrive in when it moves here (#39)."""
        storage = self._storage()
        air_only = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2026-09-07 00:00") + pd.Timedelta(hours=i)
                for i in range(3)
            ],
            "air_temp": [12.0, 12.5, 13.0],
        })
        storage.store_weather_forecast(
            air_only, datetime(2026, 9, 6, 21, 0), source="OpenWeatherMap"
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert list(result["air_temp"]) == [12.0, 12.5, 13.0]
        assert result["shortwave_radiation"].isna().all()

    def test_two_sources_can_forecast_the_same_hour(self):
        """Source is in the primary key, so neither write displaces the other."""
        storage = self._storage()
        target = [pd.Timestamp("2026-09-07 00:00")]
        storage.store_weather_forecast(
            pd.DataFrame({"datetime": target, "air_temp": [12.0]}),
            datetime(2026, 9, 6, 21, 0),
            source="OpenWeatherMap",
        )
        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": target,
                "shortwave_radiation": [400.0],
                "cloud_cover": [20.0],
            }),
            datetime(2026, 9, 6, 21, 0),
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert len(result) == 2
        assert set(result["source"]) == {"OpenWeatherMap", "Open-Meteo"}

    def test_one_source_going_quiet_does_not_hide_the_others_run(self):
        """rank() partitions by source as well as day."""
        storage = self._storage()
        storage.store_weather_forecast(
            self._forecast("2026-09-07 00:00", 2),
            datetime(2026, 9, 6, 8, 0),
            source="Open-Meteo",
        )
        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": [pd.Timestamp("2026-09-07 00:00")],
                "air_temp": [12.0],
            }),
            datetime(2026, 9, 6, 21, 0),
            source="OpenWeatherMap",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        # The later OWM run must not suppress the earlier Open-Meteo one.
        assert len(result[result["source"] == "Open-Meteo"]) == 2
        assert len(result[result["source"] == "OpenWeatherMap"]) == 1

    def test_a_second_write_merges_instead_of_being_swallowed(self):
        """
        Writing solar and then air for one source and hour must keep both.

        A plain insert collides with the primary key, and swallowing that as a
        duplicate would discard the incoming measures silently - which is how
        air temperature would vanish when it moves to this table (#39).
        """
        storage = self._storage()
        target = [pd.Timestamp("2026-09-07 00:00")]
        created = datetime(2026, 9, 6, 21, 0)

        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": target,
                "shortwave_radiation": [400.0],
                "cloud_cover": [20.0],
            }),
            created,
            source="Open-Meteo",
        )
        storage.store_weather_forecast(
            pd.DataFrame({"datetime": target, "air_temp": [12.0]}),
            created,
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert len(result) == 1
        assert result["air_temp"].iloc[0] == pytest.approx(12.0)
        assert result["shortwave_radiation"].iloc[0] == pytest.approx(400.0)
        assert result["cloud_cover"].iloc[0] == pytest.approx(20.0)

    def test_a_later_write_refreshes_values_it_carries(self):
        """A real value overwrites; NULL means 'not published', not 'clear it'."""
        storage = self._storage()
        target = [pd.Timestamp("2026-09-07 00:00")]
        created = datetime(2026, 9, 6, 21, 0)

        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": target,
                "shortwave_radiation": [400.0],
                "cloud_cover": [20.0],
            }),
            created,
            source="Open-Meteo",
        )
        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": target,
                "shortwave_radiation": [111.0],
                "cloud_cover": [88.0],
            }),
            created,
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert result["shortwave_radiation"].iloc[0] == pytest.approx(111.0)
        assert result["cloud_cover"].iloc[0] == pytest.approx(88.0)

    def test_merging_does_not_reach_across_sources(self):
        """Each source owns its own row; a merge must not touch another's."""
        storage = self._storage()
        target = [pd.Timestamp("2026-09-07 00:00")]
        created = datetime(2026, 9, 6, 21, 0)

        storage.store_weather_forecast(
            pd.DataFrame({"datetime": target, "air_temp": [12.0]}),
            created,
            source="OpenWeatherMap",
        )
        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": target,
                "shortwave_radiation": [400.0],
                "cloud_cover": [20.0],
            }),
            created,
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day().set_index("source")

        assert result.loc["OpenWeatherMap", "air_temp"] == pytest.approx(12.0)
        assert pd.isna(result.loc["OpenWeatherMap", "shortwave_radiation"])
        assert pd.isna(result.loc["Open-Meteo", "air_temp"])

    def test_one_frame_carrying_every_measure_stores_them_all(self):
        """The shape #39 must use: one write per source per run."""
        storage = self._storage()
        storage.store_weather_forecast(
            pd.DataFrame({
                "datetime": [pd.Timestamp("2026-09-07 00:00")],
                "air_temp": [12.0],
                "shortwave_radiation": [400.0],
                "cloud_cover": [20.0],
            }),
            datetime(2026, 9, 6, 21, 0),
            source="Open-Meteo",
        )

        result = storage.get_weather_forecasts_last_run_per_day()

        assert result["air_temp"].iloc[0] == pytest.approx(12.0)
        assert result["shortwave_radiation"].iloc[0] == pytest.approx(400.0)
        assert result["cloud_cover"].iloc[0] == pytest.approx(20.0)

    def test_frame_without_any_measure_column_is_rejected(self):
        storage = self._storage()
        with pytest.raises(ForecastStorageError, match="measure column"):
            storage.store_weather_forecast(
                pd.DataFrame({"datetime": [pd.Timestamp("2026-09-07")], "wind": [3.0]}),
                datetime(2026, 9, 6, 21, 0),
                source="Open-Meteo",
            )


class TestStorageReadsCoerceTimezones:
    """
    The read methods are not thin SQL wrappers.

    MotherDuck hands back timezone-AWARE timestamps; every other frame in this
    codebase is naive, and comparing the two raises. So each read does
    tz_localize(None) on its timestamp columns and pd.to_datetime on its DATE
    columns. That is the "MotherDuck returns tz-aware, local data is naive" bug
    class, and it had no coverage at all.

    A plain in-memory table cannot reproduce it: DuckDB TIMESTAMP columns come
    back naive, so tz_localize(None) is a no-op and an "is naive" assertion
    passes with the coercion deleted. These fixtures therefore declare the
    timestamp columns TIMESTAMPTZ, which is what makes fetchdf return
    datetime64[us, UTC] and the coercion load-bearing. Verified by deleting
    each tz_localize(None): the matching test below fails.
    """

    def _storage_on(self, conn):
        storage = ForecastStorage()
        storage._conn = conn
        return storage

    def _water_db(self):
        """water_temp_predictions with tz-aware timestamps, as MotherDuck returns."""
        created = pd.Timestamp("2026-09-06 21:00:00", tz="UTC")
        return raw_table(
            "water_temp_predictions",
            tz_aware(
                WATER_PREDICTIONS_COLUMNS,
                "forecast_created_timestamp", "source_air_forecast_timestamp",
            ),
            [(
                datetime(2026, 9, 6).date(), created, datetime(2026, 9, 7).date(),
                12.0, 0.02, 11.0, 24, created,
            )],
        )

    def _air_db(self):
        """air_temp_forecasts_3hourly with tz-aware timestamps."""
        return raw_table(
            "air_temp_forecasts_3hourly",
            tz_aware(
                AIR_FORECAST_3HOURLY_COLUMNS,
                "forecast_created_timestamp", "target_datetime",
            ),
            [(
                pd.Timestamp("2026-09-06 21:00:00", tz="UTC"),
                pd.Timestamp("2026-09-07 00:00:00", tz="UTC"),
                14.0, "OpenWeatherMap",
            )],
        )

    def test_water_predictions_timestamps_come_back_naive(self):
        result = self._storage_on(
            self._water_db()
        ).get_water_predictions_last_run_per_day()

        assert result["forecast_created_timestamp"].dt.tz is None
        # Stripped, not converted away: the wall-clock reading is preserved.
        assert result["forecast_created_timestamp"].iloc[0] == pd.Timestamp(
            "2026-09-06 21:00:00"
        )

    def test_water_predictions_dates_come_back_as_timestamps(self):
        """
        Callers compare these against pd.Timestamp measurement dates and index
        by them. A datetime.date object would compare unequal to a Timestamp
        of the same day and the join would silently return nothing.

        Note: on local DuckDB, DATE already arrives as datetime64, so this
        pins the contract rather than exercising the pd.to_datetime call.
        """
        result = self._storage_on(
            self._water_db()
        ).get_water_predictions_last_run_per_day()

        assert isinstance(result["target_date"].iloc[0], pd.Timestamp)
        assert isinstance(result["forecast_created_date"].iloc[0], pd.Timestamp)
        assert result["target_date"].iloc[0] == pd.Timestamp("2026-09-07")

    def test_air_forecast_target_datetimes_come_back_naive(self):
        result = self._storage_on(
            self._air_db()
        ).get_air_forecasts_3hourly_last_run_per_day()

        assert result["target_datetime"].dt.tz is None
        assert result["target_datetime"].iloc[0] == pd.Timestamp("2026-09-07 00:00")

    def test_air_forecast_creation_dates_come_back_as_timestamps(self):
        result = self._storage_on(
            self._air_db()
        ).get_air_forecasts_3hourly_last_run_per_day()

        assert isinstance(result["forecast_created_date"].iloc[0], pd.Timestamp)
        assert result["forecast_created_date"].iloc[0] == pd.Timestamp("2026-09-06")

    def test_naive_output_is_comparable_with_local_naive_frames(self):
        """
        The point of the coercion, stated as the thing that actually broke:
        a tz-aware column raises on comparison with a naive one.
        """
        result = self._storage_on(
            self._air_db()
        ).get_air_forecasts_3hourly_last_run_per_day()

        local = pd.Series([pd.Timestamp("2026-09-07 00:00")])
        assert (result["target_datetime"] == local).iloc[0]


    def test_column_order_is_stable_for_downstream_positional_reads_water(self):
        """One deliberate order assertion for the water-predictions read."""
        result = self._storage_on(
            self._water_db()
        ).get_water_predictions_last_run_per_day()

        assert list(result.columns) == [
            "forecast_created_date", "forecast_created_timestamp",
            "target_date", "forecast_temp", "horizon_days",
        ]

    def test_column_order_is_stable_for_downstream_positional_reads_air(self):
        """One deliberate order assertion for the 3-hourly air read."""
        result = self._storage_on(
            self._air_db()
        ).get_air_forecasts_3hourly_last_run_per_day()

        assert list(result.columns) == [
            "forecast_created_date", "target_datetime", "air_temp"
        ]


def _empty_scored():
    return _scored([], horizons=[])


def _empty_hourly():
    return pd.DataFrame(columns=["datetime", "air_temp"])


def _empty_measurements():
    return pd.DataFrame(columns=["date", "water_temp"])


def _replay_on_empty():
    forecaster = WaterTempForecaster()
    return replay_current_model(
        forecaster,
        _empty_measurements(),
        lambda anchor: (weather_frame(anchor, 200), "FORECAST"),
        max_horizon=2,
    )


class TestEmptyInputContract:
    """
    Every reporting function must return an EMPTY FRAME WITH ITS SCHEMA on
    empty input, never a bare DataFrame() and never a raise.

    Consolidated from four near-identical tests scattered across the classes
    above. The contract is one rule, so it reads better as one table - and a
    new function is added by adding a row, which is the point.

    Callers index these frames by name immediately (the dashboard builds a
    chart from them before checking whether anything was scored), so a
    column-less empty frame is a KeyError at render time, not a quiet no-op.
    Asserting only `.empty` would not catch that, so each case pins its
    columns as a set - order is pinned once per storage read, above.
    """

    @pytest.mark.parametrize(
        "call, expected_columns",
        [
            pytest.param(
                lambda: metrics_by_horizon(_empty_scored()),
                {"horizon_days", "mae", "bias", "rmse", "hit_rate_0_5", "n"},
                id="metrics_by_horizon",
            ),
            pytest.param(
                lambda: join_actuals(
                    pd.DataFrame(columns=[
                        "target_date", "horizon_days", "forecast_temp"
                    ]),
                    _empty_measurements(),
                ),
                {
                    "target_date", "horizon_days", "forecast_temp",
                    "actual_temp", "error", "air_source",
                },
                id="join_actuals",
            ),
            pytest.param(
                lambda: splice_air_history(
                    _empty_hourly(), _empty_hourly(), pd.Timestamp(2026, 5, 1, 7)
                ),
                {"datetime", "air_temp"},
                id="splice_air_history",
            ),
            pytest.param(
                _replay_on_empty,
                {"target_date", "horizon_days", "forecast_temp", "air_source"},
                id="replay_current_model",
            ),
        ],
    )
    def test_empty_input_returns_an_empty_frame_carrying_its_schema(
        self, call, expected_columns
    ):
        result = call()

        assert result.empty
        assert set(result.columns) == expected_columns

    def test_compute_metrics_is_the_exception_and_returns_a_dict(self):
        """
        Not part of the table above: compute_metrics returns a metrics dict,
        not a frame. On empty input every metric is NaN and n is 0 - zero, not
        NaN, because "nothing was scored" is a count, and the dashboard prints
        it.
        """
        m = compute_metrics(_empty_scored())

        assert m["n"] == 0
        assert np.isnan(m["mae"])
        assert np.isnan(m["bias"])

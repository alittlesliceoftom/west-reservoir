"""Tests for data loading and processing functions"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from data import interpolate_to_hourly, combine_hourly_temps, DataLoadError


class TestInterpolateToHourly:
    """Tests for interpolate_to_hourly function"""

    def test_empty_dataframe(self):
        """Empty input returns empty output"""
        df = pd.DataFrame(columns=["datetime", "air_temp"])
        result = interpolate_to_hourly(df)
        assert result.empty

    def test_single_row(self):
        """Single row returns single row (no interpolation possible)"""
        df = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        result = interpolate_to_hourly(df)
        assert len(result) == 1
        assert result.iloc[0]["air_temp"] == 10.0

    def test_3hourly_to_hourly(self):
        """3-hourly data interpolates to hourly"""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 15:00"),
                pd.Timestamp("2024-01-15 18:00"),
            ],
            "air_temp": [10.0, 13.0, 16.0]
        })
        result = interpolate_to_hourly(df)

        # Should have 7 rows: 12, 13, 14, 15, 16, 17, 18
        assert len(result) == 7

        # Check interpolated values
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 13:00")]["air_temp"].iloc[0] == 11.0
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 14:00")]["air_temp"].iloc[0] == 12.0
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 16:00")]["air_temp"].iloc[0] == 14.0

    def test_preserves_original_values(self):
        """Original values are preserved exactly"""
        df = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 15:00"),
            ],
            "air_temp": [10.5, 13.5]
        })
        result = interpolate_to_hourly(df)

        # Original timestamps should have exact values
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 12:00")]["air_temp"].iloc[0] == 10.5
        assert result[result["datetime"] == pd.Timestamp("2024-01-15 15:00")]["air_temp"].iloc[0] == 13.5


class TestCombineHourlyTemps:
    """Tests for combine_hourly_temps function"""

    def test_empty_both(self):
        """Both empty returns empty"""
        hist = pd.DataFrame(columns=["datetime", "air_temp"])
        fore = pd.DataFrame(columns=["datetime", "air_temp"])
        result = combine_hourly_temps(hist, fore)
        assert result.empty

    def test_empty_historical(self):
        """Empty historical returns forecast only"""
        hist = pd.DataFrame(columns=["datetime", "air_temp"])
        fore = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        result = combine_hourly_temps(hist, fore)
        assert len(result) == 1

    def test_empty_forecast(self):
        """Empty forecast returns historical only"""
        hist = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        fore = pd.DataFrame(columns=["datetime", "air_temp"])
        result = combine_hourly_temps(hist, fore)
        assert len(result) == 1

    def test_historical_takes_precedence(self):
        """Historical data overrides forecast for overlapping times"""
        hist = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 12:00"),
                pd.Timestamp("2024-01-15 13:00"),
            ],
            "air_temp": [10.0, 11.0]
        })
        fore = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 13:00"),  # Overlaps
                pd.Timestamp("2024-01-15 14:00"),
            ],
            "air_temp": [99.0, 12.0]  # 99.0 should be ignored
        })
        result = combine_hourly_temps(hist, fore)

        # Should have 3 rows
        assert len(result) == 3

        # Historical value should be used for 13:00
        val_13 = result[result["datetime"] == pd.Timestamp("2024-01-15 13:00")]["air_temp"].iloc[0]
        assert val_13 == 11.0

    def test_no_overlap(self):
        """Non-overlapping data combines correctly"""
        hist = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 10:00"),
                pd.Timestamp("2024-01-15 11:00"),
            ],
            "air_temp": [8.0, 9.0]
        })
        fore = pd.DataFrame({
            "datetime": [
                pd.Timestamp("2024-01-15 14:00"),
                pd.Timestamp("2024-01-15 15:00"),
            ],
            "air_temp": [12.0, 13.0]
        })
        result = combine_hourly_temps(hist, fore)

        # Should have gap interpolated: 10, 11, 12, 13, 14, 15 = 6 rows
        assert len(result) == 6

    def test_sorted_output(self):
        """Output is sorted by datetime"""
        hist = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 12:00")],
            "air_temp": [10.0]
        })
        fore = pd.DataFrame({
            "datetime": [pd.Timestamp("2024-01-15 15:00")],
            "air_temp": [13.0]
        })
        result = combine_hourly_temps(hist, fore)

        # Verify sorted
        assert result["datetime"].is_monotonic_increasing


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


from data import select_storable_predictions


class TestSelectStorablePredictions:
    """Only genuine forward-looking forecasts should be stored."""

    def _frame(self, rows):
        """rows: list of (date, water_temp, source)."""
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
            "source": [r[2] for r in rows],
        })

    def test_keeps_predictions_from_today_onward(self):
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2026, 5, 10), 12.0, "PREDICTED"),
            (datetime(2026, 5, 11), 12.5, "PREDICTED"),
        ])
        result = select_storable_predictions(df, today)
        assert len(result) == 2

    def test_drops_predictions_for_past_dates(self):
        """Backfilled gap-fills target dates before the run - not forecasts."""
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2024, 12, 1), 5.0, "PREDICTED"),
            (datetime(2026, 5, 9), 11.0, "PREDICTED"),
            (datetime(2026, 5, 11), 12.5, "PREDICTED"),
        ])
        result = select_storable_predictions(df, today)
        assert list(result["date"]) == [pd.Timestamp(2026, 5, 11)]

    def test_drops_non_predicted_rows(self):
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2026, 5, 11), 12.5, "PREDICTED"),
            (datetime(2026, 5, 11), 12.5, "MEASURED"),
            (datetime(2026, 5, 12), 13.0, "AIR_ONLY"),
        ])
        result = select_storable_predictions(df, today)
        assert list(result["source"]) == ["PREDICTED"]

    def test_drops_rows_with_missing_water_temp(self):
        today = pd.Timestamp(2026, 5, 10)
        df = self._frame([
            (datetime(2026, 5, 11), float("nan"), "PREDICTED"),
            (datetime(2026, 5, 12), 13.0, "PREDICTED"),
        ])
        result = select_storable_predictions(df, today)
        assert len(result) == 1

    def test_run_date_with_time_component_still_keeps_today(self):
        """A run made at 9pm must still store today's prediction."""
        df = self._frame([(datetime(2026, 5, 10), 12.0, "PREDICTED")])
        result = select_storable_predictions(df, pd.Timestamp(2026, 5, 10, 21, 30))
        assert len(result) == 1

    def test_empty_frame_returns_empty(self):
        result = select_storable_predictions(self._frame([]), pd.Timestamp(2026, 5, 10))
        assert result.empty


from data import (
    build_temperatures_frame,
    deduplicate_temperatures,
    fill_daily_from_hourly,
)


class TestFillDailyFromHourly:
    """The daily Meteostat feed lags ~2 days; hourly is more current."""

    def _daily(self, rows):
        """rows: list of (date, air_temp, air_temp_min, air_temp_max)."""
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "air_temp": [r[1] for r in rows],
            "air_temp_min": [r[2] for r in rows],
            "air_temp_max": [r[3] for r in rows],
        })

    def _hourly(self, day, temps):
        return pd.DataFrame({
            "datetime": [
                pd.Timestamp(day) + pd.Timedelta(hours=h) for h in range(len(temps))
            ],
            "air_temp": list(temps),
        })

    def test_fills_missing_days_from_hourly(self):
        daily = self._daily([(datetime(2026, 5, 1), 10.0, 8.0, 12.0)])
        hourly = self._hourly(datetime(2026, 5, 2), [float(h) for h in range(24)])

        result = fill_daily_from_hourly(daily, hourly).set_index("date")

        assert pd.Timestamp(2026, 5, 2) in result.index
        assert result.loc[pd.Timestamp(2026, 5, 2), "air_temp"] == pytest.approx(11.5)
        assert result.loc[pd.Timestamp(2026, 5, 2), "air_temp_min"] == pytest.approx(0.0)
        assert result.loc[pd.Timestamp(2026, 5, 2), "air_temp_max"] == pytest.approx(23.0)

    def test_existing_daily_values_take_precedence(self):
        daily = self._daily([(datetime(2026, 5, 1), 10.0, 8.0, 12.0)])
        hourly = self._hourly(datetime(2026, 5, 1), [99.0] * 24)

        result = fill_daily_from_hourly(daily, hourly).set_index("date")
        assert result.loc[pd.Timestamp(2026, 5, 1), "air_temp"] == pytest.approx(10.0)

    def test_rows_without_air_temp_are_dropped(self):
        daily = self._daily([
            (datetime(2026, 5, 1), float("nan"), float("nan"), float("nan"))
        ])
        hourly = pd.DataFrame(columns=["datetime", "air_temp"])

        assert fill_daily_from_hourly(daily, hourly).empty

    def test_returns_expected_columns(self):
        daily = self._daily([(datetime(2026, 5, 1), 10.0, 8.0, 12.0)])
        hourly = pd.DataFrame(columns=["datetime", "air_temp"])

        result = fill_daily_from_hourly(daily, hourly)
        assert list(result.columns) == ["date", "air_temp", "air_temp_min", "air_temp_max"]

    def test_empty_hourly_leaves_daily_untouched(self):
        daily = self._daily([(datetime(2026, 5, 1), 10.0, 8.0, 12.0)])
        result = fill_daily_from_hourly(daily, pd.DataFrame(columns=["datetime", "air_temp"]))

        assert len(result) == 1
        assert result.loc[0, "air_temp"] == pytest.approx(10.0)


class TestBuildTemperaturesFrame:

    def _water(self, rows):
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
        })

    def _air(self, rows):
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "air_temp": [r[1] for r in rows],
            "air_temp_min": [r[1] - 2 for r in rows],
            "air_temp_max": [r[1] + 2 for r in rows],
        })

    def test_measured_where_water_temp_present(self):
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 1), 10.0)]),
            self._air([(datetime(2026, 5, 1), 15.0)]),
        )
        assert list(result["source"]) == ["MEASURED"]

    def test_air_only_where_water_temp_absent(self):
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 1), 10.0)]),
            self._air([(datetime(2026, 5, 1), 15.0), (datetime(2026, 5, 2), 16.0)]),
        ).set_index("date")

        assert result.loc[pd.Timestamp(2026, 5, 2), "source"] == "AIR_ONLY"

    def test_sorted_by_date(self):
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 3), 11.0), (datetime(2026, 5, 1), 10.0)]),
            self._air([(datetime(2026, 5, 1), 15.0)]),
        )
        assert result["date"].is_monotonic_increasing

    def test_water_only_dates_are_kept(self):
        """An outer join - a measurement with no air data must survive."""
        result = build_temperatures_frame(
            self._water([(datetime(2026, 5, 9), 12.0)]),
            self._air([(datetime(2026, 5, 1), 15.0)]),
        )
        assert pd.Timestamp(2026, 5, 9) in list(result["date"])


class TestDeduplicateTemperatures:

    def _frame(self, rows):
        return pd.DataFrame({
            "date": [pd.Timestamp(r[0]) for r in rows],
            "water_temp": [r[1] for r in rows],
            "source": [r[2] for r in rows],
        })

    def test_measured_beats_air_only_on_the_same_date(self):
        df = self._frame([
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert len(result) == 1
        assert result.loc[0, "source"] == "MEASURED"

    def test_air_only_beats_predicted_on_the_same_date(self):
        df = self._frame([
            (datetime(2026, 5, 1), 9.0, "PREDICTED"),
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
        ])
        result = deduplicate_temperatures(df)

        assert len(result) == 1
        assert result.loc[0, "source"] == "AIR_ONLY"

    def test_one_row_per_date(self):
        df = self._frame([
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
            (datetime(2026, 5, 2), 11.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert len(result) == 2
        assert not result["date"].duplicated().any()

    def test_helper_column_is_not_left_behind(self):
        df = self._frame([(datetime(2026, 5, 1), 10.0, "MEASURED")])
        assert "_sort_priority" not in deduplicate_temperatures(df).columns

    def test_result_is_sorted_and_reindexed(self):
        df = self._frame([
            (datetime(2026, 5, 2), 11.0, "MEASURED"),
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
        ])
        result = deduplicate_temperatures(df)

        assert result["date"].is_monotonic_increasing
        assert list(result.index) == [0, 1]

    def test_input_frame_is_not_mutated(self):
        df = self._frame([
            (datetime(2026, 5, 1), 10.0, "MEASURED"),
            (datetime(2026, 5, 1), float("nan"), "AIR_ONLY"),
        ])
        before = df.copy()
        deduplicate_temperatures(df)

        pd.testing.assert_frame_equal(df, before)
